"""Mesh-independent RGBD material observations and shared-point rigid motion.

Pixels are tracked before any candidate is considered. Depth support is lifted
from immutable source cloud neighborhoods, NEVER by casting onto a posed mesh.
All transforms here are world-from-object; latent points are motion variables.
"""
from dataclasses import dataclass
import hashlib

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from preprocessing.geometry import project, ray_directions


@dataclass
class Tracks:
    track: np.ndarray
    frame: np.ndarray
    uv: np.ndarray
    camera_points: np.ndarray
    normals: np.ndarray
    normal_quality: np.ndarray
    heldout: np.ndarray
    count: int
    report: dict

    def selected(self, holdout=False):
        keep = self.heldout[self.track] == holdout
        old = np.unique(self.track[keep])
        lookup = np.full(self.count, -1, int); lookup[old] = np.arange(len(old))
        return Tracks(lookup[self.track[keep]], self.frame[keep], self.uv[keep],
                      self.camera_points[keep], self.normals[keep], self.normal_quality[keep],
                      np.full(len(old), holdout), len(old), self.report)


def neighborhood_lift(cloud, uv, k, radius_px=5., max_neighbors=12):
    """Intersect each RGB ray with a local observed plane when it is supported.

    Otherwise use the neighbor's camera-Z on this ray (uncertain support, not a
    fabricated depth in a missing frame). Normals only qualify on a rank-2 plane.
    """
    cloud = np.asarray(cloud, float).reshape(-1, 3)
    uv = np.asarray(uv, float).reshape(-1, 2)
    q = np.full((len(uv), 3), np.nan); normals = np.zeros_like(q); quality = np.zeros(len(uv))
    if len(cloud) < 3 or not len(uv):
        return q, normals, quality
    good = np.isfinite(cloud).all(1) & (cloud[:, 2] > .001)
    cloud = cloud[good]
    if len(cloud) < 3:
        return q, normals, quality
    pixels = project(cloud, k)
    dist, ix = cKDTree(pixels).query(uv, k=min(max_neighbors, len(cloud)), workers=1)
    dirs = ray_directions(uv, k)
    for j in range(len(uv)):
        near = cloud[ix[j][dist[j] <= radius_px]]
        if not len(near):
            continue
        center = np.median(near, axis=0); z = center[2]
        if len(near) >= 5:
            centered = near - near.mean(0)
            _, values, axes = np.linalg.svd(centered, full_matrices=False)
            if values[1] > .0005 and values[1] > .06 * values[0]:
                normal = axes[-1]
                if normal @ center > 0:
                    normal = -normal
                flatness = values[-1] / max(values[1], 1e-12)
                denom = normal @ dirs[j]
                candidate_z = normal @ near.mean(0) / denom if abs(denom) > .1 else z
                if flatness < .35 and abs(candidate_z - z) < .015:
                    z = candidate_z; normals[j] = normal
                    quality[j] = max(0., 1. - flatness / .35)
        if z > .001 and np.isfinite(z):
            q[j] = dirs[j] * z
    return q, normals, quality


def extract_tracks(inputs, stride=8, half_window=8, features=24):
    """Fixed forward/backward LK tracklets in raw SAM2 support, not mesh pixels."""
    cv2.setNumThreads(1)
    rows = []; identities = []; n = len(inputs.ids)
    anchors = sorted(set(range(0, n, stride)) | {int(inputs.anchor), n - 1})
    for anchor in anchors:
        mask = inputs.masks[anchor].copy()
        if inputs.owned_masks is not None and np.any(inputs.owned_masks[anchor]):
            mask &= inputs.owned_masks[anchor]
        mask = cv2.erode(mask.astype('uint8'), np.ones((5, 5), np.uint8)) * 255
        corners = cv2.goodFeaturesToTrack(inputs.grays[anchor], features, .015, 7, mask=mask)
        if corners is None:
            continue
        pixels = corners.reshape(-1, 2)
        observations = [{anchor: p.copy()} for p in pixels]
        for direction in (-1, 1):
            previous = anchor; current_pixels = pixels.copy(); alive = np.ones(len(pixels), bool)
            for step in range(1, half_window + 1):
                frame = anchor + direction * step
                if frame < 0 or frame >= n:
                    break
                p0 = current_pixels.astype('float32').reshape(-1, 1, 2)
                p1, status, _ = cv2.calcOpticalFlowPyrLK(inputs.grays[previous], inputs.grays[frame], p0, None,
                    winSize=(21, 21), maxLevel=3)
                back, bs, _ = cv2.calcOpticalFlowPyrLK(inputs.grays[frame], inputs.grays[previous], p1, None,
                    winSize=(21, 21), maxLevel=3)
                new = p1.reshape(-1, 2); fb = np.linalg.norm(back.reshape(-1, 2) - current_pixels, axis=1)
                pixel = np.rint(new).astype(int)
                inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < inputs.wh[0]) & (pixel[:, 1] >= 0) & (pixel[:, 1] < inputs.wh[1])
                in_mask = np.zeros(len(new), bool)
                ii = np.flatnonzero(inside)
                in_mask[ii] = inputs.masks[frame, pixel[ii, 1], pixel[ii, 0]]
                # Source ownership is fixed; no candidate-dependent mask pruning.
                if inputs.owned_masks is not None and np.any(inputs.owned_masks[frame]):
                    in_mask[ii] &= inputs.owned_masks[frame, pixel[ii, 1], pixel[ii, 0]]
                alive &= status.ravel().astype(bool) & bs.ravel().astype(bool) & (fb <= 1.5) & in_mask & np.isfinite(new).all(1)
                for j in np.flatnonzero(alive):
                    observations[j][frame] = new[j].copy()
                current_pixels = new; previous = frame
        for feature, obs in enumerate(observations):
            if len(obs) < 3:
                continue
            tid = len(identities); identities.append((anchor, feature))
            rows += [(tid, frame, uv) for frame, uv in sorted(obs.items())]
    if rows:
        track = np.array([r[0] for r in rows]); frame = np.array([r[1] for r in rows]); uv = np.array([r[2] for r in rows])
    else:
        track = np.empty(0, int); frame = np.empty(0, int); uv = np.empty((0, 2))
    q = np.full((len(rows), 3), np.nan); normal = np.zeros_like(q); quality = np.zeros(len(rows))
    for i in range(n):
        ix = np.flatnonzero(frame == i)
        q[ix], normal[ix], quality[ix] = neighborhood_lift(inputs.clouds[i], uv[ix], inputs.K[i])
    # At least two measured frames initialize a material point. All its 2D rows,
    # including intervening depth gaps, are retained. No mesh-created depth.
    supported = np.bincount(track[np.isfinite(q).all(1)], minlength=len(identities)) >= 2
    keep = supported[track]; old = np.flatnonzero(supported)
    lookup = np.full(len(identities), -1, int); lookup[old] = np.arange(len(old))
    heldout = np.array([hashlib.sha256(str(identities[j]).encode()).digest()[0] % 5 == 0 for j in old])
    report = {'detected_tracklets': len(identities), 'depth_supported_tracklets': len(old),
              'observations': int(keep.sum()), 'heldout_tracklets': int(heldout.sum()),
              'metric_observations': int(np.isfinite(q[keep]).all(1).sum()),
              'no_mesh_binding': True, 'whole_tracklet_holdout': True,
              'not_independent_material_identity_across_redetection': True,
              'frame_observation_counts': np.bincount(frame[keep], minlength=n).tolist(),
              'parameters': {'stride': stride, 'half_window': half_window, 'features': features}}
    return Tracks(lookup[track[keep]], frame[keep], uv[keep], q[keep], normal[keep], quality[keep], heldout, len(old), report)


def rigid_fit(source, target):
    a, b = np.asarray(source), np.asarray(target)
    ac, bc = a.mean(0), b.mean(0)
    u, values, vt = np.linalg.svd((a - ac).T @ (b - bc))
    if values[1] < max(1e-12, values[0] * .005):
        raise ValueError('Collinear material support')
    d = np.eye(3); d[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ d @ u.T
    return r, bc - r @ ac


def robust_rigid(source, target, trials=96, threshold=.015):
    if len(source) < 6:
        return None
    rng = np.random.default_rng(47); best = None; best_score = (-1, -np.inf)
    for _ in range(trials):
        ii = rng.choice(len(source), 3, replace=False)
        try:
            r, t = rigid_fit(source[ii], target[ii])
        except ValueError:
            continue
        error = np.linalg.norm(source @ r.T + t - target, axis=1); inliers = error < threshold
        score = (int(inliers.sum()), -float(np.minimum(error, threshold).sum()))
        if score > best_score:
            best_score, best = score, (r, t, inliers)
    if best is None or best[2].sum() < 6:
        return None
    r, t = rigid_fit(source[best[2]], target[best[2]])
    error = np.linalg.norm(source @ r.T + t - target, axis=1)
    return r, t, error < threshold, error


def rotation_edges(tracks, inputs):
    world = np.einsum('nij,nj->ni', inputs.cameras[tracks.frame, :3, :3], np.nan_to_num(tracks.camera_points)) + inputs.cameras[tracks.frame, :3, 3]
    pairs = {}
    for tid in range(tracks.count):
        ix = np.flatnonzero((tracks.track == tid) & np.isfinite(tracks.camera_points).all(1))
        byframe = {int(tracks.frame[j]): j for j in ix}
        for a in byframe:
            for gap in (4, 8):
                b = a + gap
                if b in byframe:
                    pairs.setdefault((a, b), []).append((world[byframe[a]], world[byframe[b]]))
    edges = []
    for (a, b), points in sorted(pairs.items()):
        points = np.asarray(points); result = robust_rigid(points[:, 0], points[:, 1])
        if result is None:
            continue
        r, _, good, error = result
        edges.append({'a': a, 'b': b, 'R': r, 'count': int(good.sum()), 'total': len(points),
                      'median_m': float(np.median(error[good])), 'weight': min(1., good.sum()/16.) * float(good.mean())})
    return edges


def propagated_rotation(initial, anchor, edges):
    """Maximum-support growing tree, never integrate noisy metric translation."""
    R = initial.copy(); reached = {anchor}
    while True:
        possible = [e for e in edges if (e['a'] in reached) != (e['b'] in reached)]
        if not possible:
            break
        e = max(possible, key=lambda x: x['weight'])
        if e['a'] in reached:
            R[e['b']] = e['R'] @ R[e['a']]; reached.add(e['b'])
        else:
            R[e['a']] = e['R'].T @ R[e['b']]; reached.add(e['a'])
    return R, reached


class MaterialMotionProblem:
    """Sparse SE3 + latent material point bundle adjustment on actual observations."""
    def __init__(self, inputs, tracks, R, t, edges=()):
        self.inputs, self.tracks = inputs, tracks
        self.R0, self.t0 = R.copy(), t.copy(); self.n = len(R); self.anchor = int(inputs.anchor)
        self.free = np.array([i for i in range(self.n) if i != self.anchor]); self.pose_start = np.full(self.n, -1, int)
        self.pose_start[self.free] = np.arange(len(self.free)) * 6
        self.point_start = 6 * len(self.free); self.size = self.point_start + 3 * tracks.count
        self.edges = list(edges)
        self.depth = np.isfinite(tracks.camera_points).all(1)
        self.metric_ix = np.flatnonzero(self.depth)
        self.normal_ix = np.flatnonzero(self.depth & (tracks.normal_quality > 0))
        self.frame_weights = 1. / np.sqrt(np.maximum(1, np.bincount(tracks.frame, minlength=self.n))[tracks.frame])
        self.points0 = self.initialize_points(R, t)

    def initialize_points(self, R, t):
        tr = self.tracks; world = np.einsum('nij,nj->ni', self.inputs.cameras[tr.frame, :3, :3], np.nan_to_num(tr.camera_points)) + self.inputs.cameras[tr.frame, :3, 3]
        local = np.einsum('nji,nj->ni', R[tr.frame], world - t[tr.frame])
        points = np.zeros((tr.count, 3))
        for j in range(tr.count):
            ii = (tr.track == j) & np.isfinite(tr.camera_points).all(1)
            if not np.any(ii):
                raise ValueError('Latent point has no measured initialization')
            points[j] = np.median(local[ii], axis=0)
        return points

    def initial(self):
        return np.r_[np.zeros(self.point_start), self.points0.ravel()]

    def unpack(self, x):
        R, t = self.R0.copy(), self.t0.copy()
        delta = np.asarray(x[:self.point_start]).reshape(-1, 6)
        R[self.free] = Rotation.from_rotvec(delta[:, :3]).as_matrix() @ R[self.free]
        t[self.free] += delta[:, 3:]
        return R, t, x[self.point_start:].reshape(-1, 3)

    def observations(self, R, t, points):
        tr = self.tracks; f = tr.frame
        world = np.einsum('nij,nj->ni', R[f], points[tr.track]) + t[f]
        camera = np.einsum('nji,nj->ni', self.inputs.cameras[f, :3, :3], world-self.inputs.cameras[f, :3, 3])
        pixel = (project(camera, self.inputs.K[f]) - tr.uv) / 3.
        parts = [(pixel * self.frame_weights[:, None]).ravel()]
        ix = self.metric_ix
        parts.append(((camera[ix, 2] - tr.camera_points[ix, 2]) / .025 * self.frame_weights[ix]).ravel())
        ix = self.normal_ix
        parts.append((np.sum((camera[ix] - tr.camera_points[ix]) * tr.normals[ix], axis=1) / .012 * self.frame_weights[ix] * np.sqrt(tr.normal_quality[ix])).ravel())
        parts.append((np.maximum(.01 - camera[:, 2], 0) / .01 * self.frame_weights).ravel())
        return parts

    def residual(self, x):
        R, t, points = self.unpack(x); parts = self.observations(R, t, points)
        if self.edges:
            a = np.array([e['a'] for e in self.edges]); b = np.array([e['b'] for e in self.edges]); er = np.array([e['R'] for e in self.edges])
            weight = np.sqrt([e['weight'] for e in self.edges])
            parts.append((Rotation.from_matrix((R[b] @ R[a].transpose(0, 2, 1)) @ er.transpose(0, 2, 1)).as_rotvec() / .2 * weight[:, None] * .4).ravel())
        # Weak regularity relative to initialization, not an absolute motion cap.
        correction = R @ self.R0.transpose(0, 2, 1)
        parts.append((Rotation.from_matrix(correction[1:] @ correction[:-1].transpose(0, 2, 1)).as_rotvec() / .4 * .1).ravel())
        parts.append((np.diff(t-self.t0, axis=0) / .04 * .1).ravel())
        # No strong material-coordinate prior and no mesh truth.
        return np.concatenate(parts)

    def columns(self, frame, track=None):
        cols = [] if frame == self.anchor else list(range(self.pose_start[frame], self.pose_start[frame] + 6))
        if track is not None:
            cols += list(range(self.point_start + 3*track, self.point_start+3*track+3))
        return cols

    def sparsity(self):
        tr = self.tracks; J = lil_matrix((len(MaterialMotionProblem.residual(self, self.initial())), self.size), dtype=np.int8); at = 0
        for indices, dim in [(np.arange(len(tr.frame)), 2), (self.metric_ix, 1), (self.normal_ix, 1), (np.arange(len(tr.frame)), 1)]:
            for j in indices:
                J[at:at+dim, self.columns(int(tr.frame[j]), int(tr.track[j]))] = 1; at += dim
        for edge in self.edges:
            J[at:at+3, self.columns(edge['a']) + self.columns(edge['b'])] = 1; at += 3
        for _ in range(2):
            for j in range(self.n-1):
                J[at:at+3, self.columns(j) + self.columns(j+1)] = 1; at += 3
        assert at == J.shape[0]
        return J.tocsr()

    def solve(self, max_nfev=40):
        x0 = self.initial()
        if self.tracks.count == 0:
            return self.R0.copy(), self.t0.copy(), {'status': 'no_material_support', 'nfev': 0}
        fit = least_squares(self.residual, x0, jac_sparsity=self.sparsity(), loss='soft_l1',
                            max_nfev=max_nfev, x_scale='jac', ftol=1e-5, xtol=1e-5, gtol=1e-5)
        if not np.isfinite(fit.x).all():
            raise RuntimeError('Nonfinite observation motion fit')
        R, t, _ = self.unpack(fit.x)
        return R, t, {'status': 'observation_motion_fit', 'nfev': fit.nfev,
                      'cost_before': robust_cost(self.residual(x0)), 'cost_after': robust_cost(self.residual(fit.x)),
                      'latent_points': self.tracks.count, 'no_mesh_binding': True}


def robust_cost(residual):
    residual = np.asarray(residual)
    return float(np.sum(np.sqrt(1. + residual**2) - 1.))


def fixed_track_score(inputs, tracks, R, t):
    if not tracks.count:
        return {'available': False, 'cost': 0., 'observations': 0}
    problem = MaterialMotionProblem(inputs, tracks, R, t)
    # Same metric rows initialize/project out nuisance coordinates for every pose;
    # no inlier re-selection or candidate-dependent removal from the score.
    points = problem.initialize_points(R, t)
    residual = np.concatenate(problem.observations(R, t, points))
    return {'available': True, 'cost': robust_cost(residual), 'observations': len(tracks.frame),
            'metric_observations': int(problem.depth.sum()), 'population_fixed': True}
