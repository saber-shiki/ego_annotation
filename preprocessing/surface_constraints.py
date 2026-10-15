"""Source-derived open surface envelopes and full-triangle observation factors."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from preprocessing.geometry import camera_pose, project, resized_intrinsics, surface_samples
from preprocessing.factors import subset
from preprocessing.mask_contract import FixedMaskDomain, boundary
from preprocessing.material_motion import neighborhood_lift


@dataclass
class Envelope:
    center_world: np.ndarray
    basis_world: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    observed: bool

    def residual(self, world_points):
        local = (np.asarray(world_points) - self.center_world) @ self.basis_world
        return np.maximum(self.lower - local, 0) + np.maximum(local - self.upper, 0)


def canonical_dimensions(vertices, faces):
    points = surface_samples(vertices, faces, 1024)
    _, _, axes = np.linalg.svd(points-points.mean(0), full_matrices=False)
    return np.sort(np.ptp(points @ axes.T, axis=0))[::-1]


def observed_envelope(cloud, camera, canonical_vertices, canonical_faces, dimensions=None):
    """No current pose input. Hidden dimensions use a broad shared shape prior.

    Long tangent/second tangent/away-normal axes come from the measured partial
    cloud. A near-planar cloud supplies a reliable surface normal, NOT full SO3.
    Hidden range extends away from the camera and is deliberately uncertain.
    """
    q = np.asarray(cloud, float).reshape(-1, 3)
    if len(q) < 6:
        return Envelope(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), 0., False)
    center = np.median(q, axis=0)
    _, values, vt = np.linalg.svd(q - center, full_matrices=False)
    if values[1] < 1e-6:
        return Envelope(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), 0., False)
    normal = vt[-1]
    # +z extends behind visible surface, away from camera. Not an observed backface.
    if normal @ center < 0:
        normal = -normal
    x = vt[0]; y = np.cross(normal, x); y /= np.linalg.norm(y); x = np.cross(y, normal)
    basis = np.column_stack((x, y, normal))
    local = (q - center) @ basis
    lo, hi = np.quantile(local, [.01, .99], axis=0)
    if dimensions is None:
        dimensions = canonical_dimensions(canonical_vertices, canonical_faces)
    # Tangential incomplete extents and thickness are priors, never direct sizes.
    pad = .008 + .1 * dimensions[0]
    half = np.maximum((hi[:2]-lo[:2])/2, dimensions[:2]/2)
    mid = (lo[:2]+hi[:2])/2
    lower = np.r_[mid-half-pad, lo[2]-pad]
    upper = np.r_[mid+half+pad, hi[2]+dimensions[1]+pad]
    quality = float(np.clip(1.-values[-1]/max(values[1], 1e-12), .1, 1.))
    return Envelope(camera[:3, :3] @ center + camera[:3, 3], camera[:3, :3] @ basis,
                    lower, upper, quality, True)


class SurfaceFrame:
    def __init__(self, inputs, mesh, hand_depth, i, points=64, contours=48):
        self.inputs, self.mesh, self.i = inputs, mesh, i
        self.C = inputs.cameras[i]; self.K = inputs.K[i]
        self.domain = FixedMaskDomain.from_inputs(inputs, hand_depth, i)
        self.wh = self.domain.target.shape[::-1]
        self.raster_K = resized_intrinsics(self.K, inputs.wh, self.wh)
        self.q = subset(inputs.clouds[i], points).copy()
        self.q.setflags(write=False)
        self.uv = project(self.q, self.K) if len(self.q) else np.empty((0, 2))
        _, self.normal, self.quality = neighborhood_lift(inputs.clouds[i], self.uv, self.K)
        if not hasattr(mesh, '_envelope_dimensions'):
            mesh._envelope_dimensions = canonical_dimensions(mesh.vertices, mesh.faces)
        self.envelope = observed_envelope(inputs.clouds[i], self.C, mesh.vertices, mesh.faces, mesh._envelope_dimensions)
        if not hasattr(mesh, '_envelope_samples'):
            mesh._envelope_samples = surface_samples(mesh.vertices, mesh.faces, 256)
        self.samples = mesh._envelope_samples
        self.contours = contours

    def view(self, R, t):
        r, v = camera_pose(R, t, self.C)
        return self.mesh.raster(self.raster_K, r, v, self.wh)

    def correspondences(self, R, t, view=None):
        r, v = camera_pose(R, t, self.C)
        if view is None:
            view = self.view(R, t)
        X, _, hit = self.mesh.query(self.uv, self.K, r, v)
        if np.any(~hit):
            visible = view[0][view[2]]
            if len(visible):
                _, near = cKDTree(visible @ r.T + v).query(self.q[~hit], workers=1)
                X[~hit] = visible[near]
            else:
                # Preserve population even when no projected mesh is visible.
                X[~hit] = self.samples[0]
        nb = self.mesh.normals(X) if len(X) else np.empty((0, 3))
        mb = boundary(view[2]) & self.domain.boundary_domain
        tb = boundary(self.domain.target) & self.domain.boundary_domain
        my, mx = np.nonzero(mb); ty, tx = np.nonzero(tb)
        if min(len(mx), len(tx)) >= 3:
            a = subset(np.arange(len(mx)), self.contours); b = subset(np.arange(len(tx)), self.contours)
            mxy, txy = np.c_[mx, my], np.c_[tx, ty]
            _, j = cKDTree(txy).query(mxy[a], workers=1); _, k = cKDTree(mxy).query(txy[b], workers=1)
            Y = np.r_[view[0][my[a], mx[a]], view[0][my[k], mx[k]]]
            uv = np.r_[txy[j], txy[b]]
        else:
            Y, uv = np.empty((0, 3)), np.empty((0, 2))
        return X, hit, nb, Y, uv

    def residual(self, R, t, correspondences, box_weight=0.):
        X, hit, nb, Y, uv = correspondences
        r, v = camera_pose(R, t, self.C)
        prediction = X @ r.T + v; parts = []
        if len(X):
            w = 1./np.sqrt(len(X))
            delta = prediction-self.q
            # Measured local normals, not normals defined by the old pose.
            n = self.normal.copy(); missing = self.quality <= 0
            n[missing] = np.array([0., 0., 1.])
            plane = np.sum(delta*n, axis=1)
            parts.append(plane/.012*w)
            # Missed rays still have a full visible-distance residual.
            parts.append((delta * (~hit)[:, None] / .025*w).ravel())
            model_normals = nb @ r.T
            # Unsigned normal axis: no collision/sign authority from generated faces.
            normal_error = np.cross(model_normals, n) * np.sqrt(self.quality)[:, None]
            parts.append((normal_error/.35*w).ravel())
        if len(Y):
            parts.append(((project(Y @ r.T + v, self.raster_K)-uv)/2./np.sqrt(len(Y))).ravel())
        min_z = np.min(self.samples @ r[2] + v[2])
        parts.append(np.array([max(.005-min_z, 0)/.005]))
        if box_weight:
            world = self.samples @ R.T + t
            parts.append((self.envelope.residual(world)/.02*np.sqrt(box_weight*self.envelope.confidence/len(world))).ravel())
        return np.concatenate(parts)

    def measure(self, R, t, correspondences=None, view=None):
        if view is None:
            view = self.view(R, t)
        cor = self.correspondences(R, t, view) if correspondences is None else correspondences
        X, hit = cor[:2]
        r, v = camera_pose(R, t, self.C)
        delta = X @ r.T + v-self.q
        depth = np.abs(delta[:, 2])
        outside = np.linalg.norm(self.envelope.residual(self.samples @ R.T + t), axis=1)
        return {'frame_idx': int(self.inputs.ids[self.i]), 'mask': self.domain.score(view[2]),
                'surface_samples': len(X), 'first_hit_coverage': float(hit.mean()) if len(hit) else None,
                'first_hit_depth_mean_m': float(depth[hit].mean()) if hit.any() else None,
                'all_surface_distance_mean_m': float(np.linalg.norm(delta, axis=1).mean()) if len(X) else None,
                'normal_samples': int(np.sum(self.quality > 0)),
                'normal_axis_error_mean': float(np.linalg.norm(np.cross(cor[2] @ r.T, self.normal), axis=1)[self.quality>0].mean()) if np.any(self.quality>0) else None,
                'envelope_observed': self.envelope.observed, 'envelope_confidence': self.envelope.confidence,
                'envelope_outside_mean_m': float(outside.mean()) if self.envelope.observed else None,
                'envelope_inside_fraction': float(np.mean(outside < 1e-8)) if self.envelope.observed else None,
                'fixed_population': True}
