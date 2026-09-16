"""Object-only full-triangle first-hit fitting with frozen observation envelopes.

Port of object_only_fit/solver_kernel, without historical paths or dispatchers.
Generated geometry participates in these candidate poses; it gains no physical
contact, signed-distance, or collision authority. All tensors are explicitly CPU.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
from observations.object.envelope import load_surface_sources
from observations.object.shape import record, write_mesh
from gates.object.envelope import validate_envelopes, check_references
from utils.data import load, indexed, RKEY, TKEY
from utils.geometry import resized_K, torch_rodrigues
from utils.state import sha, in_run

PROFILES = {'surface_no_box': 0., 'surface_weak_box': .15}


@dataclass(frozen=True)
class SurfaceProblem:
    annotations: Path
    refined_posterior: Path
    mesh_binding: Path
    envelope_report: Path
    appearance: Path
    run_root: Path
    object_id: str
    output_dir: Path
    profile: str = 'surface_no_box'
    outer_iterations: int = 12
    inner_iterations: int = 25


def sample_polygon(poly, count):
    poly = np.asarray(poly, float).reshape(-1, 2)
    edge = np.roll(poly, -1, axis=0) - poly
    length = np.linalg.norm(edge, axis=1)
    if len(poly) < 3 or np.any(length <= 1e-10):
        raise ValueError('degenerate silhouette polygon')
    cum = np.r_[0, np.cumsum(length)]
    pos = np.linspace(0, cum[-1], count, endpoint=False)
    ix = np.minimum(np.searchsorted(cum, pos, side='right') - 1, len(poly) - 1)
    alpha = (pos - cum[ix]) / length[ix]
    return poly[ix] + alpha[:, None] * edge[ix], ix, alpha


def project_np(pc, K):
    return pc[..., :2] / pc[..., 2:] * K[:2] + K[2:]


def robust(x):
    return 2 * (torch.sqrt(1 + x * x) - 1)


class TriangleFirstHit:
    """BVH contains EVERY original triangle; inverse similarity preserves ray t=z."""
    def __init__(self, vertices, faces):
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(faces, np.uint32)))

    def cast(self, uv, K, T_world_camera, R, t, scale, pivot):
        uv = np.asarray(uv).reshape(-1, 2)
        if not len(uv):
            return np.empty(0), np.empty((0, 3)), np.empty(0, bool)
        dc = np.c_[(uv - K[2:]) / K[:2], np.ones(len(uv))]
        invR = np.linalg.inv(R)
        origin = (invR @ (T_world_camera[:3, 3] - t) - pivot) / scale + pivot
        direction = (dc @ T_world_camera[:3, :3].T) @ invR.T / scale
        rays = np.c_[np.broadcast_to(origin, direction.shape), direction]
        result = self.scene.cast_rays(self.o3d.core.Tensor(rays.astype(np.float32)), nthreads=1)
        depth = result['t_hit'].numpy().astype(float)
        valid = np.isfinite(depth) & (depth > 0)
        q = np.zeros((len(uv), 3))
        q[valid] = origin + depth[valid, None] * direction[valid]
        return depth, q, valid


def area_samples(vertices, faces, count=1200):
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError('mesh has no nondegenerate surface area')
    rng = np.random.default_rng(7141934)
    picks = rng.choice(len(faces), count, p=areas / areas.sum())
    u, v = np.sqrt(rng.random(count)), rng.random(count)
    return np.einsum('ij,ijk->ik', np.c_[1-u, u*(1-v), u*v], tri[picks])


class PreparedSurface:
    def __init__(self, problem):
        self.problem = problem
        self.source = s = load_surface_sources(annotations=problem.annotations,
            mesh_binding=problem.mesh_binding, refined_posterior=problem.refined_posterior,
            run_root=problem.run_root, object_id=problem.object_id)
        self.envelope_path = in_run(problem.envelope_report, s['root'])
        self.envelopes = validate_envelopes(load(self.envelope_path), s)
        self.refs = copy.deepcopy(s['inputs'])
        self.refs['envelopes'] = record(self.envelope_path)
        appearance_path = in_run(problem.appearance, s['root'])
        appearance = load(appearance_path)
        if (appearance.get('schema') != 'rgb_scene_depth_order_appearance_manifest_v1' or
                appearance.get('object_id') != problem.object_id or
                appearance.get('source_annotations_sha256') != s['inputs']['annotations']['sha256'] or
                appearance.get('metric_owned_masks_consumed') is not False or
                s['binding'].get('appearance_mask_manifest') != record(appearance_path)):
            raise ValueError('appearance binding/metric separation mismatch')
        self.refs['appearance'] = record(appearance_path)
        app = indexed(appearance['frames'])
        if set(app) != set(s['frame_idx']):
            raise ValueError('appearance full timeline required')
        self.V, self.F, self.R0, self.t0 = s['vertices'], s['faces'], s['R0'], s['t0']
        self.T, self.times, self.ids = s['T'], s['times'], s['frame_idx']
        self.N = len(self.ids)
        self.pivot = self.V.mean(axis=0)
        self.box_points = area_samples(self.V, self.F)
        self.scene = TriangleFirstHit(self.V, self.F)
        self.cam_inv = np.linalg.inv(self.T[:, :3, :3])
        self.K, self.Kfit, self.S, self.lo, self.hi = [], [], [], [], []
        self.train_p = np.zeros((self.N, 192, 3))
        self.train_valid = np.zeros((self.N, 192), bool)
        self.eval_p, self.train_ids, self.eval_ids = [], [], []
        self.contours = np.zeros((self.N, 96, 2))
        self.sil_active = np.zeros(self.N, bool)
        self.masks, self.mask_K, self.direct = [], [], []
        rows = indexed(self.envelopes['frames'])
        for i, (idx, frame) in enumerate(zip(self.ids, s['frames'])):
            K = np.asarray(frame['camera']['intrinsics_fx_fy_cx_cy'], float)
            wh = [frame['source_width'], frame['source_height']]
            Kfit = resized_K(K, wh, [960, 960 * wh[1] / wh[0]])
            self.K.append(K); self.Kfit.append(Kfit)
            p = s['points_camera'][i]
            order = np.random.default_rng(62000+i).permutation(len(p))
            train, ev = order[:192], order[192:704]
            self.train_ids.append(train.tolist()); self.eval_ids.append(ev.tolist())
            self.train_p[i, :len(train)] = p[train]
            self.train_valid[i, :len(train)] = True
            self.eval_p.append(p[ev])
            e = rows[idx]
            self.direct.append(e['optimization_weight'])
            self.S.append(np.eye(4) if e['T_world_surface'] is None else e['T_world_surface'])
            self.lo.append(np.zeros(3) if e['lower_surface_m'] is None else e['lower_surface_m'])
            self.hi.append(np.zeros(3) if e['upper_surface_m'] is None else e['upper_surface_m'])
            a = app[idx]
            if a.get('time_s') != frame['time_s'] or a.get('object_id') != problem.object_id:
                raise ValueError('appearance frame identity/time mismatch')
            if a.get('status') != 'available':
                self.masks.append(None); self.mask_K.append(None)
                continue
            mask_path = in_run(a['appearance_mask_path'], s['root'])
            rgb_path = in_run(a['raw_rgb_path'], s['root'])
            if sha(mask_path) != a['appearance_mask_sha256'] or sha(rgb_path) != a['raw_rgb_sha256']:
                raise ValueError('appearance raster source bytes changed')
            self.refs[f'mask_{idx}'] = record(mask_path)
            self.refs[f'rgb_{idx}'] = record(rgb_path)
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape[::-1] != tuple(a['source_size_wh']):
                raise ValueError('appearance mask plane mismatch')
            yy, xx = np.where(m > 0)
            if len(xx) < 3:
                # Weak/empty masks carry no invented contour.
                self.masks.append(None); self.mask_K.append(None)
                continue
            hull = cv2.convexHull(np.c_[xx, yy].astype(np.float32)).reshape(-1, 2)
            if len(hull) < 3:
                self.masks.append(None); self.mask_K.append(None)
                continue
            scale_xy = np.array([960, 960 * wh[1] / wh[0]]) / np.array(a['source_size_wh'])
            hull = (hull + .5) * scale_xy - .5
            self.contours[i] = sample_polygon(hull, 96)[0]
            self.sil_active[i] = True
            self.masks.append(m > 0)
            self.mask_K.append(resized_K(K, wh, a['source_size_wh']))
        self.K, self.Kfit = np.array(self.K), np.array(self.Kfit)
        self.S, self.lo, self.hi = np.array(self.S), np.array(self.lo), np.array(self.hi)
        self.Sinv = np.linalg.inv(self.S[:, :3, :3])
        self.direct = np.asarray(self.direct)
        # No outer iteration can update the observation frame, source pose, or size prior.
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def camera_points(self, q, R, t, scale, i):
        world = (self.pivot + scale * (q - self.pivot)) @ R[i].T + t[i]
        return (world - self.T[i, :3, 3]) @ self.cam_inv[i].T

    def raycast(self, i, uv, K, R, t, scale):
        return self.scene.cast(uv, K, self.T[i], R[i], t[i], scale, self.pivot)

    def correspondences(self, R, t, scale):
        q = np.zeros_like(self.train_p)
        hit_mask = np.zeros_like(self.train_valid)
        target_uv = np.zeros((self.N, 192, 2))
        ia = np.zeros((self.N, 96), int); ib = ia.copy(); reverse = ia.copy()
        fraction = np.zeros((self.N, 96)); target = np.zeros_like(self.contours)
        for i in range(self.N):
            use = self.train_valid[i]
            if use.any():
                uv = project_np(self.train_p[i, use], self.Kfit[i])
                _, hit, valid = self.raycast(i, uv, self.Kfit[i], R, t, scale)
                q[i, use] = hit; hit_mask[i, use] = valid; target_uv[i, use] = uv
            if not self.sil_active[i]:
                continue
            pc = self.camera_points(self.V, R, t, scale, i)
            if np.any(pc[:, 2] <= .03):
                raise ValueError('candidate mesh support crosses camera near plane')
            front = np.arange(len(pc))
            uv = project_np(pc, self.Kfit[i])
            hull = cv2.convexHull(uv.astype(np.float32), returnPoints=False).reshape(-1)
            boundary, edge, alpha = sample_polygon(uv[hull], 96)
            ia[i] = front[hull[edge]]; ib[i] = front[hull[(edge+1) % len(hull)]]
            fraction[i] = alpha
            target[i] = self.contours[i][cKDTree(self.contours[i]).query(boundary)[1]]
            reverse[i] = cKDTree(boundary).query(self.contours[i])[1]
        return dict(hit_points=q, hit_mask=hit_mask, target_uv=target_uv,
                    ia=ia, ib=ib, fraction=fraction, forward_targets=target, reverse=reverse)

    def evaluate(self, R, t, scale):
        rows = []
        for i, idx in enumerate(self.ids):
            p = self.eval_p[i]
            d, _, valid = self.raycast(i, project_np(p, self.K[i]), self.K[i], R, t, scale)
            residual = d[valid] - p[valid, 2]
            row = dict(frame_idx=idx, heldout_ray_count=len(p), hit_count=int(valid.sum()),
                first_hit_depth_mae_m=float(np.mean(abs(residual))) if len(residual) else None,
                full_triangle_silhouette_iou=None)
            mask = self.masks[i]
            if mask is not None:
                h, w = mask.shape; small = (160, max(1, round(160*h/w)))
                yy, xx = np.mgrid[:small[1], :small[0]]
                K = resized_K(self.mask_K[i], [w, h], small)
                _, _, hit = self.raycast(i, np.c_[xx.ravel(), yy.ravel()], K, R, t, scale)
                pred = hit.reshape(small[1], small[0])
                target = cv2.resize(mask.astype('uint8'), small, interpolation=cv2.INTER_NEAREST_EXACT) > 0
                row['full_triangle_silhouette_iou'] = float(np.sum(pred & target) / max(1, np.sum(pred | target)))
            rows.append(row)
        return rows


def temporal_losses(dr, dt, R, t, pivot, times):
    """Timestamp derivatives, duration-weighted; historical sigmas at 30 Hz.

    Correction velocity scales: 1.5 rad/s, .30 m/s. Actual-center/matrix
    acceleration scales: 4.5 m/s² and 27 /s². Uniform 30 Hz reproduces the
    historical finite differences, but arbitrary frame IDs never enter math.
    """
    zero = (dr.sum() + dt.sum()) * 0
    if len(times) < 2:
        return zero, zero
    gaps = times[1:] - times[:-1]
    weights = gaps / gaps.sum()
    vr = (dr[1:] - dr[:-1]) / gaps[:, None] / 1.5
    vt = (dt[1:] - dt[:-1]) / gaps[:, None] / .30
    velocity = ((vr.square().mean(-1) + vt.square().mean(-1)) * weights).sum()
    if len(times) < 3:
        return velocity, zero
    center = torch.einsum('nij,j->ni', R, pivot) + t
    vc = (center[1:] - center[:-1]) / gaps[:, None]
    vm = (R[1:] - R[:-1]) / gaps[:, None, None]
    midgap = (gaps[1:] + gaps[:-1]) / 2
    ac = (vc[1:] - vc[:-1]) / midgap[:, None] / 4.5
    am = (vm[1:] - vm[:-1]) / midgap[:, None, None] / 27.
    acceleration = ((ac.square().mean(-1) + am.square().mean((1, 2))) * (midgap / midgap.sum())).sum()
    return velocity, acceleration


def optimize_surface(p, *, outer=12, inner=25, box_weight=0.):
    """Refresh BVH first-hit and bidirectional silhouette correspondences each outer."""
    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64, device='cpu')
    pivot, Rbase, tbase = to(p.pivot), to(p.R0), to(p.t0)
    camera_t, camera_inv = to(p.T[:, :3, 3]), to(p.cam_inv)
    Sinv, Sorigin, lo, hi = to(p.Sinv), to(p.S[:, :3, 3]), to(p.lo), to(p.hi)
    K, sil, box = to(p.Kfit), to(p.V), to(p.box_points)
    contours, observation, direct, times = to(p.contours), to(p.train_p), to(p.direct), to(p.times)
    active = to(p.sil_active)
    dr = torch.nn.Parameter(torch.zeros((p.N, 3), dtype=torch.float64, device='cpu'))
    dt = torch.nn.Parameter(torch.zeros_like(dr))
    logscale = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device='cpu'))
    optimizer = torch.optim.Adam([{'params': [dr, dt], 'lr': .04}, {'params': [logscale], 'lr': .008}])
    trace = []
    def state():
        return torch_rodrigues(dr * .2) @ Rbase, tbase + dt * .05, torch.exp(logscale)
    def cam(q, R, t, scale):
        q = pivot + scale * (q - pivot)
        return torch.einsum('nij,nkj->nki', camera_inv,
            torch.einsum('nij,nkj->nki', R, q.expand(p.N, -1, -1)) + t[:, None] - camera_t[:, None])
    def project(q):
        return q[..., :2] / q[..., 2:].clamp_min(.01) * K[:, None, :2] + K[:, None, 2:]
    for o in range(outer):
        with torch.no_grad():
            R, t, scale = state()
            assoc = p.correspondences(R.numpy(), t.numpy(), float(scale))
        q, mask, target_uv = (to(assoc[k]) for k in ('hit_points', 'hit_mask', 'target_uv'))
        ia, ib, rev = (torch.tensor(assoc[k], dtype=torch.long) for k in ('ia', 'ib', 'reverse'))
        frac, target = to(assoc['fraction'])[:, :, None], to(assoc['forward_targets'])
        batch = torch.arange(p.N)[:, None]
        optimizer.param_groups[0]['lr'] = .04 / (1 + .18*o)
        optimizer.param_groups[1]['lr'] = .008 / (1 + .18*o)
        for _ in range(inner):
            optimizer.zero_grad()
            R, t, scale = state()
            pc = cam(q, R, t, scale)
            Ldepth = (robust((pc[:, :, 2] - observation[:, :, 2]) / .008) * mask).sum() / mask.sum().clamp_min(1)
            Lray = (robust((project(pc) - target_uv) / 8).mean(-1) * mask).sum() / mask.sum().clamp_min(1)
            # Transform selected original hull vertices only; BVH still uses full topology.
            edge_a = project(cam(sil[ia], R, t, scale))
            edge_b = project(cam(sil[ib], R, t, scale))
            boundary = edge_a * (1-frac) + edge_b * frac
            Lsil = ((robust((boundary-target)/8).mean((1, 2)) +
                    robust((boundary[batch, rev]-contours)/8).mean((1, 2))) * active).sum() / active.sum().clamp_min(1)
            world = torch.einsum('nij,kj->nki', R, pivot + scale*(box-pivot)) + t[:, None]
            qs = torch.einsum('nij,nkj->nki', Sinv, world-Sorigin[:, None])
            violation = torch.relu(qs-hi[:, None]) + torch.relu(lo[:, None]-qs)
            Lbox = (robust(violation/.015).mean((1, 2))*direct).sum() / direct.sum().clamp_min(1)
            Lprior = (dr*.2/.3).square().mean() + (dt*.05/.10).square().mean()
            Lscale = (logscale/.2).square()
            Lvel, Lacc = temporal_losses(dr*.2, dt*.05, R, t, pivot, times)
            loss = Ldepth + .35*Lray + Lsil + box_weight*Lbox + .05*Lprior + .15*Lscale + .30*Lvel + .10*Lacc
            if not torch.isfinite(loss):
                raise ValueError('nonfinite surface objective')
            loss.backward()
            if o < 2:
                dr.grad.zero_()
            torch.nn.utils.clip_grad_norm_([dr, dt, logscale], 20.)
            optimizer.step()
            with torch.no_grad():
                dr.clamp_(-2.5, 2.5); dt.clamp_(-2.4, 2.4); logscale.clamp_(np.log(.8), np.log(1.35))
        trace.append(dict(outer=o, rotation_released=o >= 2, hit_count=int(mask.sum()),
            training_ray_count=int(p.train_valid.sum()),
            **{name: float(value.detach()) for name, value in (
                ('loss', loss), ('depth', Ldepth), ('ray', Lray), ('silhouette', Lsil),
                ('box', Lbox), ('source_pose', Lprior), ('scale_prior', Lscale),
                ('correction_velocity', Lvel), ('actual_acceleration', Lacc))}))
    with torch.no_grad():
        R, t, scale = state()
    return R.numpy(), t.numpy(), float(scale), trace


def fit_surface(problem):
    """Write a new candidate binding/report/scaled full mesh; never modify source."""
    if problem.profile not in PROFILES:
        raise ValueError('unknown named surface profile')
    if min(problem.outer_iterations, problem.inner_iterations) <= 0:
        raise ValueError('positive surface iteration budget required')
    root = Path(problem.run_root).resolve()
    output = in_run(problem.output_dir, root, must_exist=False)
    if output.exists():
        raise ValueError('new surface candidate output directory required')
    started = time.perf_counter()
    p = PreparedSurface(problem)
    check_references(p.refs, root)
    before = p.evaluate(p.R0, p.t0, 1.)
    R, t, scale, trace = optimize_surface(p, outer=problem.outer_iterations,
        inner=problem.inner_iterations, box_weight=PROFILES[problem.profile])
    after = p.evaluate(R, t, scale)
    paired = []
    for i, idx in enumerate(p.ids):
        observed = p.eval_p[i]
        uv = project_np(observed, p.K[i])
        d0, _, hit0 = p.raycast(i, uv, p.K[i], p.R0, p.t0, 1.)
        d1, _, hit1 = p.raycast(i, uv, p.K[i], R, t, scale)
        common = hit0 & hit1
        paired.append(dict(frame_idx=idx, common_hit_count=int(common.sum()),
            before_depth_mae_m=float(np.mean(abs(d0[common]-observed[common, 2]))) if common.any() else None,
            after_depth_mae_m=float(np.mean(abs(d1[common]-observed[common, 2]))) if common.any() else None))
    check_references(p.refs, root)
    output.mkdir(parents=True, exist_ok=False)
    mesh_path = output / 'object_canonical_scaled.ply'
    topology = write_mesh(mesh_path, p.pivot + scale*(p.V-p.pivot), p.F)
    pose_rows = copy.deepcopy(p.source['posterior']['pose_rows'])
    poses = indexed(pose_rows)
    for i, idx in enumerate(p.ids):
        poses[idx][RKEY], poses[idx][TKEY] = R[i].tolist(), t[i].tolist()
        poses[idx]['surface_candidate'] = dict(generated_mesh_used_as_residual_model=True,
            direct_surface_observation=bool(p.direct[i]), physical_contact_authority=False,
            uncertainty='mesh_conditioned_predicted_depth_mask_source_pose_and_occlusion_uncertainty')
    report = dict(schema='object_surface_fit_report_v1', object_id=problem.object_id,
        profile=problem.profile, box_weight=PROFILES[problem.profile], inputs=p.refs,
        before=before, after=after, paired_heldout_depth=paired,
        trace=trace, pose_rows=copy.deepcopy(pose_rows), runtime_s=time.perf_counter()-started,
        scale=scale, pivot_canonical_m=p.pivot.tolist(),
        settings=dict(outer_iterations=problem.outer_iterations, inner_iterations=problem.inner_iterations,
            depth_sigma_m=.008, pixel_sigma_at_960=8., box_sigma_m=.015,
            rotation_component_bound_rad=.5, translation_component_bound_m=.12,
            scale_bounds=[.8, 1.35], rotation_frozen_first_outer_iterations=2,
            loss_weights=dict(depth=1., ray=.35, silhouette=1., box=PROFILES[problem.profile],
                source_pose=.05, scale_prior=.15, correction_velocity=.30, actual_acceleration=.10)),
        train_indices=p.train_ids, heldout_indices=p.eval_ids,
        heldout_is_independent_measurement=False, correspondence_refreshes=len(trace),
        silhouette_training='bidirectional_convex_hull_image_factor_full_vertex_support',
        silhouette_evaluation='full_triangle_BVH_first_hit_160_width',
        temporal='timestamp_derivatives_duration_weighted_historical_sigmas_at_30Hz',
        topology=topology, complete_mesh_topology_preserved=True,
        generated_mesh_used_as_residual_model=True, annotation_ready=False,
        physical_contact_authority=False, collision_authority=False, signed_distance_authority=False,
        uncertainty='shared_predicted_depth_and_size_prior_correlated_not_metric_accuracy_proof')
    report_path = output / 'surface_fit_report.json'
    with report_path.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    # Deliberately do not deepcopy the posterior top level: its non-use-of-generated
    # geometry claims apply only to source_refined_pose_rows, never this candidate.
    binding = dict(schema='object_surface_candidate_binding_v1', object_id=problem.object_id,
        profile=problem.profile, source_binding=record(problem.mesh_binding),
        refined_posterior=record(problem.refined_posterior),
        appearance_mask_manifest=record(problem.appearance),
        source_prior=copy.deepcopy(p.source['posterior']['source_prior']),
        source_pose_rows=copy.deepcopy(p.source['posterior']['source_pose_rows']),
        source_refined_pose_rows=copy.deepcopy(p.source['posterior']['pose_rows']),
        source_visual_mesh_receipt=copy.deepcopy(p.source['binding']['visual_mesh_receipt']),
        envelope_report=record(problem.envelope_report), surface_fit_report=record(report_path),
        pose_rows=pose_rows, shared_uniform_scale=scale, scale_pivot_canonical_m=p.pivot.tolist(),
        generated_mesh_used_as_residual_model=True, annotation_ready=False, physical_authority=False,
        physical_contact_authority=False, collision_authority=False, signed_distance_authority=False,
        visual_mesh_receipt=dict(output_mesh_path=str(mesh_path), output_mesh_sha256=sha(mesh_path),
            topology=topology, source_mesh=p.refs['mesh'], shared_uniform_scale=scale,
            pivot_canonical_m=p.pivot.tolist(), generated_mesh_used_as_residual_model=True,
            annotation_ready=False, physical_authority=False,
            **{f'generated_faces_{role}_eligible': False for role in ('contact', 'collision', 'sign', 'sdf', 'nonpenetration')}),
        render_mesh_contract=dict(required_mesh_path=str(mesh_path), required_mesh_sha256=sha(mesh_path), renderer_must_match=True))
    for key in ('case', 'track_id', 'canonical_gauge'):
        if key in p.source['binding']:
            binding[key] = copy.deepcopy(p.source['binding'][key])
    from gates.object.surface_candidate import validate_surface_candidate
    validate_surface_candidate(binding, object_id=problem.object_id, run_root=root)
    with (output / 'pose_mesh.json').open('x') as handle:
        json.dump(binding, handle, indent=2, allow_nan=False)
    check_references(p.refs, root)
    return binding
