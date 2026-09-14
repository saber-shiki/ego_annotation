"""Timestamped global SO(3) followed by refreshed observed/RGB full six-DoF fitting.

Port of the published repair's relative rotation kernel and late-window residuals.
Priors are immutable; retractions are bounded per step, never in accumulated angle.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from observations.object.rgb_tracking import local_rotation_evidence, observed_rotation_authority
from utils.data import PoseNode, RGBEdge, RGBReprojectionFactor, SolverConfig
from utils.geometry import valid_rotation, apply_left_correction, pose_correction, inverse_pose
from factors.object.pose import rgb_edge_residual, source_pose_rotation, source_pose_translation
from factors.object.image import rgb_reprojection_residual, silhouette_endpoint_residual
from factors.object.geometry import first_hit_depth, observed_point_to_plane, observed_point_to_point
from factors.motion.temporal import correction_velocity, correction_acceleration
from factors.interaction.physics import nonpenetration_translation_target
from optimization.common.correspondences import build_observed_point_factors



@dataclass
class GraphConfig:
    outer_iterations: int = 8
    inner_evaluations: int = 80
    rotation_step_rad: float = .4
    translation_step_m: float = .05
    prior_rotation_rad: float = .6
    prior_translation_m: float = .08
    rgb_rotation_rad: float = .05
    rgb_translation_m: float = .025
    rotation_weight: float = 1.
    translation_weight: float = 1.
    observed_weight: float = 1.
    reprojection_weight: float = 1.
    depth_restoration_m: float = .01
    near_depth_m: float = 1e-6
    # Continuous-time integral penalties; units rad/s and m/s.
    correction_velocity_rad_s: float = 3.
    correction_velocity_m_s: float = .5
    acceleration_rad_s2: float = 8.
    acceleration_m_s2: float = 2.
    loss: str = 'soft_l1'


def validate(nodes, times, edges, anchor):
    ids = [n.frame_idx for n in nodes]
    if len(set(ids)) != len(ids) or not ids or anchor not in ids:
        raise ValueError('unique full timeline and one gauge anchor required')
    times = np.asarray(times, float)
    if times.shape != (len(nodes),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('strictly increasing finite timestamps required')
    for n in nodes:
        if not valid_rotation(n.initial_rotation) or not np.isfinite(n.initial_translation).all() or np.shape(n.initial_translation) != (3,):
            raise ValueError('malformed source/initial pose')
        if not 0 <= n.uncertainty <= 1:
            raise ValueError('invalid uncertainty')
    for e in edges:
        if e.source_frame_idx not in ids or e.target_frame_idx not in ids or e.source_frame_idx == e.target_frame_idx:
            raise ValueError('edge identity outside timeline')
        if not valid_rotation(e.rotation_source_to_target_world) or np.shape(e.translation_source_to_target_world_m) != (3,) or not np.isfinite(e.translation_source_to_target_world_m).all() or not np.isfinite(e.weight) or e.weight < 0:
            raise ValueError('malformed relative pose/precision')
    return times


def reprojection_depths(factor, poses, nodes):
    """Same endpoint/rotation eligibility as the pixel factor, without clipping."""
    rs, ts = poses[factor.source_frame_idx]
    rt, tt = poses[factor.target_frame_idx]
    if not factor.rotation_eligible:
        rs = nodes[factor.source_pos].initial_rotation
        rt = nodes[factor.target_pos].initial_rotation
    if not np.isfinite(factor.source_world).all():
        raise ValueError('nonfinite active reprojection geometry')
    local = inverse_pose(factor.source_world, rs, ts)
    camera = inverse_pose(local @ rt.T + tt, factor.target_camera_rotation, factor.target_camera_translation)
    if not np.isfinite(camera).all():
        raise ValueError('nonfinite active reprojection geometry')
    return camera[:, 2]


def depth_state(poses, nodes, reprojections, cfg):
    depths, weights = [], []
    for f in reprojections:
        w = cfg.reprojection_weight * f.edge_weight * np.clip(f.weights, 0, 1)
        active = w > 0
        if not np.any(active):
            continue
        # Zero-weight samples have no feasibility authority.
        selected = replace(f, source_world=f.source_world[active])
        depths.extend(reprojection_depths(selected, poses, nodes))
        weights.extend(w[active] / max(1, len(w)))
    return np.asarray(depths), np.asarray(weights)


def depth_violation(depths, weights, cfg):
    return float(np.sum(weights * np.maximum(cfg.near_depth_m - depths, 0)))


def conditioned_observed_pose(node, poses, reference=None, scales=None):
    r, t = poses[node.frame_idx]
    if reference is not None:
        base = reference[node.frame_idx][0]
        delta = Rotation.from_matrix(r @ base.T).as_rotvec()
        r = Rotation.from_rotvec(scales[node.frame_idx] * delta).as_matrix() @ base
    return r, t


def residual_blocks(poses, nodes, times, edges, points, reprojections, cfg, rotation_only=False,
                    observed_reference=None, observed_scales=None):
    """Each block names its incident positions for exact finite-difference sparsity."""
    positions = {n.frame_idx: i for i, n in enumerate(nodes)}
    corrections = [pose_correction(n.initial_rotation, n.initial_translation, *poses[n.frame_idx]) for n in nodes]
    blocks = []
    def add(deps, value):
        blocks.append((tuple(deps), np.asarray(value).reshape(-1)))
    for i, (n, (dr, dt)) in enumerate(zip(nodes, corrections)):
        # Missing metric/source support is a weak prior, not a fabricated measurement.
        scale = 1 + 9 * n.uncertainty
        add([i], source_pose_rotation(dr, cfg.prior_rotation_rad * scale))
        if not rotation_only:
            add([i], source_pose_translation(dt, cfg.prior_translation_m * scale))
    for e in edges:
        dr, dt = rgb_edge_residual(e, poses)
        deps = [positions[e.source_frame_idx], positions[e.target_frame_idx]]
        add(deps, math.sqrt(e.weight * cfg.rotation_weight) * dr / cfg.rgb_rotation_rad)
        if not rotation_only:
            add(deps, math.sqrt(e.weight * cfg.translation_weight) * dt / cfg.rgb_translation_m)
    if not rotation_only:
        kernel = SolverConfig(rgb_reprojection_weight_scale=cfg.reprojection_weight)
        for f in reprojections:
            if cfg.reprojection_weight * f.edge_weight <= 0 or not np.any(np.asarray(f.weights) > 0):
                continue
            active = np.asarray(f.weights) > 0
            f = replace(f, source_world=f.source_world[active], target_uv=f.target_uv[active], weights=f.weights[active])
            add([f.source_pos, f.target_pos], rgb_reprojection_residual(f, poses, nodes, kernel))
            depths = reprojection_depths(f, poses, nodes)
            weights = np.sqrt(cfg.reprojection_weight * f.edge_weight * np.clip(f.weights, 0, 1) / len(depths))
            add([f.source_pos, f.target_pos], weights * np.maximum(cfg.near_depth_m - depths, 0) / cfg.depth_restoration_m)
        for f in points:
            a, b = nodes[f.source_pos], nodes[f.target_pos]
            delta = (inverse_pose(f.source_world, *conditioned_observed_pose(a, poses, observed_reference, observed_scales))
                     - inverse_pose(f.target_world, *conditioned_observed_pose(b, poses, observed_reference, observed_scales)))
            add([f.source_pos, f.target_pos], observed_point_to_plane(delta,f.target_normals,f.weights,cfg.observed_weight))
            add([f.source_pos, f.target_pos], observed_point_to_point(delta,f.weights,cfg.observed_weight))
    vr, vt = [], []
    gaps = np.diff(times)
    for i, gap in enumerate(gaps):
        ra, ta = poses[nodes[i].frame_idx]; rb, tb = poses[nodes[i + 1].frame_idx]
        vr.append(Rotation.from_matrix(rb @ ra.T).as_rotvec() / gap)
        vt.append((tb - ta) / gap)
        ca = ra @ nodes[i].initial_rotation.T
        cb = rb @ nodes[i + 1].initial_rotation.T
        dr = Rotation.from_matrix(cb @ ca.T).as_rotvec() / gap
        dt = (corrections[i + 1][1] - corrections[i][1]) / gap
        add([i, i + 1], math.sqrt(gap) * dr / cfg.correction_velocity_rad_s)
        if not rotation_only:
            add([i, i + 1], math.sqrt(gap) * dt / cfg.correction_velocity_m_s)
    for i in range(1, len(gaps)):
        gap = (gaps[i] + gaps[i - 1]) / 2
        add([i - 1, i, i + 1], (vr[i] - vr[i - 1]) / math.sqrt(gap) / cfg.acceleration_rad_s2)
        if not rotation_only:
            add([i - 1, i, i + 1], (vt[i] - vt[i - 1]) / math.sqrt(gap) / cfg.acceleration_m_s2)
    return blocks


def robust_cost(residual, loss):
    return float(np.sum(2 * (np.sqrt(1 + residual**2) - 1) if loss == 'soft_l1' else residual**2))


def solve_scene_pose(nodes, times, edges, anchor, config=None, reprojections=()):
    cfg = config or GraphConfig()
    times = validate(nodes, times, edges, anchor)
    if cfg.loss not in {'soft_l1', 'linear'} or min(cfg.rotation_step_rad, cfg.translation_step_m, cfg.depth_restoration_m, cfg.near_depth_m) <= 0:
        raise ValueError('invalid graph configuration')
    # Work on distinct seed nodes: original arrays/rows remain immutable. Rotation
    # priors and correction velocity now refer to current RGB evidence, not ICP.
    nodes, edges, reprojections, chain = local_rotation_evidence(nodes, times, edges, anchor, reprojections)
    observed_scales = observed_rotation_authority(nodes, chain)
    poses = {n.frame_idx: (n.initial_rotation.copy(), n.initial_translation.copy()) for n in nodes}
    # Eliminating this variable fixes the sole gauge exactly, without a huge numerical weight.
    free = [i for i, n in enumerate(nodes) if n.frame_idx != anchor]
    trace = []
    snapshots = {}
    for rotation_only in (True, False):
        observed_reference = {i: (r.copy(), t.copy()) for i, (r, t) in poses.items()}
        dof = 3 if rotation_only else 6
        for outer in range(cfg.outer_iterations):
            kernel = SolverConfig(max_points=250, min_pairs=3)
            observed_poses = {n.frame_idx: conditioned_observed_pose(n, poses, observed_reference, observed_scales) for n in nodes}
            points, info = ([], {}) if rotation_only else build_observed_point_factors(nodes, observed_poses, anchor, kernel, outer)
            base = poses
            def unpack(x):
                result = dict(base)
                for j, i in enumerate(free):
                    delta = x[j * dof:(j + 1) * dof]
                    result[nodes[i].frame_idx] = apply_left_correction(*base[nodes[i].frame_idx], delta[:3], np.zeros(3) if rotation_only else delta[3:])
                return result
            def blocks(x):
                return residual_blocks(unpack(x), nodes, times, edges, points, reprojections, cfg, rotation_only,
                                       observed_reference, observed_scales)
            def fun(x):
                return np.concatenate([v for _, v in blocks(x)])
            x0 = np.zeros(len(free) * dof)
            if not len(x0):
                break
            initial_depths, depth_weights = (np.array([]), np.array([])) if rotation_only else depth_state(base, nodes, reprojections, cfg)
            violation_before = depth_violation(initial_depths, depth_weights, cfg)
            initial_blocks = blocks(x0)
            sparsity = lil_matrix((sum(len(v) for _, v in initial_blocks), len(x0)), dtype=int)
            offset = 0
            for deps, value in initial_blocks:
                for j, i in enumerate(free):
                    if i in deps:
                        sparsity[offset:offset + len(value), j*dof:(j+1)*dof] = 1
                offset += len(value)
            bounds = np.tile([cfg.rotation_step_rad] * 3 + ([] if rotation_only else [cfg.translation_step_m] * 3), len(free))
            fit = least_squares(fun, x0, bounds=(-bounds, bounds), jac_sparsity=sparsity.tocsr(),
                                loss=cfg.loss, f_scale=1, x_scale='jac', max_nfev=cfg.inner_evaluations,
                                ftol=1e-6, xtol=1e-6, gtol=1e-6)
            before = robust_cost(fun(x0), cfg.loss)
            step = fit.x
            if violation_before > 0:
                # A separate feasibility proposal avoids the singular pixel
                # objective's local minimum behind the camera. Pixels/edges and
                # observed terms remain in the full cost; acceptance explicitly
                # records restoration rather than claiming objective descent.
                def restoration(x):
                    depths, weights = depth_state(unpack(x), nodes, reprojections, cfg)
                    return np.sqrt(weights) * np.maximum(cfg.near_depth_m - depths, 0) / cfg.depth_restoration_m
                depth_sparsity = lil_matrix((len(initial_depths),len(x0)),dtype=int)
                offset = 0
                for f in reprojections:
                    count = int(np.sum(cfg.reprojection_weight * f.edge_weight * np.clip(f.weights,0,1) > 0))
                    for j, i in enumerate(free):
                        if i in (f.source_pos,f.target_pos):
                            depth_sparsity[offset:offset+count,j*dof:(j+1)*dof] = 1
                    offset += count
                restore = least_squares(restoration, x0, bounds=(-bounds, bounds),
                                        jac_sparsity=depth_sparsity.tocsr(),
                                        max_nfev=cfg.inner_evaluations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                trial_depths, _ = depth_state(unpack(step), nodes, reprojections, cfg)
                restored_depths, _ = depth_state(unpack(restore.x), nodes, reprojections, cfg)
                if depth_violation(restored_depths, depth_weights, cfg) < depth_violation(trial_depths, depth_weights, cfg):
                    step = restore.x
            accepted = False
            after = before
            for _ in range(10):
                candidate = unpack(step)
                candidate_depths, _ = (np.array([]), np.array([])) if rotation_only else depth_state(candidate, nodes, reprojections, cfg)
                violation_after = depth_violation(candidate_depths, depth_weights, cfg)
                feasible = initial_depths >= cfg.near_depth_m
                preserve_feasible = bool(np.all(candidate_depths[feasible] >= cfg.near_depth_m))
                after = robust_cost(fun(step), cfg.loss)
                # Restoration need not cross the near plane in one bounded step.
                # Once feasible, normal robust objective descent resumes.
                restoration_descent = violation_after < violation_before - 1e-12
                # An irreducible violation must not freeze independent supported
                # updates either: allow objective descent without depth regression.
                descent = restoration_descent or (violation_after <= violation_before and after <= before + 1e-9)
                accepted = preserve_feasible and np.isfinite(step).all() and np.isfinite(after) and descent
                if accepted:
                    poses = candidate
                    break
                step = step * .5
            trace.append({'stage': 'SO3' if rotation_only else 'SE3', 'outer': outer,
                          'cost_before': before, 'cost_after': after, 'accepted': bool(accepted),
                          'max_rotation_step_component_rad': float(np.max(np.abs(step.reshape(-1,dof)[:,:3]))) if accepted else 0.,
                          'max_translation_step_component_m': float(np.max(np.abs(step.reshape(-1,dof)[:,3:]))) if accepted and not rotation_only else 0.,
                          'evaluations': fit.nfev, 'restoration_evaluations': restore.nfev if violation_before > 0 else 0, 'observed': info,
                          'depth_violation_before_m': violation_before,
                          'depth_violation_after_m': violation_after if accepted else violation_before,
                          'remaining_invalid_depths': int(np.sum((candidate_depths if accepted else initial_depths) < cfg.near_depth_m))})
            if not accepted or np.max(np.abs(step)) < 1e-5:
                break
        snapshots['SO3' if rotation_only else 'SE3'] = [
            {'frame_idx': n.frame_idx, 'rotation': poses[n.frame_idx][0].tolist(),
             'translation_world_m': poses[n.frame_idx][1].tolist()} for n in nodes]
    diagnostics = []
    for e in edges:
        dr, dt = rgb_edge_residual(e, poses)
        normalized = math.sqrt(e.weight * cfg.rotation_weight) * dr / cfg.rgb_rotation_rad
        diagnostics.append({'source': e.source_frame_idx, 'target': e.target_frame_idx,
                            'rotation_error_rad': float(np.linalg.norm(dr)), 'translation_error_m': float(np.linalg.norm(dt)),
                            'robust_rotation_precision': (1 / np.sqrt(1 + normalized**2)).tolist()})
    # Connectivity concerns evidence only; temporal terms do not invent measured loops.
    connected = {anchor}
    for _ in nodes:
        for e in edges:
            if e.weight > 0 and ({e.source_frame_idx, e.target_frame_idx} & connected):
                connected.update([e.source_frame_idx, e.target_frame_idx])
    invalid_ids = set()
    for f in reprojections:
        active = cfg.reprojection_weight * f.edge_weight * np.clip(f.weights, 0, 1) > 0
        if np.any(active) and np.any(reprojection_depths(replace(f, source_world=f.source_world[active]), poses, nodes) < cfg.near_depth_m):
            invalid_ids.update((f.source_frame_idx, f.target_frame_idx))
    return poses, {'trace': trace, 'edges': diagnostics, 'gauge_anchor': anchor,
                   'local_RGB_rotation': chain, 'stage_snapshots': snapshots,
                   'observed_rotation_authority': {'policy': 'covariance_min_max_ratio_over_0.02_on_local_RGB_supported_nodes; translation_unchanged',
                                                   'scales_by_frame': observed_scales},
                   'depth_restoration': {'affected_frame_ids': sorted(invalid_ids), 'remaining_invalid': int(np.sum(depth_state(poses, nodes, reprojections, cfg)[0] < cfg.near_depth_m)),
                                         'policy': 'weighted restoration; retain uncertainty when infeasible'},
                   'relative_evidence_disconnected_ids': [n.frame_idx for n in nodes if n.frame_idx not in connected]}

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
METHOD = 'observed_surface_temporal_chain_unary_rgb_bridge_v2'

def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_mesh(path: Path) -> trimesh.Trimesh:
    import trimesh
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f'no mesh geometry in {path}')
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f'not mesh: {path}')
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f'empty mesh: {path}')
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)

def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    import trimesh
    rng = np.random.default_rng(1801)
    (points, _) = trimesh.sample.sample_surface(mesh, min(count, max(1, len(mesh.faces) * 2)), seed=rng)
    return np.asarray(points, dtype=float)

def rigid_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(src) != len(dst) or len(src) < 3:
        raise RuntimeError('rigid fit requires matched arrays with >=3 points')
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    xd = dst - mu_dst
    covariance = xs.T @ xd / len(src)
    (u, _, vt) = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = mu_dst - rotation @ mu_src
    return (rotation.astype(float), translation.astype(float))

def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation

def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))

def numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {'count': 0, 'min': None, 'median': None, 'p90': None, 'p95': None, 'mean': None, 'max': None}
    return {'count': int(len(array)), 'min': float(np.min(array)), 'median': float(np.median(array)), 'p90': float(np.percentile(array, 90)), 'p95': float(np.percentile(array, 95)), 'mean': float(np.mean(array)), 'max': float(np.max(array))}

def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    if len(query) == 0 or len(target) == 0:
        return {'count': int(len(query)), 'median_m': None, 'p90_m': None, 'p95_m': None, 'mean_m': None, 'max_m': None}
    (distances, _) = cKDTree(target).query(query, k=1, workers=-1)
    return {'count': int(len(query)), 'median_m': float(np.median(distances)), 'p90_m': float(np.percentile(distances, 90)), 'p95_m': float(np.percentile(distances, 95)), 'mean_m': float(np.mean(distances)), 'max_m': float(np.max(distances))}

def rotation_observability_summary(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        return {'observable': False, 'score': 0.0, 'eigenvalues_ascending': [], 'reason': 'too_few_points'}
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    score = float(eigenvalues[1] / max(eigenvalues[2], 1e-12))
    return {'observable': bool(np.isfinite(score) and score >= 0.02), 'score': score, 'eigenvalues_ascending': eigenvalues.astype(float).tolist(), 'threshold': 0.02, 'reason': 'second_to_first_principal_variance_ratio'}

def rigid_pose_observation_eligibility(obj: dict[str, Any], geom: dict[str, Any]) -> tuple[bool | None, list[str], list[str]]:
    """Resolve explicit P09 eligibility without rejecting legacy rows lacking the field."""
    values: list[bool] = []
    reasons: list[str] = []
    sources: list[str] = []
    for (source, payload) in (('object', obj), ('visible_geometry_candidate', geom)):
        value = payload.get('rigid_pose_observation_eligible')
        if isinstance(value, bool):
            values.append(value)
            sources.append(source)
        reason = payload.get('rigid_pose_observation_reason')
        if isinstance(reason, str) and reason and (reason not in reasons):
            reasons.append(reason)
    if False in values:
        return (False, reasons, sources)
    if True in values:
        return (True, reasons, sources)
    return (None, reasons, sources)

def correspondence_state(source: np.ndarray, target: np.ndarray, rotation: np.ndarray, translation: np.ndarray, *, frame_gap: int, trim_fraction: float, max_correspondence_m: float, min_matches: int) -> dict[str, Any]:
    transformed = apply_pose(source, rotation, translation)
    target_tree = cKDTree(target)
    (distances, target_indices) = target_tree.query(transformed, k=1, workers=-1)
    source_tree = cKDTree(transformed)
    reverse_indices = source_tree.query(target, k=1, workers=-1)[1]
    source_indices = np.arange(len(source), dtype=np.int64)
    mutual = source_indices == reverse_indices[target_indices]
    pool = np.flatnonzero(mutual)
    if len(pool) < min_matches:
        pool = source_indices
    quantile_cap = float(np.percentile(distances[pool], trim_fraction * 100.0))
    metric_cap = float(max_correspondence_m * math.sqrt(max(1, frame_gap)))
    active_cap = min(metric_cap, quantile_cap)
    keep = pool[distances[pool] <= active_cap]
    if len(keep) < min_matches:
        keep = np.argsort(distances)[:min(len(distances), max(min_matches, int(len(source) * 0.35)))]
    return {'transformed': transformed, 'distances': distances, 'target_indices': target_indices, 'mutual_count': int(np.count_nonzero(mutual)), 'keep': np.asarray(keep, dtype=np.int64), 'active_cap_m': float(active_cap), 'metric_cap_m': metric_cap, 'quantile_cap_m': quantile_cap}

def register_adjacent_observed_surfaces(source: np.ndarray, target: np.ndarray, *, frame_gap: int, iterations: int, trim_fraction: float, max_correspondence_m: float, min_matches: int, max_median_residual_m: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate a world-to-world rigid delta using only accepted observed surfels."""
    rotation = np.eye(3, dtype=float)
    translation = target.mean(axis=0) - source.mean(axis=0)
    trace: list[dict[str, Any]] = []
    for iteration in range(iterations):
        state = correspondence_state(source, target, rotation, translation, frame_gap=frame_gap, trim_fraction=trim_fraction, max_correspondence_m=max_correspondence_m, min_matches=min_matches)
        keep = state['keep']
        (rotation, translation) = rigid_umeyama(source[keep], target[state['target_indices'][keep]])
        trace.append({'iteration': int(iteration + 1), 'match_count': int(len(keep)), 'mutual_count': int(state['mutual_count']), 'active_cap_m': float(state['active_cap_m']), 'median_residual_m': float(np.median(state['distances'][keep]))})
    final = correspondence_state(source, target, rotation, translation, frame_gap=frame_gap, trim_fraction=trim_fraction, max_correspondence_m=max_correspondence_m, min_matches=min_matches)
    keep = final['keep']
    residuals = final['distances'][keep]
    median_residual = float(np.median(residuals))
    quality_passed = bool(len(keep) >= min_matches and np.isfinite(median_residual) and (median_residual <= max_median_residual_m) and np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation)) and (abs(np.linalg.det(rotation) - 1.0) <= 1e-05))
    report = {'method': 'trimmed_mutual_nearest_observed_surfel_icp', 'frame_gap': int(frame_gap), 'source_point_count': int(len(source)), 'target_point_count': int(len(target)), 'match_count': int(len(keep)), 'mutual_count': int(final['mutual_count']), 'trim_fraction': float(trim_fraction), 'active_cap_m': float(final['active_cap_m']), 'max_correspondence_m_at_gap': float(final['metric_cap_m']), 'residual_m': numeric_summary(residuals), 'delta_rotation_deg': rotation_angle_deg(rotation), 'delta_translation_m': float(np.linalg.norm(translation)), 'quality_thresholds': {'minimum_matches': int(min_matches), 'maximum_median_residual_m': float(max_median_residual_m)}, 'quality_passed': quality_passed, 'trace': trace}
    if not quality_passed:
        raise RuntimeError(f'observed-surface pairwise registration failed closed: {report}')
    return (rotation, translation, report)

def refine_translation_from_observed_anchor(anchor_canonical_points: np.ndarray, observed_world: np.ndarray, rotation: np.ndarray, initial_translation: np.ndarray, *, iterations: int, trim_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine translation at fixed rotation using no generated mesh faces."""
    translation = np.asarray(initial_translation, dtype=float).copy()
    trace: list[dict[str, Any]] = []
    for iteration in range(iterations):
        model_world = apply_pose(anchor_canonical_points, rotation, translation)
        (distances, model_indices) = cKDTree(model_world).query(observed_world, k=1, workers=-1)
        keep_count = max(30, int(len(observed_world) * trim_fraction))
        keep = np.argsort(distances)[:min(len(distances), keep_count)]
        deltas = observed_world[keep] - model_world[model_indices[keep]]
        update = np.median(deltas, axis=0)
        translation += update
        trace.append({'iteration': int(iteration + 1), 'match_count': int(len(keep)), 'median_residual_m': float(np.median(distances[keep])), 'translation_update_m': update.astype(float).tolist()})
    model_world = apply_pose(anchor_canonical_points, rotation, translation)
    final = nearest_summary(observed_world, model_world)
    return (translation, {'method': 'fixed_rotation_trimmed_anchor_observed_surfel_translation', 'trim_fraction': float(trim_fraction), 'iterations': int(iterations), 'observed_to_anchor_surface_final': final, 'trace': trace})

def refine_pose_from_observed_anchor(anchor_canonical_points: np.ndarray, observed_world: np.ndarray, initial_rotation: np.ndarray, initial_translation: np.ndarray, *, iterations: int, trim_fraction: float, damping: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Obtain an observed-only unary pose candidate in the temporal chain's local basin."""
    rotation = np.asarray(initial_rotation, dtype=float).copy()
    translation = np.asarray(initial_translation, dtype=float).copy()
    trace: list[dict[str, Any]] = []
    for iteration in range(iterations):
        model_world = apply_pose(anchor_canonical_points, rotation, translation)
        (distances, model_indices) = cKDTree(model_world).query(observed_world, k=1, workers=-1)
        keep_count = max(30, int(len(observed_world) * trim_fraction))
        keep = np.argsort(distances)[:min(len(distances), keep_count)]
        (candidate_rotation, candidate_translation) = rigid_umeyama(anchor_canonical_points[model_indices[keep]], observed_world[keep])
        rotation_delta = Rotation.from_matrix(candidate_rotation @ rotation.T).as_rotvec()
        damped_delta = Rotation.from_rotvec(float(damping) * rotation_delta).as_matrix()
        rotation = damped_delta @ rotation
        translation = (1.0 - float(damping)) * translation + float(damping) * candidate_translation
        trace.append({'iteration': int(iteration + 1), 'match_count': int(len(keep)), 'median_residual_m': float(np.median(distances[keep])), 'undamped_rotation_update_deg': float(np.degrees(np.linalg.norm(rotation_delta)))})
    final_world = apply_pose(anchor_canonical_points, rotation, translation)
    return (rotation, translation, {'method': 'temporally_initialized_trimmed_observed_anchor_surface_icp', 'iterations': int(iterations), 'trim_fraction': float(trim_fraction), 'damping': float(damping), 'observed_to_anchor_surface_final': nearest_summary(observed_world, final_world), 'trace': trace})

def frame_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for candidate in frame.get('objects', []) if isinstance(frame.get('objects'), list) else []:
        if isinstance(candidate, dict) and candidate.get('object_id') == object_id:
            return candidate
    return None

def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise RuntimeError(f'failed to read grayscale frame: {path}')
    return image

def read_mask_at_size(path: str | Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2:
        raise RuntimeError(f'failed to read object-owned mask: {path}')
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    return mask > 0

def frame_camera_for_raster(frame: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
    transform = np.asarray(camera.get('T_world_camera_metric') or [], dtype=float)
    values = np.asarray(camera.get('intrinsics_fx_fy_cx_cy') or [], dtype=float)
    source_width = int(frame.get('source_width') or width)
    source_height = int(frame.get('source_height') or height)
    if transform.shape != (4, 4) or values.shape != (4,):
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks calibrated camera state for optical-flow PnP")
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid source raster size")
    sx = float(width) / float(source_width)
    sy = float(height) / float(source_height)
    fx = float(values[0] * sx)
    fy = float(values[1] * sy)
    cx = float((values[2] + 0.5) * sx - 0.5)
    cy = float((values[3] + 0.5) * sy - 0.5)
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    return (transform, intrinsics, {'source_intrinsics_fx_fy_cx_cy': values.astype(float).tolist(), 'source_size_wh': [source_width, source_height], 'raster_size_wh': [int(width), int(height)], 'A_raster_from_source': [[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]], 'raster_intrinsics': intrinsics.astype(float).tolist(), 'pixel_center_convention': 'OpenCV half-pixel resize mapping'})

def select_flow_seed_points(frame: dict[str, Any], obj: dict[str, Any], observed_world: np.ndarray, source_rotation: np.ndarray, source_translation: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    raw_path = Path(str(frame.get('raw_frame_path') or ''))
    mask_path = Path(str(obj.get('mask_path') or ''))
    if not raw_path.is_file() or not mask_path.is_file():
        raise RuntimeError(f"optical-flow source frame lacks raw image/object-owned mask: {frame.get('frame_idx')}")
    gray = read_gray(raw_path)
    (height, width) = gray.shape
    mask = read_mask_at_size(mask_path, width, height)
    (world_camera, intrinsics, raster_contract) = frame_camera_for_raster(frame, width, height)
    camera_world = np.linalg.inv(world_camera)
    observed_camera = apply_pose(observed_world, camera_world[:3, :3], camera_world[:3, 3])
    positive = observed_camera[:, 2] > 1e-05
    uv = np.column_stack([intrinsics[0, 0] * observed_camera[:, 0] / np.maximum(observed_camera[:, 2], 1e-09) + intrinsics[0, 2], intrinsics[1, 1] * observed_camera[:, 1] / np.maximum(observed_camera[:, 2], 1e-09) + intrinsics[1, 2]])
    inside = positive & (uv[:, 0] >= 2.0) & (uv[:, 0] < width - 2.0) & (uv[:, 1] >= 2.0) & (uv[:, 1] < height - 2.0)
    canonical = (observed_world - source_translation) @ source_rotation
    uv = uv[inside]
    canonical = canonical[inside]
    rounded = np.rint(uv).astype(np.int64)
    owned = mask[rounded[:, 1], rounded[:, 0]]
    uv = uv[owned]
    canonical = canonical[owned]
    if len(uv) < 50:
        raise RuntimeError(f'too few projected object-owned flow seeds: {len(uv)}')
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient = np.hypot(gradient_x, gradient_y)
    rounded = np.rint(uv).astype(np.int64)
    scores = gradient[rounded[:, 1], rounded[:, 0]]
    order = np.argsort(scores)[::-1]
    selected = order[:min(max_points, len(order))]
    return (gray, uv[selected].astype(np.float32), canonical[selected].astype(float), {'raw_frame': str(raw_path), 'object_owned_mask': str(mask_path), 'projected_owned_seed_count': int(len(uv)), 'selected_seed_count': int(len(selected)), 'selected_gradient_summary': numeric_summary(scores[selected]), 'camera_raster_contract': raster_contract})

def estimate_optical_flow_pnp_bridge(*, source_idx: int, target_idx: int, frames: dict[int, dict[str, Any]], objects: dict[int, dict[str, Any]], observed_world: np.ndarray, source_rotation: np.ndarray, source_translation: np.ndarray, max_seed_points: int, min_tracked_points: int, min_pnp_inliers: int, min_inlier_fraction: float, max_reprojection_median_px: float, max_reprojection_p95_px: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    if source_idx == target_idx:
        raise RuntimeError('optical-flow bridge source and target must differ')
    direction = 1 if target_idx > source_idx else -1
    source_frame = frames[source_idx]
    source_obj = objects[source_idx]
    (previous_gray, source_uv, canonical_points, seed_report) = select_flow_seed_points(source_frame, source_obj, observed_world, source_rotation, source_translation, max_points=max_seed_points)
    previous_uv = source_uv.reshape(-1, 1, 2)
    active_canonical = canonical_points
    flow_trace: list[dict[str, Any]] = []
    current_uv = source_uv
    for frame_idx in range(source_idx + direction, target_idx + direction, direction):
        frame = frames.get(frame_idx)
        obj = objects.get(frame_idx)
        if frame is None or obj is None:
            raise RuntimeError(f'optical-flow bridge lacks frame/object row {frame_idx}')
        appearance = obj.get('appearance_observation')
        if isinstance(appearance, dict):
            if appearance.get('rgb_tracking_pose_evidence_eligible') is not True or appearance.get('metric_depth_pose_eligible') is not False or (not isinstance(appearance.get('ownership_contract'), dict)) or (appearance['ownership_contract'].get('mano_subtraction_completed') is not True) or (appearance['ownership_contract'].get('bbox_subtraction_used') is not False) or (str(obj.get('mask_semantics') or '') != 'projected_mano_subtracted_object_owned_appearance_mask'):
                raise RuntimeError(f'optical-flow bridge frame {frame_idx} has malformed appearance-only ownership contract')
        else:
            geom = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else {}
            if geom.get('rigid_pose_observation_eligible') is not True or not isinstance(geom.get('object_surface_ownership_filter'), dict):
                raise RuntimeError(f'optical-flow bridge frame {frame_idx} lacks accepted metric or explicit appearance ownership evidence')
        raw_path = Path(str(frame.get('raw_frame_path') or ''))
        mask_path = Path(str(obj.get('mask_path') or ''))
        if not raw_path.is_file() or not mask_path.is_file():
            raise RuntimeError(f'optical-flow bridge frame {frame_idx} lacks raw image/object-owned mask')
        current_gray = read_gray(raw_path)
        if current_gray.shape != previous_gray.shape:
            raise RuntimeError('optical-flow bridge raw raster size changed within one clip')
        (height, width) = current_gray.shape
        (next_uv, forward_status, _) = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_uv, None, winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01), minEigThreshold=1e-05)
        if next_uv is None or forward_status is None:
            raise RuntimeError(f'forward optical flow failed at frame {frame_idx}')
        (reverse_uv, reverse_status, _) = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, next_uv, None, winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01), minEigThreshold=1e-05)
        if reverse_uv is None or reverse_status is None:
            raise RuntimeError(f'reverse optical flow failed at frame {frame_idx}')
        next_points = next_uv[:, 0]
        fb_error = np.linalg.norm(reverse_uv[:, 0] - previous_uv[:, 0], axis=1)
        good = (forward_status[:, 0] > 0) & (reverse_status[:, 0] > 0) & (fb_error <= 1.5) & (next_points[:, 0] >= 0.0) & (next_points[:, 0] < width) & (next_points[:, 1] >= 0.0) & (next_points[:, 1] < height)
        owned_mask = read_mask_at_size(mask_path, width, height)
        rounded = np.rint(next_points).astype(np.int64)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        good &= owned_mask[rounded[:, 1], rounded[:, 0]]
        current_uv = next_points[good]
        active_canonical = active_canonical[good]
        flow_trace.append({'frame_idx': int(frame_idx), 'input_track_count': int(len(next_points)), 'accepted_track_count': int(len(current_uv)), 'forward_backward_error_px': numeric_summary(fb_error[good]), 'object_owned_mask': str(mask_path)})
        if len(current_uv) < min_tracked_points:
            raise RuntimeError(f'optical-flow bridge retained {len(current_uv)} < {min_tracked_points} tracks at frame {frame_idx}')
        previous_uv = current_uv.astype(np.float32).reshape(-1, 1, 2)
        previous_gray = current_gray
    target_frame = frames[target_idx]
    (target_height, target_width) = previous_gray.shape
    (world_camera, intrinsics, raster_contract) = frame_camera_for_raster(target_frame, target_width, target_height)
    (success, rotation_vector, translation_vector, inlier_indices) = cv2.solvePnPRansac(active_canonical.astype(np.float64), current_uv.astype(np.float64), intrinsics, None, flags=cv2.SOLVEPNP_EPNP, iterationsCount=300, reprojectionError=4.0, confidence=0.999)
    if not success or inlier_indices is None:
        raise RuntimeError(f'calibrated optical-flow PnP failed at bridge frame {target_idx}')
    inliers = np.asarray(inlier_indices[:, 0], dtype=np.int64)
    if len(inliers) >= 6:
        (rotation_vector, translation_vector) = cv2.solvePnPRefineLM(active_canonical[inliers].astype(np.float64), current_uv[inliers].astype(np.float64), intrinsics, None, rotation_vector, translation_vector)
    for _ in range(2):
        (all_projected, _) = cv2.projectPoints(active_canonical.astype(np.float64), rotation_vector, translation_vector, intrinsics, None)
        all_reprojection = np.linalg.norm(all_projected[:, 0] - current_uv, axis=1)
        expanded_inliers = np.flatnonzero(all_reprojection <= 4.0)
        if len(expanded_inliers) < max(6, len(inliers)):
            break
        inliers = expanded_inliers.astype(np.int64)
        (rotation_vector, translation_vector) = cv2.solvePnPRefineLM(active_canonical[inliers].astype(np.float64), current_uv[inliers].astype(np.float64), intrinsics, None, rotation_vector, translation_vector)
    rotation_camera_from_object = cv2.Rodrigues(rotation_vector)[0]
    translation_camera = np.asarray(translation_vector, dtype=float).reshape(3)
    rotation_world_from_object = world_camera[:3, :3] @ rotation_camera_from_object
    translation_world = world_camera[:3, :3] @ translation_camera + world_camera[:3, 3]
    (projected, _) = cv2.projectPoints(active_canonical[inliers].astype(np.float64), rotation_vector, translation_vector, intrinsics, None)
    reprojection = np.linalg.norm(projected[:, 0] - current_uv[inliers], axis=1)
    camera_inlier_points = apply_pose(active_canonical[inliers], rotation_camera_from_object, translation_camera)
    positive_depth_fraction = float(np.mean(camera_inlier_points[:, 2] > 0.0))
    inlier_fraction = float(len(inliers) / max(1, len(current_uv)))
    reprojection_median = float(np.median(reprojection))
    reprojection_p95 = float(np.percentile(reprojection, 95))
    observability = rotation_observability_summary(active_canonical[inliers])
    quality_passed = bool(len(current_uv) >= min_tracked_points and len(inliers) >= min_pnp_inliers and (inlier_fraction >= min_inlier_fraction) and (reprojection_median <= max_reprojection_median_px) and (reprojection_p95 <= max_reprojection_p95_px) and (positive_depth_fraction >= 0.98) and np.all(np.isfinite(rotation_world_from_object)) and np.all(np.isfinite(translation_world)) and (abs(np.linalg.det(rotation_world_from_object) - 1.0) <= 1e-05))
    report = {'method': 'object_owned_rgb_forward_backward_lk_plus_calibrated_pnp', 'source_frame_idx': int(source_idx), 'target_frame_idx': int(target_idx), 'direction': 'forward' if direction > 0 else 'backward', 'seed': seed_report, 'flow_trace': flow_trace, 'target_camera_raster_contract': raster_contract, 'target_track_count': int(len(current_uv)), 'pnp_inlier_count': int(len(inliers)), 'pnp_inlier_fraction': inlier_fraction, 'reprojection_error_px': numeric_summary(reprojection), 'positive_camera_depth_fraction': positive_depth_fraction, 'rotation_observability': observability, 'pose_evidence_canonical_points_m': active_canonical[inliers].astype(float).tolist(), 'pose_evidence_point_provenance': "Accepted metric surfels from the source depth-owned frame, converted to the shared anchor canonical frame, then tracked to the target object's MANO-subtracted RGB support; no generated point is included.", 'quality_thresholds': {'minimum_tracked_points': int(min_tracked_points), 'minimum_pnp_inliers': int(min_pnp_inliers), 'minimum_pnp_inlier_fraction': float(min_inlier_fraction), 'maximum_reprojection_median_px': float(max_reprojection_median_px), 'maximum_reprojection_p95_px': float(max_reprojection_p95_px), 'minimum_positive_camera_depth_fraction': 0.98, 'rotation_observability_diagnostic_score_threshold': 0.02, 'per_bridge_observability_hard_gate': False, 'timeline_minimum_observable_fraction_remains_fixed': 0.8}, 'rotation_observability_policy': 'A sparse bridge may remain explicitly rotation-underobservable; the downstream fixed score>=0.02 and observable-fraction>=0.80 timeline gate remains authoritative and cannot be lowered.', 'quality_passed': quality_passed, 'claim_scope': 'Direct prediction-side RGB pose observation: accepted anchor-frame metric surfels are tracked only through MANO-subtracted object-owned image support and solved with the exact calibrated pinhole camera.'}
    if not quality_passed:
        raise RuntimeError(f'optical-flow PnP bridge failed closed: {report}')
    return (rotation_world_from_object, translation_world, report, observability)

def bridge_positions_for_gap(left: int, right: int, maximum_gap: int) -> list[int]:
    gap = int(right - left)
    if gap <= maximum_gap:
        return []
    segments = int(math.ceil(gap / maximum_gap))
    positions = [int(round(left + gap * segment / segments)) for segment in range(1, segments)]
    return sorted(set((position for position in positions if left < position < right)))

def complete_translation_measurements(direct_translations: dict[int, np.ndarray], frame_indices: list[int], *, bridge_frames: set[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    direct_indices = sorted(direct_translations)
    measurements: list[np.ndarray] = []
    weights: list[float] = []
    sources: list[str] = []
    for frame_idx in frame_indices:
        if frame_idx in direct_translations:
            measurements.append(np.asarray(direct_translations[frame_idx], dtype=float))
            weights.append(0.75 if frame_idx in bridge_frames else 1.0)
            sources.append('direct_rgb_flow_pnp' if frame_idx in bridge_frames else 'direct_metric_surface')
            continue
        lower = max((idx for idx in direct_indices if idx < frame_idx), default=None)
        upper = min((idx for idx in direct_indices if idx > frame_idx), default=None)
        if lower is not None and upper is not None:
            alpha = float((frame_idx - lower) / (upper - lower))
            value = (1.0 - alpha) * direct_translations[lower] + alpha * direct_translations[upper]
            measurements.append(np.asarray(value, dtype=float))
            weights.append(0.15)
            sources.append('interpolated_regularization_support')
        else:
            nearest = min(direct_indices, key=lambda idx: abs(idx - frame_idx))
            measurements.append(np.asarray(direct_translations[nearest], dtype=float))
            weights.append(0.08)
            sources.append('edge_hold_regularization_support')
    return (np.asarray(measurements, dtype=float), np.asarray(weights, dtype=float), sources)

def regularize_translation_timeline(measurements: np.ndarray, weights: np.ndarray, *, velocity_weight: float, acceleration_weight: float, zero_prior_weight: float=0.0) -> np.ndarray:
    count = len(measurements)
    if count < 3:
        return measurements.copy()
    first_difference = np.zeros((count - 1, count), dtype=float)
    for row in range(count - 1):
        first_difference[row, row] = -1.0
        first_difference[row, row + 1] = 1.0
    second_difference = np.zeros((count - 2, count), dtype=float)
    for row in range(count - 2):
        second_difference[row, row] = 1.0
        second_difference[row, row + 1] = -2.0
        second_difference[row, row + 2] = 1.0
    weight_matrix = np.diag(weights)
    system = weight_matrix + float(zero_prior_weight) * np.eye(count, dtype=float) + float(velocity_weight) * (first_difference.T @ first_difference) + float(acceleration_weight) * (second_difference.T @ second_difference)
    rhs = weight_matrix @ measurements
    return np.column_stack([np.linalg.solve(system, rhs[:, axis]) for axis in range(3)])

def fit_observed_pose(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--completion-report', type=Path, required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=6000)
    parser.add_argument('--iterations', type=int, default=4, help='Compatibility base; observed pairwise ICP uses max(8, 3*iterations)')
    parser.add_argument('--pairwise-trim-fraction', type=float, default=0.65)
    parser.add_argument('--pairwise-max-correspondence-m', type=float, default=0.035)
    parser.add_argument('--pairwise-min-matches', type=int, default=30)
    parser.add_argument('--pairwise-max-median-residual-m', type=float, default=0.015)
    parser.add_argument('--anchor-translation-trim-fraction', type=float, default=0.8)
    parser.add_argument('--anchor-translation-iterations', type=int, default=6)
    parser.add_argument('--rotation-unary-trim-fraction', type=float, default=0.75)
    parser.add_argument('--rotation-unary-iterations', type=int, default=6)
    parser.add_argument('--rotation-unary-damping', type=float, default=0.5)
    parser.add_argument('--temporal-rotation-correction-velocity-weight', type=float, default=6.0)
    parser.add_argument('--temporal-rotation-correction-acceleration-weight', type=float, default=4.0)
    parser.add_argument('--temporal-rotation-chain-prior-weight', type=float, default=0.75)
    parser.add_argument('--maximum-rotation-unary-correction-deg', type=float, default=60.0)
    parser.add_argument('--temporal-translation-velocity-weight', type=float, default=0.1)
    parser.add_argument('--temporal-translation-acceleration-weight', type=float, default=2.0)
    parser.add_argument('--maximum-direct-pose-gap', type=int, default=10)
    parser.add_argument('--minimum-direct-pose-fraction', type=float, default=0.8)
    parser.add_argument('--flow-max-seed-points', type=int, default=1200)
    parser.add_argument('--flow-min-tracked-points', type=int, default=50)
    parser.add_argument('--flow-min-pnp-inliers', type=int, default=40)
    parser.add_argument('--flow-min-pnp-inlier-fraction', type=float, default=0.5)
    parser.add_argument('--flow-max-reprojection-median-px', type=float, default=2.5)
    parser.add_argument('--flow-max-reprojection-p95-px', type=float, default=4.0)
    parser.add_argument('--include-ineligible-rigid-pose-observations', action='store_true', help='Historical-reproduction override: consume rows explicitly rejected by P09 (forbidden in annotation runs)')
    args = parser.parse_args(argv)
    if args.include_ineligible_rigid_pose_observations:
        raise RuntimeError('--include-ineligible-rigid-pose-observations is historical-only and forbidden by the observed-only annotation contract')
    if not 0.0 < args.pairwise_trim_fraction <= 1.0:
        raise RuntimeError('--pairwise-trim-fraction must be in (0,1]')
    if not 0.0 < args.anchor_translation_trim_fraction <= 1.0:
        raise RuntimeError('--anchor-translation-trim-fraction must be in (0,1]')
    if not 0.0 < args.rotation_unary_trim_fraction <= 1.0:
        raise RuntimeError('--rotation-unary-trim-fraction must be in (0,1]')
    if not 0.0 < args.rotation_unary_damping <= 1.0:
        raise RuntimeError('--rotation-unary-damping must be in (0,1]')
    if not 0.0 < args.minimum_direct_pose_fraction <= 1.0:
        raise RuntimeError('--minimum-direct-pose-fraction must be in (0,1]')
    annotations = load_json(args.annotations)
    completion = load_json(args.completion_report)
    completion_outputs = completion.get('outputs') if isinstance(completion.get('outputs'), dict) else {}
    mesh_value = completion_outputs.get('pose_hypothesis_mesh_labeled') or completion_outputs.get('completed_mesh_labeled')
    if not mesh_value:
        raise RuntimeError('completion report lacks outputs.pose_hypothesis_mesh_labeled/completed_mesh_labeled')
    mesh_path = Path(str(mesh_value))
    generated_diagnostic_mesh = load_mesh(mesh_path)
    generated_diagnostic_samples = deterministic_sample_mesh(generated_diagnostic_mesh, args.sample_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_frames = [frame for frame in annotations.get('frames', []) if isinstance(frame, dict)]
    if not annotation_frames:
        raise RuntimeError('annotations contain no frames')
    frames = {int(frame['frame_idx']): frame for frame in annotation_frames}
    frame_indices = sorted(frames)
    objects: dict[int, dict[str, Any]] = {}
    observations: dict[int, np.ndarray] = {}
    eligibility_by_frame: dict[int, tuple[bool | None, list[str], list[str]]] = {}
    initial_rows: dict[int, dict[str, Any]] = {}
    explicit_eligible_input_count = 0
    explicit_ineligible_input_count = 0
    eligibility_unspecified_input_count = 0
    missing_observed = 0
    ineligible_reason_counts: dict[str, int] = {}
    for frame_idx in frame_indices:
        frame = frames[frame_idx]
        obj = frame_object(frame, args.object_id)
        if obj is None:
            initial_rows[frame_idx] = {'frame_idx': int(frame_idx), 'status': 'missing_object_row', 'object_id': args.object_id}
            continue
        objects[frame_idx] = obj
        geom = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else {}
        eligibility = rigid_pose_observation_eligibility(obj, geom)
        eligibility_by_frame[frame_idx] = eligibility
        observed = np.asarray(geom.get('world_vertices_sample_m') or [], dtype=float)
        if eligibility[0] is True:
            explicit_eligible_input_count += 1
        elif eligibility[0] is False:
            explicit_ineligible_input_count += 1
            for reason in eligibility[1] or ['explicit_false_without_reason']:
                ineligible_reason_counts[reason] = ineligible_reason_counts.get(reason, 0) + 1
        else:
            eligibility_unspecified_input_count += 1
        if eligibility[0] is False:
            initial_rows[frame_idx] = {'frame_idx': int(frame_idx), 'status': 'rigid_pose_observation_ineligible', 'object_id': args.object_id, 'rigid_pose_observation_eligible': False, 'rigid_pose_observation_reasons': eligibility[1], 'rigid_pose_observation_eligibility_sources': eligibility[2], 'visible_sample_count': int(len(observed)) if observed.ndim == 2 else 0, 'policy': 'explicit P09 false is a hard metric-surface measurement rejection gate'}
            continue
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 6 or (not np.all(np.isfinite(observed))):
            missing_observed += 1
            initial_rows[frame_idx] = {'frame_idx': int(frame_idx), 'status': 'missing_accepted_metric_surface', 'object_id': args.object_id, 'appearance_mask_path': obj.get('mask_path')}
            continue
        observations[frame_idx] = observed
    adapter = annotations.get('v19_visible_geometry_adapter') if isinstance(annotations.get('v19_visible_geometry_adapter'), dict) else {}
    anchor_value = adapter.get('anchor_frame_idx')
    if anchor_value is None:
        raise RuntimeError('annotations lack v19_visible_geometry_adapter.anchor_frame_idx')
    anchor_frame_idx = int(anchor_value)
    completion_inputs = completion.get('inputs') if isinstance(completion.get('inputs'), dict) else {}
    completion_binding = completion_inputs.get('selected_anchor_atomic_binding')
    evidence_path = Path(str(completion_inputs.get('candidate_evidence_report') or ''))
    expected_evidence_sha256 = str(completion_inputs.get('candidate_evidence_report_sha256') or '')
    if not evidence_path.is_file() or not expected_evidence_sha256:
        raise RuntimeError('P14 completion lacks a byte-bound selected-anchor evidence report')
    actual_evidence_sha256 = sha256_file(evidence_path)
    if actual_evidence_sha256 != expected_evidence_sha256:
        raise RuntimeError('P14 selected-anchor evidence report hash changed after P14 reference creation')
    anchor_evidence = load_json(evidence_path)
    if anchor_evidence.get('selected_anchor_atomic_binding') != completion_binding:
        raise RuntimeError('P14 completion/evidence selected-anchor bindings differ')
    evidence_selected = anchor_evidence.get('selected') if isinstance(anchor_evidence.get('selected'), dict) else {}
    evidence_geom = evidence_selected.get('visible_geometry_candidate') if isinstance(evidence_selected.get('visible_geometry_candidate'), dict) else {}
    completion_binding_centroid = np.asarray(evidence_geom.get('centroid_world_m') or [], dtype=float).reshape(-1)
    if not isinstance(completion_binding, dict) or completion_binding.get('validated') is not True or int(completion_binding.get('selected_frame_idx', -1)) != anchor_frame_idx or (int(completion_binding.get('visible_geometry_frame_idx', -1)) != anchor_frame_idx) or (int(completion_binding.get('canonical_surface_frame_idx', -1)) != anchor_frame_idx) or (completion_binding_centroid.shape != (3,)) or (not np.isfinite(completion_binding_centroid).all()):
        raise RuntimeError(f'P14 annotations/completion do not share one atomic selected anchor: {completion_binding}')
    if anchor_frame_idx not in observations:
        raise RuntimeError(f'selected anchor frame {anchor_frame_idx} lacks accepted metric surfels')
    anchor_world = observations[anchor_frame_idx]
    anchor_centroid = anchor_world.mean(axis=0)
    anchor_binding_centroid_error_m = float(np.linalg.norm(anchor_centroid - completion_binding_centroid))
    if anchor_binding_centroid_error_m > 1e-06:
        raise RuntimeError(f'P14 selected anchor surfels disagree with the atomic completion centroid by {anchor_binding_centroid_error_m} m')
    anchor_canonical = anchor_world - anchor_centroid
    anchor_extent = np.ptp(anchor_world, axis=0)
    anchor_diag = float(np.linalg.norm(anchor_extent))
    if not np.isfinite(anchor_diag) or anchor_diag <= 0.0:
        raise RuntimeError('selected anchor observed surface has invalid metric extent')
    depth_indices = sorted(observations)
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {anchor_frame_idx: (np.eye(3, dtype=float), anchor_centroid.astype(float))}
    edge_reports: list[dict[str, Any]] = []
    pairwise_iterations = max(8, int(args.iterations) * 3)
    previous = anchor_frame_idx
    for current in [idx for idx in depth_indices if idx > anchor_frame_idx]:
        (delta_rotation, delta_translation, edge_report) = register_adjacent_observed_surfaces(observations[previous], observations[current], frame_gap=current - previous, iterations=pairwise_iterations, trim_fraction=float(args.pairwise_trim_fraction), max_correspondence_m=float(args.pairwise_max_correspondence_m), min_matches=int(args.pairwise_min_matches), max_median_residual_m=float(args.pairwise_max_median_residual_m))
        (previous_rotation, previous_translation) = poses[previous]
        poses[current] = (delta_rotation @ previous_rotation, delta_rotation @ previous_translation + delta_translation)
        edge_reports.append({'source_frame_idx': previous, 'target_frame_idx': current, **edge_report})
        previous = current
    previous = anchor_frame_idx
    for current in sorted([idx for idx in depth_indices if idx < anchor_frame_idx], reverse=True):
        (delta_rotation, delta_translation, edge_report) = register_adjacent_observed_surfaces(observations[previous], observations[current], frame_gap=previous - current, iterations=pairwise_iterations, trim_fraction=float(args.pairwise_trim_fraction), max_correspondence_m=float(args.pairwise_max_correspondence_m), min_matches=int(args.pairwise_min_matches), max_median_residual_m=float(args.pairwise_max_median_residual_m))
        (previous_rotation, previous_translation) = poses[previous]
        poses[current] = (delta_rotation @ previous_rotation, delta_rotation @ previous_translation + delta_translation)
        edge_reports.append({'source_frame_idx': previous, 'target_frame_idx': current, **edge_report})
        previous = current
    preliminary_translation_reports: dict[int, dict[str, Any]] = {}
    for frame_idx in depth_indices:
        (rotation, initial_translation) = poses[frame_idx]
        (stabilized_translation, stabilization_report) = refine_translation_from_observed_anchor(anchor_canonical, observations[frame_idx], rotation, initial_translation, iterations=int(args.anchor_translation_iterations), trim_fraction=float(args.anchor_translation_trim_fraction))
        poses[frame_idx] = (rotation, stabilized_translation)
        preliminary_translation_reports[frame_idx] = stabilization_report
    chain_poses = {idx: (pose[0].copy(), pose[1].copy()) for (idx, pose) in poses.items()}
    rotation_unary_reports: dict[int, dict[str, Any]] = {}
    rotation_corrections: dict[int, np.ndarray] = {}
    for frame_idx in depth_indices:
        (chain_rotation, chain_translation) = chain_poses[frame_idx]
        (unary_rotation, _, unary_report) = refine_pose_from_observed_anchor(anchor_canonical, observations[frame_idx], chain_rotation, chain_translation, iterations=int(args.rotation_unary_iterations), trim_fraction=float(args.rotation_unary_trim_fraction), damping=float(args.rotation_unary_damping))
        correction = Rotation.from_matrix(unary_rotation @ chain_rotation.T).as_rotvec()
        rotation_corrections[frame_idx] = correction.astype(float)
        rotation_unary_reports[frame_idx] = unary_report
    (correction_measurements, correction_weights, _) = complete_translation_measurements(rotation_corrections, frame_indices, bridge_frames=set())
    regularized_corrections = regularize_translation_timeline(correction_measurements, correction_weights, velocity_weight=float(args.temporal_rotation_correction_velocity_weight), acceleration_weight=float(args.temporal_rotation_correction_acceleration_weight), zero_prior_weight=float(args.temporal_rotation_chain_prior_weight))
    frame_to_row = {frame_idx: row for (row, frame_idx) in enumerate(frame_indices)}
    regularized_corrections -= regularized_corrections[frame_to_row[anchor_frame_idx]]
    direct_rotation_correction_deg = {idx: float(np.degrees(np.linalg.norm(regularized_corrections[frame_to_row[idx]]))) for idx in depth_indices}
    if max(direct_rotation_correction_deg.values(), default=0.0) > float(args.maximum_rotation_unary_correction_deg):
        raise RuntimeError(f'observed-anchor unary rotation correction left the pairwise temporal basin: summary={numeric_summary(list(direct_rotation_correction_deg.values()))} maximum_allowed_deg={args.maximum_rotation_unary_correction_deg}')
    for frame_idx in depth_indices:
        (chain_rotation, chain_translation) = chain_poses[frame_idx]
        correction_rotation = Rotation.from_rotvec(regularized_corrections[frame_to_row[frame_idx]]).as_matrix()
        poses[frame_idx] = (correction_rotation @ chain_rotation, chain_translation)
    translation_fit_reports: dict[int, dict[str, Any]] = {}
    for frame_idx in depth_indices:
        (rotation, initial_translation) = poses[frame_idx]
        (translation, translation_report) = refine_translation_from_observed_anchor(anchor_canonical, observations[frame_idx], rotation, initial_translation, iterations=int(args.anchor_translation_iterations), trim_fraction=float(args.anchor_translation_trim_fraction))
        poses[frame_idx] = (rotation, translation)
        translation_fit_reports[frame_idx] = translation_report
    bridge_reports: dict[int, dict[str, Any]] = {}
    bridge_observability: dict[int, dict[str, Any]] = {}
    initial_direct_indices = sorted(poses)
    requested_bridge_positions: list[tuple[int, int, int]] = []
    for (left, right) in zip(initial_direct_indices[:-1], initial_direct_indices[1:]):
        for target in bridge_positions_for_gap(left, right, int(args.maximum_direct_pose_gap)):
            requested_bridge_positions.append((left, target, right))
    required_direct_count = int(math.ceil(float(args.minimum_direct_pose_fraction) * len(frame_indices) - 1e-12))
    if len(poses) + len(requested_bridge_positions) < required_direct_count:
        missing_candidates = [idx for idx in frame_indices if idx not in poses]
        ranked = sorted(missing_candidates, key=lambda idx: min((abs(idx - direct) for direct in initial_direct_indices)))
        already = {target for (_, target, _) in requested_bridge_positions}
        for target in ranked:
            if len(poses) + len(already) >= required_direct_count:
                break
            if target in already:
                continue
            left = max((idx for idx in initial_direct_indices if idx < target), default=None)
            right = min((idx for idx in initial_direct_indices if idx > target), default=None)
            if left is not None and right is not None:
                requested_bridge_positions.append((left, target, right))
                already.add(target)
    for (left, target, right) in requested_bridge_positions:
        if target in poses:
            continue
        (rotation, translation, bridge_report, observability) = estimate_optical_flow_pnp_bridge(source_idx=left, target_idx=target, frames=frames, objects=objects, observed_world=observations[left], source_rotation=poses[left][0], source_translation=poses[left][1], max_seed_points=int(args.flow_max_seed_points), min_tracked_points=int(args.flow_min_tracked_points), min_pnp_inliers=int(args.flow_min_pnp_inliers), min_inlier_fraction=float(args.flow_min_pnp_inlier_fraction), max_reprojection_median_px=float(args.flow_max_reprojection_median_px), max_reprojection_p95_px=float(args.flow_max_reprojection_p95_px))
        left_gap = max(1, target - left)
        right_gap = max(1, right - target)
        left_rotation_rate = rotation_angle_deg(rotation @ poses[left][0].T) / left_gap
        left_translation_rate = float(np.linalg.norm(translation - poses[left][1]) / left_gap)
        right_rotation_rate = rotation_angle_deg(poses[right][0] @ rotation.T) / right_gap
        right_translation_rate = float(np.linalg.norm(poses[right][1] - translation) / right_gap)
        endpoint_consistency_passed = bool(max(left_rotation_rate, right_rotation_rate) <= 20.0 and max(left_translation_rate, right_translation_rate) <= 0.1)
        bridge_report['neighbor_depth_pose_consistency'] = {'left_depth_frame_idx': int(left), 'right_depth_frame_idx': int(right), 'left_implied_rotation_deg_per_frame': float(left_rotation_rate), 'right_implied_rotation_deg_per_frame': float(right_rotation_rate), 'left_implied_translation_m_per_frame': float(left_translation_rate), 'right_implied_translation_m_per_frame': float(right_translation_rate), 'maximum_rotation_deg_per_frame': 20.0, 'maximum_translation_m_per_frame': 0.1, 'passed': endpoint_consistency_passed}
        if not endpoint_consistency_passed:
            raise RuntimeError(f'optical-flow PnP bridge endpoint consistency failed closed: {bridge_report}')
        poses[target] = (rotation, translation)
        bridge_reports[target] = bridge_report
        bridge_observability[target] = observability
    direct_indices = sorted(poses)
    direct_gaps = [right - left for (left, right) in zip(direct_indices[:-1], direct_indices[1:])]
    maximum_direct_gap = max(direct_gaps, default=0)
    if len(direct_indices) < required_direct_count or maximum_direct_gap > int(args.maximum_direct_pose_gap):
        raise RuntimeError(f'observed pose support remains below the fixed temporal contract after fail-closed RGB bridges: direct={len(direct_indices)}/{len(frame_indices)} required={required_direct_count} max_gap={maximum_direct_gap} allowed={args.maximum_direct_pose_gap}')
    direct_translations = {idx: pose[1].copy() for (idx, pose) in poses.items()}
    bridge_frames = set(bridge_reports)
    (translation_measurements, translation_weights, translation_sources) = complete_translation_measurements(direct_translations, frame_indices, bridge_frames=bridge_frames)
    regularized_translations = regularize_translation_timeline(translation_measurements, translation_weights, velocity_weight=float(args.temporal_translation_velocity_weight), acceleration_weight=float(args.temporal_translation_acceleration_weight))
    frame_to_row = {frame_idx: row for (row, frame_idx) in enumerate(frame_indices)}
    anchor_offset = anchor_centroid - regularized_translations[frame_to_row[anchor_frame_idx]]
    regularized_translations += anchor_offset
    direct_adjustments = {idx: float(np.linalg.norm(regularized_translations[frame_to_row[idx]] - direct_translations[idx])) for idx in direct_indices}
    maximum_allowed_adjustment_m = max(0.075, 0.25 * anchor_diag)
    if float(np.median(list(direct_adjustments.values()))) > 0.03 or max(direct_adjustments.values(), default=0.0) > maximum_allowed_adjustment_m:
        raise RuntimeError(f'temporal translation regularization required implausibly large corrections: summary={numeric_summary(list(direct_adjustments.values()))} max_allowed={maximum_allowed_adjustment_m}')
    for idx in direct_indices:
        poses[idx] = (poses[idx][0], regularized_translations[frame_to_row[idx]].copy())
    rows: list[dict[str, Any]] = []
    generated_diagnostic_medians: list[float] = []
    observed_anchor_medians: list[float] = []
    for frame_idx in frame_indices:
        if frame_idx not in poses:
            rows.append(initial_rows.get(frame_idx, {'frame_idx': frame_idx, 'status': 'no_direct_pose_observation'}))
            continue
        (rotation, translation) = poses[frame_idx]
        is_bridge = frame_idx in bridge_reports
        if is_bridge:
            generated_observed = nearest_summary(np.empty((0, 3), dtype=float), generated_diagnostic_samples)
            generated_reverse = nearest_summary(np.empty((0, 3), dtype=float), generated_diagnostic_samples)
            anchor_observed = nearest_summary(np.empty((0, 3), dtype=float), anchor_canonical)
            observability = bridge_observability[frame_idx]
            visible_sample_count = 0
            direct_source = 'object_owned_rgb_optical_flow_calibrated_pnp'
        else:
            observed = observations[frame_idx]
            generated_world = apply_pose(generated_diagnostic_samples, rotation, translation)
            anchor_world_model = apply_pose(anchor_canonical, rotation, translation)
            generated_observed = nearest_summary(observed, generated_world)
            generated_reverse = nearest_summary(generated_world, observed)
            anchor_observed = nearest_summary(observed, anchor_world_model)
            generated_diagnostic_medians.append(float(generated_observed['median_m']))
            observed_anchor_medians.append(float(anchor_observed['median_m']))
            observability = rotation_observability_summary(observed)
            visible_sample_count = int(len(observed))
            direct_source = 'adjacent_observed_metric_surfel_registration'
        (eligibility, eligibility_reasons, eligibility_sources) = eligibility_by_frame.get(frame_idx, (None, [], []))
        row = {'frame_idx': int(frame_idx), 'status': 'fit_to_object_owned_rgb_calibrated_pnp' if is_bridge else 'fit_to_visible_depth_samples', 'object_id': args.object_id, 'rotation_world_from_completed_canonical_matrix': rotation.astype(float).tolist(), 'translation_world_m': translation.astype(float).tolist(), 'pose_source': 'direct_observed_evidence_with_fixed_temporal_regularization', 'direct_pose_observation_source': direct_source, 'generated_geometry_pose_evidence_consumed': False, 'visible_sample_count': visible_sample_count, 'rigid_pose_observation_eligible': eligibility, 'rigid_pose_observation_reasons': eligibility_reasons, 'rigid_pose_observation_eligibility_sources': eligibility_sources, 'observed_to_mesh_initial': generated_observed, 'observed_to_mesh_final': generated_observed, 'mesh_to_observed_final': generated_reverse, 'generated_mesh_metric_role': 'post_fit_diagnostic_only_not_pose_evidence', 'observed_anchor_to_current_final': anchor_observed, 'rotation_observability': observability, 'translation_regularization': {'measurement_translation_world_m': direct_translations[frame_idx].astype(float).tolist(), 'regularized_translation_world_m': translation.astype(float).tolist(), 'adjustment_m': direct_adjustments[frame_idx], 'measurement_weight': 0.75 if is_bridge else 1.0}, 'icp_trace': translation_fit_reports.get(frame_idx, {}).get('trace', [])}
        if is_bridge:
            evidence_canonical = np.asarray(bridge_reports[frame_idx].get('pose_evidence_canonical_points_m') or [], dtype=float)
            row['rgb_optical_flow_pnp_bridge'] = bridge_reports[frame_idx]
            row['direct_pose_evidence_world_points_m'] = apply_pose(evidence_canonical, rotation, translation).astype(float).tolist()
            row['direct_pose_evidence_kind'] = 'tracked_source_metric_surfels_at_target_rgb_frame'
        else:
            row['observed_anchor_translation_fit'] = translation_fit_reports[frame_idx]
            row['preliminary_chain_translation_stabilization'] = preliminary_translation_reports[frame_idx]
            row['observed_anchor_rotation_unary_fit'] = rotation_unary_reports[frame_idx]
            row['rotation_correction_from_pairwise_chain'] = {'regularized_correction_rotvec_rad': regularized_corrections[frame_to_row[frame_idx]].astype(float).tolist(), 'regularized_correction_deg': direct_rotation_correction_deg[frame_idx], 'generated_geometry_consumed': False}
        rows.append(row)
    direct_rotation_steps: list[float] = []
    direct_translation_steps: list[float] = []
    for (left, right) in zip(direct_indices[:-1], direct_indices[1:]):
        (left_rotation, left_translation) = poses[left]
        (right_rotation, right_translation) = poses[right]
        gap = max(1, right - left)
        direct_rotation_steps.append(rotation_angle_deg(right_rotation @ left_rotation.T) / gap)
        direct_translation_steps.append(float(np.linalg.norm(right_translation - left_translation) / gap))
    report = {'method': 'fit_v18_compact_rigid_object_pose', 'status': 'ok', 'pose_estimation_method': METHOD, 'object_id': args.object_id, 'annotations': str(args.annotations), 'completion_report': str(args.completion_report), 'pose_hypothesis_mesh_labeled': str(mesh_path), 'generated_geometry_pose_evidence_consumed': False, 'generated_mesh_role': 'post_fit_distance_diagnostic_and_downstream_render_binding_only', 'generated_faces_pose_eligible': False, 'generated_faces_collision_eligible': False, 'generated_faces_contact_eligible': False, 'claim_scope': 'Shared prediction-side rigid trajectory from accepted metric surfel-to-surfel temporal registration, anchor observed-surface translation, and sparse object-owned RGB calibrated-PnP gap bridges. Generated completion faces are never pose correspondences.', 'anchor_frame_idx': int(anchor_frame_idx), 'anchor_centroid_world_m': anchor_centroid.astype(float).tolist(), 'selected_anchor_atomic_binding': completion_binding, 'selected_anchor_evidence_report': str(evidence_path), 'selected_anchor_evidence_report_sha256': actual_evidence_sha256, 'selected_anchor_binding_centroid_error_m': anchor_binding_centroid_error_m, 'anchor_observed_extent_m': anchor_extent.astype(float).tolist(), 'anchor_observed_extent_diag_m': anchor_diag, 'anchor_canonical_source': 'selected_anchor_metric_surfels_minus_anchor_centroid_world_m', 'sample_count': int(len(generated_diagnostic_samples)), 'iterations': int(args.iterations), 'pairwise_observed_surface_registration': {'edge_count': int(len(edge_reports)), 'iterations_per_edge': int(pairwise_iterations), 'trim_fraction': float(args.pairwise_trim_fraction), 'max_correspondence_m_base': float(args.pairwise_max_correspondence_m), 'minimum_matches': int(args.pairwise_min_matches), 'maximum_median_residual_m': float(args.pairwise_max_median_residual_m), 'delta_rotation_deg': numeric_summary([float(row['delta_rotation_deg']) for row in edge_reports]), 'median_residual_m': numeric_summary([float(row['residual_m']['median']) for row in edge_reports]), 'edges': edge_reports}, 'rgb_optical_flow_pnp_gap_bridges': {'enabled': True, 'requested': [{'left_depth_frame_idx': left, 'target_frame_idx': target, 'right_depth_frame_idx': right} for (left, target, right) in requested_bridge_positions], 'accepted_frame_count': int(len(bridge_reports)), 'accepted_frames': sorted(bridge_reports), 'reports': {str(idx): bridge_reports[idx] for idx in sorted(bridge_reports)}, 'generated_geometry_consumed': False}, 'rotation_unary_regularization': {'method': 'observed_anchor_surface_unary_icp_correction_over_pairwise_observed_chain', 'unary_iterations': int(args.rotation_unary_iterations), 'unary_trim_fraction': float(args.rotation_unary_trim_fraction), 'unary_damping': float(args.rotation_unary_damping), 'correction_velocity_weight': float(args.temporal_rotation_correction_velocity_weight), 'correction_acceleration_weight': float(args.temporal_rotation_correction_acceleration_weight), 'pairwise_chain_zero_correction_prior_weight': float(args.temporal_rotation_chain_prior_weight), 'anchor_correction_fixed_to_zero': True, 'direct_regularized_correction_deg': numeric_summary(list(direct_rotation_correction_deg.values())), 'maximum_allowed_correction_deg': float(args.maximum_rotation_unary_correction_deg), 'generated_geometry_consumed': False}, 'translation_regularization': {'method': 'weighted_quadratic_velocity_and_acceleration_prior', 'velocity_weight': float(args.temporal_translation_velocity_weight), 'acceleration_weight': float(args.temporal_translation_acceleration_weight), 'direct_metric_surface_weight': 1.0, 'direct_rgb_flow_pnp_weight': 0.75, 'interpolated_support_weight': 0.15, 'edge_hold_support_weight': 0.08, 'anchor_recenter_offset_m': anchor_offset.astype(float).tolist(), 'direct_adjustment_m': numeric_summary(list(direct_adjustments.values())), 'maximum_allowed_direct_adjustment_m': maximum_allowed_adjustment_m, 'measurement_source_counts': {source: int(translation_sources.count(source)) for source in sorted(set(translation_sources))}}, 'frame_fit_rows': rows, 'pose_rows': rows, 'fitted_frame_count': int(len(direct_indices)), 'direct_pose_frames': direct_indices, 'direct_metric_surface_pose_frames': sorted(observations), 'direct_rgb_flow_pnp_pose_frames': sorted(bridge_reports), 'direct_support': {'frame_count': int(len(frame_indices)), 'required_direct_count': required_direct_count, 'direct_count': int(len(direct_indices)), 'direct_fraction': float(len(direct_indices) / len(frame_indices)), 'maximum_direct_gap': int(maximum_direct_gap), 'maximum_allowed_direct_gap': int(args.maximum_direct_pose_gap)}, 'missing_initial_pose_count': 0, 'missing_observed_count': int(missing_observed), 'rigid_pose_eligibility_gate': {'include_ineligible_override': False, 'explicit_eligible_input_count': int(explicit_eligible_input_count), 'explicit_ineligible_input_count': int(explicit_ineligible_input_count), 'eligibility_unspecified_input_count': int(eligibility_unspecified_input_count), 'ineligible_observation_count': int(explicit_ineligible_input_count), 'ineligible_observation_frames': sorted((idx for (idx, value) in eligibility_by_frame.items() if value[0] is False)), 'ineligible_reason_counts': ineligible_reason_counts, 'explicit_eligible_fit_count': int(sum((eligibility_by_frame.get(idx, (None, [], []))[0] is True for idx in observations))), 'eligibility_unspecified_fit_count': int(sum((eligibility_by_frame.get(idx, (None, [], []))[0] is None for idx in observations))), 'ineligible_override_fit_count': 0, 'excluded_from_fit_count': int(explicit_ineligible_input_count), 'policy': 'explicit P09 false remains a hard metric-surface rejection; RGB bridges use appearance masks but never rejected depth'}, 'observed_pose_evidence_diagnostics': {'observed_anchor_to_current_median_summary_m': numeric_summary(observed_anchor_medians), 'direct_rotation_deg_per_frame': numeric_summary(direct_rotation_steps), 'direct_translation_m_per_frame': numeric_summary(direct_translation_steps)}, 'final_observed_to_mesh_median_summary_m': numeric_summary(generated_diagnostic_medians), 'final_observed_to_mesh_notice': 'generated mesh distance is diagnostic only and was not optimized', 'output': {'pose_hypothesis_mesh_labeled': str(mesh_path), 'collision_eligible_mesh_labeled': completion_outputs.get('collision_eligible_mesh_labeled'), 'pose_report': str(args.output_dir / 'v18_compact_rigid_object_pose_fit_report.json')}}
    output_path = args.output_dir / 'v18_compact_rigid_object_pose_fit_report.json'
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
POSE_MEASUREMENT_STATUSES = {'fit_to_visible_depth_samples', 'fit_to_visible_depth_archive_vertices', 'fit_to_object_owned_rgb_calibrated_pnp'}
CORRECTED_POSE_STATUS = 'corrected_temporal_rigid_pose_graph'
COMPLETED_POSE_STATUS = 'completed_temporal_rigid_pose_uncertain'
STRICT_ROTATION_STEP_DEG = 15.0
CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS = {'max_rotation_step_deg': 18.0, 'max_count': 2, 'max_fraction': 0.015, 'max_translation_step_m': 0.02, 'max_endpoint_observability_score': 0.03}

@dataclass(frozen=True)
class PoseObservation:
    frame_idx: int
    source_row: dict[str, Any]
    rotation_world_from_canonical: np.ndarray
    translation_world_m: np.ndarray
    translation_sigma_m: float
    rotation_sigma_rad: float
    visible_sample_count: int
    observed_points_world: np.ndarray
    nonpenetration_target_world_m: np.ndarray | None
    nonpenetration_weight: float
    nonpenetration_source_rows: int

@dataclass(frozen=True)
class ImageFactorObservation:
    frame_idx: int
    T_world_camera: np.ndarray
    K_raster: np.ndarray
    depth_points_canonical: np.ndarray
    depth_observed_z: np.ndarray
    depth_weight: np.ndarray
    silhouette_points_canonical: np.ndarray
    silhouette_target_uv: np.ndarray
    silhouette_kind: np.ndarray

def formal_object_load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def formal_object_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def validate_camera_transform(transform: np.ndarray, frame_idx: int) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f'frame {frame_idx} has invalid T_world_camera_metric')
    rotation = transform[:3, :3]
    if np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 0.0001 or abs(np.linalg.det(rotation) - 1.0) > 0.0001:
        raise RuntimeError(f'frame {frame_idx} has non-rigid T_world_camera_metric')
    if not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-06):
        raise RuntimeError(f'frame {frame_idx} has malformed homogeneous camera transform')
    return transform

def as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f'invalid {name}: expected {shape}, got {arr.shape}')
    return arr

def resize_intrinsics_xy(K_source: np.ndarray, source_size_wh: tuple[int, int], target_size_wh: tuple[int, int]) -> np.ndarray:
    sx = float(target_size_wh[0]) / float(source_size_wh[0])
    sy = float(target_size_wh[1]) / float(source_size_wh[1])
    K = np.asarray(K_source, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
    K[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return K

def formal_object_load_mesh(path: Path) -> trimesh.Trimesh:
    import trimesh
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f'no triangle mesh in {path}')
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f'invalid mesh: {path}')
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)

def formal_object_deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    import trimesh
    rng = np.random.default_rng(seed)
    (pts, _) = trimesh.sample.sample_surface(mesh, int(count), seed=rng)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or (not np.isfinite(pts).all()):
        raise RuntimeError('sampled mesh points invalid')
    return pts

def formal_object_nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    query = np.asarray(query, dtype=float)
    target = np.asarray(target, dtype=float)
    if query.ndim != 2 or target.ndim != 2 or query.shape[1:] != (3,) or (target.shape[1:] != (3,)) or (len(query) == 0) or (len(target) == 0):
        return {'count': int(len(query)) if query.ndim == 2 else 0, 'median_m': None, 'p90_m': None, 'p95_m': None, 'mean_m': None, 'max_m': None}
    (d, _) = cKDTree(target).query(query, k=1, workers=-1)
    return {'count': int(len(query)), 'median_m': float(np.median(d)), 'p90_m': float(np.percentile(d, 90.0)), 'p95_m': float(np.percentile(d, 95.0)), 'mean_m': float(np.mean(d)), 'max_m': float(np.max(d))}

def formal_object_numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'count': 0, 'median': None, 'p90': None, 'p95': None, 'max': None, 'mean': None}
    return {'count': int(arr.size), 'median': float(np.median(arr)), 'p90': float(np.percentile(arr, 90.0)), 'p95': float(np.percentile(arr, 95.0)), 'max': float(np.max(arr)), 'mean': float(np.mean(arr))}

def formal_object_apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation[None, :]

def corrected_pose(obs: PoseObservation, rot_delta: np.ndarray, trans_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_corr = Rotation.from_rotvec(rot_delta).as_matrix() @ obs.rotation_world_from_canonical
    t_corr = obs.translation_world_m + trans_delta
    return (r_corr, t_corr)

def pose_row_sigma(row: dict[str, Any], object_radius_m: float, args: argparse.Namespace) -> tuple[float, float]:
    final = row.get('observed_to_mesh_final') if isinstance(row.get('observed_to_mesh_final'), dict) else {}
    median = final.get('median_m')
    p95 = final.get('p95_m')
    visible_n = int(row.get('visible_sample_count') or final.get('count') or 0)
    finite = [float(x) for x in (median, p95) if x is not None and math.isfinite(float(x))]
    residual_scale = max(finite) if finite else float(args.default_pose_sigma_m)
    sample_factor = math.sqrt(max(1.0, min(float(visible_n), 400.0)) / 100.0)
    sigma_t = residual_scale / max(1.0, sample_factor)
    sigma_t = float(np.clip(max(float(args.min_pose_sigma_m), sigma_t), float(args.min_pose_sigma_m), float(args.max_pose_sigma_m)))
    sigma_r = sigma_t / max(float(object_radius_m), 0.001)
    sigma_r = float(np.clip(sigma_r, float(args.min_pose_rotation_sigma_rad), float(args.max_pose_rotation_sigma_rad)))
    return (sigma_t, sigma_r)

def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get('objects', []) if isinstance(frame.get('objects'), list) else []:
        if obj.get('object_id') == object_id:
            return obj
    legacy = frame.get('object') if isinstance(frame.get('object'), dict) else None
    if legacy and legacy.get('object_id') in (None, object_id):
        return legacy
    return None

def constraint_targets(path: Path | None, max_target_m: float, accepted_states: set[str]) -> dict[int, tuple[np.ndarray, float, int]]:
    if path is None:
        return {}
    data = formal_object_load_json(path)
    by_frame: dict[int, list[np.ndarray]] = {}
    for row in data.get('constraint_rows', []) if isinstance(data, dict) else []:
        state = str(row.get('candidate_application_state') or '')
        if state not in accepted_states:
            continue
        cand = np.asarray(row.get('candidate_translation_world_m') or [], dtype=float)
        if cand.shape != (3,) or not np.isfinite(cand).all():
            continue
        norm = float(np.linalg.norm(cand))
        if norm <= 0.0:
            continue
        target = -cand
        tnorm = float(np.linalg.norm(target))
        if tnorm > max_target_m:
            target = target * (max_target_m / max(tnorm, 1e-12))
        by_frame.setdefault(int(row['frame_idx']), []).append(target.astype(float))
    out: dict[int, tuple[np.ndarray, float, int]] = {}
    for (idx, targets) in by_frame.items():
        stack = np.vstack(targets)
        med = np.median(stack, axis=0)
        weight = min(1.0, math.sqrt(len(targets)))
        out[int(idx)] = (med.astype(float), float(weight), int(len(targets)))
    return out

def load_image_factor_observations(path: Path | None, observations: list[PoseObservation], args: argparse.Namespace, expected_completed_mesh: Path) -> tuple[dict[int, ImageFactorObservation], dict[str, Any]]:
    if path is None:
        return ({}, {'enabled': False})
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as data:
        required = {'metadata', 'frame_idx', 'T_world_camera', 'K_raster', 'depth_offsets', 'depth_points_canonical', 'depth_observed_z', 'silhouette_offsets', 'silhouette_points_canonical', 'silhouette_target_uv', 'silhouette_kind'}
        missing_keys = sorted(required.difference(data.files))
        if missing_keys:
            raise RuntimeError(f'image factor NPZ is missing keys: {missing_keys}')
        metadata_raw = data['metadata'][0]
        metadata = json.loads(str(metadata_raw))
        frame_idx = np.asarray(data['frame_idx'], dtype=np.int64).copy()
        T_world_camera = np.asarray(data['T_world_camera'], dtype=np.float64).copy()
        K_raster = np.asarray(data['K_raster'], dtype=np.float64).copy()
        depth_offsets = np.asarray(data['depth_offsets'], dtype=np.int64).copy()
        silhouette_offsets = np.asarray(data['silhouette_offsets'], dtype=np.int64).copy()
        depth_points = np.asarray(data['depth_points_canonical'], dtype=np.float64).copy()
        depth_z = np.asarray(data['depth_observed_z'], dtype=np.float64).copy()
        depth_weight_present = 'depth_weight' in data.files
        depth_weight = np.asarray(data['depth_weight'], dtype=np.float64).copy() if 'depth_weight' in data.files else np.ones(len(depth_z), dtype=np.float64)
        silhouette_points = np.asarray(data['silhouette_points_canonical'], dtype=np.float64).copy()
        silhouette_targets = np.asarray(data['silhouette_target_uv'], dtype=np.float64).copy()
        silhouette_kind = np.asarray(data['silhouette_kind'], dtype=np.int8).copy()
    if not isinstance(metadata, dict):
        raise RuntimeError('image factor metadata must be a JSON object')
    expected = {'annotations': str(args.annotations.expanduser().resolve()), 'pose_report': str(args.pose_report.expanduser().resolve()), 'factor_pose_report': str(args.pose_report.expanduser().resolve()), 'completed_mesh': str(expected_completed_mesh.expanduser().resolve()), 'object_id': str(args.object_id)}
    for (key, expected_value) in expected.items():
        actual = str(metadata.get(key) or '')
        if actual != expected_value:
            raise RuntimeError(f'image factor contract mismatch for {key}: expected {expected_value}, got {actual}; rebuild first-hit/silhouette factors for this exact P14/annotation/mesh input')
    strict = bool(getattr(args, 'require_strict_image_factor_contract', False))
    hash_contract = metadata.get('input_sha256') if isinstance(metadata.get('input_sha256'), dict) else {}
    if strict:
        if int(metadata.get('contract_version') or 0) < 2:
            raise RuntimeError('strict image-factor contract requires contract_version >= 2')
        if not depth_weight_present:
            raise RuntimeError('strict image-factor contract requires per-factor depth_weight')
        expected_hashes = {'annotations': args.annotations, 'pose_report': args.pose_report, 'factor_pose_report': args.pose_report, 'completed_mesh': expected_completed_mesh}
        for (key, source_path) in expected_hashes.items():
            actual_hash = str(hash_contract.get(key) or '')
            if actual_hash != formal_object_sha256_file(source_path):
                raise RuntimeError(f'image factor hash contract mismatch for {key}: expected current {formal_object_sha256_file(source_path)}, got {actual_hash}')
        for optional_key in ('hand_npz', 'mano_faces_pkl'):
            optional_path = Path(str(metadata.get(optional_key) or '')).expanduser().resolve()
            optional_hash = str(hash_contract.get(optional_key) or '')
            if not optional_path.is_file() or not optional_hash or optional_hash != formal_object_sha256_file(optional_path):
                raise RuntimeError(f'image factor {optional_key} hash contract mismatch')
        if metadata.get('generated_faces_consumed') is not False or metadata.get('collision_surface_consumed') is not False:
            raise RuntimeError('image factors must not consume generated/collision faces as pose authority')
    if frame_idx.ndim != 1 or len(frame_idx) == 0 or len(np.unique(frame_idx)) != len(frame_idx) or (not np.all(frame_idx[:-1] <= frame_idx[1:])):
        raise RuntimeError('image factor frame_idx must be a sorted unique non-empty vector')
    if T_world_camera.shape != (len(frame_idx), 4, 4) or K_raster.shape != (3, 3):
        raise RuntimeError('invalid image factor camera arrays')
    if depth_offsets.shape != (len(frame_idx) + 1,) or silhouette_offsets.shape != (len(frame_idx) + 1,):
        raise RuntimeError('invalid image factor offset arrays')
    if np.any(np.diff(depth_offsets) < 0) or np.any(np.diff(silhouette_offsets) < 0):
        raise RuntimeError('image factor offsets must be monotonic')
    if depth_offsets[-1] != len(depth_points) or depth_offsets[-1] != len(depth_z) or depth_offsets[-1] != len(depth_weight):
        raise RuntimeError('image factor depth arrays disagree with offsets')
    if silhouette_offsets[-1] != len(silhouette_points) or silhouette_offsets[-1] != len(silhouette_targets) or silhouette_offsets[-1] != len(silhouette_kind):
        raise RuntimeError('image factor silhouette arrays disagree with offsets')
    if K_raster[0, 0] <= 0.0 or K_raster[1, 1] <= 0.0 or (not np.isfinite(K_raster).all()):
        raise RuntimeError('image factor raster intrinsics are invalid')
    if not np.isfinite(T_world_camera).all() or not np.isfinite(depth_points).all() or (not np.isfinite(depth_z).all()) or (not np.isfinite(depth_weight).all()) or (not np.isfinite(silhouette_points).all()) or (not np.isfinite(silhouette_targets).all()) or (not np.isfinite(silhouette_kind).all()):
        raise RuntimeError('image factor arrays contain non-finite values')
    if np.any(depth_weight <= 0.0) or np.any(depth_weight > 1.0 + 1e-06):
        raise RuntimeError('image factor depth weights must be in (0,1]')
    if not np.isin(silhouette_kind, np.asarray([0, 1], dtype=np.int8)).all():
        raise RuntimeError('image factor silhouette_kind must contain only outside=0 or missing=1')
    annotations = formal_object_load_json(args.annotations)
    frames_by_idx = {int(frame['frame_idx']): frame for frame in annotations.get('frames', []) if isinstance(frame, dict) and frame.get('frame_idx') is not None}
    expected_frames = {int(obs.frame_idx) for obs in observations}
    actual_frames = set((int(value) for value in frame_idx.tolist()))
    if actual_frames != expected_frames:
        raise RuntimeError(f'image factor frame contract mismatch: expected {sorted(expected_frames)}, got {sorted(actual_frames)}')
    for (pos, idx) in enumerate(frame_idx.tolist()):
        frame = frames_by_idx.get(int(idx))
        if frame is None:
            raise RuntimeError(f'image factor references missing annotation frame {idx}')
        expected_T = validate_camera_transform(frame['camera']['T_world_camera_metric'], int(idx))
        if not np.allclose(T_world_camera[pos], expected_T, atol=1e-06, rtol=0.0):
            raise RuntimeError(f'image factor camera transform mismatch at frame {idx}')
        obj = annotation_object(frame, str(args.object_id))
        visible = obj.get('visible_geometry_candidate') if isinstance(obj, dict) and isinstance(obj.get('visible_geometry_candidate'), dict) else {}
        values = np.asarray(visible.get('intrinsics_fx_fy_cx_cy') or [], dtype=np.float64)
        if values.shape != (4,):
            raise RuntimeError(f'annotation lacks visible-surface intrinsics at frame {idx}')
        calibration_size = metadata.get('calibration_size_wh') or [int(frame.get('source_width') or 1408), int(frame.get('source_height') or 1408)]
        image_size = metadata.get('image_source_size_wh') or [int(frame.get('manifest_width') or 960), int(frame.get('manifest_height') or 960)]
        raster_size = metadata.get('raster_size_wh') or [int(metadata.get('raster_size') or 0), int(metadata.get('raster_size') or 0)]
        if len(calibration_size) != 2 or len(image_size) != 2 or len(raster_size) != 2:
            raise RuntimeError('image factor plane-size metadata is malformed')
        calibration_size_wh = (int(calibration_size[0]), int(calibration_size[1]))
        image_size_wh = (int(image_size[0]), int(image_size[1]))
        raster_size_wh = (int(raster_size[0]), int(raster_size[1]))
        if min(*calibration_size_wh, *image_size_wh, *raster_size_wh) <= 0:
            raise RuntimeError('image factor plane sizes must be positive')
        expected_K_image = resize_intrinsics_xy(np.asarray([[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]], dtype=np.float64), calibration_size_wh, image_size_wh)
        expected_K = resize_intrinsics_xy(expected_K_image, image_size_wh, raster_size_wh)
        if strict:
            metadata_K_calibration = np.asarray(metadata.get('K_calibration') or [], dtype=np.float64)
            metadata_K_image = np.asarray(metadata.get('K_image_source') or [], dtype=np.float64)
            if metadata_K_calibration.shape != (3, 3) or not np.allclose(metadata_K_calibration, np.asarray([[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]], dtype=np.float64), atol=1e-06, rtol=0.0):
                raise RuntimeError(f'image factor calibration intrinsics metadata mismatch at frame {idx}')
            if metadata_K_image.shape != (3, 3) or not np.allclose(metadata_K_image, expected_K_image, atol=1e-06, rtol=0.0):
                raise RuntimeError(f'image factor image-plane intrinsics metadata mismatch at frame {idx}')
            mask_contract = visible.get('mask_depth_transform_contract') if isinstance(visible.get('mask_depth_transform_contract'), dict) else {}
            declared_affine = np.asarray(mask_contract.get('A_depth_from_mask_coordinate_model') or [], dtype=np.float64)
            expected_affine = np.asarray([[float(calibration_size_wh[0]) / float(image_size_wh[0]), 0.0, 0.5 * float(calibration_size_wh[0]) / float(image_size_wh[0]) - 0.5], [0.0, float(calibration_size_wh[1]) / float(image_size_wh[1]), 0.5 * float(calibration_size_wh[1]) / float(image_size_wh[1]) - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
            if declared_affine.shape != (3, 3) or not np.allclose(declared_affine, expected_affine, atol=1e-06, rtol=0.0):
                raise RuntimeError(f'annotation mask/depth affine contract mismatch at frame {idx}')
        if not np.allclose(K_raster, expected_K, atol=1e-05, rtol=0.0):
            raise RuntimeError(f'image factor raster intrinsics mismatch at frame {idx}')
        obj = annotation_object(frame, str(args.object_id))
        mask_path = Path(str((obj or {}).get('mask_path') or '')).expanduser().resolve()
        if not mask_path.is_file():
            raise RuntimeError(f'image factor annotation mask missing at frame {idx}: {mask_path}')
        if strict:
            expected_mask_hash = str((metadata.get('mask_sha256_by_frame') or {}).get(str(idx)) or '')
            if expected_mask_hash != formal_object_sha256_file(mask_path):
                raise RuntimeError(f'image factor mask hash mismatch at frame {idx}')
            depth_path = Path(str(visible.get('depth_npz') or '')).expanduser().resolve()
            expected_depth_hash = str((metadata.get('depth_npz_sha256_by_path') or {}).get(str(depth_path)) or '')
            if not depth_path.is_file() or expected_depth_hash != formal_object_sha256_file(depth_path):
                raise RuntimeError(f'image factor depth archive hash mismatch at frame {idx}')
    by_frame: dict[int, ImageFactorObservation] = {}
    for (pos, idx) in enumerate(frame_idx.tolist()):
        (d0, d1) = (int(depth_offsets[pos]), int(depth_offsets[pos + 1]))
        (s0, s1) = (int(silhouette_offsets[pos]), int(silhouette_offsets[pos + 1]))
        factor = ImageFactorObservation(frame_idx=int(idx), T_world_camera=T_world_camera[pos], K_raster=K_raster, depth_points_canonical=depth_points[d0:d1], depth_observed_z=depth_z[d0:d1], depth_weight=depth_weight[d0:d1], silhouette_points_canonical=silhouette_points[s0:s1], silhouette_target_uv=silhouette_targets[s0:s1], silhouette_kind=silhouette_kind[s0:s1])
        if factor.depth_points_canonical.ndim != 2 or factor.depth_points_canonical.shape[1] != 3:
            raise RuntimeError(f'invalid depth factors for frame {idx}')
        if factor.silhouette_points_canonical.ndim != 2 or factor.silhouette_points_canonical.shape[1] != 3:
            raise RuntimeError(f'invalid silhouette factors for frame {idx}')
        if factor.silhouette_target_uv.shape != (len(factor.silhouette_points_canonical), 2):
            raise RuntimeError(f'invalid silhouette targets for frame {idx}')
        by_frame[int(idx)] = factor
    metadata_out = {**metadata, 'enabled': True, 'strict_contract_validated': strict, 'factor_npz': str(resolved), 'loaded_frame_count': int(len(by_frame)), 'depth_factor_count': int(sum((len(f.depth_observed_z) for f in by_frame.values()))), 'silhouette_factor_count': int(sum((len(f.silhouette_kind) for f in by_frame.values())))}
    return (by_frame, metadata_out)

def build_observations(args: argparse.Namespace, annotations: dict[str, Any], pose_report: dict[str, Any], mesh: trimesh.Trimesh) -> tuple[list[PoseObservation], list[dict[str, Any]], dict[int, tuple[np.ndarray, float, int]]]:
    import trimesh
    frames_by_idx = {int(frame.get('frame_idx')): frame for frame in annotations.get('frames', []) if isinstance(frame, dict) and frame.get('frame_idx') is not None}
    radius = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)) / 2.0)
    accepted = set(args.nonpenetration_states)
    targets = constraint_targets(args.constraint_report, float(args.max_nonpenetration_target_m), accepted)
    observations: list[PoseObservation] = []
    skipped: list[dict[str, Any]] = []
    for row in pose_report.get('pose_rows', []) if isinstance(pose_report.get('pose_rows'), list) else []:
        if str(row.get('status') or '') not in POSE_MEASUREMENT_STATUSES:
            continue
        idx = int(row['frame_idx'])
        if row.get('rigid_pose_observation_eligible') is False and (not args.include_ineligible_rigid_pose_observations):
            skipped.append({'frame_idx': idx, 'reason': 'explicit upstream rigid_pose_observation_eligible=false', 'policy': 'hard rejected; the production CLI rejects the historical override flag'})
            continue
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        try:
            rot = as_array(row.get('rotation_world_from_completed_canonical_matrix'), (3, 3), f'pose rotation frame {idx}')
            trans = as_array(row.get('translation_world_m'), (3,), f'pose translation frame {idx}')
            frame = frames_by_idx[idx]
            obj = annotation_object(frame, args.object_id)
            if obj is None:
                raise RuntimeError('annotation object missing')
            geom = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else {}
            observed = np.asarray(geom.get('world_vertices_sample_m') or [], dtype=float)
            observed_evidence_source = 'current_frame_accepted_metric_surfels'
            if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < int(args.min_visible_points) or (not np.isfinite(observed).all()):
                tracked_evidence = np.asarray(row.get('direct_pose_evidence_world_points_m') or [], dtype=float)
                tracked_kind = str(row.get('direct_pose_evidence_kind') or '')
                direct_source = str(row.get('direct_pose_observation_source') or '')
                generated_consumed = row.get('generated_geometry_pose_evidence_consumed')
                if direct_source == 'object_owned_rgb_optical_flow_calibrated_pnp' and tracked_kind == 'tracked_source_metric_surfels_at_target_rgb_frame' and (generated_consumed is False) and (tracked_evidence.ndim == 2) and (tracked_evidence.shape[1] == 3) and (len(tracked_evidence) >= int(args.min_visible_points)) and np.isfinite(tracked_evidence).all():
                    observed = tracked_evidence
                    observed_evidence_source = 'source_metric_surfels_tracked_to_current_object_owned_rgb_by_calibrated_pnp'
                else:
                    raise RuntimeError(f'insufficient visible surfels: {observed.shape}')
            (sigma_t, sigma_r) = pose_row_sigma(row, radius, args)
            target_tuple = targets.get(idx)
            if target_tuple is None:
                (target, weight, target_rows) = (None, 0.0, 0)
            else:
                (target, weight, target_rows) = target_tuple
            observations.append(PoseObservation(frame_idx=idx, source_row={**row, 'solver_observed_evidence_source': observed_evidence_source}, rotation_world_from_canonical=rot, translation_world_m=trans, translation_sigma_m=sigma_t, rotation_sigma_rad=sigma_r, visible_sample_count=int(row.get('visible_sample_count') or len(observed)), observed_points_world=observed.astype(float), nonpenetration_target_world_m=target, nonpenetration_weight=weight, nonpenetration_source_rows=target_rows))
        except Exception as exc:
            skipped.append({'frame_idx': idx, 'reason': str(exc)})
    if not observations:
        raise RuntimeError(f'no usable pose observations; skipped={skipped[:8]}')
    measurement_row_count = sum((str(row.get('status') or '') in POSE_MEASUREMENT_STATUSES for row in pose_report.get('pose_rows', []) if isinstance(row, dict)))
    if len(observations) != measurement_row_count:
        raise RuntimeError(f'P15 fail-closed observation validation rejected one or more declared direct P14 rows; accepted={len(observations)} declared={measurement_row_count} skipped={skipped[:8]}')
    if len(observations) < int(args.min_graph_frames) and (not args.complete_full_timeline_rigid_pose):
        raise RuntimeError(f'only {len(observations)} usable pose observations below min_graph_frames={args.min_graph_frames}; uncertain full-timeline completion is disabled; skipped={skipped[:8]}')
    observations.sort(key=lambda obs: obs.frame_idx)
    return (observations, skipped, targets)

def unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    arr = x.reshape(n, 6)
    return (arr[:, :3], arr[:, 3:6])

def image_factors_active(args: argparse.Namespace, image_factors: dict[int, ImageFactorObservation] | None) -> bool:
    return bool(image_factors and (float(args.image_first_hit_weight) > 0.0 or float(args.image_silhouette_weight) > 0.0))

def image_factor_blocks(obs: PoseObservation, factor: ImageFactorObservation | None, rot_delta_i: np.ndarray, trans_delta_i: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    if factor is None:
        return []
    (r, t) = corrected_pose(obs, rot_delta_i, trans_delta_i)
    blocks: list[np.ndarray] = []
    if len(factor.depth_observed_z) and float(args.image_first_hit_weight) > 0.0:
        world = formal_object_apply_pose(factor.depth_points_canonical, r, t)
        camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
        blocks.append(first_hit_depth(camera[:,2],factor.depth_observed_z,factor.depth_weight,sigma=float(args.sigma_image_first_hit_m),clip=float(args.max_image_first_hit_residual_m),weight=float(args.image_first_hit_weight)))
    if len(factor.silhouette_kind) and float(args.image_silhouette_weight) > 0.0:
        world = formal_object_apply_pose(factor.silhouette_points_canonical, r, t)
        camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
        z = np.maximum(camera[:, 2], 1e-09)
        uv = np.column_stack((factor.K_raster[0, 0] * camera[:, 0] / z + factor.K_raster[0, 2], factor.K_raster[1, 1] * camera[:, 1] / z + factor.K_raster[1, 2]))
        blocks.append(silhouette_endpoint_residual(uv,factor.silhouette_target_uv,
            sigma=float(args.sigma_image_silhouette_px),clip=float(args.max_image_silhouette_residual_px),weight=float(args.image_silhouette_weight)))
    return blocks

def residual_vector(x: np.ndarray, observations: list[PoseObservation], args: argparse.Namespace, image_factors: dict[int, ImageFactorObservation] | None=None) -> np.ndarray:
    (rot_delta, trans_delta) = unpack(x, len(observations))
    residuals: list[np.ndarray] = []
    for (i, obs) in enumerate(observations):
        residuals.append(source_pose_translation(trans_delta[i],obs.translation_sigma_m))
        residuals.append(source_pose_rotation(rot_delta[i],obs.rotation_sigma_rad))
        if obs.nonpenetration_target_world_m is not None and obs.nonpenetration_weight > 0.0:
            residuals.append(nonpenetration_translation_target(trans_delta[i],obs.nonpenetration_target_world_m,obs.nonpenetration_weight,float(args.sigma_nonpenetration_target_m)))
        if image_factors:
            residuals.extend(image_factor_blocks(obs, image_factors.get(obs.frame_idx), rot_delta[i], trans_delta[i], args))
    for i in range(1, len(observations)):
        gap = max(1, observations[i].frame_idx - observations[i - 1].frame_idx)
        scale = math.sqrt(float(gap))
        residuals.append(correction_velocity(trans_delta[i],trans_delta[i-1],gap,float(args.sigma_translation_delta_step_m)))
        residuals.append(correction_velocity(rot_delta[i],rot_delta[i-1],gap,float(args.sigma_rotation_delta_step_rad)))
    for i in range(1, len(observations) - 1):
        gap0 = max(1, observations[i].frame_idx - observations[i - 1].frame_idx)
        gap1 = max(1, observations[i + 1].frame_idx - observations[i].frame_idx)
        residuals.append(correction_acceleration(trans_delta[i-1],trans_delta[i],trans_delta[i+1],gap0,gap1,float(args.sigma_translation_delta_accel_m)))
        residuals.append(correction_acceleration(rot_delta[i-1],rot_delta[i],rot_delta[i+1],gap0,gap1,float(args.sigma_rotation_delta_accel_rad)))
    return np.concatenate([r.reshape(-1) for r in residuals]).astype(float)

def residual_sparsity(observations: list[PoseObservation], image_factors: dict[int, ImageFactorObservation] | None=None, args: argparse.Namespace | None=None) -> sparse.csr_matrix:
    n = len(observations)
    cols = n * 6
    entries: list[tuple[int, int]] = []
    row = 0

    def add(rows: range, frame_ids: list[int]) -> None:
        for rr in rows:
            for fi in frame_ids:
                for cc in range(fi * 6, fi * 6 + 6):
                    entries.append((rr, cc))
    for (i, obs) in enumerate(observations):
        add(range(row, row + 3), [i])
        row += 3
        add(range(row, row + 3), [i])
        row += 3
        if obs.nonpenetration_target_world_m is not None and obs.nonpenetration_weight > 0.0:
            add(range(row, row + 3), [i])
            row += 3
        if image_factors and args is not None:
            factor = image_factors.get(obs.frame_idx)
            if factor is not None:
                depth_rows = len(factor.depth_observed_z) if float(args.image_first_hit_weight) > 0.0 else 0
                silhouette_rows = 2 * len(factor.silhouette_kind) if float(args.image_silhouette_weight) > 0.0 else 0
                if depth_rows:
                    add(range(row, row + depth_rows), [i])
                    row += depth_rows
                if silhouette_rows:
                    add(range(row, row + silhouette_rows), [i])
                    row += silhouette_rows
    for i in range(1, n):
        add(range(row, row + 3), [i - 1, i])
        row += 3
        add(range(row, row + 3), [i - 1, i])
        row += 3
    for i in range(1, n - 1):
        add(range(row, row + 3), [i - 1, i, i + 1])
        row += 3
        add(range(row, row + 3), [i - 1, i, i + 1])
        row += 3
    (rr, cc) = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, cols))

def image_factor_metrics(x: np.ndarray, observations: list[PoseObservation], image_factors: dict[int, ImageFactorObservation], args: argparse.Namespace) -> dict[str, Any]:
    if not image_factors_active(args, image_factors):
        return {'enabled': False}
    (rot_delta, trans_delta) = unpack(x, len(observations))
    depth_errors: list[np.ndarray] = []
    depth_weight_values: list[np.ndarray] = []
    outside_errors: list[np.ndarray] = []
    missing_errors: list[np.ndarray] = []
    normalized_blocks: list[np.ndarray] = []
    per_frame: dict[str, Any] = {}
    depth_count = 0
    silhouette_count = 0
    for (i, obs) in enumerate(observations):
        factor = image_factors.get(obs.frame_idx)
        if factor is None:
            continue
        (r, t) = corrected_pose(obs, rot_delta[i], trans_delta[i])
        frame_depth: np.ndarray | None = None
        frame_outside: np.ndarray | None = None
        frame_missing: np.ndarray | None = None
        if len(factor.depth_observed_z):
            world = formal_object_apply_pose(factor.depth_points_canonical, r, t)
            camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
            frame_depth = camera[:, 2] - factor.depth_observed_z
            depth_errors.append(frame_depth)
            depth_weight_values.append(factor.depth_weight)
            depth_count += len(frame_depth)
        if len(factor.silhouette_kind):
            world = formal_object_apply_pose(factor.silhouette_points_canonical, r, t)
            camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
            z = np.maximum(camera[:, 2], 1e-09)
            uv = np.column_stack((factor.K_raster[0, 0] * camera[:, 0] / z + factor.K_raster[0, 2], factor.K_raster[1, 1] * camera[:, 1] / z + factor.K_raster[1, 2]))
            norms = np.linalg.norm(uv - factor.silhouette_target_uv, axis=1)
            outside = norms[factor.silhouette_kind == 0]
            missing = norms[factor.silhouette_kind == 1]
            if len(outside):
                outside_errors.append(outside)
                frame_outside = outside
            if len(missing):
                missing_errors.append(missing)
                frame_missing = missing
            silhouette_count += len(norms)
        blocks = image_factor_blocks(obs, factor, rot_delta[i], trans_delta[i], args)
        if blocks:
            normalized_blocks.append(np.concatenate([block.reshape(-1) for block in blocks]))
        per_frame[str(obs.frame_idx)] = {'first_hit_depth_error_m': formal_object_numeric_summary(frame_depth if frame_depth is not None else []), 'outside_silhouette_error_px': formal_object_numeric_summary(frame_outside if frame_outside is not None else []), 'missing_silhouette_error_px': formal_object_numeric_summary(frame_missing if frame_missing is not None else [])}
    normalized = np.concatenate(normalized_blocks) if normalized_blocks else np.empty(0, dtype=float)
    return {'enabled': True, 'frame_count': int(len(per_frame)), 'depth_factor_count': int(depth_count), 'silhouette_factor_count': int(silhouette_count), 'image_first_hit_weight': float(args.image_first_hit_weight), 'image_silhouette_weight': float(args.image_silhouette_weight), 'first_hit_depth_error_m': formal_object_numeric_summary(np.concatenate(depth_errors) if depth_errors else []), 'depth_factor_weight': formal_object_numeric_summary(np.concatenate(depth_weight_values) if depth_weight_values else []), 'outside_silhouette_error_px': formal_object_numeric_summary(np.concatenate(outside_errors) if outside_errors else []), 'missing_silhouette_error_px': formal_object_numeric_summary(np.concatenate(missing_errors) if missing_errors else []), 'normalized_image_factor_rms': float(np.sqrt(np.mean(normalized * normalized))) if len(normalized) else None, 'per_frame': per_frame}

def target_residual_summary(x: np.ndarray, observations: list[PoseObservation]) -> dict[str, Any]:
    (_, trans_delta) = unpack(x, len(observations))
    vals = []
    norms = []
    for (i, obs) in enumerate(observations):
        if obs.nonpenetration_target_world_m is None or obs.nonpenetration_weight <= 0.0:
            continue
        diff = trans_delta[i] - obs.nonpenetration_target_world_m
        vals.append(float(np.linalg.norm(diff)))
        norms.append(float(np.linalg.norm(obs.nonpenetration_target_world_m)))
    return {'target_residual_norm_m': formal_object_numeric_summary(vals), 'target_norm_m': formal_object_numeric_summary(norms), 'target_frame_count': len(vals)}

def correction_summary(x: np.ndarray, observations: list[PoseObservation]) -> dict[str, Any]:
    (rot_delta, trans_delta) = unpack(x, len(observations))
    gaps = np.asarray([observations[i].frame_idx - observations[i - 1].frame_idx for i in range(1, len(observations))], dtype=float)
    trans_norm = np.linalg.norm(trans_delta, axis=1)
    rot_norm = np.linalg.norm(rot_delta, axis=1)
    step_t = np.linalg.norm(np.diff(trans_delta, axis=0), axis=1) if len(observations) > 1 else np.asarray([])
    step_r = np.linalg.norm(np.diff(rot_delta, axis=0), axis=1) if len(observations) > 1 else np.asarray([])
    return {'translation_delta_norm_m': formal_object_numeric_summary(trans_norm), 'rotation_delta_norm_rad': formal_object_numeric_summary(rot_norm), 'translation_delta_step_norm_m': formal_object_numeric_summary(step_t), 'rotation_delta_step_norm_rad': formal_object_numeric_summary(step_r), 'visible_frame_gap': formal_object_numeric_summary(gaps)}

def formal_object_rotation_observability_summary(observations: list[PoseObservation], minimum_score: float) -> dict[str, Any]:
    rows = []
    for observation in observations:
        centered = observation.observed_points_world - observation.observed_points_world.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        largest = max(float(eigenvalues[0]), 1e-12)
        normalized = eigenvalues / largest
        score = float(min(normalized[0] - normalized[1], normalized[1] - normalized[2]))
        rows.append({'frame_idx': int(observation.frame_idx), 'normalized_covariance_eigenvalues': normalized.astype(float).tolist(), 'rotation_observability_score': score, 'observable': bool(score >= float(minimum_score))})
    observable_count = int(sum((row['observable'] for row in rows)))
    return {'minimum_score': float(minimum_score), 'observable_frame_count': observable_count, 'observation_frame_count': int(len(rows)), 'observable_fraction': float(observable_count / max(1, len(rows))), 'score_summary': formal_object_numeric_summary([row['rotation_observability_score'] for row in rows]), 'rows': rows, 'interpretation': 'Conservative covariance-eigenvalue proxy; symmetric, line-like, or near-planar-isotropic visible support cannot establish all rotation axes.'}

def sparse_conditional_rotation_tail_decision(*, step_rows: list[dict[str, Any]], accepted_rows_by_idx: dict[int, dict[str, Any]], observability: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Bound a tiny underobservable 15--18 degree tail without changing poses.

    The strict 15 degree tier remains the default. The conditional tier can
    only admit adjacent direct metric-surface rows, never interpolated/held/RGB
    or generated-geometry rows, and every excess transition must be sparse,
    translation-continuous, and weakly or marginally rotation-observable.
    """
    strict_limit = float(args.max_rotation_step_deg)
    enabled = bool(getattr(args, 'allow_sparse_conditional_rotation_tail', False))
    conditional_limit = float(getattr(args, 'conditional_max_rotation_step_deg', 18.0))
    maximum_count = int(getattr(args, 'conditional_max_rotation_step_count', 2))
    maximum_fraction = float(getattr(args, 'conditional_max_rotation_step_fraction', 0.015))
    maximum_translation_m = float(getattr(args, 'conditional_max_rotation_step_translation_m', 0.02))
    maximum_endpoint_observability = float(getattr(args, 'conditional_max_endpoint_rotation_observability_score', 0.03))
    numeric_policy = {'strict_max_rotation_step_deg': strict_limit, 'conditional_max_rotation_step_deg': conditional_limit, 'conditional_max_rotation_step_fraction': maximum_fraction, 'conditional_max_translation_step_m': maximum_translation_m, 'conditional_max_endpoint_observability_score': maximum_endpoint_observability}
    if not all((math.isfinite(value) for value in numeric_policy.values())):
        raise RuntimeError(f'non-finite conditional rotation-tail policy: {numeric_policy}')
    if enabled and (not math.isclose(strict_limit, STRICT_ROTATION_STEP_DEG, abs_tol=1e-12)):
        raise RuntimeError(f'conditional rotation tail requires the fixed {STRICT_ROTATION_STEP_DEG:g}-degree strict tier')
    if conditional_limit < strict_limit:
        raise RuntimeError('conditional rotation-step limit must not be below the strict limit')
    if maximum_count < 0 or not 0.0 <= maximum_fraction <= 1.0 or maximum_translation_m < 0.0 or (maximum_endpoint_observability < 0.0):
        raise RuntimeError('invalid conditional rotation-tail bounds')
    configured = {'max_rotation_step_deg': conditional_limit, 'max_count': maximum_count, 'max_fraction': maximum_fraction, 'max_translation_step_m': maximum_translation_m, 'max_endpoint_observability_score': maximum_endpoint_observability}
    broadened = {key: value for (key, value) in configured.items() if value > CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS[key] + 1e-12}
    if broadened:
        raise RuntimeError(f'conditional rotation-tail CLI exceeds immutable policy ceilings: configured={broadened}, ceilings={CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS}')
    observability_by_idx = {int(row['frame_idx']): row for row in observability.get('rows', []) if isinstance(row, dict) and row.get('frame_idx') is not None}
    excess = [row for row in step_rows if float(row.get('rotation_step_deg') or 0.0) > strict_limit]
    evaluated = []
    for step in excess:
        from_idx = int(step['from_frame_idx'])
        to_idx = int(step['to_frame_idx'])
        previous = accepted_rows_by_idx.get(from_idx, {})
        current = accepted_rows_by_idx.get(to_idx, {})
        previous_score = observability_by_idx.get(from_idx, {}).get('rotation_observability_score')
        current_score = observability_by_idx.get(to_idx, {}).get('rotation_observability_score')
        endpoint_scores_available = bool(isinstance(previous_score, (int, float)) and isinstance(current_score, (int, float)) and np.isfinite([previous_score, current_score]).all())
        direct_metric_rows = bool(previous.get('status') == CORRECTED_POSE_STATUS and current.get('status') == CORRECTED_POSE_STATUS and (previous.get('direct_pose_observation_source') == 'adjacent_observed_metric_surfel_registration') and (current.get('direct_pose_observation_source') == 'adjacent_observed_metric_surfel_registration') and (previous.get('rigid_pose_observation_eligible') is True) and (current.get('rigid_pose_observation_eligible') is True))
        generated_geometry_excluded = bool(previous.get('generated_geometry_pose_evidence_consumed') is False and current.get('generated_geometry_pose_evidence_consumed') is False)
        criteria = {'adjacent_frame_step': int(step.get('frame_gap', -1)) == 1, 'within_conditional_rotation_limit': float(step['rotation_step_deg']) <= conditional_limit, 'within_conditional_translation_limit': float(step['translation_step_m']) <= maximum_translation_m, 'direct_adjacent_metric_surface_rows': direct_metric_rows, 'generated_geometry_pose_evidence_excluded': generated_geometry_excluded, 'endpoint_observability_available': endpoint_scores_available, 'endpoints_weak_or_marginally_observable': bool(endpoint_scores_available and float(previous_score) <= maximum_endpoint_observability and (float(current_score) <= maximum_endpoint_observability))}
        evaluated.append({**step, 'from_rotation_observability_score': previous_score, 'to_rotation_observability_score': current_score, 'criteria': criteria, 'transition_eligible': bool(all(criteria.values()))})
    excess_fraction = float(len(excess) / max(1, len(step_rows)))
    population_criteria = {'conditional_tier_enabled': enabled, 'has_strict_exceedance': bool(excess), 'exceedance_count_within_limit': len(excess) <= maximum_count, 'exceedance_fraction_within_limit': excess_fraction <= maximum_fraction, 'all_exceedances_individually_eligible': bool(excess and all((row['transition_eligible'] for row in evaluated)))}
    accepted = bool(all(population_criteria.values()))
    return {'strict_max_rotation_step_deg': strict_limit, 'strict_gate_passed': not excess, 'conditional_tier_enabled': enabled, 'conditional_max_rotation_step_deg': conditional_limit, 'conditional_max_rotation_step_count': maximum_count, 'conditional_max_rotation_step_fraction': maximum_fraction, 'conditional_max_translation_step_m': maximum_translation_m, 'conditional_max_endpoint_rotation_observability_score': maximum_endpoint_observability, 'immutable_policy_ceilings': dict(CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS), 'strict_exceedance_count': int(len(excess)), 'strict_exceedance_fraction': excess_fraction, 'conditional_tier_applied': accepted, 'gate_passed': bool(not excess or accepted), 'acceptance_mode': 'strict_max_rotation_step' if not excess else 'conditional_sparse_underobservable_rotation_tail' if accepted else 'failed_rotation_step_gate', 'population_criteria': population_criteria, 'conditional_transitions': evaluated, 'trajectory_values_modified_or_clipped': False, 'generated_geometry_pose_evidence_consumed': False, 'uncertainty': 'A sparse direct observed-metric rotation tail exceeds the strict 15-degree tier but remains at or below the bounded 18-degree conditional tier. The original trajectory is preserved without clipping; these transitions are explicitly low confidence.' if accepted else None}

def temporal_readiness_diagnostics(*, pose_rows: list[dict[str, Any]], annotations: dict[str, Any], observations: list[PoseObservation], full_timeline_completion: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    timeline = sorted((int(frame['frame_idx']) for frame in annotations.get('frames', []) if isinstance(frame, dict) and frame.get('frame_idx') is not None and (args.frame_start is None or int(frame['frame_idx']) >= int(args.frame_start)) and (args.frame_end is None or int(frame['frame_idx']) <= int(args.frame_end))))
    timeline_set = set(timeline)
    accepted = []
    for row in pose_rows:
        idx = int(row.get('frame_idx', -1))
        if idx not in timeline_set or row.get('status') not in {CORRECTED_POSE_STATUS, COMPLETED_POSE_STATUS}:
            continue
        rotation = np.asarray(row.get('rotation_world_from_completed_canonical_matrix'), dtype=np.float64)
        translation = np.asarray(row.get('translation_world_m'), dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or (not np.isfinite(rotation).all()) or (not np.isfinite(translation).all()):
            continue
        accepted.append((idx, rotation, translation, row))
    accepted.sort(key=lambda value: value[0])
    direct_frames = sorted((int(observation.frame_idx) for observation in observations if observation.frame_idx in timeline_set))
    direct_fraction = float(len(direct_frames) / max(1, len(timeline)))
    internal_gaps = [direct_frames[i] - direct_frames[i - 1] for i in range(1, len(direct_frames))]
    max_direct_gap = max(internal_gaps) if internal_gaps else 0 if direct_frames else None
    leading_gap = direct_frames[0] - timeline[0] if direct_frames and timeline else None
    trailing_gap = timeline[-1] - direct_frames[-1] if direct_frames and timeline else None
    rotation_steps_deg = []
    translation_steps_m = []
    step_rows = []
    for (previous, current) in zip(accepted[:-1], accepted[1:]):
        (prev_idx, prev_rotation, prev_translation, _) = previous
        (idx, rotation, translation, _) = current
        relative = rotation @ prev_rotation.T
        rotation_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
        translation_m = float(np.linalg.norm(translation - prev_translation))
        rotation_steps_deg.append(rotation_deg)
        translation_steps_m.append(translation_m)
        step_rows.append({'from_frame_idx': int(prev_idx), 'to_frame_idx': int(idx), 'frame_gap': int(idx - prev_idx), 'rotation_step_deg': rotation_deg, 'translation_step_m': translation_m})
    mode_counts = full_timeline_completion.get('mode_counts') if isinstance(full_timeline_completion.get('mode_counts'), dict) else {}
    completed_count = int(full_timeline_completion.get('completed_row_count') or 0)
    hold_count = int(mode_counts.get('nearest_visible_pose_hold') or 0)
    completed_fraction = float(completed_count / max(1, len(timeline)))
    hold_fraction = float(hold_count / max(1, len(timeline)))
    observability = formal_object_rotation_observability_summary(observations, float(args.min_rotation_observability_score))
    accepted_rows_by_idx = {idx: row for (idx, _rotation, _translation, row) in accepted}
    rotation_step_gate = sparse_conditional_rotation_tail_decision(step_rows=step_rows, accepted_rows_by_idx=accepted_rows_by_idx, observability=observability, args=args)
    max_rotation_step = max(rotation_steps_deg) if rotation_steps_deg else 0.0
    max_translation_step = max(translation_steps_m) if translation_steps_m else 0.0
    failure_reasons = []
    if len(accepted) != len(timeline):
        failure_reasons.append('incomplete_full_timeline_pose_rows')
    if direct_fraction < float(args.min_direct_pose_fraction):
        failure_reasons.append('insufficient_direct_pose_fraction')
    if max_direct_gap is None or max_direct_gap > int(args.max_direct_pose_gap_frames):
        failure_reasons.append('direct_pose_gap_too_large')
    if leading_gap is None or leading_gap > int(args.max_rigid_pose_extrapolation_gap_frames):
        failure_reasons.append('leading_pose_extrapolation_too_large')
    if trailing_gap is None or trailing_gap > int(args.max_rigid_pose_extrapolation_gap_frames):
        failure_reasons.append('trailing_pose_extrapolation_too_large')
    if completed_fraction > float(args.max_completed_pose_fraction):
        failure_reasons.append('completed_pose_fraction_too_large')
    if hold_fraction > float(args.max_nearest_hold_fraction):
        failure_reasons.append('nearest_hold_fraction_too_large')
    if not rotation_step_gate['gate_passed']:
        failure_reasons.append('rotation_step_jump')
    if max_translation_step > float(args.max_translation_step_m):
        failure_reasons.append('translation_step_jump')
    if observability['observable_fraction'] < float(args.min_rotation_observable_fraction):
        failure_reasons.append('insufficient_rotation_observability')
    return {'ready': not failure_reasons, 'failure_reasons': failure_reasons, 'timeline_frame_count': int(len(timeline)), 'accepted_full_timeline_pose_count': int(len(accepted)), 'direct_pose_count': int(len(direct_frames)), 'direct_pose_fraction': direct_fraction, 'direct_frame_span': [direct_frames[0], direct_frames[-1]] if direct_frames else None, 'max_direct_frame_gap': max_direct_gap, 'leading_direct_frame_gap': leading_gap, 'trailing_direct_frame_gap': trailing_gap, 'completed_pose_count': completed_count, 'completed_pose_fraction': completed_fraction, 'nearest_hold_count': hold_count, 'nearest_hold_fraction': hold_fraction, 'completion_mode_counts': mode_counts, 'rotation_step_deg': formal_object_numeric_summary(rotation_steps_deg), 'translation_step_m': formal_object_numeric_summary(translation_steps_m), 'max_rotation_step_deg': max_rotation_step, 'max_translation_step_m': max_translation_step, 'rotation_observability': observability, 'rotation_step_gate': rotation_step_gate, 'conditional_temporal_uncertainty': rotation_step_gate.get('uncertainty'), 'thresholds': {'min_direct_pose_fraction': float(args.min_direct_pose_fraction), 'max_direct_pose_gap_frames': int(args.max_direct_pose_gap_frames), 'max_completed_pose_fraction': float(args.max_completed_pose_fraction), 'max_nearest_hold_fraction': float(args.max_nearest_hold_fraction), 'max_rotation_step_deg': float(args.max_rotation_step_deg), 'allow_sparse_conditional_rotation_tail': bool(getattr(args, 'allow_sparse_conditional_rotation_tail', False)), 'conditional_max_rotation_step_deg': float(getattr(args, 'conditional_max_rotation_step_deg', 18.0)), 'conditional_max_rotation_step_count': int(getattr(args, 'conditional_max_rotation_step_count', 2)), 'conditional_max_rotation_step_fraction': float(getattr(args, 'conditional_max_rotation_step_fraction', 0.015)), 'conditional_max_rotation_step_translation_m': float(getattr(args, 'conditional_max_rotation_step_translation_m', 0.02)), 'conditional_max_endpoint_rotation_observability_score': float(getattr(args, 'conditional_max_endpoint_rotation_observability_score', 0.03)), 'max_translation_step_m': float(args.max_translation_step_m), 'min_rotation_observable_fraction': float(args.min_rotation_observable_fraction), 'min_rotation_observability_score': float(args.min_rotation_observability_score)}, 'step_rows': step_rows}

def surface_metrics(observations: list[PoseObservation], mesh_samples: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    (rot_delta, trans_delta) = unpack(x, len(observations))
    obs_to_mesh_medians: list[float] = []
    mesh_to_obs_medians: list[float] = []
    per_frame: dict[str, Any] = {}
    for (i, obs) in enumerate(observations):
        (r, t) = corrected_pose(obs, rot_delta[i], trans_delta[i])
        mesh_world = formal_object_apply_pose(mesh_samples, r, t)
        o2m = formal_object_nearest_summary(obs.observed_points_world, mesh_world)
        m2o = formal_object_nearest_summary(mesh_world, obs.observed_points_world)
        if o2m['median_m'] is not None:
            obs_to_mesh_medians.append(float(o2m['median_m']))
        if m2o['median_m'] is not None:
            mesh_to_obs_medians.append(float(m2o['median_m']))
        per_frame[str(obs.frame_idx)] = {'observed_to_mesh': o2m, 'mesh_to_observed': m2o}
    return {'observed_to_mesh_median_m': formal_object_numeric_summary(obs_to_mesh_medians), 'mesh_to_observed_median_m': formal_object_numeric_summary(mesh_to_obs_medians), 'per_frame': per_frame}

def build_pose_rows(original_pose_rows: list[dict[str, Any]], observations: list[PoseObservation], x: np.ndarray, args: argparse.Namespace, graph_support_sufficient: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs_by_idx = {obs.frame_idx: i for (i, obs) in enumerate(observations)}
    (rot_delta, trans_delta) = unpack(x, len(observations))
    corrected_by_idx: dict[int, tuple[np.ndarray, np.ndarray, PoseObservation, np.ndarray, np.ndarray]] = {}
    out: list[dict[str, Any]] = []
    for row in original_pose_rows:
        idx = int(row.get('frame_idx', -1))
        if idx not in obs_by_idx:
            out.append(dict(row))
            continue
        i = obs_by_idx[idx]
        obs = observations[i]
        (r, t) = corrected_pose(obs, rot_delta[i], trans_delta[i])
        corrected_by_idx[idx] = (r, t, obs, rot_delta[i], trans_delta[i])
        new_row = dict(row)
        new_row['status'] = CORRECTED_POSE_STATUS
        new_row['pose_measurement_status'] = row.get('status')
        new_row['rotation_world_from_completed_canonical_matrix'] = r.astype(float).tolist()
        new_row['translation_world_m'] = t.astype(float).tolist()
        new_row['graph_support_sufficient'] = graph_support_sufficient
        new_row['temporal_pose_graph'] = {'pose_source': 'direct_visible_pose_observation_corrected', 'direct_visible_measurement': True, 'rotation_delta_rotvec_rad': rot_delta[i].astype(float).tolist(), 'translation_delta_world_m': trans_delta[i].astype(float).tolist(), 'translation_prior_sigma_m': float(obs.translation_sigma_m), 'rotation_prior_sigma_rad': float(obs.rotation_sigma_rad), 'nonpenetration_target_world_m': None if obs.nonpenetration_target_world_m is None else obs.nonpenetration_target_world_m.astype(float).tolist(), 'nonpenetration_weight': float(obs.nonpenetration_weight), 'nonpenetration_source_rows': int(obs.nonpenetration_source_rows), 'graph_support_sufficient': graph_support_sufficient, 'uncertainty': None if graph_support_sufficient else 'direct visible observation exists, but trusted graph-frame count is below the configured support minimum'}
        out.append(new_row)
    summary: dict[str, Any] = {'enabled': bool(args.complete_full_timeline_rigid_pose), 'graph_support_sufficient': graph_support_sufficient, 'completed_row_count': 0, 'direct_row_count': int(len(corrected_by_idx)), 'mode_counts': {}, 'max_interpolation_gap_frames': int(args.max_rigid_pose_interpolation_gap_frames), 'max_extrapolation_gap_frames': int(args.max_rigid_pose_extrapolation_gap_frames)}
    if not args.complete_full_timeline_rigid_pose or not corrected_by_idx:
        return (out, summary)
    key_frames = sorted(corrected_by_idx)
    key_rots = Rotation.from_matrix([corrected_by_idx[idx][0] for idx in key_frames])
    slerp = Slerp(np.asarray(key_frames, dtype=float), key_rots) if len(key_frames) >= 2 else None
    key_trans = np.asarray([corrected_by_idx[idx][1] for idx in key_frames], dtype=float)

    def add_mode(mode: str) -> None:
        counts = summary.setdefault('mode_counts', {})
        counts[mode] = int(counts.get(mode, 0)) + 1
    for row in out:
        idx = int(row.get('frame_idx', -1))
        if idx in corrected_by_idx or idx < 0:
            continue
        before = [f for f in key_frames if f < idx]
        after = [f for f in key_frames if f > idx]
        mode: str | None = None
        r_fill: np.ndarray | None = None
        t_fill: np.ndarray | None = None
        gap_frames: int | None = None
        bracket: list[int] | None = None
        if before and after:
            lo = before[-1]
            hi = after[0]
            gap_frames = int(hi - lo)
            if gap_frames <= int(args.max_rigid_pose_interpolation_gap_frames) and slerp is not None:
                alpha = float(idx - lo) / float(max(1, hi - lo))
                r_fill = slerp([float(idx)]).as_matrix()[0]
                t_lo = key_trans[key_frames.index(lo)]
                t_hi = key_trans[key_frames.index(hi)]
                t_fill = (1.0 - alpha) * t_lo + alpha * t_hi
                mode = 'interpolated_between_visible_pose_observations'
                bracket = [int(lo), int(hi)]
        if mode is None:
            nearest = min(key_frames, key=lambda f: abs(f - idx))
            dist = abs(nearest - idx)
            if dist <= int(args.max_rigid_pose_extrapolation_gap_frames):
                r_fill = corrected_by_idx[nearest][0]
                t_fill = corrected_by_idx[nearest][1]
                gap_frames = int(dist)
                mode = 'nearest_visible_pose_hold'
                bracket = [int(nearest)]
        if mode is None or r_fill is None or t_fill is None:
            continue
        old_status = row.get('status')
        row['status'] = COMPLETED_POSE_STATUS
        row['pose_measurement_status'] = old_status
        row['rotation_world_from_completed_canonical_matrix'] = r_fill.astype(float).tolist()
        row['translation_world_m'] = t_fill.astype(float).tolist()
        row['graph_support_sufficient'] = graph_support_sufficient
        row['temporal_pose_graph'] = {'pose_source': mode, 'bracket_visible_pose_frames': bracket, 'gap_frames': gap_frames, 'direct_visible_measurement': False, 'graph_support_sufficient': graph_support_sufficient, 'uncertainty': 'rigid-body temporal completion; not a direct object mask/depth observation' if graph_support_sufficient else 'insufficient trusted graph support; sparse-cluster interpolation/nearest hold is an unresolved trajectory hypothesis'}
        summary['completed_row_count'] = int(summary['completed_row_count']) + 1
        add_mode(mode)
    return (out, summary)

def fit_formal_object(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    annotations = formal_object_load_json(args.annotations)
    pose_report = formal_object_load_json(args.pose_report)
    pose_measurement_candidates = [row for row in (pose_report.get('pose_rows', []) if isinstance(pose_report.get('pose_rows'), list) else []) if isinstance(row, dict) and str(row.get('status') or '') in POSE_MEASUREMENT_STATUSES and (args.frame_start is None or int(row.get('frame_idx', -1)) >= int(args.frame_start)) and (args.frame_end is None or int(row.get('frame_idx', -1)) <= int(args.frame_end))]
    completion = formal_object_load_json(args.completion_report) if args.completion_report else {}
    if args.completion_report:
        pose_binding = pose_report.get('selected_anchor_atomic_binding')
        completion_binding = (completion.get('inputs') or {}).get('selected_anchor_atomic_binding') if isinstance(completion.get('inputs'), dict) else None
        pose_evidence_sha256 = str(pose_report.get('selected_anchor_evidence_report_sha256') or '')
        completion_evidence_sha256 = str((completion.get('inputs') or {}).get('candidate_evidence_report_sha256') or '')
        if not isinstance(pose_binding, dict) or pose_binding.get('validated') is not True or pose_binding != completion_binding or (not pose_evidence_sha256) or (pose_evidence_sha256 != completion_evidence_sha256):
            raise RuntimeError('P15 pose/completion reports do not share one byte-bound atomic selected anchor')
    completion_outputs = completion.get('outputs') if isinstance(completion.get('outputs'), dict) else {}
    completion_geometry_readiness = completion.get('geometry_readiness') if isinstance(completion.get('geometry_readiness'), dict) else {}
    completion_collision_mesh = completion_outputs.get('collision_eligible_mesh_labeled')
    completion_pose_mesh = completion_outputs.get('pose_hypothesis_mesh_labeled') or completion_outputs.get('completed_mesh_labeled')
    mesh_path = args.completed_mesh or Path(completion_pose_mesh or '')
    if not mesh_path:
        raise RuntimeError('pose-hypothesis mesh path missing; pass --completed-mesh or --completion-report')
    pose_mesh_semantics = 'explicit_cli_mesh' if args.completed_mesh is not None else 'pose_hypothesis_mesh_labeled' if completion_outputs.get('pose_hypothesis_mesh_labeled') else 'legacy_completed_mesh_labeled'
    mesh = formal_object_load_mesh(Path(mesh_path))
    (observations, skipped, targets) = build_observations(args, annotations, pose_report, mesh)
    (image_factors, image_factor_metadata) = load_image_factor_observations(args.image_factor_npz, observations, args, Path(mesh_path))
    if args.image_factor_npz is not None and (not image_factors_active(args, image_factors)):
        raise RuntimeError('--image-factor-npz was provided but both image factor weights are zero')
    graph_support_sufficient = len(observations) >= int(args.min_graph_frames)
    x0 = np.zeros(len(observations) * 6, dtype=float)
    before = residual_vector(x0, observations, args, image_factors)
    jac = residual_sparsity(observations, image_factors, args)
    if jac.shape != (len(before), len(x0)):
        raise RuntimeError(f'sparsity shape {jac.shape} != residual/vector {(len(before), len(x0))}')
    result = least_squares(lambda x: residual_vector(x, observations, args, image_factors), x0, jac_sparsity=jac, max_nfev=int(args.max_nfev), loss='soft_l1', f_scale=1.0, x_scale='jac', verbose=2 if args.verbose else 0)
    after = residual_vector(result.x, observations, args, image_factors)
    mesh_samples = formal_object_deterministic_sample_mesh(mesh, int(args.surface_metric_sample_count), int(args.seed) + 73)
    before_surface = surface_metrics(observations, mesh_samples, x0)
    after_surface = surface_metrics(observations, mesh_samples, result.x)
    before_target = target_residual_summary(x0, observations)
    after_target = target_residual_summary(result.x, observations)
    before_image = image_factor_metrics(x0, observations, image_factors, args)
    after_image = image_factor_metrics(result.x, observations, image_factors, args)
    (pose_rows, full_timeline_completion) = build_pose_rows(pose_report.get('pose_rows', []), observations, result.x, args, graph_support_sufficient)
    temporal_readiness = temporal_readiness_diagnostics(pose_rows=pose_rows, annotations=annotations, observations=observations, full_timeline_completion=full_timeline_completion, args=args)
    conditional_transitions = temporal_readiness.get('rotation_step_gate', {}).get('conditional_transitions', []) if temporal_readiness.get('rotation_step_gate', {}).get('conditional_tier_applied') is True else []
    pose_rows_by_idx = {int(row['frame_idx']): row for row in pose_rows if isinstance(row, dict) and row.get('frame_idx') is not None}
    for transition in conditional_transitions:
        row = pose_rows_by_idx.get(int(transition['to_frame_idx']))
        if row is not None:
            row['conditional_rotation_step_uncertainty'] = {'acceptance_mode': 'conditional_sparse_underobservable_rotation_tail', 'from_frame_idx': int(transition['from_frame_idx']), 'rotation_step_deg': float(transition['rotation_step_deg']), 'translation_step_m': float(transition['translation_step_m']), 'trajectory_values_modified_or_clipped': False, 'generated_geometry_pose_evidence_consumed': False, 'interpretation': 'Direct observed-metric pose retained unchanged under the bounded sparse rotation-tail tier.'}
    correction_is_exact_zero = bool(np.array_equal(result.x, np.zeros_like(result.x)))
    image_terms_active = image_factors_active(args, image_factors)
    no_active_nonpenetration_targets = not any((observation.nonpenetration_target_world_m is not None and observation.nonpenetration_weight > 0.0 for observation in observations))
    optimization_effect = {'correction_is_exact_zero': correction_is_exact_zero, 'no_active_nonpenetration_targets': no_active_nonpenetration_targets, 'image_factors_active': image_terms_active, 'zero_correction_is_structural_objective_minimum': bool(correction_is_exact_zero and no_active_nonpenetration_targets and (not image_terms_active)), 'physical_trajectory_smoothed_directly': False, 'interpretation': 'The objective regularizes correction deltas. Optional first-hit/silhouette factors are frozen local image correspondences re-linearized under each per-frame SE(3) correction; without nonzero external targets or image factors, all-zero deltas are the exact minimum. Temporal readiness must therefore be established from direct coverage, observability, and SE(3) jump diagnostics, not optimizer success.'}
    surface_before_med = before_surface['observed_to_mesh_median_m']['median']
    surface_after_med = after_surface['observed_to_mesh_median_m']['median']
    target_before_med = before_target['target_residual_norm_m']['median']
    target_after_med = after_target['target_residual_norm_m']['median']
    surface_degraded_m = None
    if surface_before_med is not None and surface_after_med is not None:
        surface_degraded_m = float(surface_after_med) - float(surface_before_med)
    target_improved = target_before_med is not None and target_after_med is not None and (float(target_after_med) < float(target_before_med))
    before_image_rms = before_image.get('normalized_image_factor_rms') if isinstance(before_image, dict) else None
    after_image_rms = after_image.get('normalized_image_factor_rms') if isinstance(after_image, dict) else None
    image_improved = before_image_rms is not None and after_image_rms is not None and (float(after_image_rms) < float(before_image_rms))
    surface_preserved = surface_degraded_m is None or surface_degraded_m <= float(args.max_surface_median_degradation_m)
    if not graph_support_sufficient:
        status = 'completed_uncertain_insufficient_trusted_pose_graph_support'
    elif not temporal_readiness['ready']:
        status = 'completed_unready_temporal_coverage_observability_or_se3_jump'
    elif result.success and surface_preserved and image_terms_active:
        status = 'corrected_pose_graph_surface_preserved_image_factors_improved' if image_improved else 'corrected_pose_graph_surface_preserved_no_image_factor_gain'
    elif result.success and surface_preserved and target_improved:
        status = 'corrected_pose_graph_surface_preserved_nonpenetration_pressure_improved'
    elif result.success and surface_preserved:
        status = 'corrected_pose_graph_surface_preserved_no_nonpenetration_gain'
    elif result.success:
        status = 'corrected_pose_graph_surface_degraded_untrusted'
    else:
        status = 'corrected_pose_graph_optimizer_incomplete'
    annotation_ready = bool(result.success and surface_preserved and graph_support_sufficient and temporal_readiness['ready'])
    rotation_acceptance_mode = str(temporal_readiness.get('rotation_step_gate', {}).get('acceptance_mode') or 'unknown_rotation_step_gate')
    annotation_readiness_mode = 'conditional_sparse_underobservable_rotation_tail' if annotation_ready and temporal_readiness.get('rotation_step_gate', {}).get('conditional_tier_applied') is True else 'strict_temporal_readiness' if annotation_ready else 'not_annotation_ready'
    for row in pose_rows:
        if row.get('status') in {CORRECTED_POSE_STATUS, COMPLETED_POSE_STATUS}:
            row['annotation_ready'] = annotation_ready
            row['annotation_readiness_mode'] = annotation_readiness_mode
    report = {'method': 'solve_v19_rigid_object_pose_graph', 'status': status, 'annotation_ready': annotation_ready, 'annotation_readiness_mode': annotation_readiness_mode, 'rotation_step_acceptance_mode': rotation_acceptance_mode, 'conditional_temporal_uncertainty': temporal_readiness.get('conditional_temporal_uncertainty'), 'claim_scope': 'Temporal rigid-object pose correction over eligible visible-frame SE(3) measurements. Visible-surface ICP rows are pose observations; nonpenetration rows exert only clipped soft pressure. Optional first-hit/silhouette rows are frozen local image correspondences over the observed-only pose-hypothesis mesh, with hand projection treated as unknown support. Annotation readiness additionally requires full timeline coverage, direct-observation density, bounded gaps/holds, conservative rotation observability, and bounded per-frame SE(3) steps. A named sparse conditional rotation tail is explicit uncertainty, not clipping or ground-truth angular velocity. Optimizer success alone is insufficient.', 'object_id': args.object_id, 'inputs': {'annotations': str(args.annotations), 'pose_report': str(args.pose_report), 'completion_report': str(args.completion_report) if args.completion_report else None, 'completed_mesh': str(mesh_path), 'pose_mesh_semantics': pose_mesh_semantics, 'collision_eligible_mesh_not_used_as_pose_body': completion_collision_mesh, 'completion_geometry_readiness': completion_geometry_readiness, 'constraint_report': str(args.constraint_report) if args.constraint_report else None, 'image_factor_npz': str(args.image_factor_npz) if args.image_factor_npz else None, 'image_factor_strict_contract_requested': bool(args.require_strict_image_factor_contract), 'image_factor_metadata': image_factor_metadata}, 'geometry_contract': {'pose_hypothesis_mesh': str(mesh_path), 'pose_mesh_semantics': pose_mesh_semantics, 'collision_eligible_mesh': completion_collision_mesh, 'completion_geometry_readiness': completion_geometry_readiness, 'pose_graph_consumes_collision_surface': False, 'claim_scope': 'P15 estimates temporal SE(3) in the P13 pose-hypothesis canonical frame. Collision/sign readiness is recorded for P16/P15b but does not replace the pose body or turn completion rows into observations.'}, 'graph_frames': [obs.frame_idx for obs in observations], 'graph_frame_count': int(len(observations)), 'graph_support': {'sufficient': graph_support_sufficient, 'configured_min_graph_frames': int(args.min_graph_frames), 'actual_graph_frames': int(len(observations)), 'frame_span': [int(observations[0].frame_idx), int(observations[-1].frame_idx)], 'continued_only_as_uncertain_full_timeline_completion': bool(not graph_support_sufficient and args.complete_full_timeline_rigid_pose)}, 'pose_observation_eligibility_policy': {'explicit_false': 'included_only_with_override' if args.include_ineligible_rigid_pose_observations else 'hard_rejected', 'missing_field': 'allowed_for_legacy_compatibility', 'include_ineligible_override': bool(args.include_ineligible_rigid_pose_observations), 'candidate_measurement_row_count': len(pose_measurement_candidates), 'explicit_eligible_candidate_count': int(sum((row.get('rigid_pose_observation_eligible') is True for row in pose_measurement_candidates))), 'eligibility_unspecified_candidate_count': int(sum((not isinstance(row.get('rigid_pose_observation_eligible'), bool) for row in pose_measurement_candidates))), 'explicit_ineligible_candidate_count': int(sum((row.get('rigid_pose_observation_eligible') is False for row in pose_measurement_candidates))), 'explicit_ineligible_skipped_count': int(sum((row.get('reason') == 'explicit upstream rigid_pose_observation_eligible=false' for row in skipped))), 'explicit_ineligible_admitted_count': int(sum((obs.source_row.get('rigid_pose_observation_eligible') is False for obs in observations)))}, 'skipped_pose_observations': skipped, 'nonpenetration_target_frame_count': int(sum((1 for obs in observations if obs.nonpenetration_target_world_m is not None))), 'nonpenetration_target_source_frame_count': int(len(targets)), 'parameters': {'max_nonpenetration_target_m': float(args.max_nonpenetration_target_m), 'sigma_nonpenetration_target_m': float(args.sigma_nonpenetration_target_m), 'sigma_translation_delta_step_m': float(args.sigma_translation_delta_step_m), 'sigma_rotation_delta_step_rad': float(args.sigma_rotation_delta_step_rad), 'sigma_translation_delta_accel_m': float(args.sigma_translation_delta_accel_m), 'sigma_rotation_delta_accel_rad': float(args.sigma_rotation_delta_accel_rad), 'max_surface_median_degradation_m': float(args.max_surface_median_degradation_m), 'min_graph_frames': int(args.min_graph_frames), 'nonpenetration_states': list(args.nonpenetration_states), 'complete_full_timeline_rigid_pose': bool(args.complete_full_timeline_rigid_pose), 'max_rigid_pose_interpolation_gap_frames': int(args.max_rigid_pose_interpolation_gap_frames), 'max_rigid_pose_extrapolation_gap_frames': int(args.max_rigid_pose_extrapolation_gap_frames), 'min_direct_pose_fraction': float(args.min_direct_pose_fraction), 'max_direct_pose_gap_frames': int(args.max_direct_pose_gap_frames), 'max_completed_pose_fraction': float(args.max_completed_pose_fraction), 'max_nearest_hold_fraction': float(args.max_nearest_hold_fraction), 'max_rotation_step_deg': float(args.max_rotation_step_deg), 'allow_sparse_conditional_rotation_tail': bool(args.allow_sparse_conditional_rotation_tail), 'conditional_max_rotation_step_deg': float(args.conditional_max_rotation_step_deg), 'conditional_max_rotation_step_count': int(args.conditional_max_rotation_step_count), 'conditional_max_rotation_step_fraction': float(args.conditional_max_rotation_step_fraction), 'conditional_max_rotation_step_translation_m': float(args.conditional_max_rotation_step_translation_m), 'conditional_max_endpoint_rotation_observability_score': float(args.conditional_max_endpoint_rotation_observability_score), 'max_translation_step_m': float(args.max_translation_step_m), 'min_rotation_observable_fraction': float(args.min_rotation_observable_fraction), 'min_rotation_observability_score': float(args.min_rotation_observability_score), 'image_first_hit_weight': float(args.image_first_hit_weight), 'image_silhouette_weight': float(args.image_silhouette_weight), 'sigma_image_first_hit_m': float(args.sigma_image_first_hit_m), 'sigma_image_silhouette_px': float(args.sigma_image_silhouette_px), 'max_image_first_hit_residual_m': float(args.max_image_first_hit_residual_m), 'max_image_silhouette_residual_px': float(args.max_image_silhouette_residual_px), 'require_strict_image_factor_contract': bool(args.require_strict_image_factor_contract)}, 'optimizer': {'success': bool(result.success), 'message': str(result.message), 'nfev': int(result.nfev), 'cost': float(result.cost), 'residual_rms_before': float(np.sqrt(np.mean(before * before))), 'residual_rms_after': float(np.sqrt(np.mean(after * after)))}, 'correction_summary': correction_summary(result.x, observations), 'optimization_effect': optimization_effect, 'temporal_readiness': temporal_readiness, 'nonpenetration_target_before': before_target, 'nonpenetration_target_after': after_target, 'image_factor_before': before_image, 'image_factor_after': after_image, 'image_factor_rms_improvement': float(before_image_rms) - float(after_image_rms) if before_image_rms is not None and after_image_rms is not None else None, 'surface_before': before_surface, 'surface_after': after_surface, 'surface_observed_to_mesh_median_degradation_m': surface_degraded_m, 'full_timeline_rigid_pose_completion': full_timeline_completion, 'pose_rows': pose_rows, 'outputs': {'pose_graph_report': str(args.output_dir / 'v19_rigid_object_pose_graph_report.json'), 'v18_compatible_pose_report': str(args.output_dir / 'v18_compact_rigid_object_pose_fit_report.json')}, 'elapsed_s': float(time.time() - started)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'v19_rigid_object_pose_graph_report.json', report)
    write_json(args.output_dir / 'v18_compact_rigid_object_pose_fit_report.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'annotation_ready', 'graph_frame_count', 'nonpenetration_target_frame_count', 'optimizer', 'correction_summary', 'optimization_effect', 'temporal_readiness', 'full_timeline_rigid_pose_completion', 'nonpenetration_target_before', 'nonpenetration_target_after', 'image_factor_before', 'image_factor_after', 'image_factor_rms_improvement', 'surface_observed_to_mesh_median_degradation_m']}, indent=2))
    return report

def formal_object_parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--pose-report', type=Path, required=True)
    p.add_argument('--completion-report', type=Path, default=None)
    p.add_argument('--completed-mesh', type=Path, default=None)
    p.add_argument('--constraint-report', type=Path, default=None)
    p.add_argument('--image-factor-npz', type=Path, default=None)
    p.add_argument('--require-strict-image-factor-contract', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--image-first-hit-weight', type=float, default=1.0)
    p.add_argument('--image-silhouette-weight', type=float, default=1.0)
    p.add_argument('--sigma-image-first-hit-m', type=float, default=0.008)
    p.add_argument('--sigma-image-silhouette-px', type=float, default=4.0)
    p.add_argument('--max-image-first-hit-residual-m', type=float, default=0.03)
    p.add_argument('--max-image-silhouette-residual-px', type=float, default=16.0)
    p.add_argument('--object-id', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frame-start', type=int, default=None)
    p.add_argument('--frame-end', type=int, default=None)
    p.add_argument('--min-graph-frames', type=int, default=8)
    p.add_argument('--include-ineligible-rigid-pose-observations', action='store_true', help='Forbidden production flag retained only so historical commands fail closed explicitly')
    p.add_argument('--min-visible-points', type=int, default=20)
    p.add_argument('--min-pose-sigma-m', type=float, default=0.004)
    p.add_argument('--max-pose-sigma-m', type=float, default=0.045)
    p.add_argument('--default-pose-sigma-m', type=float, default=0.018)
    p.add_argument('--min-pose-rotation-sigma-rad', type=float, default=0.035)
    p.add_argument('--max-pose-rotation-sigma-rad', type=float, default=0.35)
    p.add_argument('--max-nonpenetration-target-m', type=float, default=0.015)
    p.add_argument('--sigma-nonpenetration-target-m', type=float, default=0.03)
    p.add_argument('--nonpenetration-states', nargs='+', default=['candidate_coordinate_correction_visible_2d_compatible'])
    p.add_argument('--sigma-translation-delta-step-m', type=float, default=0.01)
    p.add_argument('--sigma-rotation-delta-step-rad', type=float, default=0.08)
    p.add_argument('--sigma-translation-delta-accel-m', type=float, default=0.006)
    p.add_argument('--sigma-rotation-delta-accel-rad', type=float, default=0.05)
    p.add_argument('--max-surface-median-degradation-m', type=float, default=0.003)
    p.add_argument('--complete-full-timeline-rigid-pose', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-rigid-pose-interpolation-gap-frames', type=int, default=10)
    p.add_argument('--max-rigid-pose-extrapolation-gap-frames', type=int, default=10)
    p.add_argument('--min-direct-pose-fraction', type=float, default=0.8)
    p.add_argument('--max-direct-pose-gap-frames', type=int, default=10)
    p.add_argument('--max-completed-pose-fraction', type=float, default=0.2)
    p.add_argument('--max-nearest-hold-fraction', type=float, default=0.05)
    p.add_argument('--max-rotation-step-deg', type=float, default=15.0)
    p.add_argument('--allow-sparse-conditional-rotation-tail', action=argparse.BooleanOptionalAction, default=False, help='Allow only the bounded, explicitly uncertain 15--18 degree direct metric-surface tail.')
    p.add_argument('--conditional-max-rotation-step-deg', type=float, default=18.0)
    p.add_argument('--conditional-max-rotation-step-count', type=int, default=2)
    p.add_argument('--conditional-max-rotation-step-fraction', type=float, default=0.015)
    p.add_argument('--conditional-max-rotation-step-translation-m', type=float, default=0.02)
    p.add_argument('--conditional-max-endpoint-rotation-observability-score', type=float, default=0.03)
    p.add_argument('--max-translation-step-m', type=float, default=0.05)
    p.add_argument('--min-rotation-observable-fraction', type=float, default=0.8)
    p.add_argument('--min-rotation-observability-score', type=float, default=0.02)
    p.add_argument('--surface-metric-sample-count', type=int, default=2500)
    p.add_argument('--max-nfev', type=int, default=80)
    p.add_argument('--seed', type=int, default=1907)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)
    if args.include_ineligible_rigid_pose_observations:
        raise RuntimeError('--include-ineligible-rigid-pose-observations is forbidden by the production observed-only pose contract')
    return args
