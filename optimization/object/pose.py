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
from factors.object.pose import rgb_edge_residual
from factors.object.image import rgb_reprojection_residual
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
        add([i], dr / (cfg.prior_rotation_rad * scale))
        if not rotation_only:
            add([i], dt / (cfg.prior_translation_m * scale))
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
            weights = np.sqrt(f.weights * cfg.observed_weight / max(1, len(delta)))
            # Unsigned observed point/plane alignment; normals have no inside/outside authority.
            plane = np.einsum('ij,ij->i', f.target_normals, delta)
            add([f.source_pos, f.target_pos], weights * plane / .008)
            add([f.source_pos, f.target_pos], weights[:, None] * delta / .020)
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
