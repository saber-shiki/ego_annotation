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
from rgb_scene_pose_kernels import (
    PoseNode, RGBEdge, RGBReprojectionFactor, SolverConfig, valid_rotation,
    apply_left_correction, pose_correction, inverse_pose, rgb_edge_residual,
    rgb_reprojection_residual, build_observed_point_factors,
)


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


def residual_blocks(poses, nodes, times, edges, points, reprojections, cfg, rotation_only=False):
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
            delta = inverse_pose(f.source_world, *poses[a.frame_idx]) - inverse_pose(f.target_world, *poses[b.frame_idx])
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


def solve(nodes, times, edges, anchor, config=None, reprojections=()):
    cfg = config or GraphConfig()
    times = validate(nodes, times, edges, anchor)
    if cfg.loss not in {'soft_l1', 'linear'} or min(cfg.rotation_step_rad, cfg.translation_step_m, cfg.depth_restoration_m, cfg.near_depth_m) <= 0:
        raise ValueError('invalid graph configuration')
    poses = {n.frame_idx: (n.initial_rotation.copy(), n.initial_translation.copy()) for n in nodes}
    # Eliminating this variable fixes the sole gauge exactly, without a huge numerical weight.
    free = [i for i, n in enumerate(nodes) if n.frame_idx != anchor]
    trace = []
    for rotation_only in (True, False):
        dof = 3 if rotation_only else 6
        for outer in range(cfg.outer_iterations):
            kernel = SolverConfig(max_points=250, min_pairs=3)
            points, info = ([], {}) if rotation_only else build_observed_point_factors(nodes, poses, anchor, kernel, outer)
            base = poses
            def unpack(x):
                result = dict(base)
                for j, i in enumerate(free):
                    delta = x[j * dof:(j + 1) * dof]
                    result[nodes[i].frame_idx] = apply_left_correction(*base[nodes[i].frame_idx], delta[:3], np.zeros(3) if rotation_only else delta[3:])
                return result
            def blocks(x):
                return residual_blocks(unpack(x), nodes, times, edges, points, reprojections, cfg, rotation_only)
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
                   'depth_restoration': {'affected_frame_ids': sorted(invalid_ids), 'remaining_invalid': int(np.sum(depth_state(poses, nodes, reprojections, cfg)[0] < cfg.near_depth_m)),
                                         'policy': 'weighted restoration; retain uncertainty when infeasible'},
                   'relative_evidence_disconnected_ids': [n.frame_idx for n in nodes if n.frame_idx not in connected]}
