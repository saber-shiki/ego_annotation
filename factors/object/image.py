"""Transport both endpoints before calibrated reprojection; retain per-sample normalization."""
from __future__ import annotations
from typing import Mapping, Sequence
import math
import numpy as np
from utils.data import RGBReprojectionFactor, PoseNode, SolverConfig
from utils.geometry import inverse_pose, apply_pose

def _robust_clip(values: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), -float(limit), float(limit))


def rgb_reprojection_residual(
    factor: RGBReprojectionFactor,
    pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
    nodes: Sequence[PoseNode],
    config: SolverConfig,
) -> np.ndarray:
    """Return observed RGB pixel residuals for one optional evidence factor."""
    source_node = nodes[factor.source_pos]
    target_node = nodes[factor.target_pos]
    source_R, source_t = pose_by_frame[factor.source_frame_idx]
    target_R, target_t = pose_by_frame[factor.target_frame_idx]
    if not factor.rotation_eligible:
        source_R = source_node.initial_rotation
        target_R = target_node.initial_rotation
    source_canonical = inverse_pose(factor.source_world, source_R, source_t)
    world = apply_pose(source_canonical, target_R, target_t)
    camera = inverse_pose(
        world,
        factor.target_camera_rotation,
        factor.target_camera_translation,
    )
    fx, fy, cx, cy = factor.target_intrinsics.tolist()
    z = np.asarray(camera[:, 2], dtype=np.float64)
    safe_z = np.where(np.abs(z) > 1.0e-6, z, np.where(z >= 0.0, 1.0e-6, -1.0e-6))
    projected = np.column_stack((fx * camera[:, 0] / safe_z + cx, fy * camera[:, 1] / safe_z + cy))
    error = projected - factor.target_uv
    error[~np.isfinite(error)] = float(config.max_rgb_reprojection_residual_px)
    error = _robust_clip(error, config.max_rgb_reprojection_residual_px)
    scale = math.sqrt(max(float(config.rgb_reprojection_weight_scale), 0.0) * max(float(factor.edge_weight), 0.0))
    scale *= 1.0 / max(float(config.sigma_rgb_reprojection_px), 1.0e-8)
    scale /= math.sqrt(max(1, len(error)))
    return (scale * np.sqrt(np.clip(factor.weights, 0.0, 1.0))[:, None] * error).reshape(-1)


def silhouette_endpoint_residual(projected,target,*,sigma,clip,weight):
    difference=projected-target
    scale=np.minimum(1.,clip/np.maximum(np.linalg.norm(difference,axis=1),1e-9))
    residual=difference*scale[:,None]/sigma
    return np.sqrt(weight)*residual.reshape(-1)/np.sqrt(max(1,len(difference)))
