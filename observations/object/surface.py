"""Extracted formal interval mechanisms; no historical script imports."""
from __future__ import annotations

def classify_object_vertices_against_depth(*, frame: dict[str, Any], vertices_object: np.ndarray, pose: tuple[np.ndarray, np.ndarray], depth_row: dict[str, Any] | None, support_margin_m: float, free_space_margin_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    from typing import Any
    import numpy as np
    from collections import Counter
    from utils.geometry import frame_camera_pose
    from utils.data import numeric_summary
    classes = np.full((len(vertices_object),), VERTEX_OUT_OR_INVALID, dtype=np.uint8)
    frame_idx = int(frame['frame_idx'])
    if depth_row is None:
        return (classes, {'frame_idx': frame_idx, 'state': 'missing_depth', 'projected_count': 0, 'finite_depth_count': 0, 'vertex_class_counts': {'out_or_invalid': int(len(vertices_object))}})
    (r_obj, t_obj) = pose
    vertices_world = vertices_object @ r_obj.T + t_obj[None, :]
    (r_c2w, t_c2w) = frame_camera_pose(frame)
    vertices_camera = (vertices_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row['depth'], dtype=np.float32)
    (height, width) = depth.shape
    (u, v, valid) = project_points(vertices_camera, np.asarray(depth_row['intrinsics'], dtype=float), width, height)
    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        return (classes, {'frame_idx': frame_idx, 'state': 'no_projected_object_vertices', 'projected_count': 0, 'finite_depth_count': 0, 'vertex_class_counts': {'out_or_invalid': int(len(vertices_object))}})
    z_mesh = vertices_camera[valid_indices, 2]
    z_obs = depth[v[valid_indices], u[valid_indices]].astype(float)
    finite = np.isfinite(z_obs) & (z_obs > 0.0)
    finite_indices = valid_indices[finite]
    residual = z_mesh[finite] - z_obs[finite]
    supported = np.abs(residual) <= float(support_margin_m)
    free_space = residual < -float(free_space_margin_m)
    behind = residual > float(support_margin_m)
    classes[finite_indices[supported]] = VERTEX_OBSERVED_SUPPORTED
    classes[finite_indices[free_space]] = VERTEX_FREE_SPACE_CONFLICT
    classes[finite_indices[behind]] = VERTEX_BEHIND_OBSERVED
    counter = Counter((int(x) for x in classes.tolist()))
    class_counts = {'out_or_invalid': int(counter.get(VERTEX_OUT_OR_INVALID, 0)), 'observed_supported': int(counter.get(VERTEX_OBSERVED_SUPPORTED, 0)), 'free_space_conflict': int(counter.get(VERTEX_FREE_SPACE_CONFLICT, 0)), 'behind_observed': int(counter.get(VERTEX_BEHIND_OBSERVED, 0))}
    finite_count = int(finite_indices.size)
    support_fraction = class_counts['observed_supported'] / finite_count if finite_count else None
    free_fraction = class_counts['free_space_conflict'] / finite_count if finite_count else None
    state = 'object_surface_depth_classified'
    return (classes, {'frame_idx': frame_idx, 'state': state, 'depth_source': str(depth_row['source']), 'projected_count': int(valid_indices.size), 'finite_depth_count': finite_count, 'vertex_class_counts': class_counts, 'observed_support_fraction_finite': support_fraction, 'free_space_conflict_fraction_finite': free_fraction, 'depth_residual_m': numeric_summary(residual)})
VERTEX_OBSERVED_SUPPORTED = 1
VERTEX_OUT_OR_INVALID = 0

def project_points(points_camera: np.ndarray, intr: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    (fx, fy, cx, cy) = [float(x) for x in intr]
    z = points_camera[:, 2]
    valid = z > 0.0001
    u_float = fx * points_camera[:, 0] / np.maximum(z, 1e-06) + cx
    v_float = fy * points_camera[:, 1] / np.maximum(z, 1e-06) + cy
    u = np.rint(u_float).astype(np.int32)
    v = np.rint(v_float).astype(np.int32)
    valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return (u, v, valid)
VERTEX_FREE_SPACE_CONFLICT = 2
VERTEX_BEHIND_OBSERVED = 3

def face_provenance(vertex_classes: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    import numpy as np
    tri = vertex_classes[np.asarray(faces, dtype=np.int64)]
    supported_count = np.count_nonzero(tri == VERTEX_OBSERVED_SUPPORTED, axis=1)
    free_count = np.count_nonzero(tri == VERTEX_FREE_SPACE_CONFLICT, axis=1)
    behind_count = np.count_nonzero(tri == VERTEX_BEHIND_OBSERVED, axis=1)
    invalid_count = np.count_nonzero(tri == VERTEX_OUT_OR_INVALID, axis=1)
    observed_strict = (supported_count >= 2) & (free_count == 0)
    observed_any = (supported_count >= 1) & (free_count == 0)
    free_any = free_count >= 1
    return {'observed_supported_strict': observed_strict, 'observed_supported_any': observed_any, 'free_space_conflict_any': free_any, 'supported_count': supported_count, 'free_count': free_count, 'behind_count': behind_count, 'invalid_count': invalid_count}

def load_depth_sources(paths: list[Path]) -> dict[int, dict[str, Any]]:
    from typing import Any
    import numpy as np
    from pathlib import Path
    frames: dict[int, dict[str, Any]] = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        frame_idx = np.asarray(data['frame_idx'], dtype=int)
        depth = np.asarray(data['depth'], dtype=np.float32)
        intrinsics = np.asarray(data['intrinsics_fx_fy_cx_cy'], dtype=np.float32)
        for (i, frame) in enumerate(frame_idx):
            idx = int(frame)
            if idx in frames:
                continue
            frames[idx] = {'source': str(path), 'depth': depth[i], 'intrinsics': intrinsics[i]}
    return frames
