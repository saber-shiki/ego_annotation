"""Appearance observations are distinct from metric ownership masks."""
from __future__ import annotations
from pathlib import Path
from utils.state import sha

def appearance_mask_rows(manifest, annotation_path, object_id, frame_ids, source_size):
    """Validate and bind the raw-appearance plane; never fall back to owned masks."""
    if not isinstance(manifest, dict) or manifest.get('object_id') != object_id:
        raise ValueError('appearance manifest object mismatch')
    if manifest.get('metric_owned_masks_consumed') is not False:
        raise ValueError('appearance manifest does not exclude metric-owned masks')
    if manifest.get('source_annotations_sha256') != sha(annotation_path):
        raise ValueError('appearance manifest annotations identity mismatch')
    rows = manifest.get('frames')
    if not isinstance(rows, list):
        raise ValueError('appearance manifest frames missing')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('object_id') != object_id:
            raise ValueError('invalid appearance manifest row')
        idx = int(row['frame_idx'])
        if idx in indexed:
            raise ValueError(f'duplicate appearance frame {idx}')
        path = row.get('appearance_mask_path')
        if row.get('status') == 'available':
            if row.get('visible') is not True:
                raise ValueError(f'invisible appearance mask cannot be available at frame {idx}')
            if not path or sha(path) != row.get('appearance_mask_sha256'):
                raise ValueError(f'appearance mask identity mismatch at frame {idx}')
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            declared = row.get('source_size_wh')
            if mask is None or not isinstance(declared, list) or len(declared) != 2 or (mask.shape != (int(declared[1]), int(declared[0]))) or (abs(mask.shape[1] / mask.shape[0] - float(source_size[0]) / float(source_size[1])) > 0.01):
                raise ValueError(f'appearance mask dimensions mismatch at frame {idx}')
            indexed[idx] = path if np.any(mask) else None
        elif row.get('status') == 'unavailable_not_substituted' and path is None:
            indexed[idx] = None
        else:
            raise ValueError(f'invalid appearance availability at frame {idx}')
    expected = {int(value) for value in frame_ids}
    if set(indexed) != expected:
        raise ValueError('appearance manifest frame coverage mismatch')
    return indexed
import cv2
import numpy as np
from pathlib import Path

def inactive_signed_surface_measure(frame_idx: int, support_uncertainty_m: float=0.0) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    from utils.data import numeric_summary
    return {'frame_idx': int(frame_idx), 'state': 'inactive_signed_geometry_not_ready', 'penetrating_vertex_count': 0, 'observed_supported_penetrating_vertex_count': 0, 'observed_supported_penetration_m': numeric_summary(np.asarray([], dtype=float)), 'observed_surface_support_uncertainty_m': max(0.0, float(support_uncertainty_m))}
OPENPOSE_FINGER_GROUPS = [np.arange(1, 5, dtype=np.int64), np.arange(5, 9, dtype=np.int64), np.arange(9, 13, dtype=np.int64), np.arange(13, 17, dtype=np.int64), np.arange(17, 21, dtype=np.int64)]

def full_observed_surface_measure(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, eps: float, object_translation_delta_world: np.ndarray | None=None, face_strict_observed: np.ndarray | None=None) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    import open3d as o3d
    from utils.geometry import inverse_object
    from utils.data import numeric_summary
    from utils.data import FrameHandRow
    obj_delta = np.zeros(3, dtype=float) if object_translation_delta_world is None else np.asarray(object_translation_delta_world, dtype=float)
    face_mask = row.face_strict_observed if face_strict_observed is None else np.asarray(face_strict_observed, dtype=bool)
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m + obj_delta)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    penetrating = np.where(signed > float(eps))[0]
    if penetrating.size == 0:
        return {'penetrating_vertex_count': 0, 'observed_supported_penetrating_vertex_count': 0, 'observed_supported_penetration_m': numeric_summary(np.asarray([], dtype=float))}
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest['primitive_ids'].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(face_mask))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = face_mask[primitive_ids[valid]]
    observed_depths = signed[penetrating][observed]
    return {'penetrating_vertex_count': int(penetrating.size), 'observed_supported_penetrating_vertex_count': int(np.count_nonzero(observed)), 'observed_supported_penetration_m': numeric_summary(observed_depths)}

def full_signed_topology_measure(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, eps: float, object_translation_delta_world: np.ndarray | None=None) -> dict[str, Any]:
    """Audit all topology penetration and separate locally authorized faces.

    Unsupported closure penetration is not asserted as physical collision, but
    it is unresolved geometry and therefore blocks full-MANO promotion.
    """
    from typing import Any
    import numpy as np
    import open3d as o3d
    from utils.geometry import inverse_object
    from utils.data import numeric_summary
    from utils.data import FrameHandRow
    obj_delta = np.zeros(3, dtype=float) if object_translation_delta_world is None else np.asarray(object_translation_delta_world, dtype=float)
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m + obj_delta)
    query = o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))
    signed = -scene.compute_signed_distance(query).numpy().astype(float)
    penetrating = np.where(signed > float(eps))[0]
    empty = numeric_summary(np.asarray([], dtype=float))
    if penetrating.size == 0:
        return {'topology_penetrating_vertex_count': 0, 'authorized_penetrating_vertex_count': 0, 'unauthorized_penetrating_vertex_count': 0, 'topology_penetration_m': empty, 'authorized_penetration_m': empty, 'unauthorized_penetration_m': empty, 'claim_scope': 'watertight topology audit; unsupported faces block promotion but do not assert physical collision'}
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive = closest['primitive_ids'].numpy().astype(np.int64)
    valid = (primitive >= 0) & (primitive < len(row.face_strict_observed_raw))
    authorized = np.zeros((len(penetrating),), dtype=bool)
    authorized[valid] = row.face_strict_observed_raw[primitive[valid]]
    depths = signed[penetrating]
    return {'topology_penetrating_vertex_count': int(len(penetrating)), 'authorized_penetrating_vertex_count': int(np.count_nonzero(authorized)), 'unauthorized_penetrating_vertex_count': int(np.count_nonzero(~authorized)), 'topology_penetration_m': numeric_summary(depths), 'authorized_penetration_m': numeric_summary(depths[authorized]), 'unauthorized_penetration_m': numeric_summary(depths[~authorized]), 'claim_scope': 'watertight topology audit; unsupported faces block promotion but do not assert physical collision'}

def sample_ids(n: int, count: int) -> np.ndarray:
    import numpy as np
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, count, dtype=np.int64)

def visible_ownership_masks_for_row(row: dict[str, Any] | None, cache: dict[Path, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    from typing import Any
    import numpy as np
    from pathlib import Path
    from utils.geometry import finite_image_affine
    if not isinstance(row, dict):
        return (None, None, {'state': 'missing_visible_ownership_row'})
    non_object_raw = row.get('non_object_owned_mask_path')
    constraint_raw = row.get('constraint_eligible_entity_mask_path') or row.get('adjusted_entity_mask_path') or row.get('visible_object_owned_mask_path')
    visible_object_raw = row.get('visible_object_owned_mask_path')
    depth_order_support_raw = row.get('depth_order_query_support_mask_path')
    depth_order_npz_raw = row.get('depth_order_first_surface_npz_path')
    if not isinstance(non_object_raw, str) or not Path(non_object_raw).exists():
        raise FileNotFoundError(f'visible ownership row has no readable non_object_owned_mask_path: {non_object_raw}')
    if not isinstance(constraint_raw, str) or not Path(constraint_raw).exists():
        raise FileNotFoundError(f'visible ownership row has no readable constraint_eligible_entity/adjusted_entity mask path: {constraint_raw}')
    non_object_mask = load_binary_mask(Path(non_object_raw), cache)
    constraint_mask = load_binary_mask(Path(constraint_raw), cache)
    if non_object_mask.shape != constraint_mask.shape:
        raise RuntimeError(f'visible ownership mask shape mismatch: {non_object_mask.shape} != {constraint_mask.shape}')
    plane = row.get('image_plane_transform')
    if not isinstance(plane, dict) or plane.get('camera_contract_consistent') is not True:
        raise RuntimeError('visible ownership row lacks an explicit camera-consistent image-plane transform')
    mask_size = [int(v) for v in plane.get('mask_size_wh') or []]
    actual_size = [int(non_object_mask.shape[1]), int(non_object_mask.shape[0])]
    if mask_size != actual_size:
        raise RuntimeError(f'visible ownership mask raster {actual_size} disagrees with factor contract {mask_size}')
    A_mask_from_source = finite_image_affine(plane.get('A_mask_from_source_coordinate_model'), 'visible ownership A_mask_from_source')
    raw_counts = row.get('counts')
    counts = raw_counts if isinstance(raw_counts, dict) else {}
    if depth_order_support_raw not in (None, '') and (not isinstance(depth_order_support_raw, str) or not Path(depth_order_support_raw).exists()):
        raise FileNotFoundError(f'visible ownership row has unreadable depth-order support mask: {depth_order_support_raw}')
    if depth_order_npz_raw not in (None, '') and (not isinstance(depth_order_npz_raw, str) or not Path(depth_order_npz_raw).exists()):
        raise FileNotFoundError(f'visible ownership row has unreadable depth-order first-surface NPZ: {depth_order_npz_raw}')
    return (non_object_mask, constraint_mask, {'state': 'ok', 'image_plane_transform': plane, 'A_mask_from_source_coordinate_model': A_mask_from_source.tolist(), 'non_object_owned_mask_path': non_object_raw if isinstance(non_object_raw, str) else None, 'constraint_eligible_entity_mask_path': constraint_raw if isinstance(constraint_raw, str) else None, 'visible_object_owned_mask_path': visible_object_raw if isinstance(visible_object_raw, str) else None, 'depth_order_query_support_mask_path': depth_order_support_raw if isinstance(depth_order_support_raw, str) else None, 'depth_order_query_support_mask_sha256': row.get('depth_order_query_support_mask_sha256'), 'depth_order_first_surface_npz_path': depth_order_npz_raw if isinstance(depth_order_npz_raw, str) else None, 'depth_order_first_surface_npz_sha256': row.get('depth_order_first_surface_npz_sha256'), 'depth_order_query_semantics': row.get('depth_order_query_semantics'), 'non_object_owned_px': int(counts.get('non_object_owned_px', int(non_object_mask.sum()) if non_object_mask is not None else 0)), 'visible_object_owned_px': int(counts.get('visible_object_owned_px', 0)), 'constraint_eligible_entity_px': int(counts.get('constraint_eligible_entity_px', int(constraint_mask.sum()) if constraint_mask is not None else 0))})

def load_binary_mask(mask_path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    from PIL import Image
    import numpy as np
    from pathlib import Path
    cached = cache.get(mask_path)
    if cached is not None:
        return cached
    mask = np.asarray(Image.open(mask_path).convert('L')) > 0
    cache[mask_path] = mask
    return mask
FACTOR_REQUIRED_FIELDS = ('factor_family', 'target_entity_id', 'frame_idx', 'hand_side', 'variable_affected', 'observation_type', 'residual_or_quarantine_rule', 'provenance', 'rendered_uncertainty_channel')

def hand_depth_shift_prior_for_row(row: dict[str, Any] | None) -> dict[str, Any]:
    from typing import Any
    import numpy as np
    if not isinstance(row, dict):
        return {'state': 'missing_hand_depth_shift_prior_row', 'camera_z_shift_m': 0.0, 'weight': None}
    state = str(row.get('state') or 'active_hand_depth_shift_prior')
    try:
        shift = float(row.get('camera_z_shift_m', 0.0) or 0.0)
    except Exception:
        shift = 0.0
    raw_weight = row.get('weight')
    weight: float | None
    if raw_weight in (None, ''):
        weight = None
    else:
        try:
            weight = float(raw_weight)
        except Exception as exc:
            raise ValueError(f'hand_depth_shift_prior row has invalid weight {raw_weight!r}: {row}') from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f'hand_depth_shift_prior row has invalid nonnegative finite weight {raw_weight!r}: {row}')
    return {'state': state, 'camera_z_shift_m': shift, 'weight': weight}

def visible_surface_track_mask_for_row(row: dict[str, Any] | None, cache: dict[Path, np.ndarray]) -> tuple[np.ndarray | None, dict[str, Any]]:
    from typing import Any
    import numpy as np
    from pathlib import Path
    if not isinstance(row, dict):
        return (None, {'state': 'missing_visible_surface_track_row'})
    state = str(row.get('surface_state') or row.get('state') or 'unknown')
    raw = row.get('surface_mask_path')
    if state != 'active_visible_surface':
        return (None, {'state': state, 'surface_mask_path': raw if isinstance(raw, str) else None, 'visible_surface_npz_path': row.get('visible_surface_npz_path') if isinstance(row.get('visible_surface_npz_path'), str) else None, 'valid_depth_pixels': int(row.get('valid_depth_pixels', 0) or 0), 'quarantine_hidden_volume': bool(row.get('quarantine_hidden_volume', False))})
    if not isinstance(raw, str) or not Path(raw).exists():
        raise FileNotFoundError(f'active visible-surface factor row has no readable mask path: {raw}')
    mask = load_binary_mask(Path(raw), cache)
    return (mask, {'state': state, 'surface_mask_path': raw, 'visible_surface_npz_path': row.get('visible_surface_npz_path') if isinstance(row.get('visible_surface_npz_path'), str) else None, 'valid_depth_pixels': int(row.get('valid_depth_pixels', 0) or 0), 'quarantine_hidden_volume': bool(row.get('quarantine_hidden_volume', True))})

def visible_ownership_quarantine_faces(*, frame: dict[str, Any], side: str, object_vertices: np.ndarray, object_faces: np.ndarray, object_pose: tuple[np.ndarray, np.ndarray], face_strict_observed: np.ndarray, non_object_owned_mask: np.ndarray | None, A_mask_from_source: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    from typing import Any
    import numpy as np
    import argparse
    from observations.interaction.evidence import project_world
    strict = np.asarray(face_strict_observed, dtype=bool).copy()
    if non_object_owned_mask is None or not np.any(strict):
        return (strict, 0)
    (r_obj, t_obj) = object_pose
    tri = object_vertices[object_faces]
    samples_obj = np.stack([tri[:, 0], tri[:, 1], tri[:, 2], 0.5 * (tri[:, 0] + tri[:, 1]), 0.5 * (tri[:, 1] + tri[:, 2]), 0.5 * (tri[:, 2] + tri[:, 0]), tri.mean(axis=1)], axis=1)
    samples_world = samples_obj.reshape(-1, 3) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    uv_source = project_world(samples_world, frame, side)
    inside_flat = mask_membership(non_object_owned_mask, uv_source, int(args.visible_ownership_face_overlap_dilation_px), A_mask_from_source=A_mask_from_source)
    if inside_flat.shape[0] != samples_world.shape[0]:
        return (strict, 0)
    inside = inside_flat.reshape(len(object_faces), -1).any(axis=1)
    q = strict & inside
    strict[q] = False
    return (strict, int(np.count_nonzero(q)))

def mask_membership(mask: np.ndarray, uv: np.ndarray | None, dilation_px: int=0, *, A_mask_from_source: np.ndarray | None=None) -> np.ndarray:
    import numpy as np
    from utils.geometry import transform_source_uv_to_mask
    if uv is None:
        return np.zeros((0,), dtype=bool)
    uv = np.asarray(uv, dtype=float)
    if A_mask_from_source is not None:
        transformed = transform_source_uv_to_mask(uv, A_mask_from_source)
        assert transformed is not None
        uv = transformed
    if uv.ndim != 2 or uv.shape[1] != 2:
        return np.zeros((len(uv),), dtype=bool)
    (height, width) = mask.shape
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    inside = np.zeros((len(uv),), dtype=bool)
    if not np.any(valid):
        return inside
    radius = max(0, int(dilation_px))
    x0 = np.clip(u[valid] - radius, 0, width - 1)
    x1 = np.clip(u[valid] + radius, 0, width - 1)
    y0 = np.clip(v[valid] - radius, 0, height - 1)
    y1 = np.clip(v[valid] + radius, 0, height - 1)
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    area = integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
    inside[np.where(valid)[0]] = area > 0
    return inside

def visible_surface_depth_order_constraints(*, frame: dict[str, Any], side: str, vertices_world: np.ndarray, mask: np.ndarray | None, depth_row: dict[str, Any] | None, A_mask_from_source: np.ndarray, depth_is_mask_plane: bool=False, mask_dilation_px: int | None=None, args: argparse.Namespace, enabled: bool | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from typing import Any
    import numpy as np
    import argparse
    from utils.geometry import transform_source_uv_to_mask
    from utils.geometry import world_to_camera
    from utils.data import numeric_summary
    from observations.interaction.evidence import project_world
    if enabled is None:
        enabled = bool(args.visible_surface_depth_order_term)
    if mask is None or depth_row is None or (not bool(enabled)):
        empty = np.zeros((0,), dtype=float)
        return (np.zeros((0,), dtype=np.int64), empty, empty, {'finite_inside_count': 0, 'hand_behind_observed_surface_count': 0, 'hand_in_front_of_observed_surface_count': 0, 'hand_near_observed_surface_depth_count': 0, 'depth_delta_hand_minus_surface_m': numeric_summary(empty)})
    uv = project_world(vertices_world, frame, side)
    if uv is None:
        empty = np.zeros((0,), dtype=float)
        return (np.zeros((0,), dtype=np.int64), empty, empty, {'finite_inside_count': 0, 'hand_behind_observed_surface_count': 0, 'hand_in_front_of_observed_surface_count': 0, 'hand_near_observed_surface_depth_count': 0, 'depth_delta_hand_minus_surface_m': numeric_summary(empty)})
    depth = np.asarray(depth_row.get('depth'), dtype=np.float32)
    if depth.ndim != 2:
        empty = np.zeros((0,), dtype=float)
        return (np.zeros((0,), dtype=np.int64), empty, empty, {'finite_inside_count': 0, 'hand_behind_observed_surface_count': 0, 'hand_in_front_of_observed_surface_count': 0, 'hand_near_observed_surface_depth_count': 0, 'depth_delta_hand_minus_surface_m': numeric_summary(empty)})
    (height, width) = depth.shape
    inside = mask_membership(mask, uv, int(args.visible_object_mask_dilation_px) if mask_dilation_px is None else int(mask_dilation_px), A_mask_from_source=A_mask_from_source)
    cam = world_to_camera(vertices_world, frame)
    if depth_is_mask_plane:
        uv_depth = transform_source_uv_to_mask(uv, A_mask_from_source)
        assert uv_depth is not None
    else:
        uv_depth = uv
    u = np.rint(uv_depth[:, 0]).astype(int)
    v = np.rint(uv_depth[:, 1]).astype(int)
    valid = inside & (cam[:, 2] > 1e-05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if np.any(valid):
        z_surface_all = depth[v[valid], u[valid]].astype(float)
        valid_ids = np.where(valid)[0]
        finite = np.isfinite(z_surface_all) & (z_surface_all > 1e-05)
        valid_ids = valid_ids[finite]
        z_surface_all = z_surface_all[finite]
    else:
        valid_ids = np.zeros((0,), dtype=np.int64)
        z_surface_all = np.zeros((0,), dtype=float)
    if valid_ids.size == 0:
        empty = np.zeros((0,), dtype=float)
        return (np.zeros((0,), dtype=np.int64), empty, empty, {'finite_inside_count': 0, 'hand_behind_observed_surface_count': 0, 'hand_in_front_of_observed_surface_count': 0, 'hand_near_observed_surface_depth_count': 0, 'depth_delta_hand_minus_surface_m': numeric_summary(empty)})
    delta = cam[valid_ids, 2].astype(float) - z_surface_all
    margin = float(args.visible_surface_depth_order_margin_m)
    measure = {'finite_inside_count': int(valid_ids.size), 'depth_coordinate_plane': 'mask' if depth_is_mask_plane else 'source', 'hand_behind_observed_surface_count': int(np.count_nonzero(delta > margin)), 'hand_in_front_of_observed_surface_count': int(np.count_nonzero(delta < -margin)), 'hand_near_observed_surface_depth_count': int(np.count_nonzero(np.abs(delta) <= margin)), 'depth_delta_hand_minus_surface_m': numeric_summary(delta)}
    violation = np.maximum(0.0, z_surface_all - margin - cam[valid_ids, 2].astype(float))
    order = np.argsort(violation)[::-1]
    cap = max(0, int(args.max_visible_surface_depth_vertices))
    if cap and len(order) > cap:
        order = order[:cap]
    selected_ids = valid_ids[order].astype(np.int64)
    selected_surface_depth = z_surface_all[order].astype(float)
    selected_delta = delta[order].astype(float)
    return (selected_ids, selected_surface_depth, selected_delta, measure)

def observed_constraints_for_hand(*, vertices_world: np.ndarray, pose: tuple[np.ndarray, np.ndarray], scene: Any, face_strict_observed: np.ndarray, frame_idx: int, max_constraints: int, eps: float, support_uncertainty_m: float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from typing import Any
    import numpy as np
    import open3d as o3d
    from utils.geometry import inverse_object
    from utils.data import numeric_summary
    from utils.geometry import object_vec_to_world
    (r_obj, t_obj) = pose
    vertices_object = inverse_object(vertices_world, r_obj, t_obj)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    hard_eps = float(eps) + max(0.0, float(support_uncertainty_m))
    penetrating = np.where(signed > hard_eps)[0]
    if penetrating.size == 0:
        measure = {'frame_idx': frame_idx, 'penetrating_vertex_count': 0, 'observed_supported_penetrating_vertex_count': 0, 'observed_supported_penetration_m': numeric_summary(np.asarray([], dtype=float)), 'observed_surface_support_uncertainty_m': max(0.0, float(support_uncertainty_m))}
        return (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), measure)
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest['primitive_ids'].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = face_strict_observed[primitive_ids[valid]]
    obs_idx = penetrating[observed]
    obs_depth = signed[obs_idx]
    if len(obs_idx) > int(max_constraints):
        order = np.argsort(obs_depth)[::-1][:int(max_constraints)]
        obs_idx = obs_idx[order]
        obs_depth = obs_depth[order]
    normals_world = np.zeros((0, 3), dtype=float)
    depths = np.zeros((0,), dtype=float)
    if len(obs_idx):
        closest_obs = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[obs_idx], dtype=np.float32)))['points'].numpy().astype(float)
        disp = closest_obs - vertices_object[obs_idx]
        norms = np.linalg.norm(disp, axis=1)
        good = norms > 1e-12
        normals_world = object_vec_to_world(disp[good] / norms[good, None], r_obj)
        depths = obs_depth[good]
        obs_idx = obs_idx[good]
    measure = {'frame_idx': frame_idx, 'penetrating_vertex_count': int(penetrating.size), 'observed_supported_penetrating_vertex_count': int(len(obs_idx)), 'observed_supported_penetration_m': numeric_summary(depths), 'observed_surface_support_uncertainty_m': max(0.0, float(support_uncertainty_m))}
    return (obs_idx.astype(np.int64), normals_world.astype(float), depths.astype(float), measure)

def surface_eligibility_mask_for_row(row: dict[str, Any] | None, expected_count: int, cache: dict[Path, np.ndarray]) -> tuple[np.ndarray | None, dict[str, Any]]:
    from typing import Any
    import numpy as np
    from pathlib import Path
    if not isinstance(row, dict):
        return (None, {'state': 'missing_surface_eligibility_row'})
    raw = row.get('face_state_npz_path')
    if not isinstance(raw, str) or not Path(raw).exists():
        return (None, {'state': 'missing_surface_eligibility_npz', 'face_state_npz_path': raw if isinstance(raw, str) else None})
    path = Path(raw)
    if path not in cache:
        with np.load(path) as data:
            if 'eligible_hard_observed' not in data:
                raise KeyError(f'surface eligibility NPZ lacks eligible_hard_observed: {path}')
            cache[path] = np.asarray(data['eligible_hard_observed'], dtype=bool)
    mask = cache[path]
    if mask.shape != (int(expected_count),):
        return (None, {'state': 'surface_eligibility_shape_mismatch', 'face_state_npz_path': str(path), 'mask_count': int(mask.size), 'expected_count': int(expected_count)})
    try:
        support_uncertainty_m = float(row.get('observed_surface_support_uncertainty_m', row.get('surface_support_uncertainty_m', row.get('object_support_uncertainty_m', 0.0))) or 0.0)
    except Exception:
        support_uncertainty_m = 0.0
    return (mask.copy(), {'state': 'ok', 'face_state_npz_path': str(path), 'eligible_hard_observed_count': int(np.count_nonzero(mask)), 'observed_surface_support_uncertainty_m': max(0.0, support_uncertainty_m)})

def load_depth_order_first_surface_for_row(diag: dict[str, Any], cache: dict[Path, np.ndarray], expected_mask_shape: tuple[int, int] | None) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    from typing import Any
    import numpy as np
    from pathlib import Path
    from utils.geometry import finite_image_affine
    from utils.state import sha256_file
    mask_raw = diag.get('depth_order_query_support_mask_path')
    npz_raw = diag.get('depth_order_first_surface_npz_path')
    if not isinstance(mask_raw, str) or not isinstance(npz_raw, str):
        return (None, None, {'state': 'missing_independent_depth_order_first_surface', 'raw_depth_under_hand_consumed': False})
    mask_path = Path(mask_raw)
    npz_path = Path(npz_raw)
    expected_mask_sha256 = str(diag.get('depth_order_query_support_mask_sha256') or '')
    expected_npz_sha256 = str(diag.get('depth_order_first_surface_npz_sha256') or '')
    if not expected_mask_sha256 or sha256_file(mask_path) != expected_mask_sha256:
        raise RuntimeError('depth-order support mask SHA256 binding failed')
    if not expected_npz_sha256 or sha256_file(npz_path) != expected_npz_sha256:
        raise RuntimeError('depth-order first-surface NPZ SHA256 binding failed')
    support = load_binary_mask(mask_path, cache)
    with np.load(npz_path, allow_pickle=False) as archive:
        if 'depth_mask_plane_m' not in archive.files or 'support_mask' not in archive.files:
            raise KeyError(f'depth-order first-surface NPZ lacks depth/support arrays: {npz_path}')
        depth_mask = np.asarray(archive['depth_mask_plane_m'], dtype=np.float32)
        support_npz = np.asarray(archive['support_mask'], dtype=bool)
        affine = finite_image_affine(archive['A_mask_from_source_coordinate_model'], 'depth-order NPZ A_mask_from_source')
    if depth_mask.shape != support.shape or support_npz.shape != support.shape:
        raise RuntimeError(f'depth-order support/depth shape mismatch: {support.shape}/{support_npz.shape}/{depth_mask.shape}')
    if expected_mask_shape is not None and support.shape != expected_mask_shape:
        raise RuntimeError(f'depth-order support mask shape {support.shape} differs from ownership mask {expected_mask_shape}')
    declared_affine = finite_image_affine(diag.get('A_mask_from_source_coordinate_model'), 'ownership A_mask_from_source')
    if not np.allclose(affine, declared_affine, atol=1e-12, rtol=0.0):
        raise RuntimeError('depth-order NPZ affine differs from ownership factor affine')
    if not np.array_equal(support, support_npz):
        raise RuntimeError('depth-order support PNG and NPZ support arrays are not identical')
    if np.any(support & (~np.isfinite(depth_mask) | (depth_mask <= 1e-05))):
        raise RuntimeError('depth-order support marks pixels without finite positive P09 first-surface depth')
    support &= np.isfinite(depth_mask) & (depth_mask > 1e-05)
    return (support, depth_mask, {'state': 'active_independent_P09_first_surface_depth_order_query', 'support_mask_path': str(mask_path), 'first_surface_npz_path': str(npz_path), 'support_pixel_count': int(np.count_nonzero(support)), 'raw_depth_under_hand_consumed': False, 'depth_source': 'P09 accepted first-surface camera samples'})

def hand_owned_object_depth_quarantine(*, frame: dict[str, Any], side: str, object_vertices: np.ndarray, object_faces: np.ndarray, object_pose: tuple[np.ndarray, np.ndarray], hand_vertices_world: np.ndarray, face_strict_observed: np.ndarray, depth_row: dict[str, Any] | None, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    """Quarantine object faces whose first-surface depth is plausibly hand-owned.

    Object-depth classification alone assumes a projected first surface belongs
    to the object.  During a grasp, a depth-supported hand surface can occupy the
    same pixels and be in front of, or at the same depth as, the object face. In
    that case the face is an occlusion/ownership uncertainty and should not push
    MANO as a trusted observed-object nonpenetration constraint.
    """
    from typing import Any
    import numpy as np
    import argparse
    from utils.geometry import world_to_camera
    from observations.interaction.evidence import project_world
    strict = np.asarray(face_strict_observed, dtype=bool).copy()
    if not bool(args.hand_owned_object_depth_quarantine) or depth_row is None or (not np.any(strict)):
        return (strict, 0)
    depth = np.asarray(depth_row.get('depth'), dtype=np.float32)
    if depth.ndim != 2:
        return (strict, 0)
    (height, width) = depth.shape
    (r_obj, t_obj) = object_pose
    face_centers_world = object_vertices[object_faces].mean(axis=1) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    uv_face = project_world(face_centers_world, frame, side)
    uv_hand = project_world(hand_vertices_world, frame, side)
    if uv_face is None or uv_hand is None:
        return (strict, 0)
    cam_face = world_to_camera(face_centers_world, frame)
    cam_hand = world_to_camera(hand_vertices_world, frame)
    uh = np.rint(uv_hand[:, 0]).astype(int)
    vh = np.rint(uv_hand[:, 1]).astype(int)
    hand_valid = (cam_hand[:, 2] > 1e-05) & (uh >= 0) & (uh < width) & (vh >= 0) & (vh < height)
    hand_ids = np.where(hand_valid)[0]
    if hand_ids.size == 0:
        return (strict, 0)
    uf = np.rint(uv_face[:, 0]).astype(int)
    vf = np.rint(uv_face[:, 1]).astype(int)
    face_valid = strict & (cam_face[:, 2] > 1e-05) & (uf >= 0) & (uf < width) & (vf >= 0) & (vf < height)
    if np.any(face_valid):
        face_depth = depth[vf[face_valid], uf[face_valid]].astype(float)
        tmp = np.zeros((len(face_valid),), dtype=bool)
        tmp[face_valid] = np.isfinite(face_depth) & (face_depth > 1e-05)
        face_valid = face_valid & tmp
    face_ids = np.where(face_valid)[0]
    if face_ids.size == 0:
        return (strict, 0)
    face_uv = uv_face[face_ids]
    face_z = cam_face[face_ids, 2]
    face_observed_depth = depth[vf[face_ids], uf[face_ids]].astype(float)
    q = np.zeros((face_ids.size,), dtype=bool)
    radius2 = float(args.hand_owned_quarantine_radius_px) ** 2
    z_margin = float(args.hand_owned_quarantine_depth_margin_m)
    support = float(args.hand_owned_quarantine_hand_depth_support_m)
    for hid in hand_ids.astype(int):
        d2 = np.sum((face_uv - uv_hand[hid]) ** 2, axis=1)
        hand_matches_face_depth = np.abs(float(cam_hand[hid, 2]) - face_observed_depth) <= support
        q |= (d2 <= radius2) & hand_matches_face_depth & (cam_hand[hid, 2] <= face_z - z_margin)
    quarantined_ids = face_ids[q]
    strict[quarantined_ids] = False
    return (strict, int(quarantined_ids.size))

def visible_object_mask_face_gate(*, frame: dict[str, Any], side: str, object_vertices: np.ndarray, object_faces: np.ndarray, object_pose: tuple[np.ndarray, np.ndarray], face_strict_observed: np.ndarray, mask: np.ndarray | None, A_mask_from_source: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, int, int]:
    from typing import Any
    import numpy as np
    import argparse
    from observations.interaction.evidence import project_world
    strict = np.asarray(face_strict_observed, dtype=bool).copy()
    raw_count = int(np.count_nonzero(strict))
    if mask is None or raw_count == 0:
        return (strict, raw_count, raw_count)
    if bool(args.visible_mask_quarantine_signed_mesh):
        strict[:] = False
        return (strict, raw_count, 0)
    if not bool(args.visible_object_mask_gate):
        return (strict, raw_count, raw_count)
    (r_obj, t_obj) = object_pose
    face_centers_world = object_vertices[object_faces].mean(axis=1) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    uv_source = project_world(face_centers_world, frame, side)
    inside = mask_membership(mask, uv_source, int(args.visible_object_mask_dilation_px), A_mask_from_source=A_mask_from_source)
    if inside.shape[0] == strict.shape[0]:
        strict &= inside
    return (strict, raw_count, int(np.count_nonzero(strict)))

def load_visible_object_mask_paths(report_path: Path | None) -> dict[int, Path]:
    from pathlib import Path
    from utils.data import as_list
    from utils.data import load_json
    if report_path is None:
        return {}
    if not report_path.exists():
        raise FileNotFoundError(f'missing visible object mask report: {report_path}')
    payload = load_json(report_path)
    rows = []
    if isinstance(payload, dict):
        rows.extend(as_list(payload.get('saved_mask_rows_after_start')))
        if not rows:
            rows.extend(as_list(payload.get('target_mask_rows')))
        if not rows:
            rows.extend(as_list(payload.get('surface_rows')))
    out: dict[int, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or 'frame_idx' not in row:
            continue
        mask_path = row.get('saved_mask_path') or row.get('mask_path')
        if not mask_path:
            continue
        path = Path(str(mask_path))
        if path.exists():
            out[int(row['frame_idx'])] = path
    return out

def load_hand_ray_shift_priors(path: Path | None) -> dict[tuple[int, str], float]:
    from pathlib import Path
    from utils.data import load_json
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out: dict[tuple[int, str], float] = {}
    for row in payload.get('rows', []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and isinstance(row.get('hand_ray_shift_m'), (int, float)):
            out[int(row['frame_idx']), str(row['hand_side'])] = float(row['hand_ray_shift_m'])
    return out

def project_world(points_world: np.ndarray, frame: dict[str, Any], side: str) -> np.ndarray | None:
    from typing import Any
    import numpy as np
    from gates.hand.observation import active_frame_intrinsics
    from utils.geometry import frame_camera_pose, project_world_intrinsics
    intr = active_frame_intrinsics(frame, side)
    (r_c2w, t_c2w) = frame_camera_pose(frame)
    return project_world_intrinsics(points_world, r_c2w, t_c2w, intr)
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
CONTACT_STATES = {'likely_contact', 'possible_contact', 'no_contact', 'unresolved'}
OCCLUSION_RELATIONS = {'hand_in_front_of_object', 'object_in_front_of_hand', 'object_partially_occluded_by_hand', 'no_visible_occlusion', 'unresolved'}
DEPTH_RELIABILITY_STATES = {'hand_depth_unreliable', 'object_depth_reliable', 'mixed_or_unresolved', 'not_evaluated'}
NO_QUARANTINE_OCCLUSIONS = {'object_in_front_of_hand', 'no_visible_occlusion'}
HAND_FRONT_OCCLUSIONS = {'hand_in_front_of_object', 'object_partially_occluded_by_hand'}

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []

def interaction_parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--target-entity-id', required=True)
    p.add_argument('--frame-span', type=int, nargs=2, required=True, metavar=('START', 'END'))
    p.add_argument('--sides', nargs='+', choices=('left', 'right'), default=['left', 'right'])
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--hand-radius-px', type=int, default=10)
    p.add_argument('--hand-line-radius-px', type=int, default=6)
    p.add_argument('--ownership-dilation-px', type=int, default=8)
    p.add_argument('--depth-order-query-radius-mask-px', type=float, default=6.0, help='Maximum mask-plane distance from a P09 accepted first-hit object sample at which a projected MANO vertex may query object depth. The hand pixel itself remains unknown; raw depth under the hand is never consumed.')
    p.add_argument('--contact-image-band-px', type=int, default=18)
    p.add_argument('--min-contact-image-px', type=int, default=12)
    p.add_argument('--contact-weight', type=float, default=25000.0)
    p.add_argument('--raw-proposal-weight-factor', type=float, default=0.65)
    p.add_argument('--contact-patch-band-m', type=float, default=0.18)
    p.add_argument('--contact-patch-target-margin-m', type=float, default=0.004)
    p.add_argument('--contact-support-uncertainty-m', type=float, default=0.06)
    p.add_argument('--agent-interaction-judgment', type=Path, default=None, help='Optional Pi-agent-authored interval contact/occlusion judgment JSON. When supplied, it calibrates contact priors, support uncertainty, and ownership quarantine policy instead of relying on projection adjacency alone.')
    p.add_argument('--agent-judgment-can-create-contact', action=argparse.BooleanOptionalAction, default=True, help='Allow a likely/possible agent contact judgment to emit a contact_patch prior even when the image-proximity pixel threshold is not met. The row still uses projected MANO/object geometry for targets; this only creates the prior/switch.')
    p.add_argument('--max-vertices', type=int, default=160)
    p.add_argument('--review-frames', type=int, nargs='*', default=[691, 700, 720, 725])
    p.add_argument('--require-camera-mano-contract-aligned', action=argparse.BooleanOptionalAction, default=False, help='Fail closed unless every consumed MANO row was inferred on the exact centered HaWoR image plane affinely bound to the active source [fx,fy,cx,cy] camera contract. A metadata-only K replacement is not accepted.')
    return p.parse_args(argv)

def target_object(frame: dict[str, Any], target_entity_id: str) -> dict[str, Any] | None:
    target_id = str(target_entity_id)
    bare = target_id.split(':', 1)[-1]
    for obj in as_list(frame.get('objects')):
        if not isinstance(obj, dict):
            continue
        ids = {str(obj.get('object_id')), str(obj.get('track_id')), f"object:{obj.get('track_id')}", f"object:{obj.get('object_id')}"}
        if target_id in ids or bare in ids:
            return obj
    return None

def target_matches(raw: Any, target_entity_id: str) -> bool:
    if raw in (None, ''):
        return True
    target = str(target_entity_id)
    bare = target.split(':', 1)[-1]
    val = str(raw)
    val_bare = val.split(':', 1)[-1]
    return val in {target, bare, f'object:{bare}'} or val_bare == bare

def finite_float(value: Any, *, field: str, path: Path) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f'agent interaction judgment field {field!r} is not numeric in {path}: {value!r}') from exc
    if not np.isfinite(out):
        raise ValueError(f'agent interaction judgment field {field!r} is not finite in {path}: {value!r}')
    return out

def agent_judgment_provenance(judgment: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(judgment, dict):
        return {'agent_interaction_judgment': None}
    keys = ['source_path', 'judgment_id', 'frame_start', 'frame_end', 'contact_state', 'occlusion_relation', 'depth_reliability', 'contact_prior_probability', 'contact_support_uncertainty_m', 'evidence', 'uncertainty']
    return {'agent_interaction_judgment': {k: judgment.get(k) for k in keys if k in judgment}}

def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('L')) > 0

def finite_intrinsics(value: Any, label: str) -> np.ndarray:
    intrinsics = np.asarray(value, dtype=np.float64).reshape(-1)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
        raise RuntimeError(f'{label} must be finite [fx,fy,cx,cy], got {intrinsics}')
    return intrinsics

def finite_affine(value: Any, label: str) -> np.ndarray:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise RuntimeError(f'{label} must be a finite 3x3 affine')
    if not np.allclose(affine[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise RuntimeError(f'{label} must use homogeneous affine pixel coordinates')
    if abs(float(np.linalg.det(affine))) < 1e-12:
        raise RuntimeError(f'{label} is singular')
    return affine

def mask_plane_contract(frame: dict[str, Any], obj: dict[str, Any], mask_shape: tuple[int, int]) -> dict[str, Any]:
    visible = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else {}
    transform = visible.get('mask_depth_transform_contract') if isinstance(visible.get('mask_depth_transform_contract'), dict) else {}
    if transform.get('camera_contract_consistent') is not True:
        raise RuntimeError(f"frame {frame.get('frame_idx')}: object mask lacks a camera-consistent mask/depth transform")
    mask_size = [int(v) for v in transform.get('mask_size_wh') or []]
    depth_size = [int(v) for v in transform.get('depth_size_wh') or []]
    actual_mask_size = [int(mask_shape[1]), int(mask_shape[0])]
    source_size = [int(frame.get('source_width') or 0), int(frame.get('source_height') or 0)]
    if mask_size != actual_mask_size:
        raise RuntimeError(f"frame {frame.get('frame_idx')}: mask raster {actual_mask_size} disagrees with contract {mask_size}")
    if depth_size != source_size or min(source_size) <= 0:
        raise RuntimeError(f"frame {frame.get('frame_idx')}: depth plane {depth_size} is not the source RGB plane {source_size}")
    A_depth_from_mask = finite_affine(transform.get('A_depth_from_mask_coordinate_model'), 'A_depth_from_mask_coordinate_model')
    A_mask_from_source = np.linalg.inv(A_depth_from_mask)
    pixel_center = str(transform.get('pixel_center_convention') or '')
    if pixel_center != 'integer_pixel_centers_opencv':
        raise RuntimeError(f"frame {frame.get('frame_idx')}: unsupported mask pixel-center convention {pixel_center!r}")
    return {'source_image_plane': 'source_rgb', 'mask_image_plane': str(transform.get('mask_image_plane') or 'sam2_mask'), 'depth_image_plane': str(transform.get('depth_image_plane') or 'source_rgb'), 'source_size_wh': source_size, 'mask_size_wh': mask_size, 'depth_size_wh': depth_size, 'A_mask_from_source_coordinate_model': A_mask_from_source.tolist(), 'A_depth_from_mask_coordinate_model': A_depth_from_mask.tolist(), 'pixel_center_convention': pixel_center, 'raster_resampling': transform.get('raster_resampling'), 'camera_contract_consistent': True}

def active_camera_mano_intrinsics(frame: dict[str, Any], hand: dict[str, Any], *, require_aligned: bool) -> np.ndarray:
    metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
    mano = finite_intrinsics(metric.get('current_v18_camera_intrinsics_fx_fy_cx_cy'), 'MANO inference intrinsics')
    if not require_aligned:
        return mano
    camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
    active = finite_intrinsics(camera.get('intrinsics_fx_fy_cx_cy'), 'active frame camera intrinsics')
    alignment = metric.get('camera_contract_alignment') if isinstance(metric.get('camera_contract_alignment'), dict) else hand.get('camera_contract_alignment') if isinstance(hand.get('camera_contract_alignment'), dict) else {}
    if require_aligned:
        if alignment.get('active_contract_reinference_required') is not False:
            raise RuntimeError(f"frame {frame.get('frame_idx')} {hand.get('hand_side')}: MANO was not inferred on the centered image plane bound to the active source K")
        if alignment.get('source_hawor_state_intrinsics_match_active_contract') is not True:
            raise RuntimeError(f"frame {frame.get('frame_idx')} {hand.get('hand_side')}: camera/MANO alignment is not explicitly proven")
        if alignment.get('source_hawor_full_K_bound_to_inference') is not True:
            raise RuntimeError(f"frame {frame.get('frame_idx')} {hand.get('hand_side')}: HaWoR did not prove source-K to centered-inference-plane binding")
        plane = alignment.get('hawor_camera_image_plane_contract')
        if not isinstance(plane, dict) or plane.get('validated') is not True or plane.get('status') != 'validated_source_K_affine_centered_hawor_inference_plane':
            raise RuntimeError(f"frame {frame.get('frame_idx')} {hand.get('hand_side')}: centered HaWoR image-plane binding is not validated")
        if not np.allclose(active, mano, atol=0.01, rtol=0.0):
            raise RuntimeError(f"frame {frame.get('frame_idx')} {hand.get('hand_side')}: active K {active.tolist()} != MANO K {mano.tolist()}")
    return active

def transform_source_pixels(uv_source: np.ndarray, A_mask_from_source: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv_source, dtype=np.float64)
    homogeneous = np.column_stack([uv, np.ones((len(uv),), dtype=np.float64)])
    transformed = homogeneous @ np.asarray(A_mask_from_source, dtype=np.float64).T
    return transformed[:, :2] / transformed[:, 2:3]

def project_camera_to_mask(points_camera: np.ndarray, intr: np.ndarray, A_mask_from_source: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_camera, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return (np.zeros((0,), dtype=int), np.zeros((0,), dtype=int), np.zeros((0,), dtype=bool))
    (fx, fy, cx, cy) = finite_intrinsics(intr, 'projection intrinsics')
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-05)
    uv_source = np.column_stack([fx * pts[:, 0] / np.maximum(z, 1e-06) + cx, fy * pts[:, 1] / np.maximum(z, 1e-06) + cy])
    uv_mask = transform_source_pixels(uv_source, A_mask_from_source)
    u = np.rint(uv_mask[:, 0]).astype(int)
    v = np.rint(uv_mask[:, 1]).astype(int)
    valid &= np.isfinite(uv_mask).all(axis=1) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return (u, v, valid)

def accepted_first_surface_query_raster(candidate: dict[str, Any], plane: dict[str, Any], mask_shape: tuple[int, int], radius_px: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rasterize only P09-accepted object first-hit samples for MANO queries.

    The resulting support band can overlap a projected hand even though the
    object-owned mask removed that hand region. Depth values are copied from the
    nearest accepted object sample, never from raw depth under the hand.
    """
    first_surface = candidate.get('first_surface_depth_ownership') if isinstance(candidate.get('first_surface_depth_ownership'), dict) else {}
    object_ownership = candidate.get('object_surface_ownership_filter') if isinstance(candidate.get('object_surface_ownership_filter'), dict) else {}
    if first_surface.get('enabled') is not True or first_surface.get('fail_closed') is True or object_ownership.get('fail_closed') is True:
        raise RuntimeError('P17 depth-order support requires P09 camera samples that passed first-surface depth ownership and object/hand ownership filtering')
    points = np.asarray(candidate.get('camera_vertices_sample_m') or [], dtype=np.float64)
    intr = finite_intrinsics(candidate.get('intrinsics_fx_fy_cx_cy'), 'P09 accepted first-surface intrinsics')
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32 or (not np.isfinite(points).all()):
        raise RuntimeError(f'P09 accepted first-surface camera points are invalid: {points.shape}')
    A_mask_from_source = finite_affine(plane.get('A_mask_from_source_coordinate_model'), 'A_mask_from_source_coordinate_model')
    (fx, fy, cx, cy) = intr.tolist()
    z = points[:, 2]
    positive = z > 1e-05
    uv_source = np.column_stack([fx * points[:, 0] / np.maximum(z, 1e-06) + cx, fy * points[:, 1] / np.maximum(z, 1e-06) + cy])
    uv_mask = transform_source_pixels(uv_source, A_mask_from_source)
    x = np.rint(uv_mask[:, 0]).astype(np.int64)
    y = np.rint(uv_mask[:, 1]).astype(np.int64)
    valid = positive & np.isfinite(uv_mask).all(axis=1) & (x >= 0) & (x < int(mask_shape[1])) & (y >= 0) & (y < int(mask_shape[0]))
    if int(np.count_nonzero(valid)) < 32:
        raise RuntimeError('too few P09 accepted first-surface points enter the mask plane')
    sparse = np.full(mask_shape, np.inf, dtype=np.float32)
    flat_index = y[valid] * int(mask_shape[1]) + x[valid]
    np.minimum.at(sparse.reshape(-1), flat_index, z[valid].astype(np.float32))
    seeds = np.isfinite(sparse)
    (distance, indices) = distance_transform_edt(~seeds, return_distances=True, return_indices=True)
    support = distance <= float(radius_px)
    nearest_depth = sparse[tuple(indices)]
    query_depth = np.full(mask_shape, np.nan, dtype=np.float32)
    query_depth[support] = nearest_depth[support]
    return (support, query_depth, {'source': 'P09 visible_geometry_candidate.camera_vertices_sample_m after first_surface_depth_ownership', 'accepted_first_surface_point_count': int(len(points)), 'projected_point_count': int(np.count_nonzero(valid)), 'unique_seed_pixel_count': int(np.count_nonzero(seeds)), 'query_support_pixel_count': int(np.count_nonzero(support)), 'query_radius_mask_px': float(radius_px), 'P09_first_surface_depth_ownership_state': first_surface.get('state'), 'P09_object_surface_ownership_state': object_ownership.get('state'), 'raw_depth_under_projected_hand_consumed': False, 'depth_sha256': hashlib.sha256(query_depth.astype('<f4').tobytes()).hexdigest()})

def draw_hand_mask(frame: dict[str, Any], hand: dict[str, Any], plane: dict[str, Any], width: int, height: int, *, point_radius: int, line_radius: int, require_camera_mano_aligned: bool) -> tuple[np.ndarray, dict[str, Any]]:
    metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
    intr = active_camera_mano_intrinsics(frame, hand, require_aligned=require_camera_mano_aligned)
    A_mask_from_source = finite_affine(plane.get('A_mask_from_source_coordinate_model'), 'A_mask_from_source_coordinate_model')
    joints = np.asarray(metric.get('joints_current_v18_camera_m') or [], dtype=float)
    verts = np.asarray(metric.get('vertices_camera_sample_m') or [], dtype=float)
    mask = np.zeros((height, width), dtype=np.uint8)
    joint_u = joint_v = np.zeros((0,), dtype=int)
    joint_valid = np.zeros((0,), dtype=bool)
    if joints.shape == (21, 3):
        (joint_u, joint_v, joint_valid) = project_camera_to_mask(joints, intr, A_mask_from_source, width, height)
        for (a, b) in HAND_EDGES:
            if a < len(joint_valid) and b < len(joint_valid) and joint_valid[a] and joint_valid[b]:
                cv2.line(mask, (int(joint_u[a]), int(joint_v[a])), (int(joint_u[b]), int(joint_v[b])), 255, max(1, int(line_radius)))
        for (u, v, ok) in zip(joint_u, joint_v, joint_valid):
            if ok:
                cv2.circle(mask, (int(u), int(v)), max(1, int(point_radius)), 255, -1)
    vert_valid_count = 0
    if verts.ndim == 2 and verts.shape[1] == 3:
        (vu, vv, vvld) = project_camera_to_mask(verts, intr, A_mask_from_source, width, height)
        vert_valid_count = int(np.count_nonzero(vvld))
        for (u, v, ok) in zip(vu, vv, vvld):
            if ok:
                cv2.circle(mask, (int(u), int(v)), max(1, int(point_radius // 2)), 255, -1)
    return (mask > 0, {'joint_valid_count': int(np.count_nonzero(joint_valid)), 'vertex_valid_count': int(vert_valid_count), 'hand_support_px': int(np.count_nonzero(mask)), 'active_camera_intrinsics_fx_fy_cx_cy': intr.astype(float).tolist(), 'image_plane_transform': plane})

def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0

def proximity_stats(hand_mask: np.ndarray, object_mask: np.ndarray, band_px: int) -> dict[str, Any]:
    hand_px = int(np.count_nonzero(hand_mask))
    object_px = int(np.count_nonzero(object_mask))
    overlap_px = int(np.count_nonzero(hand_mask & object_mask))
    if hand_px == 0 or object_px == 0:
        return {'hand_px': hand_px, 'object_px': object_px, 'overlap_px': overlap_px, 'near_px': 0, 'min_distance_px': None, 'median_near_distance_px': None}
    obj_u8 = object_mask.astype(np.uint8)
    dist = cv2.distanceTransform((1 - obj_u8).astype(np.uint8), cv2.DIST_L2, 3)
    dvals = dist[hand_mask]
    finite = dvals[np.isfinite(dvals)]
    near = finite <= float(band_px)
    return {'hand_px': hand_px, 'object_px': object_px, 'overlap_px': overlap_px, 'near_px': int(np.count_nonzero(near)), 'min_distance_px': float(np.min(finite)) if finite.size else None, 'median_near_distance_px': float(np.median(finite[near])) if np.any(near) else None}

def save_bool_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)

def make_review(raw_path: Path, object_mask: np.ndarray, hand_mask: np.ndarray, non_object_mask: np.ndarray, out: Path, title: str) -> None:
    img = cv2.imread(str(raw_path))
    if img is None:
        img = np.zeros((object_mask.shape[0], object_mask.shape[1], 3), dtype=np.uint8)
    if img.shape[:2] != object_mask.shape:
        img = cv2.resize(img, (object_mask.shape[1], object_mask.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = img.copy()
    overlay[object_mask] = (0.65 * overlay[object_mask] + 0.35 * np.array([40, 255, 80])).astype(np.uint8)
    overlay[hand_mask] = (0.65 * overlay[hand_mask] + 0.35 * np.array([255, 255, 0])).astype(np.uint8)
    overlay[non_object_mask] = (0.35 * overlay[non_object_mask] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.putText(overlay, title[:110], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

def interaction_build(args: argparse.Namespace) -> dict[str, Any]:
    from optimization.problems import combine_contact_prior
    from gates.interaction.contact import contact_row_state_from_judgment
    from gates.interaction.contact import load_agent_interaction_judgments
    from gates.interaction.visibility import ownership_quarantine_mode
    payload = load_json(args.annotations)
    frames = as_list(payload.get('frames')) if isinstance(payload, dict) else []
    (start, end) = [int(x) for x in args.frame_span]
    out_case = args.output_root / args.case
    mask_dir = out_case / 'ownership_masks'
    review_dir = out_case / 'review_frames'
    ownership_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    side_set = set((str(s) for s in args.sides))
    (judgment_index, judgment_summary) = load_agent_interaction_judgments(args.agent_interaction_judgment, case=str(args.case), target_entity_id=str(args.target_entity_id), frame_span=(start, end), sides=side_set)
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get('frame_idx', -1))
        if frame_idx < start or frame_idx > end:
            continue
        obj = target_object(frame, str(args.target_entity_id))
        if not isinstance(obj, dict) or not bool(obj.get('visible', True)):
            diagnostics.append({'frame_idx': frame_idx, 'state': 'missing_visible_target_object'})
            continue
        if obj.get('rigid_pose_observation_eligible') is False:
            diagnostics.append({'frame_idx': frame_idx, 'state': 'skipped_object_mask_ineligible_for_contact_ownership', 'reason': obj.get('rigid_pose_observation_reason'), 'object_area_px': obj.get('area_px')})
            continue
        mask_path_raw = obj.get('mask_path')
        if not isinstance(mask_path_raw, str) or not Path(mask_path_raw).exists():
            diagnostics.append({'frame_idx': frame_idx, 'state': 'missing_object_mask', 'mask_path': mask_path_raw})
            continue
        object_mask = load_mask(Path(mask_path_raw))
        (height, width) = object_mask.shape
        plane_contract = mask_plane_contract(frame, obj, object_mask.shape)
        candidate = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else None
        if candidate is None:
            diagnostics.append({'frame_idx': frame_idx, 'state': 'missing_P09_accepted_first_surface_for_depth_order_query'})
            continue
        candidate_K = finite_intrinsics(candidate.get('intrinsics_fx_fy_cx_cy'), 'P09 accepted first-surface intrinsics')
        frame_camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
        active_K = finite_intrinsics(frame_camera.get('intrinsics_fx_fy_cx_cy'), 'active frame camera intrinsics')
        if not np.allclose(candidate_K, active_K, atol=0.01, rtol=0.0):
            raise RuntimeError(f'frame {frame_idx}: P09 accepted first-surface K {candidate_K.tolist()} != active source K {active_K.tolist()}')
        (depth_order_query_support, depth_order_query_depth, depth_order_diag) = accepted_first_surface_query_raster(candidate, plane_contract, object_mask.shape, float(args.depth_order_query_radius_mask_px))
        object_contact_band = dilate(object_mask, int(args.contact_image_band_px))
        for hand in as_list(frame.get('hands')):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get('hand_side') or hand.get('side') or '')
            if side not in side_set:
                continue
            (hand_mask, hand_diag) = draw_hand_mask(frame, hand, plane_contract, width, height, point_radius=int(args.hand_radius_px), line_radius=int(args.hand_line_radius_px), require_camera_mano_aligned=bool(getattr(args, 'require_camera_mano_contract_aligned', False)))
            judgment = judgment_index.get((frame_idx, side))
            quarantine_mode = ownership_quarantine_mode(judgment)
            hand_for_ownership = dilate(hand_mask, int(args.ownership_dilation_px))
            if quarantine_mode == 'none':
                non_object_owned = np.zeros_like(object_mask, dtype=bool)
            else:
                non_object_owned = object_mask & hand_for_ownership
            visible_object_owned = object_mask & ~non_object_owned
            constraint_eligible = visible_object_owned.copy()
            prox = proximity_stats(hand_mask, object_mask, int(args.contact_image_band_px))
            contact_image_px = int(prox.get('overlap_px') or 0) + int(prox.get('near_px') or 0)
            image_candidate = contact_image_px >= int(args.min_contact_image_px)
            (emit_contact_row, contact_state) = contact_row_state_from_judgment(judgment, image_candidate=bool(image_candidate), allow_agent_create=bool(args.agent_judgment_can_create_contact))
            side_mask_dir = mask_dir / side
            non_object_path = side_mask_dir / f'{frame_idx:06d}_non_object_owned.png'
            visible_object_path = side_mask_dir / f'{frame_idx:06d}_visible_object_owned.png'
            constraint_path = side_mask_dir / f'{frame_idx:06d}_constraint_eligible_entity.png'
            hand_support_path = side_mask_dir / f'{frame_idx:06d}_projected_mano_hand_support.png'
            depth_order_support_path = side_mask_dir / f'{frame_idx:06d}_depth_order_query_support.png'
            depth_order_npz_path = side_mask_dir / f'{frame_idx:06d}_depth_order_first_surface.npz'
            save_bool_mask(non_object_path, non_object_owned)
            save_bool_mask(visible_object_path, visible_object_owned)
            save_bool_mask(constraint_path, constraint_eligible)
            save_bool_mask(hand_support_path, hand_mask)
            save_bool_mask(depth_order_support_path, depth_order_query_support)
            np.savez_compressed(depth_order_npz_path, depth_mask_plane_m=depth_order_query_depth.astype(np.float32), support_mask=depth_order_query_support.astype(np.uint8), A_mask_from_source_coordinate_model=np.asarray(plane_contract['A_mask_from_source_coordinate_model'], dtype=np.float64), source_size_wh=np.asarray(plane_contract['source_size_wh'], dtype=np.int32), mask_size_wh=np.asarray(plane_contract['mask_size_wh'], dtype=np.int32), source=np.asarray('P09 accepted first-surface samples; nearest sample within bounded mask-plane radius'))
            depth_order_support_sha256 = sha256_file(depth_order_support_path)
            depth_order_npz_sha256 = sha256_file(depth_order_npz_path)
            counts = {'non_object_owned_px': int(np.count_nonzero(non_object_owned)), 'visible_object_owned_px': int(np.count_nonzero(visible_object_owned)), 'constraint_eligible_entity_px': int(np.count_nonzero(constraint_eligible)), 'depth_order_query_support_px': int(np.count_nonzero(depth_order_query_support)), **hand_diag, **prox, 'contact_image_px': int(contact_image_px)}
            judgment_fields = agent_judgment_provenance(judgment)
            ownership_rows.append({'factor_family': 'visible_ownership', 'target_entity_id': str(args.target_entity_id), 'frame_idx': int(frame_idx), 'hand_side': side, 'variable_affected': 'O_t_surface_visibility', 'observation_type': 'projected_mano_visible_hand_ownership_mask', 'residual_or_quarantine_rule': 'quarantine object hard-surface constraints whose projected support overlaps the projected MANO hand support; this marks hand-owned/uncertain first-surface pixels, not accepted contact', 'provenance': {'annotations': str(args.annotations), 'object_mask_path': str(mask_path_raw), 'hand_support_mask_path': str(hand_support_path), 'reason': 'current MANO projection supplies hand-owned visible support where monocular depth at hand pixels is unreliable; Pi-agent interaction judgment configures whether this support quarantines object-surface constraints', **judgment_fields}, 'rendered_uncertainty_channel': 'visible object pixels under projected hand support are rendered/treated as ownership-uncertain rather than hard object surface when the agent interaction judgment says the hand is in front or unresolved', 'state': 'active_visible_ownership', 'agent_ownership_quarantine_mode': str(quarantine_mode), 'agent_contact_state': judgment.get('contact_state') if isinstance(judgment, dict) else None, 'agent_occlusion_relation': judgment.get('occlusion_relation') if isinstance(judgment, dict) else None, 'agent_depth_reliability': judgment.get('depth_reliability') if isinstance(judgment, dict) else None, 'image_plane_transform': plane_contract, 'non_object_owned_mask_path': str(non_object_path), 'visible_object_owned_mask_path': str(visible_object_path), 'constraint_eligible_entity_mask_path': str(constraint_path), 'adjusted_entity_mask_path': str(constraint_path), 'depth_order_query_support_mask_path': str(depth_order_support_path), 'depth_order_query_support_mask_sha256': depth_order_support_sha256, 'depth_order_first_surface_npz_path': str(depth_order_npz_path), 'depth_order_first_surface_npz_sha256': depth_order_npz_sha256, 'depth_order_query_semantics': 'projected MANO may query nearby P09-accepted object first-hit depth even where ownership mask marks hand/unknown; raw depth under hand is forbidden', 'depth_order_first_surface': depth_order_diag, 'counts': counts})
            min_dist = prox.get('min_distance_px')
            overlap_fraction = float(counts['overlap_px']) / max(1.0, float(counts['object_px']))
            near_fraction = float(counts['near_px']) / max(1.0, float(counts['hand_px']))
            if min_dist is None:
                proximity_fraction = 0.0
            else:
                proximity_fraction = max(0.0, min(1.0, 1.0 - float(min_dist) / max(1.0, float(args.contact_image_band_px))))
            image_contact_prior = max(0.05, min(0.65, 0.1 + 0.35 * proximity_fraction + 0.2 * min(1.0, 4.0 * overlap_fraction) + 0.1 * near_fraction))
            contact_prior = combine_contact_prior(float(image_contact_prior), judgment)
            support_uncertainty_m = float(args.contact_support_uncertainty_m)
            weight_multiplier = 1.0
            if isinstance(judgment, dict):
                support_uncertainty_m = float(judgment['contact_support_uncertainty_m'])
                weight_multiplier = float(judgment.get('contact_weight_multiplier', 1.0))
            active_weight = float(args.contact_weight) * max(float(args.raw_proposal_weight_factor), contact_prior) * float(weight_multiplier)
            if contact_state != 'active_contact_patch':
                active_weight = 0.0
            if emit_contact_row:
                blockers = ['no_persistent_object_frame_contact_anchor', 'raw_depth_at_hand_pixels_quarantined']
                if isinstance(judgment, dict):
                    blockers.append('agent_interval_judgment_not_a_persistent_object_frame_anchor')
                    if judgment.get('contact_state') == 'no_contact':
                        blockers.append('agent_judged_no_contact')
                    if contact_state == 'inactive_agent_contact_without_image_support':
                        blockers.append('agent_contact_judgment_without_projected_image_support')
                contact_rows.append({'factor_family': 'contact_patch', 'target_entity_id': str(args.target_entity_id), 'frame_idx': int(frame_idx), 'hand_side': side, 'variable_affected': 'H_t', 'observation_type': 'agent_judged_interaction_plus_projected_mano_object_image_adjacency', 'residual_or_quarantine_rule': 'Pi-agent interval contact/occlusion judgment sets the latent contact prior and support uncertainty; projected MANO/object geometry selects candidate local contact vertices when active; no persistent object-frame contact anchor is claimed', 'provenance': {'annotations': str(args.annotations), 'object_mask_path': str(mask_path_raw), 'hand_support_mask_path': str(hand_support_path), 'non_object_owned_mask_path': str(non_object_path), 'reason': 'Pi-agent visual interaction judgment supplies the contact/occlusion switch; projected MANO/object image adjacency supplies local geometric support and remains uncertain under raw hand-depth ambiguity', **judgment_fields}, 'rendered_uncertainty_channel': 'agent-judged contact/occlusion prior consumed as a soft graph switch; final render must still show posterior/uncertainty rather than treating the judgment as metric truth', 'state': str(contact_state), 'weight': float(active_weight), 'image_plane_transform': plane_contract, 'contact_patch_base_weight': float(args.contact_weight), 'contact_patch_band_m': float(args.contact_patch_band_m), 'contact_patch_target_margin_m': float(args.contact_patch_target_margin_m), 'contact_patch_support_uncertainty_m': float(support_uncertainty_m), 'object_support_uncertainty_m': float(support_uncertainty_m), 'contact_patch_deadband_m': float(args.contact_patch_target_margin_m + support_uncertainty_m), 'max_vertices': int(args.max_vertices), 'contact_state_prior_probability': float(contact_prior), 'contact_patch_prior_probability': float(contact_prior), 'image_contact_prior_probability': float(image_contact_prior), 'agent_contact_state': judgment.get('contact_state') if isinstance(judgment, dict) else None, 'agent_occlusion_relation': judgment.get('occlusion_relation') if isinstance(judgment, dict) else None, 'agent_depth_reliability': judgment.get('depth_reliability') if isinstance(judgment, dict) else None, 'agent_contact_weight_multiplier': float(weight_multiplier), 'contact_anchor_state': 'agent_interval_prior_no_stable_pose_anchor' if isinstance(judgment, dict) else 'image_adjacency_only_no_stable_pose_anchor', 'contact_anchor_residual_allowed': False, 'contact_anchor_blockers': blockers, 'contact_pose_anchor_key': None, 'image_contact_counts': counts, 'local_patch_sample_count': int(contact_image_px), 'local_patch_temporal_sample_count': 0, 'local_patch_support_state': 'agent_judgment_with_projected_mano_object_image_adjacency' if isinstance(judgment, dict) else 'projected_mano_object_image_adjacency', 'local_patch_support_consumed': bool(contact_state == 'active_contact_patch'), 'local_patch_support_uncertainty_m': float(support_uncertainty_m), 'global_object_support_uncertainty_m': float(support_uncertainty_m)})
            diagnostics.append({'frame_idx': frame_idx, 'hand_side': side, 'image_candidate_contact': bool(image_candidate), 'emitted_contact_row': bool(emit_contact_row), 'contact_row_state': str(contact_state), 'agent_judgment_id': judgment.get('judgment_id') if isinstance(judgment, dict) else None, 'agent_ownership_quarantine_mode': str(quarantine_mode), 'contact_prior_probability': float(contact_prior), 'image_contact_prior_probability': float(image_contact_prior), 'counts': counts})
            if frame_idx in set((int(x) for x in args.review_frames)):
                raw_path = Path(str(frame.get('raw_frame_path', '')))
                make_review(raw_path, object_mask, hand_mask, non_object_owned, review_dir / f'{frame_idx:06d}_{side}_contact_ownership.jpg', f'{frame_idx} {side} green=obj yellow=MANO red=hand-owned object')
    duplicate_keys = [(r['frame_idx'], r['hand_side']) for r in contact_rows]
    if len(duplicate_keys) != len(set(duplicate_keys)):
        raise ValueError('duplicate contact rows emitted')
    payload = {'schema': 'v19_visible_contact_ownership_factor_v2', 'method': 'v19_visible_contact_ownership_factor_from_projected_mano_object_mask_and_agent_interaction_judgment', 'case': str(args.case), 'target_entity_id': str(args.target_entity_id), 'claim_scope': 'Pi-agent interval interaction judgments supply explicit contact/occlusion priors; projected MANO/object image adjacency supplies local support and ownership masks. The report does not assert persistent contact anchors, exact hand depth, or object pose correction.', 'inputs': {'annotations': str(args.annotations), 'agent_interaction_judgment': None if args.agent_interaction_judgment is None else str(args.agent_interaction_judgment)}, 'parameters': {'frame_span': [start, end], 'sides': list(args.sides), 'hand_radius_px': int(args.hand_radius_px), 'hand_line_radius_px': int(args.hand_line_radius_px), 'ownership_dilation_px': int(args.ownership_dilation_px), 'depth_order_query_radius_mask_px': float(args.depth_order_query_radius_mask_px), 'contact_image_band_px': int(args.contact_image_band_px), 'min_contact_image_px': int(args.min_contact_image_px), 'contact_weight': float(args.contact_weight), 'contact_patch_band_m': float(args.contact_patch_band_m), 'contact_patch_target_margin_m': float(args.contact_patch_target_margin_m), 'contact_support_uncertainty_m': float(args.contact_support_uncertainty_m), 'agent_judgment_can_create_contact': bool(args.agent_judgment_can_create_contact), 'require_camera_mano_contract_aligned': bool(getattr(args, 'require_camera_mano_contract_aligned', False)), 'projection_image_plane_policy': 'project active K in source_rgb then apply exact P09 A_mask_from_source', 'depth_order_query_policy': 'use only P09 accepted first-surface object samples and a bounded nearest-sample mask-plane band; never consume raw depth under projected MANO'}, 'agent_interaction_judgment_summary': judgment_summary, 'summary': {'ownership_row_count': len(ownership_rows), 'contact_patch_row_count': len(contact_rows), 'active_contact_patch_row_count': int(sum((str(r.get('state')) == 'active_contact_patch' for r in contact_rows))), 'inactive_contact_patch_row_count': int(sum((str(r.get('state')) != 'active_contact_patch' for r in contact_rows))), 'contact_patch_state_counts': dict(Counter((str(r.get('state')) for r in contact_rows))), 'agent_ownership_quarantine_mode_counts': dict(Counter((str(r.get('agent_ownership_quarantine_mode')) for r in ownership_rows))), 'agent_contact_state_counts': dict(Counter((str(r.get('agent_contact_state')) for r in contact_rows if r.get('agent_contact_state') is not None))), 'agent_occlusion_relation_counts': dict(Counter((str(r.get('agent_occlusion_relation')) for r in ownership_rows if r.get('agent_occlusion_relation') is not None))), 'non_object_owned_px_total': int(sum(((r.get('counts') or {}).get('non_object_owned_px') or 0 for r in ownership_rows))), 'contact_image_px_total': int(sum(((r.get('image_contact_counts') or {}).get('contact_image_px') or 0 for r in contact_rows))), 'frames_with_active_contact_patch': sorted(set((int(r['frame_idx']) for r in contact_rows if str(r.get('state')) == 'active_contact_patch'))), 'frames_with_any_contact_patch_row': sorted(set((int(r['frame_idx']) for r in contact_rows)))}, 'ownership_rows': ownership_rows, 'factor_rows': contact_rows, 'diagnostics_preview': diagnostics[:160]}
    return payload

def interaction_factors_cli(argv=None) -> None:
    args = interaction_parse_args(argv)
    payload = interaction_build(args)
    out = args.output_root / args.case / 'v19_visible_contact_ownership_factor_report.json'
    write_json(out, payload)
    print(json.dumps({'status': 'ok', 'report': str(out), 'summary': payload['summary']}, indent=2))
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import cKDTree
from gates.object.geometry import load_signed_face_authority

def geometry_constraints_load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def pose_graph_readiness(report: dict[str, Any]) -> dict[str, Any]:
    """Treat an explicit unready/support-false P15 report as a physical-factor quarantine.

    Older pose reports did not publish these fields, so missing readiness remains a
    recorded legacy-compatible state rather than being silently converted to false.
    """
    graph = report.get('graph_support') if isinstance(report.get('graph_support'), dict) else {}
    annotation_ready = report.get('annotation_ready')
    support_sufficient = graph.get('sufficient')
    explicitly_unready = annotation_ready is False or support_sufficient is False
    return {'pose_report_status': report.get('status'), 'annotation_ready': annotation_ready if isinstance(annotation_ready, bool) else None, 'graph_support_sufficient': support_sufficient if isinstance(support_sufficient, bool) else None, 'explicitly_unready': explicitly_unready, 'legacy_readiness_fields_missing': annotation_ready is None and support_sufficient is None, 'policy': 'quarantine physical correction factors from unresolved object poses' if explicitly_unready else 'consume ready pose report or legacy report lacking an explicit rejection'}

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

def nearest_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {'count': 0, 'median': None, 'p90': None, 'p95': None, 'max': None, 'min': None}
    return {'count': int(values.size), 'median': float(np.median(values)), 'p90': float(np.percentile(values, 90)), 'p95': float(np.percentile(values, 95)), 'max': float(np.max(values)), 'min': float(np.min(values))}

def project(points_world: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray, intr: list[float]) -> np.ndarray:
    (fx, fy, cx, cy) = [float(x) for x in intr]
    cam = (points_world - t_c2w) @ r_c2w
    z = np.maximum(cam[:, 2], 1e-09)
    return np.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], axis=1)

def detector_bbox_in_intrinsics_grid(frame: dict[str, Any], hand_ann: dict[str, Any], intr: list[float], raw_video: dict[str, Any]) -> np.ndarray | None:
    bbox = hand_ann.get('bbox_xyxy') if isinstance(hand_ann, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    raw_w = float(raw_video.get('width') or 0.0)
    raw_h = float(raw_video.get('height') or 0.0)
    if raw_w <= 0.0 or raw_h <= 0.0:
        return None
    (_, _, cx, cy) = [float(x) for x in intr]
    scaled = np.asarray(bbox, dtype=float).copy()
    scaled[[0, 2]] *= 2.0 * cx / raw_w
    scaled[[1, 3]] *= 2.0 * cy / raw_h
    return scaled

def bbox_inside_fraction(uv: np.ndarray, bbox: np.ndarray) -> float:
    if uv.ndim != 2 or uv.shape[0] == 0:
        return 0.0
    inside = (uv[:, 0] >= bbox[0]) & (uv[:, 0] <= bbox[2]) & (uv[:, 1] >= bbox[1]) & (uv[:, 1] <= bbox[3]) & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    return float(np.mean(inside))

def visible_2d_consistency(frame: dict[str, Any], hand_ann: dict[str, Any], intr: list[float] | None, proj0: np.ndarray | None, proj1: np.ndarray | None, same_frame: bool, raw_video: dict[str, Any]) -> dict[str, Any]:
    if intr is None or proj0 is None or proj1 is None:
        return {'state': 'not_evaluated_missing_projection', 'compatible_with_visible_2d': False}
    bbox = detector_bbox_in_intrinsics_grid(frame, hand_ann, intr, raw_video)
    if bbox is None or not same_frame:
        return {'state': 'no_same_frame_detector_box_constraint', 'compatible_with_visible_2d': True}
    before = bbox_inside_fraction(proj0, bbox)
    after = bbox_inside_fraction(proj1, bbox)
    diag = float(np.linalg.norm([bbox[2] - bbox[0], bbox[3] - bbox[1]]))
    compatible = bool(after + 1e-12 >= before)
    return {'state': 'preserves_or_improves_detector_box_containment' if compatible else 'degrades_detector_box_containment', 'compatible_with_visible_2d': compatible, 'detector_bbox_intrinsics_grid_xyxy': [float(x) for x in bbox.tolist()], 'projected_joint_inside_detector_box_fraction_before': before, 'projected_joint_inside_detector_box_fraction_after': after, 'detector_bbox_diag_px_in_intrinsics_grid': diag}

def inverse_object(points_world: np.ndarray, r_w_from_c: np.ndarray, t_w: np.ndarray) -> np.ndarray:
    return (points_world - t_w) @ r_w_from_c

def forward_object_vec(vec_canonical: np.ndarray, r_w_from_c: np.ndarray) -> np.ndarray:
    return vec_canonical @ r_w_from_c.T

def hand_arrays(arrays: dict[str, np.ndarray], side: str, frame_pos: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(arrays[f'{side}_vertices_world_m'][frame_pos], dtype=float), np.asarray(arrays[f'{side}_joints_world_m'][frame_pos], dtype=float))

def frame_intrinsics(frame: dict[str, Any], side: str) -> list[float] | None:
    for hand in frame.get('hands', []) if isinstance(frame.get('hands'), list) else []:
        if hand.get('hand_side') == side:
            metric_state = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
            intr = hand.get('current_v18_camera_intrinsics_fx_fy_cx_cy') or metric_state.get('current_v18_camera_intrinsics_fx_fy_cx_cy')
            if intr and len(intr) == 4:
                return [float(x) for x in intr]
    return None

def frame_camera_pose(frame: dict[str, Any], fallback_r_c2w: np.ndarray, fallback_t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
    transform = np.asarray(camera.get('T_world_camera_metric') or [], dtype=float)
    if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
        return (transform[:3, :3], transform[:3, 3], 'annotation_frame_T_world_camera_metric')
    return (fallback_r_c2w, fallback_t_c2w, 'hawor_npz_camera_pose_fallback')

def completion_surface_contract(completion: dict[str, Any]) -> tuple[Path, str, dict[str, Any]]:
    outputs = completion.get('outputs') if isinstance(completion.get('outputs'), dict) else {}
    readiness = completion.get('geometry_readiness') if isinstance(completion.get('geometry_readiness'), dict) else {}
    explicit_collision = outputs.get('collision_eligible_mesh_labeled')
    legacy_completed = outputs.get('completed_mesh_labeled')
    mesh_value = explicit_collision or legacy_completed
    if not mesh_value:
        raise RuntimeError('completion report lacks collision-eligible or legacy completed surface mesh')
    semantics = 'collision_eligible_mesh_labeled' if explicit_collision else 'legacy_completed_mesh_labeled_unknown_collision_readiness'
    return (Path(str(mesh_value)), semantics, readiness)

def same_mesh_path(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)

def signed_geometry_source_readiness(completion_readiness: dict[str, Any], sign_mesh_source_report: Path | None, sign_mesh_is_completion_surface: bool, sign_mesh_path: Path | None=None) -> dict[str, Any]:
    if sign_mesh_source_report is not None:
        payload = geometry_constraints_load_json(sign_mesh_source_report)
        readiness = payload.get('geometry_readiness') if isinstance(payload.get('geometry_readiness'), dict) else {}
        outputs = payload.get('outputs') if isinstance(payload.get('outputs'), dict) else {}
        declared = readiness.get('signed_geometry_ready')
        expected_value = readiness.get('signed_geometry_mesh') or readiness.get('collision_eligible_mesh') or outputs.get('signed_geometry_mesh') or outputs.get('collision_eligible_mesh_labeled')
        if not expected_value:
            return {'source': str(sign_mesh_source_report), 'signed_geometry_ready': None, 'legacy_readiness_fields_missing': declared is None, 'source_mesh_contract_missing': True, 'reason': 'sign source report does not bind signed readiness to a concrete mesh path'}
        expected_path = Path(str(expected_value))
        if sign_mesh_path is None or not same_mesh_path(sign_mesh_path, expected_path):
            raise RuntimeError(f'sign mesh/source-report mismatch: signed readiness is bound to {expected_path}, not supplied sign mesh {sign_mesh_path}')
        return {'source': str(sign_mesh_source_report), 'source_mesh': str(expected_path), 'signed_geometry_ready': declared if isinstance(declared, bool) else None, 'legacy_readiness_fields_missing': declared is None, 'source_mesh_contract_missing': False}
    if sign_mesh_is_completion_surface:
        declared = completion_readiness.get('signed_geometry_ready')
        return {'source': 'completion_report.geometry_readiness', 'signed_geometry_ready': declared if isinstance(declared, bool) else None, 'legacy_readiness_fields_missing': declared is None}
    return {'source': None, 'signed_geometry_ready': None, 'legacy_readiness_fields_missing': True, 'reason': 'explicit sign mesh lacks --sign-mesh-source-report; watertightness alone cannot establish physical sign readiness'}

def build_geometric_constraints(argv=None) -> None:
    from optimization.common.solvers import least_norm_halfspace_escape
    import trimesh
    import open3d as o3d
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--hawor-npz', type=Path, required=True)
    parser.add_argument('--pose-report', type=Path, required=True)
    parser.add_argument('--completion-report', type=Path, required=True)
    parser.add_argument('--sign-mesh', type=Path, default=None, help='optional watertight aligned hidden-volume mesh for signed nonpenetration hypotheses')
    parser.add_argument('--sign-mesh-source-report', type=Path, default=None)
    parser.add_argument('--skip-signed-distance', action='store_true', help='broadphase only: do not run expensive exact signed distance')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--object-id', required=True)
    args = parser.parse_args(argv)
    annotations = geometry_constraints_load_json(args.annotations)
    pose_report = geometry_constraints_load_json(args.pose_report)
    pose_readiness = pose_graph_readiness(pose_report)
    quarantine_unready_pose = bool(pose_readiness['explicitly_unready'])
    pose_input_explicitly_ready = bool(pose_readiness['annotation_ready'] is True and pose_readiness['graph_support_sufficient'] is not False)
    completion = geometry_constraints_load_json(args.completion_report)
    (mesh_path, surface_mesh_semantics, geometry_readiness) = completion_surface_contract(completion)
    observed_band_m = float(completion.get('observed_band_m') or 0.0)
    surface_mesh = load_mesh(mesh_path)
    surface_mesh_watertight = bool(surface_mesh.is_watertight)
    surface_bounds_min = np.asarray(surface_mesh.bounds[0], dtype=float)
    surface_bounds_max = np.asarray(surface_mesh.bounds[1], dtype=float)
    (surface_samples, _) = trimesh.sample.sample_surface(surface_mesh, min(12000, max(1, len(surface_mesh.faces))), seed=np.random.default_rng(1818))
    surface_tree = cKDTree(np.asarray(surface_samples, dtype=float))
    sign_mesh_path = args.sign_mesh if args.sign_mesh is not None else mesh_path
    sign_mesh = load_mesh(sign_mesh_path)
    sign_mesh_watertight = bool(sign_mesh.is_watertight)
    sign_source_readiness = signed_geometry_source_readiness(geometry_readiness, args.sign_mesh_source_report, sign_mesh_is_completion_surface=args.sign_mesh is None, sign_mesh_path=sign_mesh_path)
    authority_payload = geometry_constraints_load_json(args.sign_mesh_source_report) if args.sign_mesh_source_report is not None else completion if args.sign_mesh is None else {}
    authority_report_path = args.sign_mesh_source_report if args.sign_mesh_source_report is not None else args.completion_report if args.sign_mesh is None else None
    signed_face_authority = load_signed_face_authority(authority_payload, source_report_path=authority_report_path, mesh_path=sign_mesh_path, mesh_face_count=len(sign_mesh.faces))
    signed_face_authority_mask = np.asarray(signed_face_authority.pop('mask'), dtype=bool)
    signed_face_authority.pop('provenance_code', None)
    sign_source_readiness['signed_face_authority'] = signed_face_authority
    signed_geometry_declared_ready = sign_source_readiness['signed_geometry_ready'] is True
    signed_geometry_query_eligible = bool(sign_mesh_watertight and signed_geometry_declared_ready and (signed_face_authority.get('usable') is True))
    signed_nonpenetration_physical_active = bool(signed_geometry_query_eligible and (not args.skip_signed_distance) and (not quarantine_unready_pose))
    sign_scene = None
    if not args.skip_signed_distance and signed_geometry_query_eligible:
        sign_scene = o3d.t.geometry.RaycastingScene()
        sign_scene.add_triangles(o3d.core.Tensor(np.asarray(sign_mesh.vertices, dtype=np.float32)), o3d.core.Tensor(np.asarray(sign_mesh.faces, dtype=np.uint32)))
    sign_bounds_min = np.asarray(sign_mesh.bounds[0], dtype=float)
    sign_bounds_max = np.asarray(sign_mesh.bounds[1], dtype=float)
    npz = np.load(args.hawor_npz, allow_pickle=True)
    arrays = {key: np.asarray(npz[key]) for key in npz.files}
    frame_indices = np.asarray(arrays['frame_idx'], dtype=int)
    frame_pos_by_idx = {int(f): i for (i, f) in enumerate(frame_indices)}
    frames_by_idx = {int(f.get('frame_idx')): f for f in annotations.get('frames', [])}
    raw_video_meta = annotations.get('raw_video') if isinstance(annotations.get('raw_video'), dict) else {}
    accepted_pose_statuses = {'fit_to_visible_depth_samples', 'fit_to_visible_depth_archive_vertices', 'corrected_temporal_rigid_pose_graph', 'completed_temporal_rigid_pose_uncertain'}
    pose_by_idx = {int(row['frame_idx']): row for row in pose_report.get('pose_rows', []) if row.get('status') in accepted_pose_statuses}
    rows = []
    corrective_rows = []
    for (frame_idx, pose) in pose_by_idx.items():
        if frame_idx not in frame_pos_by_idx:
            continue
        frame_pos = frame_pos_by_idx[frame_idx]
        frame = frames_by_idx.get(frame_idx, {})
        r_obj = np.asarray(pose['rotation_world_from_completed_canonical_matrix'], dtype=float)
        t_obj = np.asarray(pose['translation_world_m'], dtype=float)
        r_c2w_npz = np.asarray(arrays['R_c2w'][frame_pos], dtype=float)
        t_c2w_npz = np.asarray(arrays['t_c2w'][frame_pos], dtype=float)
        (r_c2w, t_c2w, camera_pose_source) = frame_camera_pose(frame, r_c2w_npz, t_c2w_npz)
        hands_by_side = {str(h.get('hand_side')): h for h in frame.get('hands', []) if isinstance(h, dict)}
        for side in ['left', 'right']:
            hand_ann = hands_by_side.get(side, {})
            metric_state = hand_ann.get('metric_mano_state') if isinstance(hand_ann.get('metric_mano_state'), dict) else {}
            ann_vertices_raw = hand_ann.get('vertices_world_sample_m') or metric_state.get('vertices_world_sample_m') or []
            ann_joints_raw = hand_ann.get('joints_current_v18_world_m') or metric_state.get('joints_current_v18_world_m') or []
            ann_vertices = np.asarray(ann_vertices_raw, dtype=float) if isinstance(hand_ann, dict) else np.asarray([], dtype=float)
            ann_joints = np.asarray(ann_joints_raw, dtype=float) if isinstance(hand_ann, dict) else np.asarray([], dtype=float)
            if ann_vertices.ndim == 2 and ann_vertices.shape[1] == 3 and (len(ann_vertices) > 0):
                verts_w = ann_vertices
                if ann_joints.ndim == 2 and ann_joints.shape[1] == 3 and (len(ann_joints) > 0):
                    joints_w = ann_joints
                else:
                    (_, joints_w) = hand_arrays(arrays, side, frame_pos)
                hand_geometry_source = 'delivered_annotation_metric_mano_vertices_world_sample_m'
            else:
                if int(arrays[f'{side}_valid'][frame_pos]) == 0:
                    continue
                (verts_w, joints_w) = hand_arrays(arrays, side, frame_pos)
                hand_geometry_source = 'hawor_npz_full_vertices_world_m_fallback'
            same_frame = bool(hand_ann.get('same_frame_detection')) if isinstance(hand_ann, dict) and hand_ann.get('same_frame_detection') is not None else bool(int(arrays[f'{side}_detected_same_frame'][frame_pos])) if f'{side}_detected_same_frame' in arrays else False
            verts_c = inverse_object(verts_w, r_obj, t_obj)
            surface_aabb = ((verts_c >= surface_bounds_min) & (verts_c <= surface_bounds_max)).all(axis=1)
            sign_aabb = ((verts_c >= sign_bounds_min) & (verts_c <= sign_bounds_max)).all(axis=1)
            (unsigned_surface_dist, _) = surface_tree.query(verts_c, k=1, workers=-1)
            near_surface = unsigned_surface_dist <= observed_band_m if observed_band_m > 0 else np.zeros(len(verts_c), dtype=bool)
            signed = np.full(len(verts_c), np.nan, dtype=float)
            penetrating = np.zeros(len(verts_c), dtype=bool)
            correction_c = np.zeros(3, dtype=float)
            correction_solver = {'solver': 'not_needed_no_authorized_penetration', 'success': True, 'constraint_count': 0}
            signed_broadphase_candidate = sign_aabb.copy()
            signed_local_authority_candidate = np.zeros(len(verts_c), dtype=bool)
            signed_local_authority_rejected = np.zeros(len(verts_c), dtype=bool)
            if signed_geometry_query_eligible and signed_broadphase_candidate.any() and (sign_scene is not None):
                broad_idx = np.where(signed_broadphase_candidate)[0]
                closest = sign_scene.compute_closest_points(o3d.core.Tensor(np.asarray(verts_c[broad_idx], dtype=np.float32)))
                primitive = closest['primitive_ids'].numpy().astype(np.int64)
                valid_primitive = (primitive >= 0) & (primitive < len(signed_face_authority_mask))
                authorized = np.zeros(len(broad_idx), dtype=bool)
                authorized[valid_primitive] = signed_face_authority_mask[primitive[valid_primitive]]
                signed_local_authority_candidate[broad_idx[authorized]] = True
                signed_local_authority_rejected[broad_idx[~authorized]] = True
            signed_query_candidate = signed_local_authority_candidate
            unresolved_topology_penetrating = np.zeros(len(verts_c), dtype=bool)
            unresolved_topology_signed = np.full(len(verts_c), np.nan, dtype=float)
            if signed_geometry_query_eligible and signed_local_authority_rejected.any() and (sign_scene is not None):
                rejected_idx = np.where(signed_local_authority_rejected)[0]
                rejected_signed = -sign_scene.compute_signed_distance(o3d.core.Tensor(np.asarray(verts_c[rejected_idx], dtype=np.float32))).numpy().astype(float)
                unresolved_topology_signed[rejected_idx] = rejected_signed
                unresolved_topology_penetrating[rejected_idx] = rejected_signed > 0.0
            if signed_geometry_query_eligible and signed_query_candidate.any() and (sign_scene is not None):
                candidate_idx = np.where(signed_query_candidate)[0]
                query_tensor = o3d.core.Tensor(np.asarray(verts_c[candidate_idx], dtype=np.float32))
                candidate_signed = -sign_scene.compute_signed_distance(query_tensor).numpy().astype(float)
                signed[candidate_idx] = candidate_signed
                penetrating[candidate_idx] = candidate_signed > 0.0
                if penetrating.any():
                    penetrating_points = o3d.core.Tensor(np.asarray(verts_c[penetrating], dtype=np.float32))
                    closest = sign_scene.compute_closest_points(penetrating_points)['points'].numpy().astype(float)
                    (correction_c, correction_solver) = least_norm_halfspace_escape(verts_c[penetrating], closest, signed[penetrating])
            correction_w = forward_object_vec(correction_c, r_obj)
            correction_norm_m = float(np.linalg.norm(correction_w))
            max_penetration_m = float(np.nanmax(signed[penetrating])) if penetrating.any() else 0.0
            constraint_count = int(correction_solver.get('constraint_count') or 0)
            amplification_bound_m = float(np.sqrt(max(1, constraint_count)) * max_penetration_m) if penetrating.any() else 0.0
            amplification_ratio = float(correction_norm_m / max_penetration_m) if max_penetration_m > 0.0 else None
            bounded_local_escape = bool(not penetrating.any() or (correction_solver.get('success') is True and correction_norm_m <= amplification_bound_m + 1e-12))
            correction_solver['max_penetration_depth_m'] = max_penetration_m
            correction_solver['translation_to_max_penetration_ratio'] = amplification_ratio
            correction_solver['orthogonal_independent_constraints_bound_m'] = amplification_bound_m
            correction_solver['bounded_local_escape'] = bounded_local_escape
            correction_solver['bounded_local_escape_semantics'] = 'accepted only when ||translation|| <= sqrt(num_penetrating_vertices) * max_penetration_depth; larger amplification indicates conflicting local escape directions or invalid sign support for a rigid hand correction'
            joints_corr_w = joints_w + correction_w
            intr = frame_intrinsics(frame, side)
            reproj_summary = None
            proj0 = None
            proj1 = None
            visible_consistency = {'state': 'not_evaluated_missing_projection', 'compatible_with_visible_2d': False}
            if intr is not None:
                proj0 = project(joints_w, r_c2w, t_c2w, intr)
                proj1 = project(joints_corr_w, r_c2w, t_c2w, intr)
                reproj_summary = nearest_summary(np.linalg.norm(proj1 - proj0, axis=1))
                visible_consistency = visible_2d_consistency(frame, hand_ann, intr, proj0, proj1, same_frame, raw_video_meta)
            if not signed_geometry_declared_ready:
                app_state = 'inactive_signed_geometry_not_ready'
                reason = 'completion/sign source does not explicitly declare signed_geometry_ready=true; watertightness, row count, or a single-view hidden prior cannot activate a signed nonpenetration correction'
                correction_solver = {'solver': 'not_run_signed_geometry_not_ready', 'success': False, 'constraint_count': 0}
            elif signed_face_authority.get('usable') is not True:
                app_state = 'inactive_missing_mesh_bound_local_signed_face_authority'
                reason = 'signed readiness cannot activate physical correction because the mesh-bound local face-authority contract is missing, unsafe, empty, or mismatched'
                correction_solver = {'solver': 'not_run_missing_local_signed_face_authority', 'success': False, 'constraint_count': 0}
            elif not sign_mesh_watertight:
                app_state = 'inactive_nonwatertight_mesh_no_signed_correction'
                reason = 'the declared sign source mesh is non-watertight, so signed nonpenetration remains inactive'
                correction_solver = {'solver': 'not_run_nonwatertight_sign_mesh', 'success': False, 'constraint_count': 0}
            elif penetrating.any() and bounded_local_escape and (visible_consistency.get('compatible_with_visible_2d') is True):
                app_state = 'candidate_coordinate_correction_visible_2d_compatible'
                reason = 'watertight sign mesh predicts local MANO/object penetration; least-norm local halfspace translation satisfies penetrating-vertex escape constraints and does not contradict available same-frame 2D hand-box evidence'
            elif penetrating.any() and correction_solver.get('success') is not True:
                app_state = 'not_applied_escape_solver_failed'
                reason = 'watertight sign mesh predicts MANO/object penetration, but the rigid hand translation halfspace solver failed; coordinate update is held as uncertainty'
            elif penetrating.any() and (not bounded_local_escape):
                app_state = 'not_applied_local_escape_amplified'
                reason = 'watertight sign mesh predicts MANO/object penetration, but satisfying local escape halfspaces requires translation larger than the penetration-depth-derived orthogonal constraint bound; this is treated as sign-support inconsistency, not H-prime'
            elif penetrating.any():
                app_state = 'not_applied_visible_2d_conflict_or_unmeasured'
                reason = 'watertight sign mesh predicts locally authorized MANO/object penetration, but available visible 2D consistency is missing or degraded; coordinate update is held'
            elif unresolved_topology_penetrating.any():
                app_state = 'unresolved_topology_penetration_nearest_face_lacks_local_authority'
                reason = 'watertight topology classifies MANO vertices inside, but their nearest boundary faces lack local first-hit authority; no physical correction is applied and signed promotion remains unresolved'
            elif signed_broadphase_candidate.any() and args.skip_signed_distance:
                app_state = 'uncertainty_signed_distance_not_evaluated_broadphase_support'
                reason = 'MANO vertices are within the observed-surface band and inside the sign-mesh AABB, but exact signed distance was intentionally skipped for broadphase measurement'
            elif near_surface.any() and sign_mesh_watertight and (not sign_aabb.any()):
                app_state = 'uncertainty_sign_mesh_missing_near_surface_support'
                reason = 'MANO vertices are within the observed-surface voxel band, but the watertight sign mesh does not cover those vertices; this falsifies using the sign mesh as a nonpenetration validator for this contact region'
            elif surface_aabb.any() and (not sign_mesh_watertight):
                app_state = 'uncertainty_only_nonwatertight_mesh_no_signed_correction'
                reason = 'surface overlap exists but no watertight sign mesh is available, so this changes uncertainty only'
            else:
                app_state = 'no_penetration_no_coordinate_change_needed'
                reason = 'sign-supporting compact-rigid mesh does not require nonpenetration correction for this hand/frame'
            diagnostic_candidate_translation_world_m = correction_w.astype(float).tolist()
            diagnostic_candidate_translation_norm_m = correction_norm_m
            diagnostic_candidate_joint_reprojection_shift_px = reproj_summary
            diagnostic_candidate_visible_2d_consistency = visible_consistency
            if quarantine_unready_pose:
                correction_w = np.zeros(3, dtype=float)
                correction_norm_m = 0.0
                reproj_summary = nearest_summary(np.zeros(len(joints_w), dtype=float)) if intr is not None else None
                visible_consistency = {'state': 'not_evaluated_physical_candidate_quarantined_unready_object_pose_trajectory', 'compatible_with_visible_2d': False}
                app_state = 'quarantined_unready_object_pose_trajectory'
                reason = 'P15 explicitly reports annotation_ready=false or insufficient graph support; object-relative MANO/contact and nonpenetration measurements are retained only as diagnostics and cannot produce a physical correction'
            row = {'frame_idx': int(frame_idx), 'hand_side': side, 'same_frame_detection': same_frame, 'object_id': args.object_id, 'status': 'mano_object_constraint_quarantined_unready_object_pose_trajectory' if quarantine_unready_pose else 'mano_object_constraint_measured' if signed_nonpenetration_physical_active else 'mano_object_unsigned_surface_measurement_signed_nonpenetration_inactive', 'annotation_ready': False, 'object_pose_input_ready_for_constraint_measurement': pose_input_explicitly_ready, 'object_pose_input_legacy_compatibility': bool(pose_readiness['legacy_readiness_fields_missing']), 'constraint_eligible_for_physical_correction': signed_nonpenetration_physical_active, 'object_pose_readiness': pose_readiness, 'surface_mesh_path': str(mesh_path), 'surface_mesh_semantics': surface_mesh_semantics, 'completion_geometry_readiness': geometry_readiness, 'signed_geometry_source_readiness': sign_source_readiness, 'signed_geometry_query_eligible': signed_geometry_query_eligible, 'signed_nonpenetration_factor_active': signed_nonpenetration_physical_active, 'sign_mesh_path': str(sign_mesh_path), 'sign_mesh_source_report': str(args.sign_mesh_source_report) if args.sign_mesh_source_report else None, 'signed_distance_semantics': 'positive_inside_negative_outside_zero_on_surface_open3d_raycasting_converted_from_negative_inside', 'completed_surface_mesh_watertight': surface_mesh_watertight, 'sign_mesh_watertight': sign_mesh_watertight, 'hand_geometry_source': hand_geometry_source, 'camera_pose_source_for_reprojection': camera_pose_source, 'hand_vertex_count': int(len(verts_w)), 'observed_band_m': observed_band_m, 'near_surface_vertex_count': int(near_surface.sum()), 'near_surface_vertex_fraction': float(near_surface.mean()), 'surface_aabb_candidate_vertex_count': int(surface_aabb.sum()), 'surface_aabb_candidate_vertex_fraction': float(surface_aabb.mean()), 'sign_aabb_candidate_vertex_count': int(sign_aabb.sum()), 'sign_aabb_candidate_vertex_fraction': float(sign_aabb.mean()), 'signed_broadphase_policy': 'all_vertices_inside_sign_mesh_AABB_then_nearest_face_authority_filter', 'signed_broadphase_candidate_vertex_count': int(signed_broadphase_candidate.sum()), 'signed_broadphase_candidate_vertex_fraction': float(signed_broadphase_candidate.mean()), 'signed_local_authority_candidate_vertex_count': int(signed_local_authority_candidate.sum()), 'signed_local_authority_rejected_vertex_count': int(signed_local_authority_rejected.sum()), 'unresolved_topology_penetrating_vertex_count': int(unresolved_topology_penetrating.sum()), 'unresolved_topology_penetration_depth_m': nearest_summary(unresolved_topology_signed[unresolved_topology_penetrating]), 'signed_query_candidate_vertex_count': int(signed_query_candidate.sum()), 'signed_query_candidate_vertex_fraction': float(signed_query_candidate.mean()), 'penetrating_vertex_count': int(penetrating.sum()), 'penetrating_vertex_fraction': float(penetrating.mean()), 'penetration_depth_m': nearest_summary(signed[penetrating]) if sign_mesh_watertight else nearest_summary(np.asarray([], dtype=float)), 'signed_distance_m': nearest_summary(signed[np.isfinite(signed)]), 'nearest_surface_unsigned_m': nearest_summary(np.asarray(unsigned_surface_dist, dtype=float)), 'candidate_translation_world_m': correction_w.astype(float).tolist(), 'candidate_translation_norm_m': correction_norm_m, 'diagnostic_candidate_translation_world_m_before_pose_quarantine': diagnostic_candidate_translation_world_m, 'diagnostic_candidate_translation_norm_m_before_pose_quarantine': diagnostic_candidate_translation_norm_m, 'diagnostic_candidate_joint_reprojection_shift_px_before_pose_quarantine': diagnostic_candidate_joint_reprojection_shift_px, 'diagnostic_candidate_visible_2d_consistency_before_pose_quarantine': diagnostic_candidate_visible_2d_consistency, 'candidate_translation_solver': correction_solver, 'candidate_joint_reprojection_shift_px': reproj_summary, 'candidate_visible_2d_consistency': visible_consistency, 'candidate_application_state': app_state, 'reason': reason}
            rows.append(row)
            if penetrating.any() and signed_nonpenetration_physical_active:
                corrective_rows.append(row)
    by_side = {}
    for side in ['left', 'right']:
        side_rows = [r for r in rows if r['hand_side'] == side]
        correction_norms = np.asarray([r['candidate_translation_norm_m'] for r in side_rows], dtype=float)
        penetrations = np.asarray([r['penetrating_vertex_count'] for r in side_rows], dtype=float)
        by_side[side] = {'measured_frames': len(side_rows), 'frames_with_any_penetration': int(np.sum(penetrations > 0)) if len(penetrations) else 0, 'candidate_translation_norm_m': nearest_summary(correction_norms)}
    report = {'method': 'build_v18_mano_object_constraint_state', 'status': 'completed_quarantined_unready_object_pose_trajectory' if quarantine_unready_pose else 'ok' if signed_nonpenetration_physical_active else 'completed_unsigned_surface_measurement_signed_nonpenetration_inactive', 'annotation_ready': False, 'object_pose_input_ready_for_constraint_measurement': pose_input_explicitly_ready, 'object_pose_input_legacy_compatibility': bool(pose_readiness['legacy_readiness_fields_missing']), 'physical_constraint_quarantined': quarantine_unready_pose, 'signed_nonpenetration_constraint_quarantined': not signed_nonpenetration_physical_active, 'unsigned_collision_surface_measurement_available': True, 'object_pose_readiness': pose_readiness, 'claim_scope': 'Object-relative MANO/contact and nonpenetration values are diagnostic-only because P15 explicitly reports an unready object trajectory; no candidate coordinate correction is exposed downstream.' if quarantine_unready_pose else 'Collision-eligible surface supports unsigned proximity measurement, but signed nonpenetration is inactive because geometry readiness is not explicitly true, the sign mesh is not watertight, or signed evaluation was skipped; no coordinate correction is exposed.' if not signed_nonpenetration_physical_active else 'Collision-eligible posed object surface supplies unsigned measurements and an explicitly ready watertight sign source supplies signed nonpenetration hypotheses. Coordinate corrections remain candidates until visible 2D consistency is inspected.', 'object_id': args.object_id, 'inputs': {'annotations': str(args.annotations), 'hawor_npz': str(args.hawor_npz), 'pose_report': str(args.pose_report), 'completion_report': str(args.completion_report), 'completed_surface_mesh': str(mesh_path), 'surface_mesh_semantics': surface_mesh_semantics, 'completion_geometry_readiness': geometry_readiness, 'sign_mesh': str(sign_mesh_path), 'sign_mesh_source_report': str(args.sign_mesh_source_report) if args.sign_mesh_source_report else None, 'skip_signed_distance': bool(args.skip_signed_distance)}, 'completed_surface_mesh_watertight': surface_mesh_watertight, 'surface_mesh_semantics': surface_mesh_semantics, 'completion_geometry_readiness': geometry_readiness, 'signed_geometry_source_readiness': sign_source_readiness, 'signed_geometry_query_eligible': signed_geometry_query_eligible, 'signed_face_authority': signed_face_authority, 'signed_nonpenetration_factor_active': signed_nonpenetration_physical_active, 'signed_nonpenetration_inactive_reason': None if signed_nonpenetration_physical_active else 'unready_object_pose' if quarantine_unready_pose else 'signed_distance_explicitly_skipped' if args.skip_signed_distance else 'signed_geometry_not_explicitly_ready_not_watertight_or_missing_local_face_authority', 'sign_mesh_watertight': sign_mesh_watertight, 'measured_pair_count': len(rows), 'candidate_correction_count': len(corrective_rows), 'diagnostic_penetrating_pair_count_before_pose_quarantine': int(sum((int(row.get('penetrating_vertex_count') or 0) > 0 for row in rows))), 'summary_by_side': by_side, 'constraint_rows': rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / 'v18_mano_object_constraint_state.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'object_id', 'surface_mesh_semantics', 'completed_surface_mesh_watertight', 'sign_mesh_watertight', 'signed_geometry_query_eligible', 'signed_nonpenetration_factor_active', 'measured_pair_count', 'candidate_correction_count', 'summary_by_side']}, indent=2))
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

def hand_visibility_load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def hand_visibility_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def hand_visibility_as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []

def normalize_zero_observation_interval(raw: Any, *, source: str) -> dict[str, Any]:
    """Normalize a visual-boundary interval into a durable factor-input record."""
    if isinstance(raw, str):
        parts = raw.split(':', 3)
        if len(parts) != 4:
            raise ValueError(f'zero-observation interval must be side:start:end:reason, got {raw!r}')
        (side, start_s, end_s, reason) = parts
        record: dict[str, Any] = {'hand_side': side, 'start_frame': int(start_s), 'end_frame': int(end_s), 'reason': reason, 'spec': raw}
    elif isinstance(raw, dict):
        record = dict(raw)
        if 'side' in record and 'hand_side' not in record:
            record['hand_side'] = record['side']
        record.setdefault('spec', f"{record.get('hand_side')}:{record.get('start_frame')}:{record.get('end_frame')}:{record.get('reason')}")
    else:
        raise TypeError(f'zero-observation interval must be string or dict, got {type(raw).__name__}')
    side = str(record.get('hand_side') or '').strip()
    if side not in {'left', 'right'}:
        raise ValueError(f'zero-observation interval side must be left/right, got {side!r} from {source}')
    start = int(record.get('start_frame'))
    end = int(record.get('end_frame'))
    if end < start:
        raise ValueError(f'zero-observation interval end before start: {record!r} from {source}')
    reason = str(record.get('reason') or '').strip()
    if not reason:
        raise ValueError(f'zero-observation interval reason must be nonempty: {record!r} from {source}')
    record['hand_side'] = side
    record['start_frame'] = start
    record['end_frame'] = end
    record['reason'] = reason
    review_frames = record.get('raw_rgb_review_frames')
    if review_frames is not None:
        if not isinstance(review_frames, list) or not all((isinstance(p, str) and Path(p).exists() for p in review_frames)):
            raise ValueError(f'zero-observation interval raw_rgb_review_frames must be existing paths: {record!r} from {source}')
    record['source'] = source
    record.setdefault('observation_type', 'explicit_visual_boundary_invalidates_visible_mano_observation_interval')
    record.setdefault('residual_or_quarantine_rule', 'set MANO joint/root/pose visible-observation weights to zero for this frame/side; preserve temporal smoothness and eligible visible object first-surface constraints')
    record.setdefault('rendered_uncertainty_channel', 'latent occluded-hand hypothesis; no contact, object ownership, object pose, nonpenetration, or known hidden-hand pose claim')
    record.setdefault('candidate_px_source_field', 'current_reproject_all_in_front_count')
    record.setdefault('candidate_unit', 'mano_vertices_in_front_of_visible_first_surface_at_current_projection')
    return record

def parse_zero_observation_interval(spec: str) -> dict[str, Any]:
    """Parse side:start:end:reason into a durable factor-input record."""
    return normalize_zero_observation_interval(spec, source='cli:--zero-observation-interval')

def load_zero_observation_interval_files(paths: list[Path]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for path in paths:
        payload = hand_visibility_load_json(path)
        if isinstance(payload, dict):
            raw_items = payload.get('intervals', payload.get('zero_observation_intervals'))
        else:
            raw_items = payload
        if not isinstance(raw_items, list):
            raise ValueError(f'zero-observation interval file must contain a list or intervals list: {path}')
        for raw in raw_items:
            intervals.append(normalize_zero_observation_interval(raw, source=str(path)))
    return intervals

def load_remeasurement_cache(path: Path, cache: dict[Path, dict[tuple[int, str], dict[str, Any]]]) -> dict[tuple[int, str], dict[str, Any]]:
    if path not in cache:
        payload = hand_visibility_load_json(path)
        rows = {}
        for row in hand_visibility_as_list(payload.get('rows')) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            rows[int(row['frame_idx']), str(row['hand_side'])] = row
        cache[path] = rows
    return cache[path]

def transition_metric_evidence(interval: dict[str, Any], frame_idx: int, hand_side: str, cache: dict[Path, dict[tuple[int, str], dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
    raw_path = interval.get('optimized_projection_remeasurement_report') or interval.get('remeasurement_report') or interval.get('metric_evidence_report')
    if raw_path in (None, ''):
        return (0, {})
    path = Path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f'zero-observation interval metric evidence report does not exist: {path}')
    rows = load_remeasurement_cache(path, cache)
    key = (int(frame_idx), str(hand_side))
    if key not in rows:
        raise KeyError(f'zero-observation interval metric evidence report {path} lacks row {key}')
    row = rows[key]
    field = str(interval.get('candidate_px_source_field') or 'current_reproject_all_in_front_count')
    if field not in row:
        raise KeyError(f'zero-observation interval metric evidence row {key} lacks field {field!r} in {path}')
    candidate_px = int(row.get(field) or 0)
    return (candidate_px, {'optimized_projection_remeasurement_report': str(path), 'optimized_projection_candidate_px_field': field, 'optimized_projection_candidate_px': candidate_px, 'optimized_projection_mask_path': row.get('mask_path'), 'optimized_projection_visible_surface_state': row.get('visible_surface_state')})

def hand_visibility_parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--ownership-factor-report', type=Path, required=True)
    p.add_argument('--target-entity-id', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--joint-observation-weight-multiplier', type=float, default=0.12)
    p.add_argument('--zero-observation-interval', action='append', default=[], help='Optional durable visual-boundary interval as side:start:end:reason. Emits zero-observation hand_observation_visibility rows from explicit evidence input rather than ownership candidates. Repeatable.')
    p.add_argument('--zero-observation-intervals-json', type=Path, action='append', default=[], help='JSON file containing visual-boundary zero-observation interval records. Each record may include metric evidence/provenance fields such as optimized_projection_remeasurement_report and raw_rgb_review_frames.')
    return p.parse_args(argv)

def build_hand_visibility(args: argparse.Namespace) -> dict[str, Any]:
    ownership = hand_visibility_load_json(args.ownership_factor_report)
    factor_rows: list[dict[str, Any]] = []
    skipped = Counter()
    for row in hand_visibility_as_list(ownership.get('ownership_rows')) if isinstance(ownership, dict) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get('target_entity_id')) != str(args.target_entity_id):
            skipped['wrong_target'] += 1
            continue
        counts = row.get('counts') if isinstance(row.get('counts'), dict) else {}
        candidate_px = int(counts.get('candidate_non_object_owned_px') or counts.get('aligned_hand_entity_overlap_px') or 0)
        hard_px = int(counts.get('non_object_owned_px') or 0)
        hard_state = str(row.get('hard_ownership_state') or '')
        prompt_independent = bool(row.get('hard_ownership_prompt_independent'))
        if candidate_px <= 0:
            skipped['no_candidate'] += 1
            continue
        if hard_px > 0 or prompt_independent:
            skipped['independent_hard_ownership_not_occlusion_conflict'] += 1
            continue
        state = 'active_hand_observation_visibility'
        factor_rows.append({'factor_family': 'hand_observation_visibility', 'target_entity_id': str(args.target_entity_id), 'frame_idx': int(row['frame_idx']), 'hand_side': str(row['hand_side']), 'variable_affected': 'H_t', 'observation_type': 'visible_object_surface_conflicts_with_mano_prompted_hand_candidate_without_independent_visible_hand', 'residual_or_quarantine_rule': 'multiply MANO joint/pose visible-observation weights by joint_observation_weight_multiplier for this frame/side; keep visible object first-surface constraints active', 'rendered_uncertainty_channel': 'ownership review frame renders MANO-prompt self-confirmed hand candidate as orange occluded_or_unresolved, not hard hand-owned', 'state': state, 'candidate_px': int(candidate_px), 'hard_non_object_owned_px': int(hard_px), 'joint_observation_weight_multiplier': float(args.joint_observation_weight_multiplier), 'source_hard_ownership_state': hard_state, 'provenance': {'ownership_factor_report': str(args.ownership_factor_report), 'ownership_review_frame_path': row.get('review_frame_path'), 'ownership_hand_prompt_source': (row.get('provenance') or {}).get('hand_prompt_source') if isinstance(row.get('provenance'), dict) else None, 'reason': 'candidate visible-hand/entity overlap came from MANO-seeded prompt without independent visible-hand confirmation'}})
    transition_intervals = [parse_zero_observation_interval(spec) for spec in hand_visibility_as_list(args.zero_observation_interval)]
    transition_intervals.extend(load_zero_observation_interval_files(list(args.zero_observation_intervals_json or [])))
    rows_by_key = {(int(r['frame_idx']), str(r['hand_side'])): r for r in factor_rows}
    transition_extension_row_count = 0
    transition_existing_row_count = 0
    remeasurement_cache: dict[Path, dict[tuple[int, str], dict[str, Any]]] = {}
    for interval in transition_intervals:
        if interval.get('target_entity_id') not in (None, '', args.target_entity_id):
            raise ValueError(f"zero-observation interval target {interval.get('target_entity_id')!r} does not match {args.target_entity_id!r}")
        for frame_idx in range(int(interval['start_frame']), int(interval['end_frame']) + 1):
            hand_side = str(interval['hand_side'])
            key = (frame_idx, hand_side)
            (candidate_px, metric_provenance) = transition_metric_evidence(interval, frame_idx, hand_side, remeasurement_cache)
            provenance = {'ownership_factor_report': str(args.ownership_factor_report), 'visual_boundary_interval_spec': str(interval['spec']), 'visual_boundary_interval_source': str(interval.get('source')), 'visual_boundary_summary': interval.get('visual_boundary_summary'), 'raw_rgb_review_frames': interval.get('raw_rgb_review_frames'), 'reason': str(interval['reason']), **metric_provenance}
            if key in rows_by_key:
                skipped['zero_observation_interval_existing_row_updated'] += 1
                transition_existing_row_count += 1
                row = rows_by_key[key]
                row['observation_type'] = str(interval['observation_type'])
                row['residual_or_quarantine_rule'] = str(interval['residual_or_quarantine_rule'])
                row['rendered_uncertainty_channel'] = str(interval['rendered_uncertainty_channel'])
                row['state'] = 'active_hand_observation_visibility'
                row['candidate_px'] = int(candidate_px if metric_provenance else row.get('candidate_px', 0) or 0)
                row['candidate_unit'] = str(interval.get('candidate_unit') or row.get('candidate_unit') or '')
                row['hard_non_object_owned_px'] = 0
                row['joint_observation_weight_multiplier'] = 0.0
                row['source_hard_ownership_state'] = str(interval.get('source_hard_ownership_state') or 'explicit_visual_boundary_interval')
                old_provenance = row.get('provenance') if isinstance(row.get('provenance'), dict) else {}
                row['provenance'] = {'pre_existing_hand_observation_visibility_provenance': old_provenance, **provenance}
                continue
            transition_extension_row_count += 1
            row = {'factor_family': 'hand_observation_visibility', 'target_entity_id': str(args.target_entity_id), 'frame_idx': int(frame_idx), 'hand_side': hand_side, 'variable_affected': 'H_t', 'observation_type': str(interval['observation_type']), 'residual_or_quarantine_rule': str(interval['residual_or_quarantine_rule']), 'rendered_uncertainty_channel': str(interval['rendered_uncertainty_channel']), 'state': 'active_hand_observation_visibility', 'candidate_px': int(candidate_px), 'candidate_unit': str(interval.get('candidate_unit') or ''), 'hard_non_object_owned_px': 0, 'joint_observation_weight_multiplier': 0.0, 'source_hard_ownership_state': str(interval.get('source_hard_ownership_state') or 'explicit_visual_boundary_interval'), 'provenance': provenance}
            rows_by_key[key] = row
            factor_rows.append(row)
    factor_rows.sort(key=lambda r: (int(r['frame_idx']), str(r['hand_side'])))
    return {'method': 'v18_hand_observation_visibility_factor', 'case': args.case, 'target_entity_id': args.target_entity_id, 'claim_scope': 'This factor downweights or zeros MANO observation anchoring for frame/side hand hypotheses that conflict with visible object first surfaces without independent visible-hand ownership evidence, including explicit visual-boundary intervals with metric first-surface conflict evidence. It does not claim contact, object pose, nonpenetration, known hidden-hand pose, or hand-owned object pixels.', 'inputs': {'ownership_factor_report': str(args.ownership_factor_report), 'zero_observation_intervals_json': [str(p) for p in list(args.zero_observation_intervals_json or [])]}, 'parameters': {'joint_observation_weight_multiplier': float(args.joint_observation_weight_multiplier), 'zero_observation_intervals': transition_intervals}, 'summary': {'factor_row_count': int(len(factor_rows)), 'transition_interval_frame_count': int(sum((int(i['end_frame']) - int(i['start_frame']) + 1 for i in transition_intervals))), 'transition_extension_row_count': int(transition_extension_row_count), 'transition_existing_row_count': int(transition_existing_row_count), 'zero_observation_row_count': int(sum((float(r.get('joint_observation_weight_multiplier', 1.0)) == 0.0 for r in factor_rows))), 'candidate_px_sum': int(sum((int(r['candidate_px']) for r in factor_rows))), 'candidate_px_sum_note': 'legacy diagnostic only; explicit visual-boundary rows may use candidate_unit other than pixels, so unit-specific sums below are authoritative', 'ownership_candidate_px_sum': int(sum((int(r.get('candidate_px') or 0) for r in factor_rows if not (isinstance(r.get('provenance'), dict) and r['provenance'].get('optimized_projection_remeasurement_report'))))), 'metric_transition_candidate_count_sum': int(sum((int(r.get('candidate_px') or 0) for r in factor_rows if isinstance(r.get('provenance'), dict) and r['provenance'].get('optimized_projection_remeasurement_report')))), 'state_counts': dict(Counter((str(r.get('state')) for r in factor_rows))), 'skipped_counts': dict(skipped)}, 'factor_rows': factor_rows}
