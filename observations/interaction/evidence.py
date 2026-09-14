from __future__ import annotations
'Appearance observations are distinct from metric ownership masks.'
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
