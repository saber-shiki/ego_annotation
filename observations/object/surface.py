from __future__ import annotations
from utils.data import read_json
from utils.data import write_json_payload
from utils.data import list_or_empty
from envs.qualification.object_observation import conservative_mask_on_depth
'Extracted formal interval mechanisms; no historical script imports.'

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
import hashlib
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
SCHEMA = 'rgb_scene_native_depth_metric_eligibility_v1'
RASTER_SCHEMA = 'rgb_scene_native_depth_metric_eligibility_raster_v1'
CLASS_CODES = {'outside_owned_appearance': 0, 'hand_clearly_front': 1, 'hand_clearly_behind': 2, 'near_depth_ambiguous': 3, 'clear_no_hand_overlap_candidate': 4, 'no_hit_uncertainty_fringe': 5, 'unsupported_depth': 6, 'invalid_mask_coordinate': 7, 'unresolved_hand_geometry': 8}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(json.dumps(list(array.shape), separators=(',', ':')).encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()

def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'metric eligibility input missing: {path}')
    return {'path': str(path), 'bytes': int(path.stat().st_size), 'sha256': sha256(path)}

class MetricEligibility:

    def __init__(self, manifest_path: Path, *, case: str, object_id: str, track_id: str, sam2_track: Path, depth_npz: Path, base_annotations: Path | None, calibration_contract: Path | None, frame_to_depth_i: dict[int, int], depth: np.ndarray):
        self.path = Path(manifest_path).expanduser().resolve()
        value = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or value.get('schema') != SCHEMA:
            raise ValueError('wrong metric eligibility manifest schema')
        if value.get('class_codes') != CLASS_CODES or value.get('policy', {}).get('physical_authority') is not False:
            raise ValueError('metric eligibility class/authority contract mismatch')
        for (key, expected) in (('case', case), ('object_id', object_id), ('track_id', track_id)):
            if value.get(key) != expected:
                raise ValueError(f'metric eligibility identity mismatch: {key}')
        expected_inputs = {'owned_sam2_track': sam2_track, 'depth_npz': depth_npz, 'base_annotations': base_annotations, 'calibration_contract': calibration_contract}
        inputs = value.get('inputs')
        if not isinstance(inputs, dict):
            raise ValueError('metric eligibility input identity missing')
        for (name, row) in inputs.items():
            if row is None:
                continue
            if not isinstance(row, dict) or not isinstance(row.get('path'), str):
                raise ValueError(f'malformed metric eligibility input: {name}')
            recorded_path = Path(row['path']).expanduser().resolve()
            if not recorded_path.is_file() or row.get('sha256') != sha256(recorded_path) or row.get('bytes') != recorded_path.stat().st_size:
                raise ValueError(f'metric eligibility recorded input changed: {name}')
        for (name, expected_path) in expected_inputs.items():
            if expected_path is None:
                if inputs.get(name) is not None:
                    raise ValueError(f'metric eligibility unexpected input: {name}')
                continue
            expected_path = Path(expected_path).expanduser().resolve()
            row = inputs.get(name)
            if not isinstance(row, dict) or row.get('path') != str(expected_path) or row.get('sha256') != sha256(expected_path) or (row.get('bytes') != expected_path.stat().st_size):
                raise ValueError(f'metric eligibility input path/hash mismatch: {name}')
        rows = value.get('frames')
        if not isinstance(rows, list) or len(rows) != len(frame_to_depth_i):
            raise ValueError('metric eligibility frame coverage mismatch')
        self.rows = {}
        self.depth = np.asarray(depth)
        self.frame_to_depth_i = frame_to_depth_i
        for row in rows:
            if not isinstance(row, dict) or type(row.get('frame_idx')) is not int:
                raise ValueError('malformed metric eligibility frame')
            idx = row['frame_idx']
            if idx in self.rows or idx not in frame_to_depth_i:
                raise ValueError('duplicate/foreign metric eligibility frame')
            if row.get('depth_array_sha256') != array_sha256(self.depth[frame_to_depth_i[idx]]):
                raise ValueError(f'metric eligibility depth row changed: {idx}')
            self.rows[idx] = row
        if set(self.rows) != set(frame_to_depth_i):
            raise ValueError('metric eligibility frame IDs mismatch')
        self.manifest = value

    def frame(self, frame_idx: int, *, owned_mask_path: Path, depth_shape: tuple[int, int], A_depth_from_mask: np.ndarray, K_depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        row = self.rows.get(frame_idx)
        if row is None:
            raise ValueError(f'metric eligibility lacks frame {frame_idx}')
        owned_mask_path = Path(owned_mask_path).expanduser().resolve()
        owned = row.get('owned_mask')
        if not isinstance(owned, dict) or owned.get('path') != str(owned_mask_path) or owned.get('sha256') != sha256(owned_mask_path):
            raise ValueError(f'metric eligibility owned mask changed: {frame_idx}')
        if np.asarray(row.get('A_depth_from_mask'), dtype=float).shape != (3, 3) or not np.allclose(np.asarray(row['A_depth_from_mask'], dtype=float), np.asarray(A_depth_from_mask, dtype=float), atol=1e-09, rtol=0) or np.asarray(row.get('K_depth'), dtype=float).shape != (3, 3) or (not np.allclose(np.asarray(row['K_depth'], dtype=float), np.asarray(K_depth, dtype=float), atol=1e-09, rtol=0)):
            raise ValueError(f'metric eligibility depth plane/affine mismatch: {frame_idx}')
        raster_path = Path(str(row.get('raster', {}).get('path', ''))).expanduser().resolve()
        raster = row.get('raster')
        if not isinstance(raster, dict) or raster.get('sha256') != sha256(raster_path) or raster.get('bytes') != raster_path.stat().st_size:
            raise ValueError(f'metric eligibility raster changed: {frame_idx}')
        with np.load(raster_path, allow_pickle=False) as blob:
            if set(blob.files) != {'schema', 'frame_idx', 'class_code', 'metric_candidate_eligible'}:
                raise ValueError('metric eligibility raster keys mismatch')
            if str(np.asarray(blob['schema']).reshape(-1)[0]) != RASTER_SCHEMA:
                raise ValueError('metric eligibility raster schema mismatch')
            if int(np.asarray(blob['frame_idx']).reshape(-1)[0]) != frame_idx:
                raise ValueError('metric eligibility raster frame mismatch')
            classes = np.asarray(blob['class_code'])
            eligible = np.asarray(blob['metric_candidate_eligible'])
        if classes.shape != depth_shape or classes.dtype != np.uint8 or eligible.shape != depth_shape or (eligible.dtype != np.bool_) or (not set(np.unique(classes)).issubset(set(CLASS_CODES.values()))):
            raise ValueError('metric eligibility raster type/shape mismatch')
        owned_mask_image = cv2.imread(str(owned_mask_path), cv2.IMREAD_GRAYSCALE)
        if owned_mask_image is None:
            raise ValueError(f'metric eligibility owned mask unreadable: {frame_idx}')
        (owned_depth, _) = conservative_mask_on_depth(owned_mask_image > 0, np.asarray(A_depth_from_mask, dtype=float), depth_shape)
        expected_eligible = np.isin(classes, [CLASS_CODES['hand_clearly_behind'], CLASS_CODES['clear_no_hand_overlap_candidate']]) & owned_depth
        if not np.array_equal(eligible, expected_eligible):
            raise ValueError('metric eligibility boolean disagrees with typed classes')
        if row.get('class_code_array_sha256') != array_sha256(classes) or row.get('metric_candidate_array_sha256') != array_sha256(eligible):
            raise ValueError('metric eligibility raster array hash mismatch')
        return (eligible, {'manifest': str(self.path), 'raster': str(raster_path), 'counts': row.get('counts'), 'class_codes': CLASS_CODES})
import argparse
import copy
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from envs.qualification.camera import SCHEMA as CAMERA_CONTRACT_V2_SCHEMA
from envs.qualification.camera import load_contract as load_camera_contract_v2
from envs.qualification.camera import plane_intrinsics as camera_contract_plane_intrinsics
from envs.qualification.camera import resize_affine as camera_contract_resize_affine
from envs.qualification.camera import sha256_file as camera_contract_sha256_file
from envs.qualification.camera import summarize_contract as summarize_camera_contract_v2

def safe_name(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', value).strip('_') or 'object'

def export_anchor_visible_surface_mesh(*, output_dir: Path, object_id: str, anchor_frame: int, anchor_centroid_world_m: np.ndarray, anchor_points_world_m: np.ndarray, min_voxel_m: float, voxel_divisor: float, poisson_depth: int, poisson_density_quantile: float) -> dict[str, Any]:
    import open3d as o3d
    "Export the selected anchor's visible surfels in object-canonical coordinates.\n\n    The downstream rigid completion/pose components assume the completed mesh is\n    in an object-canonical frame and per-frame pose rows map that canonical mesh\n    into world coordinates.  Therefore the anchor point cloud is centered at the\n    anchor-frame visible centroid rather than written in world coordinates.\n    "
    points_world = np.asarray(anchor_points_world_m, dtype=np.float64)
    if o3d is None:
        raise RuntimeError('Open3D is required to export anchor visible-surface artifacts')
    centroid = np.asarray(anchor_centroid_world_m, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) < 30:
        return {'status': 'too_few_anchor_visible_points_for_mesh_export', 'anchor_frame_idx': int(anchor_frame), 'point_count_input': int(len(points_world)) if points_world.ndim == 2 else 0, 'blockers': ['anchor visible surfels missing or below 30 points']}
    if centroid.shape != (3,) or not np.isfinite(centroid).all() or (not np.isfinite(points_world).all()):
        raise RuntimeError('anchor visible surface contains invalid coordinates')
    points_canonical = points_world - centroid[None, :]
    object_dir = output_dir / 'anchor_visible_surface_mesh' / safe_name(object_id.replace('object:', 'object_'))
    object_dir.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_canonical.astype(np.float64))
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(extent))
    voxel_size = max(float(min_voxel_m), diag / max(float(voxel_divisor), 1.0)) if diag > 0.0 else float(min_voxel_m)
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size)
    down_points = np.asarray(pcd.points, dtype=np.float64)
    out: dict[str, Any] = {'status': 'anchor_visible_surface_mesh_exported', 'coordinate_frame': 'object_canonical_anchor_centroid_frame', 'canonical_coordinate_source': 'selected_anchor_frame_visible_surfels_minus_anchor_centroid_world_m', 'anchor_frame_idx': int(anchor_frame), 'anchor_centroid_world_m': centroid.astype(float).tolist(), 'point_count_input': int(points_canonical.shape[0]), 'point_count_downsampled': int(down_points.shape[0]), 'voxel_size_m': float(voxel_size), 'canonical_bbox_min_m': down_points.min(axis=0).astype(float).tolist() if len(down_points) else None, 'canonical_bbox_max_m': down_points.max(axis=0).astype(float).tolist() if len(down_points) else None, 'fused_point_cloud_path': None, 'poisson_mesh_path': None, 'convex_hull_mesh_path': None, 'blockers': [], 'claim_scope': 'selected-frame metric visible surface for TRELLIS alignment; not complete hidden object geometry'}
    pcd_path = object_dir / f'frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_visible_points_canonical.ply'
    if not o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f'failed to write anchor visible point cloud: {pcd_path}')
    out['fused_point_cloud_path'] = str(pcd_path)
    if down_points.shape[0] < 30:
        out['status'] = 'too_few_downsampled_anchor_points_for_mesh'
        out['blockers'].append('too_few_downsampled_anchor_points_for_mesh')
        return out
    try:
        (hull, _) = pcd.compute_convex_hull()
        hull.compute_vertex_normals()
        hull_path = object_dir / f'frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_convex_hull_visible_candidate.ply'
        if o3d.io.write_triangle_mesh(str(hull_path), hull, write_ascii=False, compressed=False):
            out['convex_hull_mesh_path'] = str(hull_path)
            out['convex_hull_vertices'] = int(np.asarray(hull.vertices).shape[0])
            out['convex_hull_faces'] = int(np.asarray(hull.triangles).shape[0])
    except Exception as exc:
        out['blockers'].append(f'convex_hull_failed:{type(exc).__name__}:{exc}')
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 4.0, float(min_voxel_m) * 4.0), max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)
        (mesh, densities) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(poisson_depth))
        mesh = mesh.crop(bbox)
        densities_np = np.asarray(densities)
        if densities_np.size and np.asarray(mesh.vertices).shape[0] == densities_np.shape[0]:
            keep_threshold = float(np.quantile(densities_np, float(poisson_density_quantile)))
            mesh.remove_vertices_by_mask(densities_np < keep_threshold)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        poisson_path = object_dir / f'frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_poisson_visible_mesh.ply'
        if not o3d.io.write_triangle_mesh(str(poisson_path), mesh, write_ascii=False, compressed=False):
            raise RuntimeError('Open3D write_triangle_mesh returned false')
        out['poisson_mesh_path'] = str(poisson_path)
        out['poisson_vertices'] = int(np.asarray(mesh.vertices).shape[0])
        out['poisson_faces'] = int(np.asarray(mesh.triangles).shape[0])
    except Exception as exc:
        out['blockers'].append(f'poisson_reconstruction_failed:{type(exc).__name__}:{exc}')
        if not out.get('convex_hull_mesh_path'):
            out['status'] = 'anchor_visible_surface_point_cloud_only_mesh_failed'
        else:
            out['status'] = 'anchor_visible_surface_hull_only_poisson_failed'
    return out

def raw_frame_map(manifest_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = read_json(manifest_path)
    frames = payload.get('frames')
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f'{manifest_path} must contain a nonempty frames list')
    out: dict[int, dict[str, Any]] = {}
    for row_raw in frames:
        if not isinstance(row_raw, dict):
            continue
        idx = int(row_raw.get('frame_idx', row_raw.get('index', -1)))
        if idx < 0:
            raise RuntimeError(f'raw frame row lacks frame_idx/index: {row_raw}')
        if idx in out:
            raise RuntimeError(f'duplicate raw frame {idx} in {manifest_path}')
        out[idx] = row_raw
    return (out, payload)

def scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f'{label} must be a scalar string, got shape {array.shape}')
    return str(array.reshape(-1)[0])

def load_depth_npz(path: Path) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=True)
    required = {'frame_idx', 'depth', 'intrinsics_fx_fy_cx_cy'}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f'{path} missing depth keys: {missing}')
    frame_idx = np.asarray(blob['frame_idx'], dtype=int)
    depth = np.asarray(blob['depth'], dtype=np.float32)
    intr = np.asarray(blob['intrinsics_fx_fy_cx_cy'], dtype=np.float64)
    if depth.ndim != 3:
        raise RuntimeError(f'{path} depth must have shape [N,H,W], got {depth.shape}')
    if len(frame_idx) != depth.shape[0] or len(frame_idx) != intr.shape[0]:
        raise RuntimeError(f'{path} has inconsistent frame/depth/intrinsics rows')
    source_size = np.asarray(blob['source_size'], dtype=float) if 'source_size' in blob.files else None
    confidence = np.asarray(blob['confidence'], dtype=np.float32) if 'confidence' in blob.files else None
    if confidence is not None and confidence.shape != depth.shape:
        raise RuntimeError(f'{path} confidence shape {confidence.shape} disagrees with depth {depth.shape}')
    camera_contract_binding = None
    binding_keys = {'camera_contract_path', 'camera_contract_sha256', 'camera_contract_plane', 'A_depth_from_calibration', 'intrinsics_source', 'calibration_authority', 'depth_ray_geometry_reprojected', 'metadata_only_ray_relabel_override'}
    present_binding_keys = binding_keys.intersection(blob.files)
    if present_binding_keys and present_binding_keys != binding_keys:
        raise RuntimeError(f'{path} has an incomplete camera-contract binding; missing={sorted(binding_keys.difference(blob.files))}')
    if present_binding_keys:
        affine = np.asarray(blob['A_depth_from_calibration'], dtype=np.float64)
        if affine.shape != (3, 3) or not np.isfinite(affine).all():
            raise RuntimeError(f'{path} has invalid A_depth_from_calibration')
        camera_contract_binding = {'path': scalar_text(blob['camera_contract_path'], 'camera_contract_path'), 'sha256': scalar_text(blob['camera_contract_sha256'], 'camera_contract_sha256'), 'plane': scalar_text(blob['camera_contract_plane'], 'camera_contract_plane'), 'A_depth_from_calibration': affine.tolist(), 'intrinsics_source': scalar_text(blob['intrinsics_source'], 'intrinsics_source'), 'calibration_authority': scalar_text(blob['calibration_authority'], 'calibration_authority'), 'depth_ray_geometry_reprojected': bool(np.asarray(blob['depth_ray_geometry_reprojected']).reshape(-1)[0]), 'metadata_only_ray_relabel_override': bool(np.asarray(blob['metadata_only_ray_relabel_override']).reshape(-1)[0])}
    return {'path': str(path), 'frame_idx': frame_idx, 'depth': depth, 'confidence': confidence, 'intrinsics': intr, 'source_size': source_size, 'source_estimated_intrinsics': np.asarray(blob['source_estimated_intrinsics_fx_fy_cx_cy'], dtype=np.float64) if 'source_estimated_intrinsics_fx_fy_cx_cy' in blob.files else None, 'camera_contract_binding': camera_contract_binding, 'frame_to_i': {int(idx): int(i) for (i, idx) in enumerate(frame_idx)}}

def load_calibration_contract(path: Path | None, *, frame_ids: list[int] | None=None) -> tuple[np.ndarray | None, str | None, dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return (None, None, None, None)
    if not path.exists():
        raise FileNotFoundError(f'missing calibration contract: {path}')
    payload = read_json(path)
    v2_declared = payload.get('schema') == CAMERA_CONTRACT_V2_SCHEMA
    v2_fields_present = 'calibration_plane' in payload or 'image_planes' in payload
    if v2_declared or v2_fields_present:
        (full_payload, normalized) = load_camera_contract_v2(path, expected_frame_ids=frame_ids)
        source = str(full_payload.get('intrinsics_source') or full_payload.get('method') or 'v19_camera_contract')
        summary = summarize_camera_contract_v2(path, full_payload, normalized)
        return (None, f'camera_contract:{source}', summary, {'payload': full_payload, 'normalized': normalized})
    intr = np.asarray(payload.get('intrinsics_fx_fy_cx_cy'), dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or float(intr[0]) <= 0.0 or (float(intr[1]) <= 0.0):
        raise RuntimeError(f'calibration contract has invalid intrinsics_fx_fy_cx_cy: {path}')
    source = str(payload.get('intrinsics_source') or payload.get('method') or 'v19_calibration_contract')
    source = f'legacy_calibration_contract:{source}'
    summary = {'path': str(path), 'schema': payload.get('schema'), 'calibration_authority': payload.get('calibration_authority') or 'legacy_unspecified', 'intrinsics_fx_fy_cx_cy': [float(v) for v in intr.tolist()], 'intrinsics_source': source, 'fov_degrees': payload.get('fov_degrees'), 'aggregation': payload.get('aggregation'), 'legacy_contract_without_explicit_image_planes': True}
    return (intr.astype(float), source, summary, None)

def load_camera_npz(path: Path | None) -> dict[int, tuple[np.ndarray, str]]:
    if path is None:
        return {}
    blob = np.load(path, allow_pickle=True)
    if 'frame_idx' not in blob.files:
        raise RuntimeError(f'camera npz {path} lacks frame_idx')
    frame_idx = np.asarray(blob['frame_idx'], dtype=int)
    poses: dict[int, tuple[np.ndarray, str]] = {}
    if 'T_world_camera' in blob.files:
        mats = np.asarray(blob['T_world_camera'], dtype=float)
        source = 'camera_npz_T_world_camera'
    elif 'T_world_camera_metric_current_v18' in blob.files:
        mats = np.asarray(blob['T_world_camera_metric_current_v18'], dtype=float)
        source = 'camera_npz_T_world_camera_metric_current_v18'
    elif 'R_c2w' in blob.files and 't_c2w' in blob.files:
        r = np.asarray(blob['R_c2w'], dtype=float)
        t = np.asarray(blob['t_c2w'], dtype=float)
        if len(r) != len(frame_idx) or len(t) != len(frame_idx):
            raise RuntimeError(f'camera npz {path} has inconsistent R_c2w/t_c2w rows')
        for (i, idx) in enumerate(frame_idx):
            T = np.eye(4, dtype=float)
            T[:3, :3] = r[i]
            T[:3, 3] = t[i]
            if T.shape != (4, 4) or not np.isfinite(T).all():
                raise RuntimeError(f'camera npz {path} invalid pose for frame {idx}')
            poses[int(idx)] = (T, 'camera_npz_R_c2w_t_c2w')
        return poses
    else:
        raise RuntimeError(f'camera npz {path} lacks a supported camera pose key; expected T_world_camera, T_world_camera_metric_current_v18, or R_c2w/t_c2w')
    if mats.shape[0] != len(frame_idx) or mats.shape[1:] != (4, 4):
        raise RuntimeError(f'camera npz {path} pose shape mismatch: {mats.shape}')
    for (i, idx) in enumerate(frame_idx):
        T = np.asarray(mats[i], dtype=float)
        if not np.isfinite(T).all():
            raise RuntimeError(f'camera npz {path} invalid pose for frame {idx}')
        poses[int(idx)] = (T, source)
    return poses

def load_base_annotations(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = read_json(path)
    frames = payload.get('frames')
    if not isinstance(frames, list):
        raise RuntimeError(f'{path} must contain frames list')
    out: dict[int, dict[str, Any]] = {}
    for row in frames:
        if not isinstance(row, dict):
            continue
        idx = int(row.get('frame_idx', -1))
        if idx < 0:
            raise RuntimeError(f'base annotation row lacks frame_idx: {row}')
        if idx in out:
            raise RuntimeError(f'duplicate base annotation frame {idx}')
        out[idx] = row
    return out

def load_object_plan_record(path: Path | None, track_id: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = read_json(path)
    plan = payload.get('plan') if isinstance(payload, dict) else None
    if not isinstance(plan, dict):
        plan = payload if isinstance(payload, dict) else {}
    for row in list_or_empty(plan.get('objects')):
        if isinstance(row, dict) and str(row.get('track_id')) == str(track_id):
            return row
    return None

def load_sam2_track(args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], Path]:
    if args.sam2_track_json is not None:
        path = args.sam2_track_json
    else:
        path = args.sam2_root / args.track_id / 'sam2' / 'sam2_track.json'
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f'SAM2 track must be an object: {path}')
    out: dict[int, dict[str, Any]] = {}
    for (key, value) in payload.items():
        try:
            idx = int(key)
        except ValueError as exc:
            raise RuntimeError(f'SAM2 track key is not a frame index: {key}') from exc
        if not isinstance(value, dict):
            raise RuntimeError(f'SAM2 track row for frame {idx} is not an object')
        out[idx] = value
    return (out, path)

def frame_range(raw_frames: dict[int, dict[str, Any]], start: int | None, end: int | None) -> list[int]:
    indices = sorted(raw_frames)
    lo = indices[0] if start is None else int(start)
    hi = indices[-1] if end is None else int(end)
    if hi < lo:
        raise RuntimeError(f'invalid frame range {lo}:{hi}')
    selected = [idx for idx in indices if lo <= idx <= hi]
    if not selected:
        raise RuntimeError(f'no raw frames in range {lo}:{hi}')
    return selected

def base_camera_pose(frame: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
    for key in ('T_world_camera_metric', 'T_world_camera', 'T_world_camera_metric_current_v18'):
        value = camera.get(key)
        arr = np.asarray(value if value is not None else [], dtype=float)
        if arr.shape == (4, 4) and np.isfinite(arr).all():
            return (arr, f'base_annotations_camera_{key}')
    return None

def base_camera_intrinsics(frame: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
    value = camera.get('intrinsics_fx_fy_cx_cy')
    arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    if arr.shape == (4,) and np.isfinite(arr).all() and (float(arr[0]) > 0.0) and (float(arr[1]) > 0.0):
        source = str(camera.get('intrinsics_source') or 'base_annotations_camera_intrinsics_fx_fy_cx_cy')
        return (arr.astype(float), f'base_annotations:{source}')
    return None

def resolve_camera_pose(frame_idx: int, frame: dict[str, Any], camera_poses: dict[int, tuple[np.ndarray, str]], allow_camera_frame_world: bool) -> tuple[np.ndarray, str]:
    base = base_camera_pose(frame)
    if base is not None:
        return base
    if frame_idx in camera_poses:
        return camera_poses[frame_idx]
    if allow_camera_frame_world:
        return (np.eye(4, dtype=float), 'explicit_camera_frame_world_identity_not_metric_temporal_world')
    raise RuntimeError(f'frame {frame_idx} lacks camera/world pose. Provide --base-annotations with camera fields, --camera-npz, or explicit --allow-camera-frame-world.')

def localize_path(path: str | Path, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for (src, dst) in ((remote_root, local_root), (local_root, remote_root)):
            try:
                rel = direct.relative_to(src)
            except ValueError:
                continue
            candidate = dst / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(str(path))

def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f'failed to read mask {path}')
    return mask > 0

def bbox_xyxy_from_mask(mask: np.ndarray, source_width: int, source_height: int) -> list[float]:
    (ys, xs) = np.where(mask)
    if len(xs) == 0:
        return []
    sx = float(source_width) / float(mask.shape[1])
    sy = float(source_height) / float(mask.shape[0])
    return [float(xs.min() * sx), float(ys.min() * sy), float((xs.max() + 1) * sx), float((ys.max() + 1) * sy)]

def scaled_intrinsics_for_depth(intr: np.ndarray, depth_shape: tuple[int, int], source_size: np.ndarray | None, *, explicit_source_size_wh: tuple[int, int] | None=None) -> np.ndarray:
    (fx, fy, cx, cy) = np.asarray(intr, dtype=float).tolist()
    (h, w) = depth_shape
    if explicit_source_size_wh is not None:
        (source_w, source_h) = [float(v) for v in explicit_source_size_wh]
    elif source_size is not None and source_size.size >= 2:
        source_w = float(source_size[0])
        source_h = float(source_size[1])
    else:
        source_w = source_h = 0.0
    if source_w > 0 and source_h > 0 and (abs(source_w - w) > 1e-06 or abs(source_h - h) > 1e-06):
        sx = float(w) / source_w
        sy = float(h) / source_h
        return np.asarray([fx * sx, fy * sy, cx * sx, cy * sy], dtype=float)
    return np.asarray([fx, fy, cx, cy], dtype=float)

def load_npz_array_cached(path: Path, key: str) -> np.ndarray:
    cache = getattr(load_npz_array_cached, '_cache', None)
    if cache is None:
        cache = {}
        setattr(load_npz_array_cached, '_cache', cache)
    cache_key = (str(path.expanduser().resolve()), str(key))
    if cache_key not in cache:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f'missing hand geometry archive: {path}')
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise RuntimeError(f'hand geometry archive {path} lacks {key}')
            cache[cache_key] = np.asarray(archive[key]).copy()
    return cache[cache_key]

def projected_mano_hand_silhouette(hand: dict[str, Any], *, mask_shape: tuple[int, int], source_width: int, source_height: int, pad_px: int, A_mask_from_source: np.ndarray | None=None) -> tuple[np.ndarray, dict[str, Any]]:
    state = hand.get('metric_mano_state')
    if not isinstance(state, dict):
        raise RuntimeError('hand row lacks metric_mano_state')
    reference = state.get('vertices_reference')
    if not isinstance(reference, dict):
        raise RuntimeError('hand metric state lacks vertices_reference')
    bridge_path = Path(str(reference.get('bridge_npz') or ''))
    bridge_key = str(reference.get('bridge_vertices_camera_array') or '')
    row_index = int(reference.get('bridge_row_index', -1))
    vertices_rows = load_npz_array_cached(bridge_path, bridge_key)
    if vertices_rows.ndim != 3 or vertices_rows.shape[1:] != (778, 3) or (not 0 <= row_index < len(vertices_rows)):
        raise RuntimeError(f'invalid bridge MANO camera vertices {vertices_rows.shape} row={row_index}')
    vertices = np.asarray(vertices_rows[row_index], dtype=np.float64)
    side = str(hand.get('hand_side') or '').lower()
    if side not in {'left', 'right'}:
        raise RuntimeError(f'invalid hand side {side!r}')
    source_hawor = Path(str(reference.get('source_hawor_npz') or ''))
    faces = np.asarray(load_npz_array_cached(source_hawor, f'{side}_faces'), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or (faces.min() < 0) or (faces.max() >= len(vertices)):
        raise RuntimeError(f'invalid {side} MANO faces {faces.shape}')
    intrinsics = np.asarray(state.get('current_v18_camera_intrinsics_fx_fy_cx_cy') or state.get('source_hawor_camera_intrinsics_fx_fy_cx_cy') or [], dtype=np.float64).reshape(-1)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
        raise RuntimeError(f'invalid {side} MANO camera intrinsics: {intrinsics}')
    z = vertices[:, 2]
    positive_faces = np.all(np.isfinite(vertices[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    faces = faces[positive_faces]
    if len(faces) == 0:
        raise RuntimeError(f'{side} MANO mesh has no finite positive-depth faces')
    used_vertices = np.unique(faces.reshape(-1))
    (fx, fy, cx, cy) = intrinsics.tolist()
    uv = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    uv[used_vertices, 0] = fx * vertices[used_vertices, 0] / z[used_vertices] + cx
    uv[used_vertices, 1] = fy * vertices[used_vertices, 1] / z[used_vertices] + cy
    (mask_h, mask_w) = mask_shape
    if A_mask_from_source is None:
        A_mask_from_source = camera_contract_resize_affine((int(source_width), int(source_height)), (int(mask_w), int(mask_h)), pixel_center_convention='integer_pixel_centers_opencv')
    A_mask_from_source = np.asarray(A_mask_from_source, dtype=np.float64)
    if A_mask_from_source.shape != (3, 3) or not np.isfinite(A_mask_from_source).all() or (not np.allclose(A_mask_from_source[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0)) or (abs(float(np.linalg.det(A_mask_from_source))) < 1e-12):
        raise RuntimeError(f'invalid source-to-mask projection affine: {A_mask_from_source}')
    projected_source = uv[used_vertices]
    homogeneous = np.column_stack([projected_source, np.ones((len(projected_source),), dtype=np.float64)])
    projected_mask = homogeneous @ A_mask_from_source.T
    uv[used_vertices] = projected_mask[:, :2] / projected_mask[:, 2:3]
    if not np.isfinite(uv[used_vertices]).all():
        raise RuntimeError(f'{side} MANO projection has non-finite pixels')
    polygons = np.rint(uv[faces]).astype(np.int32)
    silhouette = np.zeros((mask_h, mask_w), dtype=np.uint8)
    cv2.fillPoly(silhouette, list(polygons), 1, lineType=cv2.LINE_8)
    if int(np.count_nonzero(silhouette)) == 0:
        raise RuntimeError(f'{side} MANO projected silhouette is empty')
    pad = int(max(0, pad_px))
    if pad:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        silhouette = cv2.dilate(silhouette, kernel, iterations=1)
    (ys, xs) = np.where(silhouette > 0)
    projected_bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    return (silhouette > 0, {'hand_side': side, 'bridge_npz': str(bridge_path.expanduser().resolve()), 'bridge_vertices_camera_array': bridge_key, 'bridge_row_index': row_index, 'source_hawor_npz': str(source_hawor.expanduser().resolve()), 'faces_array': f'{side}_faces', 'vertices': int(len(vertices)), 'faces': int(len(faces)), 'source_camera_intrinsics_fx_fy_cx_cy': intrinsics.tolist(), 'A_mask_from_source_coordinate_model': A_mask_from_source.tolist(), 'projection_policy': 'project full MANO with source K then apply exact OpenCV source-to-mask affine', 'projected_silhouette_bbox_mask_xyxy': projected_bbox, 'projected_silhouette_pixels_with_padding': int(np.count_nonzero(silhouette)), 'padding_px_in_mask_coordinates': pad, 'source_bbox_xyxy_diagnostic_only': hand.get('bbox_xyxy')})

def subtract_hand_owned_bbox_regions(mask: np.ndarray, base_frame: dict[str, Any], *, source_width: int, source_height: int, pad_px: int, enabled: bool, A_mask_from_source: np.ndarray | None=None) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract projected HaWoR/MANO hand silhouettes, never coarse bboxes.

    Rectangular hand boxes erase visible object pixels and still fail to describe
    fingers at object boundaries. Full prediction-side MANO vertices/faces provide
    a tighter ownership silhouette. Missing or malformed geometry fails the frame
    closed rather than falling back to a box or treating hand pixels as object.
    """
    if not enabled:
        return (mask, {'state': 'disabled', 'input_mask_pixels': int(mask.sum()), 'output_mask_pixels': int(mask.sum()), 'fail_closed': False})
    out = mask.copy()
    input_pixels = int(mask.sum())
    hand_rows = [hand for hand in list_or_empty(base_frame.get('hands')) if isinstance(hand, dict) and (not (hand.get('same_frame_detection') is False and (not hand.get('hawor_candidate_present'))))]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for hand in hand_rows:
        try:
            (silhouette, record) = projected_mano_hand_silhouette(hand, mask_shape=out.shape[:2], source_width=source_width, source_height=source_height, pad_px=pad_px, A_mask_from_source=A_mask_from_source)
            before = int(out.sum())
            out[silhouette] = False
            record['removed_object_mask_pixels'] = before - int(out.sum())
            records.append(record)
        except Exception as exc:
            failures.append({'hand_side': hand.get('hand_side'), 'reason': str(exc)})
    fail_closed = bool(failures)
    if fail_closed:
        out[:] = False
    output_pixels = int(out.sum())
    return (out, {'state': 'failed_closed_missing_projected_mano_hand_silhouette' if fail_closed else 'projected_mano_hand_silhouettes_subtracted' if records else 'no_detected_hands_to_subtract', 'ownership_primitive': 'projected_full_mano_triangle_silhouette', 'bbox_subtraction_used': False, 'input_mask_pixels': input_pixels, 'output_mask_pixels': output_pixels, 'removed_mask_pixels': input_pixels - output_pixels, 'detected_hand_rows': int(len(hand_rows)), 'projected_hand_silhouettes': records, 'failures': failures, 'fail_closed': fail_closed, 'padding_px_in_mask_coordinates': int(max(0, pad_px)), 'claim_scope': 'Only projected prediction-side MANO triangle silhouettes are removed. Hand bboxes are diagnostic and never ownership masks.'})

def robust_first_surface_depth_ownership(mask: np.ndarray, depth: np.ndarray, *, enabled: bool, mad_sigma: float, min_half_width_m: float, min_retained_fraction: float, fail_raw_to_robust_extent_ratio: float, intrinsics: np.ndarray | None=None, confidence: np.ndarray | None=None, confidence_seed_percentile: float=95.0, local_depth_step_max_m: float=0.005, max_removed_distance_inside_mask_px: float=10.0, max_confidence_flagged_interior_fraction: float=0.015, min_interior_confidence_flagged_fraction: float=0.95, max_unexplained_interior_pixels: int=5, min_component_pixels: int=20, max_small_component_fraction: float=0.01) -> tuple[np.ndarray, dict[str, Any]]:
    """Select depth-coherent first-surface support, with strict quarantine gates.

    A global percentile/MAD crop can delete real sloped surfaces or disconnected
    rigid parts. Instead, each 2D object-owned component gets a robust depth seed;
    support grows from that seed only through locally continuous depth and through
    pixels whose UniDepth error proxy is no worse than the seed's configured
    percentile. This preserves coherent surfaces while stopping edge bleeding.

    Removed pixels are accepted as quarantined (rather than making the frame fail)
    only when support remains high. Rejection beyond the ordinary boundary band is
    allowed solely for a tiny number of isolated samples, or for a sparse region
    whose UniDepth predicted-error proxy is overwhelmingly worse than the robust
    seed. This explicitly covers texture/glare-induced interior depth holes without
    treating a coherent second surface as background. The original P11 appearance
    mask is never rewritten.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    depth_m = np.asarray(depth, dtype=np.float64)
    valid = mask_bool & np.isfinite(depth_m) & (depth_m > 0.0)
    values = depth_m[valid]
    if values.size == 0:
        return (valid, {'enabled': bool(enabled), 'state': 'no_valid_owned_depth', 'input_valid_depth_pixels': 0, 'retained_depth_pixels': 0, 'retained_fraction': 0.0, 'fail_closed': True, 'failure_reasons': ['no_valid_owned_depth']})
    global_median = float(np.median(values))
    global_mad = float(np.median(np.abs(values - global_median)))
    if not enabled:
        return (valid, {'enabled': False, 'state': 'disabled', 'input_valid_depth_pixels': int(values.size), 'retained_depth_pixels': int(values.size), 'retained_fraction': 1.0, 'median_depth_m': global_median, 'median_absolute_deviation_m': global_mad, 'fail_closed': False, 'failure_reasons': []})
    confidence_m = None if confidence is None else np.asarray(confidence, dtype=np.float64)
    failure_reasons: list[str] = []
    if confidence_m is None or confidence_m.shape != depth_m.shape:
        failure_reasons.append('missing_or_mismatched_unidepth_confidence')
        confidence_m = np.full(depth_m.shape, np.inf, dtype=np.float64)
    elif not np.isfinite(confidence_m[valid]).all() or np.any(confidence_m[valid] <= 0.0):
        failure_reasons.append('invalid_unidepth_confidence')
    accepted = np.zeros_like(valid)
    confidence_threshold_map = np.full(depth_m.shape, np.nan, dtype=np.float64)
    component_rows: list[dict[str, Any]] = []
    small_component_pixels = 0
    (component_count, labels, stats, _) = cv2.connectedComponentsWithStats(valid.astype(np.uint8), 8)
    (height, width) = valid.shape
    neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for label in range(1, component_count):
        (x, y, w, h, area) = [int(value) for value in stats[label]]
        component = labels[y:y + h, x:x + w] == label
        if area < int(min_component_pixels):
            small_component_pixels += area
            component_rows.append({'label': label, 'pixels': area, 'state': 'dropped_small_component'})
            continue
        local_depth = depth_m[y:y + h, x:x + w]
        local_confidence = confidence_m[y:y + h, x:x + w]
        component_values = local_depth[component]
        median = float(np.median(component_values))
        mad = float(np.median(np.abs(component_values - median)))
        robust_sigma = float(1.4826 * mad)
        half_width = float(max(min_half_width_m, mad_sigma * robust_sigma))
        seed = component & (np.abs(local_depth - median) <= half_width)
        seed_count = int(np.count_nonzero(seed))
        if seed_count < 3 or not np.isfinite(local_confidence[seed]).all():
            failure_reasons.append(f'component_{label}_invalid_seed')
            component_rows.append({'label': label, 'pixels': area, 'state': 'invalid_seed'})
            continue
        confidence_threshold = float(np.percentile(local_confidence[seed], confidence_seed_percentile))
        threshold_view = confidence_threshold_map[y:y + h, x:x + w]
        threshold_view[component] = confidence_threshold
        traversable = component & (local_confidence <= confidence_threshold)
        keep = seed.copy()
        iterations = 0
        for iterations in range(max(1, w + h)):
            previous_count = int(np.count_nonzero(keep))
            adjacent = np.zeros_like(keep)
            for (dy, dx) in neighbor_offsets:
                source_y = slice(max(0, dy), min(h, h + dy))
                target_y = slice(max(0, -dy), min(h, h - dy))
                source_x = slice(max(0, dx), min(w, w + dx))
                target_x = slice(max(0, -dx), min(w, w - dx))
                adjacent[target_y, target_x] |= keep[source_y, source_x] & (np.abs(local_depth[target_y, target_x] - local_depth[source_y, source_x]) <= float(local_depth_step_max_m))
            keep |= traversable & adjacent
            if int(np.count_nonzero(keep)) == previous_count:
                break
        accepted[y:y + h, x:x + w] |= keep
        keep_count = int(np.count_nonzero(keep))
        component_rows.append({'label': label, 'bbox_xywh': [x, y, w, h], 'pixels': area, 'median_depth_m': median, 'median_absolute_deviation_m': mad, 'robust_sigma_m': robust_sigma, 'seed_half_width_m': half_width, 'seed_pixels': seed_count, 'seed_fraction': float(seed_count / max(1, area)), 'confidence_seed_percentile': float(confidence_seed_percentile), 'confidence_threshold': confidence_threshold, 'retained_pixels': keep_count, 'retained_fraction': float(keep_count / max(1, area)), 'growth_iterations': int(iterations), 'accepted_depth_summary_m': numeric_summary(local_depth[keep]), 'state': 'depth_confidence_geodesic_support'})
    retained = int(np.count_nonzero(accepted))
    retained_fraction = float(retained / max(1, values.size))
    removed = valid & ~accepted
    removed_count = int(np.count_nonzero(removed))
    distance_inside = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    removed_distances = distance_inside[removed]
    max_removed_distance = float(np.max(removed_distances)) if removed_distances.size else 0.0
    interior_removed = removed & (distance_inside > float(max_removed_distance_inside_mask_px))
    interior_removed_count = int(np.count_nonzero(interior_removed))
    interior_removed_fraction = float(interior_removed_count / max(1, values.size))
    interior_confidence = confidence_m[interior_removed]
    interior_confidence_threshold = confidence_threshold_map[interior_removed]
    confidence_flagged_interior = np.isfinite(interior_confidence) & np.isfinite(interior_confidence_threshold) & (interior_confidence > interior_confidence_threshold)
    confidence_flagged_interior_count = int(np.count_nonzero(confidence_flagged_interior))
    confidence_flagged_interior_fraction = float(confidence_flagged_interior_count / max(1, interior_removed_count))
    interior_quarantine_mode = 'none'
    interior_quarantine_validated = interior_removed_count == 0
    if 0 < interior_removed_count <= int(max_unexplained_interior_pixels):
        interior_quarantine_validated = True
        interior_quarantine_mode = 'isolated_raster_samples'
    elif interior_removed_count > 0:
        interior_quarantine_validated = bool(interior_removed_fraction <= float(max_confidence_flagged_interior_fraction) and confidence_flagged_interior_fraction >= float(min_interior_confidence_flagged_fraction))
        interior_quarantine_mode = 'sparse_unidepth_predicted_error_flagged_interior_holes' if interior_quarantine_validated else 'unresolved_interior_rejection'
    small_component_fraction = float(small_component_pixels / max(1, values.size))
    raw_extent = backprojected_extent(valid, depth_m, intrinsics)
    accepted_extent = backprojected_extent(accepted, depth_m, intrinsics)
    raw_diag = float(np.linalg.norm(raw_extent))
    accepted_diag = float(np.linalg.norm(accepted_extent))
    raw_to_accepted_ratio = float(raw_diag / max(accepted_diag, 1e-12))
    if retained < 3 or retained_fraction < float(min_retained_fraction):
        failure_reasons.append('insufficient_depth_coherent_first_surface_support')
    if not np.isfinite(accepted_diag) or accepted_diag <= 0.0:
        failure_reasons.append('invalid_accepted_backprojected_extent')
    if small_component_fraction > float(max_small_component_fraction):
        failure_reasons.append('too_much_support_in_dropped_small_components')
    if not interior_quarantine_validated:
        failure_reasons.append('rejected_interior_depth_not_sparse_and_confidence_flagged')
    tail_dominates_raw_extent = bool(raw_to_accepted_ratio > float(fail_raw_to_robust_extent_ratio))
    return (accepted, {'enabled': True, 'state': 'fail_closed_depth_ownership_unresolved' if failure_reasons else 'validated_sparse_interior_and_boundary_depth_quarantined' if interior_removed_count else 'validated_boundary_depth_tail_quarantined' if removed_count else 'all_owned_depth_coherent', 'method': 'per_component_mad_seed_confidence_geodesic_growth', 'input_valid_depth_pixels': int(values.size), 'retained_depth_pixels': retained, 'removed_depth_pixels': removed_count, 'retained_fraction': retained_fraction, 'removed_fraction': float(removed_count / max(1, values.size)), 'median_depth_m': global_median, 'median_absolute_deviation_m': global_mad, 'mad_sigma': float(mad_sigma), 'minimum_half_width_m': float(min_half_width_m), 'confidence_semantics': 'UniDepth exp(logconfidence); larger values predict larger depth error', 'confidence_seed_percentile': float(confidence_seed_percentile), 'local_depth_step_max_m': float(local_depth_step_max_m), 'max_removed_distance_inside_mask_px': float(max_removed_distance_inside_mask_px), 'max_confidence_flagged_interior_fraction': float(max_confidence_flagged_interior_fraction), 'min_interior_confidence_flagged_fraction': float(min_interior_confidence_flagged_fraction), 'max_unexplained_interior_pixels': int(max_unexplained_interior_pixels), 'component_count': int(component_count - 1), 'component_rows': component_rows, 'small_component_pixels': int(small_component_pixels), 'small_component_fraction': small_component_fraction, 'raw_backprojected_extent_m': raw_extent.astype(float).tolist(), 'raw_backprojected_extent_diag_m': raw_diag, 'robust_backprojected_extent_m': accepted_extent.astype(float).tolist(), 'robust_backprojected_extent_diag_m': accepted_diag, 'raw_to_robust_extent_diag_ratio': raw_to_accepted_ratio, 'fail_raw_to_robust_extent_ratio': float(fail_raw_to_robust_extent_ratio), 'raw_rejected_depth_dominates_extent': tail_dominates_raw_extent, 'raw_tail_dominates_extent_but_is_quarantined': bool(tail_dominates_raw_extent and (not failure_reasons)), 'removed_pixel_distance_inside_owned_mask_px': numeric_summary(removed_distances), 'interior_rejection_quarantine': {'boundary_band_distance_px': float(max_removed_distance_inside_mask_px), 'interior_removed_pixels': interior_removed_count, 'interior_removed_fraction_of_valid_owned_depth': interior_removed_fraction, 'interior_removed_depth_summary_m': numeric_summary(depth_m[interior_removed]), 'interior_removed_confidence_summary': numeric_summary(interior_confidence), 'confidence_flagged_interior_pixels': confidence_flagged_interior_count, 'confidence_flagged_interior_fraction': confidence_flagged_interior_fraction, 'max_confidence_flagged_interior_fraction': float(max_confidence_flagged_interior_fraction), 'min_interior_confidence_flagged_fraction': float(min_interior_confidence_flagged_fraction), 'max_unexplained_interior_pixels': int(max_unexplained_interior_pixels), 'validated': bool(interior_quarantine_validated), 'mode': interior_quarantine_mode, 'semantics': 'Pixels beyond the ordinary boundary band remain excluded only when isolated or when a sparse region is overwhelmingly marked by UniDepth as higher predicted error than its component seed. This is explicit unresolved-depth quarantine, not object-mask erosion.'}, 'fail_closed': bool(failure_reasons), 'failure_reasons': failure_reasons, 'claim_scope': 'Depth first-surface surfel ownership only. Excluded owned depth remains unresolved; the P11 RGB/mask appearance contract is not eroded or rewritten.'})

def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    return {'count': int(array.size), 'min': float(np.min(array)), 'median': float(np.median(array)), 'p95': float(np.percentile(array, 95.0)), 'max': float(np.max(array))}

def backprojected_extent(mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray | None=None) -> np.ndarray:
    (ys, xs) = np.where(mask)
    if len(xs) == 0:
        return np.zeros(3, dtype=np.float64)
    z = depth[ys, xs].astype(np.float64)
    if intrinsics is None:
        (h, w) = depth.shape
        fx = fy = max(1.0, float(w))
        (cx, cy) = (0.5 * (w - 1), 0.5 * (h - 1))
    else:
        intr = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
        if intr.shape != (4,) or not np.isfinite(intr).all() or intr[0] <= 0.0 or (intr[1] <= 0.0):
            raise RuntimeError(f'invalid intrinsics for depth ownership: {intr}')
        (fx, fy, cx, cy) = intr.tolist()
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    return points.max(axis=0) - points.min(axis=0)

def choose_visible_points(valid: np.ndarray, depth: np.ndarray, intr: np.ndarray, T_world_camera: np.ndarray, *, pixel_stride: int, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stride = max(1, int(pixel_stride))
    sampled = np.zeros_like(valid, dtype=bool)
    sampled[::stride, ::stride] = valid[::stride, ::stride]
    (ys, xs) = np.where(sampled)
    if len(xs) == 0:
        (ys, xs) = np.where(valid)
    if len(xs) > int(max_points):
        order = rng.choice(len(xs), size=int(max_points), replace=False)
        ys = ys[order]
        xs = xs[order]
    z = depth[ys, xs].astype(float)
    (fx, fy, cx, cy) = intr.astype(float).tolist()
    X = (xs.astype(float) - cx) * z / fx
    Y = (ys.astype(float) - cy) * z / fy
    cam = np.column_stack([X, Y, z])
    hom = np.column_stack([cam, np.ones(len(cam), dtype=float)])
    world = (hom @ T_world_camera.T)[:, :3]
    keep = np.isfinite(world).all(axis=1) & np.isfinite(cam).all(axis=1)
    cam = cam[keep]
    world = world[keep]
    summary = {'valid_depth_mask_pixels': int(valid.sum()), 'sampled_points_before_finite_filter': int(len(z)), 'sampled_points': int(len(world)), 'pixel_stride': int(stride), 'max_points': int(max_points)}
    return (cam.astype(float), world.astype(float), summary)

def normalize01(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value) or not np.isfinite(lo) or (not np.isfinite(hi)) or (hi <= lo):
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

def raw_image_path_from_row(raw_row: dict[str, Any]) -> str:
    return str(raw_row.get('raw_frame_path') or raw_row.get('rgb') or raw_row.get('path') or '')

def candidate_mask_summary(mask: np.ndarray, source_width: int, source_height: int) -> dict[str, Any]:
    bbox = bbox_xyxy_from_mask(mask, source_width, source_height)
    area = int(mask.sum())
    (h, w) = mask.shape[:2]
    component_count = 0
    largest_component_px = 0
    if area > 0:
        (n, labels) = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        component_count = max(0, int(n) - 1)
        if n > 1:
            counts = np.bincount(labels.reshape(-1))[1:]
            largest_component_px = int(counts.max()) if counts.size else 0
    if len(bbox) >= 4:
        (x0, y0, x1, y1) = [float(v) for v in bbox[:4]]
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        margin = 0.03 * float(min(max(1, source_width), max(1, source_height)))
        touches_border = bool(x0 <= margin or y0 <= margin or x1 >= float(source_width) - margin or (y1 >= float(source_height) - margin))
        bbox_area = float(bw * bh)
    else:
        x0 = y0 = x1 = y1 = bw = bh = bbox_area = 0.0
        touches_border = True
    return {'mask_area_px': area, 'component_count': int(component_count), 'largest_component_px': int(largest_component_px), 'bbox_xyxy': bbox, 'bbox_width_px': float(bw), 'bbox_height_px': float(bh), 'bbox_area_px': float(bbox_area), 'touches_or_near_image_border': bool(touches_border)}

def anchor_conditioning_coherence(row: dict[str, Any], *, maximum_mask_area_px: float, maximum_visible_depth_points: float, minimum_population_fraction: float=0.2, minimum_mask_area_px: int=1000, minimum_visible_depth_points: int=100) -> dict[str, Any]:
    """Identify coherent single-image completion anchors without choosing one.

    A MANO-subtracted mask may legitimately become disconnected and remains
    valid appearance/metric evidence.  For a single-image completion model,
    however, a supported one-component alternative is less ambiguous than two
    object fragments separated by a hand.  This category-agnostic signal only
    prioritizes the review list; visual inspection still owns the decision.
    """
    area_threshold = max(int(minimum_mask_area_px), int(np.ceil(float(minimum_population_fraction) * max(maximum_mask_area_px, 0.0))))
    point_threshold = max(int(minimum_visible_depth_points), int(np.ceil(float(minimum_population_fraction) * max(maximum_visible_depth_points, 0.0))))
    component_count = int(row.get('component_count') or 0)
    mask_area = int(row.get('mask_area_px') or 0)
    visible_points = int(row.get('visible_depth_vertex_count') or 0)
    one_component = component_count == 1
    support_sufficient = bool(mask_area >= area_threshold and visible_points >= point_threshold)
    non_border = not bool(row.get('touches_or_near_image_border'))
    return {'preferred': bool(one_component and support_sufficient and non_border), 'one_owned_component': one_component, 'support_sufficient': support_sufficient, 'non_border': non_border, 'mask_area_px': mask_area, 'minimum_mask_area_px': area_threshold, 'visible_depth_points': visible_points, 'minimum_visible_depth_points': point_threshold, 'minimum_population_fraction': float(minimum_population_fraction), 'semantics': 'Review prioritization for native single-image completion conditioning only; not metric-surface, pose, mask, or anchor acceptance.'}

def build_anchor_candidate_proposals(*, args: argparse.Namespace, object_id: str, visible_data: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Write anchor candidate evidence without choosing the anchor.

    The score is a proposal heuristic only.  It intentionally exposes the raw
    factors and review sheet so the runtime agent can choose, reject, or request
    another proposal after subjective visual/geometric inspection.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for (idx, vis) in sorted(visible_data.items()):
        mask = read_mask(Path(vis['mask_path']))
        source_width = int(vis.get('source_width') or mask.shape[1])
        source_height = int(vis.get('source_height') or mask.shape[0])
        mask_stats = candidate_mask_summary(mask, source_width, source_height)
        world_points = np.asarray(vis.get('world_points'), dtype=float)
        extent = world_points.max(axis=0) - world_points.min(axis=0) if world_points.ndim == 2 and len(world_points) else np.zeros(3, dtype=float)
        diag = float(np.linalg.norm(extent))
        ownership = vis.get('object_surface_ownership_filter') if isinstance(vis.get('object_surface_ownership_filter'), dict) else {}
        input_mask_px = int(ownership.get('input_mask_pixels') or mask_stats['mask_area_px'] or 0)
        output_mask_px = int(ownership.get('output_mask_pixels') or mask_stats['mask_area_px'] or 0)
        removed_px = int(ownership.get('removed_mask_pixels') or max(0, input_mask_px - output_mask_px))
        removed_frac = float(removed_px / input_mask_px) if input_mask_px > 0 else 0.0
        depth_p05 = float(vis.get('depth_p05_m') or np.nan)
        depth_p95 = float(vis.get('depth_p95_m') or np.nan)
        depth_spread = float(depth_p95 - depth_p05) if np.isfinite(depth_p05) and np.isfinite(depth_p95) else float('nan')
        sample_summary = vis.get('sample_summary') if isinstance(vis.get('sample_summary'), dict) else {}
        rows.append({'frame_idx': int(idx), 'raw_frame_path': raw_image_path_from_row(vis.get('raw_row') if isinstance(vis.get('raw_row'), dict) else {}), 'object_owned_mask_path': str(vis.get('mask_path')), 'raw_sam2_mask_path': str(vis.get('raw_sam2_mask_path')), 'visible_depth_vertex_count': int(len(world_points)), 'valid_depth_mask_pixels': int(sample_summary.get('valid_depth_mask_pixels') or 0), 'sampled_points': int(sample_summary.get('sampled_points') or len(world_points)), 'mask_area_px': int(mask_stats['mask_area_px']), 'raw_mask_area_px': int(input_mask_px), 'hand_owned_removed_px': int(removed_px), 'hand_owned_removed_fraction': float(removed_frac), 'component_count': int(mask_stats['component_count']), 'largest_component_px': int(mask_stats['largest_component_px']), 'bbox_xyxy': mask_stats['bbox_xyxy'], 'bbox_width_px': float(mask_stats['bbox_width_px']), 'bbox_height_px': float(mask_stats['bbox_height_px']), 'bbox_area_px': float(mask_stats['bbox_area_px']), 'touches_or_near_image_border': bool(mask_stats['touches_or_near_image_border']), 'depth_median_m': float(vis.get('depth_median_m') or np.nan), 'depth_p05_m': depth_p05, 'depth_p95_m': depth_p95, 'depth_spread_p95_p05_m': depth_spread, 'world_extent_m': extent.astype(float).tolist(), 'world_extent_diag_m': float(diag), 'intrinsics_source': str(vis.get('intrinsics_source')), 'camera_source': str(vis.get('camera_source')), 'native_depth_metric_eligibility': vis.get('native_depth_metric_eligibility')})
    if not rows:
        raise RuntimeError('no visible rows available for anchor candidate proposal')
    area_values = np.asarray([float(r['mask_area_px']) for r in rows], dtype=float)
    point_values = np.asarray([float(r['visible_depth_vertex_count']) for r in rows], dtype=float)
    diag_values = np.asarray([float(r['world_extent_diag_m']) for r in rows if float(r['world_extent_diag_m']) > 0.0], dtype=float)
    depth_spreads = np.asarray([float(r['depth_spread_p95_p05_m']) for r in rows if np.isfinite(float(r['depth_spread_p95_p05_m']))], dtype=float)
    median_diag = float(np.median(diag_values)) if diag_values.size else 0.0
    depth_spread_hi = float(np.percentile(depth_spreads, 90.0)) if depth_spreads.size else 1.0
    area_hi = float(area_values.max()) if area_values.size else 1.0
    point_hi = float(point_values.max()) if point_values.size else 1.0
    for r in rows:
        coherence = anchor_conditioning_coherence(r, maximum_mask_area_px=area_hi, maximum_visible_depth_points=point_hi)
        r['conditioning_coherence'] = coherence
        r['conditioning_coherence_preferred'] = bool(coherence['preferred'])
        area_score = float(r['mask_area_px']) / max(area_hi, 1.0)
        point_score = float(r['visible_depth_vertex_count']) / max(point_hi, 1.0)
        hand_clean_score = 1.0 - float(np.clip(r['hand_owned_removed_fraction'], 0.0, 1.0))
        border_score = 0.0 if bool(r['touches_or_near_image_border']) else 1.0
        comp = max(1, int(r['component_count']))
        component_score = float(max(0.0, 1.0 - 0.25 * (comp - 1)))
        diag = float(r['world_extent_diag_m'])
        extent_score = float(np.exp(-abs(np.log(max(diag, 1e-09) / max(median_diag, 1e-09))))) if median_diag > 0 else 0.0
        spread = float(r['depth_spread_p95_p05_m'])
        depth_score = 1.0 - normalize01(spread, 0.0, max(depth_spread_hi, 1e-06)) if np.isfinite(spread) else 0.0
        proposal_score = 0.24 * area_score + 0.16 * point_score + 0.18 * hand_clean_score + 0.16 * border_score + 0.14 * extent_score + 0.07 * component_score + 0.05 * depth_score
        r['proposal_score'] = float(proposal_score)
        r['proposal_score_terms'] = {'mask_area_score': float(area_score), 'visible_depth_vertex_count_score': float(point_score), 'hand_clean_score': float(hand_clean_score), 'not_near_image_border_score': float(border_score), 'extent_consistency_score': float(extent_score), 'single_component_score': float(component_score), 'depth_stability_score': float(depth_score)}
        r['score_interpretation'] = 'heuristic proposal score only; agent visual/geometric judgment must choose the anchor'
    preferred_count = int(sum((bool(row['conditioning_coherence_preferred']) for row in rows)))
    ranked = sorted(rows, key=lambda r: (0 if bool(r['conditioning_coherence_preferred']) else 1, -float(r['proposal_score']), int(r['frame_idx'])))
    top_k = max(1, int(args.anchor_candidate_count))
    min_gap = max(0, int(args.anchor_candidate_min_gap))
    review_rows: list[dict[str, Any]] = []
    for row in ranked:
        if all((abs(int(row['frame_idx']) - int(prev['frame_idx'])) >= min_gap for prev in review_rows)):
            review_rows.append(row)
        if len(review_rows) >= top_k:
            break
    if len(review_rows) < top_k:
        seen = {int(r['frame_idx']) for r in review_rows}
        for row in ranked:
            if int(row['frame_idx']) not in seen:
                review_rows.append(row)
                seen.add(int(row['frame_idx']))
            if len(review_rows) >= top_k:
                break
    report_path = args.output_dir / 'anchor_candidate_proposals.json'
    review_path = args.output_dir / 'anchor_candidate_review.jpg'
    review_status = render_anchor_candidate_review(review_rows=review_rows, review_path=review_path, panel_width=int(args.anchor_candidate_panel_width))
    report = {'method': 'build_v19_visible_geometry_from_sam2_depth_anchor_candidate_proposals', 'status': 'ok', 'object_id': object_id, 'track_id': args.track_id, 'claim_scope': 'Anchor candidates for agent subjective selection. This report does not choose or accept an anchor frame.', 'selection_required': True, 'selection_instruction': 'Inspect anchor_candidate_review.jpg and ranked_candidates; write an agent anchor decision before running P09/P11 with --anchor-frame/--selected-frame-idx. When supported single-component conditioning_coherence_preferred candidates exist, choose among them unless image inspection finds wrong ownership or inadequate target identity; disconnected rows remain valid evidence but are ambiguous single-image completion anchors.', 'proposal_score_policy': 'Supported non-border one-component owned masks are reviewed first, then a weighted heuristic over owned mask area, visible depth support, low hand-owned removal, non-border support, metric extent consistency, component count, and depth stability; neither ordering nor score is an acceptance gate.', 'conditioning_coherence_policy': {'preferred_candidate_count': preferred_count, 'review_order': 'supported_single_owned_component_first_then_proposal_score', 'minimum_population_fraction': 0.2, 'minimum_mask_area_px_floor': 1000, 'minimum_visible_depth_points_floor': 100, 'claim_scope': 'single-image completion conditioning review priority only'}, 'candidate_count': int(len(rows)), 'review_candidate_count': int(len(review_rows)), 'review_diversity_min_frame_gap': int(min_gap), 'outputs': {'anchor_candidate_proposals': str(report_path), 'anchor_candidate_review': str(review_path) if review_status.get('status') == 'ok' else None}, 'review_status': review_status, 'ranked_candidates': ranked, 'review_candidates': review_rows, 'score_population_summary': {'max_mask_area_px': int(area_hi), 'max_visible_depth_vertex_count': int(point_hi), 'median_world_extent_diag_m': float(median_diag), 'depth_spread_p90_m': float(depth_spread_hi)}}
    write_json_payload(report_path, report)
    return report

def render_anchor_candidate_review(*, review_rows: list[dict[str, Any]], review_path: Path, panel_width: int) -> dict[str, Any]:
    panels: list[np.ndarray] = []
    blockers: list[str] = []
    panel_width = max(220, int(panel_width))
    for (rank, row) in enumerate(review_rows, start=1):
        raw_path = Path(str(row.get('raw_frame_path') or ''))
        mask_path = Path(str(row.get('object_owned_mask_path') or ''))
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR) if raw_path.is_file() else None
        if raw is None:
            blockers.append(f"missing_raw_frame:{row.get('frame_idx')}:{raw_path}")
            raw = np.full((480, 480, 3), 240, dtype=np.uint8)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if mask is None:
            blockers.append(f"missing_object_owned_mask:{row.get('frame_idx')}:{mask_path}")
            mask_bool_img = np.zeros(raw.shape[:2], dtype=bool)
        else:
            if mask.shape[:2] != raw.shape[:2]:
                mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool_img = mask > 0
        overlay = raw.copy()
        tint = overlay.copy()
        tint[mask_bool_img] = (40, 220, 70)
        overlay[mask_bool_img] = cv2.addWeighted(tint, 0.45, overlay, 0.55, 0)[mask_bool_img]
        (cnts, _) = cv2.findContours(mask_bool_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)
        scale = panel_width / float(max(1, overlay.shape[1]))
        ph = int(round(overlay.shape[0] * scale))
        panel_img = cv2.resize(overlay, (panel_width, ph), interpolation=cv2.INTER_AREA)
        label_h = 112
        panel = np.full((ph + label_h, panel_width, 3), 255, dtype=np.uint8)
        panel[:ph] = panel_img

        def put(line: str, y: int, color: tuple[int, int, int]=(0, 0, 0)) -> None:
            cv2.putText(panel, line[:72], (6, ph + y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
        preferred = bool(row.get('conditioning_coherence_preferred'))
        put(f"rank {rank} frame {row.get('frame_idx')} score {float(row.get('proposal_score', 0.0)):.3f} coherent {preferred}", 16, (20, 120, 20) if preferred else (0, 0, 0))
        put(f"mask {int(row.get('mask_area_px', 0))}px pts {int(row.get('visible_depth_vertex_count', 0))} border {bool(row.get('touches_or_near_image_border'))}", 34)
        put(f"hand_removed {100.0 * float(row.get('hand_owned_removed_fraction', 0.0)):.1f}% comps {int(row.get('component_count', 0))}", 52)
        ext = row.get('world_extent_m') if isinstance(row.get('world_extent_m'), list) else []
        if len(ext) >= 3:
            put(f'extent m {float(ext[0]):.3f},{float(ext[1]):.3f},{float(ext[2]):.3f}', 70)
        put(f"depth p05-p95 {float(row.get('depth_spread_p95_p05_m', 0.0)):.3f}m", 88)
        put('agent must inspect; score is not acceptance', 106, (40, 40, 180))
        panels.append(panel)
    if not panels:
        return {'status': 'no_review_candidates', 'blockers': blockers}
    cols = min(3, len(panels))
    rows_img: list[np.ndarray] = []
    for start in range(0, len(panels), cols):
        chunk = panels[start:start + cols]
        max_h = max((p.shape[0] for p in chunk))
        padded = []
        for p in chunk:
            if p.shape[0] < max_h:
                pad = np.full((max_h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
        while len(padded) < cols:
            padded.append(np.full((max_h, panel_width, 3), 255, dtype=np.uint8))
        rows_img.append(np.hstack(padded))
    sheet = np.vstack(rows_img)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(review_path), sheet)
    return {'status': 'ok' if ok else 'write_failed', 'path': str(review_path), 'blockers': blockers[:20]}

def object_row_not_visible(object_id: str, track_id: str, status: str, reason: str, mask_path: str | None=None) -> dict[str, Any]:
    row: dict[str, Any] = {'object_id': object_id, 'track_id': track_id, 'label': track_id, 'status': status, 'visible': False, 'v19_visible_geometry_adapter': {'state': status, 'reason': reason}}
    if mask_path:
        row['mask_path'] = mask_path
    return row

def remove_existing_object(objects: list[Any], object_id: str, track_id: str) -> list[Any]:
    out = []
    for obj in objects:
        if not isinstance(obj, dict):
            out.append(obj)
            continue
        if obj.get('object_id') == object_id or obj.get('track_id') == track_id:
            continue
        out.append(obj)
    return out

def build_visible_geometry(args: argparse.Namespace) -> dict[str, Any]:
    from envs.qualification.camera import validate_depth_camera_contract_binding
    (raw_frames, raw_payload) = raw_frame_map(args.raw_frame_manifest)
    depth = load_depth_npz(args.depth_npz)
    (calibration_intrinsics, calibration_source, calibration_summary, camera_contract_v2) = load_calibration_contract(args.calibration_contract, frame_ids=sorted(raw_frames))
    depth_contract_binding_validation = validate_depth_camera_contract_binding(depth=depth, camera_contract_v2=camera_contract_v2, calibration_contract_path=args.calibration_contract, depth_image_plane=args.depth_image_plane, allow_implicit_depth_resize=bool(args.allow_implicit_depth_resize))
    camera_poses = load_camera_npz(args.camera_npz)
    base_frames = load_base_annotations(args.base_annotations)
    (sam2, sam2_path) = load_sam2_track(args)
    object_plan_record = load_object_plan_record(args.object_plan, args.track_id)
    object_id = args.object_id or f'object:{args.track_id}'
    metric_eligibility_path = getattr(args, 'metric_eligibility_manifest', None)
    metric_eligibility = MetricEligibility(metric_eligibility_path, case=args.case, object_id=object_id, track_id=args.track_id, sam2_track=sam2_path, depth_npz=args.depth_npz, base_annotations=args.base_annotations, calibration_contract=args.calibration_contract, frame_to_depth_i=depth['frame_to_i'], depth=depth['depth']) if metric_eligibility_path is not None else None
    indices = frame_range(raw_frames, args.frame_start, args.frame_end)
    output_indices = sorted(raw_frames) if bool(args.preserve_source_index) else list(indices)
    rng = np.random.default_rng(int(args.seed))
    visible_data: dict[int, dict[str, Any]] = {}
    appearance_data: dict[int, dict[str, Any]] = {}
    skipped_rows: list[dict[str, Any]] = []
    for idx in indices:
        track_row = sam2.get(idx, {})
        if not track_row.get('visible') or not track_row.get('mask_path'):
            skipped_rows.append({'frame_idx': idx, 'status': 'not_visible_in_sam2'})
            continue
        depth_i = depth['frame_to_i'].get(idx)
        if depth_i is None:
            skipped_rows.append({'frame_idx': idx, 'status': 'missing_depth_row'})
            continue
        source_mask_path = str(track_row['mask_path'])
        mask_path = localize_path(source_mask_path, args.remote_root, args.local_root)
        mask = read_mask(mask_path)
        depth_m = np.asarray(depth['depth'][depth_i], dtype=float)
        raw_row = raw_frames[idx]
        base_frame = copy.deepcopy(base_frames.get(idx, {'frame_idx': idx}))
        source_width = int(raw_row.get('source_width') or raw_payload.get('video', {}).get('width') or raw_row.get('manifest_width') or mask.shape[1])
        source_height = int(raw_row.get('source_height') or raw_payload.get('video', {}).get('height') or raw_row.get('manifest_height') or mask.shape[0])
        A_mask_from_source_for_ownership = camera_contract_resize_affine((int(source_width), int(source_height)), (int(mask.shape[1]), int(mask.shape[0])), pixel_center_convention=args.pixel_center_convention)
        if camera_contract_v2 is not None:
            (_, ownership_source_plane) = camera_contract_plane_intrinsics(camera_contract_v2['payload'], camera_contract_v2['normalized'], plane_name='source_rgb', actual_size_wh=(int(source_width), int(source_height)), allow_implicit_resize=False)
            (_, ownership_mask_plane) = camera_contract_plane_intrinsics(camera_contract_v2['payload'], camera_contract_v2['normalized'], plane_name=args.mask_image_plane, actual_size_wh=(int(mask.shape[1]), int(mask.shape[0])), allow_implicit_resize=bool(args.allow_implicit_mask_resize))
            A_source_from_calibration = np.asarray(ownership_source_plane['A_actual_plane_from_calibration'], dtype=np.float64)
            A_mask_from_calibration = np.asarray(ownership_mask_plane['A_actual_plane_from_calibration'], dtype=np.float64)
            expected_A_mask_from_source = A_mask_from_calibration @ np.linalg.inv(A_source_from_calibration)
            if not np.allclose(A_mask_from_source_for_ownership, expected_A_mask_from_source, atol=1e-09, rtol=0.0):
                raise RuntimeError(f'P09 MANO ownership projection affine disagrees with the camera contract: resize={A_mask_from_source_for_ownership.tolist()} contract={expected_A_mask_from_source.tolist()}')
        (mask_owned, ownership_summary) = subtract_hand_owned_bbox_regions(mask, base_frame, source_width=source_width, source_height=source_height, pad_px=int(args.hand_bbox_exclusion_pad_px), enabled=bool(args.exclude_hand_bboxes), A_mask_from_source=A_mask_from_source_for_ownership)
        owned_mask_path = args.output_dir / 'object_owned_masks' / f'{idx:06d}_{safe_name(object_id)}_object_owned_mask.png'
        owned_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(owned_mask_path), mask_owned.astype(np.uint8) * 255):
            raise RuntimeError(f'failed to write object-owned mask: {owned_mask_path}')
        if ownership_summary.get('fail_closed') is True:
            skipped_rows.append({'frame_idx': idx, 'status': 'projected_mano_hand_ownership_failed_closed', 'object_surface_ownership_filter': ownership_summary, 'owned_mask_path': str(owned_mask_path)})
            continue
        appearance_data[idx] = {'mask_path': str(owned_mask_path), 'source_mask_path': source_mask_path, 'object_surface_ownership_filter': ownership_summary, 'ownership_contract': {'mano_subtraction_completed': True, 'ownership_primitive': ownership_summary.get('ownership_primitive'), 'bbox_subtraction_used': False, 'fail_closed': False}, 'source_width': int(source_width), 'source_height': int(source_height), 'claim_scope': 'Prediction-side RGB/object appearance support after projected MANO silhouette subtraction. This support remains valid when metric depth ownership fails and does not promote rejected depth.'}
        if mask_owned.shape != depth_m.shape:
            A_depth_from_mask = camera_contract_resize_affine((int(mask_owned.shape[1]), int(mask_owned.shape[0])), (int(depth_m.shape[1]), int(depth_m.shape[0])), pixel_center_convention=args.pixel_center_convention)
            mask_depth_owned = cv2.resize(mask_owned.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT) > 0
        else:
            A_depth_from_mask = np.eye(3, dtype=np.float64)
            mask_depth_owned = mask_owned
        mask_depth_transform_contract = {'mask_image_plane': args.mask_image_plane, 'depth_image_plane': args.depth_image_plane, 'mask_size_wh': [int(mask_owned.shape[1]), int(mask_owned.shape[0])], 'depth_size_wh': [int(depth_m.shape[1]), int(depth_m.shape[0])], 'A_depth_from_mask_coordinate_model': A_depth_from_mask.tolist(), 'raster_resampling': 'cv2.resize INTER_NEAREST_EXACT; discrete samples follow the declared OpenCV half-pixel coordinate model', 'interpolation': 'cv2.INTER_NEAREST_EXACT', 'pixel_center_convention': args.pixel_center_convention}
        intrinsics_transform_contract: dict[str, Any]
        if camera_contract_v2 is not None:
            (raw_intrinsics, intrinsics_transform_contract) = camera_contract_plane_intrinsics(camera_contract_v2['payload'], camera_contract_v2['normalized'], plane_name=args.depth_image_plane, actual_size_wh=(int(depth_m.shape[1]), int(depth_m.shape[0])), allow_implicit_resize=bool(args.allow_implicit_depth_resize))
            intr = np.asarray(raw_intrinsics, dtype=float)
            intrinsics_source = calibration_source or 'camera_contract_v2'
            (_, mask_plane_transform_contract) = camera_contract_plane_intrinsics(camera_contract_v2['payload'], camera_contract_v2['normalized'], plane_name=args.mask_image_plane, actual_size_wh=(int(mask_owned.shape[1]), int(mask_owned.shape[0])), allow_implicit_resize=bool(args.allow_implicit_mask_resize))
            A_depth_from_calibration = np.asarray(intrinsics_transform_contract['A_actual_plane_from_calibration'], dtype=np.float64)
            A_mask_from_calibration = np.asarray(mask_plane_transform_contract['A_actual_plane_from_calibration'], dtype=np.float64)
            expected_A_depth_from_mask = A_depth_from_calibration @ np.linalg.inv(A_mask_from_calibration)
            if not np.allclose(A_depth_from_mask, expected_A_depth_from_mask, atol=1e-09, rtol=0.0):
                raise RuntimeError(f'declared camera-contract mask/depth planes disagree with the actual cv2 mask resize: actual={A_depth_from_mask.tolist()} expected={expected_A_depth_from_mask.tolist()}')
            mask_depth_transform_contract['camera_contract_consistent'] = True
            mask_depth_transform_contract['mask_plane_transform'] = mask_plane_transform_contract
        elif calibration_intrinsics is not None:
            raw_intrinsics = calibration_intrinsics
            intrinsics_source = calibration_source or 'legacy_calibration_contract'
            intr = scaled_intrinsics_for_depth(raw_intrinsics, depth_m.shape, depth['source_size'])
            intrinsics_transform_contract = {'mode': 'legacy_source_size_scaling', 'depth_image_plane': args.depth_image_plane, 'legacy_contract_without_explicit_image_planes': True}
        else:
            base_intr = base_camera_intrinsics(base_frame)
            if base_intr is not None:
                (raw_intrinsics, intrinsics_source) = base_intr
            else:
                raw_intrinsics = depth['intrinsics'][depth_i]
                intrinsics_source = 'depth_npz_intrinsics_fx_fy_cx_cy'
            intr = scaled_intrinsics_for_depth(raw_intrinsics, depth_m.shape, depth['source_size'])
            intrinsics_transform_contract = {'mode': 'legacy_no_camera_contract_fallback', 'depth_image_plane': args.depth_image_plane, 'source_size': depth['source_size'].astype(float).tolist() if depth['source_size'] is not None else None}
        valid = mask_depth_owned & np.isfinite(depth_m) & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
        metric_eligibility_summary = None
        if metric_eligibility is not None:
            (eligible, metric_eligibility_summary) = metric_eligibility.frame(idx, owned_mask_path=mask_path, depth_shape=depth_m.shape, A_depth_from_mask=A_depth_from_mask, K_depth=np.asarray([[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]], dtype=float))
            valid &= eligible
        confidence_m = np.asarray(depth['confidence'][depth_i], dtype=float) if depth.get('confidence') is not None else None
        (robust_valid, depth_ownership_summary) = robust_first_surface_depth_ownership(valid, depth_m, enabled=bool(args.robust_first_surface_depth_ownership), mad_sigma=float(args.first_surface_mad_sigma), min_half_width_m=float(args.first_surface_min_half_width_m), min_retained_fraction=float(args.first_surface_min_retained_fraction), fail_raw_to_robust_extent_ratio=float(args.first_surface_fail_raw_to_robust_extent_ratio), intrinsics=intr, confidence=confidence_m, confidence_seed_percentile=float(args.first_surface_confidence_seed_percentile), local_depth_step_max_m=float(args.first_surface_local_depth_step_max_m), max_removed_distance_inside_mask_px=float(args.first_surface_max_removed_distance_inside_mask_px), max_confidence_flagged_interior_fraction=float(args.first_surface_max_confidence_flagged_interior_fraction), min_interior_confidence_flagged_fraction=float(args.first_surface_min_interior_confidence_flagged_fraction), max_unexplained_interior_pixels=int(args.first_surface_max_unexplained_interior_pixels), min_component_pixels=int(args.first_surface_min_component_pixels), max_small_component_fraction=float(args.first_surface_max_small_component_fraction))
        if depth_ownership_summary.get('fail_closed') is True:
            skipped_rows.append({'frame_idx': idx, 'status': 'first_surface_depth_ownership_failed_closed', 'depth_ownership': depth_ownership_summary, 'owned_mask_path': str(owned_mask_path), 'appearance_support_preserved': True})
            continue
        valid = robust_valid
        if int(valid.sum()) < int(args.min_valid_points):
            skipped_rows.append({'frame_idx': idx, 'status': 'too_few_valid_mask_depth_pixels', 'valid_pixels': int(valid.sum())})
            continue
        (T_world_camera, camera_source) = resolve_camera_pose(idx, base_frame, camera_poses, bool(args.allow_camera_frame_world))
        (camera_points, world_points, sample_summary) = choose_visible_points(valid, depth_m, intr, T_world_camera, pixel_stride=int(args.pixel_stride), max_points=int(args.max_points), rng=rng)
        if len(world_points) < int(args.min_valid_points):
            skipped_rows.append({'frame_idx': idx, 'status': 'too_few_sampled_visible_points', 'sampled_points': int(len(world_points))})
            continue
        visible_data[idx] = {'mask_path': str(owned_mask_path), 'source_mask_path': source_mask_path, 'raw_sam2_mask_path': str(mask_path), 'track_row': track_row, 'raw_row': raw_row, 'camera_points': camera_points, 'world_points': world_points, 'intrinsics': intr, 'intrinsics_source': intrinsics_source, 'intrinsics_transform_contract': intrinsics_transform_contract, 'mask_depth_transform_contract': mask_depth_transform_contract, 'T_world_camera': T_world_camera, 'camera_source': camera_source, 'sample_summary': sample_summary, 'depth_median_m': float(np.median(depth_m[valid])), 'depth_p05_m': float(np.percentile(depth_m[valid], 5.0)), 'depth_p95_m': float(np.percentile(depth_m[valid], 95.0)), 'mask_shape': list(mask_owned.shape), 'raw_sam2_mask_shape': list(mask.shape), 'depth_shape': list(depth_m.shape), 'object_surface_ownership_filter': ownership_summary, 'first_surface_depth_ownership': depth_ownership_summary, 'native_depth_metric_eligibility': metric_eligibility_summary, 'source_width': int(source_width), 'source_height': int(source_height)}
    if not visible_data:
        raise RuntimeError(f'no visible metric geometry rows for track_id={args.track_id}. Check SAM2 masks, depth archive coverage, frame range, and camera-pose availability.')
    anchor_candidate_report = build_anchor_candidate_proposals(args=args, object_id=object_id, visible_data=visible_data)
    if bool(args.propose_anchor_candidates_only):
        return {'method': 'build_v19_visible_geometry_from_sam2_depth', 'status': 'anchor_candidates_proposed_only', 'case': args.case, 'track_id': args.track_id, 'object_id': object_id, 'claim_scope': 'Candidate anchor evidence only. No canonical anchor mesh, visible-geometry annotations, completion, pose, or render state was produced.', 'outputs': anchor_candidate_report.get('outputs', {}), 'candidate_count': anchor_candidate_report.get('candidate_count'), 'review_candidate_count': anchor_candidate_report.get('review_candidate_count')}
    if args.anchor_frame is not None:
        anchor = int(args.anchor_frame)
        if anchor not in visible_data:
            raise RuntimeError(f'--anchor-frame {anchor} has no visible metric geometry')
    else:
        if bool(args.require_anchor_frame):
            outputs = anchor_candidate_report.get('outputs', {}) if isinstance(anchor_candidate_report, dict) else {}
            raise RuntimeError(f"--anchor-frame is required for this run. Inspect anchor candidates and rerun with an agent-selected anchor. candidate_report={outputs.get('anchor_candidate_proposals')} review={outputs.get('anchor_candidate_review')}")
        anchor = max(visible_data, key=lambda idx: len(visible_data[idx]['world_points']))
    anchor_points = np.asarray(visible_data[anchor]['world_points'], dtype=float)
    anchor_centroid = anchor_points.mean(axis=0)
    anchor_extent_m = anchor_points.max(axis=0) - anchor_points.min(axis=0)
    anchor_diag_m = float(np.linalg.norm(anchor_extent_m))
    if anchor_diag_m <= 0.0 or not np.isfinite(anchor_diag_m):
        raise RuntimeError(f'anchor frame {anchor} has invalid metric extent')
    population_extents = np.asarray([np.ptp(np.asarray(row['world_points'], dtype=float), axis=0) for row in visible_data.values()], dtype=float)
    population_sorted_extents = np.sort(population_extents, axis=1)[:, ::-1]
    population_sorted_extent_median = np.median(population_sorted_extents, axis=0)
    population_diag = np.linalg.norm(population_extents, axis=1)
    population_diag_median = float(np.median(population_diag))
    if population_sorted_extent_median.shape != (3,) or not np.isfinite(population_sorted_extent_median).all() or np.any(population_sorted_extent_median <= 0.0) or (not np.isfinite(population_diag_median)) or (population_diag_median <= 0.0):
        raise RuntimeError('visible metric extent population is invalid')
    anchor_mesh_reconstruction = export_anchor_visible_surface_mesh(output_dir=args.output_dir, object_id=object_id, anchor_frame=int(anchor), anchor_centroid_world_m=anchor_centroid, anchor_points_world_m=anchor_points, min_voxel_m=float(args.anchor_mesh_min_voxel_m), voxel_divisor=float(args.anchor_mesh_voxel_divisor), poisson_depth=int(args.anchor_mesh_poisson_depth), poisson_density_quantile=float(args.anchor_mesh_poisson_density_quantile))
    last_pose: dict[str, Any] | None = None
    output_frames: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    camera_source_counts: Counter[str] = Counter()
    for idx in output_indices:
        raw_row = raw_frames[idx]
        frame = copy.deepcopy(base_frames.get(idx, {'frame_idx': idx}))
        frame['frame_idx'] = int(idx)
        raw_text = str(frame.get('raw_frame_path') or raw_row.get('rgb') or '')
        frame['raw_frame_path'] = str(localize_path(raw_text, args.remote_root, args.local_root)) if raw_text else raw_text
        frame['source_width'] = int(raw_row.get('source_width') or raw_payload.get('video', {}).get('width') or raw_row.get('manifest_width') or 0)
        frame['source_height'] = int(raw_row.get('source_height') or raw_payload.get('video', {}).get('height') or raw_row.get('manifest_height') or 0)
        try:
            (T_world_camera, camera_source) = resolve_camera_pose(idx, frame, camera_poses, bool(args.allow_camera_frame_world))
            camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
            camera = dict(camera)
            camera['T_world_camera_metric'] = T_world_camera.astype(float).tolist()
            camera['position_world_m'] = T_world_camera[:3, 3].astype(float).tolist()
            camera['v19_camera_pose_source'] = camera_source
            frame['camera'] = camera
            camera_source_counts[camera_source] += 1
        except RuntimeError:
            if idx in visible_data:
                raise
        objects = remove_existing_object(list_or_empty(frame.get('objects')), object_id, args.track_id)
        vis = visible_data.get(idx)
        if vis is None:
            track_row = sam2.get(idx, {})
            appearance = appearance_data.get(idx)
            appearance_mask = str(appearance.get('mask_path')) if isinstance(appearance, dict) and appearance.get('mask_path') else str(track_row.get('mask_path')) if track_row.get('mask_path') else None
            row = object_row_not_visible(object_id, args.track_id, 'not_visible_or_no_metric_depth', 'SAM2/depth did not provide an accepted visible metric surface for this frame', appearance_mask)
            if isinstance(appearance, dict):
                row['appearance_observation'] = {**appearance, 'status': 'object_owned_rgb_appearance_support_without_accepted_metric_depth', 'metric_depth_pose_eligible': False, 'rgb_tracking_pose_evidence_eligible': True}
                row['object_surface_ownership_filter'] = appearance.get('object_surface_ownership_filter')
                row['mask_path'] = appearance_mask
                row['mask_semantics'] = 'projected_mano_subtracted_object_owned_appearance_mask'
            if bool(args.carry_invisible_pose) and last_pose is not None:
                row['reconstructed_geometry_pose'] = {**last_pose, 'pose_source': 'carried_nearest_visible_centroid_uncertain_v19_adapter', 'pose_uncertainty': 'no current visible metric surface; pose is a carried prior for rendering/optimization initialization only'}
            objects.append(row)
            frame['objects'] = objects
            output_frames.append(frame)
            rows.append({'frame_idx': idx, 'status': 'not_visible_or_no_metric_depth'})
            continue
        world_points = np.asarray(vis['world_points'], dtype=float)
        cam_points = np.asarray(vis['camera_points'], dtype=float)
        centroid = world_points.mean(axis=0)
        world_extent_m = world_points.max(axis=0) - world_points.min(axis=0)
        sorted_extent_m = np.sort(world_extent_m)[::-1]
        extent_ratio_diag = float(max(np.linalg.norm(world_extent_m) / max(population_diag_median, 1e-09), population_diag_median / max(np.linalg.norm(world_extent_m), 1e-09)))
        extent_ratio_axis = np.maximum(sorted_extent_m / np.maximum(population_sorted_extent_median, 1e-06), population_sorted_extent_median / np.maximum(sorted_extent_m, 1e-06))
        max_extent_ratio_axis = float(np.max(extent_ratio_axis))
        rigid_pose_observation_eligible = bool(np.isfinite(extent_ratio_diag) and np.isfinite(max_extent_ratio_axis) and (extent_ratio_diag <= float(args.rigid_extent_ratio_max)) and (max_extent_ratio_axis <= float(args.rigid_extent_axis_ratio_max)))
        rigid_pose_observation_reason = 'metric_extent_consistent_with_orientation_invariant_visible_population_reference' if rigid_pose_observation_eligible else 'systematic_mask_extent_inconsistent_with_orientation_invariant_visible_population_reference_probable_hand_background_leakage'
        raw_mask = read_mask(Path(vis['mask_path']))
        owned_bbox = bbox_xyxy_from_mask(raw_mask, int(frame['source_width']), int(frame['source_height']))
        owned_area_source_px = float(int(raw_mask.sum()) * float(frame['source_width']) / float(max(1, raw_mask.shape[1])) * float(frame['source_height']) / float(max(1, raw_mask.shape[0])))
        if not owned_bbox:
            raise RuntimeError(f'frame {idx} object-owned mask unexpectedly has no support')
        pose = {'rotation_world_from_canonical_matrix': np.eye(3, dtype=float).tolist(), 'translation_world_m': centroid.astype(float).tolist(), 'pose_source': 'v19_visible_geometry_adapter_centroid_initial_pose_not_final_rigid_pose', 'pose_uncertainty': 'centroid-only initialization from current visible SAM2/depth surfels; rigid completion and pose fitting must refine it', 'anchor_frame_idx': int(anchor), 'anchor_centroid_world_m': anchor_centroid.astype(float).tolist()}
        last_pose = pose
        geom = {'status': 'visible_surface_from_sam2_mask_metric_depth', 'source': 'build_v19_visible_geometry_from_sam2_depth', 'frame_idx': int(idx), 'object_id': object_id, 'track_id': args.track_id, 'mask_path': str(vis['mask_path']), 'source_mask_path': str(vis.get('source_mask_path')), 'source_width': int(vis.get('source_width') or frame.get('source_width') or 0), 'source_height': int(vis.get('source_height') or frame.get('source_height') or 0), 'depth_npz': str(args.depth_npz), 'depth_frame_index': int(idx), 'camera_pose_source': vis['camera_source'], 'intrinsics_fx_fy_cx_cy': np.asarray(vis['intrinsics'], dtype=float).tolist(), 'intrinsics_source': vis.get('intrinsics_source'), 'intrinsics_transform_contract': vis.get('intrinsics_transform_contract'), 'mask_depth_transform_contract': vis.get('mask_depth_transform_contract'), 'vertex_count': int(len(world_points)), 'world_vertices_sample_m': world_points.astype(float).tolist(), 'camera_vertices_sample_m': cam_points.astype(float).tolist(), 'centroid_world_m': centroid.astype(float).tolist(), 'world_extent_m': world_extent_m.astype(float).tolist(), 'anchor_extent_world_m': anchor_extent_m.astype(float).tolist(), 'extent_population_sorted_axis_median_m': population_sorted_extent_median.astype(float).tolist(), 'extent_population_diag_median_m': population_diag_median, 'extent_consistency_basis': 'frame-population median of orientation-invariant sorted extents and extent diagonal', 'extent_ratio_to_population_diag': extent_ratio_diag, 'extent_ratio_to_population_sorted_axis': extent_ratio_axis.astype(float).tolist(), 'rigid_pose_observation_eligible': rigid_pose_observation_eligible, 'rigid_pose_observation_reason': rigid_pose_observation_reason, 'depth_median_m': float(vis['depth_median_m']), 'depth_p05_m': float(vis['depth_p05_m']), 'depth_p95_m': float(vis['depth_p95_m']), 'sample_summary': vis['sample_summary'], 'object_surface_ownership_filter': vis.get('object_surface_ownership_filter'), 'first_surface_depth_ownership': vis.get('first_surface_depth_ownership'), 'native_depth_metric_eligibility': vis.get('native_depth_metric_eligibility'), 'claim_scope': 'visible metric surface measurement only; hand-owned and non-first-surface depth pixels are excluded before metric lifting; not hidden geometry and not final object pose'}
        row_obj = {'object_id': object_id, 'track_id': args.track_id, 'label': args.track_id, 'description': (object_plan_record or {}).get('description'), 'status': 'visible_metric_surface_measurement', 'visible': True, 'mask_path': str(vis['mask_path']), 'bbox_xyxy': [float(x) for x in owned_bbox[:4]], 'area_px': owned_area_source_px, 'raw_sam2_bbox_xyxy': vis['track_row'].get('bbox_xyxy'), 'raw_sam2_area_px': vis['track_row'].get('area_px'), 'mask_metadata_coordinate_frame': 'source_image_pixels_recomputed_from_object_owned_mask', 'depth_m': float(vis['depth_median_m']), 'visible_geometry_candidate': geom, 'reconstructed_geometry_pose': pose, 'v19_physical_model': (object_plan_record or {}).get('physical_model'), 'rigid_pose_observation_eligible': rigid_pose_observation_eligible, 'rigid_pose_observation_reason': rigid_pose_observation_reason}
        objects.append(row_obj)
        frame['objects'] = objects
        output_frames.append(frame)
        rows.append({'frame_idx': int(idx), 'status': 'visible_metric_surface_measurement', 'vertex_count': int(len(world_points)), 'mask_path': str(vis['mask_path']), 'owned_mask_area_source_px': owned_area_source_px, 'owned_mask_bbox_source_xyxy': [float(x) for x in owned_bbox[:4]], 'depth_median_m': float(vis['depth_median_m']), 'centroid_world_m': centroid.astype(float).tolist(), 'world_extent_m': world_extent_m.astype(float).tolist(), 'extent_ratio_to_population_diag': extent_ratio_diag, 'extent_ratio_to_population_sorted_axis_max': max_extent_ratio_axis, 'rigid_pose_observation_eligible': rigid_pose_observation_eligible, 'rigid_pose_observation_reason': rigid_pose_observation_reason, 'camera_pose_source': vis['camera_source'], 'intrinsics_source': vis.get('intrinsics_source'), 'intrinsics_transform_contract': vis.get('intrinsics_transform_contract'), 'mask_depth_transform_contract': vis.get('mask_depth_transform_contract')})
    annotations = {'frames': output_frames, 'v19_visible_geometry_adapter': {'method': 'build_v19_visible_geometry_from_sam2_depth', 'case': args.case, 'track_id': args.track_id, 'object_id': object_id, 'raw_frame_manifest': str(args.raw_frame_manifest), 'sam2_track': str(sam2_path), 'depth_npz': str(args.depth_npz), 'base_annotations': str(args.base_annotations) if args.base_annotations else None, 'camera_npz': str(args.camera_npz) if args.camera_npz else None, 'calibration_contract': str(args.calibration_contract) if args.calibration_contract else None, 'depth_image_plane': args.depth_image_plane, 'mask_image_plane': args.mask_image_plane, 'pixel_center_convention': args.pixel_center_convention, 'metric_eligibility_manifest': str(metric_eligibility_path) if metric_eligibility_path else None, 'metric_eligibility_manifest_sha256': camera_contract_sha256_file(metric_eligibility_path) if metric_eligibility_path else None, 'depth_camera_contract_binding_validation': depth_contract_binding_validation, 'anchor_frame_idx': int(anchor), 'claim_scope': 'visible metric surfel and initial-pose adapter for rigid branch; downstream completion/pose/interval solvers must produce the physical object pose claim'}}
    if isinstance(raw_payload.get('video'), dict):
        annotations['raw_video'] = raw_payload['video']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / 'annotations_v19_visible_geometry.json'
    write_json_payload(annotations_path, annotations)
    visible_mask_rows = [{'frame_idx': int(row['frame_idx']), 'target_entity_id': object_id, 'object_id': object_id, 'track_id': args.track_id, 'status': 'v19_visible_metric_surface_mask', 'measurement_type': 'sam2_mask_with_metric_depth_support', 'mask_path': str(row['mask_path']), 'saved_mask_path': str(row['mask_path']), 'mask_area_px': row.get('owned_mask_area_source_px'), 'mask_bbox_xyxy': row.get('owned_mask_bbox_source_xyxy'), 'raw_sam2_mask_area_px': None if sam2.get(int(row['frame_idx']), {}).get('area_px') is None else float(sam2[int(row['frame_idx'])]['area_px']), 'depth_median_m': row.get('depth_median_m'), 'visible_vertex_count': row.get('vertex_count'), 'coordinate_frame': 'source_image_mask_plus_metric_depth', 'claim_scope': 'visible model mask measurement with metric-depth support; not hidden geometry or object pose'} for row in rows if row.get('status') == 'visible_metric_surface_measurement' and row.get('mask_path')]
    visible_mask_report_path = args.output_dir / 'v19_visible_mask_report.json'
    visible_mask_report = {'method': 'build_v19_visible_geometry_from_sam2_depth_visible_mask_report', 'status': 'ok', 'case': args.case, 'target_entity_id': object_id, 'track_id': args.track_id, 'claim_scope': 'SAM2 mask rows with metric depth support for downstream visible-surface/ownership factor builders. This report is a mask measurement index, not object pose.', 'surface_rows': visible_mask_rows}
    write_json_payload(visible_mask_report_path, visible_mask_report)
    report = {'method': 'build_v19_visible_geometry_from_sam2_depth', 'status': 'ok', 'case': args.case, 'track_id': args.track_id, 'object_id': object_id, 'claim_scope': 'SAM2 masks plus metric depth/camera are lifted to visible object surfels and centroid initial poses. This is measurement/adaptation input for the rigid branch, not final object pose.', 'inputs': {'raw_frame_manifest': str(args.raw_frame_manifest), 'sam2_track': str(sam2_path), 'depth_npz': str(args.depth_npz), 'base_annotations': str(args.base_annotations) if args.base_annotations else None, 'camera_npz': str(args.camera_npz) if args.camera_npz else None, 'calibration_contract': str(args.calibration_contract) if args.calibration_contract else None, 'depth_image_plane': args.depth_image_plane, 'mask_image_plane': args.mask_image_plane, 'pixel_center_convention': args.pixel_center_convention, 'object_plan': str(args.object_plan) if args.object_plan else None, 'remote_root': str(args.remote_root) if args.remote_root else None, 'local_root': str(args.local_root) if args.local_root else None, 'metric_eligibility_manifest': str(metric_eligibility_path) if metric_eligibility_path else None, 'metric_eligibility_manifest_sha256': camera_contract_sha256_file(metric_eligibility_path) if metric_eligibility_path else None}, 'outputs': {'annotations': str(annotations_path), 'depth_fused_report': str(args.output_dir / 'v19_visible_geometry_depth_fused_report.json'), 'visible_mask_report': str(visible_mask_report_path), 'anchor_candidate_proposals': anchor_candidate_report.get('outputs', {}).get('anchor_candidate_proposals') if isinstance(anchor_candidate_report, dict) else None, 'anchor_candidate_review': anchor_candidate_report.get('outputs', {}).get('anchor_candidate_review') if isinstance(anchor_candidate_report, dict) else None, 'anchor_visible_surface_mesh': anchor_mesh_reconstruction.get('poisson_mesh_path') or anchor_mesh_reconstruction.get('convex_hull_mesh_path') or anchor_mesh_reconstruction.get('fused_point_cloud_path')}, 'requested_frame_start': int(indices[0]), 'requested_frame_end': int(indices[-1]), 'requested_frame_count': int(len(indices)), 'output_frame_count': int(len(output_indices)), 'preserve_source_index': bool(args.preserve_source_index), 'visible_metric_frame_count': int(sum((1 for row in rows if row.get('status') == 'visible_metric_surface_measurement'))), 'anchor_frame_idx': int(anchor), 'anchor_centroid_world_m': anchor_centroid.astype(float).tolist(), 'anchor_extent_world_m': anchor_extent_m.astype(float).tolist(), 'extent_consistency_reference': {'method': 'orientation_invariant_visible_population_median', 'sorted_axis_extent_median_m': population_sorted_extent_median.astype(float).tolist(), 'extent_diag_median_m': population_diag_median, 'frame_count': int(len(population_extents)), 'anchor_not_used_as_extent_gate_reference': True}, 'anchor_visible_surface_mesh_reconstruction': anchor_mesh_reconstruction, 'camera_pose_source_counts': dict(camera_source_counts), 'intrinsics_source_counts': dict(Counter((str(vis.get('intrinsics_source')) for vis in visible_data.values()))), 'calibration_contract': calibration_summary, 'depth_camera_contract_binding_validation': depth_contract_binding_validation, 'parameters': {'allow_implicit_depth_resize': bool(args.allow_implicit_depth_resize), 'allow_implicit_mask_resize': bool(args.allow_implicit_mask_resize), 'depth_image_plane': args.depth_image_plane, 'mask_image_plane': args.mask_image_plane, 'pixel_center_convention': args.pixel_center_convention, 'pixel_stride': int(args.pixel_stride), 'max_points': int(args.max_points), 'min_valid_points': int(args.min_valid_points), 'min_depth_m': float(args.min_depth_m), 'max_depth_m': float(args.max_depth_m), 'allow_camera_frame_world': bool(args.allow_camera_frame_world), 'carry_invisible_pose': bool(args.carry_invisible_pose), 'preserve_source_index': bool(args.preserve_source_index), 'exclude_hand_regions': bool(args.exclude_hand_bboxes), 'hand_ownership_primitive': 'projected_full_hawor_mano_triangle_silhouette', 'hand_bbox_subtraction_used': False, 'hand_silhouette_exclusion_pad_px': int(args.hand_bbox_exclusion_pad_px), 'native_depth_metric_eligibility_enabled': metric_eligibility is not None, 'robust_first_surface_depth_ownership': bool(args.robust_first_surface_depth_ownership), 'first_surface_mad_sigma': float(args.first_surface_mad_sigma), 'first_surface_min_half_width_m': float(args.first_surface_min_half_width_m), 'first_surface_min_retained_fraction': float(args.first_surface_min_retained_fraction), 'first_surface_confidence_seed_percentile': float(args.first_surface_confidence_seed_percentile), 'first_surface_local_depth_step_max_m': float(args.first_surface_local_depth_step_max_m), 'first_surface_max_removed_distance_inside_mask_px': float(args.first_surface_max_removed_distance_inside_mask_px), 'first_surface_max_confidence_flagged_interior_fraction': float(args.first_surface_max_confidence_flagged_interior_fraction), 'first_surface_min_interior_confidence_flagged_fraction': float(args.first_surface_min_interior_confidence_flagged_fraction), 'first_surface_max_unexplained_interior_pixels': int(args.first_surface_max_unexplained_interior_pixels), 'first_surface_min_component_pixels': int(args.first_surface_min_component_pixels), 'first_surface_max_small_component_fraction': float(args.first_surface_max_small_component_fraction), 'first_surface_fail_raw_to_robust_extent_ratio': float(args.first_surface_fail_raw_to_robust_extent_ratio), 'rigid_extent_ratio_max': float(args.rigid_extent_ratio_max), 'rigid_extent_axis_ratio_max': float(args.rigid_extent_axis_ratio_max), 'anchor_candidate_count': int(args.anchor_candidate_count), 'anchor_candidate_min_gap': int(args.anchor_candidate_min_gap), 'require_anchor_frame': bool(args.require_anchor_frame)}, 'rows': rows, 'skipped_rows_preview': skipped_rows[:200]}
    report_path = args.output_dir / 'v19_visible_geometry_adapter_report.json'
    write_json_payload(report_path, report)
    depth_fused = {'method': 'build_v19_visible_geometry_from_sam2_depth_depth_fused_compat_report', 'status': 'ok', 'case': args.case, 'claim': 'Compatibility report for V18 rigid evidence bundle. Visible geometry lives in annotations; no hidden geometry is claimed here.', 'object_rows': [{'object_id': object_id, 'track_id': args.track_id, 'frame_surface_rows': report['visible_metric_frame_count'], 'visible_geometry_adapter_report': str(report_path), 'visible_mask_report': str(visible_mask_report_path), 'annotations': str(annotations_path), 'mesh_reconstruction': anchor_mesh_reconstruction, 'object_geometry_complete': False, 'hidden_geometry_reconstructed': False, 'complete_object_pose_ready': False}]}
    write_json_payload(args.output_dir / 'v19_visible_geometry_depth_fused_report.json', depth_fused)
    return report

def surface_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--track-id', required=True)
    parser.add_argument('--object-id', default=None)
    parser.add_argument('--raw-frame-manifest', type=Path, required=True)
    parser.add_argument('--sam2-root', type=Path, default=Path('.'))
    parser.add_argument('--sam2-track-json', type=Path, default=None)
    parser.add_argument('--metric-eligibility-manifest', type=Path, default=None, help='Optional hash-bound native-depth eligibility raster manifest. Default None preserves legacy P09 behavior.')
    parser.add_argument('--depth-npz', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base-annotations', type=Path, default=None)
    parser.add_argument('--camera-npz', type=Path, default=None)
    parser.add_argument('--calibration-contract', type=Path, default=None, help='V19 camera contract. V2 contracts explicitly map calibration pixels into the selected depth image plane; legacy constant-K contracts remain supported.')
    parser.add_argument('--depth-image-plane', default='source_rgb', help='Named image plane in a V2 camera contract corresponding to the depth raster.')
    parser.add_argument('--mask-image-plane', default='sam2_mask', help='Named image plane in a V2 camera contract corresponding to the SAM2/object-owned mask raster.')
    parser.add_argument('--pixel-center-convention', choices=['integer_pixel_centers_opencv', 'pixel_corner_origin'], default='integer_pixel_centers_opencv')
    parser.add_argument('--allow-implicit-depth-resize', action='store_true', help='Compatibility override when the actual depth raster differs from the declared plane. Prefer an explicit image plane in the contract.')
    parser.add_argument('--allow-implicit-mask-resize', action='store_true', help='Compatibility override when the actual mask raster differs from the declared plane. Prefer an explicit image plane in the contract.')
    parser.add_argument('--object-plan', type=Path, default=None)
    parser.add_argument('--remote-root', type=Path, default=None, help='Remote path prefix to localize mask/raw paths from server-produced manifests')
    parser.add_argument('--local-root', type=Path, default=None, help='Local path prefix corresponding to --remote-root')
    parser.add_argument('--frame-start', type=int, default=None)
    parser.add_argument('--frame-end', type=int, default=None)
    parser.add_argument('--anchor-frame', type=int, default=None)
    parser.add_argument('--propose-anchor-candidates-only', action='store_true', help='Lift visible mask/depth rows, write anchor_candidate_proposals.json and anchor_candidate_review.jpg, then stop before exporting canonical anchor geometry. The agent must inspect and choose an anchor.')
    parser.add_argument('--require-anchor-frame', action='store_true', help='Fail instead of falling back to the max-point frame when --anchor-frame is absent. Use after candidate proposal so anchor choice is explicit.')
    parser.add_argument('--anchor-candidate-count', type=int, default=12, help='Number of diversified candidate frames to show in the visual anchor review sheet.')
    parser.add_argument('--anchor-candidate-min-gap', type=int, default=8, help='Minimum frame gap used when diversifying review-sheet anchor candidates; ranked JSON still contains every candidate.')
    parser.add_argument('--anchor-candidate-panel-width', type=int, default=360, help='Width in pixels for each candidate panel in anchor_candidate_review.jpg.')
    parser.add_argument('--pixel-stride', type=int, default=4)
    parser.add_argument('--max-points', type=int, default=2500)
    parser.add_argument('--min-valid-points', type=int, default=50)
    parser.add_argument('--min-depth-m', type=float, default=0.05)
    parser.add_argument('--max-depth-m', type=float, default=4.0)
    parser.add_argument('--allow-camera-frame-world', action='store_true', help='Explicitly use each camera frame as its own world frame when no world camera pose is available. This is not valid for temporal metric world claims.')
    parser.add_argument('--carry-invisible-pose', action='store_true', help='Carry the nearest visible centroid pose into invisible frames as an uncertain initialization only.')
    parser.add_argument('--preserve-source-index', action=argparse.BooleanOptionalAction, default=True, help="Write one output frame row per raw source frame so annotations['frames'][frame_idx] remains valid for V18 rigid tools.")
    parser.add_argument('--exclude-hand-bboxes', '--exclude-hand-regions', dest='exclude_hand_bboxes', action=argparse.BooleanOptionalAction, default=True, help='Subtract projected full HaWoR/MANO triangle silhouettes before lifting object depth; coarse hand bboxes are never ownership masks.')
    parser.add_argument('--hand-bbox-exclusion-pad-px', type=int, default=4, help='Legacy option name: padding in mask pixels around projected MANO hand silhouettes (not bboxes).')
    parser.add_argument('--robust-first-surface-depth-ownership', action=argparse.BooleanOptionalAction, default=True, help='Select confidence/depth-geodesic first-surface surfels per connected object component; appearance masks remain unchanged.')
    parser.add_argument('--first-surface-mad-sigma', type=float, default=2.5, help='Per connected component: MAD sigma used only for the initial depth seed before confidence/geodesic growth.')
    parser.add_argument('--first-surface-min-half-width-m', type=float, default=0.03, help='Minimum per-component seed half-width in meters.')
    parser.add_argument('--first-surface-min-retained-fraction', type=float, default=0.9, help='Fail closed when validated depth-coherent support retains less than this fraction of valid owned depth.')
    parser.add_argument('--first-surface-confidence-seed-percentile', type=float, default=95.0, help='Maximum traversable UniDepth error proxy is derived from this percentile of each component seed.')
    parser.add_argument('--first-surface-local-depth-step-max-m', type=float, default=0.005, help='Maximum adjacent-pixel depth step during geodesic support growth.')
    parser.add_argument('--first-surface-max-removed-distance-inside-mask-px', type=float, default=10.0, help='Distance defining the ordinary boundary quarantine band; deeper rejection must pass sparse/confidence-flagged interior gates.')
    parser.add_argument('--first-surface-max-confidence-flagged-interior-fraction', type=float, default=0.015, help='Maximum fraction of valid owned depth that may be quarantined beyond the boundary band when UniDepth flags it as high predicted error.')
    parser.add_argument('--first-surface-min-interior-confidence-flagged-fraction', type=float, default=0.95, help='Required fraction of nontrivial interior rejection whose UniDepth predicted-error proxy exceeds its component seed threshold.')
    parser.add_argument('--first-surface-max-unexplained-interior-pixels', type=int, default=5, help='Permit only this many isolated interior raster samples without the confidence-flagged proof.')
    parser.add_argument('--first-surface-min-component-pixels', type=int, default=20)
    parser.add_argument('--first-surface-max-small-component-fraction', type=float, default=0.01)
    parser.add_argument('--first-surface-fail-raw-to-robust-extent-ratio', type=float, default=2.0, help='Diagnostic tail-dominance ratio; a larger raw extent is accepted only when support and boundary-localization quarantine gates pass.')
    parser.add_argument('--seed', type=int, default=1901)
    parser.add_argument('--anchor-mesh-min-voxel-m', type=float, default=0.002)
    parser.add_argument('--anchor-mesh-voxel-divisor', type=float, default=80.0)
    parser.add_argument('--anchor-mesh-poisson-depth', type=int, default=7)
    parser.add_argument('--anchor-mesh-poisson-density-quantile', type=float, default=0.02)
    parser.add_argument('--rigid-extent-ratio-max', type=float, default=2.75, help='Mark visible surfaces whose orientation-invariant extent diagonal differs from the robust visible-population median by more than this factor as ineligible for rigid pose fitting.')
    parser.add_argument('--rigid-extent-axis-ratio-max', type=float, default=3.25, help='Axis-wise companion to --rigid-extent-ratio-max for detecting elongated hand/background leakage.')
    return parser.parse_args(argv)
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from envs.qualification.camera import K_from_intrinsics
from envs.qualification.camera import load_contract
from envs.qualification.camera import plane_intrinsics
depth_order_SCHEMA = 'rgb_scene_depth_order_owned_sam2_v1'

def depth_order_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'required input is not a file: {path}')
    return {'path': str(path), 'bytes': int(path.stat().st_size), 'sha256': depth_order_sha256(path)}

def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON object required: {path}')
    return value

def indexed_frames(rows: Any, label: str) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{label} must have nonempty frames')
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get('frame_idx'), bool):
            raise ValueError(f'malformed {label} frame')
        idx = int(row['frame_idx'])
        if idx in result:
            raise ValueError(f'duplicate {label} frame {idx}')
        result[idx] = row
    return result

def indexed_track(value: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for (key, row) in value.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'SAM2 track key is not an integer: {key!r}') from exc
        if str(idx) != str(key) or idx in result or (not isinstance(row, dict)):
            raise ValueError(f'malformed/duplicate SAM2 frame {key!r}')
        result[idx] = row
    if not result:
        raise ValueError('empty SAM2 track')
    return result

def front_only(mask: np.ndarray, hand_z: np.ndarray, depth_z: np.ndarray, *, tolerance_m: float=0.005, supported: np.ndarray | None=None) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    hand_z = np.asarray(hand_z, dtype=np.float64)
    depth_z = np.asarray(depth_z, dtype=np.float64)
    if mask.ndim != 2 or hand_z.shape != mask.shape or depth_z.shape != mask.shape or (not np.isfinite(tolerance_m)) or (tolerance_m < 0):
        raise ValueError('depth-order raster/tolerance mismatch')
    supported = np.isfinite(depth_z) & (depth_z > 0) if supported is None else np.asarray(supported, dtype=bool)
    if supported.shape != mask.shape:
        raise ValueError('support raster mismatch')
    hit = np.isfinite(hand_z) & (hand_z > 0)
    valid = hit & np.isfinite(depth_z) & (depth_z > 0) & supported
    delta = np.zeros(mask.shape, dtype=np.float64)
    np.subtract(hand_z, depth_z, out=delta, where=valid)
    front = valid & (delta < -tolerance_m)
    behind = valid & (delta > tolerance_m)
    near = valid & ~front & ~behind
    classes = {'front': mask & front, 'behind': mask & behind, 'near': mask & near, 'no_hit': mask & ~hit, 'unsupported_depth': mask & hit & ~valid}
    counts = {'raw_pixels': int(mask.sum()), 'removed_front_pixels': int(classes['front'].sum()), 'preserved_behind_pixels': int(classes['behind'].sum()), 'preserved_near_pixels': int(classes['near'].sum()), 'preserved_no_hit_pixels': int(classes['no_hit'].sum()), 'preserved_unsupported_depth_pixels': int(classes['unsupported_depth'].sum())}
    return (mask & ~front, counts, classes)

def sample_depth_at_mask_pixels(depth: np.ndarray, A_depth_from_mask: np.ndarray, mask_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest discrete depth samples under the declared half-pixel affine."""
    depth = np.asarray(depth, dtype=np.float64)
    affine = np.asarray(A_depth_from_mask, dtype=np.float64)
    if depth.ndim != 2 or affine.shape != (3, 3) or (not np.isfinite(affine).all()):
        raise ValueError('malformed depth/affine contract')
    (yy, xx) = np.mgrid[:mask_shape[0], :mask_shape[1]]
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ affine.T
    valid_h = np.isfinite(points).all(axis=2) & (np.abs(points[..., 2]) > 1e-12)
    source_x = np.rint(np.divide(points[..., 0], points[..., 2], out=np.zeros_like(points[..., 0], dtype=float), where=valid_h)).astype(int)
    source_y = np.rint(np.divide(points[..., 1], points[..., 2], out=np.zeros_like(points[..., 1], dtype=float), where=valid_h)).astype(int)
    inside = valid_h & (source_x >= 0) & (source_x < depth.shape[1]) & (source_y >= 0) & (source_y < depth.shape[0])
    sampled = np.full(mask_shape, np.nan, dtype=np.float64)
    sampled[inside] = depth[source_y[inside], source_x[inside]]
    return (sampled, inside)

def mask_geometry(mask: np.ndarray, A_mask_from_source: np.ndarray) -> dict[str, Any]:
    (ys, xs) = np.where(mask)
    if not len(xs):
        return {'visible': False, 'area_px': 0.0, 'bbox_xyxy': None, 'center_xy': None}
    inverse = np.linalg.inv(A_mask_from_source)
    corners = np.asarray([[xs.min(), ys.min(), 1.0], [xs.max() + 1, ys.max() + 1, 1.0]]) @ inverse.T
    corners = corners[:, :2] / corners[:, 2:3]
    center = np.asarray([xs.mean(), ys.mean(), 1.0]) @ inverse.T
    center = center[:2] / center[2]
    return {'visible': True, 'area_px': float(len(xs) / abs(np.linalg.det(A_mask_from_source[:2, :2]))), 'bbox_xyxy': [float(corners[0, 0]), float(corners[0, 1]), float(corners[1, 0]), float(corners[1, 1])], 'center_xy': center.astype(float).tolist()}

def hand_geometry(frame: dict[str, Any], bridge: Any, hawor: Any, bridge_path: Path, hawor_path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    geometry: list[tuple[np.ndarray, np.ndarray]] = []
    identities = []
    hands = frame.get('hands')
    if not isinstance(hands, list):
        raise ValueError(f"frame {frame.get('frame_idx')} hands must be a list")
    for hand in hands:
        if not isinstance(hand, dict):
            raise ValueError('malformed hand annotation')
        state = hand.get('metric_mano_state')
        reference = state.get('vertices_reference') if isinstance(state, dict) else None
        if not isinstance(reference, dict):
            identities.append({'side': hand.get('hand_side'), 'status': 'missing_metric_mano_preserved'})
            continue
        side = str(hand.get('hand_side') or '').lower()
        if side not in {'left', 'right'}:
            raise ValueError('invalid hand side')
        if Path(str(reference.get('bridge_npz', ''))).resolve() != bridge_path:
            raise ValueError('hand bridge identity mismatch')
        if Path(str(reference.get('source_hawor_npz', ''))).resolve() != hawor_path:
            raise ValueError('hand HaWoR identity mismatch')
        row = reference.get('bridge_row_index')
        key = reference.get('bridge_vertices_camera_array')
        if isinstance(row, bool) or not isinstance(row, int) or (not isinstance(key, str)) or (key not in bridge.files):
            raise ValueError('invalid MANO bridge reference')
        vertices_array = bridge[key]
        if row < 0 or row >= len(vertices_array):
            raise ValueError('MANO bridge row outside archive')
        frame_idx = int(frame['frame_idx'])
        if 'frame_idx' not in bridge.files or 'hand_side' not in bridge.files or int(bridge['frame_idx'][row]) != frame_idx or (str(bridge['hand_side'][row]) != side):
            raise ValueError('MANO bridge frame/side identity mismatch')
        faces_key = side + '_faces'
        if faces_key not in hawor.files:
            raise ValueError(f'HaWoR archive lacks {faces_key}')
        vertices = np.asarray(vertices_array[row], dtype=np.float64)
        faces = np.asarray(hawor[faces_key])
        geometry.append((vertices, faces))
        identities.append({'side': side, 'status': 'full_mano_surface', 'bridge_row_index': row, 'vertices_array': key, 'faces_array': faces_key})
    return (geometry, identities)

def build_depth_order(args: argparse.Namespace) -> dict[str, Any]:
    from envs.qualification.object_observation import classify_native_depth
    from utils.candidate_geometry import nearest_hand_first_hits
    from envs.qualification.camera import validate_depth_camera_contract_binding
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f'depth-order output must be new: {output_dir}')
    base_path = args.base_annotations.expanduser().resolve()
    raw_track_path = args.raw_sam2_track.expanduser().resolve()
    depth_path = args.depth_npz.expanduser().resolve()
    calibration_path = args.calibration_contract.expanduser().resolve()
    bridge_path = args.mano_bridge.expanduser().resolve()
    hawor_path = args.hawor_npz.expanduser().resolve()
    input_records = {name: record(path) for (name, path) in {'base_annotations': base_path, 'raw_sam2_track': raw_track_path, 'depth_npz': depth_path, 'calibration_contract': calibration_path, 'mano_bridge': bridge_path, 'hawor_npz': hawor_path}.items()}
    base_payload = load_object(base_path)
    if base_payload.get('case') != args.case:
        raise ValueError('base annotation case mismatch')
    frames = indexed_frames(base_payload.get('frames'), 'base annotations')
    raw_payload = load_object(raw_track_path)
    track = indexed_track(raw_payload)
    if set(frames) != set(track):
        raise ValueError('base annotations/raw SAM2 frame coverage mismatch')
    depth = load_depth_npz(depth_path)
    if set(depth['frame_to_i']) != set(frames):
        raise ValueError('base annotations/depth frame coverage mismatch')
    (camera_payload, normalized) = load_contract(calibration_path, expected_frame_ids=sorted(frames))
    validate_depth_camera_contract_binding(depth=depth, camera_contract_v2={'payload': camera_payload, 'normalized': normalized}, calibration_contract_path=calibration_path, depth_image_plane=args.depth_image_plane, allow_implicit_depth_resize=False)
    (depth_intrinsics, depth_plane) = plane_intrinsics(camera_payload, normalized, plane_name=args.depth_image_plane, actual_size_wh=(int(depth['depth'].shape[2]), int(depth['depth'].shape[1])))
    A_depth_from_calibration = np.asarray(depth_plane['A_actual_plane_from_calibration'], dtype=np.float64)
    output_dir.mkdir(parents=True)
    mask_dir = output_dir / 'sam2_masks'
    mask_dir.mkdir()
    eligibility_dir = output_dir / 'native_depth_metric_eligibility'
    eligibility_dir.mkdir()
    output_track: dict[str, Any] = {}
    rows = []
    eligibility_rows = []
    bridge = np.load(bridge_path, mmap_mode='r', allow_pickle=False)
    hawor = np.load(hawor_path, mmap_mode='r', allow_pickle=False)
    try:
        for frame_idx in sorted(frames):
            raw_row = track[frame_idx]
            output_row = json.loads(json.dumps(raw_row))
            source_mask_value = raw_row.get('mask_path')
            visible = raw_row.get('visible') is True
            if source_mask_value is None:
                if visible:
                    raise ValueError(f'visible raw SAM2 frame {frame_idx} lacks mask')
                output_row['depth_order_ownership'] = {'schema': depth_order_SCHEMA, 'status': 'raw_track_not_visible', 'source_mask': None}
                output_track[str(frame_idx)] = output_row
                rows.append({'frame_idx': frame_idx, 'status': 'raw_track_not_visible'})
                depth_row = np.asarray(depth['depth'][depth['frame_to_i'][frame_idx]])
                raster_path = eligibility_dir / f'{frame_idx:06d}.npz'
                class_code = np.zeros(depth_row.shape, dtype=np.uint8)
                eligible = np.zeros(depth_row.shape, dtype=bool)
                np.savez_compressed(raster_path, schema=np.asarray(RASTER_SCHEMA), frame_idx=np.asarray(frame_idx, dtype=np.int64), class_code=class_code, metric_candidate_eligible=eligible)
                eligibility_rows.append({'frame_idx': frame_idx, 'status': 'raw_track_not_visible', 'owned_mask': None, 'depth_array_sha256': array_sha256(depth_row), 'class_code_array_sha256': array_sha256(class_code), 'metric_candidate_array_sha256': array_sha256(eligible), 'raster': record(raster_path), 'counts': {name: int(np.count_nonzero(class_code == code)) for (name, code) in CLASS_CODES.items()}})
                continue
            source_mask_path = Path(source_mask_value).expanduser().resolve()
            source_ref = record(source_mask_path)
            image = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.ndim != 2:
                raise ValueError(f'unreadable raw SAM2 mask at frame {frame_idx}')
            raw_mask = image > 0
            if visible and (not raw_mask.any()):
                raise ValueError(f'visible raw SAM2 mask is empty at frame {frame_idx}')
            (mask_intrinsics, mask_plane) = plane_intrinsics(camera_payload, normalized, plane_name=args.mask_image_plane, actual_size_wh=(int(raw_mask.shape[1]), int(raw_mask.shape[0])))
            A_mask_from_calibration = np.asarray(mask_plane['A_actual_plane_from_calibration'], dtype=np.float64)
            A_depth_from_mask = A_depth_from_calibration @ np.linalg.inv(A_mask_from_calibration)
            (sampled_depth, raster_inside) = sample_depth_at_mask_pixels(depth['depth'][depth['frame_to_i'][frame_idx]], A_depth_from_mask, raw_mask.shape)
            (hands, hand_identities) = hand_geometry(frames[frame_idx], bridge, hawor, bridge_path, hawor_path)
            (nearest_z, hand_reports) = nearest_hand_first_hits(hands, K_from_intrinsics(mask_intrinsics), raw_mask.shape, args.near_m)
            supported = raster_inside & np.isfinite(sampled_depth) & (sampled_depth > 0)
            (owned, counts, _classes) = front_only(raw_mask, nearest_z, sampled_depth, tolerance_m=args.depth_order_tolerance_m, supported=supported)
            destination = mask_dir / f'{frame_idx:06d}.png'
            if not cv2.imwrite(str(destination), owned.astype(np.uint8) * 255):
                raise RuntimeError(f'failed to write owned mask {destination}')
            destination_ref = record(destination)
            depth_row = np.asarray(depth['depth'][depth['frame_to_i'][frame_idx]])
            (nearest_depth_z, native_hand_reports) = nearest_hand_first_hits(hands, K_from_intrinsics(depth_intrinsics), depth_row.shape, args.near_m)
            declared_hands = frames[frame_idx].get('hands')
            complete_hand_geometry = isinstance(declared_hands, list) and len(declared_hands) > 0 and (len(hands) == len(declared_hands)) and all((identity.get('status') == 'full_mano_surface' for identity in hand_identities))
            validated_hand_absence = isinstance(declared_hands, list) and len(declared_hands) == 0
            (class_code, metric_eligible, metric_counts) = classify_native_depth(owned_mask=owned, raw_mask=raw_mask, hand_z=nearest_depth_z, depth_z=depth_row, A_depth_from_mask=A_depth_from_mask, complete_hand_geometry=complete_hand_geometry, validated_hand_absence=validated_hand_absence, tolerance_m=args.depth_order_tolerance_m, uncertainty_padding_mask_px=args.uncertainty_padding_mask_px)
            raster_path = eligibility_dir / f'{frame_idx:06d}.npz'
            np.savez_compressed(raster_path, schema=np.asarray(RASTER_SCHEMA), frame_idx=np.asarray(frame_idx, dtype=np.int64), class_code=class_code, metric_candidate_eligible=metric_eligible)
            hand_surface_arrays = [{'vertices_sha256': array_sha256(vertices), 'vertices_shape': list(np.asarray(vertices).shape), 'faces_sha256': array_sha256(faces), 'faces_shape': list(np.asarray(faces).shape)} for (vertices, faces) in hands]
            eligibility_rows.append({'frame_idx': frame_idx, 'status': 'native_depth_classified', 'owned_mask': destination_ref, 'source_mask': source_ref, 'depth_array_sha256': array_sha256(depth_row), 'depth_size_wh': [int(depth_row.shape[1]), int(depth_row.shape[0])], 'mask_size_wh': [int(raw_mask.shape[1]), int(raw_mask.shape[0])], 'A_depth_from_mask': A_depth_from_mask.tolist(), 'K_depth': K_from_intrinsics(depth_intrinsics).tolist(), 'complete_hand_geometry': complete_hand_geometry, 'validated_hand_absence': validated_hand_absence, 'hand_identities': hand_identities, 'hand_surface_arrays': hand_surface_arrays, 'hand_raster_reports': native_hand_reports, 'class_code_array_sha256': array_sha256(class_code), 'metric_candidate_array_sha256': array_sha256(metric_eligible), 'raster': record(raster_path), 'counts': metric_counts})
            geometry = mask_geometry(owned, A_mask_from_calibration)
            output_row.update(geometry)
            output_row['mask_path'] = str(destination)
            output_row['depth_order_ownership'] = {'schema': depth_order_SCHEMA, 'status': 'front_only_first_hit_classified' if visible else 'invisible_row_mask_preserved_unconsumed', 'source_mask': source_ref, 'owned_mask': destination_ref, 'depth_order_tolerance_m': float(args.depth_order_tolerance_m), 'classes': counts, 'metric_eligibility': {'supported_raw_pixels': int((raw_mask & supported).sum()), 'unsupported_raw_pixels': int((raw_mask & ~supported).sum()), 'policy': 'owned appearance support is not metric eligibility; P09 robust depth quarantine remains required'}, 'hands': hand_identities}
            output_track[str(frame_idx)] = output_row
            rows.append({'frame_idx': frame_idx, 'status': output_row['depth_order_ownership']['status'], 'source_mask': source_ref, 'owned_mask': destination_ref, 'mask_size_wh': [int(raw_mask.shape[1]), int(raw_mask.shape[0])], 'depth_size_wh': [int(depth['depth'].shape[2]), int(depth['depth'].shape[1])], 'A_depth_from_mask': A_depth_from_mask.tolist(), 'K_mask': K_from_intrinsics(mask_intrinsics).tolist(), 'counts': counts, 'metric_supported_raw_pixels': int((raw_mask & supported).sum()), 'hand_raster_reports': hand_reports, 'hand_identities': hand_identities})
    finally:
        bridge.close()
        hawor.close()
    track_path = output_dir / 'sam2_track.json'
    track_path.write_text(json.dumps(output_track, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    track_ref = record(track_path)
    eligibility_manifest = {'schema': SCHEMA, 'status': 'ok', 'case': args.case, 'object_id': args.object_id, 'track_id': args.track_id, 'native_raster_plane': args.depth_image_plane, 'class_codes': CLASS_CODES, 'policy': {'metric_candidates': 'hand clearly behind or clear no-hand non-overlap only; existing P09 robust depth/confidence filtering still required', 'ineligible': 'front, near, unsupported depth, invalid mask coordinates, unresolved hand geometry, and no-hit uncertainty fringe', 'ordinary_object_body': 'clear no-hit support outside the coordinate-bound projected-hand uncertainty band remains a candidate', 'appearance_masks_unchanged_by_metric_eligibility': True, 'physical_authority': False}, 'parameters': {'depth_order_tolerance_m': float(args.depth_order_tolerance_m), 'uncertainty_padding_mask_px': int(args.uncertainty_padding_mask_px), 'depth_image_plane': args.depth_image_plane, 'mask_image_plane': args.mask_image_plane}, 'inputs': {**input_records, 'owned_sam2_track': track_ref}, 'frames': eligibility_rows}
    eligibility_manifest_path = output_dir / 'native_depth_metric_eligibility.json'
    eligibility_manifest_path.write_text(json.dumps(eligibility_manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    eligibility_manifest_ref = record(eligibility_manifest_path)
    totals = {key: int(sum(((row.get('counts') or {}).get(key, 0) for row in rows))) for key in ('raw_pixels', 'removed_front_pixels', 'preserved_behind_pixels', 'preserved_near_pixels', 'preserved_no_hit_pixels', 'preserved_unsupported_depth_pixels')}
    report = {'schema': depth_order_SCHEMA, 'status': 'ok', 'method': 'perspective_correct_full_mano_first_hit_front_only', 'case': args.case, 'object_id': args.object_id, 'track_id': args.track_id, 'claim_scope': 'derived P09 mask support only; raw SAM2 appearance is immutable and separately bound', 'inputs': input_records, 'outputs': {'owned_sam2_track': track_ref, 'owned_mask_dir': str(mask_dir), 'native_depth_metric_eligibility': eligibility_manifest_ref, 'native_depth_metric_eligibility_dir': str(eligibility_dir)}, 'policy': {'front': 'remove only when nearest MANO z < measured depth z - tolerance', 'behind': 'preserve when nearest MANO z > measured depth z + tolerance', 'near': 'preserve when absolute z difference is within tolerance', 'no_hit_or_invalid_depth': 'preserve appearance; not automatically metric eligible', 'triangle_depth': 'perspective-correct inverse-z barycentric first hit', 'multiple_hands': 'nearest positive first hit over all declared full MANO triangle surfaces', 'bbox_or_centroid_substitution': False}, 'parameters': {'depth_order_tolerance_m': float(args.depth_order_tolerance_m), 'near_m': float(args.near_m), 'depth_image_plane': args.depth_image_plane, 'mask_image_plane': args.mask_image_plane, 'uncertainty_padding_mask_px': int(args.uncertainty_padding_mask_px)}, 'frame_count': len(rows), 'totals': totals, 'rows': rows}
    report_path = output_dir / 'depth_order_ownership_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report

def depth_order_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-annotations', 'raw-sam2-track', 'depth-npz', 'calibration-contract', 'mano-bridge', 'hawor-npz', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('case', 'object-id', 'track-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--depth-image-plane', default='source_rgb')
    parser.add_argument('--mask-image-plane', default='sam2_mask')
    parser.add_argument('--depth-order-tolerance-m', type=float, default=0.005)
    parser.add_argument('--near-m', type=float, default=1e-06)
    parser.add_argument('--uncertainty-padding-mask-px', type=int, default=4)
    args = parser.parse_args(argv)
    if not np.isfinite(args.depth_order_tolerance_m) or args.depth_order_tolerance_m < 0 or (not np.isfinite(args.near_m)) or (args.near_m <= 0) or (args.uncertainty_padding_mask_px < 0):
        parser.error('invalid depth-order tolerance or near plane')
    return args
import argparse
import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

def safe_id(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', value).strip('_')

def mask_bool(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.asarray(img, dtype=np.uint8) > 0

def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    (ys, xs) = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

def square_pad_bbox(box: tuple[int, int, int, int], width: int, height: int, pad_frac: float=0.08) -> tuple[int, int, int, int]:
    (x0, y0, x1, y1) = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    side = int(round(max(bw, bh) * (1.0 + 2.0 * pad_frac)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    sx0 = int(round(cx - side / 2.0))
    sy0 = int(round(cy - side / 2.0))
    sx1 = sx0 + side
    sy1 = sy0 + side
    if sx0 < 0:
        sx1 -= sx0
        sx0 = 0
    if sy0 < 0:
        sy1 -= sy0
        sy0 = 0
    if sx1 > width:
        sx0 -= sx1 - width
        sx1 = width
    if sy1 > height:
        sy0 -= sy1 - height
        sy1 = height
    sx0 = max(0, sx0)
    sy0 = max(0, sy0)
    sx1 = min(width, sx1)
    sy1 = min(height, sy1)
    return (sx0, sy0, sx1, sy1)

def make_rgba_crop(raw_path: Path, mask_path: Path, out_path: Path) -> dict[str, Any]:
    raw = Image.open(raw_path).convert('RGB')
    mask_img = Image.open(mask_path).convert('L').resize(raw.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_img, dtype=np.uint8) > 0
    box = bbox_from_mask(mask)
    if box is None:
        raise RuntimeError(f'empty mask: {mask_path}')
    crop_box = square_pad_bbox(box, raw.width, raw.height)
    raw_arr = np.asarray(raw, dtype=np.uint8).copy()
    raw_arr[~mask] = 0
    rgba = Image.fromarray(np.dstack([raw_arr, mask.astype(np.uint8) * 255]), mode='RGBA')
    crop = rgba.crop(crop_box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return {'raw_image': str(raw_path), 'mask': str(mask_path), 'crop_rgba': str(out_path), 'source_size': [raw.width, raw.height], 'mask_bbox_xyxy': list(box), 'crop_bbox_xyxy': list(crop_box), 'mask_area_px': int(mask.sum()), 'crop_size': list(crop.size)}

def find_depth_fused_object(report_path: Path, object_id: str) -> dict[str, Any]:
    report = read_json(report_path)
    for row in report.get('object_rows', []):
        if isinstance(row, dict) and row.get('object_id') == object_id:
            return row
    return {}

def build_selected_anchor_mesh_reconstruction(*, selected: dict[str, Any], output_dir: Path, object_id: str, min_voxel_m: float=0.002, voxel_divisor: float=80.0, poisson_depth: int=7, poisson_density_quantile: float=0.02) -> dict[str, Any]:
    import open3d as o3d
    "Materialize canonical observed geometry from the exact selected P11 row.\n\n    An anchor override is atomic: RGB, mask, camera, metric surfels, centroid,\n    and canonical observed mesh must all name the same frame. Reusing P09's mesh\n    for another frame mixes object poses and invalidates P13 alignment/residuals.\n    "
    if o3d is None:
        raise RuntimeError('Open3D is required to bind the selected P11 canonical surface')
    frame_idx = int(selected['frame_idx'])
    geom = selected.get('visible_geometry_candidate')
    if not isinstance(geom, dict):
        raise RuntimeError('selected P11 row lacks visible_geometry_candidate')
    geom_frame_idx = int(geom.get('frame_idx', -1))
    if geom_frame_idx != frame_idx:
        raise RuntimeError(f'selected P11 frame {frame_idx} disagrees with visible geometry frame {geom_frame_idx}')
    if geom.get('rigid_pose_observation_eligible') is not True:
        raise RuntimeError('selected P11 row is not eligible metric-surface evidence')
    points_world = np.asarray(geom.get('world_vertices_sample_m') or [], dtype=np.float64)
    centroid_world = np.asarray(geom.get('centroid_world_m') or [], dtype=np.float64).reshape(-1)
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) < 30 or (not np.isfinite(points_world).all()):
        raise RuntimeError('selected P11 row lacks at least 30 finite world surfels')
    if centroid_world.shape != (3,) or not np.isfinite(centroid_world).all():
        raise RuntimeError('selected P11 row lacks a finite metric centroid')
    recomputed_centroid = points_world.mean(axis=0)
    centroid_error_m = float(np.linalg.norm(recomputed_centroid - centroid_world))
    if centroid_error_m > 1e-06:
        raise RuntimeError(f'selected P11 centroid disagrees with its world surfels by {centroid_error_m} m')
    points_canonical = points_world - centroid_world[None, :]
    mesh_dir = output_dir / 'selected_anchor_visible_surface_mesh' / safe_id(object_id)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_canonical)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(extent))
    voxel_size_m = max(float(min_voxel_m), diag / max(float(voxel_divisor), 1.0))
    pcd = pcd.voxel_down_sample(voxel_size_m)
    down_points = np.asarray(pcd.points, dtype=np.float64)
    if len(down_points) < 30:
        raise RuntimeError('selected P11 surfels collapse below 30 points after fixed voxelization')
    stem = f'frame_{frame_idx:06d}_{safe_id(object_id)}_selected_anchor'
    pcd_path = mesh_dir / f'{stem}_visible_points_canonical.ply'
    if not o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f'failed to write selected P11 point cloud: {pcd_path}')
    (hull, _) = pcd.compute_convex_hull()
    hull.compute_vertex_normals()
    hull_path = mesh_dir / f'{stem}_convex_hull_visible_candidate.ply'
    if not o3d.io.write_triangle_mesh(str(hull_path), hull, write_ascii=False, compressed=False):
        raise RuntimeError(f'failed to write selected P11 convex hull: {hull_path}')
    poisson_path: Path | None = None
    poisson_error: str | None = None
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size_m * 4.0, float(min_voxel_m) * 4.0), max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)
        (poisson, densities) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(poisson_depth))
        poisson = poisson.crop(bbox)
        density = np.asarray(densities)
        if density.size and len(poisson.vertices) == len(density):
            threshold = float(np.quantile(density, float(poisson_density_quantile)))
            poisson.remove_vertices_by_mask(density < threshold)
        poisson.remove_degenerate_triangles()
        poisson.remove_duplicated_triangles()
        poisson.remove_duplicated_vertices()
        poisson.remove_non_manifold_edges()
        if len(poisson.vertices) == 0 or len(poisson.triangles) == 0:
            raise RuntimeError('empty Poisson result')
        poisson.compute_vertex_normals()
        poisson_path = mesh_dir / f'{stem}_poisson_visible_mesh.ply'
        if not o3d.io.write_triangle_mesh(str(poisson_path), poisson, write_ascii=False, compressed=False):
            raise RuntimeError('Open3D write_triangle_mesh returned false')
    except Exception as exc:
        poisson_error = f'{type(exc).__name__}:{exc}'
        poisson_path = None
    return {'status': 'selected_p11_anchor_visible_surface_mesh_exported' if poisson_path is not None else 'selected_p11_anchor_visible_surface_hull_only_poisson_failed', 'coordinate_frame': 'object_canonical_selected_anchor_centroid_frame', 'canonical_coordinate_source': 'selected P11 frame world_vertices_sample_m minus that same row centroid_world_m', 'atomic_anchor_binding': True, 'anchor_frame_idx': frame_idx, 'anchor_centroid_world_m': centroid_world.astype(float).tolist(), 'centroid_recomputed_error_m': centroid_error_m, 'point_count_input': int(len(points_world)), 'point_count_downsampled': int(len(down_points)), 'voxel_size_m': float(voxel_size_m), 'canonical_bbox_min_m': down_points.min(axis=0).astype(float).tolist(), 'canonical_bbox_max_m': down_points.max(axis=0).astype(float).tolist(), 'fused_point_cloud_path': str(pcd_path), 'poisson_mesh_path': str(poisson_path) if poisson_path is not None else None, 'convex_hull_mesh_path': str(hull_path), 'poisson_error': poisson_error, 'claim_scope': 'selected-frame partial metric observed surface; no hidden geometry'}

def validate_selected_anchor_binding(*, selected: dict[str, Any], mesh_reconstruction: dict[str, Any]) -> dict[str, Any]:
    frame_idx = int(selected['frame_idx'])
    geom = selected.get('visible_geometry_candidate') or {}
    selected_centroid = np.asarray(geom.get('centroid_world_m') or [], dtype=np.float64)
    mesh_centroid = np.asarray(mesh_reconstruction.get('anchor_centroid_world_m') or [], dtype=np.float64)
    centroid_error_m = float(np.linalg.norm(selected_centroid - mesh_centroid)) if selected_centroid.shape == (3,) and mesh_centroid.shape == (3,) else float('inf')
    binding = {'selected_frame_idx': frame_idx, 'visible_geometry_frame_idx': int(geom.get('frame_idx', -1)), 'canonical_surface_frame_idx': int(mesh_reconstruction.get('anchor_frame_idx', -1)), 'selected_vs_canonical_centroid_error_m': centroid_error_m, 'required_same_frame': True, 'required_centroid_tolerance_m': 1e-06}
    if binding['visible_geometry_frame_idx'] != frame_idx or binding['canonical_surface_frame_idx'] != frame_idx or (not np.isfinite(centroid_error_m)) or (centroid_error_m > 1e-06):
        raise RuntimeError(f'non-atomic selected P11 anchor binding: {binding}')
    binding['validated'] = True
    return binding

def build_anchor_evidence(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--depth-fused-report', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--selected-frame-idx', type=int, default=None, help='documented override when the top support mask is visually/geometrically invalid')
    parser.add_argument('--selection-note', default=None)
    args = parser.parse_args(argv)
    ann = read_json(args.annotations)
    object_safe = safe_id(args.object_id.replace('object:', 'object_'))
    out_dir = args.output_root / args.case / object_safe / 'evidence_bundle'
    crop_dir = out_dir / 'crops'
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for frame in ann.get('frames', []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get('frame_idx'))
        for obj in frame.get('objects', []) if isinstance(frame.get('objects'), list) else []:
            if not isinstance(obj, dict) or obj.get('object_id') != args.object_id:
                continue
            geom = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else {}
            raw_text = str(frame.get('raw_frame_path') or '')
            mask_text = str(obj.get('mask_path') or '')
            if not raw_text or not mask_text:
                continue
            raw_path = Path(raw_text)
            mask_path = Path(mask_text)
            if not raw_path.is_file() or not mask_path.is_file():
                continue
            try:
                mask = mask_bool(mask_path)
                mask_area = int(mask.sum())
            except Exception:
                mask_area = 0
            visible_vertices = int(geom.get('vertex_count') or len(geom.get('world_vertices_sample_m') or []))
            candidates.append({'frame_idx': frame_idx, 'raw_frame_path': str(raw_path), 'mask_path': str(mask_path), 'visible_depth_vertex_count': visible_vertices, 'mask_area_px': mask_area, 'camera': frame.get('camera'), 'visible_geometry_candidate': geom, 'object_bbox_xyxy': obj.get('bbox_xyxy')})
    if not candidates:
        raise RuntimeError(f'no evidence candidates for {args.case} {args.object_id}')
    candidates.sort(key=lambda r: (-int(r['visible_depth_vertex_count']), -int(r['mask_area_px']), int(r['frame_idx'])))
    if args.selected_frame_idx is not None:
        matches = [c for c in candidates if int(c['frame_idx']) == int(args.selected_frame_idx)]
        if not matches:
            raise RuntimeError(f'selected override frame {args.selected_frame_idx} is not a candidate for {args.object_id}')
        selected = dict(matches[0])
        selection_rule = 'documented_selected_frame_override_due_invalid_top_mask_or_conditioning_evidence'
    else:
        selected = dict(candidates[0])
        selection_rule = 'max_visible_depth_vertex_count_then_mask_area_then_earliest_frame'
    selected_crop = crop_dir / f"frame_{int(selected['frame_idx']):06d}_{object_safe}_rgba.png"
    selected['trellis_conditioning_crop'] = make_rgba_crop(Path(selected['raw_frame_path']), Path(selected['mask_path']), selected_crop)
    depth_fused = find_depth_fused_object(args.depth_fused_report, args.object_id)
    selected_mesh_reconstruction = build_selected_anchor_mesh_reconstruction(selected=selected, output_dir=out_dir, object_id=args.object_id)
    anchor_binding = validate_selected_anchor_binding(selected=selected, mesh_reconstruction=selected_mesh_reconstruction)
    depth_fused_selected = copy.deepcopy(depth_fused)
    original_mesh_reconstruction = copy.deepcopy(depth_fused.get('mesh_reconstruction')) if isinstance(depth_fused.get('mesh_reconstruction'), dict) else None
    depth_fused_selected['mesh_reconstruction'] = selected_mesh_reconstruction
    depth_fused_selected['original_p09_mesh_reconstruction_diagnostic_only'] = original_mesh_reconstruction
    partial_mesh_paths = {key: selected_mesh_reconstruction[key] for key in ['fused_point_cloud_path', 'poisson_mesh_path', 'convex_hull_mesh_path'] if selected_mesh_reconstruction.get(key)}
    report = {'method': 'build_v18_compact_rigid_evidence_bundle', 'status': 'ok', 'case': args.case, 'object_id': args.object_id, 'output_dir': str(out_dir), 'selection_rule': selection_rule, 'selection_note': args.selection_note, 'selected_frame_idx': int(selected['frame_idx']), 'selected_anchor_atomic_binding': anchor_binding, 'selected': selected, 'candidate_count': len(candidates), 'candidate_summary': {'visible_depth_vertex_count_max': int(candidates[0]['visible_depth_vertex_count']), 'visible_depth_vertex_count_median': float(np.median([c['visible_depth_vertex_count'] for c in candidates])), 'mask_area_px_max': int(max((c['mask_area_px'] for c in candidates)))}, 'depth_fused_object_row': depth_fused_selected, 'partial_metric_geometry_paths': partial_mesh_paths, 'all_candidate_frames': [{k: c[k] for k in ['frame_idx', 'visible_depth_vertex_count', 'mask_area_px', 'raw_frame_path', 'mask_path']} for c in candidates]}
    (out_dir / 'evidence_bundle_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
import numpy as np
from PIL import Image
FALSE_READY: dict[str, bool] = {'annotation_ready': False, 'deliverable_ready': False, 'accuracy_target_met': False, 'object_geometry_complete': False, 'object_pose_requirement_met': False, 'rigid_pose_requirement_met': False, 'v3_solver_complete': False}
STATUS = 'v18_part_visible_surface_evidence'
CLAIM = 'This artifact extracts bounded depth-backed visible surfaces for accepted part/segment masks. The surfaces are part-level visible geometry evidence only; they do not reconstruct hidden geometry, estimate part pose, or complete object pose.'

def resize_bool_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape_hw:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from utils.data import as_list as visible_track_as_list
from utils.geometry import frame_camera_pose
from utils.data import load_json as visible_track_load_json
from utils.data import numeric_summary as visible_track_numeric_summary
from utils.data import write_json as visible_track_write_json
STATE_ACTIVE = 'active_visible_surface'
STATE_NO_MASK = 'no_visible_surface_observation'
STATE_EMPTY_MASK = 'empty_visible_surface_mask'
STATE_MISSING_DEPTH = 'missing_metric_depth'
STATE_NO_DEPTH_SUPPORT = 'visible_mask_without_metric_depth_support'

def coherent_first_surface_mask(*, mask: np.ndarray, depth_row: dict[str, Any], enabled: bool, min_gap_m: float, min_retained_fraction: float, min_removed_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the nearest coherent depth component inside a model mask.

    A visible-surface mask is an object/part support region, but depth inside it
    can contain holes, background, or mixed transparent/edge pixels. The first
    physical surface that can occlude a hand is the nearest coherent component,
    not a disconnected far-depth tail. Smooth depth variation is preserved; only
    a clear upper-tail depth discontinuity creates a filtered solver mask.
    """
    depth = np.asarray(depth_row['depth'], dtype=np.float32)
    mask_d = resize_bool_mask(mask.astype(bool), depth.shape)
    valid = mask_d & np.isfinite(depth) & (depth > 1e-05)
    z = np.sort(depth[valid].astype(float))
    diag: dict[str, Any] = {'enabled': bool(enabled), 'input_mask_area_px_depth_grid': int(mask_d.sum()), 'input_valid_depth_pixels': int(z.size), 'state': 'disabled' if not enabled else 'no_filter_applied', 'cutoff_depth_m': None, 'retained_valid_depth_pixels': int(z.size), 'removed_valid_depth_pixels': 0, 'removed_valid_depth_fraction': 0.0, 'depth_m_before': visible_track_numeric_summary(z)}
    if not enabled or z.size < 32:
        return (mask_d, diag)
    gaps = np.diff(z)
    if gaps.size == 0:
        return (mask_d, diag)
    lo = max(0, int(0.5 * z.size) - 1)
    hi = max(lo + 1, int(0.995 * z.size) - 1)
    cand = gaps[lo:hi]
    if cand.size == 0:
        return (mask_d, diag)
    gap_idx = lo + int(np.argmax(cand))
    max_gap = float(gaps[gap_idx])
    positive = cand[cand > 0]
    median_gap = float(np.median(positive)) if positive.size else 0.0
    required_gap = max(float(min_gap_m), 25.0 * median_gap)
    cutoff = float(z[gap_idx])
    retained = depth[valid] <= cutoff
    retained_count = int(np.count_nonzero(retained))
    removed_count = int(z.size - retained_count)
    retained_fraction = float(retained_count / max(1, z.size))
    removed_fraction = float(removed_count / max(1, z.size))
    q90 = float(np.percentile(z, 90.0))
    q95 = float(np.percentile(z, 95.0))
    q90_removed = int(np.count_nonzero(z > q90))
    q90_removed_fraction = float(q90_removed / max(1, z.size))
    diag.update({'candidate_gap_m': max_gap, 'median_upper_tail_gap_m': median_gap, 'required_gap_m': required_gap, 'candidate_cutoff_depth_m': cutoff, 'candidate_retained_fraction': retained_fraction, 'candidate_removed_fraction': removed_fraction, 'q90_depth_m': q90, 'q95_depth_m': q95, 'q95_minus_q90_m': float(q95 - q90)})
    if max_gap < required_gap or retained_fraction < float(min_retained_fraction) or removed_fraction < float(min_removed_fraction):
        if q95 - q90 >= float(min_gap_m) and 1.0 - q90_removed_fraction >= float(min_retained_fraction) and (q90_removed_fraction >= float(min_removed_fraction)):
            cutoff = q90
            retained = depth[valid] <= cutoff
            retained_count = int(np.count_nonzero(retained))
            removed_count = int(z.size - retained_count)
            retained_fraction = float(retained_count / max(1, z.size))
            removed_fraction = float(removed_count / max(1, z.size))
            diag['state'] = 'applied_upper_tail_quantile_depth_filter'
        else:
            diag['state'] = 'no_clear_disconnected_far_depth_component'
            return (mask_d, diag)
    filtered = np.zeros_like(mask_d, dtype=bool)
    valid_indices = np.argwhere(valid)
    keep_indices = valid_indices[retained]
    filtered[keep_indices[:, 0], keep_indices[:, 1]] = True
    applied_state = str(diag.get('state') or '')
    if not applied_state.startswith('applied_'):
        applied_state = 'applied_disconnected_far_depth_tail_filter'
    diag.update({'state': applied_state, 'cutoff_depth_m': cutoff, 'retained_valid_depth_pixels': retained_count, 'removed_valid_depth_pixels': removed_count, 'removed_valid_depth_fraction': removed_fraction, 'depth_m_after': visible_track_numeric_summary(depth[filtered].astype(float))})
    return (filtered, diag)

def write_mask(mask: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    return path

def parse_spans(values: list[list[int]] | None) -> list[tuple[int, int]]:
    if not values:
        raise RuntimeError('at least one --frame-span START END is required')
    spans: list[tuple[int, int]] = []
    for pair in values:
        if len(pair) != 2:
            raise RuntimeError(f'invalid --frame-span value: {pair}')
        (a, b) = (int(pair[0]), int(pair[1]))
        if b < a:
            raise RuntimeError(f'decreasing frame span {a}:{b}')
        spans.append((a, b))
    return spans

def frame_ids_from_spans(spans: list[tuple[int, int]]) -> list[int]:
    out: set[int] = set()
    for (a, b) in spans:
        out.update(range(a, b + 1))
    return sorted(out)

def load_frames(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(f['frame_idx']): f for f in visible_track_as_list(annotations.get('frames')) if isinstance(f, dict) and f.get('frame_idx') is not None}

def entity_tokens(entity_id: str) -> set[str]:
    raw = str(entity_id)
    tokens = {raw}
    if raw.startswith('object:'):
        tokens.add(raw.split(':', 1)[1])
    else:
        tokens.add(f'object:{raw}')
    return tokens

def row_matches_target(row: dict[str, Any], target_entity_id: str) -> bool:
    tokens = entity_tokens(target_entity_id)
    for key in ('target_entity_id', 'object_id', 'entity_id', 'track_id'):
        value = row.get(key)
        if isinstance(value, str):
            return value in tokens
    return True

def load_mask_rows(report_path: Path, target_entity_id: str) -> dict[int, dict[str, Any]]:
    if not report_path.exists():
        raise FileNotFoundError(f'missing visible mask report: {report_path}')
    payload = visible_track_load_json(report_path)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ('saved_mask_rows_after_start', 'target_mask_rows', 'surface_rows'):
            rows.extend([r for r in visible_track_as_list(payload.get(key)) if isinstance(r, dict) and row_matches_target(r, target_entity_id)])
            if rows:
                break
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get('frame_idx') is None:
            continue
        raw = row.get('saved_mask_path') or row.get('mask_path')
        if not isinstance(raw, str):
            continue
        path = Path(raw)
        if path.exists():
            frame_idx = int(row['frame_idx'])
            if frame_idx in out:
                raise RuntimeError(f'duplicate visible mask row for target {target_entity_id} frame {frame_idx} in {report_path}')
            out[frame_idx] = dict(row)
    return out

def load_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f'could not decode visible surface mask: {path}')
    return arr > 0

def load_ownership_rows(report_path: Path | None, target_entity_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    if report_path is None:
        return {}
    payload = visible_track_load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in visible_track_as_list(payload.get('ownership_rows') if isinstance(payload, dict) else None):
        if not isinstance(row, dict) or not row_matches_target(row, target_entity_id):
            continue
        if row.get('frame_idx') is None or row.get('hand_side') is None:
            continue
        key = (int(row['frame_idx']), str(row['hand_side']))
        if key in out:
            raise RuntimeError(f'duplicate ownership row for target {target_entity_id} frame/side {key} in {report_path}')
        out[key] = dict(row)
    return out

def lift_mask_depth_samples(*, frame: dict[str, Any], frame_idx: int, mask: np.ndarray, depth_row: dict[str, Any], stride: int, max_samples: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depth = np.asarray(depth_row['depth'], dtype=np.float32)
    mask_d = resize_bool_mask(mask.astype(bool), depth.shape)
    valid = mask_d & np.isfinite(depth) & (depth > 1e-05)
    (ys_all, xs_all) = np.nonzero(valid)
    if xs_all.size == 0:
        empty = {'pixel_uv': np.zeros((0, 2), dtype=np.float32), 'depth_m': np.zeros((0,), dtype=np.float32), 'points_camera_m': np.zeros((0, 3), dtype=np.float32), 'points_world_m': np.zeros((0, 3), dtype=np.float32)}
        return (empty, {'surface_state': STATE_NO_DEPTH_SUPPORT, 'mask_area_px_raw': int(mask.sum()), 'mask_area_px_depth_grid': int(mask_d.sum()), 'valid_depth_pixels': 0, 'sample_count': 0, 'depth_m': visible_track_numeric_summary(np.zeros((0,), dtype=float))})
    stride = max(1, int(stride))
    sampled_grid = np.zeros_like(valid, dtype=bool)
    sampled_grid[::stride, ::stride] = True
    sampled = valid & sampled_grid
    (ys, xs) = np.nonzero(sampled)
    if xs.size == 0:
        (ys, xs) = (ys_all, xs_all)
    if int(max_samples) > 0 and xs.size > int(max_samples):
        order = np.linspace(0, xs.size - 1, int(max_samples), dtype=np.int64)
        ys = ys[order]
        xs = xs[order]
    z = depth[ys, xs].astype(np.float64)
    intr = np.asarray(depth_row['intrinsics'], dtype=np.float64).reshape(-1)
    if intr.size == 9:
        (fx, fy, cx, cy) = (float(intr[0]), float(intr[4]), float(intr[2]), float(intr[5]))
    else:
        (fx, fy, cx, cy) = [float(v) for v in intr[:4]]
    x_cam = (xs.astype(np.float64) - cx) * z / fx
    y_cam = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.column_stack([x_cam, y_cam, z])
    (r_c2w, t_c2w) = frame_camera_pose(frame)
    pts_world = pts_cam @ r_c2w.T + t_c2w[None, :]
    arrays = {'pixel_uv': np.column_stack([xs, ys]).astype(np.float32), 'depth_m': z.astype(np.float32), 'points_camera_m': pts_cam.astype(np.float32), 'points_world_m': pts_world.astype(np.float32)}
    summary = {'surface_state': STATE_ACTIVE, 'mask_area_px_raw': int(mask.sum()), 'mask_area_px_depth_grid': int(mask_d.sum()), 'valid_depth_pixels': int(xs_all.size), 'sample_count': int(xs.size), 'depth_m': visible_track_numeric_summary(z.astype(float)), 'bbox_world_min_m': pts_world.min(axis=0).astype(float).tolist(), 'bbox_world_max_m': pts_world.max(axis=0).astype(float).tolist(), 'extent_world_m': (pts_world.max(axis=0) - pts_world.min(axis=0)).astype(float).tolist()}
    return (arrays, summary)

def render_review(*, frame: dict[str, Any], frame_idx: int, mask: np.ndarray, samples: dict[str, np.ndarray], output_path: Path, title: str) -> bool:
    raw = frame.get('raw_frame_path')
    image = cv2.imread(str(raw), cv2.IMREAD_COLOR) if isinstance(raw, str) else None
    if image is None:
        return False
    mask_img = resize_bool_mask(mask.astype(bool), image.shape[:2])
    overlay = image.copy()
    overlay[mask_img] = (180, 0, 180)
    canvas = cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)
    uv = np.asarray(samples.get('pixel_uv', np.zeros((0, 2))), dtype=float)
    if uv.size:
        if mask.shape[:2] != image.shape[:2]:
            sx = image.shape[1] / float(mask.shape[1])
            sy = image.shape[0] / float(mask.shape[0])
            uv = uv * np.asarray([sx, sy], dtype=float)[None, :]
        max_pts = 4500
        if len(uv) > max_pts:
            uv = uv[np.linspace(0, len(uv) - 1, max_pts, dtype=np.int64)]
        for (u, v) in uv.astype(int):
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(canvas, (int(u), int(v)), 1, (0, 220, 0), -1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), canvas))

def build_visible_surface_track(args: argparse.Namespace) -> dict[str, Any]:
    spans = parse_spans(args.frame_span)
    frame_ids = frame_ids_from_spans(spans)
    annotations = visible_track_load_json(args.annotations)
    frames = load_frames(annotations)
    mask_rows = load_mask_rows(args.visible_mask_report, str(args.target_entity_id))
    ownership_rows = load_ownership_rows(args.visible_ownership_factor_report, str(args.target_entity_id))
    depth_rows = load_depth_sources(list(args.depth_npz))
    case_root = args.output_root / args.case
    sample_dir = case_root / 'visible_surface_samples'
    review_dir = case_root / 'review_frames'
    filtered_mask_dir = case_root / 'first_surface_masks'
    sample_dir.mkdir(parents=True, exist_ok=True)
    sides = list(args.sides)
    frame_surface_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    review_frames = set((int(f) for f in args.review_frames))
    for frame_idx in frame_ids:
        frame = frames.get(frame_idx)
        mask_row = mask_rows.get(frame_idx)
        depth_row = depth_rows.get(frame_idx)
        frame_surface: dict[str, Any]
        surface_npz_path: Path | None = None
        mask_path: Path | None = None
        if frame is None:
            frame_surface = {'frame_idx': int(frame_idx), 'surface_state': 'missing_annotation_frame'}
        elif mask_row is None:
            frame_surface = {'frame_idx': int(frame_idx), 'surface_state': STATE_NO_MASK}
        else:
            mask_path = Path(str(mask_row.get('saved_mask_path') or mask_row.get('mask_path')))
            mask = load_mask(mask_path)
            if not np.any(mask):
                frame_surface = {'frame_idx': int(frame_idx), 'surface_state': STATE_EMPTY_MASK, 'surface_mask_path': str(mask_path), 'mask_area_px_raw': 0, 'valid_depth_pixels': 0, 'sample_count': 0}
            elif depth_row is None:
                frame_surface = {'frame_idx': int(frame_idx), 'surface_state': STATE_MISSING_DEPTH, 'surface_mask_path': str(mask_path), 'mask_area_px_raw': int(mask.sum()), 'valid_depth_pixels': 0, 'sample_count': 0}
            else:
                (depth_mask, depth_filter_diag) = coherent_first_surface_mask(mask=mask, depth_row=depth_row, enabled=bool(args.coherent_first_surface_depth_filter), min_gap_m=float(args.first_surface_depth_gap_min_m), min_retained_fraction=float(args.first_surface_min_retained_fraction), min_removed_fraction=float(args.first_surface_min_removed_fraction))
                factor_mask_path = mask_path
                if str(depth_filter_diag.get('state') or '').startswith('applied_'):
                    factor_mask_path = write_mask(depth_mask, filtered_mask_dir / f'{frame_idx:06d}_first_surface_depth_filtered.png')
                (samples, summary) = lift_mask_depth_samples(frame=frame, frame_idx=frame_idx, mask=depth_mask, depth_row=depth_row, stride=int(args.sample_stride), max_samples=int(args.max_samples_per_frame))
                frame_surface = {'frame_idx': int(frame_idx), 'surface_mask_path': str(factor_mask_path), 'raw_surface_mask_path': str(mask_path), 'depth_filter': depth_filter_diag, **summary}
                if frame_surface['surface_state'] == STATE_ACTIVE:
                    surface_npz_path = sample_dir / f'{frame_idx:06d}_visible_surface_samples.npz'
                    np.savez_compressed(surface_npz_path, **samples, metadata_json=json.dumps({'frame_idx': int(frame_idx), 'surface_mask_path': str(factor_mask_path), 'raw_surface_mask_path': str(mask_path), 'depth_filter': depth_filter_diag, 'claim': 'visible first-surface samples only; no hidden geometry or object pose'}))
                    frame_surface['visible_surface_npz_path'] = str(surface_npz_path)
                    if frame_idx in review_frames:
                        review_path = review_dir / f'{frame_idx:06d}_visible_surface_track.jpg'
                        ok = render_review(frame=frame, frame_idx=frame_idx, mask=depth_mask, samples=samples, output_path=review_path, title=f'visible surface track frame {frame_idx}')
                        if ok:
                            frame_surface['review_path'] = str(review_path)
                else:
                    frame_surface['visible_surface_npz_path'] = None
        state_counts[str(frame_surface.get('surface_state'))] += 1
        frame_surface_rows.append(frame_surface)
        for side in sides:
            factor_surface = dict(frame_surface)
            ownership_row = ownership_rows.get((int(frame_idx), str(side)))
            ownership_mask_path = None
            if frame is not None and depth_row is not None and (ownership_row is not None) and isinstance(ownership_row.get('adjusted_entity_mask_path'), str) and Path(str(ownership_row.get('adjusted_entity_mask_path'))).exists():
                ownership_mask_path = Path(str(ownership_row['adjusted_entity_mask_path']))
                ownership_mask = load_mask(ownership_mask_path)
                if np.any(ownership_mask):
                    (ownership_depth_mask, ownership_depth_filter_diag) = coherent_first_surface_mask(mask=ownership_mask, depth_row=depth_row, enabled=bool(args.coherent_first_surface_depth_filter), min_gap_m=float(args.first_surface_depth_gap_min_m), min_retained_fraction=float(args.first_surface_min_retained_fraction), min_removed_fraction=float(args.first_surface_min_removed_fraction))
                    factor_mask_path = ownership_mask_path
                    if str(ownership_depth_filter_diag.get('state') or '').startswith('applied_'):
                        factor_mask_path = write_mask(ownership_depth_mask, filtered_mask_dir / f'{frame_idx:06d}_{side}_ownership_first_surface_depth_filtered.png')
                    (ownership_samples, ownership_summary) = lift_mask_depth_samples(frame=frame, frame_idx=frame_idx, mask=ownership_depth_mask, depth_row=depth_row, stride=int(args.sample_stride), max_samples=int(args.max_samples_per_frame))
                    factor_surface = {'frame_idx': int(frame_idx), 'surface_mask_path': str(factor_mask_path), 'raw_surface_mask_path': str(ownership_mask_path), 'depth_filter': ownership_depth_filter_diag, **ownership_summary}
                    if factor_surface['surface_state'] == STATE_ACTIVE:
                        side_npz_path = sample_dir / f'{frame_idx:06d}_{side}_ownership_filtered_visible_surface_samples.npz'
                        np.savez_compressed(side_npz_path, **ownership_samples, metadata_json=json.dumps({'frame_idx': int(frame_idx), 'hand_side': str(side), 'surface_mask_path': str(factor_mask_path), 'raw_surface_mask_path': str(ownership_mask_path), 'depth_filter': ownership_depth_filter_diag, 'visible_ownership_factor_report': str(args.visible_ownership_factor_report), 'claim': 'visible first-surface samples after side-specific hand-owned-pixel quarantine; no hidden geometry or object pose'}))
                        factor_surface['visible_surface_npz_path'] = str(side_npz_path)
                        if frame_idx in review_frames:
                            review_path = review_dir / f'{frame_idx:06d}_{side}_ownership_filtered_visible_surface_track.jpg'
                            ok = render_review(frame=frame, frame_idx=frame_idx, mask=ownership_depth_mask, samples=ownership_samples, output_path=review_path, title=f'ownership-filtered visible surface f{frame_idx} {side}')
                            if ok:
                                factor_surface['review_path'] = str(review_path)
                    else:
                        factor_surface['visible_surface_npz_path'] = None
                else:
                    factor_surface = {'frame_idx': int(frame_idx), 'surface_state': STATE_EMPTY_MASK, 'surface_mask_path': str(ownership_mask_path), 'mask_area_px_raw': 0, 'valid_depth_pixels': 0, 'sample_count': 0, 'visible_surface_npz_path': None}
            active = factor_surface.get('surface_state') == STATE_ACTIVE
            factor_rows.append({'factor_family': 'visible_surface_track', 'target_entity_id': args.target_entity_id, 'frame_idx': int(frame_idx), 'hand_side': side, 'variable_affected': 'H_t_and_constraint_eligibility', 'observation_type': 'model_mask_metric_depth_visible_first_surface' if ownership_mask_path is None else 'ownership_filtered_model_mask_metric_depth_visible_first_surface', 'surface_state': str(factor_surface.get('surface_state')), 'residual_or_quarantine_rule': 'if active, MANO vertices projecting into the visible first-surface mask are constrained by one-sided depth order; hidden signed-volume nonpenetration is quarantined for this target/frame', 'quarantine_hidden_volume': bool(active and args.quarantine_hidden_volume), 'surface_mask_path': factor_surface.get('surface_mask_path'), 'visible_surface_npz_path': factor_surface.get('visible_surface_npz_path'), 'valid_depth_pixels': int(factor_surface.get('valid_depth_pixels', 0) or 0), 'sample_count': int(factor_surface.get('sample_count', 0) or 0), 'mask_area_px_raw': int(factor_surface.get('mask_area_px_raw', 0) or 0), 'raw_surface_mask_path': factor_surface.get('raw_surface_mask_path'), 'depth_filter': factor_surface.get('depth_filter'), 'depth_order_margin_m': float(args.depth_order_margin_m), 'provenance': {'annotations': str(args.annotations), 'visible_mask_report': str(args.visible_mask_report), 'visible_ownership_factor_report': None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report), 'depth_npz': [str(p) for p in args.depth_npz], 'mask_source_row': mask_row if mask_row is not None else None, 'ownership_source_row': ownership_row if ownership_row is not None else None}, 'rendered_uncertainty_channel': 'visible first-surface mask/depth and MANO depth-order residual; no hidden volume accepted'})
    active_rows = [r for r in frame_surface_rows if r.get('surface_state') == STATE_ACTIVE]
    report = {'method': 'build_v18_visible_surface_track_factor', 'status': 'ok', 'claim_scope': 'Visible first-surface track factor only: model-produced mask plus metric depth samples. It does not assert hidden geometry, object pose, contact, or signed nonpenetration.', 'case': args.case, 'target_entity_id': args.target_entity_id, 'inputs': {'annotations': str(args.annotations), 'visible_mask_report': str(args.visible_mask_report), 'depth_npz': [str(p) for p in args.depth_npz]}, 'parameters': {'frame_spans': [[int(a), int(b)] for (a, b) in spans], 'sides': sides, 'sample_stride': int(args.sample_stride), 'max_samples_per_frame': int(args.max_samples_per_frame), 'quarantine_hidden_volume': bool(args.quarantine_hidden_volume), 'depth_order_margin_m': float(args.depth_order_margin_m), 'coherent_first_surface_depth_filter': bool(args.coherent_first_surface_depth_filter), 'first_surface_depth_gap_min_m': float(args.first_surface_depth_gap_min_m), 'first_surface_min_retained_fraction': float(args.first_surface_min_retained_fraction), 'first_surface_min_removed_fraction': float(args.first_surface_min_removed_fraction)}, 'summary': {'frame_count_requested': int(len(frame_ids)), 'surface_frame_count': int(len(frame_surface_rows)), 'factor_row_count': int(len(factor_rows)), 'state_counts': dict(state_counts), 'valid_depth_pixels': visible_track_numeric_summary(np.asarray([r.get('valid_depth_pixels', 0) for r in active_rows], dtype=float)), 'sample_count': visible_track_numeric_summary(np.asarray([r.get('sample_count', 0) for r in active_rows], dtype=float)), 'mask_area_px_raw': visible_track_numeric_summary(np.asarray([r.get('mask_area_px_raw', 0) for r in active_rows], dtype=float))}, 'surface_rows': frame_surface_rows, 'factor_rows': factor_rows}
    out = case_root / 'v18_visible_surface_track_factor_report.json'
    visible_track_write_json(out, report)
    print(json.dumps({'status': 'ok', 'report': str(out), 'summary': report['summary']}, indent=2)[:6000])
    return report

def visible_track_parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--target-entity-id', required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--visible-mask-report', type=Path, required=True)
    p.add_argument('--visible-ownership-factor-report', type=Path, default=None, help='Optional visible ownership factor report. When a frame/side has adjusted_entity_mask_path, build side-specific visible surface samples from object-owned pixels only.')
    p.add_argument('--depth-npz', type=Path, action='append', required=True)
    p.add_argument('--frame-span', nargs=2, type=int, action='append', required=True)
    p.add_argument('--sides', nargs='+', choices=('left', 'right'), default=['left', 'right'])
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--sample-stride', type=int, default=6)
    p.add_argument('--max-samples-per-frame', type=int, default=6000)
    p.add_argument('--depth-order-margin-m', type=float, default=0.01)
    p.add_argument('--coherent-first-surface-depth-filter', action=argparse.BooleanOptionalAction, default=False, help='When a model mask contains a disconnected far-depth tail, replace its solver mask with the nearest coherent first-surface component before emitting visible-surface factors.')
    p.add_argument('--first-surface-depth-gap-min-m', type=float, default=0.05, help='Minimum metric depth discontinuity required before pruning far-depth mask pixels as non-first-surface outliers.')
    p.add_argument('--first-surface-min-retained-fraction', type=float, default=0.5, help='Do not apply first-surface pruning if the nearest component would retain less than this fraction of valid mask depth pixels.')
    p.add_argument('--first-surface-min-removed-fraction', type=float, default=0.01, help='Do not write a filtered mask unless the far-depth component removes at least this fraction of valid mask depth pixels.')
    p.add_argument('--quarantine-hidden-volume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--review-frames', nargs='*', type=int, default=[])
    return p.parse_args(argv)
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from utils.data import as_list as face_eligibility_as_list
from utils.geometry import frame_camera_pose as face_eligibility_frame_camera_pose
from utils.data import load_json as face_eligibility_load_json
from utils.data import load_mesh
from utils.data import numeric_summary as face_eligibility_numeric_summary
from utils.data import pose_map
from utils.data import write_json as face_eligibility_write_json
STATE_OBSERVED = 'observed_depth_supported'
STATE_HAND_QUARANTINED = 'hand_owned_depth_quarantined'
STATE_FREE = 'free_space_rejected'
STATE_HIDDEN = 'hidden_unvalidated'
STATE_OUTSIDE = 'outside_view'
STATE_UNRESOLVED = 'unresolved'

def face_eligibility_parse_spans(values: list[list[int]] | None) -> list[tuple[int, int]]:
    if not values:
        raise RuntimeError('at least one --frame-span START END is required')
    spans: list[tuple[int, int]] = []
    for pair in values:
        if len(pair) != 2:
            raise RuntimeError(f'invalid frame span: {pair}')
        (a, b) = (int(pair[0]), int(pair[1]))
        if b < a:
            raise RuntimeError(f'invalid decreasing span {a}:{b}')
        spans.append((a, b))
    return spans

def face_eligibility_frame_ids_from_spans(spans: list[tuple[int, int]]) -> list[int]:
    ids: set[int] = set()
    for (a, b) in spans:
        ids.update(range(a, b + 1))
    return sorted(ids)

def face_eligibility_load_frames(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(frame['frame_idx']): frame for frame in face_eligibility_as_list(annotations.get('frames')) if isinstance(frame, dict) and frame.get('frame_idx') is not None}

def face_support_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[np.asarray(faces, dtype=np.int64)]
    return np.stack([tri[:, 0], tri[:, 1], tri[:, 2], 0.5 * (tri[:, 0] + tri[:, 1]), 0.5 * (tri[:, 1] + tri[:, 2]), 0.5 * (tri[:, 2] + tri[:, 0]), tri.mean(axis=1)], axis=1)

def classify_faces_against_depth(*, frame: dict[str, Any], vertices_object: np.ndarray, faces: np.ndarray, pose: tuple[np.ndarray, np.ndarray], depth_row: dict[str, Any] | None, support_margin_m: float, free_space_margin_m: float, min_supported_samples: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_faces = int(len(faces))
    if depth_row is None:
        masks = {STATE_OBSERVED: np.zeros(n_faces, dtype=bool), STATE_FREE: np.zeros(n_faces, dtype=bool), STATE_HIDDEN: np.zeros(n_faces, dtype=bool), STATE_OUTSIDE: np.ones(n_faces, dtype=bool), STATE_UNRESOLVED: np.zeros(n_faces, dtype=bool)}
        return (masks, {'state': 'missing_depth', 'frame_idx': int(frame['frame_idx']), 'face_count': n_faces})
    (r_obj, t_obj) = pose
    samples_obj = face_support_samples(vertices_object, faces)
    flat_obj = samples_obj.reshape(-1, 3)
    flat_world = flat_obj @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    (r_c2w, t_c2w) = face_eligibility_frame_camera_pose(frame)
    flat_cam = (flat_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row['depth'], dtype=np.float32)
    (height, width) = depth.shape
    (u, v, valid) = project_points(flat_cam, np.asarray(depth_row['intrinsics'], dtype=float), width, height)
    z_obs = np.full((flat_cam.shape[0],), np.nan, dtype=float)
    if np.any(valid):
        z = depth[v[valid], u[valid]].astype(float)
        finite = np.isfinite(z) & (z > 1e-05)
        valid_ids = np.where(valid)[0]
        z_obs[valid_ids[finite]] = z[finite]
    finite = np.isfinite(z_obs)
    residual = flat_cam[:, 2].astype(float) - z_obs
    finite_2d = finite.reshape(n_faces, -1)
    residual_2d = residual.reshape(n_faces, -1)
    supported = finite_2d & (np.abs(residual_2d) <= float(support_margin_m))
    free = finite_2d & (residual_2d < -float(free_space_margin_m))
    behind = finite_2d & (residual_2d > float(support_margin_m))
    supported_count = np.count_nonzero(supported, axis=1)
    free_count = np.count_nonzero(free, axis=1)
    behind_count = np.count_nonzero(behind, axis=1)
    finite_count = np.count_nonzero(finite_2d, axis=1)
    free_mask = free_count > 0
    observed_mask = (supported_count >= int(min_supported_samples)) & ~free_mask
    outside_mask = finite_count == 0
    hidden_mask = (finite_count > 0) & (behind_count >= np.maximum(1, finite_count - supported_count)) & ~observed_mask & ~free_mask
    unresolved_mask = ~(observed_mask | free_mask | outside_mask | hidden_mask)
    masks = {STATE_OBSERVED: observed_mask.astype(bool), STATE_FREE: free_mask.astype(bool), STATE_HIDDEN: hidden_mask.astype(bool), STATE_OUTSIDE: outside_mask.astype(bool), STATE_UNRESOLVED: unresolved_mask.astype(bool)}
    summary = {'state': 'face_support_depth_classified', 'frame_idx': int(frame['frame_idx']), 'depth_source': str(depth_row.get('source')), 'face_count': n_faces, 'sample_count': int(flat_cam.shape[0]), 'finite_sample_count': int(np.count_nonzero(finite)), 'face_state_counts': {key: int(np.count_nonzero(mask)) for (key, mask) in masks.items()}, 'sample_depth_residual_m': face_eligibility_numeric_summary(residual[finite]), 'supported_sample_count': face_eligibility_numeric_summary(supported_count.astype(float)), 'free_space_sample_count': face_eligibility_numeric_summary(free_count.astype(float)), 'behind_sample_count': face_eligibility_numeric_summary(behind_count.astype(float))}
    return (masks, summary)

def nested_get(row: dict[str, Any], dotted: str) -> Any:
    cur: Any = row
    for key in dotted.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur

def load_pose_support_uncertainty(pose_report: dict[str, Any], *, stat: str, default_m: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in face_eligibility_as_list(pose_report.get('pose_rows')) if isinstance(pose_report, dict) else []:
        if not isinstance(row, dict) or row.get('frame_idx') is None:
            continue
        raw = nested_get(row, stat)
        if raw is None:
            raw = default_m
        try:
            val = float(raw)
        except Exception:
            val = float(default_m)
        out[int(row['frame_idx'])] = max(0.0, val)
    return out

def load_visible_ownership_rows(path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = face_eligibility_load_json(path)
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in face_eligibility_as_list(payload.get('ownership_rows')) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get('frame_idx') is not None and (row.get('hand_side') is not None):
            rows[int(row['frame_idx']), str(row['hand_side'])] = row
    return rows

def load_bool_mask(path: str | None) -> np.ndarray | None:
    if not isinstance(path, str) or not path:
        return None
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f'explicit ownership mask path is missing: {mask_path}')
    import cv2
    arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f'explicit ownership mask path could not be decoded: {mask_path}')
    return arr > 0

def mask_membership(mask: np.ndarray, uv: np.ndarray, dilation_px: int) -> np.ndarray:
    if mask is None or uv.ndim != 2 or uv.shape[1] != 2:
        return np.zeros((len(uv),), dtype=bool)
    (h, w) = mask.shape
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside = np.zeros((len(uv),), dtype=bool)
    if not np.any(valid):
        return inside
    radius = max(0, int(dilation_px))
    x0 = np.clip(u[valid] - radius, 0, w - 1)
    x1 = np.clip(u[valid] + radius, 0, w - 1)
    y0 = np.clip(v[valid] - radius, 0, h - 1)
    y1 = np.clip(v[valid] + radius, 0, h - 1)
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    area = integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
    inside[np.where(valid)[0]] = area > 0
    return inside

def ownership_face_quarantine(*, frame: dict[str, Any], vertices_object: np.ndarray, faces: np.ndarray, pose: tuple[np.ndarray, np.ndarray], depth_row: dict[str, Any] | None, ownership_row: dict[str, Any] | None, dilation_px: int) -> np.ndarray:
    mask = load_bool_mask(None if ownership_row is None else ownership_row.get('non_object_owned_mask_path'))
    if mask is None or depth_row is None:
        return np.zeros((len(faces),), dtype=bool)
    (r_obj, t_obj) = pose
    samples_obj = face_support_samples(vertices_object, faces)
    flat_world = samples_obj.reshape(-1, 3) @ np.asarray(r_obj, dtype=float).T + np.asarray(t_obj, dtype=float)[None, :]
    (r_c2w, t_c2w) = face_eligibility_frame_camera_pose(frame)
    flat_cam = (flat_world - t_c2w[None, :]) @ r_c2w
    depth = np.asarray(depth_row['depth'], dtype=np.float32)
    (height, width) = depth.shape
    (u, v, valid) = project_points(flat_cam, np.asarray(depth_row['intrinsics'], dtype=float), width, height)
    uv = np.column_stack([u.astype(float), v.astype(float)])
    inside = mask_membership(mask, uv, int(dilation_px)) & valid
    if inside.shape[0] != flat_cam.shape[0]:
        return np.zeros((len(faces),), dtype=bool)
    return inside.reshape(len(faces), -1).any(axis=1)

def write_face_masks(path: Path, masks: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: mask.astype(np.bool_) for (key, mask) in masks.items()})

def build_face_eligibility(args: argparse.Namespace) -> dict[str, Any]:
    spans = face_eligibility_parse_spans(args.frame_span)
    frame_ids = face_eligibility_frame_ids_from_spans(spans)
    annotations = face_eligibility_load_json(args.annotations)
    frames = face_eligibility_load_frames(annotations)
    pose_report = face_eligibility_load_json(args.pose_report)
    poses = pose_map(pose_report)
    support_uncertainty_by_frame = load_pose_support_uncertainty(pose_report, stat=str(args.surface_support_uncertainty_stat), default_m=float(args.default_surface_support_uncertainty_m))
    mesh = load_mesh(args.completed_mesh)
    vertices_object = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    depth_rows = load_depth_sources(list(args.depth_npz))
    ownership_rows = load_visible_ownership_rows(args.visible_ownership_factor_report)
    output_case = args.output_root / args.case
    output_case.mkdir(parents=True, exist_ok=True)
    factor_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    state_counter: Counter[str] = Counter()
    for frame_idx in frame_ids:
        frame = frames.get(frame_idx)
        pose = poses.get(frame_idx)
        if frame is None or pose is None:
            state_counter['missing_frame_or_pose'] += 1
            continue
        (base_masks, summary) = classify_faces_against_depth(frame=frame, vertices_object=vertices_object, faces=faces, pose=pose, depth_row=depth_rows.get(frame_idx), support_margin_m=float(args.support_margin_m), free_space_margin_m=float(args.free_space_margin_m), min_supported_samples=int(args.min_supported_samples))
        frame_summaries.append(summary)
        surface_support_uncertainty_m = float(support_uncertainty_by_frame.get(frame_idx, float(args.default_surface_support_uncertainty_m)))
        for side in args.sides:
            hand_owned = ownership_face_quarantine(frame=frame, vertices_object=vertices_object, faces=faces, pose=pose, depth_row=depth_rows.get(frame_idx), ownership_row=ownership_rows.get((frame_idx, side)), dilation_px=int(args.ownership_face_overlap_dilation_px))
            masks = {key: value.copy() for (key, value) in base_masks.items()}
            if np.any(hand_owned):
                masks[STATE_HAND_QUARANTINED] = hand_owned
                masks[STATE_OBSERVED] &= ~hand_owned
            else:
                masks[STATE_HAND_QUARANTINED] = hand_owned
            eligible_hard = masks[STATE_OBSERVED].copy()
            mask_path = output_case / 'face_state_masks' / side / f'{frame_idx:06d}_surface_eligibility_faces.npz'
            write_face_masks(mask_path, {**masks, 'eligible_hard_observed': eligible_hard})
            counts = {key: int(np.count_nonzero(value)) for (key, value) in masks.items()}
            counts['eligible_hard_observed'] = int(np.count_nonzero(eligible_hard))
            factor_rows.append({'factor_family': 'surface_eligibility', 'target_entity_id': args.object_id, 'frame_idx': int(frame_idx), 'hand_side': side, 'variable_affected': 'constraint_eligibility', 'observation_type': 'posed_mesh_face_support_samples_x_metric_depth', 'residual_or_quarantine_rule': 'only eligible_hard_observed faces may produce MANO nonpenetration; observed-surface residuals are hard only beyond observed_surface_support_uncertainty_m, while free-space, hidden, outside-view, unresolved, and hand-owned faces are not hard constraints', 'provenance': {'annotations': str(args.annotations), 'pose_report': str(args.pose_report), 'completed_mesh': str(args.completed_mesh), 'depth_npz': [str(p) for p in args.depth_npz], 'visible_ownership_factor_report': None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report)}, 'rendered_uncertainty_channel': 'surface eligibility should render eligible observed faces separately from free-space rejected, hidden, outside-view, unresolved, hand-owned, and support-uncertain faces', 'face_state_npz_path': str(mask_path), 'observed_surface_support_uncertainty_m': float(surface_support_uncertainty_m), 'surface_support_uncertainty_m': float(surface_support_uncertainty_m), 'surface_support_uncertainty_stat': str(args.surface_support_uncertainty_stat), 'face_count': int(len(faces)), 'counts': counts})
            for (key, value) in counts.items():
                state_counter[f'{side}:{key}'] += int(value)
    report = {'method': 'build_v18_surface_eligibility_factor', 'status': 'ok', 'claim_scope': 'Reusable face/patch eligibility factor records; not a MANO correction by themselves.', 'case': args.case, 'target_entity_id': args.object_id, 'inputs': {'annotations': str(args.annotations), 'pose_report': str(args.pose_report), 'completed_mesh': str(args.completed_mesh), 'depth_npz': [str(p) for p in args.depth_npz], 'visible_ownership_factor_report': None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report)}, 'parameters': {'frame_spans': [[int(a), int(b)] for (a, b) in spans], 'sides': list(args.sides), 'support_margin_m': float(args.support_margin_m), 'free_space_margin_m': float(args.free_space_margin_m), 'min_supported_samples': int(args.min_supported_samples), 'ownership_face_overlap_dilation_px': int(args.ownership_face_overlap_dilation_px), 'surface_support_uncertainty_stat': str(args.surface_support_uncertainty_stat), 'default_surface_support_uncertainty_m': float(args.default_surface_support_uncertainty_m)}, 'summary': {'frame_count_requested': int(len(frame_ids)), 'factor_row_count': int(len(factor_rows)), 'state_counts': dict(state_counter), 'frame_face_state_counts': {key: face_eligibility_numeric_summary(np.asarray([s.get('face_state_counts', {}).get(key, 0) for s in frame_summaries], dtype=float)) for key in [STATE_OBSERVED, STATE_FREE, STATE_HIDDEN, STATE_OUTSIDE, STATE_UNRESOLVED]}, 'observed_surface_support_uncertainty_m': face_eligibility_numeric_summary(np.asarray([float(r.get('observed_surface_support_uncertainty_m', 0.0)) for r in factor_rows], dtype=float))}, 'frame_summaries': frame_summaries, 'factor_rows': factor_rows}
    report_path = output_case / 'v18_surface_eligibility_factor_report.json'
    face_eligibility_write_json(report_path, report)
    print(json.dumps({'status': 'ok', 'report': str(report_path), 'factor_rows': len(factor_rows)}, indent=2))
    return report

def face_eligibility_parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--object-id', required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--pose-report', type=Path, required=True)
    p.add_argument('--completed-mesh', type=Path, required=True)
    p.add_argument('--depth-npz', type=Path, action='append', required=True)
    p.add_argument('--frame-span', nargs=2, type=int, action='append', required=True)
    p.add_argument('--sides', nargs='+', choices=('left', 'right'), default=['left', 'right'])
    p.add_argument('--visible-ownership-factor-report', type=Path, default=None)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--support-margin-m', type=float, default=0.015)
    p.add_argument('--free-space-margin-m', type=float, default=0.025)
    p.add_argument('--min-supported-samples', type=int, default=2)
    p.add_argument('--ownership-face-overlap-dilation-px', type=int, default=2)
    p.add_argument('--surface-support-uncertainty-stat', default='observed_to_mesh_final.p95_m', help='Dotted pose_rows[] field emitted as observed_surface_support_uncertainty_m for solver nonpenetration slack.')
    p.add_argument('--default-surface-support-uncertainty-m', type=float, default=0.0)
    return p.parse_args(argv)
import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.spatial import cKDTree
POSE_MEASUREMENT_STATUSES = {'fit_to_visible_depth_samples', 'fit_to_visible_depth_archive_vertices', 'fit_to_object_owned_rgb_calibrated_pnp'}

@dataclass
class FactorBuildFrame:
    frame_idx: int
    T_world_camera: np.ndarray
    rotation_world_object: np.ndarray
    translation_world_object: np.ndarray
    observed_camera: np.ndarray
    observed_uv: np.ndarray
    observed_depth_confidence: np.ndarray
    ownership_quality_weight: float
    image_source_size_wh: tuple[int, int]
    calibration_size_wh: tuple[int, int]
    K_calibration: np.ndarray
    K_image_source: np.ndarray
    mask_depth_transform_contract: dict[str, Any] | None
    mask_path: str
    depth_npz_path: str
    target_mask: np.ndarray
    hand_unknown: np.ndarray
    target_or_unknown: np.ndarray

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_mano_faces(path: Path) -> np.ndarray:
    resolved = path.expanduser().resolve()
    with resolved.open('rb') as handle:
        payload = pickle.load(handle, encoding='latin1')
    faces = np.asarray(payload.get('f') if isinstance(payload, dict) else [], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or (np.min(faces) < 0) or (np.max(faces) >= 778):
        raise RuntimeError(f'invalid MANO triangle topology: {resolved}')
    return faces

def image_targets_load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise RuntimeError(f'expected JSON object: {path}')
    return value

def image_targets_load_mesh(path: Path) -> trimesh.Trimesh:
    import trimesh
    mesh = trimesh.load(path.expanduser().resolve(), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f'invalid triangle mesh: {path}')
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError(f'non-finite mesh vertices: {path}')
    return mesh

def world_to_camera(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]

def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]

def canonical_to_camera(canonical: np.ndarray, rotation_world_object: np.ndarray, translation_world_object: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    world = np.asarray(canonical) @ rotation_world_object.T + translation_world_object[None, :]
    return world_to_camera(world, T_world_camera)

def resize_intrinsics_xy(K_source: np.ndarray, source_size_wh: tuple[int, int], target_size_wh: tuple[int, int]) -> np.ndarray:
    (source_width, source_height) = (int(source_size_wh[0]), int(source_size_wh[1]))
    (target_width, target_height) = (int(target_size_wh[0]), int(target_size_wh[1]))
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or (target_height <= 0):
        raise RuntimeError(f'invalid image-plane sizes: source={source_size_wh}, target={target_size_wh}')
    sx = float(target_width) / float(source_width)
    sy = float(target_height) / float(source_height)
    K = np.asarray(K_source, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
    K[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return K

def resize_intrinsics(K_source: np.ndarray, source_size: int, raster_size: int) -> np.ndarray:
    return resize_intrinsics_xy(K_source, (int(source_size), int(source_size)), (int(raster_size), int(raster_size)))

def resize_mask(mask: np.ndarray, raster_size: int) -> np.ndarray:
    interpolation = getattr(cv2, 'INTER_NEAREST_EXACT', cv2.INTER_NEAREST)
    return cv2.resize(mask.astype(np.uint8), (raster_size, raster_size), interpolation=interpolation) > 0

def project_camera(points_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    z = np.maximum(points[:, 2], 1e-09)
    return np.column_stack((K[0, 0] * points[:, 0] / z + K[0, 2], K[1, 1] * points[:, 1] / z + K[1, 2]))

def deterministic_indices(count: int, max_count: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, num=max_count)).astype(np.int64))

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

class BatchFirstHitRasterizer:

    def __init__(self, faces: np.ndarray, raster_size: int, K_raster: np.ndarray, device: str):
        import torch
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
        self.device = device
        self.faces = torch.tensor(faces, dtype=torch.int64, device=device)
        self.size = int(raster_size)
        cameras = cameras_from_opencv_projection(torch.eye(3, dtype=torch.float32, device=device)[None], torch.zeros(1, 3, dtype=torch.float32, device=device), torch.tensor(K_raster, dtype=torch.float32, device=device)[None], torch.tensor([[self.size, self.size]], dtype=torch.float32, device=device))
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=self.size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False, bin_size=0))

    def render(self, vertices_camera: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from pytorch3d.structures import Meshes
        verts = [torch.tensor(v, dtype=torch.float32, device=self.device) for v in vertices_camera]
        with torch.no_grad():
            fragments = self.rasterizer(Meshes(verts=verts, faces=[self.faces] * len(verts)))
        zbuf = fragments.zbuf[..., 0].detach().cpu().numpy().astype(np.float64)
        pix = fragments.pix_to_face[..., 0].detach().cpu().numpy()
        valid = (pix >= 0) & np.isfinite(zbuf) & (zbuf > 1e-06)
        return (np.where(valid, zbuf, np.nan), valid)

def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get('objects', []) if isinstance(frame.get('objects'), list) else []:
        if obj.get('object_id') == object_id:
            return obj
    legacy = frame.get('object') if isinstance(frame.get('object'), dict) else None
    if legacy and legacy.get('object_id') in (None, object_id):
        return legacy
    return None

def build_hand_unknown_mask(hand_npz: Any, hand_rows_by_frame: dict[int, list[int]], mano_faces: np.ndarray, frame_idx: int, T_world_camera: np.ndarray, K_raster: np.ndarray, raster_size: int, dilation_px: int) -> np.ndarray:
    """Rasterize the complete projected MANO triangle silhouette as unknown support.

    A convex hull of MANO vertices can mark large background wedges as hand.  The
    factor contract needs the actual triangle silhouette: pixels covered by the
    projected hand are unknown, while pixels outside it may be used as known
    background for the object silhouette term.
    """
    unknown = np.zeros((raster_size, raster_size), dtype=np.uint8)
    faces = np.asarray(mano_faces, dtype=np.int64)
    for row_pos in hand_rows_by_frame.get(int(frame_idx), []):
        vertices_world = np.asarray(hand_npz['vertices_current_v18_world_from_hawor_projection_relift_m'][row_pos], dtype=np.float64)
        vertices_camera = world_to_camera(vertices_world, T_world_camera)
        valid_vertex = np.isfinite(vertices_camera).all(axis=1) & (vertices_camera[:, 2] > 1e-06)
        valid_faces = valid_vertex[faces].all(axis=1)
        if not np.any(valid_faces):
            continue
        face_vertices = vertices_camera[faces[valid_faces]]
        projected = np.empty((len(face_vertices), 3, 2), dtype=np.float64)
        z = face_vertices[:, :, 2]
        projected[:, :, 0] = K_raster[0, 0] * face_vertices[:, :, 0] / z + K_raster[0, 2]
        projected[:, :, 1] = K_raster[1, 1] * face_vertices[:, :, 1] / z + K_raster[1, 2]
        finite = np.isfinite(projected).all(axis=(1, 2))
        projected = projected[finite]
        if len(projected) == 0:
            continue
        projected = np.clip(projected, -2.0 * raster_size, 3.0 * raster_size)
        polygons = [np.rint(poly).astype(np.int32) for poly in projected]
        cv2.fillPoly(unknown, polygons, 1)
    if dilation_px > 0:
        radius = int(dilation_px)
        unknown = cv2.dilate(unknown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    return unknown > 0

def image_targets_load_frames(args: argparse.Namespace) -> tuple[list[FactorBuildFrame], np.ndarray]:
    annotations = image_targets_load_json(args.annotations)
    measurement_report = image_targets_load_json(args.pose_report)
    factor_pose_report = image_targets_load_json(args.factor_pose_report) if args.factor_pose_report else measurement_report
    measurement_rows = {int(row['frame_idx']): row for row in measurement_report.get('pose_rows', []) if isinstance(row, dict) and str(row.get('status') or '') in POSE_MEASUREMENT_STATUSES}
    factor_pose_rows = {int(row['frame_idx']): row for row in factor_pose_report.get('pose_rows', []) if isinstance(row, dict) and row.get('frame_idx') is not None}
    hand_npz = np.load(args.hand_npz)
    mano_faces = load_mano_faces(args.mano_faces_pkl)
    hand_rows_by_frame: dict[int, list[int]] = {}
    for (pos, frame_idx) in enumerate(np.asarray(hand_npz['frame_idx']).astype(int).tolist()):
        hand_rows_by_frame.setdefault(int(frame_idx), []).append(int(pos))
    depth_archives: dict[str, Any] = {}
    frames: list[FactorBuildFrame] = []
    K_raster: np.ndarray | None = None
    raster_size_wh = (int(args.raster_size), int(args.raster_size))
    for frame in annotations.get('frames', []):
        if not isinstance(frame, dict) or frame.get('frame_idx') is None:
            continue
        idx = int(frame['frame_idx'])
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        if idx not in measurement_rows or idx not in factor_pose_rows:
            continue
        obj = annotation_object(frame, args.object_id)
        if obj is None:
            continue
        visible = obj.get('visible_geometry_candidate') if isinstance(obj.get('visible_geometry_candidate'), dict) else None
        if not isinstance(visible, dict):
            continue
        ownership = visible.get('first_surface_depth_ownership')
        if not isinstance(ownership, dict) or ownership.get('enabled') is not True or ownership.get('fail_closed') is True:
            continue
        if float(ownership.get('retained_fraction') or 0.0) < float(args.min_ownership_fraction):
            continue
        mask_path = Path(str(visible.get('mask_path') or obj.get('mask_path') or '')).expanduser().resolve()
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None or mask_raw.ndim != 2:
            raise RuntimeError(f'failed to read object-owned mask: {mask_path}')
        (image_height, image_width) = (int(mask_raw.shape[0]), int(mask_raw.shape[1]))
        raw_path = Path(str(frame.get('raw_frame_path') or '')).expanduser().resolve()
        raw_gray = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        if raw_gray is None or raw_gray.ndim != 2:
            raise RuntimeError(f'failed to read raw RGB frame for image-plane contract: {raw_path}')
        if raw_gray.shape != mask_raw.shape:
            raise RuntimeError(f'raw RGB shape {raw_gray.shape} != object mask shape {mask_raw.shape} at frame {idx}')
        image_size_wh = (image_width, image_height)
        calibration_width = int(frame.get('source_width') or args.source_size)
        calibration_height = int(frame.get('source_height') or args.source_size)
        (fx, fy, cx, cy) = np.asarray(visible['intrinsics_fx_fy_cx_cy'], dtype=np.float64).tolist()
        K_calibration = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        K_image = resize_intrinsics_xy(K_calibration, (calibration_width, calibration_height), image_size_wh)
        K_current = resize_intrinsics_xy(K_image, image_size_wh, raster_size_wh)
        if K_raster is None:
            K_raster = K_current
        elif not np.allclose(K_current, K_raster, atol=1e-06):
            raise RuntimeError('per-frame intrinsics differ; factor batch contract is invalid')
        mask_depth_contract = visible.get('mask_depth_transform_contract')
        if isinstance(mask_depth_contract, dict):
            declared_mask_size = tuple((int(v) for v in mask_depth_contract.get('mask_size_wh', [])))
            declared_depth_size = tuple((int(v) for v in mask_depth_contract.get('depth_size_wh', [])))
            if declared_mask_size not in {image_size_wh, ()} or declared_depth_size not in {(calibration_width, calibration_height), ()}:
                raise RuntimeError(f'mask/depth plane contract size mismatch at frame {idx}: mask={declared_mask_size}, image={image_size_wh}, depth={declared_depth_size}, calibration={(calibration_width, calibration_height)}')
            declared_affine = np.asarray(mask_depth_contract.get('A_depth_from_mask_coordinate_model') or [], dtype=np.float64)
            expected_affine = np.asarray([[float(calibration_width) / float(image_width), 0.0, 0.5 * float(calibration_width) / float(image_width) - 0.5], [0.0, float(calibration_height) / float(image_height), 0.5 * float(calibration_height) / float(image_height) - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
            if declared_affine.shape == (3, 3) and (not np.allclose(declared_affine, expected_affine, atol=1e-06, rtol=0.0)):
                raise RuntimeError(f'mask/depth affine contract mismatch at frame {idx}')
        observed = np.asarray(visible.get('camera_vertices_sample_m') or [], dtype=np.float64)
        valid_observed = np.isfinite(observed).all(axis=1) & (observed[:, 2] > 1e-06)
        observed = observed[valid_observed]
        if len(observed) < int(args.min_observed_points):
            continue
        confidence_path = Path(str(visible.get('depth_npz') or '')).expanduser().resolve()
        if not confidence_path.is_file():
            raise RuntimeError(f'depth confidence archive missing for frame {idx}: {confidence_path}')
        confidence_key = str(confidence_path)
        if confidence_key not in depth_archives:
            depth_archives[confidence_key] = np.load(confidence_path, allow_pickle=False)
        depth_archive = depth_archives[confidence_key]
        confidence_frames = np.asarray(depth_archive['frame_idx'], dtype=np.int64)
        matching = np.flatnonzero(confidence_frames == idx)
        if len(matching) != 1:
            raise RuntimeError(f'depth confidence archive has no unique frame {idx}: {confidence_path}')
        depth_frame = int(matching[0])
        confidence_map = np.asarray(depth_archive['confidence'][depth_frame], dtype=np.float64)
        if confidence_map.shape != (calibration_height, calibration_width):
            raise RuntimeError(f'depth confidence shape {confidence_map.shape} != calibration plane {(calibration_height, calibration_width)} for frame {idx}')
        source_uv = project_camera(observed, K_calibration)
        source_px = np.rint(source_uv).astype(np.int64)
        source_px[:, 0] = np.clip(source_px[:, 0], 0, calibration_width - 1)
        source_px[:, 1] = np.clip(source_px[:, 1], 0, calibration_height - 1)
        finite_confidence = confidence_map[np.isfinite(confidence_map)]
        fallback_confidence = float(np.median(finite_confidence)) if len(finite_confidence) else 1.0
        observed_confidence = confidence_map[source_px[:, 1], source_px[:, 0]]
        observed_confidence = np.nan_to_num(observed_confidence, nan=fallback_confidence, posinf=float(np.max(finite_confidence)) if len(finite_confidence) else fallback_confidence, neginf=float(np.min(finite_confidence)) if len(finite_confidence) else fallback_confidence)
        uv = project_camera(observed, K_current)
        inside = (uv[:, 0] >= 0.0) & (uv[:, 0] < args.raster_size) & (uv[:, 1] >= 0.0) & (uv[:, 1] < args.raster_size)
        observed = observed[inside]
        uv = uv[inside]
        observed_confidence = observed_confidence[inside]
        if len(observed) < int(args.min_observed_points):
            continue
        target_mask = resize_mask(mask_raw > 0, int(args.raster_size))
        T_world_camera = validate_camera_transform(frame['camera']['T_world_camera_metric'], idx)
        hand_unknown = build_hand_unknown_mask(hand_npz, hand_rows_by_frame, mano_faces, idx, T_world_camera, K_current, int(args.raster_size), int(args.hand_unknown_dilation_px))
        pose_row = factor_pose_rows[idx]
        rotation = np.asarray(pose_row['rotation_world_from_completed_canonical_matrix'], dtype=np.float64)
        translation = np.asarray(pose_row['translation_world_m'], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or (not np.isfinite(rotation).all()) or (not np.isfinite(translation).all()):
            raise RuntimeError(f'invalid factor pose row for frame {idx}')
        frames.append(FactorBuildFrame(frame_idx=idx, T_world_camera=T_world_camera, rotation_world_object=rotation, translation_world_object=translation, observed_camera=observed, observed_uv=uv, observed_depth_confidence=observed_confidence, ownership_quality_weight=float(np.clip(1.0 - float(ownership.get('removed_fraction') or 0.0) / max(float(args.max_removed_fraction_for_full_weight), 1e-06), float(args.min_depth_factor_weight), 1.0)), image_source_size_wh=image_size_wh, calibration_size_wh=(calibration_width, calibration_height), K_calibration=K_calibration, K_image_source=K_image, mask_depth_transform_contract=mask_depth_contract if isinstance(mask_depth_contract, dict) else None, mask_path=str(mask_path), depth_npz_path=str(confidence_path), target_mask=target_mask, hand_unknown=hand_unknown, target_or_unknown=target_mask | hand_unknown))
    if K_raster is None or len(frames) < int(args.min_frames):
        raise RuntimeError(f'only {len(frames)} eligible factor frames, need {args.min_frames}')
    return (frames, K_raster)

def canonical_point_map(zbuf: np.ndarray, rendered: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray, rotation_world_object: np.ndarray, translation_world_object: np.ndarray) -> np.ndarray:
    (vv, uu) = np.nonzero(rendered)
    z = zbuf[vv, uu]
    points_camera = np.column_stack(((uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z))
    points_world = camera_to_world(points_camera, T_world_camera)
    points_canonical = (points_world - translation_world_object[None, :]) @ rotation_world_object
    out = np.full(zbuf.shape + (3,), np.nan, dtype=np.float64)
    out[vv, uu] = points_canonical
    return out

def build_frame_factors(frame: FactorBuildFrame, zbuf: np.ndarray, rendered: np.ndarray, K: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    point_map = canonical_point_map(zbuf, rendered, K, frame.T_world_camera, frame.rotation_world_object, frame.translation_world_object)
    observed_idx = deterministic_indices(len(frame.observed_camera), int(args.max_observed_factors_per_frame))
    observed = frame.observed_camera[observed_idx]
    observed_uv = frame.observed_uv[observed_idx]
    observed_confidence = frame.observed_depth_confidence[observed_idx]
    observed_px = np.rint(observed_uv).astype(np.int64)
    observed_px[:, 0] = np.clip(observed_px[:, 0], 0, args.raster_size - 1)
    observed_px[:, 1] = np.clip(observed_px[:, 1], 0, args.raster_size - 1)
    hit = rendered[observed_px[:, 1], observed_px[:, 0]]
    observed_hand_unknown = frame.hand_unknown[observed_px[:, 1], observed_px[:, 0]]
    depth_points = point_map[observed_px[hit, 1], observed_px[hit, 0]]
    depth_z = observed[hit, 2]
    depth_confidence = observed_confidence[hit]
    valid_depth = np.isfinite(depth_points).all(axis=1) & np.isfinite(depth_z) & (depth_z > 1e-06) & np.isfinite(depth_confidence)
    depth_points = depth_points[valid_depth]
    depth_z = depth_z[valid_depth]
    depth_confidence = depth_confidence[valid_depth]
    (conf_lo, conf_hi) = np.percentile(depth_confidence, [10.0, 90.0]) if len(depth_confidence) else (0.0, 1.0)
    conf_bad = np.clip((depth_confidence - conf_lo) / max(conf_hi - conf_lo, 1e-06), 0.0, 1.0)
    confidence_weight = 1.0 - (1.0 - float(args.min_depth_factor_weight)) * conf_bad
    boundary_distance = cv2.distanceTransform(frame.target_mask.astype(np.uint8), cv2.DIST_L2, 3)
    hit_px = observed_px[hit][valid_depth]
    boundary_weight = np.where(boundary_distance[hit_px[:, 1], hit_px[:, 0]] <= float(args.boundary_downweight_radius_px), float(args.boundary_weight), 1.0)
    depth_weights = np.clip(float(frame.ownership_quality_weight) * confidence_weight * boundary_weight, float(args.min_depth_factor_weight), 1.0)
    sil_points: list[np.ndarray] = []
    sil_targets: list[np.ndarray] = []
    sil_kinds: list[int] = []
    outside = rendered & ~frame.target_or_unknown
    outside_coords = np.column_stack(np.nonzero(outside)[::-1])
    if len(outside_coords) and np.any(frame.target_or_unknown):
        target_coords = np.column_stack(np.nonzero(frame.target_or_unknown)[::-1])
        selected = deterministic_indices(len(outside_coords), int(args.max_outside_silhouette_factors_per_frame))
        selected_coords = outside_coords[selected]
        (_, nearest) = cKDTree(target_coords).query(selected_coords, k=1, workers=-1)
        nearest_target = target_coords[nearest]
        points = point_map[selected_coords[:, 1], selected_coords[:, 0]]
        valid = np.isfinite(points).all(axis=1)
        sil_points.append(points[valid])
        sil_targets.append(nearest_target[valid].astype(np.float64))
        sil_kinds.append(np.zeros(int(np.count_nonzero(valid)), dtype=np.int8))
    missing_uv = observed_uv[~hit & ~observed_hand_unknown]
    if len(missing_uv) and np.any(rendered):
        rendered_coords = np.column_stack(np.nonzero(rendered)[::-1])
        selected = deterministic_indices(len(missing_uv), int(args.max_missing_silhouette_factors_per_frame))
        selected_uv = missing_uv[selected]
        (_, nearest) = cKDTree(rendered_coords).query(selected_uv, k=1, workers=-1)
        nearest_rendered = rendered_coords[nearest]
        points = point_map[nearest_rendered[:, 1], nearest_rendered[:, 0]]
        valid = np.isfinite(points).all(axis=1)
        sil_points.append(points[valid])
        sil_targets.append(selected_uv[valid].astype(np.float64))
        sil_kinds.append(np.ones(int(np.count_nonzero(valid)), dtype=np.int8))
    if sil_points:
        sil_point_arr = np.vstack(sil_points)
        sil_target_arr = np.vstack(sil_targets)
        sil_kind_arr = np.concatenate(sil_kinds)
    else:
        sil_point_arr = np.empty((0, 3), dtype=np.float64)
        sil_target_arr = np.empty((0, 2), dtype=np.float64)
        sil_kind_arr = np.empty((0,), dtype=np.int8)
    intersection = int(np.count_nonzero(rendered & frame.target_mask))
    union = int(np.count_nonzero(rendered | frame.target_mask))
    metrics = {'frame_idx': int(frame.frame_idx), 'observed_factor_count': int(len(depth_z)), 'depth_factor_weight_summary': {'median': float(np.median(depth_weights)) if len(depth_weights) else None, 'p10': float(np.percentile(depth_weights, 10.0)) if len(depth_weights) else None, 'p90': float(np.percentile(depth_weights, 90.0)) if len(depth_weights) else None, 'min': float(np.min(depth_weights)) if len(depth_weights) else None, 'max': float(np.max(depth_weights)) if len(depth_weights) else None}, 'ownership_quality_weight': float(frame.ownership_quality_weight), 'observed_depth_confidence_median': float(np.median(depth_confidence)) if len(depth_confidence) else None, 'missing_silhouette_factor_count': int(np.count_nonzero(sil_kind_arr == 1)), 'outside_silhouette_factor_count': int(np.count_nonzero(sil_kind_arr == 0)), 'rendered_pixels': int(np.count_nonzero(rendered)), 'target_pixels': int(np.count_nonzero(frame.target_mask)), 'hand_unknown_pixels': int(np.count_nonzero(frame.target_or_unknown & ~frame.target_mask)), 'initial_silhouette_iou': float(intersection / max(1, union)), 'initial_outside_pixels': int(np.count_nonzero(outside)), 'initial_observed_hit_fraction': float(np.mean(hit)) if len(hit) else 0.0, 'observed_hand_unknown_excluded_count': int(np.count_nonzero(~hit & observed_hand_unknown)), 'known_pixels_contract': 'hand-projected pixels are excluded from known-background factors'}
    return ({'depth_points': depth_points.astype(np.float64), 'depth_z': depth_z.astype(np.float64), 'depth_weight': depth_weights.astype(np.float64), 'silhouette_points': sil_point_arr.astype(np.float64), 'silhouette_target_uv': sil_target_arr.astype(np.float64), 'silhouette_kind': sil_kind_arr.astype(np.int8)}, metrics)

def image_targets_parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--pose-report', type=Path, required=True, help='Original P14 measurement report defining factor frames')
    p.add_argument('--factor-pose-report', type=Path, default=None, help='Optional current P15 pose report used only to freeze correspondences')
    p.add_argument('--completed-mesh', type=Path, required=True)
    p.add_argument('--hand-npz', type=Path, required=True)
    p.add_argument('--mano-faces-pkl', type=Path, required=True, help='MANO model pickle containing the 778-vertex triangle topology')
    p.add_argument('--object-id', required=True)
    p.add_argument('--output-npz', type=Path, required=True)
    p.add_argument('--output-json', type=Path, required=True)
    p.add_argument('--frame-start', type=int, default=None)
    p.add_argument('--frame-end', type=int, default=None)
    p.add_argument('--raster-size', type=int, default=256)
    p.add_argument('--source-size', type=int, default=1408)
    p.add_argument('--min-frames', type=int, default=8)
    p.add_argument('--min-observed-points', type=int, default=20)
    p.add_argument('--min-ownership-fraction', type=float, default=0.8)
    p.add_argument('--hand-unknown-dilation-px', type=int, default=1)
    p.add_argument('--boundary-downweight-radius-px', type=float, default=1.5)
    p.add_argument('--boundary-weight', type=float, default=0.65)
    p.add_argument('--min-depth-factor-weight', type=float, default=0.25)
    p.add_argument('--max-removed-fraction-for-full-weight', type=float, default=0.1)
    p.add_argument('--max-observed-factors-per-frame', type=int, default=192)
    p.add_argument('--max-outside-silhouette-factors-per-frame', type=int, default=128)
    p.add_argument('--max-missing-silhouette-factors-per-frame', type=int, default=128)
    p.add_argument('--render-batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(argv)
    if not 0.0 < float(args.boundary_weight) <= 1.0:
        raise RuntimeError('--boundary-weight must be in (0,1]')
    if not 0.0 < float(args.min_depth_factor_weight) <= 1.0:
        raise RuntimeError('--min-depth-factor-weight must be in (0,1]')
    if float(args.max_removed_fraction_for_full_weight) <= 0.0:
        raise RuntimeError('--max-removed-fraction-for-full-weight must be positive')
    return args

def build_pose_image_targets(argv=None) -> None:
    args = image_targets_parse_args(argv)
    mesh = image_targets_load_mesh(args.completed_mesh)
    if not args.mano_faces_pkl.is_file():
        raise RuntimeError(f'MANO topology pickle does not exist: {args.mano_faces_pkl}')
    (frames, K_raster) = image_targets_load_frames(args)
    rasterizer = BatchFirstHitRasterizer(np.asarray(mesh.faces, dtype=np.int64), int(args.raster_size), K_raster, str(args.device))
    frame_idx: list[int] = []
    T_world_camera_all: list[np.ndarray] = []
    depth_offsets = [0]
    silhouette_offsets = [0]
    depth_points_all: list[np.ndarray] = []
    depth_z_all: list[np.ndarray] = []
    depth_weight_all: list[np.ndarray] = []
    silhouette_points_all: list[np.ndarray] = []
    silhouette_target_all: list[np.ndarray] = []
    silhouette_kind_all: list[np.ndarray] = []
    per_frame_metrics: list[dict[str, Any]] = []
    for start in range(0, len(frames), int(args.render_batch_size)):
        chunk = frames[start:start + int(args.render_batch_size)]
        vertices_camera = [canonical_to_camera(np.asarray(mesh.vertices, dtype=np.float64), frame.rotation_world_object, frame.translation_world_object, frame.T_world_camera) for frame in chunk]
        (zbuf, rendered) = rasterizer.render(vertices_camera)
        for (local_i, frame) in enumerate(chunk):
            (factors, metrics) = build_frame_factors(frame, zbuf[local_i], rendered[local_i], K_raster, args)
            frame_idx.append(int(frame.frame_idx))
            T_world_camera_all.append(frame.T_world_camera)
            depth_points_all.append(factors['depth_points'])
            depth_z_all.append(factors['depth_z'])
            depth_weight_all.append(factors['depth_weight'])
            silhouette_points_all.append(factors['silhouette_points'])
            silhouette_target_all.append(factors['silhouette_target_uv'])
            silhouette_kind_all.append(factors['silhouette_kind'])
            depth_offsets.append(depth_offsets[-1] + len(factors['depth_points']))
            silhouette_offsets.append(silhouette_offsets[-1] + len(factors['silhouette_points']))
            per_frame_metrics.append(metrics)
        print(f'[factors] rendered {min(start + len(chunk), len(frames))}/{len(frames)} frames', flush=True)
    metadata = {'method': 'build_depth_order_p15_first_hit_silhouette_factors', 'contract_version': 2, 'factor_semantics': 'frozen local first-hit and silhouette correspondences for P15 per-frame SE(3) corrections', 'annotations': str(args.annotations.expanduser().resolve()), 'pose_report': str(args.pose_report.expanduser().resolve()), 'factor_pose_report': str((args.factor_pose_report or args.pose_report).expanduser().resolve()), 'completed_mesh': str(args.completed_mesh.expanduser().resolve()), 'hand_npz': str(args.hand_npz.expanduser().resolve()), 'mano_faces_pkl': str(args.mano_faces_pkl.expanduser().resolve()), 'input_sha256': {'annotations': sha256_file(args.annotations), 'pose_report': sha256_file(args.pose_report), 'factor_pose_report': sha256_file(args.factor_pose_report or args.pose_report), 'completed_mesh': sha256_file(args.completed_mesh), 'hand_npz': sha256_file(args.hand_npz), 'mano_faces_pkl': sha256_file(args.mano_faces_pkl)}, 'object_id': str(args.object_id), 'calibration_size_wh': list(frames[0].calibration_size_wh), 'image_source_size_wh': list(frames[0].image_source_size_wh), 'raster_size_wh': [int(args.raster_size), int(args.raster_size)], 'K_calibration': frames[0].K_calibration.astype(float).tolist(), 'K_image_source': frames[0].K_image_source.astype(float).tolist(), 'K_raster': np.asarray(K_raster, dtype=float).tolist(), 'mask_depth_transform_contract': frames[0].mask_depth_transform_contract, 'mask_sha256_by_frame': {str(frame.frame_idx): sha256_file(Path(frame.mask_path)) for frame in frames}, 'depth_npz_sha256_by_path': {frame.depth_npz_path: sha256_file(Path(frame.depth_npz_path)) for frame in frames}, 'raster_size': int(args.raster_size), 'hand_unknown_dilation_px': int(args.hand_unknown_dilation_px), 'hand_unknown_method': 'projected_mano_triangle_silhouette', 'missing_coverage_hand_unknown_excluded': True, 'boundary_downweight_radius_px': float(args.boundary_downweight_radius_px), 'boundary_weight': float(args.boundary_weight), 'min_depth_factor_weight': float(args.min_depth_factor_weight), 'max_removed_fraction_for_full_weight': float(args.max_removed_fraction_for_full_weight), 'max_observed_factors_per_frame': int(args.max_observed_factors_per_frame), 'max_outside_silhouette_factors_per_frame': int(args.max_outside_silhouette_factors_per_frame), 'max_missing_silhouette_factors_per_frame': int(args.max_missing_silhouette_factors_per_frame), 'generated_faces_consumed': False, 'collision_surface_consumed': False}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, metadata=np.asarray([json.dumps(metadata)]), frame_idx=np.asarray(frame_idx, dtype=np.int64), T_world_camera=np.asarray(T_world_camera_all, dtype=np.float64), K_raster=np.asarray(K_raster, dtype=np.float64), depth_offsets=np.asarray(depth_offsets, dtype=np.int64), depth_points_canonical=np.vstack(depth_points_all) if depth_points_all else np.empty((0, 3)), depth_observed_z=np.concatenate(depth_z_all) if depth_z_all else np.empty((0,)), depth_weight=np.concatenate(depth_weight_all) if depth_weight_all else np.empty((0,)), silhouette_offsets=np.asarray(silhouette_offsets, dtype=np.int64), silhouette_points_canonical=np.vstack(silhouette_points_all) if silhouette_points_all else np.empty((0, 3)), silhouette_target_uv=np.vstack(silhouette_target_all) if silhouette_target_all else np.empty((0, 2)), silhouette_kind=np.concatenate(silhouette_kind_all) if silhouette_kind_all else np.empty((0,)))
    summary = {**metadata, 'frame_count': int(len(frame_idx)), 'depth_factor_count': int(depth_offsets[-1]), 'depth_weight_summary': {'median': float(np.median(np.concatenate(depth_weight_all))) if depth_weight_all else None, 'p10': float(np.percentile(np.concatenate(depth_weight_all), 10.0)) if depth_weight_all else None, 'p90': float(np.percentile(np.concatenate(depth_weight_all), 90.0)) if depth_weight_all else None, 'min': float(np.min(np.concatenate(depth_weight_all))) if depth_weight_all else None, 'max': float(np.max(np.concatenate(depth_weight_all))) if depth_weight_all else None}, 'silhouette_factor_count': int(silhouette_offsets[-1]), 'outside_silhouette_factor_count': int(np.sum([m['outside_silhouette_factor_count'] for m in per_frame_metrics])), 'missing_silhouette_factor_count': int(np.sum([m['missing_silhouette_factor_count'] for m in per_frame_metrics])), 'initial_silhouette_iou_median': float(np.median([m['initial_silhouette_iou'] for m in per_frame_metrics])), 'initial_observed_hit_fraction_median': float(np.median([m['initial_observed_hit_fraction'] for m in per_frame_metrics])), 'frames': per_frame_metrics, 'outputs': {'factor_npz': str(args.output_npz), 'factor_report': str(args.output_json)}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['frame_count', 'depth_factor_count', 'silhouette_factor_count', 'initial_silhouette_iou_median', 'initial_observed_hit_fraction_median', 'outputs']}, indent=2))


def build_rigidity_motion_evidence(run_root, object_id, plan_path, raw_manifest, track_path, owned_track_path, output_dir):
    """Contiguous raw/object-owned crop pairs for the selected tracked whole.

    Crops have one fixed image-plane window per ten-frame sequence. We do not
    align, warp or infer component motion from silhouettes; the judge sees RGB.
    """
    from utils.state import require_run_artifact, in_run, sha
    root=Path(run_root).resolve();output=Path(output_dir)
    manifest=read_json(raw_manifest);frames=manifest['frames'];ids=[int(row['frame_idx']) for row in frames]
    if ids != list(range(len(ids))):raise ValueError('rigidity evidence requires contiguous complete raw timeline')
    receipts={}
    for path,role,selected in [(raw_manifest,'raw_manifest',None),(plan_path,'judgment',None),
            (track_path,'object_masks',object_id),(owned_track_path,'owned_masks',object_id)]:
        receipts[str(path)]=require_run_artifact(path,root,role,object_id=selected,frame_ids=ids)
    objects=[obj for obj in read_json(plan_path)['objects'] if obj['object_id']==object_id]
    if len(objects)!=1:raise ValueError('rigidity requires exactly the selected object plan')
    track=read_json(track_path);owned=read_json(owned_track_path)
    output.mkdir(parents=True,exist_ok=False)
    batches=[];images=[]
    for start in range(0,len(frames),10):
        chunk=frames[start:start+10];samples=[];boxes=[]
        for row in chunk:
            frame=int(row['frame_idx']);path=in_run(row['raw_frame_path'],root)
            if sha(path)!=row['raw_frame_sha256']:raise ValueError('rigidity RGB differs from decoded source receipt')
            rgb=cv2.imread(str(path))
            if rgb is None:raise ValueError('rigidity raw frame is undecodable')
            masks=[];mask_sources=[]
            for label,source in [('selected_track',track),('object_owned',owned)]:
                mask_name=source.get(str(frame),{}).get('mask_path')
                if mask_name:
                    mask_path=in_run(mask_name,root);mask=cv2.imread(str(mask_path),0)
                    if mask is None:raise ValueError('rigidity declared mask is undecodable')
                    mask_sources.append({'role':label,'path':str(mask_path),'sha256':sha(mask_path),'size_wh':[mask.shape[1],mask.shape[0]]})
                    mask=cv2.resize(mask,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                else:
                    mask=np.zeros(rgb.shape[:2],dtype=bool)
                    mask_sources.append({'role':label,'status':'no_visible_mask'})
                masks.append(mask)
            yy,xx=np.nonzero(masks[0]|masks[1])
            if len(xx):boxes.append((int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1))
            samples.append((row,rgb,masks,mask_sources))
        height,width=samples[0][1].shape[:2]
        if boxes:
            x0,y0=min(b[0] for b in boxes),min(b[1] for b in boxes)
            x1,y1=max(b[2] for b in boxes),max(b[3] for b in boxes)
            pad=max(24,int(.2*max(x1-x0,y1-y0)))
            x0,y0,x1,y1=max(0,x0-pad),max(0,y0-pad),min(width,x1+pad),min(height,y1+pad)
        else:x0,y0,x1,y1=0,0,width,height
        sheet=np.full((2*552,5*272,3),245,dtype=np.uint8);evidence_rows=[]
        for index,(row,rgb,masks,mask_sources) in enumerate(samples):
            outlined=rgb.copy()
            for mask,color in zip(masks,[(255,160,0),(0,255,0)]):
                contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(outlined,contours,-1,color,2)
            left,top=(index%5)*272,(index//5)*552
            for offset,image,label in [(0,rgb,'RAW'),(272,outlined,'TRACK blue / OWNED green')]:
                crop=image[y0:y1,x0:x1];scale=min(256/crop.shape[1],244/crop.shape[0])
                tile=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
                sheet[top+offset+24:top+offset+24+tile.shape[0],left:left+tile.shape[1]]=tile
                cv2.putText(sheet,f"{int(row['frame_idx'])} {label}",(left+2,top+offset+17),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
            evidence_rows.append({'frame_idx':int(row['frame_idx']),'raw_frame_sha256':row['raw_frame_sha256'],
                'mask_sources':mask_sources})
        path=output/f'motion_{start:06d}_{int(chunk[-1]["frame_idx"]):06d}.jpg'
        if not cv2.imwrite(str(path),sheet):raise RuntimeError('rigidity motion sheet write failed')
        images.append(path);batches.append({'frame_ids':[int(row['frame_idx']) for row in chunk],
            'crop_xyxy_manifest_pixels':[x0,y0,x1,y1],'sheet':str(path),'sha256':sha(path),'frames':evidence_rows})
    evidence={'schema':'tracked_whole_rigidity_motion_evidence_v1','object_id':object_id,'frame_ids':ids,
        'selected_object':objects[0],'source_run_id':read_json(root/'input/runtime_input_contract.json')['run_id'],
        'input_video_sha256':manifest['input_video_sha256'],'producer_receipts':receipts,
        'image_layout':'Each ten-consecutive-frame sheet is read left-to-right, then next row block. Each frame has raw RGB above, track/owned contours below. One fixed crop window and scale per sheet; no motion alignment.',
        'mask_semantics':'Masks identify the selected tracked whole and its visible first surface. They are not evidence that attached components moved relative to the body.',
        'coverage':'all raw frames, without temporal subsampling','motion_sequences':batches}
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
    return evidence,images
