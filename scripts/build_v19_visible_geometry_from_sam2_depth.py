#!/usr/bin/env python3
from __future__ import annotations

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

try:
    import open3d as o3d
except ImportError:  # Contract resolution/backprojection helpers remain CPU-testable without Open3D.
    o3d = None

from v19_camera_contract import SCHEMA as CAMERA_CONTRACT_V2_SCHEMA
from v19_camera_contract import load_contract as load_camera_contract_v2
from v19_camera_contract import plane_intrinsics as camera_contract_plane_intrinsics
from v19_camera_contract import resize_affine as camera_contract_resize_affine
from v19_camera_contract import sha256_file as camera_contract_sha256_file
from v19_camera_contract import summarize_contract as summarize_camera_contract_v2


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "object"


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def export_anchor_visible_surface_mesh(
    *,
    output_dir: Path,
    object_id: str,
    anchor_frame: int,
    anchor_centroid_world_m: np.ndarray,
    anchor_points_world_m: np.ndarray,
    min_voxel_m: float,
    voxel_divisor: float,
    poisson_depth: int,
    poisson_density_quantile: float,
) -> dict[str, Any]:
    """Export the selected anchor's visible surfels in object-canonical coordinates.

    The downstream rigid completion/pose components assume the completed mesh is
    in an object-canonical frame and per-frame pose rows map that canonical mesh
    into world coordinates.  Therefore the anchor point cloud is centered at the
    anchor-frame visible centroid rather than written in world coordinates.
    """
    points_world = np.asarray(anchor_points_world_m, dtype=np.float64)
    if o3d is None:
        raise RuntimeError("Open3D is required to export anchor visible-surface artifacts")
    centroid = np.asarray(anchor_centroid_world_m, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) < 30:
        return {
            "status": "too_few_anchor_visible_points_for_mesh_export",
            "anchor_frame_idx": int(anchor_frame),
            "point_count_input": int(len(points_world)) if points_world.ndim == 2 else 0,
            "blockers": ["anchor visible surfels missing or below 30 points"],
        }
    if centroid.shape != (3,) or not np.isfinite(centroid).all() or not np.isfinite(points_world).all():
        raise RuntimeError("anchor visible surface contains invalid coordinates")

    points_canonical = points_world - centroid[None, :]
    object_dir = output_dir / "anchor_visible_surface_mesh" / safe_name(object_id.replace("object:", "object_"))
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
    out: dict[str, Any] = {
        "status": "anchor_visible_surface_mesh_exported",
        "coordinate_frame": "object_canonical_anchor_centroid_frame",
        "canonical_coordinate_source": "selected_anchor_frame_visible_surfels_minus_anchor_centroid_world_m",
        "anchor_frame_idx": int(anchor_frame),
        "anchor_centroid_world_m": centroid.astype(float).tolist(),
        "point_count_input": int(points_canonical.shape[0]),
        "point_count_downsampled": int(down_points.shape[0]),
        "voxel_size_m": float(voxel_size),
        "canonical_bbox_min_m": down_points.min(axis=0).astype(float).tolist() if len(down_points) else None,
        "canonical_bbox_max_m": down_points.max(axis=0).astype(float).tolist() if len(down_points) else None,
        "fused_point_cloud_path": None,
        "poisson_mesh_path": None,
        "convex_hull_mesh_path": None,
        "blockers": [],
        "claim_scope": "selected-frame metric visible surface for TRELLIS alignment; not complete hidden object geometry",
    }
    pcd_path = object_dir / f"frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_visible_points_canonical.ply"
    if not o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write anchor visible point cloud: {pcd_path}")
    out["fused_point_cloud_path"] = str(pcd_path)
    if down_points.shape[0] < 30:
        out["status"] = "too_few_downsampled_anchor_points_for_mesh"
        out["blockers"].append("too_few_downsampled_anchor_points_for_mesh")
        return out
    try:
        hull, _ = pcd.compute_convex_hull()
        hull.compute_vertex_normals()
        hull_path = object_dir / f"frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_convex_hull_visible_candidate.ply"
        if o3d.io.write_triangle_mesh(str(hull_path), hull, write_ascii=False, compressed=False):
            out["convex_hull_mesh_path"] = str(hull_path)
            out["convex_hull_vertices"] = int(np.asarray(hull.vertices).shape[0])
            out["convex_hull_faces"] = int(np.asarray(hull.triangles).shape[0])
    except Exception as exc:
        out["blockers"].append(f"convex_hull_failed:{type(exc).__name__}:{exc}")
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 4.0, float(min_voxel_m) * 4.0), max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(poisson_depth))
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
        poisson_path = object_dir / f"frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_poisson_visible_mesh.ply"
        if not o3d.io.write_triangle_mesh(str(poisson_path), mesh, write_ascii=False, compressed=False):
            raise RuntimeError("Open3D write_triangle_mesh returned false")
        out["poisson_mesh_path"] = str(poisson_path)
        out["poisson_vertices"] = int(np.asarray(mesh.vertices).shape[0])
        out["poisson_faces"] = int(np.asarray(mesh.triangles).shape[0])
    except Exception as exc:
        out["blockers"].append(f"poisson_reconstruction_failed:{type(exc).__name__}:{exc}")
        if not out.get("convex_hull_mesh_path"):
            out["status"] = "anchor_visible_surface_point_cloud_only_mesh_failed"
        else:
            out["status"] = "anchor_visible_surface_hull_only_poisson_failed"
    return out


def raw_frame_map(manifest_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = load_json(manifest_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{manifest_path} must contain a nonempty frames list")
    out: dict[int, dict[str, Any]] = {}
    for row_raw in frames:
        if not isinstance(row_raw, dict):
            continue
        idx = int(row_raw.get("frame_idx", row_raw.get("index", -1)))
        if idx < 0:
            raise RuntimeError(f"raw frame row lacks frame_idx/index: {row_raw}")
        if idx in out:
            raise RuntimeError(f"duplicate raw frame {idx} in {manifest_path}")
        out[idx] = row_raw
    return out, payload


def scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{label} must be a scalar string, got shape {array.shape}")
    return str(array.reshape(-1)[0])


def load_depth_npz(path: Path) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=True)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{path} missing depth keys: {missing}")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    depth = np.asarray(blob["depth"], dtype=np.float32)
    intr = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    if depth.ndim != 3:
        raise RuntimeError(f"{path} depth must have shape [N,H,W], got {depth.shape}")
    if len(frame_idx) != depth.shape[0] or len(frame_idx) != intr.shape[0]:
        raise RuntimeError(f"{path} has inconsistent frame/depth/intrinsics rows")
    source_size = np.asarray(blob["source_size"], dtype=float) if "source_size" in blob.files else None
    confidence = np.asarray(blob["confidence"], dtype=np.float32) if "confidence" in blob.files else None
    if confidence is not None and confidence.shape != depth.shape:
        raise RuntimeError(f"{path} confidence shape {confidence.shape} disagrees with depth {depth.shape}")
    camera_contract_binding = None
    binding_keys = {
        "camera_contract_path",
        "camera_contract_sha256",
        "camera_contract_plane",
        "A_depth_from_calibration",
        "intrinsics_source",
        "calibration_authority",
        "depth_ray_geometry_reprojected",
        "metadata_only_ray_relabel_override",
    }
    present_binding_keys = binding_keys.intersection(blob.files)
    if present_binding_keys and present_binding_keys != binding_keys:
        raise RuntimeError(
            f"{path} has an incomplete camera-contract binding; missing={sorted(binding_keys.difference(blob.files))}"
        )
    if present_binding_keys:
        affine = np.asarray(blob["A_depth_from_calibration"], dtype=np.float64)
        if affine.shape != (3, 3) or not np.isfinite(affine).all():
            raise RuntimeError(f"{path} has invalid A_depth_from_calibration")
        camera_contract_binding = {
            "path": scalar_text(blob["camera_contract_path"], "camera_contract_path"),
            "sha256": scalar_text(blob["camera_contract_sha256"], "camera_contract_sha256"),
            "plane": scalar_text(blob["camera_contract_plane"], "camera_contract_plane"),
            "A_depth_from_calibration": affine.tolist(),
            "intrinsics_source": scalar_text(blob["intrinsics_source"], "intrinsics_source"),
            "calibration_authority": scalar_text(blob["calibration_authority"], "calibration_authority"),
            "depth_ray_geometry_reprojected": bool(np.asarray(blob["depth_ray_geometry_reprojected"]).reshape(-1)[0]),
            "metadata_only_ray_relabel_override": bool(np.asarray(blob["metadata_only_ray_relabel_override"]).reshape(-1)[0]),
        }
    return {
        "path": str(path),
        "frame_idx": frame_idx,
        "depth": depth,
        "confidence": confidence,
        "intrinsics": intr,
        "source_size": source_size,
        "source_estimated_intrinsics": (
            np.asarray(blob["source_estimated_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
            if "source_estimated_intrinsics_fx_fy_cx_cy" in blob.files
            else None
        ),
        "camera_contract_binding": camera_contract_binding,
        "frame_to_i": {int(idx): int(i) for i, idx in enumerate(frame_idx)},
    }


def load_calibration_contract(
    path: Path | None,
    *,
    frame_ids: list[int] | None = None,
) -> tuple[np.ndarray | None, str | None, dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None, None, None
    if not path.exists():
        raise FileNotFoundError(f"missing calibration contract: {path}")
    payload = load_json(path)
    v2_declared = payload.get("schema") == CAMERA_CONTRACT_V2_SCHEMA
    v2_fields_present = "calibration_plane" in payload or "image_planes" in payload
    if v2_declared or v2_fields_present:
        full_payload, normalized = load_camera_contract_v2(path, expected_frame_ids=frame_ids)
        source = str(full_payload.get("intrinsics_source") or full_payload.get("method") or "v19_camera_contract")
        summary = summarize_camera_contract_v2(path, full_payload, normalized)
        return None, f"camera_contract:{source}", summary, {
            "payload": full_payload,
            "normalized": normalized,
        }
    intr = np.asarray(payload.get("intrinsics_fx_fy_cx_cy"), dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or float(intr[0]) <= 0.0 or float(intr[1]) <= 0.0:
        raise RuntimeError(f"calibration contract has invalid intrinsics_fx_fy_cx_cy: {path}")
    source = str(payload.get("intrinsics_source") or payload.get("method") or "v19_calibration_contract")
    source = f"legacy_calibration_contract:{source}"
    summary = {
        "path": str(path),
        "schema": payload.get("schema"),
        "calibration_authority": payload.get("calibration_authority") or "legacy_unspecified",
        "intrinsics_fx_fy_cx_cy": [float(v) for v in intr.tolist()],
        "intrinsics_source": source,
        "fov_degrees": payload.get("fov_degrees"),
        "aggregation": payload.get("aggregation"),
        "legacy_contract_without_explicit_image_planes": True,
    }
    return intr.astype(float), source, summary, None


def validate_depth_camera_contract_binding(
    *,
    depth: dict[str, Any],
    camera_contract_v2: dict[str, Any] | None,
    calibration_contract_path: Path | None,
    depth_image_plane: str,
    allow_implicit_depth_resize: bool,
) -> dict[str, Any]:
    if camera_contract_v2 is None:
        return {
            "status": "not_required_without_v2_camera_contract",
            "archive_binding": depth.get("camera_contract_binding"),
        }
    if calibration_contract_path is None:
        raise RuntimeError("internal error: V2 camera contract has no source path")
    binding = depth.get("camera_contract_binding")
    if not isinstance(binding, dict):
        raise RuntimeError(
            "a V2 camera contract requires a depth archive produced by adapt_v19_depth_to_camera_contract.py; "
            "the archive has no complete camera-contract binding"
        )
    expected_contract_hash = camera_contract_sha256_file(calibration_contract_path)
    if binding["sha256"] != expected_contract_hash:
        raise RuntimeError(
            "depth archive camera-contract hash disagrees with --calibration-contract: "
            f"depth={binding['sha256']} supplied={expected_contract_hash}"
        )
    if binding["plane"] != depth_image_plane:
        raise RuntimeError(
            f"depth archive is bound to plane {binding['plane']!r}, not requested {depth_image_plane!r}"
        )
    expected_depth_intrinsics, expected_depth_transform = camera_contract_plane_intrinsics(
        camera_contract_v2["payload"],
        camera_contract_v2["normalized"],
        plane_name=depth_image_plane,
        actual_size_wh=(int(depth["depth"].shape[2]), int(depth["depth"].shape[1])),
        allow_implicit_resize=bool(allow_implicit_depth_resize),
    )
    if not np.allclose(depth["intrinsics"], expected_depth_intrinsics[None, :], atol=1.0e-6, rtol=0.0):
        raise RuntimeError("depth archive intrinsics rows disagree with the requested V2 contract plane")
    if not np.allclose(
        np.asarray(binding["A_depth_from_calibration"], dtype=np.float64),
        np.asarray(expected_depth_transform["A_actual_plane_from_calibration"], dtype=np.float64),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise RuntimeError("depth archive A_depth_from_calibration disagrees with the requested V2 contract plane")
    if binding.get("metadata_only_ray_relabel_override") is True or binding.get("depth_ray_geometry_reprojected") is not True:
        # A byte-identical z raster with a changed K is not a geometric camera
        # adaptation. The only safe unchanged-raster case is exact source/resolved
        # K equality, represented by intrinsics_override_applied=false.
        source_intrinsics = depth.get("source_estimated_intrinsics")
        same_rays = bool(
            source_intrinsics is not None
            and np.asarray(source_intrinsics).shape == np.asarray(depth["intrinsics"]).shape
            and np.allclose(source_intrinsics, depth["intrinsics"], atol=1.0e-6, rtol=0.0)
        )
        if not same_rays:
            raise RuntimeError(
                "depth archive only relabeled K metadata: dense z was not reprojected onto the requested camera rays"
            )
    return {
        **binding,
        "status": "exact_camera_contract_hash_plane_intrinsics_and_affine_match",
        "supplied_contract_path": str(calibration_contract_path),
        "supplied_contract_sha256": expected_contract_hash,
    }


def load_camera_npz(path: Path | None) -> dict[int, tuple[np.ndarray, str]]:
    if path is None:
        return {}
    blob = np.load(path, allow_pickle=True)
    if "frame_idx" not in blob.files:
        raise RuntimeError(f"camera npz {path} lacks frame_idx")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    poses: dict[int, tuple[np.ndarray, str]] = {}
    if "T_world_camera" in blob.files:
        mats = np.asarray(blob["T_world_camera"], dtype=float)
        source = "camera_npz_T_world_camera"
    elif "T_world_camera_metric_current_v18" in blob.files:
        mats = np.asarray(blob["T_world_camera_metric_current_v18"], dtype=float)
        source = "camera_npz_T_world_camera_metric_current_v18"
    elif "R_c2w" in blob.files and "t_c2w" in blob.files:
        r = np.asarray(blob["R_c2w"], dtype=float)
        t = np.asarray(blob["t_c2w"], dtype=float)
        if len(r) != len(frame_idx) or len(t) != len(frame_idx):
            raise RuntimeError(f"camera npz {path} has inconsistent R_c2w/t_c2w rows")
        for i, idx in enumerate(frame_idx):
            T = np.eye(4, dtype=float)
            T[:3, :3] = r[i]
            T[:3, 3] = t[i]
            if T.shape != (4, 4) or not np.isfinite(T).all():
                raise RuntimeError(f"camera npz {path} invalid pose for frame {idx}")
            poses[int(idx)] = (T, "camera_npz_R_c2w_t_c2w")
        return poses
    else:
        raise RuntimeError(
            f"camera npz {path} lacks a supported camera pose key; expected T_world_camera, "
            "T_world_camera_metric_current_v18, or R_c2w/t_c2w"
        )
    if mats.shape[0] != len(frame_idx) or mats.shape[1:] != (4, 4):
        raise RuntimeError(f"camera npz {path} pose shape mismatch: {mats.shape}")
    for i, idx in enumerate(frame_idx):
        T = np.asarray(mats[i], dtype=float)
        if not np.isfinite(T).all():
            raise RuntimeError(f"camera npz {path} invalid pose for frame {idx}")
        poses[int(idx)] = (T, source)
    return poses


def load_base_annotations(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"{path} must contain frames list")
    out: dict[int, dict[str, Any]] = {}
    for row in frames:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("frame_idx", -1))
        if idx < 0:
            raise RuntimeError(f"base annotation row lacks frame_idx: {row}")
        if idx in out:
            raise RuntimeError(f"duplicate base annotation frame {idx}")
        out[idx] = row
    return out


def load_object_plan_record(path: Path | None, track_id: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, dict):
        plan = payload if isinstance(payload, dict) else {}
    for row in as_list(plan.get("objects")):
        if isinstance(row, dict) and str(row.get("track_id")) == str(track_id):
            return row
    return None


def load_sam2_track(args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], Path]:
    if args.sam2_track_json is not None:
        path = args.sam2_track_json
    else:
        path = args.sam2_root / args.track_id / "sam2" / "sam2_track.json"
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"SAM2 track must be an object: {path}")
    out: dict[int, dict[str, Any]] = {}
    for key, value in payload.items():
        try:
            idx = int(key)
        except ValueError as exc:
            raise RuntimeError(f"SAM2 track key is not a frame index: {key}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"SAM2 track row for frame {idx} is not an object")
        out[idx] = value
    return out, path


def frame_range(raw_frames: dict[int, dict[str, Any]], start: int | None, end: int | None) -> list[int]:
    indices = sorted(raw_frames)
    lo = indices[0] if start is None else int(start)
    hi = indices[-1] if end is None else int(end)
    if hi < lo:
        raise RuntimeError(f"invalid frame range {lo}:{hi}")
    selected = [idx for idx in indices if lo <= idx <= hi]
    if not selected:
        raise RuntimeError(f"no raw frames in range {lo}:{hi}")
    return selected


def base_camera_pose(frame: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    for key in ("T_world_camera_metric", "T_world_camera", "T_world_camera_metric_current_v18"):
        value = camera.get(key)
        arr = np.asarray(value if value is not None else [], dtype=float)
        if arr.shape == (4, 4) and np.isfinite(arr).all():
            return arr, f"base_annotations_camera_{key}"
    return None


def base_camera_intrinsics(frame: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    value = camera.get("intrinsics_fx_fy_cx_cy")
    arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    if arr.shape == (4,) and np.isfinite(arr).all() and float(arr[0]) > 0.0 and float(arr[1]) > 0.0:
        source = str(camera.get("intrinsics_source") or "base_annotations_camera_intrinsics_fx_fy_cx_cy")
        return arr.astype(float), f"base_annotations:{source}"
    return None


def resolve_camera_pose(
    frame_idx: int,
    frame: dict[str, Any],
    camera_poses: dict[int, tuple[np.ndarray, str]],
    allow_camera_frame_world: bool,
) -> tuple[np.ndarray, str]:
    base = base_camera_pose(frame)
    if base is not None:
        return base
    if frame_idx in camera_poses:
        return camera_poses[frame_idx]
    if allow_camera_frame_world:
        return np.eye(4, dtype=float), "explicit_camera_frame_world_identity_not_metric_temporal_world"
    raise RuntimeError(
        f"frame {frame_idx} lacks camera/world pose. Provide --base-annotations with camera fields, "
        "--camera-npz, or explicit --allow-camera-frame-world."
    )


def localize_path(path: str | Path, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for src, dst in ((remote_root, local_root), (local_root, remote_root)):
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
        raise RuntimeError(f"failed to read mask {path}")
    return mask > 0


def bbox_xyxy_from_mask(mask: np.ndarray, source_width: int, source_height: int) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    sx = float(source_width) / float(mask.shape[1])
    sy = float(source_height) / float(mask.shape[0])
    return [float(xs.min() * sx), float(ys.min() * sy), float((xs.max() + 1) * sx), float((ys.max() + 1) * sy)]


def scaled_intrinsics_for_depth(
    intr: np.ndarray,
    depth_shape: tuple[int, int],
    source_size: np.ndarray | None,
    *,
    explicit_source_size_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(intr, dtype=float).tolist()
    h, w = depth_shape
    if explicit_source_size_wh is not None:
        source_w, source_h = [float(v) for v in explicit_source_size_wh]
    elif source_size is not None and source_size.size >= 2:
        source_w = float(source_size[0])
        source_h = float(source_size[1])
    else:
        source_w = source_h = 0.0
    if source_w > 0 and source_h > 0 and (abs(source_w - w) > 1e-6 or abs(source_h - h) > 1e-6):
        sx = float(w) / source_w
        sy = float(h) / source_h
        return np.asarray([fx * sx, fy * sy, cx * sx, cy * sy], dtype=float)
    return np.asarray([fx, fy, cx, cy], dtype=float)


def load_npz_array_cached(path: Path, key: str) -> np.ndarray:
    cache = getattr(load_npz_array_cached, "_cache", None)
    if cache is None:
        cache = {}
        setattr(load_npz_array_cached, "_cache", cache)
    cache_key = (str(path.expanduser().resolve()), str(key))
    if cache_key not in cache:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"missing hand geometry archive: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise RuntimeError(f"hand geometry archive {path} lacks {key}")
            cache[cache_key] = np.asarray(archive[key]).copy()
    return cache[cache_key]


def projected_mano_hand_silhouette(
    hand: dict[str, Any],
    *,
    mask_shape: tuple[int, int],
    source_width: int,
    source_height: int,
    pad_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    state = hand.get("metric_mano_state")
    if not isinstance(state, dict):
        raise RuntimeError("hand row lacks metric_mano_state")
    reference = state.get("vertices_reference")
    if not isinstance(reference, dict):
        raise RuntimeError("hand metric state lacks vertices_reference")
    bridge_path = Path(str(reference.get("bridge_npz") or ""))
    bridge_key = str(reference.get("bridge_vertices_camera_array") or "")
    row_index = int(reference.get("bridge_row_index", -1))
    vertices_rows = load_npz_array_cached(bridge_path, bridge_key)
    if vertices_rows.ndim != 3 or vertices_rows.shape[1:] != (778, 3) or not (0 <= row_index < len(vertices_rows)):
        raise RuntimeError(f"invalid bridge MANO camera vertices {vertices_rows.shape} row={row_index}")
    vertices = np.asarray(vertices_rows[row_index], dtype=np.float64)
    side = str(hand.get("hand_side") or "").lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"invalid hand side {side!r}")
    source_hawor = Path(str(reference.get("source_hawor_npz") or ""))
    faces = np.asarray(load_npz_array_cached(source_hawor, f"{side}_faces"), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise RuntimeError(f"invalid {side} MANO faces {faces.shape}")
    intrinsics = np.asarray(
        state.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
        or state.get("source_hawor_camera_intrinsics_fx_fy_cx_cy")
        or [],
        dtype=np.float64,
    ).reshape(-1)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
        raise RuntimeError(f"invalid {side} MANO camera intrinsics: {intrinsics}")
    z = vertices[:, 2]
    positive_faces = np.all(np.isfinite(vertices[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    faces = faces[positive_faces]
    if len(faces) == 0:
        raise RuntimeError(f"{side} MANO mesh has no finite positive-depth faces")
    used_vertices = np.unique(faces.reshape(-1))
    fx, fy, cx, cy = intrinsics.tolist()
    uv = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    uv[used_vertices, 0] = fx * vertices[used_vertices, 0] / z[used_vertices] + cx
    uv[used_vertices, 1] = fy * vertices[used_vertices, 1] / z[used_vertices] + cy
    mask_h, mask_w = mask_shape
    sx = float(mask_w) / float(max(1, source_width))
    sy = float(mask_h) / float(max(1, source_height))
    uv[:, 0] *= sx
    uv[:, 1] *= sy
    if not np.isfinite(uv[used_vertices]).all():
        raise RuntimeError(f"{side} MANO projection has non-finite pixels")
    polygons = np.rint(uv[faces]).astype(np.int32)
    silhouette = np.zeros((mask_h, mask_w), dtype=np.uint8)
    cv2.fillPoly(silhouette, list(polygons), 1, lineType=cv2.LINE_8)
    if int(np.count_nonzero(silhouette)) == 0:
        raise RuntimeError(f"{side} MANO projected silhouette is empty")
    pad = int(max(0, pad_px))
    if pad:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        silhouette = cv2.dilate(silhouette, kernel, iterations=1)
    ys, xs = np.where(silhouette > 0)
    projected_bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    return silhouette > 0, {
        "hand_side": side,
        "bridge_npz": str(bridge_path.expanduser().resolve()),
        "bridge_vertices_camera_array": bridge_key,
        "bridge_row_index": row_index,
        "source_hawor_npz": str(source_hawor.expanduser().resolve()),
        "faces_array": f"{side}_faces",
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "source_camera_intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "projected_silhouette_bbox_mask_xyxy": projected_bbox,
        "projected_silhouette_pixels_with_padding": int(np.count_nonzero(silhouette)),
        "padding_px_in_mask_coordinates": pad,
        "source_bbox_xyxy_diagnostic_only": hand.get("bbox_xyxy"),
    }


def subtract_hand_owned_bbox_regions(
    mask: np.ndarray,
    base_frame: dict[str, Any],
    *,
    source_width: int,
    source_height: int,
    pad_px: int,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract projected HaWoR/MANO hand silhouettes, never coarse bboxes.

    Rectangular hand boxes erase visible object pixels and still fail to describe
    fingers at object boundaries. Full prediction-side MANO vertices/faces provide
    a tighter ownership silhouette. Missing or malformed geometry fails the frame
    closed rather than falling back to a box or treating hand pixels as object.
    """
    if not enabled:
        return mask, {
            "state": "disabled",
            "input_mask_pixels": int(mask.sum()),
            "output_mask_pixels": int(mask.sum()),
            "fail_closed": False,
        }
    out = mask.copy()
    input_pixels = int(mask.sum())
    hand_rows = [
        hand for hand in as_list(base_frame.get("hands"))
        if isinstance(hand, dict)
        and not (hand.get("same_frame_detection") is False and not hand.get("hawor_candidate_present"))
    ]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for hand in hand_rows:
        try:
            silhouette, record = projected_mano_hand_silhouette(
                hand,
                mask_shape=out.shape[:2],
                source_width=source_width,
                source_height=source_height,
                pad_px=pad_px,
            )
            before = int(out.sum())
            out[silhouette] = False
            record["removed_object_mask_pixels"] = before - int(out.sum())
            records.append(record)
        except Exception as exc:
            failures.append({"hand_side": hand.get("hand_side"), "reason": str(exc)})
    fail_closed = bool(failures)
    if fail_closed:
        # Do not return a partially-owned mask when one detected hand could not be
        # represented. The caller records diagnostics and rejects this frame.
        out[:] = False
    output_pixels = int(out.sum())
    return out, {
        "state": (
            "failed_closed_missing_projected_mano_hand_silhouette"
            if fail_closed else
            "projected_mano_hand_silhouettes_subtracted"
            if records else
            "no_detected_hands_to_subtract"
        ),
        "ownership_primitive": "projected_full_mano_triangle_silhouette",
        "bbox_subtraction_used": False,
        "input_mask_pixels": input_pixels,
        "output_mask_pixels": output_pixels,
        "removed_mask_pixels": input_pixels - output_pixels,
        "detected_hand_rows": int(len(hand_rows)),
        "projected_hand_silhouettes": records,
        "failures": failures,
        "fail_closed": fail_closed,
        "padding_px_in_mask_coordinates": int(max(0, pad_px)),
        "claim_scope": "Only projected prediction-side MANO triangle silhouettes are removed. Hand bboxes are diagnostic and never ownership masks.",
    }


def robust_first_surface_depth_ownership(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    enabled: bool,
    mad_sigma: float,
    min_half_width_m: float,
    min_retained_fraction: float,
    fail_raw_to_robust_extent_ratio: float,
    intrinsics: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    confidence_seed_percentile: float = 95.0,
    local_depth_step_max_m: float = 0.005,
    max_removed_distance_inside_mask_px: float = 10.0,
    max_confidence_flagged_interior_fraction: float = 0.015,
    min_interior_confidence_flagged_fraction: float = 0.95,
    max_unexplained_interior_pixels: int = 5,
    min_component_pixels: int = 20,
    max_small_component_fraction: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any]]:
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
        return valid, {
            "enabled": bool(enabled),
            "state": "no_valid_owned_depth",
            "input_valid_depth_pixels": 0,
            "retained_depth_pixels": 0,
            "retained_fraction": 0.0,
            "fail_closed": True,
            "failure_reasons": ["no_valid_owned_depth"],
        }
    global_median = float(np.median(values))
    global_mad = float(np.median(np.abs(values - global_median)))
    if not enabled:
        return valid, {
            "enabled": False,
            "state": "disabled",
            "input_valid_depth_pixels": int(values.size),
            "retained_depth_pixels": int(values.size),
            "retained_fraction": 1.0,
            "median_depth_m": global_median,
            "median_absolute_deviation_m": global_mad,
            "fail_closed": False,
            "failure_reasons": [],
        }

    confidence_m = None if confidence is None else np.asarray(confidence, dtype=np.float64)
    failure_reasons: list[str] = []
    if confidence_m is None or confidence_m.shape != depth_m.shape:
        failure_reasons.append("missing_or_mismatched_unidepth_confidence")
        confidence_m = np.full(depth_m.shape, np.inf, dtype=np.float64)
    elif not np.isfinite(confidence_m[valid]).all() or np.any(confidence_m[valid] <= 0.0):
        failure_reasons.append("invalid_unidepth_confidence")

    accepted = np.zeros_like(valid)
    confidence_threshold_map = np.full(depth_m.shape, np.nan, dtype=np.float64)
    component_rows: list[dict[str, Any]] = []
    small_component_pixels = 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), 8)
    height, width = valid.shape
    neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for label in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[label]]
        component = labels[y : y + h, x : x + w] == label
        if area < int(min_component_pixels):
            small_component_pixels += area
            component_rows.append({"label": label, "pixels": area, "state": "dropped_small_component"})
            continue
        local_depth = depth_m[y : y + h, x : x + w]
        local_confidence = confidence_m[y : y + h, x : x + w]
        component_values = local_depth[component]
        median = float(np.median(component_values))
        mad = float(np.median(np.abs(component_values - median)))
        robust_sigma = float(1.4826 * mad)
        half_width = float(max(min_half_width_m, mad_sigma * robust_sigma))
        seed = component & (np.abs(local_depth - median) <= half_width)
        seed_count = int(np.count_nonzero(seed))
        if seed_count < 3 or not np.isfinite(local_confidence[seed]).all():
            failure_reasons.append(f"component_{label}_invalid_seed")
            component_rows.append({"label": label, "pixels": area, "state": "invalid_seed"})
            continue
        confidence_threshold = float(np.percentile(local_confidence[seed], confidence_seed_percentile))
        threshold_view = confidence_threshold_map[y : y + h, x : x + w]
        threshold_view[component] = confidence_threshold
        traversable = component & (local_confidence <= confidence_threshold)
        keep = seed.copy()
        iterations = 0
        for iterations in range(max(1, w + h)):
            previous_count = int(np.count_nonzero(keep))
            adjacent = np.zeros_like(keep)
            for dy, dx in neighbor_offsets:
                source_y = slice(max(0, dy), min(h, h + dy))
                target_y = slice(max(0, -dy), min(h, h - dy))
                source_x = slice(max(0, dx), min(w, w + dx))
                target_x = slice(max(0, -dx), min(w, w - dx))
                adjacent[target_y, target_x] |= (
                    keep[source_y, source_x]
                    & (np.abs(local_depth[target_y, target_x] - local_depth[source_y, source_x]) <= float(local_depth_step_max_m))
                )
            keep |= traversable & adjacent
            if int(np.count_nonzero(keep)) == previous_count:
                break
        accepted[y : y + h, x : x + w] |= keep
        keep_count = int(np.count_nonzero(keep))
        component_rows.append({
            "label": label,
            "bbox_xywh": [x, y, w, h],
            "pixels": area,
            "median_depth_m": median,
            "median_absolute_deviation_m": mad,
            "robust_sigma_m": robust_sigma,
            "seed_half_width_m": half_width,
            "seed_pixels": seed_count,
            "seed_fraction": float(seed_count / max(1, area)),
            "confidence_seed_percentile": float(confidence_seed_percentile),
            "confidence_threshold": confidence_threshold,
            "retained_pixels": keep_count,
            "retained_fraction": float(keep_count / max(1, area)),
            "growth_iterations": int(iterations),
            "accepted_depth_summary_m": numeric_summary(local_depth[keep]),
            "state": "depth_confidence_geodesic_support",
        })

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
    confidence_flagged_interior = (
        np.isfinite(interior_confidence)
        & np.isfinite(interior_confidence_threshold)
        & (interior_confidence > interior_confidence_threshold)
    )
    confidence_flagged_interior_count = int(np.count_nonzero(confidence_flagged_interior))
    confidence_flagged_interior_fraction = float(
        confidence_flagged_interior_count / max(1, interior_removed_count)
    )
    interior_quarantine_mode = "none"
    interior_quarantine_validated = interior_removed_count == 0
    if 0 < interior_removed_count <= int(max_unexplained_interior_pixels):
        interior_quarantine_validated = True
        interior_quarantine_mode = "isolated_raster_samples"
    elif interior_removed_count > 0:
        interior_quarantine_validated = bool(
            interior_removed_fraction <= float(max_confidence_flagged_interior_fraction)
            and confidence_flagged_interior_fraction >= float(min_interior_confidence_flagged_fraction)
        )
        interior_quarantine_mode = (
            "sparse_unidepth_predicted_error_flagged_interior_holes"
            if interior_quarantine_validated else
            "unresolved_interior_rejection"
        )
    small_component_fraction = float(small_component_pixels / max(1, values.size))

    raw_extent = backprojected_extent(valid, depth_m, intrinsics)
    accepted_extent = backprojected_extent(accepted, depth_m, intrinsics)
    raw_diag = float(np.linalg.norm(raw_extent))
    accepted_diag = float(np.linalg.norm(accepted_extent))
    raw_to_accepted_ratio = float(raw_diag / max(accepted_diag, 1.0e-12))
    if retained < 3 or retained_fraction < float(min_retained_fraction):
        failure_reasons.append("insufficient_depth_coherent_first_surface_support")
    if not np.isfinite(accepted_diag) or accepted_diag <= 0.0:
        failure_reasons.append("invalid_accepted_backprojected_extent")
    if small_component_fraction > float(max_small_component_fraction):
        failure_reasons.append("too_much_support_in_dropped_small_components")
    if not interior_quarantine_validated:
        failure_reasons.append("rejected_interior_depth_not_sparse_and_confidence_flagged")
    tail_dominates_raw_extent = bool(raw_to_accepted_ratio > float(fail_raw_to_robust_extent_ratio))
    return accepted, {
        "enabled": True,
        "state": (
            "fail_closed_depth_ownership_unresolved"
            if failure_reasons else
            "validated_sparse_interior_and_boundary_depth_quarantined"
            if interior_removed_count else
            "validated_boundary_depth_tail_quarantined"
            if removed_count else
            "all_owned_depth_coherent"
        ),
        "method": "per_component_mad_seed_confidence_geodesic_growth",
        "input_valid_depth_pixels": int(values.size),
        "retained_depth_pixels": retained,
        "removed_depth_pixels": removed_count,
        "retained_fraction": retained_fraction,
        "removed_fraction": float(removed_count / max(1, values.size)),
        "median_depth_m": global_median,
        "median_absolute_deviation_m": global_mad,
        "mad_sigma": float(mad_sigma),
        "minimum_half_width_m": float(min_half_width_m),
        "confidence_semantics": "UniDepth exp(logconfidence); larger values predict larger depth error",
        "confidence_seed_percentile": float(confidence_seed_percentile),
        "local_depth_step_max_m": float(local_depth_step_max_m),
        "max_removed_distance_inside_mask_px": float(max_removed_distance_inside_mask_px),
        "max_confidence_flagged_interior_fraction": float(max_confidence_flagged_interior_fraction),
        "min_interior_confidence_flagged_fraction": float(min_interior_confidence_flagged_fraction),
        "max_unexplained_interior_pixels": int(max_unexplained_interior_pixels),
        "component_count": int(component_count - 1),
        "component_rows": component_rows,
        "small_component_pixels": int(small_component_pixels),
        "small_component_fraction": small_component_fraction,
        "raw_backprojected_extent_m": raw_extent.astype(float).tolist(),
        "raw_backprojected_extent_diag_m": raw_diag,
        "robust_backprojected_extent_m": accepted_extent.astype(float).tolist(),
        "robust_backprojected_extent_diag_m": accepted_diag,
        "raw_to_robust_extent_diag_ratio": raw_to_accepted_ratio,
        "fail_raw_to_robust_extent_ratio": float(fail_raw_to_robust_extent_ratio),
        "raw_rejected_depth_dominates_extent": tail_dominates_raw_extent,
        "raw_tail_dominates_extent_but_is_quarantined": bool(tail_dominates_raw_extent and not failure_reasons),
        "removed_pixel_distance_inside_owned_mask_px": numeric_summary(removed_distances),
        "interior_rejection_quarantine": {
            "boundary_band_distance_px": float(max_removed_distance_inside_mask_px),
            "interior_removed_pixels": interior_removed_count,
            "interior_removed_fraction_of_valid_owned_depth": interior_removed_fraction,
            "interior_removed_depth_summary_m": numeric_summary(depth_m[interior_removed]),
            "interior_removed_confidence_summary": numeric_summary(interior_confidence),
            "confidence_flagged_interior_pixels": confidence_flagged_interior_count,
            "confidence_flagged_interior_fraction": confidence_flagged_interior_fraction,
            "max_confidence_flagged_interior_fraction": float(max_confidence_flagged_interior_fraction),
            "min_interior_confidence_flagged_fraction": float(min_interior_confidence_flagged_fraction),
            "max_unexplained_interior_pixels": int(max_unexplained_interior_pixels),
            "validated": bool(interior_quarantine_validated),
            "mode": interior_quarantine_mode,
            "semantics": "Pixels beyond the ordinary boundary band remain excluded only when isolated or when a sparse region is overwhelmingly marked by UniDepth as higher predicted error than its component seed. This is explicit unresolved-depth quarantine, not object-mask erosion.",
        },
        "fail_closed": bool(failure_reasons),
        "failure_reasons": failure_reasons,
        "claim_scope": "Depth first-surface surfel ownership only. Excluded owned depth remains unresolved; the P11 RGB/mask appearance contract is not eroded or rewritten.",
    }


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def backprojected_extent(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray | None = None,
) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros(3, dtype=np.float64)
    z = depth[ys, xs].astype(np.float64)
    if intrinsics is None:
        # Tests and proposal-only diagnostics may use normalized rays. Production
        # P09 passes the exact active depth-plane K below.
        h, w = depth.shape
        fx = fy = max(1.0, float(w))
        cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    else:
        intr = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
        if intr.shape != (4,) or not np.isfinite(intr).all() or intr[0] <= 0.0 or intr[1] <= 0.0:
            raise RuntimeError(f"invalid intrinsics for depth ownership: {intr}")
        fx, fy, cx, cy = intr.tolist()
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    return points.max(axis=0) - points.min(axis=0)


def choose_visible_points(
    valid: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    T_world_camera: np.ndarray,
    *,
    pixel_stride: int,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stride = max(1, int(pixel_stride))
    sampled = np.zeros_like(valid, dtype=bool)
    sampled[::stride, ::stride] = valid[::stride, ::stride]
    ys, xs = np.where(sampled)
    if len(xs) == 0:
        ys, xs = np.where(valid)
    if len(xs) > int(max_points):
        order = rng.choice(len(xs), size=int(max_points), replace=False)
        ys = ys[order]
        xs = xs[order]
    z = depth[ys, xs].astype(float)
    fx, fy, cx, cy = intr.astype(float).tolist()
    X = (xs.astype(float) - cx) * z / fx
    Y = (ys.astype(float) - cy) * z / fy
    cam = np.column_stack([X, Y, z])
    hom = np.column_stack([cam, np.ones(len(cam), dtype=float)])
    world = (hom @ T_world_camera.T)[:, :3]
    keep = np.isfinite(world).all(axis=1) & np.isfinite(cam).all(axis=1)
    cam = cam[keep]
    world = world[keep]
    summary = {
        "valid_depth_mask_pixels": int(valid.sum()),
        "sampled_points_before_finite_filter": int(len(z)),
        "sampled_points": int(len(world)),
        "pixel_stride": int(stride),
        "max_points": int(max_points),
    }
    return cam.astype(float), world.astype(float), summary


def normalize01(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def raw_image_path_from_row(raw_row: dict[str, Any]) -> str:
    return str(raw_row.get("raw_frame_path") or raw_row.get("rgb") or raw_row.get("path") or "")


def candidate_mask_summary(mask: np.ndarray, source_width: int, source_height: int) -> dict[str, Any]:
    bbox = bbox_xyxy_from_mask(mask, source_width, source_height)
    area = int(mask.sum())
    h, w = mask.shape[:2]
    component_count = 0
    largest_component_px = 0
    if area > 0:
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        component_count = max(0, int(n) - 1)
        if n > 1:
            counts = np.bincount(labels.reshape(-1))[1:]
            largest_component_px = int(counts.max()) if counts.size else 0
    if len(bbox) >= 4:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        margin = 0.03 * float(min(max(1, source_width), max(1, source_height)))
        touches_border = bool(x0 <= margin or y0 <= margin or x1 >= float(source_width) - margin or y1 >= float(source_height) - margin)
        bbox_area = float(bw * bh)
    else:
        x0 = y0 = x1 = y1 = bw = bh = bbox_area = 0.0
        touches_border = True
    return {
        "mask_area_px": area,
        "component_count": int(component_count),
        "largest_component_px": int(largest_component_px),
        "bbox_xyxy": bbox,
        "bbox_width_px": float(bw),
        "bbox_height_px": float(bh),
        "bbox_area_px": float(bbox_area),
        "touches_or_near_image_border": bool(touches_border),
    }


def build_anchor_candidate_proposals(
    *,
    args: argparse.Namespace,
    object_id: str,
    visible_data: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Write anchor candidate evidence without choosing the anchor.

    The score is a proposal heuristic only.  It intentionally exposes the raw
    factors and review sheet so the runtime agent can choose, reject, or request
    another proposal after subjective visual/geometric inspection.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, vis in sorted(visible_data.items()):
        mask = read_mask(Path(vis["mask_path"]))
        source_width = int(vis.get("source_width") or mask.shape[1])
        source_height = int(vis.get("source_height") or mask.shape[0])
        mask_stats = candidate_mask_summary(mask, source_width, source_height)
        world_points = np.asarray(vis.get("world_points"), dtype=float)
        extent = world_points.max(axis=0) - world_points.min(axis=0) if world_points.ndim == 2 and len(world_points) else np.zeros(3, dtype=float)
        diag = float(np.linalg.norm(extent))
        ownership = vis.get("object_surface_ownership_filter") if isinstance(vis.get("object_surface_ownership_filter"), dict) else {}
        input_mask_px = int(ownership.get("input_mask_pixels") or mask_stats["mask_area_px"] or 0)
        output_mask_px = int(ownership.get("output_mask_pixels") or mask_stats["mask_area_px"] or 0)
        removed_px = int(ownership.get("removed_mask_pixels") or max(0, input_mask_px - output_mask_px))
        removed_frac = float(removed_px / input_mask_px) if input_mask_px > 0 else 0.0
        depth_p05 = float(vis.get("depth_p05_m") or np.nan)
        depth_p95 = float(vis.get("depth_p95_m") or np.nan)
        depth_spread = float(depth_p95 - depth_p05) if np.isfinite(depth_p05) and np.isfinite(depth_p95) else float("nan")
        sample_summary = vis.get("sample_summary") if isinstance(vis.get("sample_summary"), dict) else {}
        rows.append(
            {
                "frame_idx": int(idx),
                "raw_frame_path": raw_image_path_from_row(vis.get("raw_row") if isinstance(vis.get("raw_row"), dict) else {}),
                "object_owned_mask_path": str(vis.get("mask_path")),
                "raw_sam2_mask_path": str(vis.get("raw_sam2_mask_path")),
                "visible_depth_vertex_count": int(len(world_points)),
                "valid_depth_mask_pixels": int(sample_summary.get("valid_depth_mask_pixels") or 0),
                "sampled_points": int(sample_summary.get("sampled_points") or len(world_points)),
                "mask_area_px": int(mask_stats["mask_area_px"]),
                "raw_mask_area_px": int(input_mask_px),
                "hand_owned_removed_px": int(removed_px),
                "hand_owned_removed_fraction": float(removed_frac),
                "component_count": int(mask_stats["component_count"]),
                "largest_component_px": int(mask_stats["largest_component_px"]),
                "bbox_xyxy": mask_stats["bbox_xyxy"],
                "bbox_width_px": float(mask_stats["bbox_width_px"]),
                "bbox_height_px": float(mask_stats["bbox_height_px"]),
                "bbox_area_px": float(mask_stats["bbox_area_px"]),
                "touches_or_near_image_border": bool(mask_stats["touches_or_near_image_border"]),
                "depth_median_m": float(vis.get("depth_median_m") or np.nan),
                "depth_p05_m": depth_p05,
                "depth_p95_m": depth_p95,
                "depth_spread_p95_p05_m": depth_spread,
                "world_extent_m": extent.astype(float).tolist(),
                "world_extent_diag_m": float(diag),
                "intrinsics_source": str(vis.get("intrinsics_source")),
                "camera_source": str(vis.get("camera_source")),
            }
        )
    if not rows:
        raise RuntimeError("no visible rows available for anchor candidate proposal")
    area_values = np.asarray([float(r["mask_area_px"]) for r in rows], dtype=float)
    point_values = np.asarray([float(r["visible_depth_vertex_count"]) for r in rows], dtype=float)
    diag_values = np.asarray([float(r["world_extent_diag_m"]) for r in rows if float(r["world_extent_diag_m"]) > 0.0], dtype=float)
    depth_spreads = np.asarray([float(r["depth_spread_p95_p05_m"]) for r in rows if np.isfinite(float(r["depth_spread_p95_p05_m"]))], dtype=float)
    median_diag = float(np.median(diag_values)) if diag_values.size else 0.0
    depth_spread_hi = float(np.percentile(depth_spreads, 90.0)) if depth_spreads.size else 1.0
    area_hi = float(area_values.max()) if area_values.size else 1.0
    point_hi = float(point_values.max()) if point_values.size else 1.0
    for r in rows:
        area_score = float(r["mask_area_px"]) / max(area_hi, 1.0)
        point_score = float(r["visible_depth_vertex_count"]) / max(point_hi, 1.0)
        hand_clean_score = 1.0 - float(np.clip(r["hand_owned_removed_fraction"], 0.0, 1.0))
        border_score = 0.0 if bool(r["touches_or_near_image_border"]) else 1.0
        comp = max(1, int(r["component_count"]))
        component_score = float(max(0.0, 1.0 - 0.25 * (comp - 1)))
        diag = float(r["world_extent_diag_m"])
        extent_score = float(np.exp(-abs(np.log(max(diag, 1.0e-9) / max(median_diag, 1.0e-9))))) if median_diag > 0 else 0.0
        spread = float(r["depth_spread_p95_p05_m"])
        depth_score = 1.0 - normalize01(spread, 0.0, max(depth_spread_hi, 1.0e-6)) if np.isfinite(spread) else 0.0
        proposal_score = (
            0.24 * area_score
            + 0.16 * point_score
            + 0.18 * hand_clean_score
            + 0.16 * border_score
            + 0.14 * extent_score
            + 0.07 * component_score
            + 0.05 * depth_score
        )
        r["proposal_score"] = float(proposal_score)
        r["proposal_score_terms"] = {
            "mask_area_score": float(area_score),
            "visible_depth_vertex_count_score": float(point_score),
            "hand_clean_score": float(hand_clean_score),
            "not_near_image_border_score": float(border_score),
            "extent_consistency_score": float(extent_score),
            "single_component_score": float(component_score),
            "depth_stability_score": float(depth_score),
        }
        r["score_interpretation"] = "heuristic proposal score only; agent visual/geometric judgment must choose the anchor"
    ranked = sorted(rows, key=lambda r: (-float(r["proposal_score"]), int(r["frame_idx"])))
    top_k = max(1, int(args.anchor_candidate_count))
    min_gap = max(0, int(args.anchor_candidate_min_gap))
    review_rows: list[dict[str, Any]] = []
    for row in ranked:
        if all(abs(int(row["frame_idx"]) - int(prev["frame_idx"])) >= min_gap for prev in review_rows):
            review_rows.append(row)
        if len(review_rows) >= top_k:
            break
    if len(review_rows) < top_k:
        seen = {int(r["frame_idx"]) for r in review_rows}
        for row in ranked:
            if int(row["frame_idx"]) not in seen:
                review_rows.append(row)
                seen.add(int(row["frame_idx"]))
            if len(review_rows) >= top_k:
                break
    report_path = args.output_dir / "anchor_candidate_proposals.json"
    review_path = args.output_dir / "anchor_candidate_review.jpg"
    review_status = render_anchor_candidate_review(
        review_rows=review_rows,
        review_path=review_path,
        panel_width=int(args.anchor_candidate_panel_width),
    )
    report = {
        "method": "build_v19_visible_geometry_from_sam2_depth_anchor_candidate_proposals",
        "status": "ok",
        "object_id": object_id,
        "track_id": args.track_id,
        "claim_scope": "Anchor candidates for agent subjective selection. This report does not choose or accept an anchor frame.",
        "selection_required": True,
        "selection_instruction": "Inspect anchor_candidate_review.jpg and ranked_candidates; write an agent anchor decision before running P09/P11 with --anchor-frame/--selected-frame-idx.",
        "proposal_score_policy": "weighted heuristic over owned mask area, visible depth support, low hand-owned removal, non-border support, metric extent consistency, component count, and depth stability; score is not an acceptance gate.",
        "candidate_count": int(len(rows)),
        "review_candidate_count": int(len(review_rows)),
        "review_diversity_min_frame_gap": int(min_gap),
        "outputs": {
            "anchor_candidate_proposals": str(report_path),
            "anchor_candidate_review": str(review_path) if review_status.get("status") == "ok" else None,
        },
        "review_status": review_status,
        "ranked_candidates": ranked,
        "review_candidates": review_rows,
        "score_population_summary": {
            "max_mask_area_px": int(area_hi),
            "max_visible_depth_vertex_count": int(point_hi),
            "median_world_extent_diag_m": float(median_diag),
            "depth_spread_p90_m": float(depth_spread_hi),
        },
    }
    write_json(report_path, report)
    return report


def render_anchor_candidate_review(*, review_rows: list[dict[str, Any]], review_path: Path, panel_width: int) -> dict[str, Any]:
    panels: list[np.ndarray] = []
    blockers: list[str] = []
    panel_width = max(220, int(panel_width))
    for rank, row in enumerate(review_rows, start=1):
        raw_path = Path(str(row.get("raw_frame_path") or ""))
        mask_path = Path(str(row.get("object_owned_mask_path") or ""))
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
        cnts, _ = cv2.findContours(mask_bool_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)
        scale = panel_width / float(max(1, overlay.shape[1]))
        ph = int(round(overlay.shape[0] * scale))
        panel_img = cv2.resize(overlay, (panel_width, ph), interpolation=cv2.INTER_AREA)
        label_h = 112
        panel = np.full((ph + label_h, panel_width, 3), 255, dtype=np.uint8)
        panel[:ph] = panel_img
        def put(line: str, y: int, color: tuple[int, int, int] = (0, 0, 0)) -> None:
            cv2.putText(panel, line[:72], (6, ph + y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
        put(f"rank {rank} frame {row.get('frame_idx')} score {float(row.get('proposal_score',0.0)):.3f}", 16)
        put(f"mask {int(row.get('mask_area_px',0))}px pts {int(row.get('visible_depth_vertex_count',0))} border {bool(row.get('touches_or_near_image_border'))}", 34)
        put(f"hand_removed {100.0*float(row.get('hand_owned_removed_fraction',0.0)):.1f}% comps {int(row.get('component_count',0))}", 52)
        ext = row.get("world_extent_m") if isinstance(row.get("world_extent_m"), list) else []
        if len(ext) >= 3:
            put(f"extent m {float(ext[0]):.3f},{float(ext[1]):.3f},{float(ext[2]):.3f}", 70)
        put(f"depth p05-p95 {float(row.get('depth_spread_p95_p05_m',0.0)):.3f}m", 88)
        put("agent must inspect; score is not acceptance", 106, (40, 40, 180))
        panels.append(panel)
    if not panels:
        return {"status": "no_review_candidates", "blockers": blockers}
    cols = min(3, len(panels))
    rows_img: list[np.ndarray] = []
    for start in range(0, len(panels), cols):
        chunk = panels[start:start + cols]
        max_h = max(p.shape[0] for p in chunk)
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
    return {"status": "ok" if ok else "write_failed", "path": str(review_path), "blockers": blockers[:20]}


def object_row_not_visible(object_id: str, track_id: str, status: str, reason: str, mask_path: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "object_id": object_id,
        "track_id": track_id,
        "label": track_id,
        "status": status,
        "visible": False,
        "v19_visible_geometry_adapter": {
            "state": status,
            "reason": reason,
        },
    }
    if mask_path:
        row["mask_path"] = mask_path
    return row


def remove_existing_object(objects: list[Any], object_id: str, track_id: str) -> list[Any]:
    out = []
    for obj in objects:
        if not isinstance(obj, dict):
            out.append(obj)
            continue
        if obj.get("object_id") == object_id or obj.get("track_id") == track_id:
            continue
        out.append(obj)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_frames, raw_payload = raw_frame_map(args.raw_frame_manifest)
    depth = load_depth_npz(args.depth_npz)
    calibration_intrinsics, calibration_source, calibration_summary, camera_contract_v2 = load_calibration_contract(
        args.calibration_contract,
        frame_ids=sorted(raw_frames),
    )
    depth_contract_binding_validation = validate_depth_camera_contract_binding(
        depth=depth,
        camera_contract_v2=camera_contract_v2,
        calibration_contract_path=args.calibration_contract,
        depth_image_plane=args.depth_image_plane,
        allow_implicit_depth_resize=bool(args.allow_implicit_depth_resize),
    )
    camera_poses = load_camera_npz(args.camera_npz)
    base_frames = load_base_annotations(args.base_annotations)
    sam2, sam2_path = load_sam2_track(args)
    object_plan_record = load_object_plan_record(args.object_plan, args.track_id)
    object_id = args.object_id or f"object:{args.track_id}"
    indices = frame_range(raw_frames, args.frame_start, args.frame_end)
    output_indices = sorted(raw_frames) if bool(args.preserve_source_index) else list(indices)
    rng = np.random.default_rng(int(args.seed))

    visible_data: dict[int, dict[str, Any]] = {}
    appearance_data: dict[int, dict[str, Any]] = {}
    skipped_rows: list[dict[str, Any]] = []
    for idx in indices:
        track_row = sam2.get(idx, {})
        if not track_row.get("visible") or not track_row.get("mask_path"):
            skipped_rows.append({"frame_idx": idx, "status": "not_visible_in_sam2"})
            continue
        depth_i = depth["frame_to_i"].get(idx)
        if depth_i is None:
            skipped_rows.append({"frame_idx": idx, "status": "missing_depth_row"})
            continue
        source_mask_path = str(track_row["mask_path"])
        mask_path = localize_path(source_mask_path, args.remote_root, args.local_root)
        mask = read_mask(mask_path)
        depth_m = np.asarray(depth["depth"][depth_i], dtype=float)
        raw_row = raw_frames[idx]
        base_frame = copy.deepcopy(base_frames.get(idx, {"frame_idx": idx}))
        source_width = int(raw_row.get("source_width") or raw_payload.get("video", {}).get("width") or raw_row.get("manifest_width") or mask.shape[1])
        source_height = int(raw_row.get("source_height") or raw_payload.get("video", {}).get("height") or raw_row.get("manifest_height") or mask.shape[0])
        mask_owned, ownership_summary = subtract_hand_owned_bbox_regions(
            mask,
            base_frame,
            source_width=source_width,
            source_height=source_height,
            pad_px=int(args.hand_bbox_exclusion_pad_px),
            enabled=bool(args.exclude_hand_bboxes),
        )
        owned_mask_path = args.output_dir / "object_owned_masks" / f"{idx:06d}_{safe_name(object_id)}_object_owned_mask.png"
        owned_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(owned_mask_path), mask_owned.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write object-owned mask: {owned_mask_path}")
        if ownership_summary.get("fail_closed") is True:
            skipped_rows.append({
                "frame_idx": idx,
                "status": "projected_mano_hand_ownership_failed_closed",
                "object_surface_ownership_filter": ownership_summary,
                "owned_mask_path": str(owned_mask_path),
            })
            continue
        appearance_data[idx] = {
            "mask_path": str(owned_mask_path),
            "source_mask_path": source_mask_path,
            "object_surface_ownership_filter": ownership_summary,
            "ownership_contract": {
                "mano_subtraction_completed": True,
                "ownership_primitive": ownership_summary.get("ownership_primitive"),
                "bbox_subtraction_used": False,
                "fail_closed": False,
            },
            "source_width": int(source_width),
            "source_height": int(source_height),
            "claim_scope": (
                "Prediction-side RGB/object appearance support after projected MANO silhouette subtraction. "
                "This support remains valid when metric depth ownership fails and does not promote rejected depth."
            ),
        }
        if mask_owned.shape != depth_m.shape:
            A_depth_from_mask = camera_contract_resize_affine(
                (int(mask_owned.shape[1]), int(mask_owned.shape[0])),
                (int(depth_m.shape[1]), int(depth_m.shape[0])),
                pixel_center_convention=args.pixel_center_convention,
            )
            mask_depth_owned = cv2.resize(
                mask_owned.astype(np.uint8),
                (depth_m.shape[1], depth_m.shape[0]),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ) > 0
        else:
            A_depth_from_mask = np.eye(3, dtype=np.float64)
            mask_depth_owned = mask_owned
        mask_depth_transform_contract = {
            "mask_image_plane": args.mask_image_plane,
            "depth_image_plane": args.depth_image_plane,
            "mask_size_wh": [int(mask_owned.shape[1]), int(mask_owned.shape[0])],
            "depth_size_wh": [int(depth_m.shape[1]), int(depth_m.shape[0])],
            "A_depth_from_mask_coordinate_model": A_depth_from_mask.tolist(),
            "raster_resampling": "cv2.resize INTER_NEAREST_EXACT; discrete samples follow the declared OpenCV half-pixel coordinate model",
            "interpolation": "cv2.INTER_NEAREST_EXACT",
            "pixel_center_convention": args.pixel_center_convention,
        }
        intrinsics_transform_contract: dict[str, Any]
        if camera_contract_v2 is not None:
            raw_intrinsics, intrinsics_transform_contract = camera_contract_plane_intrinsics(
                camera_contract_v2["payload"],
                camera_contract_v2["normalized"],
                plane_name=args.depth_image_plane,
                actual_size_wh=(int(depth_m.shape[1]), int(depth_m.shape[0])),
                allow_implicit_resize=bool(args.allow_implicit_depth_resize),
            )
            intr = np.asarray(raw_intrinsics, dtype=float)
            intrinsics_source = calibration_source or "camera_contract_v2"
            _, mask_plane_transform_contract = camera_contract_plane_intrinsics(
                camera_contract_v2["payload"],
                camera_contract_v2["normalized"],
                plane_name=args.mask_image_plane,
                actual_size_wh=(int(mask_owned.shape[1]), int(mask_owned.shape[0])),
                allow_implicit_resize=bool(args.allow_implicit_mask_resize),
            )
            A_depth_from_calibration = np.asarray(
                intrinsics_transform_contract["A_actual_plane_from_calibration"], dtype=np.float64
            )
            A_mask_from_calibration = np.asarray(
                mask_plane_transform_contract["A_actual_plane_from_calibration"], dtype=np.float64
            )
            expected_A_depth_from_mask = A_depth_from_calibration @ np.linalg.inv(A_mask_from_calibration)
            if not np.allclose(A_depth_from_mask, expected_A_depth_from_mask, atol=1.0e-9, rtol=0.0):
                raise RuntimeError(
                    "declared camera-contract mask/depth planes disagree with the actual cv2 mask resize: "
                    f"actual={A_depth_from_mask.tolist()} expected={expected_A_depth_from_mask.tolist()}"
                )
            mask_depth_transform_contract["camera_contract_consistent"] = True
            mask_depth_transform_contract["mask_plane_transform"] = mask_plane_transform_contract
        elif calibration_intrinsics is not None:
            raw_intrinsics = calibration_intrinsics
            intrinsics_source = calibration_source or "legacy_calibration_contract"
            intr = scaled_intrinsics_for_depth(raw_intrinsics, depth_m.shape, depth["source_size"])
            intrinsics_transform_contract = {
                "mode": "legacy_source_size_scaling",
                "depth_image_plane": args.depth_image_plane,
                "legacy_contract_without_explicit_image_planes": True,
            }
        else:
            base_intr = base_camera_intrinsics(base_frame)
            if base_intr is not None:
                raw_intrinsics, intrinsics_source = base_intr
            else:
                raw_intrinsics = depth["intrinsics"][depth_i]
                intrinsics_source = "depth_npz_intrinsics_fx_fy_cx_cy"
            intr = scaled_intrinsics_for_depth(raw_intrinsics, depth_m.shape, depth["source_size"])
            intrinsics_transform_contract = {
                "mode": "legacy_no_camera_contract_fallback",
                "depth_image_plane": args.depth_image_plane,
                "source_size": depth["source_size"].astype(float).tolist() if depth["source_size"] is not None else None,
            }
        valid = mask_depth_owned & np.isfinite(depth_m) & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
        confidence_m = (
            np.asarray(depth["confidence"][depth_i], dtype=float)
            if depth.get("confidence") is not None
            else None
        )
        robust_valid, depth_ownership_summary = robust_first_surface_depth_ownership(
            valid,
            depth_m,
            enabled=bool(args.robust_first_surface_depth_ownership),
            mad_sigma=float(args.first_surface_mad_sigma),
            min_half_width_m=float(args.first_surface_min_half_width_m),
            min_retained_fraction=float(args.first_surface_min_retained_fraction),
            fail_raw_to_robust_extent_ratio=float(args.first_surface_fail_raw_to_robust_extent_ratio),
            intrinsics=intr,
            confidence=confidence_m,
            confidence_seed_percentile=float(args.first_surface_confidence_seed_percentile),
            local_depth_step_max_m=float(args.first_surface_local_depth_step_max_m),
            max_removed_distance_inside_mask_px=float(args.first_surface_max_removed_distance_inside_mask_px),
            max_confidence_flagged_interior_fraction=float(args.first_surface_max_confidence_flagged_interior_fraction),
            min_interior_confidence_flagged_fraction=float(args.first_surface_min_interior_confidence_flagged_fraction),
            max_unexplained_interior_pixels=int(args.first_surface_max_unexplained_interior_pixels),
            min_component_pixels=int(args.first_surface_min_component_pixels),
            max_small_component_fraction=float(args.first_surface_max_small_component_fraction),
        )
        if depth_ownership_summary.get("fail_closed") is True:
            skipped_rows.append({
                "frame_idx": idx,
                "status": "first_surface_depth_ownership_failed_closed",
                "depth_ownership": depth_ownership_summary,
                "owned_mask_path": str(owned_mask_path),
                "appearance_support_preserved": True,
            })
            continue
        valid = robust_valid
        if int(valid.sum()) < int(args.min_valid_points):
            skipped_rows.append({"frame_idx": idx, "status": "too_few_valid_mask_depth_pixels", "valid_pixels": int(valid.sum())})
            continue
        T_world_camera, camera_source = resolve_camera_pose(idx, base_frame, camera_poses, bool(args.allow_camera_frame_world))
        camera_points, world_points, sample_summary = choose_visible_points(
            valid,
            depth_m,
            intr,
            T_world_camera,
            pixel_stride=int(args.pixel_stride),
            max_points=int(args.max_points),
            rng=rng,
        )
        if len(world_points) < int(args.min_valid_points):
            skipped_rows.append({"frame_idx": idx, "status": "too_few_sampled_visible_points", "sampled_points": int(len(world_points))})
            continue
        visible_data[idx] = {
            "mask_path": str(owned_mask_path),
            "source_mask_path": source_mask_path,
            "raw_sam2_mask_path": str(mask_path),
            "track_row": track_row,
            "raw_row": raw_row,
            "camera_points": camera_points,
            "world_points": world_points,
            "intrinsics": intr,
            "intrinsics_source": intrinsics_source,
            "intrinsics_transform_contract": intrinsics_transform_contract,
            "mask_depth_transform_contract": mask_depth_transform_contract,
            "T_world_camera": T_world_camera,
            "camera_source": camera_source,
            "sample_summary": sample_summary,
            "depth_median_m": float(np.median(depth_m[valid])),
            "depth_p05_m": float(np.percentile(depth_m[valid], 5.0)),
            "depth_p95_m": float(np.percentile(depth_m[valid], 95.0)),
            "mask_shape": list(mask_owned.shape),
            "raw_sam2_mask_shape": list(mask.shape),
            "depth_shape": list(depth_m.shape),
            "object_surface_ownership_filter": ownership_summary,
            "first_surface_depth_ownership": depth_ownership_summary,
            "source_width": int(source_width),
            "source_height": int(source_height),
        }

    if not visible_data:
        raise RuntimeError(
            f"no visible metric geometry rows for track_id={args.track_id}. "
            "Check SAM2 masks, depth archive coverage, frame range, and camera-pose availability."
        )

    anchor_candidate_report = build_anchor_candidate_proposals(args=args, object_id=object_id, visible_data=visible_data)
    if bool(args.propose_anchor_candidates_only):
        return {
            "method": "build_v19_visible_geometry_from_sam2_depth",
            "status": "anchor_candidates_proposed_only",
            "case": args.case,
            "track_id": args.track_id,
            "object_id": object_id,
            "claim_scope": "Candidate anchor evidence only. No canonical anchor mesh, visible-geometry annotations, completion, pose, or render state was produced.",
            "outputs": anchor_candidate_report.get("outputs", {}),
            "candidate_count": anchor_candidate_report.get("candidate_count"),
            "review_candidate_count": anchor_candidate_report.get("review_candidate_count"),
        }

    if args.anchor_frame is not None:
        anchor = int(args.anchor_frame)
        if anchor not in visible_data:
            raise RuntimeError(f"--anchor-frame {anchor} has no visible metric geometry")
    else:
        if bool(args.require_anchor_frame):
            outputs = anchor_candidate_report.get("outputs", {}) if isinstance(anchor_candidate_report, dict) else {}
            raise RuntimeError(
                "--anchor-frame is required for this run. Inspect anchor candidates and rerun with an agent-selected anchor. "
                f"candidate_report={outputs.get('anchor_candidate_proposals')} review={outputs.get('anchor_candidate_review')}"
            )
        anchor = max(visible_data, key=lambda idx: len(visible_data[idx]["world_points"]))
    anchor_points = np.asarray(visible_data[anchor]["world_points"], dtype=float)
    anchor_centroid = anchor_points.mean(axis=0)
    anchor_extent_m = anchor_points.max(axis=0) - anchor_points.min(axis=0)
    anchor_diag_m = float(np.linalg.norm(anchor_extent_m))
    if anchor_diag_m <= 0.0 or not np.isfinite(anchor_diag_m):
        raise RuntimeError(f"anchor frame {anchor} has invalid metric extent")
    population_extents = np.asarray(
        [
            np.ptp(np.asarray(row["world_points"], dtype=float), axis=0)
            for row in visible_data.values()
        ],
        dtype=float,
    )
    population_sorted_extents = np.sort(population_extents, axis=1)[:, ::-1]
    population_sorted_extent_median = np.median(population_sorted_extents, axis=0)
    population_diag = np.linalg.norm(population_extents, axis=1)
    population_diag_median = float(np.median(population_diag))
    if (
        population_sorted_extent_median.shape != (3,)
        or not np.isfinite(population_sorted_extent_median).all()
        or np.any(population_sorted_extent_median <= 0.0)
        or not np.isfinite(population_diag_median)
        or population_diag_median <= 0.0
    ):
        raise RuntimeError("visible metric extent population is invalid")
    anchor_mesh_reconstruction = export_anchor_visible_surface_mesh(
        output_dir=args.output_dir,
        object_id=object_id,
        anchor_frame=int(anchor),
        anchor_centroid_world_m=anchor_centroid,
        anchor_points_world_m=anchor_points,
        min_voxel_m=float(args.anchor_mesh_min_voxel_m),
        voxel_divisor=float(args.anchor_mesh_voxel_divisor),
        poisson_depth=int(args.anchor_mesh_poisson_depth),
        poisson_density_quantile=float(args.anchor_mesh_poisson_density_quantile),
    )

    last_pose: dict[str, Any] | None = None
    output_frames: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    camera_source_counts: Counter[str] = Counter()
    for idx in output_indices:
        raw_row = raw_frames[idx]
        frame = copy.deepcopy(base_frames.get(idx, {"frame_idx": idx}))
        frame["frame_idx"] = int(idx)
        raw_text = str(frame.get("raw_frame_path") or raw_row.get("rgb") or "")
        frame["raw_frame_path"] = str(localize_path(raw_text, args.remote_root, args.local_root)) if raw_text else raw_text
        frame["source_width"] = int(raw_row.get("source_width") or raw_payload.get("video", {}).get("width") or raw_row.get("manifest_width") or 0)
        frame["source_height"] = int(raw_row.get("source_height") or raw_payload.get("video", {}).get("height") or raw_row.get("manifest_height") or 0)
        try:
            T_world_camera, camera_source = resolve_camera_pose(idx, frame, camera_poses, bool(args.allow_camera_frame_world))
            camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
            camera = dict(camera)
            camera["T_world_camera_metric"] = T_world_camera.astype(float).tolist()
            camera["position_world_m"] = T_world_camera[:3, 3].astype(float).tolist()
            camera["v19_camera_pose_source"] = camera_source
            frame["camera"] = camera
            camera_source_counts[camera_source] += 1
        except RuntimeError:
            if idx in visible_data:
                raise
        objects = remove_existing_object(as_list(frame.get("objects")), object_id, args.track_id)
        vis = visible_data.get(idx)
        if vis is None:
            track_row = sam2.get(idx, {})
            appearance = appearance_data.get(idx)
            appearance_mask = (
                str(appearance.get("mask_path"))
                if isinstance(appearance, dict) and appearance.get("mask_path")
                else (str(track_row.get("mask_path")) if track_row.get("mask_path") else None)
            )
            row = object_row_not_visible(
                object_id,
                args.track_id,
                "not_visible_or_no_metric_depth",
                "SAM2/depth did not provide an accepted visible metric surface for this frame",
                appearance_mask,
            )
            if isinstance(appearance, dict):
                row["appearance_observation"] = {
                    **appearance,
                    "status": "object_owned_rgb_appearance_support_without_accepted_metric_depth",
                    "metric_depth_pose_eligible": False,
                    "rgb_tracking_pose_evidence_eligible": True,
                }
                row["object_surface_ownership_filter"] = appearance.get("object_surface_ownership_filter")
                row["mask_path"] = appearance_mask
                row["mask_semantics"] = "projected_mano_subtracted_object_owned_appearance_mask"
            if bool(args.carry_invisible_pose) and last_pose is not None:
                row["reconstructed_geometry_pose"] = {
                    **last_pose,
                    "pose_source": "carried_nearest_visible_centroid_uncertain_v19_adapter",
                    "pose_uncertainty": "no current visible metric surface; pose is a carried prior for rendering/optimization initialization only",
                }
            objects.append(row)
            frame["objects"] = objects
            output_frames.append(frame)
            rows.append({"frame_idx": idx, "status": "not_visible_or_no_metric_depth"})
            continue
        world_points = np.asarray(vis["world_points"], dtype=float)
        cam_points = np.asarray(vis["camera_points"], dtype=float)
        centroid = world_points.mean(axis=0)
        world_extent_m = world_points.max(axis=0) - world_points.min(axis=0)
        sorted_extent_m = np.sort(world_extent_m)[::-1]
        extent_ratio_diag = float(
            max(
                np.linalg.norm(world_extent_m) / max(population_diag_median, 1.0e-9),
                population_diag_median / max(np.linalg.norm(world_extent_m), 1.0e-9),
            )
        )
        extent_ratio_axis = np.maximum(
            sorted_extent_m / np.maximum(population_sorted_extent_median, 1.0e-6),
            population_sorted_extent_median / np.maximum(sorted_extent_m, 1.0e-6),
        )
        max_extent_ratio_axis = float(np.max(extent_ratio_axis))
        rigid_pose_observation_eligible = bool(
            np.isfinite(extent_ratio_diag)
            and np.isfinite(max_extent_ratio_axis)
            and extent_ratio_diag <= float(args.rigid_extent_ratio_max)
            and max_extent_ratio_axis <= float(args.rigid_extent_axis_ratio_max)
        )
        rigid_pose_observation_reason = (
            "metric_extent_consistent_with_orientation_invariant_visible_population_reference"
            if rigid_pose_observation_eligible
            else "systematic_mask_extent_inconsistent_with_orientation_invariant_visible_population_reference_probable_hand_background_leakage"
        )
        raw_mask = read_mask(Path(vis["mask_path"]))
        owned_bbox = bbox_xyxy_from_mask(raw_mask, int(frame["source_width"]), int(frame["source_height"]))
        owned_area_source_px = float(
            int(raw_mask.sum())
            * float(frame["source_width"]) / float(max(1, raw_mask.shape[1]))
            * float(frame["source_height"]) / float(max(1, raw_mask.shape[0]))
        )
        if not owned_bbox:
            raise RuntimeError(f"frame {idx} object-owned mask unexpectedly has no support")
        pose = {
            "rotation_world_from_canonical_matrix": np.eye(3, dtype=float).tolist(),
            "translation_world_m": centroid.astype(float).tolist(),
            "pose_source": "v19_visible_geometry_adapter_centroid_initial_pose_not_final_rigid_pose",
            "pose_uncertainty": "centroid-only initialization from current visible SAM2/depth surfels; rigid completion and pose fitting must refine it",
            "anchor_frame_idx": int(anchor),
            "anchor_centroid_world_m": anchor_centroid.astype(float).tolist(),
        }
        last_pose = pose
        geom = {
            "status": "visible_surface_from_sam2_mask_metric_depth",
            "source": "build_v19_visible_geometry_from_sam2_depth",
            "frame_idx": int(idx),
            "object_id": object_id,
            "track_id": args.track_id,
            "mask_path": str(vis["mask_path"]),
            "source_mask_path": str(vis.get("source_mask_path")),
            "source_width": int(vis.get("source_width") or frame.get("source_width") or 0),
            "source_height": int(vis.get("source_height") or frame.get("source_height") or 0),
            "depth_npz": str(args.depth_npz),
            "depth_frame_index": int(idx),
            "camera_pose_source": vis["camera_source"],
            "intrinsics_fx_fy_cx_cy": np.asarray(vis["intrinsics"], dtype=float).tolist(),
            "intrinsics_source": vis.get("intrinsics_source"),
            "intrinsics_transform_contract": vis.get("intrinsics_transform_contract"),
            "mask_depth_transform_contract": vis.get("mask_depth_transform_contract"),
            "vertex_count": int(len(world_points)),
            "world_vertices_sample_m": world_points.astype(float).tolist(),
            "camera_vertices_sample_m": cam_points.astype(float).tolist(),
            "centroid_world_m": centroid.astype(float).tolist(),
            "world_extent_m": world_extent_m.astype(float).tolist(),
            "anchor_extent_world_m": anchor_extent_m.astype(float).tolist(),
            "extent_population_sorted_axis_median_m": population_sorted_extent_median.astype(float).tolist(),
            "extent_population_diag_median_m": population_diag_median,
            "extent_consistency_basis": "frame-population median of orientation-invariant sorted extents and extent diagonal",
            "extent_ratio_to_population_diag": extent_ratio_diag,
            "extent_ratio_to_population_sorted_axis": extent_ratio_axis.astype(float).tolist(),
            "rigid_pose_observation_eligible": rigid_pose_observation_eligible,
            "rigid_pose_observation_reason": rigid_pose_observation_reason,
            "depth_median_m": float(vis["depth_median_m"]),
            "depth_p05_m": float(vis["depth_p05_m"]),
            "depth_p95_m": float(vis["depth_p95_m"]),
            "sample_summary": vis["sample_summary"],
            "object_surface_ownership_filter": vis.get("object_surface_ownership_filter"),
            "first_surface_depth_ownership": vis.get("first_surface_depth_ownership"),
            "claim_scope": "visible metric surface measurement only; hand-owned and non-first-surface depth pixels are excluded before metric lifting; not hidden geometry and not final object pose",
        }
        row_obj = {
            "object_id": object_id,
            "track_id": args.track_id,
            "label": args.track_id,
            "description": (object_plan_record or {}).get("description"),
            "status": "visible_metric_surface_measurement",
            "visible": True,
            "mask_path": str(vis["mask_path"]),
            "bbox_xyxy": [float(x) for x in owned_bbox[:4]],
            "area_px": owned_area_source_px,
            "raw_sam2_bbox_xyxy": vis["track_row"].get("bbox_xyxy"),
            "raw_sam2_area_px": vis["track_row"].get("area_px"),
            "mask_metadata_coordinate_frame": "source_image_pixels_recomputed_from_object_owned_mask",
            "depth_m": float(vis["depth_median_m"]),
            "visible_geometry_candidate": geom,
            "reconstructed_geometry_pose": pose,
            "v19_physical_model": (object_plan_record or {}).get("physical_model"),
            "rigid_pose_observation_eligible": rigid_pose_observation_eligible,
            "rigid_pose_observation_reason": rigid_pose_observation_reason,
        }
        objects.append(row_obj)
        frame["objects"] = objects
        output_frames.append(frame)
        rows.append(
            {
                "frame_idx": int(idx),
                "status": "visible_metric_surface_measurement",
                "vertex_count": int(len(world_points)),
                "mask_path": str(vis["mask_path"]),
                "owned_mask_area_source_px": owned_area_source_px,
                "owned_mask_bbox_source_xyxy": [float(x) for x in owned_bbox[:4]],
                "depth_median_m": float(vis["depth_median_m"]),
                "centroid_world_m": centroid.astype(float).tolist(),
                "world_extent_m": world_extent_m.astype(float).tolist(),
                "extent_ratio_to_population_diag": extent_ratio_diag,
                "extent_ratio_to_population_sorted_axis_max": max_extent_ratio_axis,
                "rigid_pose_observation_eligible": rigid_pose_observation_eligible,
                "rigid_pose_observation_reason": rigid_pose_observation_reason,
                "camera_pose_source": vis["camera_source"],
                "intrinsics_source": vis.get("intrinsics_source"),
                "intrinsics_transform_contract": vis.get("intrinsics_transform_contract"),
                "mask_depth_transform_contract": vis.get("mask_depth_transform_contract"),
            }
        )

    annotations = {
        "frames": output_frames,
        "v19_visible_geometry_adapter": {
            "method": "build_v19_visible_geometry_from_sam2_depth",
            "case": args.case,
            "track_id": args.track_id,
            "object_id": object_id,
            "raw_frame_manifest": str(args.raw_frame_manifest),
            "sam2_track": str(sam2_path),
            "depth_npz": str(args.depth_npz),
            "base_annotations": str(args.base_annotations) if args.base_annotations else None,
            "camera_npz": str(args.camera_npz) if args.camera_npz else None,
            "calibration_contract": str(args.calibration_contract) if args.calibration_contract else None,
            "depth_image_plane": args.depth_image_plane,
            "mask_image_plane": args.mask_image_plane,
            "pixel_center_convention": args.pixel_center_convention,
            "depth_camera_contract_binding_validation": depth_contract_binding_validation,
            "anchor_frame_idx": int(anchor),
            "claim_scope": "visible metric surfel and initial-pose adapter for rigid branch; downstream completion/pose/interval solvers must produce the physical object pose claim",
        },
    }
    if isinstance(raw_payload.get("video"), dict):
        annotations["raw_video"] = raw_payload["video"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / "annotations_v19_visible_geometry.json"
    write_json(annotations_path, annotations)

    visible_mask_rows = [
        {
            "frame_idx": int(row["frame_idx"]),
            "target_entity_id": object_id,
            "object_id": object_id,
            "track_id": args.track_id,
            "status": "v19_visible_metric_surface_mask",
            "measurement_type": "sam2_mask_with_metric_depth_support",
            "mask_path": str(row["mask_path"]),
            "saved_mask_path": str(row["mask_path"]),
            "mask_area_px": row.get("owned_mask_area_source_px"),
            "mask_bbox_xyxy": row.get("owned_mask_bbox_source_xyxy"),
            "raw_sam2_mask_area_px": None if sam2.get(int(row["frame_idx"]), {}).get("area_px") is None else float(sam2[int(row["frame_idx"])] ["area_px"]),
            "depth_median_m": row.get("depth_median_m"),
            "visible_vertex_count": row.get("vertex_count"),
            "coordinate_frame": "source_image_mask_plus_metric_depth",
            "claim_scope": "visible model mask measurement with metric-depth support; not hidden geometry or object pose",
        }
        for row in rows
        if row.get("status") == "visible_metric_surface_measurement" and row.get("mask_path")
    ]
    visible_mask_report_path = args.output_dir / "v19_visible_mask_report.json"
    visible_mask_report = {
        "method": "build_v19_visible_geometry_from_sam2_depth_visible_mask_report",
        "status": "ok",
        "case": args.case,
        "target_entity_id": object_id,
        "track_id": args.track_id,
        "claim_scope": "SAM2 mask rows with metric depth support for downstream visible-surface/ownership factor builders. This report is a mask measurement index, not object pose.",
        "surface_rows": visible_mask_rows,
    }
    write_json(visible_mask_report_path, visible_mask_report)

    report = {
        "method": "build_v19_visible_geometry_from_sam2_depth",
        "status": "ok",
        "case": args.case,
        "track_id": args.track_id,
        "object_id": object_id,
        "claim_scope": "SAM2 masks plus metric depth/camera are lifted to visible object surfels and centroid initial poses. This is measurement/adaptation input for the rigid branch, not final object pose.",
        "inputs": {
            "raw_frame_manifest": str(args.raw_frame_manifest),
            "sam2_track": str(sam2_path),
            "depth_npz": str(args.depth_npz),
            "base_annotations": str(args.base_annotations) if args.base_annotations else None,
            "camera_npz": str(args.camera_npz) if args.camera_npz else None,
            "calibration_contract": str(args.calibration_contract) if args.calibration_contract else None,
            "depth_image_plane": args.depth_image_plane,
            "mask_image_plane": args.mask_image_plane,
            "pixel_center_convention": args.pixel_center_convention,
            "object_plan": str(args.object_plan) if args.object_plan else None,
            "remote_root": str(args.remote_root) if args.remote_root else None,
            "local_root": str(args.local_root) if args.local_root else None,
        },
        "outputs": {
            "annotations": str(annotations_path),
            "depth_fused_report": str(args.output_dir / "v19_visible_geometry_depth_fused_report.json"),
            "visible_mask_report": str(visible_mask_report_path),
            "anchor_candidate_proposals": anchor_candidate_report.get("outputs", {}).get("anchor_candidate_proposals") if isinstance(anchor_candidate_report, dict) else None,
            "anchor_candidate_review": anchor_candidate_report.get("outputs", {}).get("anchor_candidate_review") if isinstance(anchor_candidate_report, dict) else None,
            "anchor_visible_surface_mesh": anchor_mesh_reconstruction.get("poisson_mesh_path") or anchor_mesh_reconstruction.get("convex_hull_mesh_path") or anchor_mesh_reconstruction.get("fused_point_cloud_path"),
        },
        "requested_frame_start": int(indices[0]),
        "requested_frame_end": int(indices[-1]),
        "requested_frame_count": int(len(indices)),
        "output_frame_count": int(len(output_indices)),
        "preserve_source_index": bool(args.preserve_source_index),
        "visible_metric_frame_count": int(sum(1 for row in rows if row.get("status") == "visible_metric_surface_measurement")),
        "anchor_frame_idx": int(anchor),
        "anchor_centroid_world_m": anchor_centroid.astype(float).tolist(),
        "anchor_extent_world_m": anchor_extent_m.astype(float).tolist(),
        "extent_consistency_reference": {
            "method": "orientation_invariant_visible_population_median",
            "sorted_axis_extent_median_m": population_sorted_extent_median.astype(float).tolist(),
            "extent_diag_median_m": population_diag_median,
            "frame_count": int(len(population_extents)),
            "anchor_not_used_as_extent_gate_reference": True,
        },
        "anchor_visible_surface_mesh_reconstruction": anchor_mesh_reconstruction,
        "camera_pose_source_counts": dict(camera_source_counts),
        "intrinsics_source_counts": dict(Counter(str(vis.get("intrinsics_source")) for vis in visible_data.values())),
        "calibration_contract": calibration_summary,
        "depth_camera_contract_binding_validation": depth_contract_binding_validation,
        "parameters": {
            "allow_implicit_depth_resize": bool(args.allow_implicit_depth_resize),
            "allow_implicit_mask_resize": bool(args.allow_implicit_mask_resize),
            "depth_image_plane": args.depth_image_plane,
            "mask_image_plane": args.mask_image_plane,
            "pixel_center_convention": args.pixel_center_convention,
            "pixel_stride": int(args.pixel_stride),
            "max_points": int(args.max_points),
            "min_valid_points": int(args.min_valid_points),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "allow_camera_frame_world": bool(args.allow_camera_frame_world),
            "carry_invisible_pose": bool(args.carry_invisible_pose),
            "preserve_source_index": bool(args.preserve_source_index),
            "exclude_hand_regions": bool(args.exclude_hand_bboxes),
            "hand_ownership_primitive": "projected_full_hawor_mano_triangle_silhouette",
            "hand_bbox_subtraction_used": False,
            "hand_silhouette_exclusion_pad_px": int(args.hand_bbox_exclusion_pad_px),
            "robust_first_surface_depth_ownership": bool(args.robust_first_surface_depth_ownership),
            "first_surface_mad_sigma": float(args.first_surface_mad_sigma),
            "first_surface_min_half_width_m": float(args.first_surface_min_half_width_m),
            "first_surface_min_retained_fraction": float(args.first_surface_min_retained_fraction),
            "first_surface_confidence_seed_percentile": float(args.first_surface_confidence_seed_percentile),
            "first_surface_local_depth_step_max_m": float(args.first_surface_local_depth_step_max_m),
            "first_surface_max_removed_distance_inside_mask_px": float(args.first_surface_max_removed_distance_inside_mask_px),
            "first_surface_max_confidence_flagged_interior_fraction": float(args.first_surface_max_confidence_flagged_interior_fraction),
            "first_surface_min_interior_confidence_flagged_fraction": float(args.first_surface_min_interior_confidence_flagged_fraction),
            "first_surface_max_unexplained_interior_pixels": int(args.first_surface_max_unexplained_interior_pixels),
            "first_surface_min_component_pixels": int(args.first_surface_min_component_pixels),
            "first_surface_max_small_component_fraction": float(args.first_surface_max_small_component_fraction),
            "first_surface_fail_raw_to_robust_extent_ratio": float(args.first_surface_fail_raw_to_robust_extent_ratio),
            "rigid_extent_ratio_max": float(args.rigid_extent_ratio_max),
            "rigid_extent_axis_ratio_max": float(args.rigid_extent_axis_ratio_max),
            "anchor_candidate_count": int(args.anchor_candidate_count),
            "anchor_candidate_min_gap": int(args.anchor_candidate_min_gap),
            "require_anchor_frame": bool(args.require_anchor_frame),
        },
        "rows": rows,
        "skipped_rows_preview": skipped_rows[:200],
    }
    report_path = args.output_dir / "v19_visible_geometry_adapter_report.json"
    write_json(report_path, report)

    depth_fused = {
        "method": "build_v19_visible_geometry_from_sam2_depth_depth_fused_compat_report",
        "status": "ok",
        "case": args.case,
        "claim": "Compatibility report for V18 rigid evidence bundle. Visible geometry lives in annotations; no hidden geometry is claimed here.",
        "object_rows": [
            {
                "object_id": object_id,
                "track_id": args.track_id,
                "frame_surface_rows": report["visible_metric_frame_count"],
                "visible_geometry_adapter_report": str(report_path),
                "visible_mask_report": str(visible_mask_report_path),
                "annotations": str(annotations_path),
                "mesh_reconstruction": anchor_mesh_reconstruction,
                "object_geometry_complete": False,
                "hidden_geometry_reconstructed": False,
                "complete_object_pose_ready": False,
            }
        ],
    }
    write_json(args.output_dir / "v19_visible_geometry_depth_fused_report.json", depth_fused)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, default=Path("."))
    parser.add_argument("--sam2-track-json", type=Path, default=None)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-annotations", type=Path, default=None)
    parser.add_argument("--camera-npz", type=Path, default=None)
    parser.add_argument("--calibration-contract", type=Path, default=None, help="V19 camera contract. V2 contracts explicitly map calibration pixels into the selected depth image plane; legacy constant-K contracts remain supported.")
    parser.add_argument("--depth-image-plane", default="source_rgb", help="Named image plane in a V2 camera contract corresponding to the depth raster.")
    parser.add_argument("--mask-image-plane", default="sam2_mask", help="Named image plane in a V2 camera contract corresponding to the SAM2/object-owned mask raster.")
    parser.add_argument("--pixel-center-convention", choices=["integer_pixel_centers_opencv", "pixel_corner_origin"], default="integer_pixel_centers_opencv")
    parser.add_argument("--allow-implicit-depth-resize", action="store_true", help="Compatibility override when the actual depth raster differs from the declared plane. Prefer an explicit image plane in the contract.")
    parser.add_argument("--allow-implicit-mask-resize", action="store_true", help="Compatibility override when the actual mask raster differs from the declared plane. Prefer an explicit image plane in the contract.")
    parser.add_argument("--object-plan", type=Path, default=None)
    parser.add_argument("--remote-root", type=Path, default=None, help="Remote path prefix to localize mask/raw paths from server-produced manifests")
    parser.add_argument("--local-root", type=Path, default=None, help="Local path prefix corresponding to --remote-root")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--anchor-frame", type=int, default=None)
    parser.add_argument("--propose-anchor-candidates-only", action="store_true", help="Lift visible mask/depth rows, write anchor_candidate_proposals.json and anchor_candidate_review.jpg, then stop before exporting canonical anchor geometry. The agent must inspect and choose an anchor.")
    parser.add_argument("--require-anchor-frame", action="store_true", help="Fail instead of falling back to the max-point frame when --anchor-frame is absent. Use after candidate proposal so anchor choice is explicit.")
    parser.add_argument("--anchor-candidate-count", type=int, default=12, help="Number of diversified candidate frames to show in the visual anchor review sheet.")
    parser.add_argument("--anchor-candidate-min-gap", type=int, default=8, help="Minimum frame gap used when diversifying review-sheet anchor candidates; ranked JSON still contains every candidate.")
    parser.add_argument("--anchor-candidate-panel-width", type=int, default=360, help="Width in pixels for each candidate panel in anchor_candidate_review.jpg.")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--min-valid-points", type=int, default=50)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--allow-camera-frame-world", action="store_true", help="Explicitly use each camera frame as its own world frame when no world camera pose is available. This is not valid for temporal metric world claims.")
    parser.add_argument("--carry-invisible-pose", action="store_true", help="Carry the nearest visible centroid pose into invisible frames as an uncertain initialization only.")
    parser.add_argument("--preserve-source-index", action=argparse.BooleanOptionalAction, default=True, help="Write one output frame row per raw source frame so annotations['frames'][frame_idx] remains valid for V18 rigid tools.")
    parser.add_argument(
        "--exclude-hand-bboxes",
        "--exclude-hand-regions",
        dest="exclude_hand_bboxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract projected full HaWoR/MANO triangle silhouettes before lifting object depth; coarse hand bboxes are never ownership masks.",
    )
    parser.add_argument(
        "--hand-bbox-exclusion-pad-px",
        type=int,
        default=4,
        help="Legacy option name: padding in mask pixels around projected MANO hand silhouettes (not bboxes).",
    )
    parser.add_argument("--robust-first-surface-depth-ownership", action=argparse.BooleanOptionalAction, default=True, help="Select confidence/depth-geodesic first-surface surfels per connected object component; appearance masks remain unchanged.")
    parser.add_argument("--first-surface-mad-sigma", type=float, default=2.5, help="Per connected component: MAD sigma used only for the initial depth seed before confidence/geodesic growth.")
    parser.add_argument("--first-surface-min-half-width-m", type=float, default=0.03, help="Minimum per-component seed half-width in meters.")
    parser.add_argument("--first-surface-min-retained-fraction", type=float, default=0.90, help="Fail closed when validated depth-coherent support retains less than this fraction of valid owned depth.")
    parser.add_argument("--first-surface-confidence-seed-percentile", type=float, default=95.0, help="Maximum traversable UniDepth error proxy is derived from this percentile of each component seed.")
    parser.add_argument("--first-surface-local-depth-step-max-m", type=float, default=0.005, help="Maximum adjacent-pixel depth step during geodesic support growth.")
    parser.add_argument("--first-surface-max-removed-distance-inside-mask-px", type=float, default=10.0, help="Distance defining the ordinary boundary quarantine band; deeper rejection must pass sparse/confidence-flagged interior gates.")
    parser.add_argument("--first-surface-max-confidence-flagged-interior-fraction", type=float, default=0.015, help="Maximum fraction of valid owned depth that may be quarantined beyond the boundary band when UniDepth flags it as high predicted error.")
    parser.add_argument("--first-surface-min-interior-confidence-flagged-fraction", type=float, default=0.95, help="Required fraction of nontrivial interior rejection whose UniDepth predicted-error proxy exceeds its component seed threshold.")
    parser.add_argument("--first-surface-max-unexplained-interior-pixels", type=int, default=5, help="Permit only this many isolated interior raster samples without the confidence-flagged proof.")
    parser.add_argument("--first-surface-min-component-pixels", type=int, default=20)
    parser.add_argument("--first-surface-max-small-component-fraction", type=float, default=0.01)
    parser.add_argument("--first-surface-fail-raw-to-robust-extent-ratio", type=float, default=2.0, help="Diagnostic tail-dominance ratio; a larger raw extent is accepted only when support and boundary-localization quarantine gates pass.")
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--anchor-mesh-min-voxel-m", type=float, default=0.002)
    parser.add_argument("--anchor-mesh-voxel-divisor", type=float, default=80.0)
    parser.add_argument("--anchor-mesh-poisson-depth", type=int, default=7)
    parser.add_argument("--anchor-mesh-poisson-density-quantile", type=float, default=0.02)
    parser.add_argument("--rigid-extent-ratio-max", type=float, default=2.75, help="Mark visible surfaces whose orientation-invariant extent diagonal differs from the robust visible-population median by more than this factor as ineligible for rigid pose fitting.")
    parser.add_argument("--rigid-extent-axis-ratio-max", type=float, default=3.25, help="Axis-wise companion to --rigid-extent-ratio-max for detecting elongated hand/background leakage.")
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "skipped_rows_preview"}}, indent=2))


if __name__ == "__main__":
    main()
