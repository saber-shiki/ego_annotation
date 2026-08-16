#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import open3d as o3d
except ImportError:  # Evidence selection remains importable for lightweight contract tests.
    o3d = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def mask_bool(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8) > 0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_pad_bbox(box: tuple[int, int, int, int], width: int, height: int, pad_frac: float = 0.08) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
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
    return sx0, sy0, sx1, sy1


def make_rgba_crop(raw_path: Path, mask_path: Path, out_path: Path) -> dict[str, Any]:
    raw = Image.open(raw_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L").resize(raw.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_img, dtype=np.uint8) > 0
    box = bbox_from_mask(mask)
    if box is None:
        raise RuntimeError(f"empty mask: {mask_path}")
    crop_box = square_pad_bbox(box, raw.width, raw.height)
    raw_arr = np.asarray(raw, dtype=np.uint8).copy()
    raw_arr[~mask] = 0
    rgba = Image.fromarray(np.dstack([raw_arr, mask.astype(np.uint8) * 255]), mode="RGBA")
    crop = rgba.crop(crop_box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return {
        "raw_image": str(raw_path),
        "mask": str(mask_path),
        "crop_rgba": str(out_path),
        "source_size": [raw.width, raw.height],
        "mask_bbox_xyxy": list(box),
        "crop_bbox_xyxy": list(crop_box),
        "mask_area_px": int(mask.sum()),
        "crop_size": list(crop.size),
    }


def find_depth_fused_object(report_path: Path, object_id: str) -> dict[str, Any]:
    report = load_json(report_path)
    for row in report.get("object_rows", []):
        if isinstance(row, dict) and row.get("object_id") == object_id:
            return row
    return {}


def build_selected_anchor_mesh_reconstruction(
    *,
    selected: dict[str, Any],
    output_dir: Path,
    object_id: str,
    min_voxel_m: float = 0.002,
    voxel_divisor: float = 80.0,
    poisson_depth: int = 7,
    poisson_density_quantile: float = 0.02,
) -> dict[str, Any]:
    """Materialize canonical observed geometry from the exact selected P11 row.

    An anchor override is atomic: RGB, mask, camera, metric surfels, centroid,
    and canonical observed mesh must all name the same frame. Reusing P09's mesh
    for another frame mixes object poses and invalidates P13 alignment/residuals.
    """
    if o3d is None:
        raise RuntimeError("Open3D is required to bind the selected P11 canonical surface")
    frame_idx = int(selected["frame_idx"])
    geom = selected.get("visible_geometry_candidate")
    if not isinstance(geom, dict):
        raise RuntimeError("selected P11 row lacks visible_geometry_candidate")
    geom_frame_idx = int(geom.get("frame_idx", -1))
    if geom_frame_idx != frame_idx:
        raise RuntimeError(
            f"selected P11 frame {frame_idx} disagrees with visible geometry frame {geom_frame_idx}"
        )
    if geom.get("rigid_pose_observation_eligible") is not True:
        raise RuntimeError("selected P11 row is not eligible metric-surface evidence")
    points_world = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float64)
    centroid_world = np.asarray(geom.get("centroid_world_m") or [], dtype=np.float64).reshape(-1)
    if (
        points_world.ndim != 2
        or points_world.shape[1] != 3
        or len(points_world) < 30
        or not np.isfinite(points_world).all()
    ):
        raise RuntimeError("selected P11 row lacks at least 30 finite world surfels")
    if centroid_world.shape != (3,) or not np.isfinite(centroid_world).all():
        raise RuntimeError("selected P11 row lacks a finite metric centroid")
    recomputed_centroid = points_world.mean(axis=0)
    centroid_error_m = float(np.linalg.norm(recomputed_centroid - centroid_world))
    if centroid_error_m > 1.0e-6:
        raise RuntimeError(
            f"selected P11 centroid disagrees with its world surfels by {centroid_error_m} m"
        )

    points_canonical = points_world - centroid_world[None, :]
    mesh_dir = output_dir / "selected_anchor_visible_surface_mesh" / safe_id(object_id)
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
        raise RuntimeError("selected P11 surfels collapse below 30 points after fixed voxelization")

    stem = f"frame_{frame_idx:06d}_{safe_id(object_id)}_selected_anchor"
    pcd_path = mesh_dir / f"{stem}_visible_points_canonical.ply"
    if not o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write selected P11 point cloud: {pcd_path}")

    hull, _ = pcd.compute_convex_hull()
    hull.compute_vertex_normals()
    hull_path = mesh_dir / f"{stem}_convex_hull_visible_candidate.ply"
    if not o3d.io.write_triangle_mesh(str(hull_path), hull, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write selected P11 convex hull: {hull_path}")

    poisson_path: Path | None = None
    poisson_error: str | None = None
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(voxel_size_m * 4.0, float(min_voxel_m) * 4.0), max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(20)
        poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(poisson_depth)
        )
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
            raise RuntimeError("empty Poisson result")
        poisson.compute_vertex_normals()
        poisson_path = mesh_dir / f"{stem}_poisson_visible_mesh.ply"
        if not o3d.io.write_triangle_mesh(str(poisson_path), poisson, write_ascii=False, compressed=False):
            raise RuntimeError("Open3D write_triangle_mesh returned false")
    except Exception as exc:
        poisson_error = f"{type(exc).__name__}:{exc}"
        poisson_path = None

    return {
        "status": (
            "selected_p11_anchor_visible_surface_mesh_exported"
            if poisson_path is not None
            else "selected_p11_anchor_visible_surface_hull_only_poisson_failed"
        ),
        "coordinate_frame": "object_canonical_selected_anchor_centroid_frame",
        "canonical_coordinate_source": (
            "selected P11 frame world_vertices_sample_m minus that same row centroid_world_m"
        ),
        "atomic_anchor_binding": True,
        "anchor_frame_idx": frame_idx,
        "anchor_centroid_world_m": centroid_world.astype(float).tolist(),
        "centroid_recomputed_error_m": centroid_error_m,
        "point_count_input": int(len(points_world)),
        "point_count_downsampled": int(len(down_points)),
        "voxel_size_m": float(voxel_size_m),
        "canonical_bbox_min_m": down_points.min(axis=0).astype(float).tolist(),
        "canonical_bbox_max_m": down_points.max(axis=0).astype(float).tolist(),
        "fused_point_cloud_path": str(pcd_path),
        "poisson_mesh_path": str(poisson_path) if poisson_path is not None else None,
        "convex_hull_mesh_path": str(hull_path),
        "poisson_error": poisson_error,
        "claim_scope": "selected-frame partial metric observed surface; no hidden geometry",
    }


def validate_selected_anchor_binding(
    *, selected: dict[str, Any], mesh_reconstruction: dict[str, Any]
) -> dict[str, Any]:
    frame_idx = int(selected["frame_idx"])
    geom = selected.get("visible_geometry_candidate") or {}
    selected_centroid = np.asarray(geom.get("centroid_world_m") or [], dtype=np.float64)
    mesh_centroid = np.asarray(mesh_reconstruction.get("anchor_centroid_world_m") or [], dtype=np.float64)
    centroid_error_m = (
        float(np.linalg.norm(selected_centroid - mesh_centroid))
        if selected_centroid.shape == (3,) and mesh_centroid.shape == (3,)
        else float("inf")
    )
    binding = {
        "selected_frame_idx": frame_idx,
        "visible_geometry_frame_idx": int(geom.get("frame_idx", -1)),
        "canonical_surface_frame_idx": int(mesh_reconstruction.get("anchor_frame_idx", -1)),
        "selected_vs_canonical_centroid_error_m": centroid_error_m,
        "required_same_frame": True,
        "required_centroid_tolerance_m": 1.0e-6,
    }
    if (
        binding["visible_geometry_frame_idx"] != frame_idx
        or binding["canonical_surface_frame_idx"] != frame_idx
        or not np.isfinite(centroid_error_m)
        or centroid_error_m > 1.0e-6
    ):
        raise RuntimeError(f"non-atomic selected P11 anchor binding: {binding}")
    binding["validated"] = True
    return binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--depth-fused-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_compact_rigid_completion"))
    parser.add_argument("--selected-frame-idx", type=int, default=None, help="documented override when the top support mask is visually/geometrically invalid")
    parser.add_argument("--selection-note", default=None)
    args = parser.parse_args()

    ann = load_json(args.annotations)
    object_safe = safe_id(args.object_id.replace("object:", "object_"))
    out_dir = args.output_root / args.case / object_safe / "evidence_bundle"
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for frame in ann.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx"))
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if not isinstance(obj, dict) or obj.get("object_id") != args.object_id:
                continue
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            raw_text = str(frame.get("raw_frame_path") or "")
            mask_text = str(obj.get("mask_path") or "")
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
            visible_vertices = int(geom.get("vertex_count") or len(geom.get("world_vertices_sample_m") or []))
            candidates.append(
                {
                    "frame_idx": frame_idx,
                    "raw_frame_path": str(raw_path),
                    "mask_path": str(mask_path),
                    "visible_depth_vertex_count": visible_vertices,
                    "mask_area_px": mask_area,
                    "camera": frame.get("camera"),
                    "visible_geometry_candidate": geom,
                    "object_bbox_xyxy": obj.get("bbox_xyxy"),
                }
            )

    if not candidates:
        raise RuntimeError(f"no evidence candidates for {args.case} {args.object_id}")
    candidates.sort(key=lambda r: (-int(r["visible_depth_vertex_count"]), -int(r["mask_area_px"]), int(r["frame_idx"])))
    if args.selected_frame_idx is not None:
        matches = [c for c in candidates if int(c["frame_idx"]) == int(args.selected_frame_idx)]
        if not matches:
            raise RuntimeError(f"selected override frame {args.selected_frame_idx} is not a candidate for {args.object_id}")
        selected = dict(matches[0])
        selection_rule = "documented_selected_frame_override_due_invalid_top_mask_or_conditioning_evidence"
    else:
        selected = dict(candidates[0])
        selection_rule = "max_visible_depth_vertex_count_then_mask_area_then_earliest_frame"
    selected_crop = crop_dir / f"frame_{int(selected['frame_idx']):06d}_{object_safe}_rgba.png"
    selected["trellis_conditioning_crop"] = make_rgba_crop(Path(selected["raw_frame_path"]), Path(selected["mask_path"]), selected_crop)

    depth_fused = find_depth_fused_object(args.depth_fused_report, args.object_id)
    selected_mesh_reconstruction = build_selected_anchor_mesh_reconstruction(
        selected=selected,
        output_dir=out_dir,
        object_id=args.object_id,
    )
    anchor_binding = validate_selected_anchor_binding(
        selected=selected,
        mesh_reconstruction=selected_mesh_reconstruction,
    )
    depth_fused_selected = copy.deepcopy(depth_fused)
    original_mesh_reconstruction = (
        copy.deepcopy(depth_fused.get("mesh_reconstruction"))
        if isinstance(depth_fused.get("mesh_reconstruction"), dict)
        else None
    )
    depth_fused_selected["mesh_reconstruction"] = selected_mesh_reconstruction
    depth_fused_selected["original_p09_mesh_reconstruction_diagnostic_only"] = original_mesh_reconstruction
    partial_mesh_paths = {
        key: selected_mesh_reconstruction[key]
        for key in ["fused_point_cloud_path", "poisson_mesh_path", "convex_hull_mesh_path"]
        if selected_mesh_reconstruction.get(key)
    }
    report = {
        "method": "build_v18_compact_rigid_evidence_bundle",
        "status": "ok",
        "case": args.case,
        "object_id": args.object_id,
        "output_dir": str(out_dir),
        "selection_rule": selection_rule,
        "selection_note": args.selection_note,
        "selected_frame_idx": int(selected["frame_idx"]),
        "selected_anchor_atomic_binding": anchor_binding,
        "selected": selected,
        "candidate_count": len(candidates),
        "candidate_summary": {
            "visible_depth_vertex_count_max": int(candidates[0]["visible_depth_vertex_count"]),
            "visible_depth_vertex_count_median": float(np.median([c["visible_depth_vertex_count"] for c in candidates])),
            "mask_area_px_max": int(max(c["mask_area_px"] for c in candidates)),
        },
        "depth_fused_object_row": depth_fused_selected,
        "partial_metric_geometry_paths": partial_mesh_paths,
        "all_candidate_frames": [
            {k: c[k] for k in ["frame_idx", "visible_depth_vertex_count", "mask_area_px", "raw_frame_path", "mask_path"]}
            for c in candidates
        ],
    }
    (out_dir / "evidence_bundle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
