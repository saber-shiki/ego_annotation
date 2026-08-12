#!/usr/bin/env python3
"""Render and measure raw-SAM3D mask-conditioning A/B without P13.

This is a shape-only P12 diagnostic. Each mesh is centered and normalized by its
own longest local extent for geometric comparison; native pose/scale changes are
reported separately and never used as metric V19 placement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from PIL import Image

SCHEMA = "v19_experimental_p12_raw_sam_mask_ab_v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"not a triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"mesh has no vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"mesh has no triangular faces: {path}")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"mesh has non-finite vertices: {path}")
    return loaded


def mask_stats(path: Path) -> dict[str, Any]:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape_h_w": [int(v) for v in mask.shape],
        "pixels": int(mask.sum()),
        "array": mask,
    }


def compare_masks(old_path: Path, new_path: Path) -> dict[str, Any]:
    old = mask_stats(old_path)
    new = mask_stats(new_path)
    old_array = old.pop("array")
    new_array = new.pop("array")
    if old_array.shape != new_array.shape:
        raise RuntimeError(f"mask shape mismatch: {old_array.shape} vs {new_array.shape}")
    intersection = int(np.logical_and(old_array, new_array).sum())
    union = int(np.logical_or(old_array, new_array).sum())
    xor = int(np.logical_xor(old_array, new_array).sum())
    return {
        "old_raw_sam2": old,
        "new_object_owned": new,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "xor_pixels": xor,
        "iou": float(intersection / union) if union else 1.0,
    }


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    longest = float(extent.max())
    if not np.isfinite(longest) or longest <= 0:
        raise RuntimeError("mesh has invalid longest extent")
    return (vertices - center) / longest, np.asarray(mesh.faces, dtype=np.int64), center, longest


def simplify(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= target_faces:
        return vertices.copy(), faces.copy()
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    out_vertices = np.asarray(simplified.vertices, dtype=np.float64)
    out_faces = np.asarray(simplified.triangles, dtype=np.int64)
    if len(out_vertices) == 0 or len(out_faces) == 0:
        raise RuntimeError("Open3D simplification returned an empty mesh")
    return out_vertices, out_faces


VIEWS = [
    ("top XY / thin axis Z", (0, 1, 2)),
    ("long side XZ", (0, 2, 1)),
    ("end YZ", (1, 2, 0)),
]


def projected_coordinates(
    vertices: np.ndarray, axes: tuple[int, int, int], size: int, margin: int
) -> tuple[np.ndarray, np.ndarray]:
    u_axis, v_axis, depth_axis = axes
    span = 1.08
    scale = float(size - 2 * margin) / span
    uv = np.empty((len(vertices), 2), dtype=np.float64)
    uv[:, 0] = size * 0.5 + vertices[:, u_axis] * scale
    uv[:, 1] = size * 0.5 - vertices[:, v_axis] * scale
    depth = vertices[:, depth_axis]
    return uv, depth


def render_mesh_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    axes: tuple[int, int, int],
    size: int,
    base_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((size, size, 3), (244, 245, 242), dtype=np.uint8)
    silhouette = np.zeros((size, size), dtype=np.uint8)
    uv, depth = projected_coordinates(vertices, axes, size, margin=32)
    face_vertices = vertices[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1)
    valid_normal = normal_norm > 1e-12
    normals[valid_normal] /= normal_norm[valid_normal, None]
    facing = np.abs(normals[:, axes[2]])
    shade = 0.55 + 0.45 * facing
    order = np.argsort(depth[faces].mean(axis=1))
    base = np.asarray(base_bgr, dtype=np.float64)
    for face_id in order:
        poly = np.rint(uv[faces[int(face_id)]]).astype(np.int32)
        if np.any(poly[:, 0] < -size) or np.any(poly[:, 0] > 2 * size):
            continue
        if np.any(poly[:, 1] < -size) or np.any(poly[:, 1] > 2 * size):
            continue
        color = tuple(int(v) for v in np.clip(base * float(shade[int(face_id)]), 0, 255))
        cv2.fillConvexPoly(image, poly, color, cv2.LINE_AA)
        cv2.fillConvexPoly(silhouette, poly, 255, cv2.LINE_8)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (28, 28, 28), 2, cv2.LINE_AA)
    return image, silhouette > 0


def difference_panel(old_mask: np.ndarray, new_mask: np.ndarray) -> np.ndarray:
    image = np.full((*old_mask.shape, 3), (244, 245, 242), dtype=np.uint8)
    overlap = old_mask & new_mask
    old_only = old_mask & ~new_mask
    new_only = new_mask & ~old_mask
    image[overlap] = (190, 190, 190)
    image[old_only] = (55, 145, 235)  # orange in BGR
    image[new_only] = (70, 190, 70)
    for mask, color in [(old_mask, (30, 95, 215)), (new_mask, (40, 145, 40))]:
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    return image


def add_title(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header = 66
    canvas = np.full((image.shape[0] + header, image.shape[1], 3), (248, 248, 246), dtype=np.uint8)
    canvas[header:] = image
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (65, 65, 65), 1, cv2.LINE_AA)
    return canvas


def surface_samples_normalized(
    mesh: trimesh.Trimesh, center: np.ndarray, longest: float, count: int, seed: int
) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, int(count), seed=int(seed))
    return (np.asarray(points, dtype=np.float64) - center) / float(longest)


def distance_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def axis_profiles(points: np.ndarray, bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(-0.5, 0.5, int(bins) + 1)
    rows: list[dict[str, Any]] = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        take = (points[:, 0] >= lo) & (points[:, 0] < hi if index + 1 < len(edges) - 1 else points[:, 0] <= hi)
        subset = points[take]
        row: dict[str, Any] = {
            "bin": int(index),
            "x_low": float(lo),
            "x_high": float(hi),
            "x_center": float(0.5 * (lo + hi)),
            "samples": int(len(subset)),
        }
        if len(subset) >= 20:
            for name, axis in [("width_y", 1), ("thickness_z", 2)]:
                q01, q99 = np.percentile(subset[:, axis], [1.0, 99.0])
                row[name] = float(q99 - q01)
        rows.append(row)
    return rows


def plot_profiles(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, field, label in [
        (axes[0], "width_y", "normalized width along local Y"),
        (axes[1], "thickness_z", "normalized thickness along local Z"),
    ]:
        for rows, name, color in [
            (old_rows, "old raw-SAM2 mask", "#df7f22"),
            (new_rows, "new object-owned mask", "#269b4a"),
        ]:
            valid = [row for row in rows if field in row]
            axis.plot(
                [row["x_center"] for row in valid],
                [row[field] for row in valid],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=name,
                color=color,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[1].set_xlabel("normalized local long axis X")
    figure.suptitle("SAM3D raw-shape profile: mask-conditioning A/B")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def mesh_summary(mesh: trimesh.Trimesh, center: np.ndarray, longest: float) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    components = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extent_model_units": extent.tolist(),
        "center_used_for_shape_normalization": center.tolist(),
        "longest_extent_used_for_shape_normalization": float(longest),
        "relative_extents": (extent / longest).tolist(),
        "surface_area_model_units2": float(mesh.area),
        "normalized_surface_area": float(mesh.area / (longest * longest)),
        "signed_volume_model_units3": float(mesh.volume),
        "normalized_abs_volume": float(abs(mesh.volume) / (longest ** 3)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "connected_components": int(len(components)),
        "euler_number": int(mesh.euler_number),
    }


def quaternion_angle_degrees(old: list[float], new: list[float]) -> float:
    q0 = np.asarray(old, dtype=np.float64)
    q1 = np.asarray(new, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_report_path = require_file(args.old_sam_report, "old raw-mask SAM3D report")
    new_report_path = require_file(args.new_p12_report, "new P12 report")
    old_report = load_json(old_report_path)
    new_report = load_json(new_report_path)
    if old_report.get("status") != "ok" or new_report.get("status") != "ok":
        raise RuntimeError("old or new SAM3D report is not ok")
    new_candidate = new_report.get("candidates", {}).get("sam3d_objects")
    if not isinstance(new_candidate, dict):
        raise RuntimeError("new P12 report has no sam3d_objects candidate")

    old_mesh_path = require_file(Path(str(old_report.get("mesh", ""))), "old SAM3D mesh")
    new_mesh_path = require_file(
        Path(str(new_candidate.get("native_outputs", {}).get("raw_mesh", {}).get("path", ""))),
        "new SAM3D mesh",
    )
    old_image_path = require_file(Path(str(old_report.get("image", ""))), "old SAM3D RGB")
    old_mask_path = require_file(Path(str(old_report.get("mask", ""))), "old SAM3D mask")
    new_conditioning = new_candidate.get("conditioning", {})
    new_image_path = require_file(Path(str(new_conditioning.get("image", {}).get("path", ""))), "new SAM3D RGB")
    new_mask_path = require_file(Path(str(new_conditioning.get("mask", {}).get("path", ""))), "new SAM3D mask")
    if sha256_file(old_image_path) != sha256_file(new_image_path):
        raise RuntimeError("old and new SAM3D runs do not use byte-identical RGB")

    output_dir = prepare_output(args.output_dir)
    old_mesh = load_mesh(old_mesh_path)
    new_mesh = load_mesh(new_mesh_path)
    old_vertices, old_faces, old_center, old_longest = normalize_mesh(old_mesh)
    new_vertices, new_faces, new_center, new_longest = normalize_mesh(new_mesh)
    old_render_vertices, old_render_faces = simplify(old_vertices, old_faces, int(args.target_faces))
    new_render_vertices, new_render_faces = simplify(new_vertices, new_faces, int(args.target_faces))

    old_pose = old_report.get("pose") or {}
    new_pose = new_candidate.get("native_outputs", {}).get("native_pose") or {}
    old_samples = surface_samples_normalized(old_mesh, old_center, old_longest, int(args.surface_samples), int(args.seed))
    new_samples = surface_samples_normalized(new_mesh, new_center, new_longest, int(args.surface_samples), int(args.seed))
    old_to_new = cKDTree(new_samples).query(old_samples, k=1, workers=-1)[0]
    new_to_old = cKDTree(old_samples).query(new_samples, k=1, workers=-1)[0]
    old_profiles = axis_profiles(old_samples, int(args.profile_bins))
    new_profiles = axis_profiles(new_samples, int(args.profile_bins))

    old_panels: list[np.ndarray] = []
    new_panels: list[np.ndarray] = []
    difference_panels: list[np.ndarray] = []
    per_view_overlap: list[dict[str, Any]] = []
    for view_name, axes in VIEWS:
        old_image, old_silhouette = render_mesh_panel(
            old_render_vertices, old_render_faces, axes, int(args.panel_size), (45, 135, 235)
        )
        new_image, new_silhouette = render_mesh_panel(
            new_render_vertices, new_render_faces, axes, int(args.panel_size), (75, 190, 90)
        )
        intersection = int(np.logical_and(old_silhouette, new_silhouette).sum())
        union = int(np.logical_or(old_silhouette, new_silhouette).sum())
        per_view_overlap.append(
            {
                "view": view_name,
                "normalized_silhouette_iou": float(intersection / union) if union else 1.0,
                "old_only_pixels": int(np.logical_and(old_silhouette, ~new_silhouette).sum()),
                "new_only_pixels": int(np.logical_and(new_silhouette, ~old_silhouette).sum()),
            }
        )
        old_panels.append(add_title(old_image, f"OLD: {view_name}", "full RGB + raw SAM2 mask"))
        new_panels.append(add_title(new_image, f"NEW: {view_name}", "full RGB + object-owned mask"))
        difference_panels.append(
            add_title(
                difference_panel(old_silhouette, new_silhouette),
                f"SILHOUETTE DELTA: {view_name}",
                "orange=old only, green=new only, gray=overlap",
            )
        )

    contact_sheet = np.vstack(
        [np.hstack(old_panels), np.hstack(new_panels), np.hstack(difference_panels)]
    )
    contact_sheet_path = output_dir / "p12_raw_sam_mask_ab_three_views.png"
    if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
        raise RuntimeError(f"failed to write {contact_sheet_path}")
    profile_path = output_dir / "p12_raw_sam_mask_ab_axis_profiles.png"
    plot_profiles(old_profiles, new_profiles, profile_path)

    old_scale = np.asarray(old_pose.get("scale"), dtype=np.float64).reshape(-1)
    new_scale = np.asarray(new_pose.get("scale"), dtype=np.float64).reshape(-1)
    old_translation = np.asarray(old_pose.get("translation"), dtype=np.float64).reshape(-1)
    new_translation = np.asarray(new_pose.get("translation"), dtype=np.float64).reshape(-1)
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "render_experimental_p12_raw_sam_mask_ab",
        "claim_scope": (
            "P12 raw local-shape and native-layout delta only; longest-axis-normalized shape views "
            "are not metric placement, P13 alignment, temporal pose, or collision evidence"
        ),
        "inputs": {
            "old_report": str(old_report_path),
            "new_p12_report": str(new_report_path),
            "rgb": {
                "old": str(old_image_path),
                "new": str(new_image_path),
                "sha256": sha256_file(old_image_path),
                "byte_identical": True,
            },
            "masks": compare_masks(old_mask_path, new_mask_path),
            "seed": int(args.seed),
        },
        "old_raw_sam2_mask_mesh": {
            "path": str(old_mesh_path),
            "sha256": sha256_file(old_mesh_path),
            "summary": mesh_summary(old_mesh, old_center, old_longest),
            "native_pose": old_pose,
        },
        "new_object_owned_mask_mesh": {
            "path": str(new_mesh_path),
            "sha256": sha256_file(new_mesh_path),
            "summary": mesh_summary(new_mesh, new_center, new_longest),
            "native_pose": new_pose,
        },
        "shape_comparison": {
            "normalization": "independent_bbox_center_and_longest_local_extent",
            "surface_samples_per_mesh": int(args.surface_samples),
            "old_to_new_nearest_distance_normalized": distance_summary(old_to_new),
            "new_to_old_nearest_distance_normalized": distance_summary(new_to_old),
            "symmetric_chamfer_mean_normalized": float(0.5 * (old_to_new.mean() + new_to_old.mean())),
            "per_view_normalized_silhouette": per_view_overlap,
            "old_axis_profiles": old_profiles,
            "new_axis_profiles": new_profiles,
        },
        "native_layout_delta_not_metric_v19": {
            "quaternion_angular_delta_degrees": quaternion_angle_degrees(
                old_pose.get("rotation"), new_pose.get("rotation")
            ),
            "translation_delta_native_units": (new_translation - old_translation).tolist(),
            "translation_delta_l2_native_units": float(np.linalg.norm(new_translation - old_translation)),
            "uniform_scale_old": float(old_scale.mean()),
            "uniform_scale_new": float(new_scale.mean()),
            "uniform_scale_ratio_new_over_old": float(new_scale.mean() / old_scale.mean()),
        },
        "renderer": {
            "kind": "same_orthographic_software_renderer",
            "target_faces_per_mesh": int(args.target_faces),
            "old_render_faces": int(len(old_render_faces)),
            "new_render_faces": int(len(new_render_faces)),
            "panel_size": int(args.panel_size),
        },
        "outputs": {
            "three_view_contact_sheet": str(contact_sheet_path),
            "axis_profiles": str(profile_path),
        },
    }
    report_path = output_dir / "p12_raw_sam_mask_ab_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "masks": report["inputs"]["masks"],
                "old_summary": report["old_raw_sam2_mask_mesh"]["summary"],
                "new_summary": report["new_object_owned_mask_mesh"]["summary"],
                "shape_comparison": {
                    key: value
                    for key, value in report["shape_comparison"].items()
                    if key not in {"old_axis_profiles", "new_axis_profiles"}
                },
                "native_layout_delta": report["native_layout_delta_not_metric_v19"],
                "outputs": report["outputs"],
            },
            indent=2,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-sam-report", type=Path, required=True)
    parser.add_argument("--new-p12-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=60000)
    parser.add_argument("--surface-samples", type=int, default=60000)
    parser.add_argument("--profile-bins", type=int, default=24)
    parser.add_argument("--panel-size", type=int, default=560)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
