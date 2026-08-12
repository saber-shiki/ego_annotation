#!/usr/bin/env python3
"""Render controlled P13 candidates in one observed-surface PCA frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from render_p12_raw_sam_mask_ab import add_title, render_mesh_panel, simplify

SCHEMA = "v19_experimental_p13_controlled_geometry_prior_render_v1"
VIEWS = [
    ("observed PCA plane", (0, 1, 2)),
    ("PCA long side", (0, 2, 1)),
    ("PCA end", (1, 2, 0)),
]


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


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return mesh


def prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def observed_basis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(vertices, axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    basis = vh.copy()
    if np.linalg.det(basis) < 0:
        basis[-1] *= -1.0
    return center, basis


def transform_vertices(
    vertices: np.ndarray,
    world_center: np.ndarray,
    basis: np.ndarray,
    common_center: np.ndarray,
    common_longest: float,
) -> np.ndarray:
    q = (vertices - world_center) @ basis.T
    return (q - common_center) / common_longest


def overlay_observed(
    image: np.ndarray,
    observed_vertices: np.ndarray,
    observed_faces: np.ndarray,
    axes: tuple[int, int, int],
) -> np.ndarray:
    _, silhouette = render_mesh_panel(
        observed_vertices,
        observed_faces,
        axes,
        image.shape[0],
        (65, 190, 75),
    )
    contours, _ = cv2.findContours(
        silhouette.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (30, 125, 35), 3, cv2.LINE_AA)
    count = min(2500, len(observed_vertices))
    ids = np.linspace(0, len(observed_vertices) - 1, count, dtype=np.int64)
    size = image.shape[0]
    scale = float(size - 64) / 1.08
    u_axis, v_axis, _ = axes
    x = np.rint(size * 0.5 + observed_vertices[ids, u_axis] * scale).astype(np.int32)
    y = np.rint(size * 0.5 - observed_vertices[ids, v_axis] * scale).astype(np.int32)
    inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    for px, py in zip(x[inside], y[inside]):
        cv2.circle(image, (int(px), int(py)), 1, (25, 105, 30), -1, cv2.LINE_AA)
    return image


def render_sheet(
    candidates: list[dict[str, Any]],
    mesh_key: str,
    observed_vertices: np.ndarray,
    observed_faces: np.ndarray,
    transformed: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    output: Path,
    panel_size: int,
) -> list[dict[str, Any]]:
    colors = {
        "trellis_frozen": (195, 105, 55),
        "sam3d_old_raw_sam2_mask": (45, 135, 235),
        "sam3d_new_object_owned_mask": (190, 75, 185),
    }
    display_names = {
        "trellis_frozen": "TRELLIS frozen",
        "sam3d_old_raw_sam2_mask": "SAM3D old raw mask",
        "sam3d_new_object_owned_mask": "SAM3D new owned mask",
    }
    rows: list[np.ndarray] = []
    render_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        vertices, faces = transformed[(name, mesh_key)]
        panels = []
        for view_name, axes in VIEWS:
            panel, _ = render_mesh_panel(
                vertices,
                faces,
                axes,
                int(panel_size),
                colors.get(name, (130, 130, 200)),
            )
            panel = overlay_observed(panel, observed_vertices, observed_faces, axes)
            panels.append(
                add_title(
                    panel,
                    f"{display_names.get(name, name)}: {view_name}",
                    "shared metric-canonical PCA frame; green = observed surface",
                )
            )
        rows.append(np.hstack(panels))
        render_rows.append(
            {
                "name": name,
                "mesh_kind": mesh_key,
                "render_vertices": int(len(vertices)),
                "render_faces": int(len(faces)),
            }
        )
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to write {output}")
    return render_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_report_path = require_file(args.controlled_report, "controlled P13 report")
    source = load_json(source_report_path)
    if source.get("status") != "ok":
        raise RuntimeError(f"controlled P13 report is not ok: {source_report_path}")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("controlled P13 report has no candidates")

    output_dir = prepare_output(args.output_dir)
    meshes: dict[tuple[str, str], trimesh.Trimesh] = {}
    observed_mesh: trimesh.Trimesh | None = None
    all_vertices: list[np.ndarray] = []
    for candidate in candidates:
        name = str(candidate.get("name"))
        outputs = candidate.get("source_neutral_outputs")
        if not isinstance(outputs, dict):
            raise RuntimeError(f"candidate {name} has no source-neutral outputs")
        aligned_path = require_file(
            Path(str(outputs.get("generated_aligned_all_candidate_mesh", ""))),
            f"{name} aligned mesh",
        )
        pose_path = require_file(
            Path(str(outputs.get("pose_hypothesis_mesh", ""))), f"{name} pose mesh"
        )
        meshes[(name, "aligned_all_candidate")] = load_mesh(aligned_path)
        meshes[(name, "pose_hypothesis")] = load_mesh(pose_path)
        all_vertices.append(np.asarray(meshes[(name, "aligned_all_candidate")].vertices, dtype=np.float64))
        if observed_mesh is None:
            legacy_report = load_json(Path(str(candidate["legacy_builder_report"])))
            observed_path = require_file(
                Path(str(legacy_report.get("outputs", {}).get("observed_depth_surface_labeled_mesh", ""))),
                "P13 observed surface",
            )
            observed_mesh = load_mesh(observed_path)
    assert observed_mesh is not None
    observed_world = np.asarray(observed_mesh.vertices, dtype=np.float64)
    world_center, basis = observed_basis(observed_world)
    all_q = np.vstack([(vertices - world_center) @ basis.T for vertices in all_vertices])
    low = np.percentile(all_q, 0.1, axis=0)
    high = np.percentile(all_q, 99.9, axis=0)
    common_center = 0.5 * (low + high)
    common_longest = float(np.max(high - low))
    if not np.isfinite(common_longest) or common_longest <= 0:
        raise RuntimeError("invalid common P13 render scale")

    transformed: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, mesh in meshes.items():
        vertices = transform_vertices(
            np.asarray(mesh.vertices, dtype=np.float64),
            world_center,
            basis,
            common_center,
            common_longest,
        )
        faces = np.asarray(mesh.faces, dtype=np.int64)
        transformed[key] = simplify(vertices, faces, int(args.target_faces))
    observed_vertices = transform_vertices(
        observed_world, world_center, basis, common_center, common_longest
    )
    observed_faces = np.asarray(observed_mesh.faces, dtype=np.int64)

    aligned_output = output_dir / "p13_aligned_candidates_common_frame.png"
    pose_output = output_dir / "p13_pose_hypotheses_common_frame.png"
    rows = []
    rows.extend(
        render_sheet(
            candidates,
            "aligned_all_candidate",
            observed_vertices,
            observed_faces,
            transformed,
            aligned_output,
            int(args.panel_size),
        )
    )
    rows.extend(
        render_sheet(
            candidates,
            "pose_hypothesis",
            observed_vertices,
            observed_faces,
            transformed,
            pose_output,
            int(args.panel_size),
        )
    )
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "render_experimental_p13_controlled_geometry_prior_ab",
        "claim_scope": (
            "static common-frame P13 geometry QC only; no camera projection, temporal pose, "
            "collision readiness, or annotation-readiness claim"
        ),
        "controlled_report": str(source_report_path),
        "common_frame": {
            "world_center": world_center.tolist(),
            "observed_pca_basis_rows": basis.tolist(),
            "pca_bounds_center": common_center.tolist(),
            "common_longest_extent_m": common_longest,
            "normalization_for_render_only": True,
        },
        "renderer": {
            "kind": "same_orthographic_software_renderer",
            "target_faces_per_mesh": int(args.target_faces),
            "panel_size": int(args.panel_size),
            "rows": rows,
        },
        "outputs": {
            "aligned_candidates": str(aligned_output),
            "pose_hypotheses": str(pose_output),
        },
    }
    report_path = output_dir / "p13_controlled_geometry_prior_render_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=60000)
    parser.add_argument("--panel-size", type=int, default=520)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
