#!/usr/bin/env python3
"""Build and render an additive dual-mesh P13 hypothesis without cutting raw prior faces."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from render_p12_raw_sam_mask_ab import add_title, render_mesh_panel, simplify

SCHEMA = "v19_experimental_p13_dual_mesh_geometry_prior_v1"
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


def prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return mesh


def boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def topology(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "boundary_edges": boundary_edge_count(mesh),
        "connected_components": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
    }


def transform_mesh_preserve_topology(mesh: trimesh.Trimesh, matrix: np.ndarray) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    hom = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    transformed = (hom @ matrix.T)[:, :3]
    out = mesh.copy()
    out.vertices = transformed
    return out


def observed_basis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    basis = vh.copy()
    if np.linalg.det(basis) < 0:
        basis[-1] *= -1.0
    return center, basis


def normalized_vertices(
    mesh: trimesh.Trimesh,
    world_center: np.ndarray,
    basis: np.ndarray,
    bounds_center: np.ndarray,
    longest: float,
) -> np.ndarray:
    q = (np.asarray(mesh.vertices, dtype=np.float64) - world_center) @ basis.T
    return (q - bounds_center) / longest


def observed_first_panel(
    prior_vertices: np.ndarray,
    prior_faces: np.ndarray,
    observed_vertices: np.ndarray,
    observed_faces: np.ndarray,
    axes: tuple[int, int, int],
    size: int,
    prior_color: tuple[int, int, int],
) -> np.ndarray:
    prior_image, _ = render_mesh_panel(prior_vertices, prior_faces, axes, size, prior_color)
    observed_image, observed_mask = render_mesh_panel(
        observed_vertices, observed_faces, axes, size, (65, 190, 75)
    )
    prior_image[observed_mask] = observed_image[observed_mask]
    contours, _ = cv2.findContours(
        observed_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(prior_image, contours, -1, (25, 105, 30), 3, cv2.LINE_AA)
    return prior_image


def render_comparison(
    aligned_raw: trimesh.Trimesh,
    legacy_cut: trimesh.Trimesh,
    observed: trimesh.Trimesh,
    output: Path,
    target_faces: int,
    panel_size: int,
) -> dict[str, Any]:
    observed_world = np.asarray(observed.vertices, dtype=np.float64)
    world_center, basis = observed_basis(observed_world)
    all_q = np.vstack(
        [
            (np.asarray(mesh.vertices, dtype=np.float64) - world_center) @ basis.T
            for mesh in [aligned_raw, legacy_cut, observed]
        ]
    )
    low = np.percentile(all_q, 0.1, axis=0)
    high = np.percentile(all_q, 99.9, axis=0)
    bounds_center = 0.5 * (low + high)
    longest = float(np.max(high - low))
    if not np.isfinite(longest) or longest <= 0:
        raise RuntimeError("invalid dual-mesh render scale")

    raw_vertices = normalized_vertices(aligned_raw, world_center, basis, bounds_center, longest)
    cut_vertices = normalized_vertices(legacy_cut, world_center, basis, bounds_center, longest)
    observed_vertices = normalized_vertices(observed, world_center, basis, bounds_center, longest)
    raw_vertices, raw_faces = simplify(
        raw_vertices, np.asarray(aligned_raw.faces, dtype=np.int64), target_faces
    )
    cut_vertices, cut_faces = simplify(
        cut_vertices, np.asarray(legacy_cut.faces, dtype=np.int64), target_faces
    )
    observed_faces = np.asarray(observed.faces, dtype=np.int64)

    rows = []
    for title, vertices, faces, color, subtitle in [
        (
            "DUAL: intact aligned SAM prior",
            raw_vertices,
            raw_faces,
            (190, 75, 185),
            "generated underlay is intact; observed metric surface rendered on top",
        ),
        (
            "LEGACY: generated faces cut near observed band",
            cut_vertices,
            cut_faces,
            (45, 135, 235),
            "legacy concatenate can expose white seam where both layers are absent",
        ),
    ]:
        panels = []
        for view_name, axes in VIEWS:
            panel = observed_first_panel(
                vertices,
                faces,
                observed_vertices,
                observed_faces,
                axes,
                panel_size,
                color,
            )
            panels.append(add_title(panel, f"{title}: {view_name}", subtitle))
        rows.append(np.hstack(panels))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to write {output}")
    return {
        "world_center": world_center.tolist(),
        "observed_pca_basis_rows": basis.tolist(),
        "bounds_center": bounds_center.tolist(),
        "longest_extent_m": longest,
        "target_faces": int(target_faces),
        "panel_size": int(panel_size),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    controlled_path = require_file(args.controlled_report, "controlled P13 report")
    controlled = load_json(controlled_path)
    if controlled.get("status") != "ok":
        raise RuntimeError(f"controlled P13 report is not ok: {controlled_path}")
    matches = [row for row in controlled.get("candidates", []) if row.get("name") == args.candidate]
    if len(matches) != 1:
        raise RuntimeError(f"candidate {args.candidate!r} appears {len(matches)} times")
    candidate = matches[0]
    raw_mesh_path = require_file(Path(str(candidate.get("raw_mesh", {}).get("path", ""))), "raw prior mesh")
    legacy_path = require_file(
        Path(str(candidate.get("source_neutral_outputs", {}).get("pose_hypothesis_mesh", ""))),
        "legacy P13 pose hypothesis",
    )
    observed_source_path = require_file(
        Path(str(candidate.get("source_neutral_outputs", {}).get("collision_eligible_mesh", ""))),
        "observed-only collision surface",
    )
    matrix = np.asarray(
        load_json(Path(str(candidate["legacy_builder_report"])))
        .get("metric_alignment", {})
        .get("matrix_model_to_canonical"),
        dtype=np.float64,
    )
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("candidate has invalid P13 model-to-canonical matrix")

    output_dir = prepare_output(args.output_dir)
    raw_mesh = load_mesh(raw_mesh_path)
    aligned_raw = transform_mesh_preserve_topology(raw_mesh, matrix)
    legacy_cut = load_mesh(legacy_path)
    observed = load_mesh(observed_source_path)
    raw_topology = topology(raw_mesh)
    aligned_topology = topology(aligned_raw)
    if raw_topology != aligned_topology:
        raise RuntimeError(
            f"global alignment changed raw mesh topology: raw={raw_topology} aligned={aligned_topology}"
        )

    aligned_path = output_dir / "generated_aligned_raw_render_prior_intact.ply"
    observed_path = output_dir / "observed_metric_surface.ply"
    collision_path = output_dir / "collision_eligible_observed_surface.ply"
    aligned_raw.export(str(aligned_path))
    shutil.copyfile(observed_source_path, observed_path)
    shutil.copyfile(observed_source_path, collision_path)

    scene_prior = aligned_raw.copy()
    scene_prior.visual.face_colors = np.tile(
        np.asarray([[185, 75, 190, 145]], dtype=np.uint8), (len(scene_prior.faces), 1)
    )
    scene_observed = observed.copy()
    scene_observed.visual.face_colors = np.tile(
        np.asarray([[65, 195, 75, 255]], dtype=np.uint8), (len(scene_observed.faces), 1)
    )
    scene = trimesh.Scene()
    scene.add_geometry(scene_prior, node_name="generated_aligned_raw_render_prior_intact")
    scene.add_geometry(scene_observed, node_name="observed_metric_surface_overlay")
    scene_path = output_dir / "dual_mesh_render_scene.glb"
    scene.export(str(scene_path))

    review_path = output_dir / "dual_mesh_vs_legacy_observed_first.png"
    render_contract = render_comparison(
        aligned_raw,
        legacy_cut,
        observed,
        review_path,
        int(args.target_faces),
        int(args.panel_size),
    )
    render_state = {
        "schema": "v19_experimental_dual_mesh_render_state_v1",
        "render_order_back_to_front": [
            {
                "role": "generated_complete_prior_underlay",
                "mesh": str(aligned_path),
                "face_deletion_applied": False,
            },
            {
                "role": "observed_metric_surface_overlay",
                "mesh": str(observed_path),
                "depth_priority": "observed_first_when_coincident_or_supported",
            },
        ],
        "physical_surface": {
            "mesh": str(collision_path),
            "source": "observed_metric_surface_only",
            "generated_faces_collision_eligible": False,
            "signed_geometry_ready": False,
        },
    }
    render_state_path = output_dir / "dual_mesh_render_state.json"
    render_state_path.write_text(json.dumps(render_state, indent=2), encoding="utf-8")

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "build_experimental_p13_dual_mesh_geometry_prior",
        "claim_scope": (
            "static dual-mesh pose/render hypothesis; raw generated topology is preserved and "
            "observed metric geometry remains a separate overlay/collision surface. No temporal "
            "pose, signed collision, or annotation-readiness claim."
        ),
        "controlled_report": str(controlled_path),
        "candidate": args.candidate,
        "source_model": candidate.get("source_model"),
        "metric_alignment_matrix_model_to_canonical": matrix.tolist(),
        "inputs": {
            "raw_prior": {"path": str(raw_mesh_path), "sha256": sha256_file(raw_mesh_path)},
            "legacy_cut_pose_hypothesis": str(legacy_path),
            "observed_metric_surface": str(observed_source_path),
        },
        "topology": {
            "raw_prior": raw_topology,
            "aligned_raw_render_prior": aligned_topology,
            "legacy_cut_pose_hypothesis": topology(legacy_cut),
            "observed_metric_surface": topology(observed),
        },
        "invariants": {
            "raw_face_count_preserved": True,
            "raw_vertex_count_preserved": True,
            "raw_watertightness_preserved": True,
            "generated_face_deletion_applied": False,
            "generated_faces_collision_eligible": False,
        },
        "render_contract": render_contract,
        "outputs": {
            "aligned_raw_render_prior": str(aligned_path),
            "observed_metric_surface": str(observed_path),
            "collision_eligible_observed_surface": str(collision_path),
            "dual_mesh_scene_glb": str(scene_path),
            "render_state": str(render_state_path),
            "comparison_review": str(review_path),
        },
    }
    report_path = output_dir / "p13_dual_mesh_geometry_prior_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--candidate", default="sam3d_new_object_owned_mask")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=60000)
    parser.add_argument("--panel-size", type=int, default=540)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
