#!/usr/bin/env python3
"""Measure observed/generated boundary seams in controlled P13 pose hypotheses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

SCHEMA = "v19_experimental_p13_controlled_seam_diagnostic_v1"
THRESHOLDS_M = [0.001, 0.002, 0.005, 0.010, 0.020, 0.030]


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


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return mesh


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def distance_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": 0}
    out: dict[str, Any] = {
        "count": int(len(finite)),
        "mean_m": float(np.mean(finite)),
        "median_m": float(np.median(finite)),
        "p90_m": float(np.percentile(finite, 90.0)),
        "p95_m": float(np.percentile(finite, 95.0)),
        "max_m": float(np.max(finite)),
    }
    out["fraction_within"] = {
        f"{int(round(threshold * 1000))}mm": float(np.mean(finite <= threshold))
        for threshold in THRESHOLDS_M
    }
    return out


def closest_distances(target: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.float64)
    _, distance, _ = trimesh.proximity.closest_point(target, points)
    return np.asarray(distance, dtype=np.float64)


def mesh_topology(mesh: trimesh.Trimesh) -> dict[str, Any]:
    edges = boundary_edges(mesh)
    components = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "boundary_edges": int(len(edges)),
        "boundary_vertices": int(len(np.unique(edges))) if len(edges) else 0,
        "connected_components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def evaluate_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    name = str(candidate.get("name"))
    outputs = candidate.get("source_neutral_outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError(f"candidate {name} has no source-neutral outputs")
    completed_path = require_file(
        Path(str(outputs.get("pose_hypothesis_mesh", ""))), f"{name} pose-hypothesis mesh"
    )
    legacy_report_path = require_file(
        Path(str(candidate.get("legacy_builder_report", ""))), f"{name} legacy P13 report"
    )
    legacy_report = load_json(legacy_report_path)
    label_path = require_file(
        Path(str(legacy_report.get("outputs", {}).get("completed_face_labels", ""))),
        f"{name} completed face labels",
    )
    label_report = load_json(label_path)
    labels = np.asarray(label_report.get("labels") or [], dtype=object)
    mesh = load_mesh(completed_path)
    if len(labels) != len(mesh.faces):
        raise RuntimeError(
            f"{name} label/face mismatch: labels={len(labels)} faces={len(mesh.faces)}"
        )
    observed_ids = np.flatnonzero(labels == "observed_depth_surface")
    generated_ids = np.flatnonzero(
        np.isin(
            labels,
            [
                "trellis_inferred_hidden_surface",
                "generated_hidden_surface",
                "sam3d_inferred_hidden_surface",
            ],
        )
    )
    if len(observed_ids) == 0 or len(generated_ids) == 0:
        raise RuntimeError(
            f"{name} requires both observed and generated faces: "
            f"observed={len(observed_ids)} generated={len(generated_ids)}"
        )
    observed = mesh.submesh([observed_ids], append=True, repair=False)
    generated = mesh.submesh([generated_ids], append=True, repair=False)
    observed_edges = boundary_edges(observed)
    generated_edges = boundary_edges(generated)
    observed_boundary_points = (
        np.asarray(observed.vertices, dtype=np.float64)[np.unique(observed_edges)]
        if len(observed_edges)
        else np.empty((0, 3), dtype=np.float64)
    )
    generated_boundary_points = (
        np.asarray(generated.vertices, dtype=np.float64)[np.unique(generated_edges)]
        if len(generated_edges)
        else np.empty((0, 3), dtype=np.float64)
    )
    observed_to_generated = closest_distances(generated, observed_boundary_points)
    generated_to_observed = closest_distances(observed, generated_boundary_points)
    combined = np.concatenate([observed_to_generated, generated_to_observed])
    row = {
        "name": name,
        "source_model": candidate.get("source_model"),
        "pose_hypothesis_mesh": str(completed_path),
        "face_labels": str(label_path),
        "whole_pose_hypothesis_topology": mesh_topology(mesh),
        "observed_submesh_topology": mesh_topology(observed),
        "generated_submesh_topology": mesh_topology(generated),
        "observed_boundary_to_generated_surface": distance_summary(observed_to_generated),
        "generated_boundary_to_observed_surface": distance_summary(generated_to_observed),
        "symmetric_boundary_to_counterpart_surface": distance_summary(combined),
        "interpretation": (
            "Boundary distances include true observed/generated interfaces and any cut/free-space "
            "boundary elsewhere. Fractions are diagnostics, not readiness gates."
        ),
    }
    return row, {
        "observed_to_generated": observed_to_generated,
        "generated_to_observed": generated_to_observed,
    }


def plot_cdfs(rows: list[dict[str, Any]], arrays: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    colors = {
        "trellis_frozen": "#356fc2",
        "sam3d_old_raw_sam2_mask": "#df7f22",
        "sam3d_new_object_owned_mask": "#a03daa",
    }
    labels = {
        "trellis_frozen": "TRELLIS frozen",
        "sam3d_old_raw_sam2_mask": "SAM3D old raw mask",
        "sam3d_new_object_owned_mask": "SAM3D new owned mask",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, key, title in [
        (axes[0], "observed_to_generated", "observed boundary → generated surface"),
        (axes[1], "generated_to_observed", "generated boundary → observed surface"),
    ]:
        for row in rows:
            name = row["name"]
            values = arrays[name][key]
            values = np.sort(values[np.isfinite(values)] * 1000.0)
            if len(values) == 0:
                continue
            cdf = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
            axis.plot(values, cdf, label=labels.get(name, name), color=colors.get(name), linewidth=2)
        axis.set_xlim(0, 30)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.set_xlabel("nearest counterpart surface distance (mm)")
        axis.set_title(title)
    axes[0].set_ylabel("boundary-vertex CDF")
    axes[1].legend(loc="lower right")
    figure.suptitle("Controlled P13 observed/generated seam diagnostic")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = require_file(args.controlled_report, "controlled P13 report")
    source = load_json(source_path)
    if source.get("status") != "ok":
        raise RuntimeError(f"controlled P13 report is not ok: {source_path}")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("controlled P13 report has no candidates")
    output_dir = prepare_output(args.output_dir)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        row, candidate_arrays = evaluate_candidate(candidate)
        rows.append(row)
        arrays[row["name"]] = candidate_arrays
    plot_path = output_dir / "p13_controlled_seam_cdf.png"
    plot_cdfs(rows, arrays, plot_path)
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "evaluate_experimental_p13_controlled_seams",
        "claim_scope": (
            "unsigned observed/generated boundary proximity diagnostic; does not establish "
            "watertightness, signed geometry, contact, or collision readiness"
        ),
        "controlled_report": str(source_path),
        "thresholds_m": THRESHOLDS_M,
        "candidates": rows,
        "outputs": {"cdf_plot": str(plot_path)},
    }
    report_path = output_dir / "p13_controlled_seam_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
