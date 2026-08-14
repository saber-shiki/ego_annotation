#!/usr/bin/env python3
"""Write a source-neutral observed-surface-only completion contract for pose fitting.

The generated single-image priors remain render hypotheses.  This adapter binds
P14/P15 pose fitting to the shared prediction-side metric observation surface so
the SAM3D and TRELLIS render branches consume exactly the same object trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import trimesh


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [
        row
        for row in report.get("candidates", [])
        if isinstance(row, dict) and str(row.get("name")) == name
    ]
    if len(rows) != 1:
        raise RuntimeError(f"controlled report candidate {name!r} appears {len(rows)} times")
    return rows[0]


def mesh_summary(path: Path) -> dict[str, Any]:
    geometry = trimesh.load(path, force="mesh", process=False)
    if not isinstance(geometry, trimesh.Trimesh) or len(geometry.vertices) == 0 or len(geometry.faces) == 0:
        raise RuntimeError(f"observed surface is not a non-empty triangle mesh: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "vertices": int(len(geometry.vertices)),
        "faces": int(len(geometry.faces)),
        "watertight": bool(geometry.is_watertight),
        "winding_consistent": bool(geometry.is_winding_consistent),
        "bounds_m": geometry.bounds.astype(float).tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    controlled_path = require_file(args.controlled_report, "controlled P13 report")
    controlled = load_json(controlled_path)
    if controlled.get("status") != "ok":
        raise RuntimeError(f"controlled P13 report is not ok: {controlled_path}")
    candidate = candidate_by_name(controlled, args.candidate)
    candidate_outputs = (
        candidate.get("source_neutral_outputs")
        if isinstance(candidate.get("source_neutral_outputs"), dict)
        else {}
    )
    candidate_collision = require_file(
        Path(str(candidate_outputs.get("collision_eligible_mesh", ""))),
        "candidate observed-only collision surface",
    )
    observed_mesh = require_file(args.observed_mesh, "shared observed-only mesh")
    candidate_hash = sha256_file(candidate_collision)
    observed_hash = sha256_file(observed_mesh)
    if candidate_hash != observed_hash:
        raise RuntimeError(
            "shared observed mesh is not byte-identical to the controlled P13 collision surface: "
            f"candidate={candidate_collision} shared={observed_mesh}"
        )

    builder_report_path = require_file(
        Path(str(candidate.get("legacy_builder_report", ""))), "candidate P13 builder report"
    )
    builder = load_json(builder_report_path)
    observed_band_m = float(builder.get("observed_band_m") or 0.0)
    summary = mesh_summary(observed_mesh)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.replace:
        raise RuntimeError(f"refusing to overwrite output: {output}")

    report = {
        "schema": "v19_observed_metric_surface_only_pose_reference_v1",
        "status": "ok",
        "method": "build_v19_observed_only_completion_reference",
        "created_unix_s": time.time(),
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Prediction-side observed metric surface only, used as the common rigid-pose reference. "
            "No generated hidden face from either geometry backend is a pose, collision, contact, or sign observation."
        ),
        "inputs": {
            "controlled_report": str(controlled_path),
            "controlled_candidate": str(args.candidate),
            "candidate_builder_report": str(builder_report_path),
            "candidate_collision_surface": str(candidate_collision),
            "shared_observed_mesh": str(observed_mesh),
        },
        "source_model": "prediction_side_observed_metric_surface",
        "observed_band_m": observed_band_m,
        "outputs": {
            "pose_hypothesis_mesh_labeled": str(observed_mesh),
            "completed_mesh_labeled": str(observed_mesh),
            "collision_eligible_mesh_labeled": str(observed_mesh),
            "observed_depth_surface_labeled_mesh": str(observed_mesh),
        },
        "geometry_readiness": {
            "pose_hypothesis_source": "prediction_side_observed_metric_surface_only",
            "collision_surface_source": "prediction_side_observed_metric_surface_only",
            "collision_eligible_mesh": str(observed_mesh),
            "generated_hidden_surface_included": False,
            "generated_faces_collision_eligible": False,
            "generated_faces_contact_eligible": False,
            "signed_geometry_ready": False,
            "annotation_ready": False,
            "uncertainty": "partial visible observation surface; unsigned physical proximity only",
        },
        "accepted_body_semantics": {
            "observed_depth_surface_faces_accepted": int(summary["faces"]),
            "generated_hidden_surface_faces_accepted": 0,
        },
        "mesh_summary": summary,
        "source_artifacts_mutated": False,
        "outputs_report": str(output),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "mesh_summary": summary,
            },
            indent=2,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--candidate", default="sam3d_new_object_owned_mask")
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
