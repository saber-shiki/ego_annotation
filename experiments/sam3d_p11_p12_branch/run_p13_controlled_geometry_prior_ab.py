#!/usr/bin/env python3
"""Run a controlled P13 comparison with backend-correct geometry adapters.

Both candidates consume the same canonical observed evidence and the same
render-only face-labeling builder. TRELLIS keeps its generic RMS/PCA/ICP adapter;
SAM3D must first pass the native-pose sensor-metric bridge and then enters the
builder as a verified metric-canonical mesh with identity alignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

SCHEMA = "v19_experimental_p13_controlled_geometry_prior_ab_v2"


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


def require_executable_preserve_symlink(path: Path, description: str) -> Path:
    # Resolving a venv's bin/python symlink can turn it into the system Python
    # and discard pyvenv.cfg discovery. Keep the supplied absolute path.
    path = Path(os.path.abspath(path.expanduser()))
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"{description} is not executable: {path}")
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


def safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return out.strip("_") or "candidate"


def parse_candidate(raw: str) -> dict[str, Any]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise RuntimeError(
            "--candidate must be name|source_model|mesh_path|source_report_path"
        )
    name, source_model, mesh_path, source_report_path = parts
    return {
        "name": safe_id(name),
        "source_model": source_model,
        "mesh": require_file(Path(mesh_path), f"{name} mesh"),
        "source_report": require_file(Path(source_report_path), f"{name} source report"),
    }


def count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(k): int(v) for k, v in value.items()}


def metric_stats(value: Any) -> dict[str, float | int | None]:
    if not isinstance(value, dict):
        return {}
    keys = ["count", "mean_m", "median_m", "p90_m", "p95_m", "max_m"]
    return {key: value.get(key) for key in keys if key in value}


def source_neutral_summary(
    candidate: dict[str, Any],
    builder_report_path: Path,
    compatibility_report_path: Path,
    command: list[str],
    elapsed_s: float,
    stdout_path: Path,
    stderr_path: Path,
    native_metric_bridge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = load_json(builder_report_path)
    status = str(report.get("status") or "")
    if not status.startswith("ok"):
        raise RuntimeError(f"P13 builder report is not successful: {builder_report_path}: {status}")
    alignment = report.get("metric_alignment") if isinstance(report.get("metric_alignment"), dict) else {}
    labels = report.get("face_label_counts") if isinstance(report.get("face_label_counts"), dict) else {}
    all_candidate_counts = count_map(labels.get("trellis_all_candidate"))
    completed_counts = count_map(labels.get("completed_mesh"))
    total_faces = int(sum(all_candidate_counts.values()))

    outputs = report.get("outputs") if isinstance(report.get("outputs"), dict) else {}
    aligned_mesh = require_file(
        Path(str(outputs.get("trellis_aligned_all_candidate_labeled_mesh", ""))),
        f"{candidate['name']} P13 aligned candidate mesh",
    )
    completed_mesh = require_file(
        Path(str(outputs.get("pose_hypothesis_mesh_labeled") or outputs.get("completed_mesh_labeled") or "")),
        f"{candidate['name']} P13 pose-hypothesis mesh",
    )
    collision_mesh_text = str(outputs.get("collision_eligible_mesh_labeled") or "")
    collision_mesh = require_file(Path(collision_mesh_text), "P13 collision mesh") if collision_mesh_text else None

    aliases = {
        "generated_aligned_all_candidate_mesh": str(aligned_mesh),
        "pose_hypothesis_mesh": str(completed_mesh),
        "collision_eligible_mesh": str(collision_mesh) if collision_mesh is not None else None,
    }
    fraction_by_label = {
        label.replace("trellis_", "generated_"): float(count / total_faces) if total_faces else 0.0
        for label, count in all_candidate_counts.items()
    }
    generated_hidden_count = int(
        completed_counts.get("trellis_inferred_hidden_surface", 0)
        + completed_counts.get("generated_hidden_surface", 0)
        + completed_counts.get("sam3d_inferred_hidden_surface", 0)
    )
    filter_state = report.get("silhouette_free_space_filter")
    slab_state = report.get("planar_slab_support_filter")
    mesh_counts = report.get("mesh_counts") if isinstance(report.get("mesh_counts"), dict) else {}
    return {
        "name": candidate["name"],
        "source_model": candidate["source_model"],
        "status": "ok_controlled_p13_candidate",
        "raw_mesh": {
            "path": str(candidate["mesh"]),
            "sha256": sha256_file(candidate["mesh"]),
            "bytes": int(candidate["mesh"].stat().st_size),
        },
        "raw_source_report": str(candidate["source_report"]),
        "compatibility_report": str(compatibility_report_path),
        "legacy_builder_report": str(builder_report_path),
        "legacy_builder_status": status,
        "legacy_field_notice": (
            "The shared V18 labeling builder retains trellis_* field names for compatibility. "
            "For SAM3D, metric alignment is a verified identity because the native bridge already produced canonical meters; labels are not source attribution."
        ),
        "metric_alignment": {
            "mode": alignment.get("mode", "legacy_generic_rms_pca_icp"),
            "scale": alignment.get("scale"),
            "rotation": alignment.get("rotation"),
            "translation": alignment.get("translation"),
            "initial_scale_from_rms_radius": alignment.get("initial_scale_from_rms_radius"),
            "observed_to_generated_initial": metric_stats(
                alignment.get("observed_to_trellis_stats_initial")
            ),
            "observed_to_generated_final": metric_stats(
                alignment.get("observed_to_trellis_stats_final")
            ),
            "generated_to_observed_final": metric_stats(
                alignment.get("trellis_to_observed_stats_final")
            ),
            "icp_refinement_stats": alignment.get("icp_refinement_stats"),
        },
        "native_metric_bridge": native_metric_bridge,
        "face_semantics": {
            "raw_legacy_counts": all_candidate_counts,
            "source_neutral_fraction_by_label": fraction_by_label,
            "completed_raw_legacy_counts": completed_counts,
            "generated_hidden_faces_in_pose_hypothesis": generated_hidden_count,
            "free_space_rejected_faces": int(
                all_candidate_counts.get("free_space_rejected", 0)
            ),
            "observed_region_overwritten_candidate_faces": int(
                all_candidate_counts.get("observed_region_overwritten_candidate", 0)
            ),
            "unsupported_uncertain_faces": int(
                all_candidate_counts.get("unsupported_uncertain", 0)
            ),
            "total_raw_candidate_faces": total_faces,
        },
        "silhouette_free_space_filter": filter_state,
        "planar_slab_support_filter": slab_state,
        "mesh_counts": mesh_counts,
        "geometry_readiness": report.get("geometry_readiness"),
        "source_neutral_outputs": aliases,
        "invocation": {
            "command": command,
            "command_shell_escaped": shlex.join(command),
            "elapsed_s": float(elapsed_s),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_report = require_file(args.evidence_report, "canonical P11 evidence report")
    builder_script = require_file(args.builder_script, "P13 labeling/alignment builder")
    sam3d_bridge_script = require_file(args.sam3d_bridge_script, "SAM3D native metric bridge")
    python = require_executable_preserve_symlink(args.python, "P13 Python interpreter")
    candidates = [parse_candidate(raw) for raw in args.candidate]
    names = [candidate["name"] for candidate in candidates]
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate candidate names: {names}")
    if len(candidates) < 2:
        raise RuntimeError("controlled P13 comparison requires at least two candidates")

    output_dir = prepare_output(args.output_dir)
    evidence_hash_before = sha256_file(evidence_report)
    builder_hash_before = sha256_file(builder_script)
    sam3d_bridge_hash_before = sha256_file(sam3d_bridge_script)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dir = output_dir / candidate["name"]
        candidate_dir.mkdir(parents=True)
        compatibility_report_path = candidate_dir / "p12_source_legacy_builder_compatibility.json"
        native_metric_bridge_summary = None
        if candidate["source_model"] == "sam3d_objects":
            bridge_output = candidate_dir / "sam3d_native_metric_bridge"
            bridge_stdout = candidate_dir / "sam3d_native_metric_bridge_stdout.json"
            bridge_stderr = candidate_dir / "sam3d_native_metric_bridge_stderr.txt"
            bridge_command = [
                str(python),
                str(sam3d_bridge_script),
                "--evidence-report",
                str(evidence_report),
                "--p12-report",
                str(candidate["source_report"]),
                "--output-dir",
                str(bridge_output),
            ]
            bridge_start = time.perf_counter()
            bridge_completed = subprocess.run(bridge_command, capture_output=True, text=True)
            bridge_elapsed_s = time.perf_counter() - bridge_start
            bridge_stdout.write_text(bridge_completed.stdout, encoding="utf-8")
            bridge_stderr.write_text(bridge_completed.stderr, encoding="utf-8")
            if bridge_completed.returncode != 0:
                raise RuntimeError(
                    f"SAM3D native metric bridge failed with code {bridge_completed.returncode}: {bridge_stderr}"
                )
            bridge_report_path = require_file(
                bridge_output / "sam3d_native_sensor_metric_canonical_bridge_report.json",
                "SAM3D native metric bridge report",
            )
            bridge_report = load_json(bridge_report_path)
            compatibility_report_path = require_file(
                bridge_output / "p13_metric_canonical_input.json",
                "SAM3D metric-canonical P13 input",
            )
            bridge_raw_mesh = Path(str((bridge_report.get("inputs") or {}).get("raw_mesh") or "")).resolve()
            if bridge_raw_mesh != candidate["mesh"].resolve() or str((bridge_report.get("inputs") or {}).get("raw_mesh_sha256")) != sha256_file(candidate["mesh"]):
                raise RuntimeError("SAM3D native metric bridge did not consume the declared controlled raw mesh")
            native_metric_bridge_summary = {
                "status": bridge_report.get("status"),
                "report": str(bridge_report_path),
                "report_sha256": sha256_file(bridge_report_path),
                "command": bridge_command,
                "elapsed_s": float(bridge_elapsed_s),
                "stdout": str(bridge_stdout),
                "stderr": str(bridge_stderr),
                "projection_validation": bridge_report.get("projection_validation"),
                "sensor_metric_scene_similarity": bridge_report.get("sensor_metric_scene_similarity"),
                "geometry_validation": bridge_report.get("geometry_validation"),
            }
        else:
            compatibility = {
                "schema": "v19_experimental_p12_source_legacy_p13_compatibility_v1",
                "status": "ok",
                "method": "write_isolated_legacy_p13_source_adapter",
                "source_model": candidate["source_model"],
                "mesh": str(candidate["mesh"]),
                "raw_source_report": str(candidate["source_report"]),
                "claim_scope": (
                    "compatibility input for the TRELLIS generic P13 adapter only; "
                    "trellis field naming is not source attribution"
                ),
            }
            compatibility_report_path.write_text(json.dumps(compatibility, indent=2), encoding="utf-8")
        builder_output = candidate_dir / "legacy_builder_output"
        command = [
            str(python),
            str(builder_script),
            "--evidence-report",
            str(evidence_report),
            "--trellis-report",
            str(compatibility_report_path),
            "--output-dir",
            str(builder_output),
            "--observed-band-scale",
            str(float(args.observed_band_scale)),
            "--silhouette-free-space-filter",
            "--silhouette-dilate-px",
            str(int(args.silhouette_dilate_px)),
            "--planar-slab-support-filter",
            "--planar-slab-eigenvalue-ratio-max",
            str(float(args.planar_slab_eigenvalue_ratio_max)),
            "--planar-slab-min-band-m",
            str(float(args.planar_slab_min_band_m)),
            "--planar-slab-max-band-m",
            str(float(args.planar_slab_max_band_m)),
        ]
        if candidate["source_model"] == "sam3d_objects":
            command.append("--input-mesh-already-metric-canonical")
        stdout_path = candidate_dir / "builder_stdout.json"
        stderr_path = candidate_dir / "builder_stderr.txt"
        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        elapsed_s = time.perf_counter() - start
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            failure = {
                "schema": SCHEMA,
                "status": "candidate_failed",
                "failed_candidate": candidate["name"],
                "source_model": candidate["source_model"],
                "command": command,
                "returncode": int(completed.returncode),
                "elapsed_s": float(elapsed_s),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "source_artifacts_mutated": False,
            }
            (output_dir / "p13_controlled_geometry_prior_ab_report.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                f"P13 candidate {candidate['name']} failed with code {completed.returncode}: {stderr_path}"
            )
        builder_report_path = require_file(
            builder_output / "v18_compact_rigid_trellis_completion_report.json",
            f"{candidate['name']} P13 builder report",
        )
        rows.append(
            source_neutral_summary(
                candidate,
                builder_report_path,
                compatibility_report_path,
                command,
                elapsed_s,
                stdout_path,
                stderr_path,
                native_metric_bridge_summary,
            )
        )

    if sha256_file(evidence_report) != evidence_hash_before:
        raise RuntimeError("canonical evidence report changed during controlled P13 run")
    if sha256_file(builder_script) != builder_hash_before:
        raise RuntimeError("P13 builder script changed during controlled P13 run")
    if sha256_file(sam3d_bridge_script) != sam3d_bridge_hash_before:
        raise RuntimeError("SAM3D native metric bridge script changed during controlled P13 run")

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "run_experimental_p13_controlled_geometry_prior_ab",
        "claim_scope": (
            "Controlled backend-correct P13 comparison: candidates share canonical observed evidence and face-labeling semantics, "
            "while each uses its declared geometry adapter. TRELLIS uses generic metric alignment; SAM3D preserves its native pose "
            "through a fail-closed sensor-metric bridge and identity canonical ingestion. Generated faces are render hypotheses only."
        ),
        "common_contract": {
            "evidence_report": str(evidence_report),
            "evidence_sha256": evidence_hash_before,
            "builder_script": str(builder_script),
            "builder_sha256": builder_hash_before,
            "sam3d_native_metric_bridge_script": str(sam3d_bridge_script),
            "sam3d_native_metric_bridge_sha256": sam3d_bridge_hash_before,
            "python": str(python),
            "parameters": {
                "observed_band_scale": float(args.observed_band_scale),
                "silhouette_free_space_filter": True,
                "silhouette_dilate_px": int(args.silhouette_dilate_px),
                "planar_slab_support_filter": True,
                "planar_slab_eigenvalue_ratio_max": float(
                    args.planar_slab_eigenvalue_ratio_max
                ),
                "planar_slab_min_band_m": float(args.planar_slab_min_band_m),
                "planar_slab_max_band_m": float(args.planar_slab_max_band_m),
                "promote_single_view_hidden_prior_to_collision": False,
            },
        },
        "candidates": rows,
        "source_artifacts_mutated": False,
    }
    report_path = output_dir / "p13_controlled_geometry_prior_ab_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        "status": "ok",
        "candidates": [
            {
                "name": row["name"],
                "source_model": row["source_model"],
                "elapsed_s": row["invocation"]["elapsed_s"],
                "observed_to_generated_final": row["metric_alignment"][
                    "observed_to_generated_final"
                ],
                "generated_to_observed_final": row["metric_alignment"][
                    "generated_to_observed_final"
                ],
                "face_semantics": row["face_semantics"],
                "outputs": row["source_neutral_outputs"],
            }
            for row in rows
        ],
        "report": str(report_path),
    }
    print(json.dumps(compact, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--builder-script", type=Path, required=True)
    parser.add_argument("--sam3d-bridge-script", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observed-band-scale", type=float, default=math.sqrt(3.0))
    parser.add_argument("--silhouette-dilate-px", type=int, default=16)
    parser.add_argument("--planar-slab-eigenvalue-ratio-max", type=float, default=0.04)
    parser.add_argument("--planar-slab-min-band-m", type=float, default=0.018)
    parser.add_argument("--planar-slab-max-band-m", type=float, default=0.055)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
