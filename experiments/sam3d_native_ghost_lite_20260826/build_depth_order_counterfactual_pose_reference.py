#!/usr/bin/env python3
"""Build a depth-order counterfactual observed-only P14/P15 pose reference.

This adapter binds the depth-order counterfactual visible-geometry run to the
same P14/P15 input contract as the baseline, without consuming generated hidden
surfaces. It writes:

* a selected-anchor evidence report carrying the atomic anchor binding;
* an observed-only completion report whose pose/canonical/collision surface is
  the counterfactual anchor Poisson mesh.

The report is prediction-side and CPU-only. Generated SAM3D/TRELLIS faces remain
outside the pose body.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

BASE = Path(
    "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/"
    "backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/"
    "P0014_84ea2dcc_carton_milk_f2370_2519"
)
CF = BASE / "experiments/sam3d_native_ghost_lite_20260826/depth_order_counterfactual_masks_v1"
SCHEMA = "v19_depth_order_counterfactual_observed_pose_reference_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="P0014_84ea2dcc_carton_milk_f2370_2519")
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument(
        "--visible-annotations",
        type=Path,
        default=CF / "visible_geometry_front_only/annotations_v19_visible_geometry.json",
    )
    parser.add_argument(
        "--visible-adapter-report",
        type=Path,
        default=CF / "visible_geometry_front_only/v19_visible_geometry_adapter_report.json",
    )
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--output-dir", type=Path, default=CF / "pose_reference_front_only")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def prepare_output(path: Path, replace: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(require_file(path, "JSON").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path, "artifact").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = require_file(path, "input")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def frame_by_idx(annotations: dict[str, Any], frame_idx: int) -> dict[str, Any]:
    for frame in annotations.get("frames", []):
        if int(frame.get("frame_idx", -1)) == int(frame_idx):
            return frame
    raise RuntimeError(f"anchor frame absent from annotations: {frame_idx}")


def mesh_summary(path: Path) -> dict[str, Any]:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise RuntimeError(f"observed surface is not a non-empty triangle mesh: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bounds_m": np.asarray(mesh.bounds, dtype=float).tolist(),
    }


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args.output_dir, bool(args.replace))
    annotations_path = require_file(args.visible_annotations, "counterfactual visible annotations")
    adapter_path = require_file(args.visible_adapter_report, "counterfactual visible adapter report")
    annotations = load_json(annotations_path)
    adapter = load_json(adapter_path)
    frame = frame_by_idx(annotations, args.anchor_frame)
    objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
    if len(objects) != 1:
        raise RuntimeError(f"anchor frame must contain exactly one object row, got {len(objects)}")
    visible = objects[0].get("visible_geometry_candidate")
    if not isinstance(visible, dict):
        raise RuntimeError("anchor frame lacks visible_geometry_candidate")
    reconstruction = adapter.get("anchor_visible_surface_mesh_reconstruction")
    if not isinstance(reconstruction, dict):
        raise RuntimeError("adapter report lacks anchor_visible_surface_mesh_reconstruction")
    if int(reconstruction.get("anchor_frame_idx", -1)) != int(args.anchor_frame):
        raise RuntimeError("adapter anchor reconstruction does not match requested anchor")
    centroid = np.asarray(visible.get("centroid_world_m"), dtype=np.float64)
    reconstruction_centroid = np.asarray(reconstruction.get("anchor_centroid_world_m"), dtype=np.float64)
    if centroid.shape != (3,) or reconstruction_centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise RuntimeError("invalid anchor centroid")
    centroid_error = float(np.linalg.norm(centroid - reconstruction_centroid))
    if centroid_error > 1.0e-9:
        raise RuntimeError(f"annotation/adapter anchor centroid mismatch: {centroid_error}")
    poisson_path = require_file(Path(str(reconstruction.get("poisson_mesh_path"))), "counterfactual anchor Poisson mesh")
    point_path = require_file(Path(str(reconstruction.get("fused_point_cloud_path"))), "counterfactual anchor visible points")

    binding = {
        "selected_frame_idx": int(args.anchor_frame),
        "visible_geometry_frame_idx": int(args.anchor_frame),
        "canonical_surface_frame_idx": int(args.anchor_frame),
        "selected_vs_canonical_centroid_error_m": 0.0,
        "required_same_frame": True,
        "required_centroid_tolerance_m": 1.0e-6,
        "validated": True,
        "ownership_contract": "depth_order_front_only_counterfactual",
    }
    selected = dict(objects[0])
    for key in ("frame_idx", "time_s", "raw_frame_path", "source_width", "source_height", "manifest_width", "manifest_height", "camera"):
        if key in frame:
            selected[key] = frame[key]
    evidence = {
        "schema": "v19_depth_order_counterfactual_anchor_evidence_v1",
        "status": "ok",
        "method": "build_depth_order_counterfactual_anchor_evidence",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "created_unix_s": time.time(),
        "claim_scope": (
            "Counterfactual selected-anchor evidence for P14/P15. The anchor surface is prediction-side "
            "visible depth support after removing only clearly front-hand first-hit pixels."
        ),
        "selected_frame_idx": int(args.anchor_frame),
        "selected_anchor_atomic_binding": binding,
        "selected": selected,
        "inputs": {
            "visible_annotations": file_ref(annotations_path),
            "visible_adapter_report": file_ref(adapter_path),
            "anchor_poisson_mesh": file_ref(poisson_path),
            "anchor_visible_points": file_ref(point_path),
        },
        "source_artifacts_mutated": False,
    }
    evidence_path = output_dir / "counterfactual_anchor_evidence_report.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    evidence_hash = sha256_file(evidence_path)
    summary = mesh_summary(poisson_path)
    completion = {
        "schema": "v19_observed_metric_surface_only_pose_reference_v1",
        "status": "ok",
        "method": "build_depth_order_counterfactual_observed_only_pose_reference",
        "created_unix_s": time.time(),
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Counterfactual prediction-side observed metric surface only, used as the P14/P15 pose reference. "
            "No generated hidden face is a pose, collision, contact, or sign observation."
        ),
        "inputs": {
            "controlled_report": None,
            "controlled_candidate": None,
            "candidate_builder_report": None,
            "candidate_evidence_report": str(evidence_path),
            "candidate_evidence_report_sha256": evidence_hash,
            "selected_anchor_atomic_binding": binding,
            "candidate_collision_surface": str(poisson_path),
            "shared_observed_mesh": str(poisson_path),
            "visible_annotations": str(annotations_path),
            "visible_adapter_report": str(adapter_path),
        },
        "source_model": "prediction_side_depth_order_counterfactual_observed_metric_surface",
        "observed_band_m": 0.0,
        "outputs": {
            "pose_hypothesis_mesh_labeled": str(poisson_path),
            "completed_mesh_labeled": str(poisson_path),
            "collision_eligible_mesh_labeled": str(poisson_path),
            "observed_depth_surface_labeled_mesh": str(poisson_path),
        },
        "geometry_readiness": {
            "pose_hypothesis_source": "prediction_side_observed_metric_surface_only",
            "collision_surface_source": "prediction_side_observed_metric_surface_only",
            "collision_eligible_mesh": str(poisson_path),
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
    }
    completion_path = output_dir / "counterfactual_observed_only_completion_report.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "inputs": {"visible_annotations": file_ref(annotations_path), "visible_adapter_report": file_ref(adapter_path)},
        "outputs": {
            "anchor_evidence_report": str(evidence_path),
            "anchor_evidence_report_sha256": evidence_hash,
            "completion_report": str(completion_path),
            "completion_report_sha256": sha256_file(completion_path),
            "observed_pose_reference_mesh": str(poisson_path),
            "observed_pose_reference_mesh_sha256": sha256_file(poisson_path),
        },
        "anchor_frame": int(args.anchor_frame),
        "anchor_centroid_world_m": centroid.astype(float).tolist(),
        "mesh_summary": summary,
    }
    report_path = output_dir / "counterfactual_pose_reference_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "report": str(report_path), "completion_report": str(completion_path), "anchor_evidence": str(evidence_path)}, indent=2))


if __name__ == "__main__":
    main()
