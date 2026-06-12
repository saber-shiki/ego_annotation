#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_part_object_blocker_manifest"
CLAIM = (
    "This manifest records object-level blockers for V18 part/relative-motion objects. It prevents partial "
    "part evidence from being interpreted as hidden geometry, part pose, contact ownership, or final object pose."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def rows_by_object(rows: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = require_dict(raw, "row")
        out[str(row.get("object_id"))] = row
    return out


def list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def blocker_state(part_row: dict[str, Any], candidate_row: dict[str, Any] | None, subset_records: list[dict[str, Any]], qc_row: dict[str, Any] | None) -> tuple[str, list[str], list[str]]:
    blockers = set(list_str(part_row.get("blockers")))
    next_evidence = set(list_str(part_row.get("required_next_evidence")))
    accepted_tracks = require_int(part_row.get("accepted_part_track_count"), "accepted part track count")
    if accepted_tracks == 0:
        blockers.add("missing_accepted_part_mask_evidence")
        next_evidence.add("obtain model-produced part plan and tracked part masks overlapping the object")
        return "blocked_missing_part_mask_evidence", sorted(blockers), sorted(next_evidence)
    if candidate_row is not None and require_int(candidate_row.get("rejected_candidate_count", 0), "rejected candidate count") > 0:
        blockers.add("part_model_residual_probes_rejected")
        next_evidence.add("repair rejected part residual probes by improving sparse masks or collecting more shared-frame part surfaces")
        return "blocked_part_model_residual_probes_rejected", sorted(blockers), sorted(next_evidence)
    if candidate_row is not None and subset_records:
        blockers.update(
            {
                "visible_subset_only_not_whole_object",
                "hidden_geometry_not_completed",
                "part_pose_not_estimated",
                "contact_ownership_not_validated",
            }
        )
        if qc_row is not None and qc_row.get("part_motion_qc_state") == "part_motion_confounded_by_sparse_tracks_with_some_stable_support":
            blockers.add("variable_part_motion_confounded_by_sparse_tracks")
            next_evidence.add("improve sparse part masks before interpreting variable part-pair distances")
        next_evidence.add("validate or model only robust visible subset without promoting to object pose")
        return "partial_visible_subset_only_blocked_no_pose", sorted(blockers), sorted(next_evidence)
    blockers.add("part_model_candidate_missing")
    next_evidence.add("extract depth-backed part surfaces and repeat part-motion QC")
    return "blocked_no_part_model_candidate", sorted(blockers), sorted(next_evidence)


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    part_split_path = args.part_split_root / case / "v18_part_split_evidence_report.json"
    completion_path = args.completion_gate_root / case / "v18_object_completion_gate_report.json"
    qc_path = args.part_motion_qc_root / case / "v18_part_motion_qc_report.json"
    candidates_path = args.part_model_candidates_root / case / "v18_part_model_candidates_report.json"
    subset_path = args.visible_part_subset_root / case / "v18_visible_part_subset_archive_report.json"
    part_split = require_dict(load_json(part_split_path), f"{case} part split")
    completion = require_dict(load_json(completion_path), f"{case} completion gate")
    qc = require_dict(load_json(qc_path), f"{case} part motion qc")
    candidates = require_dict(load_json(candidates_path), f"{case} part model candidates")
    subset = require_dict(load_json(subset_path), f"{case} visible part subset")
    completion_by_object = rows_by_object(require_list(completion.get("object_rows"), "completion rows"))
    qc_by_object = rows_by_object(require_list(qc.get("object_rows"), "qc rows"))
    candidate_by_object = rows_by_object(require_list(candidates.get("object_rows"), "candidate object rows"))
    subset_records_by_object: dict[str, list[dict[str, Any]]] = {}
    for raw_record in require_list(subset.get("candidate_records"), "subset candidate records"):
        record = require_dict(raw_record, "subset candidate record")
        subset_records_by_object.setdefault(str(record.get("object_id")), []).append(record)
    object_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for raw_part_row in require_list(part_split.get("object_rows"), "part split object rows"):
        part_row = require_dict(raw_part_row, "part split object row")
        object_id = str(part_row.get("object_id"))
        completion_row = completion_by_object.get(object_id, {})
        qc_row = qc_by_object.get(object_id)
        candidate_row = candidate_by_object.get(object_id)
        subset_records = subset_records_by_object.get(object_id, [])
        state, blockers, next_evidence = blocker_state(part_row, candidate_row, subset_records, qc_row)
        state_counts[state] += 1
        object_rows.append(
            {
                "object_id": object_id,
                "track_id": part_row.get("track_id"),
                "name": part_row.get("name"),
                "model_physical_state_type": part_row.get("model_physical_state_type"),
                "completion_gate_state": completion_row.get("completion_gate_state", part_row.get("completion_gate_state")),
                "part_split_evidence_state": part_row.get("part_split_evidence_state"),
                "part_object_blocker_state": state,
                "accepted_part_track_count": part_row.get("accepted_part_track_count"),
                "accepted_part_track_labels": part_row.get("accepted_part_track_labels"),
                "rejected_part_model_candidate_count": require_int(candidate_row.get("rejected_candidate_count", 0), "rejected candidate count") if candidate_row else 0,
                "visible_subset_candidate_count": len(subset_records),
                "visible_subset_rows": sum(require_int(record.get("archive_row_count"), "archive row count") for record in subset_records),
                "visible_subset_vertices": sum(require_int(record.get("vertex_count"), "vertex count") for record in subset_records),
                "visible_subset_faces": sum(require_int(record.get("face_count"), "face count") for record in subset_records),
                "part_motion_qc_state": qc_row.get("part_motion_qc_state") if qc_row else None,
                "blockers": blockers,
                "required_next_evidence": next_evidence,
                "hidden_geometry_reconstructed": False,
                "articulation_model_ready": False,
                "part_pose_ready": False,
                "contact_ownership_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_object_blocker_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "part_split_evidence": str(part_split_path),
            "object_completion_gate": str(completion_path),
            "part_motion_qc": str(qc_path),
            "part_model_candidates": str(candidates_path),
            "visible_part_subset_archive": str(subset_path),
        },
        "required_part_object_count": len(object_rows),
        "part_object_blocker_state_counts": dict(sorted(state_counts.items())),
        "object_rows": object_rows,
        "rejected_part_model_candidate_count": sum(require_int(row.get("rejected_part_model_candidate_count"), "rejected candidate count") for row in object_rows),
        "hidden_geometry_reconstructed_count": 0,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "contact_ownership_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_object_blocker_manifest_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["part_object_blocker_state_counts"])
    summary = {
        "method": "build_v18_part_object_blocker_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "required_part_object_count": sum(require_int(report.get("required_part_object_count"), "required part object count") for report in reports),
        "part_object_blocker_state_counts": dict(sorted(state_counts.items())),
        "rejected_part_model_candidate_count": sum(require_int(report.get("rejected_part_model_candidate_count"), "rejected candidate count") for report in reports),
        "hidden_geometry_reconstructed_count": 0,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "contact_ownership_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_object_blocker_manifest_report.json"),
                "required_part_object_count": report["required_part_object_count"],
                "part_object_blocker_state_counts": report["part_object_blocker_state_counts"],
                "rejected_part_model_candidate_count": report.get("rejected_part_model_candidate_count"),
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_object_blocker_manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-split-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_split_evidence"))
    parser.add_argument("--completion-gate-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_object_completion_gate"))
    parser.add_argument("--part-motion-qc-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_qc"))
    parser.add_argument("--part-model-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--visible-part-subset-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_part_subset_archive"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
