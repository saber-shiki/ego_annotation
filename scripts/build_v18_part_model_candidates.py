#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict, deque
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

STATUS = "v18_part_model_candidates"
CLAIM = (
    "This artifact records bounded visible part-model candidates from robust stable part-surface relationships. "
    "It does not complete hidden geometry, fit articulation, estimate part pose, or satisfy object-pose requirements."
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


def stable_components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        component: list[str] = []
        q: deque[str] = deque([start])
        seen.add(start)
        while q:
            item = q.popleft()
            component.append(item)
            for nxt in sorted(graph[item]):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if len(component) >= 2:
            components.append(sorted(component))
    return components


def rows_for_component(surface_rows: list[dict[str, Any]], object_id: str, labels: set[str]) -> list[dict[str, Any]]:
    return [row for row in surface_rows if str(row.get("object_id")) == object_id and str(row.get("part_track_label")) in labels]


def rejected_probe_from_pair(index: int, object_id: str, pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": f"{object_id}::rejected_pair_residual_probe::{index:02d}",
        "object_id": object_id,
        "candidate_type": "two_part_relative_motion_residual_probe",
        "candidate_state": "rejected_residual_probe_not_part_model",
        "part_track_labels": [str(pair.get("part_a")), str(pair.get("part_b"))],
        "shared_frame_count": pair.get("shared_frame_count"),
        "frame_min": pair.get("frame_min"),
        "frame_max": pair.get("frame_max"),
        "center_distance_m": pair.get("center_distance_m"),
        "p95_minus_p05_distance_m": pair.get("p95_minus_p05_distance_m"),
        "pair_motion_state": pair.get("pair_motion_state"),
        "pair_qc_state": pair.get("pair_qc_state"),
        "rejection_reasons": list(pair.get("qc_blockers", [])) if isinstance(pair.get("qc_blockers"), list) else [str(pair.get("pair_qc_state"))],
        "eligible_for_hidden_geometry_completion": False,
        "articulation_model_ready": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }


def rejected_probe_from_single(index: int, object_id: str, part: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": f"{object_id}::rejected_single_part_surface_probe::{index:02d}",
        "object_id": object_id,
        "candidate_type": "single_part_visible_surface_residual_probe",
        "candidate_state": "rejected_single_part_not_split_model",
        "part_track_labels": [str(part.get("part_track_label"))],
        "part_surface_quality": part.get("part_surface_quality"),
        "quality_metrics": part.get("quality_metrics"),
        "rejection_reasons": ["requires_at_least_two_semantic_part_tracks_for_part_or_articulation_model"],
        "eligible_for_hidden_geometry_completion": False,
        "articulation_model_ready": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }


def candidate_from_component(index: int, object_id: str, labels: list[str], rows: list[dict[str, Any]], source_edges: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sorted({require_int(row.get("frame_idx"), "frame_idx") for row in rows})
    blockers = [
        "visible_surface_subset_only",
        "hidden_geometry_not_completed",
        "whole_object_not_modeled",
        "confounded_variable_part_pairs_excluded",
        "no_part_pose_estimator_applied",
        "no_articulation_parameter_fit",
    ]
    return {
        "candidate_id": f"{object_id}::visible_rigid_subset::{index:02d}",
        "object_id": object_id,
        "candidate_type": "robust_stable_visible_part_subset",
        "model_scope": "visible_surface_subset_only",
        "part_track_labels": labels,
        "stable_pair_edges": [
            {"part_a": edge.get("part_a"), "part_b": edge.get("part_b"), "pair_qc_state": edge.get("pair_qc_state")}
            for edge in source_edges
        ],
        "surface_frame_count": len(rows),
        "unique_frame_count": len(frames),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "total_vertices": sum(require_int(row.get("vertices"), "vertices") for row in rows),
        "total_faces": sum(require_int(row.get("faces"), "faces") for row in rows),
        "candidate_ready_for_bounded_visible_subset_model": True,
        "eligible_for_hidden_geometry_completion": False,
        "articulation_model_ready": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
        "completion_blockers": blockers,
    }


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    qc_path = args.part_motion_qc_root / case / "v18_part_motion_qc_report.json"
    surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    qc = require_dict(load_json(qc_path), f"{case} part motion qc")
    surfaces = require_dict(load_json(surfaces_path), f"{case} part surfaces")
    surface_rows = [require_dict(raw, "surface row") for raw in require_list(surfaces.get("surface_rows"), "surface rows")]
    object_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    object_state_counts: Counter[str] = Counter()
    for raw_obj in require_list(qc.get("object_rows"), "qc object rows"):
        obj = require_dict(raw_obj, "qc object row")
        object_id = str(obj.get("object_id"))
        stable_edges: list[tuple[str, str]] = []
        stable_edge_rows: list[dict[str, Any]] = []
        confounded_variable_count = 0
        rejected_obj_candidates: list[dict[str, Any]] = []
        pair_rows = [require_dict(raw_pair, "qc pair row") for raw_pair in require_list(obj.get("pair_rows"), "qc pair rows")]
        for raw_pair in pair_rows:
            pair = require_dict(raw_pair, "qc pair row")
            pair = require_dict(raw_pair, "qc pair row")
            if pair.get("pair_qc_state") == "stable_pair_supported_by_robust_surfaces":
                a = str(pair.get("part_a"))
                b = str(pair.get("part_b"))
                stable_edges.append((a, b))
                stable_edge_rows.append(pair)
            else:
                if pair.get("pair_qc_state") == "variable_pair_confounded_by_part_surface_quality":
                    confounded_variable_count += 1
                rejected_obj_candidates.append(rejected_probe_from_pair(len(rejected_obj_candidates) + 1, object_id, pair))
        components = stable_components(stable_edges)
        obj_candidates: list[dict[str, Any]] = []
        for index, labels in enumerate(components, start=1):
            component_edges = [
                edge
                for edge in stable_edge_rows
                if str(edge.get("part_a")) in labels and str(edge.get("part_b")) in labels
            ]
            candidate = candidate_from_component(index, object_id, labels, rows_for_component(surface_rows, object_id, set(labels)), component_edges)
            obj_candidates.append(candidate)
            candidates.append(candidate)
        if not pair_rows:
            for raw_part in require_list(obj.get("part_rows"), "qc part rows"):
                rejected_obj_candidates.append(rejected_probe_from_single(len(rejected_obj_candidates) + 1, object_id, require_dict(raw_part, "qc part row")))
        rejected_candidates.extend(rejected_obj_candidates)
        if obj_candidates:
            state = "visible_stable_part_subset_candidates_only"
        elif rejected_obj_candidates:
            state = "part_model_residual_probes_rejected"
        elif confounded_variable_count:
            state = "no_model_candidate_due_confounded_variable_pairs"
        else:
            state = "no_part_model_candidate"
        object_state_counts[state] += 1
        object_rows.append(
            {
                "object_id": object_id,
                "source_part_motion_qc_state": obj.get("part_motion_qc_state"),
                "part_model_candidate_state": state,
                "stable_component_count": len(components),
                "confounded_variable_pair_count": confounded_variable_count,
                "candidate_ids": [candidate["candidate_id"] for candidate in obj_candidates],
                "rejected_candidate_ids": [candidate["candidate_id"] for candidate in rejected_obj_candidates],
                "rejected_candidate_count": len(rejected_obj_candidates),
                "hidden_geometry_reconstructed": False,
                "articulation_model_ready": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_model_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"v18_part_motion_qc": str(qc_path), "v18_part_visible_surfaces": str(surfaces_path)},
        "candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "object_state_counts": dict(sorted(object_state_counts.items())),
        "object_rows": object_rows,
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "visible_subset_model_candidate_count": len(candidates),
        "hidden_geometry_completion_candidate_count": 0,
        "articulation_model_candidate_count": 0,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_model_candidates_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["object_state_counts"])
    summary = {
        "method": "build_v18_part_model_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "candidate_count": sum(require_int(report.get("candidate_count"), "candidate_count") for report in reports),
        "rejected_candidate_count": sum(require_int(report.get("rejected_candidate_count"), "rejected_candidate_count") for report in reports),
        "visible_subset_model_candidate_count": sum(require_int(report.get("visible_subset_model_candidate_count"), "visible_subset_count") for report in reports),
        "hidden_geometry_completion_candidate_count": 0,
        "articulation_model_candidate_count": 0,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "object_state_counts": dict(sorted(state_counts.items())),
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_model_candidates_report.json"),
                "candidate_count": report["candidate_count"],
                "rejected_candidate_count": report["rejected_candidate_count"],
                "object_state_counts": report["object_state_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_model_candidates_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-motion-qc-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_qc"))
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
