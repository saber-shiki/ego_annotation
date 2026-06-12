#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

STATUS = "v18_occlusion_pose_fill_gate"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def occlusion_owner_index(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    report = load_json(path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for graph in report.get("hand_graphs", []):
        if not isinstance(graph, dict):
            continue
        for raw in graph.get("assignments", []):
            if not isinstance(raw, dict) or not isinstance(raw.get("frame_idx"), int):
                continue
            out[(int(raw["frame_idx"]), str(raw.get("hand_side")))] = raw
    return out


def gate_row(hand: dict[str, Any], owner: dict[str, Any] | None) -> dict[str, Any]:
    frame_idx_raw = hand.get("frame_idx")
    frame_idx = int(frame_idx_raw) if isinstance(frame_idx_raw, int) else -1
    hand_side = str(hand.get("hand_side"))
    blockers: list[str] = []
    owner_accepted = bool(owner and owner.get("accepted_occlusion_owner") is True)
    hawor_available = bool(hand.get("hawor_measurement_available") is True)
    hawor_candidate = bool(hand.get("hawor_candidate_present") is True)
    interior_depth = bool(hand.get("interior_metric_depth_compatible") is True)
    baseline_accepted = bool(hand.get("temporal_occlusion_pose_accepted") is True)
    if not owner_accepted:
        blockers.append("accepted_occlusion_owner_missing")
    if not hawor_available:
        blockers.append("hawor_measurement_missing_for_frame_side")
    if not hawor_candidate:
        blockers.append("hawor_candidate_missing_for_frame_side")
    if not interior_depth:
        blockers.append("interior_metric_depth_not_compatible")
    if not baseline_accepted:
        blockers.append("hand_baseline_temporal_occlusion_pose_not_accepted")
    for raw_blocker in hand.get("acceptance_blockers", []):
        if isinstance(raw_blocker, str) and raw_blocker not in blockers:
            blockers.append(raw_blocker)
    accepted = owner_accepted and hawor_available and hawor_candidate and interior_depth and baseline_accepted and not blockers
    return {
        "frame_idx": frame_idx,
        "hand_side": hand_side,
        "pose_fill_gate_claim": "accepted_pose_fill_through_occlusion" if accepted else "pose_fill_blocked_not_accepted",
        "pose_fill_through_occlusion_accepted": accepted,
        "accepted_occlusion_owner": owner_accepted,
        "chosen_owner_object_id": owner.get("chosen_owner_object_id") if isinstance(owner, dict) else None,
        "hand_baseline_state": hand.get("hand_baseline_state"),
        "hawor_measurement_available": hawor_available,
        "hawor_candidate_present": hawor_candidate,
        "hawor_evidence_role": hand.get("hawor_evidence_role"),
        "interior_metric_depth_compatible": interior_depth,
        "hand_baseline_temporal_occlusion_pose_accepted": baseline_accepted,
        "blockers": blockers,
        "source_hand_baseline_row": hand,
        "source_occlusion_owner_assignment": owner,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    hand_path = args.hand_baseline_root / case / "v18_hand_baseline_branch.json"
    owner_path = args.occlusion_owner_graph_root / case / "v18_occlusion_owner_graph_report.json"
    hand_report = load_json(hand_path)
    owners = occlusion_owner_index(owner_path)
    rows: list[dict[str, Any]] = []
    blocker_counts: Counter[str] = Counter()
    for frame in hand_report.get("frames", []):
        if not isinstance(frame, dict):
            continue
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict) or not isinstance(hand.get("frame_idx"), int):
                continue
            key = (int(hand["frame_idx"]), str(hand.get("hand_side")))
            row = gate_row(hand, owners.get(key))
            rows.append(row)
            for blocker in row["blockers"]:
                blocker_counts[str(blocker)] += 1
    accepted = sum(1 for row in rows if row.get("pose_fill_through_occlusion_accepted") is True)
    candidate = sum(1 for row in rows if row.get("hawor_candidate_present") is True or row.get("hawor_measurement_available") is True)
    out = {
        "method": "build_v18_occlusion_pose_fill_gate",
        "status": STATUS,
        "claim": "Gates pose fill-through-occlusion using accepted occlusion ownership plus accepted occluded-hand baseline evidence. It preserves blockers and does not fill poses when either side is unsupported.",
        "case": case,
        "sources": {"hand_baseline_branch": str(hand_path), "occlusion_owner_graph": str(owner_path)},
        "row_count": len(rows),
        "pose_fill_candidate_rows": candidate,
        "pose_fill_through_occlusion_accepted_rows": accepted,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "rows": rows,
        "pose_fill_through_occlusion_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_occlusion_pose_fill_gate_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_occlusion_pose_fill_gate",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {"case": r["case"], "row_count": r["row_count"], "pose_fill_candidate_rows": r["pose_fill_candidate_rows"], "pose_fill_through_occlusion_accepted_rows": r["pose_fill_through_occlusion_accepted_rows"]}
            for r in reports
        ],
        "claim_scope": "explicit_occlusion_pose_fill_gate_no_unsupported_fill",
    }
    write_json(args.output_root / "v18_occlusion_pose_fill_gate_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-baseline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_hand_baseline_branch"))
    parser.add_argument("--occlusion-owner-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_pose_fill_gate"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
