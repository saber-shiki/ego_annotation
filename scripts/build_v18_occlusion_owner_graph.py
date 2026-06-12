#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

STATUS = "v18_occlusion_owner_graph"
NONE_OWNER = "__none__"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def mesh_support(row: dict[str, Any]) -> float:
    support = row.get("mesh_contact_temporal_support")
    if isinstance(support, dict):
        return max(0.0, min(1.0, finite_float(support.get("max_support"), 0.0)))
    return 0.0


def candidate_energy(row: dict[str, Any]) -> float:
    iou = max(0.0, min(1.0, finite_float(row.get("bbox_iou"), 0.0)))
    coverage = max(0.0, min(1.0, finite_float(row.get("hand_box_coverage_by_object_box"), 0.0)))
    support = mesh_support(row)
    depth_accepted = bool(row.get("accepted_occlusion_owner") is True)
    support_score = max(0.50 * coverage + 0.30 * iou + 0.20 * support, support * 0.75)
    energy = (1.0 - support_score) ** 2
    if not depth_accepted:
        energy += 0.35
    else:
        energy *= 0.35
    return float(energy)


def none_energy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    best = max(max(0.0, min(1.0, 0.5 * finite_float(r.get("hand_box_coverage_by_object_box"), 0.0) + 0.3 * finite_float(r.get("bbox_iou"), 0.0) + 0.2 * mesh_support(r))) for r in rows)
    return float(0.20 + 0.85 * best)


def transition_energy(prev: str, cur: str, switch_penalty: float, onoff_penalty: float) -> float:
    if prev == cur:
        return 0.0
    if prev == NONE_OWNER or cur == NONE_OWNER:
        return onoff_penalty
    return switch_penalty


def solve_sequence(hand_side: str, rows_by_frame: dict[int, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    frame_indices = sorted(rows_by_frame)
    if not frame_indices:
        return {"hand_side": hand_side, "assignments": [], "accepted_occlusion_owner_rows": 0, "selected_owner_rows": 0}
    row_by_state: dict[int, dict[str, dict[str, Any]]] = {}
    unary_by_frame: dict[int, dict[str, float]] = {}
    for frame_idx in frame_indices:
        best_rows: dict[str, dict[str, Any]] = {}
        for row in rows_by_frame[frame_idx]:
            object_id = str(row.get("object_id"))
            prev = best_rows.get(object_id)
            if prev is None or candidate_energy(row) < candidate_energy(prev):
                best_rows[object_id] = row
        row_by_state[frame_idx] = best_rows
        unary = {obj: candidate_energy(row) for obj, row in best_rows.items()}
        unary[NONE_OWNER] = none_energy(rows_by_frame[frame_idx])
        unary_by_frame[frame_idx] = unary
    costs: list[dict[str, float]] = [dict(unary_by_frame[frame_indices[0]])]
    back: list[dict[str, str | None]] = [{state: None for state in unary_by_frame[frame_indices[0]]}]
    for i in range(1, len(frame_indices)):
        cur_costs: dict[str, float] = {}
        cur_back: dict[str, str | None] = {}
        frame_gap = frame_indices[i] - frame_indices[i - 1]
        use_temporal_transition = frame_gap <= args.max_temporal_gap_frames
        for cur, unary in unary_by_frame[frame_indices[i]].items():
            best_prev = None
            best_cost = float("inf")
            for prev, prev_cost in costs[i - 1].items():
                temporal = transition_energy(prev, cur, args.object_switch_penalty, args.owner_onoff_penalty) if use_temporal_transition else 0.0
                total = prev_cost + temporal + unary
                if total < best_cost:
                    best_cost = total
                    best_prev = prev
            cur_costs[cur] = best_cost
            cur_back[cur] = best_prev
        costs.append(cur_costs)
        back.append(cur_back)
    final = min(costs[-1], key=lambda state: costs[-1][state])
    path = [final]
    for i in range(len(frame_indices) - 1, 0, -1):
        prev = back[i][path[-1]]
        path.append(prev if prev is not None else NONE_OWNER)
    path.reverse()
    assignments: list[dict[str, Any]] = []
    accepted = 0
    selected = 0
    for i, frame_idx in enumerate(frame_indices):
        chosen = path[i]
        energies = unary_by_frame[frame_idx]
        ranked = sorted(energies.items(), key=lambda kv: kv[1])
        next_best = next((energy for state, energy in ranked if state != chosen), None)
        margin = (float(next_best) - float(energies[chosen])) if next_best is not None else None
        row = row_by_state[frame_idx].get(chosen) if chosen != NONE_OWNER else None
        accepted_row = bool(row and row.get("accepted_occlusion_owner") is True and margin is not None and margin >= args.accept_energy_margin)
        if chosen != NONE_OWNER:
            selected += 1
        if accepted_row:
            accepted += 1
        assignments.append(
            {
                "frame_idx": frame_idx,
                "hand_side": hand_side,
                "chosen_owner_object_id": None if chosen == NONE_OWNER else chosen,
                "chosen_unary_energy": float(energies[chosen]),
                "next_best_unary_energy": float(next_best) if next_best is not None else None,
                "unary_energy_margin": margin,
                "previous_candidate_frame_gap": (frame_idx - frame_indices[i - 1]) if i > 0 else None,
                "temporal_transition_applied": bool(i > 0 and (frame_idx - frame_indices[i - 1]) <= args.max_temporal_gap_frames),
                "accepted_occlusion_owner": accepted_row,
                "occlusion_owner_claim": "accepted_occlusion_owner_by_depth_order_temporal_graph" if accepted_row else ("temporal_graph_selected_not_accepted" if chosen != NONE_OWNER else "no_occlusion_owner_selected"),
                "source_row": row,
                "candidate_unary_energies": {None if state == NONE_OWNER else state: float(energy) for state, energy in ranked},
            }
        )
    return {"hand_side": hand_side, "assignment_count": len(assignments), "selected_owner_rows": selected, "accepted_occlusion_owner_rows": accepted, "assignments": assignments}


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.occlusion_mesh_root / case / "v18_occlusion_mesh_owner_evidence_report.json"
    source = load_json(source_path)
    by_hand_frame: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for raw in source.get("rows", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("frame_idx"), int):
            continue
        by_hand_frame[str(raw.get("hand_side"))][int(raw["frame_idx"])].append(raw)
    hand_graphs = [solve_sequence(hand_side, frames, args) for hand_side, frames in sorted(by_hand_frame.items())]
    assignment_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for graph in hand_graphs:
        for assignment in graph.get("assignments", []):
            if not isinstance(assignment, dict):
                continue
            chosen = assignment.get("chosen_owner_object_id")
            if chosen is not None:
                assignment_by_key[(int(assignment["frame_idx"]), str(assignment["hand_side"]), str(chosen))] = assignment
    annotated_rows: list[dict[str, Any]] = []
    selected = 0
    accepted = 0
    for raw in source.get("rows", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("frame_idx"), int):
            continue
        key = (int(raw["frame_idx"]), str(raw.get("hand_side")), str(raw.get("object_id")))
        assignment = assignment_by_key.get(key)
        selected_row = assignment is not None
        accepted_row = bool(assignment and assignment.get("accepted_occlusion_owner") is True)
        selected += int(selected_row)
        accepted += int(accepted_row)
        annotated_rows.append({**raw, "temporal_graph_assignment": assignment, "selected_by_occlusion_graph": selected_row, "accepted_occlusion_owner": accepted_row, "occlusion_owner_claim": "accepted_occlusion_owner_by_depth_order_temporal_graph" if accepted_row else ("temporal_graph_selected_not_accepted" if selected_row else "not_selected_by_occlusion_graph")})
    out = {
        "method": "build_v18_occlusion_owner_graph",
        "status": STATUS,
        "claim": "Solves a temporal object-or-none occlusion-owner graph over bounded candidates with mesh-contact support. It selects candidates for evidence but accepts ownership only when source depth-order evidence accepted it.",
        "case": case,
        "sources": {"occlusion_mesh_owner_evidence": str(source_path)},
        "parameters": {"object_switch_penalty": args.object_switch_penalty, "owner_onoff_penalty": args.owner_onoff_penalty, "accept_energy_margin": args.accept_energy_margin, "max_temporal_gap_frames": args.max_temporal_gap_frames},
        "hand_graphs": hand_graphs,
        "rows": annotated_rows,
        "selected_occlusion_owner_rows": selected,
        "accepted_occlusion_owner_rows": accepted,
        "occlusion_ownership_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_occlusion_owner_graph_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {"method": "build_v18_occlusion_owner_graph", "status": STATUS, "case_count": len(reports), "cases": [{"case": r["case"], "selected_occlusion_owner_rows": r["selected_occlusion_owner_rows"], "accepted_occlusion_owner_rows": r["accepted_occlusion_owner_rows"], "occlusion_ownership_complete": r["occlusion_ownership_complete"]} for r in reports], "claim_scope": "temporal_occlusion_owner_selection_not_unsupported_acceptance"}
    write_json(args.output_root / "v18_occlusion_owner_graph_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occlusion-mesh-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_mesh_owner_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--object-switch-penalty", type=float, default=0.40)
    parser.add_argument("--owner-onoff-penalty", type=float, default=0.25)
    parser.add_argument("--accept-energy-margin", type=float, default=0.25)
    parser.add_argument("--max-temporal-gap-frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
