#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def validate_case(path: Path) -> dict[str, Any]:
    report = load_json(path)
    case = str(report.get("case"))
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: rows missing")
    require(report.get("occlusion_ownership_complete") is False, f"{case}: overclaims complete occlusion ownership")
    selected = 0
    accepted = 0
    max_gap = int(report.get("parameters", {}).get("max_temporal_gap_frames", 30)) if isinstance(report.get("parameters"), dict) else 30
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        if row.get("selected_by_occlusion_graph") is True:
            selected += 1
            assignment_raw = row.get("temporal_graph_assignment")
            assignment_check: dict[str, Any] = assignment_raw if isinstance(assignment_raw, dict) else {}
            require(isinstance(row.get("temporal_graph_assignment"), dict), f"{case}: selected row lacks assignment")
            gap = assignment_check.get("previous_candidate_frame_gap")
            if isinstance(gap, int) and gap > max_gap:
                require(assignment_check.get("temporal_transition_applied") is False, f"{case}: temporal transition applied across large gap")
        if row.get("accepted_occlusion_owner") is True:
            accepted += 1
            assignment_raw = row.get("temporal_graph_assignment")
            assignment: dict[str, Any] = assignment_raw if isinstance(assignment_raw, dict) else {}
            source_raw = assignment.get("source_row")
            source: dict[str, Any] = source_raw if isinstance(source_raw, dict) else {}
            require(source.get("accepted_occlusion_owner") is True, f"{case}: unsupported accepted occlusion owner")
        else:
            require(row.get("occlusion_owner_claim") in {"temporal_graph_selected_not_accepted", "not_selected_by_occlusion_graph"}, f"{case}: unsupported nonaccepted claim")
    require(selected == int(report.get("selected_occlusion_owner_rows", -1)), f"{case}: selected count mismatch")
    require(accepted == int(report.get("accepted_occlusion_owner_rows", -1)), f"{case}: accepted count mismatch")
    return {"case": case, "selected_occlusion_owner_rows": selected, "accepted_occlusion_owner_rows": accepted, "occlusion_ownership_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_occlusion_owner_graph_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
