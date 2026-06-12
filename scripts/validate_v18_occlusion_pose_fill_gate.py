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
    require(report.get("pose_fill_through_occlusion_complete") is False, f"{case}: overclaims complete pose fill")
    accepted = 0
    candidate = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        if row.get("hawor_candidate_present") is True or row.get("hawor_measurement_available") is True:
            candidate += 1
        if row.get("pose_fill_through_occlusion_accepted") is True:
            accepted += 1
            require(row.get("accepted_occlusion_owner") is True, f"{case}: pose fill accepted without accepted owner")
            require(row.get("hand_baseline_temporal_occlusion_pose_accepted") is True, f"{case}: pose fill accepted without accepted hand baseline")
            require(not row.get("blockers"), f"{case}: pose fill accepted despite blockers")
        else:
            require(row.get("pose_fill_gate_claim") == "pose_fill_blocked_not_accepted", f"{case}: unsupported pose fill claim")
            require(isinstance(row.get("blockers"), list) and len(row.get("blockers")) > 0, f"{case}: blocked pose fill lacks blockers")
    require(accepted == int(report.get("pose_fill_through_occlusion_accepted_rows", -1)), f"{case}: accepted count mismatch")
    require(candidate == int(report.get("pose_fill_candidate_rows", -1)), f"{case}: candidate count mismatch")
    require(accepted == 0, f"{case}: unexpected accepted pose fill without reviewed support")
    return {"case": case, "row_count": len(rows), "pose_fill_candidate_rows": candidate, "pose_fill_through_occlusion_accepted_rows": accepted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_pose_fill_gate"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_occlusion_pose_fill_gate_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
