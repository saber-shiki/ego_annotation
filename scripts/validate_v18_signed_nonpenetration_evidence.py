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
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: signed rows missing")
    require(report.get("signed_nonpenetration_complete") is False, f"{case}: overclaims complete signed nonpenetration")
    accepted = int(report.get("accepted_contact_rows", -1))
    require(accepted >= len(rows), f"{case}: more signed rows than accepted contacts")
    evaluated = 0
    penetration = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        require(row.get("signed_nonpenetration_complete") is False, f"{case}: row overclaims complete nonpenetration")
        claim = row.get("signed_nonpenetration_claim")
        if claim in {"local_normal_penetration_evidence", "local_normal_no_penetration_beyond_tolerance_evidence"}:
            evaluated += 1
            require(row.get("local_signed_distance_semantics") == "nearest_face_centroid_normal_projection_not_watertight_sdf", f"{case}: signed semantics missing")
            require(isinstance(row.get("min_local_signed_distance_m"), (int, float)), f"{case}: missing signed distance")
            if row.get("local_penetration_detected") is True:
                penetration += 1
        else:
            require(claim == "blocked", f"{case}: unsupported signed claim {claim}")
    require(evaluated == int(report.get("evaluated_signed_rows", -1)), f"{case}: evaluated count mismatch")
    require(penetration == int(report.get("local_penetration_detected_rows", -1)), f"{case}: penetration count mismatch")
    return {"case": case, "signed_rows": len(rows), "evaluated_signed_rows": evaluated, "local_penetration_detected_rows": penetration, "signed_nonpenetration_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_signed_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_signed_nonpenetration_evidence_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
