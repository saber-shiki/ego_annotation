#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    require(report.get("triangle_nonpenetration_complete") is False, f"{case}: overclaims complete triangle nonpenetration")
    evaluated = 0
    penetration = 0
    watertight = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        require(row.get("triangle_nonpenetration_complete") is False, f"{case}: row overclaims complete nonpenetration")
        claim = row.get("triangle_nonpenetration_claim")
        if claim in {"local_triangle_penetration_evidence", "local_triangle_no_penetration_beyond_tolerance_evidence"}:
            evaluated += 1
            require("not_watertight_sdf" in str(row.get("local_triangle_signed_distance_semantics")), f"{case}: signed semantics not scoped")
            require(isinstance(row.get("mesh_watertight_by_edges"), bool), f"{case}: missing watertight diagnostic")
            if row.get("mesh_watertight_by_edges") is True:
                watertight += 1
            for key in ["min_triangle_unsigned_distance_m", "min_local_triangle_signed_distance_m", "negative_triangle_signed_distance_fraction"]:
                val = row.get(key)
                require(isinstance(val, (float, int)) and math.isfinite(float(val)), f"{case}: invalid {key}")
            frac = float(row.get("negative_triangle_signed_distance_fraction"))
            require(0.0 <= frac <= 1.0, f"{case}: invalid negative fraction")
            if row.get("local_triangle_penetration_detected") is True:
                penetration += 1
        else:
            require(claim == "blocked", f"{case}: unsupported claim {claim}")
    require(evaluated == int(report.get("evaluated_triangle_rows", -1)), f"{case}: evaluated count mismatch")
    require(penetration == int(report.get("local_triangle_penetration_detected_rows", -1)), f"{case}: penetration count mismatch")
    require(watertight == int(report.get("mesh_watertight_rows", -1)), f"{case}: watertight count mismatch")
    require(watertight == 0, f"{case}: unexpected watertight rows need separate acceptance review")
    return {"case": case, "triangle_rows": len(rows), "evaluated_triangle_rows": evaluated, "local_triangle_penetration_detected_rows": penetration, "mesh_watertight_rows": watertight, "triangle_nonpenetration_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_triangle_nonpenetration_evidence_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
