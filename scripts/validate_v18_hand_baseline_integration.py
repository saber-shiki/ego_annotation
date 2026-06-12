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
    ann = load_json(path)
    case = str(ann.get("case"))
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, f"{case}: missing frames")
    modules_raw = ann.get("modules")
    modules: dict[str, Any] = modules_raw if isinstance(modules_raw, dict) else {}
    require("hand_baseline_evidence" in str(modules.get("hand_branch")), f"{case}: hand module does not report baseline integration")
    hand_rows = 0
    baseline_rows = 0
    hawor_rows = 0
    wilor_rows = 0
    blocker_rows = 0
    accepted_occlusion_pose = 0
    missing_rows = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            hand_rows += 1
            baseline_raw = hand.get("hand_baseline_branch")
            baseline: dict[str, Any] = baseline_raw if isinstance(baseline_raw, dict) else {}
            if baseline.get("hand_baseline_state"):
                baseline_rows += 1
            if baseline.get("state") == "missing_hand_baseline_branch_row":
                missing_rows += 1
            if baseline.get("hawor_candidate_present") is True:
                hawor_rows += 1
            if baseline.get("wilor_measurement_available") is True:
                wilor_rows += 1
            blockers = baseline.get("acceptance_blockers")
            if isinstance(blockers, list) and len(blockers) > 0:
                blocker_rows += 1
            if baseline.get("temporal_occlusion_pose_accepted") is True:
                accepted_occlusion_pose += 1
            require(baseline.get("pose_claim") in {"no_occluded_pose_accepted_from_current_hand_baseline", None}, f"{case}: unsupported hand pose claim")
    require(hand_rows > 0, f"{case}: no hands")
    require(baseline_rows == hand_rows, f"{case}: not every hand has baseline row")
    require(missing_rows == 0, f"{case}: missing baseline integration rows")
    require(blocker_rows > 0, f"{case}: no blockers preserved")
    require(accepted_occlusion_pose == 0, f"{case}: accepted occlusion hand pose unexpectedly present")
    return {"case": case, "hand_rows": hand_rows, "baseline_rows": baseline_rows, "hawor_rows": hawor_rows, "wilor_rows": wilor_rows, "blocker_rows": blocker_rows, "accepted_occlusion_pose_rows": accepted_occlusion_pose}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "annotations_v18_full.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
