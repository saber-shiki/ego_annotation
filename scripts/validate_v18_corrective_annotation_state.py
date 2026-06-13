#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_case(case: str, root: Path, expected_root: Path, failures: list[str]) -> dict[str, Any]:
    path = root / case / "annotations_v18_corrective_state.json"
    require(path.exists(), f"{case}: missing {path}", failures)
    if not path.exists():
        return {"case": case, "status": "missing"}
    ann = load_json(path)
    src_path = expected_root / case / "annotations_v18_full.json"
    require(src_path.exists(), f"{case}: missing source annotation {src_path}", failures)
    src = load_json(src_path) if src_path.exists() else {}
    frames = ann.get("frames", [])
    counts = ann.get("counts", {}) if isinstance(ann.get("counts"), dict) else {}
    expected_count = src.get("frame_count")
    require(isinstance(frames, list), f"{case}: frames is not a list", failures)
    require(len(frames) == expected_count, f"{case}: frame count {len(frames)} != source {expected_count}", failures)
    require(ann.get("status") == "corrective_state_delta_not_full_v18_closure", f"{case}: incorrect scoped status", failures)
    require(int(counts.get("graph_shifted_mano_states", 0)) > 0, f"{case}: no graph-shifted MANO states", failures)
    require(int(counts.get("graph_hand_states", 0)) == int(counts.get("graph_shifted_mano_states", -1)), f"{case}: graph hand count != shifted MANO count", failures)
    require(int(counts.get("graph_object_se3_states", 0)) > 0, f"{case}: no graph object SE3 states", failures)
    if case == "trash_1050":
        require(int(counts.get("hawor_prior_states", 0)) == 182, f"{case}: expected 182 HaWoR prior states", failures)
        require("object:pink_lid_trash_can_second" in ann.get("rigid_candidate_ids", []), f"{case}: missing pink lid rigid candidate", failures)
    if case == "task5_tomato_960":
        require(ann.get("hawor_measurement_rows") == 0, f"{case}: expected zero HaWoR measurement rows", failures)
        require(int(counts.get("hawor_provisioning_failed_hand_states", 0)) == 1920, f"{case}: expected 1920 HaWoR provisioning-failure hand states", failures)
        require("object:obj_tomato" in ann.get("rigid_candidate_ids", []), f"{case}: missing tomato generic rigid candidate", failures)
        tomato_pose_rows = 0
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for obj in frame.get("objects", []):
                if isinstance(obj, dict) and obj.get("object_id") == "object:obj_tomato":
                    attempt = obj.get("generic_rigid_se3_attempt", {}) if isinstance(obj.get("generic_rigid_se3_attempt"), dict) else {}
                    if attempt.get("stable_pose6_world_from_object") is not None:
                        tomato_pose_rows += 1
        require(tomato_pose_rows > 0, f"{case}: no tomato stable rigid pose rows", failures)
    return {"case": case, "frame_count": len(frames), "counts": counts, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    failures: list[str] = []
    cases = [validate_case(case, args.root, args.source_root, failures) for case in args.cases]
    report = {"status": "ok" if not failures else "failed", "cases": cases, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
