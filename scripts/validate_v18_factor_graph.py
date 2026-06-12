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
    case = ann.get("case")
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, f"{case}: frames missing")
    fg = ann.get("factor_graph_summary")
    require(isinstance(fg, dict), f"{case}: factor_graph_summary missing")
    objective = fg.get("objective")
    require(isinstance(objective, dict), f"{case}: objective missing")
    energy_initial = float(objective.get("energy_initial"))
    energy_after = float(objective.get("energy_after"))
    require(energy_after <= energy_initial + 1e-6, f"{case}: graph energy did not decrease")
    variable_counts = fg.get("variable_counts")
    factor_counts = fg.get("factor_counts")
    require(isinstance(variable_counts, dict) and isinstance(factor_counts, dict), f"{case}: counts missing")
    for key in ["camera_depth_correction", "hand_state", "object_se3", "part_se3", "contact_switch"]:
        require(int(variable_counts.get(key, 0)) > 0, f"{case}: missing {key} variables")
    for key in ["camera_depth_correction_observation", "hand_state_observation", "object_se3_observation", "part_se3_observation", "contact_switch_discrete"]:
        require(int(factor_counts.get(key, 0)) > 0, f"{case}: missing {key} factors")
    implemented_status = fg.get("implemented_variable_status")
    spec_gaps = fg.get("spec_factor_gaps_remaining")
    require(isinstance(implemented_status, dict), f"{case}: implemented variable status missing")
    require("camera_depth_correction" in implemented_status and "observed_depth_scale_correction" in str(implemented_status.get("camera_depth_correction")), f"{case}: camera/depth correction observation status not explicit")
    require("part_se3" in implemented_status and "pca_rotvec" in str(implemented_status.get("part_se3")), f"{case}: part SE3 PCA status not explicit")
    require(isinstance(spec_gaps, list) and len(spec_gaps) > 0, f"{case}: spec factor gaps missing")
    require(any("camera_depth_correction_is_scale_only" in str(gap) for gap in spec_gaps), f"{case}: camera/depth correction limitation not explicit")
    require(any("visible_surface_PCA" in str(gap) for gap in spec_gaps), f"{case}: visible-surface part SE3 limitation not explicit")
    inference = fg.get("inference")
    require(isinstance(inference, dict), f"{case}: inference missing")
    require("SciPy" in str(inference.get("continuous_method")), f"{case}: continuous solve is not SciPy-backed")
    series = inference.get("series_summaries")
    require(isinstance(series, dict) and len(series) > 0, f"{case}: series summaries missing")
    object_se3_series = {k: v for k, v in series.items() if str(k).startswith("object_se3::") and isinstance(v, dict)}
    part_se3_series = {k: v for k, v in series.items() if str(k).startswith("part_se3::") and isinstance(v, dict)}
    require(len(object_se3_series) > 0, f"{case}: object SE3 series missing")
    require(len(part_se3_series) > 0, f"{case}: part SE3 series missing")
    object_6d_count = sum(1 for v in object_se3_series.values() if int(v.get("dimension", 0)) == 6)
    part_6d_count = sum(1 for v in part_se3_series.values() if int(v.get("dimension", 0)) == 6)
    require(object_6d_count > 0, f"{case}: no 6D object SE3 series")
    require(part_6d_count > 0, f"{case}: no 6D part SE3 series")
    frame_with_graph = 0
    for frame in frames:
        g = frame.get("factor_graph_solution")
        if isinstance(g, dict) and isinstance(g.get("variables"), dict) and isinstance(g.get("objective"), dict):
            frame_with_graph += 1
    require(frame_with_graph == len(frames), f"{case}: not every frame has graph solution")
    return {
        "case": case,
        "frame_count": len(frames),
        "energy_initial": energy_initial,
        "energy_after": energy_after,
        "energy_delta": energy_initial - energy_after,
        "variable_counts": variable_counts,
        "factor_counts": factor_counts,
        "object_6d_series_count": object_6d_count,
        "part_6d_series_count": part_6d_count,
        "frame_with_graph_count": frame_with_graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        rows.append(validate_case(args.root / case / "annotations_v18_full.json"))
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
