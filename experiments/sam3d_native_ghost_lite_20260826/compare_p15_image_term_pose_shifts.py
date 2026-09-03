#!/usr/bin/env python3
"""Summarize P15 pose shifts introduced by first-hit/silhouette image factors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return {int(row["frame_idx"]): row for row in data.get("pose_rows", []) if isinstance(row, dict)}


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def compare(a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]]) -> dict[str, Any]:
    frames = sorted(set(a) & set(b))
    trans = []
    rot = []
    per_frame = {}
    for idx in frames:
        ta = np.asarray(a[idx]["translation_world_m"], dtype=np.float64)
        tb = np.asarray(b[idx]["translation_world_m"], dtype=np.float64)
        Ra = np.asarray(a[idx]["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        Rb = np.asarray(b[idx]["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        dt = float(np.linalg.norm(ta - tb))
        dr = float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))
        trans.append(dt)
        rot.append(dr)
        per_frame[str(idx)] = {"translation_m": dt, "rotation_deg": dr}
    windows = {}
    for lo, hi in [(0, 83), (84, 127), (128, 149), (132, 149)]:
        selected = [i for i in frames if lo <= i <= hi]
        if not selected:
            continue
        windows[f"{lo}-{hi}"] = {
            "translation_m": stats(np.asarray([per_frame[str(i)]["translation_m"] for i in selected])),
            "rotation_deg": stats(np.asarray([per_frame[str(i)]["rotation_deg"] for i in selected])),
        }
    return {
        "translation_m": stats(np.asarray(trans)),
        "rotation_deg": stats(np.asarray(rot)),
        "windows": windows,
        "per_frame": per_frame,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--front-only-no-image", type=Path, required=True)
    p.add_argument("--image-iter1", type=Path, required=True)
    p.add_argument("--image-iter2", type=Path, required=True)
    p.add_argument("--old-owned-mask", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    rows = {
        "front_only_no_image": load_rows(args.front_only_no_image),
        "image_iter1": load_rows(args.image_iter1),
        "image_iter2": load_rows(args.image_iter2),
        "old_owned_mask": load_rows(args.old_owned_mask),
    }
    report = {
        "method": "compare_p15_image_term_pose_shifts",
        "inputs": {
            "front_only_no_image": str(args.front_only_no_image),
            "image_iter1": str(args.image_iter1),
            "image_iter2": str(args.image_iter2),
            "old_owned_mask": str(args.old_owned_mask),
        },
        "comparisons": {
            "image_iter1_vs_front_only_no_image": compare(rows["image_iter1"], rows["front_only_no_image"]),
            "image_iter2_vs_front_only_no_image": compare(rows["image_iter2"], rows["front_only_no_image"]),
            "image_iter2_vs_image_iter1": compare(rows["image_iter2"], rows["image_iter1"]),
            "image_iter2_vs_old_owned_mask": compare(rows["image_iter2"], rows["old_owned_mask"]),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
