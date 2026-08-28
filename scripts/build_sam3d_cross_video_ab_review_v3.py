#!/usr/bin/env python3
"""Build encoded before/after review videos and contact sheets for a benchmark.

Inputs are already-rendered full-duration overlays. This script does not alter
geometry, poses, masks, or metrics; it only puts frozen shared-Sim(3) before
and after outputs next to each other for direct visual review.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def build_video(before: Path, after: Path, output: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(before), "-i", str(after),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)


def build_sheet(before: Path, after: Path, frame_indices: list[int], output: Path) -> None:
    before_cap = cv2.VideoCapture(str(before))
    after_cap = cv2.VideoCapture(str(after))
    rows = []
    for frame_idx in frame_indices:
        pair = []
        for label, cap in (("BEFORE", before_cap), ("AFTER", after_cap)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {frame_idx} from {label}")
            image = cv2.resize(image, (360, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(
                image, f"{label} f{frame_idx:03d}", (10, 344),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
            )
            pair.append(image)
        rows.append(np.hstack(pair))
    before_cap.release()
    after_cap.release()
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"cannot write {output}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    outputs = []
    for case in config["cases"]:
        name = str(case["name"])
        case_dir = args.benchmark_root / name
        before = case_dir / "before_shared_sim3" / "optimized_object_camera_overlay.mp4"
        after = case_dir / "after_shared_sim3" / "optimized_object_camera_overlay.mp4"
        if not before.is_file() or not after.is_file():
            raise RuntimeError(f"missing before/after videos for {name}")
        output = case_dir / "before_vs_after_shared_sim3.mp4"
        build_video(before, after, output)
        frames = sorted({0, int(case["anchor_frame"]), 30, 60, 90, 120, 149})
        sheet = case_dir / "before_vs_after_contact_sheet.png"
        build_sheet(before, after, frames, sheet)
        outputs.append({
            "name": name,
            "before": str(before),
            "after": str(after),
            "before_vs_after": str(output),
            "contact_sheet": str(sheet),
            "ffprobe": probe(output),
        })
        print(name, output)
    report = {
        "schema": "sam3d_cross_video_before_after_review_v1",
        "diagnostic_only": True,
        "outputs": outputs,
        "claim_scope": "Encoded visual A/B only; no geometry, pose, mask, or metric changes.",
    }
    report_path = args.benchmark_root / "before_vs_after_review_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
