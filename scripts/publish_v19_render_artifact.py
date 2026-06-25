#!/usr/bin/env python3
"""Publish a V19 render branch as clear full-duration user-facing videos.

The script does not change physical state.  It consumes existing state-driven
render videos, adds a stable explanatory legend/metric banner, writes a
publication report, optional review stills, and can atomically update canonical
``v19_overlay.mp4``, ``v19_world.mp4``, and ``v19_side_by_side.mp4`` symlinks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIDEO_NAMES = {
    "overlay": "v19_overlay.mp4",
    "world": "v19_world.mp4",
    "side_by_side": "v19_side_by_side.mp4",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def metric_value(summary: dict[str, Any], key: str, stat: str = "median") -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    raw = value.get(stat)
    if raw is None:
        return None
    try:
        val = float(raw)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def side_summary(interval_state: dict[str, Any], side: str) -> dict[str, Any]:
    for row in interval_state.get("intervals", []) if isinstance(interval_state.get("intervals"), list) else []:
        if isinstance(row, dict) and str(row.get("interval_id", "")).startswith(f"{side}_"):
            return row
    return {}


def summarize_interval(interval_state: Path | None) -> dict[str, Any]:
    if interval_state is None:
        return {"summary_text": "interval metrics unavailable", "sides": {}}
    payload = load_json(interval_state)
    sides: dict[str, Any] = {}
    tokens: list[str] = []
    for side in ("left", "right"):
        row = side_summary(payload, side)
        gap = metric_value(row, "contact_patch_final_abs_normal_gap_m")
        shift = metric_value(row, "visible_joint_shift_max_px")
        trans = metric_value(row, "translation_delta_norm_m")
        closed = row.get("active_set_closed")
        sides[side] = {
            "contact_patch_final_abs_normal_gap_m_median": gap,
            "visible_joint_shift_max_px_median": shift,
            "translation_delta_norm_m_median": trans,
            "active_set_closed": closed,
        }
        gap_txt = "gap=?" if gap is None else f"gap {gap * 1000.0:.1f}mm"
        shift_txt = "shift=?" if shift is None else f"shift {shift:.1f}px"
        closed_txt = f"closed {closed}" if closed is not None else "closed ?"
        tokens.append(f"{side[0].upper()}: {gap_txt}, {shift_txt}, {closed_txt}")
    return {"summary_text": " | ".join(tokens), "sides": sides, "interval_state": str(interval_state)}


def put_text_fit(image: np.ndarray, text: str, org: tuple[int, int], max_width: int, scale: float, color: tuple[int, int, int], thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    use_scale = scale
    while use_scale > 0.35:
        (tw, _), _ = cv2.getTextSize(text, font, use_scale, thickness)
        if tw <= max_width:
            break
        use_scale *= 0.92
    cv2.putText(image, text, org, font, use_scale, color, thickness, cv2.LINE_AA)


def banner(width: int, title: str, subtitle: str, metrics: str, height: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:] = (12, 12, 12)
    cv2.rectangle(out, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    put_text_fit(out, title, (14, 27), width - 28, 0.68, (255, 255, 255), 2)
    put_text_fit(out, subtitle, (14, 51), width - 28, 0.48, (215, 230, 255), 1)
    put_text_fit(out, metrics, (14, height - 13), width - 28, 0.45, (160, 240, 255), 1)
    return out


def annotate_frame(frame: np.ndarray, title: str, subtitle: str, metrics: str, banner_h: int) -> np.ndarray:
    top = banner(frame.shape[1], title, subtitle, metrics, banner_h)
    return np.vstack([top, frame])


def write_video_with_banner(
    src: Path,
    dst: Path,
    title: str,
    subtitle: str,
    metrics: str,
    still_frames: set[int],
    still_dir: Path,
    banner_h: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source video {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height + banner_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open writer {dst}")
    written = 0
    stills: list[str] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = annotate_frame(frame, title, subtitle, metrics, banner_h)
            writer.write(annotated)
            if written in still_frames:
                still_dir.mkdir(parents=True, exist_ok=True)
                out = still_dir / f"{dst.stem}_frame_{written:06d}.jpg"
                if not cv2.imwrite(str(out), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {out}")
                stills.append(str(out))
            written += 1
    finally:
        writer.release()
        cap.release()
    return {
        "source": str(src),
        "output": str(dst),
        "source_frames_reported": frame_count,
        "frames_written": int(written),
        "fps": fps,
        "source_size": [width, height],
        "output_size": [width, height + banner_h],
        "stills": stills,
    }


def atomic_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(f".{link.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link)


def publish(args: argparse.Namespace) -> dict[str, Any]:
    metrics = summarize_interval(args.interval_state)
    subtitle = args.subtitle or "green=rigid object | orange halo=current/source MANO | cyan/yellow=optimized interval hypothesis | UNCERTAIN=not accepted contact closure"
    title = args.title
    out_dir = args.output_dir
    still_dir = out_dir / "review_stills"
    stills = {int(x) for x in args.still_frames}
    videos = {
        "overlay": args.overlay,
        "world": args.world,
        "side_by_side": args.side_by_side,
    }
    outputs: dict[str, Any] = {}
    for kind, src in videos.items():
        dst = out_dir / VIDEO_NAMES[kind]
        outputs[kind] = write_video_with_banner(
            src=src,
            dst=dst,
            title=title if kind != "world" else f"{title} — world view",
            subtitle=subtitle,
            metrics=metrics["summary_text"],
            still_frames=stills,
            still_dir=still_dir,
            banner_h=int(args.banner_height),
        )
    canonical_updates: dict[str, str] = {}
    if args.canonical_dir is not None:
        for kind, name in VIDEO_NAMES.items():
            target = (out_dir / name).resolve()
            link = args.canonical_dir / name
            if args.replace_canonical:
                atomic_symlink(target, link)
            canonical_updates[str(link)] = str(target)
    report = {
        "status": "ok",
        "method": "publish_v19_render_artifact",
        "claim_scope": "presentation/publishing of existing state-driven render; physical marks come from source render videos",
        "title": title,
        "subtitle": subtitle,
        "metrics": metrics,
        "outputs": outputs,
        "canonical_updates": canonical_updates,
        "replace_canonical": bool(args.replace_canonical),
    }
    write_json(out_dir / "v19_published_render_report.json", report)
    print(json.dumps(report, indent=2)[:20000])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--side-by-side", type=Path, required=True)
    parser.add_argument("--interval-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--replace-canonical", action="store_true")
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle")
    parser.add_argument("--banner-height", type=int, default=78)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[690, 700, 720, 725])
    return parser.parse_args()


def main() -> None:
    publish(parse_args())


if __name__ == "__main__":
    main()
