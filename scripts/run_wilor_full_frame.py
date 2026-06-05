#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from run_v1_wilor_colmap import (
    DEFAULT_CLIP,
    DEFAULT_MANO_RIGHT,
    DEFAULT_WILOR_ROOT,
    caption_for_frame,
    ensure_wilor_assets,
    load_actions,
    load_wilor_backend,
    open_video,
    run_wilor_on_frame,
)


def explicit_hand_sides(actions: list[dict]) -> set[str] | None:
    sides: set[str] = set()
    for action in actions:
        raw = action.get("hand_sides")
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise RuntimeError(f"hand_sides must be a list when present: {action}")
        for item in raw:
            side = str(item).lower()
            if side not in {"left", "right"}:
                raise RuntimeError(f"unknown hand side in hand_sides: {item}")
            sides.add(side)
    return sides or None


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_wilor_assets(args.wilor_root, args.mano_right)
    model, cfg, detector, device = load_wilor_backend(args.wilor_root)

    json_path = args.actions_json if args.actions_json is not None else args.clip.with_suffix(".json")
    actions = load_actions(json_path)
    allowed_sides = explicit_hand_sides(actions)
    cap, info = open_video(args.clip)
    frame_start = 0 if args.frame_start is None else int(args.frame_start)
    frame_end = info.frame_count - 1 if args.frame_end is None else int(args.frame_end)
    if frame_start < 0 or frame_end < frame_start or frame_end >= info.frame_count:
        raise RuntimeError(f"invalid frame window {frame_start}:{frame_end} for {info.frame_count} frames")
    if args.max_frames is not None:
        frame_end = min(frame_end, frame_start + int(args.max_frames) - 1)
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start):
        raise RuntimeError(f"failed to seek to frame {frame_start}")

    frames = []
    started = time.time()
    detected_frames = 0
    detected_hands = 0
    filtered_hands = 0
    frame_idx = frame_start
    pbar = tqdm(total=frame_end - frame_start + 1, desc="wilor_full_frame")
    try:
        while frame_idx <= frame_end:
            ok, frame = cap.read()
            if not ok:
                break
            hands = run_wilor_on_frame(model, cfg, detector, device, frame, args.rescale_factor, args.batch_size)
            if allowed_sides is not None:
                before = len(hands)
                hands = [hand for hand in hands if str(hand.get("side", "")).lower() in allowed_sides]
                filtered_hands += before - len(hands)
            if hands:
                detected_frames += 1
                detected_hands += len(hands)
            frames.append(
                {
                    "frame_idx": frame_idx,
                    "time_s": frame_idx / info.fps,
                    "caption": caption_for_frame(actions, frame_idx),
                    "raw_hands": hands,
                }
            )
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        del model, detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not frames:
        raise RuntimeError("WiLoR received no frames")

    raw_path = args.output_dir / "wilor_raw.json"
    raw_path.write_text(json.dumps({"video": info.__dict__, "frames": frames}, indent=2), encoding="utf-8")

    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "video": info.__dict__,
        "processed_frames": len(frames),
        "source_frame_range": [int(frames[0]["frame_idx"]), int(frames[-1]["frame_idx"])],
        "full_source_timeline": bool(
            args.frame_start is None and args.frame_end is None and args.max_frames is None and len(frames) == info.frame_count
        ),
        "frames_with_hands": detected_frames,
        "hand_detection_rate": detected_frames / max(1, len(frames)),
        "detected_hands": detected_hands,
        "mean_hands_per_frame": detected_hands / max(1, len(frames)),
        "explicit_hand_sides": sorted(allowed_sides) if allowed_sides is not None else None,
        "filtered_hands_by_explicit_side": filtered_hands,
        "elapsed_s": time.time() - started,
        "raw_path": str(raw_path),
    }
    (args.output_dir / "wilor_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/examples/tomato_v1_full/wilor"))
    parser.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    parser.add_argument("--mano-right", type=Path, default=DEFAULT_MANO_RIGHT)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--actions-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
