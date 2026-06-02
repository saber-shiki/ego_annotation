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


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_wilor_assets(args.wilor_root, args.mano_right)
    model, cfg, detector, device = load_wilor_backend(args.wilor_root)

    json_path = args.clip.with_suffix(".json")
    actions = load_actions(json_path)
    cap, info = open_video(args.clip)

    frames = []
    started = time.time()
    detected_frames = 0
    detected_hands = 0
    frame_idx = 0
    pbar = tqdm(total=info.frame_count, desc="wilor_full_frame")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            hands = run_wilor_on_frame(model, cfg, detector, device, frame, args.rescale_factor, args.batch_size)
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
        "full_source_timeline": bool(args.max_frames is None and len(frames) == info.frame_count),
        "frames_with_hands": detected_frames,
        "hand_detection_rate": detected_frames / max(1, len(frames)),
        "detected_hands": detected_hands,
        "mean_hands_per_frame": detected_hands / max(1, len(frames)),
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
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
