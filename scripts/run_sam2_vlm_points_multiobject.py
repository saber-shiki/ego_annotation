#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


SAM2_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)


COLORS_BGR = [
    (60, 80, 255),
    (80, 230, 80),
    (255, 150, 60),
    (60, 220, 255),
    (255, 80, 220),
    (180, 120, 255),
    (120, 255, 220),
]


@dataclass(frozen=True)
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class Track:
    obj_id: int
    track_id: str
    description: str
    prompt_path: Path
    prompts: dict[int, dict]
    payload: dict


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_video(path: Path) -> tuple[cv2.VideoCapture, ClipInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    info = ClipInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {info}")
    return cap, info


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read video frame {frame_idx}")
    return frame


def put_caption(frame: np.ndarray, caption: str, frame_idx: int) -> None:
    text = f"{frame_idx:04d}  {caption}"
    cv2.rectangle(frame, (0, frame.shape[0] - 34), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def prompt_rows(payload: dict) -> dict[int, dict]:
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError("point prompt payload has no point_prompts list")
    return {int(row["frame_idx"]): row for row in rows}


def load_tracks(point_root: Path) -> list[Track]:
    files = sorted(point_root.glob("*/object_point_prompts_vlm.json"))
    if not files:
        raise RuntimeError(f"no object_point_prompts_vlm.json files under {point_root}")
    tracks = []
    seen = set()
    for i, path in enumerate(files, start=1):
        payload = load_json(path)
        track_id = str(payload["track_id"])
        if track_id in seen:
            raise RuntimeError(f"duplicate track_id: {track_id}")
        seen.add(track_id)
        tracks.append(
            Track(
                obj_id=i,
                track_id=track_id,
                description=str(payload["description"]),
                prompt_path=path,
                prompts=prompt_rows(payload),
                payload=payload,
            )
        )
    return tracks


def selected_frames(frame_start: int, frame_end: int) -> list[dict]:
    if frame_end < frame_start:
        raise RuntimeError(f"invalid frame range {frame_start}:{frame_end}")
    return [{"frame_idx": idx} for idx in range(frame_start, frame_end + 1)]


def extract_frames(clip: Path, frames: list[dict], output_dir: Path, image_width: int) -> Path:
    frame_dir = output_dir / "sam2_shared_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(clip)
    image_height = int(round(info.height * image_width / info.width))
    try:
        for local_idx, frame in enumerate(frames):
            source_idx = int(frame["frame_idx"])
            image = read_video_frame(cap, source_idx)
            resized = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            path = frame_dir / f"{local_idx:06d}.jpg"
            if not cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"failed to write {path}")
    finally:
        cap.release()
    return frame_dir


def scaled_points(points: list[dict], prompt_size: tuple[int, int], video_size: tuple[int, int]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    scale = np.asarray([video_size[0] / prompt_size[0], video_size[1] / prompt_size[1]], dtype=np.float32)
    return np.asarray([[float(point["x"]) * scale[0], float(point["y"]) * scale[1]] for point in points], dtype=np.float32)


def prompt_points(track: Track, source_idx: int, tracks: list[Track], prompt_size: tuple[int, int], video_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    prompt = track.prompts[source_idx]
    positives = prompt.get("positive_points", [])
    negatives = prompt.get("negative_points", [])
    if not positives:
        raise RuntimeError(f"prompt frame {prompt['frame_idx']} has no positive points")
    pos = scaled_points(positives, prompt_size, video_size)
    own_neg = scaled_points(negatives, prompt_size, video_size)
    competing_pos = []
    for other in tracks:
        if other.track_id == track.track_id:
            continue
        other_prompt = other.prompts.get(source_idx)
        if other_prompt and other_prompt.get("target_visible") and other_prompt.get("positive_points"):
            competing_pos.append(scaled_points(other_prompt["positive_points"], prompt_size, video_size))
    extra_neg = np.vstack(competing_pos).astype(np.float32) if competing_pos else np.zeros((0, 2), dtype=np.float32)
    neg = np.vstack([own_neg, extra_neg]).astype(np.float32) if len(own_neg) or len(extra_neg) else np.zeros((0, 2), dtype=np.float32)
    points = np.vstack([pos, neg]).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)])
    return points, labels


def mask_box(mask: np.ndarray) -> tuple[list[float] | None, float, np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, 0.0, None
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    center = np.asarray([float(xs.mean()), float(ys.mean())], dtype=float)
    return box, float(xs.size), center


def point_hits(mask: np.ndarray, points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return int(mask[y, x].sum())


def prompt_contract(mask: np.ndarray, points: np.ndarray, labels: np.ndarray) -> dict:
    pos = points[labels == 1]
    neg = points[labels == 0]
    return {
        "positive_hits": point_hits(mask, pos),
        "positive_points": int(len(pos)),
        "negative_hits": point_hits(mask, neg),
        "negative_points": int(len(neg)),
        "satisfies_prompt_contract": bool(point_hits(mask, pos) == len(pos) and point_hits(mask, neg) == 0),
    }


def add_prompt_frames(
    predictor,
    state,
    tracks: list[Track],
    frames: list[dict],
    prompt_frames: list[int],
    prompt_size: tuple[int, int],
    video_size: tuple[int, int],
) -> list[dict]:
    selected = [int(frame["frame_idx"]) for frame in frames]
    local_by_source = {source_idx: local for local, source_idx in enumerate(selected)}
    reports = []
    for source_idx in prompt_frames:
        if source_idx not in local_by_source:
            continue
        for track in tracks:
            prompt = track.prompts.get(source_idx)
            if not prompt or not prompt.get("target_visible") or not prompt.get("positive_points"):
                continue
            points, labels = prompt_points(track, source_idx, tracks, prompt_size, video_size)
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_by_source[source_idx],
                obj_id=track.obj_id,
                points=points,
                labels=labels,
            )
            ids = [int(v) for v in out_obj_ids]
            report = {
                "frame_idx": int(source_idx),
                "track_id": track.track_id,
                "obj_id": int(track.obj_id),
                "sam2_out_frame_idx": int(out_frame_idx),
                "object_ids_after_prompt": ids,
            }
            if track.obj_id in ids:
                obj_pos = ids.index(track.obj_id)
                mask = (out_mask_logits[obj_pos, 0].detach().cpu().numpy() > 0.0)
                report.update(prompt_contract(mask, points, labels))
                report["area_px"] = int(mask.sum())
            reports.append(report)
            print(
                json.dumps(
                    {
                        "event": "prompt_added",
                        "frame_idx": int(source_idx),
                        "track_id": track.track_id,
                        "obj_id": int(track.obj_id),
                        "area_px": int(report.get("area_px", 0)),
                        "prompt_contract": bool(report.get("satisfies_prompt_contract", False)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return reports


def write_track_results(
    args: argparse.Namespace,
    tracks: list[Track],
    frames: list[dict],
    propagated: dict[int, dict[int, np.ndarray]],
    prompt_reports: list[dict],
    prompt_frames: list[int],
    video_size: tuple[int, int],
    source_size: tuple[int, int],
) -> dict[str, dict[int, dict]]:
    sx = source_size[0] / float(video_size[0])
    sy = source_size[1] / float(video_size[1])
    all_results: dict[str, dict[int, dict]] = {}
    for track in tracks:
        out_dir = args.output_root / track.track_id / "sam2"
        mask_dir = out_dir / "sam2_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        results: dict[int, dict] = {}
        for frame in frames:
            source_idx = int(frame["frame_idx"])
            mask = propagated.get(source_idx, {}).get(track.obj_id)
            if mask is None or int(mask.sum()) == 0:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            box_small, area_small, center_small = mask_box(mask)
            if box_small is None or center_small is None:
                results[source_idx] = {"visible": False, "area_px": 0.0}
                continue
            mask_path = mask_dir / f"{source_idx:06d}.png"
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise RuntimeError(f"failed to write {mask_path}")
            results[source_idx] = {
                "visible": True,
                "bbox_xyxy": [float(box_small[0] * sx), float(box_small[1] * sy), float(box_small[2] * sx), float(box_small[3] * sy)],
                "center_xy": [float(center_small[0] * sx), float(center_small[1] * sy)],
                "area_px": float(area_small * sx * sy),
                "mask_path": str(mask_path),
            }
        visible = sum(1 for row in results.values() if row.get("visible"))
        track_reports = [row for row in prompt_reports if row["track_id"] == track.track_id]
        qc = {
            "status": "ok",
            "backend": "SAM2 multi-object propagation from VLM point prompts",
            "clip": str(args.clip),
            "point_prompts": str(track.prompt_path),
            "track_id": track.track_id,
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "frames": len(frames),
            "prompt_frames": prompt_frames,
            "visible_frames": visible,
            "checkpoint": str(args.checkpoint),
            "model_cfg": args.model_cfg,
            "non_overlap_masks": True,
            "prompt_contract_reports": track_reports,
            "outputs": {
                "sam2_track": str(out_dir / "sam2_track.json"),
                "sam2_masks": str(mask_dir),
            },
        }
        (out_dir / "sam2_track.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "qc_sam2_vlm_points_track.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        all_results[track.track_id] = results
    return all_results


def render_combined(args: argparse.Namespace, tracks: list[Track], frames: list[dict], all_results: dict[str, dict[int, dict]]) -> Path:
    cap, info = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    writer_path = args.output_root / "sam2_multiobject_overlay.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (args.render_width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer {writer_path}")
    try:
        for frame in frames:
            source_idx = int(frame["frame_idx"])
            image = read_video_frame(cap, source_idx)
            image = cv2.resize(image, (args.render_width, height), interpolation=cv2.INTER_AREA)
            for i, track in enumerate(tracks):
                result = all_results[track.track_id].get(source_idx, {})
                if not result.get("visible") or not result.get("mask_path"):
                    continue
                mask = cv2.imread(str(result["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"failed to read {result['mask_path']}")
                mask = cv2.resize(mask, (args.render_width, height), interpolation=cv2.INTER_NEAREST) > 0
                tint = np.zeros_like(image)
                tint[:, :] = np.asarray(COLORS_BGR[i % len(COLORS_BGR)], dtype=np.uint8)
                image[mask] = cv2.addWeighted(image, 0.58, tint, 0.42, 0.0)[mask]
            y = 18
            for i, track in enumerate(tracks):
                color = COLORS_BGR[i % len(COLORS_BGR)]
                cv2.rectangle(image, (8, y - 10), (22, y + 4), color, -1)
                cv2.putText(image, track.track_id[:38], (30, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18
            put_caption(image, "SAM2 multi-surface non-overlap", source_idx)
            writer.write(image)
    finally:
        writer.release()
        cap.release()
    return writer_path


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(args.point_root)
    frames = selected_frames(int(args.frame_start), int(args.frame_end))
    cap, info = open_video(args.clip)
    cap.release()
    video_height = int(round(info.height * int(args.sam2_image_width) / info.width))
    video_size = (int(args.sam2_image_width), video_height)
    prompt_size = (int(tracks[0].payload["prompt_image_width"]), int(round(int(tracks[0].payload["prompt_image_width"]) * info.height / info.width)))
    for track in tracks:
        if int(track.payload["prompt_image_width"]) != prompt_size[0]:
            raise RuntimeError("all prompt files must use the same prompt image width")
    frame_dir = extract_frames(args.clip, frames, args.output_root, int(args.sam2_image_width))
    selected = {int(frame["frame_idx"]) for frame in frames}
    prompt_frames = sorted(
        {
            frame_idx
            for track in tracks
            for frame_idx, prompt in track.prompts.items()
            if frame_idx in selected and prompt.get("target_visible") and prompt.get("positive_points")
        }
    )
    if not prompt_frames:
        raise RuntimeError("no visible prompt frames inside selected range")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM2 video predictor requires CUDA for this pipeline")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=device, vos_optimized=False)
    predictor.non_overlap_masks = True
    predictor.add_all_frames_to_correct_as_cond = True
    propagated: dict[int, dict[int, np.ndarray]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        prompt_reports = add_prompt_frames(predictor, state, tracks, frames, prompt_frames, prompt_size, video_size)
        selected_frames_list = [int(frame["frame_idx"]) for frame in frames]
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            source_idx = selected_frames_list[int(out_frame_idx)]
            ids = [int(v) for v in out_obj_ids]
            per_frame: dict[int, np.ndarray] = {}
            for obj_pos, obj_id in enumerate(ids):
                per_frame[obj_id] = (out_mask_logits[obj_pos, 0].detach().cpu().numpy() > 0.0).astype(np.uint8)
            propagated[source_idx] = per_frame
            if int(out_frame_idx) % 10 == 0 or int(out_frame_idx) == len(selected_frames_list) - 1:
                print(json.dumps({"event": "propagated", "local_frame": int(out_frame_idx), "source_frame": int(source_idx), "object_ids": ids}, sort_keys=True), flush=True)
    all_results = write_track_results(args, tracks, frames, propagated, prompt_reports, prompt_frames, video_size, (info.width, info.height))
    overlay = render_combined(args, tracks, frames, all_results)
    summary = {
        "status": "ok",
        "backend": "SAM2 multi-object propagation from VLM point prompts",
        "clip": str(args.clip),
        "point_root": str(args.point_root),
        "output_root": str(args.output_root),
        "track_ids": [track.track_id for track in tracks],
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": len(frames),
        "prompt_frames": prompt_frames,
        "visible_frames_by_track": {
            track.track_id: int(sum(1 for row in all_results[track.track_id].values() if row.get("visible")))
            for track in tracks
        },
        "prompt_contract_satisfied": int(sum(1 for row in prompt_reports if row.get("satisfies_prompt_contract"))),
        "prompt_contract_reports": len(prompt_reports),
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "non_overlap_masks": True,
        "overlay": str(overlay),
        "elapsed_s": time.time() - started,
    }
    (args.output_root / "qc_sam2_multiobject_points.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "track_ids"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--point-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--sam2-image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
