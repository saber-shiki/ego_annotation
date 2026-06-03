#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from fuse_v1_full_fidelity import DEFAULT_CLIP, open_video, put_caption, read_video_frame
from run_sam2_object_track import extract_frames, mask_box


def import_samwise(repo: Path):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import opts  # noqa: PLC0415
    from datasets.transform_utils import VideoEvalDataset  # noqa: PLC0415
    from models.samwise import build_samwise  # noqa: PLC0415
    from util.misc import on_load_checkpoint  # noqa: PLC0415

    return opts, VideoEvalDataset, build_samwise, on_load_checkpoint


def build_args(args: argparse.Namespace, opts) -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[opts.get_args_parser()])
    model_args = parser.parse_args([])
    model_args.device = args.device
    model_args.resume = str(args.checkpoint)
    model_args.HSA = bool(args.hsa)
    model_args.use_cme_head = bool(args.use_cme_head)
    model_args.eval_clip_window = int(args.eval_clip_window)
    model_args.num_workers = int(args.num_workers)
    model_args.threshold = float(args.threshold)
    model_args.sam2_version = args.sam2_version
    return model_args


def load_model(args: argparse.Namespace):
    opts, VideoEvalDataset, build_samwise, on_load_checkpoint = import_samwise(args.samwise_repo)
    model_args = build_args(args, opts)
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_samwise(model_args)
    model.to(torch.device(model_args.device))
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"SAMWISE checkpoint has no model key: {args.checkpoint}")
    if list(checkpoint["model"].keys())[0].startswith("module"):
        checkpoint["model"] = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    checkpoint = on_load_checkpoint(model, checkpoint)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, model_args, VideoEvalDataset


def compute_masks(model, model_args: argparse.Namespace, VideoEvalDataset, text_prompt: str, frame_dir: Path) -> np.ndarray:
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"no extracted frames in {frame_dir}")
    ext = frame_paths[0].suffix
    frames_list = [path.stem for path in frame_paths]
    vd = VideoEvalDataset(str(frame_dir), frames_list, ext=ext)
    dl = DataLoader(vd, batch_size=int(model_args.eval_clip_window), num_workers=int(model_args.num_workers), shuffle=False)
    origin_w, origin_h = vd.origin_w, vd.origin_h
    all_pred_masks = []
    for imgs, clip_frames_ids in dl:
        clip_frames_ids = clip_frames_ids.tolist()
        imgs = imgs.to(model_args.device)
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(model_args.device)
        target = {"size": size, "frame_ids": clip_frames_ids}
        with torch.no_grad():
            outputs = model([imgs], [text_prompt], [target])
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks.unsqueeze(0)
        pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode="bilinear", align_corners=False)
        pred_masks = (pred_masks.sigmoid() > float(model_args.threshold))[0].detach().cpu()
        all_pred_masks.append(pred_masks)
    return torch.cat(all_pred_masks, dim=0).numpy()


def render(args: argparse.Namespace, frames: list[dict], results: dict[int, dict]) -> Path:
    cap, info = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    writer_path = args.output_dir / "samwise_referring_overlay.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (args.render_width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer {writer_path}")
    review_indices = set(int(part) for raw in args.review_frames for part in raw.split(",") if part.strip())
    still_dir = args.output_dir / "review_stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        source_idx = int(frame["frame_idx"])
        image = read_video_frame(cap, source_idx)
        image = cv2.resize(image, (args.render_width, height), interpolation=cv2.INTER_AREA)
        result = results.get(source_idx, {})
        if result.get("visible") and result.get("mask_path"):
            mask = cv2.imread(str(result["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"failed to read {result['mask_path']}")
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (args.render_width, height), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0
            tint = np.zeros_like(image)
            tint[:, :, 0] = 255
            tint[:, :, 2] = 255
            image[mask_bool] = cv2.addWeighted(image, 0.55, tint, 0.45, 0.0)[mask_bool]
        put_caption(image, f"SAMWISE text track: {args.text_prompt}", source_idx)
        writer.write(image)
        if source_idx in review_indices:
            if not cv2.imwrite(str(still_dir / f"{source_idx:06d}.jpg"), image):
                raise RuntimeError(f"failed to write review still for frame {source_idx}")
    writer.release()
    cap.release()
    return writer_path


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("SAMWISE inference requires CUDA for this pipeline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [{"frame_idx": idx} for idx in range(int(args.frame_start), int(args.frame_end) + 1)]
    frame_dir = extract_frames(args.clip, frames, list(range(len(frames))), args.output_dir, int(args.image_width))
    model, model_args, VideoEvalDataset = load_model(args)
    masks = compute_masks(model, model_args, VideoEvalDataset, args.text_prompt, frame_dir)
    mask_dir = args.output_dir / "samwise_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict] = {}
    for local_idx, frame in enumerate(frames):
        source_idx = int(frame["frame_idx"])
        mask = np.asarray(masks[local_idx]).astype(np.uint8)
        box, area, center = mask_box(mask)
        if box is None:
            results[source_idx] = {"visible": False, "area_px": 0.0}
            continue
        mask_path = mask_dir / f"{source_idx:06d}.png"
        if not cv2.imwrite(str(mask_path), mask * 255):
            raise RuntimeError(f"failed to write {mask_path}")
        results[source_idx] = {
            "visible": True,
            "bbox_xyxy": [float(v) for v in box],
            "center_xy": center.astype(float).tolist(),
            "area_px": float(area),
            "mask_path": str(mask_path),
        }
    video = render(args, frames, results)
    areas = [float(row["area_px"]) for row in results.values() if row.get("visible")]
    qc = {
        "status": "raw_inference_completed_semantics_unverified",
        "annotation_ready": False,
        "mesh_ready": False,
        "semantic_acceptance_required": True,
        "backend": "SAMWISE text-driven video segmentation",
        "clip": str(args.clip),
        "text_prompt": args.text_prompt,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": len(frames),
        "visible_frames": sum(1 for row in results.values() if row.get("visible")),
        "visible_rate": sum(1 for row in results.values() if row.get("visible")) / max(1, len(frames)),
        "area_median": None if not areas else float(np.median(areas)),
        "area_p05": None if not areas else float(np.percentile(areas, 5)),
        "area_p95": None if not areas else float(np.percentile(areas, 95)),
        "threshold": float(args.threshold),
        "eval_clip_window": int(args.eval_clip_window),
        "samwise_repo": str(args.samwise_repo),
        "checkpoint": str(args.checkpoint),
        "elapsed_s": time.time() - started,
        "outputs": {
            "track": str(args.output_dir / "samwise_track.json"),
            "masks": str(mask_dir),
            "overlay": str(video),
            "review_stills": str(args.output_dir / "review_stills"),
        },
        "acceptance_note": (
            "Visibility, area, and overlay generation only prove model execution. "
            "Masks must pass visual or VLM semantic verification before mesh reconstruction."
        ),
    }
    (args.output_dir / "samwise_track.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output_dir / "qc_samwise_referring_masks.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samwise-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--review-frames", nargs="+", default=["678,720,797,858,880,900,918"])
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-version", choices=["tiny", "base", "large"], default="base")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-clip-window", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hsa", action="store_true", default=True)
    parser.add_argument("--use-cme-head", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
