#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build reusable visible ownership factors for MANO/object-surface constraints.

This is not a tomato/trash-specific diagnostic.  It creates a generic factor
record that asks, for a hand side and manipulated surface entity in a frame,
whether first-surface pixels are visibly hand-owned, visibly object/part-owned,
a mixed boundary, or unresolved.  The intended downstream use is to quarantine
hard object nonpenetration/depth-order constraints on pixels that the visible
hand or a mixed boundary plausibly owns.

Inputs are model-produced or model-derived observations already present in V18:
annotation hand boxes, SAM2 hand masks prompted from those boxes, object/part
visible masks, metric depth, MANO projection support, and raw RGB frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_owlv2_sam2_part_tracks import import_sam2  # noqa: E402
from build_v18_part_visible_surfaces import load_metric_depth, resize_bool_mask  # noqa: E402
from build_v18_temporal_mano_articulated_interval_state import (  # noqa: E402
    bridge_vertices_and_joints,
    project_world,
    world_to_camera,
)
from build_v18_temporal_mano_translation_interval_state import as_list, load_json, numeric_summary, write_json  # noqa: E402

DEFAULT_SAM2_CHECKPOINT = Path("/data2/ego_annotation_outputs/checkpoints/sam2.1_hiera_small.pt")
DEFAULT_SAM2_REPO = Path("third_party/sam2")
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_visible_ownership_factor_v1")


def parse_spans(values: list[list[int]] | None) -> list[tuple[int, int]]:
    if not values:
        raise RuntimeError("at least one --frame-span START END is required")
    spans: list[tuple[int, int]] = []
    for pair in values:
        if len(pair) != 2:
            raise RuntimeError(f"invalid frame span: {pair}")
        a, b = int(pair[0]), int(pair[1])
        if b < a:
            raise RuntimeError(f"invalid decreasing frame span: {a}:{b}")
        spans.append((a, b))
    return spans


def frame_set_from_spans(spans: list[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for a, b in spans:
        out.update(range(a, b + 1))
    return out


def in_spans(frame_idx: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= frame_idx <= b for a, b in spans)


def load_frames(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(frame["frame_idx"]): frame
        for frame in as_list(annotations.get("frames"))
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }


def raw_video_dir(annotations: dict[str, Any]) -> Path:
    frames = as_list(annotations.get("frames"))
    if not frames:
        raise RuntimeError("annotations contain no frames")
    first = frames[0]
    if not isinstance(first, dict) or not isinstance(first.get("raw_frame_path"), str):
        raise RuntimeError("first annotation frame lacks raw_frame_path")
    return Path(first["raw_frame_path"]).parent


def raw_size(frames: dict[int, dict[str, Any]], frame_idx: int) -> tuple[int, int]:
    frame = frames[frame_idx]
    path = Path(str(frame["raw_frame_path"]))
    image = Image.open(path)
    return image.size


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray | None:
    arr = box.astype(np.float32).copy()
    arr[[0, 2]] = np.clip(arr[[0, 2]], 0.0, float(width - 1))
    arr[[1, 3]] = np.clip(arr[[1, 3]], 0.0, float(height - 1))
    if not np.isfinite(arr).all() or arr[2] <= arr[0] or arr[3] <= arr[1]:
        return None
    return arr


def scale_hand_box(
    box: list[float],
    *,
    image_size: tuple[int, int],
    detector_size: tuple[int, int],
    margin_ratio: float,
) -> list[float] | None:
    arr = np.asarray(box, dtype=np.float32)
    if arr.shape != (4,) or not np.isfinite(arr).all() or arr[2] <= arr[0] or arr[3] <= arr[1]:
        return None
    # Detector boxes in these annotations are usually in the detector source
    # resolution and may extend slightly outside it.  If the box already fits the
    # raw image, leave it in raw coordinates; otherwise scale from detector size.
    image_w, image_h = image_size
    if arr[2] <= image_w * 1.05 and arr[3] <= image_h * 1.05:
        source_w, source_h = image_w, image_h
    else:
        source_w, source_h = detector_size
    w = float(arr[2] - arr[0])
    h = float(arr[3] - arr[1])
    margin = float(margin_ratio) * max(w, h)
    arr += np.asarray([-margin, -margin, margin, margin], dtype=np.float32)
    scale = np.asarray([image_w / float(source_w), image_h / float(source_h), image_w / float(source_w), image_h / float(source_h)], dtype=np.float32)
    return None if (clipped := clip_box(arr * scale, image_w, image_h)) is None else [float(v) for v in clipped.tolist()]


def hand_by_side(frame: dict[str, Any], side: str) -> dict[str, Any] | None:
    for hand in as_list(frame.get("hands")):
        if isinstance(hand, dict) and str(hand.get("hand_side")) == side:
            return hand
    return None


def collect_hand_prompts(args: argparse.Namespace, annotations: dict[str, Any], frames: dict[int, dict[str, Any]], spans: list[tuple[int, int]]) -> dict[str, list[dict[str, Any]]]:
    frame_ids = sorted(frame_set_from_spans(spans))
    if not frame_ids:
        return {side: [] for side in args.sides}
    image_size = raw_size(frames, frame_ids[0])
    detector_size = (int(args.hand_box_source_width), int(args.hand_box_source_height))
    prompts: dict[str, list[dict[str, Any]]] = {side: [] for side in args.sides}
    stride = max(1, int(args.hand_prompt_stride))
    for side in args.sides:
        side_frames = []
        for frame_idx in frame_ids:
            frame = frames.get(frame_idx)
            if frame is None:
                continue
            hand = hand_by_side(frame, side)
            if hand is None or not isinstance(hand.get("bbox_xyxy"), list):
                continue
            box = scale_hand_box(
                hand["bbox_xyxy"],
                image_size=image_size,
                detector_size=detector_size,
                margin_ratio=float(args.hand_box_margin_ratio),
            )
            if box is None:
                continue
            side_frames.append((frame_idx, box, hand))
        if not side_frames:
            continue
        selected: list[tuple[int, list[float], dict[str, Any]]] = []
        seen: set[int] = set()
        # Periodic prompts plus span endpoints keep the video track anchored
        # without prompting every frame.
        for frame_idx, box, hand in side_frames:
            if frame_idx % stride == 0:
                selected.append((frame_idx, box, hand))
                seen.add(frame_idx)
        for start, end in spans:
            candidates = [row for row in side_frames if start <= row[0] <= end]
            if not candidates:
                continue
            for row in (candidates[0], candidates[-1]):
                if row[0] not in seen:
                    selected.append(row)
                    seen.add(row[0])
        selected.sort(key=lambda row: row[0])
        prompts[side] = [
            {
                "frame_idx": int(frame_idx),
                "hand_side": side,
                "bbox_xyxy": [float(v) for v in box],
                "source_bbox_xyxy": hand.get("bbox_xyxy"),
                "visibility_state": hand.get("visibility_state"),
                "confidence": hand.get("confidence"),
            }
            for frame_idx, box, hand in selected
        ]
    return prompts


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError(f"failed to write mask: {path}")


def load_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L")) > 0
    if shape_hw is not None:
        arr = resize_bool_mask(arr, shape_hw)
    return arr


def run_sam2_hand_masks(args: argparse.Namespace, annotations: dict[str, Any], prompts: dict[str, list[dict[str, Any]]], spans: list[tuple[int, int]], output_case: Path) -> dict[tuple[int, str], Path]:
    if args.reuse_hand_mask_root is not None:
        out: dict[tuple[int, str], Path] = {}
        for side in args.sides:
            for path in sorted((args.reuse_hand_mask_root / side).glob("*.png")):
                try:
                    frame_idx = int(path.stem)
                except ValueError:
                    continue
                if in_spans(frame_idx, spans):
                    out[(frame_idx, side)] = path
        if out:
            return out
        raise RuntimeError(f"--reuse-hand-mask-root has no usable masks in spans: {args.reuse_hand_mask_root}")

    if not any(prompts.values()):
        raise RuntimeError("no hand prompts could be built from annotation hand boxes")
    predictor = import_sam2(args)
    state = predictor.init_state(
        str(raw_video_dir(annotations)),
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        offload_state_to_cpu=bool(args.offload_state_to_cpu),
    )
    obj_id_by_side = {side: i + 1 for i, side in enumerate(args.sides)}
    for side, rows in prompts.items():
        obj_id = obj_id_by_side[side]
        for row in rows:
            predictor.add_new_points_or_box(
                state,
                frame_idx=int(row["frame_idx"]),
                obj_id=obj_id,
                box=np.asarray(row["bbox_xyxy"], dtype=np.float32),
            )
    target_frames = frame_set_from_spans(spans)
    masks: dict[tuple[int, str], np.ndarray] = {}
    side_by_obj_id = {obj_id: side for side, obj_id in obj_id_by_side.items()}
    with torch.inference_mode(), torch.autocast(str(args.device), dtype=torch.bfloat16, enabled=str(args.device) == "cuda"):
        for reverse in (False, True):
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=None, reverse=reverse):
                if int(frame_idx) not in target_frames or mask_logits is None:
                    continue
                for i, obj_id in enumerate(obj_ids):
                    side = side_by_obj_id.get(int(obj_id))
                    if side is None:
                        continue
                    mask = (mask_logits[i].detach().cpu().numpy() > 0.0)
                    if mask.ndim == 3:
                        mask = mask[0]
                    if int(mask.sum()) < int(args.min_hand_mask_area_px):
                        continue
                    masks[(int(frame_idx), side)] = mask.astype(bool)
    out_paths: dict[tuple[int, str], Path] = {}
    for (frame_idx, side), mask in sorted(masks.items()):
        path = output_case / "visible_hand_masks" / side / f"{frame_idx:06d}.png"
        save_mask(path, mask)
        out_paths[(frame_idx, side)] = path
    return out_paths


def annotation_entity_mask_paths(frames: dict[int, dict[str, Any]], target_entity_id: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for frame_idx, frame in frames.items():
        for obj in as_list(frame.get("objects")):
            if not isinstance(obj, dict) or str(obj.get("object_id")) != target_entity_id:
                continue
            mask_path = obj.get("mask_path")
            if obj.get("renderable_mask") is True and isinstance(mask_path, str) and Path(mask_path).exists():
                out[int(frame_idx)] = Path(mask_path)
    return out


def report_entity_mask_paths(path: Path | None) -> dict[int, Path]:
    if path is None:
        return {}
    payload = load_json(path)
    rows = []
    for key in ("saved_mask_rows_after_start", "target_mask_rows", "mask_rows", "track_rows"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(value)
    out: dict[int, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        raw = row.get("saved_mask_path") or row.get("mask_path")
        if not isinstance(raw, str):
            continue
        p = Path(raw)
        if p.exists():
            out[int(row["frame_idx"])] = p
    return out


def depth_row_for(depth: dict[str, Any], frame_idx: int) -> dict[str, Any] | None:
    idx = depth["frame_to_i"].get(int(frame_idx))
    if idx is None:
        return None
    return {
        "depth": np.asarray(depth["depth"][idx], dtype=np.float32),
        "intrinsics": np.asarray(depth["intrinsics"][idx], dtype=np.float64),
    }


def rasterize_mano_support(
    *,
    frame: dict[str, Any],
    side: str,
    hand: dict[str, Any],
    depth_row: dict[str, Any] | None,
    shape_hw: tuple[int, int],
    bridge_cache: dict[Path, Any],
    radius_px: int,
    depth_support_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    support_u8 = np.zeros(shape_hw, dtype=np.uint8)
    arrays = bridge_vertices_and_joints(hand, bridge_cache)
    if arrays is None:
        return support_u8.astype(bool), {"state": "missing_mano_bridge", "projected_vertex_count": 0, "depth_supported_vertex_count": 0}
    vertices, _joints = arrays
    uv = project_world(vertices, frame, side)
    if uv is None:
        return support_u8.astype(bool), {"state": "missing_projection", "projected_vertex_count": 0, "depth_supported_vertex_count": 0}
    cam = world_to_camera(vertices, frame)
    h, w = shape_hw
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    valid = (cam[:, 2] > 1.0e-5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    projected_count = int(np.count_nonzero(valid))
    depth_residual = np.full((len(vertices),), np.nan, dtype=float)
    if depth_row is not None:
        depth = np.asarray(depth_row["depth"], dtype=np.float32)
        if depth.shape != shape_hw:
            depth_img = Image.fromarray(depth)
            depth = np.asarray(depth_img.resize((w, h), Image.Resampling.NEAREST), dtype=np.float32)
        valid_depth = np.zeros((len(vertices),), dtype=bool)
        if np.any(valid):
            z = depth[v[valid], u[valid]].astype(float)
            finite = np.isfinite(z) & (z > 1.0e-5)
            ids = np.where(valid)[0]
            valid_depth[ids[finite]] = True
            depth_residual[ids[finite]] = cam[ids[finite], 2] - z[finite]
        valid = valid_depth & (np.abs(depth_residual) <= float(depth_support_m))
    supported_count = int(np.count_nonzero(valid))
    radius = max(0, int(radius_px))
    for x, y in zip(u[valid], v[valid]):
        cv2.circle(support_u8, (int(x), int(y)), radius, 1, thickness=-1)
    return support_u8.astype(bool), {
        "state": "ok",
        "projected_vertex_count": projected_count,
        "depth_supported_vertex_count": supported_count,
        "depth_residual_m": numeric_summary(depth_residual[np.isfinite(depth_residual)]),
    }


def render_review_panel(
    *,
    frame_path: Path,
    hand_mask: np.ndarray,
    entity_mask: np.ndarray,
    visible_hand_owned: np.ndarray,
    visible_object_owned: np.ndarray,
    mixed_boundary: np.ndarray,
    occluded_or_unresolved: np.ndarray,
    output_path: Path,
    label: str,
) -> None:
    image = Image.open(frame_path).convert("RGB")
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    colors = [
        (hand_mask, np.asarray([0, 255, 255], dtype=np.float32), 0.30),
        (entity_mask, np.asarray([0, 255, 0], dtype=np.float32), 0.24),
        (visible_hand_owned, np.asarray([255, 0, 255], dtype=np.float32), 0.70),
        (visible_object_owned, np.asarray([0, 180, 0], dtype=np.float32), 0.45),
        (mixed_boundary, np.asarray([255, 255, 0], dtype=np.float32), 0.65),
        (occluded_or_unresolved, np.asarray([255, 80, 0], dtype=np.float32), 0.70),
    ]
    for mask, color, alpha in colors:
        if mask.shape != overlay.shape[:2]:
            mask = resize_bool_mask(mask, overlay.shape[:2])
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, out.width, 30], fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    spans = parse_spans(args.frame_span)
    annotations = load_json(args.annotations)
    frames = load_frames(annotations)
    target_frames = sorted(frame_set_from_spans(spans))
    missing_frames = [idx for idx in target_frames if idx not in frames]
    if missing_frames:
        raise RuntimeError(f"annotations missing target frames: {missing_frames[:8]}")
    output_case = args.output_root / args.case
    output_case.mkdir(parents=True, exist_ok=True)
    prompts = collect_hand_prompts(args, annotations, frames, spans)
    hand_mask_paths = run_sam2_hand_masks(args, annotations, prompts, spans, output_case)
    annotation_masks = annotation_entity_mask_paths(frames, args.target_entity_id)
    report_masks = report_entity_mask_paths(args.visible_entity_mask_report)
    depth = load_metric_depth(args.depth_npz)
    mask_cache: dict[Path, np.ndarray] = {}
    bridge_cache: dict[Path, Any] = {}
    ownership_rows: list[dict[str, Any]] = []
    review_paths: list[str] = []
    state_counts = Counter()
    review_limit = max(0, int(args.max_review_frames))
    for frame_idx in target_frames:
        frame = frames[frame_idx]
        entity_path = report_masks.get(frame_idx) or annotation_masks.get(frame_idx)
        if entity_path is None:
            state_counts["missing_entity_mask"] += len(args.sides)
            continue
        if entity_path not in mask_cache:
            mask_cache[entity_path] = load_mask(entity_path)
        entity_mask = mask_cache[entity_path]
        frame_image_size = Image.open(frame["raw_frame_path"]).size
        shape_hw = (frame_image_size[1], frame_image_size[0])
        entity_mask = resize_bool_mask(entity_mask, shape_hw)
        drow = depth_row_for(depth, frame_idx)
        for side in args.sides:
            hand_path = hand_mask_paths.get((frame_idx, side))
            hand = hand_by_side(frame, side)
            if hand_path is None or hand is None:
                state_counts["missing_hand_mask_or_hand"] += 1
                continue
            if hand_path not in mask_cache:
                mask_cache[hand_path] = load_mask(hand_path)
            hand_mask = resize_bool_mask(mask_cache[hand_path], shape_hw)
            mano_support, mano_diag = rasterize_mano_support(
                frame=frame,
                side=side,
                hand=hand,
                depth_row=drow,
                shape_hw=shape_hw,
                bridge_cache=bridge_cache,
                radius_px=int(args.mano_projection_dilation_px),
                depth_support_m=float(args.mano_depth_support_m),
            )
            overlap = hand_mask & entity_mask
            visible_hand_owned = overlap & mano_support
            mixed_boundary = overlap & ~mano_support
            visible_object_owned = entity_mask & ~overlap
            classified = visible_hand_owned | mixed_boundary | visible_object_owned
            occluded_or_unresolved = entity_mask & ~classified
            non_object_owned = visible_hand_owned | mixed_boundary
            adjusted_entity = visible_object_owned
            mask_dir = output_case / "ownership_masks" / side
            paths = {
                "visible_hand_owned_mask_path": mask_dir / f"{frame_idx:06d}_visible_hand_owned.png",
                "visible_object_owned_mask_path": mask_dir / f"{frame_idx:06d}_visible_object_owned.png",
                "mixed_boundary_mask_path": mask_dir / f"{frame_idx:06d}_mixed_boundary.png",
                "occluded_or_unresolved_mask_path": mask_dir / f"{frame_idx:06d}_occluded_or_unresolved.png",
                "non_object_owned_mask_path": mask_dir / f"{frame_idx:06d}_non_object_owned.png",
                "adjusted_entity_mask_path": mask_dir / f"{frame_idx:06d}_adjusted_entity_object_owned.png",
            }
            save_mask(paths["visible_hand_owned_mask_path"], visible_hand_owned)
            save_mask(paths["visible_object_owned_mask_path"], visible_object_owned)
            save_mask(paths["mixed_boundary_mask_path"], mixed_boundary)
            save_mask(paths["occluded_or_unresolved_mask_path"], occluded_or_unresolved)
            save_mask(paths["non_object_owned_mask_path"], non_object_owned)
            save_mask(paths["adjusted_entity_mask_path"], adjusted_entity)
            review_path = None
            if len(review_paths) < review_limit and (np.any(overlap) or frame_idx in {a for a, _ in spans} or frame_idx in {b for _, b in spans}):
                review_path = output_case / "review_frames" / f"{frame_idx:06d}_{side}_ownership.jpg"
                render_review_panel(
                    frame_path=Path(frame["raw_frame_path"]),
                    hand_mask=hand_mask,
                    entity_mask=entity_mask,
                    visible_hand_owned=visible_hand_owned,
                    visible_object_owned=visible_object_owned,
                    mixed_boundary=mixed_boundary,
                    occluded_or_unresolved=occluded_or_unresolved,
                    output_path=review_path,
                    label=f"{args.case} f{frame_idx} {side}: cyan hand, green object, magenta hand-owned, yellow mixed, orange unresolved",
                )
                review_paths.append(str(review_path))
            row = {
                "factor_family": "visible_ownership",
                "frame_idx": int(frame_idx),
                "hand_side": side,
                "target_entity_id": args.target_entity_id,
                "variable_affected": "constraint_eligibility",
                "observation_type": "sam2_visible_hand_mask_x_visible_entity_mask_x_metric_depth_mano_projection",
                "residual_or_quarantine_rule": "hard object/depth-order constraints must not use pixels marked non_object_owned_mask_path; visible_object_owned_mask_path remains eligible first surface; occluded_or_unresolved_mask_path is not eligible for hard object support",
                "rendered_uncertainty_channel": "review frames use magenta for visible_hand_owned, yellow for mixed_boundary/non-object quarantine, green for visible_object_owned, and orange for occluded_or_unresolved",
                "provenance": {
                    "annotations": str(args.annotations),
                    "raw_frame_path": str(frame["raw_frame_path"]),
                    "hand_mask_path": str(hand_path),
                    "entity_mask_path": str(entity_path),
                    "entity_mask_source": "visible_entity_mask_report" if frame_idx in report_masks else "annotation_object_mask",
                    "depth_npz": str(args.depth_npz),
                },
                **{key: str(path) for key, path in paths.items()},
                "counts": {
                    "hand_mask_px": int(hand_mask.sum()),
                    "entity_mask_px": int(entity_mask.sum()),
                    "hand_entity_overlap_px": int(overlap.sum()),
                    "visible_hand_owned_px": int(visible_hand_owned.sum()),
                    "mixed_boundary_px": int(mixed_boundary.sum()),
                    "visible_object_owned_px": int(visible_object_owned.sum()),
                    "occluded_or_unresolved_px": int(occluded_or_unresolved.sum()),
                    "non_object_owned_px": int(non_object_owned.sum()),
                },
                "fractions": {
                    "entity_non_object_owned_fraction": float(non_object_owned.sum() / max(1, int(entity_mask.sum()))),
                    "entity_visible_hand_owned_fraction": float(visible_hand_owned.sum() / max(1, int(entity_mask.sum()))),
                    "entity_mixed_boundary_fraction": float(mixed_boundary.sum() / max(1, int(entity_mask.sum()))),
                    "hand_overlap_fraction": float(overlap.sum() / max(1, int(hand_mask.sum()))),
                },
                "mano_projection_support": mano_diag,
                "review_frame_path": None if review_path is None else str(review_path),
            }
            ownership_rows.append(row)
            state_counts["ownership_rows"] += 1
    report = {
        "method": "build_v18_visible_ownership_factor",
        "status": "ok",
        "claim_scope": (
            "Reusable visible ownership factor records. These masks are not final annotation claims by themselves; "
            "they are solver inputs for quarantining object/part constraints on first-surface pixels visibly owned by the hand or mixed boundary."
        ),
        "case": args.case,
        "target_entity_id": args.target_entity_id,
        "inputs": {
            "annotations": str(args.annotations),
            "depth_npz": str(args.depth_npz),
            "visible_entity_mask_report": None if args.visible_entity_mask_report is None else str(args.visible_entity_mask_report),
            "reuse_hand_mask_root": None if args.reuse_hand_mask_root is None else str(args.reuse_hand_mask_root),
        },
        "parameters": {
            "frame_spans": [[int(a), int(b)] for a, b in spans],
            "sides": list(args.sides),
            "hand_prompt_stride": int(args.hand_prompt_stride),
            "hand_box_source_size": [int(args.hand_box_source_width), int(args.hand_box_source_height)],
            "hand_box_margin_ratio": float(args.hand_box_margin_ratio),
            "mano_projection_dilation_px": int(args.mano_projection_dilation_px),
            "mano_depth_support_m": float(args.mano_depth_support_m),
        },
        "prompt_rows": [row for rows in prompts.values() for row in rows],
        "summary": {
            "frame_count_requested": int(len(target_frames)),
            "hand_mask_count": int(len(hand_mask_paths)),
            "ownership_row_count": int(len(ownership_rows)),
            "state_counts": dict(state_counts),
            "counts": {
                key: numeric_summary(np.asarray([row["counts"][key] for row in ownership_rows], dtype=float))
                for key in [
                    "hand_entity_overlap_px",
                    "visible_hand_owned_px",
                    "mixed_boundary_px",
                    "visible_object_owned_px",
                    "occluded_or_unresolved_px",
                    "non_object_owned_px",
                ]
            },
            "fractions": {
                key: numeric_summary(np.asarray([row["fractions"][key] for row in ownership_rows], dtype=float))
                for key in [
                    "entity_non_object_owned_fraction",
                    "entity_visible_hand_owned_fraction",
                    "entity_mixed_boundary_fraction",
                    "hand_overlap_fraction",
                ]
            },
        },
        "ownership_rows": ownership_rows,
        "review_frames": review_paths,
    }
    report_path = output_case / "v18_visible_ownership_factor_report.json"
    write_json(report_path, report)
    print(json.dumps({"status": "ok", "report": str(report_path), "ownership_rows": len(ownership_rows)}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--target-entity-id", required=True)
    p.add_argument("--depth-npz", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--frame-span", nargs=2, type=int, action="append", metavar=("START", "END"), required=True)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--visible-entity-mask-report", type=Path, default=None, help="Optional report containing saved visible object/part mask paths. If absent, annotation object masks for --target-entity-id are used.")
    p.add_argument("--reuse-hand-mask-root", type=Path, default=None, help="Optional root with side/frame hand masks to reuse instead of running SAM2.")
    p.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    p.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_CFG)
    p.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vos-optimized", action="store_true")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--offload-state-to-cpu", action="store_true")
    p.add_argument("--hand-prompt-stride", type=int, default=12)
    p.add_argument("--hand-box-source-width", type=int, default=1280)
    p.add_argument("--hand-box-source-height", type=int, default=720)
    p.add_argument("--hand-box-margin-ratio", type=float, default=0.04)
    p.add_argument("--min-hand-mask-area-px", type=int, default=80)
    p.add_argument("--mano-projection-dilation-px", type=int, default=5)
    p.add_argument("--mano-depth-support-m", type=float, default=0.040)
    p.add_argument("--max-review-frames", type=int, default=24)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
