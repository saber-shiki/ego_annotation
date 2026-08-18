#!/usr/bin/env python3
"""Build object-centric visual QA from final encoded P14-stage A/B videos.

The input is the already encoded camera comparison MP4, not pre-encode frames.
Cropping is review-only and centered on the prediction-side object-owned mask;
it never changes a pose or creates metric evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CASES = ("milk", "soup", "mug", "bbq", "spatula")
LABELS = ("RGB REFERENCE", "P14 PAIRWISE CHAIN", "P14 REGULARIZED", "P15 COMPLETED")
BORDERS = ((220, 220, 220), (255, 235, 40), (60, 255, 60), (255, 70, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-ab-root", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=480)
    parser.add_argument("--minimum-source-crop-px", type=int, default=320)
    parser.add_argument("--object-extent-multiplier", type=float, default=2.8)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path | str, description: str) -> Path:
    output = Path(path).expanduser().resolve(strict=True)
    if not output.is_file():
        raise RuntimeError(f"missing {description}: {output}")
    return output


def put_text(image: np.ndarray, value: str, origin: tuple[int, int], scale: float = 0.46) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (238, 238, 238),
        1,
        cv2.LINE_AA,
    )


def object_mask_path(frame: dict[str, Any], object_id: str) -> Path:
    for row in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(row, dict) and row.get("object_id") == object_id:
            return require_file(row.get("mask_path"), "object-owned mask")
    raise RuntimeError(f"frame {frame.get('frame_idx')} lacks object {object_id}")


def square_crop(mask: np.ndarray, minimum: int, multiplier: float) -> tuple[int, int, int, int, dict[str, Any]]:
    y, x = np.nonzero(mask > 0)
    if len(x) == 0:
        raise RuntimeError("empty object-owned mask")
    x0, x1 = int(x.min()), int(x.max()) + 1
    y0, y1 = int(y.min()), int(y.max()) + 1
    width = x1 - x0
    height = y1 - y0
    side = int(math.ceil(max(float(minimum), float(multiplier) * max(width, height))))
    side = min(side, mask.shape[0], mask.shape[1])
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    left = int(round(cx - 0.5 * side))
    top = int(round(cy - 0.5 * side))
    left = min(max(0, left), mask.shape[1] - side)
    top = min(max(0, top), mask.shape[0] - side)
    return left, top, left + side, top + side, {
        "mask_bbox_xyxy": [x0, y0, x1, y1],
        "crop_xyxy": [left, top, left + side, top + side],
        "source_crop_side_px": side,
    }


def stage_metrics_text(row: dict[str, Any], stage_index: int) -> str:
    if stage_index == 1:
        if not row.get("pairwise_chain_pose_available"):
            return "pose N/A"
        return (
            f"to P14: {float(row.get('chain_to_p14_rotation_deg') or 0.0):.1f}deg, "
            f"{1000.0 * float(row.get('chain_to_p14_translation_m') or 0.0):.1f}mm"
        )
    if stage_index == 2:
        if not row.get("p14_pose_available"):
            return "pose N/A"
        residual = row.get("p14_observed_surface_median_m")
        iou = row.get("p14_mask_iou")
        values = []
        if residual is not None:
            values.append(f"surface med {1000.0 * float(residual):.1f}mm")
        if iou is not None:
            values.append(f"IoU {float(iou):.3f}")
        return " | ".join(values) or "direct P14 pose"
    if stage_index == 3:
        if row.get("p15_completed_pose"):
            return f"UNCERTAIN {row.get('p15_completion_source') or 'temporal completion'}"
        return "exact same SE(3) as P14"
    delta = row.get("chain_to_p14_observed_surface_median_delta_m")
    iou_delta = row.get("chain_to_p14_mask_iou_delta")
    values = []
    if delta is not None:
        values.append(f"P14-chain residual d={1000.0 * float(delta):+.1f}mm")
    if iou_delta is not None:
        values.append(f"IoU d={float(iou_delta):+.3f}")
    return " | ".join(values) or "prediction-side RGB + owned mask"


def make_zoom_frame(
    encoded_frame: np.ndarray,
    mask: np.ndarray,
    frame_row: dict[str, Any],
    crop_size: int,
    minimum_source_crop_px: int,
    object_extent_multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = encoded_frame.shape[:2]
    if width % 4 != 0:
        raise RuntimeError(f"encoded comparison width is not divisible by four: {width}")
    panel_width = width // 4
    if panel_width != height:
        raise RuntimeError(f"expected square camera panels, got {panel_width}x{height}")
    mask_panel = cv2.resize(mask, (panel_width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    left, top, right, bottom, crop_report = square_crop(
        mask_panel, int(minimum_source_crop_px), float(object_extent_multiplier)
    )
    panels: list[np.ndarray] = []
    for stage_index in range(4):
        panel = encoded_frame[:, stage_index * panel_width : (stage_index + 1) * panel_width]
        crop = panel[top:bottom, left:right]
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        header = np.full((64, crop_size, 3), 16, dtype=np.uint8)
        put_text(header, LABELS[stage_index], (10, 24), 0.48)
        put_text(header, stage_metrics_text(frame_row, stage_index)[:68], (10, 49), 0.34)
        tile = np.vstack([header, crop])
        cv2.rectangle(tile, (1, 1), (tile.shape[1] - 2, tile.shape[0] - 2), BORDERS[stage_index], 3)
        panels.append(tile)
    output = np.hstack(panels)
    crop_report.update(
        {
            "encoded_frame_size_wh": [width, height],
            "panel_size_wh": [panel_width, height],
            "zoom_panel_size_wh": [crop_size, crop_size + 64],
        }
    )
    return output, crop_report


def encode_video(frame_dir: Path, output_path: Path, fps: float, frame_count: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-framerate",
            str(float(fps)),
            "-i",
            str(frame_dir / "%06d.jpg"),
            "-frames:v",
            str(int(frame_count)),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            str(output_path),
        ],
        check=True,
    )


def decode_video(path: Path, expected_frames: int, expected_fps: float) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    metadata_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError(f"empty decoded frame {decoded} in {path}")
        decoded += 1
    capture.release()
    passed = bool(
        metadata_count == expected_frames
        and decoded == expected_frames
        and abs(fps - expected_fps) <= 1.0e-3
    )
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata_frame_count": metadata_count,
        "decoded_frame_count": decoded,
        "fps": fps,
        "width": width,
        "height": height,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"video decode QC failed: {report}")
    return report


def select_frames(report: dict[str, Any]) -> tuple[list[int], dict[int, list[str]]]:
    rows = report["frame_rows"]
    reasons: dict[int, list[str]] = {}

    def add(frame_idx: int | None, reason: str) -> None:
        if frame_idx is None or not 0 <= int(frame_idx) < 150:
            return
        reasons.setdefault(int(frame_idx), [])
        if reason not in reasons[int(frame_idx)]:
            reasons[int(frame_idx)].append(reason)

    edge = report["stage_reconstruction"].get("edge_reconstruction_validation")
    if isinstance(edge, dict):
        add(int(edge["anchor_frame_idx"]), "atomic anchor")
    valid = [
        row
        for row in rows
        if row.get("chain_to_p14_observed_surface_median_delta_m") is not None
    ]
    if valid:
        best = min(valid, key=lambda row: float(row["chain_to_p14_observed_surface_median_delta_m"]))
        worst = max(valid, key=lambda row: float(row["chain_to_p14_observed_surface_median_delta_m"]))
        max_rotation = max(valid, key=lambda row: float(row["chain_to_p14_rotation_deg"]))
        max_translation = max(valid, key=lambda row: float(row["chain_to_p14_translation_m"]))
        add(int(best["frame_idx"]), "largest P14 surface-residual improvement")
        add(int(worst["frame_idx"]), "largest P14 surface-residual degradation")
        add(int(max_rotation["frame_idx"]), "largest chain-to-P14 rotation")
        add(int(max_translation["frame_idx"]), "largest chain-to-P14 translation")
        worst_temporal = max(
            valid,
            key=lambda row: float(row.get("temporal_translation_regularization_residual_delta_m") or -math.inf),
        )
        add(int(worst_temporal["frame_idx"]), "largest temporal-translation residual degradation")
    completed = [int(row["frame_idx"]) for row in rows if row.get("p15_completed_pose")]
    if completed:
        add(completed[0], "first P15 completed frame")
        add(completed[len(completed) // 2], "middle P15 completed frame")
        add(completed[-1], "last P15 completed frame")
    bridges = [
        int(row["frame_idx"])
        for row in rows
        if row.get("p14_direct_pose_observation_source")
        == "object_owned_rgb_optical_flow_calibrated_pnp"
    ]
    for frame_idx in bridges:
        add(frame_idx, "direct RGB-PnP bridge")
    for frame_idx in (0, 75, 149):
        add(frame_idx, "timeline reference")
    return sorted(reasons), reasons


def build_contact_sheet(
    zoom_dir: Path,
    selected: list[int],
    reasons: dict[int, list[str]],
    output_path: Path,
) -> None:
    tiles: list[np.ndarray] = []
    tile_width = 1440
    for frame_idx in selected:
        image = cv2.imread(str(zoom_dir / f"{frame_idx:06d}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"missing zoom frame {frame_idx}")
        height = int(round(image.shape[0] * tile_width / image.shape[1]))
        image = cv2.resize(image, (tile_width, height), interpolation=cv2.INTER_AREA)
        header = np.full((44, tile_width, 3), 18, dtype=np.uint8)
        put_text(header, f"frame {frame_idx:03d}: {'; '.join(reasons[frame_idx])}"[:145], (12, 29), 0.50)
        tiles.append(np.vstack([header, image]))
    if not tiles:
        raise RuntimeError("no selected zoom frames")
    blank = np.zeros_like(tiles[0])
    rows = []
    for index in range(0, len(tiles), 2):
        chunk = tiles[index : index + 2]
        if len(chunk) == 1:
            chunk.append(blank.copy())
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write {output_path}")


def build_encoded_orthographic_contact_sheet(
    video_path: Path,
    selected: list[int],
    reasons: dict[int, list[str]],
    output_path: Path,
    view_label: str,
) -> None:
    selected_set = set(selected)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open encoded {view_label} video {video_path}")
    decoded: dict[int, np.ndarray] = {}
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in selected_set:
            decoded[frame_idx] = frame
        frame_idx += 1
    capture.release()
    if frame_idx != 150 or sorted(decoded) != selected:
        raise RuntimeError(
            f"encoded world contact-sheet decode mismatch: count={frame_idx} selected={sorted(decoded)}"
        )
    target_width = 1440
    tiles: list[np.ndarray] = []
    for frame_idx in selected:
        image = decoded[frame_idx]
        target_height = int(round(image.shape[0] * target_width / image.shape[1]))
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        header = np.full((44, target_width, 3), 16, dtype=np.uint8)
        put_text(
            header,
            f"encoded {view_label.upper()} frame {frame_idx:03d}: {'; '.join(reasons[frame_idx])}"[:150],
            (12, 29),
            0.48,
        )
        tiles.append(np.vstack([header, image]))
    blank = np.zeros_like(tiles[0])
    rows: list[np.ndarray] = []
    for index in range(0, len(tiles), 2):
        chunk = tiles[index : index + 2]
        if len(chunk) == 1:
            chunk.append(blank.copy())
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write {output_path}")


def contiguous_ranges(values: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for value in sorted(values):
        if not ranges or value != ranges[-1][-1] + 1:
            ranges.append([value])
        else:
            ranges[-1].append(value)
    return ranges


def build_completion_neighborhood_sheet(
    zoom_dir: Path,
    frame_rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any] | None:
    completed = [int(row["frame_idx"]) for row in frame_rows if row.get("p15_completed_pose")]
    if not completed:
        return None
    ranges = contiguous_ranges(completed)
    panel_crop_size = 320
    panel_header = 46
    sequence_rows: list[np.ndarray] = []
    sequence_reports: list[dict[str, Any]] = []
    for sequence in ranges:
        first, last = sequence[0], sequence[-1]
        indices = list(sequence)
        if first > 0:
            indices.insert(0, first - 1)
        if last < 149:
            indices.append(last + 1)
        tiles: list[np.ndarray] = []
        for frame_idx in indices:
            image = cv2.imread(str(zoom_dir / f"{frame_idx:06d}.jpg"), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"missing zoom frame {frame_idx}")
            panel_width = image.shape[1] // 4
            p15_panel = image[:, 3 * panel_width : 4 * panel_width]
            crop = p15_panel[64:, :]
            crop = cv2.resize(crop, (panel_crop_size, panel_crop_size), interpolation=cv2.INTER_AREA)
            header = np.full((panel_header, panel_crop_size, 3), 16, dtype=np.uint8)
            is_completed = bool(frame_rows[frame_idx].get("p15_completed_pose"))
            label = "COMPLETED/UNCERTAIN" if is_completed else "DIRECT BOUNDARY"
            put_text(header, f"f{frame_idx:03d} | {label}", (8, 27), 0.40)
            tile = np.vstack([header, crop])
            border = (255, 70, 255) if is_completed else (60, 255, 60)
            cv2.rectangle(tile, (1, 1), (tile.shape[1] - 2, tile.shape[0] - 2), border, 3)
            tiles.append(tile)
        strip = np.hstack(tiles)
        sequence_header = np.full((42, strip.shape[1], 3), 12, dtype=np.uint8)
        put_text(
            sequence_header,
            f"completed interval {first}-{last}; green endpoints are direct when present; magenta rows are uncertain",
            (10, 27),
            0.43,
        )
        sequence_rows.append(np.vstack([sequence_header, strip]))
        sequence_reports.append(
            {
                "completed_interval": [first, last],
                "displayed_frame_ids": indices,
                "left_boundary_is_direct": bool(first > 0 and not frame_rows[first - 1].get("p15_completed_pose")),
                "right_boundary_is_direct": bool(last < 149 and not frame_rows[last + 1].get("p15_completed_pose")),
            }
        )
    target_width = max(row.shape[1] for row in sequence_rows)
    padded: list[np.ndarray] = []
    for row in sequence_rows:
        if row.shape[1] < target_width:
            pad = np.full((row.shape[0], target_width - row.shape[1], 3), 12, dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)
    sheet = np.vstack(padded)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write {output_path}")
    return {
        "path": str(output_path),
        "completed_frame_count": len(completed),
        "completed_intervals": sequence_reports,
        "interpretation": (
            "Visual continuity review only. A smooth completed row remains uncertain temporal completion and is not direct pose evidence."
        ),
    }


def render_case(root: Path, case_name: str, args: argparse.Namespace) -> dict[str, Any]:
    case_report_path = require_file(
        root / "cases" / case_name / "P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json",
        f"{case_name} stage report",
    )
    report = load_json(case_report_path)
    camera_video = require_file(report["outputs"]["camera_video"], f"{case_name} camera video")
    world_video = require_file(report["outputs"]["world_video"], f"{case_name} world video")
    side_video = require_file(report["outputs"]["side_video"], f"{case_name} side video")
    annotation_path = require_file(report["inputs"]["annotations"], f"{case_name} annotations")
    annotations = load_json(annotation_path)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations["frames"]
        if isinstance(frame, dict)
    }
    object_id = str(report["object_id"])
    case_qa_dir = root / "encoded_video_object_centric_qa" / case_name
    zoom_dir = case_qa_dir / "zoom_frames"
    if case_qa_dir.exists():
        if not args.replace:
            raise RuntimeError(f"refusing to overwrite {case_qa_dir}")
        shutil.rmtree(case_qa_dir)
    zoom_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(camera_video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open {camera_video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if source_count != 150 or abs(source_fps - 30.0) > 1.0e-3:
        raise RuntimeError(f"unexpected source video contract: count={source_count} fps={source_fps}")
    crop_rows: list[dict[str, Any]] = []
    for frame_idx in range(150):
        ok, encoded_frame = capture.read()
        if not ok or encoded_frame is None:
            raise RuntimeError(f"failed to decode {camera_video} frame {frame_idx}")
        mask_path = object_mask_path(frames[frame_idx], object_id)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_path}")
        zoom, crop_report = make_zoom_frame(
            encoded_frame,
            mask,
            report["frame_rows"][frame_idx],
            int(args.crop_size),
            int(args.minimum_source_crop_px),
            float(args.object_extent_multiplier),
        )
        frame_path = zoom_dir / f"{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(frame_path), zoom, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise RuntimeError(f"failed to write {frame_path}")
        crop_report.update(
            {
                "frame_idx": frame_idx,
                "object_owned_mask": str(mask_path),
                "output": str(frame_path),
            }
        )
        crop_rows.append(crop_report)
    extra_ok, _ = capture.read()
    capture.release()
    if extra_ok:
        raise RuntimeError(f"source video {camera_video} has more than 150 decodable frames")

    zoom_video = root / "videos" / f"{case_name}_object_centric_zoom_pairwise_p14_p15.mp4"
    encode_video(zoom_dir, zoom_video, source_fps, 150)
    zoom_qc = decode_video(zoom_video, 150, source_fps)
    source_qc = decode_video(camera_video, 150, source_fps)
    world_qc = decode_video(world_video, 150, source_fps)
    side_qc = decode_video(side_video, 150, source_fps)
    selected, reasons = select_frames(report)
    sheet_path = case_qa_dir / f"{case_name}_encoded_object_centric_extrema.jpg"
    build_contact_sheet(zoom_dir, selected, reasons, sheet_path)
    world_sheet_path = case_qa_dir / f"{case_name}_encoded_world_extrema.jpg"
    build_encoded_orthographic_contact_sheet(
        world_video, selected, reasons, world_sheet_path, "world"
    )
    side_sheet_path = case_qa_dir / f"{case_name}_encoded_side_extrema.jpg"
    build_encoded_orthographic_contact_sheet(
        side_video, selected, reasons, side_sheet_path, "side"
    )
    completion_sheet_path = case_qa_dir / f"{case_name}_p15_completed_interval_neighborhoods.jpg"
    completion_neighborhoods = build_completion_neighborhood_sheet(
        zoom_dir, report["frame_rows"], completion_sheet_path
    )
    return {
        "case": case_name,
        "object_id": object_id,
        "claim_scope": (
            "Object-centric crop of the final encoded camera A/B video. The crop is centered only for review and does not change pose, camera, MANO, geometry, masks, or metrics."
        ),
        "inputs": {
            "case_stage_report": str(case_report_path),
            "case_stage_report_sha256": sha256_file(case_report_path),
            "encoded_camera_video": str(camera_video),
            "encoded_camera_video_sha256": sha256_file(camera_video),
            "encoded_world_video": str(world_video),
            "encoded_world_video_sha256": sha256_file(world_video),
            "encoded_side_video": str(side_video),
            "encoded_side_video_sha256": sha256_file(side_video),
            "annotations": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
        },
        "crop_contract": {
            "center_source": "prediction_side_object_owned_appearance_mask_bbox",
            "minimum_source_crop_px": int(args.minimum_source_crop_px),
            "object_extent_multiplier": float(args.object_extent_multiplier),
            "crop_is_pose_or_metric_evidence": False,
            "identical_source_crop_applied_to_all_four_panels": True,
            "input_is_final_encoded_mp4": True,
        },
        "source_video_decode_qc": source_qc,
        "source_world_video_decode_qc": world_qc,
        "source_side_video_decode_qc": side_qc,
        "zoom_video_decode_qc": zoom_qc,
        "selected_extrema_frames": selected,
        "selected_extrema_reasons": {str(key): value for key, value in reasons.items()},
        "p15_completed_interval_neighborhoods": completion_neighborhoods,
        "outputs": {
            "zoom_frames": str(zoom_dir),
            "zoom_video": str(zoom_video),
            "encoded_extrema_contact_sheet": str(sheet_path),
            "encoded_world_extrema_contact_sheet": str(world_sheet_path),
            "encoded_side_extrema_contact_sheet": str(side_sheet_path),
            "p15_completed_interval_neighborhood_sheet": (
                str(completion_sheet_path) if completion_neighborhoods is not None else None
            ),
        },
        "crop_rows": crop_rows,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    root = args.stage_ab_root.expanduser().resolve(strict=True)
    reports = [render_case(root, case_name, args) for case_name in CASES]
    output_dir = root / "encoded_video_object_centric_qa"
    report = {
        "schema": "hot3d_p14_stage_ab_encoded_object_centric_visual_qa_v1",
        "status": "encoded_zoom_artifacts_complete_visual_review_pending",
        "stage_ab_root": str(root),
        "qa_script": str(Path(__file__).resolve()),
        "qa_script_sha256": sha256_file(Path(__file__).resolve()),
        "case_count": len(reports),
        "source_camera_video_count": len(reports),
        "source_world_video_count": len(reports),
        "source_side_video_count": len(reports),
        "zoom_video_count": len(reports),
        "source_camera_decoded_frame_count": sum(
            row["source_video_decode_qc"]["decoded_frame_count"] for row in reports
        ),
        "source_world_decoded_frame_count": sum(
            row["source_world_video_decode_qc"]["decoded_frame_count"] for row in reports
        ),
        "source_side_decoded_frame_count": sum(
            row["source_side_video_decode_qc"]["decoded_frame_count"] for row in reports
        ),
        "zoom_decoded_frame_count": sum(
            row["zoom_video_decode_qc"]["decoded_frame_count"] for row in reports
        ),
        "cases": reports,
        "visual_review_pending": True,
        "elapsed_s": time.time() - started,
    }
    report_path = output_dir / "ENCODED_OBJECT_CENTRIC_QA_REPORT.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "source_camera_decoded_frames": report["source_camera_decoded_frame_count"],
                "source_world_decoded_frames": report["source_world_decoded_frame_count"],
                "source_side_decoded_frames": report["source_side_decoded_frame_count"],
                "zoom_decoded_frames": report["zoom_decoded_frame_count"],
                "zoom_videos": [row["outputs"]["zoom_video"] for row in reports],
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
