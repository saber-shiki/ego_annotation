#!/usr/bin/env python3
"""Render full-duration actual-video comparisons for P14/P15 object state and P18 hands.

Top row overlays the observed target Mask, frozen P14->P15 completed-Mesh
projection, pose-gated P14->P15 projection, and the original P19 metric-world
render on the real raw RGB timeline. Sparse released object-Mask GT is drawn
only on its five annotated frames.

Bottom row overlays HaWoR, rejected pre-gate P18 solver candidates, emitted
post-gate P18 output, and P18b hand skeletons on the same RGB. Independent 3-D
hand-GT errors are shown as text when released GT exists, but GT joints are not
projected onto raw RGB because the local shard lacks the Aria VRS distortion
calibration needed for that pixel mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from compare_current_outputs_to_gt import (
    load_mesh,
    pose_map,
    projected_mesh_mask,
)
from evaluate_benchmark import load_hawor_joint_map, load_interval_joint_map

PANEL_SIZE = 480
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def target_object(frame: dict[str, Any], object_id: str) -> dict[str, Any]:
    matches = [row for row in frame.get("objects", []) if str(row.get("object_id")) == object_id]
    if len(matches) != 1:
        raise RuntimeError(f"frame {frame.get('frame_idx')} expected one object {object_id}, found {len(matches)}")
    return matches[0]


def read_mask(path: Path, size: int = PANEL_SIZE) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def letterbox(image: np.ndarray, size: int = PANEL_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def add_header(panel: np.ndarray, title: str, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    result = panel.copy()
    height = 72
    cv2.rectangle(result, (0, 0), (result.shape[1], height), (0, 0, 0), -1)
    cv2.putText(result, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    for index, line in enumerate(lines[:2]):
        cv2.putText(
            result,
            line,
            (8, 46 + 19 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    cv2.line(result, (result.shape[1] - 1, 0), (result.shape[1] - 1, result.shape[0] - 1), (90, 90, 90), 1)
    return result


def overlay_masks(
    image: np.ndarray,
    prediction: np.ndarray,
    prediction_color: tuple[int, int, int],
    observed: np.ndarray | None,
    gt: np.ndarray | None,
) -> np.ndarray:
    layer = image.copy()
    layer[prediction] = prediction_color
    if gt is not None:
        layer[gt] = (35, 220, 35)
        layer[prediction & gt] = (245, 245, 245)
    result = cv2.addWeighted(image, 0.52, layer, 0.48, 0.0)
    pred_contours, _ = cv2.findContours(prediction.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, pred_contours, -1, prediction_color, 2, cv2.LINE_AA)
    if observed is not None:
        observed_contours, _ = cv2.findContours(observed.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, observed_contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
    if gt is not None:
        gt_contours, _ = cv2.findContours(gt.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, gt_contours, -1, (35, 255, 35), 3, cv2.LINE_AA)
    return result


def observed_panel(image: np.ndarray, observed: np.ndarray, gt: np.ndarray | None, frame_idx: int) -> np.ndarray:
    layer = image.copy()
    layer[observed] = (255, 255, 0)
    result = cv2.addWeighted(image, 0.64, layer, 0.36, 0.0)
    observed_contours, _ = cv2.findContours(observed.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, observed_contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
    gt_text = "green=sparse GT" if gt is not None else "GT unavailable this frame"
    if gt is not None:
        gt_contours, _ = cv2.findContours(gt.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, gt_contours, -1, (35, 255, 35), 3, cv2.LINE_AA)
    return add_header(result, "Observed target on actual RGB", [f"f={frame_idx} cyan=SAM2/object-owned", gt_text], (255, 255, 0))


def p15_source(pose: dict[str, Any]) -> str:
    source = str((pose.get("temporal_pose_graph") or {}).get("pose_source", "unknown"))
    if source.startswith("direct_visible_pose_observation"):
        return "direct"
    if source == "interpolated_between_visible_pose_observations":
        return "interpolated"
    if source == "nearest_visible_pose_hold":
        return "nearest hold"
    return source


def project_world_joints(
    joints_world: np.ndarray,
    frame: dict[str, Any],
    raw_video: dict[str, Any] | None,
    size: int = PANEL_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray((frame.get("camera") or {}).get("T_world_camera_metric") or [], dtype=np.float64)
    if transform.shape != (4, 4):
        raise RuntimeError(f"frame {frame.get('frame_idx')} lacks T_world_camera_metric")
    # Reuse the exact projection contract used by the rigid-state evaluator.
    from render_v19_rigid_state_artifact import scaled_intrinsics_for_frame, world_points_to_camera, project_camera_points

    intrinsics, _report = scaled_intrinsics_for_frame(frame, size, size, raw_video)
    camera = world_points_to_camera(joints_world, transform)
    u, v, z, valid = project_camera_points(camera, intrinsics, size, size)
    return np.c_[u, v], valid & (z > 0.01)


def draw_hand_skeleton(
    image: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for first, second in HAND_EDGES:
        if valid[first] and valid[second]:
            p0 = tuple(np.rint(points[first]).astype(int))
            p1 = tuple(np.rint(points[second]).astype(int))
            cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)
    for index, point in enumerate(points):
        if valid[index]:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 3 if index else 5, color, -1, cv2.LINE_AA)


def weighted_hand_error_map(report: dict[str, Any], metric: str) -> dict[int, float]:
    by_frame: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in report.get("per_frame", []):
        value = row.get(metric)
        if row.get("status") == "scored" and value is not None:
            by_frame[int(row["local_frame_idx"])].append((float(value), int(row["eligible_gt_joint_count"])))
    output: dict[int, float] = {}
    for frame, values in by_frame.items():
        total_weight = sum(weight for _value, weight in values)
        output[frame] = sum(value * weight for value, weight in values) / total_weight
    return output


def hand_panel(
    image: np.ndarray,
    frame: dict[str, Any],
    raw_video: dict[str, Any] | None,
    frame_idx: int,
    candidate: dict[tuple[int, str], np.ndarray],
    baseline: dict[tuple[int, str], np.ndarray],
    candidate_color: tuple[int, int, int],
    title: str,
    gt_absolute_error: float | None,
    show_baseline_underlay: bool,
) -> tuple[np.ndarray, float]:
    result = image.copy()
    max_delta_mm = 0.0
    for side in ("left", "right"):
        key = (frame_idx, side)
        if key not in candidate:
            continue
        candidate_points, candidate_valid = project_world_joints(candidate[key], frame, raw_video)
        if show_baseline_underlay and key in baseline:
            baseline_points, baseline_valid = project_world_joints(baseline[key], frame, raw_video)
            draw_hand_skeleton(result, baseline_points, baseline_valid, (255, 255, 0), 1)
            common = candidate_valid & baseline_valid
            for source, target in zip(baseline_points[common], candidate_points[common]):
                cv2.line(
                    result,
                    tuple(np.rint(source).astype(int)),
                    tuple(np.rint(target).astype(int)),
                    (30, 30, 235),
                    1,
                    cv2.LINE_AA,
                )
            max_delta_mm = max(max_delta_mm, float(np.max(np.linalg.norm(candidate[key] - baseline[key], axis=1))) * 1000.0)
        draw_hand_skeleton(result, candidate_points, candidate_valid, candidate_color, 2)
    gt_line = (
        f"independent 3D GT abs MPJPE={gt_absolute_error:.1f} mm (not drawn)"
        if gt_absolute_error is not None
        else "independent 3D hand GT unavailable this frame"
    )
    return add_header(result, title, [gt_line, f"max state delta vs HaWoR={max_delta_mm:.2f} mm"], candidate_color), max_delta_mm


def load_p18_output_and_pregate(
    path: Path,
) -> tuple[
    dict[tuple[int, str], np.ndarray],
    dict[tuple[int, str], np.ndarray],
    dict[tuple[int, str], float],
]:
    payload = load_json(path)
    output: dict[tuple[int, str], np.ndarray] = {}
    pregate: dict[tuple[int, str], np.ndarray] = {}
    shifts_mm: dict[tuple[int, str], float] = {}
    for row in payload.get("per_frame_states", []):
        joints = np.asarray(row.get("optimized_joints_world_m", []), dtype=np.float64)
        if joints.shape != (21, 3) or not np.isfinite(joints).all():
            continue
        key = (int(row["frame_idx"]), str(row["hand_side"]))
        shift = np.asarray((row.get("output_translation_gate") or {}).get("applied_world_shift_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        if shift.shape != (3,) or not np.isfinite(shift).all():
            raise RuntimeError(f"invalid P18 output-gate shift at {key}")
        output[key] = joints
        pregate[key] = joints - shift[None, :]
        shifts_mm[key] = float(np.linalg.norm(shift) * 1000.0)
    return output, pregate, shifts_mm


def encode_video(frame_dir: Path, output: Path, fps: float, crop: str | None = None) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-i", str(frame_dir / "frame_%06d.jpg"),
    ]
    if crop is not None:
        command += ["-vf", crop]
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(output)]
    run(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, required=True)
    parser.add_argument("--frozen-p15", type=Path, required=True)
    parser.add_argument("--fixed-p15", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--p18-raw", type=Path, required=True)
    parser.add_argument("--p18b", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--current-output-evaluation", type=Path, required=True)
    parser.add_argument("--runtime-world-frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", default="tire_lever")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.replace:
            raise FileExistsError(f"output exists: {output_dir}; pass --replace")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    annotations = load_json(args.annotations)
    frames = {int(row["frame_idx"]): row for row in annotations["frames"]}
    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else None
    fps = float((raw_video or {}).get("fps", 30.0))
    frame_count = int((raw_video or {}).get("frame_count", len(frames)))
    vertices, faces, mesh_report = load_mesh(args.completed_mesh)
    frozen_poses = pose_map(load_json(args.frozen_p15))
    fixed_report = load_json(args.fixed_p15)
    fixed_poses = pose_map(fixed_report)
    hawor, _camera = load_hawor_joint_map(args.hawor_npz)
    p18_raw, p18_pregate, p18_gate_shift_mm = load_p18_output_and_pregate(args.p18_raw)
    p18b = load_interval_joint_map(args.p18b)
    evaluation = load_json(args.current_output_evaluation)

    hand_reports = evaluation["hand_pose"]
    hand_abs = {
        "hawor": weighted_hand_error_map(hand_reports["hawor_baseline"], "absolute_mpjpe_mm"),
        "p18_raw": weighted_hand_error_map(hand_reports["frozen_p18_raw"], "absolute_mpjpe_mm"),
        "p18b": weighted_hand_error_map(hand_reports["frozen_p18b_canonical"], "absolute_mpjpe_mm"),
    }
    projection_eval = evaluation["projected_mesh_visible_mask_diagnostic"]["methods"]
    iou_maps = {
        "frozen": {int(row["local_frame_idx"]): float(row["iou"]) for row in projection_eval["frozen_v1"]["per_frame"]},
        "fixed": {int(row["local_frame_idx"]): float(row["iou"]) for row in projection_eval["pose_gate_fixed"]["per_frame"]},
    }
    gt_payload = load_json(args.ground_truth_dir / "object_visible_mask_gt.json")
    gt_masks = {
        int(row["local_frame_idx"]): args.ground_truth_dir / "object_visible_masks_960" / row["runtime_mask_file"]
        for row in gt_payload["frames"]
    }

    selected_indices = [0, 30, 60, 90, 115, 120, 149]
    selected_frames: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    max_p18_output_delta_mm = 0.0
    max_p18_pregate_delta_mm = 0.0
    max_p18_gate_shift_mm = 0.0
    with tempfile.TemporaryDirectory(prefix="p14_p15_p18_actual_video_") as temporary:
        frame_dir = Path(temporary)
        for frame_idx in range(frame_count):
            frame = frames[frame_idx]
            raw = cv2.imread(str(frame["raw_frame_path"]), cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(frame["raw_frame_path"])
            raw = cv2.resize(raw, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
            observed_path = Path(target_object(frame, args.object_id)["mask_path"])
            observed = read_mask(observed_path)
            gt = read_mask(gt_masks[frame_idx]) if frame_idx in gt_masks else None

            observed_rgb_panel = observed_panel(raw, observed, gt, frame_idx)
            object_panels = [observed_rgb_panel]
            projection_rows: dict[str, Any] = {}
            for label, poses, color in (
                ("frozen", frozen_poses, (220, 40, 220)),
                ("fixed", fixed_poses, (0, 165, 255)),
            ):
                pose = poses[frame_idx]
                prediction, projection = projected_mesh_mask(
                    vertices, faces, pose, frame, raw_video, PANEL_SIZE, PANEL_SIZE
                )
                panel = overlay_masks(raw, prediction, color, observed, gt)
                iou = iou_maps[label].get(frame_idx)
                iou_text = f"sparse GT IoU@960={iou:.3f}" if iou is not None else "sparse GT unavailable"
                ready = pose.get("annotation_ready")
                title = "Frozen P14->P15 Mesh" if label == "frozen" else "Fixed P14->P15 | POSE UNREADY"
                panel = add_header(
                    panel,
                    title,
                    [f"f={frame_idx} source={p15_source(pose)} | cyan=observed", iou_text],
                    color,
                )
                object_panels.append(panel)
                projection_rows[label] = {
                    "pose_source": p15_source(pose),
                    "annotation_ready": ready,
                    "sparse_gt_iou_960": iou,
                    "projected_mask_area_480": int(np.count_nonzero(prediction)),
                    "projection": projection,
                }

            world_path = args.runtime_world_frames_dir / f"{frame_idx:06d}.jpg"
            world_image = cv2.imread(str(world_path), cv2.IMREAD_COLOR)
            if world_image is None:
                raise FileNotFoundError(world_path)
            world_image = letterbox(world_image)
            world_panel = add_header(
                world_image,
                "P19 metric-world render",
                ["green=completed Mesh body", "shows sheet shape and hand/object separation"],
                (80, 220, 80),
            )
            object_panels.append(world_panel)

            hawor_panel, _ = hand_panel(
                raw, frame, raw_video, frame_idx, hawor, hawor, (255, 255, 0),
                "HaWoR source hand state", hand_abs["hawor"].get(frame_idx), False,
            )
            pregate_panel, pregate_delta = hand_panel(
                raw, frame, raw_video, frame_idx, p18_pregate, hawor, (30, 30, 235),
                "P18 pre-gate candidate | REJECTED", None, True,
            )
            raw_panel, raw_delta = hand_panel(
                raw, frame, raw_video, frame_idx, p18_raw, hawor, (0, 165, 255),
                "P18 emitted after translation gate", hand_abs["p18_raw"].get(frame_idx), True,
            )
            p18b_panel, p18b_delta = hand_panel(
                raw, frame, raw_video, frame_idx, p18b, hawor, (255, 100, 0),
                "P18b canonical | source preserved", hand_abs["p18b"].get(frame_idx), True,
            )
            frame_gate_shift = max(
                [value for (frame_number, _side), value in p18_gate_shift_mm.items() if frame_number == frame_idx]
                or [0.0]
            )
            max_p18_output_delta_mm = max(max_p18_output_delta_mm, raw_delta, p18b_delta)
            max_p18_pregate_delta_mm = max(max_p18_pregate_delta_mm, pregate_delta)
            max_p18_gate_shift_mm = max(max_p18_gate_shift_mm, frame_gate_shift)
            object_row = np.concatenate(object_panels, axis=1)
            hand_row = np.concatenate([hawor_panel, pregate_panel, raw_panel, p18b_panel], axis=1)
            combined = np.concatenate([object_row, hand_row], axis=0)
            cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}.jpg"), combined, [cv2.IMWRITE_JPEG_QUALITY, 93])
            if frame_idx in selected_indices:
                selected_frames.append(cv2.resize(combined, (960, 480), interpolation=cv2.INTER_AREA))
            per_frame.append(
                {
                    "frame_idx": frame_idx,
                    "raw_frame_path": frame["raw_frame_path"],
                    "runtime_world_frame_path": str(world_path),
                    "observed_mask_path": str(observed_path),
                    "sparse_gt_mask_path": str(gt_masks[frame_idx]) if frame_idx in gt_masks else None,
                    "object": projection_rows,
                    "hand": {
                        "hawor_absolute_mpjpe_mm": hand_abs["hawor"].get(frame_idx),
                        "p18_raw_absolute_mpjpe_mm": hand_abs["p18_raw"].get(frame_idx),
                        "p18b_absolute_mpjpe_mm": hand_abs["p18b"].get(frame_idx),
                        "p18_pregate_max_joint_state_delta_vs_hawor_mm": pregate_delta,
                        "p18_output_gate_max_translation_shift_mm": frame_gate_shift,
                        "p18_raw_emitted_max_joint_state_delta_vs_hawor_mm": raw_delta,
                        "p18b_max_joint_state_delta_vs_hawor_mm": p18b_delta,
                    },
                }
            )

        combined_video = output_dir / "p14_p15_p18_actual_video_comparison.mp4"
        object_video = output_dir / "p14_p15_object_mesh_actual_video_comparison.mp4"
        hand_video = output_dir / "p18_hand_actual_video_comparison.mp4"
        encode_video(frame_dir, combined_video, fps)
        encode_video(frame_dir, object_video, fps, f"crop={PANEL_SIZE*4}:{PANEL_SIZE}:0:0")
        encode_video(frame_dir, hand_video, fps, f"crop={PANEL_SIZE*4}:{PANEL_SIZE}:0:{PANEL_SIZE}")

    columns = 3
    rows = (len(selected_frames) + columns - 1) // columns
    sheet = np.full((rows * 480, columns * 960, 3), 238, dtype=np.uint8)
    for index, image in enumerate(selected_frames):
        y = index // columns * 480
        x = index % columns * 960
        sheet[y : y + 480, x : x + 960] = image
    sheet_path = output_dir / "actual_video_comparison_selected_frames.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])

    source_files = {
        "script": Path(__file__).resolve(),
        "annotations": args.annotations.resolve(),
        "completed_mesh": args.completed_mesh.resolve(),
        "frozen_p15": args.frozen_p15.resolve(),
        "fixed_p15": args.fixed_p15.resolve(),
        "hawor_npz": args.hawor_npz.resolve(),
        "p18_raw": args.p18_raw.resolve(),
        "p18b": args.p18b.resolve(),
        "object_visible_mask_gt": (args.ground_truth_dir / "object_visible_mask_gt.json").resolve(),
        "current_output_evaluation": args.current_output_evaluation.resolve(),
    }
    output_files = {
        "combined_video": output_dir / "p14_p15_p18_actual_video_comparison.mp4",
        "object_video": output_dir / "p14_p15_object_mesh_actual_video_comparison.mp4",
        "hand_video": output_dir / "p18_hand_actual_video_comparison.mp4",
        "selected_frames_sheet": sheet_path,
    }
    provenance = {
        "source_file_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "runtime_world_frames_tree_sha256": sha256_tree(list(args.runtime_world_frames_dir.glob("*.jpg"))),
        "sparse_gt_masks_tree_sha256": sha256_tree(list((args.ground_truth_dir / "object_visible_masks_960").glob("*.png"))),
        "output_file_sha256": {name: sha256_file(path) for name, path in output_files.items()},
    }

    manifest = {
        "status": "actual_video_phase_comparison_rendered",
        "claim_scope": {
            "object": "Actual raw RGB plus observed Mask and completed-Mesh projections. Sparse visible-Mask GT appears on five frames. This exposes visible mismatch but does not isolate geometry from pose/camera/occlusion and is not object SE(3) GT.",
            "hand": "Actual raw RGB plus HaWoR, rejected pre-gate P18 solver candidate, emitted gated P18 output, and P18b skeletons under the runtime estimated camera. Independent 3-D GT errors appear as text for emitted states; GT skeletons are not projected to raw RGB because VRS distortion calibration is unavailable.",
        },
        "important_findings_encoded_in_video": [
            "Frozen P15 directly consumes 33 rows but 27 were explicitly rejected upstream; apparent overlap is not trusted pose evidence.",
            "Fixed P15 has six direct rows clustered at 115-123 and is annotation_ready=false; most timeline rows are nearest holds.",
            "P15 performed zero correction in both states (nfev=1, cost=0).",
            "P18 pre-gate solver candidates require translation shifts up to the reported maximum; the emitted state applies the output gate, while P18b preserves HaWoR exactly.",
        ],
        "inputs": {
            "annotations": str(args.annotations.resolve()),
            "completed_mesh": str(args.completed_mesh.resolve()),
            "frozen_p15": str(args.frozen_p15.resolve()),
            "fixed_p15": str(args.fixed_p15.resolve()),
            "hawor_npz": str(args.hawor_npz.resolve()),
            "p18_raw": str(args.p18_raw.resolve()),
            "p18b": str(args.p18b.resolve()),
            "ground_truth_dir": str(args.ground_truth_dir.resolve()),
            "current_output_evaluation": str(args.current_output_evaluation.resolve()),
            "runtime_world_frames_dir": str(args.runtime_world_frames_dir.resolve()),
        },
        "video": {
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": frame_count / fps,
            "combined_size": [PANEL_SIZE * 4, PANEL_SIZE * 2],
            "row_size": [PANEL_SIZE * 4, PANEL_SIZE],
        },
        "mesh": mesh_report,
        "provenance": provenance,
        "fixed_annotation_ready": fixed_report.get("annotation_ready"),
        "maximum_p18_pregate_joint_state_delta_vs_hawor_mm": max_p18_pregate_delta_mm,
        "maximum_p18_output_gate_translation_shift_mm": max_p18_gate_shift_mm,
        "maximum_p18_emitted_joint_state_delta_vs_hawor_mm": max_p18_output_delta_mm,
        "selected_source_frame_mapping": [
            {"sheet_index": index, "source_frame_idx": frame_idx}
            for index, frame_idx in enumerate(selected_indices)
        ],
        "outputs": {name: str(path) for name, path in output_files.items()},
        "per_frame": per_frame,
    }
    write_json(output_dir / "actual_video_comparison_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "frame_count": frame_count,
                "duration_s": frame_count / fps,
                "fixed_annotation_ready": manifest["fixed_annotation_ready"],
                "maximum_p18_pregate_joint_state_delta_vs_hawor_mm": max_p18_pregate_delta_mm,
                "maximum_p18_output_gate_translation_shift_mm": max_p18_gate_shift_mm,
                "maximum_p18_emitted_joint_state_delta_vs_hawor_mm": max_p18_output_delta_mm,
                "outputs": manifest["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
