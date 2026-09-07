#!/usr/bin/env python3
"""Post-freeze HOT3D rotation evaluation for V20 pose reports.

Prediction and HOT3D use different object canonical frames.  For each candidate
this evaluator converts object poses to their respective camera frames, fits
one fixed object-local transform at the anchor frame, and measures trajectory
rotation/translation error thereafter.  GT is evaluation-only and is never
written into a prediction pose report or candidate-generation artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def rotation_angle_deg(transform: np.ndarray) -> float:
    return float(
        np.degrees(
            Rotation.from_matrix(
                np.asarray(transform, dtype=np.float64)[:3, :3]
            ).magnitude()
        )
    )


def summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=/path/report.json")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("candidate label/path cannot be empty")
    return label, Path(path)


def load_prediction_cameras(path: Path) -> dict[int, np.ndarray]:
    annotations = load_json(path)
    result: dict[int, np.ndarray] = {}
    for frame in annotations.get("frames", []):
        if not isinstance(frame, dict) or frame.get("frame_idx") is None:
            continue
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        value = camera.get("T_world_camera_metric") or camera.get("T_world_camera")
        transform = np.asarray(value or [], dtype=np.float64)
        if transform.shape == (4, 4) and np.isfinite(transform).all():
            result[int(frame["frame_idx"])] = transform
    return result


def load_prediction_objects(report: dict[str, Any]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for row in report.get("pose_rows", []):
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        rotation = np.asarray(
            row.get("rotation_world_from_completed_canonical_matrix") or [],
            dtype=np.float64,
        )
        translation = np.asarray(row.get("translation_world_m") or [], dtype=np.float64)
        if (
            rotation.shape == (3, 3)
            and translation.shape == (3,)
            and np.isfinite(rotation).all()
            and np.isfinite(translation).all()
        ):
            result[int(row["frame_idx"])] = make_transform(rotation, translation)
    return result


def load_gt(
    path: Path, object_name: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        names = np.asarray(archive["object_names"]).astype(str).tolist()
        if object_name not in names:
            raise RuntimeError(f"GT object {object_name!r} not found in {names}")
        object_position = names.index(object_name)
        frame_ids = np.asarray(archive["frame_idx"], dtype=np.int64)
        cameras = np.asarray(archive["T_world_from_camera"], dtype=np.float64)
        objects = np.asarray(archive["T_world_from_object"], dtype=np.float64)
        available = np.asarray(archive["object_pose_available"], dtype=bool)
        source_ids = np.asarray(archive["source_frame_idx"], dtype=np.int64)
    gt_cameras: dict[int, np.ndarray] = {}
    gt_objects: dict[int, np.ndarray] = {}
    mapping: dict[str, int] = {}
    for position, frame_idx in enumerate(frame_ids.tolist()):
        if not bool(available[position, object_position]):
            continue
        gt_cameras[int(frame_idx)] = cameras[position]
        gt_objects[int(frame_idx)] = objects[position, object_position]
        mapping[str(int(frame_idx))] = int(source_ids[position])
    return gt_cameras, gt_objects, {
        "object_name": object_name,
        "object_index": object_position,
        "frame_to_source_frame": mapping,
    }


def evaluate_candidate(
    report: dict[str, Any],
    prediction_cameras: dict[int, np.ndarray],
    gt_cameras: dict[int, np.ndarray],
    gt_objects: dict[int, np.ndarray],
    anchor_frame: int,
) -> dict[str, Any]:
    prediction_objects = load_prediction_objects(report)
    common = sorted(
        set(prediction_objects)
        .intersection(prediction_cameras)
        .intersection(gt_cameras)
        .intersection(gt_objects)
    )
    if anchor_frame not in common:
        raise RuntimeError(f"candidate lacks evaluable anchor frame {anchor_frame}")
    prediction_camera_object = {
        frame_idx: np.linalg.inv(prediction_cameras[frame_idx])
        @ prediction_objects[frame_idx]
        for frame_idx in common
    }
    gt_camera_object = {
        frame_idx: np.linalg.inv(gt_cameras[frame_idx]) @ gt_objects[frame_idx]
        for frame_idx in common
    }
    object_frame_alignment = (
        np.linalg.inv(prediction_camera_object[anchor_frame])
        @ gt_camera_object[anchor_frame]
    )
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for frame_idx in common:
        aligned_prediction = (
            prediction_camera_object[frame_idx] @ object_frame_alignment
        )
        error = np.linalg.inv(gt_camera_object[frame_idx]) @ aligned_prediction
        relative_prediction = (
            np.linalg.inv(prediction_camera_object[anchor_frame])
            @ prediction_camera_object[frame_idx]
        )
        relative_gt = (
            np.linalg.inv(gt_camera_object[anchor_frame])
            @ gt_camera_object[frame_idx]
        )
        rotation_error = rotation_angle_deg(error)
        translation_error = float(np.linalg.norm(error[:3, 3]))
        rotation_errors.append(rotation_error)
        translation_errors.append(translation_error)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "rotation_error_deg": rotation_error,
                "translation_error_m": translation_error,
                "prediction_relative_rotation_angle_deg": rotation_angle_deg(
                    relative_prediction
                ),
                "gt_relative_rotation_angle_deg": rotation_angle_deg(relative_gt),
            }
        )
    return {
        "schema": report.get("schema"),
        "status": report.get("status"),
        "row_count": int(len(common)),
        "frame_min": int(common[0]),
        "frame_max": int(common[-1]),
        "anchor_frame": int(anchor_frame),
        "object_local_alignment_pred_to_gt": object_frame_alignment.astype(
            float
        ).tolist(),
        "frame0_aligned_rotation_error_deg": summary(rotation_errors),
        "frame0_aligned_translation_error_m": summary(translation_errors),
        "endpoint": rows[-1],
        "worst_rotation_frames": [
            int(row["frame_idx"])
            for row in sorted(
                rows, key=lambda item: item["rotation_error_deg"], reverse=True
            )[:10]
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--gt-npz", type=Path, required=True)
    parser.add_argument("--object-name", default="carton_milk")
    parser.add_argument("--anchor-frame", type=int, default=0)
    parser.add_argument(
        "--candidate", action="append", type=parse_candidate, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = args.annotations.expanduser().resolve()
    gt_path = args.gt_npz.expanduser().resolve()
    prediction_cameras = load_prediction_cameras(annotations)
    gt_cameras, gt_objects, gt_metadata = load_gt(gt_path, args.object_name)
    runs: dict[str, Any] = {}
    input_hashes: dict[str, str] = {
        "annotations": sha256_file(annotations),
        "gt_npz": sha256_file(gt_path),
    }
    for label, candidate_path in args.candidate:
        if label in runs:
            raise RuntimeError(f"duplicate candidate label: {label}")
        resolved = candidate_path.expanduser().resolve()
        report = load_json(resolved)
        result = evaluate_candidate(
            report,
            prediction_cameras,
            gt_cameras,
            gt_objects,
            int(args.anchor_frame),
        )
        runs[label] = {"pose_report": str(resolved), **result}
        input_hashes[f"candidate:{label}"] = sha256_file(resolved)
    ranking = sorted(
        runs,
        key=lambda label: float(
            runs[label]["frame0_aligned_rotation_error_deg"]["median"]
        ),
    )
    output = {
        "schema": "v20_hot3d_object_rotation_postfreeze_evaluation_v1",
        "status": "evaluation_complete",
        "prediction_gt_separation": {
            "gt_used_as_solver_input": False,
            "gt_used_for_initialization": False,
            "gt_used_for_candidate_generation": False,
            "gt_used_only_for_postfreeze_evaluation": True,
        },
        "method": (
            "camera-relative object pose with one fixed object-local transform "
            "fitted at the anchor frame"
        ),
        "inputs": {
            "annotations": str(annotations),
            "gt_npz": str(gt_path),
            "object_name": args.object_name,
            "anchor_frame": int(args.anchor_frame),
            "sha256": input_hashes,
        },
        "gt_metadata": gt_metadata,
        "rotation_median_ranking": ranking,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "ranking": [
                    {
                        "label": label,
                        "rotation_error_deg": runs[label][
                            "frame0_aligned_rotation_error_deg"
                        ],
                        "endpoint": runs[label]["endpoint"],
                    }
                    for label in ranking
                ],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
