#!/usr/bin/env python3
"""Finalize the read-only HOT3D P14-stage A/B after explicit image review.

This script does not run prediction, alter a finalized case, or infer a visual
verdict from numeric thresholds.  It independently verifies source P14/P15 SE(3),
raw HaWoR versus delivered full MANO vertices, P16's inactive correction state,
encoded videos, keyframe GLBs, and a pre-captured source-hash baseline.  A human
image-read review JSON is a required input and remains distinct from mechanical
QC.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

CASE_ORDER = ("milk", "soup", "mug", "bbq", "spatula")
SCHEMA = "hot3d_p14_stage_ab_final_review_v1"
POSE_ROTATION_KEY = "rotation_world_from_completed_canonical_matrix"
POSE_TRANSLATION_KEY = "translation_world_m"
EXCLUDED_INTERMEDIATE_DIR_NAMES = {
    "camera_frames",
    "world_frames",
    "side_frames",
    "zoom_frames",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--stage-ab-root", type=Path, required=True)
    parser.add_argument("--manual-visual-review", type=Path, required=True)
    parser.add_argument("--source-hash-baseline", type=Path, required=True)
    parser.add_argument("--diagnostic-code-commit", required=True)
    parser.add_argument("--replace-final-reports", action="store_true")
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
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"missing {description}: {resolved}")
    return resolved


def require_inside(path: Path, root: Path, description: str) -> None:
    if not path.is_relative_to(root):
        raise RuntimeError(f"{description} escapes expected root: {path} not under {root}")


def numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    if len(finite) == 0:
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def pose_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    rotation_raw = row.get(POSE_ROTATION_KEY)
    translation_raw = row.get(POSE_TRANSLATION_KEY)
    if rotation_raw is None and translation_raw is None:
        return None
    rotation = np.asarray(rotation_raw, dtype=np.float64)
    translation = np.asarray(translation_raw, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError(f"malformed pose row frame={row.get('frame_idx')}")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise RuntimeError(f"nonfinite pose row frame={row.get('frame_idx')}")
    return rotation, translation


def rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(second @ first.T).magnitude()))


def contiguous_intervals(values: list[int]) -> list[list[int]]:
    intervals: list[list[int]] = []
    for value in sorted(values):
        if not intervals or value != intervals[-1][-1] + 1:
            intervals.append([value])
        else:
            intervals[-1].append(value)
    return intervals


def extrema(rows: list[dict[str, Any]], key: str, mode: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    if not valid:
        return None
    selected = min(valid, key=lambda row: float(row[key])) if mode == "min" else max(
        valid, key=lambda row: float(row[key])
    )
    return {"frame_idx": int(selected["frame_idx"]), "value": float(selected[key])}


def parse_hash_baseline(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"malformed source hash baseline line {line_number}")
        source = Path(parts[1]).expanduser().resolve(strict=True)
        actual = sha256_file(source)
        rows.append(
            {
                "path": str(source),
                "expected_sha256": parts[0],
                "actual_sha256": actual,
                "size_bytes": source.stat().st_size,
                "matched": actual == parts[0],
            }
        )
    if not rows or not all(row["matched"] for row in rows):
        raise RuntimeError("source immutability baseline mismatch")
    return rows


def decode_video(path: Path, expected_frames: int = 150, expected_fps: float = 30.0) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open encoded video: {path}")
    metadata_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError(f"empty frame {decoded} in {path}")
        decoded += 1
    capture.release()
    passed = bool(
        metadata_frames == expected_frames
        and decoded == expected_frames
        and abs(fps - expected_fps) <= 1.0e-3
    )
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "metadata_frame_count": metadata_frames,
        "decoded_frame_count": decoded,
        "fps": fps,
        "width": width,
        "height": height,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"encoded video QC failed: {report}")
    return report


def audit_p14_vs_p15(case_name: str, run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    p14_path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P14_observed_pose_fit/v18_compact_rigid_object_pose_fit_report.json",
        f"{case_name} P14 report",
    )
    p15_path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json",
        f"{case_name} P15 report",
    )
    p14 = load_json(p14_path)
    p15 = load_json(p15_path)
    p14_rows = {
        int(row["frame_idx"]): row
        for row in p14.get("pose_rows", [])
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    p15_rows = {
        int(row["frame_idx"]): row
        for row in p15.get("pose_rows", [])
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    if sorted(p15_rows) != list(range(150)):
        raise RuntimeError(f"{case_name}: P15 does not contain exactly 150 frames")

    direct_rows: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    generated_pose_rows = 0
    max_rotation_matrix_abs = 0.0
    max_translation_abs = 0.0
    max_rotation_deg = 0.0
    correction_rotation_norms: list[float] = []
    correction_translation_norms: list[float] = []
    completion_sources: collections.Counter[str] = collections.Counter()
    for frame_idx in range(150):
        p15_row = p15_rows[frame_idx]
        temporal = p15_row.get("temporal_pose_graph") if isinstance(
            p15_row.get("temporal_pose_graph"), dict
        ) else {}
        direct = temporal.get("direct_visible_measurement") is True
        p15_pose = pose_arrays(p15_row)
        if p15_pose is None:
            raise RuntimeError(f"{case_name}: P15 frame {frame_idx} lacks pose")
        p14_row = p14_rows.get(frame_idx, {})
        p14_pose = pose_arrays(p14_row)
        if p15_row.get("generated_geometry_pose_evidence_consumed") is True:
            generated_pose_rows += 1
        if direct:
            if p14_pose is None:
                raise RuntimeError(f"{case_name}: direct P15 frame {frame_idx} lacks P14 pose")
            rotation_matrix_abs = float(np.max(np.abs(p15_pose[0] - p14_pose[0])))
            translation_abs = float(np.max(np.abs(p15_pose[1] - p14_pose[1])))
            rotation_deg = rotation_delta_deg(p14_pose[0], p15_pose[0])
            max_rotation_matrix_abs = max(max_rotation_matrix_abs, rotation_matrix_abs)
            max_translation_abs = max(max_translation_abs, translation_abs)
            max_rotation_deg = max(max_rotation_deg, rotation_deg)
            if not np.array_equal(p15_pose[0], p14_pose[0]) or not np.array_equal(
                p15_pose[1], p14_pose[1]
            ):
                raise RuntimeError(f"{case_name}: P15 changed direct P14 SE(3) at frame {frame_idx}")
            rotation_correction = np.asarray(
                temporal.get("rotation_delta_rotvec_rad", [0.0, 0.0, 0.0]), dtype=np.float64
            )
            translation_correction = np.asarray(
                temporal.get("translation_delta_world_m", [0.0, 0.0, 0.0]), dtype=np.float64
            )
            correction_rotation_norms.append(float(np.linalg.norm(rotation_correction)))
            correction_translation_norms.append(float(np.linalg.norm(translation_correction)))
            direct_rows.append(
                {
                    "frame_idx": frame_idx,
                    "rotation_matrix_max_abs": rotation_matrix_abs,
                    "rotation_angle_deg": rotation_deg,
                    "translation_max_abs_m": translation_abs,
                    "rotation_correction_norm_rad": correction_rotation_norms[-1],
                    "translation_correction_norm_m": correction_translation_norms[-1],
                    "direct_pose_observation_source": p15_row.get(
                        "direct_pose_observation_source"
                    ),
                }
            )
        else:
            if p14_pose is not None:
                raise RuntimeError(
                    f"{case_name}: completed P15 frame {frame_idx} backfills a frame with a P14 pose"
                )
            source = str(temporal.get("pose_source") or "missing")
            uncertainty = temporal.get("uncertainty")
            if source not in {
                "interpolated_between_visible_pose_observations",
                "nearest_visible_pose_hold",
            }:
                raise RuntimeError(f"{case_name}: unsupported P15 completion source {source}")
            if not isinstance(uncertainty, str) or not uncertainty:
                raise RuntimeError(f"{case_name}: completed frame {frame_idx} lacks uncertainty")
            if p15_row.get("status") != "completed_temporal_rigid_pose_uncertain":
                raise RuntimeError(f"{case_name}: completed frame {frame_idx} is not marked uncertain")
            completion_sources[source] += 1
            completed_rows.append(
                {
                    "frame_idx": frame_idx,
                    "pose_source": source,
                    "bracket_visible_pose_frames": temporal.get(
                        "bracket_visible_pose_frames"
                    ),
                    "gap_frames": temporal.get("gap_frames"),
                    "uncertainty": uncertainty,
                    "pose_measurement_status": p15_row.get("pose_measurement_status"),
                }
            )

    step_rows: list[dict[str, Any]] = []
    for destination in range(1, 150):
        first = pose_arrays(p15_rows[destination - 1])
        second = pose_arrays(p15_rows[destination])
        assert first is not None and second is not None
        source_completed = not (
            (p15_rows[destination - 1].get("temporal_pose_graph") or {}).get(
                "direct_visible_measurement"
            )
            is True
        )
        destination_completed = not (
            (p15_rows[destination].get("temporal_pose_graph") or {}).get(
                "direct_visible_measurement"
            )
            is True
        )
        step_rows.append(
            {
                "source_frame_idx": destination - 1,
                "destination_frame_idx": destination,
                "source_completed": source_completed,
                "destination_completed": destination_completed,
                "transition_class": (
                    ("C" if source_completed else "D")
                    + ("C" if destination_completed else "D")
                ),
                "touches_completed_pose": source_completed or destination_completed,
                "rotation_step_deg": rotation_delta_deg(first[0], second[0]),
                "translation_step_m": float(np.linalg.norm(second[1] - first[1])),
            }
        )
    boundary_steps = [row for row in step_rows if row["touches_completed_pose"]]
    direct_steps = [row for row in step_rows if not row["touches_completed_pose"]]
    completed_frames = [row["frame_idx"] for row in completed_rows]
    intervals = contiguous_intervals(completed_frames)
    identity_report = {
        "case": case_name,
        "inputs": {
            "p14_report": str(p14_path),
            "p14_report_sha256": sha256_file(p14_path),
            "p15_report": str(p15_path),
            "p15_report_sha256": sha256_file(p15_path),
        },
        "direct_row_count": len(direct_rows),
        "completed_row_count": len(completed_rows),
        "direct_p14_p15_exact_array_identity": True,
        "maximum_direct_rotation_matrix_abs_difference": max_rotation_matrix_abs,
        "maximum_direct_rotation_angle_difference_deg": max_rotation_deg,
        "maximum_direct_translation_abs_difference_m": max_translation_abs,
        "maximum_p15_rotation_correction_norm_rad": max(correction_rotation_norms, default=0.0),
        "maximum_p15_translation_correction_norm_m": max(
            correction_translation_norms, default=0.0
        ),
        "generated_geometry_pose_evidence_row_count": generated_pose_rows,
        "direct_rows": direct_rows,
    }
    completion_report = {
        "case": case_name,
        "completed_frame_count": len(completed_rows),
        "completed_frames": completed_frames,
        "completed_intervals": [
            {
                "first_frame_idx": interval[0],
                "last_frame_idx": interval[-1],
                "frame_count": len(interval),
                "frame_ids": interval,
                "left_direct_boundary": interval[0] > 0
                and interval[0] - 1 not in completed_frames,
                "right_direct_boundary": interval[-1] < 149
                and interval[-1] + 1 not in completed_frames,
            }
            for interval in intervals
        ],
        "completion_source_counts": dict(sorted(completion_sources.items())),
        "all_completed_rows_explicitly_uncertain": True,
        "completed_rows_do_not_backfill_P14": True,
        "completed_rows": completed_rows,
        "boundary_step_rotation_deg": numeric_summary(
            [row["rotation_step_deg"] for row in boundary_steps]
        ),
        "boundary_step_translation_m": numeric_summary(
            [row["translation_step_m"] for row in boundary_steps]
        ),
        "direct_to_direct_step_rotation_deg": numeric_summary(
            [row["rotation_step_deg"] for row in direct_steps]
        ),
        "direct_to_direct_step_translation_m": numeric_summary(
            [row["translation_step_m"] for row in direct_steps]
        ),
        "boundary_steps": boundary_steps,
        "interpretation": (
            "Temporal smoothness and bounded steps do not make a completed row direct or prove pose accuracy."
        ),
    }
    return identity_report, completion_report


def add_stage_completion_diagnostics(
    stage_root: Path, case_name: str, completion_report: dict[str, Any]
) -> dict[str, Any]:
    """Bind encoded-A/B frame diagnostics to the P15 completion audit.

    These appearance-IoU and unsigned vertex-proximity values are diagnostics
    only.  They neither promote completed rows to direct evidence nor establish
    signed contact, penetration, or nonpenetration.
    """
    stage_report_path = require_file(
        stage_root
        / "cases"
        / case_name
        / "P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json",
        f"{case_name} stage A/B report for completion diagnostics",
    )
    stage_report = load_json(stage_report_path)
    raw_rows = stage_report.get("frame_rows")
    if not isinstance(raw_rows, list):
        raise RuntimeError(f"{case_name}: stage A/B report has no frame rows")
    rows = {
        int(row["frame_idx"]): row
        for row in raw_rows
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    if sorted(rows) != list(range(150)) or len(raw_rows) != 150:
        raise RuntimeError(f"{case_name}: stage A/B report must contain exactly 150 frame rows")

    expected_completed = set(int(value) for value in completion_report["completed_frames"])
    frame_diagnostics: list[dict[str, Any]] = []
    by_frame: dict[int, dict[str, Any]] = {}
    for frame_idx in range(150):
        row = rows[frame_idx]
        completed = row.get("p15_completed_pose")
        direct = row.get("p15_direct_visible_measurement")
        if not isinstance(completed, bool) or not isinstance(direct, bool):
            raise RuntimeError(f"{case_name}: malformed completion flags at frame {frame_idx}")
        if completed == direct or completed != (frame_idx in expected_completed):
            raise RuntimeError(
                f"{case_name}: stage/P15 completion classification mismatch at frame {frame_idx}"
            )
        iou = float(row.get("p15_mask_iou"))
        if not math.isfinite(iou) or not 0.0 <= iou <= 1.0:
            raise RuntimeError(f"{case_name}: malformed P15 mask IoU at frame {frame_idx}")

        all_stage_metrics = row.get("unsigned_hand_object_vertex_metrics")
        if not isinstance(all_stage_metrics, dict):
            raise RuntimeError(f"{case_name}: missing unsigned proximity at frame {frame_idx}")
        p15_metrics = all_stage_metrics.get("p15_completed")
        if not isinstance(p15_metrics, dict):
            raise RuntimeError(f"{case_name}: missing P15 unsigned proximity at frame {frame_idx}")
        role = str(p15_metrics.get("metric_role") or "")
        if "unsigned" not in role or "not signed contact" not in role:
            raise RuntimeError(f"{case_name}: unsigned proximity role is not fail-closed")
        hands = p15_metrics.get("hands")
        if not isinstance(hands, dict) or set(hands) != {"left", "right"}:
            raise RuntimeError(f"{case_name}: P15 proximity lacks both MANO hands")
        minima: dict[str, float] = {}
        for side in ("left", "right"):
            hand = hands[side]
            if not isinstance(hand, dict):
                raise RuntimeError(f"{case_name}: malformed {side} proximity at frame {frame_idx}")
            value = float(hand.get("minimum_vertex_distance_m"))
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    f"{case_name}: malformed {side} minimum distance at frame {frame_idx}"
                )
            minima[side] = value
        diagnostic = {
            "frame_idx": frame_idx,
            "completed": completed,
            "direct": direct,
            "p15_appearance_mask_iou": iou,
            "unsigned_minimum_vertex_distance_m": {
                **minima,
                "either_hand": min(minima.values()),
            },
        }
        frame_diagnostics.append(diagnostic)
        by_frame[frame_idx] = diagnostic

    transition_rows: list[dict[str, Any]] = []
    for step in completion_report["boundary_steps"]:
        source = by_frame[int(step["source_frame_idx"])]
        destination = by_frame[int(step["destination_frame_idx"])]
        per_hand_delta = {
            side: float(
                destination["unsigned_minimum_vertex_distance_m"][side]
                - source["unsigned_minimum_vertex_distance_m"][side]
            )
            for side in ("left", "right")
        }
        source_minimum = float(
            source["unsigned_minimum_vertex_distance_m"]["either_hand"]
        )
        destination_minimum = float(
            destination["unsigned_minimum_vertex_distance_m"]["either_hand"]
        )
        iou_delta = float(
            destination["p15_appearance_mask_iou"] - source["p15_appearance_mask_iou"]
        )
        transition_rows.append(
            {
                **step,
                "source_p15_appearance_mask_iou": source["p15_appearance_mask_iou"],
                "destination_p15_appearance_mask_iou": destination[
                    "p15_appearance_mask_iou"
                ],
                "p15_appearance_mask_iou_delta": iou_delta,
                "p15_appearance_mask_iou_absolute_change": abs(iou_delta),
                "source_unsigned_either_hand_minimum_vertex_distance_m": source_minimum,
                "destination_unsigned_either_hand_minimum_vertex_distance_m": (
                    destination_minimum
                ),
                "unsigned_either_hand_minimum_vertex_distance_delta_m": (
                    destination_minimum - source_minimum
                ),
                "unsigned_either_hand_minimum_vertex_distance_absolute_change_m": abs(
                    destination_minimum - source_minimum
                ),
                "unsigned_minimum_vertex_distance_delta_by_hand_m": per_hand_delta,
            }
        )

    transition_summaries: dict[str, Any] = {}
    for transition_class in ("DC", "CC", "CD"):
        selected = [
            row for row in transition_rows if row["transition_class"] == transition_class
        ]
        transition_summaries[transition_class] = {
            "count": len(selected),
            "rotation_step_deg": numeric_summary(
                [row["rotation_step_deg"] for row in selected]
            ),
            "translation_step_m": numeric_summary(
                [row["translation_step_m"] for row in selected]
            ),
            "p15_appearance_mask_iou_delta": numeric_summary(
                [row["p15_appearance_mask_iou_delta"] for row in selected]
            ),
            "p15_appearance_mask_iou_absolute_change": numeric_summary(
                [row["p15_appearance_mask_iou_absolute_change"] for row in selected]
            ),
            "unsigned_either_hand_minimum_vertex_distance_delta_m": numeric_summary(
                [
                    row["unsigned_either_hand_minimum_vertex_distance_delta_m"]
                    for row in selected
                ]
            ),
            "unsigned_either_hand_minimum_vertex_distance_absolute_change_m": numeric_summary(
                [
                    row[
                        "unsigned_either_hand_minimum_vertex_distance_absolute_change_m"
                    ]
                    for row in selected
                ]
            ),
        }

    def summarize_frames(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "frame_count": len(selected),
            "p15_appearance_mask_iou": numeric_summary(
                [row["p15_appearance_mask_iou"] for row in selected]
            ),
            "unsigned_either_hand_minimum_vertex_distance_m": numeric_summary(
                [
                    row["unsigned_minimum_vertex_distance_m"]["either_hand"]
                    for row in selected
                ]
            ),
            "unsigned_left_minimum_vertex_distance_m": numeric_summary(
                [row["unsigned_minimum_vertex_distance_m"]["left"] for row in selected]
            ),
            "unsigned_right_minimum_vertex_distance_m": numeric_summary(
                [row["unsigned_minimum_vertex_distance_m"]["right"] for row in selected]
            ),
        }

    interval_rows: list[dict[str, Any]] = []
    for interval in completion_report["completed_intervals"]:
        first = int(interval["first_frame_idx"])
        last = int(interval["last_frame_idx"])
        entry = next(
            (
                row
                for row in transition_rows
                if row["destination_frame_idx"] == first
                and row["transition_class"] == "DC"
            ),
            None,
        )
        exit_step = next(
            (
                row
                for row in transition_rows
                if row["source_frame_idx"] == last
                and row["transition_class"] == "CD"
            ),
            None,
        )
        internal_steps = [
            row
            for row in transition_rows
            if first <= row["source_frame_idx"]
            and row["destination_frame_idx"] <= last
            and row["transition_class"] == "CC"
        ]
        interval_rows.append(
            {
                **interval,
                "entry_direct_to_completion_step": entry,
                "internal_completion_to_completion_steps": internal_steps,
                "exit_completion_to_direct_step": exit_step,
                "endpoint_hold_without_right_direct_boundary": bool(
                    last == 149 and exit_step is None
                ),
            }
        )

    completed_frame_rows = [row for row in frame_diagnostics if row["completed"]]
    direct_frame_rows = [row for row in frame_diagnostics if row["direct"]]
    completion_report["completed_intervals"] = interval_rows
    completion_report["stage_report_diagnostic_input"] = {
        "path": str(stage_report_path),
        "sha256": sha256_file(stage_report_path),
    }
    completion_report["frame_diagnostic_summaries"] = {
        "completed": summarize_frames(completed_frame_rows),
        "direct": summarize_frames(direct_frame_rows),
    }
    completion_report["completion_transition_summaries"] = transition_summaries
    completion_report["completion_transition_rows"] = transition_rows
    completion_report["frame_diagnostics"] = frame_diagnostics
    completion_report["diagnostic_semantics"] = {
        "appearance_mask_iou_is_metric_pose_evidence": False,
        "unsigned_vertex_proximity_is_signed_contact_evidence": False,
        "unsigned_vertex_proximity_is_penetration_or_nonpenetration_evidence": False,
        "visual_or_numeric_continuity_promotes_completion_to_direct": False,
    }
    return completion_report


def audit_hand_vertices(case_name: str, run_root: Path) -> dict[str, Any]:
    hawor_path = require_file(
        run_root / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz",
        f"{case_name} HaWoR archive",
    )
    bridge_path = require_file(
        run_root / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz",
        f"{case_name} MANO bridge",
    )
    p16_path = require_file(
        run_root
        / "experiments/sam3d_trellis_controlled/P16_unsigned_mano_object/v18_mano_object_constraint_state.json",
        f"{case_name} P16 state",
    )
    side_values: dict[str, list[float]] = {"left": [], "right": []}
    with np.load(hawor_path, allow_pickle=False) as hawor, np.load(
        bridge_path, allow_pickle=False
    ) as bridge:
        frame_ids = bridge["frame_idx"]
        sides = bridge["hand_side"]
        source_indices = bridge["source_frame_index"]
        delivered = bridge[
            "vertices_current_v18_world_from_hawor_projection_relift_m"
        ]
        if delivered.shape != (300, 778, 3):
            raise RuntimeError(f"{case_name}: malformed delivered MANO array {delivered.shape}")
        for row_index, (frame_idx, side, source_idx) in enumerate(
            zip(frame_ids, sides, source_indices)
        ):
            side_name = str(side)
            if side_name not in side_values:
                raise RuntimeError(f"{case_name}: malformed hand side {side_name}")
            raw = hawor[f"{side_name}_vertices_world_m"][int(source_idx)]
            if int(frame_idx) != int(hawor["frame_idx"][int(source_idx)]):
                raise RuntimeError(f"{case_name}: MANO bridge frame map mismatch")
            side_values[side_name].append(
                float(np.max(np.abs(raw.astype(np.float64) - delivered[row_index].astype(np.float64))))
            )
        topology = {
            side: {
                "vertices_per_frame": int(hawor[f"{side}_vertices_world_m"].shape[1]),
                "face_count": int(len(hawor[f"{side}_faces"])),
            }
            for side in ("left", "right")
        }
    p16 = load_json(p16_path)
    constraints = p16.get("constraint_rows") if isinstance(p16.get("constraint_rows"), list) else []
    if len(constraints) != 300:
        raise RuntimeError(f"{case_name}: P16 must contain 300 constraint rows")
    correction_norms: list[float] = []
    application_states: collections.Counter[str] = collections.Counter()
    for row in constraints:
        vector = np.asarray(row.get("candidate_translation_world_m"), dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise RuntimeError(f"{case_name}: malformed P16 candidate translation")
        correction_norms.append(float(np.linalg.norm(vector)))
        application_states[str(row.get("candidate_application_state") or "missing")] += 1
        if row.get("signed_nonpenetration_factor_active") is not False:
            raise RuntimeError(f"{case_name}: P16 unexpectedly activated signed nonpenetration")
    max_vertex_abs = max(max(values) for values in side_values.values())
    max_correction = max(correction_norms)
    if max_vertex_abs != 0.0 or max_correction != 0.0:
        raise RuntimeError(f"{case_name}: hand state changed")
    if p16.get("candidate_correction_count") != 0:
        raise RuntimeError(f"{case_name}: P16 candidate correction count is not zero")
    return {
        "case": case_name,
        "inputs": {
            "hawor_world_hands": str(hawor_path),
            "hawor_world_hands_sha256": sha256_file(hawor_path),
            "delivered_mano_bridge": str(bridge_path),
            "delivered_mano_bridge_sha256": sha256_file(bridge_path),
            "p16_constraint_state": str(p16_path),
            "p16_constraint_state_sha256": sha256_file(p16_path),
        },
        "bridge_row_count": sum(len(values) for values in side_values.values()),
        "side_row_counts": {side: len(values) for side, values in side_values.items()},
        "topology": topology,
        "maximum_raw_hawor_to_delivered_mano_vertex_abs_difference_m": max_vertex_abs,
        "maximum_by_side_m": {
            side: max(values) for side, values in side_values.items()
        },
        "raw_hawor_and_delivered_mano_vertices_exact_array_identity": True,
        "p16_constraint_row_count": len(constraints),
        "p16_candidate_correction_count": int(p16["candidate_correction_count"]),
        "maximum_p16_candidate_translation_norm_m": max_correction,
        "signed_nonpenetration_factor_active": bool(
            p16.get("signed_nonpenetration_factor_active")
        ),
        "p16_application_state_counts": dict(sorted(application_states.items())),
        "optimized_hand_mesh_exists": False,
        "interpretation": (
            "Any hand-placement error visible in this A/B is inherited from HaWoR/MANO/camera alignment; P14/P15 and P16 did not move MANO vertices."
        ),
    }


def build_internal_regularization_report(
    stage_root: Path, run_roots: dict[str, Path]
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_name in CASE_ORDER:
        case_report_path = require_file(
            stage_root
            / "cases"
            / case_name
            / "P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json",
            f"{case_name} stage A/B report",
        )
        report = load_json(case_report_path)
        inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
        p14_path = require_file(inputs.get("p14_report"), f"{case_name} bound P14 report")
        if sha256_file(p14_path) != inputs.get("p14_report_sha256"):
            raise RuntimeError(f"{case_name}: stage report P14 hash mismatch")
        reconstruction = report.get("stage_reconstruction") or {}
        rows = report.get("frame_rows") if isinstance(report.get("frame_rows"), list) else []
        if len(rows) != 150:
            raise RuntimeError(f"{case_name}: stage report lacks 150 frame rows")
        modern = reconstruction.get("pairwise_chain_available") is True
        if modern:
            edge_validation = reconstruction.get("edge_reconstruction_validation") or {}
            if edge_validation.get("passed") is not True:
                raise RuntimeError(f"{case_name}: pairwise edge reconstruction did not pass")
        elif case_name != "milk" or reconstruction.get("legacy_chain_substitution_performed") is not False:
            raise RuntimeError(f"{case_name}: unexpected unavailable pairwise schema")

        p14 = load_json(p14_path)
        p14_rows = {
            int(row["frame_idx"]): row
            for row in p14.get("pose_rows", [])
            if isinstance(row, dict) and row.get("frame_idx") is not None
        }
        annotation_path = require_file(inputs.get("annotations"), f"{case_name} annotations")
        annotations = load_json(annotation_path)
        annotation_frames = {
            int(frame["frame_idx"]): frame
            for frame in annotations.get("frames", [])
            if isinstance(frame, dict) and frame.get("frame_idx") is not None
        }
        camera_delta_rows: list[dict[str, Any]] = []
        for row in rows:
            measurement_raw = row.get("p14_measurement_translation_world_m")
            if measurement_raw is None:
                continue
            frame_idx = int(row["frame_idx"])
            p14_pose = pose_arrays(p14_rows.get(frame_idx, {}))
            if p14_pose is None:
                raise RuntimeError(f"{case_name}: measurement row {frame_idx} lacks P14 pose")
            measurement = np.asarray(measurement_raw, dtype=np.float64)
            delta_world = p14_pose[1] - measurement
            camera = annotation_frames[frame_idx].get("camera") or {}
            transform = np.asarray(
                camera.get("T_world_camera_metric") or camera.get("T_world_camera"),
                dtype=np.float64,
            )
            if transform.shape != (4, 4):
                raise RuntimeError(f"{case_name}: malformed camera at frame {frame_idx}")
            delta_camera = delta_world @ transform[:3, :3]
            camera_delta_rows.append(
                {
                    "frame_idx": frame_idx,
                    "measurement_to_final_delta_world_m": delta_world.tolist(),
                    "measurement_to_final_delta_camera_xyz_m": delta_camera.tolist(),
                    "norm_m": float(np.linalg.norm(delta_world)),
                    "camera_z_abs_m": float(abs(delta_camera[2])),
                }
            )
        largest_camera_z = (
            max(camera_delta_rows, key=lambda row: row["camera_z_abs_m"])
            if camera_delta_rows
            else None
        )
        numeric = report.get("numeric_comparison") or {}
        cases.append(
            {
                "case": case_name,
                "schema_mode": reconstruction.get("mode"),
                "pairwise_chain_available": modern,
                "legacy_chain_substitution_performed": reconstruction.get(
                    "legacy_chain_substitution_performed"
                ),
                "edge_reconstruction_validation": reconstruction.get(
                    "edge_reconstruction_validation"
                ),
                "stage_summaries": {
                    key: numeric.get(key)
                    for key in (
                        "chain_to_p14_rotation_deg",
                        "chain_to_p14_translation_m",
                        "preliminary_chain_translation_stabilization_m",
                        "post_rotation_anchor_translation_fit_m",
                        "measurement_to_p14_temporal_translation_regularization_m",
                        "preliminary_translation_stabilization_residual_delta_m",
                        "regularized_rotation_residual_delta_m",
                        "post_rotation_translation_fit_residual_delta_m",
                        "temporal_translation_regularization_residual_delta_m",
                        "chain_to_p14_observed_surface_median_delta_m",
                        "chain_to_p14_mask_iou_delta",
                    )
                },
                "fraction_direct_metric_frames_with_lower_final_p14_median_residual": numeric.get(
                    "fraction_direct_metric_frames_with_lower_p14_median_residual"
                ),
                "fraction_comparable_frames_with_higher_final_p14_mask_iou": numeric.get(
                    "fraction_comparable_frames_with_higher_p14_mask_iou"
                ),
                "extrema": {
                    "largest_chain_to_p14_rotation_deg": extrema(
                        rows, "chain_to_p14_rotation_deg", "max"
                    ),
                    "largest_chain_to_p14_translation_m": extrema(
                        rows, "chain_to_p14_translation_m", "max"
                    ),
                    "largest_temporal_translation_shift_m": extrema(
                        rows,
                        "measurement_to_p14_temporal_translation_regularization_m",
                        "max",
                    ),
                    "largest_temporal_translation_residual_degradation_m": extrema(
                        rows, "temporal_translation_regularization_residual_delta_m", "max"
                    ),
                    "largest_net_surface_residual_improvement_m": extrema(
                        rows, "chain_to_p14_observed_surface_median_delta_m", "min"
                    ),
                    "largest_net_surface_residual_degradation_m": extrema(
                        rows, "chain_to_p14_observed_surface_median_delta_m", "max"
                    ),
                    "largest_camera_z_component_of_temporal_translation": largest_camera_z,
                },
                "camera_space_temporal_translation_rows": camera_delta_rows,
                "source_report": str(case_report_path),
                "source_report_sha256": sha256_file(case_report_path),
            }
        )
    return {
        "schema": "hot3d_p14_internal_regularization_delta_v1",
        "status": "mechanically_verified",
        "claim_scope": (
            "Exact saved-stage reconstruction where modern fields exist; Milk remains explicitly unavailable. Negative residual delta is improvement."
        ),
        "translation_naming": {
            "p14_measurement_translation_world_m": (
                "pre-temporal-regularization measurement after any preliminary stabilization and fixed-rotation observed-anchor translation fits"
            ),
            "raw_pairwise_chain_translation": (
                "reconstructed by subtracting both saved translation-fit traces; never inferred from P15"
            ),
        },
        "mask_iou_is_pose_evidence": False,
        "generated_geometry_consumed": False,
        "GT_consumed": False,
        "cases": cases,
    }


def audit_glbs(stage_root: Path) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    total_available = 0
    total_unavailable = 0
    direct_identity_count = 0
    loaded_count = 0
    for case_name in CASE_ORDER:
        index_path = require_file(
            stage_root / "glb" / case_name / "KEYFRAME_STAGE_GLB_INDEX.json",
            f"{case_name} GLB index",
        )
        index = load_json(index_path)
        entries = index.get("entries") if isinstance(index.get("entries"), list) else []
        by_frame_stage = {
            (int(row["frame_idx"]), str(row["stage"])): row for row in entries
        }
        loaded_rows: list[dict[str, Any]] = []
        for row in entries:
            if row.get("available") is not True:
                total_unavailable += 1
                continue
            path = require_file(row.get("path"), f"{case_name} stage GLB")
            require_inside(path, stage_root, f"{case_name} GLB")
            if sha256_file(path) != row.get("sha256"):
                raise RuntimeError(f"{case_name}: GLB hash mismatch {path}")
            with path.open("rb") as handle:
                if handle.read(4) != b"glTF":
                    raise RuntimeError(f"{case_name}: invalid GLB magic {path}")
            scene = trimesh.load(path, process=False)
            if not isinstance(scene, trimesh.Scene):
                raise RuntimeError(f"{case_name}: keyframe GLB is not a scene {path}")
            expected = {
                "00_observed_metric_surface_overlay",
                "01_mano_left_full_surface",
                "02_mano_right_full_surface",
            }
            if set(scene.geometry) != expected:
                raise RuntimeError(f"{case_name}: GLB layer mismatch {path}")
            left = scene.geometry["01_mano_left_full_surface"]
            right = scene.geometry["02_mano_right_full_surface"]
            if (len(left.vertices), len(left.faces), len(right.vertices), len(right.faces)) != (
                778,
                1538,
                778,
                1538,
            ):
                raise RuntimeError(f"{case_name}: GLB MANO topology mismatch {path}")
            if row.get("generated_geometry_loaded_or_exported") is not False:
                raise RuntimeError(f"{case_name}: GLB exported generated geometry")
            loaded_rows.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "stage": str(row["stage"]),
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "geometry_names": sorted(scene.geometry),
                    "object_vertices": int(
                        len(scene.geometry["00_observed_metric_surface_overlay"].vertices)
                    ),
                    "object_faces": int(
                        len(scene.geometry["00_observed_metric_surface_overlay"].faces)
                    ),
                }
            )
            total_available += 1
            loaded_count += 1
        for frame_idx in index.get("selected_frame_ids", []):
            p14 = by_frame_stage.get((int(frame_idx), "p14_regularized"))
            p15 = by_frame_stage.get((int(frame_idx), "p15_completed"))
            if (
                isinstance(p15, dict)
                and p15.get("copied_from_exact_direct_p14_glb") is not None
            ):
                if not isinstance(p14, dict) or p14.get("available") is not True:
                    raise RuntimeError(f"{case_name}: copied P15 GLB lacks P14 source")
                if p14.get("sha256") != p15.get("sha256"):
                    raise RuntimeError(f"{case_name}: direct P14/P15 GLBs differ")
                direct_identity_count += 1
        case_rows.append(
            {
                "case": case_name,
                "index": str(index_path),
                "index_sha256": sha256_file(index_path),
                "selected_frame_count": len(index.get("selected_frame_ids", [])),
                "declared_entry_count": len(entries),
                "available_glb_count": sum(row.get("available") is True for row in entries),
                "unavailable_stage_count": sum(row.get("available") is not True for row in entries),
                "loaded_rows": loaded_rows,
            }
        )
    return {
        "schema": "hot3d_p14_stage_keyframe_glb_qc_v1",
        "status": "passed",
        "case_count": len(case_rows),
        "available_glb_count": total_available,
        "unavailable_stage_count": total_unavailable,
        "successfully_reloaded_glb_count": loaded_count,
        "byte_identical_direct_p14_p15_glb_pair_count": direct_identity_count,
        "full_mano_topology_required": {"vertices_per_hand": 778, "faces_per_hand": 1538},
        "generated_geometry_loaded_or_exported": False,
        "cases": case_rows,
    }


def collect_selected_deliverables(root: Path, excluded_names: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name in excluded_names:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_INTERMEDIATE_DIR_NAMES for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def main() -> None:
    args = parse_args()
    collection_root = args.collection_root.expanduser().resolve(strict=True)
    stage_root = args.stage_ab_root.expanduser().resolve(strict=True)
    manual_path = require_file(args.manual_visual_review, "manual image-read review")
    require_inside(manual_path, stage_root, "manual review")
    baseline_path = require_file(args.source_hash_baseline, "pre-render source hash baseline")
    aggregate_path = require_file(
        stage_root / "COLLECTION_P14_STAGE_AB_REPORT.json", "collection stage A/B report"
    )
    encoded_qa_path = require_file(
        stage_root
        / "encoded_video_object_centric_qa/ENCODED_OBJECT_CENTRIC_QA_REPORT.json",
        "encoded object-centric QA report",
    )
    aggregate = load_json(aggregate_path)
    encoded_qa = load_json(encoded_qa_path)
    manual = load_json(manual_path)
    if manual.get("status") != "image_read_review_complete":
        raise RuntimeError("manual visual review is not complete")
    if set((manual.get("cases") or {}).keys()) != set(CASE_ORDER):
        raise RuntimeError("manual visual review must contain all five cases")
    if aggregate.get("global_contract", {}).get("finalized_case_roots_modified") is not False:
        raise RuntimeError("stage report does not preserve finalized-root immutability")
    if aggregate.get("global_contract", {}).get("generated_geometry_loaded_or_rendered") is not False:
        raise RuntimeError("stage report loaded generated geometry")

    final_paths = [
        stage_root / "numeric/p14_internal_regularization_delta.json",
        stage_root / "numeric/p14_vs_p15_exact_se3_delta.json",
        stage_root / "numeric/hand_vertex_delta.json",
        stage_root / "numeric/interpolation_frame_audit.json",
        stage_root / "numeric/source_immutability_audit.json",
        stage_root / "numeric/artifact_qc.json",
        stage_root / "FINAL_REVIEW_REPORT.json",
        stage_root / "FINAL_REVIEW_REPORT_ZH.md",
        stage_root / "FINAL_ARTIFACT_SHA256_INDEX.json",
        stage_root / "FINAL_REVIEW_DONE.json",
    ]
    if not args.replace_final_reports:
        existing = [path for path in final_paths if path.exists()]
        if existing:
            raise RuntimeError(f"refusing to overwrite final reports: {existing}")

    run_roots: dict[str, Path] = {}
    for case_name in CASE_ORDER:
        link = collection_root / "runs" / case_name
        if not link.is_symlink():
            raise RuntimeError(f"collection run is not a symlink: {link}")
        run_roots[case_name] = link.resolve(strict=True)

    internal = build_internal_regularization_report(stage_root, run_roots)
    identity_cases: list[dict[str, Any]] = []
    completion_cases: list[dict[str, Any]] = []
    hand_cases: list[dict[str, Any]] = []
    for case_name in CASE_ORDER:
        identity, completion = audit_p14_vs_p15(case_name, run_roots[case_name])
        identity_cases.append(identity)
        completion_cases.append(
            add_stage_completion_diagnostics(stage_root, case_name, completion)
        )
        hand_cases.append(audit_hand_vertices(case_name, run_roots[case_name]))
    identity_report = {
        "schema": "hot3d_p14_vs_p15_exact_se3_delta_v1",
        "status": "passed",
        "case_count": 5,
        "total_direct_row_count": sum(row["direct_row_count"] for row in identity_cases),
        "total_completed_row_count": sum(row["completed_row_count"] for row in identity_cases),
        "all_direct_rows_exact_array_identity": all(
            row["direct_p14_p15_exact_array_identity"] for row in identity_cases
        ),
        "physical_trajectory_smoothed_directly_by_P15": False,
        "cases": identity_cases,
    }
    completion_report = {
        "schema": "hot3d_p15_interpolation_frame_audit_v1",
        "status": "passed_with_completion_remaining_uncertain",
        "case_count": 5,
        "total_completed_frame_count": sum(
            row["completed_frame_count"] for row in completion_cases
        ),
        "completion_does_not_modify_direct_P14_rows": True,
        "completed_rows_are_not_direct_pose_evidence": True,
        "cases": completion_cases,
    }
    hand_report = {
        "schema": "hot3d_hand_vertex_delta_v1",
        "status": "passed_exact_identity_no_P16_correction",
        "case_count": 5,
        "total_hand_frame_rows": sum(row["bridge_row_count"] for row in hand_cases),
        "global_maximum_raw_hawor_to_delivered_mano_vertex_abs_difference_m": max(
            row["maximum_raw_hawor_to_delivered_mano_vertex_abs_difference_m"]
            for row in hand_cases
        ),
        "global_maximum_P16_candidate_translation_norm_m": max(
            row["maximum_p16_candidate_translation_norm_m"] for row in hand_cases
        ),
        "cases": hand_cases,
    }
    source_rows = parse_hash_baseline(baseline_path)
    source_report = {
        "schema": "hot3d_p14_stage_ab_source_immutability_audit_v1",
        "status": "passed",
        "baseline_file": str(baseline_path),
        "baseline_file_sha256": sha256_file(baseline_path),
        "entry_count": len(source_rows),
        "mismatch_count": 0,
        "finalized_case_roots_modified": False,
        "rows": source_rows,
    }

    video_rows: list[dict[str, Any]] = []
    encoded_cases = {
        str(row["case"]): row
        for row in encoded_qa.get("cases", [])
        if isinstance(row, dict) and row.get("case") is not None
    }
    for case_name in CASE_ORDER:
        case_report = load_json(
            stage_root
            / "cases"
            / case_name
            / "P14_PAIRWISE_REGULARIZED_P15_AB_REPORT.json"
        )
        for view in ("camera", "world", "side"):
            video_rows.append(
                decode_video(require_file(case_report["outputs"][f"{view}_video"], f"{case_name} {view} video"))
            )
        video_rows.append(
            decode_video(
                require_file(
                    encoded_cases[case_name]["outputs"]["zoom_video"],
                    f"{case_name} encoded zoom video",
                )
            )
        )
    glb_report = audit_glbs(stage_root)
    reviewed_paths: list[dict[str, Any]] = []
    for row in manual.get("reviewed_artifacts", []):
        if not isinstance(row, dict) or row.get("path") is None:
            raise RuntimeError("malformed manual reviewed_artifacts row")
        path = require_file(row["path"], "manual reviewed image")
        require_inside(path, stage_root, "manual reviewed image")
        reviewed_paths.append(
            {
                **row,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if len(reviewed_paths) < 15:
        raise RuntimeError("manual review must bind at least 15 image artifacts")
    artifact_qc = {
        "schema": "hot3d_p14_stage_ab_artifact_qc_v1",
        "status": "passed",
        "video_count": len(video_rows),
        "decoded_video_frame_count": sum(row["decoded_frame_count"] for row in video_rows),
        "all_videos_150_frames_30_fps": all(row["passed"] for row in video_rows),
        "videos": video_rows,
        "glb_qc": glb_report,
        "manual_image_read_review": {
            "path": str(manual_path),
            "sha256": sha256_file(manual_path),
            "reviewed_artifact_count": len(reviewed_paths),
            "reviewed_artifacts": reviewed_paths,
        },
    }

    numeric_dir = stage_root / "numeric"
    numeric_dir.mkdir(parents=True, exist_ok=True)
    write_json(final_paths[0], internal)
    write_json(final_paths[1], identity_report)
    write_json(final_paths[2], hand_report)
    write_json(final_paths[3], completion_report)
    write_json(final_paths[4], source_report)
    write_json(final_paths[5], artifact_qc)

    conclusions = manual.get("global_conclusions") or {}
    final_report = {
        "schema": SCHEMA,
        "status": "complete_with_p14_temporal_translation_outliers_and_milk_schema_limit",
        "claim_scope": (
            "Read-only causal A/B of saved object-pose stages. This report does not modify or replace finalized D19 outputs or the five-case collection."
        ),
        "collection_root": str(collection_root),
        "stage_ab_root": str(stage_root),
        "diagnostic_code_commit": str(args.diagnostic_code_commit),
        "finalizer": str(Path(__file__).resolve()),
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        "mechanical_findings": {
            "P14_P15_direct_SE3_exact_identity": True,
            "P15_completed_frame_count": identity_report["total_completed_row_count"],
            "raw_HaWoR_to_delivered_MANO_global_max_vertex_delta_m": hand_report[
                "global_maximum_raw_hawor_to_delivered_mano_vertex_abs_difference_m"
            ],
            "P16_global_max_candidate_translation_norm_m": hand_report[
                "global_maximum_P16_candidate_translation_norm_m"
            ],
            "encoded_video_count": artifact_qc["video_count"],
            "decoded_video_frame_count": artifact_qc["decoded_video_frame_count"],
            "successfully_reloaded_keyframe_glb_count": glb_report[
                "successfully_reloaded_glb_count"
            ],
            "source_hash_mismatch_count": source_report["mismatch_count"],
        },
        "image_read_findings": conclusions,
        "case_image_read_verdicts": manual["cases"],
        "causal_attribution": {
            "direct_frame_error_can_be_caused_by_P15_completion": False,
            "completed_frame_error_can_be_caused_by_P15_completion": True,
            "P15_completion_accuracy_proven_by_visual_continuity": False,
            "P14_internal_regularization_is_a_direct_frame_pose_variable": True,
            "dominant_numeric_direct_frame_concern": (
                "final P14 temporal translation regularizer; preceding preliminary translation, unary rotation, and post-rotation translation stages usually reduce median accepted-surface residual"
            ),
            "hand_pose_changed_by_P14_P15_or_P16": False,
            "generated_shape_can_explain_this_A_B": False,
            "milk_P14_internal_attribution_available": False,
        },
        "limitations": [
            "No GT pose, GT MANO, CAD, GT mask, or generated hidden geometry was consumed.",
            "Appearance-mask IoU is diagnostic only and is not metric pose evidence.",
            "Observed surfaces are partial and unsigned; proximity is not signed contact or penetration evidence.",
            "Visual continuity of P15 interpolation does not establish direct pose accuracy.",
            "Milk's old P14 schema did not preserve pairwise/correction traces; no modern replay was substituted.",
        ],
        "reports": {
            "p14_internal_regularization_delta": str(final_paths[0]),
            "p14_vs_p15_exact_se3_delta": str(final_paths[1]),
            "hand_vertex_delta": str(final_paths[2]),
            "interpolation_frame_audit": str(final_paths[3]),
            "source_immutability_audit": str(final_paths[4]),
            "artifact_qc": str(final_paths[5]),
            "manual_visual_review": str(manual_path),
            "collection_stage_report": str(aggregate_path),
            "encoded_visual_qa": str(encoded_qa_path),
        },
        "immutability": {
            "finalized_case_roots_modified": False,
            "collection_modified": False,
            "diagnostic_is_post_processing_only": True,
        },
    }
    write_json(final_paths[6], final_report)

    markdown_lines = [
        "# P14 pairwise chain → P14 regularized → P15 completed 最终验证结论",
        "",
        f"状态：`{final_report['status']}`",
        "",
        "## 核心归因",
        "",
        "- **P15 没有修改任何 P14 direct pose。** 五例所有 direct rows 的 rotation matrix 与 translation 均逐元素完全相同；P15 的数值修正也为零。",
        f"- **P15 只新增 {identity_report['total_completed_row_count']} 个 uncertain completion rows。** 它们只能解释原本缺测帧，不能解释 direct 帧偏差。",
        "- **现代四例的主要 direct-frame 风险来自 P14 最后的 temporal translation regularizer。** 前置 translation stabilization、rotation unary 和 post-rotation translation fit 的 median accepted-surface residual 通常下降；最后 temporal translation stage 的 median residual 在四例都上升。",
        "- **手没有被优化或移动。** 原始 HaWoR 与正式 MANO bridge 的 150×2×778 顶点逐元素一致，P16 的 150×2 corrections 全为零。若三栏中的手持续偏，原因应回到 HaWoR/MANO、相机主点、遮挡或 handedness，而不是 P14/P15/P16。",
        "- **本 A/B 不含 generated geometry。** 因此这里看到的差异不是 SAM3D/TRELLIS hidden-shape underlay 造成的。",
        "",
        "## Case 结论",
        "",
    ]
    for case_name in CASE_ORDER:
        verdict = manual["cases"][case_name]
        markdown_lines.extend(
            [
                f"### {case_name}",
                "",
                str(verdict.get("summary_zh") or verdict.get("summary") or "见 JSON 人工审阅记录。"),
                "",
            ]
        )
    markdown_lines.extend(
        [
            "## 证据边界",
            "",
            "- P15 completed 区间在已编码视频中视觉连续且未见单帧瞬移/180° flip；这只是 continuity review，不等于 direct accuracy。",
            "- 2D overlay 会低估 camera-z/world-z 平移；因此同时审阅了 world、side 和 full-topology GLB。",
            "- Milk 缺失现代 P14 trace，pairwise 栏保持 N/A；没有重跑新算法后冒充旧 stage。",
            "- unsigned hand-object vertex distance 不是 signed contact、penetration 或 nonpenetration。",
            "",
            "## 入口",
            "",
            "- `videos/`：camera/world/side/zoom，共 20 个完整 150-frame、30 FPS 视频。",
            "- `glb/<case>/`：关键帧 full-topology object + 左右 MANO stage snapshots。",
            "- `encoded_video_object_centric_qa/`：最终编码视频 extrema 与 P15 completion neighborhood sheets。",
            "- `numeric/`：P14 内部、P14↔P15、手顶点、interpolation、immutability 和 artifact QC。",
            "- `FINAL_REVIEW_REPORT.json`：结构化最终结论。",
            "",
        ]
    )
    final_paths[7].write_text("\n".join(markdown_lines), encoding="utf-8")

    excluded = {final_paths[8].name, final_paths[9].name}
    selected_files = collect_selected_deliverables(stage_root, excluded)
    hash_rows = [
        {
            "path": str(path.relative_to(stage_root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in selected_files
    ]
    hash_index = {
        "schema": "hot3d_p14_stage_ab_final_artifact_sha256_index_v1",
        "scope": (
            "All selected reports, videos, review sheets, and keyframe GLBs; per-frame camera/world/side/zoom JPEG intermediates excluded."
        ),
        "entry_count": len(hash_rows),
        "excluded_intermediate_directory_names": sorted(EXCLUDED_INTERMEDIATE_DIR_NAMES),
        "entries": hash_rows,
    }
    write_json(final_paths[8], hash_index)
    done = {
        "schema": SCHEMA,
        "status": final_report["status"],
        "visual_review_pending": False,
        "manual_image_read_review_complete": True,
        "case_count": 5,
        "video_count": artifact_qc["video_count"],
        "decoded_video_frame_count": artifact_qc["decoded_video_frame_count"],
        "keyframe_glb_count": glb_report["available_glb_count"],
        "finalized_case_roots_modified": False,
        "collection_modified": False,
        "diagnostic_code_commit": str(args.diagnostic_code_commit),
        "final_review_report": str(final_paths[6]),
        "final_review_report_sha256": sha256_file(final_paths[6]),
        "final_review_report_zh": str(final_paths[7]),
        "final_review_report_zh_sha256": sha256_file(final_paths[7]),
        "final_artifact_sha256_index": str(final_paths[8]),
        "final_artifact_sha256_index_sha256": sha256_file(final_paths[8]),
        "source_immutability_audit_sha256": sha256_file(final_paths[4]),
        "artifact_qc_sha256": sha256_file(final_paths[5]),
    }
    write_json(final_paths[9], done)
    print(json.dumps(done, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
