#!/usr/bin/env python3
"""Audit and optionally prepare the curated Ego-Exo4D rigid multi-clip suite.

This is benchmark curation/evaluation-side code. It may read relation masks,
hand GT, and camera GT, but writes them only under a benchmark root that is
physically separate from prediction inputs. Each prediction case directory
contains only input.mp4 and PREDICTION_INPUT_MANIFEST.json.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prepare_benchmark import bbox_xyxy, decode_egoexo_mask, load_json, prepare, sha256_file, write_json
from self_test_coordinate_contract import run_test as run_coordinate_contract_test


def ffprobe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def select_aria_stream(target_streams: dict[str, Any]) -> str:
    candidates = [name for name in target_streams if name.startswith("aria") and name.endswith("_214-1")]
    if not candidates:
        raise RuntimeError(f"target has no Aria RGB relation-mask stream: {sorted(target_streams)}")
    return max(candidates, key=lambda name: len(target_streams[name].get("annotation", {})))


def choose_camera_id(camera_payload: dict[str, Any], stream: str) -> str:
    expected = stream.removesuffix("_214-1")
    if expected in camera_payload:
        return expected
    candidates = [name for name in camera_payload if name.startswith("aria")]
    if len(candidates) != 1:
        raise RuntimeError(f"cannot map stream {stream} to camera payload: {candidates}")
    return candidates[0]


def selected_person_joint_count(people: list[dict[str, Any]]) -> int:
    return max((len(person.get("annotation3D", {})) for person in people), default=0)


def temporal_bins(frames: list[int], start: int, frame_count: int, bin_count: int = 5) -> list[int]:
    bins = {
        min(bin_count - 1, max(0, (frame - start) * bin_count // frame_count))
        for frame in frames
        if start <= frame < start + frame_count
    }
    return sorted(bins)


def audit_masks(
    annotations: dict[str, Any],
    source_start: int,
    frame_count: int,
    required_local_frames: list[int],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    required_source = [source_start + frame for frame in required_local_frames]
    missing = [frame for frame in required_source if str(frame) not in annotations]
    if missing:
        raise RuntimeError(f"missing required sparse object masks at source frames {missing}")
    decoded: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    centers: list[list[float]] = []
    for local_frame, source_frame in zip(required_local_frames, required_source, strict=True):
        mask = decode_egoexo_mask(annotations[str(source_frame)])
        decoded[source_frame] = mask
        ys, xs = np.where(mask > 0)
        if not len(xs):
            raise RuntimeError(f"empty target mask at source frame {source_frame}")
        bbox = bbox_xyxy(mask)
        assert bbox is not None
        center = [float(xs.mean()), float(ys.mean())]
        centers.append(center)
        rows.append(
            {
                "local_frame_idx": local_frame,
                "source_frame_idx": source_frame,
                "source_grid": [int(mask.shape[1]), int(mask.shape[0])],
                "area_px": int(mask.sum()),
                "area_fraction": float(mask.mean()),
                "bbox_xyxy": bbox,
                "centroid_xy_px": center,
                "touches_image_border": bool(
                    bbox[0] <= 1
                    or bbox[1] <= 1
                    or bbox[2] >= mask.shape[1] - 1
                    or bbox[3] >= mask.shape[0] - 1
                ),
            }
        )
    center_array = np.asarray(centers, dtype=np.float64)
    grid_diagonal = math.sqrt(2.0) * float(rows[0]["source_grid"][0])
    center_range = float(np.linalg.norm(center_array.max(axis=0) - center_array.min(axis=0)))
    return (
        {
            "claim_scope": "selection diagnostic from five sparse visible masks; not object motion or pose GT",
            "required_source_frames": required_source,
            "available_track_mask_frame_count": len(annotations),
            "mean_area_fraction": float(np.mean([row["area_fraction"] for row in rows])),
            "min_area_fraction": float(np.min([row["area_fraction"] for row in rows])),
            "max_area_fraction": float(np.max([row["area_fraction"] for row in rows])),
            "centroid_range_px": center_range,
            "centroid_range_normalized_by_grid_diagonal": center_range / grid_diagonal,
            "border_touch_frame_count": sum(row["touches_image_border"] for row in rows),
            "frames": rows,
        },
        decoded,
    )


def render_review_sheet(
    video_path: Path,
    case: dict[str, Any],
    stream: str,
    masks: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    panels_raw: list[np.ndarray] = []
    panels_mask: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(video_path))
    for source_frame in sorted(masks):
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read {video_path} frame {source_frame}")
        mask = cv2.resize(masks[source_frame], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        overlay = image.copy()
        red = image.copy()
        red[mask] = (20, 20, 240)
        overlay = cv2.addWeighted(overlay, 0.45, red, 0.55, 0.0)
        for target, label in ((panels_raw, "raw"), (panels_mask, "evaluation-only visible Mask")):
            panel = image.copy() if label == "raw" else overlay.copy()
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"{label} source={source_frame}",
                (5, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            target.append(cv2.resize(panel, (320, 320), interpolation=cv2.INTER_AREA))
    capture.release()
    body = np.concatenate(
        [np.concatenate(panels_raw, axis=1), np.concatenate(panels_mask, axis=1)],
        axis=0,
    )
    header = np.zeros((82, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{case['role']} | {case['take_name']} | {case['target_track_evaluation_only']}",
        (8, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"{case['source_frame_start']}-{case['source_frame_start'] + 149} | stream={stream}",
        (8, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate([header, body], axis=0), [cv2.IMWRITE_JPEG_QUALITY, 93])


def audit_case(
    case: dict[str, Any],
    contract: dict[str, Any],
    dataset_root: Path,
    take: dict[str, Any],
    relation: dict[str, Any],
    sha_cache: dict[Path, str],
    review_root: Path | None,
) -> tuple[dict[str, Any], str]:
    frame_count = int(contract["frame_count"])
    source_start = int(case["source_frame_start"])
    source_end = source_start + frame_count - 1
    target_track = str(case["target_track_evaluation_only"])
    if target_track not in relation.get("object_masks", {}):
        raise RuntimeError(f"{case['case_id']}: target track not found: {target_track}")
    stream = select_aria_stream(relation["object_masks"][target_track])
    target_annotations = relation["object_masks"][target_track][stream]["annotation"]
    mask_audit, decoded_masks = audit_masks(
        target_annotations,
        source_start,
        frame_count,
        [int(value) for value in contract["required_sparse_mask_local_frames"]],
    )

    hand_path = dataset_root / "annotations" / "ego_pose" / case["split"] / "hand" / "annotation" / f"{case['take_uid']}.json"
    camera_path = dataset_root / "annotations" / "ego_pose" / case["split"] / "camera_pose" / f"{case['take_uid']}.json"
    if not hand_path.exists() or not camera_path.exists():
        raise RuntimeError(f"{case['case_id']}: hand/camera GT is missing")
    hand_payload = load_json(hand_path)
    hand_counts = {
        int(frame): selected_person_joint_count(people)
        for frame, people in hand_payload.items()
        if source_start <= int(frame) <= source_end
    }
    frames_any = sorted(frame for frame, count in hand_counts.items() if count > 0)
    frames_15 = sorted(frame for frame, count in hand_counts.items() if count >= 15)
    hand_audit = {
        "claim_scope": "released sparse named 3-D joints; selection coverage only, not MANO/contact GT",
        "annotated_frame_count": len(frames_any),
        "frames_with_at_least_15_selected_person_3d_joints": len(frames_15),
        "occupied_temporal_bins_any_joints": temporal_bins(frames_any, source_start, frame_count),
        "occupied_temporal_bins_15plus_joints": temporal_bins(frames_15, source_start, frame_count),
        "first_annotated_source_frame": min(frames_any) if frames_any else None,
        "last_annotated_source_frame": max(frames_any) if frames_any else None,
    }

    camera_payload = load_json(camera_path)
    camera_id = choose_camera_id(camera_payload, stream)
    extrinsic_frames = set(map(int, camera_payload[camera_id]["camera_extrinsics"]))
    missing_camera_frames = [frame for frame in range(source_start, source_end + 1) if frame not in extrinsic_frames]
    if missing_camera_frames:
        raise RuntimeError(f"{case['case_id']}: {len(missing_camera_frames)} camera frames missing")

    source_video = dataset_root / take["root_dir"] / "frame_aligned_videos" / "downscaled" / "448" / f"{stream}.mp4"
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    probe = ffprobe_video(source_video)
    source_frames = int(probe.get("nb_frames") or 0)
    if source_frames and source_end >= source_frames:
        raise RuntimeError(f"{case['case_id']}: source interval exceeds video length {source_frames}")
    vrs_relative = take.get("vrs_relative_path")
    local_vrs = dataset_root / take["root_dir"] / str(vrs_relative) if vrs_relative else None

    def cached_sha(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in sha_cache:
            sha_cache[resolved] = sha256_file(resolved)
        return sha_cache[resolved]

    review_path = review_root / f"{case['case_id']}.jpg" if review_root is not None else None
    if review_path is not None:
        render_review_sheet(source_video, case, stream, decoded_masks, review_path)
    audit = {
        "case_id": case["case_id"],
        "evaluation_label": case["evaluation_label"],
        "role": case["role"],
        "split": case["split"],
        "take_uid": case["take_uid"],
        "take_name": case["take_name"],
        "task_name": take.get("task_name"),
        "participant_uid": take.get("participant_uid"),
        "capture_uid": take.get("capture_uid"),
        "university_name": take.get("university_name"),
        "target_track_evaluation_only": target_track,
        "target_hint_prediction_side": case["target_hint_prediction_side"],
        "runtime_object_id": case["runtime_object_id"],
        "source_interval": [source_start, source_end],
        "source_stream": stream,
        "source_video": str(source_video),
        "source_video_sha256": cached_sha(source_video),
        "source_video_ffprobe": probe,
        "camera": {
            "camera_id": camera_id,
            "complete_extrinsic_frame_count": frame_count,
            "missing_extrinsic_frame_count": 0,
            "coordinate_scope": "official rectified annotation camera; raw view remains uncalibrated without VRS",
        },
        "hand_gt_coverage": hand_audit,
        "visible_mask_gt": mask_audit,
        "calibration": {
            "metadata_has_trimmed_vrs": bool(take.get("has_trimmed_vrs")),
            "metadata_vrs_relative_path": vrs_relative,
            "local_vrs_exists": bool(local_vrs is not None and local_vrs.exists()),
            "local_vrs_path": str(local_vrs) if local_vrs is not None else None,
        },
        "selection": {
            "strata": case["strata"],
            "rigidity_assumption": case["rigidity_assumption"],
            "reason": case["selection_reason"],
            "curator_review": "raw RGB plus five evaluation-only visible masks inspected",
            "review_sheet_evaluation_only": str(review_path) if review_path is not None else None,
        },
        "annotation_sha256": {
            "hand": cached_sha(hand_path),
            "camera": cached_sha(camera_path),
        },
        "status": "selected_case_contract_valid",
    }
    return audit, stream


def preparation_args(
    case: dict[str, Any],
    dataset_root: Path,
    prediction_root: Path,
    benchmark_root: Path,
) -> Namespace:
    return Namespace(
        dataset_root=dataset_root,
        split=case["split"],
        take_uid=case["take_uid"],
        source_frame_start=int(case["source_frame_start"]),
        frame_count=150,
        target_track=case["target_track_evaluation_only"],
        target_description=case["target_hint_prediction_side"],
        rigidity_assumption=case["rigidity_assumption"],
        case_id=case["case_id"],
        output_size=960,
        input_dir=prediction_root / case["case_id"],
        ground_truth_dir=benchmark_root / case["case_id"] / "ground_truth_v2_rectified_camera",
        replace=True,
    )


def validate_prepared_frame_alignment(
    source_video: Path,
    prepared_video: Path,
    source_start: int,
    frame_count: int,
) -> dict[str, Any]:
    source = cv2.VideoCapture(str(source_video))
    prepared = cv2.VideoCapture(str(prepared_video))
    checks: list[dict[str, Any]] = []
    for local_frame, expected_source_frame in ((0, source_start), (frame_count - 1, source_start + frame_count - 1)):
        prepared.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
        ok, prepared_image = prepared.read()
        if not ok:
            raise RuntimeError(f"cannot read prepared frame {local_frame}: {prepared_video}")
        prepared_448 = cv2.resize(prepared_image, (448, 448), interpolation=cv2.INTER_AREA)
        neighbor_mae: dict[int, float] = {}
        for source_frame in range(expected_source_frame - 2, expected_source_frame + 3):
            source.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, source_image = source.read()
            if not ok:
                raise RuntimeError(f"cannot read source frame {source_frame}: {source_video}")
            neighbor_mae[source_frame] = float(
                np.mean(np.abs(prepared_448.astype(np.float32) - source_image.astype(np.float32)))
            )
        best_source_frame = min(neighbor_mae, key=neighbor_mae.get)
        if best_source_frame != expected_source_frame:
            raise RuntimeError(
                f"prepared/source frame alignment mismatch at local {local_frame}: "
                f"expected {expected_source_frame}, best {best_source_frame}, MAE={neighbor_mae}"
            )
        checks.append(
            {
                "local_frame_idx": local_frame,
                "expected_source_frame_idx": expected_source_frame,
                "best_matching_source_frame_idx_within_plus_minus_2": best_source_frame,
                "expected_frame_mean_absolute_bgr_error_after_960_to_448_resize": neighbor_mae[expected_source_frame],
                "neighbor_errors": {str(key): value for key, value in sorted(neighbor_mae.items())},
            }
        )
    source.release()
    prepared.release()
    return {
        "status": "pass",
        "method": "prepared endpoint frames downsampled to 448 and compared against source frame plus/minus two",
        "checks": checks,
        "maximum_expected_frame_mae": max(row["expected_frame_mean_absolute_bgr_error_after_960_to_448_resize"] for row in checks),
    }


def validate_prediction_isolation(input_dir: Path, evaluation_track: str) -> dict[str, Any]:
    names = sorted(path.name for path in input_dir.iterdir())
    expected = ["PREDICTION_INPUT_MANIFEST.json", "input.mp4"]
    if names != expected:
        raise RuntimeError(f"prediction input sidecars differ from {expected}: {input_dir}: {names}")
    manifest_path = input_dir / "PREDICTION_INPUT_MANIFEST.json"
    text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(text)
    if evaluation_track in text:
        raise RuntimeError(f"evaluation-only track leaked into prediction manifest: {input_dir}")
    if manifest.get("evaluation_relation_track_published_to_predictor") is not False:
        raise RuntimeError(f"prediction manifest does not explicitly suppress relation track: {input_dir}")
    forbidden_identity_keys = sorted({"take_uid", "take_name", "target_track"}.intersection(manifest))
    if forbidden_identity_keys:
        raise RuntimeError(f"prediction manifest publishes evaluation/dataset identity keys: {forbidden_identity_keys}")
    return {
        "status": "prediction_evaluation_physically_isolated",
        "files": names,
        "prediction_manifest": str(manifest_path),
        "input_video": str(input_dir / "input.mp4"),
        "input_video_sha256": sha256_file(input_dir / "input.mp4"),
        "evaluation_relation_track_absent": True,
        "take_uid_and_take_name_keys_absent": True,
    }


def cross_case_audit(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len({case["case_id"] for case in cases}) != len(cases):
        raise RuntimeError("duplicate case_id in suite")
    if len({case["take_uid"] for case in cases}) != len(cases):
        raise RuntimeError("suite must use one interval per take")
    role_counts = Counter(case["role"] for case in cases)
    participants_by_role: dict[str, set[Any]] = defaultdict(set)
    captures_by_role: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        participants_by_role[row["role"]].add(row["participant_uid"])
        captures_by_role[row["role"]].add(row["capture_uid"])
    dev_roles = {"development", "development_reference_consumed"}
    dev_participants = set().union(*(participants_by_role[role] for role in dev_roles))
    holdout_participants = participants_by_role["locked_internal_holdout"]
    dev_captures = set().union(*(captures_by_role[role] for role in dev_roles))
    holdout_captures = captures_by_role["locked_internal_holdout"]
    overlap_participants = sorted(dev_participants.intersection(holdout_participants), key=str)
    overlap_captures = sorted(dev_captures.intersection(holdout_captures), key=str)
    if overlap_participants or overlap_captures:
        raise RuntimeError(
            f"development/holdout identity leakage: participants={overlap_participants}, captures={overlap_captures}"
        )
    return {
        "case_count": len(cases),
        "role_counts": dict(sorted(role_counts.items())),
        "unique_take_count": len({case["take_uid"] for case in cases}),
        "unique_participant_count": len({row["participant_uid"] for row in rows}),
        "unique_capture_count": len({row["capture_uid"] for row in rows}),
        "development_holdout_participant_overlap": overlap_participants,
        "development_holdout_capture_overlap": overlap_captures,
        "universities": dict(sorted(Counter(row["university_name"] for row in rows).items())),
        "tasks": dict(sorted(Counter(row["task_name"] for row in rows).items())),
        "local_vrs_available_case_count": sum(row["calibration"]["local_vrs_exists"] for row in rows),
        "all_cases_have_five_required_masks": all(
            len(row["visible_mask_gt"]["required_source_frames"]) == 5 for row in rows
        ),
        "all_cases_have_complete_camera_extrinsics": all(
            row["camera"]["complete_extrinsic_frame_count"] == 150 for row in rows
        ),
        "status": "suite_selection_contract_valid",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=Path(__file__).with_name("multiclip_rigid_suite_v1.json"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nas-106/ego4d"))
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--render-selection-review", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    suite_config_path = args.suite_config.resolve()
    suite = load_json(suite_config_path)
    cases = suite["cases"]
    if args.prepare and (args.prediction_root is None or args.benchmark_root is None):
        raise RuntimeError("--prepare requires --prediction-root and --benchmark-root")
    prediction_root = args.prediction_root.resolve() if args.prediction_root is not None else None
    benchmark_root = args.benchmark_root.resolve() if args.benchmark_root is not None else None
    if prediction_root is not None and benchmark_root is not None:
        if prediction_root == benchmark_root or prediction_root in benchmark_root.parents or benchmark_root in prediction_root.parents:
            raise RuntimeError("prediction and benchmark roots must be physically separate")
    output_manifest = args.output_manifest.resolve()
    if output_manifest.exists() and not args.replace:
        raise FileExistsError(f"output manifest exists: {output_manifest}; pass --replace")
    if args.prepare:
        assert prediction_root is not None and benchmark_root is not None
        for root in (prediction_root, benchmark_root):
            if root.exists():
                if not args.replace:
                    raise FileExistsError(f"suite root exists: {root}; pass --replace")
                shutil.rmtree(root)
            root.mkdir(parents=True)
    review_root = (
        benchmark_root / "_selection_review_evaluation_only"
        if args.render_selection_review and benchmark_root is not None
        else output_manifest.parent / "selection_review_evaluation_only"
        if args.render_selection_review
        else None
    )

    take_rows = load_json(dataset_root / "takes.json")
    takes = {row["take_uid"]: row for row in take_rows}
    official_split = load_json(dataset_root / "annotations" / "splits.json")["take_uid_to_split"]
    sha_cache: dict[Path, str] = {}
    audit_rows: list[dict[str, Any]] = []
    preparation_rows: list[dict[str, Any]] = []
    for split in sorted({case["split"] for case in cases}):
        relation_path = dataset_root / "annotations" / f"relations_{split}.json"
        relation_payload = load_json(relation_path)
        relation_annotations = relation_payload["annotations"]
        atomic_path = dataset_root / "annotations" / f"atomic_descriptions_{split}.json"
        atomic_payload = load_json(atomic_path) if atomic_path.exists() else None
        for case in [item for item in cases if item["split"] == split]:
            if official_split.get(case["take_uid"]) != split:
                raise RuntimeError(
                    f"official split mismatch for {case['case_id']}: config={split}, official={official_split.get(case['take_uid'])}"
                )
            take = takes.get(case["take_uid"])
            if take is None or take.get("take_name") != case["take_name"]:
                raise RuntimeError(f"take identity mismatch for {case['case_id']}")
            relation = relation_annotations.get(case["take_uid"])
            if relation is None:
                raise RuntimeError(f"relations annotation missing for {case['case_id']}")
            audit, _stream = audit_case(
                case,
                suite["clip_contract"],
                dataset_root,
                take,
                relation,
                sha_cache,
                review_root,
            )
            audit_rows.append(audit)
            if args.prepare:
                assert prediction_root is not None and benchmark_root is not None
                case_args = preparation_args(case, dataset_root, prediction_root, benchmark_root)
                manifest = prepare(
                    case_args,
                    take_override=take,
                    relation_annotations_override=relation_annotations,
                    atomic_descriptions_override=atomic_payload,
                    sha256_cache=sha_cache,
                )
                isolation = validate_prediction_isolation(
                    case_args.input_dir,
                    case["target_track_evaluation_only"],
                )
                coordinate_test = run_coordinate_contract_test(case_args.ground_truth_dir)
                if coordinate_test["status"] != "pass":
                    raise RuntimeError(f"coordinate contract self-test failed for {case['case_id']}: {coordinate_test}")
                coordinate_test_path = case_args.ground_truth_dir / "coordinate_contract_self_test.json"
                write_json(coordinate_test_path, coordinate_test)
                frame_alignment = validate_prepared_frame_alignment(
                    Path(audit["source_video"]),
                    case_args.input_dir / "input.mp4",
                    int(case["source_frame_start"]),
                    int(suite["clip_contract"]["frame_count"]),
                )
                preparation_rows.append(
                    {
                        "case_id": case["case_id"],
                        "evaluation_label": case["evaluation_label"],
                        "role": case["role"],
                        "prediction_input_dir": str(case_args.input_dir),
                        "ground_truth_dir": str(case_args.ground_truth_dir),
                        "benchmark_manifest": str(case_args.ground_truth_dir / "BENCHMARK_MANIFEST.json"),
                        "prepared_video_sha256": manifest["prediction_video"]["sha256"],
                        "isolation": isolation,
                        "coordinate_contract_self_test": {
                            "status": coordinate_test["status"],
                            "path": str(coordinate_test_path),
                            "checks": coordinate_test["checks"],
                        },
                        "source_frame_alignment": frame_alignment,
                        "status": "prediction_and_evaluation_bundles_prepared",
                    }
                )
        del relation_payload, relation_annotations, atomic_payload

    cross = cross_case_audit(cases, audit_rows)
    result = {
        "status": "egoexo4d_rigid_multiclip_suite_prepared" if args.prepare else "egoexo4d_rigid_multiclip_suite_audited",
        "suite_id": suite["suite_id"],
        "suite_config": str(suite_config_path),
        "suite_config_sha256": sha256_file(suite_config_path),
        "dataset_root": str(dataset_root),
        "selection_scope": {
            "available_gt": [
                "five sparse raw-view visible object masks per case",
                "sparse named hand joints",
                "150/150 camera extrinsics",
            ],
            "not_available": [
                "object CAD/metric mesh",
                "object per-frame SE(3)",
                "MANO surface",
                "metric contact/nonpenetration",
                "local Aria VRS calibration",
            ],
            "selection_metrics_are_not_pose_gt": True,
        },
        "cross_case_audit": cross,
        "cases": audit_rows,
        "preparation": {
            "requested": bool(args.prepare),
            "prediction_root": str(prediction_root) if prediction_root is not None else None,
            "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
            "case_count": len(preparation_rows),
            "all_prediction_cases_isolated": (
                all(row["isolation"]["status"] == "prediction_evaluation_physically_isolated" for row in preparation_rows)
                if preparation_rows
                else None
            ),
            "all_coordinate_contract_self_tests_pass": (
                all(row["coordinate_contract_self_test"]["status"] == "pass" for row in preparation_rows)
                if preparation_rows
                else None
            ),
            "all_endpoint_source_frame_alignment_checks_pass": (
                all(row["source_frame_alignment"]["status"] == "pass" for row in preparation_rows)
                if preparation_rows
                else None
            ),
            "maximum_endpoint_expected_frame_mae": (
                max(row["source_frame_alignment"]["maximum_expected_frame_mae"] for row in preparation_rows)
                if preparation_rows
                else None
            ),
            "cases": preparation_rows,
            "runtime_runs_launched": 0,
        },
        "next_step": "Use development cases for staged P00-P14 diagnostics and exact-state correction work; do not inspect locked holdout pipeline outputs until mechanisms and suite-level acceptance logic are frozen.",
    }
    write_json(output_manifest, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "suite_id": result["suite_id"],
                "case_count": cross["case_count"],
                "role_counts": cross["role_counts"],
                "unique_participant_count": cross["unique_participant_count"],
                "local_vrs_available_case_count": cross["local_vrs_available_case_count"],
                "prepared_case_count": len(preparation_rows),
                "runtime_runs_launched": 0,
                "output_manifest": str(output_manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
