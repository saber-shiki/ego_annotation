#!/usr/bin/env python3
"""Prepare an isolated Ego-Exo4D hand-object benchmark clip and GT bundle.

The prediction input contains raw, distorted Aria RGB plus a target text hint;
take identity and the evaluation-only relation track are not published there.
Evaluation annotations (3-D hand keypoints, rectified-view 2-D hand keypoints,
per-frame camera extrinsics, and sparse raw-view visible object masks) are
written to a separate directory so a V19 runtime can run blind.

Important camera contract: Ego-Exo4D ego-pose intrinsics describe the official
512x512 undistorted linear camera, not the downscaled raw Aria RGB MP4. The
local shard has no VRS calibration needed to reproduce that rectification, so
this adapter does not publish those rectified intrinsics as prediction-side raw
RGB intrinsics. Ego-Exo4D also does not provide object CAD/6-DoF/contact/
nonpenetration GT; the availability report states those omissions explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LZ_URI_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-$"
LZ_URI_BASE = {char: index for index, char in enumerate(LZ_URI_ALPHABET)}
EGO4D_REFERENCE_COMMIT = "4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb"
HAWOR_OPENPOSE_SUFFIXES = [
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_4",
    "index_1",
    "index_2",
    "index_3",
    "index_4",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_4",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_4",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_4",
]

# Official Ego-Exo4D extraction uses aria_original_to_extracted:
#   x_extracted = H - y_original; y_extracted = x_original.
# Ignoring the one-pixel image-coordinate offset, camera vectors therefore use
# X_extracted = ANNOTATION_CAMERA_TO_EXTRACTED_CAMERA @ X_annotation.
ANNOTATION_CAMERA_TO_EXTRACTED_CAMERA = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout


def lz_decompress_uri_component(encoded: str) -> str:
    """Decode lz-string's compressToEncodedURIComponent representation."""
    encoded = encoded.replace(" ", "+")
    if not encoded:
        return ""
    length = len(encoded)
    data_value = LZ_URI_BASE[encoded[0]]
    data_position = 32
    data_index = 1

    def read_bits(count: int) -> int:
        nonlocal data_value, data_position, data_index
        bits = 0
        power = 1
        max_power = 1 << count
        while power != max_power:
            residual = data_value & data_position
            data_position >>= 1
            if data_position == 0:
                data_position = 32
                data_value = LZ_URI_BASE[encoded[data_index]] if data_index < length else 0
                data_index += 1
            if residual:
                bits |= power
            power <<= 1
        return bits

    dictionary: dict[int, str | int] = {0: 0, 1: 1, 2: 2}
    enlarge_in = 4
    dictionary_size = 4
    num_bits = 3
    first_code = read_bits(2)
    if first_code == 0:
        first = chr(read_bits(8))
    elif first_code == 1:
        first = chr(read_bits(16))
    elif first_code == 2:
        return ""
    else:  # pragma: no cover - impossible for two bits
        raise RuntimeError(f"invalid first lz-string code: {first_code}")
    dictionary[3] = first
    word = first
    result = [first]
    while True:
        code = read_bits(num_bits)
        if code == 0:
            dictionary[dictionary_size] = chr(read_bits(8))
            code = dictionary_size
            dictionary_size += 1
            enlarge_in -= 1
        elif code == 1:
            dictionary[dictionary_size] = chr(read_bits(16))
            code = dictionary_size
            dictionary_size += 1
            enlarge_in -= 1
        elif code == 2:
            return "".join(result)
        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1
        if code in dictionary:
            entry = str(dictionary[code])
        elif code == dictionary_size:
            entry = word + word[0]
        else:
            raise RuntimeError(f"invalid lz-string dictionary code: {code}")
        result.append(entry)
        dictionary[dictionary_size] = word + entry[0]
        dictionary_size += 1
        enlarge_in -= 1
        word = entry
        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1


def decode_coco_compressed_counts(encoded_counts: str) -> list[int]:
    """Decode COCO maskApi.c's compressed RLE counts string."""
    counts: list[int] = []
    position = 0
    while position < len(encoded_counts):
        value = 0
        shift_index = 0
        more = True
        while more:
            code = ord(encoded_counts[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift_index)
            more = bool(code & 0x20)
            if not more and (code & 0x10):
                value |= -1 << (5 * (shift_index + 1))
            shift_index += 1
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise RuntimeError(f"negative decoded COCO run: {value}")
        counts.append(value)
    return counts


def decode_egoexo_mask(row: dict[str, Any]) -> np.ndarray:
    height = int(row["height"])
    width = int(row["width"])
    compressed_counts = lz_decompress_uri_component(str(row["encodedMask"]))
    counts = decode_coco_compressed_counts(compressed_counts)
    expected = height * width
    if sum(counts) != expected:
        raise RuntimeError(f"mask RLE sum {sum(counts)} does not match {width}x{height}={expected}")
    flat = np.zeros(expected, dtype=np.uint8)
    cursor = 0
    foreground = False
    for count in counts:
        if foreground:
            flat[cursor : cursor + count] = 1
        cursor += count
        foreground = not foreground
    return flat.reshape((height, width), order="F")


def bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def find_take(dataset_root: Path, take_uid: str) -> dict[str, Any]:
    rows = load_json(dataset_root / "takes.json")
    matches = [row for row in rows if row.get("take_uid") == take_uid]
    if len(matches) != 1:
        raise RuntimeError(f"expected one take_uid={take_uid}, found {len(matches)}")
    return matches[0]


def choose_aria_camera(camera_payload: dict[str, Any], target_stream: str) -> str:
    stream_camera = target_stream.removesuffix("_214-1")
    if stream_camera in camera_payload:
        return stream_camera
    candidates = [key for key in camera_payload if key.startswith("aria")]
    if len(candidates) != 1:
        raise RuntimeError(f"cannot resolve Aria camera for stream {target_stream}: {candidates}")
    return candidates[0]


def select_person(people: list[dict[str, Any]], camera_id: str) -> tuple[int, dict[str, Any]]:
    if not people:
        raise RuntimeError("cannot select from empty hand annotation list")
    ranked = sorted(
        enumerate(people),
        key=lambda item: (
            len(item[1].get("annotation3D", {})),
            len(item[1].get("annotation2D", {}).get(camera_id, {})),
            -item[0],
        ),
        reverse=True,
    )
    return ranked[0]


def project_world(points_world: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    points_camera = homogeneous @ world_to_camera.T
    projected = points_camera @ intrinsics.T
    return projected[:, :2] / projected[:, 2:3]


def find_atomic_descriptions(
    dataset_root: Path,
    split: str,
    take_uid: str,
    start_s: float,
    end_s: float,
    payload_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    path = dataset_root / "annotations" / f"atomic_descriptions_{split}.json"
    if payload_override is None and not path.exists():
        return []
    payload = payload_override if payload_override is not None else load_json(path)
    annotations = payload.get("annotations", [])
    if isinstance(annotations, dict):
        candidates = [annotations[take_uid]] if take_uid in annotations else []
    else:
        candidates = [row for row in annotations if row.get("take_uid") == take_uid]
    result: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, list):
            records.extend(row for row in candidate if isinstance(row, dict))
        elif isinstance(candidate, dict):
            records.append(candidate)
    for candidate in records:
        for row in candidate.get("descriptions", []):
            timestamp = float(row.get("timestamp", -1.0))
            if start_s <= timestamp <= end_s:
                result.append(
                    {
                        "timestamp_s": timestamp,
                        "text": row.get("text"),
                        "ego_visible": row.get("ego_visible"),
                        "unsure": row.get("unsure"),
                    }
                )
    return sorted(result, key=lambda row: row["timestamp_s"])


def build_preview(
    video_path: Path,
    mask_rows: list[dict[str, Any]],
    mask_dir: Path,
    output: Path,
) -> None:
    mask_by_frame = {int(row["local_frame_idx"]): row for row in mask_rows}
    selected_frames = sorted(set(mask_by_frame) | {0, 30, 60, 90, 120, 149})
    capture = cv2.VideoCapture(str(video_path))
    panels: list[np.ndarray] = []
    for frame_idx in selected_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = capture.read()
        if not ok:
            continue
        if frame_idx in mask_by_frame:
            mask = cv2.imread(str(mask_dir / mask_by_frame[frame_idx]["runtime_mask_file"]), cv2.IMREAD_GRAYSCALE) > 0
            red = image.copy()
            red[mask] = (20, 20, 240)
            image = cv2.addWeighted(image, 0.45, red, 0.55, 0)
        cv2.rectangle(image, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"local={frame_idx} source={frame_idx + int(mask_rows[0]['clip_source_frame_start']) if mask_rows else frame_idx}",
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "red=raw-view visible-mask GT; rectified hand UV intentionally not overlaid",
            (8, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(cv2.resize(image, (480, 480), interpolation=cv2.INTER_AREA))
    capture.release()
    if not panels:
        return
    columns = 3
    rows = int(math.ceil(len(panels) / columns))
    sheet = np.full((rows * 480, columns * 480, 3), 242, dtype=np.uint8)
    for index, panel in enumerate(panels):
        y = (index // columns) * 480
        x = (index % columns) * 480
        sheet[y : y + 480, x : x + 480] = panel
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def prepare(
    args: argparse.Namespace,
    *,
    take_override: dict[str, Any] | None = None,
    relation_annotations_override: dict[str, Any] | None = None,
    atomic_descriptions_override: dict[str, Any] | None = None,
    sha256_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    input_dir = args.input_dir.resolve()
    gt_dir = args.ground_truth_dir.resolve()
    if input_dir == gt_dir or input_dir in gt_dir.parents or gt_dir in input_dir.parents:
        raise RuntimeError("prediction input and ground-truth directories must be isolated")
    for path in (input_dir, gt_dir):
        if path.exists():
            if not args.replace:
                raise FileExistsError(f"refusing to overwrite existing directory: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True)

    take = take_override if take_override is not None else find_take(dataset_root, args.take_uid)
    if str(take.get("take_uid")) != args.take_uid:
        raise RuntimeError(f"take override UID mismatch: expected {args.take_uid}, got {take.get('take_uid')}")
    relation_path = dataset_root / "annotations" / f"relations_{args.split}.json"
    hand_path = dataset_root / "annotations" / "ego_pose" / args.split / "hand" / "annotation" / f"{args.take_uid}.json"
    camera_path = dataset_root / "annotations" / "ego_pose" / args.split / "camera_pose" / f"{args.take_uid}.json"
    relation_annotations = (
        relation_annotations_override
        if relation_annotations_override is not None
        else load_json(relation_path)["annotations"]
    )
    relation = relation_annotations[args.take_uid]
    if args.target_track not in relation.get("object_masks", {}):
        raise RuntimeError(f"target track not found: {args.target_track}")
    target_streams = relation["object_masks"][args.target_track]
    stream_candidates = [name for name in target_streams if name.startswith("aria") and name.endswith("_214-1")]
    if not stream_candidates:
        raise RuntimeError(f"target track has no Aria RGB masks: {list(target_streams)}")
    target_stream = max(stream_candidates, key=lambda name: len(target_streams[name].get("annotation", {})))
    camera_payload = load_json(camera_path)
    camera_id = choose_aria_camera(camera_payload, target_stream)
    source_video = dataset_root / take["root_dir"] / "frame_aligned_videos" / "downscaled" / "448" / f"{target_stream}.mp4"
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    source_frame_end = args.source_frame_start + args.frame_count - 1
    rectified_intrinsics = np.asarray(camera_payload[camera_id]["camera_intrinsics"], dtype=np.float64)
    raw_resize_scale = float(args.output_size) / 448.0

    output_video = input_dir / "input.mp4"
    filter_graph = (
        f"trim=start_frame={args.source_frame_start}:end_frame={source_frame_end + 1},"
        f"setpts=PTS-STARTPTS,scale={args.output_size}:{args.output_size}:flags=lanczos"
    )
    ffmpeg_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-vf",
        filter_graph,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "10",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        str(output_video),
    ]
    run(ffmpeg_command)
    probe = json.loads(
        run(
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
                str(output_video),
            ]
        )
    )["streams"][0]
    if int(probe["nb_frames"]) != args.frame_count:
        raise RuntimeError(f"prepared video frame count mismatch: {probe}")
    if int(probe["width"]) != args.output_size or int(probe["height"]) != args.output_size:
        raise RuntimeError(f"prepared video size mismatch: {probe}")

    prediction_manifest = {
        "status": "prediction_input_ready_raw_distorted_aria_rgb",
        "case_id": args.case_id,
        "target_hint": args.target_description,
        "evaluation_relation_track_published_to_predictor": False,
        "clip": {
            "source_frame_start_inclusive": args.source_frame_start,
            "source_frame_end_inclusive": source_frame_end,
            "local_frame_start": 0,
            "local_frame_end": args.frame_count - 1,
            "fps": 30.0,
        },
        "video": str(output_video),
        "video_sha256": sha256_file(output_video),
        "camera_model_contract": {
            "input_view": "raw distorted Aria RGB in Ego-Exo4D extracted orientation, resized 448x448 to runtime size",
            "raw_distortion_calibration_available_in_local_shard": False,
            "pinhole_intrinsics_published_to_predictor": False,
            "reason": "released ego-pose K is for the official 512x512 undistorted linear view and must not be applied directly to raw RGB",
            "required_to_rectify": "take VRS/no-image-stream VRS device calibration plus projectaria_tools",
        },
        "blind_evaluation_policy": "Runtime may consume RGB and the target text hint only. Rectified-view K, camera extrinsics, hand annotations, and object masks are evaluation-only.",
    }
    write_json(input_dir / "PREDICTION_INPUT_MANIFEST.json", prediction_manifest)

    extrinsics_payload = camera_payload[camera_id]["camera_extrinsics"]
    camera_rows: list[dict[str, Any]] = []
    for local_frame in range(args.frame_count):
        source_frame = args.source_frame_start + local_frame
        if str(source_frame) not in extrinsics_payload:
            raise RuntimeError(f"missing camera extrinsics for source frame {source_frame}")
        camera_rows.append(
            {
                "local_frame_idx": local_frame,
                "source_frame_idx": source_frame,
                "timestamp_s_in_take": source_frame / 30.0,
                "world_to_camera_3x4": extrinsics_payload[str(source_frame)],
            }
        )
    write_json(
        gt_dir / "camera_pose_gt.json",
        {
            "status": "ok",
            "source": "Ego-Exo4D v2 ego_pose camera_pose annotation",
            "coordinate_contract": {
                "extrinsics": "world_to_annotation_camera_3x4; X_annotation_camera = E @ [X_world_m, 1]",
                "world_units": "meter",
                "rectified_annotation_image_grid": [512, 512],
                "prediction_raw_extracted_image_grid": [args.output_size, args.output_size],
                "annotation_camera_to_prediction_extracted_camera_3x3": ANNOTATION_CAMERA_TO_EXTRACTED_CAMERA.tolist(),
                "prediction_camera_vector_adapter": "for row vectors, X_annotation = X_prediction_extracted @ annotation_camera_to_prediction_extracted_camera_3x3",
                "raw_to_rectified_pixel_mapping_available": False,
            },
            "camera_id": camera_id,
            "intrinsics_rectified_512_3x3": rectified_intrinsics.tolist(),
            "intrinsics_scope": "official Ego-Exo4D 512x512 undistorted linear camera used for pose annotation; not raw MP4 intrinsics",
            "official_adapter_provenance": {
                "repository": "https://github.com/facebookresearch/Ego4d",
                "reference_commit": EGO4D_REFERENCE_COMMIT,
                "code": "ego4d/internal/human_pose/undistort_to_halo.py and ego4d/internal/human_pose/utils.py::aria_original_to_extracted",
                "operation": "rotate extracted image back by 90 degrees, then distort_by_calibration from raw camera-rgb to a 512x512 linear camera with focal 150",
                "missing_local_input": "Aria VRS device calibration",
            },
            "frames": camera_rows,
        },
    )

    hand_payload = load_json(hand_path)
    hand_rows: list[dict[str, Any]] = []
    reprojection_errors: list[float] = []
    per_side_joint_rows = {"left": 0, "right": 0}
    person_row_count = 0
    multi_person_frame_count = 0
    frames_with_at_least_one_hand_15_joints = 0
    for source_frame_text, people in hand_payload.items():
        source_frame = int(source_frame_text)
        if not (args.source_frame_start <= source_frame <= source_frame_end):
            continue
        person_row_count += len(people)
        multi_person_frame_count += int(len(people) > 1)
        person_index, person = select_person(people, camera_id)
        world_to_camera = np.asarray(extrinsics_payload[source_frame_text], dtype=np.float64)
        hands: dict[str, Any] = {}
        for side in ("left", "right"):
            joints: list[dict[str, Any]] = []
            for output_index, suffix in enumerate(HAWOR_OPENPOSE_SUFFIXES):
                name = f"{side}_{suffix}"
                point3d = person.get("annotation3D", {}).get(name)
                point2d = person.get("annotation2D", {}).get(camera_id, {}).get(name)
                if not isinstance(point3d, dict):
                    continue
                xyz = np.asarray([point3d["x"], point3d["y"], point3d["z"]], dtype=np.float64)
                uv_dataset = [float(point2d["x"]), float(point2d["y"])] if isinstance(point2d, dict) else None
                projected = project_world(xyz[None, :], world_to_camera, rectified_intrinsics)[0]
                error = float(np.linalg.norm(projected - np.asarray(uv_dataset))) if uv_dataset is not None else None
                if error is not None:
                    reprojection_errors.append(error)
                joints.append(
                    {
                        "name": name,
                        "hawor_openpose_index": output_index,
                        "xyz_world_m": xyz.tolist(),
                        "num_views_for_3d": int(point3d.get("num_views_for_3d", 0)),
                        "uv_rectified_512_px": uv_dataset,
                        "released_3d_to_2d_reprojection_error_px_rectified_512": error,
                    }
                )
            if joints:
                hands[side] = {"joint_count": len(joints), "joints": joints}
                per_side_joint_rows[side] += len(joints)
        frames_with_at_least_one_hand_15_joints += int(any(hand["joint_count"] >= 15 for hand in hands.values()))
        hand_rows.append(
            {
                "local_frame_idx": source_frame - args.source_frame_start,
                "source_frame_idx": source_frame,
                "timestamp_s_in_take": source_frame / 30.0,
                "selected_person_index": person_index,
                "person_candidate_count": len(people),
                "selection_rule": "maximum available 3D joints, then maximum Aria 2D joints, then lowest person index",
                "hands": hands,
            }
        )
    hand_rows.sort(key=lambda row: row["local_frame_idx"])
    write_json(
        gt_dir / "hand_pose_gt.json",
        {
            "status": "ok",
            "source": "Ego-Exo4D v2 ego_pose hand annotation; multi-view 3D keypoints and per-camera 2D keypoints",
            "claim_scope": "sparse named 21-joint hand keypoints; not MANO parameters, MANO vertices, contact patches, or force",
            "joint_order_contract": HAWOR_OPENPOSE_SUFFIXES,
            "coordinate_contract": {
                "xyz": "Ego-Exo4D world coordinates in meters",
                "uv_rectified": "official 512x512 undistorted linear Aria camera; not the raw 448/960 RGB grid",
                "prediction_camera_adapter": "rotate HaWoR extracted-view camera vectors into the annotation camera with the fixed matrix stored in camera_pose_gt.json",
                "raw_rgb_2d_comparison": "unsupported without the missing VRS distortion calibration",
            },
            "coverage": {
                "clip_frame_count": args.frame_count,
                "annotated_frame_count": len(hand_rows),
                "annotated_frame_fraction": len(hand_rows) / args.frame_count,
                "person_row_count": person_row_count,
                "multi_person_frame_count": multi_person_frame_count,
                "frames_with_at_least_one_hand_15_joints": frames_with_at_least_one_hand_15_joints,
                "joint_rows_by_side": per_side_joint_rows,
                "released_reprojection_error_px_rectified_512": {
                    "count": len(reprojection_errors),
                    "median": float(np.median(reprojection_errors)) if reprojection_errors else None,
                    "p95": float(np.percentile(reprojection_errors, 95)) if reprojection_errors else None,
                    "max": float(np.max(reprojection_errors)) if reprojection_errors else None,
                },
            },
            "frames": hand_rows,
        },
    )

    original_mask_dir = gt_dir / "object_visible_masks_1408"
    runtime_mask_dir = gt_dir / f"object_visible_masks_{args.output_size}"
    original_mask_dir.mkdir(parents=True)
    runtime_mask_dir.mkdir(parents=True)
    target_annotations = target_streams[target_stream]["annotation"]
    mask_rows: list[dict[str, Any]] = []
    for source_frame_text, mask_row in sorted(target_annotations.items(), key=lambda item: int(item[0])):
        source_frame = int(source_frame_text)
        if not (args.source_frame_start <= source_frame <= source_frame_end):
            continue
        mask = decode_egoexo_mask(mask_row)
        local_frame = source_frame - args.source_frame_start
        runtime_mask = cv2.resize(mask, (args.output_size, args.output_size), interpolation=cv2.INTER_NEAREST)
        original_name = f"frame_{local_frame:06d}_source_{source_frame:06d}.png"
        runtime_name = f"frame_{local_frame:06d}.png"
        cv2.imwrite(str(original_mask_dir / original_name), mask * 255)
        cv2.imwrite(str(runtime_mask_dir / runtime_name), runtime_mask * 255)
        mask_rows.append(
            {
                "local_frame_idx": local_frame,
                "source_frame_idx": source_frame,
                "timestamp_s_in_take": source_frame / 30.0,
                "clip_source_frame_start": args.source_frame_start,
                "source_mask_file": original_name,
                "runtime_mask_file": runtime_name,
                "source_grid": [int(mask.shape[1]), int(mask.shape[0])],
                "runtime_grid": [args.output_size, args.output_size],
                "source_area_px": int(mask.sum()),
                "runtime_area_px": int(runtime_mask.sum()),
                "source_bbox_xyxy": bbox_xyxy(mask),
                "runtime_bbox_xyxy": bbox_xyxy(runtime_mask),
            }
        )
    write_json(
        gt_dir / "object_visible_mask_gt.json",
        {
            "status": "ok",
            "source": "Ego-Exo4D v2 relations object_masks encodedMask annotation",
            "target_track": args.target_track,
            "camera_stream": target_stream,
            "claim_scope": "sparse visible 2D masks, normally sampled at 1 Hz; not amodal masks, depth, object 6-DoF, CAD, or canonical geometry",
            "coverage": {
                "clip_frame_count": args.frame_count,
                "annotated_mask_frame_count": len(mask_rows),
                "annotated_mask_frame_fraction": len(mask_rows) / args.frame_count,
            },
            "frames": mask_rows,
        },
    )

    interval_start_s = args.source_frame_start / 30.0
    interval_end_s = source_frame_end / 30.0
    descriptions = find_atomic_descriptions(
        dataset_root,
        args.split,
        args.take_uid,
        interval_start_s - 2.0,
        interval_end_s + 2.0,
        payload_override=atomic_descriptions_override,
    )
    availability = {
        "status": "partial_ground_truth_available",
        "dataset_identity": "Ego-Exo4D v2 subset stored under nas-106/ego4d",
        "take": {
            "take_uid": args.take_uid,
            "take_name": take["take_name"],
            "task_name": take.get("task_name"),
            "split": args.split,
        },
        "target": {
            "description": args.target_description,
            "relation_track": args.target_track,
            "rigidity_assumption": args.rigidity_assumption,
        },
        "available_for_quantitative_evaluation": {
            "hand_2d_named_keypoints_rectified_512": True,
            "hand_2d_named_keypoints_raw_rgb": False,
            "hand_3d_named_keypoints_world_m": True,
            "camera_intrinsics_rectified_512_evaluation_only": True,
            "camera_intrinsics_raw_rgb_prediction_view": False,
            "camera_extrinsics_world_to_camera": True,
            "sparse_visible_object_masks": True,
            "object_identity_text": True,
            "temporal_action_descriptions": True,
        },
        "not_available_as_released_ground_truth": {
            "mano_parameters": True,
            "mano_vertices": True,
            "dense_depth": True,
            "object_cad_or_metric_mesh": True,
            "object_metric_dimensions": True,
            "object_6dof_pose": True,
            "object_3d_surface_points": True,
            "metric_contact_points_or_patches": True,
            "contact_force": True,
            "signed_distance_or_nonpenetration": True,
            "occlusion_owner_per_pixel": True,
        },
        "evaluation_consequence": "The run can be scored for raw-view 2D object segmentation, 3D hand localization after the official fixed camera-axis adapter, rectified-view hand reprojection, and camera trajectory. Raw-RGB hand reprojection is unsupported without VRS calibration. There is no independent full physical GT for P13/P14 object pose, P16 nonpenetration, P17 metric contact, or completed object geometry.",
        "atomic_descriptions_near_clip": descriptions,
    }
    write_json(gt_dir / "GROUND_TRUTH_AVAILABILITY.json", availability)
    build_preview(output_video, mask_rows, runtime_mask_dir, gt_dir / "ground_truth_preview.jpg")

    def cached_sha256(path: Path) -> str:
        resolved = path.resolve()
        if sha256_cache is None:
            return sha256_file(resolved)
        if resolved not in sha256_cache:
            sha256_cache[resolved] = sha256_file(resolved)
        return sha256_cache[resolved]

    manifest = {
        "status": "egoexo4d_rigid_benchmark_prepared",
        "case_id": args.case_id,
        "prediction_input_dir": str(input_dir),
        "ground_truth_dir": str(gt_dir),
        "isolation_verified": input_dir != gt_dir and input_dir not in gt_dir.parents and gt_dir not in input_dir.parents,
        "coordinate_contract_revision": "v2_rectified_pose_vs_raw_rgb_separation",
        "invalidated_v1_assumption": "ego-pose K/UV are not on the 448x448 raw RGB grid; they belong to an official 512x512 undistorted linear view",
        "source": {
            "dataset_root": str(dataset_root),
            "take_uid": args.take_uid,
            "take_name": take["take_name"],
            "split": args.split,
            "source_video": str(source_video),
            "source_video_sha256": cached_sha256(source_video),
            "relations_json_sha256": cached_sha256(relation_path),
            "hand_json_sha256": cached_sha256(hand_path),
            "camera_json_sha256": cached_sha256(camera_path),
        },
        "clip": {
            "source_frame_start_inclusive": args.source_frame_start,
            "source_frame_end_inclusive": source_frame_end,
            "frame_count": args.frame_count,
            "fps": 30.0,
            "runtime_size": [args.output_size, args.output_size],
        },
        "prediction_video": {
            "path": str(output_video),
            "sha256": sha256_file(output_video),
            "ffprobe": probe,
            "ffmpeg_command": ffmpeg_command,
        },
        "ground_truth_files": {
            "availability": str(gt_dir / "GROUND_TRUTH_AVAILABILITY.json"),
            "camera_pose": str(gt_dir / "camera_pose_gt.json"),
            "hand_pose": str(gt_dir / "hand_pose_gt.json"),
            "object_visible_masks": str(gt_dir / "object_visible_mask_gt.json"),
            "preview": str(gt_dir / "ground_truth_preview.jpg"),
        },
    }
    write_json(gt_dir / "BENCHMARK_MANIFEST.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nas-106/ego4d"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--take-uid", default="a89215f1-92fe-4207-9293-62ce108171da")
    parser.add_argument("--source-frame-start", type=int, default=2040)
    parser.add_argument("--frame-count", type=int, default=150)
    parser.add_argument("--target-track", default="yellow bicycle tire lever_0")
    parser.add_argument("--target-description", default="yellow rigid plastic bicycle tire lever")
    parser.add_argument(
        "--rigidity-assumption",
        default="curator-reviewed approximately rigid target over the selected interval",
        help="Evaluation-only statement explaining why the selected target is treated as rigid.",
    )
    parser.add_argument("--case-id", default="egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189")
    parser.add_argument("--output-size", type=int, default=960)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = prepare(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
