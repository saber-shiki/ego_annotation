#!/usr/bin/env python3
"""Scan Ego-Exo4D for five-second partial-GT hand-object candidate windows.

The scan is category-agnostic. It requires five non-empty one-Hz relation masks,
complete camera extrinsics, an available raw Aria RGB MP4, and reports released
hand-joint coverage. Rigidity and whether the target is actually manipulated
remain curator judgments from raw RGB plus evaluation-only mask review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_benchmark import (
    decode_coco_compressed_counts,
    load_json,
    lz_decompress_uri_component,
    write_json,
)


def foreground_area(mask_row: dict[str, Any]) -> int:
    encoded = lz_decompress_uri_component(str(mask_row["encodedMask"]))
    counts = decode_coco_compressed_counts(encoded)
    expected = int(mask_row["height"]) * int(mask_row["width"])
    if sum(counts) != expected:
        raise RuntimeError(f"invalid relation mask RLE: sum={sum(counts)}, expected={expected}")
    return int(sum(counts[1::2]))


def selected_person_joint_count(people: list[dict[str, Any]]) -> int:
    return max((len(person.get("annotation3D", {})) for person in people), default=0)


def candidate_score(
    start: int,
    hand_counts: dict[int, int],
    frame_count: int,
) -> tuple[int, int, int, int, int]:
    rows = [(frame, count) for frame, count in hand_counts.items() if start <= frame < start + frame_count and count > 0]
    frames_15 = [frame for frame, count in rows if count >= 15]
    bins = {
        min(4, max(0, (frame - start) * 5 // frame_count))
        for frame, _count in rows
    }
    return len(frames_15), len(rows), len(bins), sum(count for _frame, count in rows), -start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nas-106/ego4d"))
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--frame-count", type=int, default=150)
    parser.add_argument("--mask-step-frames", type=int, default=30)
    parser.add_argument("--required-mask-count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    takes = {row["take_uid"]: row for row in load_json(dataset_root / "takes.json")}
    all_candidates: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    for split in args.splits:
        relations = load_json(dataset_root / "annotations" / f"relations_{split}.json")["annotations"]
        hand_dir = dataset_root / "annotations" / "ego_pose" / split / "hand" / "annotation"
        camera_dir = dataset_root / "annotations" / "ego_pose" / split / "camera_pose"
        overlap = sorted(
            set(relations)
            .intersection(path.stem for path in hand_dir.glob("*.json"))
            .intersection(path.stem for path in camera_dir.glob("*.json"))
        )
        track_stream_count = 0
        rejected_no_five_frame_run = 0
        rejected_all_candidate_windows_empty = 0
        rejected_missing_camera_or_video = 0
        split_candidates: list[dict[str, Any]] = []
        for take_uid in overlap:
            take = takes[take_uid]
            hand_payload = load_json(hand_dir / f"{take_uid}.json")
            hand_counts = {
                int(frame): selected_person_joint_count(people)
                for frame, people in hand_payload.items()
            }
            camera_payload = load_json(camera_dir / f"{take_uid}.json")
            aria_cameras = [name for name in camera_payload if name.startswith("aria")]
            for target_track, streams in relations[take_uid].get("object_masks", {}).items():
                for stream, stream_payload in streams.items():
                    if not (stream.startswith("aria") and stream.endswith("_214-1")):
                        continue
                    track_stream_count += 1
                    camera_id = stream.removesuffix("_214-1")
                    if camera_id not in camera_payload:
                        if len(aria_cameras) == 1:
                            camera_id = aria_cameras[0]
                        else:
                            rejected_missing_camera_or_video += 1
                            continue
                    annotations = stream_payload.get("annotation", {})
                    frames = sorted(map(int, annotations))
                    frame_set = set(frames)
                    starts = [
                        frame
                        for frame in frames
                        if all(frame + args.mask_step_frames * offset in frame_set for offset in range(args.required_mask_count))
                    ]
                    if not starts:
                        rejected_no_five_frame_run += 1
                        continue
                    extrinsics = set(map(int, camera_payload[camera_id].get("camera_extrinsics", {})))
                    source_video = (
                        dataset_root
                        / take["root_dir"]
                        / "frame_aligned_videos"
                        / "downscaled"
                        / "448"
                        / f"{stream}.mp4"
                    )
                    valid_starts = [
                        start
                        for start in starts
                        if source_video.exists()
                        and all(frame in extrinsics for frame in range(start, start + args.frame_count))
                    ]
                    if not valid_starts:
                        rejected_missing_camera_or_video += 1
                        continue
                    ranked_starts = sorted(
                        valid_starts,
                        key=lambda start: candidate_score(start, hand_counts, args.frame_count),
                        reverse=True,
                    )
                    selected_start = None
                    selected_areas: list[int] = []
                    for start in ranked_starts:
                        areas = [
                            foreground_area(annotations[str(start + args.mask_step_frames * offset)])
                            for offset in range(args.required_mask_count)
                        ]
                        if all(area > 0 for area in areas):
                            selected_start = start
                            selected_areas = areas
                            break
                    if selected_start is None:
                        rejected_all_candidate_windows_empty += 1
                        continue
                    score = candidate_score(selected_start, hand_counts, args.frame_count)
                    vrs_relative = take.get("vrs_relative_path")
                    local_vrs = dataset_root / take["root_dir"] / str(vrs_relative) if vrs_relative else None
                    row = {
                        "split": split,
                        "take_uid": take_uid,
                        "take_name": take["take_name"],
                        "task_name": take.get("task_name"),
                        "participant_uid": take.get("participant_uid"),
                        "capture_uid": take.get("capture_uid"),
                        "university_name": take.get("university_name"),
                        "target_track_evaluation_only": target_track,
                        "source_stream": stream,
                        "source_frame_start": selected_start,
                        "source_frame_end": selected_start + args.frame_count - 1,
                        "required_mask_source_frames": [
                            selected_start + args.mask_step_frames * offset
                            for offset in range(args.required_mask_count)
                        ],
                        "required_mask_area_px": selected_areas,
                        "required_mask_mean_area_fraction": sum(selected_areas)
                        / len(selected_areas)
                        / (
                            int(annotations[str(selected_start)]["height"])
                            * int(annotations[str(selected_start)]["width"])
                        ),
                        "available_track_mask_frame_count": len(annotations),
                        "valid_consecutive_window_count_before_nonempty_check": len(valid_starts),
                        "hand_frames_with_15plus_3d_joints": score[0],
                        "hand_annotated_frame_count": score[1],
                        "hand_occupied_temporal_bin_count": score[2],
                        "source_video": str(source_video),
                        "metadata_has_trimmed_vrs": bool(take.get("has_trimmed_vrs")),
                        "metadata_vrs_relative_path": vrs_relative,
                        "local_vrs_exists": bool(local_vrs is not None and local_vrs.exists()),
                    }
                    split_candidates.append(row)
        split_candidates.sort(
            key=lambda row: (
                row["hand_occupied_temporal_bin_count"],
                row["hand_frames_with_15plus_3d_joints"],
                row["hand_annotated_frame_count"],
                row["required_mask_mean_area_fraction"],
            ),
            reverse=True,
        )
        all_candidates.extend(split_candidates)
        split_reports[split] = {
            "relations_hand_camera_take_overlap_count": len(overlap),
            "aria_object_track_stream_count": track_stream_count,
            "selected_candidate_track_stream_count": len(split_candidates),
            "rejected_no_five_frame_run": rejected_no_five_frame_run,
            "rejected_all_candidate_windows_empty": rejected_all_candidate_windows_empty,
            "rejected_missing_camera_or_video": rejected_missing_camera_or_video,
        }
    report = {
        "status": "egoexo4d_partial_gt_candidate_scan_complete",
        "method": "category-agnostic five-nonempty-mask run plus camera/video contract; rigidity requires curator review",
        "dataset_root": str(dataset_root),
        "contract": {
            "frame_count": args.frame_count,
            "mask_step_frames": args.mask_step_frames,
            "required_mask_count": args.required_mask_count,
            "object_pose_or_contact_gt_used": False,
            "target_category_filter_used": False,
        },
        "split_reports": split_reports,
        "candidate_count": len(all_candidates),
        "candidates": all_candidates,
    }
    write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_count": len(all_candidates),
                "split_reports": split_reports,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
