#!/usr/bin/env python3
"""Standalone appearance adapter for depth-ordered P09.

It validates the raw SAM2, derived owned-SAM2 and preprocessor identities, then
uses raw masks for appearance scoring/copies and corrected P09 annotations only
for same-frame metric alignment.  Raw appearance is never metric eligibility.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from build_v19_demo_visual_inputs import (
    SAFE_ID,
    in_run_file,
    indexed_rows,
    load,
    metric_alignment,
    metric_object,
    rgb_path,
    score_mask,
    sha256,
    track_rows,
)

SCHEMA = "rgb_scene_depth_order_appearance_manifest_v1"
OWNERSHIP_SCHEMA = "rgb_scene_depth_order_owned_sam2_v1"


def require_ref(value: object, expected: Path, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} record missing")
    path = Path(value.get("path", "")).expanduser().resolve()
    if path != expected.resolve() or not path.is_file() or value.get("sha256") != sha256(path):
        raise ValueError(f"{label} path/hash mismatch")
    return value


def validate_binding(annotation_payload: dict, ownership: dict, *, raw_track: Path,
                     annotations: Path, case_id: str, object_id: str,
                     track_id: str) -> dict:
    adapter = annotation_payload.get("v19_visible_geometry_adapter")
    if (ownership.get("schema") != OWNERSHIP_SCHEMA or ownership.get("status") != "ok"
            or not isinstance(adapter, dict)):
        raise ValueError("corrected ownership/P09 contracts missing")
    expected_ids = {"case": case_id, "object_id": object_id, "track_id": track_id}
    if any(ownership.get(key) != value or adapter.get(key) != value
           for key, value in expected_ids.items()):
        raise ValueError("raw/owned/P09 IDs differ")
    raw_ref = require_ref((ownership.get("inputs") or {}).get("raw_sam2_track"), raw_track,
                          "raw SAM2 track")
    owned_value = (ownership.get("outputs") or {}).get("owned_sam2_track")
    if not isinstance(owned_value, dict):
        raise ValueError("owned SAM2 track record missing")
    owned_path = Path(owned_value.get("path", "")).expanduser().resolve()
    owned_ref = require_ref(owned_value, owned_path, "owned SAM2 track")
    if Path(adapter.get("sam2_track", "")).expanduser().resolve() != owned_path:
        raise ValueError("corrected P09 did not consume owned SAM2 track")
    for key in ("base_annotations", "depth_npz", "calibration_contract"):
        producer = (ownership.get("inputs") or {}).get(key)
        path = Path(adapter.get(key, "")).expanduser().resolve()
        require_ref(producer, path, key)
    return {
        "preprocessor": {"path": str(ownership["_path"]),
                         "sha256": sha256(ownership["_path"])},
        "raw_sam2_track": raw_ref,
        "owned_sam2_track": owned_ref,
        "corrected_p09_annotations": {"path": str(annotations), "sha256": sha256(annotations)},
    }


def build(args: argparse.Namespace) -> tuple[dict, dict]:
    root = args.run_root.resolve()
    if not SAFE_ID.fullmatch(args.object_id):
        raise ValueError("unsafe object ID")
    output = args.output_dir.resolve()
    candidate_root = getattr(args, 'candidate_root', None)
    allowed = (candidate_root.resolve() if candidate_root is not None else
               root / "measurements" / "demo_visual" / args.object_id)
    if candidate_root is None and output.parent != allowed:
        raise ValueError("new output under measurements/demo_visual/<object> required")
    if candidate_root is not None:
        allowed = allowed / "measurements" / "demo_visual" / args.object_id
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("new output under measurements/demo_visual/<object> required")
    raw_manifest_path = in_run_file(args.raw_frame_manifest, root, "raw frame manifest")
    annotations_path = in_run_file(args.annotations, root, "corrected P09 annotations")
    raw_track_path = in_run_file(args.raw_sam2_track, root, "raw SAM2 track")
    ownership_path = in_run_file(args.depth_order_report, root, "depth-order report")
    annotation_payload = load(annotations_path)
    ownership = load(ownership_path)
    ownership["_path"] = ownership_path
    binding = validate_binding(annotation_payload, ownership, raw_track=raw_track_path,
                               annotations=annotations_path, case_id=args.case_id,
                               object_id=args.object_id, track_id=args.track_id)
    raw = indexed_rows(load(raw_manifest_path).get("frames"), "raw manifest")
    annotations = indexed_rows(annotation_payload.get("frames"), "annotations")
    track = track_rows(load(raw_track_path))
    if set(raw) != set(annotations) or set(raw) != set(track):
        raise ValueError("raw/annotation/raw-SAM2 frame coverage mismatch")

    mask_dir = output / "appearance_masks"
    mask_dir.mkdir(parents=True)
    rows = []
    ranked = []
    for idx in sorted(raw, key=lambda value: float(annotations[value]["time_s"])):
        ann, track_row = annotations[idx], track[idx]
        image_path = rgb_path(raw[idx], root)
        if candidate_root is not None:
            candidate_image = candidate_root.resolve() / "state" / "native_reuse_inputs" / ("rgb" + image_path.suffix.lower())
            if not candidate_image.is_file():
                raise ValueError("candidate RGB materialization missing")
            image_path = candidate_image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable RGB at frame {idx}")
        source_value = track_row.get("mask_path")
        visible = track_row.get("visible") is True
        row = {"frame_idx": idx, "time_s": float(ann["time_s"]),
               "object_id": args.object_id, "visible": visible,
               "semantic_role": "raw_appearance_silhouette_only_not_metric_surface",
               "raw_rgb_path": str(image_path), "raw_rgb_sha256": sha256(image_path)}
        if source_value is None or not visible:
            if visible:
                raise ValueError(f"visible raw SAM2 frame {idx} lacks mask")
            row.update(status="unavailable_not_substituted", source_mask_path=None,
                       appearance_mask_path=None, source_mask_sha256=None,
                       appearance_mask_sha256=None, anchor_eligible=False,
                       rejection="track_marks_frame_not_visible")
            rows.append(row)
            continue
        source = in_run_file(source_value, root, "raw appearance mask")
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2] or not mask.any():
            raise ValueError(f"raw appearance mask invalid at frame {idx}")
        destination = mask_dir / f"frame_{idx:010d}{source.suffix.lower()}"
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError("raw appearance copy changed bytes")
        metrics = score_mask(image, mask, min_area_fraction=args.min_area_fraction,
                             min_border_fraction=args.min_border_fraction,
                             min_sharpness=args.min_sharpness)
        alignment = metric_alignment(metric_object(ann, args.object_id), ann)
        eligible = bool(metrics["eligible"] and alignment is not None)
        rejection = metrics["rejection"] if not metrics["eligible"] else (
            None if alignment is not None else "same_frame_corrected_metric_alignment_unavailable")
        row.update(status="available", source_mask_path=str(source),
                   appearance_mask_path=str(destination), source_mask_sha256=sha256(source),
                   appearance_mask_sha256=sha256(destination),
                   source_size_wh=[mask.shape[1], mask.shape[0]], **metrics)
        row["anchor_eligible"] = eligible
        row["rejection"] = rejection
        rows.append(row)
        if eligible:
            ranked.append((float(metrics["score"]), idx, row, alignment))
    if not ranked:
        raise ValueError("no raw appearance anchor has corrected metric alignment")
    _, selected_idx, selected, alignment = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    manifest = {
        "schema": SCHEMA, "authority": "demo_only_visual_hypothesis",
        "case": args.case_id, "object_id": args.object_id, "track_id": args.track_id,
        "frame_count": len(rows), "source_track": str(raw_track_path),
        "source_track_sha256": sha256(raw_track_path),
        "raw_frame_manifest": str(raw_manifest_path),
        "raw_frame_manifest_sha256": sha256(raw_manifest_path),
        "source_annotations": str(annotations_path),
        "source_annotations_sha256": sha256(annotations_path),
        "depth_order_binding": binding,
        "raw_appearance_and_metric_eligibility_are_distinct": True,
        "metric_owned_masks_consumed": False,
        "frames": rows,
    }
    anchor = {
        "schema": "rgb_scene_depth_order_appearance_anchor_v1",
        "authority": "demo_only_visual_hypothesis", "object_id": args.object_id,
        "selected_frame_idx": selected_idx, "selection": selected,
        "appearance_conditioning": {"raw_rgb": selected["raw_rgb_path"],
                                    "raw_rgb_sha256": selected["raw_rgb_sha256"],
                                    "raw_sam2_mask": selected["appearance_mask_path"],
                                    "raw_sam2_mask_sha256": selected["appearance_mask_sha256"]},
        "metric_alignment": alignment, "planes_are_distinct": True,
        "physical_authority": False,
        "candidate_scores": [{key: row.get(key) for key in
                              ("frame_idx", "anchor_eligible", "rejection", "score", "score_terms")}
                             for row in rows],
    }
    (output / "appearance_mask_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "anchor_decision.json").write_text(json.dumps(anchor, indent=2) + "\n")
    return manifest, anchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "raw-frame-manifest", "annotations", "raw-sam2-track",
                 "depth-order-report", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("object-id", "case-id", "track-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--min-area-fraction", type=float, default=.002)
    parser.add_argument("--min-border-fraction", type=float, default=.005)
    parser.add_argument("--min-sharpness", type=float, default=2.)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args)
    print(args.output_dir / "appearance_mask_manifest.json")
