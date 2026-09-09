#!/usr/bin/env python3
"""Build isolated all-frame appearance evidence for the additive visual demo.

This stage deliberately keeps raw SAM2 appearance silhouettes separate from the
object-owned masks and metric surfels in the core visible-geometry annotations.
It performs no model inference and grants no physical/contact authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def in_run_file(path: str | Path, run_root: Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_relative_to(run_root) or not value.is_file():
        raise ValueError(f"{label} missing or outside run root: {path}")
    return value


def indexed_rows(rows, label: str) -> dict[int, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must contain nonempty rows")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get("frame_idx"), bool):
            raise ValueError(f"invalid {label} row")
        frame_idx = int(row["frame_idx"])
        if frame_idx in result:
            raise ValueError(f"duplicate {label} frame {frame_idx}")
        result[frame_idx] = row
    return result


def track_rows(track: dict) -> dict[int, dict]:
    result = {}
    for key, row in track.items():
        try:
            frame_idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SAM2 track key is not an integer: {key!r}") from exc
        if str(frame_idx) != str(key) or frame_idx in result or not isinstance(row, dict):
            raise ValueError(f"invalid/duplicate SAM2 track frame {key!r}")
        result[frame_idx] = row
    if not result:
        raise ValueError("empty SAM2 track")
    return result


def rgb_path(row: dict, root: Path) -> Path:
    raw = row.get("raw_frame_path") or row.get("rgb") or row.get("image_path")
    if not raw:
        raise ValueError(f"raw frame {row.get('frame_idx')} has no RGB path")
    return in_run_file(raw, root, "raw RGB")


def metric_object(row: dict, object_id: str) -> dict | None:
    objects = row.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"annotation frame {row.get('frame_idx')} has invalid objects")
    matches = [obj for obj in objects if isinstance(obj, dict) and obj.get("object_id") == object_id]
    if len(matches) > 1:
        raise ValueError(f"duplicate object {object_id!r} at frame {row.get('frame_idx')}")
    return matches[0] if matches else None


def metric_alignment(obj: dict | None, frame: dict) -> dict | None:
    if obj is None:
        return None
    geom = obj.get("visible_geometry_candidate")
    camera = frame.get("camera")
    if not isinstance(geom, dict) or not isinstance(camera, dict):
        return None
    world = np.asarray(geom.get("world_vertices_sample_m", []), dtype=float)
    cam = np.asarray(geom.get("camera_vertices_sample_m", []), dtype=float)
    centroid = np.asarray(geom.get("centroid_world_m", []), dtype=float)
    transform = np.asarray(camera.get("T_world_camera_metric", []), dtype=float)
    if (world.ndim != 2 or world.shape[1:] != (3,) or len(world) < 3
            or cam.shape != world.shape or centroid.shape != (3,)
            or transform.shape != (4, 4) or not all(np.isfinite(x).all() for x in (world, cam, centroid, transform))):
        return None
    return {
        "camera": camera,
        "centroid_world_m": centroid.tolist(),
        "world_surfels_m": world.tolist(),
        "camera_surfels_m": cam.tolist(),
        "owned_metric_mask_path": obj.get("mask_path"),
        "role": "same_frame_metric_alignment_only_not_appearance_conditioning",
    }


def score_mask(image: np.ndarray, mask: np.ndarray, *, min_area_fraction: float,
               min_border_fraction: float, min_sharpness: float) -> dict:
    support = mask > 0
    height, width = support.shape
    area = int(support.sum())
    ys, xs = np.where(support)
    if not area:
        return {"eligible": False, "rejection": "empty_mask", "area_px": 0}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    area_fraction = area / float(width * height)
    border_fraction = min(x0, y0, width - 1 - x1, height - 1 - y1) / float(min(width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian[dilated])) if np.any(dilated) else 0.0
    reasons = []
    if area_fraction < min_area_fraction:
        reasons.append("mask_too_small")
    if border_fraction < min_border_fraction:
        reasons.append("mask_near_image_border")
    if sharpness < min_sharpness:
        reasons.append("appearance_crop_too_blurred")
    # Fixed normalized terms. Log compression prevents sharpness magnitude from
    # overwhelming silhouette area/clearance while remaining label-independent.
    sharpness_term = min(1.0, np.log1p(max(0.0, sharpness)) / np.log(1001.0))
    clearance_term = min(1.0, border_fraction / 0.2)
    area_term = min(1.0, area_fraction / 0.25)
    score = 0.45 * area_term + 0.35 * sharpness_term + 0.20 * clearance_term
    return {
        "eligible": not reasons,
        "rejection": None if not reasons else "+".join(reasons),
        "area_px": area,
        "area_fraction": area_fraction,
        "bbox_xyxy": [x0, y0, x1 + 1, y1 + 1],
        "border_clearance_fraction": border_fraction,
        "laplacian_variance_dilated_mask": sharpness,
        "score_terms": {
            "area": area_term,
            "sharpness": sharpness_term,
            "border_clearance": clearance_term,
            "weights": {"area": 0.45, "sharpness": 0.35, "border_clearance": 0.20},
        },
        "score": float(score),
    }


def _build(run_root: Path, object_id: str, raw_manifest_path: Path, annotations_path: Path,
          track_path: Path, output_dir: Path, *, case_id: str | None = None,
          track_id: str | None = None, published_output_dir: Path | None = None, min_area_fraction: float = 0.002,
          min_border_fraction: float = 0.005, min_sharpness: float = 2.0) -> tuple[dict, dict]:
    root = run_root.resolve()
    if not SAFE_ID.fullmatch(object_id):
        raise ValueError("unsafe object id")
    output_dir = output_dir.resolve()
    allowed = root / "measurements" / "demo_visual" / object_id
    if not output_dir.is_relative_to(allowed) or output_dir.exists():
        raise ValueError("new output directory under measurements/demo_visual/<object> required")
    raw_manifest_path = in_run_file(raw_manifest_path, root, "raw frame manifest")
    annotations_path = in_run_file(annotations_path, root, "annotations")
    track_path = in_run_file(track_path, root, "SAM2 track")
    raw = indexed_rows(load(raw_manifest_path).get("frames"), "raw manifest")
    annotation_payload = load(annotations_path)
    adapter = annotation_payload.get("v19_visible_geometry_adapter")
    if case_id is not None or track_id is not None:
        if (not isinstance(adapter, dict) or adapter.get("case") != case_id
                or adapter.get("track_id") != track_id
                or adapter.get("object_id") != object_id
                or Path(adapter.get("sam2_track", "")).expanduser().resolve() != track_path):
            raise ValueError("appearance track does not match bound visible-geometry adapter")
    annotations = indexed_rows(annotation_payload.get("frames"), "annotations")
    track = track_rows(load(track_path))
    expected = set(raw)
    if set(annotations) != expected or set(track) != expected:
        raise ValueError("raw/annotation/SAM2 track frame coverage mismatch")

    mask_dir = output_dir / "appearance_masks"
    mask_dir.mkdir(parents=True)
    records = []
    ranked = []
    for frame_idx in sorted(expected, key=lambda idx: float(annotations[idx]["time_s"])):
        raw_row, ann_row, track_row = raw[frame_idx], annotations[frame_idx], track[frame_idx]
        if int(ann_row["frame_idx"]) != frame_idx:
            raise ValueError("annotation frame mismatch")
        image_path = rgb_path(raw_row, root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable raw RGB: {image_path}")
        source_mask = track_row.get("mask_path")
        visible = track_row.get("visible") is True
        record = {
            "frame_idx": frame_idx,
            "time_s": float(ann_row["time_s"]),
            "object_id": object_id,
            "visible": visible,
            "semantic_role": "appearance_silhouette_only_not_metric_surface",
            "raw_rgb_path": str(image_path),
            "raw_rgb_sha256": sha256(image_path),
        }
        if source_mask is None:
            if visible:
                raise ValueError(f"visible SAM2 frame {frame_idx} lacks mask")
            record.update(status="unavailable_not_substituted", source_mask_path=None,
                          appearance_mask_path=None, source_mask_sha256=None,
                          appearance_mask_sha256=None, anchor_eligible=False,
                          rejection="track_marks_frame_not_visible")
            records.append(record)
            continue
        source = in_run_file(source_mask, root, "SAM2 appearance mask")
        if not visible:
            record.update(
                status="unavailable_not_substituted",
                source_mask_path=str(source),
                source_mask_sha256=sha256(source),
                appearance_mask_path=None,
                appearance_mask_sha256=None,
                anchor_eligible=False,
                rejection="invisible_track_row_has_stale_mask_not_consumed",
            )
            records.append(record)
            continue
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError(f"mask/RGB dimensions differ at frame {frame_idx}")
        if not np.any(mask) and visible:
            raise ValueError(f"visible SAM2 mask is empty at frame {frame_idx}")
        destination = mask_dir / f"frame_{frame_idx:010d}{source.suffix.lower()}"
        shutil.copyfile(source, destination)
        source_hash = sha256(source)
        copy_hash = sha256(destination)
        if source_hash != copy_hash:
            raise RuntimeError("appearance mask copy is not byte-identical")
        metrics = score_mask(image, mask, min_area_fraction=min_area_fraction,
                             min_border_fraction=min_border_fraction,
                             min_sharpness=min_sharpness)
        alignment = metric_alignment(metric_object(ann_row, object_id), ann_row)
        eligible = bool(metrics["eligible"] and alignment is not None)
        rejection = metrics["rejection"] if not metrics["eligible"] else (
            None if alignment is not None else "same_frame_metric_alignment_unavailable")
        record.update(
            status="available",
            source_mask_path=str(source),
            appearance_mask_path=str((published_output_dir or output_dir) / "appearance_masks" / destination.name),
            source_mask_sha256=source_hash,
            appearance_mask_sha256=copy_hash,
            source_size_wh=[int(mask.shape[1]), int(mask.shape[0])],
            **metrics,
        )
        record["anchor_eligible"] = eligible
        record["rejection"] = rejection
        records.append(record)
        if eligible:
            ranked.append((float(metrics["score"]), frame_idx, record, alignment))
    if not ranked:
        raise ValueError("no appearance anchor passes generic quality and same-frame metric floors")
    # Highest score, then earliest frame ID for deterministic ties.
    _, selected_idx, selected, alignment = sorted(ranked, key=lambda value: (-value[0], value[1]))[0]
    manifest = {
        "schema": "v19_demo_appearance_mask_manifest_v2",
        "authority": "demo_only_visual_hypothesis",
        "object_id": object_id,
        "frame_count": len(records),
        "source_track": str(track_path),
        "source_track_sha256": sha256(track_path),
        "raw_frame_manifest": str(raw_manifest_path),
        "raw_frame_manifest_sha256": sha256(raw_manifest_path),
        "source_annotations": str(annotations_path),
        "source_annotations_sha256": sha256(annotations_path),
        "metric_owned_masks_consumed": False,
        "core_sources_modified": False,
        "frames": records,
    }
    anchor = {
        "schema": "v19_demo_visual_anchor_v2",
        "authority": "demo_only_visual_hypothesis",
        "object_id": object_id,
        "selected_frame_idx": selected_idx,
        "selection": selected,
        "appearance_conditioning": {
            "raw_rgb": selected["raw_rgb_path"],
            "raw_rgb_sha256": selected["raw_rgb_sha256"],
            "raw_sam2_mask": selected["appearance_mask_path"],
            "raw_sam2_mask_sha256": selected["appearance_mask_sha256"],
        },
        "metric_alignment": alignment,
        "planes_are_distinct": True,
        "sam3d_reconditioning": {
            "status": "optional_external_model_stage_not_executed_by_this_cpu_builder",
            "required_authority": "demo_only_visual_hypothesis",
            "conditioning_must_use": "appearance_conditioning",
            "metric_bridge_must_use": "metric_alignment",
            "core_completion_must_not_be_overwritten": True,
        },
        "thresholds": {
            "min_area_fraction": min_area_fraction,
            "min_border_clearance_fraction": min_border_fraction,
            "min_laplacian_variance": min_sharpness,
        },
        "candidate_scores": [
            {k: row.get(k) for k in ("frame_idx", "anchor_eligible", "rejection", "score", "score_terms")}
            for row in records
        ],
        "physical_authority": False,
        "generated_faces_contact_eligible": False,
        "generated_faces_collision_eligible": False,
        "generated_faces_sdf_eligible": False,
        "generated_faces_nonpenetration_eligible": False,
    }
    (output_dir / "appearance_mask_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "anchor_decision.json").write_text(json.dumps(anchor, indent=2) + "\n")
    return manifest, anchor


def build(run_root, object_id, raw_manifest_path, annotations_path, track_path,
          output_dir, **kwargs):
    """Create-only atomic D00 publication; failed staging remains debug evidence."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("appearance output already exists")
    staging = output_dir.with_name("." + output_dir.name + ".staging-" + uuid.uuid4().hex)
    published = kwargs.pop("published_output_dir", None) or output_dir
    try:
        result = _build(run_root, object_id, raw_manifest_path, annotations_path,
                        track_path, staging, published_output_dir=published, **kwargs)
        if output_dir.exists():
            raise ValueError("appearance output appeared during staging")
        staging.rename(output_dir)
        return result
    except Exception:
        if staging.exists():
            staging.rename(staging.with_name(staging.name.replace(".staging-", ".failed-")))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "raw-frame-manifest", "annotations", "sam2-track", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--published-output-dir", type=Path)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--min-area-fraction", type=float, default=0.002)
    parser.add_argument("--min-border-fraction", type=float, default=0.005)
    parser.add_argument("--min-sharpness", type=float, default=2.0)
    args = parser.parse_args()
    build(args.run_root, args.object_id, args.raw_frame_manifest, args.annotations,
          args.sam2_track, args.output_dir, case_id=args.case_id,
          track_id=args.track_id, published_output_dir=args.published_output_dir, min_area_fraction=args.min_area_fraction,
          min_border_fraction=args.min_border_fraction, min_sharpness=args.min_sharpness)
    print(args.output_dir / "appearance_mask_manifest.json")


if __name__ == "__main__":
    main()
