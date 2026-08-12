#!/usr/bin/env python3
"""Materialize additive, native-format P11 inputs for TRELLIS and SAM3D.

The canonical P11 evidence report is read-only. This script copies its selected
TRELLIS crop and selected full RGB/object-owned mask into an experiment root and
records byte-level provenance. It never rewrites the source report or files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "v19_experimental_p11_dual_geometry_inputs_v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def prepare_new_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty experiment output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("selected object mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def copy_byte_identical(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"copy hash mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "copy": str(destination),
        "sha256": source_hash,
        "bytes": int(source.stat().st_size),
        "byte_identical": True,
    }


def compare_masks(reference_path: Path, comparison_path: Path) -> dict[str, Any]:
    reference = load_mask(reference_path)
    comparison = load_mask(comparison_path)
    if reference.shape != comparison.shape:
        return {
            "reference": str(reference_path),
            "comparison": str(comparison_path),
            "same_shape": False,
            "reference_shape_h_w": [int(v) for v in reference.shape],
            "comparison_shape_h_w": [int(v) for v in comparison.shape],
        }
    intersection = int(np.logical_and(reference, comparison).sum())
    union = int(np.logical_or(reference, comparison).sum())
    xor = int(np.logical_xor(reference, comparison).sum())
    return {
        "reference": str(reference_path),
        "comparison": str(comparison_path),
        "same_shape": True,
        "shape_h_w": [int(v) for v in reference.shape],
        "reference_pixels": int(reference.sum()),
        "comparison_pixels": int(comparison.sum()),
        "xor_pixels": xor,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / union) if union else 1.0,
        "byte_identical": bool(sha256_file(reference_path) == sha256_file(comparison_path)),
    }


def make_mask_review(rgb_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = Image.open(rgb_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")
    if rgb.size != mask_image.size:
        raise RuntimeError(f"RGB/mask size mismatch: {rgb.size} vs {mask_image.size}")

    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    overlay = rgb_arr.astype(np.float32)
    tint = np.asarray([20.0, 210.0, 60.0], dtype=np.float32)
    overlay[mask] = 0.62 * overlay[mask] + 0.38 * tint

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )
    boundary = mask & ~eroded
    overlay[boundary] = np.asarray([255.0, 230.0, 20.0], dtype=np.float32)
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")

    max_panel = 640
    scale = min(1.0, max_panel / max(rgb.size))
    panel_size = (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale))))
    panels = [rgb.resize(panel_size), overlay_image.resize(panel_size)]
    canvas = Image.new("RGB", (panel_size[0] * 2, panel_size[1] + 36), (245, 245, 242))
    canvas.paste(panels[0], (0, 36))
    canvas.paste(panels[1], (panel_size[0], 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "full RGB (unchanged)", fill=(20, 20, 20))
    draw.text((panel_size[0] + 8, 10), "object-owned mask overlay", fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_report_path = require_file(args.evidence_report, "canonical P11 evidence report")
    evidence = load_json(evidence_report_path)
    selected = evidence.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError(f"P11 report has no selected object: {evidence_report_path}")

    frame_idx = int(evidence.get("selected_frame_idx", selected.get("frame_idx", -1)))
    if frame_idx < 0 or int(selected.get("frame_idx", frame_idx)) != frame_idx:
        raise RuntimeError("P11 selected frame fields are missing or inconsistent")

    raw_rgb = require_file(Path(str(selected.get("raw_frame_path", ""))), "selected full RGB")
    selected_mask = require_file(Path(str(selected.get("mask_path", ""))), "selected object-owned mask")
    trellis_info = selected.get("trellis_conditioning_crop")
    if not isinstance(trellis_info, dict):
        raise RuntimeError("P11 selected row has no trellis_conditioning_crop")
    trellis_crop = require_file(Path(str(trellis_info.get("crop_rgba", ""))), "TRELLIS conditioning crop")

    visible_geometry = selected.get("visible_geometry_candidate")
    if not isinstance(visible_geometry, dict):
        visible_geometry = {}
    geometry_mask_text = str(visible_geometry.get("mask_path") or "")
    mask_provenance: dict[str, Any]
    if geometry_mask_text:
        geometry_mask = require_file(Path(geometry_mask_text), "visible-geometry object-owned mask")
        selected_hash = sha256_file(selected_mask)
        geometry_hash = sha256_file(geometry_mask)
        mask_provenance = {
            "selected_mask": str(selected_mask),
            "visible_geometry_mask": str(geometry_mask),
            "selected_sha256": selected_hash,
            "visible_geometry_sha256": geometry_hash,
            "byte_identical": bool(selected_hash == geometry_hash),
        }
        if selected_hash != geometry_hash and not args.allow_unverified_mask_provenance:
            raise RuntimeError(
                "selected.mask_path is not byte-identical to "
                "selected.visible_geometry_candidate.mask_path; refusing SAM3D input"
            )
    else:
        mask_provenance = {
            "selected_mask": str(selected_mask),
            "visible_geometry_mask": None,
            "byte_identical": None,
            "verification": "missing_visible_geometry_mask_path",
        }
        if not args.allow_unverified_mask_provenance:
            raise RuntimeError(
                "selected.visible_geometry_candidate.mask_path is missing; use "
                "--allow-unverified-mask-provenance only for an explicit legacy experiment"
            )

    rgb_image = Image.open(raw_rgb).convert("RGB")
    mask = load_mask(selected_mask)
    if (rgb_image.height, rgb_image.width) != mask.shape:
        raise RuntimeError(
            f"full RGB and object-owned mask are not pixel-aligned: "
            f"RGB={(rgb_image.height, rgb_image.width)} mask={mask.shape}"
        )
    bbox = mask_bbox(mask)

    output_dir = prepare_new_output_dir(args.output_dir)
    inputs_dir = output_dir / "inputs"
    rgb_copy = inputs_dir / f"full_rgb{raw_rgb.suffix.lower() or '.img'}"
    mask_copy = inputs_dir / f"object_owned_mask{selected_mask.suffix.lower() or '.img'}"
    trellis_copy = inputs_dir / f"trellis_conditioning_rgba{trellis_crop.suffix.lower() or '.img'}"

    rgb_record = copy_byte_identical(raw_rgb, rgb_copy)
    mask_record = copy_byte_identical(selected_mask, mask_copy)
    trellis_record = copy_byte_identical(trellis_crop, trellis_copy)

    review_path = output_dir / "review" / "sam3d_native_input_mask_review.png"
    make_mask_review(rgb_copy, mask_copy, review_path)

    comparison = None
    if args.comparison_mask is not None:
        comparison_path = require_file(args.comparison_mask, "comparison mask")
        comparison = compare_masks(selected_mask, comparison_path)

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "build_experimental_p11_dual_geometry_inputs",
        "claim_scope": (
            "additive P11 conditioning split only; canonical evidence and model outputs are read-only"
        ),
        "source_evidence_report": str(evidence_report_path),
        "case": evidence.get("case"),
        "object_id": evidence.get("object_id"),
        "selected_frame_idx": frame_idx,
        "selection_rule": evidence.get("selection_rule"),
        "selection_note": evidence.get("selection_note"),
        "mask_provenance": mask_provenance,
        "selected_input_summary": {
            "image_shape_h_w": [int(rgb_image.height), int(rgb_image.width)],
            "mask_area_px": int(mask.sum()),
            "mask_bbox_xyxy": bbox,
        },
        "conditioning_contracts": {
            "trellis_native": {
                "input_kind": "object_isolated_rgba_crop",
                "image": str(trellis_copy),
                "source_copy": trellis_record,
                "external_depth_or_pointmap": None,
                "pre_model_transform": "canonical_P11_crop_only",
            },
            "sam3d_objects_native": {
                "input_kind": "full_scene_rgb_plus_binary_object_owned_mask",
                "image": str(rgb_copy),
                "mask": str(mask_copy),
                "image_source_copy": rgb_record,
                "mask_source_copy": mask_record,
                "external_pointmap": None,
                "pre_model_crop": None,
                "pre_model_rotation": None,
                "pre_model_rectification": None,
                "model_internal_preprocessing": "unmodified_SAM3D_internal_MoGe_then_joint_mask_crop",
            },
        },
        "comparison_mask_diagnostic": comparison,
        "review": str(review_path),
        "source_artifacts_mutated": False,
    }
    report_path = output_dir / "p11_dual_geometry_inputs_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison-mask",
        type=Path,
        default=None,
        help="Optional raw/pre-ownership mask recorded only as a diagnostic; never used for conditioning.",
    )
    parser.add_argument(
        "--allow-unverified-mask-provenance",
        action="store_true",
        help="Explicit legacy-only override when visible_geometry_candidate.mask_path is absent or differs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
