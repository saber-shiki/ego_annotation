#!/usr/bin/env python3
"""Build depth-order-aware counterfactual object masks.

This consumes the frozen hand-depth-order diagnostic machinery and writes two
counterfactual SAM2-track-compatible mask streams:

* `front_only`: remove only pixels whose MANO first hit is clearly in front of
  the measured object depth;
* `front_plus_ambiguous`: additionally remove near-contact/ambiguous pixels.

Behind-object hand projections and padding-only/no-hit pixels are preserved as
object support. Frames lacking an existing visible-geometry row are copied from
raw SAM2 unchanged and explicitly marked unprocessed. Prediction state is read
only. CPU-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve()
DIAGNOSTIC_PATH = HERE.parent / "diagnose_hand_depth_order_ownership.py"
BASE = Path(
    "/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/"
    "backend_tests/20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/"
    "P0014_84ea2dcc_carton_milk_f2370_2519"
)
GHOST = BASE / "experiments/sam3d_native_ghost_lite_20260826"
SCHEMA = "v19_depth_order_counterfactual_masks_v1"


def load_diagnostic_module() -> Any:
    spec = importlib.util.spec_from_file_location("hand_depth_order_ownership_diag", DIAGNOSTIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load diagnostic module: {DIAGNOSTIC_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=BASE / "measurements/object_geometry/visible_geometry/carton_milk/annotations_v19_visible_geometry.json",
    )
    parser.add_argument(
        "--sam2-track-json",
        type=Path,
        default=BASE / "measurements/object_tracks/sam2_owlv2_box_points/carton_milk/sam2/sam2_track.json",
    )
    parser.add_argument("--output-dir", type=Path, default=GHOST / "depth_order_counterfactual_masks_v1")
    parser.add_argument("--depth-order-tolerance-m", type=float, default=0.005)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def prepare_output(path: Path, replace: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(require_file(path, "JSON").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with require_file(path, "artifact").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = require_file(path, "input")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(require_file(path, "mask")), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return image > 0


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if not len(xs):
        return {"visible": False, "area_px": 0.0, "bbox_xyxy": None, "center_xy": None}
    return {
        "visible": True,
        "area_px": float(len(xs)),
        "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        "center_xy": [float(xs.mean()), float(ys.mean())],
    }


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {path}")


def summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def draw_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color_img = np.zeros_like(image)
    color_img[:] = color
    image[mask] = cv2.addWeighted(image, 1.0 - alpha, color_img, alpha, 0.0)[mask]


def crop(image: np.ndarray, masks: list[np.ndarray], pad: int = 75, size: int = 420) -> np.ndarray:
    combined = np.zeros(image.shape[:2], dtype=bool)
    for mask in masks:
        combined |= mask
    ys, xs = np.where(combined)
    if not len(xs):
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad + 1)
    y1 = min(image.shape[0], int(ys.max()) + pad + 1)
    c = image[y0:y1, x0:x1]
    scale = size / max(c.shape[:2])
    r = cv2.resize(c, (max(1, round(c.shape[1] * scale)), max(1, round(c.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    y = (size - r.shape[0]) // 2
    x = (size - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def make_review(rows: list[dict[str, Any]], review_frames: list[int], output: Path) -> dict[str, Any]:
    selected = [row for row in rows if int(row["frame_idx"]) in set(review_frames)]
    tiles: list[np.ndarray] = []
    for row in selected:
        rgb = cv2.imread(str(require_file(Path(row["rgb_path"]), "review RGB")), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"failed to read RGB: {row['rgb_path']}")
        raw = row["raw_mask"]
        current = row["current_owned_mask"]
        front_only = row["front_only_mask"]
        behind = row["behind_mask"]
        ambiguous = row["ambiguous_mask"]
        front = row["front_mask"]
        p1 = rgb.copy()
        overlay(p1, raw, (255, 80, 30), 0.16)
        overlay(p1, current, (65, 205, 75), 0.30)
        draw_contours(p1, raw, (255, 80, 30), 2)
        draw_contours(p1, current, (65, 205, 75), 2)
        p2 = rgb.copy()
        overlay(p2, behind, (65, 205, 75), 0.52)
        overlay(p2, ambiguous, (35, 220, 255), 0.52)
        overlay(p2, front, (40, 40, 235), 0.52)
        draw_contours(p2, raw, (255, 80, 30), 1)
        p3 = rgb.copy()
        overlay(p3, raw, (255, 80, 30), 0.16)
        overlay(p3, front_only, (65, 205, 75), 0.30)
        draw_contours(p3, raw, (255, 80, 30), 2)
        draw_contours(p3, front_only, (65, 205, 75), 2)
        for panel, title in zip(
            (p1, p2, p3),
            (
                f"f{row['frame_idx']:03d} current owned",
                f"f{row['frame_idx']:03d} classes: green behind, cyan near, red front",
                f"f{row['frame_idx']:03d} front-only counterfactual",
            ),
        ):
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (12, 12, 12), -1)
            cv2.putText(panel, title[:100], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.hstack([crop(p, [raw, current, front_only]) for p in (p1, p2, p3)]))
    if not tiles:
        raise RuntimeError("no review rows selected")
    sheet = np.vstack(tiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write review: {output}")
    return {"path": str(output), "bytes": int(output.stat().st_size), "sha256": sha256_file(output)}


def main() -> None:
    started = time.time()
    args = parse_args()
    diag = load_diagnostic_module()
    output_dir = prepare_output(args.output_dir, bool(args.replace))
    annotations_path = require_file(args.annotations, "visible-geometry annotations")
    annotations = load_json(annotations_path)
    source_track_path = require_file(args.sam2_track_json, "source SAM2 track")
    source_track = load_json(source_track_path)
    frames = [frame for frame in annotations.get("frames", []) if isinstance(frame, dict)]
    if not frames:
        raise RuntimeError("annotations contain no frames")
    first_visible = next(
        (frame["objects"][0]["visible_geometry_candidate"] for frame in frames if (frame.get("objects") or [{}])[0].get("visible_geometry_candidate")),
        None,
    )
    if first_visible is None:
        raise RuntimeError("annotations contain no visible geometry rows")
    depth_path = require_file(Path(str(first_visible["depth_npz"])), "depth archive")
    depth_archive = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    bridge_cache: dict[Path, Any] = {}
    hawor_cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    processed: dict[int, dict[str, Any]] = {}
    try:
        depth_row_by_idx = {int(value): i for i, value in enumerate(np.asarray(depth_archive["frame_idx"], dtype=np.int64).tolist())}
        diag_args = SimpleNamespace(depth_order_tolerance_m=float(args.depth_order_tolerance_m))
        for frame in frames:
            frame_idx = int(frame["frame_idx"])
            visible = (frame.get("objects") or [{}])[0].get("visible_geometry_candidate")
            track_row = source_track.get(str(frame_idx))
            if not isinstance(track_row, dict) or not track_row.get("mask_path"):
                continue
            raw_mask = read_mask(Path(str(track_row["mask_path"])))
            if isinstance(visible, dict):
                row = diag.process_frame(frame, depth_archive, depth_row_by_idx, bridge_cache, hawor_cache, diag_args)
                visual = row["_visual"]
                front = visual["front"]
                behind = visual["behind"]
                ambiguous = visual["ambiguous"]
                no_hit = visual["no_hit"]
                current_owned = visual["owned_mask"]
                raw_mask = visual["raw_mask"]
                front_only = raw_mask & ~front
                front_plus_ambiguous = raw_mask & ~(front | ambiguous)
                status = "depth_order_classified"
            else:
                front = np.zeros_like(raw_mask)
                behind = np.zeros_like(raw_mask)
                ambiguous = np.zeros_like(raw_mask)
                no_hit = np.zeros_like(raw_mask)
                current_owned = raw_mask
                front_only = raw_mask
                front_plus_ambiguous = raw_mask
                status = "no_visible_geometry_row_raw_mask_copied"
            row_out = {
                "frame_idx": frame_idx,
                "status": status,
                "rgb_path": str(frame.get("raw_frame_path")),
                "raw_mask": raw_mask,
                "current_owned_mask": current_owned,
                "front_only_mask": front_only,
                "front_plus_ambiguous_mask": front_plus_ambiguous,
                "front_mask": front,
                "behind_mask": behind,
                "ambiguous_mask": ambiguous,
                "no_hit_mask": no_hit,
                "counts": {
                    "raw_pixels": int(np.count_nonzero(raw_mask)),
                    "current_owned_pixels": int(np.count_nonzero(current_owned)),
                    "front_only_pixels": int(np.count_nonzero(front_only)),
                    "front_plus_ambiguous_pixels": int(np.count_nonzero(front_plus_ambiguous)),
                    "front_removed_pixels": int(np.count_nonzero(front)),
                    "behind_preserved_pixels": int(np.count_nonzero(behind)),
                    "ambiguous_preserved_pixels": int(np.count_nonzero(ambiguous)),
                    "no_hit_preserved_pixels": int(np.count_nonzero(no_hit)),
                },
            }
            rows.append(row_out)
            processed[frame_idx] = row_out
    finally:
        depth_archive.close()
        for cache in (bridge_cache, hawor_cache):
            for archive in cache.values():
                archive.close()

    variants = {
        "front_only": "remove only hand_in_front first-hit pixels; preserve behind, ambiguous, padding-only/no-hit",
        "front_plus_ambiguous": "remove hand_in_front and ambiguous_near_contact; preserve behind and padding-only/no-hit",
    }
    variant_reports: dict[str, Any] = {}
    for variant, semantics in variants.items():
        mask_dir = output_dir / variant / "sam2_masks"
        track_out = json.loads(json.dumps(source_track))
        for frame_key, source_row in source_track.items():
            frame_idx = int(frame_key)
            row = processed.get(frame_idx)
            if row is None:
                continue
            mask = row["front_only_mask"] if variant == "front_only" else row["front_plus_ambiguous_mask"]
            mask_path = mask_dir / f"{frame_idx:06d}.png"
            write_mask(mask_path, mask)
            geometry = mask_geometry(mask)
            track_row = dict(source_row)
            track_row["mask_path"] = str(mask_path)
            track_row.update(geometry)
            track_row["counterfactual_mask_provenance"] = {
                "schema": SCHEMA,
                "variant": variant,
                "source_mask_path": str(source_row.get("mask_path")),
                "depth_order_tolerance_m": float(args.depth_order_tolerance_m),
                "status": row["status"],
            }
            track_out[frame_key] = track_row
        track_path = output_dir / variant / "sam2_track_depth_order_counterfactual.json"
        track_path.write_text(json.dumps(track_out, indent=2), encoding="utf-8")
        retained = [
            row["counts"]["front_only_pixels" if variant == "front_only" else "front_plus_ambiguous_pixels"] / max(1, row["counts"]["raw_pixels"])
            for row in rows
        ]
        recovered = [
            (row["counts"]["front_only_pixels" if variant == "front_only" else "front_plus_ambiguous_pixels"] - row["counts"]["current_owned_pixels"])
            for row in rows
        ]
        variant_reports[variant] = {
            "semantics": semantics,
            "mask_dir": str(mask_dir),
            "track_json": str(track_path),
            "track_json_sha256": sha256_file(track_path),
            "retained_raw_fraction": summarize(retained),
            "recovered_pixels_vs_current_owned": summarize([float(v) for v in recovered]),
            "frame_count": len(rows),
        }

    review_frames = [0, 30, 50, 70, 92, 103, 110, 121, 130, 146, 149]
    review = make_review(rows, review_frames, output_dir / "counterfactual_mask_review.jpg")
    serializable_rows = []
    for row in rows:
        out = dict(row)
        for key in ("raw_mask", "current_owned_mask", "front_only_mask", "front_plus_ambiguous_mask", "front_mask", "behind_mask", "ambiguous_mask", "no_hit_mask"):
            out.pop(key, None)
        serializable_rows.append(out)
    report_path = output_dir / "depth_order_counterfactual_masks_report.json"
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "depth_order_aware_counterfactual_object_masks",
        "claim_scope": "Counterfactual mask inputs for visible-geometry experiments; prediction inputs were read-only.",
        "compute_contract": {"local_gpu_used": False, "model_inference_run": False, "prediction_inputs_mutated": False},
        "inputs": {
            "annotations": file_ref(annotations_path),
            "sam2_track": file_ref(source_track_path),
            "depth_archive": file_ref(depth_path),
            "diagnostic_script": file_ref(DIAGNOSTIC_PATH),
        },
        "depth_order_tolerance_m": float(args.depth_order_tolerance_m),
        "variants": variant_reports,
        "rows": serializable_rows,
        "review_image": review,
        "elapsed_s": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "report": str(report_path), "review": review["path"], "variants": variant_reports}, indent=2))


if __name__ == "__main__":
    main()
