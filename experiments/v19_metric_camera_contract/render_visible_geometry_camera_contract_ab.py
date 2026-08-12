#!/usr/bin/env python3
"""Render a prediction-only old-K vs sensor-K visible-geometry reprojection review.

Both branches are projected with the resolved sensor camera into the same manifest
RGB plane. The object-owned mask is review evidence only; no evaluator/GT input
is consumed. This diagnoses whether changing the camera contract actually changes
the visible 3D mechanism rather than only JSON metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_geometry(frame: dict[str, Any], object_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for row in frame.get("objects") or []:
        if not isinstance(row, dict):
            continue
        if row.get("object_id") == object_id or row.get("track_id") == object_id:
            geometry = row.get("visible_geometry_candidate")
            if not isinstance(geometry, dict):
                raise RuntimeError(f"frame {frame.get('frame_idx')} object has no visible geometry")
            return row, geometry
    raise RuntimeError(f"frame {frame.get('frame_idx')} lacks object {object_id}")


def project(points_camera: np.ndarray, K: np.ndarray, A_manifest_from_source: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-8)
    points = points[valid]
    uv_source = np.column_stack(
        [
            K[0] * points[:, 0] / points[:, 2] + K[2],
            K[1] * points[:, 1] / points[:, 2] + K[3],
            np.ones(len(points), dtype=np.float64),
        ]
    )
    return (uv_source @ A_manifest_from_source.T)[:, :2]


def draw_points(image: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    height, width = image.shape[:2]
    rounded = np.rint(uv).astype(np.int64)
    keep = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    for x, y in rounded[keep]:
        cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def mask_membership(uv: np.ndarray, mask: np.ndarray) -> float:
    rounded = np.rint(uv).astype(np.int64)
    keep = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    if not np.any(keep):
        return 0.0
    valid = rounded[keep]
    return float(np.mean(mask[valid[:, 1], valid[:, 0]]))


def distance_to_mask(uv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    rounded = np.rint(uv).astype(np.int64)
    result = np.full(len(rounded), np.nan, dtype=np.float64)
    keep = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    valid = rounded[keep]
    result[keep] = distance[valid[:, 1], valid[:, 0]]
    return result


def banner(tile: np.ndarray, lines: list[str]) -> np.ndarray:
    top = np.full((70, tile.shape[1], 3), 18, dtype=np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(top, text, (10, 22 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    return np.vstack([top, tile])


def render(args: argparse.Namespace) -> dict[str, Any]:
    old = load_json(args.old_annotations)
    new = load_json(args.new_annotations)
    manifest = load_json(args.raw_frame_manifest)
    contract = load_json(args.camera_contract)
    old_frames = {int(row["frame_idx"]): row for row in old.get("frames") or []}
    new_frames = {int(row["frame_idx"]): row for row in new.get("frames") or []}
    raw_frames = {int(row["frame_idx"]): row for row in manifest.get("frames") or []}
    frame_ids = sorted(set(old_frames) & set(new_frames) & set(raw_frames))
    if not frame_ids:
        raise RuntimeError("no common frames")
    K = np.asarray(contract["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    A_manifest = np.asarray(contract["image_planes"]["manifest_rgb"]["A_plane_from_calibration"], dtype=np.float64)
    if K.shape != (4,) or A_manifest.shape != (3, 3):
        raise RuntimeError("invalid camera contract")

    output_dir = args.output_dir.resolve()
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    rows: list[dict[str, Any]] = []
    review_frames = set(int(v) for v in args.review_frames)
    review_images: list[np.ndarray] = []
    review_labels: list[str] = []
    for frame_idx in frame_ids:
        raw_path = Path(str(raw_frames[frame_idx].get("raw_frame_path") or raw_frames[frame_idx].get("rgb")))
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(raw_path)
        old_obj, old_geom = object_geometry(old_frames[frame_idx], args.object_id)
        new_obj, new_geom = object_geometry(new_frames[frame_idx], args.object_id)
        old_mask_path = Path(str(old_obj["mask_path"]))
        new_mask_path = Path(str(new_obj["mask_path"]))
        old_mask_bytes_equal = old_mask_path.read_bytes() == new_mask_path.read_bytes()
        mask = cv2.imread(str(new_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(new_mask_path)
        mask_bool = mask > 0
        if mask_bool.shape != image.shape[:2]:
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        old_points = np.asarray(old_geom["camera_vertices_sample_m"], dtype=np.float64)
        new_points = np.asarray(new_geom["camera_vertices_sample_m"], dtype=np.float64)
        old_uv = project(old_points, K, A_manifest)
        new_uv = project(new_points, K, A_manifest)
        old_inside = mask_membership(old_uv, mask_bool)
        new_inside = mask_membership(new_uv, mask_bool)
        old_distance = distance_to_mask(old_uv, mask_bool)
        new_distance = distance_to_mask(new_uv, mask_bool)
        old_med_distance = float(np.nanmedian(old_distance))
        new_med_distance = float(np.nanmedian(new_distance))

        tint = image.copy()
        tint[mask_bool] = (40, 110, 40)
        base = image.copy()
        base[mask_bool] = cv2.addWeighted(tint, 0.30, image, 0.70, 0.0)[mask_bool]
        old_tile = base.copy()
        new_tile = base.copy()
        draw_points(old_tile, old_uv, (30, 40, 235), int(args.point_radius))
        draw_points(new_tile, new_uv, (30, 220, 255), int(args.point_radius))
        old_tile = banner(
            old_tile,
            [
                f"frame {frame_idx:03d}  OLD estimated-K 3D -> SENSOR K",
                f"inside owned mask {old_inside:.3f} | outside-distance median {old_med_distance:.2f}px",
                "red: reprojected visible 3D | green: prediction-owned support",
            ],
        )
        new_tile = banner(
            new_tile,
            [
                f"frame {frame_idx:03d}  NEW sensor-K 3D -> SENSOR K",
                f"inside owned mask {new_inside:.3f} | outside-distance median {new_med_distance:.2f}px",
                "yellow: reprojected visible 3D | green: prediction-owned support",
            ],
        )
        combined = np.hstack([old_tile, new_tile])
        output_path = frame_dir / f"{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(output_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
            raise RuntimeError(f"failed to write {output_path}")
        if writer is None:
            writer = cv2.VideoWriter(
                str(output_dir / "old_estimated_k_vs_sensor_k_visible_geometry_reprojection.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps),
                (combined.shape[1], combined.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError("failed to open review video writer")
        writer.write(combined)
        if frame_idx in review_frames:
            review_images.append(combined)
            review_labels.append(str(frame_idx))
        rows.append(
            {
                "frame_idx": frame_idx,
                "old_mask_bytes_equal_new_mask": old_mask_bytes_equal,
                "old_points_projected_with_sensor_k_inside_owned_mask_fraction": old_inside,
                "new_points_projected_with_sensor_k_inside_owned_mask_fraction": new_inside,
                "inside_fraction_delta_new_minus_old": new_inside - old_inside,
                "old_projected_outside_distance_median_px": old_med_distance,
                "new_projected_outside_distance_median_px": new_med_distance,
            }
        )
    if writer is not None:
        writer.release()

    if review_images:
        target_width = min(1920, max(image.shape[1] for image in review_images))
        normalized = []
        for image in review_images:
            if image.shape[1] != target_width:
                image = cv2.resize(image, (target_width, int(round(image.shape[0] * target_width / image.shape[1]))), interpolation=cv2.INTER_AREA)
            normalized.append(image)
        sheet = np.vstack(normalized)
        cv2.imwrite(str(output_dir / "multiframe_reprojection_qc.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    old_inside_values = np.asarray([row["old_points_projected_with_sensor_k_inside_owned_mask_fraction"] for row in rows])
    new_inside_values = np.asarray([row["new_points_projected_with_sensor_k_inside_owned_mask_fraction"] for row in rows])
    report = {
        "status": "ok",
        "method": "render_v19_camera_contract_visible_geometry_ab",
        "claim_scope": (
            "prediction-only mechanism review: old and new visible 3D are projected through the same resolved sensor K; "
            "the object-owned mask is prediction evidence, not GT; no evaluator labels are consumed"
        ),
        "inputs": {
            "old_annotations": str(args.old_annotations.resolve()),
            "new_annotations": str(args.new_annotations.resolve()),
            "raw_frame_manifest": str(args.raw_frame_manifest.resolve()),
            "camera_contract": str(args.camera_contract.resolve()),
        },
        "outputs": {
            "video": str(output_dir / "old_estimated_k_vs_sensor_k_visible_geometry_reprojection.mp4"),
            "multiframe_qc": str(output_dir / "multiframe_reprojection_qc.jpg"),
            "frames": str(frame_dir),
        },
        "frame_count": len(rows),
        "sensor_intrinsics_fx_fy_cx_cy": K.tolist(),
        "all_object_owned_masks_byte_equal": bool(all(row["old_mask_bytes_equal_new_mask"] for row in rows)),
        "inside_owned_mask_fraction": {
            "old_projected_with_sensor_k_median": float(np.median(old_inside_values)),
            "old_projected_with_sensor_k_p10": float(np.percentile(old_inside_values, 10.0)),
            "new_projected_with_sensor_k_median": float(np.median(new_inside_values)),
            "new_projected_with_sensor_k_p10": float(np.percentile(new_inside_values, 10.0)),
            "new_minus_old_median": float(np.median(new_inside_values - old_inside_values)),
            "new_better_frame_count": int(np.count_nonzero(new_inside_values > old_inside_values)),
            "equal_frame_count": int(np.count_nonzero(new_inside_values == old_inside_values)),
        },
        "rows": rows,
    }
    write_json(output_dir / "camera_contract_visible_geometry_ab_report.json", report)
    report["outputs_sha256"] = {
        "video": sha256_file(Path(report["outputs"]["video"])),
        "multiframe_qc": sha256_file(Path(report["outputs"]["multiframe_qc"])),
    }
    write_json(output_dir / "camera_contract_visible_geometry_ab_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-annotations", type=Path, required=True)
    parser.add_argument("--new-annotations", type=Path, required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-frames", type=int, nargs="+", default=[0, 50, 109, 149])
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    report = render(parse_args())
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
