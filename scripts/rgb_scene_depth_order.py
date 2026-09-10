#!/usr/bin/env python3
"""Build a hash-bound SAM2 track with front-only MANO depth ownership.

The raw SAM2 masks remain immutable appearance evidence.  For metric lifting,
only pixels where the nearest actual MANO triangle first hit is clearly closer
than same-ray metric depth are removed.  Behind-hand, near-contact, invalid-depth
and no-hit pixels remain in the appearance-shaped owned mask; downstream P09
still applies its existing robust depth-support quarantine before lifting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_v19_visible_geometry_from_sam2_depth import (
    load_depth_npz,
    validate_depth_camera_contract_binding,
)
from v19_camera_contract import K_from_intrinsics, load_contract, plane_intrinsics

SCHEMA = "rgb_scene_depth_order_owned_sam2_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"required input is not a file: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256(path)}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def indexed_frames(rows: Any, label: str) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must have nonempty frames")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get("frame_idx"), bool):
            raise ValueError(f"malformed {label} frame")
        idx = int(row["frame_idx"])
        if idx in result:
            raise ValueError(f"duplicate {label} frame {idx}")
        result[idx] = row
    return result


def indexed_track(value: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for key, row in value.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SAM2 track key is not an integer: {key!r}") from exc
        if str(idx) != str(key) or idx in result or not isinstance(row, dict):
            raise ValueError(f"malformed/duplicate SAM2 frame {key!r}")
        result[idx] = row
    if not result:
        raise ValueError("empty SAM2 track")
    return result


def first_hit_z(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray,
                shape: tuple[int, int], near_m: float = 1e-6) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize camera-z using perspective-correct screen barycentrics.

    Triangles crossing the near plane are clipped in camera coordinates.  A
    nonfinite face is diagnosed and skipped (therefore cannot erase appearance),
    while malformed array/topology/camera contracts are mechanical failures.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces)
    K = np.asarray(K, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise ValueError("vertices must have shape [V,3]")
    if (faces_array.ndim != 2 or faces_array.shape[1:] != (3,) or not len(faces_array)
            or faces_array.dtype.kind not in "iu" or int(faces_array.min()) < 0
            or int(faces_array.max()) >= len(vertices)):
        raise ValueError("malformed triangle topology")
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-12)
            or float(K[0, 0]) <= 0 or float(K[1, 1]) <= 0
            or abs(float(K[1, 0])) > 1e-12):
        raise ValueError("invalid pinhole camera K")
    if (not isinstance(shape, tuple) or len(shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
            or not np.isfinite(near_m) or near_m <= 0):
        raise ValueError("invalid raster shape or near plane")

    faces = faces_array.astype(np.int64, copy=False)
    zbuf = np.full(shape, np.inf, dtype=np.float64)
    report = {
        "faces_total": int(len(faces)), "nonfinite_faces": 0,
        "behind_near_faces": 0, "clipped_faces": 0,
        "degenerate_faces": 0, "offscreen_faces": 0,
    }
    height, width = shape
    for face in faces:
        triangle = vertices[face]
        if not np.isfinite(triangle).all():
            report["nonfinite_faces"] += 1
            continue
        if np.all(triangle[:, 2] < near_m):
            report["behind_near_faces"] += 1
            continue
        polygon = list(triangle)
        if np.any(triangle[:, 2] < near_m):
            report["clipped_faces"] += 1
            polygon = []
            for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
                start_in, end_in = start[2] >= near_m, end[2] >= near_m
                if start_in:
                    polygon.append(start)
                if start_in != end_in:
                    alpha = (near_m - start[2]) / (end[2] - start[2])
                    polygon.append(start + alpha * (end - start))
        for fan_index in range(1, len(polygon) - 1):
            tri = np.asarray([polygon[0], polygon[fan_index], polygon[fan_index + 1]])
            homogeneous = tri @ K.T
            uv = homogeneous[:, :2] / homogeneous[:, 2:3]
            lower = np.maximum(np.floor(uv.min(axis=0)), [0, 0]).astype(int)
            upper = np.minimum(np.ceil(uv.max(axis=0)), [width - 1, height - 1]).astype(int)
            if np.any(lower > upper):
                report["offscreen_faces"] += 1
                continue
            x0, y0 = lower
            x1, y1 = upper
            a, b, c = uv
            denominator = ((b[1] - c[1]) * (a[0] - c[0])
                           + (c[0] - b[0]) * (a[1] - c[1]))
            if abs(float(denominator)) < 1e-12:
                report["degenerate_faces"] += 1
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            wa = ((b[1] - c[1]) * (xx - c[0])
                  + (c[0] - b[0]) * (yy - c[1])) / denominator
            wb = ((c[1] - a[1]) * (xx - c[0])
                  + (a[0] - c[0]) * (yy - c[1])) / denominator
            wc = 1.0 - wa - wb
            inside = (wa >= -1e-10) & (wb >= -1e-10) & (wc >= -1e-10)
            inverse_z = wa / tri[0, 2] + wb / tri[1, 2] + wc / tri[2, 2]
            depth = np.divide(1.0, inverse_z, out=np.full_like(inverse_z, np.inf),
                              where=inverse_z > 0)
            view = zbuf[y0:y1 + 1, x0:x1 + 1]
            np.minimum(view, np.where(inside, depth, np.inf), out=view)
    report["hit_pixels"] = int(np.isfinite(zbuf).sum())
    return zbuf, report


def nearest_hand_first_hits(hands: list[tuple[np.ndarray, np.ndarray]], K: np.ndarray,
                            shape: tuple[int, int], near_m: float = 1e-6
                            ) -> tuple[np.ndarray, list[dict[str, int]]]:
    nearest = np.full(shape, np.inf, dtype=np.float64)
    reports = []
    for vertices, faces in hands:
        hand_z, report = first_hit_z(vertices, faces, K, shape, near_m)
        np.minimum(nearest, hand_z, out=nearest)
        reports.append(report)
    return nearest, reports


def front_only(mask: np.ndarray, hand_z: np.ndarray, depth_z: np.ndarray, *,
               tolerance_m: float = 0.005, supported: np.ndarray | None = None
               ) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    hand_z = np.asarray(hand_z, dtype=np.float64)
    depth_z = np.asarray(depth_z, dtype=np.float64)
    if (mask.ndim != 2 or hand_z.shape != mask.shape or depth_z.shape != mask.shape
            or not np.isfinite(tolerance_m) or tolerance_m < 0):
        raise ValueError("depth-order raster/tolerance mismatch")
    supported = ((np.isfinite(depth_z) & (depth_z > 0)) if supported is None
                 else np.asarray(supported, dtype=bool))
    if supported.shape != mask.shape:
        raise ValueError("support raster mismatch")
    hit = np.isfinite(hand_z) & (hand_z > 0)
    valid = hit & np.isfinite(depth_z) & (depth_z > 0) & supported
    delta = np.zeros(mask.shape, dtype=np.float64)
    np.subtract(hand_z, depth_z, out=delta, where=valid)
    front = valid & (delta < -tolerance_m)
    behind = valid & (delta > tolerance_m)
    near = valid & ~front & ~behind
    classes = {
        "front": mask & front,
        "behind": mask & behind,
        "near": mask & near,
        "no_hit": mask & ~hit,
        "unsupported_depth": mask & hit & ~valid,
    }
    counts = {
        "raw_pixels": int(mask.sum()),
        "removed_front_pixels": int(classes["front"].sum()),
        "preserved_behind_pixels": int(classes["behind"].sum()),
        "preserved_near_pixels": int(classes["near"].sum()),
        "preserved_no_hit_pixels": int(classes["no_hit"].sum()),
        "preserved_unsupported_depth_pixels": int(classes["unsupported_depth"].sum()),
    }
    return mask & ~front, counts, classes


def sample_depth_at_mask_pixels(depth: np.ndarray, A_depth_from_mask: np.ndarray,
                                mask_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest discrete depth samples under the declared half-pixel affine."""
    depth = np.asarray(depth, dtype=np.float64)
    affine = np.asarray(A_depth_from_mask, dtype=np.float64)
    if depth.ndim != 2 or affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise ValueError("malformed depth/affine contract")
    yy, xx = np.mgrid[:mask_shape[0], :mask_shape[1]]
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ affine.T
    valid_h = np.isfinite(points).all(axis=2) & (np.abs(points[..., 2]) > 1e-12)
    source_x = np.rint(np.divide(points[..., 0], points[..., 2],
                                 out=np.zeros_like(points[..., 0], dtype=float), where=valid_h)).astype(int)
    source_y = np.rint(np.divide(points[..., 1], points[..., 2],
                                 out=np.zeros_like(points[..., 1], dtype=float), where=valid_h)).astype(int)
    inside = (valid_h & (source_x >= 0) & (source_x < depth.shape[1])
              & (source_y >= 0) & (source_y < depth.shape[0]))
    sampled = np.full(mask_shape, np.nan, dtype=np.float64)
    sampled[inside] = depth[source_y[inside], source_x[inside]]
    return sampled, inside


def mask_geometry(mask: np.ndarray, A_mask_from_source: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if not len(xs):
        return {"visible": False, "area_px": 0.0, "bbox_xyxy": None, "center_xy": None}
    inverse = np.linalg.inv(A_mask_from_source)
    corners = np.asarray([[xs.min(), ys.min(), 1.0], [xs.max() + 1, ys.max() + 1, 1.0]]) @ inverse.T
    corners = corners[:, :2] / corners[:, 2:3]
    center = np.asarray([xs.mean(), ys.mean(), 1.0]) @ inverse.T
    center = center[:2] / center[2]
    return {
        "visible": True,
        "area_px": float(len(xs) / abs(np.linalg.det(A_mask_from_source[:2, :2]))),
        "bbox_xyxy": [float(corners[0, 0]), float(corners[0, 1]),
                       float(corners[1, 0]), float(corners[1, 1])],
        "center_xy": center.astype(float).tolist(),
    }


def hand_geometry(frame: dict[str, Any], bridge: Any, hawor: Any,
                  bridge_path: Path, hawor_path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    geometry: list[tuple[np.ndarray, np.ndarray]] = []
    identities = []
    hands = frame.get("hands")
    if not isinstance(hands, list):
        raise ValueError(f"frame {frame.get('frame_idx')} hands must be a list")
    for hand in hands:
        if not isinstance(hand, dict):
            raise ValueError("malformed hand annotation")
        state = hand.get("metric_mano_state")
        reference = state.get("vertices_reference") if isinstance(state, dict) else None
        if not isinstance(reference, dict):
            # Missing geometry is uncertainty: it cannot remove raw appearance.
            identities.append({"side": hand.get("hand_side"), "status": "missing_metric_mano_preserved"})
            continue
        side = str(hand.get("hand_side") or "").lower()
        if side not in {"left", "right"}:
            raise ValueError("invalid hand side")
        if Path(str(reference.get("bridge_npz", ""))).resolve() != bridge_path:
            raise ValueError("hand bridge identity mismatch")
        if Path(str(reference.get("source_hawor_npz", ""))).resolve() != hawor_path:
            raise ValueError("hand HaWoR identity mismatch")
        row = reference.get("bridge_row_index")
        key = reference.get("bridge_vertices_camera_array")
        if isinstance(row, bool) or not isinstance(row, int) or not isinstance(key, str) or key not in bridge.files:
            raise ValueError("invalid MANO bridge reference")
        vertices_array = bridge[key]
        if row < 0 or row >= len(vertices_array):
            raise ValueError("MANO bridge row outside archive")
        frame_idx = int(frame["frame_idx"])
        if ("frame_idx" not in bridge.files or "hand_side" not in bridge.files
                or int(bridge["frame_idx"][row]) != frame_idx
                or str(bridge["hand_side"][row]) != side):
            raise ValueError("MANO bridge frame/side identity mismatch")
        faces_key = side + "_faces"
        if faces_key not in hawor.files:
            raise ValueError(f"HaWoR archive lacks {faces_key}")
        vertices = np.asarray(vertices_array[row], dtype=np.float64)
        faces = np.asarray(hawor[faces_key])
        geometry.append((vertices, faces))
        identities.append({"side": side, "status": "full_mano_surface", "bridge_row_index": row,
                           "vertices_array": key, "faces_array": faces_key})
    return geometry, identities


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"depth-order output must be new: {output_dir}")
    base_path = args.base_annotations.expanduser().resolve()
    raw_track_path = args.raw_sam2_track.expanduser().resolve()
    depth_path = args.depth_npz.expanduser().resolve()
    calibration_path = args.calibration_contract.expanduser().resolve()
    bridge_path = args.mano_bridge.expanduser().resolve()
    hawor_path = args.hawor_npz.expanduser().resolve()
    input_records = {name: record(path) for name, path in {
        "base_annotations": base_path, "raw_sam2_track": raw_track_path,
        "depth_npz": depth_path, "calibration_contract": calibration_path,
        "mano_bridge": bridge_path, "hawor_npz": hawor_path,
    }.items()}
    base_payload = load_object(base_path)
    if base_payload.get("case") != args.case:
        raise ValueError("base annotation case mismatch")
    frames = indexed_frames(base_payload.get("frames"), "base annotations")
    raw_payload = load_object(raw_track_path)
    track = indexed_track(raw_payload)
    if set(frames) != set(track):
        raise ValueError("base annotations/raw SAM2 frame coverage mismatch")

    depth = load_depth_npz(depth_path)
    if set(depth["frame_to_i"]) != set(frames):
        raise ValueError("base annotations/depth frame coverage mismatch")
    camera_payload, normalized = load_contract(calibration_path, expected_frame_ids=sorted(frames))
    validate_depth_camera_contract_binding(
        depth=depth, camera_contract_v2={"payload": camera_payload, "normalized": normalized},
        calibration_contract_path=calibration_path, depth_image_plane=args.depth_image_plane,
        allow_implicit_depth_resize=False,
    )
    depth_intrinsics, depth_plane = plane_intrinsics(
        camera_payload, normalized, plane_name=args.depth_image_plane,
        actual_size_wh=(int(depth["depth"].shape[2]), int(depth["depth"].shape[1])),
    )
    A_depth_from_calibration = np.asarray(depth_plane["A_actual_plane_from_calibration"], dtype=np.float64)

    output_dir.mkdir(parents=True)
    mask_dir = output_dir / "sam2_masks"
    mask_dir.mkdir()
    output_track: dict[str, Any] = {}
    rows = []
    bridge = np.load(bridge_path, mmap_mode="r", allow_pickle=False)
    hawor = np.load(hawor_path, mmap_mode="r", allow_pickle=False)
    try:
        for frame_idx in sorted(frames):
            raw_row = track[frame_idx]
            output_row = json.loads(json.dumps(raw_row))
            source_mask_value = raw_row.get("mask_path")
            visible = raw_row.get("visible") is True
            if source_mask_value is None:
                if visible:
                    raise ValueError(f"visible raw SAM2 frame {frame_idx} lacks mask")
                output_row["depth_order_ownership"] = {
                    "schema": SCHEMA, "status": "raw_track_not_visible", "source_mask": None,
                }
                output_track[str(frame_idx)] = output_row
                rows.append({"frame_idx": frame_idx, "status": "raw_track_not_visible"})
                continue
            source_mask_path = Path(source_mask_value).expanduser().resolve()
            source_ref = record(source_mask_path)
            image = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.ndim != 2:
                raise ValueError(f"unreadable raw SAM2 mask at frame {frame_idx}")
            raw_mask = image > 0
            if visible and not raw_mask.any():
                raise ValueError(f"visible raw SAM2 mask is empty at frame {frame_idx}")
            mask_intrinsics, mask_plane = plane_intrinsics(
                camera_payload, normalized, plane_name=args.mask_image_plane,
                actual_size_wh=(int(raw_mask.shape[1]), int(raw_mask.shape[0])),
            )
            A_mask_from_calibration = np.asarray(mask_plane["A_actual_plane_from_calibration"], dtype=np.float64)
            A_depth_from_mask = A_depth_from_calibration @ np.linalg.inv(A_mask_from_calibration)
            sampled_depth, raster_inside = sample_depth_at_mask_pixels(
                depth["depth"][depth["frame_to_i"][frame_idx]], A_depth_from_mask, raw_mask.shape)
            hands, hand_identities = hand_geometry(
                frames[frame_idx], bridge, hawor, bridge_path, hawor_path)
            nearest_z, hand_reports = nearest_hand_first_hits(
                hands, K_from_intrinsics(mask_intrinsics), raw_mask.shape, args.near_m)
            supported = raster_inside & np.isfinite(sampled_depth) & (sampled_depth > 0)
            owned, counts, _classes = front_only(
                raw_mask, nearest_z, sampled_depth,
                tolerance_m=args.depth_order_tolerance_m, supported=supported)
            destination = mask_dir / f"{frame_idx:06d}.png"
            if not cv2.imwrite(str(destination), owned.astype(np.uint8) * 255):
                raise RuntimeError(f"failed to write owned mask {destination}")
            destination_ref = record(destination)
            geometry = mask_geometry(owned, A_mask_from_calibration)
            output_row.update(geometry)
            output_row["mask_path"] = str(destination)
            output_row["depth_order_ownership"] = {
                "schema": SCHEMA,
                "status": "front_only_first_hit_classified" if visible else "invisible_row_mask_preserved_unconsumed",
                "source_mask": source_ref,
                "owned_mask": destination_ref,
                "depth_order_tolerance_m": float(args.depth_order_tolerance_m),
                "classes": counts,
                "metric_eligibility": {
                    "supported_raw_pixels": int((raw_mask & supported).sum()),
                    "unsupported_raw_pixels": int((raw_mask & ~supported).sum()),
                    "policy": "owned appearance support is not metric eligibility; P09 robust depth quarantine remains required",
                },
                "hands": hand_identities,
            }
            output_track[str(frame_idx)] = output_row
            rows.append({
                "frame_idx": frame_idx, "status": output_row["depth_order_ownership"]["status"],
                "source_mask": source_ref, "owned_mask": destination_ref,
                "mask_size_wh": [int(raw_mask.shape[1]), int(raw_mask.shape[0])],
                "depth_size_wh": [int(depth["depth"].shape[2]), int(depth["depth"].shape[1])],
                "A_depth_from_mask": A_depth_from_mask.tolist(),
                "K_mask": K_from_intrinsics(mask_intrinsics).tolist(),
                "counts": counts, "metric_supported_raw_pixels": int((raw_mask & supported).sum()),
                "hand_raster_reports": hand_reports, "hand_identities": hand_identities,
            })
    finally:
        bridge.close()
        hawor.close()

    track_path = output_dir / "sam2_track.json"
    track_path.write_text(json.dumps(output_track, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    track_ref = record(track_path)
    totals = {key: int(sum((row.get("counts") or {}).get(key, 0) for row in rows))
              for key in ("raw_pixels", "removed_front_pixels", "preserved_behind_pixels",
                          "preserved_near_pixels", "preserved_no_hit_pixels",
                          "preserved_unsupported_depth_pixels")}
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "perspective_correct_full_mano_first_hit_front_only",
        "case": args.case, "object_id": args.object_id, "track_id": args.track_id,
        "claim_scope": "derived P09 mask support only; raw SAM2 appearance is immutable and separately bound",
        "inputs": input_records,
        "outputs": {"owned_sam2_track": track_ref, "owned_mask_dir": str(mask_dir)},
        "policy": {
            "front": "remove only when nearest MANO z < measured depth z - tolerance",
            "behind": "preserve when nearest MANO z > measured depth z + tolerance",
            "near": "preserve when absolute z difference is within tolerance",
            "no_hit_or_invalid_depth": "preserve appearance; not automatically metric eligible",
            "triangle_depth": "perspective-correct inverse-z barycentric first hit",
            "multiple_hands": "nearest positive first hit over all declared full MANO triangle surfaces",
            "bbox_or_centroid_substitution": False,
        },
        "parameters": {"depth_order_tolerance_m": float(args.depth_order_tolerance_m),
                       "near_m": float(args.near_m), "depth_image_plane": args.depth_image_plane,
                       "mask_image_plane": args.mask_image_plane},
        "frame_count": len(rows), "totals": totals, "rows": rows,
    }
    report_path = output_dir / "depth_order_ownership_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-annotations", "raw-sam2-track", "depth-npz", "calibration-contract",
                 "mano-bridge", "hawor-npz", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("case", "object-id", "track-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--depth-image-plane", default="source_rgb")
    parser.add_argument("--mask-image-plane", default="sam2_mask")
    parser.add_argument("--depth-order-tolerance-m", type=float, default=0.005)
    parser.add_argument("--near-m", type=float, default=1e-6)
    args = parser.parse_args()
    if (not np.isfinite(args.depth_order_tolerance_m) or args.depth_order_tolerance_m < 0
            or not np.isfinite(args.near_m) or args.near_m <= 0):
        parser.error("invalid depth-order tolerance or near plane")
    return args


def main() -> None:
    report = build(parse_args())
    print(json.dumps({k: report[k] for k in ("status", "method", "totals")}, indent=2))


if __name__ == "__main__":
    main()
