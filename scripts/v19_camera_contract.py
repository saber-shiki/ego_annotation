#!/usr/bin/env python3
"""Source-neutral V19 camera and image-transform contract helpers.

The contract describes pinhole intrinsics in one declared calibration image
plane and explicit affine transforms into decoded RGB, mask, render, and depth
rasters. Dataset sensor calibration and estimated calibration use the same
schema; provenance and authority differ, downstream projection does not.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "v19_camera_image_transform_contract_v2"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_intrinsics(value: Any, label: str) -> np.ndarray:
    intrinsics = np.asarray(value, dtype=np.float64).reshape(-1)
    if (
        intrinsics.shape != (4,)
        or not np.isfinite(intrinsics).all()
        or float(intrinsics[0]) <= 0.0
        or float(intrinsics[1]) <= 0.0
    ):
        raise RuntimeError(f"{label} must be finite [fx,fy,cx,cy], got {intrinsics}")
    return intrinsics


def finite_affine(value: Any, label: str) -> np.ndarray:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise RuntimeError(f"{label} must be a finite 3x3 matrix, got {affine.shape}")
    if abs(float(affine[2, 0])) > 1.0e-12 or abs(float(affine[2, 1])) > 1.0e-12 or abs(float(affine[2, 2]) - 1.0) > 1.0e-12:
        raise RuntimeError(f"{label} must be an affine homogeneous pixel transform")
    if abs(float(np.linalg.det(affine))) < 1.0e-12:
        raise RuntimeError(f"{label} is singular")
    return affine


def size_wh(value: Any, label: str) -> tuple[int, int]:
    if isinstance(value, dict):
        width = int(value.get("width") or 0)
        height = int(value.get("height") or 0)
    else:
        values = np.asarray(value, dtype=np.int64).reshape(-1)
        if values.shape != (2,):
            raise RuntimeError(f"{label} must be [width,height]")
        width, height = [int(v) for v in values.tolist()]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"{label} has invalid size {(width, height)}")
    return width, height


def K_from_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = finite_intrinsics(intrinsics, "intrinsics")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def intrinsics_from_K(K: np.ndarray, label: str = "K") -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise RuntimeError(f"{label} must be a finite 3x3 matrix")
    if abs(float(K[0, 1])) > 1.0e-9 or abs(float(K[1, 0])) > 1.0e-9 or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1.0e-9):
        raise RuntimeError(f"{label} must be zero-skew pinhole K")
    return finite_intrinsics([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], label)


def resize_affine(
    source_size_wh: tuple[int, int],
    target_size_wh: tuple[int, int],
    *,
    pixel_center_convention: str = "integer_pixel_centers_opencv",
) -> np.ndarray:
    source_width, source_height = size_wh(source_size_wh, "source size")
    target_width, target_height = size_wh(target_size_wh, "target size")
    sx = float(target_width) / float(source_width)
    sy = float(target_height) / float(source_height)
    if pixel_center_convention == "integer_pixel_centers_opencv":
        # OpenCV samples target pixel j at source (j + 0.5) / scale - 0.5.
        # Therefore a projected source coordinate maps forward as
        # j = scale * (x + 0.5) - 0.5.
        tx = 0.5 * (sx - 1.0)
        ty = 0.5 * (sy - 1.0)
    elif pixel_center_convention == "pixel_corner_origin":
        tx = 0.0
        ty = 0.0
    else:
        raise RuntimeError(f"unsupported resize pixel-center convention: {pixel_center_convention!r}")
    return np.asarray(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_intrinsics(intrinsics: np.ndarray, affine: np.ndarray) -> np.ndarray:
    transformed = finite_affine(affine, "image affine") @ K_from_intrinsics(intrinsics)
    return intrinsics_from_K(transformed, "transformed K")


def fov_degrees(width: int, height: int, intrinsics: np.ndarray) -> dict[str, float]:
    fx, fy, _, _ = finite_intrinsics(intrinsics, "intrinsics")
    return {
        "horizontal": float(2.0 * math.degrees(math.atan(float(width) / (2.0 * fx)))),
        "vertical": float(2.0 * math.degrees(math.atan(float(height) / (2.0 * fy)))),
    }


def normalize_authority(authority: str) -> str:
    allowed = {
        "dataset_sensor_calibration",
        "prediction_side_sensor_metadata",
        "estimated_from_rgb_depth_model",
        "explicit_user_calibration",
    }
    value = str(authority)
    if value not in allowed:
        raise RuntimeError(f"unsupported calibration authority {value!r}; expected one of {sorted(allowed)}")
    return value


def plane_record(
    *,
    name: str,
    width: int,
    height: int,
    A_from_calibration: np.ndarray,
    transform_kind: str,
    interpolation: str | None,
    pixel_center_convention: str,
) -> dict[str, Any]:
    return {
        "plane_name": str(name),
        "width": int(width),
        "height": int(height),
        "A_plane_from_calibration": finite_affine(A_from_calibration, f"{name} affine").tolist(),
        "transform_kind": str(transform_kind),
        "interpolation": interpolation,
        "pixel_center_convention": str(pixel_center_convention),
    }


def validate_contract(payload: dict[str, Any], *, expected_frame_ids: list[int] | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("camera contract must be an object")
    schema = str(payload.get("schema") or "")
    if schema != SCHEMA:
        raise RuntimeError(f"unsupported camera contract schema {schema!r}; expected {SCHEMA!r}")
    authority = normalize_authority(str(payload.get("calibration_authority")))
    calibration = payload.get("calibration_plane")
    if not isinstance(calibration, dict):
        raise RuntimeError("camera contract lacks calibration_plane")
    calibration_size = size_wh(calibration, "calibration plane")
    intrinsics = finite_intrinsics(payload.get("intrinsics_fx_fy_cx_cy"), "contract intrinsics")
    K = intrinsics_from_K(payload.get("K"), "contract K") if payload.get("K") is not None else intrinsics
    if not np.allclose(intrinsics, K, atol=1.0e-7, rtol=0.0):
        raise RuntimeError("contract K disagrees with intrinsics_fx_fy_cx_cy")
    model = str(payload.get("intrinsics_model") or "")
    intrinsics_coordinate_plane = str(payload.get("intrinsics_coordinate_plane") or "")
    if intrinsics_coordinate_plane != "calibration":
        raise RuntimeError(
            "v2 contract intrinsics_fx_fy_cx_cy/K must be declared in the calibration plane; "
            f"got {intrinsics_coordinate_plane!r}"
        )
    if model not in {"constant_pinhole_fx_fy_cx_cy", "per_frame_pinhole_fx_fy_cx_cy"}:
        raise RuntimeError(f"unsupported intrinsics model: {model!r}")
    if model != "constant_pinhole_fx_fy_cx_cy":
        raise RuntimeError("v2 downstream currently requires one fixed output pinhole plane")
    planes = payload.get("image_planes")
    if not isinstance(planes, dict) or not planes:
        raise RuntimeError("camera contract lacks image_planes")
    normalized_planes: dict[str, Any] = {}
    for name, raw in planes.items():
        if not isinstance(raw, dict):
            raise RuntimeError(f"image plane {name!r} is not an object")
        width, height = size_wh(raw, f"image plane {name}")
        affine = finite_affine(raw.get("A_plane_from_calibration"), f"image plane {name} affine")
        normalized_planes[str(name)] = {
            **raw,
            "plane_name": str(raw.get("plane_name") or name),
            "width": width,
            "height": height,
            "A_plane_from_calibration": affine.tolist(),
            "intrinsics_fx_fy_cx_cy": transform_intrinsics(intrinsics, affine).tolist(),
        }
    frame_ids = [int(v) for v in payload.get("frame_ids") or []]
    if not frame_ids:
        raise RuntimeError("camera contract has no frame_ids")
    if expected_frame_ids is not None and frame_ids != [int(v) for v in expected_frame_ids]:
        raise RuntimeError("camera contract frame_ids do not match expected source timeline")
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("camera contract has duplicate frame_ids")
    return {
        "schema": schema,
        "calibration_authority": authority,
        "intrinsics_model": model,
        "intrinsics_coordinate_plane": intrinsics_coordinate_plane,
        "intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "calibration_size_wh": [int(calibration_size[0]), int(calibration_size[1])],
        "image_planes": normalized_planes,
        "frame_ids": frame_ids,
    }


def require_ordered_frame_subset(contract_frame_ids: list[int], selected_frame_ids: list[int]) -> None:
    """Bind a depth chunk to an ordered subset of an immutable camera timeline."""
    contract_positions = {int(frame): position for position, frame in enumerate(contract_frame_ids)}
    missing = [int(frame) for frame in selected_frame_ids if int(frame) not in contract_positions]
    if missing:
        raise RuntimeError(f"camera contract misses selected frames: {missing[:10]}")
    positions = [contract_positions[int(frame)] for frame in selected_frame_ids]
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise RuntimeError("selected frames are not an ordered unique subset of the camera contract timeline")


def load_contract(path: Path, *, expected_frame_ids: list[int] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(path)
    normalized = validate_contract(payload, expected_frame_ids=expected_frame_ids)
    return payload, normalized


def plane_intrinsics(
    payload: dict[str, Any],
    normalized: dict[str, Any],
    *,
    plane_name: str,
    actual_size_wh: tuple[int, int] | None = None,
    allow_implicit_resize: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    planes = normalized["image_planes"]
    if plane_name not in planes:
        raise RuntimeError(f"camera contract has no image plane {plane_name!r}; available={sorted(planes)}")
    plane = planes[plane_name]
    declared_size = (int(plane["width"]), int(plane["height"]))
    affine = finite_affine(plane["A_plane_from_calibration"], f"{plane_name} affine")
    implicit_resize = np.eye(3, dtype=np.float64)
    if actual_size_wh is not None:
        actual = size_wh(actual_size_wh, "actual plane size")
        if actual != declared_size:
            if not allow_implicit_resize:
                raise RuntimeError(
                    f"actual {plane_name} size {actual} differs from declared {declared_size}; "
                    "record an explicit plane or pass the compatibility override"
                )
            implicit_resize = resize_affine(
                declared_size,
                actual,
                pixel_center_convention=str(plane.get("pixel_center_convention") or "integer_pixel_centers_opencv"),
            )
    combined = implicit_resize @ affine
    intrinsics = transform_intrinsics(
        np.asarray(normalized["intrinsics_fx_fy_cx_cy"], dtype=np.float64), combined
    )
    return intrinsics, {
        "calibration_authority": normalized["calibration_authority"],
        "intrinsics_source": str(payload.get("intrinsics_source") or payload.get("method") or "camera_contract"),
        "calibration_plane_size_wh": normalized["calibration_size_wh"],
        "declared_plane_name": plane_name,
        "declared_plane_size_wh": [declared_size[0], declared_size[1]],
        "actual_plane_size_wh": list(actual_size_wh) if actual_size_wh is not None else [declared_size[0], declared_size[1]],
        "A_declared_plane_from_calibration": affine.tolist(),
        "A_actual_plane_from_declared_plane": implicit_resize.tolist(),
        "A_actual_plane_from_calibration": combined.tolist(),
        "implicit_resize_compatibility_override": bool(not np.allclose(implicit_resize, np.eye(3))),
        "pixel_center_convention": plane.get("pixel_center_convention"),
        "transform_kind": plane.get("transform_kind"),
    }


def summarize_contract(path: Path, payload: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "calibration_authority": normalized["calibration_authority"],
        "intrinsics_model": normalized["intrinsics_model"],
        "intrinsics_coordinate_plane": normalized["intrinsics_coordinate_plane"],
        "intrinsics_fx_fy_cx_cy": normalized["intrinsics_fx_fy_cx_cy"],
        "calibration_size_wh": normalized["calibration_size_wh"],
        "image_plane_names": sorted(normalized["image_planes"]),
        "intrinsics_source": payload.get("intrinsics_source"),
        "source_contract": payload.get("source_contract"),
        "fallback": payload.get("fallback"),
    }
