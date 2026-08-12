#!/usr/bin/env python3
"""Resolve a source-neutral V19 camera/image-transform contract.

Preferred mode consumes prediction-side dataset/sensor calibration. When that is
unavailable, the legacy robust UniDepth aggregation remains an explicit fallback
behind the same output schema and canonical filename. The resolver never reads
object, hand, CAD, pose, contact, or evaluator annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import build_v19_calibration_contract as legacy
from v19_camera_contract import (
    SCHEMA,
    K_from_intrinsics,
    finite_affine,
    finite_intrinsics,
    fov_degrees,
    load_json,
    normalize_authority,
    plane_record,
    resize_affine,
    sha256_file,
    size_wh,
    transform_intrinsics,
    validate_contract,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_affine(values: list[float] | None, label: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.size != 9:
        raise RuntimeError(f"{label} requires 9 values in row-major order")
    return finite_affine(array.reshape(3, 3), label)


def manifest_sizes(raw_manifest_path: Path) -> tuple[list[int], tuple[int, int], tuple[int, int], dict[str, Any]]:
    frame_ids, source_size, payload = legacy.raw_frames(raw_manifest_path)
    rows = [row for row in payload.get("frames", []) if isinstance(row, dict)]
    manifest_widths = [int(row.get("manifest_width") or 0) for row in rows]
    manifest_heights = [int(row.get("manifest_height") or 0) for row in rows]
    manifest_width = next((value for value in manifest_widths if value > 0), source_size[0])
    manifest_height = next((value for value in manifest_heights if value > 0), source_size[1])
    if any(value > 0 and value != manifest_width for value in manifest_widths):
        raise RuntimeError("raw frame manifest has varying manifest widths")
    if any(value > 0 and value != manifest_height for value in manifest_heights):
        raise RuntimeError("raw frame manifest has varying manifest heights")
    return frame_ids, source_size, (manifest_width, manifest_height), payload


def validate_sensor_timeline_and_source(
    *,
    raw_payload: dict[str, Any],
    source_payload: dict[str, Any],
    expected_frame_ids: list[int],
    prediction_source_video: Path | None,
    sensor_source_video: Path | None,
    time_tolerance_s: float,
) -> dict[str, Any]:
    source_frames = source_payload.get("frames")
    source_by_idx: dict[int, dict[str, Any]] = {}
    if isinstance(source_frames, list) and source_frames:
        for row in source_frames:
            if not isinstance(row, dict):
                continue
            idx = int(row.get("frame_idx", row.get("index", -1)))
            if idx < 0 or idx in source_by_idx:
                raise RuntimeError(f"sensor metadata has invalid/duplicate frame index {idx}")
            source_by_idx[idx] = row
        expected = [int(v) for v in expected_frame_ids]
        missing = [idx for idx in expected if idx not in source_by_idx]
        extras = sorted(set(source_by_idx).difference(expected))
        if missing or extras:
            raise RuntimeError(f"sensor metadata timeline mismatch: missing={missing[:20]} extras={extras[:20]}")
    else:
        expected = [int(v) for v in expected_frame_ids]
    raw_rows = {
        int(row.get("frame_idx", row.get("index", -1))): row
        for row in raw_payload.get("frames", [])
        if isinstance(row, dict)
    }
    tolerance = float(time_tolerance_s)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise RuntimeError("sensor timeline time tolerance must be finite and nonnegative")
    time_deltas: list[float] = []
    time_rows_checked = 0
    for idx in expected:
        raw_time = raw_rows.get(idx, {}).get("time_s")
        sensor_time = source_by_idx.get(idx, {}).get("time_s")
        if sensor_time is None:
            continue
        if raw_time is None:
            raise RuntimeError(f"raw metadata frame {idx} lacks time_s while sensor metadata provides it")
        delta = abs(float(raw_time) - float(sensor_time))
        if not np.isfinite(delta) or delta > tolerance:
            raise RuntimeError(
                f"raw/sensor time_s mismatch at frame {idx}: raw={raw_time} sensor={sensor_time} tolerance={tolerance}"
            )
        time_deltas.append(delta)
        time_rows_checked += 1
    if source_by_idx and 0 < time_rows_checked < len(expected):
        raise RuntimeError(
            f"sensor metadata has partial time_s coverage: {time_rows_checked}/{len(expected)} rows"
        )

    raw_input_video_text = raw_payload.get("input_video")
    raw_input_video = Path(str(raw_input_video_text)).expanduser().resolve() if raw_input_video_text else None
    prediction_video = prediction_source_video.expanduser().resolve() if prediction_source_video is not None else raw_input_video
    if prediction_video is None:
        raise RuntimeError("sensor-first contract requires --prediction-source-video or raw manifest input_video")
    if sensor_source_video is None:
        raise RuntimeError("sensor-first contract requires --sensor-source-video for source identity validation")
    sensor_video = sensor_source_video.expanduser().resolve()
    if not prediction_video.is_file() or not sensor_video.is_file():
        raise FileNotFoundError(
            f"source video validation files missing: prediction={prediction_video} sensor={sensor_video}"
        )
    prediction_hash = sha256_file(prediction_video)
    sensor_hash = sha256_file(sensor_video)
    if prediction_hash != sensor_hash:
        raise RuntimeError(
            "prediction input video and sensor-metadata source video are not byte-identical: "
            f"prediction={prediction_hash} sensor={sensor_hash}"
        )
    if source_by_idx and time_rows_checked == len(expected):
        status = "exact_source_video_hash_and_frame_time_timeline_match"
    elif source_by_idx:
        status = "exact_source_video_hash_and_frame_id_timeline_match_time_unavailable"
    else:
        status = "exact_source_video_hash_match_sensor_frame_timeline_unavailable"
    return {
        "status": status,
        "prediction_source_video": {"path": str(prediction_video), "sha256": prediction_hash},
        "sensor_metadata_source_video": {"path": str(sensor_video), "sha256": sensor_hash},
        "frame_count": len(expected),
        "sensor_metadata_frame_rows_present": bool(source_by_idx),
        "frame_ids_exact": True if source_by_idx else None,
        "time_s_tolerance": tolerance,
        "time_s_rows_checked": int(time_rows_checked),
        "time_s_max_abs_delta": float(max(time_deltas, default=0.0)) if time_deltas else None,
        "time_s_all_available_rows_match": True if time_rows_checked else None,
    }


def source_intrinsics(
    source_payload: dict[str, Any],
    *,
    frame_intrinsics_key: str | None,
    expected_frame_ids: list[int],
    fixed_intrinsics_tolerance_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    value = source_payload.get("intrinsics_fx_fy_cx_cy")
    if value is not None:
        return finite_intrinsics(value, "source calibration intrinsics"), {
            "mode": "top_level_intrinsics_fx_fy_cx_cy",
            "frame_intrinsics_key": None,
        }
    if source_payload.get("K") is not None:
        K = np.asarray(source_payload["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise RuntimeError("source calibration K must be 3x3")
        return finite_intrinsics([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], "source calibration K"), {
            "mode": "top_level_K",
            "frame_intrinsics_key": None,
        }
    frames = source_payload.get("frames")
    if isinstance(frames, list) and frames:
        key = str(frame_intrinsics_key or "intrinsics_fx_fy_cx_cy")
        rows_by_idx: dict[int, np.ndarray] = {}
        missing_rows: list[int] = []
        for row in frames:
            if not isinstance(row, dict):
                continue
            idx = int(row.get("frame_idx", row.get("index", -1)))
            if idx < 0:
                raise RuntimeError("source calibration frame row lacks frame_idx/index")
            if idx in rows_by_idx:
                raise RuntimeError(f"source calibration has duplicate frame {idx}")
            if row.get(key) is None:
                missing_rows.append(idx)
                continue
            rows_by_idx[idx] = finite_intrinsics(row.get(key), f"source frame {idx} intrinsics key {key!r}")
        expected = [int(v) for v in expected_frame_ids]
        missing = [idx for idx in expected if idx not in rows_by_idx]
        extras = sorted(set(rows_by_idx).difference(expected))
        if missing or extras:
            raise RuntimeError(
                f"source calibration frame intrinsics key {key!r} does not match prediction timeline; "
                f"missing={missing[:20]} extras={extras[:20]} rows_missing_key={missing_rows[:20]}"
            )
        array = np.asarray([rows_by_idx[idx] for idx in expected], dtype=np.float64)
        reference = np.median(array, axis=0)
        max_abs = np.max(np.abs(array - reference[None, :]), axis=0)
        tolerance = float(fixed_intrinsics_tolerance_px)
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise RuntimeError("fixed intrinsics tolerance must be finite and nonnegative")
        if np.any(max_abs > tolerance):
            raise RuntimeError(
                f"source calibration varies beyond fixed pinhole tolerance {tolerance} px: max_abs={max_abs.tolist()}"
            )
        return finite_intrinsics(reference, "source frame intrinsics median"), {
            "mode": "per_frame_metadata_fixed_pinhole_median",
            "frame_intrinsics_key": key,
            "frame_count": int(len(array)),
            "max_abs_delta_from_median_px": max_abs.tolist(),
            "fixed_intrinsics_tolerance_px": tolerance,
        }
    raise RuntimeError("source calibration lacks intrinsics_fx_fy_cx_cy/K or frame intrinsics")


def source_calibration_size(source_payload: dict[str, Any]) -> tuple[int, int]:
    if isinstance(source_payload.get("calibration_plane"), dict):
        return size_wh(source_payload["calibration_plane"], "source calibration plane")
    width = int(source_payload.get("image_width") or source_payload.get("width") or 0)
    height = int(source_payload.get("image_height") or source_payload.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    source_size = source_payload.get("source_size")
    if source_size is not None:
        return size_wh(source_size, "source calibration size")
    frames = source_payload.get("frames")
    if isinstance(frames, list) and frames and isinstance(frames[0], dict):
        first = frames[0]
        width = int(first.get("source_width") or first.get("manifest_width") or 0)
        height = int(first.get("source_height") or first.get("manifest_height") or 0)
        if width > 0 and height > 0:
            return width, height
    raise RuntimeError("source calibration lacks image/calibration size")


def infer_source_affine(
    calibration_size: tuple[int, int],
    source_size: tuple[int, int],
    explicit: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if explicit is not None:
        return explicit, "explicit_A_source_from_calibration"
    if calibration_size == source_size:
        return np.eye(3, dtype=np.float64), "identity_same_calibration_and_source_plane"
    raise RuntimeError(
        f"calibration plane {calibration_size} differs from source RGB plane {source_size}; "
        "provide --source-from-calibration-affine to record crop/pad/resize explicitly"
    )


def build_planes(
    *,
    calibration_size: tuple[int, int],
    source_size: tuple[int, int],
    manifest_size: tuple[int, int],
    A_source_from_calibration: np.ndarray,
    A_render_from_source: np.ndarray,
    pixel_center_convention: str,
) -> dict[str, Any]:
    A_manifest_from_source = resize_affine(
        source_size,
        manifest_size,
        pixel_center_convention=pixel_center_convention,
    )
    A_manifest_from_calibration = A_manifest_from_source @ A_source_from_calibration
    A_render_from_calibration = A_render_from_source @ A_source_from_calibration
    return {
        "calibration": plane_record(
            name="calibration",
            width=calibration_size[0],
            height=calibration_size[1],
            A_from_calibration=np.eye(3),
            transform_kind="identity",
            interpolation=None,
            pixel_center_convention=pixel_center_convention,
        ),
        "source_rgb": plane_record(
            name="source_rgb",
            width=source_size[0],
            height=source_size[1],
            A_from_calibration=A_source_from_calibration,
            transform_kind="explicit_source_image_transform",
            interpolation="dataset_or_video_adapter_declared",
            pixel_center_convention=pixel_center_convention,
        ),
        "manifest_rgb": plane_record(
            name="manifest_rgb",
            width=manifest_size[0],
            height=manifest_size[1],
            A_from_calibration=A_manifest_from_calibration,
            transform_kind="source_to_manifest_resize",
            interpolation="cv2.INTER_AREA_for_RGB",
            pixel_center_convention=pixel_center_convention,
        ),
        "sam2_mask": plane_record(
            name="sam2_mask",
            width=manifest_size[0],
            height=manifest_size[1],
            A_from_calibration=A_manifest_from_calibration,
            transform_kind="source_to_manifest_resize",
            interpolation="nearest_for_masks",
            pixel_center_convention=pixel_center_convention,
        ),
        "render": plane_record(
            name="render",
            width=manifest_size[0],
            height=manifest_size[1],
            A_from_calibration=A_render_from_calibration,
            transform_kind="explicit_render_from_source_transform",
            interpolation="renderer_raster",
            pixel_center_convention=pixel_center_convention,
        ),
    }


def save_intrinsics_npz(
    *,
    path: Path,
    frame_ids: list[int],
    calibration_intrinsics: np.ndarray,
    calibration_size: tuple[int, int],
    source_intrinsics: np.ndarray,
    source_size: tuple[int, int],
    contract_path: Path,
    intrinsics_source: str,
    authority: str,
) -> None:
    rows = np.repeat(source_intrinsics[None, :], len(frame_ids), axis=0).astype(np.float32)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_ids, dtype=np.int32),
        intrinsics_fx_fy_cx_cy=rows,
        constant_intrinsics_fx_fy_cx_cy=source_intrinsics.astype(np.float32),
        calibration_intrinsics_fx_fy_cx_cy=np.asarray(calibration_intrinsics, dtype=np.float32),
        calibration_size=np.asarray(calibration_size, dtype=np.int32),
        source_intrinsics_fx_fy_cx_cy=np.asarray(source_intrinsics, dtype=np.float32),
        source_size=np.asarray(source_size, dtype=np.int32),
        calibration_contract_json=np.asarray([str(contract_path)]),
        intrinsics_source=np.asarray([intrinsics_source]),
        calibration_authority=np.asarray([authority]),
    )


def sensor_contract(args: argparse.Namespace) -> dict[str, Any]:
    frame_ids, source_size, manifest_size, raw_payload = manifest_sizes(args.raw_frame_manifest)
    source_path = args.sensor_calibration_contract.resolve()
    source_payload = load_json(source_path)
    source_timeline_validation = validate_sensor_timeline_and_source(
        raw_payload=raw_payload,
        source_payload=source_payload,
        expected_frame_ids=frame_ids,
        prediction_source_video=args.prediction_source_video,
        sensor_source_video=args.sensor_source_video,
        time_tolerance_s=float(args.sensor_timeline_time_tolerance_s),
    )
    intrinsics_calibration, intrinsics_extraction = source_intrinsics(
        source_payload,
        frame_intrinsics_key=args.sensor_frame_intrinsics_key,
        expected_frame_ids=frame_ids,
        fixed_intrinsics_tolerance_px=float(args.fixed_intrinsics_tolerance_px),
    )
    calibration_size = source_calibration_size(source_payload)
    A_source, source_affine_reason = infer_source_affine(
        calibration_size,
        source_size,
        parse_affine(args.source_from_calibration_affine, "source-from-calibration affine"),
    )
    A_render_from_source = parse_affine(args.render_from_source_affine, "render-from-source affine")
    if A_render_from_source is None:
        A_render_from_source = resize_affine(
            source_size,
            manifest_size,
            pixel_center_convention=args.pixel_center_convention,
        )
    authority = normalize_authority(args.sensor_calibration_authority)
    source_plane_intrinsics = transform_intrinsics(intrinsics_calibration, A_source)
    intrinsics_source = str(
        source_payload.get("intrinsics_source")
        or source_payload.get("source")
        or source_payload.get("method")
        or "prediction_side_sensor_calibration"
    )
    contract_path = args.output_dir.resolve() / "v19_camera_calibration_contract.json"
    intrinsics_npz_path = args.output_dir.resolve() / "v19_camera_calibration_intrinsics.npz"
    contract = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "resolve_v19_camera_contract",
        "case": args.case,
        "claim_scope": (
            "prediction-side camera calibration and explicit image-plane transforms only; "
            "no object/hand/CAD/pose/contact/evaluator labels are consumed"
        ),
        "calibration_authority": authority,
        "source_contract": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "method": source_payload.get("method"),
            "schema": source_payload.get("schema"),
        },
        "fallback": {"used": False, "reason": None},
        "inputs": {
            "raw_frame_manifest": str(args.raw_frame_manifest.resolve()),
            "sensor_calibration_contract": str(source_path),
            "unidepth_npz": None,
        },
        "outputs": {
            "contract_json": str(contract_path),
            "intrinsics_npz": str(intrinsics_npz_path),
        },
        "raw_video": raw_payload.get("video") if isinstance(raw_payload.get("video"), dict) else None,
        "frame_ids": frame_ids,
        "intrinsics_model": "constant_pinhole_fx_fy_cx_cy",
        "intrinsics_coordinate_plane": "calibration",
        "intrinsics_vary_per_frame": False,
        "calibration_plane": {
            "width": calibration_size[0],
            "height": calibration_size[1],
            "pixel_center_convention": args.pixel_center_convention,
            "camera_coordinate_convention": args.camera_coordinate_convention,
            "depth_convention": args.depth_convention,
        },
        "intrinsics_fx_fy_cx_cy": intrinsics_calibration.tolist(),
        "K": K_from_intrinsics(intrinsics_calibration).tolist(),
        "focal_px": float(intrinsics_calibration[0]),
        "focal_geom_px": float(np.sqrt(intrinsics_calibration[0] * intrinsics_calibration[1])),
        "fov_degrees": fov_degrees(calibration_size[0], calibration_size[1], intrinsics_calibration),
        "intrinsics_source": intrinsics_source,
        "image_planes": build_planes(
            calibration_size=calibration_size,
            source_size=source_size,
            manifest_size=manifest_size,
            A_source_from_calibration=A_source,
            A_render_from_source=A_render_from_source,
            pixel_center_convention=args.pixel_center_convention,
        ),
        "image_transform_contract": {
            "formula": "K_plane = A_plane_from_calibration @ K_calibration",
            "source_affine_reason": source_affine_reason,
            "no_implicit_crop_pad_rotation": True,
        },
        "source_plane_intrinsics_fx_fy_cx_cy": source_plane_intrinsics.tolist(),
        "provenance": {
            "dataset_name": args.dataset_name,
            "sensor_stream": args.sensor_stream,
            "calibration_is_prediction_side_metadata": True,
            "evaluator_gt_consumed": False,
            "intrinsics_extraction": intrinsics_extraction,
            "source_timeline_validation": source_timeline_validation,
        },
    }
    validate_contract(contract, expected_frame_ids=frame_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(contract_path, contract)
    save_intrinsics_npz(
        path=intrinsics_npz_path,
        frame_ids=frame_ids,
        calibration_intrinsics=intrinsics_calibration,
        calibration_size=calibration_size,
        source_intrinsics=source_plane_intrinsics,
        source_size=source_size,
        contract_path=contract_path,
        intrinsics_source=intrinsics_source,
        authority=authority,
    )
    return contract


def fallback_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.unidepth_npz is None:
        raise RuntimeError("sensor calibration is unavailable and --unidepth-npz fallback was not supplied")
    frame_ids, source_size, manifest_size, _ = manifest_sizes(args.raw_frame_manifest)
    selected_frame_idx, selected_intrinsics = legacy.select_rows(
        *legacy.load_unidepth_intrinsics(args.unidepth_npz)[:2],
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    intrinsics, aggregation = legacy.aggregate_intrinsics(
        selected_intrinsics,
        method=args.aggregation,
        trim_low=float(args.trim_low),
        trim_high=float(args.trim_high),
    )
    if args.square_focal:
        focal = float(np.sqrt(intrinsics[0] * intrinsics[1]))
        intrinsics[0] = focal
        intrinsics[1] = focal
    if args.center_principal_point:
        intrinsics[2] = float(source_size[0]) / 2.0
        intrinsics[3] = float(source_size[1]) / 2.0
    A_render_from_source = parse_affine(args.render_from_source_affine, "render-from-source affine")
    if A_render_from_source is None:
        A_render_from_source = resize_affine(
            source_size,
            manifest_size,
            pixel_center_convention=args.pixel_center_convention,
        )
    contract_path = args.output_dir.resolve() / "v19_camera_calibration_contract.json"
    intrinsics_npz_path = args.output_dir.resolve() / "v19_camera_calibration_intrinsics.npz"
    contract = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "resolve_v19_camera_contract",
        "case": args.case,
        "claim_scope": "estimated camera fallback from robust UniDepth intrinsics; same downstream contract as sensor calibration",
        "calibration_authority": "estimated_from_rgb_depth_model",
        "source_contract": None,
        "fallback": {
            "used": True,
            "reason": args.fallback_reason,
            "method": "robust_unidepth_video_constant",
        },
        "inputs": {
            "raw_frame_manifest": str(args.raw_frame_manifest.resolve()),
            "sensor_calibration_contract": None,
            "unidepth_npz": str(args.unidepth_npz.resolve()),
        },
        "outputs": {
            "contract_json": str(contract_path),
            "intrinsics_npz": str(intrinsics_npz_path),
        },
        "frame_ids": frame_ids,
        "intrinsics_model": "constant_pinhole_fx_fy_cx_cy",
        "intrinsics_coordinate_plane": "calibration",
        "intrinsics_vary_per_frame": False,
        "calibration_plane": {
            "width": source_size[0],
            "height": source_size[1],
            "pixel_center_convention": args.pixel_center_convention,
            "camera_coordinate_convention": args.camera_coordinate_convention,
            "depth_convention": args.depth_convention,
        },
        "intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "K": K_from_intrinsics(intrinsics).tolist(),
        "focal_px": float(intrinsics[0]),
        "focal_geom_px": float(np.sqrt(intrinsics[0] * intrinsics[1])),
        "fov_degrees": fov_degrees(source_size[0], source_size[1], intrinsics),
        "intrinsics_source": "v19_calibration_contract_unidepth_robust_video_constant",
        "aggregation": aggregation,
        "image_planes": build_planes(
            calibration_size=source_size,
            source_size=source_size,
            manifest_size=manifest_size,
            A_source_from_calibration=np.eye(3),
            A_render_from_source=A_render_from_source,
            pixel_center_convention=args.pixel_center_convention,
        ),
        "image_transform_contract": {
            "formula": "K_plane = A_plane_from_calibration @ K_calibration",
            "source_affine_reason": "identity_estimator_and_source_plane",
            "no_implicit_crop_pad_rotation": True,
        },
        "diagnostics": {
            "selected_frame_count": int(len(selected_frame_idx)),
            "selected_stats": {
                "fx": legacy.summarize(selected_intrinsics[:, 0]),
                "fy": legacy.summarize(selected_intrinsics[:, 1]),
                "cx": legacy.summarize(selected_intrinsics[:, 2]),
                "cy": legacy.summarize(selected_intrinsics[:, 3]),
            },
        },
        "provenance": {
            "dataset_name": args.dataset_name,
            "sensor_stream": args.sensor_stream,
            "calibration_is_prediction_side_metadata": False,
            "evaluator_gt_consumed": False,
        },
    }
    validate_contract(contract, expected_frame_ids=frame_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(contract_path, contract)
    save_intrinsics_npz(
        path=intrinsics_npz_path,
        frame_ids=frame_ids,
        calibration_intrinsics=intrinsics,
        calibration_size=source_size,
        source_intrinsics=intrinsics,
        source_size=source_size,
        contract_path=contract_path,
        intrinsics_source=contract["intrinsics_source"],
        authority=contract["calibration_authority"],
    )
    return contract


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    sensor_supplied = args.sensor_calibration_contract is not None
    if sensor_supplied:
        if not args.sensor_calibration_contract.exists():
            if not args.allow_missing_sensor_calibration_fallback:
                raise FileNotFoundError(args.sensor_calibration_contract)
            if args.unidepth_npz is None:
                raise RuntimeError("missing sensor calibration fallback requires --unidepth-npz")
            return fallback_contract(args)
        return sensor_contract(args)
    return fallback_contract(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--sensor-calibration-contract", type=Path, default=None)
    parser.add_argument(
        "--sensor-calibration-authority",
        choices=["dataset_sensor_calibration", "prediction_side_sensor_metadata", "explicit_user_calibration"],
        default="dataset_sensor_calibration",
    )
    parser.add_argument("--sensor-frame-intrinsics-key", default=None, help="Optional per-frame metadata key containing [fx,fy,cx,cy] when the sensor metadata has no top-level K.")
    parser.add_argument("--prediction-source-video", type=Path, default=None, help="Prediction input video. Sensor-first mode hashes it against --sensor-source-video; defaults to raw manifest input_video when available.")
    parser.add_argument("--sensor-source-video", type=Path, default=None, help="Local copy of the exact image-stream video associated with the supplied sensor metadata.")
    parser.add_argument("--sensor-timeline-time-tolerance-s", type=float, default=1.0e-9)
    parser.add_argument("--fixed-intrinsics-tolerance-px", type=float, default=0.01, help="Maximum per-component deviation from the per-frame median allowed for a fixed pinhole camera contract.")
    parser.add_argument("--unidepth-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--sensor-stream", default=None)
    parser.add_argument("--source-from-calibration-affine", type=float, nargs=9, default=None, metavar=("A00", "A01", "A02", "A10", "A11", "A12", "A20", "A21", "A22"))
    parser.add_argument("--render-from-source-affine", type=float, nargs=9, default=None, metavar=("A00", "A01", "A02", "A10", "A11", "A12", "A20", "A21", "A22"))
    parser.add_argument("--pixel-center-convention", default="integer_pixel_centers_opencv")
    parser.add_argument("--camera-coordinate-convention", default="x_right_y_down_z_forward")
    parser.add_argument("--depth-convention", default="camera_z_meters")
    parser.add_argument("--allow-missing-sensor-calibration-fallback", action="store_true")
    parser.add_argument("--fallback-reason", default="prediction-side sensor calibration unavailable")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--aggregation", choices=["median", "trimmed_mean"], default="median")
    parser.add_argument("--trim-low", type=float, default=0.10)
    parser.add_argument("--trim-high", type=float, default=0.90)
    parser.add_argument("--square-focal", action="store_true")
    parser.add_argument("--center-principal-point", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = resolve(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "method": report["method"],
                "case": report["case"],
                "calibration_authority": report["calibration_authority"],
                "intrinsics_fx_fy_cx_cy": report["intrinsics_fx_fy_cx_cy"],
                "fallback": report["fallback"],
                "outputs": report["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
