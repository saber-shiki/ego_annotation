#!/usr/bin/env python3
"""Copy a metric-depth archive while replacing only camera intrinsics metadata.

This is a prediction-side camera-contract adapter.  It is used when the RGB
sensor has an official pinhole calibration but the depth estimator also emits
an estimated K.  Depth samples, frame indices, raster size, and dtype remain
bit-identical as arrays; all downstream lifting must consume the official K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_contract(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = np.asarray(data.get("intrinsics_fx_fy_cx_cy"), dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[:2] <= 0.0):
        raise RuntimeError(f"invalid intrinsics_fx_fy_cx_cy in {path}")
    return values, data


def adapt(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_depth_npz.resolve()
    contract_path = args.calibration_contract.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / args.output_name
    report_path = output_dir / "v19_depth_official_intrinsics_adapter_report.json"
    if output.exists() and not args.replace:
        raise FileExistsError(f"output exists; use --replace explicitly: {output}")
    official, contract = load_contract(contract_path)
    with np.load(source) as archive:
        required = {"frame_idx", "depth", "source_size", "intrinsics_fx_fy_cx_cy"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"source depth archive missing keys: {missing}")
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    frame_idx = np.asarray(payload["frame_idx"])
    depth = np.asarray(payload["depth"])
    source_size = np.asarray(payload["source_size"])
    source_intrinsics = np.asarray(payload["intrinsics_fx_fy_cx_cy"])
    if depth.ndim != 3 or frame_idx.shape != (depth.shape[0],):
        raise RuntimeError(f"inconsistent depth/frame arrays: {depth.shape}, {frame_idx.shape}")
    if source_intrinsics.shape != (depth.shape[0], 4):
        raise RuntimeError(f"inconsistent source intrinsics shape: {source_intrinsics.shape}")
    if source_size.shape != (2,):
        raise RuntimeError(f"source_size must be [width,height], got {source_size.shape}")
    width, height = [int(v) for v in source_size.tolist()]
    if depth.shape[1:] != (height, width):
        raise RuntimeError(f"depth raster {depth.shape[1:]} disagrees with source_size {(width, height)}")
    official_rows = np.repeat(official[None].astype(np.float32), depth.shape[0], axis=0)
    payload["source_estimated_intrinsics_fx_fy_cx_cy"] = source_intrinsics
    payload["intrinsics_fx_fy_cx_cy"] = official_rows
    payload["focal_px"] = np.repeat(np.float32(np.sqrt(official[0] * official[1])), depth.shape[0])
    payload["intrinsics_source"] = np.asarray("official_prediction_side_calibration_contract")
    payload["intrinsics_override_applied"] = np.asarray(True)
    payload["calibration_contract_sha256"] = np.asarray(sha256_file(contract_path))
    payload["source_depth_archive_sha256"] = np.asarray(sha256_file(source))
    np.savez_compressed(output, **payload)

    # Reload and prove that the intervention changed metadata only.
    with np.load(output) as check:
        output_depth = np.asarray(check["depth"])
        output_frame_idx = np.asarray(check["frame_idx"])
        output_source_size = np.asarray(check["source_size"])
        output_intrinsics = np.asarray(check["intrinsics_fx_fy_cx_cy"])
    depth_equal = bool(np.array_equal(depth, output_depth, equal_nan=True))
    frame_equal = bool(np.array_equal(frame_idx, output_frame_idx))
    size_equal = bool(np.array_equal(source_size, output_source_size))
    intrinsics_equal_contract = bool(np.array_equal(output_intrinsics, official_rows))
    if not (depth_equal and frame_equal and size_equal and intrinsics_equal_contract):
        raise RuntimeError("official-intrinsics adapter invariant failed after archive reload")
    source_delta = source_intrinsics.astype(np.float64) - official[None]
    report = {
        "status": "ok",
        "method": "build_v19_depth_official_intrinsics_adapter",
        "claim_scope": "prediction-side intrinsics metadata override only; no depth values or released labels consumed",
        "source_depth_npz": str(source),
        "output_depth_npz": str(output),
        "calibration_contract": str(contract_path),
        "calibration_contract_method": contract.get("method"),
        "frame_count": int(depth.shape[0]),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "source_size_wh": [width, height],
        "official_intrinsics_fx_fy_cx_cy": official.tolist(),
        "source_estimated_intrinsics_delta_from_official": {
            "median_fx_fy_cx_cy": np.median(source_delta, axis=0).tolist(),
            "max_abs_fx_fy_cx_cy": np.max(np.abs(source_delta), axis=0).tolist(),
        },
        "array_invariants": {
            "depth_array_equal": depth_equal,
            "frame_idx_equal": frame_equal,
            "source_size_equal": size_equal,
            "output_intrinsics_equal_contract": intrinsics_equal_contract,
        },
        "source_depth_archive_sha256": sha256_file(source),
        "output_depth_archive_sha256": sha256_file(output),
        "downstream_contract": "P09, P14b, P18 and any depth lifting/projection must consume output_depth_npz so K is not estimated by UniDepth.",
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-depth-npz", type=Path, required=True)
    parser.add_argument("--calibration-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="unidepth_full_frame_depth_official_k_v1.npz")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    adapt(parse_args())
