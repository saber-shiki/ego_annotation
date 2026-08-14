#!/usr/bin/env python3
"""Adapt a metric-depth archive to one V19 camera contract.

Depth values are preserved exactly. Only camera/image-plane metadata is changed,
and both the original estimated K and the resolved contract provenance remain in
the output. This works for sensor-calibrated and estimated-fallback contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from v19_camera_contract import load_contract, plane_intrinsics, sha256_file, summarize_contract


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.view(np.uint8).tobytes())
    return digest.hexdigest()


def adapt(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_depth_npz.expanduser().resolve()
    contract_path = args.camera_contract.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    report_path = output_dir / "v19_depth_camera_contract_adapter_report.json"
    if output_path.exists() and not args.replace:
        raise FileExistsError(f"output exists; use --replace explicitly: {output_path}")

    with np.load(source, allow_pickle=False) as archive:
        required = {"frame_idx", "depth", "source_size", "intrinsics_fx_fy_cx_cy"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"source depth archive missing keys: {missing}")
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    frame_idx = np.asarray(payload["frame_idx"], dtype=np.int32)
    depth = np.asarray(payload["depth"])
    source_size = np.asarray(payload["source_size"], dtype=np.int32).reshape(-1)
    source_intrinsics = np.asarray(payload["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    if depth.ndim != 3 or frame_idx.shape != (depth.shape[0],):
        raise RuntimeError(f"inconsistent depth/frame arrays: {depth.shape}, {frame_idx.shape}")
    if source_size.shape != (2,):
        raise RuntimeError(f"source_size must be [width,height], got {source_size.shape}")
    if source_intrinsics.shape != (depth.shape[0], 4):
        raise RuntimeError(f"source intrinsics shape mismatch: {source_intrinsics.shape}")
    depth_size = (int(depth.shape[2]), int(depth.shape[1]))
    if depth_size != tuple(int(v) for v in source_size.tolist()):
        raise RuntimeError(f"depth raster {depth_size} disagrees with source_size {source_size.tolist()}")

    contract, normalized = load_contract(contract_path, expected_frame_ids=frame_idx.tolist())
    intrinsics, transform = plane_intrinsics(
        contract,
        normalized,
        plane_name=args.depth_plane,
        actual_size_wh=depth_size,
        allow_implicit_resize=bool(args.allow_implicit_depth_resize),
    )
    # Camera contracts are JSON/float64 authorities.  Do not quantize active K
    # rows to float32: at focal lengths around 1k px, a valid float32 round trip
    # can move K by O(1e-5) px and violate the downstream exact V2 binding.
    resolved_rows = np.repeat(intrinsics[None, :], depth.shape[0], axis=0).astype(np.float64)
    depth_hash_before = array_sha256(depth)
    frame_hash_before = array_sha256(frame_idx)

    payload["source_estimated_intrinsics_fx_fy_cx_cy"] = np.asarray(payload["intrinsics_fx_fy_cx_cy"])
    payload["intrinsics_fx_fy_cx_cy"] = resolved_rows
    payload["focal_px"] = np.repeat(np.float64(np.sqrt(intrinsics[0] * intrinsics[1])), depth.shape[0])
    payload["intrinsics_source"] = np.asarray(str(contract.get("intrinsics_source") or contract.get("method")))
    payload["calibration_authority"] = np.asarray(str(normalized["calibration_authority"]))
    payload["camera_contract_path"] = np.asarray(str(contract_path))
    payload["camera_contract_sha256"] = np.asarray(sha256_file(contract_path))
    payload["camera_contract_plane"] = np.asarray(args.depth_plane)
    payload["A_depth_from_calibration"] = np.asarray(transform["A_actual_plane_from_calibration"], dtype=np.float64)
    payload["intrinsics_override_applied"] = np.asarray(
        not np.array_equal(source_intrinsics, resolved_rows)
    )
    payload["source_depth_archive_sha256"] = np.asarray(sha256_file(source))
    np.savez_compressed(output_path, **payload)

    with np.load(output_path, allow_pickle=False) as check:
        output_depth = np.asarray(check["depth"])
        output_frame_idx = np.asarray(check["frame_idx"])
        output_source_size = np.asarray(check["source_size"])
        output_intrinsics = np.asarray(check["intrinsics_fx_fy_cx_cy"])
        archived_source_intrinsics = np.asarray(check["source_estimated_intrinsics_fx_fy_cx_cy"])
    invariants = {
        "depth_array_equal": bool(np.array_equal(depth, output_depth, equal_nan=True)),
        "depth_array_sha256_equal": bool(depth_hash_before == array_sha256(output_depth)),
        "frame_idx_equal": bool(np.array_equal(frame_idx, output_frame_idx)),
        "frame_idx_sha256_equal": bool(frame_hash_before == array_sha256(output_frame_idx)),
        "source_size_equal": bool(np.array_equal(source_size, output_source_size)),
        "source_intrinsics_preserved": bool(np.array_equal(source_intrinsics, archived_source_intrinsics)),
        "output_intrinsics_float64": bool(output_intrinsics.dtype == np.dtype(np.float64)),
        "output_intrinsics_equal_contract_plane": bool(np.array_equal(output_intrinsics, resolved_rows)),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"depth camera-contract adapter invariant failed: {invariants}")
    delta = source_intrinsics - intrinsics[None, :]
    report = {
        "status": "ok",
        "method": "adapt_v19_depth_to_camera_contract",
        "claim_scope": (
            "camera metadata adaptation only; metric depth values, frame rows, raster size, and dtype are unchanged; "
            "no evaluator/object/hand labels are consumed"
        ),
        "inputs": {
            "source_depth_npz": str(source),
            "camera_contract": str(contract_path),
        },
        "outputs": {
            "depth_npz": str(output_path),
            "report": str(report_path),
        },
        "frame_count": int(depth.shape[0]),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "active_intrinsics_dtype": str(resolved_rows.dtype),
        "depth_plane": args.depth_plane,
        "resolved_intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "camera_contract": summarize_contract(contract_path, contract, normalized),
        "image_transform": transform,
        "source_estimated_intrinsics_delta_from_resolved": {
            "median_fx_fy_cx_cy": np.median(delta, axis=0).tolist(),
            "max_abs_fx_fy_cx_cy": np.max(np.abs(delta), axis=0).tolist(),
        },
        "array_invariants": invariants,
        "source_depth_archive_sha256": sha256_file(source),
        "output_depth_archive_sha256": sha256_file(output_path),
        "downstream_contract": (
            "Visible geometry and any metric backprojection must consume this output or resolve the same camera contract plane directly; active K rows retain float64 contract precision."
        ),
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-depth-npz", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--depth-plane", default="source_rgb")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="unidepth_full_frame_depth_camera_contract_v2.npz")
    parser.add_argument("--allow-implicit-depth-resize", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = adapt(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
