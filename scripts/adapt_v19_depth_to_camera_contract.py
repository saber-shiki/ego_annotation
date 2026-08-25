#!/usr/bin/env python3
"""Bind a metric-depth archive to one V19 camera contract, fail closed.

Dense z values are preserved exactly. Therefore this adapter may only bind a
unchanged raster when its source rays already match the requested contract
plane (normally because UniDepth was run with that camera). A disagreeing K is
not a metadata problem and requires camera-conditioned inference or explicit
3D reprojection; the legacy override exists only for forensic reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from v19_camera_contract import (
    load_contract,
    plane_intrinsics,
    require_ordered_frame_subset,
    sha256_file,
    summarize_contract,
)


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


def scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{label} must be scalar")
    return str(array.reshape(-1)[0])


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
    source_confidence = np.asarray(payload["confidence"]) if "confidence" in payload else None
    depth_provider = scalar_text(payload["depth_provider"], "depth_provider") if "depth_provider" in payload else "unidepth"
    if depth.ndim != 3 or frame_idx.shape != (depth.shape[0],):
        raise RuntimeError(f"inconsistent depth/frame arrays: {depth.shape}, {frame_idx.shape}")
    if source_size.shape != (2,):
        raise RuntimeError(f"source_size must be [width,height], got {source_size.shape}")
    if source_intrinsics.shape != (depth.shape[0], 4):
        raise RuntimeError(f"source intrinsics shape mismatch: {source_intrinsics.shape}")
    depth_size = (int(depth.shape[2]), int(depth.shape[1]))
    if depth_size != tuple(int(v) for v in source_size.tolist()):
        raise RuntimeError(f"depth raster {depth_size} disagrees with source_size {source_size.tolist()}")

    contract, normalized = load_contract(contract_path)
    require_ordered_frame_subset(normalized["frame_ids"], frame_idx.tolist())
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
    contract_hash = sha256_file(contract_path)
    source_conditioning_mode = (
        scalar_text(payload["camera_conditioning_mode"], "camera_conditioning_mode")
        if "camera_conditioning_mode" in payload
        else "legacy_unspecified"
    )
    source_conditioning: dict[str, Any] = {"mode": source_conditioning_mode, "depth_provider": depth_provider}
    source_depth_rays_reprojected = bool(
        np.asarray(payload.get("depth_ray_geometry_reprojected", False)).reshape(-1)[0]
    )
    if source_conditioning_mode == "provided_pinhole_intrinsics":
        required_conditioning = {
            "inference_camera_contract_sha256",
            "inference_camera_output_plane",
            "inference_camera_intrinsics_fx_fy_cx_cy",
            "depth_output_quantity",
            "depth_ray_geometry_reprojected",
            "confidence",
        }
        missing_conditioning = sorted(required_conditioning - set(payload))
        if missing_conditioning:
            raise RuntimeError(f"camera-conditioned depth archive misses {missing_conditioning}")
        source_contract_hash = scalar_text(
            payload["inference_camera_contract_sha256"], "inference_camera_contract_sha256"
        )
        source_output_plane = scalar_text(payload["inference_camera_output_plane"], "inference_camera_output_plane")
        if source_contract_hash != contract_hash:
            raise RuntimeError("depth inference camera-contract hash disagrees with the requested contract")
        if source_output_plane != args.depth_plane:
            raise RuntimeError(
                f"depth inference output plane {source_output_plane!r} disagrees with requested {args.depth_plane!r}"
            )
        conditioned_rows = np.asarray(payload["inference_camera_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        if conditioned_rows.shape != source_intrinsics.shape or not np.allclose(
            conditioned_rows, source_intrinsics, atol=1.0e-6, rtol=0.0
        ):
            raise RuntimeError("camera-conditioned inference K rows disagree with source depth K rows")
        if not source_depth_rays_reprojected:
            raise RuntimeError(
                "camera-conditioned UniDepth archive did not convert metric radius onto exact output-contract rays"
            )
        source_conditioning.update({
            "camera_contract_sha256": source_contract_hash,
            "output_plane": source_output_plane,
            "intrinsics_match_source_rows": True,
            "depth_ray_geometry_reprojected": True,
            "depth_output_quantity": scalar_text(payload.get("depth_output_quantity"), "depth_output_quantity"),
        })
    elif source_conditioning_mode == "provided_pinhole_intrinsics_and_metric_extrinsics":
        required_conditioning = {
            "inference_camera_contract_sha256",
            "inference_camera_output_plane",
            "inference_camera_intrinsics_fx_fy_cx_cy",
            "inference_camera_extrinsics_w2c",
            "pose_conditioning_mode",
            "camera_trajectory_source",
            "depth_output_quantity",
            "depth_ray_geometry_reprojected",
            "confidence",
            "confidence_role",
            "overlap_consistency_passed",
            "metric_scale_mode",
            "metric_scale_source",
        }
        missing_conditioning = sorted(required_conditioning - set(payload))
        if missing_conditioning:
            raise RuntimeError(f"pose-conditioned depth archive misses {missing_conditioning}")
        source_contract_hash = scalar_text(
            payload["inference_camera_contract_sha256"], "inference_camera_contract_sha256"
        )
        source_output_plane = scalar_text(payload["inference_camera_output_plane"], "inference_camera_output_plane")
        if source_contract_hash != contract_hash:
            raise RuntimeError("pose-conditioned depth camera-contract hash disagrees with the requested contract")
        if source_output_plane != args.depth_plane:
            raise RuntimeError(
                f"pose-conditioned depth output plane {source_output_plane!r} disagrees with requested {args.depth_plane!r}"
            )
        conditioned_rows = np.asarray(payload["inference_camera_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        if conditioned_rows.shape != source_intrinsics.shape or not np.allclose(
            conditioned_rows, source_intrinsics, atol=1.0e-6, rtol=0.0
        ):
            raise RuntimeError("pose-conditioned inference K rows disagree with source depth K rows")
        conditioned_extrinsics = np.asarray(payload["inference_camera_extrinsics_w2c"], dtype=np.float64)
        if conditioned_extrinsics.shape != (len(frame_idx), 4, 4) or not np.isfinite(conditioned_extrinsics).all():
            raise RuntimeError("pose-conditioned depth archive has invalid fixed W2C rows")
        if not source_depth_rays_reprojected:
            raise RuntimeError("pose-conditioned depth was not ray-remapped onto the exact output contract")
        confidence_role = scalar_text(payload["confidence_role"], "confidence_role")
        if confidence_role != "predicted_error_proxy_higher_is_worse":
            raise RuntimeError(f"unsupported pose-conditioned confidence role {confidence_role!r}")
        if not bool(np.asarray(payload["overlap_consistency_passed"]).reshape(-1)[0]):
            raise RuntimeError("pose-conditioned depth archive failed overlapping-window consistency")
        metric_scale_mode = scalar_text(payload["metric_scale_mode"], "metric_scale_mode")
        if metric_scale_mode != "nested_metric_branch":
            raise RuntimeError(
                f"pose-conditioned depth metric scale mode is not downstream eligible: {metric_scale_mode!r}"
            )
        source_conditioning.update({
            "camera_contract_sha256": source_contract_hash,
            "output_plane": source_output_plane,
            "intrinsics_match_source_rows": True,
            "depth_ray_geometry_reprojected": True,
            "depth_output_quantity": scalar_text(payload["depth_output_quantity"], "depth_output_quantity"),
            "pose_conditioning_mode": scalar_text(payload["pose_conditioning_mode"], "pose_conditioning_mode"),
            "camera_trajectory_source": scalar_text(payload["camera_trajectory_source"], "camera_trajectory_source"),
            "fixed_extrinsics_shape": list(conditioned_extrinsics.shape),
            "confidence_role": confidence_role,
            "overlap_consistency_passed": True,
            "metric_scale_mode": metric_scale_mode,
            "metric_scale_source": scalar_text(payload["metric_scale_source"], "metric_scale_source"),
        })
    elif source_conditioning_mode not in {"model_inferred_intrinsics", "legacy_unspecified"}:
        raise RuntimeError(f"unsupported source depth camera conditioning mode {source_conditioning_mode!r}")
    # Preserve source intrinsics for provenance, but do not relabel an unchanged
    # dense z raster as if it had been geometrically reprojected onto another K.
    # A camera-aware depth model call or explicit 3D reproject/z-buffer stage is
    # required for that stronger contract.
    same_rays = bool(np.allclose(source_intrinsics, resolved_rows, atol=1.0e-6, rtol=0.0))
    if not same_rays and not bool(args.allow_metadata_only_ray_relabel):
        raise RuntimeError(
            "source depth K differs from the requested camera-contract plane, but this adapter preserves the depth "
            "raster byte-for-byte and cannot reproject it onto new sensor rays. Rerun depth with the supplied sensor "
            "camera or add an explicit 3D reproject/z-buffer adapter. Use --allow-metadata-only-ray-relabel only for "
            "forensic reproduction of legacy invalid runs."
        )
    depth_hash_before = array_sha256(depth)
    frame_hash_before = array_sha256(frame_idx)

    payload["source_depth_active_intrinsics_fx_fy_cx_cy"] = np.asarray(payload["intrinsics_fx_fy_cx_cy"])
    payload["source_estimated_intrinsics_fx_fy_cx_cy"] = np.asarray(
        payload.get("model_camera_head_intrinsics_fx_fy_cx_cy", payload["intrinsics_fx_fy_cx_cy"])
    )
    payload["intrinsics_fx_fy_cx_cy"] = resolved_rows
    payload["focal_px"] = np.repeat(np.float64(np.sqrt(intrinsics[0] * intrinsics[1])), depth.shape[0])
    payload["intrinsics_source"] = np.asarray(str(contract.get("intrinsics_source") or contract.get("method")))
    payload["depth_provider"] = np.asarray(depth_provider)
    payload["calibration_authority"] = np.asarray(str(normalized["calibration_authority"]))
    payload["camera_contract_path"] = np.asarray(str(contract_path))
    payload["camera_contract_sha256"] = np.asarray(contract_hash)
    payload["camera_contract_plane"] = np.asarray(args.depth_plane)
    payload["A_depth_from_calibration"] = np.asarray(transform["A_actual_plane_from_calibration"], dtype=np.float64)
    payload["intrinsics_override_applied"] = np.asarray(not same_rays)
    payload["depth_ray_geometry_reprojected"] = np.asarray(source_depth_rays_reprojected)
    payload["metadata_only_ray_relabel_override"] = np.asarray(bool(not same_rays and args.allow_metadata_only_ray_relabel))
    payload["source_depth_archive_sha256"] = np.asarray(sha256_file(source))
    np.savez_compressed(output_path, **payload)

    with np.load(output_path, allow_pickle=False) as check:
        output_depth = np.asarray(check["depth"])
        output_frame_idx = np.asarray(check["frame_idx"])
        output_source_size = np.asarray(check["source_size"])
        output_intrinsics = np.asarray(check["intrinsics_fx_fy_cx_cy"])
        archived_source_intrinsics = np.asarray(check["source_depth_active_intrinsics_fx_fy_cx_cy"])
        archived_model_estimated_intrinsics = np.asarray(check["source_estimated_intrinsics_fx_fy_cx_cy"])
        output_confidence = np.asarray(check["confidence"]) if "confidence" in check.files else None
    model_estimated_intrinsics = np.asarray(
        payload.get("model_camera_head_intrinsics_fx_fy_cx_cy", source_intrinsics), dtype=np.float64
    )
    invariants = {
        "depth_array_equal": bool(np.array_equal(depth, output_depth, equal_nan=True)),
        "depth_array_sha256_equal": bool(depth_hash_before == array_sha256(output_depth)),
        "frame_idx_equal": bool(np.array_equal(frame_idx, output_frame_idx)),
        "frame_idx_sha256_equal": bool(frame_hash_before == array_sha256(output_frame_idx)),
        "source_size_equal": bool(np.array_equal(source_size, output_source_size)),
        "source_active_intrinsics_preserved": bool(np.array_equal(source_intrinsics, archived_source_intrinsics)),
        "source_model_estimated_intrinsics_preserved": bool(
            np.array_equal(model_estimated_intrinsics, archived_model_estimated_intrinsics)
        ),
        "confidence_array_equal": bool(
            (source_confidence is None and output_confidence is None)
            or (
                source_confidence is not None
                and output_confidence is not None
                and np.array_equal(source_confidence, output_confidence, equal_nan=True)
            )
        ),
        "output_intrinsics_float64": bool(output_intrinsics.dtype == np.dtype(np.float64)),
        "output_intrinsics_equal_contract_plane": bool(np.array_equal(output_intrinsics, resolved_rows)),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"depth camera-contract adapter invariant failed: {invariants}")
    delta = model_estimated_intrinsics - intrinsics[None, :]
    report = {
        "status": "ok",
        "method": "adapt_v19_depth_to_camera_contract",
        "claim_scope": (
            "fail-closed ray-contract binding; metric depth values, frame rows, raster size, and dtype are unchanged; "
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
        "depth_provider": depth_provider,
        "source_depth_camera_conditioning": source_conditioning,
        "depth_ray_geometry_reprojected": source_depth_rays_reprojected,
        "source_and_resolved_rays_match": same_rays,
        "metadata_only_ray_relabel_override": bool(not same_rays and args.allow_metadata_only_ray_relabel),
        "resolved_intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "camera_contract": summarize_contract(contract_path, contract, normalized),
        "image_transform": transform,
        "source_model_estimated_intrinsics_delta_from_resolved": {
            "median_fx_fy_cx_cy": np.median(delta, axis=0).tolist(),
            "max_abs_fx_fy_cx_cy": np.max(np.abs(delta), axis=0).tolist(),
        },
        "array_invariants": invariants,
        "source_depth_archive_sha256": sha256_file(source),
        "output_depth_archive_sha256": sha256_file(output_path),
        "downstream_contract": (
            "Visible geometry may consume this byte-identical archive only when source_and_resolved_rays_match=true. "
            "A metadata-only override is forensic legacy reproduction and must fail downstream metric backprojection."
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
    parser.add_argument(
        "--allow-metadata-only-ray-relabel",
        action="store_true",
        help="Forensic legacy reproduction only: preserve z bytes while replacing a disagreeing K. Downstream metric backprojection must reject this state.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = adapt(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
