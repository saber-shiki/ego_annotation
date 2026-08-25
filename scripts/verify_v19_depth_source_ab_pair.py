#!/usr/bin/env python3
"""Verify one frozen-upstream UniDepth/DA3 depth-provider pair, fail closed."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FREEZE_SCHEMA = "v19_depth_source_ab_freeze_contract_v1"
PAIR_SCHEMA = "v19_depth_source_ab_pair_contract_v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array))
    return digest.hexdigest()


def require_file(path: Path, role: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {role}: {path}")
    return path


def scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{name} must be scalar, got {array.shape}")
    return str(array.reshape(-1)[0])


def scalar_bool(value: np.ndarray, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{name} must be scalar, got {array.shape}")
    return bool(array.reshape(-1)[0])


def verify_frozen_assets(contract: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    rows = []
    for key in ("fixed_assets", "rgb_assets", "sam2_mask_assets"):
        value = contract.get(key)
        if not isinstance(value, list):
            raise RuntimeError(f"freeze contract lacks {key}")
        rows.extend(value)
    required_fixed_roles = {
        "source_video",
        "raw_frame_manifest",
        "official_camera_contract",
        "camera_intrinsics_archive",
        "hawor_prediction_archive",
        "object_plan",
        "owlv2_prompt_contract",
        "owlv2_report",
        "sam2_track",
        "base_annotations",
        "base_annotations_report",
        "mano_bridge",
        "depth_source_independent_anchor_decision",
        "depth_source_independent_anchor_review",
    }
    fixed_roles = [str(row.get("role")) for row in contract.get("fixed_assets") or []]
    if set(fixed_roles) != required_fixed_roles or len(fixed_roles) != len(required_fixed_roles):
        raise RuntimeError("freeze contract fixed asset roles are incomplete or duplicated")
    seen: set[str] = set()
    for row in rows:
        path = Path(str(row.get("path"))).expanduser().resolve()
        if str(path) in seen:
            failures.append({"path": str(path), "reason": "duplicate_asset_path"})
            continue
        seen.add(str(path))
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
        elif int(path.stat().st_size) != int(row.get("bytes", -1)):
            failures.append({"path": str(path), "reason": "size_mismatch"})
        elif sha256_file(path) != row.get("sha256"):
            failures.append({"path": str(path), "reason": "sha256_mismatch"})
    if failures:
        raise RuntimeError(f"frozen shared upstream changed: {failures[:10]}")
    counts = contract.get("asset_counts") or {}
    if int(counts.get("total", -1)) != len(rows):
        raise RuntimeError("freeze contract asset count is inconsistent")
    return {"status": "ok", "checked": len(rows), "failures": []}


def camera_from_contract(contract: dict[str, Any]) -> tuple[Path, str, np.ndarray, list[int]]:
    candidates = [
        row for row in contract["fixed_assets"] if row.get("role") == "official_camera_contract"
    ]
    if len(candidates) != 1:
        raise RuntimeError("freeze contract must bind exactly one official camera contract")
    path = require_file(Path(candidates[0]["path"]), "official camera contract")
    expected_hash = str(contract.get("camera_contract_sha256"))
    if sha256_file(path) != expected_hash:
        raise RuntimeError("official camera contract hash changed")
    K = np.asarray(contract.get("official_source_intrinsics_fx_fy_cx_cy"), dtype=np.float64)
    frame_ids = [int(value) for value in contract.get("frame_ids") or []]
    if K.shape != (4,) or not frame_ids:
        raise RuntimeError("freeze contract lacks official K/timeline")
    return path, expected_hash, K, frame_ids


def expected_w2c(contract: dict[str, Any], frame_ids: list[int]) -> tuple[np.ndarray, Path]:
    candidates = [
        row for row in contract["fixed_assets"] if row.get("role") == "hawor_prediction_archive"
    ]
    if len(candidates) != 1:
        raise RuntimeError("freeze contract must bind exactly one HaWoR archive")
    path = require_file(Path(candidates[0]["path"]), "HaWoR archive")
    with np.load(path, allow_pickle=False) as archive:
        source_frames = np.asarray(archive["frame_idx"], dtype=np.int64)
        if source_frames.tolist() != frame_ids:
            raise RuntimeError("frozen HaWoR timeline changed")
        R_c2w = np.asarray(archive["R_c2w"], dtype=np.float64)
        t_c2w = np.asarray(archive["t_c2w"], dtype=np.float64)
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(frame_ids), axis=0)
    c2w[:, :3, :3] = R_c2w
    c2w[:, :3, 3] = t_c2w
    return np.linalg.inv(c2w), path


def verify_depth_archive(
    path: Path,
    expected_provider: str,
    camera_hash: str,
    official_K: np.ndarray,
    frame_ids: list[int],
    expected_w2c_rows: np.ndarray,
    frozen_hawor_path: Path,
    frozen_source_size: list[int],
) -> dict[str, Any]:
    path = require_file(path, f"{expected_provider} depth archive")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "frame_idx",
            "depth",
            "confidence",
            "source_size",
            "intrinsics_fx_fy_cx_cy",
            "camera_conditioning_mode",
            "depth_ray_geometry_reprojected",
            "inference_camera_contract_sha256",
            "inference_camera_output_plane",
            "camera_contract_sha256",
            "camera_contract_plane",
            "metadata_only_ray_relabel_override",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"{expected_provider} archive misses {missing}")
        archive_frames = np.asarray(archive["frame_idx"], dtype=np.int64)
        depth = np.asarray(archive["depth"])
        confidence = np.asarray(archive["confidence"])
        source_size = np.asarray(archive["source_size"], dtype=np.int64)
        intrinsics = np.asarray(archive["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        provider = scalar_text(archive["depth_provider"], "depth_provider") if "depth_provider" in archive.files else "unidepth"
        conditioning = scalar_text(archive["camera_conditioning_mode"], "camera_conditioning_mode")
        contract_hash = scalar_text(archive["inference_camera_contract_sha256"], "camera contract hash")
        output_plane = scalar_text(archive["inference_camera_output_plane"], "camera output plane")
        bound_contract_hash = scalar_text(archive["camera_contract_sha256"], "bound camera contract hash")
        bound_plane = scalar_text(archive["camera_contract_plane"], "bound camera contract plane")
        rays_reprojected = scalar_bool(archive["depth_ray_geometry_reprojected"], "depth ray geometry")
        metadata_override = scalar_bool(archive["metadata_only_ray_relabel_override"], "metadata override")
        values = {key: np.asarray(archive[key]) for key in archive.files}
    if provider != expected_provider:
        raise RuntimeError(f"depth provider mismatch: {provider!r} != {expected_provider!r}")
    if archive_frames.tolist() != frame_ids or depth.ndim != 3 or depth.shape[0] != len(frame_ids):
        raise RuntimeError(f"{provider} timeline/depth shape disagrees with freeze contract")
    if confidence.shape != depth.shape:
        raise RuntimeError(f"{provider} confidence shape disagrees with depth")
    if source_size.tolist() != [int(depth.shape[2]), int(depth.shape[1])]:
        raise RuntimeError(f"{provider} source_size disagrees with depth raster")
    if source_size.tolist() != [int(value) for value in frozen_source_size]:
        raise RuntimeError(f"{provider} source_size disagrees with the frozen RGB source plane")
    if intrinsics.shape != (len(frame_ids), 4) or not np.allclose(
        intrinsics, official_K[None], atol=1.0e-6, rtol=0.0
    ):
        raise RuntimeError(f"{provider} active rays disagree with frozen official K")
    if (
        contract_hash != camera_hash
        or bound_contract_hash != camera_hash
        or output_plane != "source_rgb"
        or bound_plane != "source_rgb"
        or not rays_reprojected
        or metadata_override
    ):
        raise RuntimeError(f"{provider} archive is not exact-ray bound to the frozen official camera")
    if not np.isfinite(depth).any() or not np.isfinite(confidence).any():
        raise RuntimeError(f"{provider} archive has no finite depth/confidence")

    provider_contract: dict[str, Any]
    if provider == "unidepth":
        if conditioning != "provided_pinhole_intrinsics":
            raise RuntimeError(f"UniDepth conditioning mode is invalid: {conditioning!r}")
        provider_contract = {
            "model_role": "external_metric_depth_baseline",
            "camera_conditioning_mode": conditioning,
            "confidence_role": "provider_native_predicted_error_higher_is_worse",
        }
    elif provider == "depth_anything_3":
        required_da3 = {
            "pose_conditioning_mode",
            "camera_trajectory_source",
            "inference_camera_extrinsics_w2c",
            "confidence_role",
            "overlap_consistency_passed",
            "overlap_consistency_required",
            "metric_scale_mode",
            "metric_scale_source",
        }
        missing_da3 = sorted(required_da3 - set(values))
        if missing_da3:
            raise RuntimeError(f"DA3 archive misses {missing_da3}")
        if conditioning != "provided_pinhole_intrinsics_and_metric_extrinsics":
            raise RuntimeError(f"DA3 conditioning mode is invalid: {conditioning!r}")
        scale_mode = scalar_text(values["metric_scale_mode"], "metric_scale_mode")
        confidence_role = scalar_text(values["confidence_role"], "confidence_role")
        trajectory_source = Path(
            scalar_text(values["camera_trajectory_source"], "camera_trajectory_source")
        ).expanduser().resolve()
        archive_w2c = np.asarray(values["inference_camera_extrinsics_w2c"], dtype=np.float64)
        if scale_mode != "nested_metric_branch":
            raise RuntimeError(f"DA3 metric scale mode is not eligible: {scale_mode!r}")
        if confidence_role != "predicted_error_proxy_higher_is_worse":
            raise RuntimeError(f"DA3 confidence role is invalid: {confidence_role!r}")
        if not scalar_bool(values["overlap_consistency_passed"], "overlap consistency"):
            raise RuntimeError("DA3 overlap consistency failed")
        if trajectory_source != frozen_hawor_path:
            raise RuntimeError("DA3 camera trajectory source is not the frozen HaWoR archive")
        if archive_w2c.shape != expected_w2c_rows.shape or not np.allclose(
            archive_w2c, expected_w2c_rows, atol=1.0e-6, rtol=0.0
        ):
            raise RuntimeError("DA3 conditioned W2C differs from the frozen HaWoR trajectory")
        provider_contract = {
            "model_role": "external_metric_depth_intervention",
            "camera_conditioning_mode": conditioning,
            "pose_conditioning_mode": scalar_text(values["pose_conditioning_mode"], "pose_conditioning_mode"),
            "camera_trajectory_source": str(trajectory_source),
            "metric_scale_mode": scale_mode,
            "metric_scale_source": scalar_text(values["metric_scale_source"], "metric_scale_source"),
            "overlap_consistency_passed": True,
            "overlap_consistency_required": scalar_bool(
                values["overlap_consistency_required"], "overlap consistency required"
            ),
            "confidence_role": confidence_role,
            "fixed_w2c_max_abs_error": float(np.max(np.abs(archive_w2c - expected_w2c_rows))),
        }
    else:
        raise RuntimeError(f"unsupported depth provider {provider!r}")

    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "provider": provider,
        "frame_count": len(frame_ids),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_array_sha256": array_sha256(depth),
        "confidence_array_sha256": array_sha256(confidence),
        "intrinsics_array_sha256": array_sha256(intrinsics),
        "camera_contract_sha256": contract_hash,
        "camera_output_plane": output_plane,
        "depth_ray_geometry_reprojected": True,
        "metadata_only_ray_relabel_override": False,
        **provider_contract,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = require_file(args.freeze_contract, "shared-upstream freeze contract")
    freeze = load_json(freeze_path)
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != "frozen_shared_upstream_ready_for_depth_provider_pair":
        raise RuntimeError("shared-upstream freeze contract is not ready")
    policy = freeze.get("branch_policy") if isinstance(freeze.get("branch_policy"), dict) else {}
    if policy.get("allowed_variable") != "external_metric_depth_provider_and_its_declared_confidence_semantics":
        raise RuntimeError("freeze contract does not isolate the external metric-depth provider")
    if policy.get("shared_assets_must_be_read_not_regenerated") is not True:
        raise RuntimeError("freeze contract permits branch-local upstream regeneration")
    if policy.get("sam3d_pointmap", "missing") is not None:
        raise RuntimeError("freeze contract does not preserve SAM3D pointmap=None")
    anchor = freeze.get("anchor") if isinstance(freeze.get("anchor"), dict) else {}
    if anchor.get("depth_source_independent") is not True:
        raise RuntimeError("freeze contract anchor is not depth-source independent")
    role_set = {str(value) for value in anchor.get("selection_evidence_roles") or []}
    if not role_set or not role_set.issubset({"rgb", "sam2_mask", "projected_mano_silhouette", "semantic_identity"}):
        raise RuntimeError("freeze contract anchor uses provider-dependent evidence roles")
    forbidden = {str(value) for value in anchor.get("forbidden_evidence_roles_acknowledged") or []}
    if not {"metric_depth", "depth_confidence", "provider_specific_p09_score"}.issubset(forbidden):
        raise RuntimeError("freeze contract anchor does not exclude provider-dependent evidence")
    frozen_assets = verify_frozen_assets(freeze)
    fixed_by_role = {str(row["role"]): row for row in freeze["fixed_assets"]}
    anchor_decision = load_json(Path(fixed_by_role["depth_source_independent_anchor_decision"]["path"]))
    if (
        int(anchor_decision.get("selected_anchor_frame_idx", -1)) != int(anchor.get("frame_idx", -2))
        or anchor_decision.get("selection_evidence_roles") != anchor.get("selection_evidence_roles")
        or anchor_decision.get("forbidden_evidence_roles_acknowledged")
        != anchor.get("forbidden_evidence_roles_acknowledged")
        or Path(str(anchor_decision.get("review_image"))).expanduser().resolve()
        != Path(str(anchor.get("review_path"))).expanduser().resolve()
        or str(anchor_decision.get("review_image_sha256")) != str(anchor.get("review_sha256"))
    ):
        raise RuntimeError("freeze contract anchor summary disagrees with its byte-bound decision")
    identity = {
        "schema": FREEZE_SCHEMA,
        "case_id": freeze.get("case_id"),
        "object_id": freeze.get("object_id"),
        "frame_ids": freeze.get("frame_ids"),
        "source_size_wh": freeze.get("source_size_wh"),
        "official_source_intrinsics_fx_fy_cx_cy": freeze.get("official_source_intrinsics_fx_fy_cx_cy"),
        "camera_contract_sha256": freeze.get("camera_contract_sha256"),
        "hawor_archive_sha256": freeze.get("hawor_archive_sha256"),
        "anchor_frame": (freeze.get("anchor") or {}).get("frame_idx"),
        "fixed_assets": freeze.get("fixed_assets"),
        "rgb_assets": freeze.get("rgb_assets"),
        "sam2_mask_assets": freeze.get("sam2_mask_assets"),
    }
    recomputed_contract_id = canonical_sha256(identity)
    if freeze.get("contract_id") != recomputed_contract_id:
        raise RuntimeError("freeze contract ID does not match its canonical shared-upstream identity")
    _camera_path, camera_hash, official_K, frame_ids = camera_from_contract(freeze)
    frozen_source_size = [int(value) for value in freeze.get("source_size_wh") or []]
    if len(frozen_source_size) != 2:
        raise RuntimeError("freeze contract lacks frozen source size")
    w2c, hawor_path = expected_w2c(freeze, frame_ids)
    branches = {
        "unidepth_official_K": verify_depth_archive(
            args.unidepth_depth,
            "unidepth",
            camera_hash,
            official_K,
            frame_ids,
            w2c,
            hawor_path,
            frozen_source_size,
        ),
        "da3_nested_official_K_hawor_conditioned": verify_depth_archive(
            args.da3_depth,
            "depth_anything_3",
            camera_hash,
            official_K,
            frame_ids,
            w2c,
            hawor_path,
            frozen_source_size,
        ),
    }
    if branches["unidepth_official_K"]["depth_array_sha256"] == branches[
        "da3_nested_official_K_hawor_conditioned"
    ]["depth_array_sha256"]:
        raise RuntimeError("UniDepth and DA3 depth arrays are byte-identical; intervention is not active")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.replace:
        raise RuntimeError(f"pair contract exists; use --replace explicitly: {output}")
    report = {
        "schema": PAIR_SCHEMA,
        "status": "ready_for_frozen_upstream_depth_provider_branches",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "prediction-side pairing contract only; no reference labels or backend ranking consumed",
        "freeze_contract": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
            "contract_id": freeze.get("contract_id"),
            "frozen_asset_verification": frozen_assets,
        },
        "case_id": freeze.get("case_id"),
        "object_id": freeze.get("object_id"),
        "anchor": freeze.get("anchor"),
        "frame_count": len(frame_ids),
        "camera_contract_sha256": camera_hash,
        "official_source_intrinsics_fx_fy_cx_cy": official_K.tolist(),
        "fixed_hawor_w2c_array_sha256": array_sha256(w2c.astype(np.float32)),
        "only_intervention": "external_metric_depth_provider_and_declared_provider_specific_confidence_semantics",
        "branches": branches,
        "required_downstream_policy": {
            "shared_RGB_K_HaWoR_object_plan_OWLv2_SAM2_base_annotations_anchor": True,
            "rebuild_P09_and_later_depth_dependent_state_per_branch": True,
            "same_anchor_frame_required": int(freeze["anchor"]["frame_idx"]),
            "same_geometry_backend_seed_and_config_required": True,
            "SAM3D_full_RGB_object_owned_mask_pointmap_none": True,
            "provider_specific_confidence_threshold_calibration_required": True,
            "cross_provider_numeric_confidence_threshold_reuse_forbidden": True,
        },
    }
    write_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "freeze_contract_id": freeze.get("contract_id"),
        "frame_count": len(frame_ids),
        "anchor_frame": freeze["anchor"]["frame_idx"],
        "branches": {name: {"provider": row["provider"], "sha256": row["sha256"]} for name, row in branches.items()},
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-contract", type=Path, required=True)
    parser.add_argument("--unidepth-depth", type=Path, required=True)
    parser.add_argument("--da3-depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
