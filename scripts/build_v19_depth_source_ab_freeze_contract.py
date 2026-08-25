#!/usr/bin/env python3
"""Build a fail-closed shared-upstream contract for a depth-provider A/B.

The contract is created once after shared prediction-side P01/P03b/P04/P05-P08
assets and one depth-source-independent anchor decision exist.  It binds every
RGB and SAM2 mask actually referenced by the branch inputs.  It never copies
assets and never consumes evaluation labels, released poses, or CAD geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "v19_depth_source_ab_freeze_contract_v1"
ANCHOR_SCHEMA = "v19_depth_source_ab_anchor_decision_v1"
ALLOWED_ANCHOR_EVIDENCE_ROLES = {
    "rgb",
    "sam2_mask",
    "projected_mano_silhouette",
    "semantic_identity",
}
REQUIRED_FORBIDDEN_ANCHOR_EVIDENCE_ROLES = {
    "metric_depth",
    "depth_confidence",
    "provider_specific_p09_score",
}


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


def require_file(path: Path, role: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {role}: {path}")
    return path


def asset(path: Path, role: str, frame_idx: int | None = None) -> dict[str, Any]:
    path = require_file(path, role)
    row: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if frame_idx is not None:
        row["frame_idx"] = int(frame_idx)
    return row


def scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"{name} must be scalar, got {array.shape}")
    return str(array.reshape(-1)[0])


def validate_anchor(
    anchor: dict[str, Any], object_id: str, frame_ids: list[int], visible: dict[int, bool]
) -> tuple[int, Path]:
    if anchor.get("schema") != ANCHOR_SCHEMA or anchor.get("status") != "ok":
        raise RuntimeError(f"anchor must use {ANCHOR_SCHEMA} with status=ok")
    if str(anchor.get("object_id")) != object_id:
        raise RuntimeError(f"anchor object mismatch: {anchor.get('object_id')!r} != {object_id!r}")
    if anchor.get("depth_source_independent") is not True:
        raise RuntimeError("anchor must explicitly declare depth_source_independent=true")
    roles = anchor.get("selection_evidence_roles")
    if not isinstance(roles, list) or not roles:
        raise RuntimeError("anchor must declare non-empty selection_evidence_roles")
    role_set = {str(value) for value in roles}
    unsupported = sorted(role_set - ALLOWED_ANCHOR_EVIDENCE_ROLES)
    if unsupported:
        raise RuntimeError(f"anchor uses unsupported/provider-dependent evidence roles: {unsupported}")
    forbidden = anchor.get("forbidden_evidence_roles_acknowledged")
    if not isinstance(forbidden, list) or not REQUIRED_FORBIDDEN_ANCHOR_EVIDENCE_ROLES.issubset(
        {str(value) for value in forbidden}
    ):
        raise RuntimeError(
            "anchor must acknowledge exclusion of metric_depth, depth_confidence, and provider_specific_p09_score"
        )
    frame = int(anchor.get("selected_anchor_frame_idx", -1))
    if frame not in frame_ids:
        raise RuntimeError(f"anchor frame {frame} is outside the shared timeline")
    if not visible.get(frame, False):
        raise RuntimeError(f"anchor frame {frame} is not visible in the frozen SAM2 track")
    review_path = require_file(Path(str(anchor.get("review_image"))), "depth-source-independent anchor review")
    review_hash = str(anchor.get("review_image_sha256") or "")
    if review_hash != sha256_file(review_path):
        raise RuntimeError("anchor review SHA256 is missing or does not match the reviewed image")
    return frame, review_path


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = require_file(args.manifest, "raw frame manifest")
    camera_path = require_file(args.camera_contract, "official camera contract")
    camera_npz_path = require_file(args.camera_intrinsics, "camera intrinsics archive")
    hawor_path = require_file(args.hawor_npz, "HaWoR prediction archive")
    object_plan_path = require_file(args.object_plan, "object plan")
    owlv2_prompt_path = require_file(args.owlv2_prompt, "OWLv2 prompt contract")
    owlv2_report_path = require_file(args.owlv2_report, "OWLv2 report")
    sam2_track_path = require_file(args.sam2_track, "SAM2 track")
    base_annotations_path = require_file(args.base_annotations, "base annotations")
    base_report_path = require_file(args.base_report, "base annotations report")
    mano_bridge_path = require_file(args.mano_bridge, "MANO bridge")
    anchor_path = require_file(args.anchor_decision, "depth-source-independent anchor decision")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.replace:
        raise RuntimeError(f"freeze contract exists; use --replace explicitly: {output}")

    manifest = load_json(manifest_path)
    frames = manifest.get("frames")
    if manifest.get("status") != "ok" or not isinstance(frames, list) or not frames:
        raise RuntimeError("raw frame manifest is not ready")
    frame_ids = [int(row["frame_idx"]) for row in frames]
    if frame_ids != sorted(set(frame_ids)):
        raise RuntimeError("raw frame timeline must be strictly increasing and unique")
    rgb_paths = [require_file(Path(str(row["rgb"])), f"RGB frame {idx}") for idx, row in zip(frame_ids, frames)]
    if any(Path(str(row.get("raw_frame_path"))).expanduser().resolve() != path for row, path in zip(frames, rgb_paths)):
        raise RuntimeError("raw manifest rgb/raw_frame_path rows disagree")
    source_videos = {str(Path(str(row["source_video"])).expanduser().resolve()) for row in frames}
    if len(source_videos) != 1:
        raise RuntimeError(f"raw manifest has multiple source videos: {sorted(source_videos)}")
    source_video = require_file(Path(next(iter(source_videos))), "source video")
    source_size = [int(frames[0]["source_width"]), int(frames[0]["source_height"])]
    if any([int(row["source_width"]), int(row["source_height"])] != source_size for row in frames):
        raise RuntimeError("raw manifest source size varies across frames")

    camera = load_json(camera_path)
    camera_frame_ids = [int(value) for value in camera.get("frame_ids") or []]
    if camera.get("status") != "ok" or camera_frame_ids != frame_ids:
        raise RuntimeError("official camera contract timeline disagrees with raw manifest")
    camera_K = np.asarray(camera.get("source_plane_intrinsics_fx_fy_cx_cy"), dtype=np.float64)
    if camera_K.shape != (4,) or not np.isfinite(camera_K).all() or np.any(camera_K[:2] <= 0.0):
        raise RuntimeError("camera contract lacks valid source-plane pinhole intrinsics")
    if str(camera.get("calibration_authority")) not in {
        "prediction_side_sensor_metadata",
        "dataset_sensor_calibration",
        "explicit_user_calibration",
    }:
        raise RuntimeError("camera contract is not prediction-side sensor metadata")

    with np.load(camera_npz_path, allow_pickle=False) as archive:
        if "frame_idx" in archive.files and np.asarray(archive["frame_idx"], dtype=np.int64).tolist() != frame_ids:
            raise RuntimeError("camera intrinsics archive timeline disagrees with raw manifest")

    with np.load(hawor_path, allow_pickle=False) as archive:
        required = {"frame_idx", "R_c2w", "t_c2w", "camera_intrinsics_fx_fy_cx_cy"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"HaWoR archive misses {missing}")
        hawor_frames = np.asarray(archive["frame_idx"], dtype=np.int64)
        R_c2w = np.asarray(archive["R_c2w"], dtype=np.float64)
        t_c2w = np.asarray(archive["t_c2w"], dtype=np.float64)
        hawor_K = np.asarray(archive["camera_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        if hawor_frames.tolist() != frame_ids or R_c2w.shape != (len(frame_ids), 3, 3) or t_c2w.shape != (
            len(frame_ids),
            3,
        ):
            raise RuntimeError("HaWoR camera trajectory disagrees with the shared timeline")
        if hawor_K.shape != (len(frame_ids), 4) or not np.allclose(hawor_K, camera_K[None], atol=1.0e-6, rtol=0.0):
            raise RuntimeError("HaWoR source-plane K disagrees with the official camera contract")
        if not np.isfinite(R_c2w).all() or not np.isfinite(t_c2w).all():
            raise RuntimeError("HaWoR trajectory contains non-finite values")
        video_sha = scalar_text(archive["video_sha256"], "HaWoR video_sha256") if "video_sha256" in archive.files else None
    source_video_sha = sha256_file(source_video)
    if video_sha is not None and video_sha != source_video_sha:
        raise RuntimeError("HaWoR source-video hash disagrees with the raw source video")

    plan = load_json(object_plan_path)
    plan_objects = [row for row in plan.get("objects") or [] if str(row.get("object_id")) == args.object_id]
    if (
        plan.get("status") != "ok"
        or str(plan.get("case")) != args.case_id
        or len(plan_objects) != 1
        or str(plan_objects[0].get("track_id")) != args.object_id
    ):
        raise RuntimeError(f"object plan does not bind case/object/track {args.case_id!r}/{args.object_id!r}")

    owlv2_prompt = load_json(owlv2_prompt_path)
    if (
        str(owlv2_prompt.get("case_id")) != args.case_id
        or str(owlv2_prompt.get("object_id")) != args.object_id
        or str(owlv2_prompt.get("track_id")) != args.object_id
        or owlv2_prompt.get("prompt_source") != "owlv2_text_grounded_detector_boxes"
        or not isinstance(owlv2_prompt.get("point_prompts"), list)
        or not owlv2_prompt["point_prompts"]
    ):
        raise RuntimeError("OWLv2 prompt contract does not bind the shared case/object/track")
    prompt_frames = [int(row["frame_idx"]) for row in owlv2_prompt["point_prompts"]]
    if len(prompt_frames) != len(set(prompt_frames)) or not set(prompt_frames).issubset(frame_ids):
        raise RuntimeError("OWLv2 prompt frames are duplicated or outside the shared timeline")

    owlv2_report = load_json(owlv2_report_path)
    if (
        owlv2_report.get("status") != "ok"
        or str(owlv2_report.get("case_id")) != args.case_id
        or str(owlv2_report.get("object_id")) != args.object_id
        or str(owlv2_report.get("track_id")) != args.object_id
        or Path(str(owlv2_report.get("raw_frame_manifest"))).expanduser().resolve() != manifest_path
        or Path(str(owlv2_report.get("output_prompt_json"))).expanduser().resolve() != owlv2_prompt_path
        or [int(value) for value in owlv2_report.get("prompt_frames") or []] != prompt_frames
        or owlv2_report.get("missing_prompt_frames") not in ([], None)
    ):
        raise RuntimeError("OWLv2 report is not a successful binding of the shared prompt contract")

    track = load_json(sam2_track_path)
    track_ids = sorted(int(value) for value in track)
    if track_ids != frame_ids:
        raise RuntimeError("SAM2 track timeline disagrees with raw manifest")
    mask_paths: list[Path] = []
    visible: dict[int, bool] = {}
    for frame in frame_ids:
        row = track[str(frame)]
        if not isinstance(row, dict):
            raise RuntimeError(f"SAM2 row {frame} is malformed")
        visible[frame] = bool(row.get("visible"))
        mask_paths.append(require_file(Path(str(row.get("mask_path"))), f"SAM2 mask {frame}"))

    base_report = load_json(base_report_path)
    base_outputs = base_report.get("outputs") if isinstance(base_report.get("outputs"), dict) else {}
    if (
        base_report.get("status") != "ok"
        or str(base_report.get("case")) != args.case_id
        or int(base_report.get("frame_count", -1)) != len(frame_ids)
        or Path(str(base_outputs.get("annotations"))).expanduser().resolve() != base_annotations_path
        or Path(str(base_outputs.get("mano_bridge"))).expanduser().resolve() != mano_bridge_path
    ):
        raise RuntimeError("base annotations report does not bind the shared case/annotations/MANO bridge")

    base = load_json(base_annotations_path)
    base_frames = base.get("frames")
    if base.get("status") != "ok" or not isinstance(base_frames, list):
        raise RuntimeError("base annotations are not ready")
    if [int(row["frame_idx"]) for row in base_frames] != frame_ids:
        raise RuntimeError("base annotation timeline disagrees with raw manifest")
    for row, rgb in zip(base_frames, rgb_paths):
        if Path(str(row.get("raw_frame_path"))).expanduser().resolve() != rgb:
            raise RuntimeError(f"base annotation RGB path disagrees at frame {row.get('frame_idx')}")
    base_inputs = base.get("v19_inputs") if isinstance(base.get("v19_inputs"), dict) else {}
    expected_base_paths = {
        "raw_frame_manifest": manifest_path,
        "calibration_contract": camera_path,
        "hawor_npz": hawor_path,
        "object_plan": object_plan_path,
        "mano_bridge_npz": mano_bridge_path,
    }
    for key, expected in expected_base_paths.items():
        actual = base_inputs.get(key)
        if not actual or Path(str(actual)).expanduser().resolve() != expected:
            raise RuntimeError(f"base annotation input {key} is not bound to the shared asset")

    anchor = load_json(anchor_path)
    anchor_frame, anchor_review_path = validate_anchor(anchor, args.object_id, frame_ids, visible)

    fixed_assets = [
        asset(source_video, "source_video"),
        asset(manifest_path, "raw_frame_manifest"),
        asset(camera_path, "official_camera_contract"),
        asset(camera_npz_path, "camera_intrinsics_archive"),
        asset(hawor_path, "hawor_prediction_archive"),
        asset(object_plan_path, "object_plan"),
        asset(owlv2_prompt_path, "owlv2_prompt_contract"),
        asset(owlv2_report_path, "owlv2_report"),
        asset(sam2_track_path, "sam2_track"),
        asset(base_annotations_path, "base_annotations"),
        asset(base_report_path, "base_annotations_report"),
        asset(mano_bridge_path, "mano_bridge"),
        asset(anchor_path, "depth_source_independent_anchor_decision"),
        asset(anchor_review_path, "depth_source_independent_anchor_review"),
    ]
    rgb_assets = [asset(path, "raw_rgb", frame) for frame, path in zip(frame_ids, rgb_paths)]
    mask_assets = [asset(path, "sam2_mask", frame) for frame, path in zip(frame_ids, mask_paths)]
    identity = {
        "schema": SCHEMA,
        "case_id": str(args.case_id),
        "object_id": str(args.object_id),
        "frame_ids": frame_ids,
        "source_size_wh": source_size,
        "official_source_intrinsics_fx_fy_cx_cy": camera_K.tolist(),
        "camera_contract_sha256": sha256_file(camera_path),
        "hawor_archive_sha256": sha256_file(hawor_path),
        "anchor_frame": anchor_frame,
        "fixed_assets": fixed_assets,
        "rgb_assets": rgb_assets,
        "sam2_mask_assets": mask_assets,
    }
    contract_id = canonical_sha256(identity)
    report = {
        "schema": SCHEMA,
        "status": "frozen_shared_upstream_ready_for_depth_provider_pair",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract_id,
        "claim_scope": (
            "prediction-side shared-upstream byte contract only; no reference pose, CAD, released depth, "
            "evaluation label, or provider-specific P09 evidence is consumed"
        ),
        "case_id": str(args.case_id),
        "object_id": str(args.object_id),
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "source_size_wh": source_size,
        "official_source_intrinsics_fx_fy_cx_cy": camera_K.tolist(),
        "camera_contract_sha256": sha256_file(camera_path),
        "hawor_archive_sha256": sha256_file(hawor_path),
        "anchor": {
            "frame_idx": anchor_frame,
            "decision_path": str(anchor_path),
            "review_path": str(anchor_review_path),
            "review_sha256": sha256_file(anchor_review_path),
            "depth_source_independent": True,
            "selection_evidence_roles": anchor["selection_evidence_roles"],
            "forbidden_evidence_roles_acknowledged": anchor["forbidden_evidence_roles_acknowledged"],
        },
        "asset_counts": {
            "fixed": len(fixed_assets),
            "raw_rgb": len(rgb_assets),
            "sam2_mask": len(mask_assets),
            "total": len(fixed_assets) + len(rgb_assets) + len(mask_assets),
        },
        "fixed_assets": fixed_assets,
        "rgb_assets": rgb_assets,
        "sam2_mask_assets": mask_assets,
        "branch_policy": {
            "allowed_variable": "external_metric_depth_provider_and_its_declared_confidence_semantics",
            "required_branches": ["unidepth_official_K", "da3_nested_official_K_hawor_conditioned"],
            "shared_assets_must_be_read_not_regenerated": True,
            "p03b_p04_p05_p06_p07_p08_anchor_rerun_inside_branch_forbidden": True,
            "provider_dependent_anchor_reranking_forbidden": True,
            "sam3d_pointmap": None,
        },
    }
    write_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "contract_id": contract_id,
        "frame_count": len(frame_ids),
        "asset_counts": report["asset_counts"],
        "anchor_frame": anchor_frame,
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--camera-intrinsics", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--owlv2-prompt", type=Path, required=True)
    parser.add_argument("--owlv2-report", type=Path, required=True)
    parser.add_argument("--sam2-track", type=Path, required=True)
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--mano-bridge", type=Path, required=True)
    parser.add_argument("--anchor-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
