"""Versioned, hash-bound source-observation manifests.

The manifest is the only output contract of the raw-video main entry. It binds
exogenous perception, direct registration, and reconstructed geometry without
including a refined/fitted/policy annotation state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.state import sha

SCHEMA = "ego.annotation.observation_bundle.v1"
BUNDLE_VERSION = 1
FORBIDDEN_ARTIFACT_ROLES = {
    "optimized_pose",
    "optimized_hands",
    "accepted_mano",
    "policy_checkpoint",
    "policy_annotation",
    "render",
}


def artifact_record(run_root: Path, path: Path, role: str) -> dict[str, Any]:
    root = Path(run_root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source artifact escapes run root: {resolved}") from exc
    if role in FORBIDDEN_ARTIFACT_ROLES:
        raise ValueError(f"optimized/policy artifact role is forbidden in source bundle: {role}")
    if not resolved.is_file():
        raise FileNotFoundError(f"source artifact is not a file: {resolved}")
    return {"path": relative.as_posix(), "sha256": sha(resolved), "role": role}


def build_source_observation_manifest(
    *,
    run_root: Path,
    case: str,
    object_id: str,
    track_id: str,
    frame_ids: list[int],
    time_s: list[float],
    source_timing: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, str]],
    mano_assets: Path | None = None,
    wilor_root: Path | None = None,
) -> dict[str, Any]:
    if not frame_ids or frame_ids != sorted(set(frame_ids)):
        raise ValueError("source observation frame_ids must be non-empty, sorted, and unique")
    if len(time_s) != len(frame_ids):
        raise ValueError("source observation timestamps must match frame_ids")
    required = {
        "runtime_input_contract",
        "code_manifest",
        "model_asset_manifest",
        "raw_video",
        "raw_frames",
        "camera_calibration",
        "conditioned_depth",
        "hawor_world",
        "mano_bridge",
        "visible_annotations",
        "completion_report",
        "reconstructed_mesh",
        "appearance_anchor_decision",
        "interaction_judgments",
        "interaction_factors",
    }
    missing = sorted(required.difference(artifacts))
    if missing:
        raise ValueError(f"source observation bundle is missing required artifacts: {missing}")
    records = {
        name: artifact_record(run_root, path, role)
        for name, (path, role) in sorted(artifacts.items())
    }
    runtime_record = records["runtime_input_contract"]
    runtime_path = Path(run_root).resolve() / runtime_record["path"]
    runtime_contract = json.loads(runtime_path.read_text(encoding="utf-8"))
    bundle_id = str(runtime_contract.get("run_id") or records["runtime_input_contract"]["sha256"])
    sources = {
        "annotations": records["visible_annotations"],
        "mano_archive": records["hawor_world"],
        "object_mesh": records["reconstructed_mesh"],
        "mano_bridge": records["mano_bridge"],
        "contact_factors": records["interaction_factors"],
        "completion_report": records["completion_report"],
        "conditioned_depth": records["conditioned_depth"],
        "camera_calibration": records["camera_calibration"],
        "raw_object_masks": records["raw_object_masks"],
        "owned_object_masks": records["owned_object_masks"],
        "metric_eligibility": records["metric_eligibility"],
        "interaction_judgments": records["interaction_judgments"],
        "appearance_anchor_decision": records["appearance_anchor_decision"],
    }
    if mano_assets is not None:
        asset_root = Path(mano_assets).expanduser().resolve()
        left = asset_root / "MANO_LEFT.pkl"
        right = asset_root / "MANO_RIGHT.pkl"
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"MANO asset directory lacks left/right models: {asset_root}")
        sources["mano_assets"] = {
            "path": str(asset_root),
            "left_sha256": sha(left),
            "right_sha256": sha(right),
            "wilor_root": str(Path(wilor_root).expanduser().resolve()) if wilor_root is not None else None,
        }
    return {
        "schema": SCHEMA,
        "bundle_version": BUNDLE_VERSION,
        "bundle_id": bundle_id,
        "clip_id": str(case),
        "source_video_id": records["raw_video"]["sha256"],
        "identity": {
            "source_video": records["raw_video"],
            "code": records["code_manifest"],
            "config": records["model_asset_manifest"],
        },
        "sources": sources,
        "frame_range": [int(frame_ids[0]), int(frame_ids[-1])],
        "entity_slots": {"hands": ["left", "right"], "objects": [str(object_id)]},
        "prediction_side_only": True,
        "contains_optimized_state": False,
        "contains_policy_results": False,
        "legacy_optimizer_required": False,
        "initial_state_authority": "raw_perception_and_direct_observation_registration",
        "case": str(case),
        "object_id": str(object_id),
        "track_id": str(track_id),
        "frame_ids": [int(value) for value in frame_ids],
        "time_s": [float(value) for value in time_s],
        "source_timing": dict(source_timing),
        "artifacts": records,
        "provenance": {
            "prediction_only": True,
            "legacy_optimizer_required": False,
            "object_pose_role": "reconstructed_geometry_pose_source_initializer_not_solved_trajectory",
        },
        "initial_state": {
            "hands": ["hawor_world", "mano_bridge"],
            "object_geometry": ["visible_annotations", "completion_report", "reconstructed_mesh"],
            "object_pose": ["visible_annotations"],
            "camera_and_depth": ["camera_calibration", "conditioned_depth"],
            "evidence": ["raw_object_masks", "owned_object_masks", "metric_eligibility", "interaction_judgments", "interaction_factors"],
        },
    }


def write_source_observation_manifest(output: Path, **kwargs: Any) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"source observation manifest is create-only: {output}")
    payload = build_source_observation_manifest(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return payload


def load_source_observation_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("unsupported source observation bundle schema/version")
    if payload.get("contains_optimized_state") is not False or payload.get("contains_policy_results") is not False:
        raise ValueError("source observation bundle contains downstream state")
    root = path.parent
    for name, record in payload.get("artifacts", {}).items():
        artifact = (root / record["path"]).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact {name} escapes bundle root") from exc
        if record.get("role") in FORBIDDEN_ARTIFACT_ROLES:
            raise ValueError(f"artifact {name} has downstream role")
        if not artifact.is_file() or sha(artifact) != record.get("sha256"):
            raise ValueError(f"artifact {name} is missing or changed")
    return payload
