#!/usr/bin/env python3
"""Load and validate mesh-bound per-face signed-distance authority.

A watertight mesh may carry an inside/outside topology without every closure
face being a physically observed collision boundary.  This module keeps those
two contracts separate.  New shared signed-geometry reports must bind a boolean
``signed_distance_eligible`` array to the exact topology mesh bytes.  Legacy
reports that do not opt into the per-face contract retain their historical
all-face behaviour, but that compatibility state is explicit in the returned
report and is never used by the HOT3D shared signed-geometry path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

LOCAL_AUTHORITY_POLICY = "local_observation_authority_faces_only"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"signed face authority field {field} must be scalar, got {array.shape}")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def load_signed_face_authority(
    source_report: dict[str, Any],
    *,
    source_report_path: Path | None,
    mesh_path: Path,
    mesh_face_count: int,
) -> dict[str, Any]:
    """Return a validated authority mask and an audit dictionary.

    New reports opt in with either ``signed_face_authority_required=true`` or
    ``signed_geometry_consumer_policy=local_observation_authority_faces_only``.
    If such a report is malformed, ``usable`` is false; callers must deactivate
    signed physical factors rather than falling back to all faces.
    """
    readiness = (
        source_report.get("geometry_readiness")
        if isinstance(source_report.get("geometry_readiness"), dict)
        else {}
    )
    outputs = source_report.get("outputs") if isinstance(source_report.get("outputs"), dict) else {}
    policy = str(readiness.get("signed_geometry_consumer_policy") or "")
    shared_proxy_source = str(readiness.get("signed_geometry_source") or "")
    is_shared_proxy_source = (
        shared_proxy_source
        == "shared_prediction_mask_depth_direct_pose_voxel_reconstruction"
    )
    unsafe_shared_all_faces_policy = bool(
        policy == "all_faces_of_validated_shared_proxy_signed_eligible"
        or (is_shared_proxy_source and policy != LOCAL_AUTHORITY_POLICY)
    )
    required = bool(
        readiness.get("signed_face_authority_required") is True
        or policy == LOCAL_AUTHORITY_POLICY
        or unsafe_shared_all_faces_policy
    )
    declared_ready = readiness.get("signed_geometry_ready") is True
    mesh_path = mesh_path.expanduser().resolve()
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise RuntimeError(f"signed topology mesh is missing or empty: {mesh_path}")
    mesh_sha256 = sha256_file(mesh_path)

    authority_value = (
        readiness.get("signed_face_authority_npz")
        or outputs.get("signed_face_authority_npz")
    )
    expected_authority_sha256 = str(
        readiness.get("signed_face_authority_npz_sha256")
        or (source_report.get("output_sha256") or {}).get("signed_face_authority_npz")
        or ""
    )
    expected_mesh_sha256 = str(
        readiness.get("signed_face_authority_mesh_sha256")
        or ""
    )

    if unsafe_shared_all_faces_policy:
        return {
            "usable": False,
            "required": True,
            "state": "inactive_unsafe_shared_proxy_all_faces_policy_requires_rebuild",
            "consumer_policy": policy,
            "source_report": None if source_report_path is None else str(source_report_path),
            "source_mesh": str(mesh_path),
            "source_mesh_sha256": mesh_sha256,
            "face_count": int(mesh_face_count),
            "eligible_face_count": 0,
            "eligible_face_fraction": 0.0,
            "mask": np.zeros((int(mesh_face_count),), dtype=bool),
        }

    if not required:
        return {
            "usable": True,
            "required": False,
            "state": "legacy_all_faces_without_local_authority_contract",
            "consumer_policy": policy or "legacy_unspecified_all_faces",
            "source_report": None if source_report_path is None else str(source_report_path),
            "source_mesh": str(mesh_path),
            "source_mesh_sha256": mesh_sha256,
            "face_count": int(mesh_face_count),
            "eligible_face_count": int(mesh_face_count),
            "eligible_face_fraction": 1.0 if int(mesh_face_count) else 0.0,
            "mask": np.ones((int(mesh_face_count),), dtype=bool),
        }

    if not declared_ready:
        return {
            "usable": False,
            "required": True,
            "state": "inactive_signed_geometry_not_declared_ready",
            "consumer_policy": policy,
            "source_report": None if source_report_path is None else str(source_report_path),
            "source_mesh": str(mesh_path),
            "source_mesh_sha256": mesh_sha256,
            "face_count": int(mesh_face_count),
            "eligible_face_count": 0,
            "eligible_face_fraction": 0.0,
            "mask": np.zeros((int(mesh_face_count),), dtype=bool),
        }
    if not authority_value:
        return {
            "usable": False,
            "required": True,
            "state": "missing_required_signed_face_authority_npz",
            "consumer_policy": policy,
            "source_report": None if source_report_path is None else str(source_report_path),
            "source_mesh": str(mesh_path),
            "source_mesh_sha256": mesh_sha256,
            "face_count": int(mesh_face_count),
            "eligible_face_count": 0,
            "eligible_face_fraction": 0.0,
            "mask": np.zeros((int(mesh_face_count),), dtype=bool),
        }

    authority_path = Path(str(authority_value)).expanduser().resolve()
    if not authority_path.is_file() or authority_path.stat().st_size <= 0:
        raise RuntimeError(f"required signed face authority NPZ is missing or empty: {authority_path}")
    authority_sha256 = sha256_file(authority_path)
    if not expected_authority_sha256:
        raise RuntimeError("signed face authority report does not bind the NPZ SHA256")
    if authority_sha256 != expected_authority_sha256:
        raise RuntimeError(
            "signed face authority NPZ hash mismatch: "
            f"expected={expected_authority_sha256} actual={authority_sha256}"
        )
    if not expected_mesh_sha256:
        raise RuntimeError(
            "signed face authority report does not bind the topology mesh SHA256"
        )
    if expected_mesh_sha256 != mesh_sha256:
        raise RuntimeError(
            "signed face authority report is bound to different topology mesh bytes: "
            f"expected={expected_mesh_sha256} actual={mesh_sha256}"
        )

    with np.load(authority_path, allow_pickle=False) as archive:
        required_arrays = {
            "face_id",
            "signed_distance_eligible",
            "provenance_code",
            "source_mesh_sha256",
            "source_mesh_face_count",
            "authority_schema",
            "authority_role",
            "readiness_passed",
        }
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise RuntimeError(f"signed face authority NPZ lacks arrays {missing}: {authority_path}")
        face_id = np.asarray(archive["face_id"], dtype=np.int64)
        eligible = np.asarray(archive["signed_distance_eligible"], dtype=bool)
        archive_mesh_sha256 = _scalar_text(archive["source_mesh_sha256"], "source_mesh_sha256")
        archive_face_count = int(np.asarray(archive["source_mesh_face_count"]).reshape(-1)[0])
        provenance_code = np.asarray(archive["provenance_code"], dtype=np.uint8)
        authority_schema = _scalar_text(
            archive["authority_schema"], "authority_schema"
        )
        authority_role = _scalar_text(
            archive["authority_role"], "authority_role"
        )
        readiness_passed = bool(
            np.asarray(archive["readiness_passed"]).reshape(-1)[0]
        )
    expected_ids = np.arange(int(mesh_face_count), dtype=np.int64)
    if face_id.shape != expected_ids.shape or not np.array_equal(face_id, expected_ids):
        raise RuntimeError(
            f"signed face authority IDs do not exactly cover mesh faces: {face_id.shape} vs {expected_ids.shape}"
        )
    if eligible.shape != (int(mesh_face_count),):
        raise RuntimeError(
            f"signed face authority mask shape {eligible.shape} != mesh face count {mesh_face_count}"
        )
    if authority_schema != "v19_mesh_bound_signed_face_authority_v1":
        raise RuntimeError(
            f"unsupported signed face authority schema {authority_schema!r}"
        )
    if authority_role != "selected_collision_surface":
        raise RuntimeError(
            f"signed face authority has non-consumable role {authority_role!r}"
        )
    if not readiness_passed:
        raise RuntimeError(
            "signed face authority sidecar did not pass geometry readiness"
        )
    if provenance_code.shape != eligible.shape:
        raise RuntimeError("signed face authority provenance_code shape mismatch")
    if not np.array_equal(eligible, provenance_code == 3):
        raise RuntimeError(
            "signed face authority mask disagrees with local-authority provenance code"
        )
    if archive_face_count != int(mesh_face_count):
        raise RuntimeError(
            f"signed face authority declared face count {archive_face_count} != mesh face count {mesh_face_count}"
        )
    if archive_mesh_sha256 != mesh_sha256:
        raise RuntimeError(
            "signed face authority NPZ is bound to different topology mesh bytes: "
            f"archive={archive_mesh_sha256} actual={mesh_sha256}"
        )
    eligible_count = int(np.count_nonzero(eligible))
    return {
        "usable": bool(eligible_count > 0),
        "required": True,
        "state": "active_mesh_bound_local_signed_face_authority" if eligible_count else "inactive_zero_eligible_signed_faces",
        "consumer_policy": policy,
        "source_report": None if source_report_path is None else str(source_report_path),
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": mesh_sha256,
        "authority_npz": str(authority_path),
        "authority_npz_sha256": authority_sha256,
        "face_count": int(mesh_face_count),
        "eligible_face_count": eligible_count,
        "eligible_face_fraction": float(eligible_count / max(int(mesh_face_count), 1)),
        "mask": eligible,
        "provenance_code": provenance_code,
    }
