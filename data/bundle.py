from __future__ import annotations

"""Immutable, lazy observation bundles for annotation episodes.

The manifest is prediction-side only. It names source observations and assets; it
never requires a solver trajectory or optimizer status. Heavy arrays remain in
their original archives and are read lazily.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import json
import threading

import numpy as np

SCHEMA = "ego.annotation.observation_bundle.v1"
HAND_SLOTS = ("left", "right")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _absolute(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _validate_identity(value: Mapping[str, Any], label: str) -> None:
    for key in ("path", "sha256"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"{label} requires {key}")
    if len(value["sha256"]) != 64:
        raise ValueError(f"{label} requires a SHA256 digest")


def _positive_gt_flags(value: Any, path: str = "$") -> list[str]:
    flags: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in {"gt_consumed", "uses_gt", "used_gt", "ground_truth_consumed",
                              "gt_used_for_initialization", "gt_used_as_solver_input"} and child is True:
                flags.append(child_path)
            flags.extend(_positive_gt_flags(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            flags.extend(_positive_gt_flags(child, f"{path}[{index}]"))
    return flags


def validate_manifest(manifest: Mapping[str, Any], manifest_path: str | Path | None = None) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported observation bundle schema: {manifest.get('schema')!r}")
    gt_flags = _positive_gt_flags(manifest)
    if gt_flags:
        raise ValueError(f"observation bundle contains positive GT provenance flags: {gt_flags[:8]}")
    for key in ("bundle_id", "clip_id", "source_video_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"manifest requires nonempty {key}")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("manifest requires identity")
    for key in ("source_video", "code", "config"):
        value = identity.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"identity requires {key}")
        _validate_identity(value, f"identity.{key}")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("manifest requires sources")
    for required in ("annotations", "mano_archive", "object_mesh"):
        value = sources.get(required)
        if not isinstance(value, Mapping):
            raise ValueError(f"sources requires {required}")
        _validate_identity(value, f"sources.{required}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("prediction_only") is not True or provenance.get("legacy_optimizer_required") is not False:
        raise ValueError("manifest provenance must be prediction-only and optimizer-independent")
    forbidden = ("/pose_fits/", "/mano_interval_correction/", "/render_state/",
                 "/p15b_shared_signed_geometry/")
    for label, value in sources.items():
        if isinstance(value, Mapping) and isinstance(value.get("path"), str):
            normalized = str(value["path"]).replace("\\", "/")
            if any(marker in normalized for marker in forbidden):
                raise ValueError(f"source {label} points to forbidden solved/downstream artifact: {normalized}")
    frame_range = manifest.get("frame_range")
    if (not isinstance(frame_range, list) or len(frame_range) != 2
            or any(type(v) is not int for v in frame_range) or frame_range[1] < frame_range[0]):
        raise ValueError("frame_range must be inclusive integer [start, end]")
    slots = manifest.get("entity_slots")
    if not isinstance(slots, Mapping) or slots.get("hands") != list(HAND_SLOTS):
        raise ValueError("entity_slots.hands must be ['left', 'right']")
    objects = slots.get("objects")
    if not isinstance(objects, list) or not objects or any(not isinstance(v, str) or not v for v in objects):
        raise ValueError("entity_slots.objects must be a nonempty string list")
    if len(set(objects)) != len(objects):
        raise ValueError("duplicate object slots")
    if manifest_path is not None:
        base = Path(manifest_path).resolve().parent
        for label, value in sources.items():
            if isinstance(value, Mapping) and isinstance(value.get("path"), str):
                path = _absolute(base, value["path"])
                if label == "mano_assets":
                    if not path.is_dir():
                        raise FileNotFoundError(f"missing MANO asset directory: {path}")
                    continue
                if not path.is_file():
                    raise FileNotFoundError(f"missing {label}: {path}")
                if value.get("sha256") and file_sha256(path) != value["sha256"]:
                    raise ValueError(f"{label} content identity mismatch: {path}")


def prepared_manifest(
    *,
    bundle_id: str,
    clip_id: str,
    source_video_id: str,
    source_video: str | Path,
    annotations: str | Path,
    mano_archive: str | Path,
    object_mesh: str | Path,
    object_id: str,
    frame_range: tuple[int, int],
    code_identity: tuple[str, str],
    config_identity: tuple[str, str],
    mano_bridge: str | Path | None = None,
    contact_factors: str | Path | None = None,
    completion_report: str | Path | None = None,
    mano_assets: str | Path | None = None,
    signed_face_authority: str | Path | None = None,
) -> dict[str, Any]:
    """Build a manifest over existing prepared observations without copying data.

    This adapter is also the compatibility path for pre-export prepared runs.
    The object baseline is the source centroid/full-pose initializer, while the
    named completed mesh supplies the actual reconstructed geometry.
    """
    def artifact(path: str | Path) -> dict[str, str]:
        p = Path(path).expanduser().resolve()
        return {"path": str(p), "sha256": file_sha256(p)}

    sources: dict[str, Any] = {
        "annotations": artifact(annotations),
        "mano_archive": artifact(mano_archive),
        "object_mesh": artifact(object_mesh),
    }
    for name, path in (("mano_bridge", mano_bridge), ("contact_factors", contact_factors),
                       ("completion_report", completion_report), ("signed_face_authority", signed_face_authority)):
        if path is not None:
            sources[name] = artifact(path)
    if mano_assets is not None:
        asset_root = Path(mano_assets).expanduser().resolve()
        sources["mano_assets"] = {
            "path": str(asset_root),
            "left_sha256": file_sha256(asset_root / "MANO_LEFT.pkl"),
            "right_sha256": file_sha256(asset_root / "MANO_RIGHT.pkl"),
        }
    return {
        "schema": SCHEMA,
        "bundle_id": bundle_id,
        "clip_id": clip_id,
        "source_video_id": source_video_id,
        "identity": {
            "source_video": artifact(source_video),
            "code": {"path": code_identity[0], "sha256": code_identity[1]},
            "config": {"path": config_identity[0], "sha256": config_identity[1]},
        },
        "sources": sources,
        "frame_range": [int(frame_range[0]), int(frame_range[1])],
        "entity_slots": {"hands": list(HAND_SLOTS), "objects": [str(object_id)]},
        "provenance": {
            "prediction_only": True,
            "legacy_optimizer_required": False,
            "object_pose_role": "source_observation_initializer_not_solved_trajectory",
            "object_geometry_role": "completed_reconstructed_mesh_for_policy_pose_hypothesis",
            "signed_geometry_role": "requires separate mesh-bound authority and readiness; otherwise inactive",
        },
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class EpisodeBundle:
    manifest_path: Path
    manifest: Mapping[str, Any]
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    @classmethod
    def open(cls, path: str | Path, *, verify: bool = True) -> "EpisodeBundle":
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(payload, manifest_path if verify else None)
        # Round-trip JSON detaches caller-owned mutable containers.
        immutable = MappingProxyType(json.loads(json.dumps(payload, allow_nan=False)))
        return cls(manifest_path, immutable)

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def source_video_id(self) -> str:
        return str(self.manifest["source_video_id"])

    def source_path(self, name: str) -> Path:
        value = self.manifest["sources"].get(name)
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise KeyError(f"bundle has no source {name!r}")
        return _absolute(self.manifest_path.parent, value["path"])

    def load(self) -> Mapping[str, Any]:
        """Load the declared clip slice once; all returned ndarrays are read-only."""
        with self._lock:
            loaded = self._cache.get("arrays")
            if loaded is None:
                loaded = MappingProxyType(_load_prepared_sources(self))
                self._cache["arrays"] = loaded
            return loaded


def _load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        import trimesh
    except ImportError as exc:  # production must surface a missing real geometry loader
        raise RuntimeError("trimesh is required to load reconstructed object geometry") from exc
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"object mesh contains no triangles: {path}")
        geom = trimesh.util.concatenate(meshes)
    vertices = np.asarray(geom.vertices, dtype=np.float32)
    faces = np.asarray(geom.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid object mesh: {path}")
    return vertices, faces, bool(geom.is_watertight)


def _load_prepared_sources(bundle: EpisodeBundle) -> dict[str, Any]:
    manifest = bundle.manifest
    start, end = map(int, manifest["frame_range"])
    annotations = json.loads(bundle.source_path("annotations").read_text(encoding="utf-8"))
    frames_by_id = {int(row.get("frame_idx", pos)): row for pos, row in enumerate(annotations.get("frames", []))}
    frame_ids = np.arange(start, end + 1, dtype=np.int64)
    if any(int(i) not in frames_by_id for i in frame_ids):
        raise ValueError("annotations do not cover declared frame_range")
    frames = [frames_by_id[int(i)] for i in frame_ids]
    declared_mano = bundle.source_path("mano_archive")
    embedded_mano_paths: set[Path] = set()
    for frame in frames:
        for hand in frame.get("hands", []):
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            params = metric.get("mano_params") if isinstance(metric.get("mano_params"), dict) else {}
            reference = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
            raw = params.get("source_hawor_npz") or reference.get("source_hawor_npz")
            if isinstance(raw, str):
                path = Path(raw).expanduser().resolve()
                if path.is_dir():
                    candidates = sorted(path.glob("*with_track_support*.npz")) + sorted(path.glob("*.npz"))
                    if len(candidates):
                        path = candidates[0].resolve()
                embedded_mano_paths.add(path)
    if embedded_mano_paths and embedded_mano_paths != {declared_mano}:
        raise ValueError(f"manifest MANO archive does not match annotation-bound source pointer: {embedded_mano_paths}")
    times = np.asarray([row["time_s"] for row in frames], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("bundle timestamps must be finite and strictly increasing")
    camera_T = np.asarray([
        row["camera"].get("T_world_camera_metric", row["camera"].get("T_world_camera"))
        for row in frames
    ], dtype=np.float64)
    intrinsics = np.asarray([row["camera"]["intrinsics_fx_fy_cx_cy"] for row in frames], dtype=np.float64)
    if camera_T.shape != (len(frames), 4, 4) or intrinsics.shape != (len(frames), 4):
        raise ValueError("invalid camera timeline")
    size = np.asarray([frames[0].get("source_width"), frames[0].get("source_height")], dtype=np.int64)

    with np.load(bundle.source_path("mano_archive"), allow_pickle=False) as archive:
        source_ids = np.asarray(archive["frame_idx"], dtype=np.int64)
        lookup = {int(v): i for i, v in enumerate(source_ids.tolist())}
        if any(int(v) not in lookup for v in frame_ids):
            raise ValueError("MANO archive does not cover declared frame_range")
        row_ids = np.asarray([lookup[int(v)] for v in frame_ids], dtype=np.int64)
        hands: dict[str, dict[str, np.ndarray]] = {}
        for side in HAND_SLOTS:
            required = [f"{side}_{suffix}" for suffix in (
                "root_orient_axis_angle", "hand_pose_axis_angle", "betas", "trans_world_m",
                "vertices_world_m", "joints_world_m", "valid", "faces")]
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise ValueError(f"MANO archive lacks {missing}")
            hands[side] = {
                "root_orient": _frozen_array(archive[f"{side}_root_orient_axis_angle"][row_ids], np.float32),
                "articulation": _frozen_array(archive[f"{side}_hand_pose_axis_angle"][row_ids], np.float32),
                "betas": _frozen_array(archive[f"{side}_betas"][row_ids], np.float32),
                "translation": _frozen_array(archive[f"{side}_trans_world_m"][row_ids], np.float32),
                "vertices": _frozen_array(archive[f"{side}_vertices_world_m"][row_ids], np.float32),
                "joints": _frozen_array(archive[f"{side}_joints_world_m"][row_ids], np.float32),
                "valid": _frozen_array(archive[f"{side}_valid"][row_ids].astype(bool)),
                "faces": _frozen_array(archive[f"{side}_faces"], np.int64),
            }

    object_slots = list(manifest["entity_slots"]["objects"])
    object_id = object_slots[0]
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    object_valid: list[bool] = []
    observed_points: list[np.ndarray] = []
    for row in frames:
        matches = [value for value in row.get("objects", [])
                   if str(value.get("object_id", value.get("track_id", ""))).split(":")[-1]
                   == str(object_id).split(":")[-1]]
        if len(matches) != 1:
            rotations.append(np.eye(3)); translations.append(np.zeros(3)); object_valid.append(False)
            observed_points.append(np.zeros((0, 3), dtype=np.float32)); continue
        obj = matches[0]
        pose = obj.get("reconstructed_geometry_pose") or {}
        R = np.asarray(pose.get("rotation_world_from_canonical_matrix", np.eye(3)), dtype=np.float64)
        t = np.asarray(pose.get("translation_world_m", [np.nan] * 3), dtype=np.float64)
        geom = obj.get("visible_geometry_candidate") or {}
        points = np.asarray(geom.get("world_vertices_sample_m", []), dtype=np.float32).reshape(-1, 3)
        valid = R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all()
        rotations.append(R if valid else np.eye(3)); translations.append(t if valid else np.zeros(3)); object_valid.append(valid)
        observed_points.append(points)
    mesh_vertices, mesh_faces, mesh_watertight = _load_mesh_arrays(bundle.source_path("object_mesh"))
    signed_geometry_ready = False
    try:
        completion_payload = json.loads(bundle.source_path("completion_report").read_text(encoding="utf-8"))
        readiness = completion_payload.get("geometry_readiness") if isinstance(completion_payload.get("geometry_readiness"), dict) else {}
        signed_geometry_ready = readiness.get("signed_geometry_ready") is True
    except KeyError:
        completion_payload = {}

    signed_authority = np.zeros(len(mesh_faces), dtype=bool)
    try:
        authority_path = bundle.source_path("signed_face_authority")
        with np.load(authority_path, allow_pickle=False) as authority_archive:
            face_id = np.asarray(authority_archive["face_id"], dtype=np.int64)
            eligible = np.asarray(authority_archive["signed_distance_eligible"], dtype=bool)
        if not np.array_equal(face_id, np.arange(len(mesh_faces), dtype=np.int64)) or eligible.shape != (len(mesh_faces),):
            raise ValueError("signed face authority does not exactly cover object mesh topology")
        signed_authority = eligible.copy()
    except KeyError:
        pass
    signed_authority.setflags(write=False)

    contact: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        factor_path = bundle.source_path("contact_factors")
    except KeyError:
        factor_path = None
    if factor_path is not None:
        report = json.loads(factor_path.read_text(encoding="utf-8"))
        for row in report.get("factor_rows", []):
            if row.get("factor_family") == "contact_patch":
                key = (int(row["frame_idx"]), str(row["hand_side"]))
                if key[0] < start or key[0] > end:
                    continue
                contact[key] = {
                    "state": str(row.get("state", "inactive_missing_support")),
                    "prior": float(row.get("contact_patch_prior_probability", 0.0)),
                    "weight": float(row.get("weight", 0.0)),
                    "band_m": float(row.get("contact_patch_band_m", 0.0)),
                    "margin_m": float(row.get("contact_patch_target_margin_m", 0.0)),
                    "support_uncertainty_m": float(row.get("contact_patch_support_uncertainty_m", 0.0)),
                    "anchor_allowed": bool(row.get("contact_anchor_residual_allowed", False)),
                    "blockers": tuple(map(str, row.get("contact_anchor_blockers", []))),
                }
        frame_position = {int(value): pos for pos, value in enumerate(frame_ids.tolist())}
        for row in report.get("ownership_rows", []):
            key = (int(row.get("frame_idx", -1)), str(row.get("hand_side", "")))
            if key[0] not in frame_position or key[1] not in hands:
                continue
            entry = contact.setdefault(key, {"state": "inactive_missing_contact_row", "prior": 0.0,
                "weight": 0.0, "band_m": 0.0, "margin_m": 0.0, "support_uncertainty_m": 0.0,
                "anchor_allowed": False, "blockers": ()})
            npz_raw = row.get("depth_order_first_surface_npz_path")
            expected_sha = str(row.get("depth_order_first_surface_npz_sha256") or "")
            if not isinstance(npz_raw, str) or not expected_sha:
                entry["depth_order_state"] = "inactive_missing_independent_first_surface"
                continue
            npz_path = Path(npz_raw).expanduser().resolve()
            if file_sha256(npz_path) != expected_sha:
                raise ValueError(f"depth-order first-surface identity mismatch: {npz_path}")
            with np.load(npz_path, allow_pickle=False) as depth_archive:
                depth = np.asarray(depth_archive["depth_mask_plane_m"], dtype=np.float32)
                support = np.asarray(depth_archive["support_mask"], dtype=bool)
                affine = np.asarray(depth_archive["A_mask_from_source_coordinate_model"], dtype=np.float64)
            pos = frame_position[key[0]]
            vertices = hands[key[1]]["vertices"][pos]
            cam = (vertices - camera_T[pos, :3, 3][None, :]) @ camera_T[pos, :3, :3]
            fx, fy, cx, cy = intrinsics[pos]
            source_uv = np.column_stack([fx * cam[:, 0] / np.maximum(cam[:, 2], 1e-6) + cx,
                                         fy * cam[:, 1] / np.maximum(cam[:, 2], 1e-6) + cy])
            homogeneous = np.column_stack([source_uv, np.ones(len(source_uv))]) @ affine.T
            uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-12)
            u = np.rint(uv[:, 0]).astype(int); v = np.rint(uv[:, 1]).astype(int)
            valid = ((cam[:, 2] > 1e-5) & (u >= 0) & (u < depth.shape[1]) &
                     (v >= 0) & (v < depth.shape[0]))
            ids = np.flatnonzero(valid)
            if len(ids):
                ids = ids[support[v[ids], u[ids]]]
                z = depth[v[ids], u[ids]].astype(np.float64)
                finite = np.isfinite(z) & (z > 1e-5)
                ids, z = ids[finite], z[finite]
            else:
                z = np.zeros((0,), dtype=np.float64)
            entry.update({
                "depth_order_state": "active_independent_P09_first_surface_depth_order_query" if len(ids) else "inactive_no_supported_projected_vertices",
                "depth_order_vertex_ids": _frozen_array(ids, np.int64),
                "depth_order_surface_depth_m": _frozen_array(z, np.float64),
                "translation_support_count": int(len(ids)),
                "depth_order_npz_path": str(npz_path), "depth_order_npz_sha256": expected_sha,
            })
    frozen_contact = {key: MappingProxyType(value) for key, value in contact.items()}

    result: dict[str, Any] = {
        "frame_idx": _frozen_array(frame_ids), "time_s": _frozen_array(times),
        "T_world_camera": _frozen_array(camera_T), "intrinsics": _frozen_array(intrinsics),
        "image_size_wh": _frozen_array(size),
        "hands": MappingProxyType({side: MappingProxyType(value) for side, value in hands.items()}),
        "objects": MappingProxyType({object_id: MappingProxyType({
            "rotation": _frozen_array(rotations), "translation": _frozen_array(translations),
            "valid": _frozen_array(object_valid), "vertices_canonical": _frozen_array(mesh_vertices),
            "faces": _frozen_array(mesh_faces), "watertight": mesh_watertight,
            "signed_geometry_ready": signed_geometry_ready,
            "signed_authority": signed_authority,
            "observed_points_world": tuple(_frozen_array(v) for v in observed_points),
            "pose_provenance": "source_observation_initializer_not_solved_trajectory",
            "geometry_provenance": "completed_reconstructed_mesh",
        })}),
        "contact": MappingProxyType(frozen_contact),
        "identity": manifest["identity"], "provenance": manifest.get("provenance", {}),
    }
    return result
