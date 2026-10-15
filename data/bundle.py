from __future__ import annotations

"""Immutable source-observation bundles for full annotation episodes.

The bundle records source measurements and qualification state. It never needs a
solver trajectory: candidate geometry belongs to :mod:`envs.state`, while factor
membership and source evidence remain immutable here.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import json
import threading

import numpy as np

from data.immutable import freeze, thaw, frozen_array
from data.interaction import load_interaction_evidence
from data.timeline import declared_ids, join_annotation_rows, timeline_times
from observations.contracts import SUPPORTED_SCHEMAS, validate_v2

SCHEMA = "ego.annotation.observation_bundle.v1"
HAND_SLOTS = ("left", "right")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    return frozen_array(value, dtype)


def _absolute(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _artifact(path: str | Path) -> dict[str, str]:
    value = Path(path).expanduser().resolve()
    return {"path": str(value), "sha256": file_sha256(value)}


def _entity_alias(value: str) -> str:
    return str(value).split(":", 1)[-1]


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


def _require_artifact(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64:
        raise ValueError(f"{label} requires path and SHA256")


def _validate_source_value(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            _require_artifact(value, label)
            return
        for key, child in value.items():
            _validate_source_value(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_source_value(child, f"{label}[{index}]")


def _verify_source_value(value: Any, base: Path, label: str) -> None:
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            _require_artifact(value, label)
            path = _absolute(base, value["path"])
            if not path.is_file():
                raise FileNotFoundError(f"missing {label}: {path}")
            if file_sha256(path) != value["sha256"]:
                raise ValueError(f"{label} content identity mismatch: {path}")
            return
        for key, child in value.items():
            _verify_source_value(child, base, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_source_value(child, base, f"{label}[{index}]")


def validate_manifest(manifest: Mapping[str, Any], manifest_path: str | Path | None = None) -> None:
    manifest = thaw(manifest)
    if manifest.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported observation bundle schema: {manifest.get('schema')!r}")
    if manifest.get("contains_optimized_state") is True or manifest.get("contains_policy_results") is True:
        raise ValueError("source observation bundle declares downstream optimized/policy state")
    if manifest.get("prediction_side_only") not in (None, True):
        raise ValueError("bundle must be prediction-side only")
    flags = _positive_gt_flags(manifest)
    if flags:
        raise ValueError(f"observation bundle contains positive GT provenance flags: {flags[:8]}")
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
        _require_artifact(value, f"identity.{key}")
    artifacts = manifest.get("artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, Mapping):
            raise ValueError("artifacts must be a mapping when supplied")
        for name, record in artifacts.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"artifact {name} is not a mapping")
            _require_artifact(record, f"artifacts.{name}")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("manifest requires sources")
    from observations.object.rgb_evidence import SCHEMA as RGB_SCHEMA, EXTENSION
    rgb_names = {'rgb_source_npz', 'rgb_source_report'}
    extensions = manifest.get('extensions', {})
    if not isinstance(extensions, Mapping):
        raise ValueError('observation extensions must be a mapping')
    rgb_extension = extensions.get(EXTENSION)
    if rgb_names.intersection(sources) or rgb_extension is not None:
        if not rgb_names.issubset(sources) or rgb_extension != RGB_SCHEMA:
            raise ValueError('RGB source extension requires versioned paired NPZ/report')
        for name, role in (('rgb_source_npz', 'rgb_source_arrays'), ('rgb_source_report', 'rgb_source_report')):
            if not isinstance(sources[name], Mapping) or sources[name].get('role') != role:
                raise ValueError('RGB source artifact role mismatch')
        for name in ('conditioned_depth', 'raw_frames', 'surface_report'):
            if name not in sources:
                raise ValueError(f'RGB source extension missing bound input: {name}')
    if 'visual_geometry_report' in sources and manifest.get('schema') != 'ego.annotation.observation_bundle.v2':
        raise ValueError('complete visual geometry requires an explicit v2 bundle')
    for required in ("annotations", "mano_archive", "object_mesh"):
        value = sources.get(required)
        if not isinstance(value, Mapping):
            raise ValueError(f"sources requires {required}")
        _require_artifact(value, f"sources.{required}")
    _validate_source_value({key: value for key, value in sources.items() if key != "mano_assets"}, "sources")
    assets = sources.get("mano_assets")
    if assets is not None:
        if not isinstance(assets, Mapping) or not isinstance(assets.get("path"), str):
            raise ValueError("mano_assets requires a path")
        for side in ("left_sha256", "right_sha256", "wrapper_sha256"):
            if not isinstance(assets.get(side), str) or len(assets[side]) != 64:
                raise ValueError(f"mano_assets requires {side}")
        if not isinstance(assets.get("wrapper_path"), str):
            raise ValueError("mano_assets requires wrapper_path")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("prediction_only") is not True or provenance.get("legacy_optimizer_required") is not False:
        raise ValueError("manifest provenance must be prediction-only and optimizer-independent")
    forbidden = ("/pose_fits/", "/mano_interval_correction/", "/render_state/", "/p15b_shared_signed_geometry/")
    for text in json.dumps(sources, sort_keys=True).replace("\\", "/").split('"'):
        if any(marker in text for marker in forbidden):
            raise ValueError(f"source points to forbidden solved/downstream artifact: {text}")
    frame_range = manifest.get("frame_range")
    if (not isinstance(frame_range, list) or len(frame_range) != 2 or any(type(v) is not int for v in frame_range)
            or frame_range[1] < frame_range[0]):
        raise ValueError("frame_range must be inclusive integer [start, end]")
    if "frame_ids" in manifest:
        declared_ids(manifest)
    if "time_s" in manifest:
        timeline_times(manifest["time_s"], len(declared_ids(manifest)))
    validate_v2(manifest)
    slots = manifest.get("entity_slots")
    if not isinstance(slots, Mapping) or slots.get("hands") != list(HAND_SLOTS):
        raise ValueError("entity_slots.hands must be ['left', 'right']")
    objects = slots.get("objects")
    if not isinstance(objects, list) or not objects or any(not isinstance(v, str) or not v for v in objects) or len(set(objects)) != len(objects):
        raise ValueError("entity_slots.objects must be a unique nonempty string list")
    if manifest_path is not None:
        base = Path(manifest_path).resolve().parent
        _verify_source_value(identity, base, "identity")
        if artifacts is not None:
            _verify_source_value(artifacts, base, "artifacts")
        for name, value in sources.items():
            if name != "mano_assets":
                _verify_source_value(value, base, f"sources.{name}")
        if assets is not None:
            root = _absolute(base, assets["path"])
            if not root.is_dir():
                raise FileNotFoundError(f"missing MANO asset directory: {root}")
            for filename, key in (("MANO_LEFT.pkl", "left_sha256"), ("MANO_RIGHT.pkl", "right_sha256")):
                if file_sha256(root / filename) != assets[key]:
                    raise ValueError(f"MANO asset identity mismatch: {root / filename}")
            wrapper = _absolute(base, assets["wrapper_path"])
            if file_sha256(wrapper) != assets["wrapper_sha256"]:
                raise ValueError(f"WiLoR wrapper identity mismatch: {wrapper}")


def prepared_manifest(
    *, bundle_id: str, clip_id: str, source_video_id: str, source_video: str | Path,
    annotations: str | Path, mano_archive: str | Path,
    object_mesh: str | Path | Mapping[str, str | Path], object_id: str | Sequence[str],
    frame_range: tuple[int, int], code_identity: tuple[str, str], config_identity: tuple[str, str],
    mano_bridge: str | Path | None = None, contact_factors: str | Path | None = None,
    factor_reports: Sequence[str | Path] = (), completion_report: str | Path | None = None,
    mano_assets: str | Path | None = None, signed_face_authority: str | Path | Mapping[str, str | Path] | None = None,
    collision_mesh: str | Path | Mapping[str, str | Path] | None = None,
    frame_ids: Sequence[int] | None = None,
    interaction_judgments: str | Path | None = None,
) -> dict[str, Any]:
    """Create a source-only manifest over existing prepared observations.

    ``object_mesh`` may map every declared object ID to a pose mesh. Optional
    collision meshes and signed authority artifacts remain independently bound.
    """
    object_ids = [object_id] if isinstance(object_id, str) else list(object_id)
    if not object_ids:
        raise ValueError("at least one object_id is required")
    def per_object(value: Any, entity: str) -> Any:
        return value.get(entity) if isinstance(value, Mapping) else value
    sources: dict[str, Any] = {"annotations": _artifact(annotations), "mano_archive": _artifact(mano_archive)}
    first_mesh = per_object(object_mesh, object_ids[0])
    if first_mesh is None:
        raise ValueError("object_mesh lacks first declared entity")
    sources["object_mesh"] = _artifact(first_mesh)
    completion_collision: Path | None = None
    if completion_report is not None:
        completion_payload = json.loads(Path(completion_report).read_text(encoding="utf-8"))
        outputs = completion_payload.get("outputs") if isinstance(completion_payload.get("outputs"), Mapping) else {}
        raw_collision = outputs.get("collision_eligible_mesh_labeled")
        if not isinstance(raw_collision, str) or not Path(raw_collision).is_file():
            raise ValueError("completion_report requires a readable collision_eligible_mesh_labeled output")
        completion_collision = Path(raw_collision).resolve()
    assets: dict[str, Any] = {}
    for entity in object_ids:
        pose = per_object(object_mesh, entity)
        if pose is None:
            raise ValueError(f"object_mesh lacks declared entity {entity!r}")
        item: dict[str, Any] = {"pose_mesh": _artifact(pose)}
        collision = per_object(collision_mesh, entity)
        if collision is None and completion_collision is not None:
            if len(object_ids) != 1:
                raise ValueError("multi-object completion reports require an explicit collision_mesh mapping")
            collision = completion_collision
        if collision is not None:
            item["collision_mesh"] = _artifact(collision)
        elif completion_report is not None:
            raise ValueError(f"missing manifest-bound collision mesh for {entity!r}")
        authority = per_object(signed_face_authority, entity)
        if authority is not None:
            item["signed_face_authority"] = _artifact(authority)
        assets[entity] = item
    sources["object_assets"] = assets
    for name, path in (("mano_bridge", mano_bridge), ("contact_factors", contact_factors), ("completion_report", completion_report), ("interaction_judgments", interaction_judgments)):
        if path is not None:
            sources[name] = _artifact(path)
    if factor_reports:
        sources["factor_reports"] = [_artifact(path) for path in factor_reports]
    if mano_assets is not None:
        root = Path(mano_assets).expanduser().resolve()
        wrapper = root.parent / "wilor" / "models" / "mano_wrapper.py"
        sources["mano_assets"] = {"path": str(root), "left_sha256": file_sha256(root / "MANO_LEFT.pkl"),
                                  "right_sha256": file_sha256(root / "MANO_RIGHT.pkl"),
                                  "wrapper_path": str(wrapper), "wrapper_sha256": file_sha256(wrapper)}
    return {
        "schema": SCHEMA, "bundle_id": bundle_id, "clip_id": clip_id, "source_video_id": source_video_id,
        "identity": {"source_video": _artifact(source_video), "code": {"path": code_identity[0], "sha256": code_identity[1]},
                     "config": {"path": config_identity[0], "sha256": config_identity[1]}},
        "sources": sources, "frame_range": [int(frame_range[0]), int(frame_range[1])],
        **({"frame_ids": list(frame_ids)} if frame_ids is not None else {}),
        "entity_slots": {"hands": list(HAND_SLOTS), "objects": [str(value) for value in object_ids]},
        "provenance": {"prediction_only": True, "legacy_optimizer_required": False,
                       "object_pose_role": "source_observation_initializer_not_solved_trajectory",
                       "object_geometry_role": "completed_reconstructed_mesh_for_policy_pose_hypothesis",
                       "signed_geometry_role": "requires mesh-bound collision authority and readiness"},
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    manifest = thaw(manifest)
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
        return cls(manifest_path, freeze(json.loads(json.dumps(payload, allow_nan=False))))

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
        with self._lock:
            if "arrays" not in self._cache:
                self._cache["arrays"] = MappingProxyType(_load_prepared_sources(self))
            return self._cache["arrays"]

    def load_observations(self) -> Mapping[str, Any]:
        """Load preprocessing views without creating the legacy solver contact patch.

        This is not an environment reset and does not load MANO model weights.
        Independent P17 first-surface queries remain available; frozen legacy
        nearest-triangle membership is explicitly marked as not prepared.
        """
        with self._lock:
            if "observation_arrays" not in self._cache:
                self._cache["observation_arrays"] = MappingProxyType(
                    _load_prepared_sources(self, prepare_legacy_contacts=False))
            return self._cache["observation_arrays"]


def _load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    import trimesh
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [value for value in geom.geometry.values() if isinstance(value, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"mesh contains no triangles: {path}")
        geom = trimesh.util.concatenate(meshes)
    vertices, faces = np.asarray(geom.vertices, np.float32), np.asarray(geom.faces, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid triangle mesh: {path}")
    return vertices, faces, bool(geom.is_watertight)


def _object_asset(bundle: EpisodeBundle, entity: str) -> Mapping[str, Any]:
    assets = bundle.manifest["sources"].get("object_assets")
    if isinstance(assets, Mapping) and isinstance(assets.get(entity), Mapping):
        return assets[entity]
    return MappingProxyType({"pose_mesh": bundle.manifest["sources"]["object_mesh"]})


def _top_level_collision_artifact(bundle: EpisodeBundle, entity: str) -> Mapping[str, Any] | None:
    records = bundle.manifest.get("artifacts")
    if not isinstance(records, Mapping):
        return None
    named = {"collision_eligible_mesh", "collision_mesh", f"{entity}_collision_eligible_mesh"}
    matches = []
    for name, record in records.items():
        if not isinstance(record, Mapping):
            continue
        if name in named or record.get("role") in {"collision_eligible_mesh", "collision_mesh"}:
            matches.append(record)
    if len(matches) > 1:
        raise ValueError(f"ambiguous top-level collision artifacts for {entity!r}")
    return matches[0] if matches else None


def _source_artifact_path(bundle: EpisodeBundle, value: Mapping[str, Any]) -> Path:
    return _absolute(bundle.manifest_path.parent, value["path"])


def _factor_reports(bundle: EpisodeBundle) -> list[dict[str, Any]]:
    sources = bundle.manifest["sources"]
    values: list[Mapping[str, Any]] = []
    for name in ("contact_factors", "factor_reports"):
        raw = sources.get(name)
        if isinstance(raw, Mapping) and "path" in raw:
            values.append(raw)
        elif isinstance(raw, (list, tuple)):
            values.extend(value for value in raw if isinstance(value, Mapping))
    reports = []
    seen: set[Path] = set()
    for value in values:
        path = _source_artifact_path(bundle, value)
        if path in seen:
            continue
        seen.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"factor report is not an object: {path}")
        reports.append(payload)
    return reports


def _camera_mano_contract(frame: Mapping[str, Any], side: str, intrinsics: np.ndarray, *, required: bool) -> tuple[bool, str]:
    rows = [hand for hand in frame.get("hands", []) if hand.get("hand_side") == side]
    if len(rows) != 1:
        return (False, "missing_annotation_hand_row")
    metric = rows[0].get("metric_mano_state") if isinstance(rows[0].get("metric_mano_state"), Mapping) else {}
    alignment = metric.get("camera_contract_alignment") if isinstance(metric.get("camera_contract_alignment"), Mapping) else {}
    if not alignment:
        return (not required, "missing_camera_mano_contract")
    if alignment.get('backend') == 'wilor':
        from utils.hand_contract import independent_alignment_valid
        try:
            independent_alignment_valid(alignment, intrinsics)
        except (ValueError, OSError, KeyError) as exc:
            return (False, 'independent_wilor_contract_invalid:' + str(exc))
        source = np.asarray(metric.get('current_v18_camera_intrinsics_fx_fy_cx_cy', []), float)
        if source.shape != (4,) or not np.allclose(source, intrinsics, atol=.01, rtol=0):
            return (False, 'annotation_mano_intrinsics_differ_from_camera')
        return (True, 'validated_independent_wilor_camera_mano_contract')
    if alignment.get("active_contract_reinference_required") is not False:
        return (False, "camera_mano_reinference_required")
    if alignment.get("source_hawor_state_intrinsics_match_active_contract") is not True:
        return (False, "source_hawor_intrinsics_not_bound_to_active_camera")
    if alignment.get("source_hawor_full_K_bound_to_inference") is not True:
        return (False, "source_hawor_full_K_bound_to_inference_not_proven")
    plane = alignment.get("hawor_camera_image_plane_contract")
    if (not isinstance(plane, Mapping) or plane.get("validated") is not True
            or plane.get("status") != "validated_source_K_affine_centered_hawor_inference_plane"):
        return (False, "hawor_camera_image_plane_contract_not_validated")
    source = np.asarray(metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy", []), float)
    if source.shape != (4,) or not np.allclose(source, intrinsics, atol=.01, rtol=0):
        return (False, "annotation_mano_intrinsics_differ_from_camera")
    return (True, "validated_source_camera_mano_contract")


def _row_target(row: Mapping[str, Any], aliases: Mapping[str, str]) -> str | None:
    raw = row.get("target_entity_id", row.get("object_id", row.get("track_id")))
    if raw is None:
        return None
    text = str(raw)
    return aliases.get(text, aliases.get(_entity_alias(text)))


def _frozen_row(value: Mapping[str, Any], *, report_index: int, row_kind: str) -> Mapping[str, Any]:
    item = json.loads(json.dumps(dict(value), allow_nan=False))
    item["_report_index"] = report_index
    item["_row_kind"] = row_kind
    return freeze(item)


def _first_vector(row: Mapping[str, Any], names: Sequence[str], shape: tuple[int, ...], default: np.ndarray) -> np.ndarray:
    for name in names:
        if name in row:
            value = np.asarray(row[name], dtype=np.float32)
            if value.shape == shape and np.isfinite(value).all():
                return value
    return default


def _load_prepared_sources(bundle: EpisodeBundle, *, prepare_legacy_contacts: bool = True) -> dict[str, Any]:
    manifest = bundle.manifest
    start, end = map(int, manifest["frame_range"])
    annotations = json.loads(bundle.source_path("annotations").read_text(encoding="utf-8"))
    frame_ids, times, frames = join_annotation_rows(manifest, annotations.get("frames", []))
    camera_T = np.asarray([row["camera"].get("T_world_camera_metric", row["camera"].get("T_world_camera")) for row in frames], dtype=np.float64)
    intrinsics = np.asarray([row["camera"]["intrinsics_fx_fy_cx_cy"] for row in frames], dtype=np.float64)
    if camera_T.shape != (len(frames), 4, 4) or intrinsics.shape != (len(frames), 4) or not np.isfinite(camera_T).all() or not np.isfinite(intrinsics).all():
        raise ValueError("invalid camera timeline")
    rgb_evidence = None  # Explicit missing capability in v1/prepared bundles without the extension.
    if 'rgb_source_npz' in manifest['sources']:
        from data.rgb import load_rgb_source_evidence
        for name in ('rgb_source_npz', 'rgb_source_report'):
            if file_sha256(bundle.source_path(name)) != manifest['sources'][name]['sha256']:
                raise ValueError(f'RGB source artifact changed: {name}')
        rgb_evidence = load_rgb_source_evidence(bundle.source_path('rgb_source_npz'),
            bundle.source_path('rgb_source_report'), frame_ids=frame_ids, times=times,
            frames=frames, object_ids=list(manifest['entity_slots']['objects']),
            expected_inputs={key: bundle.source_path(name) for key, name in (
                ('annotations', 'annotations'), ('depth_npz', 'conditioned_depth'),
                ('raw_frame_manifest', 'raw_frames'), ('p09_adapter_report', 'surface_report'))})
    motion_preprocessing = None
    anchor = annotations.get('v19_visible_geometry_adapter', {}).get('anchor_frame_idx')
    if rgb_evidence is not None and anchor is not None:
        from data.motion import prepare_motion
        motion_preprocessing = prepare_motion(rgb_evidence, frames, rgb_evidence.object_id, anchor)
    interaction_payload = None
    if "interaction_judgments" in manifest["sources"]:
        interaction_path = bundle.source_path("interaction_judgments")
        if file_sha256(interaction_path) != manifest["sources"]["interaction_judgments"]["sha256"]:
            raise ValueError("interaction source identity changed before loading")
        interaction_payload = json.loads(interaction_path.read_text())
        if 'typed_interaction' in manifest.get('capabilities', ()):
            from observations.interaction.adapter import SCHEMA as INTERACTION_SCHEMA
            if interaction_payload.get('schema') != INTERACTION_SCHEMA:
                raise ValueError('typed_interaction capability requires versioned source declarations')
    interaction = load_interaction_evidence(interaction_payload, frame_ids, list(manifest["entity_slots"]["objects"]))
    from data.interaction_features import compile_interaction_timeline
    interaction_timelines = MappingProxyType({entity: compile_interaction_timeline(interaction, frame_ids, entity)
        for entity in manifest["entity_slots"]["objects"]})
    size = np.asarray([frames[0].get("source_width"), frames[0].get("source_height")], dtype=np.int64)

    declared_mano = bundle.source_path("mano_archive")
    embedded: set[Path] = set()
    for frame in frames:
        for hand in frame.get("hands", []):
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), Mapping) else {}
            params = metric.get("mano_params") if isinstance(metric.get("mano_params"), Mapping) else {}
            reference = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), Mapping) else {}
            raw = params.get("source_hawor_npz") or reference.get("source_hawor_npz")
            if isinstance(raw, str):
                path = Path(raw).expanduser().resolve()
                if path.is_dir():
                    candidates = sorted(path.glob("*with_track_support*.npz")) + sorted(path.glob("*.npz"))
                    if candidates:
                        path = candidates[0].resolve()
                embedded.add(path)
    if embedded and embedded != {declared_mano}:
        raise ValueError(f"manifest MANO archive does not match annotation-bound source pointer: {embedded}")

    with np.load(declared_mano, allow_pickle=False) as archive:
        source_ids = np.asarray(archive["frame_idx"], dtype=np.int64)
        lookup = {int(value): index for index, value in enumerate(source_ids.tolist())}
        if any(int(value) not in lookup for value in frame_ids):
            raise ValueError("MANO archive does not cover declared frame_range")
        source_rows = np.asarray([lookup[int(value)] for value in frame_ids], dtype=np.int64)
        hands: dict[str, dict[str, Any]] = {}
        for side in HAND_SLOTS:
            required = [f"{side}_{suffix}" for suffix in ("root_orient_axis_angle", "hand_pose_axis_angle", "betas", "trans_world_m", "vertices_world_m", "joints_world_m", "valid", "faces")]
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise ValueError(f"MANO archive lacks {missing}")
            valid = np.asarray(archive[f"{side}_valid"][source_rows], bool)
            camera_status = []
            for pos, frame in enumerate(frames):
                ok, reason = _camera_mano_contract(frame, side, intrinsics[pos], required=bool(valid[pos]))
                if valid[pos] and not ok:
                    raise ValueError(f"frame={frame_ids[pos]} {side}: {reason}")
                camera_status.append(reason)
            hands[side] = {"root_orient": _frozen_array(archive[f"{side}_root_orient_axis_angle"][source_rows], np.float32),
                "articulation": _frozen_array(archive[f"{side}_hand_pose_axis_angle"][source_rows], np.float32),
                "betas": _frozen_array(archive[f"{side}_betas"][source_rows], np.float32),
                "translation": _frozen_array(archive[f"{side}_trans_world_m"][source_rows], np.float32),
                "vertices": _frozen_array(archive[f"{side}_vertices_world_m"][source_rows], np.float32),
                "joints": _frozen_array(archive[f"{side}_joints_world_m"][source_rows], np.float32),
                "valid": _frozen_array(valid), "faces": _frozen_array(archive[f"{side}_faces"], np.int64),
                "camera_mano_contract_state": tuple(camera_status)}

    object_ids = list(manifest["entity_slots"]["objects"])
    aliases = {alias: entity for entity in object_ids for alias in (entity, _entity_alias(entity))}
    objects: dict[str, dict[str, Any]] = {}
    for entity in object_ids:
        asset = _object_asset(bundle, entity)
        pose_path = _source_artifact_path(bundle, asset["pose_mesh"])
        pose_vertices, pose_faces, pose_watertight = _load_mesh_arrays(pose_path)
        collision_value = asset.get("collision_mesh") or _top_level_collision_artifact(bundle, entity)
        completion_payload: Mapping[str, Any] = {}
        try:
            completion_payload = json.loads(bundle.source_path("completion_report").read_text(encoding="utf-8"))
        except KeyError:
            pass
        if collision_value is None and "completion_report" in bundle.manifest["sources"]:
            raise ValueError(f"completion-backed object {entity!r} lacks a manifest-bound collision mesh")
        if collision_value is not None:
            collision_path = _source_artifact_path(bundle, collision_value)
            collision_vertices, collision_faces, collision_watertight = _load_mesh_arrays(collision_path)
            collision_sha = file_sha256(collision_path)
            collision_role = "declared_collision_eligible_mesh"
        else:
            collision_vertices, collision_faces, collision_watertight = pose_vertices, pose_faces, pose_watertight
            collision_sha, collision_role = file_sha256(pose_path), "pose_mesh_fallback_no_collision_mesh_declared"
        rotations=[]; translations=[]; present=[]; pose_valid=[]; metric_eligible=[]; raw_points=[]; admitted=[]; states=[]
        for frame in frames:
            matches = [value for value in frame.get("objects", []) if aliases.get(str(value.get("object_id", value.get("track_id", ""))), aliases.get(_entity_alias(str(value.get("object_id", value.get("track_id", ""))))) ) == entity]
            if len(matches) > 1:
                raise ValueError(f"duplicate object entity {entity} in one frame")
            if not matches:
                rotations.append(np.eye(3)); translations.append(np.zeros(3)); present.append(False); pose_valid.append(False); metric_eligible.append(False); raw_points.append(np.zeros((0,3),np.float32)); admitted.append(np.zeros((0,3),np.float32)); states.append("inactive_absent_entity"); continue
            obj=matches[0]; pose=obj.get("reconstructed_geometry_pose") or {}; geom=obj.get("visible_geometry_candidate") or {}
            R=np.asarray(pose.get("rotation_world_from_canonical_matrix",np.eye(3)),float); t=np.asarray(pose.get("translation_world_m",[np.nan]*3),float)
            valid=R.shape==(3,3) and t.shape==(3,) and np.isfinite(R).all() and np.isfinite(t).all()
            points=np.asarray(geom.get("world_vertices_sample_m",[]),np.float32).reshape(-1,3)
            eligible=bool(obj.get("rigid_pose_observation_eligible",geom.get("rigid_pose_observation_eligible",True))) and valid and len(points)>=3
            rotations.append(R if valid else np.eye(3)); translations.append(t if valid else np.zeros(3)); present.append(True); pose_valid.append(valid); metric_eligible.append(eligible); raw_points.append(points); admitted.append(points if eligible else np.zeros((0,3),np.float32)); states.append("active_metric_P09" if eligible else "inactive_P09_ineligible_or_missing_pose")
        readiness=completion_payload.get("geometry_readiness") if isinstance(completion_payload.get("geometry_readiness"),Mapping) else {}
        authority=np.zeros(len(collision_faces),bool)
        authority_state="inactive_missing_signed_face_authority"
        raw_authority=asset.get("signed_face_authority")
        if raw_authority is not None:
            authority_path=_source_artifact_path(bundle,raw_authority)
            with np.load(authority_path,allow_pickle=False) as z:
                ids=np.asarray(z["face_id"],np.int64); eligible=np.asarray(z["signed_distance_eligible"],bool)
                mesh_hash=str(np.asarray(z.get("source_mesh_sha256",[""])).reshape(-1)[0]); face_count=int(np.asarray(z.get("source_mesh_face_count",[-1])).reshape(-1)[0])
            if not np.array_equal(ids,np.arange(len(collision_faces))) or eligible.shape!=(len(collision_faces),) or mesh_hash!=collision_sha or face_count!=len(collision_faces):
                raise ValueError(f"signed authority does not bind exact collision mesh for {entity}")
            authority=eligible; authority_state="active_mesh_bound_local_face_authority" if authority.any() else "inactive_zero_signed_faces"
        objects[entity]={"entity_id":entity,"present":_frozen_array(present),"pose_valid":_frozen_array(pose_valid),"valid":_frozen_array(np.asarray(present)&np.asarray(pose_valid)),"metric_observation_eligible":_frozen_array(metric_eligible),"rotation":_frozen_array(rotations),"translation":_frozen_array(translations),"pose_vertices_canonical":_frozen_array(pose_vertices),"pose_faces":_frozen_array(pose_faces),"pose_mesh_sha256":file_sha256(pose_path),"pose_mesh_face_count":int(len(pose_faces)),"vertices_canonical":_frozen_array(pose_vertices),"faces":_frozen_array(pose_faces),"collision_vertices_canonical":_frozen_array(collision_vertices),"collision_faces":_frozen_array(collision_faces),"collision_mesh_sha256":collision_sha,"collision_mesh_face_count":int(len(collision_faces)),"collision_mesh_role":collision_role,"watertight":bool(collision_watertight),"signed_geometry_ready":readiness.get("signed_geometry_ready") is True,"signed_authority":_frozen_array(authority),"signed_authority_state":authority_state,"raw_observed_points_world":tuple(_frozen_array(value) for value in raw_points),"observed_points_world":tuple(_frozen_array(value) for value in admitted),"observed_point_state":tuple(states),"pose_provenance":"source_observation_initializer_not_solved_trajectory","geometry_provenance":"completed_reconstructed_pose_mesh"}

    visual_geometry = None
    if 'visual_geometry_report' in manifest['sources']:
        if len(object_ids) != 1:
            raise ValueError('current complete visual observation supports one selected object')
        from data.geometry import load_visual_geometry
        for name in ('visual_geometry_report','visual_mesh_camera','visual_native_mesh','visual_native_qc','appearance_masks'):
            if file_sha256(bundle.source_path(name)) != manifest['sources'][name]['sha256']:
                raise ValueError('visual source artifact changed: '+name)
        visual_geometry = load_visual_geometry(bundle.source_path('visual_geometry_report'),
            frame_ids=frame_ids,frames=frames,object_id=object_ids[0],expected_inputs={
                'annotations':bundle.source_path('annotations'),
                'appearance':bundle.source_path('appearance_masks'),
                'decision':bundle.source_path('appearance_anchor_decision'),
                'native_qc':bundle.source_path('visual_native_qc'),
                'native_mesh':bundle.source_path('visual_native_mesh'),
                'output_mesh':bundle.source_path('visual_mesh_camera')})

    source_visual = appearance_evidence = None
    if manifest.get('schema') == 'ego.annotation.observation_bundle.v2':
        from data.visual_preprocessing import source_hand_views, load_appearance_observations
        source_visual = source_hand_views(hands, frame_ids, camera_T, intrinsics, size)
        if 'appearance_masks' in manifest['sources']:
            appearance_evidence = load_appearance_observations(bundle.source_path('appearance_masks'),
                annotations_path=bundle.source_path('annotations'), frames=frames, object_id=object_ids[0])

    reports=_factor_reports(bundle)
    typed: dict[tuple[int,str|None,str|None,str],list[Mapping[str,Any]]]={}
    by_family: dict[str,list[Mapping[str,Any]]]={}
    for report_index, report in enumerate(reports):
        for kind in ("factor_rows","ownership_rows"):
            for raw in report.get(kind,[]):
                if not isinstance(raw,Mapping): continue
                frame=int(raw.get("frame_idx",-1)); side=str(raw["hand_side"]) if raw.get("hand_side") in HAND_SLOTS else None; target=_row_target(raw,aliases); family=str(raw.get("factor_family","unknown"))
                if frame not in frame_ids: continue
                frozen=_frozen_row(raw,report_index=report_index,row_kind=kind)
                typed.setdefault((frame,side,target,family),[]).append(frozen); by_family.setdefault(family,[]).append(frozen)
    typed_frozen=MappingProxyType({key:tuple(value) for key,value in typed.items()})
    by_family_frozen=MappingProxyType({key:tuple(value) for key,value in by_family.items()})

    # Row/joint/finger/ray evidence stays in hand dictionaries for reward profiles.
    for side, hand in hands.items():
        n=len(frame_ids); joint=np.zeros((n,21),np.float32); pose=np.zeros((n,15),np.float32); row_vis=np.zeros(n,np.float32); depth_res=np.full((n,21),np.nan,np.float32); mult=np.ones(n,np.float32); ray=np.zeros((n,3),np.float32); ray_source=np.full(n,np.nan,np.float32); ray_weight=np.zeros(n,np.float32); states=[]
        for i,frame in enumerate(frame_ids):
            candidates=[row for (f,s,_t,fam),rows in typed_frozen.items() if f==int(frame) and s==side for row in rows if fam in {"hand_observation_visibility","hand_depth_shift_prior"}]
            default=float(hand["valid"][i]); row_vis[i]=default; joint[i]=default; pose[i]=default; states.append("source_validity_only_no_joint_factor_row" if default else "inactive_hand_absent")
            for row in candidates:
                family=row.get("factor_family")
                if family=="hand_observation_visibility":
                    multiplier=float(row.get("joint_observation_weight_multiplier",row.get("weight_multiplier",1.0))); mult[i]=np.clip(multiplier,0,1); row_vis[i]=mult[i]; joint[i]=_first_vector(row,("joint_visibility_weights","joint_visibility"),(21,),joint[i]); pose[i]=_first_vector(row,("pose_visibility_weights","pose_visibility"),(15,),pose[i]); depth_res[i]=_first_vector(row,("joint_depth_residual_m",),(21,),depth_res[i]); states[-1]=str(row.get("state","active_hand_observation_visibility"))
                elif family=="hand_depth_shift_prior":
                    ray[i]=_first_vector(row,("hand_ray_shift_prior_world_m","translation_world_m","target_world_m"),(3,),ray[i]); ray_source[i]=float(row.get("hand_ray_shift_prior_source_m",np.nan)); ray_weight[i]=float(row.get("hand_ray_shift_prior_weight",row.get("weight",0.0)))
        hand.update({"row_visibility":_frozen_array(row_vis),"joint_visibility":_frozen_array(joint),"pose_visibility":_frozen_array(pose),"joint_depth_residual_m":_frozen_array(depth_res),"hand_observation_visibility_weight_multiplier":_frozen_array(mult),"hand_ray_shift_prior_world_m":_frozen_array(ray),"hand_ray_shift_prior_source_m":_frozen_array(ray_source),"hand_ray_shift_prior_weight":_frozen_array(ray_weight),"visibility_state":tuple(states)})

    # Independent P09 first-surface depth evidence is frozen and bound to factor/NPZ hashes.
    contact: dict[tuple[int,str,str],dict[str,Any]]={}; depth_by_hand={side:[None]*len(frame_ids) for side in HAND_SLOTS}; frame_position={int(value):i for i,value in enumerate(frame_ids)}
    for (frame,side,target,family),rows in typed_frozen.items():
        if side not in HAND_SLOTS: continue
        if family=="contact_patch" and target is not None:
            row=rows[-1]; contact[(frame,side,target)]={"target_entity_id":target,"state":str(row.get("state","inactive_missing_support")),"prior":float(row.get("contact_patch_prior_probability",0.0)),"weight":float(row.get("weight",0.0)),"band_m":float(row.get("contact_patch_band_m",0.0)),"margin_m":float(row.get("contact_patch_target_margin_m",0.0)),"support_uncertainty_m":float(row.get("contact_patch_support_uncertainty_m",0.0)),"anchor_allowed":bool(row.get("contact_anchor_residual_allowed",False)),"blockers":tuple(map(str,row.get("contact_anchor_blockers",[]))),"factor_row":row}
        if family=="visible_ownership":
            row=rows[-1]; pos=frame_position[frame]; npz_raw=row.get("depth_order_first_surface_npz_path"); expected=str(row.get("depth_order_first_surface_npz_sha256") or "")
            if not isinstance(npz_raw,str) or not expected:
                depth_by_hand[side][pos]={"state":"inactive_missing_independent_first_surface"}; continue
            path=Path(npz_raw).resolve()
            if file_sha256(path)!=expected: raise ValueError(f"depth-order identity mismatch: {path}")
            with np.load(path,allow_pickle=False) as z:
                depth=np.asarray(z["depth_mask_plane_m"],np.float32); support=np.asarray(z["support_mask"],bool); affine=np.asarray(z["A_mask_from_source_coordinate_model"],float)
            plane=row.get("image_plane_transform") if isinstance(row.get("image_plane_transform"),Mapping) else {}
            declared=np.asarray(plane.get("A_mask_from_source_coordinate_model",affine),float)
            if declared.shape!=(3,3) or not np.allclose(declared,affine,atol=1e-12,rtol=0): raise ValueError("factor-row depth affine differs from bound NPZ")
            vertices=hands[side]["vertices"][pos]; cam=(vertices-camera_T[pos,:3,3])@camera_T[pos,:3,:3]; fx,fy,cx,cy=intrinsics[pos]; uv=np.column_stack([fx*cam[:,0]/np.maximum(cam[:,2],1e-6)+cx,fy*cam[:,1]/np.maximum(cam[:,2],1e-6)+cy]); q=np.column_stack([uv,np.ones(len(uv))])@affine.T; q=q[:,:2]/np.maximum(q[:,2:3],1e-12); u=np.rint(q[:,0]).astype(int);v=np.rint(q[:,1]).astype(int); valid=(cam[:,2]>1e-5)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0]); ids=np.flatnonzero(valid); ids=ids[support[v[ids],u[ids]]] if len(ids) else ids; zvals=depth[v[ids],u[ids]].astype(float) if len(ids) else np.zeros(0); finite=np.isfinite(zvals)&(zvals>1e-5); ids,zvals=ids[finite],zvals[finite]
            depth_by_hand[side][pos]={"state":"active_independent_P09_first_surface_depth_order_query" if len(ids) else "inactive_no_supported_projected_vertices","vertex_ids":_frozen_array(ids,np.int64),"surface_depth_m":_frozen_array(zvals,np.float64),"initial_delta_m":_frozen_array(cam[ids,2]-zvals,np.float64),"translation_support_count":int(len(ids)),"npz_path":str(path),"npz_sha256":expected,"factor_row":row}

    if not prepare_legacy_contacts:
        for entry in contact.values():
            entry['membership_state'] = 'not_prepared_observation_inspection'
    else:
        # Freeze default contact membership against source MANO and source target mesh once.
        try:
            from utils.candidate_geometry import make_scene, contact_patch_targets_from_vertices
            for (frame,side,target),entry in contact.items():
                pos=frame_position[frame]; obj=objects[target]
                if entry["state"]!="active_contact_patch" or not hands[side]["valid"][pos] or not obj["valid"][pos] or not len(obj["collision_faces"]):
                    entry.update({"vertex_ids":_frozen_array([],np.int64),"targets_object":_frozen_array(np.zeros((0,3))),"normals_object":_frozen_array(np.zeros((0,3))),"distances_m":_frozen_array([]),"initial_contact_geometry_target_probability":float(entry["prior"]),"membership_state":"inactive_contact_or_target_unavailable"}); continue
                scene=make_scene(obj["collision_vertices_canonical"],obj["collision_faces"])
                triangles=np.asarray(obj["collision_vertices_canonical"])[np.asarray(obj["collision_faces"],np.int64)]
                normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
                ids,targets,normals_world,dist=contact_patch_targets_from_vertices(
                    hands[side]["vertices"][pos],scene,obj["rotation"][pos],obj["translation"][pos],
                    np.ones(len(obj["collision_faces"]),bool),normals,
                    max_vertices=int(entry["factor_row"].get("max_vertices",160)),band_m=float(entry["band_m"]))
                R=obj["rotation"][pos];t=obj["translation"][pos]
                deadband=max(float(entry["margin_m"]+entry["support_uncertainty_m"]),1e-6)
                target_probability=float(np.exp(-.5*(float(np.median(dist))/deadband)**2)) if len(dist) else float(entry["prior"])
                entry.update({"vertex_ids":_frozen_array(ids,np.int64),"targets_object":_frozen_array((targets-t)@R if len(targets) else np.zeros((0,3))),"normals_object":_frozen_array(normals_world@R if len(normals_world) else np.zeros((0,3))),"distances_m":_frozen_array(dist),"initial_contact_geometry_target_probability":target_probability,"membership_state":"active_frozen_source_patch" if len(ids) else "inactive_no_source_patch_vertices"})
        except ImportError as exc:
            raise RuntimeError("source contact patch freezing requires declared geometry runtime") from exc

    # Copy frozen first-surface rows into every target-pair contact record; this
    # preserves the legacy single-target compatibility view without changing pair identity.
    for (frame, side, _target), entry in contact.items():
        depth = depth_by_hand[side][frame_position[frame]]
        if depth is not None:
            entry.update({"depth_order_state": depth["state"],
                          "depth_order_vertex_ids": depth.get("vertex_ids", _frozen_array([], np.int64)),
                          "depth_order_surface_depth_m": depth.get("surface_depth_m", _frozen_array([], np.float64)),
                          "translation_support_count": depth.get("translation_support_count", 0)})
    # Attach per-hand frozen depth rows and a backward-compatible unambiguous contact alias.
    for side,hand in hands.items():
        rows=depth_by_hand[side]; hand["depth_order"] = tuple(MappingProxyType(row or {"state":"inactive_missing_visible_ownership_row"}) for row in rows)
    frozen_contact={key:MappingProxyType(value) for key,value in contact.items()}
    legacy_contact={}
    for (frame,side,target),value in frozen_contact.items():
        matches=[v for (f,s,_),v in frozen_contact.items() if f==frame and s==side]
        if len(matches)==1: legacy_contact[frame,side]=value

    return {"annotation_frames":tuple(freeze(frame) for frame in frames),"legacy_contact_patch_prepared":prepare_legacy_contacts,"source_visual_preprocessing":source_visual,"appearance_evidence":appearance_evidence,"visual_geometry":visual_geometry,"motion_preprocessing":motion_preprocessing,"rgb_evidence":rgb_evidence,"interaction_timelines":interaction_timelines,"interaction_evidence":interaction,"frame_idx":_frozen_array(frame_ids),"time_s":_frozen_array(times),"T_world_camera":_frozen_array(camera_T),"intrinsics":_frozen_array(intrinsics),"image_size_wh":_frozen_array(size),"hands":MappingProxyType({side:MappingProxyType(value) for side,value in hands.items()}),"objects":MappingProxyType({entity:MappingProxyType(value) for entity,value in objects.items()}),"contact":MappingProxyType(frozen_contact),"contact_legacy":MappingProxyType(legacy_contact),"factor_rows":typed_frozen,"factor_rows_by_family":by_family_frozen,"identity":manifest["identity"],"provenance":manifest.get("provenance",{})}
