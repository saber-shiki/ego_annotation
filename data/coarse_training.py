"""Explicit derived coarse-state training adapter; the raw v2 guards stay intact."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from data.bundle import EpisodeBundle, file_sha256
from data.dataset import source_video_split
from data.immutable import freeze, frozen_array

INDEX_SCHEMA = "ego.annotation.coarse_training_index.v1"
ADAPTER = "coarse_scene_v1_source_v2_rgb"
VISIBILITY = ("visible", "partially_visible", "occluded", "out_of_frame", "unresolved")
CONTACT = ("likely_contact", "possible_contact", "no_contact", "unresolved")
EXTRA_FEATURES = 71


def _path(base, value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def _bound(path, expected):
    if not expected or file_sha256(path) != expected:
        raise ValueError(f"coarse training artifact hash mismatch: {path}")


def validate_coarse(payload, path):
    if payload.get("schema") != "ego.annotation.monocular_coarse_scene.v1":
        raise ValueError("explicit coarse_scene.v1 initializer required")
    for key, expected in (("prediction_side_only", True), ("contains_optimized_state", True),
                          ("contains_policy_results", False), ("GT_or_CAD_inputs", False),
                          ("additional_camera_inputs", False), ("source_MANO_unchanged", True),
                          ("signed_geometry_ready", False)):
        if payload.get(key) is not expected:
            raise ValueError(f"unsupported coarse initializer authority: {key}")
    if payload.get("canonical_frame") != "centered_native_appearance_camera_axes":
        raise ValueError("unsupported coarse canonical frame")
    source = _path(path.parent, payload["source_observations"])
    arrays = _path(path.parent, payload["arrays"]["path"])
    _bound(source, payload["source_bundle_sha256"])
    _bound(arrays, payload["arrays"]["sha256"])
    return source, arrays


def validate_archive(archive, source, payload):
    ids = np.asarray(source["frame_idx"])
    n = len(ids)
    for key, reference in (("frame_idx", ids), ("time_s", source["time_s"]),
                           ("T_world_camera", source["T_world_camera"]),
                           ("intrinsics_source", source["intrinsics"])):
        if not np.array_equal(archive[key], reference):
            raise ValueError(f"coarse/source timeline or camera mismatch: {key}")
    if list(payload["frame_ids"]) != ids.tolist() or not np.array_equal(payload["time_s"], source["time_s"]):
        raise ValueError("coarse manifest timeline mismatch")
    sizes = np.asarray(archive["image_size_wh"])
    if sizes.shape != (n, 2) or not np.all(sizes == source["image_size_wh"]):
        raise ValueError("coarse raster mismatch")
    for side in ("left", "right"):
        for field in ("vertices", "joints", "faces", "valid"):
            if not np.array_equal(archive[f"{side}_{field}"], source["hands"][side][field]):
                raise ValueError(f"coarse changed source MANO: {side}/{field}")
    for key, shape in (("object_rotation", (n, 3, 3)), ("object_translation_world", (n, 3)),
                       ("object_valid", (n,)), ("source_pose_available", (n,)),
                       ("source_metric_eligible", (n,)), ("translation_depth_inferred", (n,))):
        a = archive[key]
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"invalid coarse array: {key}")
    rotations = archive["object_rotation"]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6):
        raise ValueError("coarse rotation is not SO(3)")
    v, f = archive["vertices_canonical"], archive["faces"]
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError("complete finite native vertices required")
    if f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in "iu" or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("complete native triangles required")
    visual = source.get("visual_geometry")
    if visual is None:
        raise ValueError("coarse adapter requires the complete native v2 geometry")
    pivot = np.asarray(payload["canonical_native_pivot_camera_m"], dtype=float)
    expected_vertices = (visual.vertices_metric_camera - pivot) * float(payload["residual_uniform_object_scale"])
    if not np.array_equal(f, visual.faces) or not np.allclose(v, expected_vertices, rtol=0, atol=1e-9):
        raise ValueError("coarse mesh changed full native topology or declared uniform scale")
    if len(source["objects"]) != 1 or payload["object_id"] not in source["objects"]:
        raise ValueError("coarse object identity mismatch")
    obj = source["objects"][payload["object_id"]]
    if not np.array_equal(archive["source_pose_available"], obj["pose_valid"]) or not np.array_equal(archive["source_metric_eligible"], obj["metric_observation_eligible"]):
        raise ValueError("source validity was upgraded by preprocessing")
    if not np.array_equal(archive["translation_depth_inferred"], ~archive["source_metric_eligible"].astype(bool)):
        raise ValueError("inferred depth must remain distinct from observations")


def freeze_contacts(source, obj):
    """Unsigned anchors on actual coarse triangles, NOT signed/observed authority."""
    from utils.candidate_geometry import make_scene, contact_patch_targets_from_vertices
    scene = make_scene(obj["vertices_canonical"], obj["faces"])
    triangles = obj["vertices_canonical"][obj["faces"]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    pos = {int(f): i for i, f in enumerate(source["frame_idx"])}
    result = {}
    for key, frozen in source["contact"].items():
        frame, side, target = key
        i = pos[frame]
        row = dict(frozen)
        ids, targets, ns, dist = np.empty(0, int), np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
        if target == obj["entity_id"] and row["state"] == "active_contact_patch" and source["hands"][side]["valid"][i] and obj["valid"][i]:
            ids, targets, ns, dist = contact_patch_targets_from_vertices(
                source["hands"][side]["vertices"][i], scene, obj["rotation"][i], obj["translation"][i],
                np.ones(len(obj["faces"]), bool), normals,
                max_vertices=int(row["factor_row"].get("max_vertices", 160)), band_m=float(row["band_m"]))
        R, t = obj["rotation"][i], obj["translation"][i]
        deadband = max(float(row["margin_m"] + row["support_uncertainty_m"]), 1e-6)
        row.update(vertex_ids=ids, targets_object=(targets - t) @ R, normals_object=ns @ R,
                   distances_m=dist, initial_contact_geometry_target_probability=float(np.exp(-.5 * (np.median(dist) / deadband) ** 2)) if len(dist) else row["prior"],
                   membership_state="active_unsigned_coarse_native_prior_patch" if len(ids) else "inactive_no_coarse_native_patch",
                   geometry_authority="generated_native_unsigned_prior_not_observed_or_signed")
        result[key] = freeze(row)
    return freeze(result)


def source_rgb_rows(source, entity):
    """Reuse qualified independent source-world RGB likelihoods, never pose-fit rows."""
    evidence, motion = source["rgb_evidence"], source["motion_preprocessing"]
    if evidence is None or motion is None:
        raise ValueError("coarse RGB training adapter requires independent v2 RGB evidence")
    weights = motion.qualification["relative_edge_weights"]
    pixels = motion.qualification["reprojection_edge_weights"]
    if len(weights) != len(evidence.edges) or len(pixels) != len(evidence.edges):
        raise ValueError("RGB qualification edge order mismatch")
    rows = {}
    summary = np.zeros((len(source["frame_idx"]), 9), np.float32)
    lookup = {int(f): i for i, f in enumerate(source["frame_idx"])}
    incoming = {}
    for e, weight, pixel_weight in zip(evidence.edges, weights, pixels):
        a, b = e.source_frame_idx, e.target_frame_idx
        if weight > 0:
            row = {"factor_family": "object_rgb_edge", "frame_idx": b, "target_entity_id": entity,
                   "source_frame_idx": a, "target_frame_idx": b, "rotation_only": True,
                   "rotation_source_to_target_world": e.rotation_source_to_target_world,
                   "translation_source_to_target_world_m": e.translation_source_to_target_world_m,
                   "weight": float(weight), "diagnostics": dict(e.diagnostics)}
            rows.setdefault((b, None, entity, "object_rgb_edge"), []).append(freeze(row))
        if pixel_weight > 0:
            row = {"factor_family": "object_rgb_reprojection", "frame_idx": b, "target_entity_id": entity,
                   "source_frame_idx": a, "target_frame_idx": b, "source_world": e.source_world,
                   "target_uv": e.target_uv, "weights": e.weights,
                   "target_camera_rotation": e.target_camera[:3, :3], "target_camera_translation": e.target_camera[:3, 3],
                   "target_intrinsics": e.target_intrinsics, "edge_weight": float(pixel_weight), "rotation_eligible": True}
            rows.setdefault((b, None, entity, "object_rgb_reprojection"), []).append(freeze(row))
        if a < b and weight > 0:
            incoming.setdefault(lookup[b], []).append((e, weight))
    for i, edges in incoming.items():
        w = np.asarray([weight for _, weight in edges]); w /= w.sum()
        summary[i, :3] = sum(weight * Rotation.from_matrix(e.rotation_source_to_target_world).as_rotvec() for (e, _), weight in zip(edges, w))
        summary[i, 3:6] = sum(weight * e.translation_source_to_target_world_m / .01 for (e, _), weight in zip(edges, w))
        summary[i, 6:] = [np.log1p(len(edges)), np.log1p(sum(len(e.weights) for e, _ in edges)), sum(weight * float(e.diagnostics["reprojection_median_px"]) for (e, _), weight in zip(edges, w))]
    return rows, frozen_array(summary)


class CoarseTrainingBundle:
    def __init__(self, manifest_path, expected_hash):
        self.manifest_path = Path(manifest_path).resolve()
        _bound(self.manifest_path, expected_hash)
        self.coarse = json.loads(self.manifest_path.read_text())
        source, self.array_path = validate_coarse(self.coarse, self.manifest_path)
        self.source = EpisodeBundle.open(source)
        if self.source.manifest.get("schema") != "ego.annotation.observation_bundle.v2" or self.source.manifest.get("preprocessing_only") is not True:
            raise ValueError("explicit untouched preprocessing-only v2 source required")
        self.bundle_id = "coarse-" + expected_hash[:24]
        self.source_video_id = self.source.source_video_id
        def absolute_artifacts(value):
            if isinstance(value, Mapping):
                return {key: str(_path(self.source.manifest_path.parent, item)) if key == "path" and isinstance(item, str)
                        else absolute_artifacts(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [absolute_artifacts(item) for item in value]
            return value
        self.manifest = freeze({**dict(self.source.manifest), "sources": absolute_artifacts(self.source.manifest["sources"]),
                                "schema": "ego.annotation.coarse_training_episode.v1",
                                "bundle_id": self.bundle_id, "clip_id": self.source.manifest["clip_id"] + "/coarse-training",
                                "preprocessing_only": False, "contains_optimized_state": True,
                                "training_adapter": ADAPTER, "coarse_manifest_sha256": expected_hash})
        self._cache = None

    def load(self):
        if self._cache is not None:
            return self._cache
        _bound(self.array_path, self.coarse["arrays"]["sha256"])
        source = self.source.load_observations()
        with np.load(self.array_path, allow_pickle=False) as z:
            archive = {k: np.array(z[k]) for k in z.files}
        validate_archive(archive, source, self.coarse)
        entity = self.coarse["object_id"]
        obj = dict(source["objects"][entity])
        obj.update(rotation=archive["object_rotation"], translation=archive["object_translation_world"],
                   present=archive["object_valid"], valid=archive["object_valid"], pose_valid=archive["object_valid"],
                   pose_vertices_canonical=archive["vertices_canonical"], pose_faces=archive["faces"],
                   vertices_canonical=archive["vertices_canonical"], faces=archive["faces"],
                   pose_mesh_sha256=self.coarse["arrays"]["sha256"], pose_mesh_face_count=len(archive["faces"]),
                   collision_vertices_canonical=archive["vertices_canonical"], collision_faces=archive["faces"],
                   collision_mesh_sha256=self.coarse["arrays"]["sha256"], collision_mesh_face_count=len(archive["faces"]),
                   collision_mesh_role="generated_native_unsigned_contact_prior_only", watertight=False,
                   signed_geometry_ready=False, signed_authority=np.zeros(len(archive["faces"]), bool),
                   signed_authority_state="inactive_coarse_generated_mesh_has_no_signed_authority",
                   pose_provenance="explicit_R6_derived_initializer_not_raw_observation",
                   geometry_provenance="complete_native_mesh_fixed_preprocessed_uniform_scale")
        contact = freeze_contacts(source, obj)
        legacy = {(f, side): row for (f, side, target), row in contact.items()}
        # Canonical/source-object rows cannot be silently moved to the coarse frame.
        factors = {k: list(rows) for k, rows in source["factor_rows"].items()
                   if k[3] not in {"object_image", "object_first_hit_silhouette", "first_hit_depth", "silhouette_endpoint",
                                   "object_rgb_edge", "rgb_rotation", "object_rgb_reprojection", "rgb_reprojection"}}
        rgb_rows, rgb_summary = source_rgb_rows(source, entity)
        for key, rows in rgb_rows.items():
            factors.setdefault(key, []).extend(rows)
        adapter = {"schema": "ego.annotation.coarse_state_adapter.v1", "name": ADAPTER,
                   "coarse_manifest": str(self.manifest_path), "coarse_manifest_sha256": file_sha256(self.manifest_path),
                   "coarse_arrays_sha256": self.coarse["arrays"]["sha256"], "source_v2": str(self.source.manifest_path),
                   "source_v2_sha256": self.coarse["source_bundle_sha256"], "contains_optimized_initial_state": True,
                   "physical_contact_authority": False, "signed_geometry_ready": False,
                   "RGB_translation_edge_enabled": False, "source_RGB_reprojection_enabled": True}
        extra = {"source_pose_available": archive["source_pose_available"],
                 "source_metric_eligible": archive["source_metric_eligible"],
                 "translation_depth_inferred": archive["translation_depth_inferred"],
                 "interaction": source["interaction_timelines"][entity], "rgb_summary": rgb_summary}
        self._cache = freeze({**dict(source), "objects": {entity: obj}, "contact": contact,
                              "contact_legacy": legacy, "factor_rows": factors,
                              "identity": {**dict(source["identity"]), "derived_initializer": adapter},
                              "provenance": {**dict(source["provenance"]), "preprocessed_state_adapter": adapter, "coarse_features": extra}})
        return self._cache


class CoarseTrainingDataset(Sequence):
    compatibility_contract = {"adapter": ADAPTER, "bundle_schemas": ["ego.annotation.coarse_training_episode.v1"],
                              "source_schema": "ego.annotation.observation_bundle.v2", "extra_observation_features": EXTRA_FEATURES}

    def __init__(self, index, *, split="all", seed=0, test_fraction=.2):
        if split not in {"all", "train", "test"}:
            raise ValueError("invalid dataset split")
        index = Path(index)
        payload = json.loads(index.read_text())
        if payload.get("schema") != INDEX_SCHEMA or payload.get("adapter") != ADAPTER or not payload.get("episodes"):
            raise ValueError("explicit coarse training index required; raw v2 is not a training index")
        self._records = tuple({**r, "manifest": str(_path(index.parent, r["manifest"]))} for r in payload["episodes"]
                              if split == "all" or source_video_split(r["source_video_id"], seed=seed, test_fraction=test_fraction) == split)
        self._cache = {}

    def __len__(self):
        return len(self._records)

    def __getitem__(self, i):
        record = self._records[i]
        if i not in self._cache:
            b = CoarseTrainingBundle(record["manifest"], record["manifest_sha256"])
            if b.bundle_id != record["bundle_id"] or b.source_video_id != record["source_video_id"]:
                raise ValueError("coarse index identity mismatch")
            self._cache[i] = b
        return self._cache[i]

    @property
    def records(self):
        return self._records

    @property
    def source_video_ids(self):
        return frozenset(r["source_video_id"] for r in self._records)


def build_index(paths):
    rows = []
    for path in paths:
        p = Path(path).resolve(); digest = file_sha256(p)
        coarse = json.loads(p.read_text()); source, _ = validate_coarse(coarse, p)
        b = EpisodeBundle.open(source)
        if b.manifest.get("schema") != "ego.annotation.observation_bundle.v2":
            raise ValueError("source must be v2")
        rows.append({"manifest": str(p), "manifest_sha256": digest, "bundle_id": "coarse-" + digest[:24],
                     "source_video_id": b.source_video_id, "frame_count": len(coarse["frame_ids"])})
    return {"schema": INDEX_SCHEMA, "adapter": ADAPTER, "episodes": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_index(args.manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(value, f, indent=2); f.write("\n")
