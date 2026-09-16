from __future__ import annotations

"""Domain assembly, semantic action filtering, and physical candidate decoding."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import inspect
import math

import numpy as np

from actions.scaling import ActionScales, GateTrace, filter_scaled_action
from actions.space import ActionLayout
from data.bundle import EpisodeBundle
from envs.state import AnnotationState, EpisodeStatic


def _cfg(cfg: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _load_mano_models(bundle: EpisodeBundle, cfg: Mapping[str, Any] | Any, device: str) -> Mapping[str, Any]:
    supplied = _cfg(cfg, "mano_models")
    if supplied is not None:
        if set(supplied) != {"left", "right"}:
            raise ValueError("mano_models must contain left and right")
        return MappingProxyType(dict(supplied))
    sources = bundle.manifest["sources"]
    assets = sources.get("mano_assets")
    if not isinstance(assets, Mapping):
        raise RuntimeError("present hand slots require real MANO models or a manifest mano_assets binding")
    asset_root = Path(assets["path"])
    default_wilor_root = assets.get("wilor_root") or asset_root.parent
    wilor_root = Path(_cfg(cfg, "wilor_root", default_wilor_root))
    # Baseline import; migration rewrite is listed in the integration report.
    import inspect
    for name, value in (("bool", np.bool_), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("unicode", str), ("str", str)):
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    from utils.mano import load_wilor_mano_class
    mano_cls = load_wilor_mano_class(wilor_root)
    models = {}
    for side in ("left", "right"):
        path = asset_root / f"MANO_{side.upper()}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"missing licensed MANO asset: {path}")
        model = mano_cls(model_path=str(path), is_rhand=(side == "right"), use_pca=False,
                         flat_hand_mean=False, batch_size=1).to(device)
        if side == "left" and bool(_cfg(cfg, "hawor_left_shapedirs_x_fix", True)):
            import torch
            with torch.no_grad():
                model.shapedirs[:, 0, :] *= -1
        model.eval()
        models[side] = model
    return MappingProxyType(models)


def build_episode(bundle: EpisodeBundle, cfg: Mapping[str, Any] | Any, device: str = "cpu") -> EpisodeStatic:
    arrays = bundle.load()
    max_objects = int(_cfg(cfg, "max_objects", 1))
    if max_objects < len(arrays["objects"]):
        raise ValueError("max_objects is smaller than declared object entities")
    contact_count = int(_cfg(cfg, "contact_variables_per_hand", 1))
    object_slots = tuple(f"object:{i}" for i in range(max_objects))
    layout = ActionLayout(object_slots=object_slots, contact_variables_per_hand=contact_count)
    source_objects = list(arrays["objects"].items())
    objects: dict[str, Mapping[str, Any]] = {}
    count = len(arrays["frame_idx"])
    for slot_index, slot_name in enumerate(object_slots):
        if slot_index < len(source_objects):
            entity_id, value = source_objects[slot_index]
            objects[slot_name] = MappingProxyType({**dict(value), "entity_id": entity_id})
        else:
            objects[slot_name] = MappingProxyType({
                "entity_id": None, "rotation": np.tile(np.eye(3), (count, 1, 1)),
                "translation": np.zeros((count, 3)), "valid": np.zeros(count, bool),
                "vertices_canonical": np.zeros((0, 3), np.float32), "faces": np.zeros((0, 3), np.int64),
                "watertight": False, "signed_geometry_ready": False,
                "signed_authority": np.zeros((0,), dtype=bool),
                "observed_points_world": tuple(np.zeros((0, 3)) for _ in range(count)),
                "pose_provenance": "absent_entity_slot", "geometry_provenance": "absent_entity_slot",
            })
    models = _load_mano_models(bundle, cfg, device)
    episode = EpisodeStatic(
        bundle_id=bundle.bundle_id, clip_id=str(bundle.manifest["clip_id"]),
        source_video_id=bundle.source_video_id, frame_idx=np.asarray(arrays["frame_idx"]),
        time_s=np.asarray(arrays["time_s"]), T_world_camera=np.asarray(arrays["T_world_camera"]),
        intrinsics=np.asarray(arrays["intrinsics"]), image_size_wh=np.asarray(arrays["image_size_wh"]),
        hands=arrays["hands"], objects=MappingProxyType(objects), contact=arrays["contact"],
        layout=layout, mano_models=models,
        provenance=MappingProxyType({**dict(arrays["provenance"]), "sources": dict(bundle.manifest["sources"])}),
        identity=arrays["identity"],
    )
    if bool(_cfg(cfg, "verify_mano_replay", True)):
        _verify_mano_replay(episode, device=device, tolerance_m=float(_cfg(cfg, "mano_replay_tolerance_m", 1e-4)))
    return episode


def _model_forward(model: Any, root: np.ndarray, pose: np.ndarray, betas: np.ndarray,
                   translation: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    root_t = torch.as_tensor(np.array(root, copy=True), dtype=torch.float32, device=device).reshape(1, 1, 3)
    pose_t = torch.as_tensor(np.array(pose, copy=True), dtype=torch.float32, device=device).reshape(1, 15, 3)
    beta_t = torch.as_tensor(np.array(betas, copy=True), dtype=torch.float32, device=device).reshape(1, -1)
    trans_t = torch.as_tensor(np.array(translation, copy=True), dtype=torch.float32, device=device).reshape(1, 3)
    def matrices(rotvec: Any):
        x, y, z = rotvec.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack([torch.stack([zero, -z, y], dim=-1),
                            torch.stack([z, zero, -x], dim=-1),
                            torch.stack([-y, x, zero], dim=-1)], dim=-2)
        return torch.linalg.matrix_exp(skew)
    with torch.no_grad():
        try:
            output = model(global_orient=matrices(root_t), hand_pose=matrices(pose_t), betas=beta_t,
                           transl=trans_t, return_verts=True, pose2rot=False)
        except (TypeError, RuntimeError):
            try:
                output = model(global_orient=root_t.reshape(1, 3), hand_pose=pose_t.reshape(1, 45), betas=beta_t,
                               transl=trans_t, return_verts=True)
            except TypeError:
                output = model(global_orient=root_t.reshape(1, 3), hand_pose=pose_t.reshape(1, 45), betas=beta_t)
                output.vertices = output.vertices + trans_t[:, None]
                output.joints = output.joints + trans_t[:, None]
    vertices = output.vertices
    joints = output.joints
    # Standard MANO exposes 16 joints. Match the established 21-joint order using real tips.
    if joints.shape[1] < 21 and vertices.shape[1] == 778:
        from utils.mano import JOINT_ORDER, TIP_VERTICES
        joints = torch.cat([joints[:, :16], vertices[:, TIP_VERTICES]], dim=1)[:, JOINT_ORDER]
    elif joints.shape[1] > 21:
        joints = joints[:, :21]
    return (vertices[0].detach().cpu().numpy().astype(np.float32),
            joints[0].detach().cpu().numpy().astype(np.float32))


def _verify_mano_replay(episode: EpisodeStatic, *, device: str, tolerance_m: float) -> None:
    for side, values in episode.hands.items():
        valid = np.flatnonzero(values["valid"])
        if not len(valid):
            continue
        for i in (int(valid[0]), int(valid[-1])):
            vertices, _ = _model_forward(episode.mano_models[side], values["root_orient"][i],
                                         values["articulation"][i], values["betas"][i],
                                         values["translation"][i], device)
            error = np.linalg.norm(vertices - values["vertices"][i], axis=1)
            if not np.isfinite(error).all() or float(error.max()) > tolerance_m:
                raise RuntimeError(f"{side} MANO replay mismatch max={float(error.max()):.6g}m")


def _rotvec_matrix(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        skew = np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]],
                         [-vector[1], vector[0], 0.0]])
        return np.eye(3) + skew
    axis = vector / angle
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _matrix_rotvec(matrix: np.ndarray) -> np.ndarray:
    R = np.asarray(matrix, dtype=np.float64)
    cosine = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    axis = np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / max(2.0 * np.sin(angle), 1e-12) * angle


def action_gate_mask(state: AnnotationState, cfg: Mapping[str, Any] | Any) -> tuple[np.ndarray, tuple[str, ...]]:
    episode = state.episode
    i = state.position
    hand_present = [bool(episode.hands[side]["valid"][i]) for side in episode.layout.hand_slots]
    object_present = [bool(episode.objects[name]["valid"][i]) for name in episode.layout.object_slots]
    contact_active = []
    frame_id = int(episode.frame_idx[i])
    for side, present in zip(episode.layout.hand_slots, hand_present):
        row = episode.contact.get((frame_id, side), {})
        contact_active.append(bool(present and any(object_present) and row.get("state") == "active_contact_patch"))
    allowed = episode.layout.presence_mask(hand_present, object_present, contact_active)
    reasons = np.full(episode.layout.size, "allowed", dtype=object)
    reasons[~allowed] = "absent_entity_or_inactive_contact"
    if bool(_cfg(cfg, "freeze_translation_without_visible_surface_support", False)):
        threshold = int(_cfg(cfg, "translation_gate_min_visible_surface_depth_vertices", 0))
        for side, present in zip(episode.layout.hand_slots, hand_present):
            if not present:
                continue
            support = int(episode.contact.get((frame_id, side), {}).get("translation_support_count", 0))
            if support <= threshold:
                block = episode.layout.block(f"hand.{side}.translation")
                allowed[block.slice] = False
                reasons[block.slice] = "missing_visible_surface_translation_support"
    explicit = _cfg(cfg, "hard_action_mask")
    if explicit is not None:
        mask = np.asarray(explicit, dtype=bool)
        if mask.ndim == 2:
            mask = mask[i]
        if mask.shape != (episode.layout.size,):
            raise ValueError("hard_action_mask shape mismatch")
        newly = allowed & ~mask
        allowed &= mask
        reasons[newly] = "configured_semantic_hard_gate"
    return allowed, tuple(map(str, reasons.tolist()))


def filter_action(state: AnnotationState, raw_action: Sequence[float], cfg: Mapping[str, Any] | Any) -> tuple[np.ndarray, GateTrace]:
    scales_value = _cfg(cfg, "action_scales", {})
    scales = scales_value if isinstance(scales_value, ActionScales) else ActionScales(**dict(scales_value))
    allowed, reasons = action_gate_mask(state, cfg)
    trace = filter_scaled_action(raw_action, state.episode.layout, scales, allowed,
                                 reason_by_dimension=reasons)
    return trace.executed_delta.copy(), trace


def decode_candidate(state: AnnotationState, executed_delta: Sequence[float],
                     cfg: Mapping[str, Any] | Any, device: str = "cpu") -> None:
    episode = state.episode
    if state.done:
        raise RuntimeError("cannot decode action after episode completion")
    i = state.position
    decoded = episode.layout.decode(executed_delta)
    for side in episode.layout.hand_slots:
        if not bool(episode.hands[side]["valid"][i]):
            continue
        root_delta = decoded[f"hand.{side}.root"]
        pose_delta = decoded[f"hand.{side}.articulation"]
        trans_delta = decoded[f"hand.{side}.translation"]
        base_root = _rotvec_matrix(episode.hands[side]["root_orient"][i])
        state.hand_root[side][i] = _matrix_rotvec(_rotvec_matrix(root_delta) @ base_root)
        base_pose = episode.hands[side]["articulation"][i].reshape(15, 3)
        delta_pose = pose_delta.reshape(15, 3)
        state.hand_articulation[side][i] = np.stack([
            _matrix_rotvec(_rotvec_matrix(delta_pose[j]) @ _rotvec_matrix(base_pose[j]))
            for j in range(15)
        ]).reshape(45)
        state.hand_translation[side][i] = episode.hands[side]["translation"][i] + trans_delta
        vertices, joints = _model_forward(episode.mano_models[side], state.hand_root[side][i],
                                          state.hand_articulation[side][i], episode.hands[side]["betas"][i],
                                          state.hand_translation[side][i], device)
        state.hand_vertices[side][i] = vertices
        state.hand_joints[side][i] = joints
    for slot_name in episode.layout.object_slots:
        value = episode.objects[slot_name]
        if not bool(value["valid"][i]):
            continue
        dR = _rotvec_matrix(decoded[f"object.{slot_name}.rotation"])
        state.object_rotation[slot_name][i] = dR @ np.asarray(value["rotation"][i])
        delta_t = decoded[f"object.{slot_name}.translation"].astype(np.float64)
        if _cfg(cfg, "object_translation_frame", "world") == "camera":
            delta_t = episode.T_world_camera[i, :3, :3] @ delta_t
        state.object_translation[slot_name][i] = np.asarray(value["translation"][i]) + delta_t
    for slot, side in enumerate(episode.layout.hand_slots):
        if episode.layout.contact_variables_per_hand:
            frame_id = int(episode.frame_idx[i])
            prior = float(np.clip(episode.contact.get((frame_id, side), {}).get("prior", 0.5), 1e-4, 1.0 - 1e-4))
            base_logit = float(np.log(prior / (1.0 - prior)))
            state.contact_logits[i, slot] = base_logit + decoded[f"contact.{side}"]
    state.applied[i] = True
    state.term_cache.pop(i, None)


def transformed_object_geometry(state: AnnotationState, frame_position: int, slot_name: str) -> tuple[np.ndarray, np.ndarray]:
    value = state.episode.objects[slot_name]
    vertices = np.asarray(value["vertices_canonical"], dtype=np.float64)
    R = state.object_rotation[slot_name][frame_position]
    t = state.object_translation[slot_name][frame_position]
    return vertices @ R.T + t[None, :], np.asarray(value["faces"], dtype=np.int64)


def refresh_candidate_terms(state: AnnotationState, cfg: Mapping[str, Any] | Any,
                            frame_position: int | None = None) -> dict[str, Any]:
    """Refresh candidate-dependent unsigned hand/object matches.

    Signed factors are never synthesized. Non-watertight or unauthorized meshes
    explicitly disable them in the returned trace.
    """
    i = state.position if frame_position is None else int(frame_position)
    episode = state.episode
    result: dict[str, Any] = {"contact": {}, "signed": {}, "object_observed": {}}
    for slot_name in episode.layout.object_slots:
        obj = episode.objects[slot_name]
        if not bool(obj["valid"][i]):
            result["signed"][slot_name] = {"state": "inactive_absent_object", "hands": {}}
            continue
        authority = np.asarray(obj.get("signed_authority", []), dtype=bool)
        signed_eligible = bool(obj.get("watertight", False) and obj.get("signed_geometry_ready", False)
                               and len(authority) == len(obj["faces"]) and np.any(authority))
        signed_entry: dict[str, Any] = {"state": "active_declared_mesh_bound_local_face_authority" if signed_eligible
            else "inactive_nonwatertight_or_missing_local_face_authority", "hands": {}}
        result["signed"][slot_name] = signed_entry
        if signed_eligible:
            from types import SimpleNamespace
            from utils.candidate_geometry import (make_scene, active_constraints_from_vertices,
                dense_observed_surface_constraints_from_vertices)
            scene = make_scene(obj["vertices_canonical"], obj["faces"])
            for side in episode.layout.hand_slots:
                if not bool(episode.hands[side]["valid"][i]):
                    continue
                row = SimpleNamespace(
                    object_rotation_world_from_object=state.object_rotation[slot_name][i],
                    object_translation_world_m=state.object_translation[slot_name][i],
                    face_strict_observed=authority,
                    current_vertices_world=episode.hands[side]["vertices"][i],
                    observed_surface_support_uncertainty_m=float(_cfg(cfg, "signed_surface_support_uncertainty_m", 0.0)),
                )
                active_ids, active_normals, active_depths = active_constraints_from_vertices(
                    state.hand_vertices[side][i], row, scene,
                    int(_cfg(cfg, "max_constraints_per_frame", 256)),
                    float(_cfg(cfg, "penetration_epsilon_m", 0.001)),
                    episode.hands[side]["vertices"][i])
                dense_ids, dense_normals, dense_depths = dense_observed_surface_constraints_from_vertices(
                    state.hand_vertices[side][i], row, scene, episode.hands[side]["vertices"][i])
                signed_entry["hands"][side] = {
                    "active_ids": active_ids, "active_normals": active_normals, "active_depths": active_depths,
                    "dense_ids": dense_ids, "dense_normals": dense_normals, "dense_depths": dense_depths,
                    "uncertainty_m": row.observed_surface_support_uncertainty_m,
                }
        observed = np.asarray(obj["observed_points_world"][i], dtype=np.float64)
        if len(observed):
            R = state.object_rotation[slot_name][i]
            t = state.object_translation[slot_name][i]
            result["object_observed"][slot_name] = (observed - t[None, :]) @ R
    # Contact correspondences use the same current nearest-triangle helper as the legacy fit.
    if not episode.layout.object_slots:
        return result
    slot_name = episode.layout.object_slots[0]
    obj = episode.objects[slot_name]
    if not bool(obj["valid"][i]) or not len(obj["faces"]):
        return result
    active_sides = [side for side in episode.layout.hand_slots
                    if episode.contact.get((int(episode.frame_idx[i]), side), {}).get("state") == "active_contact_patch"
                    and bool(episode.hands[side]["valid"][i])]
    if not active_sides:
        return result
    try:
        from utils.candidate_geometry import make_scene, contact_patch_targets_from_vertices
        scene = make_scene(obj["vertices_canonical"], obj["faces"])
        triangles = np.asarray(obj["vertices_canonical"])[np.asarray(obj["faces"], dtype=np.int64)]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)
        authority = np.ones(len(obj["faces"]), dtype=bool)  # unsigned proximity only
        R = state.object_rotation[slot_name][i]
        t = state.object_translation[slot_name][i]
        for side in active_sides:
            row = episode.contact[(int(episode.frame_idx[i]), side)]
            ids, targets_world, normals_world, distances = contact_patch_targets_from_vertices(
                state.hand_vertices[side][i], scene, R, t, authority, normals,
                max_vertices=int(_cfg(cfg, "max_contact_patch_vertices", 160)),
                band_m=float(row.get("band_m", _cfg(cfg, "contact_patch_band_m", 0.18))),
            )
            targets_object = (targets_world - t[None, :]) @ R if len(targets_world) else targets_world
            normals_object = normals_world @ R if len(normals_world) else normals_world
            result["contact"][side] = {
                "vertex_ids": ids, "targets_object": targets_object, "normals_object": normals_object,
                "targets_world": targets_object @ R.T + t[None, :] if len(targets_object) else targets_object,
                "normals_world": normals_object @ R.T if len(normals_object) else normals_object,
                "distances_m": distances, "state": "active_unsigned_nearest_triangle" if len(ids) else "inactive_no_near_triangle",
                "weight": float(row.get("weight", 0.0)), "margin_m": float(row.get("margin_m", 0.0)),
                "support_uncertainty_m": float(row.get("support_uncertainty_m", 0.0)),
            }
    except ImportError as exc:
        raise RuntimeError("active contact geometry requires Open3D and migrated correspondence helpers") from exc
    return result
