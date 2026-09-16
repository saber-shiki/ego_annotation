from __future__ import annotations

"""Gymnasium annotation environment over complete video/declared-clip episodes."""

from pathlib import Path
from typing import Any, Mapping, Sequence
from hashlib import sha256
import json
import numpy as np

import gymnasium as gym

from actions.space import ActionLayout
from data.bundle import EpisodeBundle
from envs.dynamics import build_episode, decode_candidate, filter_action
from envs.observation import ObservationLayout, observation_layout, pack_observation
from envs.spaces import action_space, observation_space
from envs.state import AnnotationState, initial_state
from rewards.evaluator import RewardEvaluator


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class AnnotationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: Sequence[EpisodeBundle], cfg: Mapping[str, Any] | Any,
                 device: str = "cpu") -> None:
        super().__init__()
        if len(dataset) == 0:
            raise ValueError("AnnotationEnv requires a nonempty episode dataset")
        self.dataset = dataset
        self.cfg = cfg
        self.device = str(device)
        self.max_objects = int(self._get("max_objects", 1))
        self.contact_variables_per_hand = int(self._get("contact_variables_per_hand", 1))
        declared_layout = ActionLayout(object_slots=tuple(f"object:{i}" for i in range(self.max_objects)),
                                       contact_variables_per_hand=self.contact_variables_per_hand)
        declared_obs = ObservationLayout(declared_layout.size, 2, self.max_objects,
                                         self.contact_variables_per_hand,
                                         int(self._get("observation_point_samples", 8)))
        self.action_space = action_space(declared_layout)
        self.observation_space = observation_space(declared_obs.size)
        self.state: AnnotationState | None = None
        self.episode_index: int | None = None
        self.elapsed_steps = 0
        self.max_episode_steps = self._get("max_episode_steps", None)
        if self.max_episode_steps is not None and int(self.max_episode_steps) <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.evaluator = RewardEvaluator(cfg, self.device)
        self.reward_discount_contract = 1.0

    def _get(self, name: str, default: Any) -> Any:
        return self.cfg.get(name, default) if isinstance(self.cfg, Mapping) else getattr(self.cfg, name, default)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "episode_index" in options:
            index = int(options["episode_index"])
            if index < 0 or index >= len(self.dataset):
                raise IndexError(index)
        else:
            index = int(self.np_random.integers(0, len(self.dataset)))
        episode = build_episode(self.dataset[index], self.cfg, self.device)
        if episode.layout.size != self.action_space.shape[0] or observation_layout(episode).size != self.observation_space.shape[0]:
            raise RuntimeError("episode entity configuration violates constructor-declared spaces")
        self.episode_index = index
        self.state = initial_state(episode)
        self.elapsed_steps = 0
        self._terminal = False
        observation = pack_observation(self.state)
        return observation, {
            "bundle_id": episode.bundle_id, "clip_id": episode.clip_id,
            "source_video_id": episode.source_video_id,
            "frame_idx": int(episode.frame_idx[0]), "time_s": float(episode.time_s[0]),
            "reward_discount_contract": "gamma=1 required for exact final-objective equivalence",
            "entity_slots": {
                "hands": list(episode.layout.hand_slots),
                "objects": [episode.objects[name].get("entity_id") for name in episode.layout.object_slots],
            },
        }

    def step(self, raw_action):
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self.state.done or getattr(self, "_terminal", False):
            raise RuntimeError("step called after terminated/truncated episode; reset required")
        state = self.state
        position = state.position
        frame_idx = int(state.episode.frame_idx[position])
        executed, trace = filter_action(state, raw_action, self.cfg)
        decode_candidate(state, executed, self.cfg, self.device)
        state.last_gate_trace = trace
        reward, ledger = self.evaluator.transition(state, position + 1)
        state.position = position + 1
        self.elapsed_steps += 1
        terminated = bool(state.position >= state.episode.frame_count)
        truncated = bool(not terminated and self.max_episode_steps is not None
                         and self.elapsed_steps >= int(self.max_episode_steps))
        self._terminal = bool(terminated or truncated)
        observation = pack_observation(state, terminal=self._terminal)
        info = {
            "bundle_id": state.episode.bundle_id,
            "frame_idx": frame_idx,
            "time_s": float(state.episode.time_s[position]),
            "raw_action": trace.raw_action.copy(),
            "executed_action": trace.normalized_executed_action.copy(),
            "executed_delta": trace.executed_delta.copy(),
            "gate_trace": trace.document(),
            "factor_ledger": ledger.document(),
            "objective": ledger.objective,
            "natural_frame_end": terminated,
            "artificial_cap": truncated,
            "reward_discount_contract": "gamma=1 required for exact final-objective equivalence",
        }
        if self._terminal:
            info["final_observation"] = observation.copy()
        return observation, float(reward), terminated, truncated, info

    def export_annotation(self, path: str | Path | None = None) -> dict[str, Any]:
        """Export an action-driven candidate for the direct policy consumer.

        ``path`` accepts a JSON path, NPZ path, or stem and produces three new
        companion artifacts: the full geometry NPZ, the evaluator adapter JSON,
        and a receipt binding their exact SHA256 values. The geometry has direct
        candidate display authority; it is deliberately separate from formal
        physical acceptance and therefore cannot trigger legacy source-HaWoR
        substitution.
        """
        if self.state is None:
            raise RuntimeError("reset must be called before export")
        state = self.state
        episode = state.episode
        object_slots = [name for name in episode.layout.object_slots if episode.objects[name].get("entity_id") is not None]
        if not object_slots:
            raise RuntimeError("cannot export episode with no declared object entity")
        object_slot = object_slots[0]
        obj = episode.objects[object_slot]
        arrays: dict[str, np.ndarray] = {
            "frame_idx": np.asarray(episode.frame_idx, dtype=np.int64),
            "time_s": np.asarray(episode.time_s, dtype=np.float64),
            "T_world_camera": np.asarray(episode.T_world_camera, dtype=np.float64),
            "intrinsics_render": np.asarray(episode.intrinsics, dtype=np.float64),
            "render_size_wh": np.asarray(episode.image_size_wh, dtype=np.int64),
            "object_vertices_canonical": np.asarray(obj["vertices_canonical"], dtype=np.float32),
            "object_faces": np.asarray(obj["faces"], dtype=np.int64),
            "object_rotation": np.asarray(state.object_rotation[object_slot], dtype=np.float64),
            "object_translation_world": np.asarray(state.object_translation[object_slot], dtype=np.float64),
            "object_valid": np.asarray(obj["valid"], dtype=bool),
            "object_pose_valid": np.asarray(obj["valid"], dtype=bool),
            "object_entity_ids": np.asarray([episode.objects[slot]["entity_id"] for slot in object_slots]),
        }
        object_trajectories: dict[str, Any] = {}
        for object_index, slot in enumerate(object_slots):
            value = episode.objects[slot]
            prefix = f"object_{object_index}"
            arrays[f"{prefix}_vertices_canonical"] = np.asarray(value["pose_vertices_canonical"], dtype=np.float32)
            arrays[f"{prefix}_faces"] = np.asarray(value["pose_faces"], dtype=np.int64)
            arrays[f"{prefix}_rotation"] = np.asarray(state.object_rotation[slot], dtype=np.float64)
            arrays[f"{prefix}_translation_world"] = np.asarray(state.object_translation[slot], dtype=np.float64)
            arrays[f"{prefix}_present"] = np.asarray(value.get("present", value["valid"]), dtype=bool)
            arrays[f"{prefix}_pose_valid"] = np.asarray(value["pose_valid"], dtype=bool)
            arrays[f"{prefix}_metric_observation_eligible"] = np.asarray(value["metric_observation_eligible"], dtype=bool)
            rows = [{"frame_idx": int(frame),
                     "rotation_world_from_completed_canonical_matrix": state.object_rotation[slot][i].astype(float).tolist(),
                     "translation_world_m": state.object_translation[slot][i].astype(float).tolist(),
                     "status": "policy_rollout_candidate"}
                    for i, frame in enumerate(episode.frame_idx)]
            object_trajectories[value["entity_id"]] = {"object_slot": slot, "array_prefix": prefix,
                "object_id": value["entity_id"], "pose_rows": rows,
                "pose_mesh_sha256": value.get("pose_mesh_sha256"),
                "collision_mesh_sha256": value.get("collision_mesh_sha256")}
        per_frame_states = []
        for side in episode.layout.hand_slots:
            arrays[f"{side}_valid"] = np.asarray(episode.hands[side]["valid"], dtype=bool)
            # Renderer consumer authority for this direct policy-candidate layer.
            # This does not claim signed/physical acceptance.
            arrays[f"{side}_display_valid"] = np.asarray(episode.hands[side]["valid"], dtype=bool)
            arrays[f"{side}_vertices_world"] = np.asarray(state.hand_vertices[side], dtype=np.float32)
            arrays[f"{side}_joints_world"] = np.asarray(state.hand_joints[side], dtype=np.float32)
            arrays[f"{side}_faces"] = np.asarray(episode.hands[side]["faces"], dtype=np.int64)
            for i, frame in enumerate(episode.frame_idx):
                if episode.hands[side]["valid"][i]:
                    per_frame_states.append({
                        "frame_idx": int(frame), "hand_side": side,
                        "optimized_joints_world_m": state.hand_joints[side][i].astype(float).tolist(),
                        "policy_candidate_vertices_array": f"{side}_vertices_world",
                        "policy_candidate_row": i,
                    })
        sources = episode.provenance.get("sources", {})
        annotations = sources.get("annotations", {}).get("path") if isinstance(sources, Mapping) else None
        pose_rows = object_trajectories[obj["entity_id"]]["pose_rows"]
        document: dict[str, Any] = {
            "schema": "ego.annotation.policy_rollout.v1",
            "status": "policy_candidate",
            "annotation_ready": False,
            "acceptance_substitution_applied": False,
            "bundle_id": episode.bundle_id,
            "clip_id": episode.clip_id,
            "source_video_id": episode.source_video_id,
            "inputs": {"annotations": annotations, "identity": dict(episode.identity)},
            "policy_rollout_array_layer": {
                "role": "direct_policy_candidate_full_geometry",
                "path": None,
                "sha256": None,
                "display_authority": "policy_candidate_geometry_not_formal_acceptance",
            },
            "display_validity_policy": {
                "object_pose_valid": "policy candidate object pose is finite and present",
                "hand_display_valid": "policy candidate full MANO geometry is present",
                "formal_acceptance_substitution": False,
            },
            "per_frame_states": per_frame_states,
            "object_pose_trajectory": {"object_id": obj.get("entity_id"), "pose_rows": pose_rows},
            "object_pose_trajectories": object_trajectories,
            "uncertainty": {
                "claim": "policy candidate; downstream acceptance remains separate",
                "source_geometry": obj.get("geometry_provenance"),
                "source_pose": obj.get("pose_provenance"),
            },
        }
        if path is not None:
            requested = Path(path)
            if requested.suffix.lower() == ".npz":
                npz_path = requested
                json_path = requested.with_suffix(".json")
            elif requested.suffix.lower() == ".json":
                json_path = requested
                npz_path = requested.with_suffix(".npz")
            else:
                json_path = requested.with_suffix(".json")
                npz_path = requested.with_suffix(".npz")
            receipt_path = json_path.with_name(json_path.stem + "_integrity.json")
            for artifact in (json_path, npz_path, receipt_path):
                if artifact.exists():
                    raise ValueError(f"policy rollout export requires new artifact path: {artifact}")
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, **arrays)
            npz_sha256 = _file_sha256(npz_path)
            document["policy_rollout_array_layer"].update({"path": str(npz_path), "sha256": npz_sha256})
            document["companion_integrity_receipt"] = {
                "path": str(receipt_path),
                "schema": "ego.annotation.policy_rollout_integrity.v1",
            }
            json_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            json_sha256 = _file_sha256(json_path)
            receipt = {
                "schema": "ego.annotation.policy_rollout_integrity.v1",
                "role": "bind_direct_policy_candidate_json_and_npz",
                "files": {
                    "policy_rollout_json": {"path": str(json_path), "sha256": json_sha256},
                    "policy_rollout_npz": {"path": str(npz_path), "sha256": npz_sha256},
                },
                "acceptance_substitution_applied": False,
            }
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            receipt_sha256 = _file_sha256(receipt_path)
            return {
                "json": str(json_path), "npz": str(npz_path), "integrity": str(receipt_path),
                "hashes": {"json_sha256": json_sha256, "npz_sha256": npz_sha256,
                           "integrity_sha256": receipt_sha256},
                "document": document, "integrity_document": receipt,
            }
        return {"arrays": arrays, "document": document}
