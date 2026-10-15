"""Normal full-action PPO with an explicitly trusted generated-mesh barrier.

This is NOT a source signed-authority override. It is a separately named
operator-approved experimental hypothesis appended to unchanged formal costs.
"""
from __future__ import annotations
from dataclasses import replace

import numpy as np
from data.immutable import freeze
from envs.coarse_annotation_env import CoarseAnnotationEnv
from rewards.evaluator import RewardEvaluator
from utils.generated_collision import CONTRACT, GeneratedCollisionScene


class GeneratedCollisionReward(RewardEvaluator):
    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        options = cfg.get("generated_collision", {})
        if options.get("contract") != CONTRACT or options.get("trust_complete_generated_mesh") is not True:
            raise ValueError("explicit operator-trusted generated collision hypothesis required")
        self.options = options
        if options.get("epsilon_m") != .0001 or options.get("nsamples") != 5:
            raise ValueError("v1 reviewed only at 0.1mm tolerance / five sign rays")
        if not np.isfinite(options["weight"]) or options["weight"] <= 0:
            raise ValueError("positive finite collision weight required")
        self._scenes = {}

    def scene_for(self, obj):
        key = obj["pose_mesh_sha256"]
        if key not in self._scenes:
            self._scenes[key] = GeneratedCollisionScene(obj["vertices_canonical"], obj["faces"])
        return self._scenes[key]

    def _signed_physics(self, state, end, add, activation):
        # Preserve all ordinary signed gating/traces. The new term is explicitly
        # hypothetical, not silently using all generated faces as observations.
        super()._signed_physics(state, end, add, activation)
        episode = state.episode
        for i in range(end):
            frame = int(episode.frame_idx[i])
            cache = state.term_cache.setdefault(i, {})
            values = cache.setdefault("generated_collision_hypothesis", {})
            for slot in episode.layout.object_slots:
                obj = episode.objects[slot]
                if not obj["valid"][i]:
                    continue
                scene = self.scene_for(obj)
                for side in episode.layout.hand_slots:
                    if not episode.hands[side]["valid"][i]:
                        continue
                    key = (slot, side)
                    if key not in values:
                        faces = episode.hands[side]["faces"]
                        faces = faces[i] if faces.ndim == 3 else faces
                        row = scene.query(state.hand_vertices[side][i], faces,
                            state.object_rotation[slot][i], state.object_translation[slot][i])
                        depth = row["depth_m"]
                        residual = np.maximum(depth - self.options["epsilon_m"], 0.)
                        positive = int(np.count_nonzero(residual))
                        numerator = float(self.options["weight"] * np.square(residual).sum())
                        values[key] = (numerator / max(positive, 1), {
                            "numerator": numerator, "denominator": max(positive, 1),
                            "active_count": len(depth), "positive_count": positive,
                            "max_inside_depth_mm": float(max(0., depth.max()) * 1000),
                            "reduction": "positive_residual_count_on_full_MANO_vertices_and_all_triangle_centroids",
                        })
                    value, stats = values[key]
                    name = f"hypothesis/{slot}/{side}/penetration/frame={frame}"
                    add(name, value, **stats)
                    activation[name] = "active_operator_trusted_complete_generated_mesh_not_observed_signed_authority"


class GeneratedCollisionCoarseEnv(CoarseAnnotationEnv):
    def __init__(self, dataset, cfg, device="cpu"):
        super().__init__(dataset, cfg, device)
        if cfg.get("hard_action_mask") is not None or cfg.get("freeze_translation_without_visible_surface_support", False):
            raise ValueError("full-action experiment forbids additional hand-freeze masks")
        self.evaluator = GeneratedCollisionReward(cfg, device)
        if self.action_space.shape != (118,):
            raise ValueError("v1 full-action hypothesis requires 118D / five contact logits per hand")

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        episode = self.state.episode
        topology = {slot: self.evaluator.scene_for(obj).topology for slot, obj in episode.objects.items() if obj.get("entity_id")}
        adapter = {**dict(episode.provenance["preprocessed_state_adapter"]),
                   "generated_collision_hypothesis": {**self.cfg["generated_collision"], "topology": topology},
                   "policy_action_dim": 118, "hands_frozen": False,
                   "contact_deadband_unchanged": True, "source_signed_authority_promoted": False}
        self.state.episode = replace(episode, provenance=freeze({**dict(episode.provenance), "preprocessed_state_adapter": adapter}))
        info.update(generated_collision_hypothesis=CONTRACT, hands_frozen=False)
        return observation, info

    def step(self, action):
        result = super().step(action)
        ledger = result[-1]["factor_ledger"]
        original = sum(v for k, v in ledger["terms"].items() if not k.startswith("hypothesis/"))
        collision = sum(v for k, v in ledger["terms"].items() if k.startswith("hypothesis/"))
        result[-1].update(original_formal_objective=original, generated_collision_objective=collision,
                          collision_authority="operator_trusted_generated_mesh_hypothesis_only")
        return result
