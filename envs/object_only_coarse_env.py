"""Six-dimensional latent-action PPO over a frozen coarse object initializer.

All physical factors remain unchanged. Hands/contact have no policy coordinates.
PPO likelihood is over raw Gaussian latents, not over the mapped physical delta.
"""
from __future__ import annotations
from dataclasses import replace
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from data.immutable import freeze
from envs.coarse_annotation_env import CoarseAnnotationEnv

CONTRACT = "coarse_object_only_radial_tanh_zero_relative.v1"


def radial_tanh(vector, limit):
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all() or not np.isfinite(limit) or limit <= 0:
        raise ValueError("finite 3D vector and positive trust radius required")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return value.copy()
    return value * (float(limit) * np.tanh(norm / float(limit)) / norm)


def physical_delta(latent, cfg):
    value = np.asarray(latent, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("object policy requires six finite raw latent actions")
    units = cfg["action_scales"]
    bounds = cfg["object_only"]
    return np.r_[radial_tanh(value[:3] * float(units["object_rotation_rad"]), math.radians(float(bounds["max_rotation_deg"]))),
                 radial_tanh(value[3:] * float(units["object_translation_m"]), float(bounds["max_translation_m"]))]


class ObjectOnlyCoarseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset, cfg, device="cpu"):
        super().__init__()
        self.cfg, self.device, self.dataset = cfg, str(device), dataset
        options = cfg.get("object_only", {})
        if options.get("contract") != CONTRACT:
            raise ValueError("explicit object-only policy/reward contract required")
        if cfg.get("max_episode_steps") is not None:
            raise ValueError("zero-relative object experiment requires complete natural episodes")
        for key in ("max_rotation_deg", "max_translation_m"):
            if not np.isfinite(options[key]) or options[key] <= 0:
                raise ValueError("trust radii must be positive finite")
        if cfg.get("object_translation_frame", "world") != "world":
            raise ValueError("object-only v1 translation is in world coordinates")
        self._inner = CoarseAnnotationEnv(dataset, cfg, device)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = self._inner.observation_space
        self.reward_discount_contract = 1.0
        self._zero_cache = {}
        self.zero_prefix_objectives = None
        self.zero_final_ledger = None
        self._scale = float(cfg.get("reward_scale", 1.))
        self._units = np.r_[[cfg["action_scales"]["object_rotation_rad"]] * 3,
                            [cfg["action_scales"]["object_translation_m"]] * 3].astype(np.float64)

    @property
    def state(self):
        return self._inner.state

    @property
    def episode_index(self):
        return self._inner.episode_index

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = dict(options or {})
        if "episode_index" not in opts:
            opts["episode_index"] = int(self.np_random.integers(len(self.dataset)))
        observation, info = self._inner.reset(seed=seed, options=opts)
        index = self.episode_index
        if index not in self._zero_cache:
            values = [0.]
            count = self.state.episode.frame_count
            for _ in range(count):
                _, _, done, truncated, zero_info = self._inner.step(np.zeros(self.state.episode.layout.size, np.float32))
                if truncated:
                    raise RuntimeError("zero reference cannot truncate")
                values.append(float(zero_info["objective"]))
            if not done or len(values) != count + 1:
                raise RuntimeError("zero reference is not complete")
            self._zero_cache[index] = (np.asarray(values, np.float64), zero_info["factor_ledger"])
            observation, info = self._inner.reset(seed=seed, options=opts)
        values, ledger = self._zero_cache[index]
        self.zero_prefix_objectives, self.zero_final_ledger = values, ledger
        # Bind the new subspace/transform in the exported candidate provenance.
        episode = self.state.episode
        adapter = {**dict(episode.provenance["preprocessed_state_adapter"]),
                   "policy_subspace": CONTRACT, "latent_action_dim": 6,
                   "hand_and_contact_actions": "absent_not_sampled",
                   "object_correction_reference": "same_frame_frozen_R6_initializer",
                   "max_rotation_norm_deg": float(self.cfg["object_only"]["max_rotation_deg"]),
                   "max_translation_norm_m": float(self.cfg["object_only"]["max_translation_m"]),
                   "reward_convention": "sum(r)=-reward_scale*(J(final)-J0(final)), gamma=1",
                   "raw_objective_is_unmodified": True}
        self.state.episode = replace(episode, provenance=freeze({**dict(episode.provenance), "preprocessed_state_adapter": adapter}))
        info.update(policy_subspace=CONTRACT, latent_action_dim=6,
                    reward_discount_contract="gamma=1 required for exact zero-relative final-objective equivalence")
        return observation, info

    def step(self, action):
        if self.state is None:
            raise RuntimeError("reset required")
        latent = np.asarray(action, np.float32)
        delta = physical_delta(latent, self.cfg)
        layout = self.state.episode.layout
        expanded = np.zeros(layout.size, np.float32)
        r = layout.block("object.object:0.rotation").slice
        t = layout.block("object.object:0.translation").slice
        expanded[r] = delta[:3] / self._units[:3]
        expanded[t] = delta[3:] / self._units[3:]
        position = self.state.position
        observation, raw_reward, done, truncated, info = self._inner.step(expanded)
        baseline_before = float(self.zero_prefix_objectives[position])
        baseline = float(self.zero_prefix_objectives[position + 1])
        reward = float(raw_reward + self._scale * (baseline - baseline_before))
        executed = np.r_[info["executed_delta"][r], info["executed_delta"][t]].astype(np.float64)
        info.update(full_gate_trace=info["gate_trace"], raw_action=latent.copy(),
                    executed_action=(executed / self._units).astype(np.float32), executed_delta=executed,
                    original_reward=raw_reward, zero_prefix_objective=baseline,
                    zero_final_objective=float(self.zero_prefix_objectives[-1]),
                    objective_relative_to_zero=float(info["objective"] - baseline),
                    reward_discount_contract="sum(r)=-reward_scale*(J(final)-J0(final)), gamma=1")
        info["gate_trace"] = {"contract": CONTRACT, "raw_latent": latent.tolist(),
                              "requested_physical_delta": delta.tolist(), "executed_physical_delta": executed.tolist(),
                              "log_probability_space": "six_raw_Gaussian_latents"}
        return observation, reward, done, truncated, info

    def export_annotation(self, path: str | Path | None = None):
        return self._inner.export_annotation(path)

    def close(self):
        self._inner.close()
