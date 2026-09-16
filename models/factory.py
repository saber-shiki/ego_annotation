"""Model construction shared by training and checkpoint-backed rollout."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from models.policy import ResidualGaussianPolicy
from models.value import AnnotationValue


def _mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return deepcopy(dict(value))


def build_models(env, cfg: Mapping[str, object]):
    """Build PPO policy/value models from actual wrapped environment spaces."""
    model_cfg = _mapping(cfg.get("model"), "model config")
    encoder_cfg = _mapping(model_cfg.get("encoder"), "model.encoder")
    policy_cfg = _mapping(model_cfg.get("policy"), "model.policy")
    value_cfg = _mapping(model_cfg.get("value"), "model.value")

    shared_encoder_keys = {"hidden_dims", "activation", "layer_norm"}
    unknown = set(encoder_cfg) - shared_encoder_keys
    if unknown:
        raise ValueError(f"unknown model.encoder fields: {sorted(unknown)}")

    observation_layout = model_cfg.get("observation_feature_groups")
    if observation_layout is None:
        observation_layout = getattr(env, "observation_feature_groups", None)
    state_layout = model_cfg.get("state_feature_groups")
    if state_layout is None:
        state_layout = getattr(env, "state_feature_groups", None)
    policy_kwargs = {**encoder_cfg, **policy_cfg, "feature_groups": observation_layout}
    value_kwargs = {
        **encoder_cfg,
        **value_cfg,
        "feature_groups": state_layout if value_cfg.get("use_states", False) else observation_layout,
    }

    policy = ResidualGaussianPolicy(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        **policy_kwargs,
    )
    value = AnnotationValue(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        **value_kwargs,
    )
    return {"policy": policy, "value": value}
