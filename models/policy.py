"""Stochastic residual policy for skrl PPO."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, Model

from models.encoders import GroupedFeatureEncoder


class ResidualGaussianPolicy(GaussianMixin, Model):
    """Gaussian actor over the environment-defined residual action space.

    Samples remain unfiltered so PPO stores the sampled action and matching log
    probability. The annotation environment owns physical action filtering and
    must retain the raw/applied distinction in its exported state.
    """

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        layer_norm: bool = False,
        feature_groups: Sequence[Mapping[str, object]] | None = None,
        initial_log_std: float | Sequence[float] = -1.0,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
        zero_mean_init: bool = True,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_mean_actions=False,
            clip_log_std=True,
            min_log_std=float(min_log_std),
            max_log_std=float(max_log_std),
            reduction="sum",
        )
        self.encoder = GroupedFeatureEncoder(
            self.num_observations,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            feature_groups=feature_groups,
        )
        self.mean_head = nn.Linear(self.encoder.output_dim, self.num_actions)
        if zero_mean_init:
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        initial = torch.as_tensor(initial_log_std, dtype=torch.float32)
        if initial.ndim == 0:
            initial = initial.repeat(self.num_actions)
        if initial.shape != (self.num_actions,) or not torch.isfinite(initial).all():
            raise ValueError("initial_log_std must be finite scalar or one value per policy action")
        if torch.any(initial < float(min_log_std)) or torch.any(initial > float(max_log_std)):
            raise ValueError("initial_log_std must lie inside declared policy bounds")
        self.log_std_parameter = nn.Parameter(initial.clone())

    def compute(self, inputs, role):
        observations = inputs["observations"]
        mean_actions = self.mean_head(self.encoder(observations))
        return mean_actions, {"log_std": self.log_std_parameter}
