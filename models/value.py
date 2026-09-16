"""Causal value model for skrl PPO."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model

from models.encoders import GroupedFeatureEncoder


class AnnotationValue(DeterministicMixin, Model):
    """Value function over observations, or an explicit causal state space.

    `use_states` is accepted only when the environment actually supplies a
    state space and state tensor. The stock Gymnasium wrapper does not invent
    privileged state; the default therefore evaluates the same causal packed
    observation seen by the actor.
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
        use_states: bool = False,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        if use_states and state_space is None:
            raise ValueError("critic use_states requires an environment state_space")
        self.use_states = bool(use_states)
        input_dim = self.num_states if self.use_states else self.num_observations
        self.encoder = GroupedFeatureEncoder(
            input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            feature_groups=feature_groups,
        )
        self.value_head = nn.Linear(self.encoder.output_dim, 1)

    def compute(self, inputs, role):
        values = inputs.get("states") if self.use_states else inputs["observations"]
        if values is None:
            raise ValueError("critic configured for states but environment returned no state")
        return self.value_head(self.encoder(values)), {}
