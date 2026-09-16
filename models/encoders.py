"""Configurable encoders for flattened causal annotation observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


@dataclass(frozen=True)
class FeatureGroup:
    """A named half-open slice in the environment's flattened observation."""

    name: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start


def _hidden_dims(values: Iterable[int]) -> tuple[int, ...]:
    dims = tuple(int(value) for value in values)
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("encoder hidden_dims must contain positive integers")
    return dims


def parse_feature_groups(input_dim: int, values: Sequence[Mapping[str, object]] | None) -> tuple[FeatureGroup, ...]:
    """Validate a complete, non-overlapping layout or infer one dynamic group.

    The environment owns the observation packing. A configured layout is checked
    against its actual flattened Gymnasium space; no hand/object feature count is
    embedded in the model.
    """
    if input_dim <= 0:
        raise ValueError("encoder input dimension must be positive")
    if not values:
        return (FeatureGroup("causal_observation", 0, input_dim),)
    groups = tuple(
        FeatureGroup(name=str(value["name"]), start=int(value["start"]), stop=int(value["stop"])) for value in values
    )
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("feature group names must be unique")
    cursor = 0
    for group in sorted(groups, key=lambda item: item.start):
        if group.start != cursor or group.stop <= group.start or group.stop > input_dim:
            raise ValueError("feature groups must form a contiguous partition of the flattened input")
        cursor = group.stop
    if cursor != input_dim:
        raise ValueError("feature groups must cover the flattened input exactly")
    return groups


class MLPEncoder(nn.Module):
    """Small configurable MLP used independently for each declared feature group."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "elu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = (int(input_dim),) + _hidden_dims(hidden_dims)
        try:
            activation_type = _ACTIVATIONS[activation.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported encoder activation: {activation}") from exc
        layers: list[nn.Module] = []
        for source, target in zip(dims, dims[1:]):
            layers.append(nn.Linear(source, target))
            if layer_norm:
                layers.append(nn.LayerNorm(target))
            layers.append(activation_type())
        self.network = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("encoder expects a flattened [batch, features] tensor")
        return self.network(values)


class GroupedFeatureEncoder(nn.Module):
    """Encode configured causal feature groups without assuming entity counts."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int],
        activation: str = "elu",
        layer_norm: bool = False,
        feature_groups: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.groups = parse_feature_groups(self.input_dim, feature_groups)
        self.encoders = nn.ModuleDict(
            {
                group.name: MLPEncoder(
                    group.width,
                    hidden_dims,
                    activation=activation,
                    layer_norm=layer_norm,
                )
                for group in self.groups
            }
        )
        self.output_dim = sum(encoder.output_dim for encoder in self.encoders.values())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"packed observation shape mismatch: expected [batch, {self.input_dim}], got {tuple(values.shape)}"
            )
        encoded = [self.encoders[group.name](values[:, group.start : group.stop]) for group in self.groups]
        return torch.cat(encoded, dim=-1)
