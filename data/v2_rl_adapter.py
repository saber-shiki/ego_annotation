"""Explicit boundary for v2 observations entering trainable RL.

The v2 observation bundle is intentionally not consumable by the legacy
``EpisodeDataset``.  This module makes that boundary machine-readable instead
of encouraging callers to delete the guard or reinterpret v2 as v1.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

V2_SCHEMA = "ego.annotation.observation_bundle.v2"
RL_ADAPTER_CONTRACT_SCHEMA = "ego.annotation.rl_observation_adapter_contract.v1"
RL_OUTPUT_SCHEMA = "ego.annotation.observation_bundle.v3_rl"
REQUIRED_CONTRACT_FIELDS = (
    "action_layout_version",
    "observation_layout_version",
    "reward_profile",
    "evidence_qualification",
    "visibility_uncertainty",
    "factor_activation_provenance",
    "geometry_roles",
    "source_hashes",
)


@dataclass(frozen=True)
class V2RLAdapterContract:
    input_path: str
    input_schema: str
    output_schema: str
    status: str
    missing_contract_fields: tuple[str, ...]
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": RL_ADAPTER_CONTRACT_SCHEMA,
            "input": {"path": self.input_path, "schema": self.input_schema},
            "output_schema": self.output_schema,
            "status": self.status,
            "missing_contract_fields": list(self.missing_contract_fields),
            "reason": self.reason,
            "trainable": self.status == "ready",
        }


def inspect_v2_manifest(path: str | Path) -> V2RLAdapterContract:
    """Inspect, but do not promote, a v2 preprocessing-only manifest."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("v2 observation manifest must be a JSON object")
    schema = str(payload.get("schema", ""))
    if schema != V2_SCHEMA:
        raise ValueError(f"expected {V2_SCHEMA}, got {schema!r}")
    if payload.get("preprocessing_only") is not True:
        raise ValueError("v2 observation manifest must declare preprocessing_only=true")
    declared = payload.get("rl_training_contract")
    declared = declared if isinstance(declared, Mapping) else {}
    missing = tuple(field for field in REQUIRED_CONTRACT_FIELDS if field not in declared)
    return V2RLAdapterContract(
        input_path=str(source),
        input_schema=schema,
        output_schema=RL_OUTPUT_SCHEMA,
        status="blocked_not_trainable" if missing else "ready_pending_runtime_adapter",
        missing_contract_fields=missing,
        reason=(
            "v2 remains preprocessing-only until an explicit adapter emits the "
            "versioned RL observation/action/reward contract"
            if missing else
            "contract fields are declared, but no promotion is performed by this inspector"
        ),
    )


def require_trainable_v2_adapter(path: str | Path) -> V2RLAdapterContract:
    """Fail closed; this function is the future promotion seam."""
    contract = inspect_v2_manifest(path)
    if contract.status != "ready_pending_runtime_adapter":
        raise RuntimeError(
            "v2 observation bundle is not trainable: "
            + ", ".join(contract.missing_contract_fields)
        )
    raise NotImplementedError(
        "the explicit v2->v3_rl runtime adapter is not implemented; "
        "do not bypass EpisodeDataset's preprocessing_only guard"
    )
