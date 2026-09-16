"""Deterministic checkpoint-backed annotation rollout using training builders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from train import (
    SUPPORTED_SKRL_VERSION,
    _jsonable_dataset_contract,
    _jsonable_dataset_identity,
    _sha256,
    _write_json_create,
    build_agent,
    build_env,
    build_trainer,
    load_runtime_config,
)


def _load_provenance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != "ego.annotation.skrl_checkpoint/1.0.0":
        raise ValueError("unsupported or missing skrl checkpoint provenance")
    return value


def _verify_checkpoint_contract(
    *,
    provenance: Mapping[str, object],
    checkpoint: Path,
    config_sha256: str,
    env,
) -> None:
    import skrl

    errors: list[str] = []
    if provenance.get("config_sha256") != config_sha256:
        errors.append("configuration hash differs from training")
    if provenance.get("skrl_version") != SUPPORTED_SKRL_VERSION or skrl.__version__ != SUPPORTED_SKRL_VERSION:
        errors.append("skrl version differs from training/supported runtime")
    checkpoint_record = provenance.get("checkpoint", {})
    if not isinstance(checkpoint_record, Mapping) or checkpoint_record.get("sha256") != _sha256(checkpoint):
        errors.append("checkpoint hash differs from provenance")
    if provenance.get("dataset_contract") != _jsonable_dataset_contract(env):
        errors.append("dataset contract differs from training")
    for name, actual in (
        ("observation_space", repr(env.observation_space)),
        ("state_space", repr(env.state_space)),
        ("action_space", repr(env.action_space)),
    ):
        if provenance.get(name) != actual:
            errors.append(f"{name} differs from training")
    if errors:
        raise ValueError("checkpoint/runtime mismatch: " + "; ".join(errors))


def _snapshot_preprocessors(agent):
    import torch

    return {
        name: {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in module.state_dict().items()}
        for name, module in agent.checkpoint_modules.items()
        if name.endswith("_preprocessor")
    }


def _assert_preprocessors_unchanged(agent, before) -> None:
    import torch

    after = _snapshot_preprocessors(agent)
    if before.keys() != after.keys():
        raise RuntimeError("preprocessor modules changed during evaluation")
    for name in before:
        if before[name].keys() != after[name].keys() or any(
            not torch.equal(before[name][key], after[name][key]) for key in before[name]
        ):
            raise RuntimeError(f"preprocessor statistics changed during evaluation: {name}")


def rollout(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    split: str,
    checkpoint_path: str | Path,
    provenance_path: str | Path,
    output_path: str | Path,
    device: str = "cpu",
) -> tuple[Path, Path]:
    cfg, config_sha256 = load_runtime_config(config_path)
    checkpoint = Path(checkpoint_path)
    provenance = _load_provenance(Path(provenance_path))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"annotation output is create-only: {output}")

    env = build_env(cfg, dataset_path=dataset_path, split=split, device=device)
    _verify_checkpoint_contract(
        provenance=provenance,
        checkpoint=checkpoint,
        config_sha256=config_sha256,
        env=env,
    )
    agent = build_agent(
        env,
        cfg,
        training=False,
        experiment_dir=output.parent / "runs",
        experiment_name="rollout",
    )

    rollout_cfg = cfg.get("rollout", {})
    configured_timesteps = rollout_cfg.get("timesteps") if isinstance(rollout_cfg, Mapping) else None
    if configured_timesteps is None:
        configured_timesteps = getattr(env, "num_annotation_steps", None)
    if not isinstance(configured_timesteps, int) or configured_timesteps <= 0:
        raise ValueError("AnnotationEnv must expose positive num_annotation_steps when rollout.timesteps is null")

    trainer = build_trainer(env, agent, cfg, evaluation=True, timesteps=configured_timesteps)
    agent.load(str(checkpoint))  # restores policy/value, optimizer and preprocessor statistics
    preprocessor_state = _snapshot_preprocessors(agent)
    trainer.eval()  # deterministic mean_actions because stochastic_evaluation is rejected
    _assert_preprocessors_unchanged(agent, preprocessor_state)

    output.parent.mkdir(parents=True, exist_ok=True)
    export_result = env.export_annotation(output)
    if not output.exists():
        raise RuntimeError("AnnotationEnv.export_annotation did not create the requested artifact")
    sidecar = output.with_name(output.name + ".provenance.json")
    _write_json_create(
        sidecar,
        {
            "schema": "ego.annotation.skrl_rollout/1.0.0",
            "checkpoint_sha256": _sha256(checkpoint),
            "training_config_sha256": config_sha256,
            "skrl_version": SUPPORTED_SKRL_VERSION,
            "dataset_contract": _jsonable_dataset_contract(env),
            "dataset_identity": _jsonable_dataset_identity(env),
            "split": str(split),
            "timesteps": configured_timesteps,
            "stochastic_evaluation": False,
            "preprocessors_frozen_during_evaluation": True,
            "action_selection": "Gaussian mean_actions; physical filtering performed and exported by AnnotationEnv",
            "annotation_output": {"filename": output.name, "sha256": _sha256(output)},
            "export_result": export_result,
        },
    )
    env.close()
    return output, sidecar


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output, provenance = rollout(
        config_path=args.config,
        dataset_path=args.dataset,
        split=args.split,
        checkpoint_path=args.checkpoint,
        provenance_path=args.checkpoint_provenance,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps({"annotation": str(output), "provenance": str(provenance)}, indent=2))


if __name__ == "__main__":
    main()
