"""Train the annotation residual policy with official skrl PPO 2.1.0."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SUPPORTED_SKRL_VERSION = "2.1.0"


def _canonical(value: Mapping[str, object]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def load_runtime_config(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("PPO annotation config must be a mapping")
    configured = str(value.get("runtime", {}).get("skrl_version", ""))
    installed = importlib.metadata.version("skrl")
    if configured != SUPPORTED_SKRL_VERSION or installed != configured:
        raise RuntimeError(
            f"skrl version mismatch: config={configured!r}, installed={installed!r}, supported={SUPPORTED_SKRL_VERSION!r}"
        )
    return value, hashlib.sha256(raw).hexdigest()


def import_symbol(specification: str):
    try:
        module_name, symbol_name = specification.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"expected module:symbol import, got {specification!r}") from exc
    return getattr(importlib.import_module(module_name), symbol_name)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _dataset_provenance(
    dataset,
    *,
    dataset_path: Path,
    split: str,
    dataset_class: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return stable compatibility separately from mutable training identity."""
    index = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(index, Mapping) or not isinstance(index.get("schema"), str):
        raise ValueError("dataset index requires a versioned schema")
    declared = getattr(dataset, "compatibility_contract", None)
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise ValueError("dataset compatibility_contract must be a mapping")
        dataset_shape = _canonical(declared)
    else:
        records = getattr(dataset, "records", ())
        schemas: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("dataset records must be mappings")
            manifest = Path(record["manifest"])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or not isinstance(payload.get("schema"), str):
                raise ValueError("episode manifest requires a versioned schema")
            schemas.add(str(payload["schema"]))
        dataset_shape = {"bundle_schemas": sorted(schemas)}

    compatibility = {
        "dataset_class": dataset_class,
        "index_schema": str(index["schema"]),
        **dataset_shape,
    }
    records = getattr(dataset, "records", ())
    manifests = sorted(
        str(record["manifest_sha256"])
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("manifest_sha256"), str)
    )
    identity = {
        "index_sha256": _sha256(dataset_path),
        "split": str(split),
        "source_video_ids": sorted(map(str, getattr(dataset, "source_video_ids", ()))),
        "manifest_sha256": manifests,
    }
    return compatibility, identity


def build_env(
    cfg: Mapping[str, object],
    *,
    dataset_path: str | Path,
    split: str,
    device: str = "cpu",
    dataset=None,
    env_type=None,
):
    """Construct the agreed read-only episode dataset and Gymnasium environment."""
    from skrl.envs.wrappers.torch import wrap_env

    runtime_cfg = _mapping(cfg.get("runtime"), "runtime config")
    dataset_path = Path(dataset_path)
    dataset_class = str(runtime_cfg["dataset_class"])
    if dataset is None:
        dataset_type = import_symbol(dataset_class)
        dataset = dataset_type(dataset_path, split=str(split))
    if env_type is None:
        env_type = import_symbol(str(runtime_cfg["environment_class"]))
    raw_env = env_type(dataset, _mapping(cfg.get("environment"), "environment config"), device=device)
    wrapped = wrap_env(raw_env, wrapper="gymnasium", verbose=False)
    dataset_contract, dataset_identity = _dataset_provenance(
        dataset,
        dataset_path=dataset_path,
        split=split,
        dataset_class=dataset_class,
    )
    wrapped.dataset_contract = dataset_contract
    wrapped.dataset_identity = dataset_identity
    wrapped.episode_dataset = dataset

    use_states = bool(_mapping(cfg.get("model"), "model config").get("value", {}).get("use_states", False))
    if use_states and wrapped.state_space is None:
        raise ValueError("critic use_states=true but AnnotationEnv exposes no state_space")
    return wrapped


def _ppo_cfg(env, cfg: Mapping[str, object], *, experiment_dir: Path, experiment_name: str):
    from skrl.agents.torch.ppo import PPO_CFG
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    values = _mapping(cfg.get("ppo"), "ppo config")
    allowed = {field.name for field in dataclasses.fields(PPO_CFG)} - {
        "experiment",
        "observation_preprocessor",
        "observation_preprocessor_kwargs",
        "state_preprocessor",
        "state_preprocessor_kwargs",
        "value_preprocessor",
        "value_preprocessor_kwargs",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown PPO_CFG fields: {sorted(unknown)}")
    agent_cfg = PPO_CFG(**values)
    agent_cfg.observation_preprocessor = RunningStandardScaler
    agent_cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": env.device}
    if bool(_mapping(cfg.get("model"), "model config").get("value", {}).get("use_states", False)):
        agent_cfg.state_preprocessor = RunningStandardScaler
        agent_cfg.state_preprocessor_kwargs = {"size": env.state_space, "device": env.device}
    agent_cfg.value_preprocessor = RunningStandardScaler
    agent_cfg.value_preprocessor_kwargs = {"size": 1, "device": env.device}
    contract = _mapping(cfg.get("runtime"), "runtime config").get("training_contract")
    if contract is not None:
        from models.fixed_preprocessor import CONTRACT, FixedAsinh
        if contract != CONTRACT or cfg.get("environment", {}).get("training_contract") != contract:
            raise ValueError("unknown or environment-inconsistent fixed PPO contract")
        if not cfg.get("model", {}).get("value", {}).get("zero_value_init", False):
            raise ValueError("fixed PPO v1 requires zero initialized critic")
        agent_cfg.observation_preprocessor = FixedAsinh
        agent_cfg.observation_preprocessor_kwargs = {}
        agent_cfg.state_preprocessor = None
        agent_cfg.value_preprocessor = None
    agent_cfg.experiment.directory = str(experiment_dir)
    agent_cfg.experiment.experiment_name = experiment_name
    experiment_cfg = _mapping(cfg.get("experiment"), "experiment config")
    agent_cfg.experiment.write_interval = experiment_cfg.get("write_interval", "auto")
    agent_cfg.experiment.checkpoint_interval = experiment_cfg.get("checkpoint_interval", "auto")
    agent_cfg.experiment.store_separately = False
    return agent_cfg


def build_agent(
    env,
    cfg: Mapping[str, object],
    *,
    training: bool,
    experiment_dir: str | Path,
    experiment_name: str,
):
    """Build the same official PPO structure for learning or loaded inference."""
    from skrl.agents.torch.ppo import PPO
    from skrl.memories.torch import RandomMemory

    from models.factory import build_models

    models = build_models(env, cfg)
    agent_cfg = _ppo_cfg(env, cfg, experiment_dir=Path(experiment_dir), experiment_name=experiment_name)
    if not training and agent_cfg.random_timesteps:
        raise ValueError("rollout requires ppo.random_timesteps=0; random warmup actions are not deterministic inference")
    memory = (
        RandomMemory(
            memory_size=agent_cfg.rollouts,
            num_envs=env.num_envs,
            device=env.device,
            replacement=False,
        )
        if training
        else None
    )
    return PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )


def build_trainer(env, agent, cfg: Mapping[str, object], *, timesteps: int | None = None):
    from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg

    section = _mapping(cfg.get("trainer"), "trainer config")
    configured_timesteps = section.get("timesteps") if timesteps is None else timesteps
    if configured_timesteps is None:
        raise ValueError("training timesteps must be explicit")
    return SequentialTrainer(
        env=env,
        agents=agent,
        cfg=SequentialTrainerCfg(
            timesteps=int(configured_timesteps),
            headless=bool(section.get("headless", True)),
            render_interval=int(section.get("render_interval", 1)),
            disable_progressbar=bool(section.get("disable_progressbar", False)),
            close_environment_at_exit=False,
        ),
    )


def _jsonable_mapping_attribute(env, name: str) -> dict[str, Any]:
    value = getattr(env, name, None)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"AnnotationEnv must expose nonempty {name} provenance")
    return _canonical(value)


def _jsonable_dataset_identity(env) -> dict[str, Any]:
    return _jsonable_mapping_attribute(env, "dataset_identity")


def _jsonable_dataset_contract(env) -> dict[str, Any]:
    return _jsonable_mapping_attribute(env, "dataset_contract")


def compatibility_contract(env, cfg: Mapping[str, object]) -> dict[str, Any]:
    runtime = _mapping(cfg.get("runtime"), "runtime config")
    return {
        "dataset": _jsonable_dataset_contract(env),
        "environment_class": str(runtime["environment_class"]),
        "environment_config": _canonical(_mapping(cfg.get("environment"), "environment config")),
        "model_config": _canonical(_mapping(cfg.get("model"), "model config")),
        **({"training_contract": runtime["training_contract"]} if "training_contract" in runtime else {}),
        "observation_space": repr(env.observation_space),
        "state_space": repr(env.state_space),
        "action_space": repr(env.action_space),
    }


def _assert_finite_value(value, label: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite checkpoint tensor: {label}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_value(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_value(child, f"{label}[{index}]")


def _assert_module_state_finite(module, name: str) -> None:
    if module is not None and hasattr(module, "state_dict"):
        _assert_finite_value(module.state_dict(), name)


def assert_agent_checkpoint_finite(agent) -> None:
    for name, module in agent.checkpoint_modules.items():
        _assert_module_state_finite(module, name)


def finite_policy_parameter_changed(agent, initial_policy) -> bool:
    import torch

    changed = False
    for name, current in agent.policy.state_dict().items():
        before = initial_policy[name]
        if not torch.isfinite(current).all() or not torch.isfinite(before).all():
            raise RuntimeError(f"non-finite policy parameter during optimizer update: {name}")
        changed = changed or bool(torch.any(current != before).item())
    return changed


def checkpoint_provenance(
    *,
    checkpoint: Path,
    config_sha256: str,
    cfg: Mapping[str, object],
    env,
    split: str,
) -> dict[str, Any]:
    import skrl
    import torch

    return {
        "schema": "ego.annotation.skrl_checkpoint/2.0.0",
        "checkpoint": {"filename": checkpoint.name, "sha256": _sha256(checkpoint)},
        "training_config_sha256": config_sha256,
        "skrl_version": skrl.__version__,
        "torch_version": torch.__version__,
        "compatibility": compatibility_contract(env, cfg),
        "training_identity": {"dataset": _jsonable_dataset_identity(env), "split": str(split)},
        "ppo": _canonical(_mapping(cfg.get("ppo"), "ppo config")),
        "preprocessors": ({"observation": "models.fixed_preprocessor.FixedAsinh", "state": None, "value": None}
            if cfg.get("runtime", {}).get("training_contract") else {
            "observation": "skrl.resources.preprocessors.torch.RunningStandardScaler",
            "state": (
                "skrl.resources.preprocessors.torch.RunningStandardScaler"
                if bool(_mapping(cfg.get("model"), "model config").get("value", {}).get("use_states", False))
                else None
            ),
            "value": "skrl.resources.preprocessors.torch.RunningStandardScaler",
        }),
        "optimizer_update_changed_policy": True,
        "checkpoint_tensors_finite": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_create(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def train(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    split: str,
    output_dir: str | Path,
    device: str = "cpu",
) -> tuple[Path, Path]:
    cfg, config_sha256 = load_runtime_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    env = build_env(cfg, dataset_path=dataset_path, split=split, device=device)
    agent = build_agent(env, cfg, training=True, experiment_dir=output_dir / "runs", experiment_name="train")
    initial_policy = {name: value.detach().clone() for name, value in agent.policy.state_dict().items()}
    trainer = build_trainer(env, agent, cfg)
    trainer.train()
    assert_agent_checkpoint_finite(agent)
    if not finite_policy_parameter_changed(agent, initial_policy):
        raise RuntimeError("training completed without a finite PPO policy parameter update")

    checkpoint = output_dir / "agent.pt"
    agent.save(str(checkpoint))
    provenance_path = output_dir / "agent.provenance.json"
    _write_json_create(
        provenance_path,
        checkpoint_provenance(
            checkpoint=checkpoint,
            config_sha256=config_sha256,
            cfg=cfg,
            env=env,
            split=split,
        ),
    )
    env.close()
    return checkpoint, provenance_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint, provenance = train(
        config_path=args.config,
        dataset_path=args.dataset,
        split=args.split,
        output_dir=args.output,
        device=args.device,
    )
    print(json.dumps({"checkpoint": str(checkpoint), "provenance": str(provenance)}, indent=2))


if __name__ == "__main__":
    main()
