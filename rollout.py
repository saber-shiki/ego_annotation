"""Deterministic, complete checkpoint-backed annotation rollout."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from train import (
    SUPPORTED_SKRL_VERSION,
    _jsonable_dataset_identity,
    _sha256,
    _write_json_create,
    build_agent,
    build_env,
    compatibility_contract,
    load_runtime_config,
)


_EXPORT_SCHEMA = "ego.annotation.skrl_rollout_manifest/2.0.0"


def _load_provenance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != "ego.annotation.skrl_checkpoint/2.0.0":
        raise ValueError("unsupported or missing skrl checkpoint provenance")
    return value


def _verify_checkpoint_contract(*, provenance: Mapping[str, object], checkpoint: Path, cfg, env) -> None:
    import skrl

    errors: list[str] = []
    if provenance.get("skrl_version") != SUPPORTED_SKRL_VERSION or skrl.__version__ != SUPPORTED_SKRL_VERSION:
        errors.append("skrl version differs from training/supported runtime")
    checkpoint_record = provenance.get("checkpoint", {})
    if not isinstance(checkpoint_record, Mapping) or checkpoint_record.get("sha256") != _sha256(checkpoint):
        errors.append("checkpoint hash differs from provenance")
    if provenance.get("compatibility") != compatibility_contract(env, cfg):
        errors.append("dataset/environment/model compatibility differs from checkpoint")
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


def _reset_episode(env, episode_index: int):
    """Reset the underlying Gymnasium episode by index then tensorize as skrl does.

    GymnasiumWrapper.reset does not accept `options`, so passing the explicit
    episode index through it is impossible. This uses the wrapper's own space
    conversion helpers; environment stepping remains through `env.step`.
    """
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

    raw_env = env._unwrapped
    observation, info = raw_env.reset(options={"episode_index": int(episode_index)})
    tensor = tensorize_space(env.observation_space, observation, device=env.device)
    return flatten_tensorized_space(tensor), info


def _require_finite(value, label: str) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite rollout {label}")


def _slug(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return normalized or "episode"


def _discover_export_files(result: object, requested: Path) -> list[Path]:
    """Discover JSON/geometry files from the environment's explicit export result."""
    paths = {requested.resolve()}

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                if str(child_key) == "document":
                    continue
                visit(child, str(child_key))
        elif isinstance(value, (str, Path)) and key.lower() in {"path", "json", "npz", "artifact"}:
            path = Path(value).expanduser().resolve()
            if path.is_file() and path.suffix.lower() in {".json", ".npz"}:
                paths.add(path)

    visit(result)
    existing = sorted((path for path in paths if path.is_file()), key=lambda path: str(path))
    suffixes = {path.suffix.lower() for path in existing}
    if ".json" not in suffixes or ".npz" not in suffixes:
        raise RuntimeError("AnnotationEnv.export_annotation must return/create both JSON and geometry NPZ artifacts")
    return existing


def _artifact_record(path: Path, root: Path) -> dict[str, str]:
    try:
        label = str(path.relative_to(root))
    except ValueError:
        label = str(path)
    return {"path": label, "sha256": _sha256(path), "format": path.suffix.lower().lstrip(".")}


def _run_episode(env, agent, *, episode_index: int, output: Path, global_timestep: int) -> tuple[dict[str, Any], int]:
    import torch

    observations, reset_info = _reset_episode(env, episode_index)
    bundle_id = str(reset_info.get("bundle_id", f"episode-{episode_index}"))
    frame_count = 0
    while True:
        _require_finite(observations, "observation")
        states = env.state()
        if states is not None:
            _require_finite(states, "state")
        with torch.no_grad():
            actions, outputs = agent.act(
                observations,
                states,
                timestep=global_timestep,
                timesteps=global_timestep + 1,
            )
        if "mean_actions" not in outputs:
            raise RuntimeError("Gaussian rollout policy did not return mean_actions")
        actions = outputs["mean_actions"]
        _require_finite(actions, "mean action")
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        _require_finite(rewards, "reward")
        _require_finite(next_observations, "next observation")
        for name in ("raw_action", "executed_action"):
            if name in info:
                value = torch.as_tensor(info[name])
                _require_finite(value, name)
        frame_count += 1
        global_timestep += 1
        if bool(truncated.any()):
            raise RuntimeError(
                f"episode {bundle_id!r} truncated after {frame_count} frames; refusing incomplete annotation export"
            )
        if bool(terminated.any()):
            export_result = env.export_annotation(output)
            if not output.exists():
                raise RuntimeError("AnnotationEnv.export_annotation did not create its requested policy artifact")
            artifacts = [_artifact_record(path, output.parent.parent) for path in _discover_export_files(export_result, output)]
            return {
                "episode_index": episode_index,
                "bundle_id": bundle_id,
                "clip_id": reset_info.get("clip_id"),
                "source_video_id": reset_info.get("source_video_id"),
                "frame_count": frame_count,
                "completion": "natural_termination",
                "artifacts": artifacts,
            }, global_timestep
        observations = next_observations


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
        raise FileExistsError(f"rollout manifest is create-only: {output}")
    rollout_cfg = cfg.get("rollout", {})
    if isinstance(rollout_cfg, Mapping) and bool(rollout_cfg.get("stochastic_evaluation", False)):
        raise ValueError("annotation rollout requires stochastic_evaluation=false and uses Gaussian mean_actions")
    episodes_root = output.parent / f"{output.stem}.episodes"
    if episodes_root.exists():
        raise FileExistsError(f"rollout episode directory is create-only: {episodes_root}")

    env = build_env(cfg, dataset_path=dataset_path, split=split, device=device)
    _verify_checkpoint_contract(provenance=provenance, checkpoint=checkpoint, cfg=cfg, env=env)
    agent = build_agent(
        env,
        cfg,
        training=False,
        experiment_dir=output.parent / "runs",
        experiment_name="rollout",
    )
    agent.load(str(checkpoint))
    agent.enable_training_mode(False, apply_to_models=True)
    preprocessor_state = _snapshot_preprocessors(agent)

    dataset = env.episode_dataset
    if len(dataset) <= 0:
        raise ValueError("rollout dataset is empty")
    episodes_root.mkdir(parents=True)
    global_timestep = 0
    episodes: list[dict[str, Any]] = []
    for episode_index in range(len(dataset)):
        record = getattr(dataset, "records", ())
        expected = record[episode_index] if episode_index < len(record) else {}
        expected_bundle = expected.get("bundle_id") if isinstance(expected, Mapping) else None
        episode_output = episodes_root / f"{episode_index:05d}_{_slug(expected_bundle or 'episode')}.json"
        entry, global_timestep = _run_episode(
            env,
            agent,
            episode_index=episode_index,
            output=episode_output,
            global_timestep=global_timestep,
        )
        if expected_bundle is not None and entry["bundle_id"] != expected_bundle:
            raise RuntimeError("environment reset episode identity disagrees with dataset record")
        episodes.append(entry)
    _assert_preprocessors_unchanged(agent, preprocessor_state)

    manifest = {
        "schema": _EXPORT_SCHEMA,
        "checkpoint_sha256": _sha256(checkpoint),
        "training_config_sha256": provenance["training_config_sha256"],
        "inference_config_sha256": config_sha256,
        "skrl_version": SUPPORTED_SKRL_VERSION,
        "compatibility": compatibility_contract(env, cfg),
        "training_identity": provenance["training_identity"],
        "inference_identity": _jsonable_dataset_identity(env),
        "split": str(split),
        "episode_count": len(episodes),
        "executed_frame_count": global_timestep,
        "stochastic_evaluation": False,
        "random_timesteps": 0,
        "preprocessors_frozen_during_evaluation": True,
        "action_selection": "loaded Gaussian mean_actions; environment filters and exports raw/executed traces",
        "episodes": episodes,
    }
    _write_json_create(output, manifest)
    sidecar = output.with_name(output.name + ".provenance.json")
    _write_json_create(
        sidecar,
        {
            "schema": "ego.annotation.skrl_rollout_provenance/2.0.0",
            "manifest": _artifact_record(output, output.parent),
            "episodes": episodes,
            "geometry_and_json_artifacts_bound": True,
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
    print(json.dumps({"manifest": str(output), "provenance": str(provenance)}, indent=2))


if __name__ == "__main__":
    main()
