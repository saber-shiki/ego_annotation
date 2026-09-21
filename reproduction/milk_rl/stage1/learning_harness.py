"""Experiment-only learning-curve harness around the frozen skrl training API.

This module never changes PPO, the environment, reward, action layout, or source
snapshot.  It installs one post-interaction callback on the original agent and
runs each deterministic evaluation in a fresh subprocess.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import importlib.metadata
import platform
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from types import MethodType
from typing import Any, Callable, Mapping

sys.dont_write_bytecode = True
for _name, _value_env in {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
    os.environ[_name] = _value_env

SOURCE_ROOT = Path("/mnt/user-home/kupingxin/ego_annotation_runtime/rl_milk_minfix_v1_20260918")
DEFAULT_CONFIG = SOURCE_ROOT / "configs/ppo_annotation.yaml"
DEFAULT_INDEX = Path("/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/indexes/milk_823a906_minfix_v1.json")
DEFAULT_OUTPUT = Path("/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_learning_stage1_1024_v1")
ZERO_SUMMARY = Path("/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_rl_smoke_minfix_v1/zero/summary.json")
PREVIOUS_256_OBJECTIVE = 8900.215
BOUNDARIES = (256, 512, 768, 1024)


def _source_imports() -> None:
    source = str(SOURCE_ROOT)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(child) for child in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _write_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def set_deterministic_cpu(seed: int = 0) -> None:
    import numpy as np
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be hidden")


def rng_hashes() -> dict[str, str]:
    import numpy as np
    import torch
    values = {
        "python": pickle.dumps(random.getstate()),
        "numpy": pickle.dumps(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().cpu().numpy().tobytes(),
    }
    return {name: hashlib.sha256(value).hexdigest() for name, value in values.items()}


def environment_snapshot(env) -> dict[str, Any]:
    raw = getattr(env, "_unwrapped", env)
    state = getattr(raw, "state", None)
    payload = {
        "episode_index": getattr(raw, "episode_index", None),
        "elapsed_steps": getattr(raw, "elapsed_steps", None),
        "position": getattr(state, "position", None),
        "environment_rng": _json(raw.np_random.bit_generator.state) if hasattr(raw, "np_random") else None,
    }
    encoded = json.dumps(_json(payload), sort_keys=True, allow_nan=False).encode()
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _finite_checkpoint(agent) -> None:
    _source_imports()
    from train import assert_agent_checkpoint_finite
    assert_agent_checkpoint_finite(agent)


def install_post_interaction_callback(agent, callback: Callable[..., None], boundaries=BOUNDARIES):
    """Install exactly one callback after the original method, with no trainer restart."""
    if getattr(agent, "_learning_callback_installed", False):
        raise RuntimeError("post-interaction callback already installed")
    agent._learning_callback_installed = True
    original = agent.post_interaction
    seen: set[int] = set()
    boundaries = frozenset(int(value) for value in boundaries)

    def wrapped(self, *, timestep: int, timesteps: int) -> None:
        completed = int(timestep) + 1
        if completed in seen:
            raise RuntimeError(f"duplicate post-interaction checkpoint callback at {completed}")
        original(timestep=timestep, timesteps=timesteps)
        if completed in boundaries:
            seen.add(completed)
            callback(agent=self, completed_steps=completed, timesteps=int(timesteps))

    agent.post_interaction = MethodType(wrapped, agent)
    return seen


def _value(info: Mapping[str, Any], key: str, default=None):
    value = info.get(key, default)
    return _json(value)


class TrainingDiagnostics:
    units = {
        "hand_root": "rad", "hand_articulation": "rad", "hand_translation": "m",
        "object_rotation": "rad", "object_translation": "m", "contact": "logit",
    }

    def __init__(self, env, path: Path):
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.metrics: dict[str, list[float]] = {}
        self.block_values: dict[str, dict[str, list[float]]] = {}
        self.env = env
        self.blocks = self._blocks(env)

    @staticmethod
    def _blocks(env):
        raw = getattr(env, "_unwrapped", env)
        try:
            from actions.space import ActionLayout
            cfg = getattr(raw, "cfg", {})
            max_objects = int(cfg.get("max_objects", 1)) if isinstance(cfg, Mapping) else 1
            contacts = int(cfg.get("contact_variables_per_hand", 1)) if isinstance(cfg, Mapping) else 1
            return ActionLayout(object_slots=tuple(f"object:{i}" for i in range(max_objects)), contact_variables_per_hand=contacts).blocks
        except ImportError:
            return ()

    def record_step(self, action, info: Mapping[str, Any], reward: Any, terminated: Any) -> None:
        import numpy as np
        raw = np.asarray(_value(info, "raw_action", action), dtype=float).reshape(-1)
        executed = np.asarray(_value(info, "executed_action", action), dtype=float).reshape(-1)
        physical = np.asarray(_value(info, "executed_delta", executed), dtype=float).reshape(-1)
        gate = info.get("gate_trace", {})
        allowed = np.asarray(_value(gate, "allowed_mask", np.ones(raw.shape, dtype=bool)), dtype=bool).reshape(-1)
        reasons = list(_value(gate, "reasons", ["unknown"] * len(raw)))
        row = {"raw": raw.tolist(), "executed": executed.tolist(), "physical_delta": physical.tolist(),
               "allowed_mask": allowed.tolist(), "reasons": reasons, "reward": _value({}, "x", reward),
               "objective": _value(info, "objective"), "terminated": bool(np.asarray(terminated).any()),
               "bundle_id": info.get("bundle_id"), "frame_idx": info.get("frame_idx"), "step": len(self.rows) + 1}
        json.dumps(_json(row), allow_nan=False)  # fail immediately, including training observations below
        self.rows.append(row)
        for block in self.blocks:
            values = self.block_values.setdefault(block.kind, {"raw": [], "executed": [], "physical_delta": [], "zeroed": 0, "filtered": 0, "total": 0})
            sl = slice(block.start, block.stop)
            for name, array in (("raw", raw[sl]), ("executed", executed[sl]), ("physical_delta", physical[sl])):
                values[name].extend(float(v) for v in array)
            values["zeroed"] += int(np.count_nonzero((np.abs(raw[sl]) > 0) & (np.abs(executed[sl]) == 0)))
            values["filtered"] += int(np.count_nonzero(~allowed[sl]))
            values["total"] += int(block.stop - block.start)

    def track(self, tag: str, value: float) -> None:
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite PPO metric: {tag}")
        if tag.startswith(("Loss /", "Policy /", "KL /")):
            self.metrics.setdefault(tag, []).append(float(value))

    def finish(self) -> None:
        blocks = {}
        import numpy as np
        for kind, values in self.block_values.items():
            blocks[kind] = {
                "distributions": {name: {"mean": float(np.mean(values[name])), "std": float(np.std(values[name])),
                    "min": float(np.min(values[name])), "max": float(np.max(values[name])),
                    "rms": float(np.sqrt(np.mean(np.square(values[name]))))}
                    for name in ("raw", "executed", "physical_delta")},
                "raw_executed_units": "normalized_unbounded",
                "physical_delta_units": self.units[kind],
                "filtered_fraction": values["filtered"] / max(1, values["total"]),
                "raw_abs_mean": float(sum(abs(v) for v in values["raw"]) / max(1, len(values["raw"]))),
                "executed_abs_mean": float(sum(abs(v) for v in values["executed"]) / max(1, len(values["executed"]))),
                "physical_delta_abs_mean": float(sum(abs(v) for v in values["physical_delta"]) / max(1, len(values["physical_delta"]))),
                "zeroed_fraction": values["zeroed"] / max(1, values["total"]),
            }
        _write_create(self.path, {"schema": "ego.annotation.learning_training_diagnostics/1", "steps": self.rows,
                                  "training_episode_end_objectives": [row["objective"] for row in self.rows if row["terminated"]],
                                  "action_blocks": blocks, "KL": "unavailable: skrl 2.1 update keeps KL local, not tracked",
                                  "ppo_metrics": self.metrics or {
                                      "status": "unavailable", "unavailable": ["policy_loss", "value_loss", "KL", "policy_std"]
                                  }})


def instrument_training_env(env, diagnostics: TrainingDiagnostics):
    original = env.step
    def step(self, action):
        result = original(action)
        if not __import__("torch").isfinite(result[0]).all():
            raise RuntimeError("non-finite training observation")
        if bool(result[3].any()):
            raise RuntimeError("training episode unexpectedly truncated")
        diagnostics.record_step(action, result[-1], result[1], result[2])
        return result
    env.step = MethodType(step, env)
    return env


def aggregate_final_ledgers(ledgers: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only one final ledger per episode; prefix ledgers are never summed."""
    if not ledgers:
        raise ValueError("evaluation produced no final ledgers")
    terms: dict[str, float] = {}
    grouped: dict[str, float] = {}
    objective = 0.0
    for ledger in ledgers:
        value = float(ledger["objective"])
        if not (value == value and abs(value) != float("inf")):
            raise RuntimeError("non-finite final objective")
        terms_sum = float(ledger.get("baseline_constant", 0)) + sum(float(v) for v in ledger.get("terms", {}).values())
        if not math.isclose(value, terms_sum, rel_tol=1e-9, abs_tol=1e-8):
            raise RuntimeError("final ledger terms do not reconstruct objective")
        objective += value
        for name, term in ledger.get("terms", {}).items():
            terms[name] = terms.get(name, 0.0) + float(term)
            family = str(name).split("/", 1)[0]
            grouped[family] = grouped.get(family, 0.0) + float(term)
    json.dumps({"objective": objective, "terms": terms}, allow_nan=False)
    detailed_families = {}
    for name, value in terms.items():
        parts = name.split("/")
        family = "/".join([parts[0], *parts[2:3]]) if len(parts) >= 3 else name
        detailed_families[family] = detailed_families.get(family, 0.0) + value
    return {"objective_by_named_term_family": detailed_families, "episode_count": len(ledgers), "final_objective": objective, "final_terms": terms,
            "objective_by_term_family": grouped}


def _evaluator_build_env(collector):
    _source_imports()
    import rollout as original_rollout
    original = original_rollout.build_env
    def build_env(*args, **kwargs):
        env = original(*args, **kwargs)
        old_step = env.step
        def step(self, action):
            result = old_step(action)
            info = result[-1]
            collector.append({"bundle_id": _value(info, "bundle_id"), "raw": _value(info, "raw_action"), "executed": _value(info, "executed_action"),
                              "physical_delta": _value(info, "executed_delta"),
                              "gate_trace": _value(info, "gate_trace", {}), "reward": _value({}, "x", result[1]),
                              "objective": _value(info, "objective"), "factor_ledger": _value(info, "factor_ledger", {}),
                              "terminated": bool(result[2].any()), "truncated": bool(result[3].any())})
            return result
        env.step = MethodType(step, env)
        return env
    original_rollout.build_env = build_env
    return original_rollout


def evaluator_main(args) -> None:
    set_deterministic_cpu(0)
    collector: list[dict[str, Any]] = []
    original_rollout = _evaluator_build_env(collector)
    started = time.monotonic()
    manifest, sidecar = original_rollout.rollout(config_path=args.config, dataset_path=args.dataset, split=args.split,
                                                   checkpoint_path=args.checkpoint, provenance_path=args.provenance,
                                                   output_path=args.manifest, device=args.device)
    manifest_value = json.loads(manifest.read_text())
    artifact_paths = []
    for episode in manifest_value.get("episodes", []):
        for artifact in episode.get("artifacts", []):
            if not isinstance(artifact, Mapping):
                raise RuntimeError("rollout manifest artifact mapping is malformed")
            artifact_path = Path(str(artifact["path"]))
            resolved = artifact_path if artifact_path.is_absolute() else manifest.parent / artifact_path
            if not resolved.is_file() or sha256(resolved) != artifact.get("sha256"):
                raise RuntimeError(f"rollout artifact hash/path verification failed: {resolved}")
            artifact_paths.append({"path": str(resolved), "sha256": sha256(resolved), "format": artifact.get("format")})
    final_by_episode: dict[str, Mapping[str, Any]] = {}
    for row in collector:
        ledger = row.get("factor_ledger") or {}
        if row.get("terminated") and ledger.get("end_position") is not None:
            final_by_episode[str(row.get("bundle_id"))] = ledger
    # Natural completion means the last ledger in each episode is terminal. The
    # source rollout asserts termination and complete export before returning.
    if len(final_by_episode) != manifest_value["episode_count"]:
        raise RuntimeError("terminal ledgers disagree with exported episodes")
    for episode in manifest_value["episodes"]:
        if final_by_episode[episode["bundle_id"]]["end_position"] != episode["frame_count"]:
            raise RuntimeError("final ledger/export length mismatch")
    if not final_by_episode:
        raise RuntimeError("evaluator collected no factor ledgers")
    aggregate = aggregate_final_ledgers([final_by_episode[key] for key in sorted(final_by_episode)])
    diagnostics = {"schema": "ego.annotation.learning_evaluation_diagnostics/1", "steps": collector,
                   "final_ledgers": [final_by_episode[key] for key in sorted(final_by_episode)],
                   "artifact_outputs": artifact_paths, **aggregate, "manifest": str(manifest), "sidecar": str(sidecar),
                   "elapsed_seconds": time.monotonic() - started}
    _write_create(args.diagnostics, diagnostics)
    print(json.dumps({"manifest": str(manifest), "sidecar": str(sidecar), "diagnostics": str(args.diagnostics),
                      "artifact_outputs": artifact_paths, "final_objective": aggregate["final_objective"]}, sort_keys=True))


def run_evaluator(args, *, timeout: float = 3600.0) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--evaluator", "--config", str(args.config),
               "--dataset", str(args.dataset), "--split", str(args.split), "--checkpoint", str(args.checkpoint),
               "--provenance", str(args.provenance), "--manifest", str(args.manifest),
               "--diagnostics", str(args.diagnostics), "--device", str(args.device)]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    try:
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        (args.manifest.parent / "evaluator.timeout.log").write_text(str(exc))
        raise RuntimeError(f"evaluator timeout after {timeout}s") from exc
    (args.manifest.parent / "evaluator.stdout.log").write_text(result.stdout)
    (args.manifest.parent / "evaluator.stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("evaluator failed; training stopped: " + (result.stderr[-2000:] or result.stdout[-2000:]))
    if not args.diagnostics.exists():
        raise RuntimeError("evaluator exited successfully without diagnostics")
    result_value = json.loads(result.stdout.strip().splitlines()[-1])
    if result_value["diagnostics"] != str(args.diagnostics) or result_value["manifest"] != str(args.manifest):
        raise RuntimeError("evaluator output mapping mismatch")
    return result_value


def read_zero_baseline() -> dict[str, Any]:
    summary = json.loads(ZERO_SUMMARY.read_text())
    trace = ZERO_SUMMARY.with_name("trace.jsonl")
    last = None
    with trace.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    ledger = last["factor_ledger"]
    if ledger["end_position"] != 150 or not summary["terminated"]:
        raise RuntimeError("zero baseline is not complete")
    aggregate = aggregate_final_ledgers([ledger])
    if not math.isclose(aggregate["final_objective"], summary["final_objective"], rel_tol=1e-10):
        raise RuntimeError("zero baseline final ledger mismatch")
    files = [ZERO_SUMMARY, trace, *[ZERO_SUMMARY.parent / name for name in
             ("candidate.json", "candidate.npz", "candidate_integrity.json")]]
    return {**aggregate, "final_ledger": ledger, "sha256": {str(p): sha256(p) for p in files}}


def runtime_identity():
    import torch
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED")},
            "scripts": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            "source_files": {str(p.relative_to(SOURCE_ROOT)): sha256(p) for p in sorted(SOURCE_ROOT.rglob("*"))
                             if p.is_file() and "__pycache__" not in p.parts}}


def effective_training_config(cfg):
    result = copy.deepcopy(cfg)
    result["trainer"]["timesteps"] = 1024
    result["trainer"]["disable_progressbar"] = True
    # Avoid automatic best-reward checkpoints; only post-update periodic checkpoints
    # are candidates. These settings affect logging/storage, not PPO updates.
    result["experiment"] = {"write_interval": 0, "checkpoint_interval": 0}
    return result


def train_main(args) -> None:
    _source_imports()
    set_deterministic_cpu(args.seed)
    from train import build_agent, build_env, build_trainer, checkpoint_provenance, load_runtime_config
    cfg, source_config_sha = load_runtime_config(args.config)
    cfg = effective_training_config(cfg)
    baseline = read_zero_baseline()
    run_started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"create-only output already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    _write_create(args.output / "zero_baseline.json", baseline)
    identity = runtime_identity()
    index_value = json.loads(args.dataset.read_text())
    if len(index_value["episodes"]) != 1 or index_value["episodes"][0]["frame_count"] != 150:
        raise RuntimeError("expected single 150-frame milk index")
    identity["input_hashes"] = {str(p): sha256(p) for p in
        [args.dataset, Path(index_value["episodes"][0]["manifest"])]}
    _write_create(args.output / "runtime_identity.json", identity)
    import yaml
    effective_config = args.output / "effective_config.yaml"
    with effective_config.open("x") as handle:
        yaml.safe_dump(cfg, handle)
    effective_config_sha = sha256(effective_config)
    _write_create(args.output / "run_config.json", {"source_config": str(args.config), "source_config_sha256": source_config_sha,
        "effective_config_sha256": effective_config_sha,
        "effective_trainer_timesteps": 1024, "cfg": cfg, "seed": args.seed, "device": args.device,
        "split": args.split, "source_root": str(SOURCE_ROOT), "python": sys.version,
        "torch": __import__("torch").__version__, "skrl": __import__("skrl").__version__})
    env = build_env(cfg, dataset_path=args.dataset, split=args.split, device=args.device)
    if env.action_space.shape != (118,) or env.observation_space.shape != (480,):
        raise RuntimeError("real milk dimensions changed")
    env._unwrapped._np_random = __import__("numpy").random.default_rng(0)
    diagnostics = TrainingDiagnostics(env, args.output / "training_diagnostics.json")
    instrument_training_env(env, diagnostics)
    agent = build_agent(env, cfg, training=True, experiment_dir=args.output / "runs", experiment_name="train")
    original_track_data = agent.track_data
    def track_data(self, tag, value):
        original_track_data(tag, value)
        diagnostics.track(str(tag), float(value))
    agent.track_data = MethodType(track_data, agent)
    previous_policy = {name: value.detach().clone() for name, value in agent.policy.state_dict().items()}
    trainer = build_trainer(env, agent, cfg)
    checkpoint_records: list[dict[str, Any]] = []

    def checkpoint_callback(*, agent, completed_steps: int, timesteps: int) -> None:
        nonlocal previous_policy
        _finite_checkpoint(agent)
        if agent._rollout != completed_steps or completed_steps % agent.cfg.rollouts:
            raise RuntimeError("checkpoint is not at completed PPO update boundary")
        from train import finite_policy_parameter_changed
        if not finite_policy_parameter_changed(agent, previous_policy):
            raise RuntimeError(f"policy did not change by checkpoint {completed_steps}")
        step_dir = args.output / "checkpoints" / f"step_{completed_steps:04d}"
        step_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = step_dir / "agent.pt"
        agent.save(str(checkpoint))
        _finite_checkpoint(agent)
        provenance = checkpoint.with_name("agent.provenance.json")
        _write_create(provenance, checkpoint_provenance(checkpoint=checkpoint, config_sha256=effective_config_sha,
            cfg=cfg, env=env, split=args.split))
        provenance_value = json.loads(provenance.read_text())
        if provenance_value["checkpoint"]["sha256"] != sha256(checkpoint):
            raise RuntimeError(f"checkpoint hash mismatch at {completed_steps}")
        previous_policy = {name: value.detach().clone() for name, value in agent.policy.state_dict().items()}
        before_rng, before_env = rng_hashes(), environment_snapshot(env)
        started = time.monotonic()
        eval_args = argparse.Namespace(config=effective_config, dataset=args.dataset, split=args.split, checkpoint=checkpoint,
            provenance=provenance, manifest=step_dir / "evaluation.manifest.json",
            diagnostics=step_dir / "evaluation.diagnostics.json", device=args.device)
        result = run_evaluator(eval_args, timeout=args.evaluator_timeout)
        after_rng, after_env = rng_hashes(), environment_snapshot(env)
        evaluated_manifest = json.loads(eval_args.manifest.read_text())
        if evaluated_manifest["executed_frame_count"] != 150 or evaluated_manifest["episode_count"] != 1:
            raise RuntimeError("milk evaluator did not export exactly one 150-frame episode")
        if sha256(checkpoint) != provenance_value["checkpoint"]["sha256"]:
            raise RuntimeError("checkpoint mutated during evaluation")
        if before_rng != after_rng:
            raise RuntimeError(f"evaluator changed training RNG state at {completed_steps}")
        if before_env["sha256"] != after_env["sha256"]:
            raise RuntimeError(f"evaluator changed training environment position at {completed_steps}")
        record = {"steps": completed_steps, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "provenance": str(provenance), "provenance_sha256": sha256(provenance), "evaluation": result,
                  "training_rng_before": before_rng, "training_rng_after": after_rng,
                  "environment_before": before_env, "environment_after": after_env,
                  "elapsed_seconds": time.monotonic() - started}
        checkpoint_records.append(record)
        _write_create(step_dir / "receipt.json", record)

    install_post_interaction_callback(agent, checkpoint_callback)
    try:
        trainer.train()
        diagnostics.finish()
        _finite_checkpoint(agent)
        _write_create(args.output / "checkpoints.json", {"checkpoints": checkpoint_records})
        evaluations = []
        for record in checkpoint_records:
            diagnostics_path = Path(record["evaluation"]["diagnostics"])
            value = json.loads(diagnostics_path.read_text())
            evaluations.append({"steps": record["steps"], "checkpoint": record["checkpoint"],
                                "checkpoint_sha256": record["checkpoint_sha256"], "final_objective": value["final_objective"],
                                "objective_by_term_family": value["objective_by_term_family"],
                                "objective_by_named_term_family": value["objective_by_named_term_family"]})
        if len(evaluations) != len(BOUNDARIES):
            raise RuntimeError("training completed without all four evaluation checkpoints")
        best = min(evaluations, key=lambda row: row["final_objective"])
        immutable_hashes = {**baseline["sha256"], **identity["input_hashes"],
            **{str(SOURCE_ROOT / name): digest for name, digest in identity["source_files"].items()}}
        if any(sha256(Path(name)) != digest for name, digest in immutable_hashes.items()):
            raise RuntimeError("read-only source/input/baseline artifact changed during run")
        zero = baseline["final_objective"]
        for row in evaluations:
            row["delta_vs_zero"] = row["final_objective"] - zero
            row["delta_vs_previous_256"] = row["final_objective"] - PREVIOUS_256_OBJECTIVE
            row["named_term_family_delta_vs_zero"] = {key: row["objective_by_named_term_family"].get(key, 0) -
                baseline["objective_by_named_term_family"].get(key, 0) for key in
                row["objective_by_named_term_family"].keys() | baseline["objective_by_named_term_family"].keys()}
            row["term_family_delta_vs_zero"] = {key: value - baseline["objective_by_term_family"].get(key, 0)
                                                for key, value in row["objective_by_term_family"].items()}
        _write_create(args.output / "summary.json", {"status": "passed", "scope": "same-video optimization; not convergence/generalization",
            "source_config_sha256": source_config_sha, "dataset_index_sha256": sha256(args.dataset),
            "checkpoints": evaluations, "best_checkpoint": best, "zero_final_objective": zero,
            "previous_256_final_objective": PREVIOUS_256_OBJECTIVE, "selection": "minimum FINAL ledger objective",
            "annotation_quality_verified": False, "timesteps": 1024,
            "elapsed_seconds": time.monotonic() - run_started, "continuation_4096": "not authorized or implemented"})
    except Exception as exc:
        if not diagnostics.path.exists():
            diagnostics.finish()
        try:
            _write_create(args.output / "run.failure.json", {"status": "failed", "error": repr(exc),
                                                               "checkpoints_completed": checkpoint_records})
        except FileExistsError:
            pass
        raise
    finally:
        env.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluator", action="store_true")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--dataset", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--split", default="all")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--diagnostics", type=Path)
    p.add_argument("--evaluator-timeout", type=float, default=3600.0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.device != "cpu" or args.seed != 0 or args.split != "all":
        raise ValueError("experiment fixed to CPU, seed0, split all")
    if args.evaluator_timeout <= 0:
        raise ValueError("evaluator timeout must be positive")
    if args.evaluator:
        for name in ("checkpoint", "provenance", "manifest", "diagnostics"):
            if getattr(args, name) is None:
                raise ValueError(f"--{name} is required for evaluator")
        evaluator_main(args)
    else:
        if args.config.resolve() != DEFAULT_CONFIG.resolve() or args.dataset.resolve() != DEFAULT_INDEX.resolve():
            raise ValueError("real training config/index are frozen")
        if args.output.resolve() != DEFAULT_OUTPUT.resolve():
            raise ValueError("experiment output is fixed and create-only")
        train_main(args)


if __name__ == "__main__":
    main()
