"""API-only skrl lifecycle tests.

Fixtures prove checkpoint/inference control flow only. They intentionally do not
claim MANO, reconstructed geometry, factors, rendering, or annotation quality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
import yaml

from models.factory import build_models
from rollout import rollout
from train import (
    assert_agent_checkpoint_finite,
    build_agent,
    build_env,
    finite_policy_parameter_changed,
    load_runtime_config,
    train,
)


class FixtureDataset:
    def __init__(self, index_path, *, split):
        self.path = Path(index_path)
        self.split = str(split)
        self.payload = json.loads(self.path.read_text())
        self.records = tuple(dict(row) for row in self.payload["episodes"])
        self.source_video_ids = frozenset(str(row["source_video_id"]) for row in self.records)
        self.compatibility_contract = {
            "bundle_schemas": [self.payload["bundle_schema"]],
            "packing_version": self.payload["packing_version"],
            "action_layout_version": self.payload["action_layout_version"],
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class FixtureAnnotationEnv(gym.Env):
    """Destructive-reset stand-in for the annotation environment interface."""

    metadata = {}

    def __init__(self, dataset, cfg, device="cpu"):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.device = device
        self.observation_space = gym.spaces.Box(-10, 10, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)
        self.observation_feature_groups = [
            {"name": "observed", "start": 0, "stop": 2},
            {"name": "causal_history", "start": 2, "stop": 4},
        ]
        self.current = None
        self.position = 0
        self.actions = []
        self.export_calls = []

    def _observation(self):
        return np.array([self.position, 1.0, max(0, self.position - 1), 1.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int((options or {}).get("episode_index", 0))
        self.current = dict(self.dataset[index])
        self.position = 0
        # Intentionally erase completed state: export after reset must fail.
        self.actions = []
        return self._observation(), {
            "bundle_id": self.current["bundle_id"],
            "clip_id": self.current["clip_id"],
            "source_video_id": self.current["source_video_id"],
        }

    def step(self, action):
        if self.current is None:
            raise RuntimeError("reset required")
        raw = np.asarray(action, dtype=np.float32)
        applied = np.clip(raw, -0.25, 0.25)
        self.actions.append({"raw": raw.tolist(), "executed": applied.tolist()})
        self.position += 1
        terminated = self.position >= int(self.current["frame_count"])
        reward = 1.0 - float(np.square(applied).sum())
        return self._observation(), reward, terminated, False, {}

    def export_annotation(self, output):
        if self.current is None or self.position != int(self.current["frame_count"]):
            raise RuntimeError("completed policy state required before export")
        output = Path(output)
        npz = output.with_suffix(".npz")
        document = {
            "schema": "api_fixture_not_domain_annotation",
            "bundle_id": self.current["bundle_id"],
            "actions": self.actions,
        }
        output.write_text(json.dumps(document, indent=2) + "\n")
        np.savez_compressed(
            npz,
            policy_raw_actions=np.asarray([row["raw"] for row in self.actions], dtype=np.float32),
            policy_executed_actions=np.asarray([row["executed"] for row in self.actions], dtype=np.float32),
        )
        self.export_calls.append(self.current["bundle_id"])
        return {"json": str(output), "npz": str(npz), "document": document}


def fixture_config() -> dict:
    return {
        "runtime": {
            "skrl_version": "2.1.0",
            "dataset_class": "models.test_skrl_runtime:FixtureDataset",
            "environment_class": "models.test_skrl_runtime:FixtureAnnotationEnv",
        },
        "environment": {"fixture_only": True, "layout": "fixture-action-v1"},
        "model": {
            "encoder": {"hidden_dims": [16, 8], "activation": "elu", "layer_norm": False},
            "observation_feature_groups": None,
            "state_feature_groups": None,
            "policy": {
                "initial_log_std": -1.0,
                "min_log_std": -5.0,
                "max_log_std": 1.0,
                "zero_mean_init": True,
            },
            "value": {"use_states": False},
        },
        "ppo": {
            "rollouts": 4,
            "learning_epochs": 2,
            "mini_batches": 2,
            "discount_factor": 1.0,
            "gae_lambda": 0.95,
            "learning_rate": 0.001,
            "random_timesteps": 0,
            "time_limit_bootstrap": True,
        },
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        "trainer": {"timesteps": 16, "headless": True, "disable_progressbar": True},
        "rollout": {"headless": True, "disable_progressbar": True, "stochastic_evaluation": False},
    }


def write_index(root: Path, name: str, episodes, *, schema="fixture.index.v1", packing="fixture-pack-v1") -> Path:
    path = root / name
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "bundle_schema": "fixture.bundle.v1",
                "packing_version": packing,
                "action_layout_version": "fixture-action-v1",
                "episodes": episodes,
            },
            sort_keys=True,
        )
    )
    return path


def rows(*specifications):
    return [
        {
            "bundle_id": bundle,
            "clip_id": f"clip-{bundle}",
            "source_video_id": video,
            "frame_count": frames,
            "manifest_sha256": hashlib.sha256(bundle.encode()).hexdigest(),
        }
        for bundle, video, frames in specifications
    ]


class ModelContractTest(unittest.TestCase):
    def test_dynamic_spaces_zero_mean_but_stochastic_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = write_index(Path(temporary), "index.json", rows(("one", "video-1", 2)))
            dataset = FixtureDataset(index, split="train")
            env = build_env(fixture_config(), dataset_path=index, split="train", dataset=dataset, env_type=FixtureAnnotationEnv)
            models = build_models(env, fixture_config())
            observations, _ = env.reset()
            torch.manual_seed(5)
            actions, outputs = models["policy"].act({"observations": observations}, role="policy")
            self.assertEqual(tuple(actions.shape), (1, 3))
            self.assertTrue(torch.equal(outputs["mean_actions"], torch.zeros_like(outputs["mean_actions"])))
            self.assertFalse(torch.equal(actions, torch.zeros_like(actions)))
            self.assertEqual(models["policy"].encoder.input_dim, 4)
            self.assertEqual(models["policy"].num_actions, 3)
            self.assertIsNone(env.state_space)
            self.assertIsNone(env.state())
            env.close()

    def test_environment_filter_preserves_raw_policy_log_probability(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = write_index(Path(temporary), "index.json", rows(("one", "video-1", 2)))
            dataset = FixtureDataset(index, split="train")
            env = build_env(fixture_config(), dataset_path=index, split="train", dataset=dataset, env_type=FixtureAnnotationEnv)
            policy = build_models(env, fixture_config())["policy"]
            with torch.no_grad():
                policy.mean_head.weight.zero_()
                policy.mean_head.bias.fill_(2.0)
                policy.log_std_parameter.fill_(-20.0)
            observations, _ = env.reset()
            with torch.no_grad():
                raw, outputs = policy.act({"observations": observations}, role="policy")
                expected = torch.distributions.Normal(outputs["mean_actions"], outputs["log_std"].exp()).log_prob(raw).sum(-1, keepdim=True)
                env.step(raw)
            recorded = env._unwrapped.actions[-1]
            self.assertTrue(np.all(np.asarray(recorded["raw"]) > 1.0))
            self.assertTrue(np.allclose(recorded["executed"], [0.25, 0.25, 0.25]))
            self.assertTrue(torch.allclose(outputs["log_prob"], expected))
            env.close()

    def test_value_rejects_unavailable_privileged_state(self):
        config = fixture_config()
        config["model"]["value"]["use_states"] = True
        with tempfile.TemporaryDirectory() as temporary:
            index = write_index(Path(temporary), "index.json", rows(("one", "video-1", 2)))
            dataset = FixtureDataset(index, split="train")
            with self.assertRaisesRegex(ValueError, "state_space"):
                build_env(config, dataset_path=index, split="train", dataset=dataset, env_type=FixtureAnnotationEnv)

    def test_finite_checkpoint_and_update_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = write_index(Path(temporary), "index.json", rows(("one", "video-1", 2)))
            dataset = FixtureDataset(index, split="train")
            env = build_env(fixture_config(), dataset_path=index, split="train", dataset=dataset, env_type=FixtureAnnotationEnv)
            agent = build_agent(env, fixture_config(), training=True, experiment_dir=Path(temporary), experiment_name="finite")
            initial = {name: value.detach().clone() for name, value in agent.policy.state_dict().items()}
            with torch.no_grad():
                next(iter(agent.policy.parameters())).fill_(float("nan"))
            with self.assertRaisesRegex(RuntimeError, "non-finite checkpoint tensor"):
                assert_agent_checkpoint_finite(agent)
            with self.assertRaisesRegex(RuntimeError, "non-finite policy parameter"):
                finite_policy_parameter_changed(agent, initial)
            clean = build_agent(env, fixture_config(), training=True, experiment_dir=Path(temporary), experiment_name="finite-clean")
            with torch.no_grad():
                clean._observation_preprocessor.running_variance.fill_(float("nan"))
            with self.assertRaisesRegex(RuntimeError, "observation_preprocessor"):
                assert_agent_checkpoint_finite(clean)
            env.close()


class TrainRolloutContractTest(unittest.TestCase):
    def test_complete_multi_episode_rollout_new_compatible_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_index = write_index(root, "training.json", rows(("train", "train-video", 4)))
            inference_rows = rows(("held-a", "video-a", 2), ("held-b", "video-b", 5), ("held-c", "video-c", 3))
            inference_index = write_index(root, "inference.json", inference_rows)
            config = root / "ppo.yaml"
            config.write_text(yaml.safe_dump(fixture_config(), sort_keys=True))

            checkpoint, provenance = train(config_path=config, dataset_path=training_index, split="train", output_dir=root / "training")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_report = json.loads(provenance.read_text())
            self.assertTrue(checkpoint_report["optimizer_update_changed_policy"])
            self.assertTrue(checkpoint_report["checkpoint_tensors_finite"])
            self.assertEqual(set(saved), {"policy", "value", "optimizer", "observation_preprocessor", "value_preprocessor"})
            self.assertGreater(float(saved["observation_preprocessor"]["current_count"]), 1.0)

            manifest, sidecar = rollout(
                config_path=config,
                dataset_path=inference_index,
                split="test",
                checkpoint_path=checkpoint,
                provenance_path=provenance,
                output_path=root / "heldout.manifest.json",
            )
            result = json.loads(manifest.read_text())
            self.assertEqual(result["episode_count"], 3)
            self.assertEqual(result["executed_frame_count"], 10)
            self.assertEqual([row["bundle_id"] for row in result["episodes"]], ["held-a", "held-b", "held-c"])
            self.assertTrue(all(row["completion"] == "natural_termination" for row in result["episodes"]))
            self.assertEqual(result["training_identity"]["dataset"]["index_sha256"], hashlib.sha256(training_index.read_bytes()).hexdigest())
            self.assertEqual(result["inference_identity"]["index_sha256"], hashlib.sha256(inference_index.read_bytes()).hexdigest())
            self.assertTrue(result["preprocessors_frozen_during_evaluation"])
            for episode in result["episodes"]:
                self.assertEqual({artifact["format"] for artifact in episode["artifacts"]}, {"json", "npz"})
                archive = next(artifact for artifact in episode["artifacts"] if artifact["format"] == "npz")
                with np.load(root / archive["path"], allow_pickle=False) as arrays:
                    self.assertEqual(len(arrays["policy_raw_actions"]), episode["frame_count"])
                    self.assertTrue(np.any(np.abs(arrays["policy_raw_actions"]) > 0))
            sidecar_value = json.loads(sidecar.read_text())
            self.assertTrue(sidecar_value["geometry_and_json_artifacts_bound"])
            self.assertEqual(len(sidecar_value["episodes"]), 3)

    def test_rollout_rejects_incompatible_contract_and_random_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_index = write_index(root, "training.json", rows(("train", "train-video", 4)))
            incompatible_index = write_index(root, "bad.json", rows(("held", "video", 2)), packing="other-pack-v1")
            config = root / "ppo.yaml"
            config.write_text(yaml.safe_dump(fixture_config(), sort_keys=True))
            checkpoint, provenance = train(config_path=config, dataset_path=training_index, split="train", output_dir=root / "training")
            with self.assertRaisesRegex(ValueError, "compatibility"):
                rollout(
                    config_path=config,
                    dataset_path=incompatible_index,
                    split="test",
                    checkpoint_path=checkpoint,
                    provenance_path=provenance,
                    output_path=root / "bad.manifest.json",
                )
            stochastic = fixture_config()
            stochastic["rollout"]["stochastic_evaluation"] = True
            stochastic_path = root / "stochastic.yaml"
            stochastic_path.write_text(yaml.safe_dump(stochastic, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "stochastic_evaluation=false"):
                rollout(
                    config_path=stochastic_path,
                    dataset_path=training_index,
                    split="test",
                    checkpoint_path=checkpoint,
                    provenance_path=provenance,
                    output_path=root / "stochastic.manifest.json",
                )
            warmup = fixture_config()
            warmup["ppo"]["random_timesteps"] = 1
            warmup_path = root / "warmup.yaml"
            warmup_path.write_text(yaml.safe_dump(warmup, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "random_timesteps=0"):
                rollout(
                    config_path=warmup_path,
                    dataset_path=training_index,
                    split="test",
                    checkpoint_path=checkpoint,
                    provenance_path=provenance,
                    output_path=root / "warmup.manifest.json",
                )

    def test_installed_api_is_exact_2_1(self):
        cfg, digest = load_runtime_config(Path(__file__).parents[1] / "configs" / "ppo_annotation.yaml")
        self.assertEqual(cfg["runtime"]["skrl_version"], "2.1.0")
        self.assertEqual(len(digest), 64)
        from skrl.agents.torch.ppo import PPO_CFG
        import skrl.agents.torch.ppo as ppo_module

        self.assertTrue(hasattr(PPO_CFG(), "gae_lambda"))
        self.assertFalse(hasattr(ppo_module, "PPO_DEFAULT_CONFIG"))


if __name__ == "__main__":
    unittest.main()
