"""API-only fixtures for skrl model/train/rollout semantics.

These synthetic tensors validate library plumbing only; they do not validate
MANO, object geometry, factor quality, or annotation quality.
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
from train import build_agent, build_env, load_runtime_config, train


class FixtureDataset:
    def __init__(self, bundle_path, *, split):
        self.path = Path(bundle_path)
        self.split = split
        self.payload = json.loads(self.path.read_text())
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()


class FixtureAnnotationEnv(gym.Env):
    """Tiny causal residual system used only to test the integration API."""

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
        self.num_annotation_steps = int(dataset.payload["steps"])
        self.dataset_contract = {
            "schema": dataset.payload["schema"],
            "bundle_sha256": dataset.sha256,
        }
        self.dataset_identity = {**self.dataset_contract, "split": dataset.split}
        self._episode_actions = []
        self._completed_actions = []
        self._position = 0

    def _observation(self):
        return np.array([self._position, 1.0, max(0, self._position - 1), 1.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._position = 0
        self._episode_actions = []
        return self._observation(), {}

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32)
        applied = np.clip(raw, -0.25, 0.25)
        self._episode_actions.append({"raw": raw.tolist(), "applied": applied.tolist()})
        self._position += 1
        terminated = self._position >= self.num_annotation_steps
        if terminated:
            self._completed_actions = list(self._episode_actions)
        reward = -float(np.square(applied).sum())
        return self._observation(), reward, terminated, False, {}

    def export_annotation(self, output):
        output = Path(output)
        actions = self._completed_actions or self._episode_actions
        payload = {
            "schema": "api_fixture_not_domain_annotation",
            "split": self.dataset.split,
            "actions": actions,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        return {"row_count": len(actions)}


def fixture_config() -> dict:
    return {
        "runtime": {
            "skrl_version": "2.1.0",
            "dataset_class": "models.test_skrl_runtime:FixtureDataset",
            "environment_class": "models.test_skrl_runtime:FixtureAnnotationEnv",
        },
        "environment": {"fixture_only": True},
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
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 0.001,
            "time_limit_bootstrap": True,
        },
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        "trainer": {"timesteps": 16, "headless": True, "disable_progressbar": True},
        "rollout": {
            "timesteps": None,
            "headless": True,
            "disable_progressbar": True,
            "stochastic_evaluation": False,
        },
    }


class ModelContractTest(unittest.TestCase):
    def test_dynamic_spaces_zero_mean_but_stochastic_samples(self):
        dataset = FixtureDataset(self._bundle(), split="train")
        env = build_env(
            fixture_config(),
            dataset_path=dataset.path,
            split="train",
            dataset=dataset,
            env_type=FixtureAnnotationEnv,
        )
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
        dataset = FixtureDataset(self._bundle(), split="train")
        env = build_env(
            fixture_config(),
            dataset_path=dataset.path,
            split="train",
            dataset=dataset,
            env_type=FixtureAnnotationEnv,
        )
        policy = build_models(env, fixture_config())["policy"]
        with torch.no_grad():
            policy.mean_head.weight.zero_()
            policy.mean_head.bias.fill_(2.0)
            policy.log_std_parameter.fill_(-20.0)
        observations, _ = env.reset()
        with torch.no_grad():
            raw, outputs = policy.act({"observations": observations}, role="policy")
            expected = torch.distributions.Normal(outputs["mean_actions"], outputs["log_std"].exp()).log_prob(raw).sum(
                -1, keepdim=True
            )
            env.step(raw)
        recorded = env._unwrapped._episode_actions[-1]
        self.assertTrue(np.all(np.asarray(recorded["raw"]) > 1.0))
        self.assertTrue(np.allclose(recorded["applied"], [0.25, 0.25, 0.25]))
        self.assertTrue(torch.allclose(outputs["log_prob"], expected))
        env.close()

    def test_value_rejects_unavailable_privileged_state(self):
        config = fixture_config()
        config["model"]["value"]["use_states"] = True
        dataset = FixtureDataset(self._bundle(), split="train")
        with self.assertRaisesRegex(ValueError, "state_space"):
            build_env(config, dataset_path=dataset.path, split="train", dataset=dataset, env_type=FixtureAnnotationEnv)

    def _bundle(self):
        root = Path(tempfile.mkdtemp())
        path = root / "bundle.json"
        path.write_text(json.dumps({"schema": "fixture/1", "steps": 4}))
        return path


class TrainRolloutContractTest(unittest.TestCase):
    def test_actual_ppo_checkpoint_and_deterministic_rollout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps({"schema": "fixture/1", "steps": 4}))
            config = root / "ppo.yaml"
            config.write_text(yaml.safe_dump(fixture_config(), sort_keys=True))

            checkpoint, provenance = train(
                config_path=config,
                dataset_path=bundle,
                split="train",
                output_dir=root / "training",
            )
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_report = json.loads(provenance.read_text())
            self.assertTrue(checkpoint_report["optimizer_update_changed_policy"])
            self.assertEqual(
                set(saved),
                {"policy", "value", "optimizer", "observation_preprocessor", "value_preprocessor"},
            )
            self.assertGreater(float(saved["observation_preprocessor"]["current_count"]), 1.0)

            output, sidecar = rollout(
                config_path=config,
                dataset_path=bundle,
                split="heldout",
                checkpoint_path=checkpoint,
                provenance_path=provenance,
                output_path=root / "heldout.annotation.json",
            )
            annotations = json.loads(output.read_text())
            self.assertEqual(annotations["split"], "heldout")
            self.assertEqual(len(annotations["actions"]), 4)
            self.assertTrue(all(set(row) == {"raw", "applied"} for row in annotations["actions"]))
            report = json.loads(sidecar.read_text())
            self.assertFalse(report["stochastic_evaluation"])
            self.assertTrue(report["preprocessors_frozen_during_evaluation"])
            self.assertEqual(report["timesteps"], 4)

            second_output, _ = rollout(
                config_path=config,
                dataset_path=bundle,
                split="heldout",
                checkpoint_path=checkpoint,
                provenance_path=provenance,
                output_path=root / "heldout-repeat.annotation.json",
            )
            self.assertEqual(annotations["actions"], json.loads(second_output.read_text())["actions"])

            modified = fixture_config()
            modified["ppo"]["gae_lambda"] = 0.9
            mismatched_config = root / "mismatch.yaml"
            mismatched_config.write_text(yaml.safe_dump(modified, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "configuration hash differs"):
                rollout(
                    config_path=mismatched_config,
                    dataset_path=bundle,
                    split="heldout",
                    checkpoint_path=checkpoint,
                    provenance_path=provenance,
                    output_path=root / "must-not-exist.json",
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
