from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import learning_harness as harness


class FakeAgent:
    def __init__(self):
        self.original_calls = []
        self.post_interaction = self._post

    def _post(self, *, timestep, timesteps):
        self.original_calls.append((timestep, timesteps))


class FakeEnv:
    def __init__(self):
        self._unwrapped = self
        self.episode_index = 0
        self.elapsed_steps = 3
        self.state = SimpleNamespace(position=3)


class HarnessContractTest(unittest.TestCase):
    def test_post_update_timing_count_and_no_duplicate_calls(self):
        agent = FakeAgent()
        seen = []
        harness.install_post_interaction_callback(
            agent, lambda **kwargs: seen.append(kwargs["completed_steps"]), boundaries=(2,)
        )
        agent.post_interaction(timestep=0, timesteps=2)
        agent.post_interaction(timestep=1, timesteps=2)
        self.assertEqual(agent.original_calls, [(0, 2), (1, 2)])
        self.assertEqual(seen, [2])
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            agent.post_interaction(timestep=1, timesteps=2)
        self.assertEqual(agent.original_calls, [(0, 2), (1, 2)])

    def test_rng_and_environment_snapshots_are_noninterfering(self):
        random.seed(0)
        np.random.seed(0)
        env = FakeEnv()
        before_rng = harness.rng_hashes()
        before_env = harness.environment_snapshot(env)
        after_rng = harness.rng_hashes()
        after_env = harness.environment_snapshot(env)
        self.assertEqual(before_rng, after_rng)
        self.assertEqual(before_env["sha256"], after_env["sha256"])

    def test_final_ledger_aggregation_does_not_sum_prefixes(self):
        result = harness.aggregate_final_ledgers([
            {"objective": 7.0, "terms": {"hand/a": 2.0, "object/b": 5.0}},
            {"objective": 3.0, "terms": {"hand/a": 1.0, "contact/c": 2.0}},
        ])
        self.assertEqual(result["final_objective"], 10.0)
        self.assertEqual(result["final_terms"], {"hand/a": 3.0, "object/b": 5.0, "contact/c": 2.0})
        self.assertEqual(result["objective_by_term_family"], {"hand": 3.0, "object": 5.0, "contact": 2.0})

    def test_original_ppo_model_environment_config_are_unchanged(self):
        import copy
        harness._source_imports()
        from train import load_runtime_config
        original, _ = load_runtime_config(harness.DEFAULT_CONFIG)
        before = copy.deepcopy(original)
        effective = harness.effective_training_config(original)
        self.assertEqual(original, before)
        for key in original.keys() - {'trainer', 'experiment'}:
            self.assertEqual(effective[key], original[key])
        self.assertEqual(effective['ppo']['rollouts'], 128)
        self.assertEqual(effective['ppo']['learning_epochs'], 8)
        self.assertEqual(effective['ppo']['mini_batches'], 8)
        self.assertEqual(effective['ppo']['discount_factor'], 1)
        self.assertEqual(effective['trainer']['timesteps'], 1024)
        self.assertEqual(effective['experiment'], {'write_interval': 0, 'checkpoint_interval': 0})

    def test_final_ledger_aggregation_rejects_nonfinite_and_inconsistent_terms(self):
        for ledger in ({'objective': float('nan'), 'terms': {}},
                       {'objective': 1, 'terms': {'bad': 5}}):
            with self.assertRaises(RuntimeError):
                harness.aggregate_final_ledgers([ledger])

    def test_training_diagnostic_output_mapping_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            diagnostics = harness.TrainingDiagnostics(FakeEnv(), path)
            diagnostics.blocks = [SimpleNamespace(kind='hand_translation', start=0, stop=2)]
            diagnostics.record_step(
                np.array([1.0, 0.0]),
                {"raw_action": np.array([1.0, 0.0]), "executed_action": np.array([0.5, 0.0]),
                 "executed_delta": np.array([0.01, 0.0]), "gate_trace": {"allowed_mask": [True, False], "reasons": ["allowed", "masked"]}},
                -2.0, True,
            )
            diagnostics.finish()
            output = json.loads(path.read_text())
            self.assertEqual(output["steps"][0]["physical_delta"], [0.01, 0.0])
            self.assertIn("ppo_metrics", output)
            block = output['action_blocks']['hand_translation']
            self.assertEqual(block['physical_delta_units'], 'm')
            self.assertEqual(block['raw_executed_units'], 'normalized_unbounded')
            self.assertEqual(block['filtered_fraction'], .5)
            self.assertEqual(block['zeroed_fraction'], 0)
            self.assertAlmostEqual(block['physical_delta_abs_mean'], .005)

    def test_evaluator_failure_is_raised_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(config=Path(directory) / "cfg", dataset=Path(directory) / "idx", split="all",
                                   checkpoint=Path(directory) / "ckpt", provenance=Path(directory) / "prov",
                                   manifest=Path(directory) / "manifest", diagnostics=Path(directory) / "diag", device="cpu")
            original = harness.subprocess.run
            try:
                harness.subprocess.run = lambda *a, **k: SimpleNamespace(returncode=9, stderr="boom", stdout="")
                with self.assertRaisesRegex(RuntimeError, "training stopped"):
                    harness.run_evaluator(args, timeout=1)
            finally:
                harness.subprocess.run = original

class SkrlSyntheticLifecycleTest(unittest.TestCase):
    def test_true_updates_saved_rollout_noninterference_and_failure_stop(self):
        # 16 planned synthetic steps, stopped at 12; no actual MANO/domain data.
        import torch
        import yaml
        harness._source_imports()
        harness.set_deterministic_cpu()
        from train import (build_env, build_agent, build_trainer, checkpoint_provenance,
                           assert_agent_checkpoint_finite, finite_policy_parameter_changed)
        from synthetic_fixture import config, write_index, rows
        with tempfile.TemporaryDirectory(prefix='learning-harness-api-', dir='/tmp') as directory:
            root = Path(directory)
            cfg = config()
            cfg['trainer']['timesteps'] = 16
            index = write_index(root, 'index.json', rows(('a', 'v', 3), ('b', 'v', 3)))
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(cfg))
            env = build_env(cfg, dataset_path=index, split='all', device='cpu')
            agent = build_agent(env, cfg, training=True, experiment_dir=root, experiment_name='synthetic')
            trainer = build_trainer(env, agent, cfg)
            initial = {k: v.clone() for k, v in agent.policy.state_dict().items()}
            original_post, original_update = agent.post_interaction, agent.update
            calls, updates, checkpoints = [], [], []
            def post(**kwargs):
                calls.append(kwargs['timestep'])
                original_post(**kwargs)
            def update(**kwargs):
                updates.append(kwargs['timestep'])
                original_update(**kwargs)
            agent.post_interaction, agent.update = post, update
            optimizer_id = id(agent.optimizer)
            def callback(*, agent, completed_steps, timesteps):
                self.assertEqual(calls[-1], completed_steps - 1)
                self.assertEqual(updates[-1], completed_steps - 1)
                self.assertEqual(id(agent.optimizer), optimizer_id)
                assert_agent_checkpoint_finite(agent)
                self.assertTrue(finite_policy_parameter_changed(agent, initial))
                checkpoints.append(completed_steps)
                folder = root / str(completed_steps)
                folder.mkdir()
                checkpoint = folder / 'agent.pt'
                agent.save(str(checkpoint))
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                self.assertIn('optimizer', saved)
                self.assertIn('observation_preprocessor', saved)
                provenance = folder / 'provenance.json'
                harness._write_create(provenance, checkpoint_provenance(checkpoint=checkpoint,
                    config_sha256=harness.sha256(config_path), cfg=cfg, env=env, split='all'))
                args = SimpleNamespace(config=config_path, dataset=index, split='all', checkpoint=checkpoint,
                    provenance=provenance, manifest=folder/'eval.json', diagnostics=folder/'diag.json', device='cpu')
                if completed_steps == 12:
                    args.checkpoint = folder/'missing.pt'
                    harness.run_evaluator(args, timeout=30)  # must abort trainer before step13
                    self.fail('evaluator failure was skipped')
                before_rng, before_env = harness.rng_hashes(), harness.environment_snapshot(env)
                before_policy = {k: v.clone() for k, v in agent.policy.state_dict().items()}
                outputs = harness.run_evaluator(args, timeout=30)
                self.assertEqual(harness.rng_hashes(), before_rng)
                self.assertEqual(harness.environment_snapshot(env), before_env)
                self.assertTrue(all(torch.equal(v, agent.policy.state_dict()[k]) for k, v in before_policy.items()))
                diag = json.loads(args.diagnostics.read_text())
                self.assertEqual(diag['final_objective'], 6)  # NOT sum of prefixes (12)
                self.assertEqual(len(diag['final_ledgers']), 2)
                self.assertEqual([s['terminated'] for s in diag['steps']], [False, False, True] * 2)
                self.assertEqual(len(outputs['artifact_outputs']), 4)
                self.assertEqual({p['format'] for p in outputs['artifact_outputs']}, {'json', 'npz'})
                self.assertTrue(all(Path(p['path']).is_file() for p in outputs['artifact_outputs']))
                manifest = json.loads(args.manifest.read_text())
                self.assertTrue(manifest['preprocessors_frozen_during_evaluation'])
                self.assertEqual(manifest['executed_frame_count'], 6)
            harness.install_post_interaction_callback(agent, callback, boundaries=(4, 8, 12, 16))
            try:
                with self.assertRaisesRegex(RuntimeError, 'training stopped'):
                    trainer.train()
                self.assertEqual(calls, list(range(12)))
                self.assertEqual(updates, [3, 7, 11])
                self.assertEqual(checkpoints, [4, 8, 12])
            finally:
                env.close()

    def test_timeout_raises(self):
        from unittest.mock import patch
        import subprocess
        args = SimpleNamespace(config=Path('/tmp/cfg'), dataset=Path('/tmp/idx'), split='all',
            checkpoint=Path('/tmp/ckpt'), provenance=Path('/tmp/prov'), manifest=Path('/tmp/m'),
            diagnostics=Path('/tmp/d'), device='cpu')
        with patch.object(harness.subprocess, 'run', side_effect=subprocess.TimeoutExpired('test', 1)):
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                harness.run_evaluator(args, timeout=1)


if __name__ == "__main__":
    unittest.main()
