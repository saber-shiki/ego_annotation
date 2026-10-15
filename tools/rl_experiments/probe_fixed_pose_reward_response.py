"""Frozen-pose reward direction tests; never fit a policy or export annotation.

This preserves the hand translation probes used for the PPO diagnostic.
Only a declared frame's metric MANO translation changes. The queried objective
is the unchanged full prefix ledger (NOT a per-frame cost). Zero archive and
original observations/native object triangles remain immutable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from data.coarse_training import CoarseTrainingDataset
from envs.reward_ablation_coarse_env import RewardAblationCoarseEnv
from envs.dynamics import decode_candidate
from rewards.ablation import grouped


def response(run):
    cfg = json.loads((run / 'matrix_contract.json').read_text())['base_config']
    dataset = CoarseTrainingDataset(run / 'index.json', split='all')
    env = RewardAblationCoarseEnv(dataset, cfg['environment'])
    env.reset(seed=0)
    state = env.state
    with np.load(run / 'zero/candidate.npz', allow_pickle=False) as archive:
        for side in ('left', 'right'):
            state.hand_vertices[side] = np.array(archive[side + '_vertices_world'])
            state.hand_joints[side] = np.array(archive[side + '_joints_world'])
    rows = []
    try:
        for frame in (0, 120):
            state.position = frame
            state.term_cache.clear()
            baseline = grouped(env.evaluator.objective(state, frame + 1).terms)
            if frame == 0:
                probes = [(side, name, direction, amount)
                          for side, name, direction in [('left', 'world_z', np.array([0., 0., 1.])),
                                                        ('right', 'world_y', np.array([0., 1., 0.]))]
                          for amount in (-.002, -.001, -.0005, 0., .0005, .001, .002)]
            else:
                R = state.episode.T_world_camera[frame, :3, :3]
                probes = [('left', 'camera_z', R @ np.array([0., 0., 1.]), amount)
                          for amount in (-.002, -.001, 0., .001, .002)]
                probes += [('left', 'camera_x', R @ np.array([1., 0., 0.]), amount)
                           for amount in (-.01, -.002, 0., .002, .01)]
            for side, name, direction, amount in probes:
                changed = state.copy()
                delta = np.zeros(118)
                block = changed.episode.layout.block(f'hand.{side}.translation')
                delta[block.slice] = direction * amount
                decode_candidate(changed, delta, cfg['environment'])
                after = grouped(env.evaluator.objective(changed, frame + 1).terms)
                rows.append({'frame': frame, 'side': side, 'direction': name, 'translation_mm': amount * 1000,
                             'group_cost_delta': {group: after[group] - baseline[group] for group in baseline},
                             'full_J_delta': sum(after.values()) - sum(baseline.values())})
        return {'scope': 'controlled frozenZero geometric-response test, no policy update, no new annotation output',
                'pose_sources': 'same frozen source and native geometry; zero replayed vertices; only declared translation variable perturbed',
                'rows': rows}
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(response(args.run), stream, indent=2, allow_nan=False)
        stream.write('\n')
