"""Three dimensionless geometric costs for an explicit R6 debugging experiment.

R6 is a noisy reference, not GT. Closed generated triangles define a collision
hypothesis, not observed signed authority. Native object moments exactly sum ALL
vertices; no geometric proxy, sampled object, or mesh repair is introduced.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from rewards.evaluator import FactorLedger, _project
from utils.generated_collision import GeneratedCollisionScene

CONTRACT = 'r6_three_geometric_rewards.v1'
GROUPS = ('reference_alignment', 'nonpenetration_contact', 'temporal_continuity')
DEFAULTS = {
    'contract': CONTRACT, 'trust_complete_generated_mesh': True,
    'reference_sigma_m': .005, 'pixel_sigma_px': 5., 'penetration_sigma_m': .0005,
    'penetration_epsilon_m': .0001, 'deepest_weight': .1,
    'contact_separation_margin_m': .002, 'contact_sigma_m': .005,
    'correction_velocity_sigma_m_s': .03, 'correction_acceleration_sigma_m_s2': 1.,
    'contact_logit_reference_weight': .05,
    'weights': {'reference_alignment': 1., 'nonpenetration_contact': 5., 'temporal_continuity': .25},
}


def squared_norm_mean(values):
    values = np.asarray(values, np.float64)
    return float(np.square(values).sum(axis=-1).mean()) if len(values) else 0.


@dataclass(frozen=True)
class NativeMoments:
    mean: np.ndarray
    second: np.ndarray
    count: int

    @classmethod
    def from_vertices(cls, vertices):
        v = np.asarray(vertices, np.float64)
        if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
            raise ValueError('complete finite native vertices required')
        return cls(v.mean(axis=0), v.T @ v / len(v), len(v))

    def displacement_squared(self, A, b):
        A, b = np.asarray(A, np.float64), np.asarray(b, np.float64)
        result = np.trace(A.T @ self.second @ A) + 2 * (self.mean @ A) @ b + b @ b
        return max(0., float(result))


def penetration_cost(depth, *, epsilon_m, sigma_m, deepest_weight):
    residual = np.maximum(np.asarray(depth, np.float64) - epsilon_m, 0.) / sigma_m
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError('fixed complete finite collision samples required')
    return float(np.square(residual).mean() + deepest_weight * residual.max() ** 2)


def continuity_samples(values, time_s, frame_idx, valid, velocity_sigma, acceleration_sigma, norm):
    """Return ADDITIVE prefix costs with FIXED full-timeline denominators."""
    n = len(time_s); speed = 0.; acceleration = 0.
    for i in range(1, len(values)):
        if not (valid[i] and valid[i-1] and frame_idx[i] == frame_idx[i-1] + 1):
            continue
        dt = time_s[i] - time_s[i-1]
        if dt <= 0: raise ValueError('strictly increasing timestamps required')
        v = (values[i] - values[i-1]) / dt
        speed += norm(v) / velocity_sigma ** 2 / max(n - 1, 1)
        if i >= 2 and valid[i-2] and frame_idx[i-1] == frame_idx[i-2] + 1:
            prevdt = time_s[i-1] - time_s[i-2]
            a = (v - (values[i-1] - values[i-2]) / prevdt) / ((dt + prevdt) * .5)
            acceleration += norm(a) / acceleration_sigma ** 2 / max(n - 2, 1)
    return speed + acceleration


class ThreeRewardEvaluator:
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg; self.options = cfg.get('three_rewards', {})
        if self.options != DEFAULTS:
            raise ValueError('explicit fixed v1 three-reward config required; no silent tuning')
        if cfg.get('reward_scale') != 1.:
            raise ValueError('three normalized reward groups require reward_scale=1')
        self.scenes = {}; self.moments = {}; self.zero_prefix = {}

    def scene_for(self, obj):
        key = obj['pose_mesh_sha256']
        if key not in self.scenes:
            self.scenes[key] = GeneratedCollisionScene(obj['vertices_canonical'], obj['faces'])
            self.moments[key] = NativeMoments.from_vertices(obj['vertices_canonical'])
        return self.scenes[key]

    def frame_values(self, state, i):
        cache = state.term_cache.setdefault(i, {})
        if 'three_rewards' in cache: return cache['three_rewards']
        e = state.episode; o = self.options; alignment = []; physical = []; contact_values = []
        details = {'hand_surface': [], 'hand_pixel': [], 'object_surface': [], 'relative_interaction': [], 'penetration': [], 'contact_separation': []}
        for side in e.layout.hand_slots:
            hand = e.hands[side]
            if not hand['valid'][i]: continue
            surface = squared_norm_mean(state.hand_vertices[side][i] - hand['vertices'][i]) / o['reference_sigma_m'] ** 2
            alignment.append(surface); details['hand_surface'].append(surface)
            uv, _ = _project(state.hand_joints[side][i], e.T_world_camera[i], e.intrinsics[i])
            ref_uv, _ = _project(hand['joints'][i], e.T_world_camera[i], e.intrinsics[i])
            vis = np.asarray(hand['joint_visibility'][i], np.float64)
            pixel = float((vis * np.square(uv - ref_uv).sum(axis=-1)).sum() / max(vis.sum(), 1.) / o['pixel_sigma_px'] ** 2)
            alignment.append(pixel); details['hand_pixel'].append(pixel)
            priors = [row.get('prior', .5) for (frame, row_side, _), row in e.contacts.items() if frame == int(e.frame_idx[i]) and row_side == side]
            p = np.clip(np.mean(priors) if priors else .5, 1e-4, 1-1e-4)
            index = e.layout.hand_slots.index(side)
            contact_values.append(float(np.square(state.contact_logits[i, index] - np.log(p/(1-p))).mean()))
        for slot in e.layout.object_slots:
            obj = e.objects[slot]
            if not obj['valid'][i]: continue
            scene = self.scene_for(obj); moments = self.moments[obj['pose_mesh_sha256']]
            R, t = state.object_rotation[slot][i], state.object_translation[slot][i]
            R0, t0 = np.asarray(obj['rotation'][i]), np.asarray(obj['translation'][i])
            displacement = moments.displacement_squared(R.T - R0.T, t - t0) / o['reference_sigma_m'] ** 2
            alignment.append(displacement); details['object_surface'].append(displacement)
            for side in e.layout.hand_slots:
                hand = e.hands[side]
                if not hand['valid'][i]: continue
                v = np.asarray(state.hand_vertices[side][i], np.float64)
                reference = np.asarray(hand['vertices'][i], np.float64)
                relative = squared_norm_mean((v - t) @ R - (reference - t0) @ R0) / o['reference_sigma_m'] ** 2
                alignment.append(relative); details['relative_interaction'].append(relative)
                faces = np.asarray(hand['faces']); faces = faces[i] if faces.ndim == 3 else faces
                depth = scene.query(v, faces, R, t)['depth_m']
                collision = penetration_cost(depth, epsilon_m=o['penetration_epsilon_m'], sigma_m=o['penetration_sigma_m'], deepest_weight=o['deepest_weight'])
                gap_cost = 0.
                row = e.contacts.get((int(e.frame_idx[i]), side, obj.get('entity_id')))
                if row is not None:
                    ids = np.asarray(row.get('vertex_ids', []), np.int64)
                    anchors = np.asarray(row.get('targets_object', []), np.float64).reshape(-1, 3)
                    if len(ids):
                        current_gap = np.linalg.norm(v[ids] - (anchors @ R.T + t), axis=1)
                        base_gap = np.linalg.norm(reference[ids] - (anchors @ R0.T + t0), axis=1)
                        residual = np.maximum(current_gap - base_gap - o['contact_separation_margin_m'], 0.) / o['contact_sigma_m']
                        gap_cost = float(row.get('prior', .5)) * float(np.square(residual).mean())
                physical.append(collision + gap_cost); details['penetration'].append(collision); details['contact_separation'].append(gap_cost)
        values = {'reference_alignment': float(np.mean(alignment)) if alignment else 0.,
                  'nonpenetration_contact': float(np.mean(physical)) if physical else 0.}
        values['reference_alignment'] += o['contact_logit_reference_weight'] * (float(np.mean(contact_values)) if contact_values else 0.)
        cache['three_rewards'] = (values, details)
        return cache['three_rewards']

    def temporal(self, state, end):
        e, o = state.episode, self.options; values = []
        for side in e.layout.hand_slots:
            h = e.hands[side]
            if not np.any(h['valid']): continue
            corrections = np.asarray(state.hand_vertices[side][:end], float) - np.asarray(h['vertices'][:end], float)
            values.append(continuity_samples(corrections, e.time_s, e.frame_idx, h['valid'],
                                            o['correction_velocity_sigma_m_s'], o['correction_acceleration_sigma_m_s2'], squared_norm_mean))
        for slot in e.layout.object_slots:
            obj = e.objects[slot]
            if not np.any(obj['valid']): continue
            self.scene_for(obj); moments = self.moments[obj['pose_mesh_sha256']]
            # Each correction is an affine map of the SAME native vertices.
            transforms = np.zeros((end, 4, 3))
            transforms[:, :3] = np.asarray(state.object_rotation[slot][:end]).transpose(0, 2, 1) - np.asarray(obj['rotation'][:end]).transpose(0, 2, 1)
            transforms[:, 3] = state.object_translation[slot][:end] - obj['translation'][:end]
            values.append(continuity_samples(transforms, e.time_s, e.frame_idx, obj['valid'],
                                            o['correction_velocity_sigma_m_s'], o['correction_acceleration_sigma_m_s2'],
                                            lambda x: moments.displacement_squared(x[:3], x[3])))
        return float(np.mean(values)) if values else 0.

    def objective(self, state, end_position):
        if not 0 <= end_position <= state.episode.frame_count: raise ValueError('prefix out of range')
        n = state.episode.frame_count; totals = {g: 0. for g in GROUPS}; subcosts = {}
        for i in range(end_position):
            row, details = self.frame_values(state, i)
            for g, value in row.items(): totals[g] += value / n
            for k, vals in details.items(): subcosts[k] = subcosts.get(k, 0.) + (float(np.mean(vals)) if vals else 0.) / n
        totals['temporal_continuity'] = self.temporal(state, end_position)
        terms = {g: totals[g] * self.options['weights'][g] for g in GROUPS}
        if not np.isfinite(list(terms.values())).all(): raise FloatingPointError('nonfinite three reward cost')
        return FactorLedger(end_position, terms, statistics={'unweighted_groups': totals, 'subcosts': subcosts},
                            activation={'contract': CONTRACT, 'reference_is_GT': False, 'signed_source_authority_promoted': False,
                                        'penetration_reduction': 'fixed_complete_samples_mean_plus_deepest', 'timeline_denominator': n},
                            profile=CONTRACT, reward_scale=1.)

    def zero_for(self, state):
        from envs.state import initial_state
        from envs.dynamics import decode_candidate
        key = state.episode.bundle_id
        if key not in self.zero_prefix:
            baseline = initial_state(state.episode)
            for i in range(baseline.episode.frame_count):
                baseline.position = i
                decode_candidate(baseline, np.zeros(baseline.episode.layout.size), self.cfg)
            costs = [self.objective(baseline, i).objective for i in range(baseline.episode.frame_count + 1)]
            self.zero_prefix[key] = np.asarray(costs)
        return self.zero_prefix[key]

    def transition(self, state, end_position):
        ledger = self.objective(state, end_position)
        baseline = self.zero_for(state)
        centered = ledger.objective - baseline[end_position]
        reward = -(centered - state.previous_objective)
        state.previous_objective = centered; state.ledger = ledger
        return float(reward), ledger
