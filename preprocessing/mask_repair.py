"""Opt-in full-mesh mask repair; the default producer remains unchanged.

Reproduce the existing fixed-UniDepth producer inside the fresh run as the seed.
Then refine each frame's rotation and translation, retaining its ONE shared scale,
source depth and metric MANO. All output proposals are checked against frozen seed
mask evidence and fixed source surface samples. No previous output/GT is an input.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import traceback

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from preprocessing.factors import hand_depths, subset
from preprocessing.geometry import TriangleMesh, camera_pose, world_pose, project, resized_intrinsics
from preprocessing.inputs import load_inputs, sha
from preprocessing.mask_contract import FixedMaskDomain, boundary, nonregression
from preprocessing.surface_pipeline import run_visible_surface, write_json, code_identity
from preprocessing.visible_surface import cloud_digest


def mask_merit(evidence):
    b = evidence['boundary']
    boundary_score = 0. if b['recall'] is None else b['recall'] + b['precision']
    return evidence['visible']['iou'] + .15 * boundary_score + .1 * evidence['raw']['recall']


class FrameRepair:
    def __init__(self, inputs, mesh, domain, i, R, t, max_points=160):
        self.inputs, self.mesh, self.domain, self.i = inputs, mesh, domain, i
        self.K = inputs.K[i]
        self.wh = domain.target.shape[::-1]
        self.raster_K = resized_intrinsics(self.K, inputs.wh, self.wh)
        self.C = inputs.cameras[i]
        self.world_R0, self.world_t0 = np.asarray(R).copy(), np.asarray(t).copy()
        self.R0, self.t0 = camera_pose(R, t, self.C)
        # The same source samples score every pose. Candidate visibility never removes samples.
        self.q = subset(np.asarray(inputs.clouds[i], float), max_points).copy()
        self.q.setflags(write=False)
        self.uv = project(self.q, self.K) if len(self.q) else np.empty((0, 2))
        self.reference = self.measure(self.R0, self.t0)
        self.neighbor_corrections = []

    def temporal_residual(self, R, t):
        """SO(3)/world-translation correction edges, NOT absolute motion limits."""
        correction_R = (self.C[:3, :3] @ R) @ self.world_R0.T
        correction_t = self.C[:3, :3] @ (t - self.t0)
        parts = []
        for neighbor_R, neighbor_t, weight in self.neighbor_corrections:
            parts += [np.sqrt(.02 * weight) * Rotation.from_matrix(correction_R @ neighbor_R.T).as_rotvec() / .2,
                      np.sqrt(.02 * weight) * (correction_t - neighbor_t) / .01]
        return np.concatenate(parts) if parts else np.empty(0)

    def temporal_cost(self, R, t):
        residual = self.temporal_residual(R, t)
        return .5 * float(residual @ residual)

    def correspondences(self, R, t, view):
        X, _, hit = self.mesh.query(self.uv, self.K, R, t)
        if np.any(~hit):
            points = view[0][view[2]]
            if not len(points):
                raise ValueError('No visible triangles')
            _, idx = cKDTree(points @ R.T + t).query(self.q[~hit], workers=1)
            X[~hit] = points[idx]
        return X, hit

    def measure(self, R, t):
        if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min(self.mesh.vertices @ R[2] + t[2]) <= .001:
            raise ValueError('Nonprojectable full mesh')
        view = self.mesh.raster(self.raster_K, R, t, self.wh)
        mask = self.domain.score(view[2])
        X, hit = self.correspondences(R, t, view)
        distances = np.linalg.norm(X @ R.T + t - self.q, axis=1) if len(X) else np.empty(0)
        surface = {'samples': len(X), 'mean_m': float(np.mean(distances)) if len(X) else None,
                   'p90_m': float(np.percentile(distances, 90)) if len(X) else None,
                   'first_hit_coverage': float(np.mean(hit)) if len(hit) else None}
        return {'mask': mask, 'surface': surface}

    def admissible(self, current, candidate):
        for name, reference in [('frozen_seed', self.reference), ('current', current)]:
            ok, reason = nonregression(reference['mask'], candidate['mask'])
            if not ok:
                return False, name + ':' + reason
        # No cumulative IoU loss at all; other small raster tolerances remain frozen.
        if candidate['mask']['visible']['iou'] < self.reference['mask']['visible']['iou'] - 1e-12:
            return False, 'seed_visible_iou_regressed'
        b, a = self.reference['surface'], candidate['surface']
        if b['samples'] != a['samples']:
            return False, 'source_surface_population_changed'
        if b['samples']:
            if a['mean_m'] > b['mean_m'] + .003 or a['p90_m'] > b['p90_m'] + .005:
                return False, 'source_surface_regressed'
            if a['first_hit_coverage'] < b['first_hit_coverage'] - .03:
                return False, 'source_surface_coverage_regressed'
        return True, 'mask_and_surface_consistent'

    def contour_pairs(self, view, count=96):
        X, _, mask = view
        my, mx = np.nonzero(boundary(mask) & self.domain.boundary_domain)
        ty, tx = np.nonzero(boundary(self.domain.target) & self.domain.boundary_domain)
        if min(len(mx), len(tx)) < 6:
            return np.empty((0, 3)), np.empty((0, 2))
        mxy, txy = np.c_[mx, my], np.c_[tx, ty]
        mi = subset(np.arange(len(mx)), count); ti = subset(np.arange(len(tx)), count)
        _, j = cKDTree(txy).query(mxy[mi], workers=1)
        _, k = cKDTree(mxy).query(txy[ti], workers=1)
        points = np.r_[X[my[mi], mx[mi]], X[my[k], mx[k]]]
        uv = np.r_[txy[j], txy[ti]].astype(float)
        return points, uv

    def solve(self, outer_iterations=5, max_nfev=20, initial_pose=None):
        R, t = (self.R0.copy(), self.t0.copy()) if initial_pose is None else camera_pose(*initial_pose, self.C)
        current = self.reference if initial_pose is None else self.measure(R, t)
        trace = []
        if not self.reference['mask']['visible']['available']:
            return self.world_R0.copy(), self.world_t0.copy(), {'before': self.reference, 'after': current, 'iterations': [], 'reason': 'no_visible_support'}
        for outer in range(outer_iterations):
            view = self.mesh.raster(self.raster_K, R, t, self.wh)
            X, uv = self.contour_pairs(view)
            if len(X) < 6:
                break
            Y, _ = self.correspondences(R, t, view)
            normals = self.mesh.normals(Y) @ R.T if len(Y) else np.empty((0, 3))
            base_R, base_t = R.copy(), t.copy()
            # Missing metric depth is NOT replaced by silhouette-inferred depth.
            free_z = bool(len(self.q))

            def pose(x):
                r = Rotation.from_rotvec(x[:3]).as_matrix() @ base_R
                tt = base_t + np.r_[x[3:5], x[5] if free_z else 0.]
                return r, tt

            def residual(x):
                r, tt = pose(x)
                # Pixel units at the fixed scoring resolution, not candidate-selected pixels.
                contour = (project(X @ r.T + tt, self.raster_K) - uv) / 1.5
                rotation_delta = Rotation.from_matrix(r @ self.R0.T).as_rotvec()
                parts = [contour.ravel()]
                if len(Y):
                    delta = Y @ r.T + tt - self.q
                    # First-surface point-to-plane term, with fixed raw source points.
                    plane = np.sum(delta * normals, axis=1)
                    parts.append(plane / .006 * np.sqrt(len(X) / len(Y)))
                if self.neighbor_corrections:
                    # Normalize observation populations so sparse priors cannot be
                    # overwhelmed merely by requesting more contour samples.
                    parts = [p / np.sqrt(len(X)) for p in parts]
                    parts += [.1 * rotation_delta / .35,
                              .1 * (tt - self.t0) / np.array([.04, .04, .008]),
                              self.temporal_residual(r, tt)]
                else:
                    parts += [rotation_delta / .35, (tt - self.t0) / np.array([.04, .04, .008])]
                if not free_z:
                    parts.append(np.array([x[5] / .001]))
                return np.concatenate(parts)

            def loss(z):
                # Temporal edges are Gaussian: a rotation flip cannot disappear as
                # a saturated robust-prior outlier. Image/surface rows remain robust.
                n = 2 * len(X) + len(Y)
                rho = np.vstack((z.copy(), np.ones_like(z), np.zeros_like(z)))
                a = 1. + z[:n]
                rho[0, :n] = 2. * (np.sqrt(a) - 1.)
                rho[1, :n] = a ** -.5; rho[2, :n] = -.5 * a ** -1.5
                return rho
            fit = least_squares(residual, np.zeros(6), loss=loss if self.neighbor_corrections else 'soft_l1', max_nfev=max_nfev,
                                x_scale=np.r_[np.full(3, .1), np.full(3, .01)])
            accepted = False
            trials = []
            if not np.isfinite(fit.x).all():
                break
            for fraction in (1., .5, .25, .125):
                cr, ct = pose(fit.x * fraction)
                try:
                    candidate = self.measure(cr, ct)
                    ok, reason = self.admissible(current, candidate)
                except ValueError as exc:
                    trials.append({'fraction': fraction, 'accepted': False, 'reason': str(exc)})
                    continue
                gain = mask_merit(candidate['mask']) - mask_merit(current['mask'])
                motion_increase = self.temporal_cost(cr, ct) - self.temporal_cost(R, t)
                joint_gain = gain - motion_increase
                ok = ok and joint_gain > 1e-6
                trials.append({'fraction': fraction, 'accepted': bool(ok),
                               'reason': reason if joint_gain > 1e-6 else 'no_joint_image_temporal_gain',
                               'mask_merit_gain': gain, 'temporal_cost_increase': motion_increase, 'joint_gain': joint_gain})
                if ok:
                    R, t, current = cr, ct, candidate
                    accepted = True
                    break
            trace.append({'outer': outer, 'nfev': fit.nfev, 'trials': trials})
            if not accepted:
                break
        accepted_updates = sum(any(x['accepted'] for x in row['trials']) for row in trace)
        exported_R, exported_t, current, export_reason = self.export_checked(R, t, accepted_updates, initial_pose)
        if export_reason != 'exported_pose_verified':
            accepted_updates = 0
        R, t = camera_pose(exported_R, exported_t, self.C)
        report = {'before': self.reference, 'after': current, 'iterations': trace,
                  'export_verification': export_reason,
                  'accepted_updates': accepted_updates,
                  'rotation_correction_deg': float(np.degrees(Rotation.from_matrix(R @ self.R0.T).magnitude())),
                  'translation_correction_m': float(np.linalg.norm(t - self.t0)), 'missing_depth_frozen': not bool(len(self.q))}
        if current is not self.reference:
            ok, reason = self.admissible(self.reference, current)
            if not ok:
                raise RuntimeError('Final mask repair violated contract: ' + reason)
        return exported_R, exported_t, report

    def export_checked(self, R, t, accepted_updates, initial_pose=None):
        """Check the exact world pose that will be saved, not its pre-export camera pose.

        A camera/world round trip can flip a grazing float32 triangle raster pixel.
        Unchanged frames preserve the exact seed arrays; failed final checks return
        that SAME-frame seed, never a copied neighboring pose.
        """
        if not accepted_updates:
            if initial_pose is not None:
                wr, wt = (v.copy() for v in initial_pose)
                return wr, wt, self.measure(*camera_pose(wr, wt, self.C)), 'initial_pose_unchanged'
            return self.world_R0.copy(), self.world_t0.copy(), self.reference, 'seed_unchanged'
        wr, wt = world_pose(R, t, self.C)
        try:
            exported = self.measure(*camera_pose(wr, wt, self.C))
            ok, reason = self.admissible(self.reference, exported)
        except ValueError as exc:
            ok, reason = False, str(exc)
        if ok:
            return wr, wt, exported, 'exported_pose_verified'
        if initial_pose is not None:
            wr, wt = (v.copy() for v in initial_pose)
            return wr, wt, self.measure(*camera_pose(wr, wt, self.C)), 'export_rejected:' + reason
        return self.world_R0.copy(), self.world_t0.copy(), self.reference, 'export_rejected:' + reason


def run(source, output, *, width=480, registration_width=192, max_nfev=24, render_width=320):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source.parent or output.is_relative_to(source.parent):
        raise ValueError('Derived output must be outside source')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); code = code_identity()
    config = {'profile': 'experimental-mask-repair', 'seed_profile': 'visible-surface',
              'mask_acceptance_contract': 'frozen_seed_raw_recall_visible_owned_boundary_and_source_surface',
              'scale_policy': 'one_frozen_seed_uniform_scale', 'rotation_refinement': True,
              'no_prediction_feedback_into_tracker': True, 'registration_width': registration_width,
              'scope': 'output_refinement_only; seed_proposals_are_not_mask_constrained',
              'temporal_correction_edges': 'SO3_and_world_translation_Gaussian', 'temporal_sweeps': 2}
    write_json(output / 'started.json', {'source': str(source), 'configuration': config, 'code_sha256': code})
    try:
        # Recompute the seed from source observations; never read evaluator/baseline files.
        run_visible_surface(source, output / 'seed', width=width, registration_width=registration_width,
                            max_nfev=max_nfev, render=False)
        inputs = load_inputs(source, width, read_owned_masks=True)
        digest = cloud_digest(inputs.clouds)
        for q in inputs.clouds:
            q.setflags(write=False)
        hd = hand_depths(inputs, registration_width)
        seed = dict(np.load(output / 'seed' / 'coarse_scene.npz', allow_pickle=False))
        archive = {k: v.copy() for k, v in seed.items()}
        mesh = TriangleMesh(seed['vertices_canonical'], seed['faces'])
        repairs = [FrameRepair(inputs, mesh, FixedMaskDomain.from_inputs(inputs, hd, i), i,
                    seed['object_rotation'][i], seed['object_translation_world'][i]) for i in range(len(inputs.ids))]
        reports = [None] * len(inputs.ids)
        dt_ref = float(np.median(np.diff(inputs.times)))
        for sweep in range(config['temporal_sweeps']):
            order = range(len(inputs.ids)) if sweep % 2 == 0 else range(len(inputs.ids)-1, -1, -1)
            for i in order:
                repair = repairs[i]
                neighbors = [j for j in (i-1, i+1) if 0 <= j < len(inputs.ids)]
                repair.neighbor_corrections = [(
                    archive['object_rotation'][j] @ seed['object_rotation'][j].T,
                    archive['object_translation_world'][j] - seed['object_translation_world'][j],
                    dt_ref / abs(float(inputs.times[i] - inputs.times[j]))) for j in neighbors]
                initial = (archive['object_rotation'][i].copy(), archive['object_translation_world'][i].copy())
                R, t, report = repair.solve(max_nfev=max_nfev, initial_pose=initial)
                archive['object_rotation'][i] = R; archive['object_translation_world'][i] = t
                if reports[i] is not None:
                    report['accepted_updates'] = report.get('accepted_updates', 0) + reports[i].get('accepted_updates', 0)
                    report['iterations'] = reports[i]['iterations'] + report['iterations']
                reports[i] = {'frame_idx': int(inputs.ids[i]), **report}
            print(f'mask correction sweep {sweep+1}/{config["temporal_sweeps"]} complete', flush=True)
        write_json(output / 'mask_repair.json', {'configuration': config, 'frames': reports})
        # Retain real bootstrap for stage-separated evaluation, with source mesh semantics.
        bootstrap = dict(np.load(output / 'seed' / 'bootstrap.npz', allow_pickle=False))
        np.savez_compressed(output / 'bootstrap.npz', **bootstrap)
        np.savez_compressed(output / 'coarse_scene.npz', **archive)
        manifest = json.loads((output / 'seed' / 'coarse_scene.json').read_text())
        manifest.update(alignment_profile=config['profile'], configuration=config,
                        rotation_source='same_frame_full_triangle_visible_boundary_and_fixed_UniDepth_refinement',
                        rotation_changed_by_registration=True, code_sha256=code,
                        translation_refit_independently=True,
                        translation_parameterization='per_frame_R_t_correction_with_missing_metric_camera_depth_frozen',
                        arrays={'path': str(output / 'coarse_scene.npz'), 'sha256': sha(output / 'coarse_scene.npz')},
                        seed_manifest=str(output / 'seed' / 'coarse_scene.json'),
                        limitations=[v for v in manifest['limitations'] if v != 'registration holds orientation hypotheses fixed'] + ['silhouette does not resolve 3D rotational ambiguity',
                            'mask contract applies to output refinement only; seed solver remains unchanged',
                            'occlusion allowance limited to fixed source MANO/owned evidence'])
        write_json(output / 'coarse_scene.json', manifest)
        from preprocessing.visible_surface import VisibleSurfaceRegistration
        measurements = {}
        for name, state in [('before', seed), ('after', archive)]:
            fit = VisibleSurfaceRegistration(inputs, state['object_rotation'], state['object_translation_world'], hd,
                range(len(inputs.ids)), fit_scale=False, initial_scale=float(state['uniform_object_scale']), max_points=512)
            measurements[name] = fit.measure(float(state['uniform_object_scale']), fit.t0)
        write_json(output / 'visible_surface_metrics.json', {'reference': 'fixed_UniDepth', **measurements})
        from preprocessing.render import render_scene
        # The standard renderer's before panel remains the actual RGB bootstrap,
        # with the source mesh, avoiding accidental mixed-scale before projections.
        render_scene(inputs, archive, output / 'renders', render_width)
        inputs.receipt.check()
        if cloud_digest(inputs.clouds) != digest or code_identity() != code:
            raise RuntimeError('Source depth or snapshot changed')
        for key in ('vertices_canonical', 'faces', 'uniform_object_scale'):
            np.testing.assert_array_equal(archive[key], seed[key])
        write_json(output / 'completed.json', {'status': 'experimental_mask_repair', 'frames': len(inputs.ids),
                   'source_unchanged': True, 'rendered': True, 'seconds': time.monotonic() - started,
                   'accepted_frame_refinements': sum(r.get('accepted_updates', 0) > 0 for r in reports)})
        print(json.dumps({'output': str(output), 'accepted_frames': sum(r.get('accepted_updates', 0) > 0 for r in reports),
                          'seconds': time.monotonic() - started}), flush=True)
        return manifest
    except Exception as exc:
        write_json(output / 'failed.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-observations', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--registration-width', type=int, default=192)
    p.add_argument('--max-nfev', type=int, default=24)
    a = p.parse_args()
    if a.registration_width < 48 or a.max_nfev < 1:
        p.error('Invalid raster or solver budget')
    run(a.source_observations, a.output_dir, registration_width=a.registration_width, max_nfev=a.max_nfev)


if __name__ == '__main__':
    cli()
