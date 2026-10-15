"""Explicit A/R6, B/observation-motion, C/B+envelope numerical experiment.

Consumes memoized immutable R6 state plus original v2 observations. Does not
change default preprocessing, run models, load GT, train PPO or alter source data.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, vstack
from scipy.spatial.transform import Rotation

from preprocessing.inputs import load_inputs, sha, reject_evaluation_path
from preprocessing.geometry import TriangleMesh
from preprocessing.factors import hand_depths
from preprocessing.mask_contract import nonregression
from preprocessing.material_motion import (extract_tracks, rotation_edges, propagated_rotation,
    MaterialMotionProblem, fixed_track_score, robust_cost)
from preprocessing.surface_constraints import SurfaceFrame
from preprocessing.surface_pipeline import write_json, code_identity
from preprocessing.visible_surface import cloud_digest


def validate_rotations(R, tolerance=1e-6):
    R = np.asarray(R)
    error = float(np.max(np.abs(R @ R.transpose(0, 2, 1)-np.eye(3))))
    if not np.isfinite(R).all() or error > tolerance or np.any(np.linalg.det(R) < .9999):
        raise ValueError('Invalid rotation beyond source precision: '+str(error))
    return error


class JointSurfaceProblem(MaterialMotionProblem):
    def __init__(self, inputs, tracks, reference_R, reference_t, edges, surfaces, correspondences, box_weight):
        super().__init__(inputs, tracks, reference_R, reference_t, edges)
        self.surfaces, self.correspondences, self.box_weight = surfaces, correspondences, box_weight

    def residual(self, x):
        parts = [super().residual(x)]
        R, t, _ = self.unpack(x)
        parts.extend(f.residual(R[i], t[i], self.correspondences[i], self.box_weight) for i, f in enumerate(self.surfaces))
        return np.concatenate(parts)

    def sparsity(self):
        # Build the material block via its own method without virtual residual size.
        base_rows = len(MaterialMotionProblem.residual(self, self.initial()))
        base = MaterialMotionProblem.sparsity(self)
        base = base[:base_rows]
        R, t, _ = self.unpack(self.initial())
        lengths = [len(f.residual(R[i], t[i], self.correspondences[i], self.box_weight)) for i, f in enumerate(self.surfaces)]
        extra = lil_matrix((sum(lengths), self.size), dtype=np.int8); at = 0
        for i, length in enumerate(lengths):
            extra[at:at+length, self.columns(i)] = 1; at += length
        return vstack((base, extra.tocsr()), format='csr')

    def pack_pose(self, R, t):
        x = self.initial(); delta = np.empty((len(self.free), 6))
        delta[:, :3] = Rotation.from_matrix(R[self.free] @ self.R0[self.free].transpose(0, 2, 1)).as_rotvec()
        delta[:, 3:] = t[self.free]-self.t0[self.free]
        x[:self.point_start] = delta.ravel()
        x[self.point_start:] = self.initialize_points(R, t).ravel()
        return x


def score(inputs, tracks, edges, surfaces, R, t, reference_R, reference_t, box_weight):
    parts = []
    material = fixed_track_score(inputs, tracks, R, t)
    train = fixed_track_score(inputs, tracks.selected(False), R, t)
    heldout = fixed_track_score(inputs, tracks.selected(True), R, t)
    rows = []
    for i, surface in enumerate(surfaces):
        view = surface.view(R[i], t[i]); cor = surface.correspondences(R[i], t[i], view)
        row = surface.measure(R[i], t[i], cor, view); rows.append(row)
        parts.append(robust_cost(surface.residual(R[i], t[i], cor, box_weight)))
    # All arms/candidates share the same source populations and fixed raster domain.
    image = sum(2.*(1.-r['mask']['visible']['iou']) + .5*(1.-r['mask']['raw']['recall']) for r in rows)
    rotation_correction = R @ reference_R.transpose(0, 2, 1)
    smooth = robust_cost(Rotation.from_matrix(rotation_correction[1:] @ rotation_correction[:-1].transpose(0, 2, 1)).as_rotvec()/.4*.1)
    smooth += robust_cost(np.diff(t-reference_t, axis=0)/.04*.1)
    edge_cost = 0.
    if edges:
        a = np.array([e['a'] for e in edges]); b = np.array([e['b'] for e in edges]); er = np.array([e['R'] for e in edges])
        edge_cost = robust_cost(Rotation.from_matrix((R[b] @ R[a].transpose(0, 2, 1)) @ er.transpose(0, 2, 1)).as_rotvec()/.2*np.sqrt([e['weight'] for e in edges])[:, None]*.4)
    return {'total': material['cost'] + sum(parts) + image + smooth + edge_cost,
            'material': material, 'training_tracklets': train, 'heldout_tracklets': heldout,
            'surface_boundary_normal_box_cost': float(sum(parts)), 'fixed_mask_cost': float(image),
            'motion_regularization_cost': smooth, 'measured_rotation_edge_cost': edge_cost, 'frames': rows}


def constrained_candidate(inputs, surfaces, reference, R, t, reference_R, reference_t):
    """Preserve R6 visible support at final exported precision, not dynamic domains.

    This is an explicit constrained experiment; rejection is retained as evidence
    of observation-vs-generated-shape conflict, not relabeled measured success.
    """
    R, t = R.copy(), t.copy(); rejected = []
    for i, f in enumerate(surfaces):
        row = f.measure(R[i], t[i])
        valid, reason = nonregression(reference['frames'][i]['mask'], row['mask'])
        rc = inputs.cameras[i, :3, :3].T @ R[i]
        tc = inputs.cameras[i, :3, :3].T @ (t[i]-inputs.cameras[i, :3, 3])
        if np.min(f.mesh.vertices @ rc[2] + tc[2]) <= .001:
            valid, reason = False, 'full_mesh_nonpositive_depth'
        if not valid:
            rejected.append({'frame_idx': int(inputs.ids[i]), 'reason': reason})
            R[i], t[i] = reference_R[i], reference_t[i]
    return R, t, rejected


def refine_candidate(inputs, train, all_tracks, edges, surfaces, reference_R, reference_t, start_R, start_t,
                     box_weight, max_nfev, outer=3, reference=None):
    R, t = start_R.copy(), start_t.copy()
    if reference is None:
        reference = score(inputs, all_tracks, edges, surfaces, reference_R, reference_t, reference_R, reference_t, box_weight)
    raw_start = score(inputs, all_tracks, edges, surfaces, R, t, reference_R, reference_t, box_weight)
    R, t, rejected_start = constrained_candidate(inputs, surfaces, reference, R, t, reference_R, reference_t)
    current = score(inputs, all_tracks, edges, surfaces, R, t, reference_R, reference_t, box_weight)
    trace = []
    for iteration in range(outer):
        correspondences = [f.correspondences(R[i], t[i]) for i, f in enumerate(surfaces)]
        problem = JointSurfaceProblem(inputs, train, reference_R, reference_t, edges, surfaces, correspondences, box_weight)
        x0 = problem.pack_pose(R, t)
        fit = least_squares(problem.residual, x0, jac_sparsity=problem.sparsity(), loss='soft_l1',
                            max_nfev=max_nfev, x_scale='jac', ftol=1e-5, xtol=1e-5, gtol=1e-5)
        if not np.isfinite(fit.x).all():
            raise RuntimeError('Nonfinite joint surface state')
        accepted = False; trials = []
        for fraction in (1., .5, .25):
            nr, nt, _ = problem.unpack(x0 + fraction*(fit.x-x0))
            nr, nt, rejected = constrained_candidate(inputs, surfaces, reference, nr, nt, reference_R, reference_t)
            measured = score(inputs, all_tracks, edges, surfaces, nr, nt, reference_R, reference_t, box_weight)
            ok = measured['total'] < current['total'] - 1e-6
            trials.append({'fraction': fraction, 'score': measured['total'], 'accepted': bool(ok), 'rejected_frames': rejected})
            if ok:
                R, t, current = nr, nt, measured; accepted = True; break
        trace.append({'outer': iteration, 'nfev': fit.nfev, 'fixed_correspondence_cost': float(fit.cost), 'trials': trials})
        if not accepted:
            break
    return R, t, current, {'raw_start_score': raw_start, 'rejected_start': rejected_start, 'iterations': trace}


def baseline_load(source, baseline, inputs):
    reject_evaluation_path(baseline)
    path = Path(baseline).resolve(); inputs.receipt.add(path)
    manifest = json.loads(path.read_text())
    if manifest['schema'] != 'ego.annotation.monocular_coarse_scene.v1' or manifest.get('alignment_profile') != 'experimental-mask-repair':
        raise ValueError('A must be explicit frozen R6 coarse scene')
    if not manifest.get('prediction_side_only') or manifest.get('GT_or_CAD_inputs'):
        raise ValueError('Invalid baseline authority')
    if manifest['source_bundle_sha256'] != sha(source):
        raise ValueError('Baseline and raw source mismatch')
    arrays = inputs.receipt.add(manifest['arrays']['path'], manifest['arrays']['sha256'])
    with np.load(arrays, allow_pickle=False) as z:
        state = {k: np.array(z[k]) for k in z.files}
    if not np.array_equal(state['frame_idx'], inputs.ids):
        raise ValueError('Baseline timeline mismatch')
    for key, value in [('T_world_camera', inputs.cameras), ('intrinsics_source', inputs.source_K), ('faces', inputs.faces)]:
        np.testing.assert_array_equal(state[key], value)
    np.testing.assert_allclose(state['vertices_canonical'], inputs.vertices*float(state['uniform_object_scale']), rtol=0, atol=1e-10)
    for side, hand in inputs.hands.items():
        for key in ('vertices', 'joints', 'faces', 'valid'):
            np.testing.assert_array_equal(state[side+'_'+key], hand[key])
    validate_rotations(state['object_rotation'])
    return manifest, state


def run(source, baseline, output, max_nfev=24, width=480, raster_width=192):
    source = Path(source).resolve(); output = Path(output).resolve()
    if output.is_relative_to(source.parent) or output.is_relative_to(Path(baseline).resolve().parent):
        raise ValueError('Fresh output must be outside source/baseline')
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); code = code_identity()
    write_json(output/'started.json', {'source': str(source), 'baseline': str(baseline), 'code_sha256': code,
               'configuration': {'arms': ['A', 'B', 'C'], 'box_weight_B': 0., 'box_weight_C': .15,
                                 'width': width, 'raster_width': raster_width, 'max_nfev': max_nfev,
                                 'outer_iterations': 3, 'candidate_starts': ['A', 'observation_motion', 'half_correction']}})
    try:
        cv2.setNumThreads(1); inputs = load_inputs(source, width, read_owned_masks=True)
        manifest, state = baseline_load(source, baseline, inputs); digest = cloud_digest(inputs.clouds)
        for q in inputs.clouds:
            q.setflags(write=False)
        tracks = extract_tracks(inputs); train = tracks.selected(False)
        edges = rotation_edges(train, inputs)
        Ra, ta = state['object_rotation'], state['object_translation_world']
        Rinit, reached = propagated_rotation(Ra, inputs.anchor, edges)
        independent = MaterialMotionProblem(inputs, train, Rinit, ta, edges)
        Rmotion, tmotion, motion_report = independent.solve(max_nfev=max_nfev*2)
        write_json(output/'motion_observations.json', {'tracks': tracks.report, 'edges': [{**e, 'R': e['R'].tolist()} for e in edges],
                   'rotation_tree_reached_frames': sorted(reached), 'fit': motion_report,
                   'material_point_role': 'latent_motion_support_not_final_geometry'})
        np.savez_compressed(output/'motion_observations.npz', track=tracks.track, frame=tracks.frame, uv=tracks.uv,
            camera_points=tracks.camera_points, normals=tracks.normals, normal_quality=tracks.normal_quality,
            heldout=tracks.heldout, object_rotation=Rmotion, object_translation_world=tmotion)
        print(f'{len(inputs.ids)} frames; tracklets={tracks.count}, measured edges={len(edges)}; independent motion solved', flush=True)
        mesh = TriangleMesh(state['vertices_canonical'], state['faces']); hd = hand_depths(inputs, raster_width)
        surfaces = [SurfaceFrame(inputs, mesh, hd, i) for i in range(len(inputs.ids))]
        envelopes = [f.envelope for f in surfaces]
        np.savez_compressed(output/'surface_envelopes.npz', center_world=[e.center_world for e in envelopes], basis_world=[e.basis_world for e in envelopes],
            lower=[e.lower for e in envelopes], upper=[e.upper for e in envelopes], confidence=[e.confidence for e in envelopes], observed=[e.observed for e in envelopes])
        correction = Rotation.from_matrix(Rmotion @ Ra.transpose(0, 2, 1)).as_rotvec()
        midpoint_R = Rotation.from_rotvec(correction*.5).as_matrix() @ Ra
        starts = [('A', Ra, ta), ('observation_motion', Rmotion, tmotion), ('half_correction', midpoint_R, (ta+tmotion)*.5)]
        states = {'A': state}; scores = {}; evidence = {}
        scores['A'] = score(inputs, tracks, edges, surfaces, Ra, ta, Ra, ta, 0.)
        for name, box_weight in [('B', 0.), ('C', .15)]:
            best_R, best_t = Ra.copy(), ta.copy()
            best_score = score(inputs, tracks, edges, surfaces, Ra, ta, Ra, ta, box_weight)
            trials = []; selected = 'A_unchanged'; reference_score = best_score
            for start_name, start_R, start_t in starts:
                R, t, measured, report = refine_candidate(inputs, train, tracks, edges, surfaces, Ra, ta,
                    start_R, start_t, box_weight, max_nfev, reference=reference_score)
                trials.append({'start': start_name, 'score': measured['total'], 'report': report})
                if measured['total'] < best_score['total']-1e-6:
                    best_R, best_t, best_score = R, t, measured; selected = start_name
                print(f'{name}/{start_name} total={measured["total"]:.4f}, selected={selected}', flush=True)
            archive = {k: v.copy() for k, v in state.items()}
            archive['object_rotation'], archive['object_translation_world'] = best_R, best_t
            states[name], scores[name] = archive, best_score
            evidence[name] = {'selected_start': selected, 'box_weight': box_weight, 'candidates': trials}
        for name, archive in states.items():
            arm = output/name; arm.mkdir()
            np.savez_compressed(arm/'coarse_scene.npz', **archive)
            m = dict(manifest); m.update(alignment_profile='experimental_observation_surface_'+name,
                arrays={'path': str(arm/'coarse_scene.npz'), 'sha256': sha(arm/'coarse_scene.npz')},
                memoized_R6_manifest=str(Path(baseline).resolve()), memoized_R6_sha256=sha(baseline),
                motion_observations=str(output/'motion_observations.npz'), code_sha256=code,
                observation_motion_no_mesh_binding=name != 'A', source_reads=inputs.receipt.files,
                limitations=manifest['limitations'] + ['single depth source remains biased', 'tracklet redetection can repeat material identities',
                    'A-fixed canonical anchor, not absolute pose calibration', 'candidate spread is not covariance'],
                signed_geometry_ready=False, collision_authority_changed=False,
                configuration={'arm': name, 'box_weight': .15 if name=='C' else 0., 'fixed_shared_scale': True})
            write_json(arm/'coarse_scene.json', m); write_json(arm/'prediction_metrics.json', scores[name])
            write_json(arm/'completed.json', {'status': 'complete_experimental_coarse_scene', 'frames': len(inputs.ids),
                'source_unchanged': True, 'prediction_side_only': True, 'rendered': False})
        write_json(output/'candidate_evidence.json', evidence)
        # Full topology and original MANO/camera/K/scale are invariant across arms.
        for name, archive in states.items():
            for key, value in state.items():
                if key not in ('object_rotation', 'object_translation_world'):
                    np.testing.assert_array_equal(archive[key], value)
            validate_rotations(archive['object_rotation'])
        from preprocessing.surface_render import render_comparison
        render_comparison(inputs, states, output/'renders')
        inputs.receipt.check()
        if cloud_digest(inputs.clouds) != digest or code_identity() != code:
            raise RuntimeError('Source cloud or code changed during prediction')
        for name in states:
            done = json.loads((output/name/'completed.json').read_text()); done['rendered'] = True
            write_json(output/name/'completed.json', done)
        write_json(output/'completed.json', {'status': 'complete_three_arm_prediction', 'frames': len(inputs.ids),
                   'seconds': time.monotonic()-started, 'source_unchanged': True, 'rendered': True,
                   'source_cloud_digest': digest, 'arms': {k: {'npz_sha256': sha(output/k/'coarse_scene.npz'),
                      'manifest_sha256': sha(output/k/'coarse_scene.json')} for k in states}})
        print(json.dumps({'output': str(output), 'seconds': time.monotonic()-started}), flush=True)
        return output
    except Exception as exc:
        write_json(output/'failed.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        gc.collect()


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-observations', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-nfev', type=int, default=24)
    args = p.parse_args()
    if args.max_nfev < 1:
        p.error('Positive budget required')
    run(args.source_observations, args.baseline, args.output_dir, max_nfev=args.max_nfev)


if __name__ == '__main__':
    cli()
