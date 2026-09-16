"""Frozen observation-oriented envelopes, ported from the September surface studies.

The plane normal faces the camera; its axes are descriptive, not object pose.
Direct bounds never consume candidate R/t/scale. Generated PCA dimensions are
one fixed, correlated size prior, not independent metric or collision evidence.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from factors.object.envelope import local_coordinates
from gates.object.geometry import validate_posterior
from gates.object.observation import retained_metric_alignment
from observations.common import pose_from_values
from observations.object.shape import mesh_arrays, record, metric_object
from utils.data import load, indexed, RKEY, TKEY
from utils.state import in_run, sha
from utils.geometry import validate_transforms

TAU = .003
COS_NORMAL = np.cos(np.deg2rad(25))

def local_normals(p, reference=None):
    ref = p if reference is None else reference
    ids = cKDTree(ref).query(p, k=min(24, len(ref)))[1]
    q = ref[ids]
    q = q - q.mean(axis=1, keepdims=True)
    eig, v = np.linalg.eigh(np.einsum('nki,nkj->nij', q, q))
    return v[:, :, 0], eig[:, 0] / np.maximum(eig.sum(axis=1), 1e-12)

def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def angle(a, b):
    return float(np.rad2deg(np.arccos(np.clip(abs(a @ b), 0, 1))))

def plane_labels(p, normals, planes):
    if not planes:
        return np.full(len(p), -1, dtype=int)
    ns = np.array([x['normal'] for x in planes]).T
    ds = np.array([x['offset'] for x in planes])
    dist = abs(p @ ns - ds)
    alignment = abs(normals @ ns)
    valid = (dist <= TAU) & (alignment >= COS_NORMAL)
    costs = np.where(valid, dist, np.inf)
    labels = costs.argmin(axis=1)
    labels[~valid.any(axis=1)] = -1
    return labels

def extract_planes(p, normals, rng):
    """RANSAC planes with local-normal support; no object class/pose initialization."""
    remaining = np.ones(len(p), bool)
    planes = []
    for _ in range(3):
        ids = np.flatnonzero(remaining)
        if len(ids) < 80:
            break
        picks = rng.choice(ids, 400, replace=True)
        candidates = normals[picks]
        offsets = np.einsum('ij,ij->i', candidates, p[picks])
        triples = rng.choice(ids, (500, 3), replace=True)
        n = normalize(np.cross(p[triples[:, 1]] - p[triples[:, 0]],
                               p[triples[:, 2]] - p[triples[:, 0]]))
        candidates = np.r_[candidates, n]
        offsets = np.r_[offsets, np.einsum('ij,ij->i', n, p[triples[:, 0]])]
        keep = np.linalg.norm(candidates, axis=1) > .9
        for old in planes:
            keep &= abs(candidates @ old['normal']) < np.cos(np.deg2rad(35))
        candidates, offsets = candidates[keep], offsets[keep]
        if not len(candidates):
            break
        dist = abs(p[ids] @ candidates.T - offsets)
        compatible = abs(normals[ids] @ candidates.T) >= COS_NORMAL
        scores = (np.maximum(0, 1 - (dist / TAU)**2) * compatible).sum(axis=0)
        best = int(scores.argmax())
        n, d = candidates[best], offsets[best]
        for __ in range(5):
            inlier = remaining & (abs(p @ n - d) <= TAU) & (abs(normals @ n) >= COS_NORMAL)
            if inlier.sum() < 80:
                break
            q = p[inlier]
            center = q.mean(axis=0)
            _, _, vt = np.linalg.svd(q - center, full_matrices=False)
            n = vt[-1]
            d = np.median(q @ n)
        inlier = remaining & (abs(p @ n - d) <= TAU) & (abs(normals @ n) >= COS_NORMAL)
        if inlier.sum() < 80:
            break
        # Refinement can drift back toward an already-extracted curved patch.
        # Do not promote that duplicate to a new independent surface.
        if any(angle(n, old['normal']) < 35 for old in planes):
            break
        patch = p[inlier] - p[inlier].mean(axis=0)
        spread = np.linalg.svd(patch, compute_uv=False) / np.sqrt(max(len(patch)-1, 1))
        if spread[1] < .005:
            break
        # Camera-facing sign is display convention, not canonical face identity.
        if n @ p[inlier].mean(axis=0) > 0:
            n, d = -n, -d
        planes.append(dict(normal=n, offset=float(d), train_support=int(inlier.sum())))
        remaining[inlier] = False
    return planes, plane_labels(p, normals, planes)

def descriptive_frame(p,raw,labels):
    if raw and (labels==0).sum()>=3:
        q=p[labels==0];n=np.asarray(raw[0]['normal']);n/=np.linalg.norm(n)
        c=np.median(q,axis=0);c+=(raw[0]['offset']-c@n)*n;method='dominant_free_angle_observed_plane'
    else:
        q=p;c=np.median(q,axis=0);_,_,v=np.linalg.svd(q-q.mean(axis=0),full_matrices=False);n=v[-1]
        if n@c>0:n=-n
        method='whole_visible_cloud_PCA_fallback_not_planar_evidence'
    _,sv,v=np.linalg.svd(q-q.mean(axis=0),full_matrices=False)
    y=v[0]-(v[0]@n)*n;y/=np.linalg.norm(y)
    if y@np.array([0.,-1.,0.])<0:y=-y
    x=np.cross(y,n);x/=np.linalg.norm(x);y=np.cross(n,x)
    T=np.eye(4);T[:3,:3]=np.column_stack([x,y,n]);T[:3,3]=c
    return T,dict(method=method,support_points=len(q),tangent_long_short_ratio=float(sv[0]/max(sv[1],1e-12)),normal_faces_camera=True)

def direct_bounds(points_world,T_world_surface,nominal_long_mid_short_m,margin_m=.008,size_slack=.10):
    """Build from observations and one fixed size prior only; no object R/t input."""
    q=local_coordinates(points_world,T_world_surface)
    lower,upper=np.quantile(q,[.01,.99],axis=0)
    long,mid,short=nominal_long_mid_short_m
    xy_nominal=np.array([mid,long])
    missing_each=np.maximum(0,xy_nominal-(upper-lower)[:2])
    padding_xy=margin_m+size_slack*xy_nominal
    lo=np.r_[lower[:2]-missing_each-padding_xy,min(lower[2],-mid*(1+size_slack))-margin_m]
    hi=np.r_[upper[:2]+missing_each+padding_xy,upper[2]+margin_m]
    return lo,hi,dict(observed_quantile_lower_surface_m=lower.tolist(),observed_quantile_upper_surface_m=upper.tolist(),
        missing_tangent_extent_each_side_m=missing_each.tolist(),tangent_padding_each_side_m=padding_xy.tolist(),
        hidden_thickness_prior_m=float(mid*(1+size_slack)),measurement_margin_m=margin_m)


def load_surface_sources(*, annotations, mesh_binding, refined_posterior, run_root, object_id):
    """Load original rgb_scene_visual_refined_binding_v1, not an edited candidate.

    All paths are current-run files. Pose rows must equal the explicit original
    refined posterior; observations are the retained P09 camera/world arrays.
    Source frame IDs need not be contiguous or ordered numerically.
    """
    root = Path(run_root).resolve()
    paths = {name: in_run(path, root) for name, path in (
        ('annotations', annotations), ('mesh_binding', mesh_binding),
        ('refined_posterior', refined_posterior))}
    ann, binding, posterior = (load(paths[k]) for k in paths)
    validate_posterior(posterior, object_id, root)
    if binding.get('schema') != 'rgb_scene_visual_refined_binding_v1' or binding.get('object_id') != object_id:
        raise ValueError('original selected-object visual binding required')
    if binding.get('refined_posterior') != record(paths['refined_posterior']):
        raise ValueError('binding/refined posterior path/hash mismatch')
    if binding.get('pose_rows') != posterior['pose_rows']:
        raise ValueError('binding pose rows differ from immutable posterior')
    if (binding.get('source_prior') != posterior['source_prior'] or
            binding.get('source_pose_rows') != posterior['source_pose_rows']):
        raise ValueError('binding original source prior changed')
    if posterior.get('evidence', {}).get('annotations_sha256') != sha(paths['annotations']):
        raise ValueError('posterior annotations changed')
    mesh_contract = binding['render_mesh_contract']
    mesh_path = in_run(mesh_contract['required_mesh_path'], root)
    receipt = binding['visual_mesh_receipt']
    if (sha(mesh_path) != mesh_contract['required_mesh_sha256'] or
            receipt.get('output_mesh_sha256') != sha(mesh_path) or
            Path(receipt['output_mesh_path']).resolve() != mesh_path):
        raise ValueError('original complete mesh binding changed')
    vertices, faces = mesh_arrays(mesh_path)
    frames, poses = indexed(ann['frames']), indexed(posterior['pose_rows'])
    if not frames or set(frames) != set(poses):
        raise ValueError('full annotation/posterior timeline required')
    ids = sorted(frames, key=lambda i: float(frames[i]['time_s']))
    times = np.array([frames[i]['time_s'] for i in ids], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('finite strictly increasing timestamps required')
    cameras, rotations, translations, points, diagnostics = [], [], [], [], []
    for idx in ids:
        frame = frames[idx]
        row = poses[idx]
        if 'time_s' in row and row['time_s'] != frame['time_s']:
            raise ValueError('source pose/annotation timestamp mismatch')
        camera = frame['camera']
        T = validate_transforms([camera.get('T_world_camera_metric', camera.get('T_world_camera'))])[0]
        pose = pose_from_values(row.get(RKEY), row.get(TKEY))
        if pose is None:
            raise ValueError('invalid source pose')
        world, reason, diagnostic = retained_metric_alignment(metric_object(frame, object_id), frame)
        if diagnostic.get('status') in {'invalid_world_points', 'invalid_camera_points',
                'nonpositive_camera_depth', 'invalid_camera_transform',
                'camera_world_count_mismatch', 'camera_world_mismatch'}:
            raise ValueError(f'invalid retained surface geometry at frame {idx}: {reason}')
        # A rejected/missing local measurement remains a timeline variable.
        pc = np.empty((0, 3)) if world is None else (world - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T
        if np.any(pc[:, 2] <= 0):
            raise ValueError('retained surface has nonpositive camera-z')
        cameras.append(T); rotations.append(pose[0]); translations.append(pose[1])
        points.append(pc); diagnostics.append({'reason': reason, 'validation': diagnostic})
    refs = {name: record(path) for name, path in paths.items()}
    refs['mesh'] = record(mesh_path)
    refs['source_prior'] = copy.deepcopy(posterior['source_prior'])
    return dict(root=root, object_id=object_id, binding=binding, posterior=posterior,
                frames=[frames[i] for i in ids], frame_idx=ids, times=times,
                vertices=vertices, faces=faces, T=np.array(cameras), R0=np.array(rotations),
                t0=np.array(translations), points_camera=points,
                observation_diagnostics=diagnostics, inputs=refs)


def build_envelopes(*, annotations, mesh_binding, refined_posterior, run_root, object_id, output):
    """Create observed_surface_envelope_v1 at a new output JSON path."""
    from gates.object.envelope import validate_envelopes
    source = load_surface_sources(annotations=annotations, mesh_binding=mesh_binding,
        refined_posterior=refined_posterior, run_root=run_root, object_id=object_id)
    output = in_run(output, source['root'], must_exist=False)
    if output.exists():
        raise ValueError('create-only envelope output required')
    v = source['vertices']
    q = v - np.median(v, axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    sizes = np.sort(np.ptp(q @ vt.T, axis=0))[::-1]
    rows, direct = [], {}
    for i, idx in enumerate(source['frame_idx']):
        p, T = source['points_camera'][i], source['T'][i]
        row = dict(frame_idx=idx, time_s=float(source['times'][i]), object_id=object_id,
            observation=source['observation_diagnostics'][i], support_points=len(p),
            per_frame_candidate_pose_used_for_construction=False,
            generated_shape_size_prior=True, calibrated_confidence_region=False,
            collision_authority=False)
        if len(p):
            rng = np.random.default_rng(714 + i)
            normals, _ = local_normals(p)
            raw, labels = extract_planes(p, normals, rng)
            Tcs, descriptor = descriptive_frame(p, raw, labels)
            S = T @ Tcs
            lo, hi, construction = direct_bounds(p @ T[:3, :3].T + T[:3, 3], S, sizes)
            row.update(source_kind='direct_observation', optimization_weight=1.,
                T_world_surface=S.tolist(), lower_surface_m=lo.tolist(), upper_surface_m=hi.tolist(),
                surface_frame=descriptor, construction=construction,
                uncertainty='predicted_metric_depth_and_fixed_generated_size_correlated_not_calibrated')
            direct[i] = row
        rows.append(row)
    for i, row in enumerate(rows):
        if i in direct:
            continue
        if not direct:
            row.update(source_kind='unresolved_no_observation', optimization_weight=0.,
                T_world_surface=None, lower_surface_m=None, upper_surface_m=None,
                uncertainty='no_direct_surface_frame_available')
            continue
        donor = min(direct, key=lambda j: (abs(source['times'][i] - source['times'][j]), j))
        Ti, Td = np.eye(4), np.eye(4)
        Ti[:3, :3], Ti[:3, 3] = source['R0'][i], source['t0'][i]
        Td[:3, :3], Td[:3, 3] = source['R0'][donor], source['t0'][donor]
        S = Ti @ np.linalg.inv(Td) @ np.asarray(direct[donor]['T_world_surface'])
        # Historical 3 mm/frame expressed at the explicit reference rate 30 Hz.
        gap_s = abs(float(source['times'][i] - source['times'][donor]))
        pad = .006 + .090 * gap_s
        row.update(source_kind='transported_temporal_prior', optimization_weight=0.,
            T_world_surface=S.tolist(),
            lower_surface_m=(np.array(direct[donor]['lower_surface_m']) - pad).tolist(),
            upper_surface_m=(np.array(direct[donor]['upper_surface_m']) + pad).tolist(),
            donor_frame_idx=source['frame_idx'][donor], gap_s=gap_s,
            source_pose_used_for_transport=True, extra_padding_m=pad,
            uncertainty='source_pose_transported_display_hypothesis_not_independent_measurement')
    result = dict(schema='observed_surface_envelope_v1', object_id=object_id,
        run_root=str(source['root']), inputs=source['inputs'], frames=rows,
        fixed_shape_pca_long_mid_short_m=sizes.tolist(), frozen_before_optimization=True,
        generated_shape_size_prior=True, calibrated_confidence_region=False,
        collision_authority=False, settings=dict(observed_quantiles=[.01, .99],
        measurement_margin_m=.008, size_relative_slack=.10, missing_padding_m_per_s=.090))
    validate_envelopes(result, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result
