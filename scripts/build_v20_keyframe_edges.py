#!/usr/bin/env python3
"""Build observed-only keyframe SE(3) edges with point-to-plane ICP and covariance.

All point clouds come from corrected P09 visible first hits.  P14 poses are only
used to place each cloud in the common initial canonical basin; generated
completion geometry is never loaded.  Edges are measurements for a later global
solver, not annotation state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "fit_to_object_owned_rgb_calibrated_pnp",
}


@dataclass
class Cloud:
    frame_idx: int
    rotation0: np.ndarray
    translation0: np.ndarray
    world: np.ndarray
    canonical0: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_pose(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64)[None, :]


def inverse_pose(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - np.asarray(t, dtype=np.float64)[None, :]) @ np.asarray(R, dtype=np.float64)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {"count": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)), "max": float(np.max(a)), "mean": float(np.mean(a))}


def deterministic_sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        return np.asarray(points, dtype=np.float64).copy()
    rng = np.random.default_rng(int(seed))
    return np.asarray(points, dtype=np.float64)[rng.choice(len(points), int(count), replace=False)]


def rigid_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(source) != len(target) or len(source) < 3:
        raise RuntimeError("rigid Umeyama requires matched arrays with >=3 points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center) / len(source)
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))


def estimate_normals(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    kk = min(max(3, int(k)), max(3, len(points)))
    _d, indices = cKDTree(points).query(points, k=kk, workers=-1)
    neighbors = points[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(1, kk - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normals = eigenvectors[:, :, 0]
    planarity = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1.0e-12)
    return normals, planarity


def mutual_correspondences(source: np.ndarray, target: np.ndarray, R: np.ndarray, t: np.ndarray, trim: float, max_distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = apply_pose(source, R, t)
    target_tree = cKDTree(target)
    distances, target_idx = target_tree.query(transformed, k=1, workers=-1)
    source_tree = cKDTree(transformed)
    reverse_idx = source_tree.query(target, k=1, workers=-1)[1]
    source_idx = np.arange(len(source), dtype=np.int64)
    mutual = reverse_idx[target_idx] == source_idx
    pool = np.flatnonzero(mutual)
    if len(pool) < 30:
        pool = np.argsort(distances)[: min(len(distances), max(30, int(len(source) * 0.2)))]
    if len(pool) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    cap = min(float(max_distance), float(np.percentile(distances[pool], 100.0 * trim)))
    keep = pool[distances[pool] <= cap]
    if len(keep) < 30:
        keep = np.argsort(distances)[: min(len(distances), max(30, int(len(source) * 0.2)))]
    return keep.astype(np.int64), target_idx[keep].astype(np.int64), distances[keep].astype(np.float64)


def point_to_plane_registration(source: np.ndarray, target: np.ndarray, *, normals: np.ndarray, planarity: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    trace=[]
    H_last=np.eye(6,dtype=np.float64)
    residual_last=np.empty(0,dtype=np.float64)
    match_last=np.empty((0,),dtype=np.int64)
    target_last=np.empty((0,),dtype=np.int64)
    for iteration in range(int(args.icp_iterations)):
        source_idx, target_idx, distances = mutual_correspondences(source,target,R,t,float(args.trim_fraction),float(args.max_correspondence_m))
        if len(source_idx)<int(args.min_matches): break
        p=apply_pose(source[source_idx],R,t);q=target[target_idx];n=normals[target_idx]
        residual=np.einsum('ij,ij->i',n,p-q)
        absr=np.abs(residual);scale=max(float(np.median(absr))*1.4826,float(args.min_noise_m))
        huber=float(args.huber_delta_m)*scale
        weights=np.ones(len(residual),dtype=np.float64);bad=absr>huber;weights[bad]=huber/np.maximum(absr[bad],1.0e-12)
        # Downweight locally ill-conditioned normal estimates, but retain a
        # point-to-point stabilizer for planar/low-normal-diversity patches.
        normal_weight=np.clip(planarity[target_idx]/max(float(args.max_planarity_ratio),1.0e-6),0.1,1.0)
        weights*=normal_weight
        J=np.zeros((len(p),6),dtype=np.float64)
        for row,(point,normal) in enumerate(zip(p,n)):
            J[row,:3]=normal @ (-skew(point))
            J[row,3:]=normal
        H=J.T@(weights[:,None]*J)+float(args.damping)*np.eye(6)
        rhs=J.T@(weights*residual)
        try: step=np.linalg.solve(H,-rhs)
        except np.linalg.LinAlgError: step=np.linalg.lstsq(H,-rhs,rcond=None)[0]
        step[:3]=np.clip(step[:3],-float(args.max_rotation_step_rad),float(args.max_rotation_step_rad))
        step[3:]=np.clip(step[3:],-float(args.max_translation_step_m),float(args.max_translation_step_m))
        dR=Rotation.from_rotvec(step[:3]).as_matrix();R=dR@R;t=dR@t+step[3:]
        H_last=H;residual_last=residual;match_last=source_idx;target_last=target_idx
        trace.append({'iteration':iteration+1,'match_count':int(len(source_idx)),'point_to_plane_abs_median':float(np.median(absr)),'point_to_plane_abs_p95':float(np.percentile(absr,95)),'step_rotation_deg':float(np.degrees(np.linalg.norm(step[:3]))),'step_translation_m':float(np.linalg.norm(step[3:])),'normal_weight_median':float(np.median(normal_weight))})
        if np.linalg.norm(step[:3])<float(args.convergence_rotation_rad) and np.linalg.norm(step[3:])<float(args.convergence_translation_m): break
    if len(match_last)>=int(args.min_matches):
        p=apply_pose(source[match_last],R,t);q=target[target_last];n=normals[target_last];plane=np.einsum('ij,ij->i',n,p-q);point=np.linalg.norm(p-q,axis=1)
        absplane=np.abs(plane);abs_point=np.abs(point)
        # Cross-check the point-to-plane solution against a rigid point-to-point
        # fit on the final correspondence set.  Low plane residual alone can hide
        # a large tangential/rotational ambiguity on partial planar surfaces.
        R_point, t_point = rigid_umeyama(source[match_last], target[target_last])
        method_rotation_disagreement = rotation_angle_deg(R @ R_point.T)
        method_translation_disagreement = float(np.linalg.norm(t - t_point))
        correction_rotation_deg = rotation_angle_deg(R)
        correction_translation_m = float(np.linalg.norm(t))
        # Approximate covariance from the final point-to-plane information.
        dof=max(1,len(plane)-6);variance=float(np.sum(plane*plane)/dof);cov=np.linalg.pinv(H_last)*max(variance,1.0e-12);info_eig=np.linalg.eigvalsh(H_last[:3,:3]);condition=float(np.max(info_eig)/max(np.min(info_eig),1.0e-12))
        overlap=float(len(match_last)/max(1,min(len(source),len(target))))
        rotation_observable=bool(np.min(info_eig)>=float(args.min_rotation_information) and condition<=float(args.max_rotation_condition))
    else:
        plane=np.empty(0);absplane=np.empty(0);abs_point=np.empty(0);cov=np.full((6,6),np.nan);info_eig=np.full(3,np.nan);condition=float('inf');overlap=0.;rotation_observable=False;method_rotation_disagreement=float('inf');method_translation_disagreement=float('inf');correction_rotation_deg=float('inf');correction_translation_m=float('inf')
    acceptance_criteria = {
        'minimum_matches': len(match_last)>=int(args.min_matches),
        'plane_median': bool(len(absplane)>0 and np.median(absplane)<=float(args.max_plane_median_m)),
        'overlap': overlap>=float(args.min_overlap),
        'rotation_information': rotation_observable,
        'bounded_canonical_rotation': correction_rotation_deg<=float(args.max_canonical_correction_deg),
        'bounded_canonical_translation': correction_translation_m<=float(args.max_canonical_correction_m),
        'point_plane_point_method_rotation_agreement': method_rotation_disagreement<=float(args.max_method_rotation_disagreement_deg),
        'point_plane_point_method_translation_agreement': method_translation_disagreement<=float(args.max_method_translation_disagreement_m),
    }
    accepted=bool(all(acceptance_criteria.values()))
    return {'rotation':R,'translation':t,'match_count':int(len(match_last)),'overlap_fraction':overlap,'point_to_plane_abs_m':numeric_summary(absplane),'point_to_point_m':numeric_summary(abs_point),'rotation_information_eigenvalues':info_eig.tolist(),'rotation_information_condition':condition,'rotation_observable':rotation_observable,'canonical_correction_rotation_deg':correction_rotation_deg,'canonical_correction_translation_m':correction_translation_m,'point_plane_point_method_rotation_disagreement_deg':method_rotation_disagreement,'point_plane_point_method_translation_disagreement_m':method_translation_disagreement,'acceptance_criteria':acceptance_criteria,'accepted':accepted,'trace':trace,'covariance_6x6':cov.tolist()}


def load_clouds(args: argparse.Namespace) -> tuple[dict[int,Cloud],dict[int,dict[str,Any]],dict[int,dict[str,Any]]]:
    ann=load_json(args.annotations);frames={int(f['frame_idx']):f for f in ann.get('frames',[]) if isinstance(f,dict) and f.get('frame_idx') is not None};pose=load_json(args.pose_report);rows={int(r['frame_idx']):r for r in pose.get('pose_rows',[]) if isinstance(r,dict) and str(r.get('status') or '') in POSE_STATUSES};clouds={};objects={}
    for idx,row in rows.items():
        if args.frame_start is not None and idx<args.frame_start or args.frame_end is not None and idx>args.frame_end:continue
        f=frames.get(idx);obj=next((o for o in f.get('objects',[]) if o.get('object_id')==args.object_id),None) if f else None
        geom=obj.get('visible_geometry_candidate') if isinstance(obj,dict) and isinstance(obj.get('visible_geometry_candidate'),dict) else {}
        pts=np.asarray(geom.get('world_vertices_sample_m') or [],dtype=np.float64)
        if pts.ndim!=2 or pts.shape[1]!=3 or len(pts)<int(args.min_points) or not np.isfinite(pts).all():continue
        R=np.asarray(row.get('rotation_world_from_completed_canonical_matrix'),dtype=np.float64);t=np.asarray(row.get('translation_world_m'),dtype=np.float64)
        if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():continue
        if obj.get('rigid_pose_observation_eligible') is False or geom.get('rigid_pose_observation_eligible') is False:continue
        sample=deterministic_sample(pts,int(args.max_points),int(args.seed)+idx);clouds[idx]=Cloud(idx,R,t,sample,inverse_pose(sample,R,t));objects[idx]=obj
    return clouds,frames,objects


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--pose-report',type=Path,required=True);p.add_argument('--object-id',required=True);p.add_argument('--output-npz',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--frame-start',type=int,default=None);p.add_argument('--frame-end',type=int,default=None);p.add_argument('--anchor-frame',type=int,default=0);p.add_argument('--keyframe-interval',type=int,default=5);p.add_argument('--max-points',type=int,default=800);p.add_argument('--min-points',type=int,default=50);p.add_argument('--max-keyframes',type=int,default=40);p.add_argument('--normal-k',type=int,default=24);p.add_argument('--icp-iterations',type=int,default=12);p.add_argument('--trim-fraction',type=float,default=.70);p.add_argument('--max-correspondence-m',type=float,default=.045);p.add_argument('--min-matches',type=int,default=50);p.add_argument('--min-overlap',type=float,default=.08);p.add_argument('--max-plane-median-m',type=float,default=.012);p.add_argument('--huber-delta-m',type=float,default=1.5);p.add_argument('--min-noise-m',type=float,default=.001);p.add_argument('--max-planarity-ratio',type=float,default=.25);p.add_argument('--damping',type=float,default=1e-6);p.add_argument('--max-rotation-step-rad',type=float,default=.12);p.add_argument('--max-translation-step-m',type=float,default=.025);p.add_argument('--convergence-rotation-rad',type=float,default=1e-5);p.add_argument('--convergence-translation-m',type=float,default=1e-5);p.add_argument('--min-rotation-information',type=float,default=1e-5);p.add_argument('--max-rotation-condition',type=float,default=1e8);p.add_argument('--max-canonical-correction-deg',type=float,default=8.0);p.add_argument('--max-canonical-correction-m',type=float,default=.020);p.add_argument('--max-method-rotation-disagreement-deg',type=float,default=3.0);p.add_argument('--max-method-translation-disagreement-m',type=float,default=.010);p.add_argument('--seed',type=int,default=20260904);args=p.parse_args()
    if args.keyframe_interval<1:raise RuntimeError('keyframe interval must be >=1')
    clouds,frames,objects=load_clouds(args);direct=sorted(clouds);anchor=int(args.anchor_frame)
    if anchor not in clouds:raise RuntimeError(f'anchor {anchor} lacks accepted cloud')
    keyframes=[idx for idx in direct if idx==anchor or (idx-anchor)%int(args.keyframe_interval)==0];
    if direct[-1] not in keyframes:keyframes.append(direct[-1])
    keyframes=sorted(set(keyframes))[:int(args.max_keyframes)]
    pairs=[]
    for a,b in zip(keyframes[:-1],keyframes[1:]):pairs.append((a,b,'keyframe_local'))
    for b in keyframes:
        if b!=anchor:pairs.append((anchor,b,'anchor_loop'))
    # Estimate normals independently for each target cloud.
    normals={};planarity={}
    for idx in sorted({b for _,b,_ in pairs}):normals[idx],planarity[idx]=estimate_normals(clouds[idx].canonical0,int(args.normal_k))
    rows=[];src_ids=[];dst_ids=[];R_all=[];t_all=[];cov_all=[];accepted_all=[]
    for no,(a,b,kind) in enumerate(pairs,1):
        result=point_to_plane_registration(
            clouds[a].canonical0,
            clouds[b].canonical0,
            normals=normals[b],
            planarity=planarity[b],
            args=args,
        )
        rows.append({'source_frame_idx':a,'target_frame_idx':b,'kind':kind,**{k:v for k,v in result.items() if k not in {'rotation','translation','covariance_6x6'}}})
        src_ids.append(a);dst_ids.append(b);R_all.append(result['rotation']);t_all.append(result['translation']);accepted_all.append(result['accepted']);cov_all.append(result['covariance_6x6'])
        print(f'[keyframe-edge] {no}/{len(pairs)} {a}->{b} {kind} accepted={result["accepted"]} matches={result["match_count"]}',flush=True)
    # T_all is retained as a named alias for backwards-readable NPZ inspection.
    meta={'schema':'v20_keyframe_observed_se3_edges_v1','annotations':str(args.annotations.expanduser().resolve()),'pose_report':str(args.pose_report.expanduser().resolve()),'object_id':args.object_id,'keyframe_interval':int(args.keyframe_interval),'keyframes':keyframes,'generated_geometry_consumed':False,'input_sha256':{'annotations':sha256_file(args.annotations),'pose_report':sha256_file(args.pose_report)}}
    args.output_npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output_npz,metadata=np.asarray([json.dumps(meta)]),source_frame_idx=np.asarray(src_ids,dtype=np.int64),target_frame_idx=np.asarray(dst_ids,dtype=np.int64),rotation_correction=np.asarray(R_all,dtype=np.float64),translation_correction_m=np.asarray(t_all,dtype=np.float64),accepted=np.asarray(accepted_all,dtype=bool),covariance_6x6=np.asarray(cov_all,dtype=np.float64))
    report={**meta,'pair_count':len(pairs),'accepted_count':int(sum(accepted_all)),'rejected_count':int(len(pairs)-sum(accepted_all)),'accepted_fraction':float(sum(accepted_all)/max(1,len(pairs))),'rotation_information_min_eigen':numeric_summary([min(r['rotation_information_eigenvalues']) for r in rows if r['rotation_observable']]),'rotation_condition':numeric_summary([r['rotation_information_condition'] for r in rows if np.isfinite(r['rotation_information_condition'])]),'point_to_plane_abs_median':numeric_summary([r['point_to_plane_abs_m']['median'] for r in rows if r['point_to_plane_abs_m']['median'] is not None]),'rows':rows,'outputs':{'npz':str(args.output_npz.expanduser().resolve()),'report':str(args.output_json.expanduser().resolve())}}
    args.output_json.parent.mkdir(parents=True,exist_ok=True);args.output_json.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:report[k] for k in ['keyframes','pair_count','accepted_count','rejected_count','accepted_fraction','rotation_information_min_eigen','rotation_condition','point_to_plane_abs_median']},indent=2))

if __name__=='__main__':main()
