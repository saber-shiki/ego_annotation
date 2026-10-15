"""Opt-in sequential repair worker. Prediction inputs only, immutable source/cache."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from dataclasses import replace
import argparse,json,time,traceback
import cv2,numpy as np
from preprocessing.inputs import load_inputs,sha,reject_evaluation_path
from preprocessing.surface_pipeline import code_identity,write_json
from preprocessing.geometry import camera_pose,resized_intrinsics,TriangleMesh
from preprocessing.visible_surface import cloud_digest
from preprocessing.factors import hand_depths
from preprocessing.shared_repair import ObjectRepair,shared_matrix

def load_state(path,source):
 path=Path(path).resolve();reject_evaluation_path(path)
 meta=json.loads((path/'coarse_scene.json').read_text())
 if meta['source_bundle_sha256']!=sha(source):raise ValueError('cache source mismatch')
 archive=path/'coarse_scene.npz'
 if meta['arrays']['sha256']!=sha(archive):raise ValueError('cache arrays changed')
 with np.load(archive,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 return a,meta

def run(args):
 source=Path(args.source).resolve();baseline=Path(args.baseline).resolve();out=Path(args.output).resolve()
 for p in (source,baseline):reject_evaluation_path(p)
 if out.is_relative_to(source.parent) or out.is_relative_to(baseline):raise ValueError('fresh output must be isolated')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=code_identity();cv2.setNumThreads(1);cv2.setRNGSeed(47)
 try:
  base,meta=load_state(baseline,source);initial,initialmeta=load_state(args.initial or baseline,source)
  inputs=load_inputs(source,480,read_owned_masks=True);digest=cloud_digest(inputs.clouds)
  for q in inputs.clouds:q.setflags(write=False)
  if not np.array_equal(inputs.ids,base['frame_idx']):raise ValueError('timeline mismatch')
  source_hashes={str(source):sha(source),str(baseline/'coarse_scene.npz'):sha(baseline/'coarse_scene.npz'),str(baseline/'coarse_scene.json'):sha(baseline/'coarse_scene.json')}
  if args.initial:
   p=Path(args.initial);source_hashes.update({str(p/'coarse_scene.npz'):sha(p/'coarse_scene.npz'),str(p/'coarse_scene.json'):sha(p/'coarse_scene.json')})
  config=dict(stage=args.stage,shape_keys=10,shape_outer=3,timeline_outer=2,max_nfev=24,scoring_width=160,shape_bounds=.25,depth_additive_bias=False,camera_unchanged=True)
  write_json(out/'started.json',dict(configuration=config,source=str(source),code_sha256=code))
  calibration=None;relative=np.ones(len(inputs.ids))
  if args.stage in ('depth_only','shape_depth','hands'):
   from preprocessing.relative_repair import estimate_calibration
   relative,calibration=estimate_calibration(inputs)
   write_json(out/'relative_depth_calibration.json',calibration)
   working=replace(inputs,clouds=[p*relative[i] for i,p in enumerate(inputs.clouds)])
  else:working=inputs
  hd=hand_depths(inputs,160);print('source loaded',len(inputs.ids),'raw depth digest',digest,flush=True)
  solver=ObjectRepair(working,base,hd,width=160)
  q=np.asarray(initial.get('shared_repair_log_lengths',np.zeros(3)),float)
  R=initial['object_rotation'].copy();t=initial['object_translation_world'].copy();shape_trace=[];pose_trace=[]
  if args.stage=='shape' or (args.stage=='shape_depth' and args.initial is None):
   valid=np.flatnonzero(inputs.cloud_valid);keys=np.unique(np.r_[inputs.anchor,valid[np.linspace(0,len(valid)-1,10).astype(int)]])
   R,t,q,shape_trace=solver.fit(R,t,q,keys,True,outer=3,max_nfev=24)
  if args.stage!='hands':R,t,q,pose_trace=solver.fit(R,t,q,np.arange(len(R)),False,outer=2,max_nfev=24,freeze_rotation=args.stage in ('depth_only','shape_depth'))
  S=shared_matrix(q,solver.basis);archive={k:v.copy() for k,v in initial.items()}
  archive.update(object_rotation=R,object_translation_world=t,vertices_canonical=base['vertices_canonical']@S.T,faces=base['faces'],shared_repair_log_lengths=q,shared_repair_basis=solver.basis,shared_repair_matrix=S,shared_shape_matrix=S*float(base['uniform_object_scale']),relative_depth_scale=relative,relative_depth_bias_m=np.zeros(len(R)),raw_depth_reference_unchanged=np.array(True),absolute_metric_accuracy_certified=np.array(False),alignment_initial_rotation=base['object_rotation'],alignment_initial_translation_world=base['object_translation_world'])
  hand_report=None
  if args.stage=='hands':
   from preprocessing.mano_repair import repair_hands
   archive,hand_report=repair_hands(inputs,archive,out)
   R=archive['object_rotation'];t=archive['object_translation_world']
  # Compare actual before/after against exactly the same raw source population.
  rawsolver=ObjectRepair(inputs,base,hd,width=160)
  before=rawsolver.correspondences(base['object_rotation'],base['object_translation_world'],np.zeros(3),range(len(R)))
  after=rawsolver.correspondences(R,t,q,range(len(R)))
  def summary(rows,RR,tt,qq):
   SS=shared_matrix(qq,solver.basis);result=[]
   for row in rows:
    i=row.frame;rc,tc=camera_pose(RR[i],tt[i],inputs.cameras[i]);p=row.depth_X@SS.T@rc.T+tc
    _,_,valid,_=rawsolver.mesh.raster(rawsolver.Kr[i],rc,tc,SS,rawsolver.wh)
    result.append(dict(frame_idx=int(inputs.ids[i]),samples=len(p),raw_depth_mean_residual_m=float(np.mean(np.linalg.norm(p-row.depth_Q,axis=1))) if len(p) else None,mask=rawsolver.domains[i].score(valid)))
   return result
  write_json(out/'source_diagnostics.json',dict(before=summary(before,base['object_rotation'],base['object_translation_world'],np.zeros(3)),after=summary(after,R,t,q),raw_depth_fixed_population=True))
  write_json(out/'solver.json',dict(shape=shape_trace,timeline=pose_trace,shape_log_lengths=q.tolist(),shared_matrix=S.tolist(),basis=solver.basis.tolist(),hand=hand_report))
  np.savez_compressed(out/'coarse_scene.npz',**archive)
  manifest=dict(schema='ego.annotation.monocular_coarse_scene.v1',alignment_profile='sequential_shared_shape_depth_mano_repair',stage=args.stage,configuration=config,source_observations=str(source),source_bundle_sha256=sha(source),memoized_baseline=str(baseline),memoized_initialization=str(args.initial or baseline),object_id=meta['object_id'],frame_ids=inputs.ids.tolist(),time_s=inputs.times.tolist(),contains_optimized_state=True,contains_policy_results=False,prediction_side_only=True,GT_or_CAD_inputs=False,additional_camera_inputs=False,absolute_metric_accuracy_certified=False,source_cloud_digest=digest,raw_depth_unchanged=True,relative_depth_calibration_enabled=calibration is not None,depth_calibration_gauge='source appearance anchor fixed, relative only',per_frame_shape=False,all_native_vertices_faces_preserved=True,shared_shape_log_lengths=q.tolist(),shared_shape_matrix=S.tolist(),source_MANO_unchanged=args.stage!='hands',camera_unchanged=True,signed_geometry_ready=False,collision_authority_changed=False,visibility_and_contact_source=str(inputs.bundle.source_path('interaction_judgments')),inferred_frames_are_not_observations=True,source_reads=inputs.receipt.files,initialization_hashes=source_hashes,code_sha256=code,arrays=dict(path=str(out/'coarse_scene.npz'),sha256=sha(out/'coarse_scene.npz')),seconds_before_render=time.monotonic()-start,limitations=['Shared affine mesh repair does not reconstruct hidden topology','Relative calibration does not fix source-anchor absolute scale','RGB/depth source observations remain uncertain','Source full MANO FK and image support may be insufficient for absolute hand depth','Generated geometry is not signed occupancy authority'])
  inputs.receipt.check()
  for p,h in source_hashes.items():
   if sha(p)!=h:raise RuntimeError('source/cache changed '+p)
  if cloud_digest(inputs.clouds)!=digest or code_identity()!=code:raise RuntimeError('raw source or immutable code changed')
  write_json(out/'coarse_scene.json',manifest)
  from preprocessing.repair_render import render_stage
  render_stage(inputs,base,archive,out/'renders')
  inputs.receipt.check()
  if code_identity()!=code:raise RuntimeError('code changed during prediction')
  write_json(out/'completed.json',dict(status='complete_derived_coarse_scene',frames=len(R),stage=args.stage,seconds=time.monotonic()-start,prediction_frozen=True,rendered=True))
  print('complete',args.stage,'seconds',time.monotonic()-start,flush=True)
 except Exception as e:
  write_json(out/'failed.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise

def cli():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--baseline',required=True);p.add_argument('--initial');p.add_argument('--output',required=True);p.add_argument('--stage',choices=['shape','depth_only','shape_depth','hands'],required=True);args=p.parse_args();run(args)
if __name__=='__main__':cli()
