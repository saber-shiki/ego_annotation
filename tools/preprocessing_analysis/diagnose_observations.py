"""Read-only five-case observation consistency diagnostics; no GT/CAD/model calls.
Run against an explicit frozen prediction/snapshot, write outside observations/runs.
These statistics measure consistency/observability, NOT physical accuracy.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def stats(values):
 a=np.asarray(values,float);a=a[np.isfinite(a)]
 return {'n':int(len(a)),'median':float(np.median(a)),'p05':float(np.percentile(a,5)),'p95':float(np.percentile(a,95)),'max':float(a.max())} if len(a) else {'n':0}


def gaps(valid):
 intervals=[];start=None
 for i,v in enumerate(list(valid)+[True]):
  if not v and start is None:start=i
  if v and start is not None:intervals.append([start,i-1]);start=None
 return intervals


def analyze(item,args):
 from preprocessing.inputs import load_inputs
 from preprocessing.factors import hand_depths
 from preprocessing.geometry import camera_pose,project,matrix_k
 import preprocessing.tracking as tracking
 out=Path(item['output']);inp=load_inputs(item['source'],480,read_owned_masks=True);hd=hand_depths(inp,128)
 n=len(inp.ids);surface_rows=[];bg_rows=[];track_rows=[];context={}
 old_solve=tracking.solve_pnp;old_contour=tracking.contour_pose_proposal
 def solve(X,uv,K,R,t):
  result=old_solve(X,uv,K,R,t);context.clear();context.update(X=X.copy(),uv=uv.copy(),result=result)
  return result
 def contour(R,t,inputs,mesh,hand_depth,i,iterations=10):
  X=context.get('X',np.empty((0,3)));uv=context.get('uv',np.empty((0,2)));pnp=context.get('result')
  ii=pnp[2] if pnp is not None else np.arange(len(X));X=X[ii];uv=uv[ii]
  row={'frame_idx':int(inp.ids[i]),'PnP_available':pnp is not None,'points':int(len(X))}
  if len(X)>=4:
   eig=np.linalg.eigvalsh(np.cov(X.T));hull=cv2.convexHull(np.asarray(uv,np.float32));area=float(cv2.contourArea(hull))
   row.update(model_point_min_over_max_eigenvalue=float(max(0,eig[0])/max(eig[-1],1e-15)),
      image_hull_over_appearance_area=area/max(1,int(inp.masks[i].sum())))
   rc,tc=camera_pose(R,t,inp.cameras[i]);P=X@rc.T+tc;at=project(P,inp.K[i]);_,z,hit=mesh.query(at,inp.K[i],rc,tc)
   behind=P[:,2]-z
   row['model_bound_points_behind_first_surface_fraction']=float(np.mean(hit&(behind>.005)))
   row['model_self_visibility_test_hit_fraction']=float(np.mean(hit))
   row['median_reprojection_px']=float(np.median(np.linalg.norm(at-uv,axis=1)))
  track_rows.append(row)
  return old_contour(R,t,inputs,mesh,hand_depth,i,iterations)
 tracking.solve_pnp=solve;tracking.contour_pose_proposal=contour
 try:
  cv2.setRNGSeed(47);R,t,_,rows=tracking.bootstrap(inp,hd,guarded_output=True)
 finally:tracking.solve_pnp=old_solve;tracking.contour_pose_proposal=old_contour
 with np.load(out/'bootstrap.npz',allow_pickle=False) as z:
  dr=float(np.max(np.abs(R-z['object_rotation'])));dt=float(np.max(np.abs(t-z['object_translation_world'])))
  if dr>1e-9 or dt>1e-9:raise ValueError(f'audit replay changed outputs: {dr} {dt}')
 entity=inp.geometry_report['object_id']
 for i,f in enumerate(inp.frames):
  obj=next((o for o in f.get('objects',[]) if o.get('object_id')==entity),{});g=obj.get('visible_geometry_candidate') or {};q=inp.clouds[i]
  own=inp.owned_masks[i];raw=inp.masks[i]
  row={'frame_idx':int(inp.ids[i]),'metric_eligible':bool(inp.cloud_valid[i]),'metric_points':len(q),'appearance_pixels':int(raw.sum()),
   'owned_over_appearance_pixels':float(own.sum()/max(1,raw.sum())),
   'explicit_ownership_filter_state':(g.get('object_surface_ownership_filter') or {}).get('state'),
   'first_surface_filter_state':(g.get('first_surface_depth_ownership') or {}).get('state'),
   'hand_confidences':[h.get('confidence') for h in f.get('hands',[])],
   'hand_sources':[h.get('hand_geometry_source') for h in f.get('hands',[])],
   'metric_rejection_reason':obj.get('rigid_pose_observation_reason')}
  if len(q):
   uv=project(q,inp.K[i]);px=np.rint((uv+.5)*np.array([128/inp.wh[0],128/inp.wh[1]])-.5).astype(int)
   valid=(px[:,0]>=0)&(px[:,0]<128)&(px[:,1]>=0)&(px[:,1]<128)
   hz=np.full(len(q),np.inf);hz[valid]=hd[i,px[valid,1],px[valid,0]]
   row.update(surface_median_camera_z_m=float(np.median(q[:,2])),surface_z_p95_p05_m=float(np.percentile(q[:,2],95)-np.percentile(q[:,2],5)),
     MANO_predicted_in_front_of_visible_surface_fraction=float(np.mean(hz<q[:,2]-.01)),
     projected_MANO_overlap_fraction=float(np.mean(np.isfinite(hz))))
  surface_rows.append(row)
 # Sparse static-background optical-flow test of camera epipolar geometry. Neither
 # this nor a camera/mano shared coordinate contract certifies metric scale.
 for i in range(1,n,3):
  exclusion=cv2.dilate(inp.masks[i-1].astype('uint8'),np.ones((25,25),np.uint8))>0
  for h in inp.frames[i-1].get('hands',[]):
   box=h.get('bbox_xyxy')
   if box is not None and len(box)==4:
    box=np.asarray(box,float)*np.tile(np.array(inp.wh)/inp.source_wh[i-1],2)
    x0,y0,x1,y1=np.rint(box).astype(int);exclusion[max(0,y0-20):min(inp.wh[1],y1+20),max(0,x0-20):min(inp.wh[0],x1+20)]=True
  uv=cv2.goodFeaturesToTrack(inp.grays[i-1],maxCorners=160,qualityLevel=.01,minDistance=8,mask=(~exclusion).astype('uint8')*255)
  if uv is None:continue
  p,ok,_=cv2.calcOpticalFlowPyrLK(inp.grays[i-1],inp.grays[i],uv,None,winSize=(25,25),maxLevel=3)
  if p is None:continue
  back,bok,_=cv2.calcOpticalFlowPyrLK(inp.grays[i],inp.grays[i-1],p,None,winSize=(25,25),maxLevel=3)
  if back is None:continue
  valid=ok.ravel().astype(bool)&bok.ravel().astype(bool)&np.isfinite(p).all((1,2))
  fb=np.linalg.norm(back.reshape(-1,2)-uv.reshape(-1,2),axis=1);valid&=fb<1.
  u=uv.reshape(-1,2)[valid];v=p.reshape(-1,2)[valid]
  if len(u)<8:continue
  rel=np.linalg.inv(inp.cameras[i])@inp.cameras[i-1];tr=rel[:3,3];rr=rel[:3,:3]
  skew=np.array([[0,-tr[2],tr[1]],[tr[2],0,-tr[0]],[-tr[1],tr[0],0.]])
  F=np.linalg.inv(matrix_k(inp.K[i])).T@skew@rr@np.linalg.inv(matrix_k(inp.K[i-1]))
  x=np.c_[u,np.ones(len(u))];y=np.c_[v,np.ones(len(v))];Fx=x@F.T;Fy=y@F
  denominator=Fx[:,0]**2+Fx[:,1]**2+Fy[:,0]**2+Fy[:,1]**2
  err=np.abs(np.sum(y*Fx,axis=1))/np.sqrt(np.maximum(denominator,1e-30))
  bg_rows.append({'frame_idx':int(inp.ids[i]),'background_candidates':len(uv),'FB_consistent_tracks':len(u),
   'camera_translation_step_m':float(np.linalg.norm(tr)),'median_camera_epipolar_Sampson_px':float(np.median(err)) if np.linalg.norm(tr)>1e-6 else None,
   'p95_camera_epipolar_Sampson_px':float(np.percentile(err,95)) if np.linalg.norm(tr)>1e-6 else None,
   'median_background_flow_px':float(np.median(np.linalg.norm(v-u,axis=1))),
   'limitation':'background selection is approximate; near-zero baseline epipolar tests are weak; no scale test'})
 track_rows.sort(key=lambda r:r['frame_idx']);good=[r for r in track_rows if r['PnP_available']]
 metric=[r for r in surface_rows if r['metric_eligible']]
 import trimesh
 mesh=trimesh.Trimesh(inp.vertices,inp.faces,process=False)
 summary={'case':item['case'],'frames':n,'metric_frames':int(inp.cloud_valid.sum()),'metric_gaps':gaps(inp.cloud_valid),
  'owned_over_appearance_pixels':stats([r['owned_over_appearance_pixels'] for r in metric]),
  'explicit_ownership_filter_states':sorted(set(str(r['explicit_ownership_filter_state']) for r in metric)),
  'PnP_frames':len(good),'PnP_inliers':stats([r['points'] for r in good]),'PnP_median_reprojection_px':stats([r['median_reprojection_px'] for r in good]),
  'point_planarity_eigenvalue_ratio':stats([r['model_point_min_over_max_eigenvalue'] for r in good]),
  'point_hull_over_mask_area':stats([r['image_hull_over_appearance_area'] for r in good]),
  'behind_first_surface_fraction':stats([r['model_bound_points_behind_first_surface_fraction'] for r in good]),
  'MANO_front_conflict_fraction':stats([r['MANO_predicted_in_front_of_visible_surface_fraction'] for r in metric]),
  'background_epipolar_pair_median_px':stats([r['median_camera_epipolar_Sampson_px'] for r in bg_rows if r['median_camera_epipolar_Sampson_px'] is not None]),
  'background_test_pairs':len(bg_rows),'camera_provenance':sorted(set(f['camera'].get('v19_camera_pose_source','unspecified') for f in inp.frames)),
  'mesh':{'vertices':len(inp.vertices),'faces':len(inp.faces),'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'signed_contact_authority':False},
  'replay_max_R_difference':dr,'replay_max_t_difference_m':dt,'GT_or_CAD_read':False,
  'interpretation':'consistency/observability diagnostics, not proof that any entire modality is trustworthy'}
 inp.receipt.check()
 result={'summary':summary,'surface_rows':surface_rows,'tracking_rows':track_rows,'background_rows':bg_rows}
 (args.output/(item['case']+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(summary),flush=True)
 return summary


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--snapshot',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
 if (args.run/'batch.exit').read_text().strip()!='0':raise ValueError('requires a completed frozen prediction')
 if args.output.resolve().is_relative_to(args.run.resolve()) or args.output.resolve().is_relative_to(args.snapshot.resolve()):raise ValueError('external output required')
 args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.snapshot.resolve()));cv2.setNumThreads(1)
 items=json.loads((args.run/'inputs.json').read_text());summary=[analyze(item,args) for item in items]
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (args.output/'completed.json').write_text(json.dumps({'cases':len(items),'prediction_modified':False,'source_modified':False,'GT_or_CAD_read':False})+'\n')

if __name__=='__main__':main()
