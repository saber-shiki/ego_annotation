"""Read-only replay of the frozen tracker, recording evidence before/after contour fitting."""
from pathlib import Path
import sys,json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
SNAP=Path('/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_visible_surface_20260927_r2')
sys.path.insert(0,str(SNAP))
from preprocessing.inputs import load_inputs
from preprocessing.factors import hand_depths,contour_factors
from preprocessing.geometry import camera_pose,project,resized_intrinsics
import preprocessing.tracking as tracker
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2/source_observations.json')
RUN=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_visible_surface_20260927_r2/milk_observation_v2_20260921_r2')
cv2.setNumThreads(1);cv2.setRNGSeed(47)
inputs=load_inputs(SOURCE,480,read_owned_masks=True);hd=hand_depths(inputs,128)
original_pnp=tracker.solve_pnp;original_fit=tracker.small_image_fit
context={};rows=[];witness={}

def pnp(points,uv,k,R,t):
    result=original_pnp(points,uv,k,R,t)
    context.clear();context.update(points=np.array(points,copy=True),uv=np.array(uv,copy=True),result=result)
    return result

def score(R,t,X,uv,i):
    if not len(X):return None
    rc,tc=camera_pose(R,t,inputs.cameras[i]);cam=X@rc.T+tc
    error=np.linalg.norm(project(cam,inputs.K[i])-uv,axis=1)
    return {'median_px':float(np.median(error)),'p90_px':float(np.percentile(error,90)),'positive_fraction':float(np.mean(cam[:,2]>0))}

def fit(R,t,inp,mesh,hand_depth,i,iterations=10):
    oldR=np.array(R,copy=True);oldt=np.array(t,copy=True)
    result=original_fit(R,t,inp,mesh,hand_depth,i,iterations)
    newR,newt,diagnostic=result
    X=context.get('points',np.empty((0,3)));uv=context.get('uv',np.empty((0,2)));proposal=context.get('result')
    ii=proposal[2] if proposal is not None else np.arange(len(X))
    row={'frame':int(inp.ids[i]),'contour_rotation_update_deg':float(np.degrees(Rotation.from_matrix(newR@oldR.T).magnitude())),
        'contour_translation_update_mm':float(np.linalg.norm(newt-oldt)*1000),'tracked_points':len(X),'PnP_inliers':len(ii),
        'tracked_reprojection_before':score(oldR,oldt,X[ii],uv[ii],i),'tracked_reprojection_after':score(newR,newt,X[ii],uv[ii],i),'fit':diagnostic}
    # Same original appearance pixels for both poses; never score on newly selected inliers.
    k=resized_intrinsics(inp.K[i],inp.wh,(128,128));mask=cv2.resize(inp.masks[i].astype('uint8'),(128,128),interpolation=cv2.INTER_NEAREST_EXACT)>0
    before_after=[]
    for rr,tt in [(oldR,oldt),(newR,newt)]:
        rc,tc=camera_pose(rr,tt,inp.cameras[i]);_,zz,rendered=mesh.raster(k,rc,tc,(128,128))
        before_after.append(float(np.count_nonzero(rendered&mask)/max(1,np.count_nonzero(rendered|mask))))
    row['whole_appearance_IoU_before_after']=before_after
    rows.append(row)
    if i>=125 or row['contour_rotation_update_deg']>15:
        witness[f'X_{i}']=X[ii];witness[f'uv_{i}']=uv[ii];witness[f'R_before_{i}']=oldR;witness[f't_before_{i}']=oldt;witness[f'R_after_{i}']=newR;witness[f't_after_{i}']=newt
    return result

tracker.solve_pnp=pnp;tracker.small_image_fit=fit
R,t,tracks,_=tracker.bootstrap(inputs,hd)
with np.load(RUN/'bootstrap.npz',allow_pickle=False) as stored:
    max_R=float(np.max(np.abs(R-stored['object_rotation'])));max_t=float(np.max(np.abs(t-stored['object_translation_world'])))
if max_R>1e-9 or max_t>1e-9:raise ValueError(f'replay mismatch {max_R} {max_t}')
report={'replay_exact_to_1e_9':True,'max_rotation_array_difference':max_R,'max_translation_array_difference_m':max_t,'rows':rows,'source_modified':False}
(ROOT/'tracking_before_after_trace.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(ROOT/'tracking_witness.npz',**witness)
for r in rows:
    if r['frame']>=140 or r['contour_rotation_update_deg']>20:print(json.dumps(r),flush=True)
