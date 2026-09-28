"""Low-texture SO(3) proposals scored with full triangle silhouettes, not proxies."""
from __future__ import annotations
import itertools
import cv2
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from preprocessing.geometry import camera_pose,resized_intrinsics,world_pose
from preprocessing.factors import ImageFactors
from preprocessing.relocalization import DescriptorBank,silhouette_score,surface_score


def cube_rotations():
    out=[]
    for perm in itertools.permutations(range(3)):
        P=np.eye(3)[:,perm]
        for signs in itertools.product([-1.,1.],repeat=3):
            R=P@np.diag(signs)
            if np.linalg.det(R)>.99:out.append(R)
    return out


def select_path(candidates,times,angular_sigma=6.,translation_sigma=.6):
    """Dynamic programming over observed keyframe hypotheses, no per-frame hard cap."""
    costs=[np.array([v['cost'] for v in candidates[0]],float)];parents=[]
    for i in range(1,len(candidates)):
        dt=max(float(times[i]-times[i-1]),1e-4);row=np.full(len(candidates[i]),np.inf);back=np.zeros(len(row),int)
        for j,current in enumerate(candidates[i]):
            options=[]
            for k,previous in enumerate(candidates[i-1]):
                angle=Rotation.from_matrix(current['R']@previous['R'].T).magnitude()
                delta=np.linalg.norm(current['t']-previous['t'])
                transition=.2*(angle/(angular_sigma*dt))**2+.04*(delta/(translation_sigma*dt))**2
                options.append(costs[-1][k]+transition)
            back[j]=int(np.argmin(options));row[j]=current['cost']+options[back[j]]
        costs.append(row);parents.append(back)
    path=[int(np.argmin(costs[-1]))]
    for back in reversed(parents):path.append(int(back[path[-1]]))
    return list(reversed(path))


def dense_acceptance(before,after,cost_before,cost_after,surface_before=(),surface_after=()):
    """Monotone measured objective, not an arbitrary minimum IoU gain."""
    if not len(before):return False,'no_dense_visible_support'
    if np.mean(cost_after)>=np.mean(cost_before)-1e-6:return False,'no_dense_objective_improvement'
    if np.median(after)<np.median(before)-1e-6:return False,'rejected_dense_image_regression'
    if len(surface_before) and np.median(surface_after)>np.median(surface_before)+.01:return False,'rejected_source_surface_regression'
    return True,'applied_multiframe_shape_hypothesis'


def interpolate_corrections(base_R,base_t,times,keys,selected):
    correction=np.asarray([c['R']@base_R[k].T for k,c in zip(keys,selected)])
    delta=np.asarray([c['t']-base_t[k] for k,c in zip(keys,selected)])
    stamps=np.asarray(times)[keys]
    if len(keys)==1:rot=np.repeat(correction[:1],len(times),axis=0)
    else:rot=Slerp(stamps,Rotation.from_matrix(correction))(np.clip(times,stamps[0],stamps[-1])).as_matrix()
    shift=np.column_stack([np.interp(times,stamps,delta[:,j]) for j in range(3)])
    return rot@base_R,base_t+shift


def align_projection_center(inputs,mesh,hand_depth,i,R,t):
    wh=(64,round(64*inputs.wh[1]/inputs.wh[0]));K=resized_intrinsics(inputs.K[i],inputs.wh,wh)
    rc,tc=camera_pose(R,t,inputs.cameras[i]);_,depth,mask=mesh.raster(K,rc,tc,wh)
    target=cv2.resize(inputs.masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
    if mask.sum()<4 or target.sum()<4:return t.copy()
    y,x=np.nonzero(mask);ty,tx=np.nonzero(target);z=float(np.median(depth[mask]))
    dt=np.array([(np.mean(tx)-np.mean(x))*z/K[0],(np.mean(ty)-np.mean(y))*z/K[1],0.])
    # Translation-only initialization for the actual mesh fit, not a centroid pose output.
    return t+inputs.cameras[i,:3,:3]@dt


def hypothesis_cost(inputs,mesh,hd,i,R,t,baseline_R):
    image=silhouette_score(inputs,mesh,hd,i,R,t)
    surface=surface_score(inputs,mesh,i,R,t)
    deviation=Rotation.from_matrix(R@baseline_R.T).magnitude()
    cost=1-image['iou']+.01*deviation**2
    if surface is not None:cost+=.04*min(surface/.025,4.)
    return float(cost),image,surface


def geometry_relocalize(inputs,mesh,hand_depth,R,t,tracks,tracking_rows):
    # Texture sufficiency is measured from source pixels, not semantic object identity.
    cache=DescriptorBank(inputs,mesh,hand_depth,R,t,[])
    keys=sorted(set(range(0,len(R),12))|{inputs.anchor,len(R)-1})
    counts=[len(cache.extract(i)[0]) for i in keys]
    report={'method':'low_texture_full_triangle_SO3_keyframe_path','interior_feature_counts':counts,
            'keyframes':[int(inputs.ids[k]) for k in keys],'applied':False,'absolute_accuracy_certified':False}
    if np.median(counts)>=12:
        report['state']='interior_texture_support_available';return R,t,tracks,tracking_rows,report
    from preprocessing.tracking import small_image_fit
    rotations=cube_rotations();B=inputs.shape_basis;states=[];baseline_cost=[]
    small_hd=np.asarray([cv2.resize(d.astype('float32'),(64,round(64*d.shape[0]/d.shape[1])),interpolation=cv2.INTER_NEAREST) for d in hand_depth])
    for i in keys:
        cost,image,surface=hypothesis_cost(inputs,mesh,hand_depth,i,R[i],t[i],R[i])
        baseline={'R':R[i].copy(),'t':t[i].copy(),'cost':cost,'iou':image['iou'],'surface_m':surface,'kind':'baseline'}
        ranked=[]
        for Q in rotations:
            candidate=R[i]@B@Q@B.T
            if Rotation.from_matrix(candidate@R[i].T).magnitude()<1e-5:continue
            tt=align_projection_center(inputs,mesh,small_hd,i,candidate,t[i])
            c,_,_=hypothesis_cost(inputs,mesh,small_hd,i,candidate,tt,R[i]);ranked.append((c,candidate,tt))
        ranked.sort(key=lambda x:x[0]);choices=[baseline]
        for _,rr,tt in ranked[:3]:
            for _ in range(2):rr,tt,_=small_image_fit(rr,tt,inputs,mesh,hand_depth,i,8)
            c,im,surf=hypothesis_cost(inputs,mesh,hand_depth,i,rr,tt,R[i])
            choices.append({'R':rr,'t':tt,'cost':c,'iou':im['iou'],'surface_m':surf,'kind':'triangle_silhouette_hypothesis'})
        states.append(choices);baseline_cost.append(cost)
    path=select_path(states,inputs.times[keys]);selected=[row[j] for row,j in zip(states,path)]
    gain=float(np.mean(baseline_cost)-np.mean([c['cost'] for c in selected]));report['mean_unary_improvement']=gain
    report['selected']=[{'frame_idx':int(inputs.ids[k]),'candidate_index':j,'kind':c['kind'],'cost':c['cost'],'iou':c['iou'],'surface_m':c['surface_m']} for k,j,c in zip(keys,path,selected)]
    if gain<=1e-6 or all(j==0 for j in path):
        report['state']='insufficient_multiframe_shape_improvement';return R,t,tracks,tracking_rows,report
    rr,tt=interpolate_corrections(R,t,inputs.times,keys,selected)
    # Independent dense image check uses the same fixed mask domain, not just keyframes.
    before=[];after=[];surface_before=[];surface_after=[];cost_before=[];cost_after=[]
    for i in range(len(R)):
        c0,a,d0=hypothesis_cost(inputs,mesh,hand_depth,i,R[i],t[i],R[i]);c1,b,d1=hypothesis_cost(inputs,mesh,hand_depth,i,rr[i],tt[i],R[i])
        if a['pixels']>=15:
            before.append(a['iou']);after.append(b['iou']);cost_before.append(c0);cost_after.append(c1)
        if d0 is not None:surface_before.append(d0);surface_after.append(d1)
    report['dense_median_IoU_before']=float(np.median(before)) if before else None
    report['dense_median_IoU_after']=float(np.median(after)) if after else None
    report['dense_objective_before']=float(np.mean(cost_before)) if cost_before else None
    report['dense_objective_after']=float(np.mean(cost_after)) if cost_after else None
    if surface_before:
        report['dense_surface_before_m']=float(np.median(surface_before));report['dense_surface_after_m']=float(np.median(surface_after))
    accepted,reason=dense_acceptance(before,after,cost_before,cost_after,surface_before,surface_after)
    if not accepted:
        report['state']=reason;return R,t,tracks,tracking_rows,report
    # All old canonical bindings are invalid after a geometry-based mode correction.
    # Rebound rows are weak per-frame visual support, not claimed material tracks.
    frames=[];points=[];pixels=[];weights=[]
    for i in range(len(R)):
        ids=np.flatnonzero(tracks.frames==i)
        if not len(ids):continue
        rc,tc=camera_pose(rr[i],tt[i],inputs.cameras[i]);X,_,valid=mesh.query(tracks.uv[ids],inputs.K[i],rc,tc);j=ids[valid]
        frames.extend([i]*len(j));points.extend(X[valid]);pixels.extend(tracks.uv[j]);weights.extend(tracks.weights[j]*.15)
    rebound=ImageFactors(np.asarray(frames,int),np.asarray(points,float).reshape(-1,3),np.asarray(pixels,float).reshape(-1,2),np.asarray(weights,float),'shape_reanchored_visual_support_not_material_tracks')
    rows=[dict(r,geometry_relocalization='interpolated_triangle_shape_hypothesis') for r in tracking_rows]
    report.update(applied=True,state='applied_multiframe_shape_hypothesis',old_binding_rows_removed=len(tracks.frames),rebound_visual_rows=len(frames))
    return rr,tt,rebound,rows,report
