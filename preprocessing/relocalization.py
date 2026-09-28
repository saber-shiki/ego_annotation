"""Descriptor-keyframe relocalization with common held-out evidence.
The local tracker is left intact. Accepted corrections replace stale factor rows.
All evidence comes from the same monocular RGB/mask/native-geometry inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.geometry import camera_pose,world_pose,matrix_k,project,resized_intrinsics
from preprocessing.factors import ImageFactors,concatenate,subset


def reprojection_score(R,t,points,uv,k,weights=None,cap_px=12.):
    """Score every supplied observation, never only candidate-selected inliers."""
    points=np.asarray(points,float);uv=np.asarray(uv,float)
    if len(points)==0:return {'count':0,'cost_px':float(cap_px),'inlier_fraction':0.,'positive_fraction':0.}
    cam=points@np.asarray(R).T+np.asarray(t);positive=np.isfinite(cam).all(1)&(cam[:,2]>.01)
    errors=np.full(len(points),cap_px,float)
    errors[positive]=np.linalg.norm(project(cam[positive],k)-uv[positive],axis=1)
    w=np.ones(len(points)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(points),) or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('invalid evidence weights')
    cost=float(np.sum(np.minimum(errors,cap_px)*w)/w.sum())
    return {'count':len(points),'cost_px':cost,'inlier_fraction':float(np.sum(w*(positive&(errors<4.)))/w.sum()),'positive_fraction':float(np.sum(w*positive)/w.sum())}


def pnp_hypotheses(points,uv,k,initial_R,initial_t,max_candidates=8):
    X=np.ascontiguousarray(points,dtype=np.float64);q=np.ascontiguousarray(uv,dtype=np.float64)
    if len(X)<6 or X.shape!=(len(q),3) or q.shape!=(len(X),2):return []
    if not np.isfinite(X).all() or not np.isfinite(q).all():return []
    K=matrix_k(k);proposals=[];support=None
    r0=Rotation.from_matrix(initial_R).as_rotvec().reshape(3,1);t0=np.array(initial_t,float,copy=True).reshape(3,1)
    try:
        ok,r,t,ii=cv2.solvePnPRansac(X,q,K,None,r0.copy(),t0.copy(),True,iterationsCount=100,reprojectionError=3.,confidence=.995,flags=cv2.SOLVEPNP_EPNP)
        if ok and ii is not None and len(ii)>=6:
            support=ii.ravel();proposals.append((r,t,'ransac_epnp'))
    except cv2.error:pass
    # Other solvers use the SAME robust support, not all original outliers.
    if support is None:return []
    xs=np.ascontiguousarray(X[support]);qs=np.ascontiguousarray(q[support])
    try:
        ok,r,t=cv2.solvePnP(xs,qs,K,None,r0.copy(),t0.copy(),True,flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:proposals.append((r,t,'iterative_guess'))
    except cv2.error:pass
    try:
        output=cv2.solvePnPGeneric(xs,qs,K,None,flags=cv2.SOLVEPNP_SQPNP)
        for r,t in zip(output[1],output[2]):proposals.append((r,t,'sqpnp'))
    except cv2.error:pass
    rng=np.random.default_rng(47)
    for _ in range(3):
        ids=rng.choice(len(xs),4,replace=False)
        try:
            output=cv2.solvePnPGeneric(np.ascontiguousarray(xs[ids]),np.ascontiguousarray(qs[ids]),K,None,flags=cv2.SOLVEPNP_AP3P)
            for r,t in zip(output[1],output[2]):proposals.append((r,t,'ap3p'))
        except cv2.error:pass
    result=[]
    for r,t,tag in proposals:
        try:r,t=cv2.solvePnPRefineLM(xs,qs,K,None,np.array(r,copy=True),np.array(t,copy=True))
        except cv2.error:continue
        R=cv2.Rodrigues(r)[0];t=np.asarray(t).ravel();score=reprojection_score(R,t,X,q,k)
        if score['positive_fraction']<.95:continue
        if any(Rotation.from_matrix(R@v[0].T).magnitude()<.01 and np.linalg.norm(t-v[1])<.002 for v in result):continue
        cam=X@R.T+t;good=(cam[:,2]>.01)&(np.linalg.norm(project(cam,k)-q,axis=1)<4.)
        result.append((R,t,good,score['cost_px'],tag))
    return sorted(result,key=lambda v:v[3])[:max_candidates]


def diverse_keys(indices,anchor,target,maximum=7):
    keys=sorted(set(map(int,indices)))
    if target in keys:keys.remove(target)
    if len(keys)<=maximum:return keys
    chosen=[anchor] if anchor in keys else []
    nearest=sorted(keys,key=lambda k:abs(k-target))
    for k in nearest[:2]:
        if k not in chosen:chosen.append(k)
    for k in [keys[0],keys[-1]]:
        if k not in chosen:chosen.append(k)
    while len(chosen)<maximum:
        remaining=[k for k in keys if k not in chosen]
        if not remaining:break
        k=max(remaining,key=lambda z:min(abs(z-c) for c in chosen));chosen.append(k)
    return sorted(chosen[:maximum])


def mutual_matches(a,b,ratio=.75):
    if a is None or b is None or len(a)<2 or len(b)<2:return np.empty((0,2),int)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def accepted(x,y):
        out={}
        for pair in matcher.knnMatch(np.ascontiguousarray(x,np.float32),np.ascontiguousarray(y,np.float32),k=2):
            if len(pair)==2 and pair[0].distance<ratio*pair[1].distance:out[pair[0].queryIdx]=pair[0].trainIdx
        return out
    forward=accepted(a,b);back=accepted(b,a)
    return np.asarray([(i,j) for i,j in forward.items() if back.get(j)==i],int).reshape(-1,2)


def replace_frame_factors(original,accepted_frames,addition):
    keep=~np.isin(original.frames,np.asarray(list(accepted_frames),int))
    retained=ImageFactors(original.frames[keep].copy(),original.points[keep].copy(),original.uv[keep].copy(),original.weights[keep].copy(),original.family)
    return concatenate([retained,addition],'local_plus_verified_keyframes')


@dataclass
class Matches:
    reference: int
    points: np.ndarray
    source_uv: np.ndarray
    target_uv: np.ndarray
    target_ids: np.ndarray


class DescriptorBank:
    def __init__(self,inputs,mesh,hand_depth,R,t,key_indices):
        self.inputs=inputs;self.mesh=mesh;self.hand_depth=hand_depth;self.R=R;self.t=t
        self.detector=cv2.SIFT_create(nfeatures=400,contrastThreshold=.015,edgeThreshold=12)
        self.features={};self.keys={}
        for i in key_indices:
            uv,desc=self.extract(i)
            if len(uv)<12:continue
            rc,tc=camera_pose(R[i],t[i],inputs.cameras[i]);X,z,good=mesh.query(uv,inputs.K[i],rc,tc)
            hd=hand_depth[i];px=np.rint((uv+.5)*np.array(hd.shape[::-1])/np.array(inputs.wh)-.5).astype(int)
            px[:,0]=np.clip(px[:,0],0,hd.shape[1]-1);px[:,1]=np.clip(px[:,1],0,hd.shape[0]-1)
            good &= ~(hd[px[:,1],px[:,0]]+.005<z)
            if good.sum()>=12:self.keys[i]=(uv[good],desc[good],X[good])

    def extract(self,i):
        if i in self.features:return self.features[i]
        distance=cv2.distanceTransform(self.inputs.masks[i].astype('uint8'),cv2.DIST_L2,3)
        mask=(distance>4).astype('uint8')*255
        kp,desc=self.detector.detectAndCompute(self.inputs.grays[i],mask)
        if desc is None:result=(np.empty((0,2)),np.empty((0,128),np.float32))
        else:
            # Collapse duplicate orientations at the same image location.
            seen=set();ids=[]
            for j in sorted(range(len(kp)),key=lambda j:-kp[j].response):
                key=tuple(np.rint(np.array(kp[j].pt)/2).astype(int))
                if key not in seen:seen.add(key);ids.append(j)
            uv=np.asarray([kp[j].pt for j in ids],float);desc=desc[ids].astype(np.float32)
            desc=np.sqrt(desc/np.maximum(desc.sum(1,keepdims=True),1e-12))
            result=(uv,desc)
        self.features[i]=result;return result

    def match(self,reference,target):
        uv0,d0,X=self.keys[reference];uv1,d1=self.extract(target);pairs=mutual_matches(d0,d1)
        if len(pairs)<12:return None
        # Reciprocal descriptor identity is independent of any pose proposal.
        return Matches(reference,X[pairs[:,0]],uv0[pairs[:,0]],uv1[pairs[:,1]],pairs[:,1])


def silhouette_score(inputs,mesh,hand_depth,i,R,t):
    wh=hand_depth.shape[1:][::-1];k=resized_intrinsics(inputs.K[i],inputs.wh,wh);r,tt=camera_pose(R,t,inputs.cameras[i])
    _,_,mask=mesh.raster(k,r,tt,wh)
    target=cv2.resize(inputs.masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
    # A fixed camera/hand domain for every candidate; a wrong pose cannot hide errors
    # by changing its own occlusion denominator.
    domain=~(cv2.dilate(np.isfinite(hand_depth[i]).astype('uint8'),np.ones((3,3),np.uint8))>0)
    active=int(np.count_nonzero(target&domain));union=np.count_nonzero((mask|target)&domain)
    return {'pixels':active,'iou':float(np.count_nonzero(mask&target&domain)/max(1,union))}


def surface_score(inputs,mesh,i,R,t):
    q=subset(inputs.clouds[i],32)
    if not len(q):return None
    C=inputs.cameras[i];x=(q@C[:3,:3].T+C[:3,3]-t)@R
    _,d=mesh.closest(x)
    return float(np.median(d))


def relocalize(inputs,mesh,hand_depth,R,t,tracks,tracking_rows):
    base_R=R.copy();base_t=t.copy();result_R=R.copy();result_t=t.copy();n=len(R)
    keys=sorted(set(range(0,n,12))|{inputs.anchor,n-1})
    bank=DescriptorBank(inputs,mesh,hand_depth,base_R,base_t,keys)
    records=[];new_factors=[];accepted=[];loop_frames=[]
    for i in range(n):
        record={'frame_idx':int(inputs.ids[i]),'state':'insufficient_independent_matches','accepted':False}
        groups=[bank.match(k,i) for k in diverse_keys(bank.keys,inputs.anchor,i)]
        groups=[g for g in groups if g is not None]
        if not groups:records.append(record);continue
        X=np.vstack([g.points for g in groups]);uv=np.vstack([g.target_uv for g in groups]);target_ids=np.concatenate([g.target_ids for g in groups])
        weights=np.concatenate([np.full(len(g.points),1/len(g.points)) for g in groups])
        # A material identity goes wholly into train or validation across all refs.
        hold=target_ids%3==0;train=~hold
        record.update(references=[int(inputs.ids[g.reference]) for g in groups],matches=len(X),validation_points=int(hold.sum()))
        if len(np.unique(target_ids[hold]))<4 or len(np.unique(target_ids[train]))<8:
            records.append(record);continue
        rc,tc=camera_pose(base_R[i],base_t[i],inputs.cameras[i])
        original=reprojection_score(rc,tc,X[hold],uv[hold],inputs.K[i],weights[hold]);record['baseline_validation']=original
        candidates=pnp_hypotheses(X[train],uv[train],inputs.K[i],rc,tc)
        for g in groups:
            use=g.target_ids%3!=0
            if use.sum()>=6:candidates+=pnp_hypotheses(g.points[use],g.target_uv[use],inputs.K[i],rc,tc,max_candidates=3)
        scored=[]
        for rr,tt,_,_,tag in candidates:
            score=reprojection_score(rr,tt,X[hold],uv[hold],inputs.K[i],weights[hold])
            if score['positive_fraction']<.98 or score['inlier_fraction']<.6:continue
            scored.append((score['cost_px'],rr,tt,score,tag))
        scored.sort(key=lambda item:item[0]);record['candidate_count']=len(candidates)
        adopted=False
        if scored and i not in bank.keys:
            cost,rr,tt,score,tag=scored[0]
            # Selection uses the same held-out pixels as the baseline, not own inliers.
            if cost<original['cost_px']*.70 and original['cost_px']-cost>.5:
                wr,wt=world_pose(rr,tt,inputs.cameras[i]);si0=silhouette_score(inputs,mesh,hand_depth,i,base_R[i],base_t[i]);si1=silhouette_score(inputs,mesh,hand_depth,i,wr,wt)
                d0=surface_score(inputs,mesh,i,base_R[i],base_t[i]);d1=surface_score(inputs,mesh,i,wr,wt)
                image_ok=si0['pixels']<15 or si1['iou']>=si0['iou']-.02
                depth_ok=d0 is None or d1<=d0*1.1+.003
                record.update(candidate_validation=score,candidate_solver=tag,silhouette_before=si0,silhouette_after=si1,surface_before_m=d0,surface_after_m=d1)
                if image_ok and depth_ok:
                    result_R[i]=wr;result_t[i]=wt;accepted.append(i);adopted=True
                    record.update(state='accepted_descriptor_relocalization',accepted=True,old_frame_bindings_replaced=True)
                else:record['state']='rejected_independent_geometry_disagreement'
            else:record['state']='no_heldout_improvement'
        else:record['state']='anchor_preserved_or_no_valid_hypothesis'
        cr,ct=camera_pose(result_R[i],result_t[i],inputs.cameras[i])
        all_score=reprojection_score(cr,ct,X,uv,inputs.K[i],weights)
        if adopted or all_score['cost_px']<2.5:
            added=0
            for g in groups:
                cam=g.points@cr.T+ct;good=(cam[:,2]>.01)&(np.linalg.norm(project(cam,inputs.K[i])-g.target_uv,axis=1)<4.)
                if good.sum()<8:continue
                quality=np.full(int(good.sum()),.6)
                # Store BOTH endpoints. These are genuinely consumed by joint refinement.
                new_factors.append(ImageFactors(np.full(good.sum(),g.reference,int),g.points[good].copy(),g.source_uv[good].copy(),quality.copy(),'descriptor_loop_source'))
                new_factors.append(ImageFactors(np.full(good.sum(),i,int),g.points[good].copy(),g.target_uv[good].copy(),quality.copy(),'descriptor_loop_target'))
                added+=int(good.sum())
            record['backend_correspondences']=added
            if added:loop_frames.append(i)
        records.append(record)
    additions=concatenate(new_factors,'verified_descriptor_keyframes')
    result_tracks=replace_frame_factors(tracks,accepted,additions)
    rows=[dict(row) for row in tracking_rows]
    for i,record in enumerate(records):
        rows[i]['descriptor_relocalization']=record
        if record['accepted']:rows[i]['kind']='descriptor_keyframe_relocalization'
    summary={'method':'mutual_interior_RootSIFT_keyframes_common_heldout_validation',
             'keyframes':[int(inputs.ids[k]) for k in bank.keys],'accepted_frames':[int(inputs.ids[i]) for i in accepted],
             'loop_supported_frames':[int(inputs.ids[i]) for i in loop_frames],'backend_factor_rows':len(additions.frames),
             'stale_factor_rows_removed':int(np.isin(tracks.frames,accepted).sum()),'absolute_pose_accuracy_certified':False,'frames':records}
    return result_R,result_t,result_tracks,rows,summary
