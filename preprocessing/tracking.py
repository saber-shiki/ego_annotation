"""Monocular model-bound tracking with native triangles and persistent LK tracks."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from preprocessing.geometry import camera_pose,world_pose,matrix_k,project,TriangleMesh,resized_intrinsics
from preprocessing.factors import ImageFactors,concatenate,contour_factors
# Relocalization candidates are separate from the unchanged robust local tracker.
from preprocessing.relocalization import pnp_hypotheses as solve_pnp_hypotheses


def projection_quality(points,uv,R,t,camera,k):
    if not len(points):return None
    rc,tc=camera_pose(R,t,camera);cam=points@rc.T+tc
    error=np.linalg.norm(project(cam,k)-uv,axis=1)
    return {'rms_px':float(np.sqrt(np.mean(error**2))),'median_px':float(np.median(error)),
            'p90_px':float(np.percentile(error,90)),'positive_fraction':float(np.mean(cam[:,2]>.01))}


def fresh_mask_iou(inputs,mesh,hand_depth,i,R,t):
    wh=hand_depth.shape[1:][::-1];k=resized_intrinsics(inputs.K[i],inputs.wh,wh)
    rc,tc=camera_pose(R,t,inputs.cameras[i]);_,_,rendered=mesh.raster(k,rc,tc,wh)
    target=cv2.resize(inputs.masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
    owned=getattr(inputs,'owned_masks',None)
    owned=np.zeros_like(target) if owned is None else cv2.resize(owned[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
    # The comparison domain is fixed by source evidence, never by candidate depth.
    domain=~np.isfinite(hand_depth[i])|owned
    if not np.any(target&domain):return None
    return float(np.count_nonzero(rendered&target&domain)/max(1,np.count_nonzero((rendered|target)&domain)))


def refinement_consistent(before,after,mask_before,mask_after,pixel_noise=.75):
    if before is not None:
        if after is None or after['positive_fraction']<before['positive_fraction']:return False,'lost_projectable_tracks'
        if after['rms_px']>np.hypot(before['rms_px'],pixel_noise):return False,'tracked_reprojection_worsened'
        if after['p90_px']>before['p90_px']+pixel_noise:return False,'tracked_reprojection_tail_worsened'
    if mask_before is not None and (mask_after is None or mask_after<mask_before-1e-6):return False,'whole_visible_mask_worsened'
    return True,'joint_evidence_consistent'


def small_image_fit(R,t,inputs,mesh,hand_depth,i,iterations=10,*,support_points=None,support_uv=None):
    """Joint image refinement; contour-only success cannot overwrite good tracking."""
    rr=inputs.source_R.copy();tt=inputs.source_t.copy();rr[i]=R;tt[i]=t
    factors,stats=contour_factors(mesh,rr,tt,inputs,hand_depth,[i],32)
    if len(factors.points)<6:return R,t,{'available':False,'reason':'no_visible_contour_support'}
    X=np.empty((0,3)) if support_points is None else np.asarray(support_points,float)
    uv=np.empty((0,2)) if support_uv is None else np.asarray(support_uv,float)
    if X.shape!=(len(uv),3) or uv.shape!=(len(X),2):raise ValueError('tracking evidence shape mismatch')
    C=inputs.cameras[i];k=inputs.K[i];base_R=R.copy();base_t=t.copy()
    def pose(x):return Rotation.from_rotvec(x[:3]).as_matrix()@base_R,base_t+x[3:]
    def residual(x):
        r,tt=pose(x);cam=(factors.points@r.T+tt-C[:3,3])@C[:3,:3]
        parts=[((project(cam,k)-factors.uv)/3.*np.sqrt(factors.weights[:,None])).ravel(),x[:3]/.25,x[3:]/.05,np.minimum(cam[:,2]-.01,0)*100]
        if len(X):
            p=(X@r.T+tt-C[:3,3])@C[:3,:3];parts.append(((project(p,k)-uv)/1.5).ravel())
        return np.concatenate(parts)
    zero=np.zeros(6);before_res=residual(zero);before_quality=projection_quality(X,uv,R,t,C,k)
    before_mask=fresh_mask_iou(inputs,mesh,hand_depth,i,R,t)
    fit=least_squares(residual,zero,loss='soft_l1',max_nfev=iterations,x_scale=np.r_[np.ones(3)*.1,np.ones(3)*.01])
    if not np.isfinite(fit.x).all():return R,t,{'available':False,'reason':'nonfinite_fit'}
    selected=zero;reason='no_consistent_update';after_quality=before_quality;after_mask=before_mask;fraction=0.
    for f in (1.,.5,.25,.125):
        trial=fit.x*f;r,tt=pose(trial);q=projection_quality(X,uv,r,tt,C,k);miou=fresh_mask_iou(inputs,mesh,hand_depth,i,r,tt)
        valid,why=refinement_consistent(before_quality,q,before_mask,miou)
        if valid:selected=trial;fraction=f;after_quality=q;after_mask=miou;reason=why;break
        reason=why
    r,tt=pose(selected)
    return r,tt,{'available':True,'nfev':fit.nfev,'joint_tracking_points':len(X),'accepted_step_fraction':fraction,'reason':reason,
        'initial_residual_rms':float(np.sqrt(np.mean(before_res**2))),'final_residual_rms':float(np.sqrt(np.mean(residual(selected)**2))),
        'tracked_reprojection_before':before_quality,'tracked_reprojection_after':after_quality,
        'fixed_mask_IoU_before':before_mask,'fixed_mask_IoU_after':after_mask}


def contour_pose_proposal(R,t,inputs,mesh,hand_depth,i,iterations=10):
    """Local visual registration; not a centroid substituted for rigid pose."""
    rr=inputs.source_R.copy();tt=inputs.source_t.copy();rr[i]=R;tt[i]=t
    factors,stats=contour_factors(mesh,rr,tt,inputs,hand_depth,[i],32)
    if len(factors.points)<6:return R,t,{'available':False}
    C=inputs.cameras[i];k=inputs.K[i];base_R=R.copy();base_t=t.copy()
    def residual(x):
        r=Rotation.from_rotvec(x[:3]).as_matrix()@base_R;v=factors.points@r.T+base_t+x[3:]
        cam=(v-C[:3,3])@C[:3,:3]
        return np.r_[((project(cam,k)-factors.uv)/3.*np.sqrt(factors.weights[:,None])).ravel(),x[:3]/.25,x[3:]/.05,np.minimum(cam[:,2]-.01,0)*100]
    before=residual(np.zeros(6));fit=least_squares(residual,np.zeros(6),loss='soft_l1',max_nfev=iterations,x_scale=np.r_[np.ones(3)*.1,np.ones(3)*.01])
    if not np.isfinite(fit.x).all():return R,t,{'available':False,'reason':'nonfinite_fit'}
    return Rotation.from_rotvec(fit.x[:3]).as_matrix()@R,t+fit.x[3:],{'available':True,'nfev':fit.nfev,'initial_residual_rms':float(np.sqrt(np.mean(before**2))),'final_residual_rms':float(np.sqrt(np.mean(residual(fit.x)**2)))}


def guarded_contour_output(before_R,before_t,after_R,after_t,inputs,mesh,hand_depth,i,X,uv,iterations,raw_fit):
    before=projection_quality(X,uv,before_R,before_t,inputs.cameras[i],inputs.K[i])
    after=projection_quality(X,uv,after_R,after_t,inputs.cameras[i],inputs.K[i])
    mb=fresh_mask_iou(inputs,mesh,hand_depth,i,before_R,before_t)
    ma=fresh_mask_iou(inputs,mesh,hand_depth,i,after_R,after_t)
    tracks_ok,why=refinement_consistent(before,after,None,None)
    # Only a contradiction across BOTH independent evidence families diagnoses
    # a corrupt contour overwrite. Mask-improving corrections may legitimately
    # repair stale 3D bindings; do not forbid them just for moving their pixels.
    corrupt=before is not None and not tracks_ok and mb is not None and ma is not None and ma<mb-1e-6
    if corrupt:
        R,t,report=small_image_fit(before_R,before_t,inputs,mesh,hand_depth,i,iterations,support_points=X,support_uv=uv)
    else:R,t,report=after_R,after_t,dict(raw_fit)
    report.update(contour_candidate_rejected=bool(corrupt),legacy_tracked_reprojection_before=before,
                  legacy_tracked_reprojection_after=after,legacy_fixed_mask_IoU_before=mb,legacy_fixed_mask_IoU_after=ma,
                  proposal_state_preserved=True)
    return R,t,report


def seed_features(inputs,mesh,i,R,t,existing_uv=None,count=160):
    mask=cv2.erode(inputs.masks[i].astype('uint8'),np.ones((5,5),np.uint8))*255
    if existing_uv is not None:
        for p in existing_uv:cv2.circle(mask,tuple(np.rint(p).astype(int)),6,0,-1)
    uv=cv2.goodFeaturesToTrack(inputs.grays[i],maxCorners=count,qualityLevel=.008,minDistance=5,mask=mask,blockSize=5)
    if uv is None:return np.zeros((0,2)),np.zeros((0,3))
    uv=uv.reshape(-1,2);rc,tc=camera_pose(R,t,inputs.cameras[i]);points,z,good=mesh.query(uv,inputs.K[i],rc,tc)
    return uv[good],points[good]


def flow_track(inputs,a,b,uv):
    if not len(uv):return np.zeros((0,2)),np.zeros(0,bool)
    p0=np.asarray(uv,np.float32).reshape(-1,1,2)
    p1,ok,err=cv2.calcOpticalFlowPyrLK(inputs.grays[a],inputs.grays[b],p0,None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    if p1 is None:return np.zeros_like(uv),np.zeros(len(uv),bool)
    back,bok,_=cv2.calcOpticalFlowPyrLK(inputs.grays[b],inputs.grays[a],p1,None,winSize=(25,25),maxLevel=3)
    p=p1.reshape(-1,2);valid=ok.ravel().astype(bool)&bok.ravel().astype(bool)&np.isfinite(p).all(1)&(np.linalg.norm(back.reshape(-1,2)-uv,axis=1)<1.5)
    h,w=inputs.grays[b].shape;px=np.rint(np.nan_to_num(p)).astype(int);valid&=(px[:,0]>=0)&(px[:,0]<w)&(px[:,1]>=0)&(px[:,1]<h)
    ii=np.flatnonzero(valid);inside=np.zeros(len(uv),bool);inside[ii]=inputs.masks[b,px[ii,1],px[ii,0]]
    return p,valid&inside


def solve_pnp(points,uv,k,initial_R,initial_t):
    if len(points)<6:return None
    rvec=Rotation.from_matrix(initial_R).as_rotvec().reshape(3,1);tvec=np.array(initial_t,float).reshape(3,1)
    try:
        ok,r,t,inliers=cv2.solvePnPRansac(np.asarray(points,np.float64),np.asarray(uv,np.float64),matrix_k(k),None,rvec,tvec,True,iterationsCount=100,reprojectionError=3.,confidence=.995,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<6:return None
        ii=inliers.ravel();r,t=cv2.solvePnPRefineLM(np.asarray(points[ii],float),np.asarray(uv[ii],float),matrix_k(k),None,r,t)
        R=cv2.Rodrigues(r)[0];t=t.ravel();cam=points@R.T+t
        if not np.isfinite(t).all() or np.mean(cam[:,2]>.01)<.95:return None
        err=np.linalg.norm(project(cam,k)-uv,axis=1)
        return R,t,ii,float(np.median(err[ii]))
    except cv2.error:return None


def bootstrap(inputs,hand_depth,frame_fit_iterations=8,max_features=160,*,refine_anchor=True,guarded_output=False):
    mesh=TriangleMesh(inputs.vertices,inputs.faces);n=len(inputs.ids);a=inputs.anchor
    R=inputs.source_R.copy();t=inputs.source_t.copy();rows=[None]*n;factors=[]
    if refine_anchor:
        R[a],t[a],anchorfit=contour_pose_proposal(R[a],t[a],inputs,mesh,hand_depth,a,15)
    else:
        anchorfit={'available':True,'source':'trusted_visible_surface_scale_translation_alignment','image_only_refit_skipped':True}
    rows[a]={'frame':int(inputs.ids[a]),'kind':'native_appearance_anchor_visual_fit' if refine_anchor else 'visible_surface_aligned_anchor','pnp_inliers':0,'fit':anchorfit}
    output_R=R.copy();output_t=t.copy()
    if guarded_output:rows[a]['unfiltered_contour_pose']={'rotation_world':R[a].tolist(),'translation_world':t[a].tolist()}
    # Direct world rotation proposals only, metric translations are intentionally not chained.
    positions={int(f):i for i,f in enumerate(inputs.ids)};edges={}
    for e in inputs.rgb.edges:
        aa=positions[e.source_frame_idx];bb=positions[e.target_frame_idx]
        edges[aa,bb]=e.rotation_source_to_target_world;edges[bb,aa]=e.rotation_source_to_target_world.T
    for direction in (1,-1):
        uv,X=seed_features(inputs,mesh,a,R[a],t[a],count=max_features)
        if len(X):factors.append(ImageFactors(np.full(len(X),a),X.copy(),uv.copy(),np.ones(len(X)), 'persistent_LK'))
        previous=a
        for i in range(a+direction,n if direction>0 else -1,direction):
            current,valid=flow_track(inputs,previous,i,uv);X=X[valid];uv=current[valid]
            # Previous world state: a prediction, not a fake observation in a missing row.
            R[i]=edges[previous,i]@R[previous] if (previous,i) in edges else R[previous]
            t[i]=t[previous]
            rc,tc=camera_pose(R[i],t[i],inputs.cameras[i]);result=solve_pnp(X,uv,inputs.K[i],rc,tc)
            support=0;kind='temporal_inference_visual_refit';error=None
            if result is not None:
                rc,tc,ii,error=result;R[i],t[i]=world_pose(rc,tc,inputs.cameras[i]);X=X[ii];uv=uv[ii];support=len(ii);kind='model_bound_LK_PnP'
            else:
                # Retain the tracked 2D evidence for the global pass, but reduce its influence.
                pass
            before_R=R[i].copy();before_t=t[i].copy()
            R[i],t[i],fit=contour_pose_proposal(R[i],t[i],inputs,mesh,hand_depth,i,frame_fit_iterations)
            if guarded_output and result is not None:
                output_R[i],output_t[i],fit=guarded_contour_output(before_R,before_t,R[i],t[i],inputs,mesh,hand_depth,i,X,uv,frame_fit_iterations,fit)
            else:output_R[i]=R[i];output_t[i]=t[i]
            if len(X):factors.append(ImageFactors(np.full(len(X),i),X.copy(),uv.copy(),np.full(len(X),1. if result else .15),'persistent_LK'))
            keyframe=(abs(i-a)%8==0 or len(uv)<max_features//2)
            if keyframe:
                more,points=seed_features(inputs,mesh,i,R[i],t[i],uv,count=max(0,max_features-len(uv))) if len(uv)<max_features else (np.zeros((0,2)),np.zeros((0,3)))
                if len(more):
                    uv=np.vstack((uv,more));X=np.vstack((X,points));factors.append(ImageFactors(np.full(len(points),i),points.copy(),more.copy(),np.ones(len(points))*.7,'keyframe_replenishment'))
            rows[i]={'frame':int(inputs.ids[i]),'kind':kind,'pnp_inliers':support,'tracked_points':len(X),'pnp_median_px':error,'keyframe_replenished':bool(keyframe),'fit':fit}
            if guarded_output:rows[i]['unfiltered_contour_pose']={'rotation_world':R[i].tolist(),'translation_world':t[i].tolist()}
            previous=i
    return output_R,output_t,concatenate(factors,'persistent_LK_and_keyframes'),rows
