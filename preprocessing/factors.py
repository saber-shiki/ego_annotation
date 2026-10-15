"""Visibility-aware image/depth correspondences from current real triangle geometry."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial import cKDTree
from preprocessing.geometry import TriangleMesh,camera_pose,resized_intrinsics,project


@dataclass
class ImageFactors:
    frames: np.ndarray
    points: np.ndarray
    uv: np.ndarray
    weights: np.ndarray
    family: str

    @classmethod
    def empty(cls,family='empty'):
        return cls(np.zeros(0,int),np.zeros((0,3)),np.zeros((0,2)),np.zeros(0),family)


def concatenate(items,family='image'):
    items=[x for x in items if len(x.frames)]
    if not items:return ImageFactors.empty(family)
    return ImageFactors(*(np.concatenate([getattr(x,k) for x in items]) for k in ('frames','points','uv','weights')),family)


def hand_depths(inputs,width=128):
    wh=(width,round(width*inputs.wh[1]/inputs.wh[0]));all_depth=[]
    for i in range(len(inputs.ids)):
        depth=np.full(wh[::-1],np.inf);k=resized_intrinsics(inputs.K[i],inputs.wh,wh);C=inputs.cameras[i]
        for hand in inputs.hands.values():
            if not hand['valid'][i]:continue
            verts=(hand['vertices'][i]-C[:3,3])@C[:3,:3]
            if not np.isfinite(verts).all():continue
            faces=hand['faces'][i] if hand['faces'].ndim==3 else hand['faces']
            mesh=TriangleMesh(verts,faces);_,z,valid=mesh.raster(k,np.eye(3),np.zeros(3),wh)
            depth=np.minimum(depth,z)
        all_depth.append(depth)
    return np.asarray(all_depth)


def boundary(mask):
    return mask & ~(cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8))>0)


def subset(a,n):
    if len(a)<=n:return a
    return a[np.linspace(0,len(a)-1,n).astype(int)]


def contour_factors(mesh,world_R,world_t,inputs,hand_depth,indices=None,max_points=48):
    rows=[];stats=[];indices=range(len(inputs.ids)) if indices is None else indices
    wh=hand_depth.shape[1:][::-1]
    for i in indices:
        k=resized_intrinsics(inputs.K[i],inputs.wh,wh);R,t=camera_pose(world_R[i],world_t[i],inputs.cameras[i])
        points,depth,rendered=mesh.raster(k,R,t,wh)
        target=cv2.resize(inputs.masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
        hd=hand_depth[i]
        # Exclude hand-touching mask edges rather than force a full mesh into visible-only mask.
        hand=cv2.dilate(np.isfinite(hd).astype('uint8'),np.ones((3,3),np.uint8))>0
        front_hand=np.isfinite(hd)&(hd<depth+.01)
        mb=boundary(rendered)&~front_hand;tb=boundary(target)&~hand
        my,mx=np.nonzero(mb);ty,tx=np.nonzero(tb)
        if len(mx)<3 or len(tx)<3:
            stats.append({'frame':int(inputs.ids[i]),'contour_samples':0,'reason':'no_visible_contour_support'});continue
        mxy=np.c_[mx,my];txy=np.c_[tx,ty]
        mi=subset(np.arange(len(mxy)),max_points);ti=subset(np.arange(len(txy)),max_points)
        dist,j=cKDTree(txy).query(mxy[mi]);rd,kidx=cKDTree(mxy).query(txy[ti])
        x=np.r_[points[my[mi],mx[mi]],points[my[kidx],mx[kidx]]]
        uv=np.r_[txy[j],txy[ti]].astype(float)
        uv=(uv+.5)*(np.asarray(inputs.wh)/np.asarray(wh))-.5
        w=np.r_[np.ones(len(mi)),np.ones(len(ti))]*.35
        rows.append(ImageFactors(np.full(len(x),i,int),x,uv,w,'visible_contour'))
        union=np.count_nonzero((rendered|target)&~front_hand)
        stats.append({'frame':int(inputs.ids[i]),'contour_samples':len(x),'visible_IoU':float(np.count_nonzero(rendered&target&~front_hand)/max(1,union))})
    return concatenate(rows,'visible_contour'),stats


def mask_alignment(mesh,world_R,world_t,inputs,hand_depth,i):
    """Full-raster evidence used only by the opt-in mask experiment/evaluator."""
    from preprocessing.mask_contract import FixedMaskDomain
    return FixedMaskDomain.from_inputs(inputs,hand_depth,i).render_score(mesh,world_R,world_t,inputs,i)


def mask_nonregression(before,after,tolerance=.015):
    """Experimental fixed-reference comparison; caller must retain the reference."""
    from preprocessing.mask_contract import nonregression
    return nonregression(before,after,tolerance=tolerance)


def depth_factors(mesh,R,t,inputs,max_points=20):
    frames=[];points=[];depths=[]
    for i,cloud in enumerate(inputs.clouds):
        if not len(cloud):continue
        cloud=subset(cloud,max_points*3);uv=project(cloud,inputs.K[i]);rr,tt=camera_pose(R[i],t[i],inputs.cameras[i])
        x,z,valid=mesh.query(uv,inputs.K[i],rr,tt)
        choose=subset(np.flatnonzero(valid),max_points)
        frames.extend([i]*len(choose));points.extend(x[choose]);depths.extend(cloud[choose,2])
    return np.asarray(frames,int),np.asarray(points,float).reshape(-1,3),np.asarray(depths,float)


def rgb_model_factors(mesh,R,t,inputs,max_points=48):
    """Rebind source RGB rays to candidate triangles, never chain noisy source depth t."""
    positions={int(f):i for i,f in enumerate(inputs.ids)};items=[]
    for e in inputs.rgb.edges:
        a=positions[e.source_frame_idx];b=positions[e.target_frame_idx]
        uv0=(e.source_uv+.5)*(np.asarray(inputs.wh)/e.source_size_wh)-.5
        uv1=(e.target_uv+.5)*(np.asarray(inputs.wh)/e.target_size_wh)-.5
        ar,at=camera_pose(R[a],t[a],inputs.cameras[a]);x,z,valid=mesh.query(uv0,inputs.K[a],ar,at)
        ids=subset(np.flatnonzero(valid),max_points)
        if not len(ids):continue
        quality=float(e.diagnostics['inlier_fraction'])*np.exp(-float(e.diagnostics['reprojection_median_px'])/3.)
        # These remain model-dependent uncertain correspondences, not metric measurements.
        weight=np.asarray(e.weights)[ids]*max(.03,quality)*.6
        for idx,uv in [(a,uv0),(b,uv1)]:items.append(ImageFactors(np.full(len(ids),idx),x[ids],uv[ids],weight,'RGB_model_reprojection'))
    return concatenate(items,'RGB_model_reprojection')


def observed_surface_factors(mesh,R,t,inputs,max_points=32):
    """Unsigned observed-to-triangle registration remains active even for missed rays."""
    frames=[];model=[];observed=[]
    for i,cloud in enumerate(inputs.clouds):
        if not len(cloud):continue
        q=subset(cloud,max_points*2);C=inputs.cameras[i]
        canonical=(q@C[:3,:3].T+C[:3,3]-t[i])@R[i]
        nearest,dist=mesh.closest(canonical)
        keep=np.flatnonzero(dist<=np.percentile(dist,80));keep=subset(keep,max_points)
        frames.extend([i]*len(keep));model.extend(nearest[keep]);observed.extend(q[keep])
    return np.asarray(frames,int),np.asarray(model,float).reshape(-1,3),np.asarray(observed,float).reshape(-1,3)
