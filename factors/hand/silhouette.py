"""Differentiable residuals for independent RGB hand silhouettes."""
from __future__ import annotations
import cv2
import numpy as np
import torch


def mask_contour(mask: np.ndarray, sample_count: int=160) -> np.ndarray:
    """Return every visible connected boundary, preserving holes/islands."""
    mask=np.asarray(mask).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours=[c.reshape(-1,2).astype(np.float32) for c in contours if len(c)>=2]
    if not contours: return np.zeros((0,2),np.float32)
    lengths=[float(np.linalg.norm(np.roll(c,-1,axis=0)-c,axis=1).sum()) for c in contours]
    total=sum(lengths)
    if total<=0: return np.concatenate(contours,axis=0)
    out=[]
    for contour,length in zip(contours,lengths):
        count=max(2,int(round(sample_count*length/total)))
        edge=np.roll(contour,-1,axis=0)-contour
        el=np.linalg.norm(edge,axis=1); cum=np.r_[0.,np.cumsum(el)]
        pos=np.linspace(0,float(cum[-1]),count,endpoint=False)
        ix=np.searchsorted(cum,pos,side='right')-1
        frac=(pos-cum[ix])/np.maximum(el[ix],1e-6)
        out.append(contour[ix]+frac[:,None]*edge[ix])
    return np.concatenate(out,axis=0).astype(np.float32)


def triangle_edge_points(vertices_uv: torch.Tensor, faces: np.ndarray|torch.Tensor, samples_per_edge: int=8) -> torch.Tensor:
    """Sample all three edges of every original triangle, retaining topology."""
    f=torch.as_tensor(faces,device=vertices_uv.device,dtype=torch.long)
    edges=torch.unique(torch.sort(torch.cat((f[:,[0,1]], f[:,[1,2]], f[:,[2,0]]),dim=0),dim=1).values,dim=0)
    a=vertices_uv[edges[:,0]]; b=vertices_uv[edges[:,1]]
    fractions=torch.linspace(0,1,max(2,int(samples_per_edge)),device=vertices_uv.device)
    return (a[:,None,:]*(1-fractions[None,:,None])+b[:,None,:]*fractions[None,:,None]).reshape(-1,2)


def smooth_robust_squared(residual: torch.Tensor, transition_px: float=5.) -> torch.Tensor:
    """Pseudo-Huber squared residual with nonzero tails beyond transition_px."""
    scale=torch.as_tensor(float(transition_px),device=residual.device,dtype=residual.dtype)
    x=residual.square()/scale.square()
    return 2*scale.square()*(torch.sqrt(1+x)-1)


def bidirectional_contour_residual(candidate_uv: torch.Tensor, raw_contour: torch.Tensor,
                                   visibility: torch.Tensor|float=1., uncertainty: torch.Tensor|float=0.,
                                   robust_px: float=5.) -> torch.Tensor:
    if candidate_uv.numel()==0 or raw_contour.numel()==0:return candidate_uv.sum()*0.
    weight=torch.as_tensor(visibility,device=candidate_uv.device,dtype=candidate_uv.dtype)
    weight=weight*(1-torch.as_tensor(uncertainty,device=candidate_uv.device,dtype=candidate_uv.dtype)).clamp(0,1)
    with torch.no_grad():
        fi=torch.cdist(candidate_uv.detach(),raw_contour.detach()).argmin(dim=1)
        ri=torch.cdist(raw_contour.detach(),candidate_uv.detach()).argmin(dim=1)
    a=torch.linalg.vector_norm(candidate_uv-raw_contour.detach()[fi],dim=1)
    b=torch.linalg.vector_norm(candidate_uv[ri]-raw_contour.detach(),dim=1)
    return weight*(smooth_robust_squared(a,robust_px).mean()+smooth_robust_squared(b,robust_px).mean())/2


def visible_boundary_associations(entities: list[dict], target_label: int,
                                  size_wh: tuple[int,int], max_samples: int=320) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """CPU z-buffer raycast returning target triangle/barycentric visible boundary.

    Each entity supplies ``vertices_camera`` (metric XYZ), ``faces`` and an
    integer ``label``.  Screen-space rasterization uses perspective-correct
    barycentric weights, so the detached associations can be replayed against
    live MANO vertices without turning internal triangle edges into targets.
    """
    width,height=map(int,size_wh)
    depth=np.full((height,width),np.inf,np.float32)
    owner=np.full((height,width),-1,np.int32)
    face_map=np.full((height,width),-1,np.int32)
    bary_map=np.zeros((height,width,3),np.float32)
    target_faces=None
    for entity in entities:
        xyz=np.asarray(entity['vertices_camera'],np.float64)
        faces=np.asarray(entity['faces'],np.int64)
        uv=np.asarray(entity['vertices_uv'],np.float64)
        label=int(entity['label'])
        if label==target_label: target_faces=faces
        for face_id,face in enumerate(faces):
            z=xyz[face,2]; p=uv[face]
            if np.any(z<=1e-6) or not np.isfinite(p).all(): continue
            xmin=max(0,int(np.floor(p[:,0].min()))); xmax=min(width-1,int(np.ceil(p[:,0].max())))
            ymin=max(0,int(np.floor(p[:,1].min()))); ymax=min(height-1,int(np.ceil(p[:,1].max())))
            if xmax<xmin or ymax<ymin: continue
            den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
            if abs(den)<1e-9: continue
            yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
            x=xx.astype(np.float64); y=yy.astype(np.float64)
            b0=((p[1,1]-p[2,1])*(x-p[2,0])+(p[2,0]-p[1,0])*(y-p[2,1]))/den
            b1=((p[2,1]-p[0,1])*(x-p[2,0])+(p[0,0]-p[2,0])*(y-p[2,1]))/den
            b2=1.-b0-b1
            inside=(b0>=-1e-6)&(b1>=-1e-6)&(b2>=-1e-6)
            inv=b0/z[0]+b1/z[1]+b2/z[2]
            zz=np.where(inv>0,1./inv,np.inf)
            sub=depth[ymin:ymax+1,xmin:xmax+1]
            take=inside&(zz<sub)
            if not np.any(take): continue
            # Perspective-correct barycentrics interpolate the live 3D ray hit.
            pc=np.stack((b0/z[0],b1/z[1],b2/z[2]),axis=-1)/inv[...,None]
            sub[take]=zz[take].astype(np.float32)
            owner[ymin:ymax+1,xmin:xmax+1][take]=label
            face_map[ymin:ymax+1,xmin:xmax+1][take]=face_id
            bary_map[ymin:ymax+1,xmin:xmax+1][take]=pc[take].astype(np.float32)
    if target_faces is None: raise ValueError('target label has no mesh entity')
    visible=(owner==int(target_label)).astype(np.uint8)
    contours,_=cv2.findContours(visible,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours=[c.reshape(-1,2) for c in contours if len(c)]
    if not contours:
        return np.zeros((0,3),np.int64),np.zeros((0,3),np.float32),np.zeros((0,2),np.float32)
    # Preserve every external boundary, hole, and disconnected island.  The
    # cap is shared proportionally only after reserving one hit per contour.
    budget=max(int(max_samples),len(contours))
    lengths=np.asarray([len(c) for c in contours],np.float64)
    allocation=np.ones(len(contours),np.int64)
    remaining=budget-len(contours)
    if remaining:
        share=remaining*lengths/lengths.sum()
        allocation+=np.floor(share).astype(np.int64)
        for i in np.argsort(-(share-np.floor(share)))[:remaining-int(np.floor(share).sum())]:
            allocation[i]+=1
    selected=[]
    for contour,count in zip(contours,allocation):
        take=min(len(contour),int(count))
        selected.append(contour if take==len(contour) else contour[np.linspace(0,len(contour)-1,take,dtype=int)])
    xy=np.concatenate(selected,axis=0)
    xs=xy[:,0];ys=xy[:,1]
    ids=face_map[ys,xs]
    good=ids>=0
    return target_faces[ids[good]],bary_map[ys[good],xs[good]],xy[good].astype(np.float32)

def timestamp_velocity_residual(values: torch.Tensor, times: torch.Tensor, active: torch.Tensor|None=None) -> torch.Tensor:
    if values.shape[0]<2:return values.sum()*0.
    dt=times[1:]-times[:-1]
    good=dt>0
    if active is not None:good=good & active[:-1] & active[1:]
    if not torch.any(good):return values.sum()*0.
    velocity=(values[1:]-values[:-1])/dt[:,None]
    return velocity[good].square().mean()
