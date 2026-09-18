"""Differentiable residuals for independent RGB hand silhouettes."""
from __future__ import annotations
import cv2
import numpy as np
import torch

def mask_contour(mask: np.ndarray, sample_count: int=160) -> np.ndarray:
    mask=np.asarray(mask).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours: return np.zeros((0,2),np.float32)
    contour=max(contours,key=cv2.contourArea).reshape(-1,2).astype(np.float32)
    if len(contour)<2:return contour
    edge=np.roll(contour,-1,axis=0)-contour; length=np.linalg.norm(edge,axis=1)
    cum=np.r_[0.,np.cumsum(length)]; total=float(cum[-1])
    if total<=0:return contour
    pos=np.linspace(0,total,int(sample_count),endpoint=False); ix=np.searchsorted(cum,pos,side='right')-1
    frac=(pos-cum[ix])/np.maximum(length[ix],1e-6)
    return contour[ix]+frac[:,None]*edge[ix]

def triangle_edge_points(vertices_uv: torch.Tensor, faces: np.ndarray|torch.Tensor, samples_per_edge: int=2) -> torch.Tensor:
    """Project all MANO triangle edges; unlike a joint/vertex hull this retains topology."""
    f=torch.as_tensor(faces,device=vertices_uv.device,dtype=torch.long)
    edges=torch.unique(torch.sort(torch.cat((f[:,:2],f[:,1:]),dim=0),dim=1).values,dim=0)
    a=vertices_uv[edges[:,0]]; b=vertices_uv[edges[:,1]]
    fractions=torch.linspace(0,1,max(2,int(samples_per_edge)),device=vertices_uv.device)
    return (a[:,None,:]*(1-fractions[None,:,None])+b[:,None,:]*fractions[None,:,None]).reshape(-1,2)

def bidirectional_contour_residual(candidate_uv: torch.Tensor, raw_contour: torch.Tensor,
                                   visibility: torch.Tensor|float=1., uncertainty: torch.Tensor|float=0.,
                                   robust_px: float=5.) -> torch.Tensor:
    """Symmetric nearest-contour residual with explicit visibility uncertainty gate."""
    if candidate_uv.numel()==0 or raw_contour.numel()==0:return candidate_uv.sum()*0.
    with torch.no_grad():
        d=torch.cdist(candidate_uv.detach(),raw_contour.detach())
        forward=d.min(dim=1).values
        reverse=d.min(dim=0).values
    weight=torch.as_tensor(visibility,device=candidate_uv.device,dtype=candidate_uv.dtype)
    weight=weight*(1-torch.as_tensor(uncertainty,device=candidate_uv.device,dtype=candidate_uv.dtype)).clamp(0,1)
    # Keep correspondence indices detached, but residual coordinates live in graph.
    fi=torch.argmin(torch.cdist(candidate_uv,raw_contour.detach()),dim=1)
    ri=torch.argmin(torch.cdist(raw_contour.detach(),candidate_uv),dim=1)
    a=torch.linalg.vector_norm(candidate_uv-raw_contour.detach()[fi],dim=1)
    b=torch.linalg.vector_norm(candidate_uv[ri]-raw_contour.detach(),dim=1)
    robust=lambda x: torch.minimum(x.square(),torch.as_tensor(robust_px**2,device=x.device,dtype=x.dtype))
    return weight*(robust(a).mean()+robust(b).mean())/2

def timestamp_velocity_residual(values: torch.Tensor, times: torch.Tensor, active: torch.Tensor|None=None) -> torch.Tensor:
    if values.shape[0]<2:return values.sum()*0.
    dt=times[1:]-times[:-1]
    good=dt>0
    if active is not None:good=good & active[:-1] & active[1:]
    if not torch.any(good):return values.sum()*0.
    velocity=(values[1:]-values[:-1])/dt[:,None]
    return velocity[good].square().mean()
