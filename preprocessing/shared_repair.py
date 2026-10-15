"""Full-native-mesh affine queries and observation-driven shared-shape/pose repair.
All queries use original triangles. A positive shared linear deformation is NOT a box.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from preprocessing.geometry import TriangleMesh,camera_pose,project,resized_intrinsics,ray_directions,surface_samples
from preprocessing.factors import subset,boundary,rgb_model_factors
from preprocessing.mask_contract import FixedMaskDomain


def shared_matrix(q,basis):
 return basis@np.diag(np.exp(q))@basis.T

class AffineMesh:
 """Ray parameter remains camera Z under any nonsingular shared mesh deformation."""
 def __init__(self,vertices,faces):
  self.native=TriangleMesh(vertices,faces);self.vertices=np.asarray(vertices);self.faces=np.asarray(faces)
 def query(self,uv,k,R,t,S):
  M=R@S;inv=np.linalg.inv(M);dirs=ray_directions(uv,k)@inv.T;origin=-np.asarray(t)@inv.T
  rays=np.c_[np.broadcast_to(origin,dirs.shape),dirs].astype(np.float32)
  hits=self.native.scene.cast_rays(self.native.o3d.core.Tensor(rays),nthreads=1)
  z=hits['t_hit'].numpy().astype(float);ok=np.isfinite(z)&(z>.001)
  X=np.zeros((len(z),3));X[ok]=origin+dirs[ok]*z[ok,None]
  normal=hits['primitive_normals'].numpy().astype(float)@np.linalg.inv(S)@R.T
  normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
  return X,z,ok,normal
 def raster(self,k,R,t,S,wh):
  w,h=map(int,wh);y,x=np.mgrid[:h,:w];p,z,v,n=self.query(np.c_[x.ravel(),y.ravel()],k,R,t,S)
  return p.reshape(h,w,3),z.reshape(h,w),v.reshape(h,w),n.reshape(h,w,3)

def observed_normals(points,query):
 if len(points)<8:return np.tile([0.,0.,-1.],(len(query),1)),np.zeros(len(query))
 _,idx=cKDTree(points).query(query,k=min(24,len(points)),workers=1)
 block=points[idx];center=block.mean(1);d=block-center[:,None];cov=np.einsum('nki,nkj->nij',d,d)/block.shape[1]
 values,vectors=np.linalg.eigh(cov);normal=vectors[:,:,0];normal[normal[:,2]>0]*=-1
 confidence=np.clip(1.-4*values[:,0]/np.maximum(values[:,1],1e-9),.05,1.)
 return normal,confidence

@dataclass
class Corr:
 frame:int
 depth_X:np.ndarray
 depth_Q:np.ndarray
 depth_normal:np.ndarray
 normal_weight:np.ndarray
 native_normals:np.ndarray
 contour_X:np.ndarray
 contour_uv:np.ndarray
 crease_X:np.ndarray
 crease_uv:np.ndarray

def native_creases(vertices,faces,limit=1200):
 import trimesh
 m=trimesh.Trimesh(vertices,faces,process=False)
 adjacent=m.face_adjacency;angles=m.face_adjacency_angles
 eligible=np.flatnonzero(angles>.40)
 if not len(eligible):return np.empty((0,3))
 edges=m.face_adjacency_edges[eligible]
 length=np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
 rng=np.random.default_rng(3015);ids=rng.choice(len(edges),min(limit,len(edges)),replace=False,p=(length+1e-9)/(length+1e-9).sum())
 u=rng.random(len(ids))
 return vertices[edges[ids,0]]*(1-u[:,None])+vertices[edges[ids,1]]*u[:,None]

class ObjectRepair:
 def __init__(self,inputs,archive,hd,width=160,points=160):
  self.inputs=inputs;self.base=archive;self.hd=hd;self.width=width;self.points=points
  self.mesh=AffineMesh(archive['vertices_canonical'],archive['faces'])
  v=surface_samples(archive['vertices_canonical'],archive['faces'],4096)
  _,B=np.linalg.eigh(np.cov(v.T));self.basis=B[:,::-1]
  if np.linalg.det(self.basis)<0:self.basis[:,-1]*=-1
  self.creases=native_creases(archive['vertices_canonical'],archive['faces'])
  self.domains=[FixedMaskDomain.from_inputs(inputs,hd,i) for i in range(len(inputs.ids))]
  self.wh=hd.shape[1:][::-1]
  self.Kr=np.asarray([resized_intrinsics(k,inputs.wh,self.wh) for k in inputs.K])
 def correspondences(self,R,t,q,indices):
  S=shared_matrix(q,self.basis);rows=[]
  for i in indices:
   C=self.inputs.cameras[i];r,tt=camera_pose(R[i],t[i],C);view=self.mesh.raster(self.Kr[i],r,tt,S,self.wh);domain=self.domains[i]
   Q=subset(self.inputs.clouds[i],self.points)
   uv=project(Q,self.inputs.K[i]);X,_,hit,hit_normals=self.mesh.query(uv,self.inputs.K[i],r,tt,S)
   if len(Q) and np.any(~hit):
    visible=view[0][view[2]];p=visible@S.T@r.T+tt
    if len(p):_,idx=cKDTree(p).query(Q[~hit],workers=1);X[~hit]=visible[idx]
   normal,nw=observed_normals(self.inputs.clouds[i],Q)
   native_normals=self.mesh.native.normals(X) if len(X) else np.empty((0,3))
   nw=nw*hit.astype(float)
   mb=boundary(view[2])&domain.boundary_domain;tb=boundary(domain.target)&domain.boundary_domain
   my,mx=np.nonzero(mb);ty,tx=np.nonzero(tb)
   cX=np.empty((0,3));cu=np.empty((0,2))
   if min(len(mx),len(tx))>=5:
    mxy=np.c_[mx,my];txy=np.c_[tx,ty];mi=subset(np.arange(len(mx)),80);ti=subset(np.arange(len(tx)),80)
    _,a=cKDTree(txy).query(mxy[mi],workers=1);_,b=cKDTree(mxy).query(txy[ti],workers=1)
    cX=np.r_[view[0][my[mi],mx[mi]],view[0][my[b],mx[b]]];cu=np.r_[txy[a],txy[ti]].astype(float)
   # Model creases are queried for true surface visibility; RGB gradients are
   # weak generic edge evidence, NOT a learned segmentation or semantic edge truth.
   eX=np.empty((0,3));eu=np.empty((0,2))
   if len(self.creases):
    cp=self.creases@S.T@r.T+tt;cu0=project(cp,self.Kr[i]);pix=np.rint(cu0).astype(int)
    valid=(cp[:,2]>.01)&(pix[:,0]>=1)&(pix[:,0]<self.wh[0]-1)&(pix[:,1]>=1)&(pix[:,1]<self.wh[1]-1)
    ids=np.flatnonzero(valid)
    if len(ids):
     _,zz,hit,_=self.mesh.query(cu0[ids],self.Kr[i],r,tt,S)
     validids=ids[hit&(np.abs(zz-cp[ids,2])<.003)]
     if len(validids):validids=validids[domain.boundary_domain[pix[validids,1],pix[validids,0]]]
     gray=cv2.resize(self.inputs.grays[i],self.wh,interpolation=cv2.INTER_AREA)
     edge=(cv2.Canny(gray,45,110)>0)&cv2.erode(domain.target.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)&domain.boundary_domain
     yy,xx=np.nonzero(edge)
     if len(xx) and len(validids):
      dist,match=cKDTree(np.c_[xx,yy]).query(cu0[validids],workers=1);good=np.flatnonzero(dist<8);good=subset(good,48)
      eX=self.creases[validids[good]];eu=np.c_[xx,yy][match[good]].astype(float)
   rows.append(Corr(int(i),X,Q,normal,nw,native_normals,cX,cu,eX,eu))
  return rows
 def fresh_merit(self,R,t,q,indices):
  S=shared_matrix(q,self.basis);values=[];ious=[];surfaces=[]
  for i in indices:
   r,tt=camera_pose(R[i],t[i],self.inputs.cameras[i]);_,_,mask,_=self.mesh.raster(self.Kr[i],r,tt,S,self.wh)
   score=self.domains[i].score(mask);iou=score['visible']['iou'];ious.append(iou)
   Q=subset(self.inputs.clouds[i],self.points);uv=project(Q,self.inputs.K[i]);X,z,hit,n=self.mesh.query(uv,self.inputs.K[i],r,tt,S)
   dist=np.full(len(Q),.04)
   if len(Q):dist[hit]=np.minimum(np.abs(z[hit]-Q[hit,2]),.04)
   surface=float(dist.mean()) if len(dist) else 0.;surfaces.append(surface)
   values.append((1-iou)*4+surface/.01)
  return float(np.mean(values)+.08*np.sum((q/.12)**2)+.2*(np.sum(q)/.10)**2),float(np.mean(ious)),float(np.mean(surfaces))
 def fit(self,R,t,q,indices,fit_shape,outer=3,max_nfev=24,freeze_rotation=False):
  indices=np.asarray(indices,int);trace=[];S0=shared_matrix(q,self.basis)
  R_ref=self.base['object_rotation'];t_ref=self.base['object_translation_world'];R=R.copy();t=t.copy();q=q.copy()
  for iteration in range(outer):
   start=time.monotonic();corr=self.correspondences(R,t,q,indices);offset=3 if fit_shape else 0;size=offset+6*len(indices);pos={i:j for j,i in enumerate(indices)}
   # RGB source endpoint identities preserved per outer pass. Native ray bindings
   # are conditional uncertain image factors, not independent metric observations.
   deformed=TriangleMesh(self.base['vertices_canonical']@shared_matrix(q,self.basis).T,self.base['faces'])
   rgb=rgb_model_factors(deformed,R,t,self.inputs,max_points=16)
   keep=np.isin(rgb.frames,indices);rf=rgb.frames[keep];rx=rgb.points[keep]@np.linalg.inv(shared_matrix(q,self.basis)).T;ru=rgb.uv[keep];rw=rgb.weights[keep]
   def unpack(x):
    qq=q+x[:3] if fit_shape else q;v=x[offset:].reshape(-1,6);RR=R.copy();tt=t.copy()
    if not freeze_rotation:RR[indices]=Rotation.from_rotvec(v[:,:3]).as_matrix()@R[indices]
    tt[indices]+=v[:,3:]
    return RR,tt,qq
   def groups(x):
    RR,tt,qq=unpack(x);S=shared_matrix(qq,self.basis);parts=[]
    def add(name,values,frames,shape=False):parts.append((name,np.asarray(values).ravel(),list(frames),shape))
    for row in corr:
     i=row.frame;rc,tc=camera_pose(RR[i],tt[i],self.inputs.cameras[i]);count=max(1,len(row.depth_X));scale=np.sqrt(count)
     P=row.depth_X@S.T@rc.T+tc;d=P-row.depth_Q
     add('surface',np.sum(d*row.depth_normal,axis=1)*np.sqrt(row.normal_weight)/.008/scale,[i],True)
     add('depth',(P[:,2]-row.depth_Q[:,2])/.015/scale*.35,[i],True)
     if len(row.native_normals):
      normals=row.native_normals@np.linalg.inv(S)@rc.T;normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
      agree=1-np.abs(np.sum(normals*row.depth_normal,axis=1))
      add('surface_normals',agree*np.sqrt(row.normal_weight)/.25/scale*.35,[i],True)
     if len(row.contour_X):add('contour',(project(row.contour_X@S.T@rc.T+tc,self.Kr[i])-row.contour_uv)/1.5/np.sqrt(len(row.contour_X)),[i],True)
     if len(row.crease_X):add('crease',(project(row.crease_X@S.T@rc.T+tc,self.Kr[i])-row.crease_uv)/1.8/np.sqrt(len(row.crease_X))*.30,[i],True)
     correction=Rotation.from_matrix(RR[i]@R_ref[i].T).as_rotvec()
     add('pose_prior',np.r_[correction/.35,(tt[i]-t_ref[i])/.06]*.12,[i])
    for i in indices:
     keep=rf==i
     if keep.any():
      rc,tc=camera_pose(RR[i],tt[i],self.inputs.cameras[i]);pp=rx[keep]@S.T@rc.T+tc
      add('rgb',(project(pp,self.inputs.K[i])-ru[keep])*np.sqrt(rw[keep,None])/3/np.sqrt(keep.sum())*.5,[i],True)
    if len(indices)==len(R):
     dt=np.diff(self.inputs.times)
     v=np.diff(tt,axis=0)/dt[:,None];acc=np.diff(v,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
     omega=Rotation.from_matrix(RR[1:]@RR[:-1].transpose(0,2,1)).as_rotvec()/dt[:,None];aa=np.diff(omega,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
     for i in range(len(R)-2):
      add('actual_acceleration',acc[i]/6*.10,[i,i+1,i+2]);add('actual_rotation_acceleration',aa[i]/140*.08,[i,i+1,i+2])
    if fit_shape:
     add('shared_shape_prior',qq/.12*.30,[],True)
     add('shared_log_volume_prior',[np.sum(qq)/.10*.5],[],True)
    return parts
   zero=np.zeros(size);initial=groups(zero);n=sum(len(v) for _,v,_,_ in initial);J=lil_matrix((n,size),dtype=np.int8);at=0
   for _,values,frames,uses_shape in initial:
    if fit_shape and uses_shape:J[at:at+len(values),:3]=1
    for i in frames:J[at:at+len(values),offset+6*pos[i]:offset+6*pos[i]+6]=1
    at+=len(values)
   residual=lambda x:np.concatenate([v for _,v,_,_ in groups(x)])
   lo=np.full(size,-np.inf);hi=np.full(size,np.inf)
   if fit_shape:lo[:3]=-.25-q;hi[:3]=.25-q
   fit=least_squares(residual,zero,jac_sparsity=J.tocsr(),loss='soft_l1',f_scale=1.,max_nfev=max_nfev,x_scale=np.r_[np.full(offset,.08),np.tile([.06,.06,.06,.01,.01,.01],len(indices))],bounds=(lo,hi),ftol=1e-5,xtol=1e-5,gtol=1e-5)
   if not np.isfinite(fit.x).all():raise ValueError('nonfinite full-mesh repair')
   before=residual(zero)
   initial_merit,initial_iou,initial_surface=self.fresh_merit(R,t,q,indices);chosen=zero;trial=[]
   for fraction in (1.,.5,.25,.125):
    RR,tt,qq=unpack(fit.x*fraction);merit,iou,surface=self.fresh_merit(RR,tt,qq,indices)
    accepted=merit<initial_merit-1e-8 and iou>=initial_iou-.003
    trial.append(dict(fraction=fraction,merit=merit,visible_iou=iou,raw_surface_m=surface,accepted=bool(accepted)))
    if accepted:chosen=fit.x*fraction;break
   after=residual(chosen);final_groups=groups(chosen)
   RR,tt,qq=unpack(chosen);R,t,q=RR,tt,qq
   trace.append(dict(outer=iteration,actual_step_fraction=float(next((x['fraction'] for x in trial if x['accepted']),0)),fresh_merit_before=initial_merit,visible_iou_before=initial_iou,trials=trial,nfev=int(fit.nfev),seconds=time.monotonic()-start,cost_before=float(np.sum(2*(np.sqrt(1+before**2)-1))*.5),cost_after=float(np.sum(2*(np.sqrt(1+after**2)-1))*.5),q=q.tolist(),families={key:dict(count=sum(len(v) for name,v,_,_ in initial if name==key),rms_before=float(np.sqrt(np.mean(np.concatenate([v for name,v,_,_ in initial if name==key])**2))),rms_after=float(np.sqrt(np.mean(np.concatenate([v for name,v,_,_ in final_groups if name==key])**2)))) for key in set(name for name,_,_,_ in initial)}))
   print('object fit',fit_shape,iteration,'shape',np.exp(q),'cost',fit.cost,'seconds',trace[-1]['seconds'],flush=True)
  return R,t,q,trace
