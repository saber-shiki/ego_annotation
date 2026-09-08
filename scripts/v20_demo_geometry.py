"""Shared demo geometry. These utilities make no measured-contact authority claim."""
from pathlib import Path
import hashlib
import json
import inspect
import numpy as np

TIP_VERTICES=np.array([744,320,443,554,671])
JOINT_ORDER=np.array([0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20])

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())
def camera(points,T):return (points-T[:3,3])@T[:3,:3]
def world(points,T):return points@T[:3,:3].T+T[:3,3]
def project(points,K):return points[...,:2]/np.maximum(points[...,2:],1e-6)*K[:2]+K[2:]
def resized_K(K,source=(1408,1408),target=(960,960)):
    s=np.asarray(target)/np.asarray(source);return np.r_[np.asarray(K)[:2]*s,(np.asarray(K)[2:]+.5)*s-.5]

def mano_models(path,device='cpu'):
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
    for name,value in [('bool',np.bool_),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
        if name not in np.__dict__:setattr(np,name,value)
    import smplx
    models={side:smplx.MANO(str(Path(path)/f'MANO_{side.upper()}.pkl'),is_rhand=(side=='right'),use_pca=False,flat_hand_mean=True).to(device) for side in ['left','right']}
    models['left'].shapedirs[:,0,:]*=-1
    return models

def torch_rodrigues(x):
    import torch
    z=torch.zeros_like(x[...,0]);a,b,c=x.unbind(-1)
    mat=torch.stack([z,-c,b,c,z,-a,-b,a,z],-1).reshape(*x.shape[:-1],3,3)
    return torch.matrix_exp(mat)

def torch_project(p,K):return p[...,:2]/p[...,2:].clamp_min(.01)*K[...,:2]+K[...,2:]

def make_scene(vertices,faces):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,dtype=np.float32)),o3d.core.Tensor(np.asarray(faces,dtype=np.uint32)))
    return scene

def closest(scene,points):
    import open3d as o3d
    shape=points.shape
    out=scene.compute_closest_points(o3d.core.Tensor(np.asarray(points,dtype=np.float32).reshape(-1,3)),nthreads=4)
    q=out['points'].numpy().reshape(shape);n=out['primitive_normals'].numpy().reshape(shape)
    return q,n
