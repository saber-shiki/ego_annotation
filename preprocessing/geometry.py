"""Pure transforms and actual-triangle queries shared by solver and renderer."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def resized_intrinsics(k, source_wh, target_wh):
    scale=np.asarray(target_wh,float)/np.asarray(source_wh,float)
    k=np.asarray(k,float)
    return np.r_[k[:2]*scale,(k[2:]+.5)*scale-.5]


def matrix_k(k):
    return np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1.]],float)


def project(points,k):
    p=np.asarray(points,float);z=p[...,2]
    return p[...,:2]/np.maximum(z[...,None],1e-5)*np.asarray(k)[...,:2]+np.asarray(k)[...,2:]


def camera_pose(world_R,world_t,camera):
    R=camera[:3,:3].T@world_R
    t=camera[:3,:3].T@(world_t-camera[:3,3])
    return R,t


def world_pose(R,t,camera):
    return camera[:3,:3]@R,camera[:3,:3]@t+camera[:3,3]


def surface_samples(vertices,faces,count=2048,seed=47):
    v=np.asarray(vertices,float);f=np.asarray(faces,int);tri=v[f]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if not len(area) or not np.isfinite(area).all() or area.sum()<=0:raise ValueError('invalid mesh area')
    rng=np.random.default_rng(seed);ids=rng.choice(len(f),count,p=area/area.sum())
    uv=rng.random((count,2));flip=uv.sum(1)>1;uv[flip]=1-uv[flip];q=tri[ids]
    points=q[:,0]+uv[:,0,None]*(q[:,1]-q[:,0])+uv[:,1,None]*(q[:,2]-q[:,0])
    return points


def shape_matrix(parameters,basis):
    """Two bounded principal log stretches, determinant exactly one."""
    a,b=np.asarray(parameters,float)
    return basis@np.diag(np.exp([a,b,-a-b]))@basis.T


def ray_directions(uv,k):
    uv=np.asarray(uv,float).reshape(-1,2)
    return np.column_stack(((uv[:,0]-k[2])/k[0],(uv[:,1]-k[3])/k[1],np.ones(len(uv))))


class TriangleMesh:
    def __init__(self,vertices,faces):
        import open3d as o3d
        self.o3d=o3d;self.vertices=np.asarray(vertices,float);self.faces=np.asarray(faces,np.int64)
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,np.float32)),o3d.core.Tensor(np.asarray(faces,np.uint32)))

    def query(self,uv,k,R,t):
        """Camera ray to canonical mesh; t_hit is camera-Z, not radial distance."""
        dirs=ray_directions(uv,k)@R;origin=-np.asarray(t)@R
        rays=np.c_[np.broadcast_to(origin,dirs.shape),dirs].astype(np.float32)
        hit=self.scene.cast_rays(self.o3d.core.Tensor(rays),nthreads=1)
        z=hit['t_hit'].numpy().astype(float);valid=np.isfinite(z)&(z>.001)
        points=np.zeros((len(z),3));points[valid]=origin+dirs[valid]*z[valid,None]
        return points,z,valid

    def closest(self,points):
        points=np.asarray(points,np.float32)
        result=self.scene.compute_closest_points(self.o3d.core.Tensor(points),nthreads=1)
        nearest=result['points'].numpy().astype(float)
        return nearest,np.linalg.norm(nearest-points,axis=1)

    def normals(self,points):
        if len(points)==0:return np.empty((0,3))
        result=self.scene.compute_closest_points(self.o3d.core.Tensor(np.asarray(points,np.float32)),nthreads=1)
        return result['primitive_normals'].numpy().astype(float)

    def raster(self,k,R,t,wh):
        w,h=map(int,wh);yy,xx=np.mgrid[:h,:w]
        p,z,v=self.query(np.c_[xx.ravel(),yy.ravel()],k,R,t)
        return p.reshape(h,w,3),z.reshape(h,w),v.reshape(h,w)


def rotation_interpolate(Rs,times,valid):
    from scipy.spatial.transform import Slerp
    ids=np.flatnonzero(valid)
    if not len(ids):raise ValueError('no rotation anchor')
    if len(ids)==1:return np.repeat(Rs[ids[0]][None],len(Rs),axis=0)
    return Slerp(np.asarray(times)[ids],Rotation.from_matrix(Rs[ids]))(np.clip(times,times[ids[0]],times[ids[-1]])).as_matrix()
