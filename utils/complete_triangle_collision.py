"""Complete native triangle intersection narrow phase, not geometric substitutes.

Triangle-center bounding spheres ONLY accelerate a conservative broad phase;
every retained pair is tested using the actual six vertices. No decimation.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from utils.generated_collision import GeneratedCollisionScene

CONTRACT='complete_native_triangle_nonpenetration.v1'


def triangles_overlap(a,b,tolerance_m=1e-8):
    """Float64 SAT, including coplanar triangles and boundary touches."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1:]!=(3,3):raise ValueError('paired triangles required')
    if not len(a):return np.zeros(0,bool)
    ea=np.roll(a,-1,axis=1)-a;eb=np.roll(b,-1,axis=1)-b
    na=np.cross(ea[:,0],ea[:,1]);nb=np.cross(eb[:,0],eb[:,1])
    cross=np.cross(ea[:,:,None,:],eb[:,None,:,:]).reshape(-1,9,3)
    axes=np.concatenate((na[:,None],nb[:,None],cross,np.cross(na[:,None],ea),np.cross(nb[:,None],eb)),axis=1)
    norm=np.linalg.norm(axes,axis=-1);valid=norm>1e-20;axes=axes/np.maximum(norm[:,:,None],1e-30)
    # Centering improves numerical precision but does not change projections.
    origin=(a.mean(1)+b.mean(1))*.5
    pa=np.einsum('nvi,nki->nkv',a-origin[:,None],axes)
    pb=np.einsum('nvi,nki->nkv',b-origin[:,None],axes)
    separated=((pa.max(-1)<pb.min(-1)-tolerance_m)|(pb.max(-1)<pa.min(-1)-tolerance_m))&valid
    return ~separated.any(-1)


def barycentric(points,triangles):
    p,a=np.asarray(points,float),np.asarray(triangles,float)
    e0=a[:,1]-a[:,0];e1=a[:,2]-a[:,0];d=p-a[:,0]
    d00=np.einsum('ni,ni->n',e0,e0);d01=np.einsum('ni,ni->n',e0,e1);d11=np.einsum('ni,ni->n',e1,e1)
    d20=np.einsum('ni,ni->n',d,e0);d21=np.einsum('ni,ni->n',d,e1)
    den=np.maximum(d00*d11-d01*d01,1e-30)
    v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
    return np.c_[1-v-w,v,w]


def intersection_barycentric(a,b):
    """Actual triangle/triangle intersection witnesses in the first triangle.

    Noncoplanar witnesses are segment/plane hits tested in the other triangle.
    Coplanar containment uses actual native vertices; borderline SAT pairs use
    a projected/clamped point ON the hand triangle, not its distant vertices.
    Final intersection decisions remain complete SAT, independent of witnesses.
    """
    a,b=np.asarray(a,float),np.asarray(b,float);n=len(a);points=[];valid=[]
    for first,second in ((a,b),(b,a)):
        normal=np.cross(second[:,1]-second[:,0],second[:,2]-second[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-30)
        dist=np.einsum('nvi,ni->nv',first-second[:,None,0],normal)
        for k in range(3):
            end=(k+1)%3;den=dist[:,k]-dist[:,end]
            t=np.divide(dist[:,k],den,out=np.zeros(n),where=np.abs(den)>1e-15)
            p=first[:,k]+t[:,None]*(first[:,end]-first[:,k]);bar=barycentric(p,second)
            good=(np.abs(den)>1e-15)&(t>=-1e-7)&(t<=1+1e-7)&(bar.min(-1)>=-1e-7)
            points.append(p);valid.append(good)
            # Vertices already on the other plane, including coplanar containment.
            p=first[:,k];bar=barycentric(p,second)
            good=(np.abs(dist[:,k])<=1e-8)&(bar.min(-1)>=-1e-7)
            points.append(p);valid.append(good)
    points=np.stack(points,axis=1);valid=np.stack(valid,axis=1);count=valid.sum(1)
    witness=np.einsum('nvc,nv->nc',points,valid)/np.maximum(count[:,None],1)
    fallback=count==0
    witness[fallback]=b[fallback].mean(1)
    bar=barycentric(witness,a);bar=np.clip(bar,0.,1.);bar/=np.maximum(bar.sum(-1,keepdims=True),1e-30)
    return bar,fallback


class CompleteTriangleScene(GeneratedCollisionScene):
    def __init__(self,vertices,faces):
        super().__init__(vertices,faces)
        self.triangles=self.vertices[self.faces].astype(float)
        self.centers=self.triangles.mean(1)
        self.radii=np.linalg.norm(self.triangles-self.centers[:,None],axis=-1).max(-1)
        self.max_radius=float(self.radii.max())
        self.lower=self.triangles.min(1);self.upper=self.triangles.max(1)
        self.tree=cKDTree(self.centers)
        self.face_normals=np.cross(self.triangles[:,1]-self.triangles[:,0],self.triangles[:,2]-self.triangles[:,0])
        self.face_normals/=np.maximum(np.linalg.norm(self.face_normals,axis=-1,keepdims=True),1e-30)

    def intersections(self,vertices_world,mano_faces,rotation,translation,*,tolerance_m=1e-8):
        vertices=(np.asarray(vertices_world,float)-translation)@rotation
        tri=vertices[np.asarray(mano_faces,int)]
        centers=tri.mean(1);radii=np.linalg.norm(tri-centers[:,None],axis=-1).max(-1)
        neighbors=self.tree.query_ball_point(centers,radii+self.max_radius+tolerance_m,workers=1,eps=0.)
        counts=np.array([len(x) for x in neighbors],int)
        if not counts.sum():return {'hand_face_ids':np.zeros(0,int),'object_face_ids':np.zeros(0,int),'broad_phase_pairs':0}
        hi=np.repeat(np.arange(len(tri)),counts);oi=np.concatenate([np.asarray(x,int) for x in neighbors if len(x)])
        lower=tri.min(1);upper=tri.max(1)
        mask=np.all((upper[hi]>=self.lower[oi]-tolerance_m)&(self.upper[oi]>=lower[hi]-tolerance_m),axis=-1)
        hi,oi=hi[mask],oi[mask];broad=len(hi);selected=[]
        for offset in range(0,len(hi),32768):
            keep=triangles_overlap(tri[hi[offset:offset+32768]],self.triangles[oi[offset:offset+32768]],tolerance_m)
            selected.append(np.flatnonzero(keep)+offset)
        ix=np.concatenate(selected) if selected else np.zeros(0,int)
        return {'hand_face_ids':hi[ix],'object_face_ids':oi[ix],'broad_phase_pairs':broad}

    def closest_signed(self,points_world,rotation,translation,*,nsamples=5):
        import open3d as o3d
        local=(np.asarray(points_world,float)-translation)@rotation
        tensor=o3d.core.Tensor(local.astype(np.float32))
        closest=self.scene.compute_closest_points(tensor,nthreads=1)
        sdf=self.scene.compute_signed_distance(tensor,nthreads=1,nsamples=nsamples).numpy().astype(float)
        q=closest['points'].numpy().astype(float);d=local-q
        length=np.linalg.norm(d,axis=-1)
        normals=closest['primitive_normals'].numpy().astype(float)
        # Distance gradients outside/inside point away from/to the nearest boundary.
        active=length>1e-7
        normals[active]=d[active]/length[active,None]*np.where(sdf[active]>=0,1.,-1.)[:,None]
        return {'signed_distance_m':sdf,'points_object':local,'closest_object':q,'normals_world':normals@rotation.T,
                'nearest_object_face_ids':closest['primitive_ids'].numpy().astype(int)}

    def directed_exit(self,points_world,direction_world,rotation,translation):
        """Trace real native triangles to the first OUTSIDE state along a mode.

        For a boundary point on the entry side, step across entry then trace exit;
        this selects a coherent side, unlike contradictory nearest-wall exits.
        """
        import open3d as o3d
        p=(np.asarray(points_world,float)-translation)@rotation
        direction=np.asarray(direction_world,float)@rotation;direction/=np.linalg.norm(direction)
        origins=p-direction[None]*5e-6;travel=np.full(len(p),-5e-6);active=np.arange(len(p))
        targets=np.full_like(p,np.nan);normals=np.full_like(p,np.nan)
        for _ in range(4):
            if not len(active):break
            rays=np.c_[origins[active],np.repeat(direction[None],len(active),axis=0)]
            hit=self.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)),nthreads=1)
            distance=hit['t_hit'].numpy().astype(float);finite=np.isfinite(distance)
            missed=active[~finite]
            if len(missed):
                near=self.closest_signed(points_world[missed],rotation,translation)
                targets[missed]=near['closest_object'];normals[missed]=near['normals_world']@rotation
            ids=active[finite];face=hit['primitive_ids'].numpy()[finite].astype(int)
            if not len(ids):break
            travel[ids]+=distance[finite]
            surface=p[ids]+travel[ids,None]*direction
            beyond=surface+direction*5e-6
            signed=self.scene.compute_signed_distance(o3d.core.Tensor(beyond.astype(np.float32)),nthreads=1,nsamples=5).numpy()
            done=signed>=0
            targets[ids[done]]=surface[done];normals[ids[done]]=self.face_normals[face[done]]
            remaining=ids[~done];travel[remaining]+=5e-6;origins[remaining]=beyond[~done];active=remaining
        if np.isnan(targets).any():raise RuntimeError('native directed exit exceeded four actual crossings')
        return {'target_world':targets@rotation.T+translation,'normals_world':normals@rotation.T,
                'travel_m':np.linalg.norm(targets-p,axis=-1)}

    def inspect_hand(self,vertices_world,faces,rotation,translation,*,nsamples=5,tolerance_m=1e-8):
        v=np.asarray(vertices_world,float);f=np.asarray(faces,int)
        points=np.concatenate((v,v[f].mean(1)))
        near=self.closest_signed(points,rotation,translation,nsamples=nsamples)
        intersection=self.intersections(v,f,rotation,translation,tolerance_m=tolerance_m)
        depth=np.maximum(-near['signed_distance_m'],0.)
        return {'near':near,'intersections':intersection,'max_inside_depth_m':float(depth.max()),
                'inside_squared_sum_m2':float(np.square(depth).sum()),'inside_count_over_10um':int((depth>1e-5).sum()),
                'intersection_pair_count':len(intersection['hand_face_ids']),
                'intersection_hand_face_count':int(len(np.unique(intersection['hand_face_ids']))),
                'nearest_surface_gap_m':float(np.min(np.abs(near['signed_distance_m'])))}
