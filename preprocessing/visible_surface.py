"""Uniform mesh scale/independent translation against FIXED UniDepth visible surfaces.
Actual triangle first hits define visibility. No CAD, GT, MoGe rewrite or depth correction.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from preprocessing.geometry import TriangleMesh,camera_pose,project,resized_intrinsics
from preprocessing.factors import boundary,subset


def cloud_digest(clouds):
    h=hashlib.sha256()
    for q in clouds:
        a=np.ascontiguousarray(q);h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return {'count':int(len(x)),'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'mean':float(np.mean(x))} if len(x) else {'count':0}


@dataclass
class SurfaceFrame:
    position: int
    R_camera: np.ndarray
    t_initial_camera: np.ndarray
    points_camera: np.ndarray
    uv: np.ndarray
    K: np.ndarray
    raster_K: np.ndarray
    raster_wh: tuple
    appearance_mask: np.ndarray
    owned_mask: np.ndarray
    hand_unknown: np.ndarray
    contour_exclusion: np.ndarray
    weight: float


class VisibleSurfaceRegistration:
    """Alternating visible correspondences and a shared positive scale + camera t's."""
    def __init__(self,inputs,R_world,t_world,hand_depth,indices,*,fit_scale=True,initial_scale=1.,weights=None,max_points=384,temporal=False,motion_reference_world=None):
        self.inputs=inputs;self.mesh=TriangleMesh(inputs.vertices,inputs.faces)
        self.indices=np.asarray(indices,int);self.fit_scale=fit_scale;self.scale0=float(initial_scale);self.temporal=temporal
        if not len(self.indices) or len(set(self.indices.tolist()))!=len(self.indices):raise ValueError('invalid registration frames')
        if not np.isfinite(initial_scale) or initial_scale<=0:raise ValueError('positive scale required')
        self.weights=np.ones(len(self.indices)) if weights is None else np.asarray(weights,float)
        if self.weights.shape!=(len(self.indices),) or np.any(self.weights<=0):raise ValueError('invalid frame weights')
        self.frames=[];self.min_model_z=[]
        for j,i in enumerate(self.indices):
            rc,tc=camera_pose(R_world[i],t_world[i],inputs.cameras[i]);q=np.asarray(inputs.clouds[i],float)
            if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or np.any(q[:,2]<=0):raise ValueError('qualified surface must be finite positive camera XYZ')
            q=subset(q,max_points).copy();wh=hand_depth.shape[1:][::-1]
            raw=cv2.resize(inputs.masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
            owned=getattr(inputs,'owned_masks',None)
            own=np.zeros_like(raw) if owned is None else cv2.resize(owned[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
            # Trusted object-owned pixels override an inconsistent predicted hand surface.
            uncertain_hand=np.isfinite(hand_depth[i])&~own
            exclusion=cv2.dilate(uncertain_hand.astype('uint8'),np.ones((3,3),np.uint8))>0
            exclusion[:2]=True;exclusion[-2:]=True;exclusion[:,:2]=True;exclusion[:,-2:]=True
            self.frames.append(SurfaceFrame(int(i),rc,tc.copy(),q,project(q,inputs.K[i]) if len(q) else np.empty((0,2)),inputs.K[i].copy(),resized_intrinsics(inputs.K[i],inputs.wh,wh),wh,raw,own,uncertain_hand,exclusion,float(self.weights[j])))
            self.min_model_z.append(float(np.min(inputs.vertices@rc[2])))
        self.t0=np.asarray([f.t_initial_camera for f in self.frames]);self.min_model_z=np.asarray(self.min_model_z)
        self.offset=1 if fit_scale else 0
        reference=t_world if motion_reference_world is None else motion_reference_world
        self.reference_camera=np.asarray([camera_pose(R_world[i],reference[i],inputs.cameras[i])[1] for i in self.indices])
        self.camera_rotations=inputs.cameras[self.indices,:3,:3]
        self.times=np.asarray(inputs.times)[self.indices]
        self.depth_bridges={};self.free_coordinates=np.ones_like(self.t0,dtype=bool)
        if temporal:
            if np.any(np.diff(self.times)<=0):raise ValueError('timeline timestamps must increase')
            measured=np.array([j for j,f in enumerate(self.frames) if len(f.points_camera)],int)
            if not len(measured):raise ValueError('timeline requires at least one qualified metric anchor')
            for j,f in enumerate(self.frames):
                if len(f.points_camera):continue
                left=measured[measured<j];right=measured[measured>j]
                if len(left) and len(right):
                    a,b=int(left[-1]),int(right[0]);w=(self.times[j]-self.times[a])/(self.times[b]-self.times[a])
                    self.depth_bridges[j]=[(a,1.-float(w)),(b,float(w))]
                else:self.depth_bridges[j]=[(int(left[-1] if len(left) else right[0]),1.)]
                self.free_coordinates[j,2]=False
        self.variables=np.full(self.t0.shape,-1,int)
        self.variables[self.free_coordinates]=np.arange(self.offset,self.offset+np.count_nonzero(self.free_coordinates))
        self.size=self.offset+int(np.count_nonzero(self.free_coordinates))
        self.acceleration_sigma_m_s2=8.

    def unpack(self,x):
        scale=float(np.exp(x[0])) if self.fit_scale else self.scale0
        t=self.t0.copy();t[self.free_coordinates]=np.asarray(x[self.offset:])
        # Interpolate the metric-reference correction, not the pose: current-frame
        # RGB motion is retained. A missing surface never creates a free depth reset.
        delta=np.einsum('nij,nj->ni',self.camera_rotations,t-self.reference_camera)
        for j,anchors in self.depth_bridges.items():
            correction=sum(w*delta[k] for k,w in anchors)
            t[j,2]=self.reference_camera[j,2]+self.camera_rotations[j,:,2]@correction
        return scale,t

    def pack(self,s,t):
        values=np.asarray(t)[self.free_coordinates]
        return np.r_[np.log(s),values] if self.fit_scale else values.copy()

    def dependencies(self,j,component=None):
        components=range(3) if component is None else [component];columns=set()
        for c in components:
            if self.variables[j,c]>=0:columns.add(int(self.variables[j,c]))
            else:
                for k,_ in self.depth_bridges[j]:columns.update(self.variables[k].tolist())
        return sorted(columns)

    def motion_groups(self,t):
        if not self.temporal or len(t)<2:return []
        dt=np.diff(self.times);delta=np.einsum('nij,nj->ni',self.camera_rotations,t-self.reference_camera)
        sigma=.03*np.sqrt(dt/max(float(np.median(dt)),1e-6))
        groups=[('correction_smooth',(np.diff(delta,axis=0)/sigma[:,None]).ravel())]
        if len(t)>2:
            world=np.einsum('nij,nj->ni',self.camera_rotations,t)+self.inputs.cameras[self.indices,:3,3]
            velocity=np.diff(world,axis=0)/dt[:,None]
            acceleration=np.diff(velocity,axis=0)/((dt[:-1]+dt[1:])*.5)[:,None]
            groups.append(('motion_acceleration',(acceleration/self.acceleration_sigma_m_s2).ravel()))
        return groups

    def loss_for_groups(self,groups):
        robust=np.concatenate([np.full(len(v),':' in name or not self.temporal,bool) for name,v in groups])
        def loss(z):
            rho=np.vstack((z.copy(),np.ones_like(z),np.zeros_like(z)))
            a=1.+z[robust];rho[0,robust]=2.*(np.sqrt(a)-1.)
            rho[1,robust]=a**-.5;rho[2,robust]=-.5*a**-1.5
            return rho
        return loss

    def view(self,frame,s,t):
        # Query an unscaled BVH. Uniformly scaling the model is equivalent to t/s.
        X,z,mask=self.mesh.raster(frame.raster_K,frame.R_camera,t/s,frame.raster_wh)
        return X,z*s,mask

    def observed_correspondences(self,frame,s,t,view):
        q=frame.points_camera
        if not len(q):return np.empty((0,3)),np.empty(0,bool)
        X,z,hit=self.mesh.query(frame.uv,frame.K,frame.R_camera,t/s)
        if np.any(~hit):
            xv,_,visible=view;points=xv[visible]
            if not len(points):raise ValueError('candidate has no projectable visible triangles')
            cam=(s*points)@frame.R_camera.T+t
            _,idx=cKDTree(cam).query(q[~hit],workers=1)
            X[~hit]=points[idx]
        return X,hit

    def contour_correspondences(self,frame,view,max_points=64):
        X,_,mask=view
        mb=boundary(mask)&~frame.contour_exclusion;tb=boundary(frame.appearance_mask)&~frame.contour_exclusion
        my,mx=np.nonzero(mb);ty,tx=np.nonzero(tb)
        if len(mx)<3 or len(tx)<3:return np.empty((0,3)),np.empty((0,2))
        mxy=np.c_[mx,my];txy=np.c_[tx,ty];mi=subset(np.arange(len(mx)),max_points);ti=subset(np.arange(len(tx)),max_points)
        _,j=cKDTree(txy).query(mxy[mi],workers=1);_,k=cKDTree(mxy).query(txy[ti],workers=1)
        points=np.r_[X[my[mi],mx[mi]],X[my[k],mx[k]]]
        uv=np.r_[txy[j],txy[ti]].astype(float)
        uv=(uv+.5)*(np.array(self.inputs.wh)/np.array(frame.raster_wh))-.5
        return points,uv

    def correspondences(self,s,t):
        depth=[];contour=[]
        for j,f in enumerate(self.frames):
            view=self.view(f,s,t[j]);X,hit=self.observed_correspondences(f,s,t[j],view)
            normals=self.mesh.normals(X)@f.R_camera.T if len(X) else np.empty((0,3))
            depth.append((X,f.points_camera,hit,normals));contour.append(self.contour_correspondences(f,view))
        return depth,contour

    def measure(self,s,t):
        rows=[];merits=[]
        for j,f in enumerate(self.frames):
            view=self.view(f,s,t[j]);X,hit=self.observed_correspondences(f,s,t[j],view)
            predicted=(s*X)@f.R_camera.T+t[j]
            distance=np.linalg.norm(predicted-f.points_camera,axis=1) if len(X) else np.empty(0)
            depth_error=predicted[:,2]-f.points_camera[:,2] if len(X) else np.empty(0)
            mask=view[2];domain=~f.hand_unknown;target=f.appearance_mask
            union=int(np.count_nonzero((mask|target)&domain));iou=float(np.count_nonzero(mask&target&domain)/max(1,union))
            coverage=float(np.mean(hit)) if len(hit) else None
            merit=.02*(1-iou) if np.any(target&domain) else 0.
            if len(distance):
                merit+=float(np.mean(np.sqrt(distance**2+.005**2)-.005))+.02*(1-coverage)
            merits.append(merit)
            rows.append({'frame_idx':int(self.inputs.ids[f.position]),'metric_samples':len(X),'first_hit_coverage':coverage,
                'visible_surface_distance_m':stats(distance),'first_hit_abs_depth_m':stats(np.abs(depth_error[hit])) if len(hit) else {'count':0},
                'visible_projection_IoU':iou,'physical_boundary_pixels':int(np.count_nonzero(boundary(target)&~f.contour_exclusion)),
                'score_population_fixed':True})
        observation_merit=float(np.average(merits,weights=self.weights))+.0003*np.log(s/self.scale0)**2
        temporal_merit=.0003*sum(float(v@v) for _,v in self.motion_groups(t))/len(self.frames)
        return {'merit_m':observation_merit+temporal_merit,'observation_merit_m':observation_merit,
                'temporal_merit_m':temporal_merit,'scale':float(s),'frames':rows}

    def residual_groups(self,x,depth,contour):
        s,t=self.unpack(x);groups=[]
        for j,f in enumerate(self.frames):
            X,q,hit,normals=depth[j];P=(s*X)@f.R_camera.T+t[j]
            w=np.sqrt(f.weight);delta=P-q
            # A ray hit identifies a surface, not a material point. Point-to-point
            # tangential residuals would incorrectly pin XY on a flat visible face.
            plane=hit&(np.linalg.norm(normals,axis=1)>.5)
            delta[plane]=normals[plane]*np.sum(normals[plane]*delta[plane],axis=1,keepdims=True)
            groups.append((f'surface:{j}',(delta/.005*w).ravel()))
            Y,uv=contour[j];C=(s*Y)@f.R_camera.T+t[j]
            groups.append((f'contour:{j}',((project(C,f.K)-uv)/2.*w).ravel()))
        groups.append(('cheirality',np.maximum(.01-(s*self.min_model_z+t[:,2]),0)/.005))
        groups.append(('translation_prior',((t-self.reference_camera)/.3).ravel()))
        if self.fit_scale:groups.append(('scale_prior',np.array([np.log(s/self.scale0)/1.5])))
        groups.extend(self.motion_groups(t))
        return groups

    def sparsity(self,groups):
        count=sum(len(v) for _,v in groups);J=lil_matrix((count,self.size),dtype=np.int8);at=0
        for name,v in groups:
            length=len(v)
            if ':' in name:
                j=int(name.split(':')[1]);J[at:at+length,self.dependencies(j)]=1
                if self.fit_scale:J[at:at+length,0]=1
            elif name=='cheirality':
                for j in range(len(self.frames)):
                    J[at+j,self.dependencies(j,2)]=1
                    if self.fit_scale:J[at+j,0]=1
            elif name=='translation_prior':
                for j in range(len(self.frames)):
                    for c in range(3):J[at+3*j+c,self.dependencies(j,c)]=1
            elif name=='scale_prior':J[at,0]=1
            elif name in ('correction_smooth','motion_acceleration'):
                span=2 if name=='correction_smooth' else 3
                for j in range(len(self.frames)-span+1):
                    columns=sorted(set(c for k in range(j,j+span) for c in self.dependencies(k)))
                    J[at+3*j:at+3*j+3,columns]=1
            at+=length
        return J.tocsr()

    def solve(self,outer_iterations=12,max_nfev=30):
        s,t=self.unpack(self.pack(self.scale0,self.t0));initial=self.measure(s,t);current=initial;trace=[]
        for outer in range(outer_iterations):
            depth,contour=self.correspondences(s,t);x0=self.pack(s,t)
            def residual(x):return np.concatenate([v for _,v in self.residual_groups(x,depth,contour)])
            groups=self.residual_groups(x0,depth,contour);lb=np.full(self.size,-np.inf);ub=np.full(self.size,np.inf)
            scales=np.full(self.size,.02)
            if self.fit_scale:lb[0]=np.log(.25);ub[0]=np.log(4.);scales[0]=.2
            fit=least_squares(residual,x0,jac_sparsity=self.sparsity(groups),method='trf',tr_solver='lsmr',loss=self.loss_for_groups(groups),f_scale=1.,x_scale=scales,bounds=(lb,ub),max_nfev=max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5)
            if not np.isfinite(fit.x).all():raise RuntimeError('nonfinite surface registration')
            accepted=False;fraction=0.
            for fraction in (1.,.5,.25,.125):
                candidate=x0+fraction*(fit.x-x0);cs,ct=self.unpack(candidate)
                try:measurement=self.measure(cs,ct)
                except ValueError:continue
                if measurement['merit_m']<current['merit_m']-1e-8:
                    s,t,current=cs,ct,measurement;accepted=True;break
            trace.append({'outer_iteration':outer,'nfev':fit.nfev,'fixed_correspondence_cost':float(fit.cost),'accepted_on_fixed_population':accepted,'step_fraction':float(fraction) if accepted else 0.,'scale':float(s),'validated_merit_m':current['merit_m']})
            if not accepted:break
        report={'method':'visible_triangle_first_surface_uniform_scale_and_independent_translation','scale_fitted':self.fit_scale,
            'scale_initial':self.scale0,'residual_uniform_scale':float(s),'scale_at_search_boundary':bool(s<.2501 or s>3.999),
            'rotation_changed':False,'depth_observations_changed':False,'full_object_dimensions_directly_observed':False,
            'scale_uniqueness_certified':False,'before':initial,'after':current,'iterations':trace,
            'motion_acceleration_sigma_m_s2':self.acceleration_sigma_m_s2 if self.temporal else None,
            'missing_metric_depth_inference':[{'frame_idx':int(self.inputs.ids[self.frames[j].position]),
                'mode':'two_sided_world_correction_bridge' if len(anchors)==2 else 'one_sided_world_correction',
                'anchors':[{'frame_idx':int(self.inputs.ids[self.frames[k].position]),'weight':w} for k,w in anchors],
                'raw_depth_created':False,'current_RGB_motion_preserved':True} for j,anchors in self.depth_bridges.items()]}
        return float(s),t,report


def camera_translations_to_world(inputs,indices,translations,reference):
    out=np.array(reference,copy=True)
    for i,t in zip(indices,translations):out[i]=inputs.cameras[i,:3,:3]@t+inputs.cameras[i,:3,3]
    return out


def select_scale_keyframes(inputs,tracking_rows,maximum=5):
    eligible=np.flatnonzero(inputs.cloud_valid);anchor=inputs.anchor
    if anchor not in eligible:raise ValueError('appearance anchor requires a qualified fixed visible surface')
    if maximum<1:raise ValueError('at least one scale keyframe is required')
    if maximum==1:return [int(anchor)]
    selected={int(anchor)}
    for block in np.array_split(eligible,max(1,maximum-1)):
        if not len(block):continue
        def quality(i):
            owned=getattr(inputs,'owned_masks',None)
            fraction=float(np.count_nonzero(owned[i])/max(1,np.count_nonzero(inputs.masks[i]))) if owned is not None else 1.
            points=float(tracking_rows[i].get('pnp_inliers',0));error=tracking_rows[i].get('pnp_median_px')
            motion=.25+min(1.,points/60.)*(np.exp(-float(error)/3.) if error is not None else .25)
            return min(1.,fraction)*motion
        selected.add(int(max(block,key=quality)))
    return sorted(selected)[:maximum]
