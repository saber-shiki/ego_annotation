"""Sparse robust SE3/image refinement with explicitly uncalibrated depth bias."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from preprocessing.geometry import TriangleMesh,shape_matrix,project
from preprocessing.factors import contour_factors,depth_factors,rgb_model_factors,concatenate,observed_surface_factors


class SceneProblem:
    def __init__(self,inputs,R,t,image,depth,*,fit_bias=False,fit_shape=False,shape_initial=None,bias_initial=None,use_depth=False,surface=None):
        self.inputs=inputs;self.R0=np.asarray(R);self.t0=np.asarray(t);self.image=image;self.depth=depth
        self.use_depth=use_depth or fit_bias
        self.surface=surface if surface is not None else (np.zeros(0,int),np.zeros((0,3)),np.zeros((0,3)))
        self.n=len(R);self.fit_bias=fit_bias;self.fit_shape=fit_shape;self.block=7 if fit_bias else 6
        self.shape0=np.zeros(2) if shape_initial is None else np.asarray(shape_initial)
        self.bias0=np.zeros(self.n) if bias_initial is None else np.asarray(bias_initial)
        self.size=self.n*self.block+(2 if fit_shape else 0)
        positions={int(f):i for i,f in enumerate(inputs.ids)}
        self.ea=np.array([positions[e.source_frame_idx] for e in inputs.rgb.edges],int);self.eb=np.array([positions[e.target_frame_idx] for e in inputs.rgb.edges],int)
        self.er=np.asarray([e.rotation_source_to_target_world for e in inputs.rgb.edges]).reshape(-1,3,3)
        self.ew=np.asarray([max(.01,float(e.diagnostics['inlier_fraction'])*np.exp(-float(e.diagnostics['reprojection_median_px'])/3.)) for e in inputs.rgb.edges])
        self.dt=np.diff(inputs.times);self.Rcam=inputs.cameras[:,:3,:3];self.tcam=inputs.cameras[:,:3,3]

    def unpack(self,x):
        blocks=x[:self.n*self.block].reshape(self.n,self.block)
        R=Rotation.from_rotvec(blocks[:,:3]).as_matrix()@self.R0;t=self.t0+blocks[:,3:6]
        bias=self.bias0+blocks[:,6] if self.fit_bias else self.bias0.copy()
        shape=self.shape0+x[-2:] if self.fit_shape else self.shape0.copy()
        return R,t,bias,shape

    def camera_points(self,points,frames,R,t,shape):
        if not len(frames):return np.zeros((0,3))
        S=shape_matrix(shape,self.inputs.shape_basis);p=points@S.T
        world=np.einsum('nij,nj->ni',R[frames],p)+t[frames]
        return np.einsum('nji,nj->ni',self.Rcam[frames],world-self.tcam[frames])

    def groups(self,x):
        R,t,bias,shape=self.unpack(x);im=self.image;f=im.frames
        cam=self.camera_points(im.points,f,R,t,shape);uv=project(cam,self.inputs.K[f])
        groups={'image':((uv-im.uv)*np.sqrt(im.weights[:,None])/3.).ravel(),
                'cheirality':np.minimum(cam[:,2]-.01,0)*300}
        if self.use_depth:
            fi,points,z=self.depth;dc=self.camera_points(points,fi,R,t,shape)
            groups['depth']=(dc[:,2]+bias[fi]-z)/.025
            sf,sx,sy=self.surface;sc=self.camera_points(sx,sf,R,t,shape)
            groups['observed_surface']=((sc-sy)/.025).ravel()
        if len(self.ea):
            rel=R[self.eb]@R[self.ea].transpose(0,2,1)
            groups['RGB_rotation']=(Rotation.from_matrix(rel@self.er.transpose(0,2,1)).as_rotvec()*np.sqrt(self.ew[:,None])/.18).ravel()
        else:groups['RGB_rotation']=np.zeros(0)
        if self.n>=3:
            step=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).as_rotvec()/self.dt[:,None]
            denom=(self.dt[1:]+self.dt[:-1])[:,None]/2.
            groups['rotation_acceleration']=(np.diff(step,axis=0)/denom/180.).ravel()
            velocity=np.diff(t,axis=0)/self.dt[:,None]
            groups['translation_acceleration']=(np.diff(velocity,axis=0)/denom/8.).ravel()
        blocks=x[:self.n*self.block].reshape(self.n,self.block)
        groups['local_pose_prior']=(blocks[:,:6]/np.array([.7,.7,.7,.15,.15,.15])).ravel()
        if self.fit_bias:
            groups['relative_bias_prior']=bias/.12
            groups['relative_bias_smooth']=np.diff(bias)/np.sqrt(self.dt/np.median(self.dt))/.012
            groups['reference_bias_gauge']=np.array([bias[self.inputs.anchor]/1e-4])
        if self.fit_shape:groups['shared_shape_prior']=shape/.08
        return groups

    def residual(self,x):return np.concatenate(list(self.groups(x).values()))

    def sparsity(self):
        groups=self.groups(np.zeros(self.size));rows=[];cols=[];offset=0;B=self.block
        def attach(local_rows,frame_ids,axes):
            lr=np.asarray(local_rows,int).ravel()+offset;fid=np.asarray(frame_ids,int).ravel()
            if len(lr)!=len(fid):raise ValueError('sparsity dimension mismatch')
            for axis in axes:rows.extend(lr.tolist());cols.extend((fid*B+axis).tolist())
        for name,value in groups.items():
            length=len(value)
            if name in ('image','cheirality'):
                frames=np.repeat(self.image.frames,2) if name=='image' else self.image.frames
                attach(np.arange(length),frames,range(6))
            elif name=='depth':attach(np.arange(length),self.depth[0],range(B))
            elif name=='observed_surface':attach(np.arange(length),np.repeat(self.surface[0],3),range(6))
            elif name=='RGB_rotation':
                for endpoint in (self.ea,self.eb):attach(np.arange(length),np.repeat(endpoint,3),range(3))
            elif name in ('rotation_acceleration','translation_acceleration'):
                axes=range(3) if name=='rotation_acceleration' else range(3,6)
                for shift in range(3):attach(np.arange(length),np.repeat(np.arange(self.n-2)+shift,3),axes)
            elif name=='local_pose_prior':
                for i in range(self.n):
                    for j in range(6):rows.append(offset+i*6+j);cols.append(i*B+j)
            elif name=='relative_bias_prior':attach(np.arange(length),np.arange(self.n),[6])
            elif name=='relative_bias_smooth':
                attach(np.arange(length),np.arange(self.n-1),[6]);attach(np.arange(length),np.arange(1,self.n),[6])
            elif name=='reference_bias_gauge':attach([0],[self.inputs.anchor],[6])
            elif name=='shared_shape_prior':
                rows.extend([offset,offset+1]);cols.extend([self.n*B,self.n*B+1])
            if self.fit_shape and name in ('image','cheirality','depth','observed_surface'):
                for c in [self.n*B,self.n*B+1]:rows.extend(range(offset,offset+length));cols.extend([c]*length)
            offset+=length
        return coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(offset,self.size)).tocsr()

    def solve(self,max_nfev=20):
        start=time.monotonic();x0=np.zeros(self.size);before=self.groups(x0)
        lower=np.full(self.size,-np.inf);upper=np.full(self.size,np.inf)
        if self.fit_shape:
            lower[-2:]=-.14-self.shape0;upper[-2:]=.14-self.shape0
        scale=np.tile(np.array([.1,.1,.1,.01,.01,.01]+([.03] if self.fit_bias else [])),self.n)
        if self.fit_shape:scale=np.r_[scale,[.03,.03]]
        fit=least_squares(self.residual,x0,jac_sparsity=self.sparsity(),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=1.,bounds=(lower,upper),x_scale=scale,max_nfev=max_nfev,ftol=1e-4,xtol=1e-4,gtol=1e-4)
        if not np.isfinite(fit.x).all():raise RuntimeError('nonfinite coarse solver result')
        after=self.groups(fit.x)
        diag={'seconds':time.monotonic()-start,'nfev':fit.nfev,'status':int(fit.status),'message':fit.message,'cost':float(fit.cost),'optimality':float(fit.optimality),
            'residuals':{k:{'count':len(v),'rms_before':float(np.sqrt(np.mean(before[k]**2))) if len(v) else None,'rms_after':float(np.sqrt(np.mean(v**2))) if len(v) else None} for k,v in after.items()}}
        return self.unpack(fit.x),diag


def refine(inputs,R,t,tracks,hand_depth,max_nfev=18):
    from dataclasses import replace
    from preprocessing.metric_alignment import estimate_relative_depth
    calibration=estimate_relative_depth(inputs)
    adjusted=replace(inputs,clouds=[p*calibration['scale'][i] for i,p in enumerate(inputs.clouds)])
    shape=np.zeros(2);bias=np.zeros(len(inputs.ids));diagnostics=[]
    mesh=TriangleMesh(inputs.vertices,inputs.faces)
    # No free per-frame additive bias can absorb a displaced object in these passes.
    for phase,use_depth,shape_fit in [('image_pose_graph',False,False),('relative_depth_consistency',True,False),('shared_geometry',True,True)]:
        current=adjusted if use_depth else inputs
        S=shape_matrix(shape,inputs.shape_basis)
        query_mesh=mesh if not np.any(shape) else TriangleMesh(inputs.vertices@S.T,inputs.faces)
        contours,contour_stats=contour_factors(query_mesh,R,t,current,hand_depth)
        rgb=rgb_model_factors(query_mesh,R,t,current)
        depth=depth_factors(query_mesh,R,t,current)
        surface=observed_surface_factors(query_mesh,R,t,current)
        invS=np.linalg.inv(S)
        contours.points=contours.points@invS.T;rgb.points=rgb.points@invS.T
        depth=(depth[0],depth[1]@invS.T,depth[2]);surface=(surface[0],surface[1]@invS.T,surface[2])
        image=concatenate([tracks,contours,rgb],phase)
        problem=SceneProblem(current,R,t,image,depth,use_depth=use_depth,surface=surface,fit_shape=shape_fit,shape_initial=shape)
        (R,t,bias,shape),diagnostic=problem.solve(max_nfev)
        diagnostic.update(phase=phase,contours=contour_stats,image_factor_count=len(image.frames),depth_samples=len(depth[0]),surface_samples=len(surface[0]),shared_log_stretch=shape.tolist(),free_depth_bias_enabled=False)
        diagnostics.append(diagnostic)
        print(f'{phase}: cost={diagnostic["cost"]:.3f} nfev={diagnostic["nfev"]} seconds={diagnostic["seconds"]:.1f}',flush=True)
    return R,t,calibration['bias_diagnostic_m'],shape,diagnostics,calibration
