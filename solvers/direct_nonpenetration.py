"""Parameter-space nonpenetration solver; no video factors and no PPO.

Linear feasibility has priority; complete triangle checks accept/reject physical
steps. Reference/temporal costs only choose smaller feasible linear steps.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog,minimize,OptimizeWarning
from rewards.three_rewards import NativeMoments
from utils.complete_triangle_collision import CompleteTriangleScene,intersection_barycentric

CONFIG={'contract':'direct_native_nonpenetration.v1','clearance_m':.0001,'inside_tolerance_m':.00001,
        'triangle_tolerance_m':1e-8,'max_iterations':12,'hand_translation_max_m':.008,
        'object_translation_max_m':.004,'root_rotation_max_rad':np.radians(5.),
        'articulation_rotation_max_rad':np.radians(8.),'object_rotation_max_rad':np.radians(5.),
        'translation_trust_m':.0015,'rotation_trust_rad':np.radians(1.5),
        'finite_difference_rad':.001,'temporal_tiebreak_weight':.1,'surface_sigma_m':.005}


def mano_batch(episode,side,i,physical):
    import torch
    q=np.asarray(physical,float).reshape(-1,51);h=episode.hands[side]
    R0=Rotation.from_rotvec(h['root_orient'][i]).as_matrix()
    P0=Rotation.from_rotvec(h['articulation'][i].reshape(15,3)).as_matrix()
    R=Rotation.from_rotvec(q[:,:3]).as_matrix()@R0
    P=Rotation.from_rotvec(q[:,3:48].reshape(-1,3)).as_matrix().reshape(-1,15,3,3)@P0
    trans=h['translation'][i]+q[:,48:51];beta=np.repeat(np.asarray(h['betas'][i])[None],len(q),axis=0)
    def t(a):return torch.tensor(np.asarray(a),dtype=torch.float32)
    with torch.no_grad():
        out=episode.mano_models[side](global_orient=t(R[:,None]),hand_pose=t(P),betas=t(beta),transl=t(trans),return_verts=True,pose2rot=False)
    return out.vertices.detach().numpy().astype(float),out.joints.detach().numpy().astype(float)


class ParameterGeometry:
    def __init__(self,episode,cfg):
        self.episode=episode;self.cfg=cfg;self.layout=episode.layout
        if self.layout.size!=118:raise ValueError('requires original 118D layout')
        self.scales=np.zeros(108);self.bounds=np.zeros(108);self.trust=np.zeros(108)
        for block in self.layout.blocks:
            if block.name.startswith('contact.'):continue
            is_trans=block.name.endswith('.translation');scale=.005 if is_trans else .05
            if block.name.startswith('hand.'):
                limit=cfg['hand_translation_max_m'] if is_trans else cfg['articulation_rotation_max_rad'] if block.name.endswith('.articulation') else cfg['root_rotation_max_rad']
            else:limit=cfg['object_translation_max_m'] if is_trans else cfg['object_rotation_max_rad']
            self.scales[block.slice]=scale;self.bounds[block.slice]=limit/np.sqrt(3)/scale
            self.trust[block.slice]=(cfg['translation_trust_m'] if is_trans else cfg['rotation_trust_rad'])/scale
        self.obj=episode.objects['object:0'];self.moments=NativeMoments.from_vertices(self.obj['vertices_canonical'])
        vv=np.asarray(self.obj['vertices_canonical'],float)
        self.moment4=np.block([[vv.T@vv/len(vv),vv.mean(0)[:,None]],[vv.mean(0)[None],np.ones((1,1))]])

    def physical(self,q):return np.r_[np.asarray(q)*self.scales,np.zeros(10)]

    def forward(self,i,q,*,jacobian=False):
        physical=self.physical(q);result={'hands':{},'physical':physical}
        for side in ('left','right'):
            offset=self.layout.block(f'hand.{side}.root').start;hq=physical[offset:offset+51]
            v,j=mano_batch(self.episode,side,i,hq)
            entry={'vertices':v[0],'joints':j[0]}
            if jacobian:
                fd=self.cfg['finite_difference_rad'];batch=np.repeat(hq[None],96,axis=0)
                for k in range(48):batch[2*k,k]+=fd;batch[2*k+1,k]-=fd
                bv,_=mano_batch(self.episode,side,i,batch)
                J=np.zeros((len(v[0]),3,51));J[:,:,:48]=((bv[::2]-bv[1::2])/(2*fd)).transpose(1,2,0)
                J[:,:,48:51]=np.eye(3)[None]
                entry['jacobian']=J*self.scales[offset:offset+51][None,None,:]
            result['hands'][side]=entry
        r=physical[102:105];t=physical[105:108]
        R=Rotation.from_rotvec(r).as_matrix()@np.asarray(self.obj['rotation'][i])
        frame=self.cfg.get('object_translation_frame','world')
        T=np.asarray(self.episode.T_world_camera[i,:3,:3]) if frame=='camera' else np.eye(3)
        translation=np.asarray(self.obj['translation'][i])+T@t
        result.update(rotation=R,translation=translation)
        if jacobian:
            D=np.zeros((6,4,3));fd=self.cfg['finite_difference_rad']
            for k in range(3):
                a=r.copy();b=r.copy();a[k]+=fd;b[k]-=fd
                D[k,:3]=((Rotation.from_rotvec(a).as_matrix()-Rotation.from_rotvec(b).as_matrix())@self.obj['rotation'][i]).T/(2*fd)*self.scales[102+k]
            D[3:,3,:]=T.T*self.scales[105:108,None]
            result['object_derivatives']=D
        return result

    def quadratic(self,i,geometry,previous):
        H=np.eye(108)*1e-6;g=np.zeros(108);w=self.cfg['temporal_tiebreak_weight'] if previous is not None else 0.
        sigma2=self.cfg['surface_sigma_m']**2
        for side in ('left','right'):
            h=geometry['hands'][side];v=h['vertices'];J=h['jacobian'];offset=self.layout.block(f'hand.{side}.root').start
            delta=v-self.episode.hands[side]['vertices'][i]
            old=previous['hand_corrections'][side] if previous is not None else 0.
            index=slice(offset,offset+51)
            H[index,index]+=(1+w)*np.einsum('vci,vcj->ij',J,J)/len(v)/sigma2
            g[index]+=np.einsum('vci,vc->i',J,(1+w)*delta-w*old)/len(v)/sigma2
        D=geometry['object_derivatives'];M=self.moment4
        C=np.vstack((geometry['rotation'].T-self.obj['rotation'][i].T,geometry['translation']-self.obj['translation'][i]))
        old=previous['object_correction'] if previous is not None else 0.
        H[102:108,102:108]+=(1+w)*np.einsum('iac,ab,jbc->ij',D,M,D)/sigma2
        g[102:108]+=np.einsum('iac,ab,bc->i',D,M,(1+w)*C-w*old)/sigma2
        return (H+H.T)*.5,g

    def correction_reference(self,i,geometry):
        return {'hand_corrections':{s:geometry['hands'][s]['vertices']-self.episode.hands[s]['vertices'][i] for s in ('left','right')},
                'object_correction':np.vstack((geometry['rotation'].T-self.obj['rotation'][i].T,geometry['translation']-self.obj['translation'][i]))}

    def surface_cost(self,i,geometry):
        hand=sum(np.square(geometry['hands'][s]['vertices']-self.episode.hands[s]['vertices'][i]).sum(-1).mean() for s in ('left','right'))
        obj=self.moments.displacement_squared(geometry['rotation'].T-self.obj['rotation'][i].T,geometry['translation']-self.obj['translation'][i])
        return float(hand+obj)


def linear_feasible_step(A,b,H,g,lower,upper):
    """Minimax feasibility LP, then min-surface quadratic in its feasible set."""
    n=len(g)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected')
        lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.c_[-A,-np.ones(len(A))],b_ub=-b,
                   bounds=list(zip(lower,upper))+[(0.,None)],method='highs',
                   options={'threads':1,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not lp.success:return np.zeros(n),{'lp_success':False,'lp_message':lp.message,'slack_mm':None,'qp_success':False}
    slack=float(lp.x[-1]);target=b-slack-1e-7
    # Scale convex objective for stable SLSQP tolerance without changing minimizer.
    objective_scale=max(1.,float(np.linalg.norm(H,ord=np.inf)))
    def fun(x):return (.5*x@H@x+g@x)/objective_scale
    def jac(x):return (H@x+g)/objective_scale
    qp=minimize(fun,lp.x[:-1],jac=jac,bounds=list(zip(lower,upper)),
                constraints=[{'type':'ineq','fun':lambda x:A@x-target,'jac':lambda x:A}],
                method='SLSQP',options={'maxiter':60,'ftol':1e-9,'disp':False})
    valid=bool(np.isfinite(qp.x).all() and np.min(A@qp.x-target)>=-1e-6)
    step=qp.x if valid else lp.x[:-1]
    return step,{'lp_success':True,'slack_mm':slack,'qp_success':bool(qp.success),'qp_feasible':valid,
                 'qp_message':qp.message,'qp_iterations':int(qp.nit)}


class DirectNonpenetrationSolver:
    def __init__(self,episode,cfg=None):
        self.cfg={**CONFIG,**dict(cfg or {})};self.geometry=ParameterGeometry(episode,self.cfg);self.episode=episode
        self.scene=CompleteTriangleScene(self.geometry.obj['vertices_canonical'],self.geometry.obj['faces'])
        self.previous=None;self.previous_q=np.zeros(108)

    def audit(self,geometry,*,nsamples=5):
        hands={s:self.scene.inspect_hand(geometry['hands'][s]['vertices'],self.episode.hands[s]['faces'],
                   geometry['rotation'],geometry['translation'],nsamples=nsamples,tolerance_m=self.cfg['triangle_tolerance_m']) for s in ('left','right')}
        count=sum(r['intersection_pair_count'] for r in hands.values());depth=max(r['max_inside_depth_m'] for r in hands.values())
        sq=sum(r['inside_squared_sum_m2'] for r in hands.values())
        merit=depth/.0001+np.sqrt(sq/(2*(778+1538)))/.0001+count/1000.
        feasible=bool(count==0 and depth<=self.cfg['inside_tolerance_m'])
        return {'hands':hands,'intersection_pair_count':count,'max_inside_depth_m':depth,'inside_squared_sum_m2':sq,'merit':float(merit),'feasible':feasible}

    def constraints(self,geometry,audit):
        R,t=geometry['rotation'],geometry['translation'];D=geometry['object_derivatives'];As=[];bs=[]
        for side in ('left','right'):
            h=geometry['hands'][side];v=h['vertices'];J=h['jacobian'];faces=np.asarray(self.episode.hands[side]['faces'],int)
            if faces.ndim==3:raise ValueError('fixed MANO topology required')
            allJ=np.concatenate((J,J[faces].mean(1)));points=np.concatenate((v,v[faces].mean(1)))
            row=audit['hands'][side];near=row['near'];sdf=near['signed_distance_m']
            ids=np.flatnonzero(sdf<self.cfg['inside_tolerance_m'])
            constraint_points=[points[ids]];constraint_J=[allJ[ids]]
            # Actual triangle/triangle intersection barycentric witnesses. Do NOT
            # extend the object's tiny local plane to distant hand vertices.
            pair=row['intersections'];hf=pair['hand_face_ids'];of=pair['object_face_ids']
            if len(hf):
                local=(v-t)@R;bar,_=intersection_barycentric(local[faces[hf]],self.scene.triangles[of])
                # Mean of witnesses on each hand face is still ON that real face.
                unique,inverse=np.unique(hf,return_inverse=True);weights=np.zeros((len(unique),3));counts=np.bincount(inverse)
                np.add.at(weights,inverse,bar);weights/=counts[:,None]
                ids=faces[unique];Pcut=np.einsum('nv,nvc->nc',weights,v[ids]);Jcut=np.einsum('nv,nvck->nck',weights,J[ids])
                constraint_points.append(Pcut);constraint_J.append(Jcut)
            P=np.concatenate(constraint_points);JJ=np.concatenate(constraint_J)
            if not len(P):continue
            # A thin generated body can put one finger across BOTH sides. Nearest
            # exits then oppose each other. Select an explicit coherent ± normal
            # mode, tracing the real mesh entry/exit rather than extrapolating.
            if len(of):mode_normals=self.scene.face_normals[of]@R.T
            else:mode_normals=near['normals_world'][ids]
            _,axes=np.linalg.eigh(mode_normals.T@mode_normals/max(1,len(mode_normals)));direction=axes[:,-1]
            exits=[self.scene.directed_exit(P,d,R,t) for d in (direction,-direction)]
            chosen=min(exits,key=lambda e:(float(e['travel_m'].max()),float(np.square(e['travel_m']).mean())))
            N=chosen['normals_world'];b=np.einsum('nc,nc->n',N,chosen['target_world']-P)+self.cfg['clearance_m']
            A=np.zeros((len(b),108));offset=self.episode.layout.block(f'hand.{side}.root').start
            A[:,offset:offset+51]=np.einsum('nc,nck->nk',N,JJ)
            local=(P-t)@R;hom=np.c_[local,np.ones(len(local))]
            objectJ=np.einsum('na,kac->nck',hom,D)
            A[:,102:108]=-np.einsum('nc,nck->nk',N,objectJ)
            As.append(A/.001);bs.append(b/.001)
        return (np.concatenate(As),np.concatenate(bs)) if As else (np.zeros((0,108)),np.zeros(0))

    def solve_frame(self,i):
        q=np.zeros(108);geometry=self.geometry.forward(i,q);audit=self.audit(geometry);initial=audit
        if i>0:
            warm=self.previous_q.copy();wg=self.geometry.forward(i,warm);wa=self.audit(wg)
            if wa['merit']<audit['merit'] or (wa['feasible'] and self.geometry.surface_cost(i,wg)<self.geometry.surface_cost(i,geometry)):
                q,geometry,audit=warm,wg,wa
        history=[]
        for iteration in range(self.cfg['max_iterations']):
            if audit['feasible']:break
            geometry=self.geometry.forward(i,q,jacobian=True)
            A,b=self.constraints(geometry,audit);H,g=self.geometry.quadratic(i,geometry,self.previous)
            lower=np.maximum(-self.geometry.trust,-self.geometry.bounds-q);upper=np.minimum(self.geometry.trust,self.geometry.bounds-q)
            step,receipt=linear_feasible_step(A,b,H,g,lower,upper)
            candidates=[]
            for fraction in (1.,.5,.25,.125):
                nq=q+fraction*step;ng=self.geometry.forward(i,nq);na=self.audit(ng)
                candidates.append((na['merit'],self.geometry.surface_cost(i,ng),fraction,nq,ng,na))
                if na['feasible']:break
            best=min(candidates,key=lambda x:(not x[-1]['feasible'],x[0],x[1]))
            accepted=bool(best[-1]['feasible'] or best[0]<audit['merit']-1e-7)
            history.append({'iteration':iteration,'constraint_count':len(b),'before_merit':audit['merit'],'after_merit':best[0],
                            'fraction':best[2],'accepted':accepted,**receipt})
            if not accepted:break
            q,geometry,audit=best[3],best[4],best[5]
        # Feasibility-preserving contraction reduces the possibility of reporting
        # a needlessly separated grasp. No claim of globally minimal solution.
        if audit['feasible']:
            for fraction in (.75,.9,.97):
                nq=q*fraction;ng=self.geometry.forward(i,nq);na=self.audit(ng)
                if na['feasible'] and self.geometry.surface_cost(i,ng)<self.geometry.surface_cost(i,geometry):q,geometry,audit=nq,ng,na
        self.previous=self.geometry.correction_reference(i,geometry);self.previous_q=q.copy()
        return q,geometry,audit,{'frame_idx':int(self.episode.frame_idx[i]),'iterations':len(history),
                 'initial_max_depth_mm':initial['max_inside_depth_m']*1000,'initial_triangle_pairs':initial['intersection_pair_count'],
                 'final_max_depth_mm':audit['max_inside_depth_m']*1000,'final_triangle_pairs':audit['intersection_pair_count'],
                 'feasible':audit['feasible'],'surface_cost_m2':self.geometry.surface_cost(i,geometry),'history':history,
                 'physical_delta_118':self.geometry.physical(q).tolist()}
