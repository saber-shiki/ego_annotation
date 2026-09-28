"""Bounded, prediction-only experiments on RGB tracklets and conditional mesh bindings.
Not a replacement for the default producer. Baseline is a frozen prediction cache.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,time,traceback,subprocess
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from preprocessing.inputs import load_inputs,sha,reject_evaluation_path
from preprocessing.geometry import TriangleMesh,camera_pose,world_pose,project,resized_intrinsics
from preprocessing.factors import hand_depths,subset
from preprocessing.tracking import flow_track
from preprocessing.relocalization import pnp_hypotheses
from preprocessing.visible_surface import VisibleSurfaceRegistration,camera_translations_to_world,select_scale_keyframes,cloud_digest
from preprocessing.surface_pipeline import code_identity,write_json
from preprocessing.render import paint


@dataclass
class Evidence:
    points: np.ndarray
    uv: np.ndarray
    identity: np.ndarray
    reference: np.ndarray
    valid_binding: np.ndarray

    @property
    def hold(self):return self.identity%3==0


def binding_visibility(mesh,X,uv,K,R,t):
    if not len(X):return np.zeros(0,bool)
    cam=X@R.T+t;pred=project(cam,K);_,z,hit=mesh.query(pred,K,R,t)
    return hit&(cam[:,2]>.01)&(np.abs(cam[:,2]-z)<.005)&(np.linalg.norm(pred-uv,axis=1)<4.)


def normalized_weights(refs):
    if not len(refs):return np.empty(0)
    _,inv,count=np.unique(refs,return_inverse=True,return_counts=True)
    w=1./count[inv];return w/w.sum()


def tracklets(inputs,mesh,R,t):
    """2D trajectories are generated without any candidate-dependent filtering."""
    n=len(inputs.ids);rows=[[] for _ in range(n)];saved=[];next_id=0
    keys=sorted(set(range(0,n,8))|{inputs.anchor,n-1})
    for ref in keys:
        mask=cv2.erode(inputs.masks[ref].astype('uint8'),np.ones((5,5),np.uint8))*255
        raw=cv2.goodFeaturesToTrack(inputs.grays[ref],maxCorners=160,qualityLevel=.008,minDistance=5,mask=mask)
        if raw is None:continue
        uv=raw.reshape(-1,2);ids=np.arange(next_id,next_id+len(uv));next_id+=len(uv)
        rc,tc=camera_pose(R[ref],t[ref],inputs.cameras[ref]);X,_,hit=mesh.query(uv,inputs.K[ref],rc,tc)
        # Never drop a 2D record merely because its conditional mesh binding missed.
        for direction in (-1,1):
            current=uv.copy();local=np.arange(len(uv));previous=ref
            for i in range(ref+direction,max(-1,ref-9) if direction<0 else min(n,ref+9),direction):
                moved,valid=flow_track(inputs,previous,i,current);local=local[valid];current=moved[valid]
                if not len(local):break
                rows[i].append((X[local].copy(),current.copy(),ids[local].copy(),np.full(len(local),ref),hit[local].copy()))
                for ident,p,bind in zip(ids[local],current,hit[local]):saved.append((int(ident),ref,i,float(p[0]),float(p[1]),bool(bind)))
                previous=i
    evidence=[]
    for records in rows:
        if records:evidence.append(Evidence(*(np.concatenate([r[k] for r in records]) for k in range(5))))
        else:evidence.append(Evidence(np.empty((0,3)),np.empty((0,2)),np.empty(0,int),np.empty(0,int),np.empty(0,bool)))
    return evidence,keys,np.asarray(saved,float).reshape(-1,6)


def fixed_rgb_score(e,R,t,K,keep):
    if not np.any(keep):return None
    X=e.points[keep];uv=e.uv[keep];valid=e.valid_binding[keep];p=X@R.T+t
    err=np.full(len(X),24.);ok=valid&(p[:,2]>.01)&np.isfinite(p).all(1)
    err[ok]=np.minimum(np.linalg.norm(project(p[ok],K)-uv[ok],axis=1),24.)
    return float(normalized_weights(e.reference[keep])@err/8.)


def observability(e,R,t,K,radius):
    keep=e.valid_binding&~e.hold
    if np.count_nonzero(keep)<6:return {'available':False,'reason':'fewer_than_six_bound_train_observations'}
    X=e.points[keep];uv=e.uv[keep];w=np.sqrt(normalized_weights(e.reference[keep]))
    def residual(x):
        rr=Rotation.from_rotvec(x[:3]).as_matrix()@R;tt=t+x[3:]*radius
        return (((project(X@rr.T+tt,K)-uv)/2.)*w[:,None]).ravel()
    zero=np.zeros(6);base=residual(zero);J=np.column_stack([(residual(np.eye(6)[j]*1e-5)-base)/1e-5 for j in range(6)])
    _,s,V=np.linalg.svd(J,full_matrices=False)
    return {'available':True,'singular_values':s.tolist(),'min_over_max':float(s[-1]/max(s[0],1e-12)),
            'weakest_direction_rot_rad_translation_per_radius':V[-1].tolist(),'translation_normalization_m':radius,
            'conditional_on_baseline_bindings':True,'covariance_calibrated':False}


def choose(scores,mode):
    totals=[]
    for s in scores:
        fields=['rgb','mask','surface']
        if mode=='no_rgb':fields.remove('rgb')
        if mode=='no_mask':fields.remove('mask')
        if mode=='no_surface':fields.remove('surface')
        # Same candidate bank; raw_mask and no_mano intentionally coincide:
        # these scoring masks are the only explicit downstream MANO coupling.
        totals.append(sum((s['raw_mask'] if name=='mask' and mode in ('no_mano','raw_mask') else s[name]) or 0. for name in fields))
    return int(np.argmin(totals)),np.asarray(totals)


def distinct(R,t,bank):
    return not any(Rotation.from_matrix(R@r.T).magnitude()<.015 and np.linalg.norm(t-v)<.003 for r,v,_ in bank)


def candidate_bank(e,mesh,K,R,t):
    train=~e.hold&e.valid_binding
    alive=binding_visibility(mesh,e.points,e.uv,K,R,t)&train
    stale=pnp_hypotheses(e.points[train],e.uv[train],K,R,t,max_candidates=4)
    life=pnp_hypotheses(e.points[alive],e.uv[alive],K,R,t,max_candidates=4)
    bank=[(R.copy(),t.copy(),'baseline')];stale_index=0;life_index=0
    for options,tag in [(stale,'stale'),(life,'lifecycle')]:
        if options:
            r,v,_,_,solver=options[0]
            if distinct(r,v,bank):bank.append((r,v,tag+':'+solver));idx=len(bank)-1
            else:idx=next(j for j,(br,bt,_) in enumerate(bank) if not distinct(r,v,[(br,bt,'')]))
            if tag=='stale':stale_index=idx
            else:life_index=idx
    extras=stale[1:]+life[1:]
    for ref in np.unique(e.reference):
        take=train&(e.reference==ref)
        extras+=pnp_hypotheses(e.points[take],e.uv[take],K,R,t,max_candidates=3)
    # One diverse additional solution after mandatory control lanes. Ranking only
    # uses training pixels, NEVER validation or GT, before common score evaluation.
    extras=sorted(extras,key=lambda v:fixed_rgb_score(e,v[0],v[1],K,train) if np.any(train) else 1e9)
    if len(bank)<4:
        for r,v,_,_,tag in extras:
            if distinct(r,v,bank):bank.append((r,v,'reference_or_alternative:'+tag))
            if len(bank)>=4:break
    info={'train_observations':int(train.sum()),'holdout_observations':int(e.hold.sum()),'unbound_observations':int((~e.valid_binding).sum()),
          'lifecycle_train_observations':int(alive.sum()),'reference_count':len(np.unique(e.reference)),
          'generated_hypotheses':len(stale)+len(life)+len(extras),'retained_candidates':len(bank),
          'stale_index':stale_index,'lifecycle_index':life_index,'support_identity':'reference_keyframe_tracklet',
          'heldout_not_used_in_new_PnP_fit':True,'baseline_and_keyframe_pose_not_statistically_heldout':True}
    return bank,info


def score_candidate(inputs,mesh,hd,i,e,R,t):
    wh=hd.shape[1:][::-1];k=resized_intrinsics(inputs.K[i],inputs.wh,wh)
    _,_,rendered=mesh.raster(k,R,t,wh)
    raw=cv2.resize(inputs.masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
    own=cv2.resize(inputs.owned_masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)>0
    domain=~np.isfinite(hd[i])|own
    def error(d):return float(1.-np.count_nonzero(rendered&raw&d)/max(1,np.count_nonzero((rendered|raw)&d))) if np.any(raw&d) else None
    q=subset(inputs.clouds[i],160);coverage=None;surface=None;median=None
    if len(q):
        _,z,hit=mesh.query(project(q,inputs.K[i]),inputs.K[i],R,t)
        dist=np.full(len(q),.06);dist[hit]=np.minimum(np.abs(z[hit]-q[hit,2]),.06)
        coverage=float(hit.mean());surface=float(np.mean(dist)/.02);median=float(np.median(dist))
    hold=e.hold
    visibility=binding_visibility(mesh,e.points,e.uv,inputs.K[i],R,t)&e.valid_binding
    return {'rgb':fixed_rgb_score(e,R,t,inputs.K[i],hold),'mask':error(domain),'raw_mask':error(np.ones_like(domain)),
      'surface':surface,'surface_coverage':coverage,'fixed_ray_capped_abs_depth_median_m':median,
      'binding_visible_fraction':float(visibility.sum()/max(1,int(e.valid_binding.sum()))),
      'fixed_holdout_count':int(hold.sum()),'surface_sample_count':len(q)}


def render_grids(inputs,base_mesh,refit_mesh,R,t,choices,refit_R,refit_t,out):
    width=240;wh=(width,round(width*inputs.wh[1]/inputs.wh[0]));h=wh[1]+36;n=len(inputs.ids)
    groups={'mechanism':(['RGB','baseline','stale','lifecycle','multi','scale_refit'],3),
            'leave_one':(['multi','no_rgb','no_mask','no_surface','no_mano','raw_mask'],3),
            'candidates':(['candidate0','candidate1','candidate2','candidate3'],2)}
    pipes={};fps=1./float(np.median(np.diff(inputs.times)));selected_images={k:[] for k in groups}
    for name,(labels,cols) in groups.items():
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width*cols}x{h*(len(labels)//cols)}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(out/(name+'.mp4'))]
        pipes[name]=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i,fid in enumerate(inputs.ids):
            rgb=cv2.resize(cv2.imread(inputs.raw_paths[i]),wh,interpolation=cv2.INTER_AREA);K=resized_intrinsics(inputs.K[i],inputs.wh,wh);C=inputs.cameras[i]
            hands=[]
            for side,color in [('left',(225,150,75)),('right',(95,185,225))]:
                hand=inputs.hands[side]
                if not hand['valid'][i]:continue
                faces=hand['faces'][i] if hand['faces'].ndim==3 else hand['faces'];hm=TriangleMesh(hand['vertices'][i],faces)
                _,z,_=hm.raster(K,C[:3,:3].T,-C[:3,:3].T@C[:3,3],wh);hands.append((z,color))
            views={}
            for j in range(4):
                im=rgb.copy();dep=np.full(wh[::-1],np.inf)
                rr,tt=camera_pose(R[i,j],t[i,j],C);_,z,_=base_mesh.raster(K,rr,tt,wh);paint(im,dep,z,(240,160,40),.55)
                for z,col in hands:paint(im,dep,z,col,.45)
                views[j]=im
            rr,tt=camera_pose(refit_R[i],refit_t[i],C);_,z,_=refit_mesh.raster(K,rr,tt,wh);im=rgb.copy();dep=np.full(wh[::-1],np.inf);paint(im,dep,z,(240,160,40),.55)
            for z,col in hands:paint(im,dep,z,col,.45)
            for name,(labels,cols) in groups.items():
                panels=[]
                for label in labels:
                    if label=='RGB':body=rgb
                    elif label=='scale_refit':body=im
                    elif label.startswith('candidate'):body=views[int(label[-1])]
                    else:body=views[int(choices[label][i])]
                    head=np.zeros((36,width,3),np.uint8);cv2.putText(head,f'f{fid} {label}',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
                    cv2.putText(head,'full mesh + source MANO; experiment',(4,29),cv2.FONT_HERSHEY_SIMPLEX,.3,(210,210,210),1)
                    panels.append(np.vstack((head,body)))
                image=np.vstack([np.hstack(panels[j:j+cols]) for j in range(0,len(panels),cols)]);pipes[name].stdin.write(image.tobytes())
                if i in (0,50,100,n-1):
                    cv2.imwrite(str(out/f'{name}_{i:03d}.jpg'),image);selected_images[name].append(image)
        for name,images in selected_images.items():cv2.imwrite(str(out/(name+'_review.jpg')),np.vstack(images))
    finally:
        for pipe in pipes.values():pipe.stdin.close()
        codes=[p.wait() for p in pipes.values()]
    if any(codes):raise RuntimeError('render encoder failed')


def run(source,baseline,output):
    start=time.monotonic();source=Path(source).resolve();baseline=Path(baseline).resolve();output=Path(output).resolve()
    reject_evaluation_path(source);reject_evaluation_path(baseline)
    if output.is_relative_to(source.parent) or output.is_relative_to(baseline):raise ValueError('fresh external output required')
    output.mkdir(parents=True,exist_ok=False);code=code_identity();cv2.setNumThreads(1);cv2.setRNGSeed(47)
    try:
        m=json.loads((baseline/'coarse_scene.json').read_text())
        if sha(source)!=m['source_bundle_sha256'] or sha(baseline/'coarse_scene.npz')!=m['arrays']['sha256']:raise ValueError('baseline/source hash mismatch')
        inputs=load_inputs(source,480,read_owned_masks=True);cloud_hash=cloud_digest(inputs.clouds)
        for q in inputs.clouds:q.setflags(write=False)
        with np.load(baseline/'coarse_scene.npz',allow_pickle=False) as z:base={k:z[k] for k in z.files}
        R0=base['object_rotation'];t0=base['object_translation_world'];scale=float(base['uniform_object_scale']);n=len(R0)
        np.testing.assert_array_equal(base['frame_idx'],inputs.ids)
        mesh=TriangleMesh(base['vertices_canonical'],inputs.faces);hd=hand_depths(inputs,128)
        e,keys,track_rows=tracklets(inputs,mesh,R0,t0);np.savez_compressed(output/'tracklets.npz',rows=track_rows,column_names=np.array(['identity','reference_position','target_position','u','v','binding_hit']))
        modes=['baseline','stale','lifecycle','multi','no_rgb','no_mask','no_surface','no_mano','raw_mask'];choices={k:np.zeros(n,int) for k in modes}
        RR=np.repeat(R0[:,None],4,axis=1);tt=np.repeat(t0[:,None],4,axis=1);valid=np.zeros((n,4),bool);reports=[]
        radius=float(np.sqrt(np.mean(np.sum(base['vertices_canonical']**2,axis=1))))
        for i in range(n):
            rc,tc=camera_pose(R0[i],t0[i],inputs.cameras[i]);bank,info=candidate_bank(e[i],mesh,inputs.K[i],rc,tc)
            scores=[];tags=[]
            for j,(r,v,tag) in enumerate(bank):
                RR[i,j],tt[i,j]=world_pose(r,v,inputs.cameras[i]);valid[i,j]=True;tags.append(tag);scores.append(score_candidate(inputs,mesh,hd,i,e[i],r,v))
            # Padded slots render baseline and are NEVER treated as extra candidates.
            choices['stale'][i]=info['stale_index'];choices['lifecycle'][i]=info['lifecycle_index']
            for mode in modes[3:]:choices[mode][i]=choose(scores,mode)[0]
            _,totals=choose(scores,'multi');order=np.argsort(totals)
            angle=max([float(np.degrees(Rotation.from_matrix(a[0]@b[0].T).magnitude())) for a in bank for b in bank])
            spread=max([float(np.linalg.norm(a[1]-b[1])*1000) for a in bank for b in bank])
            reports.append({'frame_idx':int(inputs.ids[i]),'metric_available':bool(inputs.cloud_valid[i]),'tags':tags,'binding':info,'scores':scores,
              'selection':{k:int(v[i]) for k,v in choices.items()},'normalized_total_scores':totals.tolist(),
              'best_second_score_margin':float(totals[order[1]]-totals[order[0]]) if len(order)>1 else None,
              'candidate_rotation_spread_deg':angle,'candidate_translation_spread_mm':spread,'observability':observability(e[i],rc,tc,inputs.K[i],radius)})
        idx=np.arange(n);rsel=RR[idx,choices['multi']];tsel=tt[idx,choices['multi']]
        print(f'candidate selection complete {n} frames; selected nonbaseline={(choices["multi"]!=0).sum()}',flush=True)
        tracking=json.loads((baseline/'tracking.json').read_text());scale_keys=select_scale_keyframes(inputs,tracking,5)
        weights=np.array([3. if i==inputs.anchor else .5 for i in scale_keys])
        fit=VisibleSurfaceRegistration(inputs,rsel,tsel,hd,scale_keys,fit_scale=True,initial_scale=scale,weights=weights,max_points=512)
        new_scale,kt,scale_report=fit.solve(outer_iterations=8,max_nfev=24)
        seed=camera_translations_to_world(inputs,scale_keys,kt,tsel)
        timeline=VisibleSurfaceRegistration(inputs,rsel,seed,hd,range(n),fit_scale=False,initial_scale=new_scale,temporal=True,max_points=160,motion_reference_world=tsel)
        _,ct,translation_report=timeline.solve(outer_iterations=6,max_nfev=24)
        tfinal=camera_translations_to_world(inputs,range(n),ct,seed)
        write_json(output/'scale_refit.json',scale_report);write_json(output/'translation_refit.json',translation_report)
        write_json(output/'candidates.json',reports)
        arrays={'frame_idx':inputs.ids,'R_candidates':RR,'t_candidates':tt,'candidate_valid':valid,'R_scale_refit':rsel,'t_scale_refit':tfinal,
                'baseline_scale':np.asarray(scale),'refit_scale':np.asarray(new_scale),'vertices_source':inputs.vertices,'faces':inputs.faces,
                'T_world_camera':inputs.cameras,'source_metric_eligible':inputs.cloud_valid}
        arrays.update({'choice_'+k:v for k,v in choices.items()});np.savez_compressed(output/'experiment.npz',**arrays)
        render_grids(inputs,mesh,TriangleMesh(inputs.vertices*new_scale,inputs.faces),RR,tt,choices,rsel,tfinal,output)
        inputs.receipt.check()
        if cloud_digest(inputs.clouds)!=cloud_hash or code_identity()!=code:raise RuntimeError('input or code mutated')
        if sha(baseline/'coarse_scene.npz')!=m['arrays']['sha256']:raise RuntimeError('baseline mutated')
        report={'source':str(source),'source_sha256':sha(source),'baseline':str(baseline),'baseline_archive_sha256':m['arrays']['sha256'],
          'experiment_archive_sha256':sha(output/'experiment.npz'),'frames':n,'modes':modes+['scale_refit'],'baseline_scale':scale,'refit_scale':new_scale,
          'two_dimensional_tracklet_records':len(track_rows),'tracklet_keys':keys,'new_model_inference':False,'GT_CAD_reads':False,
          'depth_cloud_hash':cloud_hash,'MANO_modified':False,'source_reads':inputs.receipt.files,'code_sha256':code,
          'seconds':time.monotonic()-start,'selection_only_ablation':True,'no_mano_does_not_undo_upstream_ownership':True,
          'holdout_scope':'new proposal fit excludes held-out tracklets; baseline pose/bindings reuse RGB and are NOT an independent validation set',
          'candidate_padding':'invalid slots repeat baseline for rendering only','default_producer_changed':False}
        write_json(output/'completed.json',report);print(json.dumps({'output':str(output),'seconds':report['seconds'],'scale':new_scale}),flush=True)
    except Exception as ex:
        write_json(output/'failed.json',{'error':str(ex),'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.baseline,a.output)
