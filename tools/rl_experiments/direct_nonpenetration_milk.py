"""One bounded full-Milk direct geometry solve; no PPO or video alignment."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from data.coarse_training import build_index,CoarseTrainingDataset
from data.bundle import file_sha256
from envs.coarse_annotation_env import CoarseAnnotationEnv
from envs.dynamics import decode_candidate
from solvers.direct_nonpenetration import DirectNonpenetrationSolver
from utils.complete_triangle_collision import CompleteTriangleScene
from tools.rl_experiments.preprocessed_learning import source_hashes,render_comparison


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def raw_env(run,cfg):
    ds=CoarseTrainingDataset(run/'index.json',split='all');env=CoarseAnnotationEnv(ds,cfg['environment']);env.reset(seed=0)
    return env,ds


def export(env,out,label):
    out.mkdir(parents=True,exist_ok=False);arrays=env.export_annotation()['arrays']
    for side in ('left','right'):
        arrays[side+'_root_orient']=env.state.hand_root[side];arrays[side+'_articulation']=env.state.hand_articulation[side]
        arrays[side+'_translation_world']=env.state.hand_translation[side];arrays[side+'_betas']=env.state.episode.hands[side]['betas']
    arrays['contact_logits']=env.state.contact_logits
    npz=out/'candidate.npz';np.savez_compressed(npz,**arrays)
    manifest=out/'candidate.json'
    write(manifest,{'schema':'ego.annotation.direct_geometric_sequence.v1','method':label,'frame_count':len(arrays['frame_idx']),
        'input_identity':dict(env.state.episode.identity),'source_signed_authority_promoted':False,
        'reference_is_GT':False,'video_alignment_used':False,'policy_checkpoint_used':False,
        'episodes':[{'artifacts':[{'format':'npz','path':npz.name,'sha256':file_sha256(npz)}]}]})
    write(out/'evaluation.json',{'manifest':str(manifest),'npz':str(npz),'objective':None,'method':label})
    return {'evaluation':str(out/'evaluation.json'),'npz':str(npz),'steps':0,'objective':0.}


def summary(audit):
    return {k:audit[k] for k in ('intersection_pair_count','max_inside_depth_m','inside_squared_sum_m2','merit','feasible')}


def preflight(args,cfg):
    args.run.mkdir(parents=True,exist_ok=False);write(args.run/'index.json',build_index([args.coarse]))
    write(args.run/'source_identity.json',source_hashes());write(args.run/'run_config.json',{'config':cfg,'runtime':str(ROOT),
       'config_sha256':file_sha256(args.config),'source_sha256':file_sha256(args.coarse),'host':__import__('os').uname().nodename,
       'device':'single_thread_CPU_no_CUDA','full_sequence_attempts':1,'frame_budget':150,'PPO_steps':0})
    env,_=raw_env(args.run,cfg);episode=env.state.episode;solver=DirectNonpenetrationSolver(episode,{**cfg['solver'],'object_translation_frame':cfg['environment']['object_translation_frame']})
    rng=np.random.default_rng(902);q=rng.normal(size=108)*.015
    for i in range(150):env.state.position=i;decode_candidate(env.state,np.zeros(118),env.cfg)
    env.state.position=150;env.elapsed_steps=150;export(env,args.run/'zero','zero_R6_real_MANO_replay')
    env.state.position=120;decode_candidate(env.state,solver.geometry.physical(q),env.cfg)
    g=solver.geometry.forward(120,q,jacobian=True)
    errors={s:float(np.linalg.norm(g['hands'][s]['vertices']-env.state.hand_vertices[s][120],axis=-1).max()) for s in ('left','right')}
    if max(errors.values())>2e-6:raise RuntimeError('batch MANO does not reproduce existing decoder '+str(errors))
    np.testing.assert_allclose(g['rotation'],env.state.object_rotation['object:0'][120],atol=2e-8)
    directions={}
    for side in ('left','right'):
        offset=episode.layout.block(f'hand.{side}.root').start;direction=rng.normal(size=51)*.001
        test=q.copy();test[offset:offset+51]+=direction;v=solver.geometry.forward(120,test)['hands'][side]['vertices']
        predicted=g['hands'][side]['vertices']+np.einsum('vck,k->vc',g['hands'][side]['jacobian'],direction)
        directions[side]=float(np.linalg.norm(v-predicted,axis=-1).max())
        if directions[side]>2e-6:raise RuntimeError('parameter Jacobian direction mismatch')
    rows=[]
    for i in (0,120):
        fresh=DirectNonpenetrationSolver(episode,{**cfg['solver'],'object_translation_frame':cfg['environment']['object_translation_frame']})
        start=time.monotonic();q,g,a,r=fresh.solve_frame(i);r.update(seconds=time.monotonic()-start)
        rows.append(r);print('PREFLIGHT FRAME '+json.dumps({k:r[k] for k in ['frame_idx','seconds','iterations','initial_triangle_pairs','final_triangle_pairs','initial_max_depth_mm','final_max_depth_mm','feasible']}),flush=True)
    write(args.run/'mechanism_preflight.json',{'MANO_decoder_max_error_m':errors,'MANO_Jacobian_direction_error_m':directions,
          'both_hands_and_object_available':True,'full_parameter_layout':118,'PPO_steps':0,'frames':rows})
    if source_hashes()!=json.loads((args.run/'source_identity.json').read_text()):raise RuntimeError('snapshot mutated')
    write(args.run/'preflight_completed.json',{'mechanism_checks_passed':True,'geometric_feasibility_of_probe_frames':[r['feasible'] for r in rows]})
    env.close()


def solve(args,cfg):
    if not (args.run/'preflight_completed.json').exists():raise RuntimeError('mechanism preflight required')
    env,_=raw_env(args.run,cfg);solver=DirectNonpenetrationSolver(env.state.episode,{**cfg['solver'],'object_translation_frame':cfg['environment']['object_translation_frame']})
    start=time.monotonic();trace=[];qall=[]
    for i in range(150):
        q,g,a,r=solver.solve_frame(i);env.state.position=i;decode_candidate(env.state,solver.geometry.physical(q),env.cfg)
        for side in ('left','right'):
            np.testing.assert_allclose(env.state.hand_vertices[side][i],g['hands'][side]['vertices'],atol=2e-6)
        r['cumulative_seconds']=time.monotonic()-start;trace.append(r);qall.append(q)
        write(args.run/'partial_trace.json',trace)
        print('FRAME '+json.dumps({k:r[k] for k in ['frame_idx','iterations','initial_triangle_pairs','final_triangle_pairs','final_max_depth_mm','feasible','cumulative_seconds']}),flush=True)
    env.state.position=150;env.elapsed_steps=150;candidate=export(env,args.run/'solved','direct_parameter_nonpenetration')
    np.save(args.run/'solved/normalized_geometric_parameters.npy',np.array(qall))
    write(args.run/'solve_trace.json',trace)
    write(args.run/'solve_completed.json',{'frames':150,'feasible_frames':sum(r['feasible'] for r in trace),
        'unresolved_frame_ids':[r['frame_idx'] for r in trace if not r['feasible']],
        'seconds':time.monotonic()-start,'candidate':candidate,'PPO_steps':0,'video_alignment_used':False})
    if source_hashes()!=json.loads((args.run/'source_identity.json').read_text()):raise RuntimeError('snapshot mutated')
    env.close()


def audit(args,cfg):
    reports={}
    for label in ['zero','solved']:
        path=args.run/label/'candidate.npz'
        with np.load(path,allow_pickle=False) as z:a=dict(z)
        scene=CompleteTriangleScene(a['object_vertices_canonical'],a['object_faces']);frames=[]
        for i,fid in enumerate(a['frame_idx']):
            hands={}
            for side in ('left','right'):
                faces=a[side+'_faces'];faces=faces[i] if faces.ndim==3 else faces
                row=scene.inspect_hand(a[side+'_vertices_world'][i],faces,a['object_rotation'][i],a['object_translation_world'][i],nsamples=11)
                pair=row['intersections'];near=row['near']
                hands[side]={'max_inside_depth_mm':row['max_inside_depth_m']*1000,'inside_squared_sum_m2':row['inside_squared_sum_m2'],
                    'inside_count_over_10um':row['inside_count_over_10um'],'triangle_intersection_pair_count':row['intersection_pair_count'],
                    'intersected_hand_face_ids':np.unique(pair['hand_face_ids']).tolist(),'intersected_object_face_ids':np.unique(pair['object_face_ids']).tolist(),
                    'inside_vertex_ids':np.flatnonzero(near['signed_distance_m'][:778]<-1e-5).tolist(),
                    'nearest_surface_gap_mm':row['nearest_surface_gap_m']*1000}
            feasible=all(h['triangle_intersection_pair_count']==0 and h['max_inside_depth_mm']<=.01 for h in hands.values())
            frames.append({'frame_idx':int(fid),'feasible':feasible,'hands':hands})
        reports[label]={'frame_count':150,'topology':scene.topology,'npz_sha256':file_sha256(path),'nsamples':11,
            'triangle_tolerance_m':1e-8,'inside_tolerance_m':1e-5,'frames':frames,
            'feasible_frame_count':sum(f['feasible'] for f in frames),'unresolved_frame_ids':[f['frame_idx'] for f in frames if not f['feasible']],
            'triangle_pair_count':sum(h['triangle_intersection_pair_count'] for f in frames for h in f['hands'].values()),
            'max_inside_depth_mm':max(h['max_inside_depth_mm'] for f in frames for h in f['hands'].values())}
        write(args.run/label/'complete_triangle_audit.json',reports[label]);print('AUDIT '+label+' '+json.dumps({k:v for k,v in reports[label].items() if k!='frames'}),flush=True)
    write(args.run/'audit_completed.json',{'scope':'all_triangle_intersections_and_generated_inside_not_GT','reports':{k:{x:v[x] for x in ['feasible_frame_count','triangle_pair_count','max_inside_depth_mm']} for k,v in reports.items()}})


def render(args,cfg):
    ds=CoarseTrainingDataset(args.run/'index.json',split='all');zero=json.loads((args.run/'zero/evaluation.json').read_text())
    solved={'evaluation':str(args.run/'solved/evaluation.json'),'steps':0,'objective':0.}
    out=args.run/'comparison';out.mkdir(exist_ok=False)
    render_comparison(out,zero,solved,None,ds[0],panel_labels=('Zero R6','direct solve'))
    write(args.run/'render_completed.json',{'frames':150,'fps':30,'physical_geometry':'complete_native_and_metric_MANO',
                  'panel_labels':['Zero R6','direct solve'],'two_distinct_geometry_panels':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True,choices=['preflight','solve','audit','render'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/direct_nonpenetration_milk.yaml');p.add_argument('--run',type=Path,required=True);p.add_argument('--coarse',type=Path)
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text());__import__('torch').set_num_threads(1);__import__('torch').set_num_interop_threads(1)
    {'preflight':preflight,'solve':solve,'audit':audit,'render':render}[args.mode](args,cfg)
