"""Read-only physical review of frozen three-reward outputs, not training inputs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from data.coarse_training import CoarseTrainingDataset
from data.bundle import file_sha256


def stats(values):
    a=np.asarray(values,float).reshape(-1)
    return {'count':int(a.size),'mean':float(a.mean()),'p50':float(np.median(a)),
            'p95':float(np.percentile(a,95)),'max':float(a.max())} if a.size else {'count':0}


def full_native_motion(a,base):
    # EXACT mean over every native vertex, not centroid motion.
    vertices=a['object_vertices_canonical'].astype(float);second=vertices.T@vertices/len(vertices);mean=vertices.mean(0)
    dt=np.diff(a['time_s'])
    out={}
    for name in ('actual','correction'):
        R=a['object_rotation'].transpose(0,2,1).astype(float);t=a['object_translation_world'].astype(float)
        if name=='correction':
            R-=base['object_rotation'].transpose(0,2,1);t-=base['object_translation_world']
        A=np.diff(R,axis=0)/dt[:,None,None];b=np.diff(t,axis=0)/dt[:,None]
        Aa=np.diff(A,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None,None];ba=np.diff(b,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
        def rms(matrices,translations):
            vals=[]
            for M,B in zip(matrices,translations):
                squared=np.trace(M.T@second@M)+2*(mean@M)@B+B@B
                vals.append(np.sqrt(max(0.,squared))*1000)
            return vals
        speed=rms(A,b);accel=rms(Aa,ba)
        # Independent explicit native-vertex cross-check of the largest-motion edge.
        i=int(np.argmax(speed));explicit=np.sqrt(np.square(vertices@A[i]+b[i]).sum(-1).mean())*1000
        np.testing.assert_allclose(speed[i],explicit,rtol=1e-10,atol=1e-9)
        out[name]={'per_edge_full_surface_RMS_speed_mm_s':stats(speed),'per_triple_full_surface_RMS_acceleration_mm_s2':stats(accel),
                   'explicit_full_native_speed_check_edge':i,'explicit_speed_mm_s':float(explicit)}
    return out


def inspect(run,output):
    source=CoarseTrainingDataset(run/'index.json',split='all')[0].load()
    z=json.loads((run/'zero/evaluation.json').read_text())
    with np.load(z['npz'],allow_pickle=False) as f:base=dict(f)
    rows=[('zero',z)]
    for s in (0,1):
        for step in (150,300,450,600):
            rows.append((f'seed{s}@{step}',json.loads((run/f'seed_{s}/checkpoints/step_{step:04d}/evaluation/evaluation.json').read_text())))
    result={}
    for label,d in rows:
        with np.load(d['npz'],allow_pickle=False) as f:a=dict(f)
        gaps={side:[] for side in ('left','right')};changes={side:[] for side in gaps};frame_rows=[]
        for i,fid in enumerate(a['frame_idx']):
            R=a['object_rotation'][i];t=a['object_translation_world'][i];R0=base['object_rotation'][i];t0=base['object_translation_world'][i]
            for (frame,side,entity),row in source['contact'].items():
                if frame!=int(fid):continue
                ids=np.asarray(row['vertex_ids'],int);anchors=np.asarray(row['targets_object'],float)
                if not len(ids):continue
                v=a[side+'_vertices_world'][i,ids];v0=base[side+'_vertices_world'][i,ids]
                current=np.linalg.norm(v-(anchors@R.T+t),axis=-1);original=np.linalg.norm(v0-(anchors@R0.T+t0),axis=-1)
                gaps[side].extend(current*1000);changes[side].extend((current-original)*1000)
                frame_rows.append({'frame_idx':int(fid),'side':side,'native_anchor_count':int(len(ids)),
                    'gap_max_mm':float(current.max()*1000),'increase_max_mm':float((current-original).max()*1000)})
        result[label]={'input_npz':d['npz'],'sha256':file_sha256(Path(d['npz'])),
            'qualified_source_patch_gap_mm':{side:stats(vals) for side,vals in gaps.items()},
            'qualified_source_patch_gap_change_mm':{side:{**stats(vals),'count_increased_over_2mm':int((np.array(vals)>2).sum())} for side,vals in changes.items()},
            'object_full_surface_motion':full_native_motion(a,base),'contact_frames':frame_rows,
            'subcosts':d['final_ledger']['statistics']['subcosts']}
    attribution=[]
    for s in (0,1):
        trace=json.loads((run/f'seed_{s}/training_trace.json').read_text())[:150]
        all_reward=np.array([r['reward'] for r in trace]);cost=np.array([[r['group_costs'][g] for g in ('reference_alignment','nonpenetration_contact','temporal_continuity')] for r in trace])
        deltas=np.diff(cost,axis=0,prepend=np.zeros((1,3)))
        # Reference/temporal Zero replay is numerical roundoff (~1e-9 total); keep physical centered component as exact residual.
        rew=np.c_[-deltas[:,0],all_reward+deltas[:,0]+deltas[:,2],-deltas[:,2]]
        np.testing.assert_allclose(rew.sum(1),all_reward,atol=1e-10)
        returns=np.cumsum(rew[::-1],axis=0)[::-1];total=returns.sum(1)
        attribution.append({'seed':s,'scope':'first_rollout_only; reward-to-go with initial zero critic gamma=lambda=1; geometry-family additive decomposition, not causal percentages',
            'groups':{g:{'reward_std':float(rew[:,i].std()),'reward_to_go_std':float(returns[:,i].std()),'return_sum':float(rew[:,i].sum())} for i,g in enumerate(('reference_alignment','nonpenetration_contact','temporal_continuity'))},
            'total_reward_std':float(all_reward.std()),'total_reward_to_go_std':float(total.std()),
            'first_rollout_zero_critic':True})
    output.write_text(json.dumps({'scope':'qualified_source_patches_not_true_contact; object_surface_motion_is_exact_all_native_vertices',
        'candidates':result,'first_update_reward_signal':attribution},indent=2)+'\n')
    for k,v in result.items():print(k,'gaps',v['qualified_source_patch_gap_change_mm'],'object actual speed',v['object_full_surface_motion']['actual']['per_edge_full_surface_RMS_speed_mm_s']['p95'])
    print('FIRST UPDATE SIGNAL',json.dumps(attribution,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.run,a.output)
