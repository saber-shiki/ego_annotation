"""External post-prediction evaluation only. Never writes into source/candidate runs."""
from pathlib import Path
import sys,json,argparse,csv,hashlib,gc
import numpy as np
AUDIT=Path('/mnt/user-home/kupingxin/ego_annotation_experiments/hot3d_s3_pose_audit_20260927')
sys.path.insert(0,str(AUDIT))
from evaluate import load_case,read,sha,pose,apply,sample,scene,distance,gauge_errors,summ
ROOT=Path(__file__).resolve().parent


def evaluate(root):
    if (root/'batch.exit').read_text().strip()!='0':raise ValueError('wait for frozen completed prediction batch')
    out=ROOT/root.name;out.mkdir(exist_ok=True);results=[]
    for item in read(root/'inputs.json'):
        case=item['case'];pred=Path(item['output']);manifest=read(pred/'coarse_scene.json');done=read(pred/'completed.json')
        if done['status']!='complete_derived_coarse_scene' or sha(pred/'coarse_scene.npz')!=manifest['arrays']['sha256']:raise ValueError('candidate not frozen')
        if not manifest['prediction_side_only'] or manifest['GT_or_CAD_inputs']:raise ValueError('wrong prediction protocol')
        c=load_case(case);old=read(AUDIT/case/'evaluation.json')
        if manifest['source_bundle_sha256']!=old['input_bundle_sha256']:raise ValueError('source run mismatch')
        z=np.load(pred/'coarse_scene.npz',allow_pickle=False);ids=list(map(int,z['frame_idx']))
        if ids!=c['ids']:raise ValueError('candidate timeline mismatch')
        P={i:pose(R,t) for i,R,t in zip(ids,z['object_rotation'],z['object_translation_world'])}
        errors=gauge_errors(P,c['C'],c['G'],c['D'],c['m'])
        initial_errors=None
        if 'alignment_initial_rotation' in z:
            initial_P={i:pose(R,t) for i,R,t in zip(ids,z['alignment_initial_rotation'],z['alignment_initial_translation_world'])}
            initial_errors=gauge_errors(initial_P,c['C'],c['G'],c['D'],c['m'])
        vertices=z['vertices_canonical'];faces=z['faces'];n=sample(vertices,faces);g=sample(c['gv'],c['gf']);ns=scene(vertices,faces);gs=scene(c['gv'],c['gf'])
        base={r['frame']:r for r in old['frame_rows']};rows=[]
        for i in ids:
            row={'frame':i,'GT_quality_eligible':c['eligible'][i]};row.update(errors[i])
            H=np.linalg.inv(c['C'][i])@P[i];G=np.linalg.inv(c['D'][i])@c['G'][i]
            dn=distance(gs,apply(np.linalg.inv(G)@H,n))*1000;dg=distance(ns,apply(np.linalg.inv(H)@G,g))*1000
            d=np.r_[dn,dg];row['raw_surface_mean_mm']=float(d.mean());row['raw_surface_p95_mm']=float(np.percentile(d,95));row['surface_within_5mm_fraction']=float(np.mean(d<=5))
            row['relative_depth_bias_m']=float(z['relative_depth_bias_m'][ids.index(i)])
            if i in base:
                for k in ['source_aligned_camera_rotation_deg','source_aligned_camera_translation_mm','seed_aligned_camera_rotation_deg','seed_aligned_camera_translation_mm','source_raw_surface_symmetric_mean_mm','seed_raw_surface_symmetric_mean_mm']:
                    if k in base[i]:row[k]=base[i][k]
            rows.append(row)
        valid=[r for r in rows if r['GT_quality_eligible']];shared=[r for r in valid if 'source_aligned_camera_rotation_deg' in r and 'seed_aligned_camera_rotation_deg' in r]
        image=read(pred/'image_diagnostics.json')
        result={'case':case,'candidate_dir':str(pred),'candidate_sha256':manifest['arrays']['sha256'],'input_bundle_sha256':manifest['source_bundle_sha256'],'GT_used_only_after_prediction':True,'frames':len(rows),'GT_quality_frames':len(valid),'common_frames':len(shared),
            'camera_rotation_deg':summ([r['camera_rotation_deg'] for r in shared]),'camera_translation_mm':summ([r['camera_translation_mm'] for r in shared]),'raw_surface_mean_mm':summ([r['raw_surface_mean_mm'] for r in shared]),
            'old_common':old['same_common_frame_comparison']['metrics'],'all_QA_camera_rotation_deg':summ([r['camera_rotation_deg'] for r in valid]),'all_QA_camera_translation_mm':summ([r['camera_translation_mm'] for r in valid]),
            'all_QA_raw_surface_mean_mm':summ([r['raw_surface_mean_mm'] for r in valid]),
            'visible_IoU':{k:summ([x['visible_IoU'] for x in image[k] if 'visible_IoU' in x]) for k in ['source','coarse']},'shared_log_stretch':z['shared_log_stretch'].tolist(),'runtime_seconds':done['seconds'],
            'metric_scope':'One fixed common GT anchor alignment for trajectory; not absolute initial pose and not symmetry-adjusted. Raw surface distance has no GT alignment, includes shape+scale+pose.','rows':rows}
        if initial_errors is not None:
            common_ids=[r['frame'] for r in shared]
            result['alignment_input_pose']={'camera_rotation_deg':summ([initial_errors[i]['camera_rotation_deg'] for i in common_ids]),'camera_translation_mm':summ([initial_errors[i]['camera_translation_mm'] for i in common_ids])}
            result['rotation_change_by_alignment_max_deg']=float(max(abs(errors[i]['camera_rotation_deg']-initial_errors[i]['camera_rotation_deg']) for i in ids))
        result['alignment_profile']=manifest.get('alignment_profile','legacy')
        result['uniform_object_scale']=manifest.get('residual_uniform_object_scale')
        (out/(case+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with (out/(case+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
        compact={k:v for k,v in result.items() if k!='rows'};results.append(compact);print(json.dumps(compact),flush=True)
        del c,z,ns,gs;gc.collect()
    (out/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_root',type=Path);a=p.parse_args();evaluate(a.run_root)
