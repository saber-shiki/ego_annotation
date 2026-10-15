"""Read-only posthoc geometry diagnostics on frozen training exports (no GT)."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation


def inspect(run,output):
    zero=json.loads((run/'zero/evaluation.json').read_text())
    with np.load(zero['npz'],allow_pickle=False) as z:base=dict(z)
    rows=[]
    for p in sorted(run.glob('checkpoints/step_*/evaluation/evaluation.json')):
        evaluation=json.loads(p.read_text());manifest_path=Path(evaluation['manifest']);manifest=json.loads(manifest_path.read_text())
        paths=[a['path'] for a in manifest['episodes'][0]['artifacts'] if a['format']=='npz']
        path=Path(paths[0]);path=path if path.is_absolute() else manifest_path.parent/path
        with np.load(path,allow_pickle=False) as z:arrays=dict(z)
        measurements={}
        for side in ('left','right'):
            valid=base[side+'_valid'].astype(bool)
            C=base['T_world_camera'];K=base['intrinsics_render']
            def projection(joints):
                cam=np.einsum('tni,tij->tnj',joints-C[:,None,:3,3],C[:,:3,:3])
                z=np.maximum(cam[:,:,2],1e-6)
                uv=cam[:,:,:2]/z[:,:,None]*K[:,None,:2]+K[:,None,2:]
                return uv,cam[:,:,2]
            buv,bz=projection(base[side+'_joints_world']);uv,z=projection(arrays[side+'_joints_world'])
            pix=np.linalg.norm(uv-buv,axis=-1);vertices=np.linalg.norm(arrays[side+'_vertices_world']-base[side+'_vertices_world'],axis=-1)
            worst=np.argsort(np.max(pix,axis=1))[::-1][:8]
            measurements[side]={'vertex_correction_mean_mm':float(vertices[valid].mean()*1000),
                'vertex_correction_max_mm':float(vertices[valid].max()*1000),
                'joint_pixel_shift_median_px':float(np.median(pix[valid])),
                'joint_pixel_shift_p95_px':float(np.percentile(pix[valid],95)),
                'joint_pixel_shift_max_px':float(pix[valid].max()),
                'joint_pixel_shift_over_12px_count':int(np.count_nonzero(pix[valid]>12)),
                'baseline_nonpositive_joint_depth_count':int(np.count_nonzero(bz[valid]<=0)),
                'candidate_nonpositive_joint_depth_count':int(np.count_nonzero(z[valid]<=0)),
                'worst_frames':[{'frame_idx':int(base['frame_idx'][i]),'max_pixel_shift':float(pix[i].max()),'minimum_joint_camera_z_m':float(z[i].min())} for i in worst]}
        angle=Rotation.from_matrix(arrays['object_rotation']@base['object_rotation'].transpose(0,2,1)).magnitude()
        dt=np.linalg.norm(arrays['object_translation_world']-base['object_translation_world'],axis=1)
        named=evaluation['named_families'];obj=float(evaluation['objective'])
        rows.append({'checkpoint':p.parts[-3],'objective':obj,'objective_delta_vs_zero':obj-zero['objective'],
            'named_term_delta_vs_zero':{k:v-zero['named_families'].get(k,0) for k,v in named.items()},
            'joint_pixel_objective_fraction':named.get('hand/joint_pixel_deadband',0)/max(obj,1e-12),
            'object_translation_correction_max_mm':float(dt.max()*1000),
            'object_rotation_correction_max_deg':float(np.degrees(angle.max())), 'hands':measurements})
    output.mkdir(parents=True,exist_ok=False)
    (output/'geometry_diagnostics.json').write_text(json.dumps({'scope':'posthoc_no_GT_not_generalization',
        'zero_objective':zero['objective'],'available_frozen_checkpoints':rows},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'checkpoints':len(rows),'objectives':[(r['checkpoint'],r['objective']) for r in rows]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();inspect(a.run,a.output)
