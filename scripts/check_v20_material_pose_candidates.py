#!/usr/bin/env python3
"""Frozen same-ray full mesh and fixed-material comparison. No optimizer/GT."""
from pathlib import Path
import argparse,json
import numpy as np
import open3d as o3d
import trimesh
import fit_v20_late_window_prediction_only_se3 as late
import v20_rgbd_material_factors as material


def summary(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return {'count':int(v.size),'median':float(np.median(v)) if len(v) else None,
            'p95':float(np.percentile(v,95)) if len(v) else None,'mean':float(np.mean(v)) if len(v) else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--material-rgbd-npz',type=Path,required=True)
    p.add_argument('--initial-pose',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--candidate',action='append',required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--object-id',default='carton_milk');a=p.parse_args()
    if a.output_dir.exists():raise ValueError('refusing to overwrite comparison')
    ann=late.load_json(a.annotations);base=late.load_json(a.initial_pose)
    late.assert_prediction_only(base,label='comparison initial',object_id=a.object_id)
    nodes,_,_,_=late.build_pose_nodes(ann,base,base,a.object_id,110,149);nodes=[n for n in nodes if 110<=n.frame_idx<=149]
    factors,info=material.load_factors(a.material_rgbd_npz,nodes,annotation_sha256=late.sha256_file(a.annotations),object_id=a.object_id,max_points=0)
    initial_rows=late._timeline_rows(base);frames={int(f['frame_idx']):f for f in ann['frames']}
    mesh=trimesh.load(a.mesh,process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices.astype('float32')),o3d.core.Tensor(faces.astype('uint32')))
    reports={};arrays={}
    for arg in a.candidate:
        name,filename=arg.split('=',1);path=Path(filename);rep=late.load_json(path)
        late.assert_prediction_only(rep,label=name,object_id=a.object_id)
        rows=late._timeline_rows(rep)
        poses={idx:late.pose_from_values(row['rotation_world_from_completed_canonical_matrix'],row['translation_world_m']) for idx,row in rows.items()}
        metric=material.metrics(factors,poses);timeline=late.pose_metrics(nodes,poses,120)
        fr=[];errs=[];lateerrs=[]
        for idx in range(150):
            obj=late._object_for_frame(frames[idx],a.object_id);world,reason,_=late.p09_geometry_validation(obj,frames[idx])
            if world is None:fr.append({'frame':idx,'metric_available':False});continue
            T=late._frame_camera_transform(frames[idx]);pc=late.world_to_camera(world,T)
            rays=pc/pc[:,2,None];R,t=poses[idx];oc=(T[:3,3]-t)@R;directions=(rays@T[:3,:3].T)@R
            ray=np.column_stack((np.broadcast_to(oc,directions.shape),directions)).astype('float32')
            z=scene.cast_rays(o3d.core.Tensor(ray),nthreads=4)['t_hit'].numpy()
            valid=np.isfinite(z)&(z>0);e=(z-pc[:,2])[valid];errs.extend(e.tolist())
            if idx>=130:lateerrs.extend(e.tolist())
            fr.append({'frame':idx,'metric_available':True,'coverage':float(valid.mean()),
                       'signed_mm':summary(1000*e),'abs_mm':summary(1000*np.abs(e)),
                       'front_gt5mm_fraction':float(np.mean(e<-.005))})
            arrays[f'{name}_{idx}_z']=z;arrays[f'{name}_{idx}_good']=valid;arrays[f'p09_{idx}_z']=pc[:,2]
        preserved=all(rows[idx]==initial_rows[idx] for idx in initial_rows if idx<110 or idx>149)
        reports[name]={'path':str(path.resolve()),'sha256':late.sha256_file(path),'status':rep.get('status'),
                       'material':metric,'timeline':timeline,'per_frame':fr,'all_signed_mm':summary(np.array(errs)*1000),
                       'all_abs_mm':summary(np.abs(errs)*1000),'late_130_149_signed_mm':summary(np.array(lateerrs)*1000),
                       'late_130_149_abs_mm':summary(np.abs(lateerrs)*1000),'outside_window_rows_preserved_vs_initial':preserved}
    baseline=next(iter(reports));deltas={}
    for name in list(reports)[1:]:
        rows=[]
        for idx in range(150):
            bk=f'{baseline}_{idx}_z';ck=f'{name}_{idx}_z'
            if bk not in arrays or ck not in arrays:continue
            common=arrays[f'{baseline}_{idx}_good']&arrays[f'{name}_{idx}_good'];zobs=arrays[f'p09_{idx}_z']
            b=arrays[bk][common]-zobs[common];c=arrays[ck][common]-zobs[common]
            rows.append({'frame':idx,'common_count':int(common.sum()),'before_abs_mm':summary(np.abs(b)*1000),
                         'after_abs_mm':summary(np.abs(c)*1000),'coverage_delta':float(arrays[f'{name}_{idx}_good'].mean()-arrays[f'{baseline}_{idx}_good'].mean())})
        deltas[name]={'baseline':baseline,'per_frame_common_hit':rows}
    a.output_dir.mkdir(parents=True);np.savez_compressed(a.output_dir/'rays.npz',**arrays)
    report={'schema':'v20_frozen_material_pose_comparison_v1','annotation_ready':False,'diagnostic_only':True,'gt_consumed':False,
            'mesh_path':str(a.mesh.resolve()),'mesh_sha256':late.sha256_file(a.mesh),'mesh_faces':len(faces),
            'candidates':reports,'common_hit_comparisons':deltas}
    (a.output_dir/'comparison.json').write_text(json.dumps(late.json_safe(report),indent=2,allow_nan=False)+'\n')
    for name,rep in reports.items():
        m=rep['material'];print(name,'pixel',round(m['material_pixel_median_px'],3),'metric mm',round(1000*m['material_metric_median_m'],3),'lateabs',rep['late_130_149_abs_mm'])
        print('firsthit',[(r['frame'],round(r['signed_mm']['median'],2),round(r['coverage'],3)) for r in rep['per_frame'] if r['frame'] in [120,130,135,140,142,144,149] and r['metric_available']])


if __name__=='__main__':main()
