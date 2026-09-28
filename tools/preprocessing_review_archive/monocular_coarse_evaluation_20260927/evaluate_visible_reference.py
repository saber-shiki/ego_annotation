"""Independent fixed-UniDepth reference checks. This script does not read CAD/GT."""
from pathlib import Path
import json,sys,argparse
import numpy as np
sys.path.insert(0,'/mnt/user-home/kupingxin/ego_annotation_preprocess_monocular')
from preprocessing.geometry import TriangleMesh,camera_pose,project
from preprocessing.visible_surface import stats
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())


def evaluate(run):
 if (run/'batch.exit').read_text().strip()!='0':raise ValueError('prediction not frozen')
 results=[]
 for item in read(run/'inputs.json'):
  out=Path(item['output']);m=read(out/'coarse_scene.json');source=Path(item['source']);bundle=read(source);ann=read(source.parent/bundle['sources']['annotations']['path'])
  with np.load(out/'coarse_scene.npz',allow_pickle=False) as z:
   fid=m['appearance_anchor_frame_idx'];p=int(np.flatnonzero(z['frame_idx']==fid)[0]);frame=next(f for f in ann['frames'] if f['frame_idx']==fid)
   obj=next(o for o in frame['objects'] if o['object_id']==m['object_id']);Q=np.asarray(obj['visible_geometry_candidate']['camera_vertices_sample_m'],float)
   mesh=TriangleMesh(z['source_vertices_canonical'],z['faces']);K=z['intrinsics_source'][p];uv=project(Q,K);C=z['T_world_camera'][p];s=float(z['uniform_object_scale'])
   rk='alignment_initial_rotation' if 'alignment_initial_rotation' in z else 'source_rotation'
   tk='alignment_initial_translation_world' if 'alignment_initial_translation_world' in z else 'source_translation_world'
   R0,t0=camera_pose(z[rk][p],z[tk][p],C);R1,t1=camera_pose(z['object_rotation'][p],z['object_translation_world'][p],C)
   np.testing.assert_allclose(R0,R1,rtol=0,atol=1e-10)
   x0,d0,h0=mesh.query(uv,K,R0,t0);x1,d1,h1=mesh.query(uv,K,R1,t1/s)
   P0=x0@R0.T+t0;P1=(s*x1)@R1.T+t1;common=h0&h1
   before=stats(np.linalg.norm(P0[common]-Q[common],axis=1)*1000);after=stats(np.linalg.norm(P1[common]-Q[common],axis=1)*1000)
   before_depth=stats(np.abs(P0[common,2]-Q[common,2])*1000);after_depth=stats(np.abs(P1[common,2]-Q[common,2])*1000)
   ext={}
   if common.sum()>1:
    ext={'observed_camera_span_mm':(np.ptp(Q[common],axis=0)*1000).tolist(),'before_same_rays_span_mm':(np.ptp(P0[common],axis=0)*1000).tolist(),'after_same_rays_span_mm':(np.ptp(P1[common],axis=0)*1000).tolist(),'definition':'same observed rays, not full-object intrinsic dimensions'}
   surface=read(out/'visible_surface_metrics.json');brows={f['frame_idx']:f for f in surface['before']['frames']};arows={f['frame_idx']:f for f in surface['after']['frames']}
   eligible=[i for i in brows if brows[i]['metric_samples']>0 and arows[i]['metric_samples']>0]
   result={'case':item['case'],'reference':'raw_Unidepth_visible_surface_not_GT_depth','residual_uniform_scale':s,'anchor_frame_idx':fid,'anchor_input_samples':len(Q),
    'anchor_coverage_before':float(h0.mean()),'anchor_coverage_after':float(h1.mean()),'anchor_common_first_hit_samples':int(common.sum()),
    'anchor_common_ray_3D_before_mm':before,'anchor_common_ray_3D_after_mm':after,'anchor_common_ray_depth_before_mm':before_depth,'anchor_common_ray_depth_after_mm':after_depth,'visible_spans':ext,
    'anchor_final_metrics_before':brows[fid],'anchor_final_metrics_after':arows[fid],
    'timeline_frame_median_surface_before_mm':stats([brows[i]['visible_surface_distance_m']['median']*1000 for i in eligible]),
    'timeline_frame_median_surface_after_mm':stats([arows[i]['visible_surface_distance_m']['median']*1000 for i in eligible]),
    'timeline_frame_coverage_before':stats([brows[i]['first_hit_coverage'] for i in eligible]),'timeline_frame_coverage_after':stats([arows[i]['first_hit_coverage'] for i in eligible]),
    'source_cloud_hash_unchanged':surface['source_point_digest_before']==surface['source_point_digest_after'],
    'scope':'anchor R unchanged, evaluates actual delivered shared scale+t on identical common-hit rays; coverage reported separately. Timeline includes tracking and translation fitting.'}
  results.append(result);print(json.dumps({k:result[k] for k in ['case','residual_uniform_scale','anchor_coverage_before','anchor_coverage_after','anchor_common_ray_3D_before_mm','anchor_common_ray_3D_after_mm','timeline_frame_median_surface_before_mm','timeline_frame_median_surface_after_mm']}),flush=True)
 dest=ROOT/run.name;dest.mkdir(exist_ok=True);(dest/'fixed_visible_reference.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run_root',type=Path);a=p.parse_args();evaluate(a.run_root)
