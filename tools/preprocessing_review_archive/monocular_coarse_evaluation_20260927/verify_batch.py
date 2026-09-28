"""Read-only completed-batch/geometry/video verifier, separate from predictions."""
from pathlib import Path
import argparse,json,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def verify(run,snapshot,worktree=None):
 if (run/'batch.exit').read_text().strip()!='0':raise ValueError('incomplete prediction batch')
 rows=[]
 for item in read(run/'inputs.json'):
  out=Path(item['output']);m=read(out/'coarse_scene.json');done=read(out/'completed.json');source=Path(item['source']);bundle=read(source)
  assert sha(source)==m['source_bundle_sha256'];assert sha(out/'coarse_scene.npz')==m['arrays']['sha256']
  for rel,h in m['code_sha256'].items():
   assert sha(snapshot/'preprocessing'/rel)==h
   if worktree:assert sha(worktree/'preprocessing'/rel)==h
  with np.load(out/'coarse_scene.npz',allow_pickle=False) as z:
   assert len(z['frame_idx'])==150 and np.array_equal(z['frame_idx'],m['frame_ids'])
   assert np.isfinite(z['object_translation_world']).all()
   assert np.allclose(z['object_rotation']@z['object_rotation'].transpose(0,2,1),np.eye(3),atol=1e-6)
   if m.get('alignment_profile')=='visible-surface':
    s=float(z['uniform_object_scale']);assert np.isfinite(s) and s>0
    np.testing.assert_allclose(z['shared_shape_matrix'],np.eye(3)*s,rtol=0,atol=1e-10)
    np.testing.assert_allclose(z['vertices_canonical'],z['source_vertices_canonical']*s,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(z['relative_depth_scale'],np.ones(len(z['frame_idx'])))
    np.testing.assert_array_equal(z['relative_depth_bias_m'],np.zeros(len(z['frame_idx'])))
    assert m['depth_correction_enabled'] is False and m['MoGe_modified'] is False
    if 'alignment_initial_rotation' in z:
     np.testing.assert_array_equal(z['object_rotation'],z['alignment_initial_rotation'])
     assert m['rotation_changed_by_registration'] is False
    measurement=read(out/'visible_surface_metrics.json')
    assert measurement['source_point_digest_before']==measurement['source_point_digest_after']==m['source_cloud_digest']
    depth_record=bundle['sources']['conditioned_depth'];assert sha(source.parent/depth_record['path'])==depth_record['sha256']
    ann=read(source.parent/bundle['sources']['annotations']['path']);h=hashlib.sha256()
    for frame in ann['frames']:
     obj=next((v for v in frame.get('objects',[]) if v.get('object_id')==m['object_id']),{});g=obj.get('visible_geometry_candidate') or {}
     q=np.asarray(g.get('camera_vertices_sample_m',[]),float).reshape(-1,3)
     eligible=obj.get('rigid_pose_observation_eligible') is True and g.get('rigid_pose_observation_eligible') is True and len(q)>=4
     q=np.ascontiguousarray(q if eligible else np.empty((0,3)));h.update(str(q.shape).encode());h.update(q.dtype.str.encode());h.update(q.tobytes())
    assert h.hexdigest()==m['source_cloud_digest']
    import trimesh
    native=trimesh.load(source.parent/bundle['sources']['visual_mesh_camera']['path'],process=False)
    np.testing.assert_array_equal(z['faces'],np.asarray(native.faces))
   else:
    assert np.isclose(np.linalg.det(z['shared_shape_matrix']),1.,atol=1e-6)
   with np.load(source.parent/bundle['sources']['mano_archive']['path'],allow_pickle=False) as hands:
    for side in ('left','right'):
     assert np.array_equal(z[side+'_vertices'],hands[side+'_vertices_world_m'],equal_nan=True)
     assert np.array_equal(z[side+'_joints'],hands[side+'_joints_world_m'],equal_nan=True)
     assert np.array_equal(z[side+'_faces'],hands[side+'_faces'])
  forbidden=[x for x in m['source_reads'] if '/hot3d_gt/' in x or '/depth_gt/' in x or 'pose_audit' in x]
  assert not forbidden
  videos=[]
  for v in [out/'renders/before_after_overlay.mp4',out/'renders/coarse_world.mp4']:
   result=subprocess.run(['ffmpeg','-hide_banner','-v','error','-threads','1','-i',str(v),'-map','0:v:0','-an','-threads','1','-f','null','-','-progress','pipe:1'],capture_output=True,text=True)
   progress={}
   for line in result.stdout.splitlines():
    if '=' in line:k,value=line.split('=',1);progress[k]=value
   assert result.returncode==0 and int(progress.get('frame',0))==150
   videos.append({'path':str(v),'sha256':sha(v),'frames':150,'time_us':int(progress['out_time_us'])})
  rows.append({'case':item['case'],'source_unchanged':True,'MANO_unchanged':True,'snapshot_verified':True,'seconds':done['seconds'],'videos':videos})
 report={'run':str(run),'snapshot':str(snapshot),'worktree_checked':str(worktree) if worktree else None,'cases':rows,'video_frames':sum(v['frames'] for r in rows for v in r['videos']),'scope':'integrity and execution, NOT precision acceptance'}
 dest=ROOT/run.name;dest.mkdir(exist_ok=True);(dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'cases':len(rows),'decoded_frames':report['video_frames'],'output':str(dest/'verification.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('snapshot',type=Path);p.add_argument('--worktree',type=Path);a=p.parse_args();verify(a.run,a.snapshot,a.worktree)
