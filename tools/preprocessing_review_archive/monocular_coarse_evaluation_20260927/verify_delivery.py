"""Validate completed candidates without modifying prediction runs."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent
RUN=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_coarse_20260927_r2')
SNAP=Path('/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_coarse_20260927_r2')
CODE=Path('/mnt/user-home/kupingxin/ego_annotation_preprocess_monocular')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
if (RUN/'batch.exit').read_text().strip()!='0':raise ValueError('incomplete batch')
results=[]
for item in read(RUN/'inputs.json'):
 out=Path(item['output']);m=read(out/'coarse_scene.json');done=read(out/'completed.json');source=Path(item['source']);bundle=read(source)
 assert sha(source)==m['source_bundle_sha256']
 assert sha(out/'coarse_scene.npz')==m['arrays']['sha256']
 for rel,h in m['code_sha256'].items():
  assert sha(SNAP/'preprocessing'/rel)==h
  assert sha(CODE/'preprocessing'/rel)==h
 z=np.load(out/'coarse_scene.npz',allow_pickle=False)
 assert np.array_equal(z['frame_idx'],m['frame_ids']) and len(z['frame_idx'])==150
 assert np.isfinite(z['object_translation_world']).all()
 assert np.allclose(z['object_rotation']@z['object_rotation'].transpose(0,2,1),np.eye(3),atol=1e-6)
 assert np.isclose(np.linalg.det(z['shared_shape_matrix']),1.,atol=1e-6)
 mano=source.parent/bundle['sources']['mano_archive']['path']
 with np.load(mano,allow_pickle=False) as h:
  for side in ['left','right']:
   assert np.array_equal(z[side+'_vertices'],h[side+'_vertices_world_m'])
   assert np.array_equal(z[side+'_joints'],h[side+'_joints_world_m'])
   assert np.array_equal(z[side+'_faces'],h[side+'_faces'])
 videos=[]
 for v in [out/'renders/before_after_overlay.mp4',out/'renders/coarse_world.mp4']:
  proc=subprocess.run(['ffmpeg','-hide_banner','-v','error','-threads','1','-i',str(v),'-map','0:v:0','-an','-threads','1','-f','null','-','-progress','pipe:1'],capture_output=True,text=True)
  values={}
  for s in proc.stdout.splitlines():
   if '=' in s:k,val=s.split('=',1);values[k]=val
  assert proc.returncode==0 and int(values.get('frame',0))==150
  videos.append({'path':str(v),'sha256':sha(v),'decoded_frames':150,'time_us':int(values['out_time_us'])})
 results.append({'case':item['case'],'source_unchanged':True,'snapshot_matches_worktree':True,'MANO_vertices_joints_faces_unchanged':True,'frames':150,'seconds':done['seconds'],'videos':videos})
report={'run_root':str(RUN),'case_count':len(results),'decoded_video_frames':sum(v['decoded_frames'] for r in results for v in r['videos']),'cases':results,'physical_accuracy_passed':False,'acceptance_scope':'execution, input/code/geometry identity and full videos only; see external GT errors'}
(ROOT/'delivery_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(results),'video_frames':report['decoded_video_frames'],'source_and_MANO_unchanged':True,'precision_qualified':False}))
