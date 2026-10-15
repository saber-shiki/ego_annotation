"""Independent post-freeze FK/mesh/video/RRD input correspondence checks."""
from pathlib import Path
import sys,json,subprocess,hashlib
import numpy as np,cv2
HOME=Path('/mnt/user-home/kupingxin');sys.path.insert(0,str(HOME/'ego_annotation_experiments/milk_R6_vs_CAD_GT_20261014'))
from evaluate_comparison import load,sha,GT,CAD,PRED,mesh,apply,read
import inspect
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
for name,value in [('bool',np.bool_),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
 if name not in np.__dict__:setattr(np,name,value)
from rerun.experimental import RrdReader
HERE=HOME/'ego_annotation_experiments/milk_preprocess_shape_depth_hands_20261015_r1'
ROOT=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/milk_shape_depth_hands_20261015_r1')
STAGES=['stage1_shape_r2','stage2_depth_only_r2','stage2_shape_depth_r2','stage3_hands_r2']

def vector(b,name,row,width):return b.column(name)[row].values.flatten().to_numpy().reshape(-1,width)
def main():
 import torch,smplx
 from smplx.lbs import batch_rodrigues
 torch.set_num_threads(1);base=load(PRED);final=load(ROOT/'stage3_hands_r2/coarse_scene.npz');gt=load(GT);cad=mesh(CAD);D=gt['T_world_from_camera'];oi=int(np.flatnonzero(gt['object_uids']=='125863066770940')[0]);G=gt['T_world_from_object'][:,oi];A=D[0]@np.linalg.inv(base['T_world_camera'][0]);checks={}
 for name in STAGES:
  root=ROOT/name;a=load(root/'coarse_scene.npz');m=read(root/'coarse_scene.json');assert sha(root/'coarse_scene.npz')==m['arrays']['sha256'];assert (root/'completed.json').exists();np.testing.assert_array_equal(a['faces'],base['faces']);np.testing.assert_array_equal(a['T_world_camera'],base['T_world_camera']);np.testing.assert_array_equal(a['intrinsics_source'],base['intrinsics_source']);np.testing.assert_array_equal(a['frame_idx'],np.arange(150));assert a['vertices_canonical'].shape==base['vertices_canonical'].shape
  np.testing.assert_allclose(a['vertices_canonical'],base['vertices_canonical']@a['shared_repair_matrix'].T,atol=1e-10)
  for s in ['left','right']:
   np.testing.assert_array_equal(a[s+'_faces'],base[s+'_faces'])
   if name!='stage3_hands_r2':np.testing.assert_array_equal(a[s+'_vertices'],base[s+'_vertices'])
  videos={}
  for file in ['before_after_overlay.mp4','before_after_world.mp4']:
   cap=cv2.VideoCapture(str(root/'renders'/file));count=0
   while True:
    ok,im=cap.read()
    if not ok:break
    count+=1
   fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert count==150 and abs(fps-30)<1e-7;videos[file]=dict(decoded_frames=count,fps=fps)
  checks[name]=dict(vertices=len(a['vertices_canonical']),faces=len(a['faces']),camera_K_unchanged=True,native_topology_unchanged=True,full_mesh_transform_replayed=True,videos=videos)
 for s in ['left','right']:
  model=smplx.MANOLayer(str(HOME/'ego_annotation_runtime/rgb_scene_standalone_candidate_9bed519_gpu5_20260910/third_party/WiLoR/mano_data'/f'MANO_{s.upper()}.pkl'),is_rhand=s=='right',use_pca=False).eval()
  if s=='left':model.shapedirs[:,0,:]*=-1
  tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
  with torch.no_grad():
   out=model(global_orient=batch_rodrigues(tensor(final[s+'_root_orient_axis_angle'])).reshape(150,1,3,3),hand_pose=batch_rodrigues(tensor(final[s+'_hand_pose_axis_angle']).reshape(-1,3)).reshape(150,15,3,3),betas=tensor(final[s+'_betas']),transl=tensor(final[s+'_trans_world_m']))
  err=float(np.max(np.abs(out.vertices.numpy()-final[s+'_vertices'])));assert err<2e-6;assert np.max(np.ptp(final[s+'_betas'],axis=0))==0
  checks['stage3_hands_r2'][s+'_FK_export_replay_max_error_m']=err;checks['stage3_hands_r2'][s+'_shape_constant']=True
 file=HERE/'final_review/baseline_repair_GT.rrd';reader=RrdReader(file);assert len(reader.recordings())==1 and len(reader.blueprints())==1;counts={};maxpose=0
 for chunk in reader.stream().to_chunks():
  entity=chunk.entity_path;b=chunk.to_record_batch();counts[entity]=counts.get(entity,0)+b.num_rows
  for family,a in [('R6',base),('repaired',final),('GT',None)]:
   if entity==f'/world/{family}/object/mesh':
    v,f=(cad.vertices,cad.faces) if a is None else (a['vertices_canonical'],a['faces']);np.testing.assert_array_equal(vector(b,'Mesh3D:vertex_positions',0,3),v.astype(np.float32));np.testing.assert_array_equal(vector(b,'Mesh3D:triangle_indices',0,3),f.astype(np.uint32))
   if entity in [f'/world/{family}/object',f'/world/{family}/camera']:
    for row,i in enumerate(b.column('frame').to_pylist()):
     if a is None:T=G[i] if entity.endswith('object') else D[i]
     elif entity.endswith('object'):
      O=np.eye(4);O[:3,:3]=a['object_rotation'][i];O[:3,3]=a['object_translation_world'][i];T=A@O
     else:T=A@a['T_world_camera'][i]
     P=np.eye(4);P[:3,:3]=vector(b,'Transform3D:mat3x3',row,9)[0].reshape(3,3,order='F');P[:3,3]=vector(b,'Transform3D:translation',row,3)[0];err=float(np.max(np.abs(P-T)));assert err<2e-7;maxpose=max(maxpose,err)
   for s in ['left','right']:
    if entity==f'/world/{family}/{s}':
     si=int(np.flatnonzero(gt['mano_sides']==s)[0]);f=gt['mano_'+s+'_faces'] if a is None else a[s+'_faces']
     for row,i in enumerate(b.column('frame').to_pylist()):
      v=gt['mano_world_vertices_m'][i,si] if a is None else apply(A,a[s+'_vertices'][i]);np.testing.assert_array_equal(vector(b,'Mesh3D:vertex_positions',row,3),v.astype(np.float32));np.testing.assert_array_equal(vector(b,'Mesh3D:triangle_indices',row,3),f.astype(np.uint32))
 for family in ['R6','repaired','GT']:
  for entity in ['object','camera','left','right']:assert counts[f'/world/{family}/{entity}']==150
 res=subprocess.run([str(HOME/'ego_skrl/.venv/bin/rerun'),'rrd','verify','--check-footers','true',str(file)],capture_output=True,text=True);(HERE/'official_rrd_verify.log').write_text(res.stdout+res.stderr);assert res.returncode==0
 verify=read(HERE/'final_review/render_verification.json')
 for path,h in verify['inputs_sha256'].items():assert sha(path)==h
 (HERE/'final_review/verification.json').write_text(json.dumps(dict(status='passed',stages=checks,RRD_full_mesh_data_equal=True,RRD_object_camera_poses_max_error=maxpose,RRD_counts=counts,RRD_hand_mesh_rows=900,RRD_object_pose_rows=450,RRD_camera_pose_rows=450,official_verify_passed=True,scope='actual state/reproducible FK/native mesh/video correspondence, not physical-contact accuracy certification'),indent=2)+'\n')
 print('All frozen stage/FK/complete video/RRD checks passed',flush=True)
if __name__=='__main__':main()
