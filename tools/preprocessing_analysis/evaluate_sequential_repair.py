#!/usr/bin/env python3
"""Read-only evaluation of the current RL R6 preprocessing state vs HOT3D CAD/GT.
No perception, policy, pose fitting, per-frame world registration, or scale fit.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import sys,json,csv,hashlib,argparse,inspect
# Legacy chumpy pickles predate Python3.11. Process-local compatibility only.
if not hasattr(inspect,'getargspec'): inspect.getargspec=inspect.getfullargspec
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import open3d as o3d
HOME=Path('/mnt/user-home/kupingxin')
REPO=HOME/'ego_annotation_rl_three_rewards'
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(HOME/'ego_annotation_experiments/hot3d_s3_pose_audit_20260927'))
from evaluate import sample,scene,distance,gtpose
from hot3d_eval.mano_fk import ManoFK,OURS21_TO_OFFICIAL19,MANO21_TO_OURS21,TIP_VERTICES
from hot3d_eval.alignment import fit_points,transform_points
from hot3d_eval.metrics_camera import evaluate_camera
HERE=Path(os.environ.get('MILK_REPAIR_OUTPUT',Path(__file__).resolve().parent)).resolve()
HERE.mkdir(parents=True,exist_ok=True)
PRED=Path(os.environ['MILK_REPAIR_PRED']).resolve()
if not (PRED.parent/'completed.json').exists():raise ValueError('Only frozen completed predictions may be evaluated')
CLIP=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/clips/P0014_84ea2dcc_carton_milk_f2370_2519')
GT=CLIP/'depth_gt/camera_object_hand_gt.npz'
CAD=Path('/mnt/nas-106/hot3d/dataset/assets/125863066770940.glb')
DATA=Path('/mnt/nas-106/hot3d/dataset/P0014_84ea2dcc')
OBS=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2')
SIDE=CLIP/'evaluation/hot3d_gt/hot3d_vrs_gt_sidecar.json'
MANO_ROOT=Path('/var/tmp/kupingxin/ego_annotation_envs/hawor_work/HaWoR/_DATA')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def apply(T,x):return np.asarray(x)@T[:3,:3].T+T[:3,3]
def ang(R):return float(np.rad2deg(Rotation.from_matrix(R).magnitude()))
def st(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 return dict(count=int(x.size),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()),rmse=float(np.sqrt(np.mean(x*x)))) if len(x) else dict(count=0)
def mesh(p):
 m=trimesh.load(p,process=False)
 return m.to_geometry() if isinstance(m,trimesh.Scene) else m

def sources():
 p,g=load(PRED),load(GT);m=read(PRED.with_suffix('.json'));cad=mesh(CAD)
 assert sha(PRED)==m['arrays']['sha256']
 assert np.array_equal(p['frame_idx'],g['frame_idx'])
 assert np.array_equal(g['source_frame_idx'],np.arange(2370,2520))
 assert np.array_equal(p['image_size_wh'],np.tile(g['image_size_wh'],(150,1)))
 assert np.max(np.abs(p['intrinsics_source']-g['intrinsics_fx_fy_cx_cy']))<1e-10
 oi=int(np.flatnonzero(g['object_uids']=='125863066770940')[0])
 O=np.tile(np.eye(4),(150,1,1));O[:,:3,:3]=p['object_rotation'];O[:,:3,3]=p['object_translation_world']
 C=p['T_world_camera'];D=g['T_world_from_camera'];G=g['T_world_from_object'][:,oi]
 for a in (O,C,D,G):
  assert np.isfinite(a).all()
  assert np.max(np.abs(a[:,:3,:3]@a[:,:3,:3].transpose(0,2,1)-np.eye(3)))<2e-4
  assert np.max(np.abs(np.linalg.det(a[:,:3,:3])-1))<2e-4
 A=D[0]@np.linalg.inv(C[0]);B=np.linalg.inv(A@O[0])@G[0]
 side=read(SIDE)
 assert np.array_equal([f['timestamp_ns'] for f in side['frames']],g['timestamp_ns'])
 for i,f in enumerate(side['frames']):
  assert np.allclose(D[i],gtpose(f['json']['cameras.json']['214-1']['T_world_from_camera']),atol=1e-8)
 return p,g,m,cad,O,C,D,G,A,B

def masks(name,timestamps):
 path=DATA/'masks'/f'{name}.csv'
 with path.open() as f:d={int(r['timestamp[ns]']):r['mask'].lower()=='true' for r in csv.DictReader(f) if r['stream_id']=='214-1'}
 return np.array([d.get(int(t),False) for t in timestamps]),path

def gt_joints(g):
 paths={'left':MANO_ROOT/'data_left/mano_left/MANO_LEFT.pkl','right':MANO_ROOT/'data/mano/MANO_RIGHT.pkl'}
 fk=ManoFK(paths);ts=set(map(int,g['timestamp_ns']));records={}
 with (DATA/'mano_hand_pose_trajectory.jsonl').open() as f:
  for line in f:
   r=json.loads(line)
   if r['timestamp_ns'] in ts:records[r['timestamp_ns']]=r['hand_poses']
 mask,_=masks('mask_hand_pose_available',g['timestamp_ns']);out={};replay={}
 for side,hid in [('left','0'),('right','1')]:
  si=int(np.flatnonzero(g['mano_sides']==side)[0])
  valid=np.array([mask[i] and hid in records.get(int(t),{}) and g['mano_available'][i,si] for i,t in enumerate(g['timestamp_ns'])])
  ids=np.flatnonzero(valid);rr=[records[int(g['timestamp_ns'][i])][hid] for i in ids]
  joints=np.full((150,21,3),np.nan);joints[ids]=fk.forward(side,rr)
  # Independent FK vertex replay validates units, wrist pivot, hand side, shape and world frame.
  torch=fk.torch;tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float32,device='cpu')
  with torch.no_grad():
   v=fk.models[side](betas=tensor([r['betas'] for r in rr]),hand_pose=tensor([r['pose'] for r in rr]),global_orient=tensor([Rotation.from_quat(np.asarray(r['wrist_xform']['q_wxyz'])[[1,2,3,0]]).as_rotvec() for r in rr]),transl=tensor([r['wrist_xform']['t_xyz'] for r in rr]),return_verts=True).vertices.numpy()
  error=float(np.max(np.linalg.norm(v-g['mano_world_vertices_m'][ids,si],axis=-1)))
  if error>1e-5:raise ValueError(f'GT FK/cache vertex disagreement: {side} {error}m')
  out[side]=(joints,valid);replay[side]={'max_vertex_replay_error_m':error,'exact_timestamp_records':len(ids)}
 return out,dict(models=fk.provenance,left_shapedirs_fixed=fk.left_shapedirs_fixed,replay=replay)

def centroid(m):
 tri=np.asarray(m.vertices)[np.asarray(m.faces)];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
 return np.average(tri.mean(1),axis=0,weights=area)
def symmetry_distance(a_scene,b_scene,a_points,b_points,T_b_from_a):
 x=distance(b_scene,apply(T_b_from_a,a_points));y=distance(a_scene,apply(np.linalg.inv(T_b_from_a),b_points))
 return np.r_[x,y]*1000

def evaluate():
 p,g,m,cad,O,C,D,G,A,B=sources();gtj,fkmeta=gt_joints(g)
 files=[PRED,PRED.with_suffix('.json'),GT,CAD,SIDE,OBS/'source_observations.json',OBS/'input/sensor_calibration.json',OBS/'input/runtime_input_contract.json',OBS/'input/raw.mp4',DATA/'mano_hand_pose_trajectory.jsonl']
 eligible=np.ones(150,bool)
 for key in ('mask_object_pose_available','mask_headset_pose_available','mask_qa_pass'):
  v,path=masks(key,g['timestamp_ns']);eligible&=v;files.append(path)
 handmask,path=masks('mask_hand_pose_available',g['timestamp_ns']);files.append(path)
 hashes={str(f):sha(f) for f in files}
 pm=trimesh.Trimesh(p['vertices_canonical'],p['faces'],process=False)
 ps=sample(pm.vertices,pm.faces,4096);gs=sample(cad.vertices,cad.faces,4096)
 pscene=scene(pm.vertices,pm.faces);gscene=scene(cad.vertices,cad.faces)
 pc=centroid(pm);gc=centroid(cad);rows=[];cache={}
 Pw=A@O;Cw=A@C
 for side in ('left','right'):
  cache[f'pred_{side}_world_vertices']=apply(A,p[f'{side}_vertices'])
  cache[f'pred_{side}_world_joints']=apply(A,p[f'{side}_joints'])
  cache[f'gt_{side}_world_joints']=gtj[side][0]
  cache[f'{side}_valid']=p[f'{side}_valid'].astype(bool)&gtj[side][1]
 for i in range(150):
  row=dict(frame_idx=i,source_frame_idx=int(g['source_frame_idx'][i]),time_s=float(p['time_s'][i]),timestamp_ns=int(g['timestamp_ns'][i]),GT_quality_eligible=bool(eligible[i]))
  row['camera_translation_mm']=float(np.linalg.norm(Cw[i,:3,3]-D[i,:3,3])*1000)
  row['camera_rotation_deg']=ang(Cw[i,:3,:3]@D[i,:3,:3].T)
  row['object_centroid_world_mm']=float(np.linalg.norm(apply(Pw[i],pc)-apply(G[i],gc))*1000)
  H=np.linalg.inv(C[i])@O[i];Hgt=np.linalg.inv(D[i])@G[i]
  row['object_centroid_camera_mm']=float(np.linalg.norm(apply(H,pc)-apply(Hgt,gc))*1000)
  row['object_centroid_camera_bias_xyz_mm']=((apply(H,pc)-apply(Hgt,gc))*1000).tolist()
  E=Pw[i]@B
  row['object_motion_world_rotation_deg']=ang(E[:3,:3]@G[i,:3,:3].T)
  row['object_motion_world_translation_mm']=float(np.linalg.norm(E[:3,3]-G[i,:3,3])*1000)
  E=np.linalg.inv(C[i])@O[i]@B
  row['object_motion_camera_rotation_deg']=ang(E[:3,:3]@Hgt[:3,:3].T)
  row['object_motion_camera_translation_mm']=float(np.linalg.norm(E[:3,3]-Hgt[:3,3])*1000)
  for frame,T in [('world',np.linalg.inv(G[i])@Pw[i]),('camera',np.linalg.inv(Hgt)@H)]:
   d=symmetry_distance(pscene,gscene,ps,gs,T)
   row[f'object_surface_{frame}_mean_mm']=float(d.mean());row[f'object_surface_{frame}_p95_mm']=float(np.percentile(d,95));row[f'object_surface_{frame}_within5mm_fraction']=float(np.mean(d<=5))
  for side in ('left','right'):
   valid=cache[f'{side}_valid'][i];row[f'{side}_valid']=bool(valid)
   if not valid:continue
   si=int(np.flatnonzero(g['mano_sides']==side)[0]);pj=cache[f'pred_{side}_world_joints'][i];gj=gtj[side][0][i]
   pv=cache[f'pred_{side}_world_vertices'][i];gv=g['mano_world_vertices_m'][i,si]
   hc=apply(np.linalg.inv(C[i]),p[f'{side}_joints'][i]);gcj=apply(np.linalg.inv(D[i]),gj)
   row[f'{side}_MPJPE19_world_mm']=float(np.linalg.norm(pj[OURS21_TO_OFFICIAL19]-gj[OURS21_TO_OFFICIAL19],axis=-1).mean()*1000)
   row[f'{side}_MPJPE19_camera_mm']=float(np.linalg.norm(hc[OURS21_TO_OFFICIAL19]-gcj[OURS21_TO_OFFICIAL19],axis=-1).mean()*1000)
   row[f'{side}_wrist_world_mm']=float(np.linalg.norm(pj[0]-gj[0])*1000)
   row[f'{side}_wrist_camera_mm']=float(np.linalg.norm(hc[0]-gcj[0])*1000)
   row[f'{side}_wrist_camera_bias_xyz_mm']=((hc[0]-gcj[0])*1000).tolist()
   row[f'{side}_wrist_centered_MPJPE19_mm']=float(np.linalg.norm((pj-pj[0])[OURS21_TO_OFFICIAL19]-(gj-gj[0])[OURS21_TO_OFFICIAL19],axis=-1).mean()*1000)
   row[f'{side}_PA_MPJPE21_mm']=float(np.linalg.norm(transform_points(fit_points(pj,gj,scale=True),pj)-gj,axis=-1).mean()*1000)
   row[f'{side}_tip_world_mm']=float(np.linalg.norm(pj[[4,8,12,16,20]]-gj[[4,8,12,16,20]],axis=-1).mean()*1000)
   row[f'{side}_MPVPE_world_mm']=float(np.linalg.norm(pv-gv,axis=-1).mean()*1000)
   row[f'{side}_MPVPE_camera_mm']=float(np.linalg.norm(apply(np.linalg.inv(C[i]),p[f'{side}_vertices'][i])-apply(np.linalg.inv(D[i]),gv),axis=-1).mean()*1000)
   k=p['intrinsics_source'][i];project=lambda x:x[:,:2]/x[:,2:]*k[:2]+k[2:]
   positive=(hc[:,2]>0)&(gcj[:,2]>0)
   row[f'{side}_joint_projection_mean_px']=float(np.linalg.norm(project(hc)[positive]-project(gcj)[positive],axis=-1).mean())
  rows.append(row)
  if (i+1)%30==0:print('evaluated',i+1,flush=True)
 selected=[r for r in rows if r['GT_quality_eligible']]
 keys=[k for k,v in rows[0].items() if isinstance(v,float) and k not in ('time_s',)]
 metrics={k:st([r[k] for r in selected if k in r]) for k in keys}
 cam_metrics=evaluate_camera(Cw,D,p['frame_idx'])
 for side in ('left','right'):
  valid=cache[f'{side}_valid'];pj=cache[f'pred_{side}_world_joints'];gj=cache[f'gt_{side}_world_joints']
  trip=valid[:-2]&valid[1:-1]&valid[2:]
  ap=np.diff(pj,n=2,axis=0)*900;ag=np.diff(gj,n=2,axis=0)*900
  metrics[f'{side}_acceleration_error_m_s2']=st(np.linalg.norm(ap[trip]-ag[trip],axis=-1).mean(1))
  metrics[f'{side}_pred_acceleration_m_s2']=st(np.linalg.norm(ap[trip],axis=-1).mean(1))
  metrics[f'{side}_GT_acceleration_m_s2']=st(np.linalg.norm(ag[trip],axis=-1).mean(1))
  metrics[f'{side}_wrist_camera_bias_mean_xyz_mm']=np.mean([r[f'{side}_wrist_camera_bias_xyz_mm'] for r in selected if r[f'{side}_valid']],axis=0).tolist()
 sizes={'pred_sorted_OBB_mm':(np.sort(pm.bounding_box_oriented.primitive.extents)*1000).tolist(),'GT_sorted_OBB_mm':(np.sort(cad.bounding_box_oriented.primitive.extents)*1000).tolist(),'pred_surface_area_cm2':float(pm.area*1e4),'GT_surface_area_cm2':float(cad.area*1e4),'pred_signed_volume_cm3':float(pm.volume*1e6),'GT_signed_volume_cm3':float(cad.volume*1e6),'pred_watertight':bool(pm.is_watertight),'GT_watertight':bool(cad.is_watertight)}
 sizes['OBB_length_ratio']=(np.array(sizes['pred_sorted_OBB_mm'])/sizes['GT_sorted_OBB_mm']).tolist()
 worst={k:max(selected,key=lambda r:r.get(k,-1))['frame_idx'] for k in ['camera_translation_mm','left_MPJPE19_world_mm','right_MPJPE19_world_mm','object_surface_camera_mean_mm','object_motion_camera_rotation_deg']}
 result=dict(schema='milk_R6_vs_HOT3D_CAD_GT.v1',compute='single-thread CPU; no model inference/training',prediction_path=str(PRED),GT_state_path=str(GT),CAD_path=str(CAD),frames=150,source_frames=[2370,2519],duration_s=5,GT_quality_frames=int(eligible.sum()),K=p['intrinsics_source'][0].tolist(),maximum_K_difference_px=float(np.max(np.abs(p['intrinsics_source']-g['intrinsics_fx_fy_cx_cy']))),source_MANO_unchanged=m['source_MANO_unchanged'],A_gt_world_from_pred_world=A.tolist(),B_pred_canonical_from_GT_object=B.tolist(),object_size=sizes,metrics=metrics,camera_metrics=cam_metrics,FK_validation=fkmeta,coverage={s:int(cache[f'{s}_valid'].sum()) for s in ('left','right')},worst_frames=worst,object_centroid_camera_bias_mean_xyz_mm=np.mean([r['object_centroid_camera_bias_xyz_mm'] for r in selected],axis=0).tolist(),input_sha256=hashes,definitions={'world':'A=GT_camera[0] @ inverse(pred_camera[0]); same single SE3 applied to all predicted cameras, hands and object. No scale fit, no per-frame registration.','camera':'Each state expressed in its own corresponding calibrated output camera at the matched frame; separates trajectory error from local reconstruction error. NOT a world-frame result.','object_rotation':'One fixed B=inverse(A @ pred_object[0]) @ GT_object[0], canonical offset removed. Scores motion only; zero error at frame0 is construction, not absolute pose correctness.','object_translation':'Motion scores use the same fixed B; geometric centroid and surface distances do NOT use B or any object alignment. Origins of generated/CAD meshes are incomparable directly.','object_surface':'4096 deterministic area samples in each direction to exact original triangles; symmetric mean over8192 distances. No ICP, scale fit or symmetry adjustment. Includes pose+shape+scale.','hands':'Official19 mapping; world/camera MPJPE preserves metric scale. Wrist-centered removes translation only. PA21 permits diagnostic per-frame Sim3 and must not be used as absolute accuracy. MPVPE compares full778 posed corresponding MANO vertices; NOT official rest-shape MPVPE.','size':'Sorted OBB dimensions, not semantic width/depth. Surface area and signed volume describe mesh construction; CAD is open and SAM3D may be double/thin shell, so volume not certified physical occupancy.','GT':'Recorded HOT3D camera, object poses, licensed MANO FK, CAD mesh. Exact matched timestamps, official QA/availability masks. GT itself is not collision-free certification.'})
 save(HERE/'metrics.json',result);save(HERE/'frame_metrics.json',rows)
 with (HERE/'frame_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
 np.savez_compressed(HERE/'aligned_comparison.npz',frame_idx=p['frame_idx'],source_frame_idx=g['source_frame_idx'],timestamp_ns=g['timestamp_ns'],time_s=p['time_s'],pred_camera_world=Cw,gt_camera_world=D,pred_object_world=Pw,gt_object_world=G,A=A,B=B,**cache)
 for f,h in hashes.items():assert sha(f)==h,f'input modified {f}'
 print(json.dumps({k:result[k] for k in ['coverage','camera_metrics','object_size','worst_frames']},indent=2),flush=True)
 print('evaluation complete',flush=True)

if __name__=='__main__':evaluate()
