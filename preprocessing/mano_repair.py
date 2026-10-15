"""Full MANO joint repair with source RGB material tracks, shared shape and motion.
No raw-depth pull on hands, whose source reliability is explicitly weak.
"""
from pathlib import Path
import inspect,json,time
import numpy as np,cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from preprocessing.geometry import TriangleMesh,project,camera_pose,ray_directions
from preprocessing.inputs import sha
TIP=np.array([744,320,443,554,671]);ORDER=np.array([0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20])

def models_from_source(inputs):
 if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
 for name,value in [('bool',np.bool_),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
  if name not in np.__dict__:setattr(np,name,value)
 import torch,smplx
 torch.set_num_threads(1)
 rec=inputs.bundle.manifest['sources']['mano_assets'];root=Path(rec['path']);models={};hashes={}
 for side in ('left','right'):
  path=root/f'MANO_{side.upper()}.pkl';h=sha(path)
  if h!=rec[side+'_sha256']:raise ValueError('MANO source model hash mismatch')
  models[side]=smplx.MANOLayer(str(path),is_rhand=side=='right',use_pca=False).cpu().eval();hashes[side]=dict(path=str(path),sha256=h)
 models['left'].shapedirs[:,0,:]*=-1
 return models,hashes

def fk(model,root,pose,beta,translation):
 from smplx.lbs import batch_rodrigues
 import torch
 n=len(root);out=model(global_orient=batch_rodrigues(root.reshape(-1,3)).reshape(n,1,3,3),hand_pose=batch_rodrigues(pose.reshape(-1,3)).reshape(n,15,3,3),betas=beta,transl=translation,return_verts=True)
 joints=torch.cat([out.joints[:,:16],out.vertices[:,TIP]],dim=1)[:,ORDER]
 return out.vertices,joints

def image_tracks(inputs,archive,side):
 """Actual RGB feature trajectories bound conditionally to full source MANO triangles.
 Bindings are not independent absolute pose observations. Whole-tracklet holdout.
 """
 records=[];n=len(inputs.ids);nextid=0
 for ref in range(0,n,10):
  if not archive[side+'_valid'][ref]:continue
  C=inputs.cameras[ref];V=archive[side+'_vertices'][ref];F=archive[side+'_faces'];mesh=TriangleMesh(V,F)
  r=C[:3,:3].T;t=-r@C[:3,3];_,hz,mask=mesh.raster(inputs.K[ref],r,t,inputs.wh)
  obj=TriangleMesh(archive['vertices_canonical'],archive['faces']);rr,tt=camera_pose(archive['object_rotation'][ref],archive['object_translation_world'][ref],C);_,oz,_=obj.raster(inputs.K[ref],rr,tt,inputs.wh)
  mask &= (hz<oz+.003)
  mask=cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8))*255
  uv=cv2.goodFeaturesToTrack(inputs.grays[ref],maxCorners=32,qualityLevel=.02,minDistance=6,mask=mask)
  if uv is None:continue
  uv=uv.reshape(-1,2);worlddirs=ray_directions(uv,inputs.K[ref])@C[:3,:3].T
  rays=np.c_[np.broadcast_to(C[:3,3],worlddirs.shape),worlddirs].astype(np.float32);hit=mesh.scene.cast_rays(mesh.o3d.core.Tensor(rays),nthreads=1)
  z=hit['t_hit'].numpy();ok=np.isfinite(z);uv=uv[ok];tris=F[hit['primitive_ids'].numpy()[ok]];weights=hit['primitive_uvs'].numpy()[ok];bary=np.c_[1-weights.sum(1),weights]
  ids=np.arange(nextid,nextid+len(uv));nextid+=len(uv)
  for direction in (-1,1):
   current=uv.copy();alive=np.arange(len(uv));previous=ref
   for i in range(ref+direction,max(-1,ref-7) if direction<0 else min(n,ref+7),direction):
    if not len(current):break
    q,status,_=cv2.calcOpticalFlowPyrLK(inputs.grays[previous],inputs.grays[i],current.astype(np.float32).reshape(-1,1,2),None,winSize=(15,15),maxLevel=2)
    back,bst,_=cv2.calcOpticalFlowPyrLK(inputs.grays[i],inputs.grays[previous],q,None,winSize=(15,15),maxLevel=2)
    q=q.reshape(-1,2);back=back.reshape(-1,2);valid=status.ravel().astype(bool)&bst.ravel().astype(bool)&(np.linalg.norm(back-current,axis=1)<1.)
    valid&=(q[:,0]>=1)&(q[:,0]<inputs.wh[0]-1)&(q[:,1]>=1)&(q[:,1]<inputs.wh[1]-1)
    alive=alive[valid];current=q[valid]
    for a,u in zip(alive,current):records.append((i,ref,int(ids[a]),tris[a].copy(),bary[a].copy(),u.copy()))
    previous=i
 if records:
  frames,refs,ids,tris,bary,uv=zip(*records)
  return dict(frames=np.array(frames),references=np.array(refs),identities=np.array(ids),triangles=np.array(tris),barycentric=np.array(bary),uv=np.array(uv))
 return dict(frames=np.zeros(0,int),references=np.zeros(0,int),identities=np.zeros(0,int),triangles=np.zeros((0,3),int),barycentric=np.zeros((0,3)),uv=np.zeros((0,2)))

def repair_hands(inputs,archive,output_dir):
 import torch
 from smplx.lbs import batch_rodrigues
 start=time.monotonic();models,hashes=models_from_source(inputs);source=inputs.bundle.source_path('mano_archive')
 with np.load(source,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
 tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cpu')
 n=len(inputs.ids);C=tensor(inputs.cameras);K=tensor(inputs.K);Rc=C[:,:3,:3];tc=C[:,:3,3]
 project_t=lambda v:v[...,:2]/v[...,2:].clamp(min=.01)*K[:,None,:2]+K[:,None,2:]
 to_camera=lambda v:torch.einsum('nji,nkj->nki',Rc,v-tc[:,None])
 state={};parameters=[];replay={};tracks={};report={}
 for side in ('left','right'):
  rot=tensor(raw[side+'_root_orient_axis_angle']);pose=tensor(raw[side+'_hand_pose_axis_angle']);trans=tensor(raw[side+'_trans_world_m']);betas=tensor(raw[side+'_betas'])
  with torch.no_grad():V,J=fk(models[side],rot,pose,betas,trans)
  err=float(torch.max(torch.abs(V-tensor(raw[side+'_vertices_world_m']))));jerr=float(torch.max(torch.abs(J-tensor(raw[side+'_joints_world_m']))))
  if err>2e-6 or jerr>2e-6:raise ValueError(f'MANO source replay mismatch {side} V={err} J={jerr}')
  replay[side]=dict(max_vertex_error_m=err,max_joint_error_m=jerr)
  track=image_tracks(inputs,archive,side);tracks[side]=track;np.savez_compressed(Path(output_dir)/f'{side}_RGB_material_tracks.npz',**track)
  valid=tensor(raw[side+'_valid']).bool();beta=torch.nn.Parameter(torch.median(betas[valid],dim=0).values);dr=torch.nn.Parameter(torch.zeros_like(rot));dp=torch.nn.Parameter(torch.zeros_like(pose));dt=torch.nn.Parameter(torch.zeros_like(trans))
  parameters.extend([dict(params=[beta],lr=.002),dict(params=[dr,dp],lr=.001),dict(params=[dt],lr=.0005)])
  state[side]=dict(rot=rot,pose=pose,trans=trans,betas=betas,beta=beta,dr=dr,dp=dp,dt=dt,valid=valid,sourceV=V.detach(),sourceJ=J.detach(),source_uv=project_t(to_camera(J)).detach())
  report[side]=dict(tracks=int(len(track['frames'])),train_tracks=int(np.sum(track['identities']%4!=0)),heldout_tracks=int(np.sum(track['identities']%4==0)))
 objectR=tensor(archive['object_rotation']);objectt=tensor(archive['object_translation_world']);object_dr=torch.nn.Parameter(torch.zeros((n,3)));object_dt=torch.nn.Parameter(torch.zeros((n,3)));parameters.extend([dict(params=[object_dr],lr=.0005),dict(params=[object_dt],lr=.0003)])
 # Fixed surface anchor identities are conditional geometry; declared digits are
 # independent image evidence, not inferred from proximity. Rebind outer passes.
 om=TriangleMesh(archive['vertices_canonical'],archive['faces']);object_points=tensor(archive['vertices_canonical'][::3000]);contact={};digits={'thumb':4,'index':8,'middle':12,'ring':16,'little':20}
 for side in ('left','right'):
  rows=[]
  for row in inputs.visibility.get('interaction_judgments',[]):
   if row['hand_side']!=side or row['contact_state']!='likely_contact':continue
   for i in range(row['frame_start'],row['frame_end']+1):
    for d in row['primary_pad_declarations']:
     if d['declared'] is not True:continue
     tip=digits[d['digit']];P=archive[side+'_joints'][i,tip];canonical=(P-archive['object_translation_world'][i])@archive['object_rotation'][i]
     near,dist=om.closest(canonical[None]);normal=om.normals(near)
     # No hard zero-gap target: weak normal-distance term with the recorded
     # conservative support prior, at least60mm here, and independent likelihood.
     rows.append((i,tip,near[0],normal[0],max(.02,float(row['contact_support_uncertainty_m'])),float(row['contact_prior_probability'])))
  if rows:
   frame,tip,X,normal,sigma,prob=zip(*rows);contact[side]=dict(frame=np.array(frame),tip=np.array(tip),X=tensor(X),normal=tensor(normal),sigma=tensor(sigma),prob=tensor(prob))
 optimizer=torch.optim.Adam(parameters);trace=[]
 def cost():
  OR=batch_rodrigues(object_dr)@objectR;OT=objectt+object_dt;terms={};output={}
  norm=lambda x:torch.mean(2*(torch.sqrt(1+x*x)-1)) if x.numel() else torch.tensor(0.)
  for side,s in state.items():
   beta=s['beta'][None].expand(n,-1);V,J=fk(models[side],s['rot']+s['dr'],s['pose']+s['dp'],beta,s['trans']+s['dt']);output[side]=(V,J)
   uv=project_t(to_camera(J));valid=s['valid'];terms[side+'_projection_prior']=.5*norm((uv[valid]-s['source_uv'][valid])/6)
   terms[side+'_source_pose_prior']=.05*norm(s['dr'][valid]/.14)+.03*norm(s['dp'][valid]/.18)+.05*norm(s['dt'][valid]/.025)
   terms[side+'_shared_shape_prior']=.05*norm((s['beta']-torch.median(s['betas'][valid],dim=0).values)/.3)
   trip=valid[:-2]&valid[1:-1]&valid[2:];acc=(J[2:]-2*J[1:-1]+J[:-2])*900
   terms[side+'_actual_world_acceleration']=.20*norm(acc[trip]/3.)
   tr=tracks[side];train=tr['identities']%4!=0
   if train.any():
    fr=tr['frames'][train];tri=tr['triangles'][train];bary=tensor(tr['barycentric'][train]);P=(V[fr[:,None],tri]*bary[:,:,None]).sum(1)
    pc=torch.einsum('nji,nj->ni',Rc[fr],P-tc[fr]);puv=pc[:,:2]/pc[:,2:].clamp(min=.01)*K[fr,:2]+K[fr,2:]
    terms[side+'_RGB_track']=.40*norm((puv-tensor(tr['uv'][train]))/2.5)
   if side in contact:
    c=contact[side];fr=c['frame'];tip=c['tip'];P=torch.einsum('nij,nj->ni',OR[fr],c['X'])+OT[fr];N=torch.einsum('nij,nj->ni',OR[fr],c['normal']);delta=J[fr,tip]-P
    terms[side+'_uncertain_declared_contact']=.08*norm((delta*N).sum(1)/c['sigma']*torch.sqrt(c['prob']))
  # Source calibrated object reprojection/depth and full native geometry anchor
  # prevent the uncertain hands from moving the object arbitrarily.
  terms['object_calibrated_pose_prior']=2.*norm(object_dt/.008)+norm(object_dr/.08)
  fullp=torch.einsum('nij,kj->nki',OR,object_points)+OT[:,None];full0=torch.einsum('nij,kj->nki',objectR,object_points)+objectt[:,None]
  terms['object_surface_prior']=norm((fullp-full0)/.01)
  terms['object_actual_world_acceleration']=.03*norm((fullp[2:]-2*fullp[1:-1]+fullp[:-2])*900/6.)
  return sum(terms.values()),terms,output,OR,OT
 with torch.no_grad():initial,initialterms,_,_,_=cost()
 for iteration in range(100):
  optimizer.zero_grad();loss,terms,_,_,_=cost()
  if not torch.isfinite(loss):raise ValueError('nonfinite MANO/object loss')
  loss.backward();torch.nn.utils.clip_grad_norm_([p for g in parameters for p in g['params']],10.);optimizer.step()
  # These are general bounded source corrections, not frame/side-specific depth fixes.
  with torch.no_grad():
   for s in state.values():s['dt'].clamp_(-.04,.04);s['dr'].clamp_(-.20,.20);s['dp'].clamp_(-.25,.25)
   object_dt.clamp_(-.015,.015);object_dr.clamp_(-.10,.10)
  if iteration%20==0 or iteration==99:
   trace.append(dict(iteration=iteration,total=float(loss),terms={k:float(v) for k,v in terms.items()}));print('MANO joint fit',iteration,float(loss),flush=True)
 with torch.no_grad():final,terms,output,OR,OT=cost()
 changed={k:v.copy() for k,v in archive.items()};changed['object_rotation']=OR.numpy().astype(float);changed['object_translation_world']=OT.numpy().astype(float)
 for side,s in state.items():
  V,J=output[side];changed[side+'_vertices']=V.numpy();changed[side+'_joints']=J.numpy();changed[side+'_source_betas']=raw[side+'_betas'];changed[side+'_betas']=np.repeat(s['beta'].detach().numpy()[None],n,axis=0)
  for key,value in [('root_orient_axis_angle',s['rot']+s['dr']),('hand_pose_axis_angle',s['pose']+s['dp']),('trans_world_m',s['trans']+s['dt'])]:changed[side+'_'+key]=value.detach().numpy()
  report[side].update(mean_full_surface_change_m=float(torch.linalg.norm(V-s['sourceV'],dim=-1).mean()),max_root_translation_correction_m=float(torch.linalg.norm(s['dt'],dim=-1).max()),constant_betas=changed[side+'_betas'][0].tolist())
  tr=tracks[side];hold=tr['identities']%4==0
  if hold.any():
   fr=tr['frames'][hold];tri=tr['triangles'][hold];bary=tr['barycentric'][hold];points=(changed[side+'_vertices'][fr[:,None],tri]*bary[:,:,None]).sum(1);cam=np.einsum('nji,nj->ni',inputs.cameras[fr,:3,:3],points-inputs.cameras[fr,:3,3]);uv=cam[:,:2]/cam[:,2:]*inputs.K[fr,:2]+inputs.K[fr,2:];error=np.linalg.norm(uv-tr['uv'][hold],axis=-1);report[side]['heldout_RGB_reprojection_mean_px']=float(error.mean())
 result=dict(method='full_MANO_FK_shared_shape_RGB_material_tracks_actual_world_motion_and_uncertain_declared_contact',source_replay=replay,models=hashes,source_MANO_archive_sha256=sha(source),before_cost=float(initial),after_cost=float(final),iterations=trace,hands=report,object_translation_max_correction_m=float(torch.linalg.norm(object_dt,dim=-1).max()),seconds=time.monotonic()-start,source_independent_2D_joint_detector_available=False,physical_contact_claim=False,signed_nonpenetration_claim=False,occlusion_uncertainty_source_preserved=True,limits=['Pixel tracks are conditional on initial MANO feature bindings, not independent absolute root truth','Declared contact support is a conservative prior, no certain zero gap imposed','Single hand beta improves consistency, not independent identity/size accuracy','No blanket Z-direction correction; real hand-object motion preserved as image evidence allows'])
 (Path(output_dir)/'mano_repair.json').write_text(json.dumps(result,indent=2)+'\n')
 return changed,result
