"""Post-freeze full mesh stage-comparison rendering and metrics (read-only).
Does not write prediction arrays or feed independent references back to producers.
"""
from pathlib import Path
import sys,json,os,subprocess
import numpy as np,cv2
from PIL import Image
HOME=Path('/mnt/user-home/kupingxin')
AUDIT=HOME/'ego_annotation_experiments/milk_R6_vs_CAD_GT_20261014'
sys.path.insert(0,str(AUDIT))
from evaluate_comparison import load,sha,apply,PRED,GT,CAD,OBS,mesh,read,save
from render_comparison import observer,rays_for,render_native,frustum,draw_lines
from evaluate_comparison import scene
HERE=HOME/'ego_annotation_experiments/milk_preprocess_shape_depth_hands_20261015_r1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
RUN=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/milk_shape_depth_hands_20261015_r1')
STAGES=[('R6',PRED),('Shape',RUN/'stage1_shape_r2/coarse_scene.npz'),('Depth only',RUN/'stage2_depth_only_r2/coarse_scene.npz'),('Shape + depth',RUN/'stage2_shape_depth_r2/coarse_scene.npz'),('MANO joint',RUN/'stage3_hands_r2/coarse_scene.npz')]
COLORS=[(38,145,197),(30,172,159),(132,111,196),(25,165,192),(17,154,183)]

def run():
 cv2.setNumThreads(1);out=HERE/'final_review';out.mkdir(exist_ok=False)
 arrays=[];hashes={};g=load(GT);cad=mesh(CAD);oi=int(np.flatnonzero(g['object_uids']=='125863066770940')[0]);D=g['T_world_from_camera'];G=g['T_world_from_object'][:,oi]
 for label,path in STAGES:
  if label!='R6' and not (path.parent/'completed.json').exists():raise ValueError('not frozen complete '+str(path))
  a=load(path);A=D[0]@np.linalg.inv(a['T_world_camera'][0]);arrays.append((a,A));hashes[str(path)]=sha(path)
 samples=[]
 for a,A in arrays:
  for i in range(0,150,10):samples.append(apply(A,a['vertices_canonical'][::500]@a['object_rotation'][i].T+a['object_translation_world'][i]))
  for s in ('left','right'):samples.append(apply(A,a[s+'_vertices'][::10].reshape(-1,3)))
 samples.extend([g['mano_world_vertices_m'][::10].reshape(-1,3)])
 for i in range(0,150,10):samples.append(apply(G[i],cad.vertices[::30]))
 pts=np.concatenate(samples);W=H=360;T,K,center,radius=observer(pts,D,W,H);rays=rays_for(T,K,W,H)
 meshes=[scene(a['vertices_canonical'],a['faces']) for a,A in arrays];gs=scene(cad.vertices,cad.faces)
 worldsize=(W*6,H+44);rgbsize=(W*5,H+44)
 pipes=[]
 for name,size in [('all_stages_world.mp4',worldsize),('all_stages_overlay.mp4',rgbsize)]:
  cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','30','-i','-','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/name)];pipes.append(subprocess.Popen(cmd,stdin=subprocess.PIPE))
 from preprocessing.geometry import TriangleMesh,resized_intrinsics,camera_pose
 from preprocessing.render import paint
 overlaymeshes=[TriangleMesh(a['vertices_canonical'],a['faces']) for a,A in arrays]
 manifest=read(OBS/'input/raw_frame_manifest/manifest.json');frames=manifest.get('frames',manifest.get('rows',[]));rawpaths={int(r['frame_idx']):r.get('raw_frame_path',r.get('rgb_path',r.get('path'))) for r in frames}
 # Use exact decoded source video; transforms do not change source RGB.
 cap=cv2.VideoCapture(str(OBS/'input/raw.mp4'));images=[];sourceimages=[]
 for i in range(150):
  ok,rgb=cv2.VideoCapture.read(cap);assert ok;rgb=cv2.resize(rgb,(W,H));imrgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB);panels=[];overlays=[]
  for j,((label,path),(a,A)) in enumerate(zip(STAGES,arrays)):
   O=np.eye(4);O[:3,:3]=a['object_rotation'][i];O[:3,3]=a['object_translation_world'][i];objects=[(meshes[j],A@O,COLORS[j])]
   oi_rgb=rgb.copy();dep=np.full((H,W),np.inf);k=resized_intrinsics(a['intrinsics_source'][i],a['image_size_wh'][i],(W,H));c=a['T_world_camera'][i];r,t=camera_pose(a['object_rotation'][i],a['object_translation_world'][i],c);_,z,_=overlaymeshes[j].raster(k,r,t,(W,H));paint(oi_rgb,dep,z,(190,160,30),.50)
   for s,col in [('left',[39,103,179]),('right',[87,165,220])]:
    V=apply(A,a[s+'_vertices'][i]);sc=scene(V,a[s+'_faces']);objects.append((sc,np.eye(4),col))
    hm=TriangleMesh(a[s+'_vertices'][i],a[s+'_faces']);r=c[:3,:3].T;t=-r@c[:3,3];_,z,_=hm.raster(k,r,t,(W,H));paint(oi_rgb,dep,z,(200,130,80) if s=='left' else (230,170,70),.40)
   image,_=render_native(objects,rays);head=np.full((44,W,3),246,np.uint8);cv2.putText(head,f'{label}   f{i:03d}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,35,45),1,cv2.LINE_AA);panels.append(np.vstack([head,image]));overlays.append(np.vstack([head,cv2.cvtColor(oi_rgb,cv2.COLOR_BGR2RGB)]))
  objects=[(gs,G[i],[233,134,34])]
  for s,col in [('left',[180,82,30]),('right',[233,170,63])]:
   si=int(np.flatnonzero(g['mano_sides']==s)[0]);objects.append((scene(g['mano_world_vertices_m'][i,si],g['mano_'+s+'_faces']),np.eye(4),col))
  image,_=render_native(objects,rays);head=np.full((44,W,3),246,np.uint8);cv2.putText(head,f'CAD + MANO GT   f{i:03d}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,35,45),1,cv2.LINE_AA);panels.append(np.vstack([head,image]));world=np.hstack(panels);overlay=np.hstack(overlays);pipes[0].stdin.write(world.tobytes());pipes[1].stdin.write(overlay.tobytes())
  if i in [0,30,60,88,94,120,130,143,144,147,149]:Image.fromarray(world).save(out/f'all_stages_world_f{i:03d}.jpg',quality=94);Image.fromarray(overlay).save(out/f'all_stages_overlay_f{i:03d}.jpg',quality=94);images.append(world);sourceimages.append(overlay)
  if (i+1)%30==0:print('comparison render',i+1,flush=True)
 cap.release()
 for p in pipes:p.stdin.close()
 codes=[p.wait() for p in pipes];assert not any(codes)
 for name in ['all_stages_world.mp4','all_stages_overlay.mp4']:
  cap=cv2.VideoCapture(str(out/name));count=0
  while True:
   ok,im=cap.read()
   if not ok:break
   count+=1
  cap.release();assert count==150
 for name,items in [('world_sheet.jpg',images),('overlay_sheet.jpg',sourceimages)]:
  thumbs=[cv2.resize(im,(im.shape[1]//2,im.shape[0]//2)) for im in items];Image.fromarray(np.vstack(thumbs)).save(out/name,quality=93)
 for path,h in hashes.items():assert sha(path)==h
 save(out/'render_verification.json',dict(all_native_object_vertices_faces_preserved=True,all_MANO_vertices_preserved=True,world_alignment='same single first-camera SE3, scale1; no object/canonical alignment',frames_per_video=150,fps=30,duration_s=5,predictions_unchanged=True,inputs_sha256=hashes,external_observer_world_pose=T.tolist(),reference_only_after_prediction_freeze=True))
 print('comparison complete',flush=True)

if __name__=='__main__':run()
