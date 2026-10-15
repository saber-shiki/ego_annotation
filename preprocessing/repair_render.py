"""Complete source RGB/world comparison driven by actual mesh/MANO arrays."""
from pathlib import Path
import json,subprocess
import numpy as np,cv2
from preprocessing.geometry import TriangleMesh,camera_pose,resized_intrinsics
from preprocessing.render import look_at,paint

def render_stage(inputs,before,after,output,width=384):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);wh=(width,round(width*inputs.wh[1]/inputs.wh[0]));fps=1/np.median(np.diff(inputs.times));H=wh[1]+42
 meshes=[TriangleMesh(a['vertices_canonical'],a['faces']) for a in (before,after)]
 points=[]
 for a in (before,after):
  for i in range(0,len(inputs.ids),10):points.append(a['vertices_canonical'][::500]@a['object_rotation'][i].T+a['object_translation_world'][i])
  for s in ('left','right'):points.append(a[s+'_vertices'][::10].reshape(-1,3))
 pts=np.concatenate(points);center=(pts.min(0)+pts.max(0))/2;radius=max(.15,np.linalg.norm(np.ptp(pts,axis=0))/2);C0=inputs.cameras[0]
 eye=center+C0[:3,:3]@np.array([.6*radius,-.45*radius,-2.9*radius]);view=look_at(eye,center,C0[:3,:3]@np.array([0.,-1.,0.]));kv=np.array([width*1.6,width*1.6,(width-1)/2,(width-1)/2])
 pipes=[]
 for name in ['before_after_overlay.mp4','before_after_world.mp4']:
  cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width*2}x{H}','-r',str(fps),'-i','-','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output/name)]
  pipes.append(subprocess.Popen(cmd,stdin=subprocess.PIPE))
 try:
  for i,fid in enumerate(inputs.ids):
   rgb=cv2.resize(cv2.imread(inputs.raw_paths[i]),wh);C=inputs.cameras[i];k=resized_intrinsics(inputs.K[i],inputs.wh,wh);over=[];world=[]
   for j,a in enumerate((before,after)):
    image=rgb.copy();depth=np.full(wh[::-1],np.inf);wi=np.full(wh[::-1]+(3,),246,np.uint8);wd=np.full(wh[::-1],np.inf)
    for im,d,cam,K in [(image,depth,C,k),(wi,wd,view,kv)]:
     r,t=camera_pose(a['object_rotation'][i],a['object_translation_world'][i],cam);_,z,_=meshes[j].raster(K,r,t,wh);paint(im,d,z,(230,170,35) if j==0 else (185,170,25),.55 if im is image else .95)
    for side,col in [('left',(200,100,50)),('right',(230,180,80))]:
     if not a[side+'_valid'][i]:continue
     hm=TriangleMesh(a[side+'_vertices'][i],a[side+'_faces'])
     for im,d,cam,K in [(image,depth,C,k),(wi,wd,view,kv)]:
      r=cam[:3,:3].T;t=-r@cam[:3,3];_,z,_=hm.raster(K,r,t,wh);paint(im,d,z,col,.45 if im is image else .95)
    head=np.full((42,width,3),25,np.uint8);cv2.putText(head,f'f{fid:03d} '+('R6 baseline' if j==0 else 'sequential repair'),(7,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1);cv2.putText(head,'Full mesh + MANO; metres; no contact claim',(7,35),cv2.FONT_HERSHEY_SIMPLEX,.32,(210,210,210),1)
    over.append(np.vstack([head,image]));world.append(np.vstack([head,wi]))
   overlay=np.hstack(over);wv=np.hstack(world);pipes[0].stdin.write(overlay.tobytes());pipes[1].stdin.write(wv.tobytes())
   if i in [0,30,60,88,94,120,130,143,144,147,len(inputs.ids)-1]:cv2.imwrite(str(output/f'overlay_f{i:03d}.jpg'),overlay);cv2.imwrite(str(output/f'world_f{i:03d}.jpg'),wv)
 finally:
  for p in pipes:p.stdin.close()
  codes=[p.wait() for p in pipes]
 if any(codes):raise RuntimeError('ffmpeg failure '+str(codes))
 for name in ['before_after_overlay.mp4','before_after_world.mp4']:
  cap=cv2.VideoCapture(str(output/name));count=0
  while True:
   ok,im=cap.read()
   if not ok:break
   count+=1
  cap.release()
  if count!=len(inputs.ids):raise ValueError('render incomplete')
 (output/'render.json').write_text(json.dumps(dict(frames=len(inputs.ids),fps=float(fps),duration_s=float(len(inputs.ids)/fps),original_triangles_preserved=True,hands_from_actual_archive=True,physical_contact_claim=False,fully_decoded=True),indent=2)+'\n')
