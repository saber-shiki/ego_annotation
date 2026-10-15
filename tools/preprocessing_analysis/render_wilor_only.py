"""Post-freeze complete native object+MANO comparisons, common world observer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from preprocessing.geometry import TriangleMesh,resized_intrinsics,camera_pose
from preprocessing.render import paint,look_at
from utils.state import sha


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def apply(T,x):return np.asarray(x)@T[:3,:3].T+T[:3,3]


def main(a):
    for _,path in a.state:
        if not (Path(path).parent/'completed.json').is_file():raise ValueError('state not complete: '+path)
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    g=load(a.gt);D=g['T_world_from_camera'];ids=g['frame_idx'];n=len(ids)
    oi=int(np.flatnonzero(g['object_uids']==a.object_uid)[0]);G=g['T_world_from_object'][:,oi]
    import trimesh
    cad=trimesh.load(a.cad,process=False)
    if isinstance(cad,trimesh.Scene):cad=cad.to_geometry()
    states=[];samples=[];hashes={str(p):sha(p) for p in [a.gt,a.cad,a.video]}
    for label,path in a.state:
        path=Path(path);p=load(path);meta=json.loads(path.with_suffix('.json').read_text())
        if sha(path)!=meta['arrays']['sha256']:raise ValueError('state arrays changed')
        if not np.array_equal(p['frame_idx'],ids):raise ValueError('state timeline differs')
        if not np.allclose(p['intrinsics_source'],g['intrinsics_fx_fy_cx_cy'],atol=1e-8,rtol=0):raise ValueError('image planes differ')
        A=D[0]@np.linalg.inv(p['T_world_camera'][0]);p['display_A']=A
        for i in range(0,n,10):samples.append(apply(A,p['vertices_canonical'][::500]@p['object_rotation'][i].T+p['object_translation_world'][i]))
        for side in ['left','right']:samples.append(apply(A,p[side+'_vertices'][p[side+'_valid'].astype(bool)][::10].reshape(-1,3)))
        states.append((label,p,TriangleMesh(p['vertices_canonical'],p['faces'])));hashes[str(path)]=sha(path)
    for i in range(0,n,10):samples.append(apply(G[i],cad.vertices[::100]))
    samples.append(g['mano_world_vertices_m'][::10].reshape(-1,3));pts=np.concatenate(samples);pts=pts[np.isfinite(pts).all(1)]
    center=(pts.min(0)+pts.max(0))/2;radius=max(.15,np.linalg.norm(np.ptp(pts,axis=0))/2)
    observer=look_at(center+D[0,:3,:3]@np.array([1.0*radius,-.5*radius,-3.0*radius]),center,D[0,:3,:3]@np.array([0.,-1.,0.]))
    w=a.width;wh=(w,w);kv=np.array([w*1.3,w*1.3,(w-1)/2,(w-1)/2]);count=len(states)+1;cad_scene=TriangleMesh(cad.vertices,cad.faces)
    pipes=[]
    for name in ['full_pipeline_overlay.mp4','full_pipeline_world.mp4']:
        cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w*count}x{w+44}',
             '-r','30','-i','-','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/name)]
        pipes.append(subprocess.Popen(cmd,stdin=subprocess.PIPE))
    cap=cv2.VideoCapture(str(a.video));sheets=[[],[]]
    def panel(i,label,p,object_mesh,reference=False):
        C=D[i] if reference else p['T_world_camera'][i]
        k=resized_intrinsics(g['intrinsics_fx_fy_cx_cy'] if reference else p['intrinsics_source'][i],g['image_size_wh'],wh)
        A=np.eye(4) if reference else p['display_A']
        O=G[i].copy() if reference else np.eye(4)
        if not reference:O[:3,:3]=p['object_rotation'][i];O[:3,3]=p['object_translation_world'][i]
        image=rgb.copy();wi=np.full((w,w,3),244,np.uint8);dep=np.full((w,w),np.inf);wd=np.full((w,w),np.inf)
        for im,zbuf,cam,K,world in [(image,dep,C,k,False),(wi,wd,observer,kv,True)]:
            Oview=A@O if world else O;r,t=camera_pose(Oview[:3,:3],Oview[:3,3],cam)
            _,z,_=object_mesh.raster(K,r,t,wh);paint(im,zbuf,z,(40,170,210) if not reference else (35,130,230),.55 if not world else .97)
        for side,col in [('left',(205,120,60)),('right',(65,180,225))]:
            si=int(np.flatnonzero(g['mano_sides']==side)[0]);valid=g['mano_available'][i,si] if reference else p[side+'_valid'][i]
            if not valid:continue
            V=g['mano_world_vertices_m'][i,si] if reference else p[side+'_vertices'][i];F=g['mano_'+side+'_faces'] if reference else p[side+'_faces']
            for im,zbuf,cam,K,world in [(image,dep,C,k,False),(wi,wd,observer,kv,True)]:
                hand=TriangleMesh(apply(A,V) if world else V,F);r=cam[:3,:3].T;t=-r@cam[:3,3]
                _,z,_=hand.raster(K,r,t,wh);paint(im,zbuf,z,col,.45 if not world else .97)
        head=np.full((44,w,3),24,np.uint8);cv2.putText(head,f'{label} f{i:03d}',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.41,(245,245,245),1)
        cv2.putText(head,'Full native mesh / MANO; scale1',(6,36),cv2.FONT_HERSHEY_SIMPLEX,.33,(220,220,220),1)
        return np.vstack([head,image]),np.vstack([head,wi])
    try:
        for i in range(n):
            ok,im=cap.read()
            if not ok:raise ValueError('source video ended early')
            rgb=cv2.resize(im,wh,interpolation=cv2.INTER_AREA);panels=[panel(i,*state) for state in states]
            panels.append(panel(i,'GT',None,cad_scene,True))
            images=[np.hstack([p[j] for p in panels]) for j in range(2)]
            for j,im in enumerate(images):
                pipes[j].stdin.write(im.tobytes())
                if i in [0,30,60,90,120,n-1]:
                    cv2.imwrite(str(a.output/f'{["overlay","world"][j]}_f{i:03d}.jpg'),im)
                    sheets[j].append(cv2.resize(im,(w*count//2,(w+44)//2)))
            if i%30==0:print('full comparison frame',i,flush=True)
    finally:
        cap.release()
        for p in pipes:p.stdin.close()
        codes=[p.wait() for p in pipes]
    if any(codes):raise ValueError('ffmpeg comparison failed')
    for j in range(2):cv2.imwrite(str(a.output/f'{["overlay","world"][j]}_sheet.jpg'),np.vstack(sheets[j]))
    for path in a.output.glob('*.mp4'):
        c=cv2.VideoCapture(str(path));decoded=0
        while c.read()[0]:decoded+=1
        c.release()
        if decoded!=n:raise ValueError('incomplete video '+str(path))
    for p,digest in hashes.items():
        if sha(p)!=digest:raise ValueError('render mutated input '+p)
    (a.output/'render.json').write_text(json.dumps(dict(frames=n,fps=30,duration_s=n/30,
        columns=[label for label,_,_ in states]+['GT'],observer_world=observer.tolist(),
        world_alignment='one first-camera SE3 per independent state, scale1; no object/per-frame registration',
        actual_native_vertices_and_faces=True,prediction_outputs_unchanged=True,input_hashes=hashes),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['video','gt','cad','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--object-uid',required=True);p.add_argument('--state',nargs=2,action='append',required=True,metavar=('LABEL','ARCHIVE'))
    p.add_argument('--width',type=int,default=320);main(p.parse_args())
