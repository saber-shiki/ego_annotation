"""Full-timeline native triangle/MANO before-after and fixed-world review videos."""
from __future__ import annotations
from pathlib import Path
import subprocess,json
import cv2
import numpy as np
from preprocessing.geometry import TriangleMesh,camera_pose,resized_intrinsics


def paint(image,depth,z,color,alpha=.65):
    visible=np.isfinite(z)&(z<depth)&(z>.001)
    if visible.any():image[visible]=(image[visible]*(1-alpha)+np.array(color)*alpha).astype('uint8');depth[visible]=z[visible]


def look_at(eye,target,up):
    z=target-eye;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    C=np.eye(4);C[:3,:3]=np.column_stack((x,y,z));C[:3,3]=eye;return C


def render_scene(inputs,archive,output,width=320):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    mesh=TriangleMesh(archive['vertices_canonical'],archive['faces']);before=TriangleMesh(inputs.vertices,inputs.faces)
    calibrated_comparison='alignment_initial_rotation' in archive
    before_R=archive['alignment_initial_rotation'] if calibrated_comparison else inputs.source_R
    before_t=archive['alignment_initial_translation_world'] if calibrated_comparison else inputs.source_t
    wh=(width,round(width*inputs.wh[1]/inputs.wh[0]));fps=1/float(np.median(np.diff(inputs.times)))
    overlay_path=output/'before_after_overlay.mp4';world_path=output/'coarse_world.mp4'
    pipes=[]
    for path,size in [(overlay_path,(2*wh[0],wh[1]+40)),(world_path,(width,width+40))]:
        command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(path)]
        pipes.append(subprocess.Popen(command,stdin=subprocess.PIPE))
    center=np.median(archive['object_translation_world'],axis=0);C0=inputs.cameras[0]
    radius=max(.2,float(np.max(np.linalg.norm(archive['object_translation_world']-center,axis=1)))+.12)
    eye=center+C0[:3,:3]@np.array([radius*.7,-radius*.25,-radius*2.5]);view=look_at(eye,center,C0[:3,:3]@np.array([0.,-1.,0.]));kv=np.array([width*1.05,width*1.05,(width-1)/2,(width-1)/2])
    try:
        for i,fid in enumerate(inputs.ids):
            rgb=cv2.imread(inputs.raw_paths[i])
            if rgb is None:raise FileNotFoundError(inputs.raw_paths[i])
            rgb=cv2.resize(rgb,wh,interpolation=cv2.INTER_AREA);k=resized_intrinsics(inputs.K[i],inputs.wh,wh);C=inputs.cameras[i]
            panels=[];world=np.full((width,width,3),242,np.uint8);wd=np.full((width,width),np.inf)
            rc,tc=camera_pose(archive['object_rotation'][i],archive['object_translation_world'][i],view);_,z,_=mesh.raster(kv,rc,tc,(width,width));paint(world,wd,z,(90,180,245),.95)
            hand_scenes=[]
            for side,col in [('left',(225,150,75)),('right',(95,185,225))]:
                hand=inputs.hands[side]
                if not hand['valid'][i]:continue
                faces=hand['faces'][i] if hand['faces'].ndim==3 else hand['faces'];hs=TriangleMesh(hand['vertices'][i],faces);hand_scenes.append((hs,col))
                _,hz,_=hs.raster(kv,view[:3,:3].T,-view[:3,:3].T@view[:3,3],(width,width));paint(world,wd,hz,col,.95)
            for is_after in (False,True):
                im=rgb.copy();dep=np.full(wh[::-1],np.inf)
                available=True if is_after or calibrated_comparison else inputs.source_valid[i]
                if available:
                    R=archive['object_rotation'][i] if is_after else before_R[i];t=archive['object_translation_world'][i] if is_after else before_t[i];m=mesh if is_after else before
                    r,tt=camera_pose(R,t,C);_,z,_=m.raster(k,r,tt,wh);paint(im,dep,z,(240,160,40),.55)
                for hs,col in hand_scenes:
                    _,hz,_=hs.raster(k,C[:3,:3].T,-C[:3,:3].T@C[:3,3],wh);paint(im,dep,hz,col,.45)
                header=np.zeros((40,width,3),np.uint8);label=('aligned: fixed UniDepth' if calibrated_comparison else 'coarse: RGB-only derived') if is_after else ('before: RGB pose proposal' if calibrated_comparison else 'source: no RGB rotation')
                cv2.putText(header,f'f{fid} {label}',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                cv2.putText(header,'source metric scale; no signed-contact claim',(5,33),cv2.FONT_HERSHEY_SIMPLEX,.32,(220,220,220),1)
                panels.append(np.vstack((header,im)))
            overlay=np.concatenate(panels,axis=1);pipes[0].stdin.write(overlay.tobytes())
            header=np.zeros((40,width,3),np.uint8);cv2.putText(header,f'f{fid} fixed-world mesh + source MANO',(5,23),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1);pipes[1].stdin.write(np.vstack((header,world)).tobytes())
            if i in sorted(set([0,len(inputs.ids)//3,2*len(inputs.ids)//3,len(inputs.ids)-1,inputs.anchor])):cv2.imwrite(str(output/f'review_{int(fid):06d}.jpg'),overlay)
    finally:
        for pipe in pipes:
            if pipe.stdin:pipe.stdin.close()
        codes=[p.wait() for p in pipes]
    if any(codes):raise RuntimeError('video encoding failed: '+str(codes))
    result={'frames':len(inputs.ids),'fps':fps,'duration_s':len(inputs.ids)/fps,'overlay':str(overlay_path),'world':str(world_path),'mesh_faces':len(archive['faces']),'MANO_unchanged':True,'physical_contact_claim':False}
    (output/'render.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
