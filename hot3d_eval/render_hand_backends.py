"""Post-freeze full-timeline actual MANO surface comparison; no prediction writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from .compare_hand_backends import load, read, apply
from preprocessing.geometry import TriangleMesh, resized_intrinsics
from preprocessing.render import paint, look_at


def main(a):
    cv2.setNumThreads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    if not a.candidate.with_suffix('.completed.json').is_file():
        raise ValueError('candidate not frozen')
    source = read(a.source_bundle)
    root = a.source_bundle.parent
    h = load(root / source['sources']['mano_archive']['path'])
    w, g, evaluation = load(a.candidate), load(a.gt_cache), read(a.evaluation / 'metrics.json')
    A, C, K = np.array(evaluation['A']), w['T_world_camera'], w['intrinsics']
    n, width = len(w['frame_idx']), a.width
    wh = (width, width)
    models = []
    for label in ['HaWoR', 'WiLoR', 'GT']:
        data = {}
        for side in ['left','right']:
            si = int(np.flatnonzero(g['mano_sides'] == side)[0])
            if label == 'HaWoR':
                camera = apply(np.linalg.inv(C), h[side+'_vertices_world_m'])
                world = apply(A, h[side+'_vertices_world_m'])
                valid, faces = h[side+'_valid'].astype(bool), h[side+'_faces']
            elif label == 'WiLoR':
                camera = w[side+'_vertices_camera_m']
                world = apply(A @ C, camera)
                valid, faces = w[side+'_valid'].astype(bool), w[side+'_faces']
            else:
                world = g['mano_world_vertices_m'][:,si]
                camera = apply(g['T_camera_from_world'], world)
                valid, faces = g['mano_available'][:,si].astype(bool), g['mano_'+side+'_faces']
            data[side] = dict(camera=camera, world=world, valid=valid, faces=faces)
        models.append((label,data))
    samples = np.concatenate([d[s]['world'][d[s]['valid']][::10].reshape(-1,3) for _,d in models for s in ['left','right']])
    lo, hi = samples.min(0), samples.max(0)
    center, radius = (lo+hi)/2, max(.15, np.linalg.norm(hi-lo)/2)
    D0 = g['T_world_from_camera'][0]
    view = look_at(center + D0[:3,:3] @ np.array([.7*radius,-.5*radius,-3.1*radius]), center, D0[:3,:3] @ np.array([0.,-1.,0.]))
    kv = np.array([width*1.4,width*1.4,(width-1)/2,(width-1)/2])
    pipes = []
    for name in ['mano_overlay.mp4','mano_fixed_world.mp4']:
        cmd = ['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width*3}x{width+44}',
               '-r','30','-i','-','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/name)]
        pipes.append(subprocess.Popen(cmd,stdin=subprocess.PIPE))
    cap = cv2.VideoCapture(str(root/source['identity']['source_video']['path']))
    sheets = [[],[]]
    try:
        for i in range(n):
            ok, rgb = cap.read()
            if not ok:
                raise ValueError('source video ended early')
            rgb = cv2.resize(rgb,wh,interpolation=cv2.INTER_AREA)
            k = resized_intrinsics(K[i], w['image_size_wh'], wh)
            overlays, worlds = [], []
            for label, data in models:
                image, world_image = rgb.copy(), np.full((width,width,3),242,np.uint8)
                depth, wd = np.full((width,width),np.inf), np.full((width,width),np.inf)
                for side,col in [('left',(210,125,65)),('right',(65,180,230))]:
                    d = data[side]
                    if not d['valid'][i]:
                        continue
                    mc = TriangleMesh(d['camera'][i], d['faces'])
                    _, z, _ = mc.raster(k,np.eye(3),np.zeros(3),wh)
                    paint(image,depth,z,col,.5)
                    mw = TriangleMesh(d['world'][i],d['faces'])
                    _,z,_ = mw.raster(kv,view[:3,:3].T,-view[:3,:3].T@view[:3,3],wh)
                    paint(world_image,wd,z,col,.95)
                header = np.full((44,width,3),25,np.uint8)
                cv2.putText(header,f'{label}  frame {i:03d}',(7,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1)
                cv2.putText(header,'Actual MANO surface; no contact claim',(7,36),cv2.FONT_HERSHEY_SIMPLEX,.32,(220,220,220),1)
                overlays.append(np.vstack([header,image]))
                worlds.append(np.vstack([header,world_image]))
            images = [np.hstack(overlays),np.hstack(worlds)]
            for j, image in enumerate(images):
                pipes[j].stdin.write(image.tobytes())
                if i in [0,30,60,90,120,n-1]:
                    name = ['overlay','world'][j]
                    cv2.imwrite(str(a.output/f'{name}_f{i:03d}.jpg'),image)
                    sheets[j].append(cv2.resize(image,(width*3//2,(width+44)//2)))
            if i%30 == 0:
                print('render',i,flush=True)
    finally:
        cap.release()
        for p in pipes:
            p.stdin.close()
        codes = [p.wait() for p in pipes]
    if any(codes):
        raise ValueError(f'ffmpeg failed {codes}')
    for j, name in enumerate(['overlay_sheet.jpg','world_sheet.jpg']):
        cv2.imwrite(str(a.output/name), np.vstack(sheets[j]))
    for path in a.output.glob('*.mp4'):
        c = cv2.VideoCapture(str(path)); count = 0
        while c.read()[0]:
            count += 1
        c.release()
        if count != n:
            raise ValueError(f'incomplete video {path} {count}/{n}')
    (a.output/'render.json').write_text(json.dumps(dict(frames=n,fps=30,duration_s=n/30,
        native_MANO_vertices=778,world_observer=view.tolist(),same_world_observer_all_backends=True,
        world_gauge='one fixed first-camera SE3; scale1',
        overlay='actual projected MANO silhouettes, not RGB-occlusion-correct rendering',
        prediction_outputs_unchanged=True),indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['source-bundle','candidate','gt-cache','evaluation','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--width',type=int,default=384)
    main(p.parse_args())
