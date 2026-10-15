"""Independent DROID-SLAM + current UniDepth metric scale, never HaWoR state."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from utils.state import sha


def resize_frame(frame, K, area=160000):
    h,w=frame.shape[:2];r=np.sqrt(area/(h*w));hh,ww=int(h*r),int(w*r)
    # Crop only after resizing. K uses the pre-crop resize, not cropped dimensions.
    image=cv2.resize(frame,(ww,hh),interpolation=cv2.INTER_AREA)
    k=np.asarray(K,float).copy();scale=np.array([ww/w,hh/h]);k[:2]*=scale;k[2:]=(k[2:]+.5)*scale-.5
    image=image[:hh-hh%8,:ww-ww%8]
    return image,k


def resize_metric_depth(depth, size):
    # UniDepth archives use float16; OpenCV resize supports float32/64, not FP16.
    return cv2.resize(np.asarray(depth,dtype=np.float32),size,interpolation=cv2.INTER_LINEAR)


def robust_depth_scale(slam_depth, metric_depth, excluded):
    s,m=np.asarray(slam_depth,float),np.asarray(metric_depth,float)
    valid=np.isfinite(s)&np.isfinite(m)&(s>.001)&(m>.05)&(m<20)&~excluded
    ratios=m[valid]/s[valid]
    ratios=ratios[np.isfinite(ratios)&(ratios>0)]
    if len(ratios)<100:
        raise ValueError('too few current background pixels to estimate independent metric scale')
    logs=np.log(ratios);med=np.median(logs);mad=np.median(abs(logs-med))
    keep=abs(logs-med)<=max(.05,3*1.4826*mad)
    return float(np.exp(np.median(logs[keep]))),dict(pixels=int(len(ratios)),retained=int(keep.sum()),
        ratio_p25=float(np.percentile(ratios,25)),ratio_p75=float(np.percentile(ratios,75)))


def run(a):
    import torch
    if not torch.cuda.is_available():
        raise ValueError('DROID requires explicitly placed server CUDA')
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    raw=json.loads(a.manifest.read_text());rows=raw['frames'];ids=np.array([r['frame_idx'] for r in rows],int)
    with np.load(a.depth,allow_pickle=False) as z:
        depth_ids=z['frame_idx'];depth=z['depth'];K=z['intrinsics_fx_fy_cx_cy'] if 'intrinsics_fx_fy_cx_cy' in z else z['intrinsics']
    if K.shape==(4,):K=np.tile(K,(len(ids),1))
    if not np.array_equal(ids,depth_ids) or K.shape!=(len(ids),4):raise ValueError('depth/K/timeline differs')
    wp=a.wilor
    with np.load(wp,allow_pickle=False) as z:hand={k:z[k] for k in z.files}
    if 'T_world_camera' in hand:raise ValueError('WiLoR input must be independently camera-local, not an inherited world track')
    if not np.array_equal(hand['frame_idx'],ids):raise ValueError('hand timeline differs')
    track=json.loads(a.object_track.read_text())
    if isinstance(track, dict) and 'frames' not in track:
        object_rows={int(fid):row for fid,row in track.items()}
    else:
        object_rows={int(r['frame_idx']):r for r in track['frames']}
    if set(object_rows)!=set(map(int,ids)):
        raise ValueError('current SAM2 track must cover the complete camera timeline')
    def stream():
        for i,row in enumerate(rows):
            image=cv2.imread(str(row['raw_frame_path']))
            if image is None:raise FileNotFoundError(row['raw_frame_path'])
            # raw manifest may be960px while depth is source1408px; exact resize K.
            source_wh=np.array([row['source_width'],row['source_height']],float)
            input_wh=np.array([image.shape[1],image.shape[0]],float)
            ki=K[i].copy();ki[:2]*=input_wh/source_wh;ki[2:]=(ki[2:]+.5)*input_wh/source_wh-.5
            image,ki=resize_frame(image,ki,a.area)
            yield int(ids[i]),torch.from_numpy(image).permute(2,0,1)[None],torch.tensor(ki,dtype=torch.float32)
    sys.path.insert(0,str(a.droid_root/'droid_slam'))
    from droid import Droid
    torch.multiprocessing.set_start_method('spawn',force=True)
    first=next(stream());hw=list(first[1].shape[-2:])
    options=SimpleNamespace(weights=str(a.weights),buffer=max(256,len(ids)+64),image_size=hw,disable_vis=True,
        beta=.3,filter_thresh=2.4,warmup=8,keyframe_thresh=4.,frontend_thresh=12.,frontend_window=25,
        frontend_radius=2,frontend_nms=1,backend_thresh=16.,backend_radius=2,backend_nms=3,
        upsample=True,stereo=False,motion_damping=.5)
    droid=Droid(options)
    for i,image,k in stream():
        droid.track(i,image,intrinsics=k)
        if i%20==0:print('DROID frame',i,flush=True)
    trajectory=droid.terminate(stream())
    count=int(droid.video.counter.value)
    keyframes=droid.video.tstamp[:count].cpu().numpy().astype(int)
    disparity=droid.video.disps_up[:count].cpu().numpy()
    if len(trajectory)!=len(ids) or not np.isfinite(trajectory).all():raise ValueError('invalid DROID dense trajectory')
    scales=[];traces=[]
    for key_index,fid in enumerate(keyframes):
        i=int(np.flatnonzero(ids==fid)[0]);disp=disparity[key_index];size=(disp.shape[1],disp.shape[0])
        # Match DROID's actual resize-then-crop plane exactly for depth/masks.
        raw_image=cv2.imread(rows[i]['raw_frame_path']);rh,rw=raw_image.shape[:2];resize=np.sqrt(a.area/(rh*rw))
        pre_size=(int(rw*resize),int(rh*resize))
        metric=resize_metric_depth(depth[i],pre_size)[:size[1],:size[0]]
        excluded=np.zeros(disp.shape,bool)
        for side in ['left','right']:
            if not hand[side+'_valid'][i]:continue
            box=hand[side+'_bbox_xyxy'][i].astype(float)
            box[[0,2]]=(box[[0,2]]+.5)*pre_size[0]/hand['image_size_wh'][0]-.5
            box[[1,3]]=(box[[1,3]]+.5)*pre_size[1]/hand['image_size_wh'][1]-.5
            x0,y0=np.maximum(np.floor(box[:2]-5),0).astype(int);x1,y1=np.minimum(np.ceil(box[2:]+5),size).astype(int)
            excluded[y0:y1,x0:x1]=True
        row=object_rows[int(fid)]
        if row.get('mask_path'):
            mask=cv2.imread(row['mask_path'],cv2.IMREAD_GRAYSCALE)
            if mask is None:raise FileNotFoundError(row['mask_path'])
            excluded|=cv2.resize(mask,pre_size,interpolation=cv2.INTER_NEAREST)[:size[1],:size[0]]>0
        excluded=cv2.dilate(excluded.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        scale,diag=robust_depth_scale(np.divide(1.,disp,out=np.full_like(disp,np.nan),where=disp>0),metric,excluded)
        scales.append(scale);traces.append(dict(frame_idx=int(fid),scale=scale,**diag))
    metric_scale=float(np.exp(np.median(np.log(scales))))
    T=np.tile(np.eye(4),(len(ids),1,1));T[:,:3,:3]=Rotation.from_quat(trajectory[:,3:7]).as_matrix();T[:,:3,3]=trajectory[:,:3]*metric_scale
    # Fresh first-camera world gauge; never alignment to an old prediction.
    T=np.linalg.inv(T[0])@T
    np.savez_compressed(a.output/'droid_camera.npz',frame_idx=ids,T_world_camera=T,R_c2w=T[:,:3,:3],t_c2w=T[:,:3,3],
        intrinsics=K,intrinsics_source=K[0],metric_scale=np.array(metric_scale),keyframe_idx=keyframes,keyframe_disparity=disparity)
    report=dict(schema='ego.annotation.independent_droid_unidepth_camera.v1',source='standalone_DROID_SLAM',
        frames=len(ids),metric_scale=metric_scale,scale_estimates=traces,elapsed_s=time.monotonic()-started,
        HaWoR_inputs=False,hand_geometry_used_for_scale=False,
        scale_source='single robust ratio of current UniDepth camera-Z and DROID background inverse-depth, current object/hand exclusions',
        limitations=['monocular depth and SLAM scale remain uncertain','DROID visual motion is unmasked; moving object/hand features can affect camera'],
        input_hashes={str(p):sha(p) for p in [a.manifest,a.depth,a.wilor,a.object_track,a.weights]},
        output_sha256=sha(a.output/'droid_camera.npz'))
    (a.output/'droid_camera.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['manifest','depth','wilor','object-track','droid-root','weights','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--area',type=int,default=160000)
    run(p.parse_args())
