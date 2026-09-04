#!/usr/bin/env python3
"""Evaluate observed-only canonical mesh pose reports against fresh RGB masks.

This is a display/diagnostic metric only.  It uses the exact 960x960 mask plane
and the declared 1408->960->raster pinhole transform; it does not alter poses or
promote generated faces to authority.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


def load_json(p: Path): return json.loads(p.expanduser().resolve().read_text())
def sha256_file(p: Path):
    h=hashlib.sha256()
    with p.expanduser().resolve().open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()
def resize_K(K,src_wh,dst_wh):
    sx=dst_wh[0]/src_wh[0];sy=dst_wh[1]/src_wh[1];Q=np.asarray(K,float).copy();Q[0,0]*=sx;Q[1,1]*=sy;Q[0,2]=sx*(Q[0,2]+.5)-.5;Q[1,2]=sy*(Q[1,2]+.5)-.5;return Q
def metrics(a,b):
    inter=np.count_nonzero(a&b);uni=np.count_nonzero(a|b);ca=np.argwhere(a);cb=np.argwhere(b)
    centroid_delta=float(np.linalg.norm(ca.mean(0)-cb.mean(0))) if len(ca) and len(cb) else None
    return {'intersection':int(inter),'union':int(uni),'iou':float(inter/uni) if uni else None,'render_pixels':int(a.sum()),'mask_pixels':int(b.sum()),'centroid_delta_px':centroid_delta}

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True);p.add_argument('--pose-report',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--object-id',required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--raster-size',type=int,default=256);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--label',default='pose')
    args=p.parse_args();a=load_json(args.annotations);frames={int(f['frame_idx']):f for f in a['frames']};pr=load_json(args.pose_report);rows={int(r['frame_idx']):r for r in pr['pose_rows'] if isinstance(r,dict) and r.get('rotation_world_from_completed_canonical_matrix') is not None}
    mesh=trimesh.load(args.mesh,force='mesh',process=False);v=np.asarray(mesh.vertices,float);faces=torch.tensor(np.asarray(mesh.faces,dtype=np.int64),device=args.device)
    # Use the calibration K from the first valid visible frame, then map it to the actual RGB/mask plane.
    first=next(f for f in a['frames'] if isinstance(f.get('objects'),list) and f['objects'] and isinstance(f['objects'][0].get('visible_geometry_candidate'),dict) and f['objects'][0]['visible_geometry_candidate'].get('intrinsics_fx_fy_cx_cy'))
    vals=np.asarray(first['objects'][0]['visible_geometry_candidate']['intrinsics_fx_fy_cx_cy'],float);Kcal=np.array([[vals[0],0,vals[2]],[0,vals[1],vals[3]],[0,0,1]],float)
    # Actual image plane is checked per frame, but all current frames are 960 square.
    size=args.raster_size; device=args.device
    out=[];all_iou=[];all_cent=[]
    for start in range(0,len(frames),args.batch_size):
        chunk=list(sorted(frames))[start:start+args.batch_size];verts=[];Ks=[];targets=[];valid_ids=[]
        for idx in chunk:
            if idx not in rows: continue
            f=frames[idx];obj=next((o for o in f.get('objects',[]) if o.get('object_id')==args.object_id),None)
            if obj is None:continue
            mask=cv2.imread(str(obj.get('mask_path')),cv2.IMREAD_GRAYSCALE)
            if mask is None:raise RuntimeError(f'missing mask frame {idx}')
            h,w=mask.shape;Kr=resize_K(Kcal,(int(f.get('source_width') or 1408),int(f.get('source_height') or 1408)),(w,h));Kr=resize_K(Kr,(w,h),(size,size))
            T=np.asarray(f['camera']['T_world_camera_metric'],float);R=np.asarray(rows[idx]['rotation_world_from_completed_canonical_matrix'],float);t=np.asarray(rows[idx]['translation_world_m'],float);world=v@R.T+t;cam=(world-T[:3,3])@T[:3,:3];verts.append(cam.astype(np.float32));Ks.append(Kr);targets.append(cv2.resize((mask>0).astype(np.uint8),(size,size),interpolation=cv2.INTER_NEAREST_EXACT)>0);valid_ids.append(idx)
        if not verts:continue
        # cameras can differ only in K; render one at a time to keep the code explicit and avoid a hidden shared-K assumption.
        for cam,Kr,target,idx in zip(verts,Ks,targets,valid_ids):
            camera=cameras_from_opencv_projection(torch.eye(3,device=device,dtype=torch.float32)[None],torch.zeros((1,3),device=device),torch.tensor(Kr,device=device,dtype=torch.float32)[None],torch.tensor([[size,size]],device=device,dtype=torch.float32))
            rast=MeshRasterizer(cameras=camera,raster_settings=RasterizationSettings(image_size=size,blur_radius=0.,faces_per_pixel=1,cull_backfaces=False,bin_size=0))
            with torch.no_grad(): frag=rast(Meshes(verts=[torch.tensor(cam,device=device)],faces=[faces]))
            hit=(frag.pix_to_face[0,:,:,0].cpu().numpy()>=0);m=metrics(hit,target);m['frame_idx']=idx;out.append(m);all_iou.append(m['iou']);all_cent.append(m['centroid_delta_px'])
        print(f'[eval] {min(start+args.batch_size,len(frames))}/{len(frames)}',flush=True)
    vals_i=np.asarray([x for x in all_iou if x is not None]);vals_c=np.asarray([x for x in all_cent if x is not None]);summary=lambda x:{'count':int(len(x)),'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'p95':float(np.percentile(x,95)),'max':float(np.max(x))} if len(x) else {'count':0}
    report={'schema':'v19_observed_mesh_pose_mask_evaluation_v1','label':args.label,'inputs':{'annotations':str(args.annotations.resolve()),'pose_report':str(args.pose_report.resolve()),'mesh':str(args.mesh.resolve()),'object_id':args.object_id,'input_sha256':{'annotations':sha256_file(args.annotations),'pose_report':sha256_file(args.pose_report),'mesh':sha256_file(args.mesh)}},'coordinate_contract':{'calibration_plane_size_wh':[1408,1408],'actual_mask_plane':'read_per_frame','raster_size_wh':[size,size],'pixel_center_convention':'OpenCV half-pixel resize','generated_geometry_consumed':False},'frame_count':len(out),'iou':summary(vals_i),'centroid_delta_px':summary(vals_c),'frames':out}
    args.output_json.parent.mkdir(parents=True,exist_ok=True);args.output_json.write_text(json.dumps(report,indent=2));print(json.dumps({'label':args.label,'frame_count':len(out),'iou':report['iou'],'centroid_delta_px':report['centroid_delta_px']},indent=2))
if __name__=='__main__':main()
