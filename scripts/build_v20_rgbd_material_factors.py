#!/usr/bin/env python3
"""Pack cached RGB material correspondences and prediction-side endpoint depth.

No inference or optimization. Source observations are immutable world points;
missing/invalid P09 at the target permits image-only, never fabricated depth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import fit_v20_late_window_prediction_only_se3 as late
from v20_prediction_contracts import assert_prediction_only
from v20_rgbd_material_factors import valid_camera


def sample_depth_members(path, queries, frame_ids, size):
    """Stream NPZ arrays one frame at a time; memory independent of clip length."""
    results={}
    with zipfile.ZipFile(path) as archive:
        for name in ('depth','confidence'):
            with archive.open(name+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=reader(stream)
                if fortran or shape!=(len(frame_ids),int(size[1]),int(size[0])) or dtype.kind!='f':
                    raise ValueError(f'invalid {name} raster contract')
                byte_count=int(np.prod(shape[1:]))*dtype.itemsize
                for frame_id in frame_ids:
                    blob=stream.read(byte_count)
                    if len(blob)!=byte_count:raise ValueError('truncated prediction depth archive')
                    if int(frame_id) not in queries:continue
                    image=np.frombuffer(blob,dtype).reshape(shape[1:])
                    for tag,pixels in queries[int(frame_id)]:results[tag,name]=image[pixels[:,1],pixels[:,0]].astype(float)
    return results


def valid_source_depth_mask(depth, cached_z, confidence):
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(cached_z) & (cached_z > 0)
    if np.any(valid & (np.abs(depth - cached_z) > 1e-5)):
        raise ValueError('recovered source observation does not match hash-bound source depth')
    return valid & np.isfinite(confidence) & (confidence >= 0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--rgb-edge-npz',type=Path,required=True)
    parser.add_argument('--rgb-edge-json',type=Path,required=True)
    parser.add_argument('--depth-npz',type=Path,required=True)
    parser.add_argument('--object-id',required=True)
    parser.add_argument('--output-npz',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--frame-start',type=int,default=100)
    parser.add_argument('--frame-end',type=int,default=149)
    parser.add_argument('--min-mask-interior-px',type=float,default=3.)
    parser.add_argument('--max-p09-support-px',type=float,default=12.)
    parser.add_argument('--max-p09-depth-disagreement-m',type=float,default=.015)
    args=parser.parse_args()
    if args.output_npz.exists() or args.output_json.exists():raise ValueError('refusing to overwrite material evidence')
    ann=late.load_json(args.annotations);ed=late.load_json(args.rgb_edge_json)
    assert_prediction_only(ann,label='material annotations',object_id=args.object_id)
    assert_prediction_only(ed,label='material RGB report',object_id=args.object_id)
    ah=late.sha256_file(args.annotations);dh=late.sha256_file(args.depth_npz)
    with np.load(args.rgb_edge_npz,allow_pickle=False) as z:
        source_world,meta=late.fixed_rgb_source_world(z)
        arrays={k:np.asarray(z[k]).copy() for k in ['source_frame_idx','target_frame_idx','accepted','reprojection_evidence_offsets','reprojection_target_uv','reprojection_target_intrinsics','reprojection_target_camera','reprojection_weights']}
    for m in (ed,meta):
        if m.get('object_id')!=args.object_id or m.get('input_sha256',{}).get('annotations')!=ah or m.get('input_sha256',{}).get('depth_npz')!=dh:
            raise ValueError('RGB observations are bound to a different annotation/object/depth source')
    if ed.get('input_sha256')!=meta.get('input_sha256'):raise ValueError('RGB NPZ/JSON provenance mismatch')
    with np.load(args.depth_npz,allow_pickle=False) as z:
        ids=np.asarray(z['frame_idx'],int);size=np.asarray(z['source_size'],int);intr=np.asarray(z['intrinsics_fx_fy_cx_cy'],float)
        quantity=str(np.asarray(z['depth_output_quantity']).item())
        if quantity!='camera_z_m_from_metric_radius_on_exact_contract_rays':raise ValueError('unsupported depth quantity; require explicit camera-Z contract')
        if bool(np.asarray(z['metadata_only_ray_relabel_override']).item()):raise ValueError('metadata-only camera ray relabel prohibited')
    if size.shape!=(2,) or intr.shape!=(len(ids),4) or len(set(ids))!=len(ids) or not np.isfinite(intr).all() or np.any(intr[:,:2]<=0):
        raise ValueError('depth camera identity invalid')
    frames={int(f['frame_idx']):f for f in ann['frames']};pos={int(i):p for p,i in enumerate(ids)}
    fdata={};mask_hashes={}
    for idx,f in frames.items():
        if not args.frame_start<=idx<=args.frame_end or idx not in pos:continue
        obj=late._object_for_frame(f,args.object_id)
        if obj is None:continue
        points,reason,validation=late.p09_geometry_validation(obj,f)
        T=late._frame_camera_transform(f);K=intr[pos[idx]]
        if T is None or not valid_camera(T):raise ValueError('invalid annotation camera')
        if not np.allclose(np.asarray(f['camera']['intrinsics_fx_fy_cx_cy']),K,atol=1e-5,rtol=0):raise ValueError('annotation/depth intrinsics mismatch')
        maskpath=Path(str(obj.get('mask_path') or ''));mask=cv2.imread(str(maskpath),cv2.IMREAD_GRAYSCALE)
        if mask is None:raise ValueError('material object-owned mask missing')
        mask_hashes[str(idx)]=late.sha256_file(maskpath)
        wh=np.array([mask.shape[1],mask.shape[0]],float)
        tree=p09z=None
        if points is not None:
            pc=late.world_to_camera(points,T);p09uv=(pc[:,:2]/pc[:,2,None]*K[:2]+K[2:]+.5)*(wh/size)-.5
            tree=cKDTree(p09uv);p09z=pc[:,2]
        fdata[idx]={'T':T,'K':K,'wh':wh,'distance':cv2.distanceTransform((mask>0).astype('uint8'),cv2.DIST_L2,5),
                    'metric':points is not None,'tree':tree,'p09z':p09z,'reason':reason,'p09_validation':validation}
    src=arrays['source_frame_idx'];dst=arrays['target_frame_idx'];off=arrays['reprojection_evidence_offsets'];n=len(src)
    if off.shape!=(n+1,) or off[0]!=0 or off[-1]!=len(source_world) or np.any(np.diff(off)<0):raise ValueError('RGB point offsets invalid')
    if arrays['reprojection_target_uv'].shape!=(len(source_world),2) or arrays['reprojection_weights'].shape!=(len(source_world),):raise ValueError('RGB UV/weights length mismatch')
    quality={(int(r['source_frame_idx']),int(r['target_frame_idx'])):r for r in ed['rows']}
    queries={};pairs=[];rejected=[]
    for i,(s,t) in enumerate(zip(src,dst)):
        s,t=int(s),int(t);l,r=map(int,off[i:i+2]);key=(s,t)
        if s not in fdata or t not in fdata:continue
        if (quality.get(key,{}).get('status')=='accepted')!=bool(arrays['accepted'][i]):raise ValueError('RGB acceptance NPZ/JSON mismatch')
        if not arrays['accepted'][i] or r-l<4:continue
        a,b=fdata[s],fdata[t]
        if not a['metric']:
            rejected.append({'source':s,'target':t,'reason':'source_metric_unavailable','source_observation_reason':a['reason']});continue
        W=source_world[l:r];pc=late.world_to_camera(W,a['T']);uvs=(pc[:,:2]/pc[:,2,None]*a['K'][:2]+a['K'][2:]+.5)*(a['wh']/size)-.5
        uvt=arrays['reprojection_target_uv'][l:r]
        Kt=np.r_[b['K'][:2]*(b['wh']/size),(b['K'][2:]+.5)*(b['wh']/size)-.5]
        if not np.allclose(Kt,arrays['reprojection_target_intrinsics'][i],atol=1e-5,rtol=0) or not np.allclose(b['T'],arrays['reprojection_target_camera'][i],atol=1e-6,rtol=0):
            raise ValueError(f'target RGB camera mismatch at {s}->{t}')
        keep=np.isfinite(W).all(1)&np.isfinite(uvt).all(1)&(pc[:,2]>0)
        for uv,fd in [(uvs,a),(uvt,b)]:
            keep&=(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<fd['wh'][0]-1)&(uv[:,1]<fd['wh'][1]-1)
            pixels=np.rint(np.clip(uv,[0,0],fd['wh']-1)).astype(int)
            keep&=fd['distance'][pixels[:,1],pixels[:,0]]>=args.min_mask_interior_px
        uvs,uvt,W,pc=uvs[keep],uvt[keep],W[keep],pc[keep];w=arrays['reprojection_weights'][l:r][keep]
        if len(W)<4:continue
        pixels_s=np.rint((uvs+.5)*(size/a['wh'])-.5).astype(int);pixels_t=np.rint((uvt+.5)*(size/b['wh'])-.5).astype(int)
        for endpoint,frame,pixels in [('s',s,pixels_s),('t',t,pixels_t)]:
            if np.any(pixels<0) or np.any(pixels[:,0]>=size[0]) or np.any(pixels[:,1]>=size[1]):raise ValueError('UV/depth pixel mismatch')
            queries.setdefault(frame,[]).append(((i,endpoint),pixels))
        pairs.append({'i':i,'s':s,'t':t,'W':W,'source_z':pc[:,2],'uvs':uvs,'uvt':uvt,'weights':w,'Kt':Kt})
    samples=sample_depth_members(args.depth_npz,queries,ids,size)
    out={k:[] for k in ['source_frame_idx','target_frame_idx','source_world','target_world','target_uv','weights','metric_valid','target_camera','target_intrinsics','edge_weight','source_metric_eligible','target_metric_eligible']};offsets=[0];rows=[]
    for pair in pairs:
        i,s,t=pair['i'],pair['s'],pair['t'];a,b=fdata[s],fdata[t];zs=samples[(i,'s'),'depth'];zt=samples[(i,'t'),'depth'];cs=samples[(i,'s'),'confidence'];ct=samples[(i,'t'),'confidence']
        source_depth_valid=valid_source_depth_mask(zs,pair['source_z'],cs)
        distance,nearest=a['tree'].query(pair['uvs']);keep=source_depth_valid&(distance<=args.max_p09_support_px)&(np.abs(zs-a['p09z'][nearest])<=args.max_p09_depth_disagreement_m)
        valid=np.zeros(len(zt),bool)
        if b['metric']:
            distance,nearest=b['tree'].query(pair['uvt']);valid=(distance<=args.max_p09_support_px)&(np.abs(zt-b['p09z'][nearest])<=args.max_p09_depth_disagreement_m)&np.isfinite(zt)&(zt>0)&np.isfinite(ct)&(ct>=0)
        W=pair['W'][keep];uv=pair['uvt'][keep];valid=valid[keep];zt=zt[keep]
        if len(W)<4:continue
        target_camera=np.column_stack(((uv-pair['Kt'][2:])/pair['Kt'][:2]*np.where(valid,zt,0)[:,None],np.where(valid,zt,0)))
        target_world=late.camera_to_world(target_camera,b['T']);target_world[~valid]=0
        # exp(logconfidence) predicts error: larger => smaller weight.
        dw=1/(1+np.maximum(cs[keep],0));dw=np.minimum(dw,np.where(valid,1/(1+np.maximum(np.nan_to_num(ct[keep],nan=1e6),0)),1))
        weights=np.clip(pair['weights'][keep]*dw,.01,1.)
        qr=quality[s,t];ew=float(np.clip(float(qr.get('inlier_fraction',0))*np.exp(-float(qr.get('reprojection_median_px',np.inf))/2),.01,1.))
        vals={'source_frame_idx':s,'target_frame_idx':t,'source_world':W,'target_world':target_world,'target_uv':uv,'weights':weights,'metric_valid':valid,'target_camera':b['T'],'target_intrinsics':pair['Kt'],'edge_weight':ew,'source_metric_eligible':True,'target_metric_eligible':b['metric']}
        for k,v in vals.items():out[k].append(v)
        offsets.append(offsets[-1]+len(W));eig=np.linalg.eigvalsh(np.cov(W.T));rank_score=float(max(0,eig[1])/max(eig[2],1e-12))
        rows.append({'source':s,'target':t,'image_points':len(W),'metric_points':int(valid.sum()),'noncollinearity_ratio':rank_score,'pnp_planarity_ratio':qr.get('source_3d_conditioning'),'edge_weight':ew,'target_metric_reason':b['reason']})
    if not rows:raise ValueError('no material observations packed')
    sha={'annotations':ah,'depth_npz':dh,'rgb_edge_npz':late.sha256_file(args.rgb_edge_npz),'rgb_edge_json':late.sha256_file(args.rgb_edge_json)}
    metadata={'schema':'v20_fixed_world_material_rgbd_v1','object_id':args.object_id,'gt_consumed':False,'diagnostic_only':True,'annotation_ready':False,'source_frame':'fixed_world_observation','input_sha256':sha,
              'inputs':{k:str(getattr(args,k).resolve()) for k in ['annotations','depth_npz','rgb_edge_npz','rgb_edge_json']},'mask_sha256':mask_hashes,
              'depth_quantity':quantity,'confidence_weight':'exp(logconfidence) is predicted error; weight proportional to 1/(1+error)',
              'source_producer_metadata':meta,'missing_target_depth':'image-only; never synthesize metric observation','generated_geometry_consumed':False}
    packed={}
    for k,v in out.items():packed[k]=np.concatenate(v,axis=0) if k in ['source_world','target_world','target_uv','weights','metric_valid'] else np.asarray(v)
    packed['offsets']=np.asarray(offsets,np.int64);packed['metadata']=np.asarray([json.dumps(metadata)])
    args.output_npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output_npz,**packed)
    report={**metadata,'parameters':{k:v for k,v in vars(args).items() if not isinstance(v,Path)},'rows':rows,'rejected':rejected,
            'bundle_sha256':late.sha256_file(args.output_npz),'summary':{'pairs':len(rows),'image_points':offsets[-1],'metric_points':int(packed['metric_valid'].sum())}}
    args.output_json.parent.mkdir(parents=True,exist_ok=True);args.output_json.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
