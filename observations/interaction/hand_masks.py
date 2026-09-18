"""Independent direct-frame RGB/SAM2 hand-mask producer and receipt validation.

HaWoR detector boxes are conditioning prompts only.  Every prediction is made
from the current decoded RGB frame through the official SAM2 image predictor;
boxes, MANO renders, joints, and earlier frames never become mask evidence.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable
import cv2
import numpy as np

HAND_VISIBILITY_STATES=frozenset({'visible','partially_visible','occluded','out_of_frame','unresolved'})
SIDES=('left','right')
ACTIVE_MASK_STATES=frozenset({'visible','partially_visible'})
UNCERTAINTY_FORMULA='clip(0.25 + 0.75 * raw_overlap_fraction_of_smaller, 0.25, 1.0)'
OVERLAP_TIE_POLICY='left_lower_sam2_object_id_wins_exact_logit_ties'
OVERLAP_OCCLUDER='other_hand_overlap_sam2_logit_partition_uncertain'


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def _digest(value: Any,label: str) -> str:
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value.lower()):
        raise ValueError(f'{label} must be a SHA256 digest')
    return value.lower()


def _record(path: Path,expected: str,label: str) -> Path:
    path=path.resolve()
    if not path.is_file():raise ValueError(f'{label} is missing: {path}')
    if _sha(path)!=_digest(expected,label+' hash'):raise ValueError(f'{label} hash mismatch')
    return path


def _rows(doc: dict) -> dict[int,dict]:
    rows=doc.get('frames')
    if not isinstance(rows,list) or not rows:raise ValueError('source RGB manifest has no frames')
    result={}
    for row in rows:
        i=int(row.get('frame_idx',row.get('index',-1)))
        if i<0 or i in result:raise ValueError('source RGB manifest frame IDs must be unique')
        result[i]=row
    return result


def _raw_path(row: dict) -> Path:
    value=row.get('raw_frame_path') or row.get('rgb')
    if not value:raise ValueError('source RGB row lacks image path')
    return Path(value).resolve()


def _validated_hawor_source_contract(raw: dict, arrays: dict[str,np.ndarray], source_size_wh: tuple[int,int]) -> dict:
    """Validate the detector archive against the manifest's source video and full-K plane."""
    from observations.common import validate_hawor_camera_image_plane_contract
    from observations.hand.hawor import FULL_K_RECTIFIED_MODE
    input_video=raw.get('input_video')
    if not input_video:raise ValueError('source RGB manifest lacks input video path')
    video=Path(input_video).resolve()
    expected_hash=_digest(raw.get('input_video_sha256'),'source video hash')
    if not video.is_file() or _sha(video)!=expected_hash:
        raise ValueError('source RGB manifest video identity mismatch')
    source_rows=_rows(raw)
    first=next(iter(source_rows.values()))
    manifest_size=(int(first.get('source_width',0)),int(first.get('source_height',0)))
    if manifest_size!=tuple(map(int,source_size_wh)) or any(
            (int(row.get('source_width',0)),int(row.get('source_height',0)))!=manifest_size
            for row in source_rows.values()):
        raise ValueError('source RGB manifest dimensions differ from HaWoR source plane')
    mode_values=np.asarray(arrays.get('camera_intrinsics_contract_mode',[])).reshape(-1)
    if len(mode_values)!=1 or str(mode_values[0])!=FULL_K_RECTIFIED_MODE:
        raise ValueError('HaWoR detector archive requires the validated full-K source-plane contract')
    intrinsics=np.asarray(arrays.get('camera_intrinsics_fx_fy_cx_cy',[]),np.float64)
    frame_count=len(np.asarray(arrays.get('frame_idx',[])))
    if intrinsics.shape!=(frame_count,4) or not np.isfinite(intrinsics).all() or not np.allclose(intrinsics,intrinsics[0],atol=1e-6,rtol=0):
        raise ValueError('HaWoR detector archive source intrinsics are invalid or not fixed')
    video_values=np.asarray(arrays.get('video_path',[])).reshape(-1)
    if len(video_values)!=1:raise ValueError('HaWoR detector archive lacks one source video path')
    try:
        contract=validate_hawor_camera_image_plane_contract(arrays,source_intrinsics=intrinsics[0],
            source_size_wh=source_size_wh,source_video=Path(str(video_values[0])).resolve(),expected_source_video=video)
    except (RuntimeError,ValueError) as exc:
        raise ValueError(f'HaWoR detector source/camera contract is invalid: {exc}') from exc
    if contract.get('validated') is not True:
        raise ValueError('HaWoR detector source/camera contract was not validated')
    return contract


def half_pixel_affine(source_wh: tuple[int,int],target_wh: tuple[int,int]) -> np.ndarray:
    sw,sh=map(float,source_wh);tw,th=map(float,target_wh)
    if min(sw,sh,tw,th)<=0:raise ValueError('image planes must be positive')
    sx,sy=tw/sw,th/sh
    return np.asarray([[sx,0.,sx*.5-.5],[0.,sy,sy*.5-.5],[0.,0.,1.]],np.float64)


def transform_points(points: np.ndarray,affine: np.ndarray) -> np.ndarray:
    p=np.asarray(points,np.float64);A=np.asarray(affine,np.float64)
    q=np.c_[p,np.ones(len(p))]@A.T
    if np.any(np.abs(q[:,2])<1e-12):raise ValueError('point transform reached infinity')
    return (q[:,:2]/q[:,2,None]).astype(np.float32)


def _support_inside_box(mask: np.ndarray,box_xyxy: Iterable[float]) -> int:
    """Count positive integer-centered pixels inside a continuous xyxy box."""
    x0,y0,x1,y1=map(float,box_xyxy);ys,xs=np.nonzero(mask)
    return int(np.count_nonzero((xs>=x0)&(xs<=x1)&(ys>=y0)&(ys<=y1)))


def _uncertainty(overlap_fraction: float) -> float:
    return float(np.clip(.25+.75*float(overlap_fraction),.25,1.))


def validate_hand_mask_receipt(manifest: str|Path, *, frame_ids: Iterable[int]|None=None,
        sides: Iterable[str]=SIDES, source_size_wh: tuple[int,int]|None=None,
        source_frame_manifest: str|Path) -> dict:
    """Bind every mask row to the authoritative frame/model/side/image plane."""
    path=Path(manifest).resolve();doc=json.loads(path.read_text())
    if doc.get('schema')!='rgb_sam2_hand_masks_v1' or doc.get('producer')!='sam2_rgb_hand_track':
        raise ValueError('hand masks must be an independent SAM2 RGB receipt')
    source_path=Path(source_frame_manifest).resolve()
    if doc.get('source_rgb_manifest_path')!=str(source_path) or _sha(source_path)!=_digest(doc.get('source_rgb_manifest_sha256'),'source RGB manifest hash'):
        raise ValueError('hand mask source RGB manifest identity mismatch')
    source_doc=json.loads(source_path.read_text());source_rows=_rows(source_doc)
    source_video_hash=_digest(source_doc.get('input_video_sha256'),'source video hash')
    if _digest(doc.get('source_video_sha256'),'source video hash')!=source_video_hash:
        raise ValueError('hand mask source video hash mismatch')
    for key in ('checkpoint','model_config','hawor_detector_npz','prompt_payload'):
        ref=doc.get(key)
        if not isinstance(ref,dict) or not {'path','sha256'} <= set(ref):raise ValueError(f'hand mask receipt lacks {key} reference')
        _record(Path(ref['path']),ref['sha256'],key)
    prompt_doc=json.loads(Path(doc['prompt_payload']['path']).read_text())
    regenerated=make_sam2_hand_prompt_payload(raw_frame_manifest=source_path,hawor_npz=doc['hawor_detector_npz']['path'])
    if regenerated!=prompt_doc:raise ValueError('SAM2 prompt payload differs from current detector/source contract')
    prompt_rows={(int(r['frame_idx']),str(r['hand_side'])):r for r in prompt_doc['tracks']}
    if len(prompt_rows)!=len(prompt_doc['tracks']):raise ValueError('duplicate SAM2 prompt frame/side')
    if prompt_doc['source_rgb_manifest_sha256']!=doc['source_rgb_manifest_sha256'] or prompt_doc['hawor_detector_npz']!=doc['hawor_detector_npz']:
        raise ValueError('SAM2 prompt source identity mismatch')
    plane=doc.get('hawor_detector_plane_contract')
    if not isinstance(plane,dict) or plane.get('box_output_plane')!='source_rgb' or plane.get('camera_intrinsics_contract_mode') is None:
        raise ValueError('hand mask receipt lacks HaWoR detector plane contract')
    declared=tuple(map(int,doc.get('source_size_wh') or []))
    if len(declared)!=2 or min(declared)<=0 or (source_size_wh is not None and declared!=tuple(map(int,source_size_wh))):
        raise ValueError('hand mask source RGB plane mismatch')
    with np.load(doc['hawor_detector_npz']['path'],allow_pickle=False) as hz:
        arrays=dict(hz)
    hawor_contract=_validated_hawor_source_contract(source_doc,arrays,declared)
    mode=str(np.asarray(arrays['camera_intrinsics_contract_mode']).reshape(-1)[0])
    A=np.asarray(arrays['A_source_from_hawor_inference'],np.float64)
    if (plane.get('camera_intrinsics_contract_mode')!=mode or
            not np.allclose(plane.get('A_source_from_hawor_inference'),A,atol=1e-12) or
            plane.get('validated_camera_contract_sha256')!=hawor_contract['contract_sha256']):
        raise ValueError('hand mask HaWoR detector plane contract mismatch')
    transform=doc.get('coordinate_transform') or {}
    affine=np.asarray(transform.get('A_mask_from_source'),dtype=float)
    if affine.shape!=(3,3) or not np.isfinite(affine).all() or abs(np.linalg.det(affine))<1e-12:
        raise ValueError('hand mask coordinate transform must be finite and invertible')
    if not np.allclose(affine[2],[0,0,1],atol=1e-12):raise ValueError('hand mask transform must be affine homogeneous')
    if transform.get('pixel_center_convention')!='integer_pixel_centers_opencv' or transform.get('mapping_kind')!='explicit_half_pixel_affine':
        raise ValueError('unsupported hand mask pixel-center convention')
    execution=doc.get('model_execution') or {}
    expected_execution={'api':'sam2.build_sam.build_sam2 + sam2.sam2_image_predictor.SAM2ImagePredictor',
        'direct_frame':True,'temporal_memory':False,'propagation':False,
        'ownership':'visual_sam2_raw_logit_partition_not_physical_depth_order'}
    if any(execution.get(k)!=v for k,v in expected_execution.items()):
        raise ValueError('hand mask receipt does not declare the direct-frame SAM2 image API contract')
    uncertainty_doc=doc.get('visibility_uncertainty') or {}
    if (uncertainty_doc.get('formula')!=UNCERTAINTY_FORMULA or uncertainty_doc.get('scope')!='heuristic_overlap_uncertainty_not_physical_occlusion'):
        raise ValueError('hand mask overlap uncertainty contract mismatch')
    ownership_doc=doc.get('ownership_partition') or {}
    if (ownership_doc.get('winner')!='higher_raw_sam2_logit' or
            ownership_doc.get('exact_tie_policy')!=OVERLAP_TIE_POLICY or
            ownership_doc.get('physical_depth_order') is not False):
        raise ValueError('hand mask visual overlap partition contract mismatch')
    predictor_inputs=doc.get('predictor_inputs')
    if not isinstance(predictor_inputs,list) or len(predictor_inputs)!=len(source_rows):
        raise ValueError('hand mask receipt lacks the exact SAM2 predictor input timeline')
    predictor_by_id={int(r.get('frame_idx',-1)):r for r in predictor_inputs}
    if len(predictor_by_id)!=len(predictor_inputs) or set(predictor_by_id)!=set(source_rows):
        raise ValueError('SAM2 predictor input frame IDs differ from source timeline')
    for i,source in source_rows.items():
        row=predictor_by_id[i];raw_path=_raw_path(source);raw_hash=source.get('raw_frame_sha256') or _sha(raw_path)
        if (row.get('path')!=str(raw_path) or row.get('sha256')!=raw_hash or
                row.get('source_frame_path')!=str(raw_path) or row.get('source_frame_sha256')!=raw_hash):
            raise ValueError('SAM2 predictor input has stale source-frame provenance')
        _record(raw_path,raw_hash,'SAM2 predictor input frame')
        if row.get('derivation')!='authoritative_raw_frame_opencv_bgr_to_rgb_direct':
            raise ValueError('SAM2 predictor input is not a direct raw-frame decode')
    expected_ids=None if frame_ids is None else [int(x) for x in frame_ids]
    wanted=set(expected_ids or []);allowed_sides=set(sides)
    rows=doc.get('frames')
    if not isinstance(rows,list):raise ValueError('hand mask receipt frames must be a list')
    seen=set();row_masks={};raw_masks={};by_key={}
    for row in rows:
        i=int(row.get('frame_idx',-1));side=str(row.get('hand_side',''));key=(i,side)
        if key in seen:raise ValueError('duplicate hand mask frame/side')
        seen.add(key);by_key[key]=row
        if side not in SIDES or side not in allowed_sides:raise ValueError('invalid hand side identity')
        if expected_ids is not None and i not in wanted:raise ValueError('hand mask frame outside source timeline')
        source=source_rows.get(i)
        if source is None:raise ValueError('hand mask frame missing from source timeline')
        raw_path=_raw_path(source);raw_hash=source.get('raw_frame_sha256') or _sha(raw_path)
        if row.get('raw_frame_path')!=str(raw_path) or row.get('raw_frame_sha256')!=raw_hash:
            raise ValueError('hand mask raw RGB frame/hash is stale or attached to wrong frame')
        _record(raw_path,raw_hash,'raw RGB frame')
        if row.get('status') not in HAND_VISIBILITY_STATES:raise ValueError('invalid hand mask visibility state')
        if not isinstance(row.get('uncertainty'),(int,float)) or not 0<=float(row['uncertainty'])<=1:raise ValueError('invalid hand mask uncertainty')
        if 'occluder' not in row:raise ValueError('hand mask row lacks occluder ownership')
        if not isinstance(row.get('detector_track_id'),str):raise ValueError('detector track ID must remain an opaque string')
        prompt=prompt_rows.get(key)
        if prompt is None or prompt['detector_track_id']!=row['detector_track_id'] or prompt['sam2_object_id']!=row.get('sam2_object_id'):
            raise ValueError('hand mask row differs from SAM2 frame/side prompt identity')
        if row.get('status') in ACTIVE_MASK_STATES and not prompt.get('target_visible'):
            raise ValueError('mask has no same-frame independent detector support')
        if row.get('sam2_object_id') not in (1,2) or row['sam2_object_id']!=(1 if side=='left' else 2):
            raise ValueError('hand mask side/object identity mismatch')
        row_size=tuple(map(int,row.get('mask_size_wh') or []))
        if row_size!=declared or not np.allclose(affine,half_pixel_affine(declared,row_size),atol=1e-9):
            raise ValueError('hand mask direct-frame raster/half-pixel affine mismatch')
        for name in ('raw_support_area_px','partitioned_area_px','raw_overlap_px','overlap_tie_px','overlap_assigned_to_side_px','source_box_raw_support_px'):
            if not isinstance(row.get(name),int) or row[name]<0:raise ValueError(f'invalid hand mask {name}')
        fraction=row.get('raw_overlap_fraction_of_smaller')
        if not isinstance(fraction,(int,float)) or not 0<=float(fraction)<=1:raise ValueError('invalid raw overlap fraction')
        if row.get('overlap_winner')!='higher_raw_sam2_logit' or row.get('overlap_tie_policy')!=OVERLAP_TIE_POLICY:
            raise ValueError('hand mask overlap winner policy mismatch')
        raw_ref=(row.get('raw_positive_mask_path'),row.get('raw_positive_mask_sha256'))
        if not prompt.get('target_visible') and raw_ref!=(None,None):
            raise ValueError('SAM2 prediction exists without a same-frame detector box')
        if raw_ref==(None,None):
            raw_image=np.zeros((declared[1],declared[0]),bool)
            if (row['raw_support_area_px']!=0 or row['source_box_raw_support_px']!=0 or
                    row.get('predicted_iou') is not None or row.get('source_box_raw_support_fraction') is not None):
                raise ValueError('missing SAM2 prediction fabricates raw diagnostics')
        elif None in raw_ref:
            raise ValueError('incomplete unconstrained SAM2 positive-mask reference')
        else:
            raw_mask_path=_record(Path(raw_ref[0]),raw_ref[1],'unconstrained SAM2 positive mask')
            loaded=cv2.imread(str(raw_mask_path),cv2.IMREAD_GRAYSCALE)
            if loaded is None or (loaded.shape[1],loaded.shape[0])!=row_size:raise ValueError('raw positive mask raster mismatch')
            raw_image=loaded>0
            if row['raw_support_area_px']!=int(raw_image.sum()):raise ValueError('raw support area diagnostic mismatch')
            score=row.get('predicted_iou')
            if not isinstance(score,(int,float)) or not np.isfinite(score) or not 0<=float(score)<=1:
                raise ValueError('invalid SAM2 predicted IoU')
            box=transform_points(np.asarray(prompt['box_xyxy'],np.float32).reshape(2,2),affine).reshape(-1)
            inside=_support_inside_box(raw_image,box)
            expected_fraction=float(inside/max(1,int(raw_image.sum())))
            if row['source_box_raw_support_px']!=inside or not np.isclose(row.get('source_box_raw_support_fraction'),expected_fraction,atol=1e-12):
                raise ValueError('source-box localization diagnostic mismatch')
        raw_masks[key]=raw_image
        if row.get('status') in ACTIVE_MASK_STATES:
            mask=_record(Path(row.get('mask_path','')),row.get('mask_sha256'),'hand mask')
            image=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            if image is None or not np.any(image):raise ValueError('visible hand mask is empty/unreadable')
            if (image.shape[1],image.shape[0])!=row_size:raise ValueError('mask size metadata mismatch')
            row_masks[key]=image>0
        else:
            if row.get('mask_path') is not None or row.get('mask_sha256') is not None:
                raise ValueError('inactive hand rows must not fabricate a mask')
            row_masks[key]=np.zeros((declared[1],declared[0]),bool)
    if expected_ids is not None:
        required={(i,s) for i in expected_ids for s in allowed_sides}
        if seen!=required:raise ValueError('hand mask receipt must exactly cover frame/side timeline')
    for i in {key[0] for key in seen}:
        if not {(i,'left'),(i,'right')} <= set(by_key):continue
        left_row,right_row=by_key[i,'left'],by_key[i,'right']
        left_raw,right_raw=raw_masks[i,'left'],raw_masks[i,'right']
        left,right=row_masks[i,'left'],row_masks[i,'right']
        overlap=left_raw&right_raw;overlap_px=int(overlap.sum())
        denominator=min(int(left_raw.sum()),int(right_raw.sum()))
        overlap_fraction=float(overlap_px/denominator) if denominator else 0.
        if np.any(left&right):raise ValueError('partitioned hand masks overlap')
        if np.any(left&~left_raw) or np.any(right&~right_raw):raise ValueError('partitioned mask leaves its unconstrained SAM2 support')
        if not np.array_equal(left|right,left_raw|right_raw):raise ValueError('partitioned masks do not preserve raw support union')
        if int((left&overlap).sum())+int((right&overlap).sum())!=overlap_px:
            raise ValueError('raw overlap pixels were not assigned exactly once')
        tie_px=left_row['overlap_tie_px']
        if tie_px!=right_row['overlap_tie_px'] or tie_px>overlap_px:raise ValueError('overlap tie diagnostic mismatch')
        for side,row,mask in (('left',left_row,left),('right',right_row,right)):
            if row['raw_overlap_px']!=overlap_px or not np.isclose(row['raw_overlap_fraction_of_smaller'],overlap_fraction,atol=1e-12):
                raise ValueError('raw overlap diagnostic mismatch')
            if row['partitioned_area_px']!=int(mask.sum()) or row['overlap_assigned_to_side_px']!=int((mask&overlap).sum()):
                raise ValueError('partition area diagnostic mismatch')
            active=bool(mask.any());expected_status=('partially_visible' if overlap_px else 'visible') if active else 'unresolved'
            if row['status']!=expected_status:raise ValueError('hand mask visibility status disagrees with partition')
            expected_uncertainty=_uncertainty(overlap_fraction) if active else 1.
            if not np.isclose(row['uncertainty'],expected_uncertainty,atol=1e-12):raise ValueError('hand mask uncertainty formula mismatch')
            expected_occluder=OVERLAP_OCCLUDER if active and overlap_px else None
            if row['occluder']!=expected_occluder:raise ValueError('hand mask overlap occluder mismatch')
    return doc


def hand_mask_rows(manifest: str|Path, *, frame_ids: Iterable[int], source_size_wh: tuple[int,int],
                   source_frame_manifest: str|Path) -> tuple[dict,dict[tuple[int,str],dict]]:
    doc=validate_hand_mask_receipt(manifest,frame_ids=frame_ids,source_size_wh=source_size_wh,
                                   source_frame_manifest=source_frame_manifest)
    return doc,{(int(r['frame_idx']),str(r['hand_side'])):r for r in doc['frames']}


def make_sam2_hand_prompt_payload(*,raw_frame_manifest: str|Path,hawor_npz: str|Path) -> dict:
    raw_path=Path(raw_frame_manifest).resolve();raw=json.loads(raw_path.read_text());frames=_rows(raw)
    first=next(iter(frames.values()))
    width=int(first.get('source_width',0));height=int(first.get('source_height',0))
    if width<=0 or height<=0:raise ValueError('raw frame manifest lacks source dimensions')
    with np.load(hawor_npz,allow_pickle=False) as z:
        ids=np.asarray(z['frame_idx']).astype(int)
        if len(np.unique(ids))!=len(ids) or set(ids)!=set(frames) or len(ids)!=len(frames):raise ValueError('HaWoR detector timeline differs from RGB manifest')
        hawor_contract=_validated_hawor_source_contract(raw,dict(z),(width,height))
        mode=str(np.asarray(z['camera_intrinsics_contract_mode']).reshape(-1)[0])
        A_source=np.asarray(z['A_source_from_hawor_inference'],np.float64)
        if A_source.shape!=(3,3) or not np.isfinite(A_source).all():raise ValueError('HaWoR detector plane affine is invalid')
        for side in SIDES:
            key=side+'_detected_same_frame'
            if key not in z or np.asarray(z[key]).shape!=(len(ids),):
                raise ValueError(f'HaWoR detector archive lacks exact {key} provenance')
        payload=[]
        for pos,frame in enumerate(ids):
            for side in SIDES:
                detected=bool(np.asarray(z[side+'_detected_same_frame'])[pos])
                box=np.asarray(z[side+'_det_box_xyxyscore'])[pos,:4]
                track=str(np.asarray(z[side+'_track_id'])[pos])
                valid=detected and np.isfinite(box).all() and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height
                payload.append({'frame_idx':int(frame),'hand_side':side,'detector_track_id':track,
                    'sam2_object_id':1 if side=='left' else 2,'target_visible':bool(valid),
                    'box_xyxy':[float(x) for x in box] if valid else None,
                    'prompt_role':'conditioning_only','prompt_source':'hawor_same_frame_detector_box'})
    return {'schema':'sam2_rgb_hand_prompt_payload_v1','coordinate_frame':f'source_video_pixels_{width}x{height}',
        'source_size_wh':[width,height],'source_video_sha256':raw.get('input_video_sha256'),
        'source_rgb_manifest_path':str(raw_path),'source_rgb_manifest_sha256':_sha(raw_path),
        'hawor_detector_npz':{'path':str(Path(hawor_npz).resolve()),'sha256':_sha(Path(hawor_npz))},
        'hawor_detector_plane_contract':{'box_output_plane':'source_rgb',
            'camera_intrinsics_contract_mode':mode,'A_source_from_hawor_inference':A_source.tolist(),
            'validated_camera_contract_sha256':hawor_contract['contract_sha256']},'tracks':payload}


def _model_reference(sam2_source: Path,model_cfg: str|Path) -> Path:
    cfg=Path(model_cfg)
    candidates=[cfg] if cfg.is_absolute() else [sam2_source/cfg,sam2_source/'sam2'/cfg]
    for candidate in candidates:
        if candidate.is_file():return candidate.resolve()
    return candidates[0].resolve()


def produce_sam2_hand_masks(*,raw_frame_manifest: str|Path,hawor_npz: str|Path,output_dir: str|Path,
        checkpoint: str|Path,model_cfg: str|Path,sam2_source: str|Path|None=None,
        device: str='cuda',predictor_factory=None) -> Path:
    """Run independent official SAM2 image predictions and partition hand overlap."""
    import torch
    raw_path=Path(raw_frame_manifest).resolve();hawor_path=Path(hawor_npz).resolve();out=Path(output_dir).resolve()
    if out.exists():raise FileExistsError('SAM2 hand output must be create-only')
    source=Path(sam2_source or os.environ.get('EGO_SAM2_SOURCE','')).resolve()
    checkpoint=Path(checkpoint).resolve();model_cfg_argument=str(model_cfg);cfg=_model_reference(source,model_cfg)
    for p,label in ((raw_path,'raw manifest'),(hawor_path,'HaWoR detector archive'),(checkpoint,'SAM2 checkpoint'),(cfg,'SAM2 model config')):
        if not p.is_file():raise FileNotFoundError(f'{label} missing: {p}')
    prompt=make_sam2_hand_prompt_payload(raw_frame_manifest=raw_path,hawor_npz=hawor_path)
    raw=json.loads(raw_path.read_text());source_rows=_rows(raw);ids=sorted(source_rows)
    source_wh=tuple(prompt['source_size_wh']);images=[]
    for i in ids:
        p=_raw_path(source_rows[i]);_record(p,source_rows[i].get('raw_frame_sha256') or _sha(p),'raw RGB frame')
        image=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'unreadable RGB frame {p}')
        if (image.shape[1],image.shape[0])!=source_wh:raise ValueError('SAM2 direct RGB frame differs from declared source raster')
        images.append((p,image))
    mask_wh=source_wh;A=half_pixel_affine(source_wh,mask_wh)
    out.mkdir(parents=True);mask_dir=out/'masks';mask_dir.mkdir();raw_mask_dir=out/'raw_positive_masks';raw_mask_dir.mkdir()
    prompt_path=out/'prompt_payload.json';prompt_path.write_text(json.dumps(prompt,indent=2,allow_nan=False)+'\n')
    predictor_inputs=[{'frame_idx':int(frame),'path':str(p),'sha256':_sha(p),
        'source_frame_path':str(p),'source_frame_sha256':_sha(p),
        'derivation':'authoritative_raw_frame_opencv_bgr_to_rgb_direct'} for frame,(p,_) in zip(ids,images)]
    if predictor_factory is None:
        sys.path.insert(0,str(source))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model=build_sam2(model_cfg_argument,str(checkpoint),device=torch.device(device))
        predictor=SAM2ImagePredictor(model)
    else:
        # CPU tests inject an assembled SAM2ImagePredictor protocol double.
        predictor=predictor_factory(model_cfg_argument,str(checkpoint),device=torch.device(device))
    prompts={(int(r['frame_idx']),r['hand_side']):r for r in prompt['tracks']}
    inferred={}
    autocast=torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()
    with torch.inference_mode(),autocast:
        for frame,(path,bgr) in zip(ids,images):
            predictor.set_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            frame_predictions={}
            for side in SIDES:
                prompt_row=prompts[(frame,side)]
                if not prompt_row['target_visible']:continue
                box=transform_points(np.asarray(prompt_row['box_xyxy'],np.float32).reshape(2,2),A).reshape(-1)
                high_res_logits,predicted_iou,_=predictor.predict(point_coords=None,point_labels=None,
                    box=box,multimask_output=False,return_logits=True)
                logits=np.asarray(high_res_logits)
                if logits.ndim==3 and logits.shape[0]==1:logits=logits[0]
                if logits.shape!=(mask_wh[1],mask_wh[0]) or not np.isfinite(logits).all():
                    raise ValueError(f'invalid direct-frame SAM2 high-resolution logits for frame {frame} {side}')
                scores=np.asarray(predicted_iou,dtype=np.float64).reshape(-1)
                if len(scores)!=1 or not np.isfinite(scores[0]) or not 0<=float(scores[0])<=1:
                    raise ValueError(f'invalid SAM2 predicted IoU for frame {frame} {side}')
                frame_predictions[side]={'logits':logits.astype(np.float32,copy=False),'predicted_iou':float(scores[0])}
            inferred[frame]=frame_predictions
    rows=[]
    for frame,(path,_) in zip(ids,images):
        prediction=inferred[frame]
        raw_masks={side:(prediction[side]['logits']>0 if side in prediction else None) for side in SIDES}
        left_raw,right_raw=raw_masks['left'],raw_masks['right']
        if left_raw is not None and right_raw is not None:
            overlap=left_raw&right_raw
            left_logits=prediction['left']['logits'];right_logits=prediction['right']['logits']
            ties=overlap&(left_logits==right_logits)
            left_wins=overlap&(left_logits>=right_logits)
            right_wins=overlap&(right_logits>left_logits)
            partitions={'left':left_raw&(~overlap|left_wins),'right':right_raw&(~overlap|right_wins)}
        else:
            overlap=np.zeros((mask_wh[1],mask_wh[0]),bool);ties=overlap.copy()
            partitions={'left':None if left_raw is None else left_raw.copy(),
                        'right':None if right_raw is None else right_raw.copy()}
        overlap_px=int(overlap.sum())
        areas=[int(mask.sum()) for mask in (left_raw,right_raw) if mask is not None]
        denominator=min(areas) if len(areas)==2 else 0
        overlap_fraction=float(overlap_px/denominator) if denominator else 0.
        for side in SIDES:
            prompt_row=prompts[(frame,side)];obj=prompt_row['sam2_object_id'];raw_mask=raw_masks[side];partition=partitions[side]
            raw_path_out=None;raw_sha=None;raw_area=0;inside=0;inside_fraction=None;score=None
            if raw_mask is not None:
                raw_path_out=raw_mask_dir/f'{frame:06d}_{side}.png'
                if not cv2.imwrite(str(raw_path_out),raw_mask.astype(np.uint8)*255):raise RuntimeError(f'failed to write {raw_path_out}')
                raw_sha=_sha(raw_path_out);raw_area=int(raw_mask.sum());score=prediction[side]['predicted_iou']
                box=transform_points(np.asarray(prompt_row['box_xyxy'],np.float32).reshape(2,2),A).reshape(-1)
                inside=_support_inside_box(raw_mask,box);inside_fraction=float(inside/max(1,raw_area))
            active=partition is not None and bool(partition.any())
            status=('partially_visible' if overlap_px else 'visible') if active else 'unresolved'
            if active:inactive_reason=None
            elif not prompt_row['target_visible']:inactive_reason='no_same_frame_detection'
            elif raw_mask is None:inactive_reason='missing_sam2_prediction'
            elif not raw_area:inactive_reason='empty_unconstrained_support'
            else:inactive_reason='empty_after_logit_partition'
            assigned=int((partition&overlap).sum()) if partition is not None else 0
            row={'frame_idx':frame,'hand_side':side,'detector_track_id':prompt_row['detector_track_id'],
                'sam2_object_id':obj,'raw_frame_path':str(path),'raw_frame_sha256':_sha(path),
                'status':status,'uncertainty':_uncertainty(overlap_fraction) if active else 1.,
                'occluder':OVERLAP_OCCLUDER if active and overlap_px else None,
                'mask_path':None,'mask_sha256':None,'mask_size_wh':list(mask_wh),'inactive_reason':inactive_reason,
                'model_call':'official_sam2_image_predictor_direct_frame_box_prompt',
                'predicted_iou':score,'raw_positive_mask_path':str(raw_path_out) if raw_path_out else None,
                'raw_positive_mask_sha256':raw_sha,'raw_support_area_px':raw_area,
                'partitioned_area_px':int(partition.sum()) if partition is not None else 0,
                'raw_overlap_px':overlap_px,'raw_overlap_fraction_of_smaller':overlap_fraction,
                'overlap_winner':'higher_raw_sam2_logit','overlap_tie_policy':OVERLAP_TIE_POLICY,
                'overlap_tie_px':int(ties.sum()),'overlap_assigned_to_side_px':assigned,
                'source_box_raw_support_px':inside,'source_box_raw_support_fraction':inside_fraction}
            if active:
                mp=mask_dir/f'{frame:06d}_{side}.png'
                if not cv2.imwrite(str(mp),partition.astype(np.uint8)*255):raise RuntimeError(f'failed to write {mp}')
                row.update(mask_path=str(mp),mask_sha256=_sha(mp))
            rows.append(row)
    receipt={'schema':'rgb_sam2_hand_masks_v1','producer':'sam2_rgb_hand_track',
        'source_video_sha256':prompt['source_video_sha256'],'source_rgb_manifest_path':str(raw_path),
        'source_rgb_manifest_sha256':_sha(raw_path),'source_size_wh':list(source_wh),
        'coordinate_transform':{'A_mask_from_source':A.tolist(),'pixel_center_convention':'integer_pixel_centers_opencv',
            'mapping_kind':'explicit_half_pixel_affine'},'checkpoint':{'path':str(checkpoint),'sha256':_sha(checkpoint)},
        'model_config':{'path':str(cfg),'sha256':_sha(cfg)},'hawor_detector_npz':{'path':str(hawor_path),'sha256':_sha(hawor_path)},
        'hawor_detector_plane_contract':prompt['hawor_detector_plane_contract'],
        'prompt_payload':{'path':str(prompt_path),'sha256':_sha(prompt_path)},'predictor_inputs':predictor_inputs,
        'model_execution':{'api':'sam2.build_sam.build_sam2 + sam2.sam2_image_predictor.SAM2ImagePredictor',
            'device':str(device),'direct_frame':True,'temporal_memory':False,'propagation':False,
            'set_image_once_per_frame':True,'predict_once_per_available_side':True,
            'raw_high_resolution_logits':True,'multimask_output':False,
            'ownership':'visual_sam2_raw_logit_partition_not_physical_depth_order'},
        'ownership_partition':{'winner':'higher_raw_sam2_logit','exact_tie_policy':OVERLAP_TIE_POLICY,
            'physical_depth_order':False,'contact_authority':False},
        'visibility_uncertainty':{'formula':UNCERTAINTY_FORMULA,
            'scope':'heuristic_overlap_uncertainty_not_physical_occlusion'},'frames':rows}
    receipt_path=out/'hand_mask_receipt.json';receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    validate_hand_mask_receipt(receipt_path,frame_ids=ids,source_size_wh=source_wh,source_frame_manifest=raw_path)
    return receipt_path


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-frame-manifest',type=Path,required=True);p.add_argument('--hawor-npz',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-cfg',required=True);p.add_argument('--sam2-source',type=Path);p.add_argument('--device',default='cuda')
    return p.parse_args(argv)


def main(argv=None):
    a=parse_args(argv)
    path=produce_sam2_hand_masks(raw_frame_manifest=a.raw_frame_manifest,hawor_npz=a.hawor_npz,
        output_dir=a.output_dir,checkpoint=a.checkpoint,model_cfg=a.model_cfg,sam2_source=a.sam2_source,device=a.device)
    print(json.dumps({'status':'ok','receipt':str(path),'receipt_sha256':_sha(path)},indent=2));return path


if __name__=='__main__':main()

__all__=['HAND_VISIBILITY_STATES','validate_hand_mask_receipt','hand_mask_rows','make_sam2_hand_prompt_payload',
         'produce_sam2_hand_masks','half_pixel_affine','transform_points']
