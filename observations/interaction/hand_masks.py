"""Independent RGB/SAM2 hand-mask producer and immutable receipt validation.

HaWoR detector boxes are conditioning prompts only.  This module executes the
official grouped SAM2 video predictor; callers never turn boxes, MANO renders,
or joints into silhouette observations.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable
import cv2
import numpy as np

HAND_VISIBILITY_STATES=frozenset({'visible','partially_visible','occluded','out_of_frame','unresolved'})
SIDES=('left','right')
ACTIVE_MASK_STATES=frozenset({'visible','partially_visible'})


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
    predictor_inputs=doc.get('predictor_inputs')
    if not isinstance(predictor_inputs,list) or len(predictor_inputs)!=len(source_rows):
        raise ValueError('hand mask receipt lacks the exact SAM2 predictor input timeline')
    predictor_by_id={int(r.get('frame_idx',-1)):r for r in predictor_inputs}
    if len(predictor_by_id)!=len(predictor_inputs) or set(predictor_by_id)!=set(source_rows):
        raise ValueError('SAM2 predictor input frame IDs differ from source timeline')
    for i,source in source_rows.items():
        row=predictor_by_id[i];raw_path=_raw_path(source);raw_hash=source.get('raw_frame_sha256') or _sha(raw_path)
        if row.get('source_frame_path')!=str(raw_path) or row.get('source_frame_sha256')!=raw_hash:
            raise ValueError('SAM2 predictor input has stale source-frame provenance')
        _record(Path(row.get('path','')),row.get('sha256'),'SAM2 predictor input frame')
        if row.get('derivation') not in {'authoritative_jpeg_symlink','opencv_decode_then_jpeg_default_parameters'}:
            raise ValueError('SAM2 predictor input derivation is unsupported')
    expected_ids=None if frame_ids is None else [int(x) for x in frame_ids]
    wanted=set(expected_ids or []);allowed_sides=set(sides)
    rows=doc.get('frames')
    if not isinstance(rows,list):raise ValueError('hand mask receipt frames must be a list')
    seen=set()
    for row in rows:
        i=int(row.get('frame_idx',-1));side=str(row.get('hand_side',''));key=(i,side)
        if key in seen:raise ValueError('duplicate hand mask frame/side')
        seen.add(key)
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
        if row.get('status') in ACTIVE_MASK_STATES:
            mask=_record(Path(row.get('mask_path','')),row.get('mask_sha256'),'hand mask')
            image=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            if image is None or not np.any(image):raise ValueError('visible hand mask is empty/unreadable')
            row_size=tuple(map(int,row.get('mask_size_wh') or []))
            if row_size!=(image.shape[1],image.shape[0]):raise ValueError('mask size metadata mismatch')
            if not np.allclose(affine,half_pixel_affine(declared,row_size),atol=1e-9):
                raise ValueError('hand mask half-pixel affine does not match declared rasters')
        elif row.get('mask_path') is not None or row.get('mask_sha256') is not None:
            raise ValueError('inactive hand rows must not fabricate a mask')
    if expected_ids is not None:
        required={(i,s) for i in expected_ids for s in allowed_sides}
        if seen!=required:raise ValueError('hand mask receipt must exactly cover frame/side timeline')
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
    """Execute official SAM2 once for both hands and write masks plus receipt."""
    import torch
    raw_path=Path(raw_frame_manifest).resolve();hawor_path=Path(hawor_npz).resolve();out=Path(output_dir).resolve()
    if out.exists():raise FileExistsError('SAM2 hand output must be create-only')
    source=Path(sam2_source or os.environ.get('EGO_SAM2_SOURCE','')).resolve()
    checkpoint=Path(checkpoint).resolve();model_cfg_argument=str(model_cfg);cfg=_model_reference(source,model_cfg)
    for p,label in ((raw_path,'raw manifest'),(hawor_path,'HaWoR detector archive'),(checkpoint,'SAM2 checkpoint'),(cfg,'SAM2 model config')):
        if not p.is_file():raise FileNotFoundError(f'{label} missing: {p}')
    prompt=make_sam2_hand_prompt_payload(raw_frame_manifest=raw_path,hawor_npz=hawor_path)
    raw=json.loads(raw_path.read_text());source_rows=_rows(raw);ids=sorted(source_rows)
    source_wh=tuple(prompt['source_size_wh'])
    images=[]
    for i in ids:
        p=_raw_path(source_rows[i]);_record(p,source_rows[i].get('raw_frame_sha256') or _sha(p),'raw RGB frame')
        image=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'unreadable RGB frame {p}')
        images.append((p,image.shape[1],image.shape[0]))
    mask_wh=(images[0][1],images[0][2])
    if any((w,h)!=mask_wh for _,w,h in images):raise ValueError('SAM2 RGB frames use inconsistent rasters')
    out.mkdir(parents=True);frame_dir=out/'sam2_frames';frame_dir.mkdir();mask_dir=out/'masks';mask_dir.mkdir()
    prompt_path=out/'prompt_payload.json';prompt_path.write_text(json.dumps(prompt,indent=2,allow_nan=False)+'\n')
    predictor_inputs=[]
    for local,(p,_,_) in enumerate(images):
        jpeg=p.suffix.lower() in {'.jpg','.jpeg'}
        target=frame_dir/f'{local:06d}{p.suffix.lower() if jpeg else ".jpg"}'
        if jpeg:os.symlink(p,target)
        else:
            image=cv2.imread(str(p))
            if image is None or not cv2.imwrite(str(target),image):raise RuntimeError(f'failed to materialize SAM2 input {target}')
        predictor_inputs.append({'frame_idx':int(ids[local]),'path':str(target),'sha256':_sha(target),
            'source_frame_path':str(p),'source_frame_sha256':_sha(p),
            'derivation':'authoritative_jpeg_symlink' if jpeg else 'opencv_decode_then_jpeg_default_parameters'})
    if predictor_factory is None:
        sys.path.insert(0,str(source))
        from sam2.build_sam import build_sam2_video_predictor
        predictor_factory=build_sam2_video_predictor
    predictor=predictor_factory(model_cfg_argument,str(checkpoint),device=torch.device(device),vos_optimized=False)
    # Returned logits must remain unconstrained so identity ambiguity can be
    # measured before any left/right ownership partition is accepted.
    predictor.non_overlap_masks=False;predictor.add_all_frames_to_correct_as_cond=True
    state=predictor.init_state(video_path=str(frame_dir),offload_video_to_cpu=True,offload_state_to_cpu=True)
    prompts={(int(r['frame_idx']),r['hand_side']):r for r in prompt['tracks']}
    inferred={}
    context=(torch.inference_mode() if predictor_factory is not None else contextlib.nullcontext())
    autocast=torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()
    with context,autocast:
        for local,frame in enumerate(ids):
            frame_masks={}
            for side in SIDES:
                row=prompts[(frame,side)]
                if not row['target_visible']:continue
                box=np.asarray(row['box_xyxy'],np.float32)
                # Boxes are in source-video pixels; model frames can be resized extractions.
                A=half_pixel_affine(source_wh,mask_wh)
                corners=transform_points(box.reshape(2,2),A).reshape(-1)
                _,object_ids,logits=predictor.add_new_points_or_box(inference_state=state,frame_idx=local,
                    obj_id=row['sam2_object_id'],points=None,labels=None,box=corners)
                frame_masks.update({int(obj):logits[j,0].detach().cpu().numpy()
                                    for j,obj in enumerate(object_ids)})
            if frame_masks:inferred[frame]=frame_masks
    rows=[];A=half_pixel_affine(source_wh,mask_wh)
    for frame,(p,_,_) in zip(ids,images):
        frame_inferred=inferred.get(frame,{})
        left_logits=frame_inferred.get(1);right_logits=frame_inferred.get(2)
        left_mask=None if left_logits is None else left_logits>0
        right_mask=None if right_logits is None else right_logits>0
        ambiguous_overlap=(prompts[(frame,'left')]['target_visible'] and prompts[(frame,'right')]['target_visible'] and
                           left_mask is not None and right_mask is not None and bool(np.any(left_mask&right_mask)))
        for side in SIDES:
            prompt_row=prompts[(frame,side)];obj=prompt_row['sam2_object_id']
            mask=left_mask if obj==1 else right_mask
            active=prompt_row['target_visible'] and mask is not None and np.any(mask) and not ambiguous_overlap
            row={'frame_idx':frame,'hand_side':side,'detector_track_id':prompt_row['detector_track_id'],
                'sam2_object_id':obj,'raw_frame_path':str(p),'raw_frame_sha256':_sha(p),
                'status':'visible' if active else 'unresolved','uncertainty':0.25 if active else 1.0,
                'occluder':None,'mask_path':None,'mask_sha256':None,'mask_size_wh':list(mask_wh),
                'inactive_reason':('ambiguous_side_overlap' if ambiguous_overlap else
                    'no_same_frame_detection_or_nonempty_mask' if not active else None),
                'model_call':'official_sam2_video_predictor_same_frame_box_conditioning'}
            if active:
                mp=mask_dir/f'{frame:06d}_{side}.png'
                if not cv2.imwrite(str(mp),mask.astype(np.uint8)*255):raise RuntimeError(f'failed to write {mp}')
                row.update(mask_path=str(mp),mask_sha256=_sha(mp))
            rows.append(row)
    receipt={'schema':'rgb_sam2_hand_masks_v1','producer':'sam2_rgb_hand_track',
        'source_video_sha256':prompt['source_video_sha256'],'source_rgb_manifest_path':str(raw_path),
        'source_rgb_manifest_sha256':_sha(raw_path),'source_size_wh':list(source_wh),
        'coordinate_transform':{'A_mask_from_source':A.tolist(),'pixel_center_convention':'integer_pixel_centers_opencv',
            'mapping_kind':'explicit_half_pixel_affine'},'checkpoint':{'path':str(checkpoint),'sha256':_sha(checkpoint)},
        'model_config':{'path':str(cfg),'sha256':_sha(cfg)},'hawor_detector_npz':{'path':str(hawor_path),'sha256':_sha(hawor_path)},
        'hawor_detector_plane_contract':prompt['hawor_detector_plane_contract'],
        'prompt_payload':{'path':str(prompt_path),'sha256':_sha(prompt_path)},
        'predictor_inputs':predictor_inputs,
        'model_execution':{'api':'sam2.build_sam.build_sam2_video_predictor','device':str(device),
            'non_overlap_masks':False,'overlap_qc_on_unconstrained_logits':True,
            'propagation_used_as_direct_observation':False},'frames':rows}
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
