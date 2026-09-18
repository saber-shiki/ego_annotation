"""Independent RGB/SAM2 hand-mask receipts and HaWoR-prompt producer seam.

Detector boxes are accepted only as conditioning prompts.  This module never
turns them into hand state or masks; the receipt must contain masks actually
produced by the independent SAM2 RGB pass.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np

HAND_VISIBILITY_STATES = frozenset({'visible','partially_visible','occluded','out_of_frame','unresolved'})
SIDES = frozenset({'left','right'})
ACTIVE_MASK_STATES = frozenset({'visible','partially_visible'})

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def _digest(value: Any, label: str) -> str:
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value.lower()):
        raise ValueError(f'{label} must be a SHA256 digest')
    return value.lower()

def _record(path: Path, expected: str, label: str) -> Path:
    if not path.is_file(): raise ValueError(f'{label} is missing: {path}')
    if _sha(path) != _digest(expected,label+' hash'): raise ValueError(f'{label} hash mismatch')
    return path

def validate_hand_mask_receipt(manifest: str|Path, *, frame_ids: Iterable[int]|None=None,
                               sides: Iterable[str]=('left','right'), source_size_wh: tuple[int,int]|None=None,
                               source_frame_manifest: str|Path|None=None) -> dict:
    """Validate a per-side/per-frame independent SAM2 mask receipt.

    A half-pixel affine is mandatory, even for equal-sized masks, so consumers
    cannot silently resize a mask into a different coordinate plane.
    """
    path=Path(manifest); doc=json.loads(path.read_text())
    if doc.get('schema') != 'rgb_sam2_hand_masks_v1': raise ValueError('unsupported hand mask receipt schema')
    if doc.get('producer') != 'sam2_rgb_hand_track': raise ValueError('hand masks must be produced by independent SAM2 RGB tracking')
    _digest(doc.get('source_video_sha256'),'source video hash')
    if not doc.get('source_rgb_manifest_sha256'): raise ValueError('hand mask receipt lacks source RGB manifest hash')
    _digest(doc['source_rgb_manifest_sha256'],'source RGB manifest hash')
    declared=doc.get('source_size_wh'); declared=tuple(map(int,declared or []))
    if len(declared)!=2 or min(declared)<=0 or (source_size_wh is not None and declared!=tuple(map(int,source_size_wh))):
        raise ValueError('hand mask source RGB plane mismatch')
    transform=doc.get('coordinate_transform') or {}
    affine=np.asarray(transform.get('A_mask_from_source'),dtype=float)
    if affine.shape!=(3,3) or not np.isfinite(affine).all(): raise ValueError('hand mask coordinate transform missing')
    if transform.get('pixel_center_convention') != 'integer_pixel_centers_opencv': raise ValueError('unsupported hand mask pixel-center convention')
    if transform.get('mapping_kind') != 'explicit_half_pixel_affine': raise ValueError('hand mask half-pixel mapping is not explicit')
    expected_ids=None if frame_ids is None else [int(x) for x in frame_ids]
    rows=doc.get('frames');
    if not isinstance(rows,list): raise ValueError('hand mask receipt frames must be a list')
    seen=set(); wanted=set(expected_ids or [])
    for row in rows:
        i=int(row.get('frame_idx',-1)); side=str(row.get('hand_side',''))
        key=(i,side)
        if key in seen: raise ValueError('duplicate hand mask frame/side')
        seen.add(key)
        if side not in SIDES or side not in set(sides): raise ValueError('invalid hand side identity')
        if expected_ids is not None and i not in wanted: raise ValueError('hand mask frame outside source timeline')
        if row.get('status') not in HAND_VISIBILITY_STATES: raise ValueError('invalid hand mask visibility state')
        if not isinstance(row.get('uncertainty'),(int,float)) or not 0<=float(row['uncertainty'])<=1: raise ValueError('invalid hand mask uncertainty')
        if 'occluder' not in row: raise ValueError('hand mask row lacks occluder ownership')
        if not row.get('raw_frame_path') or not row.get('raw_frame_sha256'):
            raise ValueError('hand mask row lacks raw RGB frame path/hash')
        _digest(row['raw_frame_sha256'],'raw frame hash')
        _record(Path(row['raw_frame_path']), row['raw_frame_sha256'], 'raw RGB frame')
        if source_frame_manifest is not None:
            source_doc=json.loads(Path(source_frame_manifest).read_text())
            source_rows={int(x['frame_idx']):x for x in source_doc.get('frames',[])}
            source=source_rows.get(i)
            if source is None or source.get('raw_frame_path') != row['raw_frame_path'] or source.get('raw_frame_sha256') != row['raw_frame_sha256']:
                raise ValueError('hand mask raw RGB frame/hash is stale')
        if row.get('status') in ACTIVE_MASK_STATES:
            mp=row.get('mask_path'); mh=row.get('mask_sha256')
            if not mp or not mh: raise ValueError('visible hand mask lacks path/hash')
            mask=_record(Path(mp),mh,'hand mask')
            image=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            if image is None or not np.any(image): raise ValueError('visible hand mask is empty/unreadable')
            row_size=tuple(map(int,row.get('mask_size_wh') or []))
            if len(row_size)!=2 or row_size!=(image.shape[1],image.shape[0]): raise ValueError('mask size metadata mismatch')
        elif row.get('mask_path') is not None:
            raise ValueError('occluded/unresolved hand rows must not fabricate a mask')
    if expected_ids is not None:
        present={(int(r['frame_idx']),str(r['hand_side'])) for r in rows}
        required={(i,s) for i in expected_ids for s in sides}
        if not required.issubset(present): raise ValueError('hand mask receipt does not cover full timeline')
    return doc

def hand_mask_rows(manifest: str|Path, *, frame_ids: Iterable[int], source_size_wh: tuple[int,int]) -> dict[tuple[int,str],dict]:
    doc=validate_hand_mask_receipt(manifest,frame_ids=frame_ids,source_size_wh=source_size_wh)
    return {(int(r['frame_idx']),str(r['hand_side'])):r for r in doc['frames']}

def make_sam2_hand_prompt_payload(*, raw_frame_manifest: str|Path, hawor_npz: str|Path) -> dict:
    """Serialize grouped left/right SAM2 prompts from HaWoR detector boxes."""
    raw=json.loads(Path(raw_frame_manifest).read_text())
    frames=raw.get('frames') or []
    if not frames: raise ValueError('raw frame manifest has no frames')
    width,height=int(frames[0].get('source_width',0)),int(frames[0].get('source_height',0))
    if width<=0 or height<=0: raise ValueError('raw frame manifest lacks source dimensions')
    with np.load(hawor_npz,allow_pickle=False) as z:
        ids=np.asarray(z['frame_idx']).astype(int)
        payload=[]
        for i,frame in enumerate(ids):
            row={'frame_idx':int(frame),'target_visible':True,'box_xyxy':None,'positive_points':[],'negative_points':[]}
            for side in ('left','right'):
                box=np.asarray(z[side+'_det_box_xyxyscore'])[i,:4]
                if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                    continue
                # Detector boxes seed SAM2 only; no box is copied into a factor.
                payload.append({'frame_idx':int(frame),'hand_side':side,'track_id':int(z[side+'_track_id']),
                    'target_visible':True,'box_xyxy':[float(x) for x in box],
                    'positive_points':[],'negative_points':[],'prompt_role':'conditioning_only'})
    return {'schema':'sam2_rgb_hand_prompt_payload_v1','coordinate_frame':f'source_video_pixels_{width}x{height}','source_size_wh':[width,height], 'source_video_sha256':raw.get('input_video_sha256'),'tracks':payload,'mask_receipt_schema':'rgb_sam2_hand_masks_v1'}

def build_sam2_hand_track_request(*, raw_frame_manifest: str|Path, hawor_npz: str|Path,
                                  output_dir: str|Path, checkpoint: str|Path, model_cfg: str,
                                  track_ids: dict[str,int]={'left':1,'right':2}) -> dict:
    """Create the exact GPU-agent request; does not import/run SAM2.

    HaWoR detector boxes are model-produced conditioning prompts only.  The
    request asks the runtime producer to write independent RGB masks and the
    receipt contract consumed above.
    """
    if set(track_ids)!=SIDES or len(set(track_ids.values()))!=2: raise ValueError('SAM2 hand tracks require distinct left/right IDs')
    manifest=Path(raw_frame_manifest); hawor=Path(hawor_npz)
    if not manifest.is_file() or not hawor.is_file(): raise ValueError('SAM2 hand producer inputs missing')
    with np.load(hawor,allow_pickle=False) as z:
        required=['frame_idx','left_det_box_xyxyscore','right_det_box_xyxyscore','left_track_id','right_track_id']
        if any(k not in z for k in required): raise ValueError('HaWoR archive lacks detector prompt contract')
        n=len(z['frame_idx'])
        if z['left_det_box_xyxyscore'].shape!=(n,5) or z['right_det_box_xyxyscore'].shape!=(n,5): raise ValueError('invalid detector prompt boxes')
    return {'schema':'sam2_rgb_hand_track_request_v1','producer':'sam2_rgb_hand_track','raw_frame_manifest':str(manifest.resolve()),'hawor_detector_npz':str(hawor.resolve()),'output_dir':str(Path(output_dir).resolve()),'checkpoint':str(Path(checkpoint).resolve()),'model_cfg':model_cfg,'tracks':[{'hand_side':s,'track_id':int(track_ids[s]),'prompt_source':'hawor_detector_box','prompt_target':'conditioning_only_not_hand_state'} for s in ('left','right')],'output_schema':'rgb_sam2_hand_masks_v1','no_gt':True}

__all__=['HAND_VISIBILITY_STATES','validate_hand_mask_receipt','hand_mask_rows','make_sam2_hand_prompt_payload','build_sam2_hand_track_request']
