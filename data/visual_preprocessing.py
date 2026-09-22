"""Static source MANO / appearance preparation, independent of any learned update.

These are observations derived from frozen source geometry. They are not a hand
fit, contact measurement, policy encoder, or candidate-dependent triangle query.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree
from data.immutable import freeze
from data.timeline import timeline_ids
from utils.geometry import resized_K, validate_transforms
from utils.mano import TIP_VERTICES
from utils.state import sha

FIT_REFERENCE_WIDTH = 960


def fit_size(source_size):
    size=np.asarray(source_size)
    if size.shape!=(2,) or np.any(size<=0) or not np.isfinite(size).all() or np.any(size!=size.astype(int)):
        raise ValueError('invalid source image size')
    return np.array([FIT_REFERENCE_WIDTH,max(1,round(size[1]*FIT_REFERENCE_WIDTH/size[0]))],np.int64)


def source_hand_views(hands, frame_ids, camera, intrinsics, source_size):
    """Full source projections + source-anatomy pads; no MANO weight load."""
    ids=timeline_ids(frame_ids);T=validate_transforms(camera)
    K=np.asarray(intrinsics,float);size=fit_size(source_size);n=len(ids)
    if T.shape!=(n,4,4) or K.shape!=(n,4) or not np.isfinite(K).all() or np.any(K[:,:2]<=0):
        raise ValueError('source hand camera timeline mismatch')
    fit_K=np.asarray([resized_K(k,source_size,size) for k in K])
    result={}
    for side,hand in hands.items():
        valid=np.asarray(hand['valid'],bool);vertices=np.asarray(hand['vertices'],float);joints=np.asarray(hand['joints'],float)
        if vertices.shape!=(n,778,3) or joints.shape!=(n,21,3) or valid.shape!=(n,):
            raise ValueError('complete time-indexed source MANO required')
        if not np.isfinite(vertices[valid]).all() or not np.isfinite(joints[valid]).all():
            raise ValueError('valid source MANO contains nonfinite geometry')
        def project(points):
            cam=np.einsum('nvj,njk->nvk',points-T[:,None,:3,3],T[:,:3,:3])
            supported=valid[:,None]&np.isfinite(cam).all(axis=2)&(cam[:,:,2]>1e-5)
            uv=np.zeros(cam.shape[:2]+(2,))
            for i in range(n):
                take=supported[i]
                uv[i,take]=cam[i,take,:2]/cam[i,take,2:]*fit_K[i,:2]+fit_K[i,2:]
            inside=supported&(uv[:,:,0]>=0)&(uv[:,:,0]<size[0])&(uv[:,:,1]>=0)&(uv[:,:,1]<size[1])
            # Source archive remains untouched. Missing views have a separate mask.
            return np.where(valid[:,None,None],cam,0.),uv,supported,inside
        vc,vu,vok,vin=project(vertices);jc,ju,jok,jin=project(joints)
        pads=np.full((n,5,12),-1,np.int64)
        for i in np.flatnonzero(valid):
            pads[i]=cKDTree(vertices[i]).query(vertices[i,np.asarray(TIP_VERTICES)],k=12)[1]
        result[side]=freeze({'source_valid':valid,'vertices_camera':vc,'joints_camera':jc,
            'vertices_uv_fit':vu,'joints_uv_fit':ju,'vertex_projection_valid':vok,
            'joint_projection_valid':jok,'vertex_in_frame':vin,'joint_in_frame':jin,
            'pad_vertex_ids':pads,'pad_membership_valid':np.broadcast_to(valid[:,None],(n,5)),
            'pad_role':'source_anatomical_membership_not_contact_measurement'})
    return freeze({'schema':'ego.annotation.source_visual_preprocessing.v1','frame_idx':np.asarray(ids),
        'fit_size_wh':size,'intrinsics_fit':fit_K,'hands':result,'source_mutated':False,
        'mano_lbs_replay_performed':False,'candidate_geometry_consumed':False})


@dataclass(frozen=True)
class AppearanceObservations:
    object_id: str
    rows: object
    frames: object

    def prepare_frame(self, frame_idx):
        """Lazy per-frame mask/contour/distance transform; no whole-video raster cache."""
        row=self.rows[frame_idx];frame=self.frames[frame_idx]
        if row['status']!='available':
            return freeze({'frame_idx':frame_idx,'available':False,'reason':row['status']})
        path=Path(row['appearance_mask_path']);source=Path(row['source_mask_path'])
        if sha(path)!=row['appearance_mask_sha256'] or sha(source)!=row['source_mask_sha256'] or sha(path)!=sha(source):
            raise ValueError('raw appearance mask bytes changed or were substituted')
        mask=cv2.imread(str(path),0)
        if mask is None or list(mask.shape[::-1])!=list(row['source_size_wh']) or not np.any(mask):
            raise ValueError('invalid source appearance mask raster')
        source_size=[frame['source_width'],frame['source_height']];size=fit_size(source_size)
        if abs(mask.shape[1]/mask.shape[0]-source_size[0]/source_size[1])>.01:
            raise ValueError('appearance aspect ratio differs from source plane')
        mask=cv2.resize(mask,tuple(size),interpolation=cv2.INTER_NEAREST)>0
        yy,xx=np.where(mask);hull=cv2.convexHull(np.column_stack([xx,yy]).astype(np.float32)).reshape(-1,2)
        contour=np.zeros((0,2),float)
        if len(hull)>=3:
            edge=np.roll(hull,-1,axis=0)-hull;length=np.linalg.norm(edge,axis=1)
            cumulative=np.r_[0.,np.cumsum(length)];positions=np.linspace(0,cumulative[-1],160,endpoint=False)
            index=np.searchsorted(cumulative,positions,side='right')-1
            contour=hull[index]+(positions-cumulative[index])[:,None]/length[index,None]*edge[index]
        distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
        return freeze({'frame_idx':frame_idx,'available':True,'fit_size_wh':size,
            'intrinsics_fit':resized_K(frame['camera']['intrinsics_fx_fy_cx_cy'],source_size,size),
            'mask':mask,'contour_uv':contour,'distance_transform_px':distance,
            'role':'raw_appearance_image_support_not_metric_surface',
            'candidate_geometry_consumed':False})


def load_appearance_observations(path, *, annotations_path, frames, object_id):
    payload=json.loads(Path(path).read_text());ids=timeline_ids([f['frame_idx'] for f in frames])
    if (payload.get('object_id')!=object_id or payload.get('metric_owned_masks_consumed') is not False
            or payload.get('source_annotations_sha256')!=sha(annotations_path)
            or Path(payload.get('source_annotations','')).resolve()!=Path(annotations_path).resolve()):
        raise ValueError('appearance/annotations role or identity mismatch')
    rows=payload['frames']
    if [r['frame_idx'] for r in rows]!=ids:raise ValueError('appearance timeline mismatch')
    for row,frame in zip(rows,frames):
        if row.get('time_s')!=frame['time_s']:raise ValueError('appearance timestamp mismatch')
        if row.get('status') not in {'available','unavailable_not_substituted'}:raise ValueError('unknown appearance availability')
        if row['status']=='available':
            for key in ['appearance_mask_path','appearance_mask_sha256','source_mask_path','source_mask_sha256','source_size_wh']:
                if key not in row:raise ValueError('missing source appearance field: '+key)
        elif row.get('appearance_mask_path') is not None:raise ValueError('unavailable appearance cannot substitute a mask')
    return AppearanceObservations(object_id,freeze({r['frame_idx']:r for r in rows}),freeze({f['frame_idx']:f for f in frames}))
