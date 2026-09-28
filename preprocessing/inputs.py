"""Hash-bound prediction-only inputs. No audit/GT dependency and no relocated bundle."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json,hashlib
import numpy as np
import cv2
from data.bundle import EpisodeBundle
from data.rgb import load_rgb_source_evidence
from preprocessing.geometry import resized_intrinsics,surface_samples


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def reject_evaluation_path(path):
    parts={x.lower() for x in Path(path).parts}
    if parts&{'evaluation','hot3d_gt','depth_gt','cad_gt'} or any('pose_audit' in x for x in parts):
        raise ValueError('evaluation-only input is forbidden')


class InputReceipt:
    def __init__(self):self.files={}
    def add(self,path,expected=None):
        path=Path(path).resolve();reject_evaluation_path(path)
        key=str(path)
        if key not in self.files:self.files[key]={'sha256':sha(path),'bytes':path.stat().st_size}
        if expected and self.files[key]['sha256']!=expected:raise ValueError('source hash mismatch: '+key)
        return path
    def check(self):
        changed=[p for p,r in self.files.items() if sha(p)!=r['sha256']]
        if changed:raise RuntimeError('source changed during preprocessing: '+str(changed[:3]))


@dataclass
class Inputs:
    bundle: object
    receipt: InputReceipt
    ids: np.ndarray
    times: np.ndarray
    cameras: np.ndarray
    K: np.ndarray
    source_K: np.ndarray
    wh: tuple
    source_wh: np.ndarray
    frames: list
    masks: np.ndarray
    grays: np.ndarray
    raw_paths: list
    mask_paths: list
    clouds: list
    cloud_valid: np.ndarray
    source_R: np.ndarray
    source_t: np.ndarray
    source_valid: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    pivot_camera: np.ndarray
    shape_basis: np.ndarray
    anchor: int
    rgb: object
    hands: dict
    visibility: object
    owned_masks: object = None
    geometry_report: object = None


def load_inputs(source,width=480,*,read_owned_masks=False):
    reject_evaluation_path(source)
    bundle=EpisodeBundle.open(source,verify=False)
    if bundle.manifest.get('schema')!='ego.annotation.observation_bundle.v2':raise ValueError('explicit v2 source required')
    receipt=InputReceipt();receipt.add(source)
    def source_path(key):
        rec=bundle.manifest['sources'][key]
        return receipt.add(bundle.source_path(key),rec['sha256'])
    def document(key):return json.loads(source_path(key).read_text())
    annotations=document('annotations');frames=annotations['frames']
    ids=np.asarray([f['frame_idx'] for f in frames],int);times=np.asarray([f['time_s'] for f in frames],float)
    if len(ids)<2 or len(set(ids))!=len(ids) or np.any(np.diff(times)<=0):raise ValueError('invalid frame timeline')
    vis=document('visual_geometry_report');entity=vis['object_id'];anchor=int(np.flatnonzero(ids==vis['appearance_anchor_frame_idx'])[0])
    if (vis.get('contains_optimized_state') is not False or vis.get('physical_authority') is not False
        or vis.get('coordinate_frame')!='opencv_camera_at_appearance_anchor'):raise ValueError('unexpected geometry role')
    import trimesh
    mesh=trimesh.load(source_path('visual_mesh_camera'),process=False)
    if isinstance(mesh,trimesh.Scene):mesh=mesh.to_geometry()
    raw_v=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces,int)
    pivot=surface_samples(raw_v,faces,4096).mean(0);vertices=raw_v-pivot
    _,basis=np.linalg.eigh(np.cov(surface_samples(vertices,faces).T));basis=basis[:,::-1]
    if np.linalg.det(basis)<0:basis[:,-1]*=-1
    cameras=np.asarray([f['camera'].get('T_world_camera_metric',f['camera']['T_world_camera']) for f in frames],float)
    srcK=np.asarray([f['camera']['intrinsics_fx_fy_cx_cy'] for f in frames],float)
    srcwh=np.asarray([[f['source_width'],f['source_height']] for f in frames],int)
    if np.any(srcwh!=srcwh[0]):raise ValueError('changing source raster unsupported')
    if not np.allclose(vis['T_world_camera_at_anchor'],cameras[anchor],rtol=0,atol=1e-8):raise ValueError('visual mesh/visible surface anchor camera mismatch')
    if not np.allclose(vis['source_K'],srcK[anchor],rtol=0,atol=1e-8):raise ValueError('visual mesh/visible surface anchor K mismatch')
    wh=(int(width),int(round(width*srcwh[0,1]/srcwh[0,0])))
    K=np.asarray([resized_intrinsics(k,s,wh) for k,s in zip(srcK,srcwh)])
    app=document('appearance_masks');apps={r['frame_idx']:r for r in app['frames']}
    if set(apps)!=set(ids):raise ValueError('appearance timeline mismatch')
    masks=[];grays=[];raw_paths=[];mask_paths=[];clouds=[];eligible=[];t=[];valid=[];owned_masks=[]
    for f in frames:
        a=apps[f['frame_idx']]
        image=receipt.add(a['raw_rgb_path'],a['raw_rgb_sha256']);mp=receipt.add(a['appearance_mask_path'],a['appearance_mask_sha256'])
        im=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE);ma=cv2.imread(str(mp),0)
        if im is None or ma is None:raise ValueError('missing source raster')
        grays.append(cv2.resize(im,wh,interpolation=cv2.INTER_AREA));masks.append(cv2.resize(ma,wh,interpolation=cv2.INTER_NEAREST_EXACT)>0)
        raw_paths.append(str(image));mask_paths.append(str(mp))
        obj=next((o for o in f.get('objects',[]) if o.get('object_id')==entity),{})
        vg=obj.get('visible_geometry_candidate') or {};po=obj.get('reconstructed_geometry_pose') or {}
        pts=np.asarray(vg.get('camera_vertices_sample_m',[]),float).reshape(-1,3)
        ok=bool(obj.get('rigid_pose_observation_eligible') is True and vg.get('rigid_pose_observation_eligible') is True and len(pts)>=4)
        clouds.append(pts if ok else np.empty((0,3)));eligible.append(ok)
        if read_owned_masks:
            owned_path=vg.get('mask_path') or obj.get('mask_path')
            if owned_path:
                path=receipt.add(owned_path)
                om=cv2.imread(str(path),0)
                if om is None:raise ValueError('declared object-owned mask is unreadable')
                owned_masks.append(cv2.resize(om,wh,interpolation=cv2.INTER_NEAREST_EXACT)>0)
            else:
                # A missing object/metric row is not manufactured as an observation.
                owned_masks.append(np.zeros(wh[::-1],bool))
        trans=po.get('translation_world_m');valid.append(trans is not None)
        t.append(np.asarray(trans,float) if trans is not None else np.full(3,np.nan))
    valid=np.asarray(valid,bool);t=np.asarray(t);eligible=np.asarray(eligible,bool)
    if not valid[anchor]:raise ValueError('appearance anchor has no source translation')
    # Re-express original no-rotation source candidate in this fixed native-centered frame.
    center_world=cameras[anchor,:3,:3]@pivot+cameras[anchor,:3,3]
    offset=center_world-t[anchor];t+=offset
    for j in range(3):t[:,j]=np.interp(times,times[valid],t[valid,j])
    R=np.repeat(cameras[anchor,None,:3,:3],len(ids),axis=0)
    rgb=load_rgb_source_evidence(source_path('rgb_source_npz'),source_path('rgb_source_report'),
        frame_ids=ids,times=times,frames=frames,object_ids=[entity],expected_inputs={
          'annotations':bundle.source_path('annotations'),'depth_npz':bundle.source_path('conditioned_depth'),
          'raw_frame_manifest':bundle.source_path('raw_frames'),'p09_adapter_report':bundle.source_path('surface_report')})
    for rec in rgb.report['inputs'].values():receipt.add(rec['path'],rec['sha256'])
    manos=source_path('mano_archive');hands={}
    with np.load(manos,allow_pickle=False) as z:
        if 'frame_idx' not in z or not np.array_equal(z['frame_idx'],ids):raise ValueError('MANO timeline mismatch')
        for side in ('left','right'):
            hands[side]={k:np.array(z[f'{side}_{s}']) for k,s in [('vertices','vertices_world_m'),('joints','joints_world_m'),('faces','faces'),('valid','valid')]}
            if hands[side]['vertices'].shape!=(len(ids),778,3):raise ValueError('MANO topology mismatch')
    visibility=document('interaction_judgments')
    return Inputs(bundle,receipt,ids,times,cameras,K,srcK,wh,srcwh,frames,np.asarray(masks),np.asarray(grays),raw_paths,mask_paths,clouds,eligible,R,t,valid,vertices,faces,pivot,basis,anchor,rgb,hands,visibility,np.asarray(owned_masks) if read_owned_masks else None,vis)
