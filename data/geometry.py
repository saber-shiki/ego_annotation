"""Read-only complete appearance geometry, separate from pose/collision meshes."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from data.immutable import freeze, frozen_array
from observations.object.visual_geometry import SCHEMA, ROLE
from observations.object.shape import mesh_arrays
from utils.geometry import apply_native_pose, validate_transforms, valid_rotation, resized_K
from utils.state import sha


@dataclass(frozen=True)
class VisualGeometryObservation:
    object_id: str
    anchor_frame_idx: int
    vertices_metric_camera: np.ndarray
    faces: np.ndarray
    T_world_camera: np.ndarray
    report: object


def load_visual_geometry(report_path, *, frame_ids, frames, object_id, expected_inputs):
    report=json.loads(Path(report_path).read_text())
    if report.get('schema')!=SCHEMA or report.get('role')!=ROLE or report.get('object_id')!=object_id:
        raise ValueError('visual geometry schema/role/object mismatch')
    if report.get('frame_ids')!=list(map(int,frame_ids)):
        raise ValueError('visual geometry timeline mismatch')
    if (report.get('coordinate_frame')!='opencv_camera_at_appearance_anchor' or report.get('units')!='meters'
            or report.get('candidate_dependency')!='none' or report.get('contains_optimized_state') is not False
            or report.get('image_scale_optimized') is not False
            or report.get('physical_authority') is not False
            or report.get('generated_faces_collision_eligible') is not False
            or report.get('generated_faces_signed_eligible') is not False):
        raise ValueError('visual geometry cannot contain optimized/signed/collision authority')
    for name,rec in {**report['inputs'],'output_mesh':report['output_mesh']}.items():
        p=Path(rec['path'])
        if not p.is_file() or sha(p)!=rec['sha256']:raise ValueError('visual input identity mismatch: '+name)
    for name,path in expected_inputs.items():
        rec=report['output_mesh'] if name=='output_mesh' else report['inputs'].get(name)
        if rec is None or Path(rec['path']).resolve()!=Path(path).resolve():
            raise ValueError('visual bundle input mismatch: '+name)
    anchor=report['appearance_anchor_frame_idx'];by_id={f['frame_idx']:f for f in frames}
    if anchor not in by_id:raise ValueError('visual anchor outside annotation')
    T=np.asarray(report['T_world_camera_at_anchor'],float);validate_transforms(T[None])
    frame=by_id[anchor];cam=frame['camera'];expected_T=cam.get('T_world_camera_metric',cam.get('T_world_camera'))
    if not np.allclose(T,expected_T,atol=1e-10,rtol=0):raise ValueError('visual anchor camera mismatch')
    app=json.loads(Path(report['inputs']['appearance']['path']).read_text())
    decision=json.loads(Path(report['inputs']['decision']['path']).read_text())
    if decision.get('object_id')!=object_id or decision.get('selected_frame_idx')!=anchor:
        raise ValueError('visual anchor decision mismatch')
    if app.get('object_id')!=object_id or app.get('metric_owned_masks_consumed') is not False:
        raise ValueError('visual appearance role mismatch')
    selected=next((r for r in app['frames'] if r['frame_idx']==anchor),None)
    if selected is None or selected.get('anchor_eligible') is not True:
        raise ValueError('visual appearance anchor unavailable')
    for name,key in [('image','raw_rgb'),('mask','appearance_mask')]:
        if (Path(selected[key+'_path']).resolve()!=Path(report['inputs'][name]['path']).resolve()
                or selected[key+'_sha256']!=report['inputs'][name]['sha256']):
            raise ValueError('visual appearance source differs from bound bytes')
    expected_K=resized_K(cam['intrinsics_fx_fy_cx_cy'],[frame['source_width'],frame['source_height']],selected['source_size_wh'])
    if (not np.allclose(report.get('source_K'),cam['intrinsics_fx_fy_cx_cy'],atol=1e-10,rtol=0)
            or not np.allclose(report.get('intrinsics_appearance'),expected_K,atol=1e-10,rtol=0)
            or report.get('appearance_size_wh')!=selected['source_size_wh']):
        raise ValueError('visual anchor image-plane/K mismatch')
    vertices,faces=mesh_arrays(report['output_mesh']['path'])
    raw,native_faces=mesh_arrays(report['source_native_mesh']['path'])
    if report['source_native_mesh']!=report['inputs']['native_mesh'] or not np.array_equal(faces,native_faces):
        raise ValueError('visual full topology/source identity mismatch')
    qc=json.loads(Path(report['inputs']['native_qc']['path']).read_text())
    if qc.get('status')!='ok' or Path(qc['mesh']).resolve()!=Path(report['inputs']['native_mesh']['path']).resolve():
        raise ValueError('visual native QC mesh mismatch')
    pose=qc['pose'];_,cv,_=apply_native_pose(raw,np.asarray(pose['rotation']),np.asarray(pose['translation']),np.asarray(pose['scale']))
    scale=report['scene_similarity']['scene_similarity_scale']
    if not np.isfinite(scale) or not .05<=scale<=5. or not np.allclose(vertices,cv*scale,atol=1e-6,rtol=0):
        raise ValueError('visual metric camera mesh replay mismatch')
    if len(vertices)!=report['topology']['vertex_count'] or len(faces)!=report['topology']['face_count']:
        raise ValueError('visual mesh counts mismatch')
    return VisualGeometryObservation(object_id,anchor,frozen_array(vertices),frozen_array(faces),frozen_array(T),freeze(report))


def bind_visual_to_candidate(observation, rotation, translation, *, candidate_revision):
    """Pure explicit binding API; caller owns the candidate snapshot, never source data.

    No global state or policy integration. Shared image scale is deliberately NOT
    estimated here. Returns a derived R4 binding with unchanged full topology.
    """
    if not isinstance(candidate_revision,str) or not candidate_revision.strip():
        raise ValueError('canonical binding requires an explicit candidate revision')
    R=np.asarray(rotation,float);t=np.asarray(translation,float)
    if not valid_rotation(R) or t.shape!=(3,) or not np.isfinite(t).all():raise ValueError('invalid binding candidate pose')
    T=observation.T_world_camera
    world=observation.vertices_metric_camera@T[:3,:3].T+T[:3,3]
    canonical=(world-t)@R
    if not np.allclose(canonical@R.T+t,world,atol=1e-8,rtol=0):raise ValueError('canonical roundtrip failed')
    return freeze({'role':'R4_candidate_dependent_canonical_binding','candidate_revision':candidate_revision,
        'anchor_frame_idx':observation.anchor_frame_idx,'rotation':R,'translation':t,
        'source_mesh_sha256':observation.report['output_mesh']['sha256'],
        'vertices_canonical':canonical,'faces':observation.faces,'physical_authority':False})
