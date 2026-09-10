#!/usr/bin/env python3
"""Dedicated complete triangle-scene export; no annotation image/video renderer."""
from __future__ import annotations
import argparse, json, re, uuid
from pathlib import Path
import numpy as np
from rgb_scene_pose_io import RKEY,TKEY,indexed,pose_from_values
from bind_rgb_scene_shape import validate_posterior
from v19_runtime_contracts import load,sha

from rgb_scene_visibility import validate_visibility, apply_display_visibility, POLICY
from rgb_scene_identity import InputIdentity, input_identity
from rgb_scene_candidates import load_candidate


def mesh(vertices,faces):
    v=np.asarray(vertices);f=np.asarray(faces)
    if v.ndim!=2 or v.shape[1:]!=(3,) or not len(v) or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or not np.issubdtype(f.dtype,np.integer) or f.min()<0 or f.max()>=len(v):
        raise ValueError('invalid complete triangle geometry')
    return v,f


def timeline(z):
    ids=np.asarray(z['frame_idx']);times=np.asarray(z['time_s'],float)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(set(ids.tolist()))!=len(ids) or not len(ids) or times.shape!=ids.shape or not np.isfinite(times).all():
        raise ValueError('invalid full scene timeline')
    order=np.argsort(times)
    if np.any(np.diff(times[order])<=0):raise ValueError('duplicate/nonincreasing times')
    n=len(ids)
    T=np.asarray(z['T_world_camera']);K=np.asarray(z['intrinsics_render']);size=np.asarray(z['render_size_wh'])
    if T.shape!=(n,4,4) or K.shape!=(n,4) or not np.isfinite(K).all() or np.any(K[:,:2]<=0) or size.shape!=(2,) or not np.issubdtype(size.dtype,np.integer) or np.any(size<=0):
        raise ValueError('invalid explicit camera/image plane')
    for t in T:
        pose_from_values(t[:3,:3],t[:3,3])
        if not np.allclose(t[3],[0,0,0,1]):raise ValueError('invalid camera homogeneous row')
    if np.asarray(z['object_rotation']).shape!=(n,3,3) or np.asarray(z['object_translation_world']).shape!=(n,3):raise ValueError('object pose length mismatch')
    if 'object_valid' not in z:raise ValueError('explicit object_valid required from standalone fit')
    valid=np.asarray(z['object_valid'])
    if valid.shape!=(n,) or valid.dtype!=bool:raise ValueError('object validity must be boolean')
    for i in np.flatnonzero(valid):pose_from_values(z['object_rotation'][i],z['object_translation_world'][i])
    for side in ('left','right'):
        hv=np.asarray(z[side+'_valid'])
        if hv.shape!=(n,) or hv.dtype!=bool or np.shape(z[side+'_vertices_world'])!=(n,778,3):raise ValueError('full side-specific MANO surface/validity required')
        for i in np.flatnonzero(hv):mesh(z[side+'_vertices_world'][i],z[side+'_faces'])
    mesh(z['object_vertices_canonical'],z['object_faces'])
    return ids,times,order,T,K,size,valid


def export(z,binding,case,object_id,track_id,output,*,raw_frames=None,calibration=None,visibility=None,provenance=None,identity=None):
    """All backing arrays are embedded in RRD or numeric JSON, never NPZ/PLY refs."""
    for value in (case,object_id,track_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value):raise ValueError('unsafe scene identity')
    if binding.get('object_id')!=object_id or binding.get('schema')!='rgb_scene_visual_refined_binding_v1':raise ValueError('typed selected-object binding required')
    if output.exists():raise ValueError('final directory already exists')
    ids,times,order,T,K,size,valid=timeline(z)
    if not isinstance(identity,InputIdentity) or identity.case!=case or binding.get('case',case)!=case:
        raise ValueError('typed immutable input identity/case mismatch')
    if identity.document()['source_timing']['frame_count'] != len(ids):
        raise ValueError('input identity/full timeline mismatch')
    visibility=validate_visibility(visibility,ids,object_id)
    expected=apply_display_visibility(z,{'object_id':object_id,'frames':[{'frame_idx':i,'visibility':v} for i,v in visibility.items()]},object_id)
    for key in ('object_valid','object_pose_valid','left_display_valid','right_display_valid','display_visibility_policy'):
        if key not in z or not np.array_equal(z[key],expected[key]):
            raise ValueError('fitted display validity/policy differs from judgment')
    poses=indexed(binding['pose_rows']);source=indexed(binding['source_pose_rows'])
    if set(poses)!=set(ids.tolist()):raise ValueError('binding/full timeline mismatch')
    for i,idx in enumerate(ids):
        r,t=pose_from_values(poses[int(idx)][RKEY],poses[int(idx)][TKEY])
        if not np.allclose(z['base_object_rotation'][i],r,atol=1e-7) or not np.allclose(z['base_object_translation_world'][i],t,atol=1e-7):
            raise ValueError('fit reset the refined posterior to a stale source')
    if raw_frames is not None and set(raw_frames)!=set(ids.tolist()):raise ValueError('RGB/full timeline mismatch')
    if provenance is None:provenance={}
    staging=output.with_name('.'+output.name+'.staging-'+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    import rerun as rr
    recording_id=str(uuid.uuid4())
    recording=rr.RecordingStream('rgb_scene_annotation',recording_id=recording_id)
    recording.save(str(staging/'world.rrd'))
    recording.log('/scene/input_identity',rr.TextDocument(json.dumps(identity.document(),sort_keys=True)),static=True)
    recording.log('/world',rr.ViewCoordinates.RDF,static=True)
    recording.log('/scene/semantics',rr.TextDocument('Complete native reconstructed object and MANO surfaces. Camera-origin hand similarity and unsigned triangle proximity are visual hypotheses, not measured contact or signed collision authority.'),static=True)
    objpath='/world/objects/'+object_id
    handpaths={s:'/world/hands/'+s for s in ('left','right')}
    def ref(path,components):return {'file':'world.rrd','recording_id':recording_id,'entity_path':path,'timeline':{'sequence':'frame','duration':'time_s'},'components':components}
    recording.log('/world/camera_trajectory',rr.LineStrips3D([T[order,:3,3]]),static=True)
    frames=[]
    mesh_present=False
    try:
        for decode_position,i in enumerate(order):
            idx=int(ids[i]);stamp=float(times[i])
            recording.set_time('frame',sequence=idx);recording.set_time('time_s',duration=stamp)
            recording.set_time('decode_position',sequence=decode_position)
            if valid[i]:
                # Temporal full mesh at reappearance avoids static geometry surviving a clear.
                recording.log(objpath,rr.Transform3D(translation=z['object_translation_world'][i],mat3x3=z['object_rotation'][i]))
                if not mesh_present:
                    recording.log(objpath,rr.Mesh3D(vertex_positions=z['object_vertices_canonical'],triangle_indices=z['object_faces'],albedo_factor=[80,188,183,255]))
                mesh_present=True
            else:
                recording.log(objpath,rr.Clear(recursive=True))
                mesh_present=False
            recording.log('/world/camera',rr.Transform3D(translation=T[i,:3,3],mat3x3=T[i,:3,:3]))
            recording.log('/world/camera/image',rr.Pinhole(image_from_camera=[[K[i,0],0,K[i,2]],[0,K[i,1],K[i,3]],[0,0,1]],resolution=size))
            if raw_frames is not None:
                import cv2
                raw=raw_frames[idx]
                if abs(float(raw['time_s'])-stamp)>1e-6:raise ValueError('RGB timestamp mismatch')
                image=cv2.imread(str(raw.get('raw_frame_path') or raw.get('rgb')))
                if image is None:raise ValueError('source RGB unavailable')
                rgb=cv2.cvtColor(cv2.resize(image,tuple(map(int,size))),cv2.COLOR_BGR2RGB)
                recording.log('/world/camera/image',rr.Image(rgb).compress(jpeg_quality=90))
            handstates={}
            for si,side in enumerate(('left','right')):
                hv=bool(z[side+'_display_valid'][i]);path=handpaths[side]
                if hv:recording.log(path,rr.Mesh3D(vertex_positions=z[side+'_vertices_world'][i],triangle_indices=z[side+'_faces'],albedo_factor=[235,183,158,255]))
                else:recording.log(path,rr.Clear(recursive=True))
                params={}
                if bool(z[side+'_valid'][i]):
                    for key in ('hand_articulation','hand_root','hand_translation_camera','hand_log_scale'):
                        if key in z:params[key]=np.asarray(z[key][i,si]).tolist()
                    for suffix in ('root_orient_axis_angle','hand_pose_axis_angle','betas','trans_world_m'):
                        key=side+'_source_'+suffix
                        if key in z:params['source_'+suffix]=np.asarray(z[key][i]).tolist()
                handstates[side]={'id':side,'surface_valid':hv,'source_surface_valid':bool(z[side+'_valid'][i]),'surface':ref(path,['Mesh3D:vertex_positions','Mesh3D:triangle_indices']) if hv else None,
                    'visibility':visibility[idx][side],
                    'parameters':params,'semantics':'MANO LBS with fixed beta; world wrist-root rotation, then camera-origin exp(log_scale) similarity and camera translation',
                    'uncertainty':'visual similarity changes metric depth/extent; occluded states are uncertain',
                    'interaction':str(z['interaction_states'][i,si]) if 'interaction_states' in z else 'unresolved'}
            frames.append({'frame_idx':idx,'time_s':stamp,'decode_position':decode_position,
                'camera':{'T_world_camera':T[i].tolist(),'intrinsics_fx_fy_cx_cy':K[i].tolist(),'image_size_wh':size.tolist(),'entity':ref('/world/camera',['Transform3D']),'uncertainty':'prediction-side calibrated or estimated camera; see calibration provenance'},
                'object':{'id':object_id,'track_id':track_id,'source_prior':source.get(idx),
                    'refined_graph_pose':{RKEY:poses[idx][RKEY],TKEY:poses[idx][TKEY]},
                    'visual_fitted_pose':{RKEY:np.asarray(z['object_rotation'][i]).tolist(),TKEY:np.asarray(z['object_translation_world'][i]).tolist()} if z['object_pose_valid'][i] else None,
                    'geometry':ref(objpath,['Mesh3D:vertex_positions','Mesh3D:triangle_indices','Transform3D']) if valid[i] else None,
                    'fitted_pose_valid':bool(z['object_pose_valid'][i]),'display_valid':bool(valid[i]),
                    'metric_observation':poses[idx].get('metric_observation',False),'uncertainty':poses[idx].get('uncertainty',1.),
                    'visibility':visibility[idx]['object']},'hands':handstates})
        recording.flush()
    finally:
        recording.disconnect()
    document={'schema':'rgb_scene_annotation_v1','scene_id':case,'object_id':object_id,'track_id':track_id,
        'input_identity':identity.document(),'display_visibility_policy':POLICY,
        'units':'meters','transform_convention':'column vectors; world_from_camera and world_from_object; left SO3 plus additive world translation',
        'canonical_gauge':binding['canonical_gauge'],
        'shape_provenance':binding.get('visual_mesh_receipt',{}),
        'calibration':calibration or {'source':'estimated','uncertainty':'unresolved'},
        'geometry_authority':'native_reconstructed_complete_triangles; unsigned visual proximity only',
        'hand_model_provenance':provenance,'source_prior_sha256':binding['source_prior']['sha256'],
        'refined_posterior_sha256':binding['refined_posterior']['sha256'],
        'recording':{'file':'world.rrd','sha256':sha(staging/'world.rrd'),'recording_id':recording_id,'application_id':'rgb_scene_annotation'},
        'frames':frames}
    # Source report fields may contain internal provenance paths. Keep numerics and
    # hashes, not loading dependencies, in the portable final annotation.
    def portable(value):
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items() if not (isinstance(v,str) and v.startswith('/'))}
        if isinstance(value,list):return [portable(v) for v in value]
        return value
    # Entity paths ARE required references, so sanitize only source/calibration provenance.
    for f in document['frames']:f['object']['source_prior']=portable(f['object']['source_prior'])
    document['shape_provenance']=portable(document['shape_provenance'])
    document['calibration']=portable(document['calibration']);document['hand_model_provenance']=portable(document['hand_model_provenance'])
    with (staging/'annotation.json').open('x') as f:json.dump(document,f,indent=2,allow_nan=False)
    if output.exists():raise ValueError('final destination appeared during export')
    staging.rename(output)
    return document


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('state','binding','fit-report','run-root','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--candidate-context',type=Path)
    a=p.parse_args();root=a.run_root.resolve();binding=load(a.binding)
    candidate=None
    if a.candidate_context:
        candidate,_=load_candidate(a.candidate_context)
        if candidate.run_root!=root: raise ValueError('candidate context belongs to another run')
    def out(rel): return candidate.path(rel) if candidate is not None else root/rel

    source=binding['refined_posterior']
    if sha(source['path'])!=source['sha256']:raise ValueError('refined state changed')
    validate_posterior(load(source['path']),binding['object_id'],root)
    fit=load(a.fit_report)
    if fit['input_hashes']['pose_report']!=sha(a.binding):raise ValueError('fitter consumed different binding')
    if fit['mesh_sha256']!=binding['render_mesh_contract']['required_mesh_sha256']:raise ValueError('fitter geometry mismatch')
    with np.load(a.state,allow_pickle=False) as archive:z=dict(archive)
    from rgb_scene_shape_kernels import mesh_arrays
    mesh_contract=binding['render_mesh_contract']
    if sha(mesh_contract['required_mesh_path'])!=mesh_contract['required_mesh_sha256']:raise ValueError('bound mesh mutated')
    vertices,faces=mesh_arrays(mesh_contract['required_mesh_path'])
    if not np.array_equal(z['object_faces'],faces) or np.shape(z['object_vertices_canonical'])!=vertices.shape or not np.allclose(z['object_vertices_canonical'],vertices,atol=1e-6,rtol=0):
        raise ValueError('fitted state dropped or changed bound native geometry')
    import rerun as rr
    bundle=Path(__file__).resolve().parents[1]
    cfg=load(bundle/'runtime/runtime_config.json')
    if rr.__version__!=cfg['scene_rerun_version']:raise ValueError('Rerun dependency differs from prepared identity')
    selection=load(out('state/object_selection.json'))
    contract=load(root/'input/runtime_input_contract.json')
    raw=indexed(load(root/'input/raw_frame_manifest/manifest.json')['frames'])
    calibration=load(root/'state/calibration/v19_camera_calibration_contract.json')
    judgment=load(out(f"state/agent_interaction_judgments/{binding['object_id']}.json"))

    from v19_runtime_assets import verify_bundle
    from rgb_scene_recovery import recovery_context
    recovery=recovery_context(root,bundle,contract,verify_bundle(bundle,asset_role='export'))
    identity=input_identity(root/'input/runtime_input_contract.json',root,bundle/'runtime/bundle_manifest.json',recovery)
    if selection.get('case')!=contract['case'] or selection.get('object_id')!=binding['object_id']:raise ValueError('runtime selection identity mismatch')
    if fit.get('display_visibility',{}).get('judgment_sha256')!=sha(out(f"state/agent_interaction_judgments/{binding['object_id']}.json")):raise ValueError('fit visibility judgment changed')
    provenance={'input_hashes':fit['input_hashes'],'replay_max_vertex_error_m':fit['zero_state_max_vertex_error_m']}
    if candidate is not None:
        provenance['candidate_identity']=load_candidate(candidate.contract_path)[1]
        reuse_receipt=out(f"state/native_reuse_receipt.json")
        if reuse_receipt.is_file(): provenance['native_reuse_receipt']=load(reuse_receipt)
    export(z,binding,contract['case'],binding['object_id'],selection.get('track_id',binding['object_id']),a.output_dir,
           raw_frames=raw,calibration=calibration,visibility=judgment,identity=identity,provenance=provenance)

if __name__=='__main__':main()
