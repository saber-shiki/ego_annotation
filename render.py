"""Complete triangle-scene JSON/RRD export and publication, preserving display authority."""
from pathlib import Path
import json, re, uuid
import numpy as np
from utils.data import indexed, RKEY, TKEY, load
from utils.state import sha, InputIdentity
from observations.common import pose_from_values
from gates.output.acceptance import mesh, timeline
from gates.interaction.visibility import validate_visibility, apply_display_visibility, POLICY


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


def export_scene(state_path,binding_path,fit_report_path,run_root,bundle_manifest,output):
    """Validate an actual fitted archive and publish complete JSON/RRD geometry."""
    from utils.state import in_run, input_identity
    from gates.object.geometry import validate_posterior
    from observations.object.shape import mesh_arrays
    root=Path(run_root).resolve()
    for path in (state_path,binding_path,fit_report_path):in_run(path,root)
    in_run(output,root,must_exist=False)
    binding=load(binding_path);fit=load(fit_report_path)
    reference=binding['refined_posterior']
    if sha(in_run(reference['path'],root))!=reference['sha256']:
        raise ValueError('refined state changed before export')
    validate_posterior(load(reference['path']),binding['object_id'],root)
    if fit['input_hashes']['pose_report']!=sha(binding_path):
        raise ValueError('fitter consumed another pose binding')
    mesh_contract=binding['render_mesh_contract']
    mesh_path=in_run(mesh_contract['required_mesh_path'],root)
    if sha(mesh_path)!=mesh_contract['required_mesh_sha256'] or fit['mesh_sha256']!=sha(mesh_path):
        raise ValueError('fitter/export mesh identity mismatch')
    vertices,faces=mesh_arrays(mesh_path)
    with np.load(state_path,allow_pickle=False) as archive:state=dict(archive)
    if not np.array_equal(state['object_faces'],faces) or not np.allclose(state['object_vertices_canonical'],vertices,atol=1e-6,rtol=0):
        raise ValueError('fit changed complete canonical mesh topology or vertices')
    judgment_path=in_run(fit['interaction_judgments'],root)
    if sha(judgment_path)!=fit['input_hashes']['interaction_judgments']:
        raise ValueError('interaction judgment changed since fit')
    identity=input_identity(root/'input/runtime_input_contract.json',root,bundle_manifest)
    raw=indexed(load(root/'input/raw_frame_manifest/manifest.json')['frames'])
    return export(state,binding,identity.case,binding['object_id'],binding.get('track_id',binding['object_id']),
        Path(output),raw_frames=raw,visibility=load(judgment_path),identity=identity,
        provenance={'fit_report_sha256':sha(fit_report_path),'input_hashes':fit['input_hashes']})


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('state','binding','fit-report','run-root','bundle-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    return export_scene(args.state,args.binding,args.fit_report,args.run_root,args.bundle_manifest,args.output)


if __name__=='__main__':main()
