"""Complete triangle-scene JSON/RRD export and publication, preserving display authority."""
from pathlib import Path
import json, re, uuid
import numpy as np
from utils.data import indexed, RKEY, TKEY, load, RawFrameSource
from utils.state import sha, InputIdentity
from observations.common import pose_from_values
from gates.output.acceptance import mesh, timeline
from gates.interaction.visibility import validate_visibility, apply_display_visibility, POLICY


def export(z,binding,case,object_id,track_id,output,*,raw_frames=None,raw_source=None,mode="complete",calibration=None,visibility=None,provenance=None,identity=None,visual_review=None):
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
    if mode not in {'complete','geometry_debug'}:raise ValueError('unknown export mode')
    raw_provenance=None
    if mode=='complete':
        from utils.state import validated_raw_frames
        verified,raw_provenance=validated_raw_frames(raw_source,identity)
        if raw_frames is not None and raw_frames!=verified:
            raise ValueError('caller RGB differs from producer-bound raw frames')
        raw_frames=verified
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
    document={'schema':'rgb_scene_annotation_v1' if mode=='complete' else 'rgb_scene_geometry_debug_v1',
        'partial':mode!='complete','annotation_ready':False,'physical_contact_authority':False,
        'physical_acceptance':{'accepted':False,'scope':'standalone_visual_candidate'},
        'visual_acceptance':visual_review or {'status':'pending_visual_review','accepted':False},
        'raw_frame_provenance':raw_provenance,'scene_id':case,'object_id':object_id,'track_id':track_id,
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


def export_scene(state_path,binding_path,fit_report_path,run_root,bundle_manifest,output,*,visual_review=None):
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
        Path(output),raw_source=RawFrameSource(root/'input/raw_frame_manifest/manifest.json',root),visibility=load(judgment_path),identity=identity,
        visual_review=visual_review,provenance={'fit_report_sha256':sha(fit_report_path),'input_hashes':fit['input_hashes'],
        'interaction_diagnostics':load(judgment_path).get('diagnostics',{})})


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('state','binding','fit-report','run-root','bundle-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    return export_scene(args.state,args.binding,args.fit_report,args.run_root,args.bundle_manifest,args.output)


if __name__=='__main__':main()


def mesh_overlay(state, position, image):
    """Render actual object/MANO triangles through the declared calibrated camera."""
    import cv2
    import open3d as o3d
    from utils.geometry import resized_K
    width=min(960,image.shape[1]);height=round(image.shape[0]*width/image.shape[1])
    image=cv2.resize(image,(width,height))
    K=resized_K(state['intrinsics_render'][position],state['render_size_wh'],(width,height))
    T=state['T_world_camera'][position]
    scene=o3d.t.geometry.RaycastingScene(nthreads=4);colors={}
    def add(vertices,faces,color):
        vertices=(np.asarray(vertices)-T[:3,3])@T[:3,:3]
        key=scene.add_triangles(o3d.core.Tensor(vertices.astype('float32')),
                                o3d.core.Tensor(np.asarray(faces,dtype='uint32')))
        colors[key]=np.asarray(color,float)
    if state['object_valid'][position]:
        add(state['object_vertices_canonical']@state['object_rotation'][position].T+
            state['object_translation_world'][position],state['object_faces'],[183,188,80])
    for side in ('left','right'):
        if state[side+'_display_valid'][position]:
            add(state[side+'_vertices_world'][position],state[side+'_faces'],[158,183,235])
    if not colors:return image
    y,x=np.mgrid[:height,:width]
    rays=np.stack([np.zeros_like(x),np.zeros_like(x),np.zeros_like(x),
                   (x-K[2])/K[0],(y-K[3])/K[1],np.ones_like(x)],axis=-1).astype('float32')
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)
    ids=hit['geometry_ids'].numpy();normals=hit['primitive_normals'].numpy()
    shading=.4+.6*np.abs(normals[:,:,2])
    result=image.astype(float)
    for key,color in colors.items():
        selected=ids==key
        result[selected]=.35*result[selected]+.65*color*shading[selected,None]
    return np.clip(result,0,255).astype('uint8')


def render_video(state_path, raw_manifest_path, output):
    """Full-timeline MP4 driven by complete triangles, not 2-D proxy boxes."""
    import cv2
    output=Path(output)
    if output.exists():raise ValueError('video output must be new')
    with np.load(state_path,allow_pickle=False) as archive:state=dict(archive)
    ids,times,order,*_=timeline(state)
    raw=indexed(load(raw_manifest_path)['frames'])
    if set(raw)!=set(map(int,ids)):raise ValueError('render raw/state timeline mismatch')
    fps=1/float(np.median(np.diff(times))) if len(times)>1 else 30.
    writer=None;reviews=[]
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        for pos in order:
            row=raw[int(ids[pos])]
            image=cv2.imread(str(row['raw_frame_path']))
            if image is None:raise ValueError('render raw RGB missing')
            rendered=mesh_overlay(state,pos,image)
            if writer is None:
                writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(rendered.shape[1],rendered.shape[0]))
                if not writer.isOpened():raise RuntimeError('MP4 encoder did not open')
            writer.write(rendered)
            review=output.parent/'review'/f'{int(ids[pos]):06d}.jpg';review.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(review),rendered):raise RuntimeError('failed to write rendered review frame')
            reviews.append({'frame_idx':int(ids[pos]),'path':str(review),'sha256':sha(review)})
    finally:
        if writer is not None:writer.release()
    return reviews


def pose_review_images(annotations, binding, hand_archive, frame_ids, output):
    """Project the actual refined full mesh and immutable source MANO for pose review."""
    import cv2
    from observations.object.shape import mesh_arrays
    from optimization.hand.mano import hand_validity
    frames=indexed(annotations['frames']);poses=indexed(binding['pose_rows'])
    ids=list(frames)
    with np.load(hand_archive,allow_pickle=False) as archive:hands=dict(archive)
    positions={int(i):j for j,i in enumerate(hands['frame_idx'])}
    order=[positions[i] for i in ids]
    vertices,faces=mesh_arrays(binding['render_mesh_contract']['required_mesh_path'])
    state={'T_world_camera':np.array([frames[i]['camera']['T_world_camera_metric'] for i in ids]),
        'intrinsics_render':np.array([frames[i]['camera']['intrinsics_fx_fy_cx_cy'] for i in ids]),
        'render_size_wh':np.array([frames[ids[0]]['source_width'],frames[ids[0]]['source_height']]),
        'object_vertices_canonical':vertices,'object_faces':faces,'object_valid':np.ones(len(ids),bool),
        'object_rotation':np.array([poses[i][RKEY] for i in ids]),
        'object_translation_world':np.array([poses[i][TKEY] for i in ids])}
    for side in ('left','right'):
        state[side+'_display_valid']=hand_validity(hands,side,len(ids))[order]
        if state[side+'_display_valid'].any():
            state[side+'_vertices_world']=hands[side+'_vertices_world_m'][order]
            state[side+'_faces']=hands[side+'_faces']
    output=Path(output);output.mkdir(parents=True,exist_ok=True);images=[]
    for frame in frame_ids:
        image=cv2.imread(frames[frame]['raw_frame_path'])
        if image is None:raise ValueError('missing raw pose-review image')
        path=output/f'{frame:06d}.jpg'
        if not cv2.imwrite(str(path),mesh_overlay(state,ids.index(frame),image)):
            raise RuntimeError('pose-review triangle rendering failed')
        images.append(path)
    return images
