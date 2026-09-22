"""Explicit synthetic v2 inspection fixture. Never a milk/model prediction.

This helper is test-only: synthetic full-778 hand arrays, synthetic native grid,
mocked feature matching, real source PnP/geometry/interaction preprocessing.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import cv2
from observations.test_rgb_source import source_fixture, run_fixture
from observations.test_visual_geometry import native_fixture
from observations.test_interaction_adapter import schema_results
from data.test_rl_data import make_synthetic_bundle
from observations.interaction.adapter import assemble_interaction_evidence
from observations.object.visual_geometry import build_visual_geometry
from observations.object.surface import SCHEMA as ELIG_SCHEMA, RASTER_SCHEMA, CLASS_CODES, array_sha256
from observations.manifest import write_source_observation_manifest
from envs.qualification.camera import SCHEMA as CAMERA_SCHEMA, resize_affine, plane_record
from utils.state import sha


def build_fixture(root, *, empty_rgb=False):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    f=source_fixture(root/'observations');ids=f['ids'];times=f['times'];object_id='object:synthetic'
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return path
    def rec(path):return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    raw_manifest=f['inputs']['raw_frame_manifest'];raw= json.loads(raw_manifest.read_text())
    track={};K=np.array([90.,90.,39.5,29.5]);source_size=(80,60);raster_size=(160,120)
    for row in raw['frames']:
        fid=row['frame_idx'];image_path=Path(row['raw_frame_path']);yy,xx=np.indices((120,160))
        image=np.stack([xx*255//159,yy*255//119,np.full_like(xx,90)],axis=-1).astype(np.uint8)
        cv2.putText(image,f'SYNTHETIC {fid}',(4,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.imwrite(str(image_path),image);row['raw_frame_sha256']=sha(image_path)
        mask_path=Path(f['frames'][ids.index(fid)]['objects'][0]['mask_path'])
        cv2.imwrite(str(mask_path),cv2.resize(cv2.imread(str(mask_path),0),raster_size,interpolation=cv2.INTER_NEAREST))
        track[str(fid)]={'visible':True,'mask_path':str(mask_path),'bbox_xyxy':[0,0,80,60],'area_px':4800}
    put(raw_manifest,raw)
    video=root/'input/raw.mp4';video.parent.mkdir(parents=True)
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),2.5,source_size)
    if not writer.isOpened():raise RuntimeError('test codec unavailable')
    for row in raw['frames']:writer.write(cv2.resize(cv2.imread(row['raw_frame_path']),source_size))
    writer.release()
    raw_track=put(root/'raw_track.json',track);owned_track=put(root/'owned_track.json',track)
    camera=put(root/'camera.json',{'schema':CAMERA_SCHEMA,'calibration_authority':'explicit_user_calibration',
        'calibration_plane':{'width':80,'height':60},'frame_ids':ids,
        'intrinsics_model':'constant_pinhole_fx_fy_cx_cy','intrinsics_coordinate_plane':'calibration',
        'intrinsics_fx_fy_cx_cy':K.tolist(),'image_planes':{
            name:plane_record(name=name,width=size[0],height=size[1],A_from_calibration=resize_affine(source_size,size),
                transform_kind='synthetic_resize',interpolation='nearest_or_area',pixel_center_convention='integer_pixel_centers_opencv')
            for name,size in [('source_rgb',source_size),('manifest_rgb',raster_size),('sam2_mask',raster_size)]},
        'synthetic_fixture':True})
    old,_=make_synthetic_bundle(root/'synthetic_hand_fixture',frame_count=2)
    old=json.loads(old.read_text());hand_ann=json.loads(Path(old['sources']['annotations']['path']).read_text())
    mano=root/'source_mano.npz'
    with np.load(old['sources']['mano_archive']['path'],allow_pickle=False) as archive:hands=dict(archive)
    hands['frame_idx']=np.array(ids);hands['time_s']=np.array(times)
    # A curved 778-vertex numeric surface with nondegenerate triangle indices.
    # It is explicitly NOT a licensed MANO model or anatomically realistic hand.
    vertex_count=778;u=np.arange(vertex_count)%39;v=np.arange(vertex_count)//39
    surface=np.column_stack([(u-19)*.004,(v-10)*.004,1.+.01*np.sin(u/5)*np.cos(v/4)]).astype(np.float32)
    faces=[]
    for a in range(vertex_count-40):
        if a%39<38:faces.extend([[a,a+1,a+39],[a+1,a+40,a+39]])
    for side,shift in [('left',-.08),('right',.08)]:
        geometry=surface.copy();geometry[:,0]+=shift
        hands[side+'_vertices_world_m']=np.repeat(geometry[None],2,axis=0)
        hands[side+'_joints_world_m']=np.repeat(geometry[np.linspace(0,777,21,dtype=int)][None],2,axis=0)
        hands[side+'_faces']=np.asarray(faces,np.int64)
    np.savez(mano,**hands)
    for i,frame in enumerate(f['frames']):
        frame['hands']=hand_ann['frames'][i]['hands']
        for hand in frame['hands']:hand['metric_mano_state']['current_v18_camera_intrinsics_fx_fy_cx_cy']=K.tolist()
        if i==0:
            obj=frame['objects'][0];g=obj['visible_geometry_candidate'];g['centroid_world_m']=np.mean(g['world_vertices_sample_m'],axis=0).tolist()
            g['mask_depth_transform_contract']={'mask_size_wh':[160,120],'depth_size_wh':[80,60],
                'A_depth_from_mask_coordinate_model':resize_affine(raster_size,source_size).tolist()}
            with np.load(f['inputs']['depth_npz'],allow_pickle=False) as archive:source_depth=np.asarray(archive['depth'][0],np.float32)
            yy,xx=np.mgrid[4:56:4,4:76:4];zz=source_depth[yy,xx]
            camera_points=np.column_stack([(xx.ravel()-K[2])*zz.ravel()/K[0],(yy.ravel()-K[3])*zz.ravel()/K[1],zz.ravel()])
            T=np.asarray(frame['camera']['T_world_camera_metric']);world_points=camera_points@T[:3,:3].T+T[:3,3]
            g.update(camera_vertices_sample_m=camera_points.tolist(),world_vertices_sample_m=world_points.tolist(),
                centroid_world_m=world_points.mean(0).tolist(),intrinsics_fx_fy_cx_cy=K.tolist())
            obj['reconstructed_geometry_pose']={'rotation_world_from_canonical_matrix':np.eye(3).tolist(),
                'translation_world_m':g['centroid_world_m'],'pose_source':'synthetic_centroid_initializer'}
    base=put(root/'base_annotations.json',{'frames':f['frames'],'synthetic_fixture':True})
    eligibility_path=root/'eligibility.json';raster=root/'eligibility_frame3.npz'
    with np.load(f['inputs']['depth_npz'],allow_pickle=False) as z:depth=np.asarray(z['depth'][0],np.float32)
    classes=np.full(depth.shape,CLASS_CODES['clear_no_hand_overlap_candidate'],np.uint8);eligible=np.ones(depth.shape,bool)
    np.savez(raster,schema=np.array(RASTER_SCHEMA),frame_idx=np.array(3),class_code=classes,metric_candidate_eligible=eligible)
    mask=Path(track['3']['mask_path'])
    put(eligibility_path,{'schema':ELIG_SCHEMA,'case':'synthetic_v2','object_id':object_id,'track_id':'synthetic_track',
        'class_codes':CLASS_CODES,'policy':{'physical_authority':False},'inputs':{
            'owned_sam2_track':rec(owned_track),'depth_npz':rec(f['inputs']['depth_npz']),
            'base_annotations':rec(base),'calibration_contract':rec(camera)},'frames':[{
                'frame_idx':3,'status':'native_depth_classified','owned_mask':rec(mask),'source_mask':rec(mask),
                'depth_array_sha256':array_sha256(depth),'raster':rec(raster),
                'A_depth_from_mask':resize_affine(raster_size,source_size).tolist(),
                'K_depth':[[90.,0.,39.5],[0.,90.,29.5],[0.,0.,1.]],
                'class_code_array_sha256':array_sha256(classes),'metric_candidate_array_sha256':array_sha256(eligible)}]})
    g=f['frames'][0]['objects'][0]['visible_geometry_candidate'];g.update(source_mask_path=str(mask),
        native_depth_metric_eligibility={'manifest':str(eligibility_path),'raster':str(raster)})
    annotations=f['inputs']['annotations'];put(annotations,{'frames':f['frames'],
        'v19_visible_geometry_adapter':{'anchor_frame_idx':3},'synthetic_fixture':True})
    adapter=json.loads(f['inputs']['p09_adapter_report'].read_text());adapter['parameters']['native_depth_metric_eligibility_enabled']=True
    adapter.update(case='synthetic_v2',object_id=object_id,track_id='synthetic_track',inputs={
        'metric_eligibility_manifest':str(eligibility_path),'metric_eligibility_manifest_sha256':sha(eligibility_path),
        'depth_npz':str(f['inputs']['depth_npz']),'sam2_track':str(owned_track),
        'base_annotations':str(base),'calibration_contract':str(camera)})
    put(f['inputs']['p09_adapter_report'],adapter)
    # Actual retained-depth replay + PnP; only pretrained feature extraction/matching is mocked.
    run_fixture(f,matches=not empty_rgb)
    bridge=root/'source_bridge.npz';bridge_ids=[];sides=[];world=[]
    for i,fid in enumerate(ids):
        for side in ('left','right'):
            if hands[side+'_valid'][i]:bridge_ids.append(fid);sides.append(side);world.append(hands[side+'_vertices_world_m'][i])
    np.savez(bridge,frame_idx=bridge_ids,hand_side=sides,vertices_current_v18_world_from_hawor_projection_relift_m=world)
    app_rows=[]
    for i,row in enumerate(raw['frames']):
        src=Path(track[str(ids[i])]['mask_path']);dest=root/f'appearance_{ids[i]}.png';shutil.copyfile(src,dest)
        app_rows.append({'frame_idx':ids[i],'time_s':times[i],'status':'available','anchor_eligible':i==0,
            'raw_rgb_path':row['raw_frame_path'],'raw_rgb_sha256':row['raw_frame_sha256'],'source_size_wh':[160,120],
            'appearance_mask_path':str(dest),'appearance_mask_sha256':sha(dest),'source_mask_path':str(src),'source_mask_sha256':sha(src)})
    appearance=put(root/'appearance.json',{'object_id':object_id,'source_annotations':str(annotations),
        'source_annotations_sha256':sha(annotations),'metric_owned_masks_consumed':False,'frames':app_rows})
    decision=put(root/'appearance_decision.json',{'object_id':object_id,'selected_frame_idx':3})
    qc,_,_=native_fixture(root/'synthetic_native',Path(app_rows[0]['raw_rgb_path']),Path(app_rows[0]['appearance_mask_path']))
    visual=build_visual_geometry(annotations=annotations,appearance=appearance,decision=decision,native_qc=qc,
        output_dir=root/'visual_observation',object_id=object_id,run_root=root)
    visual_report=json.loads(visual.read_text())
    results=schema_results(ids,object_id);sources={}
    for name,result in results.items():sources[name]=rec(put(root/'judgments'/f'{name}.json',result))
    interaction=put(root/'interaction.json',assemble_interaction_evidence(object_id=object_id,frame_ids=ids,
        batches=[{'frame_ids':ids,'judgments':results,'sources':sources}]))
    from observations.interaction.evidence import accepted_first_surface_query_raster
    affine=resize_affine(source_size,raster_size)
    support,query,diagnostics=accepted_first_surface_query_raster(g,{'A_mask_from_source_coordinate_model':affine.tolist()},(120,160),12.)
    first_surface=root/'synthetic_first_surface.npz'
    np.savez(first_surface,depth_mask_plane_m=query,support_mask=support,A_mask_from_source_coordinate_model=affine)
    support_image=root/'synthetic_first_surface_support.png';cv2.imwrite(str(support_image),support.astype(np.uint8)*255)
    ownership_rows=[];factor_rows=[]
    for side in ('left','right'):
        ownership_rows.append({'factor_family':'visible_ownership','frame_idx':3,'hand_side':side,'target_entity_id':object_id,
            'state':'active_visible_ownership','image_plane_transform':{'A_mask_from_source_coordinate_model':affine.tolist()},
            'depth_order_first_surface_npz_path':str(first_surface),'depth_order_first_surface_npz_sha256':sha(first_surface),
            'depth_order_query_support_mask_path':str(support_image),'depth_order_query_support_mask_sha256':sha(support_image),
            'depth_order_first_surface':diagnostics})
        factor_rows.append({'factor_family':'contact_patch','frame_idx':3,'hand_side':side,'target_entity_id':object_id,
            'state':'active_contact_patch','weight':1.,'contact_patch_prior_probability':.5,
            'contact_patch_band_m':.06,'contact_patch_support_uncertainty_m':.06,
            'contact_anchor_residual_allowed':False,'synthetic_fixture':True})
    factors=put(root/'factors.json',{'factor_rows':factor_rows,'ownership_rows':ownership_rows,'synthetic_fixture':True})
    pose_mesh=Path(old['sources']['object_mesh']['path'])
    completion=put(root/'completion.json',{'outputs':{'pose_hypothesis_mesh_labeled':str(pose_mesh),
        'collision_eligible_mesh_labeled':str(pose_mesh)},'geometry_readiness':{'signed_geometry_ready':False}})
    runtime=put(root/'input/runtime_input_contract.json',{'run_id':'synthetic-v2-acceptance','run_root':str(root),
        'input_video':str(video),'input_video_sha256':sha(video),'synthetic_fixture':True})
    metadata={name:put(root/(name+'.json'),{'synthetic_fixture':True,'role':name,'model_execution_performed':False})
        for name in ['code_manifest','model_asset_manifest','object_plan','candidate_identity']}
    put(metadata['object_plan'],{'objects':[{'object_id':object_id,'track_id':'synthetic_track',
        'description':'synthetic numeric inspection fixture, not a real object prediction'}],'synthetic_fixture':True})
    put(metadata['candidate_identity'],{'object_id':object_id,'track_id':'synthetic_track','synthetic_fixture':True})
    records={
        'runtime_input_contract':(runtime,'runtime_input_contract'),'code_manifest':(metadata['code_manifest'],'code_manifest'),
        'model_asset_manifest':(metadata['model_asset_manifest'],'model_asset_manifest'),'raw_video':(video,'raw_video'),
        'raw_frames':(raw_manifest,'raw_frame_manifest'),'camera_calibration':(camera,'camera_calibration'),
        'conditioned_depth':(f['inputs']['depth_npz'],'conditioned_camera_z_depth'),'hawor_world':(mano,'raw_hawor_world_mano'),
        'mano_bridge':(bridge,'raw_mano_bridge'),'object_plan':(metadata['object_plan'],'object_plan'),
        'candidate_identity':(metadata['candidate_identity'],'object_candidate_identity'),
        'raw_object_masks':(raw_track,'raw_object_masks'),'owned_object_masks':(owned_track,'depth_order_owned_object_masks'),
        'metric_eligibility':(eligibility_path,'metric_surface_eligibility'),'visible_annotations':(annotations,'visible_metric_annotations'),
        'surface_report':(f['inputs']['p09_adapter_report'],'visible_surface_report'),'completion_report':(completion,'reconstructed_shape_contract'),
        'reconstructed_mesh':(pose_mesh,'reconstructed_object_mesh'),'collision_mesh':(pose_mesh,'collision_object_mesh'),
        'appearance_masks':(appearance,'appearance_mask_observations'),'appearance_anchor_decision':(decision,'appearance_anchor_decision'),
        'interaction_judgments':(interaction,'contact_visibility_occlusion_evidence'),'interaction_factors':(factors,'qualified_contact_visibility_factors'),
        'rgb_source_npz':(f['root']/'rgb.npz','rgb_source_arrays'),'rgb_source_report':(f['root']/'rgb.json','rgb_source_report'),
        'visual_geometry_report':(visual,'visual_geometry_report'),
        'visual_mesh_camera':(Path(visual_report['output_mesh']['path']),'visual_full_metric_camera_mesh'),
        'visual_native_mesh':(Path(visual_report['source_native_mesh']['path']),'visual_full_native_mesh'),
        'visual_native_qc':(qc,'visual_native_model_prior')}
    output=root/'source_observations.json'
    payload=write_source_observation_manifest(output,run_root=root,case='synthetic_v2',object_id=object_id,
        track_id='synthetic_track',frame_ids=ids,time_s=times,source_timing={'frame_count':2,'synthetic_fixture':True},
        artifacts=records,bundle_version=2)
    # Test provenance only; no production run/producer receipt is forged.
    payload['synthetic_fixture']=True;output.write_text(json.dumps(payload,indent=2)+'\n')
    (root/'SYNTHETIC_FIXTURE.md').write_text('# Synthetic interface fixture, NOT milk/model prediction\nTwo sparse-ID frames, synthetic full-778 hand arrays and native grid.\nNo MANO/SAM3D/LightGlue weights or Astra calls.\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(build_fixture(args.output_dir))
