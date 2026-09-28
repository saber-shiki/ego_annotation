"""Trusted-UniDepth profile: adapt SAM3D scale/translation, never the depth reference."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import json,time,traceback,gc
import numpy as np
import cv2
from preprocessing import SCHEMA
from preprocessing.inputs import load_inputs,sha
from preprocessing.geometry import TriangleMesh
from preprocessing.factors import hand_depths,contour_factors
from preprocessing.tracking import bootstrap
from preprocessing.visible_surface import (VisibleSurfaceRegistration,cloud_digest,
    camera_translations_to_world,select_scale_keyframes)


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def code_identity():return {str(p.relative_to(Path(__file__).parent)):sha(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}


def run_visible_surface(source,output,*,width=480,registration_width=192,max_nfev=24,scale_keyframes=5,render=True,render_width=320):
    source=Path(source).resolve();output=Path(output).resolve()
    if output==source.parent or output.is_relative_to(source.parent):raise ValueError('new derived output must be outside source')
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=code_identity();cv2.setNumThreads(1);cv2.setRNGSeed(47)
    config={'profile':'visible-surface','width':width,'registration_width':registration_width,'max_nfev':max_nfev,'scale_keyframes':scale_keyframes,
            'depth_reference_fixed':True,'rotation_refit_in_alignment':False,'MoGe_modified':False,'anisotropic_shape_fit':False,
            'RGB_refinement':'isolated_output_repair_on_joint_evidence_contradiction','missing_metric_depth':'latent_world_correction_bridge',
            'motion_acceleration_sigma_m_s2':8.}
    write_json(output/'started.json',{'source':str(source),'source_sha256':sha(source),'code_sha256':code,'configuration':config})
    try:
        inputs=load_inputs(source,width,read_owned_masks=True);digest=cloud_digest(inputs.clouds)
        # Immutable arrays make accidental reverse-fitting of the reference fail immediately.
        for q in inputs.clouds:q.setflags(write=False)
        hd=hand_depths(inputs,registration_width);n=len(inputs.ids);a=inputs.anchor
        if not inputs.cloud_valid[a] or len(inputs.clouds[a])<6:raise ValueError('appearance anchor lacks qualified visible surface')
        print(f'loaded {n} frames; anchor={int(inputs.ids[a])}; raw UniDepth retained unchanged',flush=True)
        # Pose rotation is estimated once from the original template, with the existing
        # tracker raster. Isotropic scale fitting must not rebind and flip that rotation.
        tracking_hd=hd if registration_width==128 else hand_depths(inputs,128)
        R,t,tracks,tracking=bootstrap(inputs,tracking_hd,guarded_output=True)
        initial_R=R.copy();initial_t=t.copy()
        np.savez_compressed(output/'bootstrap.npz',frame_idx=inputs.ids,object_rotation=R,object_translation_world=t)
        write_json(output/'tracking.json',tracking)
        anchor=VisibleSurfaceRegistration(inputs,R,t,hd,[a],fit_scale=True,initial_scale=1.,max_points=768)
        s0,tc0,anchor_report=anchor.solve(outer_iterations=12,max_nfev=max_nfev)
        ta=camera_translations_to_world(inputs,[a],tc0,t)
        # Projection-preserving initialization only; subsequent t variables are free.
        camera_centers=inputs.cameras[:,:3,3]
        t=camera_centers+s0*(t-camera_centers)
        t+=ta[a]-t[a]
        del anchor;gc.collect()
        write_json(output/'anchor_alignment.json',anchor_report)
        print(f'anchor scale={s0:.6f}; merit {anchor_report["before"]["merit_m"]:.6f}->{anchor_report["after"]["merit_m"]:.6f}m',flush=True)
        keys=select_scale_keyframes(inputs,tracking,maximum=scale_keyframes)
        weights=np.array([3. if i==a else .5 for i in keys])
        multi=VisibleSurfaceRegistration(inputs,R,t,hd,keys,fit_scale=True,initial_scale=s0,weights=weights,max_points=512)
        scale,key_t,multi_report=multi.solve(outer_iterations=10,max_nfev=max_nfev)
        t_seed=camera_translations_to_world(inputs,keys,key_t,t)
        write_json(output/'shared_scale_alignment.json',multi_report)
        del multi;gc.collect()
        print(f'shared scale={scale:.6f}; keys={[int(inputs.ids[i]) for i in keys]}',flush=True)
        # Scale is now frozen for the full timeline. The only variables are translations.
        timeline=VisibleSurfaceRegistration(inputs,R,t_seed,hd,list(range(n)),fit_scale=False,initial_scale=scale,max_points=160,temporal=True,motion_reference_world=t)
        _,all_t,timeline_report=timeline.solve(outer_iterations=6,max_nfev=max_nfev)
        t_final=camera_translations_to_world(inputs,range(n),all_t,t_seed)
        write_json(output/'timeline_surface_alignment.json',timeline_report)
        del timeline;gc.collect()
        print(f'timeline merit {timeline_report["before"]["merit_m"]:.6f}->{timeline_report["after"]["merit_m"]:.6f}m',flush=True)
        # Evaluate before/after on identical raw observed samples, with this operation isolated.
        before=VisibleSurfaceRegistration(inputs,initial_R,initial_t,hd,list(range(n)),fit_scale=False,initial_scale=1.,max_points=512)
        after=VisibleSurfaceRegistration(inputs,R,t_final,hd,list(range(n)),fit_scale=False,initial_scale=scale,max_points=512)
        measurement={'reference':'fixed_UniDepth_qualified_deoccluded_surfaces','before':before.measure(1.,before.t0),'after':after.measure(scale,after.t0),
                     'not_GT_depth':True,'source_point_digest_before':digest,'source_point_digest_after':cloud_digest(inputs.clouds)}
        if measurement['source_point_digest_after']!=digest:raise RuntimeError('raw surface arrays changed')
        write_json(output/'visible_surface_metrics.json',measurement)
        del before,after;gc.collect()
        if not np.array_equal(R,initial_R):raise RuntimeError('scale/translation registration changed rotation proposals')
        vertices=inputs.vertices*scale;faces=inputs.faces
        _,old_contours=contour_factors(TriangleMesh(inputs.vertices,faces),initial_R,initial_t,inputs,hd)
        _,new_contours=contour_factors(TriangleMesh(vertices,faces),R,t_final,inputs,hd)
        write_json(output/'image_diagnostics.json',{'source':old_contours,'coarse':new_contours,'definition':'source-side visible mask diagnostic, not physical GT accuracy'})
        archive={'frame_idx':inputs.ids,'time_s':inputs.times,'T_world_camera':inputs.cameras,'intrinsics_source':inputs.source_K,'image_size_wh':inputs.source_wh,
            'object_rotation':R,'object_translation_world':t_final,'object_valid':np.ones(n,bool),'vertices_canonical':vertices,'faces':faces,
            'source_vertices_canonical':inputs.vertices,'source_rotation':inputs.source_R,'source_translation_world':inputs.source_t,
            'alignment_initial_rotation':initial_R,'alignment_initial_translation_world':initial_t,
            'source_pose_available':inputs.source_valid,'source_metric_eligible':inputs.cloud_valid,
            'translation_depth_inferred':~inputs.cloud_valid,
            'relative_depth_bias_m':np.zeros(n),'relative_depth_scale':np.ones(n),'relative_depth_support':np.zeros(n,int),
            'uniform_object_scale':np.asarray(scale),'shared_shape_matrix':np.eye(3)*scale,'shared_log_stretch':np.zeros(2),
            'original_native_pivot_camera':inputs.pivot_camera}
        for side,hand in inputs.hands.items():
            for key in ('vertices','joints','faces','valid'):archive[side+'_'+key]=hand[key]
        np.savez_compressed(output/'coarse_scene.npz',**archive)
        old_scale=float(inputs.geometry_report['scene_similarity']['scene_similarity_scale'])
        manifest={'schema':SCHEMA,'alignment_profile':'visible-surface','configuration':config,'source_observations':str(source),'source_bundle_sha256':sha(source),
            'object_id':inputs.bundle.manifest['entity_slots']['objects'][0],'frame_ids':inputs.ids.tolist(),'time_s':inputs.times.tolist(),
            'contains_optimized_state':True,'contains_policy_results':False,'prediction_side_only':True,'GT_or_CAD_inputs':False,'additional_camera_inputs':False,
            'absolute_metric_accuracy_certified':False,'metric_reference':'UniDepth raw qualified visible surface accepted as reference by user assumption',
            'depth_correction_enabled':False,'depth_source_unchanged':True,'source_cloud_digest':digest,'MoGe_modified':False,
            'original_camera_origin_scene_scale':old_scale,'residual_uniform_object_scale':scale,'combined_native_length_scale':old_scale*scale,
            'translation_refit_independently':False,'translation_parameterization':'independent_lateral_and_observed_depth_with_latent_gap_depth_correction','rotation_changed_by_registration':False,'rotation_source':'original_template_RGB_bootstrap_frozen_before_alignment','shared_scale_across_frames':True,'scale_reference_frames':[int(inputs.ids[i]) for i in keys],
            'appearance_anchor_frame_idx':int(inputs.ids[a]),'canonical_frame':'centered_native_appearance_camera_axes','canonical_native_pivot_camera_m':inputs.pivot_camera.tolist(),
            'shape_role':'full_SAM3D_prior_uniformly_registered_to_fixed_visible_surface','shape_topology_preserved':True,'anisotropic_shape_fit':False,
            'signed_geometry_ready':False,'collision_authority_changed':False,'source_MANO_unchanged':True,'inferred_frames_are_not_observations':True,
            'source_reads':inputs.receipt.files,'code_sha256':code,'arrays':{'path':str(output/'coarse_scene.npz'),'sha256':sha(output/'coarse_scene.npz')},
            'seconds_before_render':time.monotonic()-start,'limitations':['reference accuracy inherits UniDepth','hidden shape is still generated prior','registration holds orientation hypotheses fixed','not an RL training adapter or contact certificate']}
        inputs.receipt.check()
        if code!=code_identity():raise RuntimeError('preprocessing code changed during run')
        write_json(output/'coarse_scene.json',manifest)
        if render:
            from preprocessing.render import render_scene
            render_scene(inputs,archive,output/'renders',render_width)
        inputs.receipt.check()
        if cloud_digest(inputs.clouds)!=digest or code!=code_identity():raise RuntimeError('source arrays or code changed')
        write_json(output/'completed.json',{'status':'complete_derived_coarse_scene','profile':'visible-surface','frames':n,'seconds':time.monotonic()-start,
            'rendered':render,'source_unchanged':True,'raw_depth_and_surfaces_unchanged':True,'absolute_metric_accuracy_certified':False})
        print(json.dumps({'output':str(output),'uniform_scale':scale,'seconds':time.monotonic()-start}),flush=True)
        return manifest
    except Exception as exc:
        write_json(output/'failed.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
