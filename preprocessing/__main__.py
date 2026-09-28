"""Independent prediction-only coarse scene CLI; never invokes observe, PPO or GT."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,traceback
import numpy as np
import cv2
from preprocessing import SCHEMA
from preprocessing.inputs import load_inputs,sha
from preprocessing.geometry import shape_matrix,TriangleMesh
from preprocessing.factors import hand_depths,contour_factors
from preprocessing.tracking import bootstrap
from preprocessing.optimizer import refine


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def code_identity():return {str(p.relative_to(Path(__file__).parent)):sha(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}


def run(source,output,*,width=480,contour_width=128,max_nfev=18,render=True,render_width=320,enable_relocalization=True,enable_geometry=False,profile='legacy',registration_width=192,scale_keyframes=5):
    if profile=='visible-surface':
        if enable_geometry:raise ValueError('visible-surface profile does not use the geometry relocalization experiment')
        from preprocessing.surface_pipeline import run_visible_surface
        return run_visible_surface(source,output,width=width,registration_width=registration_width,max_nfev=max_nfev,scale_keyframes=scale_keyframes,render=render,render_width=render_width)
    if profile!='legacy':raise ValueError('unknown preprocessing profile')
    output=Path(output).resolve();source=Path(source).resolve()
    if enable_geometry and not enable_relocalization:raise ValueError('geometry experiment requires the relocalization stage')
    if output==source.parent or output.is_relative_to(source.parent):raise ValueError('derived output must not be placed inside immutable source')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();code=code_identity();cv2.setNumThreads(1);cv2.setRNGSeed(47)
    write_json(output/'started.json',{'source':str(source),'source_sha256':sha(source),'code_sha256':code,'input_mode':'monocular_RGB_only','state':'running','configuration':{'width':width,'contour_width':contour_width,'max_nfev':max_nfev,'enable_relocalization':enable_relocalization,'enable_geometry':enable_geometry}})
    try:
        inputs=load_inputs(source,width);print(f'loaded {len(inputs.ids)} frames, {len(inputs.faces)} triangles',flush=True)
        start=time.monotonic();hd=hand_depths(inputs,contour_width);print(f'hand occlusion geometry: {time.monotonic()-start:.1f}s',flush=True)
        source_mesh=TriangleMesh(inputs.vertices,inputs.faces)
        _,source_contours=contour_factors(source_mesh,inputs.source_R,inputs.source_t,inputs,hd)
        start=time.monotonic();R0,t0,tracks,tracking=bootstrap(inputs,hd);print(f'bootstrap: {len(tracks.frames)} observations, {time.monotonic()-start:.1f}s',flush=True)
        np.savez_compressed(output/'bootstrap.npz',frame_idx=inputs.ids,object_rotation=R0,object_translation_world=t0)
        if enable_relocalization:
            from preprocessing.relocalization import relocalize
            if enable_geometry:
                from preprocessing.shape_relocalization import geometry_relocalize
                start=time.monotonic()
                R0,t0,tracks,tracking,shape_relocalization=geometry_relocalize(inputs,source_mesh,hd,R0,t0,tracks,tracking)
                print(f'geometry relocalization: {shape_relocalization["state"]}; {time.monotonic()-start:.1f}s',flush=True)
            else:
                shape_relocalization={'applied':False,'state':'disabled_experiment_failed_precision_validation'}
            write_json(output/'shape_relocalization.json',shape_relocalization)
            start=time.monotonic()
            R0,t0,tracks,tracking,relocalization=relocalize(inputs,source_mesh,hd,R0,t0,tracks,tracking)
            write_json(output/'relocalization.json',relocalization)
            np.savez_compressed(output/'relocalized.npz',frame_idx=inputs.ids,object_rotation=R0,object_translation_world=t0)
            print(f'descriptor relocalization: {len(relocalization["accepted_frames"])} accepted frames; {relocalization["backend_factor_rows"]} loop factors; {time.monotonic()-start:.1f}s',flush=True)
        write_json(output/'tracking.json',tracking)
        R,t,bias,shape,diagnostics,calibration=refine(inputs,R0,t0,tracks,hd,max_nfev)
        write_json(output/'relative_depth_calibration.json',calibration['report'])
        S=shape_matrix(shape,inputs.shape_basis);vertices=inputs.vertices@S.T
        mesh=TriangleMesh(vertices,inputs.faces);_,final_contours=contour_factors(mesh,R,t,inputs,hd)
        if not np.allclose(R@R.transpose(0,2,1),np.eye(3),atol=1e-6) or np.any(np.linalg.det(R)<.999):raise RuntimeError('invalid solved rotations')
        if not np.isfinite(t).all() or not np.isfinite(vertices).all():raise RuntimeError('invalid solved geometry')
        archive={'frame_idx':inputs.ids,'time_s':inputs.times,'T_world_camera':inputs.cameras,'intrinsics_source':inputs.source_K,'image_size_wh':inputs.source_wh,
            'object_rotation':R,'object_translation_world':t,'object_valid':np.ones(len(R),bool),
            'vertices_canonical':vertices,'faces':inputs.faces,'source_vertices_canonical':inputs.vertices,'source_rotation':inputs.source_R,'source_translation_world':inputs.source_t,
            'source_pose_available':inputs.source_valid,'source_metric_eligible':inputs.cloud_valid,'relative_depth_bias_m':bias,'relative_depth_scale':calibration['scale'],'relative_depth_support':calibration['support'],'shared_log_stretch':shape,'shared_shape_matrix':S,
            'original_native_pivot_camera':inputs.pivot_camera}
        for side,hand in inputs.hands.items():
            for key in ('vertices','joints','faces','valid'):archive[side+'_'+key]=hand[key]
        np.savez_compressed(output/'coarse_scene.npz',**archive)
        write_json(output/'solver.json',diagnostics)
        write_json(output/'image_diagnostics.json',{'source':source_contours,'coarse':final_contours,'definition':'prediction-side visible mask IoU excludes nearer source MANO pixels; not GT pose accuracy'})
        inputs.receipt.check()
        if code!=code_identity():raise RuntimeError('preprocessing code changed during run')
        manifest={'schema':SCHEMA,'source_observations':str(source),'source_bundle_sha256':sha(source),'object_id':inputs.bundle.manifest['entity_slots']['objects'][0],
            'frame_ids':inputs.ids.tolist(),'time_s':inputs.times.tolist(),'contains_optimized_state':True,'contains_policy_results':False,'prediction_side_only':True,
            'input_mode':'monocular_RGB_only','GT_or_CAD_inputs':False,'additional_camera_inputs':False,'absolute_metric_accuracy_certified':False,
            'metric_gauge':'source_native_monocular_prediction_scale; no independent absolute scale measurement',
            'relative_depth_method':'rigid_material_chord_lengths; no free per-frame additive bias',
            'descriptor_relocalization_enabled':enable_relocalization,'experimental_geometry_relocalization_enabled':enable_geometry,
            'canonical_frame':'centered_native_appearance_camera_axes','canonical_native_pivot_camera_m':inputs.pivot_camera.tolist(),
            'appearance_anchor_frame_idx':int(inputs.ids[inputs.anchor]),'shape_role':'shared_volume_preserving_low_rank_visual_hypothesis_not_physical_surface',
            'shape_topology_preserved':True,'signed_geometry_ready':False,'collision_authority_changed':False,'source_MANO_unchanged':True,
            'pose_measured_validity_preserved':True,'inferred_frames_are_not_observations':True,
            'source_reads':inputs.receipt.files,'code_sha256':code,'configuration':{'width':width,'contour_width':contour_width,'max_nfev':max_nfev,'enable_relocalization':enable_relocalization,'enable_geometry':enable_geometry},
            'arrays':{'path':str(output/'coarse_scene.npz'),'sha256':sha(output/'coarse_scene.npz')},'seconds_before_render':time.monotonic()-started,
            'limitations':['absolute scale remains monocular-source uncertain','generated hidden geometry is not measured collision authority','low-rank shared stretch cannot restore missing topology','no reviewed RL training adapter implied']}
        write_json(output/'coarse_scene.json',manifest)
        if render:
            from preprocessing.render import render_scene
            render_scene(inputs,archive,output/'renders',render_width)
        inputs.receipt.check()
        if code!=code_identity():raise RuntimeError('preprocessing code changed during render')
        write_json(output/'completed.json',{'status':'complete_derived_coarse_scene','frames':len(inputs.ids),'source_unchanged':True,'seconds':time.monotonic()-started,'rendered':render,'absolute_metric_accuracy_certified':False})
        print(json.dumps({'output':str(output),'seconds':time.monotonic()-started}),flush=True)
        return manifest
    except Exception as exc:
        write_json(output/'failed.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'source_not_modified':True})
        raise


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-observations',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--profile',choices=['legacy','visible-surface'],default='legacy',help='visible-surface fixes UniDepth and aligns one shared object scale')
    p.add_argument('--registration-width',type=int,default=192);p.add_argument('--scale-keyframes',type=int,default=5)
    p.add_argument('--width',type=int,default=480);p.add_argument('--contour-width',type=int,default=128);p.add_argument('--max-nfev',type=int,default=18)
    p.add_argument('--render-width',type=int,default=320);p.add_argument('--no-render',action='store_true')
    p.add_argument('--no-relocalization',action='store_true',help='explicit local-tracker-only ablation')
    p.add_argument('--experimental-geometry-relocalization',action='store_true',help='opt-in failed-validation geometry hypothesis experiment; not the default')
    a=p.parse_args(argv)
    if a.width<96 or a.contour_width<48 or a.registration_width<48 or not 1<=a.scale_keyframes<=12 or a.max_nfev<1 or a.render_width<96 or a.render_width%2:raise ValueError('invalid raster/solver budget')
    run(a.source_observations,a.output_dir,width=a.width,contour_width=a.contour_width,max_nfev=a.max_nfev,render=not a.no_render,render_width=a.render_width,enable_relocalization=not a.no_relocalization,enable_geometry=a.experimental_geometry_relocalization,profile=a.profile,registration_width=a.registration_width,scale_keyframes=a.scale_keyframes)

if __name__=='__main__':cli()
