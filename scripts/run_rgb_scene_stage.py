#!/usr/bin/env python3
"""Independent raw RGB scene stage entry. Model-assisted decisions are interleaved."""
from __future__ import annotations
import argparse, json, os, runpy, sys
from pathlib import Path
from v19_runtime_contracts import load,interpreter,execution_target,runtime_environment,validate_timeline
from rgb_scene_runtime import SceneLineage,run_command,verify_resume
from rgb_scene_candidates import CandidatePaths, load_candidate
from rgb_scene_recovery import RECOVERY_FILE, require_recovery_command, require_recovery_scope

ROOT=Path(__file__).resolve().parents[1]
STAGES=('frontend','camera','geometry','initial-pose','refined-pose','appearance','native-shape','shape','visual-fit','export')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--phase',choices=('P01','P06','P07','P08','P09_DEPTH_ORDER','P09-propose','P09','P11'))
    for n in ('run-root','runtime-config'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--resume-contract',type=Path)
    p.add_argument('--recovery-contract',type=Path)
    p.add_argument('--candidate-context',type=Path,help='candidate_contract.json for an isolated continuation')
    a=p.parse_args();root=a.run_root.resolve();cfg=load(a.runtime_config)
    candidate=None; candidate_value=None
    if a.candidate_context:
        candidate,candidate_value=load_candidate(a.candidate_context)
        if candidate.run_root!=root: raise ValueError('candidate context belongs to another run')
    if a.resume_contract and a.recovery_contract:p.error('--resume-contract and --recovery-contract are mutually exclusive')
    recovery_requested=False
    if a.recovery_contract:
        if a.recovery_contract.resolve()!=root/RECOVERY_FILE:raise ValueError('recovery contract must be the applied fixed P08 transition')
        require_recovery_scope(a.stage,a.phase);recovery_requested=True
    execution_target(cfg,a.gpu)
    os.environ.pop('RGB_SCENE_VERIFIED_RESUME',None)
    if a.resume_contract:
        verify_resume(a.resume_contract,root,ROOT)
        os.environ['RGB_SCENE_VERIFIED_RESUME']='1'
    if a.stage=='frontend':
        if not a.phase:p.error('frontend requires --phase')
        script=ROOT/'scripts/run_rgb_scene_frontend.py'
        sys.argv=[str(script),'--phase',a.phase,'--run-root',str(root),'--runtime-config',str(a.runtime_config),'--gpu',str(a.gpu)]
        if a.recovery_contract:sys.argv.extend(['--recovery-contract',str(a.recovery_contract)])
        if a.candidate_context:sys.argv.extend(['--candidate-context',str(a.candidate_context)])
        runpy.run_path(str(script),run_name='__main__');return
    role={'camera':'camera','geometry':'geometry_dispatch','native-shape':'geometry','initial-pose':'pose','refined-pose':'pose','appearance':'P09','shape':'pose','visual-fit':'demo','export':'export'}[a.stage]
    lineage=SceneLineage(root,ROOT,a.runtime_config,asset_role=role,candidate=candidate)

    if recovery_requested:require_recovery_command(lineage.recovery,a.stage,a.phase)
    if candidate is not None and a.stage=='camera':
        raise ValueError('candidate continuation preserves source camera/depth/MANO prefix')
    if a.stage=='camera':
        import rgb_scene_camera
        rgb_scene_camera._LINEAGE=lineage
        rgb_scene_camera.camera_stage(a,cfg,root);return
    def out(rel): return candidate.path(rel) if candidate is not None else root/rel
    if candidate is not None:
        selection_path=out('state/object_selection.json')
        if not selection_path.is_file(): raise ValueError('candidate object selection judgment is required')
    else:
        selection_path=root/'state/object_selection.json'
    selection=load(lineage.require(selection_path));obj=selection['object_id'];case=selection['case'];anchor=(candidate_value['selected_frame_idx'] if candidate_value is not None else selection['anchor_frame']);track=selection.get('track_id',obj)

    import re
    if not all(isinstance(v,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',v) for v in (obj,case,track)):raise ValueError('unsafe case/object/track')
    plan=load(lineage.require(root/'measurements/object_candidates/object_plan_agent.json'))
    if case!=lineage.contract['case'] or len(plan['objects'])!=1 or plan['objects'][0]['object_id']!=obj or plan['objects'][0].get('track_id',obj)!=track:raise ValueError('selection differs from current object plan')
    def rp(rel): return out(rel)
    decision=load(lineage.require(out(f'state/anchor_decisions/{obj}.json') if candidate is not None else root/f'state/anchor_decisions/{obj}.json'))
    if decision['selected_anchor_frame_idx']!=anchor:raise ValueError('selection anchor mismatch')
    ann=out(f'measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json')
    lineage.require(ann)

    raw=root/'input/raw_frame_manifest/manifest.json';lineage.require(raw)
    scene=out(f'state/scene/{obj}')
    depth=root/'state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
    geom=out(f'measurements/geometry_completion/sam3d_{obj}')
    original=geom/'completion_report.json';completion=geom/'anchor_bound_completion_report.json'
    p14=out(f'measurements/pose_fits/{obj}/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json')
    rgb=out(f'measurements/pose_fits/{obj}/scene_rgb/edges.npz')
    posterior=scene/'refined_pose.json'
    appearance=out(f'measurements/demo_visual/{obj}/appearance_evidence/appearance_mask_manifest.json')
    visual_decision=out(f'state/appearance_decisions/{obj}.json')
    native=out(f'measurements/scene_native/{obj}')
    shape=out(f'measurements/scene_shape/{obj}');binding=shape/'pose_mesh.json'
    fit=out(f'measurements/scene_fit/{obj}')
    hawor=root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz'
    bridge=root/'state/base_annotations/v19_mano_bridge_from_hawor_world.npz'
    judgments=out(f'state/agent_interaction_judgments/{obj}.json')
    native_reuse=(candidate_value or {}).get('native_mode')=='reuse'

    raw=root/'input/raw_frame_manifest/manifest.json';lineage.require(raw)
    ids=validate_timeline(load(ann),load(raw));lo,hi=min(ids),max(ids)
    py=interpreter(cfg['main_python'])
    def script(name):return ROOT/'scripts'/name
    def run(key,argv,outputs,heavy=False,extra=(),environment=None):
        run_command(key,argv,outputs,root=root,cwd=ROOT,env=environment or runtime_environment(cfg,a.gpu),lineage=lineage,gpu=a.gpu if heavy else None,extra_inputs=extra)
    if a.stage=='geometry':
        evidence=(root/f'measurements/geometry_completion/rigid_evidence/{case}/{obj}/evidence_bundle/evidence_bundle_report.json' if candidate is None else out(f'measurements/geometry_completion/rigid_evidence/{case}/{obj}/evidence_bundle/evidence_bundle_report.json'))
        lineage.require(evidence)
        geometry_argv=[py,script('run_rgb_scene_reconstruction.py'),'--run-root',root,'--evidence-report',evidence,'--annotations',ann,'--output-dir',geom,'--runtime-config',a.runtime_config,'--gpu',a.gpu]
        if candidate is not None: geometry_argv.extend(['--candidate-context',str(candidate.contract_path)])
        run('scene_geometry',geometry_argv,[original],True)
    elif a.stage=='initial-pose':
        run('scene_anchor_binding',[py,script('bind_v19_completion_anchor.py'),'--completion-report',original,'--annotations',ann,'--object-id',obj,'--output',completion],[completion])
        run('scene_source_pose',[py,script('fit_v18_compact_rigid_object_pose.py'),'--annotations',ann,'--completion-report',completion,'--object-id',obj,'--output-dir',p14.parent],[p14])
    elif a.stage=='refined-pose':
        run('scene_rgb',[interpreter(cfg['rgb_python']),script('build_rgb_scene_edges.py'),'--annotations',ann,'--depth-npz',depth,'--initial-pose-report',p14,'--object-id',obj,'--frame-start',lo,'--frame-end',hi,'--keyframe-seconds',.5,'--source-gap-seconds',2.,'--keyframe-neighbor-span',4,'--lightglue-root',cfg['lightglue_root'],'--device','cuda:0','--output-npz',rgb,'--output-json',rgb.with_suffix('.json')],[rgb,rgb.with_suffix('.json')],True,extra=[ann.parent/'v19_visible_geometry_adapter_report.json',raw])
        run('scene_refined_pose',[py,script('rgb_scene_pose_io.py'),'--annotations',ann,'--source-pose',p14,'--rgb-edges',rgb,'--rgb-report',rgb.with_suffix('.json'),'--object-id',obj,'--anchor',anchor,'--run-root',root,'--output',posterior],[posterior])
    elif a.stage=='appearance':
        sam2=root/f'measurements/object_tracks/sam2_owlv2_box_points/{track}/sam2/sam2_track.json'
        ownership=root/f'measurements/object_tracks/depth_order_owned/{track}/depth_order_ownership_report.json'
        lineage.require(ownership)
        run('scene_appearance',[py,script('rgb_scene_depth_order_appearance.py'),'--run-root',root,'--object-id',obj,'--case-id',case,'--track-id',track,'--raw-frame-manifest',raw,'--annotations',ann,'--raw-sam2-track',sam2,'--depth-order-report',ownership,'--output-dir',appearance.parent],[appearance,appearance.with_name('anchor_decision.json')],extra=[ownership,sam2])
    elif a.stage=='native-shape':
        lineage.require(visual_decision);lineage.require(appearance);lineage.require(posterior)
        chosen=load(visual_decision)
        if chosen.get('object_id')!=obj:raise ValueError('appearance selection object mismatch')
        idx=chosen['selected_frame_idx']
        if candidate_value is not None and idx != candidate_value['selected_frame_idx']:
            raise ValueError('candidate appearance anchor must equal candidate selected frame')
        masks={f['frame_idx']:f for f in load(appearance)['frames']};mask=masks[idx]
        if mask.get('status')!='available':raise ValueError('selected appearance mask unavailable')
        image=lineage.require(mask['raw_rgb_path']);maskpath=lineage.require(mask['appearance_mask_path'])
        name=f'{case}_{obj}_frame_{idx}'
        if native_reuse:
            qc=native/'imported/qc_sam3d_objects_mesh_v7.json'
            lineage.require(qc)
        else:
            argv=[interpreter(cfg['sam3d_python']),script('remote_run_sam3d_objects_mesh_v7.py'),'--repo',cfg['sam3d_repo'],'--config',cfg['sam3d_config'],'--moge-checkpoint',cfg['sam3d_moge_checkpoint'],'--dinov2-repo',cfg['sam3d_dinov2_repo'],'--dinov2-checkpoint',cfg['sam3d_dinov2_checkpoint'],'--case',f'{name}|{image}|{maskpath}|42','--output-dir',native]
            run('scene_native_appearance',argv,[native/name/'qc_sam3d_objects_mesh_v7.json'],True,extra=[appearance,visual_decision,posterior,image,maskpath],environment={**runtime_environment(cfg,a.gpu),**cfg['sam3d_environment']})

    elif a.stage=='shape':
        lineage.require(visual_decision)
        idx=load(visual_decision)['selected_frame_idx']
        if candidate_value is not None and idx != candidate_value['selected_frame_idx']:
            raise ValueError('candidate shape must consume selected candidate frame')
        if native_reuse:
            qc=native/'imported/qc_sam3d_objects_mesh_v7.json'
        else:
            qc=native/f'{case}_{obj}_frame_{idx}/qc_sam3d_objects_mesh_v7.json'

        run('scene_refined_shape',[py,script('bind_rgb_scene_shape.py'),'--posterior',posterior,'--annotations',ann,'--appearance',appearance,'--decision',visual_decision,'--native-qc',qc,'--run-root',root,'--object-id',obj,'--output-dir',shape],[binding,shape/'object_canonical.ply'])
    elif a.stage=='visual-fit':
        lineage.require(judgments)
        run('scene_visual_fit',[interpreter(cfg['demo_python']),script('fit_rgb_scene_visual.py'),'--run-root',root,'--annotations',ann,'--pose-report',binding,'--mano-bridge',bridge,'--hawor-params',hawor,'--mano-models',cfg['mano_models'],'--object-id',obj,'--appearance-mask-manifest',appearance,'--interaction-judgments',judgments,'--device','cuda:0','--output-dir',fit],[fit/'demo_state.npz',fit/'fit_report.json'],True)
    elif a.stage=='export':
        export_dir=out('final')
        argv=[interpreter(cfg['demo_python']),script('export_rgb_scene.py'),'--run-root',root,'--state',fit/'demo_state.npz','--binding',binding,'--fit-report',fit/'fit_report.json','--output-dir',export_dir]
        if candidate is not None: argv.extend(['--candidate-context',str(candidate.contract_path)])
        run('scene_export',argv,[export_dir/'annotation.json',export_dir/'world.rrd'],extra=[judgments,raw,root/'state/calibration/v19_camera_calibration_contract.json'])

if __name__=='__main__':main()
