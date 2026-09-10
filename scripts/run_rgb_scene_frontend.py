#!/usr/bin/env python3
"""Fixed P01/P06-P11 producer entry. No arbitrary command or cache-registration API."""
import argparse
from pathlib import Path
from v19_runtime_contracts import load, interpreter, execution_target, runtime_environment
from rgb_scene_runtime import run_command
from rgb_scene_runtime import SceneLineage as RunLineage
from rgb_scene_recovery import RECOVERY_FILE, require_recovery_command, require_recovery_scope

ROOT=Path(__file__).resolve().parents[1]
PHASES=('P01','P06','P07','P08','P09_DEPTH_ORDER','P09-propose','P09','P11')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=PHASES,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--runtime-config',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--object-id')
    p.add_argument('--recovery-contract',type=Path)
    a=p.parse_args();root=a.run_root.resolve();cfg=load(a.runtime_config)
    recovery_requested=False
    if a.recovery_contract:
        if a.recovery_contract.resolve()!=root/RECOVERY_FILE:raise ValueError('recovery contract must be the applied fixed P08 transition')
        require_recovery_scope('frontend',a.phase);recovery_requested=True
    execution_target(cfg,a.gpu)
    asset_role='P09-propose' if a.phase=='P09_DEPTH_ORDER' else a.phase
    lineage=RunLineage(root,ROOT,a.runtime_config,asset_role=asset_role);contract=lineage.contract;case=contract['case']
    if recovery_requested:require_recovery_command(lineage.recovery,'frontend',a.phase)
    python=interpreter(cfg['main_python']);env=runtime_environment(cfg,a.gpu)
    def run(script,args,outputs,extra=(),heavy=False,exe=python):
        run_command(a.phase,[exe,ROOT/'scripts'/script,*args],outputs,root=root,cwd=ROOT,env=env,
                    lineage=lineage,extra_inputs=extra,gpu=a.gpu if heavy else None)
    video=Path(contract['input_video']);manifest=root/'input/raw_frame_manifest/manifest.json'
    if a.phase=='P01':
        run('build_v19_raw_frame_manifest.py',['--video',video,'--output-dir',manifest.parent,'--render-width',960],[manifest])
        return
    lineage.require(manifest);raw=load(manifest);end=max(f['frame_idx'] for f in raw['frames'])
    plan=root/'measurements/object_candidates/object_plan_agent.json';lineage.require(plan)
    objects=load(plan)['objects']
    if len(objects)!=1:raise ValueError('first integration requires one selected rigid object')
    obj=objects[0]
    object_id=obj['object_id']
    if a.object_id is not None and a.object_id!=object_id:raise ValueError('P05/object ID mismatch')
    if not object_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in object_id) or object_id in {'.','..'}:
        raise ValueError('unsafe object ID')
    track=obj.get('track_id',object_id)
    if not track or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in track) or track in {'.','..'}:
        raise ValueError('unsafe track ID')
    detector=root/'measurements/object_candidates/object_box_prompts_owlv2'
    sam2=root/'measurements/object_tracks/sam2_owlv2_box_points'
    calibration=root/'state/calibration/v19_camera_calibration_contract.json'
    base=root/'state/base_annotations/annotations_v19_base.json'
    depth=root/'state/calibration/depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
    owned=root/f'measurements/object_tracks/depth_order_owned/{track}'
    owned_track=owned/'sam2_track.json'
    ownership_report=owned/'depth_order_ownership_report.json'
    metric_eligibility=owned/'native_depth_metric_eligibility.json'
    visible=root/f'measurements/object_geometry/visible_geometry/{object_id}'
    proposals=root/f'measurements/object_geometry/anchor_candidates/{object_id}'
    anchor_path=root/f'state/anchor_decisions/{object_id}.json'
    if a.phase=='P06':
        interval=obj['detector_active_interval']
        if isinstance(interval,dict):start,stop=interval['start_frame'],interval['end_frame']
        else:start,stop=interval
        prompts=[v for text in obj['detector_text_prompts'] for v in ('--text-prompt',text)]
        run('build_v19_owlv2_object_box_prompts.py',[
            '--raw-frame-manifest',manifest,'--output-root',detector,
            '--review-dir',root/'renders/review_frames/P06_owlv2_object_boxes',
            '--case-id',case,'--object-id',object_id,'--track-id',track,'--description',obj['description'],
            *prompts,'--prompt-frames',','.join(map(str,obj['detector_prompt_frames'])),
            '--active-start',start,'--active-end',stop,'--owlv2-model',cfg['owlv2_model'],
            '--box-threshold',.03,'--device','cuda','--box-only'],
            [detector/object_id/'object_point_prompts_vlm.json'],[plan],True,interpreter(cfg['owlv2_python']))
    elif a.phase=='P07':
        run('run_sam2_vlm_points_multiobject.py',['--clip',video,'--point-root',detector,
            '--output-root',sam2,'--checkpoint',cfg['sam2_checkpoint'],'--frame-start',0,'--frame-end',end,
            '--sam2-image-width',960,'--render-width',960,'--use-positive-prompt-box',
            '--prompt-box-pad-ratio',.18,'--prompt-box-min-pad-px',24],
            [sam2/track/'sam2/sam2_track.json'],[plan],True)
    elif a.phase=='P08':
        run('build_v19_base_annotations.py',['--case',case,'--raw-frame-manifest',manifest,
            '--hawor-npz',root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz',
            '--object-plan',plan,'--sam2-output-root',sam2,'--calibration-contract',calibration,
            '--output-dir',base.parent],[base,base.parent/'v19_mano_bridge_from_hawor_world.npz'])
    elif a.phase=='P09_DEPTH_ORDER':
        run('rgb_scene_depth_order.py',[
            '--case',case,'--object-id',object_id,'--track-id',track,
            '--base-annotations',base,
            '--raw-sam2-track',sam2/track/'sam2/sam2_track.json',
            '--depth-npz',depth,'--calibration-contract',calibration,
            '--mano-bridge',base.parent/'v19_mano_bridge_from_hawor_world.npz',
            '--hawor-npz',root/'measurements/hand_candidates/hawor_world/hawor_world_hands.npz',
            '--output-dir',owned],
            [owned_track,ownership_report,metric_eligibility])
    elif a.phase in {'P09-propose','P09'}:
        lineage.require(owned_track);lineage.require(ownership_report);lineage.require(metric_eligibility)
        if a.phase=='P09-propose':
            out=proposals;extra=['--propose-anchor-candidates-only','--anchor-candidate-count',12];inputs=[]
            outputs=[out/'anchor_candidate_proposals.json',out/'anchor_candidate_review.jpg']
        else:
            lineage.require(anchor_path);anchor=load(anchor_path)['selected_anchor_frame_idx']
            out=visible;extra=['--anchor-frame',anchor,'--require-anchor-frame'];inputs=[anchor_path]
            outputs=[out/'v19_visible_geometry_depth_fused_report.json',out/'annotations_v19_visible_geometry.json']
        run('build_v19_visible_geometry_from_sam2_depth.py',['--case',case,'--track-id',track,'--object-id',object_id,
            '--raw-frame-manifest',manifest,'--sam2-root',sam2,'--sam2-track-json',owned_track,
            '--metric-eligibility-manifest',metric_eligibility,
            '--depth-npz',depth,'--output-dir',out,
            '--base-annotations',base,'--calibration-contract',calibration,'--depth-image-plane','source_rgb',
            '--mask-image-plane','sam2_mask','--object-plan',plan,'--preserve-source-index','--no-exclude-hand-bboxes',
            *extra],outputs,[ownership_report,metric_eligibility,*inputs])
    else:
        lineage.require(anchor_path);anchor=load(anchor_path)['selected_anchor_frame_idx']
        branch=root/f'state/physical_branch_decisions/{object_id}.json';lineage.require(branch)
        if load(branch).get('physical_branch')!='rigid':raise ValueError('rigid P11 requires explicit P10 rigid branch')
        out=root/'measurements/geometry_completion/rigid_evidence'
        run('build_v18_compact_rigid_evidence_bundle.py',['--case',case,'--object-id',object_id,
            '--annotations',visible/'annotations_v19_visible_geometry.json',
            '--depth-fused-report',visible/'v19_visible_geometry_depth_fused_report.json',
            '--output-root',out,'--selected-frame-idx',anchor,
            '--selection-note','current-run visual anchor decision'],
            [out/case/object_id/'evidence_bundle/evidence_bundle_report.json'],[anchor_path,branch])


if __name__=='__main__':main()
