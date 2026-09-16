"""Legacy numerical optimization entrypoints for source-observation bundles.

These commands preserve the grouped numerical reference implementation. Training
and policy rollout do not import this module or require its outputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np

from legacy_optimize.problems import StandaloneScene
from legacy_optimize.solve import solve
from observations.common import build_nodes, pose_from_values
from utils.data import RGBEdge, RGBReprojectionFactor, indexed, RKEY, TKEY, load
from utils.state import sha, in_run, assert_prediction_only, require_run_artifact, RunState
from utils.runtime import run_model


def load_edges(npz_path, report_path, source_path, annotations_path, nodes, root):
    ids=[n.frame_idx for n in nodes]
    source=load(source_path);object_id=source['object_id']
    npz_receipt=require_run_artifact(npz_path,root,'rgb_edges',object_id=object_id,frame_ids=ids)
    json_receipt=require_run_artifact(report_path,root,'rgb_edge_report',object_id=object_id,frame_ids=ids)
    if npz_receipt['producer_id']!=json_receipt['producer_id']:
        raise ValueError('NPZ/report producer pairing mismatch')
    report = load(report_path)
    with np.load(npz_path, allow_pickle=False) as z: data = dict(z)
    meta = json.loads(str(data['metadata'].reshape(-1)[0]))
    if meta.get('object_id')!=object_id:
        raise ValueError('RGB object identity mismatch')
    if any(report.get(key)!=value for key,value in meta.items()):
        raise ValueError('RGB NPZ/report metadata pairing mismatch')
    for key, path in [('initial_pose_report', source_path), ('annotations', annotations_path)]:
        if meta['input_sha256'][key] != sha(path) or Path(meta[key]).resolve() != Path(path).resolve():
            raise ValueError('current RGB source hash/identity mismatch')
    for key in ('depth_npz',):
        path = in_run(meta[key], root)
        require_run_artifact(path,root,'depth',frame_ids=ids)
        if sha(path) != meta['input_sha256'][key]: raise ValueError('RGB metric source mutated')
    positions = {n.frame_idx: i for i,n in enumerate(nodes)}
    source_rows = indexed(load(source_path)['pose_rows'])
    edges, projections = [], []
    offsets = data['reprojection_evidence_offsets']
    if len(offsets) != len(data['source_frame_idx'])+1 or offsets[0] != 0 or offsets[-1] != len(data['reprojection_canonical_points']) or np.any(np.diff(offsets)<0):
        raise ValueError('invalid RGB evidence offsets')
    rows = {(r['source_frame_idx'],r['target_frame_idx']):r for r in report['rows']}
    if len(rows)!=len(report['rows']):raise ValueError('duplicate RGB report edge')
    for a,b in zip(data['source_frame_idx'],data['target_frame_idx']):
        if int(a) not in positions or int(b) not in positions or (int(a),int(b)) not in rows:
            raise ValueError('RGB edge outside source timeline or report')
    for k, (a,b) in enumerate(zip(data['source_frame_idx'],data['target_frame_idx'])):
        a,b=int(a),int(b)
        if not data['accepted'][k]: continue
        row=rows[a,b]
        q=float(row['inlier_fraction'])*math.exp(-float(row['reprojection_median_px'])/2.)
        q *= max(.02,min(1.,float(row['source_3d_conditioning'])/.02))
        q *= .45 if row['topological_role']=='adjacent' else 1.
        e=RGBEdge(a,b,data['rotation_source_to_target_world'][k],data['translation_source_to_target_world_m'][k],max(.001,q),row)
        edges.append(e)
        lo,hi=map(int,offsets[k:k+2])
        if hi>lo:
            r,t=pose_from_values(source_rows[a][RKEY],source_rows[a][TKEY])
            world=data['reprojection_canonical_points'][lo:hi]@r.T+t
            take=np.linspace(0,len(world)-1,min(96,len(world)),dtype=int)
            camera=data['reprojection_target_camera'][k]
            projections.append(RGBReprojectionFactor(a,b,positions[a],positions[b],world[take],
                data['reprojection_target_uv'][lo:hi][take],data['reprojection_target_intrinsics'][k],
                camera[:3,:3],camera[:3,3],data['reprojection_weights'][lo:hi][take],e.weight))
    return edges,projections


def refine(annotations_path, source_path, edge_path, edge_report, object_id, anchor, output, root):
    root=Path(root).resolve()
    for path in (annotations_path,source_path,edge_path,edge_report):in_run(path,root)
    in_run(output,root,must_exist=False)
    if output.exists(): raise ValueError('create-only posterior required')
    before=sha(source_path)
    source=load(source_path); annotations=load(annotations_path)
    assert_prediction_only(source, label='scene source prior', object_id=object_id)
    assert_prediction_only(annotations, label='scene observed geometry', object_id=object_id)
    nodes,times=build_nodes(annotations,source,object_id)
    ids=[n.frame_idx for n in nodes]
    require_run_artifact(annotations_path,root,'observations',object_id=object_id,frame_ids=ids)
    require_run_artifact(source_path,root,'source_pose',object_id=object_id,frame_ids=ids)
    if not any(n.frame_idx==anchor and n.metric_observation for n in nodes):
        raise ValueError('gauge anchor must have current retained metric support')
    edges,reprojections=load_edges(edge_path,edge_report,source_path,annotations_path,nodes,root)
    poses,diagnostics=solve(StandaloneScene(nodes,times,edges,anchor,reprojections=reprojections))
    seed_rows = {row['frame_idx']: row for row in diagnostics['local_RGB_rotation']['seed_rows']}
    rows=[]
    for n,stamp in zip(nodes,times):
        r,t=poses[n.frame_idx]
        rows.append({'frame_idx':n.frame_idx,'time_s':float(stamp),RKEY:r.tolist(),TKEY:t.tolist(),
                     'pose_source':'global_SO3_observed_RGB_SE3_visual_posterior',
                     'metric_observation':n.metric_observation,
                     'depth_feasibility_uncertain':n.frame_idx in diagnostics['depth_restoration']['affected_frame_ids'],
                     'uncertainty':max(seed_rows[n.frame_idx]['uncertainty'], .9 if n.frame_idx in diagnostics['depth_restoration']['affected_frame_ids'] else 0., n.uncertainty, .6 if n.frame_idx in diagnostics['relative_evidence_disconnected_ids'] else .2),
                     'uncertainty_kind':'heuristic_support_uncertainty_not_marginal_covariance',
                     'relative_evidence_connected':n.frame_idx not in diagnostics['relative_evidence_disconnected_ids'],
                     'local_RGB_seed':seed_rows[n.frame_idx],
                     'source_prior':n.source_row})
    report={'schema':'rgb_scene_refined_pose_v1','object_id':object_id,
            'prediction_side_only':True,'annotation_ready':False,'physical_contact_authority':False,
            'canonical_gauge':{'name':'initial_completed_canonical','anchor_frame_idx':anchor,'anchor_reason':'current_visual_selected_retained_metric_source'},
            'source_prior':{'path':str(source_path),'sha256':before},
            'evidence':{'annotations_sha256':sha(annotations_path),'rgb_npz_sha256':sha(edge_path),'rgb_report_sha256':sha(edge_report)},
            'source_pose_rows':copy.deepcopy(source['pose_rows']),'pose_rows':rows,'solver':diagnostics}
    if sha(source_path)!=before: raise ValueError('source prior mutated')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    return report


def fit_prepared(problem, run_root):
    from envs.qualification.object_geometry import validate_posterior
    from actions.filters.visibility import validate_visibility, apply_display_visibility, POLICY
    binding=load(problem.pose_report)
    if binding.get('schema')!='rgb_scene_visual_refined_binding_v1':
        raise ValueError('typed visual-refined binding required')
    reference=binding['refined_posterior']
    if sha(reference['path'])!=reference['sha256']:
        raise ValueError('refined posterior changed')
    posterior=validate_posterior(load(reference['path']),problem.object_id,run_root)
    if binding['pose_rows']!=posterior['pose_rows']:
        raise ValueError('binding discarded refined trajectory')
    judgment=load(problem.interaction_judgments)
    validate_visibility(judgment,[r['frame_idx'] for r in binding['pose_rows']],problem.object_id)
    solve(problem)
    path=problem.output_dir/'demo_state.npz'
    with np.load(path,allow_pickle=False) as archive:
        state=apply_display_visibility(dict(archive),judgment,problem.object_id)
    np.savez_compressed(path,**state)
    path=problem.output_dir/'fit_report.json'
    report=load(path)
    report['display_visibility']={'policy':POLICY,
        'judgment_sha256':sha(problem.interaction_judgments),'source_geometry_mutated':False}
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def _bundle_artifact(bundle: dict, manifest_path: Path, name: str) -> Path:
    record = bundle.get('artifacts', {}).get(name)
    if not isinstance(record, dict) or not isinstance(record.get('path'), str):
        raise ValueError(f'source observation bundle lacks required artifact: {name}')
    path = (manifest_path.parent / record['path']).resolve()
    if not path.is_file() or record.get('sha256') != sha(path):
        raise ValueError(f'source observation artifact changed or is missing: {name}')
    return path


def _legacy_reference_assets(bundle: dict, manifest_path: Path, assets_manifest: Path) -> dict:
    contract_path = _bundle_artifact(bundle, manifest_path, 'runtime_input_contract')
    contract = load(contract_path)
    expected = contract.get('assets_manifest_sha256')
    if expected is not None and expected != sha(assets_manifest):
        raise ValueError('legacy reference assets manifest differs from source-observation run')
    assets = load(assets_manifest)
    if not isinstance(assets, dict):
        raise ValueError('legacy reference assets manifest must be a JSON object')
    return assets


def _reference_worker(command, *, root: Path, output: Path, assets: dict, gpu: int, gpu_uuid: str | None, label: str):
    code_root = Path(__file__).resolve().parent
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONPATH': str(code_root) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    if label == 'appearance_native':
        overlay = assets.get('sam3d_environment', {})
        if not isinstance(overlay, dict):
            raise ValueError('sam3d_environment must be a mapping when declared')
        environment.update({str(key): str(value) for key, value in overlay.items()})
    with output.open('x', encoding='utf-8') as log:
        resources = run_model(command, cwd=code_root, environment=environment, gpu=gpu,
                              expected_uuid=gpu_uuid, min_free_mib=int(assets.get('sam3d_min_free_mib', 12000)),
                              stdout=log)
    return resources


def reference_from_source_observations(source_manifest: Path, assets_manifest: Path, output_dir: Path, *,
                                       gpu: int, gpu_uuid: str | None=None, device: str='cpu',
                                       iterations: int=240, render_width: int | None=None) -> dict:
    """Run the retained P14→RGB/P15→shape→visual reference from verified sources.

    This command deliberately lives outside ``main.py``. It records every new
    reference stage under the original immutable source run and never feeds its
    results back into the observation bundle, training data, or rollout path.
    """
    from observations.manifest import load_source_observation_manifest
    from observations.object.pose import fit_observed_pose
    from legacy_optimize.problems import StandaloneShape, PreparedVisualFit

    source_manifest = Path(source_manifest).resolve()
    bundle = load_source_observation_manifest(source_manifest)
    root = source_manifest.parent.resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError('legacy reference output must be created inside the source-observation run root')
    if output_dir.exists():
        raise FileExistsError(f'legacy reference output is create-only: {output_dir}')
    if not isinstance(gpu, int) or gpu < 0:
        raise ValueError('legacy reference requires an explicit nonnegative GPU index')
    if iterations < 1:
        raise ValueError('legacy reference visual iterations must be positive')

    assets_manifest = Path(assets_manifest).resolve()
    assets = _legacy_reference_assets(bundle, source_manifest, assets_manifest)
    required_assets = ('rgb_python', 'lightglue_root', 'sam3d_python', 'sam3d_repo', 'sam3d_config',
                       'sam3d_moge_checkpoint', 'sam3d_dinov2_repo', 'sam3d_dinov2_checkpoint',
                       'mano_models')
    missing = [key for key in required_assets if key not in assets]
    if missing:
        raise ValueError(f'legacy reference assets manifest lacks: {missing}')

    annotations = _bundle_artifact(bundle, source_manifest, 'visible_annotations')
    completion = _bundle_artifact(bundle, source_manifest, 'completion_report')
    depth = _bundle_artifact(bundle, source_manifest, 'conditioned_depth')
    raw_manifest = _bundle_artifact(bundle, source_manifest, 'raw_frames')
    appearance = _bundle_artifact(bundle, source_manifest, 'appearance_masks')
    decision = _bundle_artifact(bundle, source_manifest, 'appearance_anchor_decision')
    mano_bridge = _bundle_artifact(bundle, source_manifest, 'mano_bridge')
    hawor = _bundle_artifact(bundle, source_manifest, 'hawor_world')
    judgments = _bundle_artifact(bundle, source_manifest, 'interaction_judgments')
    object_id = str(bundle['object_id'])
    frame_ids = [int(value) for value in bundle['frame_ids']]
    if not frame_ids:
        raise ValueError('source observation timeline is empty')

    output_dir.mkdir(parents=True)
    ledger = RunState(root)
    p14_dir = output_dir / 'p14_observed_pose'
    fit_observed_pose(['--annotations', str(annotations), '--completion-report', str(completion),
                       '--object-id', object_id, '--output-dir', str(p14_dir)])
    source_pose = p14_dir / 'v18_compact_rigid_object_pose_fit_report.json'
    ledger.record('legacy_reference_observed_pose', [(source_pose, 'source_pose')],
                  object_id=object_id, frame_ids=frame_ids, command=['ego-optimize', 'legacy-reference'])

    rgb_dir = output_dir / 'rgb_edges'
    edges = rgb_dir / 'edges.npz'
    edge_report = rgb_dir / 'edges.json'
    rgb_command = [str(assets['rgb_python']), '-m', 'observations.object.rgb_tracking',
        '--annotations', str(annotations), '--raw-frame-manifest', str(raw_manifest), '--depth-npz', str(depth),
        '--initial-pose-report', str(source_pose), '--object-id', object_id, '--frame-start', str(min(frame_ids)),
        '--frame-end', str(max(frame_ids)), '--keyframe-seconds', '0.5', '--source-gap-seconds', '2.0',
        '--keyframe-neighbor-span', '4', '--lightglue-root', str(assets['lightglue_root']), '--assets-manifest',
        str(assets_manifest), '--device', 'cuda:0', '--output-npz', str(edges), '--output-json', str(edge_report)]
    resources = _reference_worker(rgb_command, root=root, output=output_dir / 'rgb_edges.log', assets=assets,
                                  gpu=gpu, gpu_uuid=gpu_uuid, label='rgb_edges')
    if resources is not None:
        gpu_uuid = resources['uuid']
    ledger.record('legacy_reference_rgb_edges', [(edges, 'rgb_edges'), (edge_report, 'rgb_edge_report')],
                  object_id=object_id, frame_ids=frame_ids, command=rgb_command)

    posterior = output_dir / 'p15_refined_pose.json'
    completion_payload = load(completion)
    anchor = int((completion_payload.get('inputs') or {}).get('selected_anchor_atomic_binding', {}).get('selected_frame_idx', frame_ids[0]))
    refine(annotations, source_pose, edges, edge_report, object_id, anchor, posterior, root)
    ledger.record('legacy_reference_refined_pose', [(posterior, 'candidate_pose')],
                  object_id=object_id, frame_ids=frame_ids, command=['ego-optimize', 'refine-pose'])

    appearance_payload = load(appearance)
    decision_payload = load(decision)
    selected = int(decision_payload['selected_frame_idx'])
    appearance_rows = indexed(appearance_payload['frames'])
    if selected not in appearance_rows:
        raise ValueError('appearance decision selected a frame absent from the source appearance manifest')
    selected_row = appearance_rows[selected]
    native_name = f"{bundle['case']}_{object_id}_appearance_{selected}"
    native_root = output_dir / 'appearance_native'
    native_qc = native_root / native_name / 'qc_sam3d_objects_mesh_v7.json'
    native_command = [str(assets['sam3d_python']), '-m', 'observations.object.sam3d', '--repo', str(assets['sam3d_repo']),
        '--config', str(assets['sam3d_config']), '--moge-checkpoint', str(assets['sam3d_moge_checkpoint']),
        '--dinov2-repo', str(assets['sam3d_dinov2_repo']), '--dinov2-checkpoint', str(assets['sam3d_dinov2_checkpoint']),
        '--case', f"{native_name}|{selected_row['raw_rgb_path']}|{selected_row['appearance_mask_path']}|42",
        '--output-dir', str(native_root)]
    resources = _reference_worker(native_command, root=root, output=output_dir / 'appearance_native.log', assets=assets,
                                  gpu=gpu, gpu_uuid=gpu_uuid, label='appearance_native')
    if resources is not None:
        gpu_uuid = resources['uuid']
    ledger.record('legacy_reference_appearance_native', [(native_qc, 'native_geometry')],
                  object_id=object_id, frame_ids=[selected], command=native_command)

    shape_dir = output_dir / 'shape_binding'
    binding = solve(StandaloneShape(posterior, annotations, appearance, decision, native_qc,
                                    shape_dir, object_id, root))
    binding_path = shape_dir / 'pose_mesh.json'
    if binding is None or not binding_path.is_file():
        raise RuntimeError('legacy shape binding did not produce pose_mesh.json')
    ledger.record('legacy_reference_shape_binding', [(binding_path, 'pose_mesh_binding')],
                  object_id=object_id, frame_ids=frame_ids, command=['ego-optimize', 'shape-bind'])

    visual_dir = output_dir / 'visual_fit'
    visual = fit_prepared(PreparedVisualFit(annotations, binding_path, mano_bridge, hawor,
                                             Path(assets['mano_models']), visual_dir, object_id, appearance,
                                             judgments, device=device, iterations=iterations,
                                             render_width=render_width), root)
    report = {'schema': 'ego.annotation.legacy_reference/1.0.0', 'source_observations': str(source_manifest),
              'source_observations_sha256': sha(source_manifest), 'object_id': object_id,
              'output_dir': str(output_dir), 'stages': {'p14_source_pose': str(source_pose),
                  'rgb_edges': str(edges), 'p15_refined_pose': str(posterior),
                  'shape_binding': str(binding_path), 'visual_state': str(visual_dir / 'demo_state.npz'),
                  'visual_report': str(visual_dir / 'fit_report.json')},
              'visual_report': visual, 'reference_only': True,
              'training_or_rollout_dependency': False,
              'note': 'This legacy result is a reference candidate. It does not alter source observations or grant policy/physical acceptance.'}
    report_path = output_dir / 'legacy_reference.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    ledger.record('legacy_reference_complete', [(report_path, 'legacy_reference')],
                  object_id=object_id, frame_ids=frame_ids, command=['ego-optimize', 'legacy-reference'])
    return report


def formal_tail(spec):
    from types import SimpleNamespace
    from legacy_optimize.problems import FormalObjectTrajectory, FormalManoInterval
    from observations.interaction.evidence import interaction_build, interaction_parse_args, build_geometric_constraints
    from observations.object.signed_geometry import signed_proxy_build, signed_proxy_parse_args
    from envs.qualification.hand_acceptance import accept_mano_surface
    from render import build_formal_render_state, formal_state_parse_args, assemble_formal_layers, render_formal_rrd, render_formal_video
    root=Path(spec['run_root']).resolve();out=Path(spec['output']).resolve()
    in_run(out,root,must_exist=False);out.mkdir(parents=True,exist_ok=False)
    obj=spec['object_id'];case=spec['case'];ann=in_run(spec['annotations'],root)
    source=in_run(spec['source_pose'],root);completion=in_run(spec['completion_report'],root)
    hawor=in_run(spec['hawor_npz'],root);judgments=in_run(spec['interaction_judgments'],root)
    depth=[in_run(path,root) for path in spec['depth_npz']]
    images=load(ann)['frames'];frame_ids=[row['frame_idx'] for row in images]
    profile=spec.get('profile','formal')
    if profile not in {'formal','hot3d_shared','rgb_sam3d'}:raise ValueError('unknown retained formal path')
    if profile=='rgb_sam3d':
        from observations.object.rgb_tracking import initialize
        nodes,_=build_nodes(load(ann),load(source),obj)
        edge_path=in_run(spec['rgb_edges'],root);edge_report=in_run(spec['rgb_report'],root)
        load_edges(edge_path,edge_report,source,ann,nodes,root)
        with np.load(edge_path,allow_pickle=False) as archive:arrays=dict(archive)
        initialized=initialize(load(source),load(ann),obj,arrays,load(edge_report)['rows'],int(spec['anchor_frame']))
        source=out/'rgb_initialized_pose.json';source.write_text(json.dumps(initialized,indent=2,allow_nan=False))
    object_settings=spec.get('object_problem',{})
    object_problem=FormalObjectTrajectory(ann,source,completion,out/'pose',obj,
        profile=object_settings.get('profile','observed_only'),
        image_factors=in_run(object_settings['image_factors'],root) if object_settings.get('image_factors') else None,
        nonpenetration_targets=in_run(object_settings['nonpenetration_targets'],root) if object_settings.get('nonpenetration_targets') else None)
    pose_report=solve(object_problem)
    pose_path=out/'pose/v19_rigid_object_pose_graph_report.json'
    physical_completion=completion
    if profile=='hot3d_shared':
        signed_proxy_build(signed_proxy_parse_args(list(map(str,['--case',case,'--object-id',obj,
            '--annotations',ann,'--direct-pose-report',source,'--pose-report',pose_path,
            '--observed-completion-report',completion,'--depth-npz',depth[0],'--output-dir',out/'signed_geometry']))))
        physical_completion=out/'signed_geometry/shared_signed_geometry_completion_report.json'
    factor_args=interaction_parse_args(list(map(str,['--annotations',ann,'--case',case,
        '--target-entity-id','object:'+obj.removeprefix('object:'),'--frame-span',min(frame_ids),max(frame_ids),
        '--output-root',out/'interaction','--agent-interaction-judgment',judgments,
        '--require-camera-mano-contract-aligned'])))
    factor=interaction_build(factor_args)
    factor_path=out/'interaction'/case/'v19_visible_contact_ownership_factor_report.json'
    factor_path.parent.mkdir(parents=True,exist_ok=True);factor_path.write_text(json.dumps(factor,indent=2))
    mesh=load(completion)['outputs'].get('pose_hypothesis_mesh_labeled') or load(completion)['outputs']['completed_mesh_labeled']
    mano=FormalManoInterval(case,obj,ann,pose_path,Path(mesh),physical_completion,depth,out/'mano',
        Path(spec['wilor_root']),Path(spec['mano_left']),min(frame_ids),max(frame_ids),
        profile='hot3d_shared' if profile=='hot3d_shared' else spec.get('mano_problem',{}).get('profile','formal'),
        device=spec.get('device','cpu'),factor_reports=[factor_path,*[in_run(path,root) for path in spec.get('mano_problem',{}).get('factor_reports',[])]],
        hand_ray_prior=in_run(spec['mano_problem']['hand_ray_prior'],root) if spec.get('mano_problem',{}).get('hand_ray_prior') else None,
        numeric_options=spec.get('mano_problem',{}).get('numeric_options',{}))
    solve(mano)
    contact_state=out/'mano'/case/'v18_joint_mano_interval_trajectory_state.json'
    accepted=out/'accepted_mano.json'
    accept_mano_surface(list(map(str,['--contact-state',contact_state,'--hawor-npz',hawor,
        '--annotations',ann,'--case',case,'--object-id',obj,'--accept-signed-full-mano','--output',accepted])))
    build_geometric_constraints(list(map(str,['--annotations',ann,'--hawor-npz',hawor,
        '--pose-report',pose_path,'--completion-report',physical_completion,
        '--output-dir',out/'constraints','--object-id',obj])))
    constraints=out/'constraints/v18_mano_object_constraint_state.json'
    state=out/'render_state.json'
    build_formal_render_state(formal_state_parse_args(list(map(str,['--case',case,'--object-id',obj,
        '--annotations',ann,'--pose-report',pose_path,'--completion-report',completion,
        '--physical-completion-report',physical_completion,'--constraint-report',constraints,'--temporal-mano-state',accepted,'--output',state]))))
    package=assemble_formal_layers(SimpleNamespace(render_state=state,annotations=ann,hawor_npz=hawor,
        interaction_judgments=judgments,output_dir=out/'package',run_root=root,demo_state=None,demo_report=None,allow_external_replay=False))
    package_path=out/'package/annotation.json'
    if isinstance(package,(str,Path)):package_path=Path(package)
    render_formal_rrd(package_path,out/'world.rrd',layer='prediction')
    render_formal_video(package_path,in_run(spec['source_video'],root),out/'video')
    if spec.get('backend_geometry') is not None:
        from render import render_dual_backend
        render_dual_backend(package_path,in_run(spec['source_video'],root),spec['backend_geometry'],out/'dual_display')
    acceptance=load(accepted)['full_mano_acceptance']
    result={'profile':profile,'source_pose_sha256':sha(source),'pose_report':str(pose_path),
        'mano_candidate':str(contact_state),'mano_acceptance':acceptance,'package':str(package_path),
        'physical_pose_accepted':pose_report.get('annotation_ready') is True,
        'physical_mano_accepted':acceptance.get('accepted') is True}
    (out/'result.json').write_text(json.dumps(result,indent=2));return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command', required=True)
    reference=commands.add_parser('legacy-reference', help='run the retained P14→RGB/P15→shape→visual reference from source_observations.json')
    reference.add_argument('--source-observations', type=Path, required=True)
    reference.add_argument('--assets-manifest', type=Path, required=True)
    reference.add_argument('--output-dir', type=Path, required=True)
    reference.add_argument('--gpu', type=int, required=True)
    reference.add_argument('--gpu-uuid')
    reference.add_argument('--device', default='cpu')
    reference.add_argument('--iterations', type=int, default=240)
    reference.add_argument('--render-width', type=int)
    observed=commands.add_parser('observed-pose',help='build the preserved P14 direct pose reference')
    for name in ('annotations','completion-report','output-dir'):
        observed.add_argument('--'+name,type=Path,required=True)
    observed.add_argument('--object-id',required=True)
    worker=commands.add_parser('_worker',help=argparse.SUPPRESS)
    worker.add_argument('kind',choices=['rgb'])
    worker.add_argument('arguments',nargs=argparse.REMAINDER)
    pose=commands.add_parser('refine-pose', help='run legacy refinement on a source-observation bundle')
    for name in ('annotations','source-pose','rgb-edges','rgb-report','output','run-root'):
        pose.add_argument('--'+name,type=Path,required=True)
    pose.add_argument('--object-id',required=True)
    pose.add_argument('--anchor',type=int,required=True)
    visual=commands.add_parser('fit-visual',help='run legacy MANO/object fitting')
    for name in ('run-root','annotations','pose-report','mano-bridge','hawor-params','mano-models',
                 'output-dir','appearance-mask-manifest','interaction-judgments'):
        visual.add_argument('--'+name,type=Path,required=True)
    visual.add_argument('--object-id',required=True)
    visual.add_argument('--device',default='cpu')
    visual.add_argument('--iterations',type=int,default=240)
    visual.add_argument('--render-width',type=int)
    formal=commands.add_parser('fit-formal',help='solve a named prepared legacy MANO interval')
    formal.add_argument('--problem',type=Path,required=True)
    tail=commands.add_parser('formal-tail',help='run the preserved formal numerical and render tail')
    tail.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='legacy-reference':
        return reference_from_source_observations(args.source_observations, args.assets_manifest, args.output_dir,
                                                  gpu=args.gpu, gpu_uuid=args.gpu_uuid, device=args.device,
                                                  iterations=args.iterations, render_width=args.render_width)
    if args.command=='observed-pose':
        from observations.object.pose import fit_observed_pose
        return fit_observed_pose(list(map(str,['--annotations',args.annotations,'--completion-report',args.completion_report,
            '--object-id',args.object_id,'--output-dir',args.output_dir])))
    if args.command=='_worker':
        from observations.object.rgb_tracking import build_rgb_edges
        return build_rgb_edges(args.arguments)
    if args.command=='formal-tail': return formal_tail(load(args.spec))
    if args.command=='fit-formal':
        from legacy_optimize.problems import FormalManoInterval
        values=load(args.problem)
        for key in ('annotations','pose_report','completed_mesh','completion_report','output_dir','wilor_root','mano_left'):
            values[key]=Path(values[key])
        for key in ('depth_npz','factor_reports'):
            if key in values:values[key]=tuple(Path(p) for p in values[key])
        if values.get('hand_ray_prior') is not None:values['hand_ray_prior']=Path(values['hand_ray_prior'])
        return solve(FormalManoInterval(**values))
    if args.command=='refine-pose':
        return refine(args.annotations,args.source_pose,args.rgb_edges,args.rgb_report,
                      args.object_id,args.anchor,args.output,args.run_root)
    from legacy_optimize.problems import PreparedVisualFit
    values=vars(args).copy();values.pop('command');run_root=values.pop('run_root')
    return fit_prepared(PreparedVisualFit(**values),run_root)


def cli():
    main()


if __name__=='__main__':
    cli()
