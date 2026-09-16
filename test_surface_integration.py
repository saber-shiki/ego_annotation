"""Execution tests for the post-shape surface candidate consumer seam (CPU only)."""
import ast
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

import main
import render
from observations.object.envelope import build_envelopes
from observations.object.shape import record, mesh_arrays, write_mesh, build_completion_inputs, conditioning_parse_args
from optimization.object.surface_fit import fit_surface
from optimization.problems import PreparedVisualFit, StandaloneShape, ObjectSurfaceEnvelope
from optimization.solve import solve
from optimization.test_surface_fit import fixture as surface_fixture, write_json
from gates.object.surface_candidate import validate_surface_candidate, validate_surface_state
from utils.data import RKEY, TKEY, load
from utils.state import sha


def consumer_fixture(root):
    p = surface_fixture(root)
    ann = load(p.annotations)
    for frame, idx in zip(ann['frames'], [7, 19, 103]):
        frame['frame_idx'] = idx
        frame['raw_frame_path'] = str(root/'rgb.png')
    write_json(p.annotations, ann)
    prior = load(root/'prior.json')
    for row, idx in zip(prior['pose_rows'], [7, 19, 103]):
        row['frame_idx'] = idx
        row['uncertainty'] = .7
    write_json(root/'prior.json', prior)
    posterior = load(p.refined_posterior)
    posterior.update(source_prior=record(root/'prior.json'), source_pose_rows=copy.deepcopy(prior['pose_rows']),
                     pose_rows=copy.deepcopy(prior['pose_rows']),
                     evidence={'annotations_sha256':sha(p.annotations)})
    write_json(p.refined_posterior, posterior)
    appearance = load(p.appearance)
    for row, idx in zip(appearance['frames'], [7, 19, 103]):
        row.update(frame_idx=idx, visible=True, anchor_eligible=True)
    appearance['source_annotations_sha256'] = sha(p.annotations)
    write_json(p.appearance, appearance)
    binding = load(p.mesh_binding)
    binding.update(posterior)
    binding.update(schema='rgb_scene_visual_refined_binding_v1', refined_posterior=record(p.refined_posterior),
                   appearance_mask_manifest=record(p.appearance), case='case_b', track_id='track_d',
                   canonical_gauge={'name':'initial_completed_canonical', 'anchor_frame_idx':7})
    write_json(p.mesh_binding, binding)
    p.envelope_report.unlink()
    build_envelopes(annotations=p.annotations, mesh_binding=p.mesh_binding, refined_posterior=p.refined_posterior,
                    run_root=root, object_id=p.object_id, output=p.envelope_report)
    return p


class FakeMano(torch.nn.Module):
    """Differentiable 778-vertex fixture; not a replacement or claim of MANO inference."""
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(2)
        self.register_buffer('v', torch.tensor(rng.normal(size=(778, 3))*.008, dtype=torch.float32))
        self.faces = np.array([[i, i+1, i+2] for i in range(776)])

    def forward(self, global_orient, hand_pose, betas):
        v = self.v[None] + hand_pose[:, :3, None].transpose(1, 2)*.01
        return SimpleNamespace(vertices=v, joints=v[:, :16])


def hands_and_judgment(root, p):
    from optimization.hand.mano import TIP_VERTICES, JOINT_ORDER
    rows = load(p.refined_posterior)['pose_rows']
    ids = np.array([r['frame_idx'] for r in rows]); times = np.array([r['time_s'] for r in rows])
    source = {'frame_idx':ids, 'time_s':times}
    bridge_v, bridge_ids, bridge_sides = [], [], []
    models = {s:FakeMano() for s in ('left', 'right')}
    (root/'fake_models').mkdir()
    for side in models:
        (root/'fake_models'/f'MANO_{side.upper()}.pkl').write_bytes(b'synthetic differentiable fixture, not model weights')
        z = torch.zeros((3, 3)); pose = torch.zeros((3, 45)); beta = torch.zeros((3, 10))
        # Visible hand fixture just in front of the actual reconstructed sphere.
        trans = np.tile([0., 0., .82], (3, 1)).astype('float32')
        with torch.no_grad():
            o = models[side](global_orient=z, hand_pose=pose, betas=beta)
            v = o.vertices.numpy()+trans[:, None]
            j = torch.cat([o.joints, o.vertices[:, TIP_VERTICES]], dim=1)[:, JOINT_ORDER].numpy()+trans[:, None]
        for suffix, value in dict(root_orient_axis_angle=z.numpy(), hand_pose_axis_angle=pose.numpy(),
                betas=beta.numpy(), trans_world_m=trans, vertices_world_m=v, joints_world_m=j,
                valid=np.ones(3,bool), faces=models[side].faces).items():
            source[side+'_'+suffix] = value
        bridge_v.extend(v); bridge_ids.extend(ids); bridge_sides.extend([side]*3)
    np.savez(root/'hands.npz', **source)
    np.savez(root/'bridge.npz', frame_idx=bridge_ids, hand_side=bridge_sides,
             vertices_current_v18_world_from_hawor_projection_relift_m=bridge_v)
    judgment = {'object_id':p.object_id, 'frames':[{'frame_idx':int(i), 'visibility':{
        entity:{'state':'visible','occluder':None,'uncertainty':'synthetic uncertain measurement'}
        for entity in ('left','right','object')}} for i in ids],
        'interaction_judgments':[{'target_entity_id':'object:'+p.object_id,'hand_side':s,
            'frame_start':7,'frame_end':103,'contact_state':'possible_contact'} for s in ('left','right')]}
    write_json(root/'judgments.json', judgment)
    return models, judgment


def annotate_nodes():
    tree = ast.parse(Path(main.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'annotate')
    return next(n for n in fn.body if isinstance(n, ast.Try)).body


def execute_nodes(nodes, scope):
    # Execute the actual annotate stage definitions/calls, not copied pipeline logic.
    code = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(code, main.__file__, 'exec'), scope)


class SurfaceConsumerTests(unittest.TestCase):
    def test_actual_completion_input_copy_has_no_inference_claim(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'rgb.png'; mask = root/'mask.png'; crop = root/'crop.png'
            Image.fromarray(np.full((12,16,3),120,np.uint8)).save(image)
            Image.fromarray(np.full((12,16),255,np.uint8)).save(mask)
            Image.fromarray(np.full((8,8,4),255,np.uint8)).save(crop)
            evidence = write_json(root/'evidence.json', {'selected_frame_idx':48,'object_id':'thing',
                'selected':{'frame_idx':48,'raw_frame_path':str(image),'mask_path':str(mask),
                    'visible_geometry_candidate':{'mask_path':str(mask)},
                    'trellis_conditioning_crop':{'crop_rgba':str(crop)}}})
            with redirect_stdout(io.StringIO()):
                result = build_completion_inputs(conditioning_parse_args(['--evidence-report',str(evidence),
                                                                         '--output-dir',str(root/'inputs')]))
            native = result['conditioning_contracts']['sam3d_objects_native']
            self.assertNotEqual(Path(native['mask']), mask)
            self.assertEqual(Path(native['mask']).read_bytes(), mask.read_bytes())
            self.assertEqual(Path(native['image']).read_bytes(), image.read_bytes())
            self.assertIsNone(native['external_pointmap'])

    def test_shape_dispatch_native_initializer_and_expected_receipt(self):
        from utils.geometry import apply_native_pose
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = consumer_fixture(root)
            (root/'input/raw_frame_manifest').mkdir(parents=True)
            write_json(root/'input/raw_frame_manifest/manifest.json',{'frames':[{'source_width':64,'source_height':64}]})
            choice = write_json(root/'choice.json', {'object_id':'thing','selected_frame_idx':19})
            import trimesh
            native = trimesh.creation.icosphere(subdivisions=3,radius=.1)
            v,f = np.asarray(native.vertices), np.asarray(native.faces)
            write_mesh(root/'raw_native.ply',v,f)
            p3d, cv, _ = apply_native_pose(v, np.array([1.,0,0,0]), np.array([0.,0,1.]), np.array([1.]))
            write_mesh(root/'native_cv.ply', cv, f); write_mesh(root/'native_p3d.ply', p3d, f)
            qc = write_json(root/'native.json', dict(status='ok',image=str(root/'rgb.png'),mask=str(root/'mask.png'),
                mesh=str(root/'raw_native.ply'), native_pose_mesh_opencv_camera=str(root/'native_cv.ply'),
                native_pose_mesh_pytorch3d_camera=str(root/'native_p3d.ply'),
                pose={'rotation':[1.,0,0,0],'translation':[0.,0,1.],'scale':[1.]}))
            before = p.refined_posterior.read_bytes()
            problem = StandaloneShape(p.refined_posterior,p.annotations,p.appearance,choice,qc,root/'shape','thing',root,
                object_profile='surface_no_box',geometry_profile='shared_metric_pointmap',pointmap_receipt=root/'pointmap.json')
            with patch('observations.object.shape.validate_pointmap_conditioning', return_value={'verified':'fixture'}) as check, \
                 patch('optimization.object.shape.estimate_scale', side_effect=AssertionError('surface must not image-rescale')):
                result = solve(problem)
            check.assert_called_once_with(load(qc),root/'rgb.png',root/'mask.png',expected_frame_idx=19,
                expected_geometry_profile='shared_metric_pointmap',expected_pointmap_receipt=root/'pointmap.json')
            self.assertEqual(result['visual_mesh_receipt']['static_image_scale']['uniform_scale'],1.)
            self.assertEqual(result['pose_rows'],load(p.refined_posterior)['pose_rows'])
            self.assertEqual(before,p.refined_posterior.read_bytes())

    def test_candidate_prepared_actual_hand_fit_overlay_and_rrd(self):
        from test_pipeline import identity, batches
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=consumer_fixture(root)
            before = {path:path.read_bytes() for path in (p.mesh_binding,p.refined_posterior,p.annotations,p.appearance,p.envelope_report)}
            candidate = fit_surface(p)
            candidate_path = p.output_dir/'pose_mesh.json'
            validate_surface_candidate(candidate,object_id='thing',run_root=root)
            self.assertEqual(candidate['canonical_gauge'],load(p.mesh_binding)['canonical_gauge'])
            models, judgment = hands_and_judgment(root,p)
            problem = PreparedVisualFit(p.annotations,candidate_path,root/'bridge.npz',root/'hands.npz',root/'fake_models',
                root/'fit','thing',p.appearance,root/'judgments.json',iterations=3,object_profile='surface_no_box')
            argv=['fit-visual','--object-id','thing','--run-root',str(root),'--iterations','3',
                  '--object-profile','surface_no_box','--freeze-object']
            for flag, value in (('annotations',p.annotations),('pose-report',candidate_path),
                    ('mano-bridge',root/'bridge.npz'),('hawor-params',root/'hands.npz'),
                    ('mano-models',root/'fake_models'),('output-dir',root/'fit'),
                    ('appearance-mask-manifest',p.appearance),('interaction-judgments',root/'judgments.json')):
                argv.extend(['--'+flag,str(value)])
            with patch('optimization.interaction.joint_fit.mano_models',return_value=models):
                report = main.main(argv)
            state = dict(np.load(root/'fit/demo_state.npz',allow_pickle=False))
            validate_surface_state(state,candidate,run_root=root)
            self.assertTrue(report['object_frozen'])
            # Nonzero updates prove the actual hand optimizer was live while object variables stayed fixed.
            self.assertGreater(np.max(np.abs(state['hand_translation_camera'])),0.)
            self.assertGreater(np.max(np.abs(state['left_vertices_world']-state['left_base_vertices_world'])),0.)
            self.assertGreater(report['trace'][0]['visual_proximity'],0.)
            np.testing.assert_array_equal(state['object_delta'],np.zeros((3,6)))
            image=np.zeros((64,64,3),np.uint8)
            actual=render.mesh_overlay(state,0,image)
            source_state=copy.deepcopy(state)
            source_state['object_vertices_canonical']=mesh_arrays(root/'canonical.ply')[0]
            source_state['object_translation_world']=np.array([r[TKEY] for r in candidate['source_refined_pose_rows']])
            self.assertTrue(actual.any())
            self.assertTrue(np.any(actual != render.mesh_overlay(source_state,0,image)))
            # Exercise export_scene's binding/fit report/mesh guards, then use the
            # actual geometry-debug JSON/RRD writer without a raw-video fixture.
            (root/'input/raw_frame_manifest').mkdir(parents=True)
            write_json(root/'input/raw_frame_manifest/manifest.json', {'frames':[{'frame_idx':r['frame_idx'],
                'raw_frame_path':str(root/'rgb.png'),'time_s':r['time_s']} for r in candidate['pose_rows']]})
            reviews=render.render_video(root/'fit/demo_state.npz',root/'input/raw_frame_manifest/manifest.json',
                root/'video/annotation.mp4',binding_path=candidate_path,run_root=root)
            self.assertEqual([r['frame_idx'] for r in reviews],[7,19,103])
            capture=cv2.VideoCapture(str(root/'video/annotation.mp4'))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),3)
            finally:
                capture.release()
            real_export = render.export
            def debug_export(*args, **kwargs):
                kwargs['mode']='geometry_debug'; kwargs.pop('raw_source')
                return real_export(*args, **kwargs)
            with patch('utils.state.input_identity',return_value=identity()), patch.object(render,'export',side_effect=debug_export):
                doc=render.export_scene(root/'fit/demo_state.npz',candidate_path,root/'fit/fit_report.json',root,
                    root/'unused_identity_fixture.json',root/'export')
            for row,source,cand in zip(doc['frames'],candidate['source_refined_pose_rows'],candidate['pose_rows']):
                self.assertEqual(row['object']['refined_graph_pose'][TKEY],source[TKEY])
                self.assertEqual(row['object']['surface_candidate_pose'][TKEY],cand[TKEY])
                self.assertEqual(row['object']['visual_fitted_pose'][TKEY],cand[TKEY])
            chunks = batches(root/'export/world.rrd')
            arrays=[]
            for chunk in chunks:
                if (chunk.schema.metadata or {}).get(b'rerun:entity_path') != b'/world/objects/thing':
                    continue
                for field,col in zip(chunk.schema,chunk.columns):
                    if 'vertex_positions' in field.name:
                        arrays.extend(v for v in col.to_pylist() if v is not None)
            np.testing.assert_allclose(arrays[0],state['object_vertices_canonical'],atol=1e-7)
            from observations.judgments.judge_pose_trajectory import compact_trajectory_evidence, trajectory_review_sheets
            review=copy.deepcopy(next(n for n in annotate_nodes() if isinstance(n,ast.FunctionDef) and n.name=='_stage_pose_review'))
            review.decorator_list=[]
            (root/'state').mkdir(exist_ok=True)
            scope=dict(root=root,ann=p.annotations,binding=candidate_path,posterior=p.refined_posterior,
                hawor=root/'hands.npz',profiles={'object_profile':'surface_no_box'},
                compact_trajectory_evidence=compact_trajectory_evidence,pose_review_images=render.pose_review_images,
                trajectory_review_sheets=trajectory_review_sheets,load=load,Path=Path,json=json,
                raw_by_id={f['frame_idx']:f for f in load(p.annotations)['frames']},contract={'run_id':'test'},
                manifest=root/'input/raw_frame_manifest/manifest.json',record=lambda *a:None)
            execute_nodes([review],scope)
            evidence,images=scope['_stage_pose_review']()
            self.assertEqual(evidence['pose_evidence_role'],'generated_mesh_conditioned_surface_candidate')
            self.assertEqual(evidence['source_posterior']['sha256'],sha(candidate_path))
            self.assertEqual(evidence['timeline'][0][2],candidate['pose_rows'][0][TKEY])
            self.assertTrue(all(path.is_file() for path in images))
            for path,original in before.items():
                self.assertEqual(path.read_bytes(),original)
            for key in ('object_rotation','base_object_translation_world','object_delta','object_vertices_canonical'):
                bad=copy.deepcopy(state); bad[key].flat[0] += .001
                with self.subTest(key=key), self.assertRaises(ValueError):
                    render.export(bad,candidate,'case_b','thing','track_d',root/'bad',mode='geometry_debug',
                        visibility=judgment,identity=identity(),run_root=root)
            with self.assertRaisesRegex(ValueError,'matching frozen'):
                main.fit_prepared(replace(problem,object_profile='surface_weak_box',output_dir=root/'wrong'),root)
            changed=copy.deepcopy(candidate);changed['canonical_gauge']['anchor_frame_idx']=19
            with self.assertRaisesRegex(ValueError,'identity/gauge'):
                validate_surface_candidate(changed,object_id='thing',run_root=root)
            for field in ('annotations','appearance_mask_manifest'):
                other = root/(field+'_other.json')
                original = getattr(problem,field);other.write_bytes(original.read_bytes())
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'visual inputs'):
                    main.fit_prepared(replace(problem,**{field:other},output_dir=root/'wrong_inputs'),root)

    def test_annotate_both_native_jobs_consume_distinct_selected_receipts(self):
        from observations.test_pointmap import PointmapTests, FakeInference
        from observations.object.pointmap import load_pointmap
        from observations.object.sam3d import infer_shared_pointmap
        from PIL import Image
        fixture=PointmapTests();fixture.setUp()
        try:
            root=fixture.root
            copied_rgb=root/'conditioning_rgb.png';copied_mask=root/'conditioning_mask.png'
            copied_rgb.write_bytes(fixture.images[7].read_bytes());copied_mask.write_bytes(fixture.masks[7].read_bytes())
            appearance=write_json(root/'appearance.json',{'frames':[{'frame_idx':103,'object_id':'object-A',
                'appearance_mask_path':str(fixture.masks[103])}]})
            nodes=annotate_nodes(); native=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='native_job')
            calls=[]; producers=[]
            def worker(name,python,module,argv,outputs,overlay,*,input_contract):
                case=argv[argv.index('--case')+1].split('|'); pm=argv[argv.index('--pointmap-case')+1].split('|')
                self.assertEqual(pm[0],case[0]); frame=int(pm[1]);receipt=Path(pm[2])
                self.assertEqual(input_contract['pointmap_receipt']['sha256'],sha(receipt))
                self.assertEqual(input_contract['frame_idx'],frame)
                points,_=load_pointmap(receipt,frame_idx=frame,image_path=case[1],mask_path=case[2])
                inference=FakeInference()
                _,consumed=infer_shared_pointmap(inference,np.asarray(Image.open(case[1])),np.asarray(Image.open(case[2]))>0,
                    42,receipt_path=receipt,frame_idx=frame,image_path=case[1],mask_path=case[2])
                np.testing.assert_array_equal(inference._pipeline.seen,points)
                calls.append((name,frame,consumed,points.copy()))
            scope=dict(root=root,Path=Path,sha=sha,profiles={'geometry_profile':'shared_metric_pointmap'},
                args=SimpleNamespace(case='case_b'),object_id='object-A',worker=worker,
                assets={k:'fixture' for k in ('sam3d_repo','sam3d_config','sam3d_moge_checkpoint',
                    'sam3d_dinov2_repo','sam3d_dinov2_checkpoint','sam3d_python')},
                stage=lambda name,**kw:lambda fn:fn(),record=lambda name,outputs:producers.append((name,outputs)),
                conditioning={'image':str(copied_rgb),'mask':str(copied_mask)},anchor=7,
                ann=fixture.mask_source,bound=fixture.depth,camera=fixture.camera,manifest=fixture.manifest,
                geometry=root/'geometry',selected=103,selected_row={'raw_rgb_path':str(fixture.images[103]),
                    'appearance_mask_path':str(fixture.masks[103])},appearance=appearance)
            execute_nodes([native],scope)
            a=next(i for i,n in enumerate(nodes) if isinstance(n,ast.ImportFrom) and n.module=='observations.object.pointmap')
            b=next(i for i,n in enumerate(nodes) if isinstance(n,ast.FunctionDef) and n.name=='_stage_metric_geometry')
            execute_nodes(nodes[a:b],scope)
            a=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='appearance_pointmap' for t in n.targets))
            b=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='binding_dir' for t in n.targets))
            execute_nodes(nodes[a:b],scope)
            self.assertEqual([(n,f) for n,f,_,_ in calls],[('metric_native',7),('appearance_native',103)])
            self.assertEqual(calls[0][3][0,0,2],1.);self.assertEqual(calls[1][3][0,0,2],2.)
            self.assertEqual([n for n,_ in producers],['metric_pointmap','appearance_pointmap'])
            # Execute metric_geometry up to its native validator and inspect exact expected inputs.
            metric=copy.deepcopy(next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_stage_metric_geometry'))
            metric.decorator_list=[]
            class StopValidation(Exception):pass
            checker=unittest.mock.Mock(side_effect=StopValidation)
            scope['shape']=SimpleNamespace(sam3d_candidate_record=checker)
            execute_nodes([metric],scope)
            with self.assertRaises(StopValidation):scope['_stage_metric_geometry']()
            self.assertEqual(checker.call_args.kwargs,{'expected_frame_idx':7,
                'expected_geometry_profile':'shared_metric_pointmap','expected_pointmap_receipt':scope['metric_pointmap_receipt']})
        finally:
            fixture.doCleanups()

    def test_annotate_surface_stage_selection_receipts_resume_and_source_immutability(self):
        from utils.state import RunState, StageRecovery, planned_stage_contract
        nodes=annotate_nodes()
        start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='binding_dir' for t in n.targets))
        end=next(i for i,n in enumerate(nodes) if i>start and isinstance(n,ast.ImportFrom) and n.module=='render')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=consumer_fixture(root)
            originals={path:path.read_bytes() for path in (p.mesh_binding,p.refined_posterior)}
            profiles={'geometry_profile':'shared_metric_pointmap','object_profile':'surface_no_box'}
            (root/'input').mkdir()
            write_json(root/'input/runtime_input_contract.json',{'run_id':'synthetic-stage-consumer','input_video_sha256':'a'*64})
            seen=[]
            def solver(problem):
                seen.append(problem)
                if isinstance(problem,StandaloneShape):
                    # Source shape construction is covered above; return the immutable original here.
                    problem.output.mkdir(parents=True)
                    v,f=mesh_arrays(root/'canonical.ply');write_mesh(problem.output/'object_canonical.ply',v,f)
                    write_json(problem.output/'pose_mesh.json',load(p.mesh_binding))
                    return load(p.mesh_binding)
                # Run the real surface solver with a bounded synthetic fixture iteration count.
                from optimization.object.surface_fit import SurfaceProblem
                return fit_surface(SurfaceProblem(annotations=problem.annotations,refined_posterior=problem.refined_posterior,
                    mesh_binding=problem.mesh_binding,envelope_report=problem.envelope_report,appearance=problem.appearance,
                    run_root=problem.run_root,object_id=problem.object_id,output_dir=problem.output,
                    profile=problem.profile,outer_iterations=2,inner_iterations=4))
            write_json(root/'choice.json',{})
            write_json(root/'qc.json',{})
            for resume in (False,True):
                ledger=RunState(root); recovery=StageRecovery(ledger,{'test':'same-code-consumer','agent_model':'gpt-6-astra'},resume=resume)
                def stage(name,*,inputs,parameters):
                    def execute(fn):
                        return recovery.run(name,fn,contract=planned_stage_contract('main.annotate.'+name,'inprocess',
                            inputs=inputs,parameters={**parameters,'profiles':profiles},execution={'fixture':'cpu'}))
                    return execute
                def record_stage(name,outputs):
                    return ledger.record(name,outputs,object_id='thing',frame_ids=[7,19,103])
                scope=dict(root=root,ann=p.annotations,posterior=p.refined_posterior,appearance=p.appearance,
                    choice_path=root/'choice.json', native_qc=root/'qc.json',
                    appearance_receipt=None,metric_pointmap_receipt=None,object_id='thing',profiles=profiles,
                    stage=stage,record=record_stage,solve=solver,StandaloneShape=StandaloneShape,
                    ObjectSurfaceEnvelope=ObjectSurfaceEnvelope,Path=Path)
                try:
                    execute_nodes(nodes[start:end],scope)
                finally:
                    recovery.close()
                self.assertEqual(scope['binding'],root/'measurements/surface_candidate/thing/pose_mesh.json')
                result=load(scope['binding']); validate_surface_candidate(result,object_id='thing',run_root=root)
            self.assertEqual(len(seen),2)  # resume did not rebuild either create-only output
            self.assertEqual(seen[0].object_profile,'surface_no_box')
            self.assertEqual(seen[0].geometry_profile,'shared_metric_pointmap')
            for path,original in originals.items():self.assertEqual(path.read_bytes(),original)
            records=load(root/'state/producer_receipts.json')['records']
            self.assertTrue({'shape_binding','shape_mesh','surface_envelopes','surface_candidate_binding',
                             'surface_candidate_mesh','surface_fit_report'} <= {r['role'] for r in records})
            profiles['object_profile']='surface_weak_box'
            recovery=StageRecovery(RunState(root),{'test':'same-code-consumer','agent_model':'gpt-6-astra'},resume=True)
            try:
                with self.assertRaisesRegex(ValueError,'planned stage contract changed'):
                    execute_nodes(nodes[start:end],scope)
            finally:
                recovery.close()

    def test_profile_defaults_resume_and_legacy_source_equality_remain_strict(self):
        defaults=main.pipeline_profiles(SimpleNamespace())
        self.assertEqual(defaults,{'geometry_profile':'legacy_internal_moge','object_profile':'none'})
        selected={'geometry_profile':'shared_metric_pointmap','object_profile':'surface_no_box'}
        self.assertEqual(main.pipeline_profiles(SimpleNamespace(),selected),selected)
        with self.assertRaisesRegex(ValueError,'composition changed'):
            main.pipeline_profiles(SimpleNamespace(object_profile='surface_weak_box'),selected)
        with self.assertRaisesRegex(ValueError,'require shared'):
            main.pipeline_profiles(SimpleNamespace(object_profile='surface_no_box'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=consumer_fixture(root);binding=load(p.mesh_binding)
            binding['pose_rows'][0][TKEY][0]+=.01
            write_json(p.mesh_binding,binding)
            problem=SimpleNamespace(pose_report=p.mesh_binding,object_id='thing',object_profile='legacy')
            with patch.object(main,'solve',side_effect=AssertionError('must reject before optimizer')):
                with self.assertRaisesRegex(ValueError,'discarded refined'):
                    main.fit_prepared(problem,root)


if __name__ == '__main__':
    unittest.main()
