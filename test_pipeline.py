"""Grouped entry and dependency tests, independent of retired script imports."""
import ast
import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from optimization.problems import StandaloneScene
from optimization.solve import solve
from utils.data import PoseNode

class GroupedContractTests(unittest.TestCase):
    def test_all_grouped_modules_import_without_scripts(self):
        root=Path(__file__).parent
        for directory in ('utils','observations','gates','factors','optimization'):
            for path in (root/directory).rglob('*.py'):
                if not path.name.startswith('test_'):
                    importlib.import_module('.'.join(path.relative_to(root).with_suffix('').parts))

    def test_utils_do_not_import_business(self):
        for path in (Path(__file__).parent/'utils').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0],
                                     {'observations','gates','factors','optimization','main','render'})

    def test_named_problem_fixes_anchor_and_rejects_unknown(self):
        node=PoseNode(0,np.eye(3),np.array([0.,0.,1.]),None,'test',False,1.)
        poses,report=solve(StandaloneScene([node],np.array([0.]),[],0))
        np.testing.assert_array_equal(poses[0][0],node.initial_rotation)
        self.assertEqual(report['gauge_anchor'],0)
        with self.assertRaises(TypeError):solve({'formal':True,'visual':True})

    def test_prepared_entry_writes_candidate_and_preserves_source(self):
        import json
        import main
        from utils.state import sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            source=root/'source.json'; annotations=root/'annotations.json'
            depth=root/'depth.npz'; edge=root/'edges.npz'; report=root/'edges.json'
            output=root/'posterior.json'
            surface=np.array([[0.,0.,1.],[.02,0.,1.],[0.,.02,1.],[.02,.02,1.]])
            geometry={'world_vertices_sample_m':surface.tolist(),
                      'rigid_pose_observation_eligible':True}
            source.write_text(json.dumps({'object_id':'tool','pose_rows':[
                {'frame_idx':i,'rotation_world_from_completed_canonical_matrix':np.eye(3).tolist(),
                 'translation_world_m':[0.,0.,1.]} for i in range(3)]}))
            annotations.write_text(json.dumps({'frames':[
                {'frame_idx':i,'time_s':i*.1,'objects':[{'object_id':'tool',
                 'rigid_pose_observation_eligible':True,'visible_geometry_candidate':geometry}]}
                 for i in range(3)]}))
            np.savez(depth,depth=np.ones((3,2,2)))
            metadata={'object_id':'tool','initial_pose_report':str(source),'annotations':str(annotations),
                      'depth_npz':str(depth),'input_sha256':{'initial_pose_report':sha(source),
                      'annotations':sha(annotations),'depth_npz':sha(depth)}}
            edge_rows=[{'source_frame_idx':0,'target_frame_idx':1,'inlier_fraction':1.,
                        'reprojection_median_px':0.,'source_3d_conditioning':.02,
                        'topological_role':'adjacent','status':'accepted'}]
            np.savez(edge,metadata=np.array([json.dumps(metadata)]),
                     source_frame_idx=np.array([0]), target_frame_idx=np.array([1]),
                     accepted=np.array([True]), rotation_source_to_target_world=np.eye(3)[None],
                     translation_source_to_target_world_m=np.zeros((1,3)),
                     reprojection_evidence_offsets=np.array([0,4]),
                     reprojection_canonical_points=surface-[0,0,1.],
                     reprojection_target_camera=np.eye(4)[None],
                     reprojection_target_intrinsics=np.array([[100.,100.,0.,0.]]),
                     reprojection_target_uv=surface[:,:2]*100.,reprojection_weights=np.ones(4))
            report.write_text(json.dumps({**metadata,'rows':edge_rows})); before=sha(source)
            (root/'input').mkdir();(root/'state').mkdir()
            raw=root/'input/raw.mp4';raw.write_bytes(b'prepared raw fixture')
            contract={'run_id':'test-run','run_root':str(root),'input_video':str(raw),'input_video_sha256':sha(raw)}
            (root/'input/runtime_input_contract.json').write_text(json.dumps(contract))
            records=[{'path':str(path),'sha256':sha(path),'role':role,'object_id':'tool',
                      'frame_ids':[0,1,2],'producer_id':producer}
                     for path,role,producer in [(source,'source_pose','pose'),(annotations,'observations','observations'),
                     (depth,'depth','depth'),(edge,'rgb_edges','rgb'),(report,'rgb_edge_report','rgb')]]
            registry={**contract,'records':records}
            (root/'state/producer_receipts.json').write_text(json.dumps(registry))
            main.main(['refine-pose','--annotations',str(annotations),'--source-pose',str(source),
                '--rgb-edges',str(edge),'--rgb-report',str(report),'--object-id','tool',
                '--anchor','0','--output',str(output),'--run-root',str(root)])
            candidate=json.loads(output.read_text())
            self.assertEqual(len(candidate['pose_rows']),3)
            self.assertFalse(candidate['annotation_ready'])
            self.assertFalse(candidate['physical_contact_authority'])
            self.assertEqual(sha(source),before)
            with self.assertRaises(ValueError):
                main.refine(annotations,source,edge,report,'tool',0,output,root)
            nodes,_=main.build_nodes(json.loads(annotations.read_text()),json.loads(source.read_text()),'tool')
            edges,pixels=main.load_edges(edge,report,source,annotations,nodes,root)
            self.assertEqual(len(edges),1);self.assertEqual(len(pixels),1)
            from factors.object.image import rgb_reprojection_residual
            from utils.data import SolverConfig
            poses={n.frame_idx:(n.initial_rotation.copy(),n.initial_translation.copy()) for n in nodes}
            baseline=rgb_reprojection_residual(pixels[0],poses,nodes,SolverConfig(rgb_reprojection_weight_scale=1.))
            for endpoint in (0,1):
                changed={k:(r.copy(),t.copy()) for k,(r,t) in poses.items()}
                changed[endpoint][1][0]+=.001
                self.assertGreater(np.linalg.norm(rgb_reprojection_residual(pixels[0],changed,nodes,
                    SolverConfig(rgb_reprojection_weight_scale=1.))-baseline),0)
            # Byte-identical foreign data and symlinks are still another run.
            with tempfile.TemporaryDirectory() as other:
                for original in (annotations,source,edge,report):
                    foreign=Path(other)/original.name;foreign.write_bytes(original.read_bytes())
                    replacements={annotations:annotations,source:source,edge:edge,report:report}
                    replacements[original]=foreign
                    with self.assertRaisesRegex(ValueError,'escapes run root'):
                        main.refine(replacements[annotations],replacements[source],replacements[edge],
                            replacements[report],'tool',0,root/'new.json',root)
            registry['records'][-1]['producer_id']='different-rgb-producer'
            (root/'state/producer_receipts.json').write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError,'producer pairing'):
                main.load_edges(edge,report,source,annotations,nodes,root)
            registry['records'][-1]['producer_id']='rgb'
            mismatched=json.loads(report.read_text());mismatched['object_id']='other'
            report.write_text(json.dumps(mismatched));registry['records'][-1]['sha256']=sha(report)
            (root/'state/producer_receipts.json').write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError,'metadata pairing'):
                main.load_edges(edge,report,source,annotations,nodes,root)


    def test_entry_help_and_no_legacy_imports(self):
        import main
        with self.assertRaises(SystemExit) as result:main.main(['--help'])
        self.assertEqual(result.exception.code,0)
        for directory in ('utils','observations','gates','factors','optimization'):
            for path in (Path(__file__).parent/directory).rglob('*.py'):
                if path.name.startswith('test_'):continue
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node,ast.ImportFrom):
                        self.assertFalse((node.module or '').startswith(('rgb_scene','v19_','v20_')))


import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from render import export,timeline
from gates.interaction.visibility import apply_display_visibility, validate_visibility
from utils.state import InputIdentity, input_identity
from utils.data import RKEY,TKEY
from utils.state import sha


def fixture(object_id='vessel',size=(640,360)):
    n=3;ids=np.array([7,19,103]);times=np.array([.1,.43,1.7])
    rotations=Rotation.from_rotvec([[0,0,.7],[0,.4,.9],[.2,.5,1.1]]).as_matrix()
    translations=np.array([[0,0,1],[.1,0,1.1],[.2,0,1.2]])
    z={'frame_idx':ids,'time_s':times,'T_world_camera':np.tile(np.eye(4),(n,1,1)),
       'intrinsics_render':np.tile([510.,430.,size[0]/2,size[1]/2],(n,1)),
       'render_size_wh':np.array(size),'source_size_wh':np.array(size),
       'object_rotation':rotations,'base_object_rotation':rotations.copy(),
       'object_translation_world':translations,'base_object_translation_world':translations.copy(),
       'object_vertices_canonical':np.array([[0,0,0],[.1,0,0],[0,.2,0],[0,0,.3]]),
       'object_faces':np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]),
       'interaction_states':np.full((n,2),'unresolved')}
    rng=np.random.default_rng(5)
    for side in ('left','right'):
        z[side+'_valid']=np.array([True,True,True]);z[side+'_vertices_world']=rng.normal(size=(n,778,3))*.01+[0,0,1.]
        z[side+'_faces']=np.array([[i,i+1,i+2] for i in range(776)])
    binding={'schema':'rgb_scene_visual_refined_binding_v1','object_id':object_id,
        'canonical_gauge':{'name':'initial_completed_canonical','anchor_frame_idx':7},
        'source_prior':{'sha256':'a'*64},'refined_posterior':{'sha256':'b'*64},
        'source_pose_rows':[{'frame_idx':int(i),RKEY:np.eye(3).tolist(),TKEY:[0,0,1]} for i in ids],
        'pose_rows':[{'frame_idx':int(i),'time_s':float(t),RKEY:r.tolist(),TKEY:p.tolist(),'uncertainty':.7,'metric_observation':False} for i,t,r,p in zip(ids,times,rotations,translations)]}
    return apply_display_visibility(z,judgment(object_id),object_id),binding


def judgment(object_id='vessel'):
    return {'object_id':object_id,'frames':[{'frame_idx':idx,'visibility':{
        entity:{'state':'out_of_frame' if pos==1 else 'occluded' if pos==2 else 'visible',
                'occluder':'unknown' if pos==2 else None,'uncertainty':'finite inferred pose; uncertain depth ownership'}
        for entity in ('object','left','right')}} for pos,idx in enumerate((7,19,103))]}


def identity(case='case_b'):
    return InputIdentity('aa5f4afc-2dc4-4cfe-aae6-25cd98aafc11',case,'c'*64,'d'*64,None,'e'*64,
        json.dumps({'frame_count':3,'fps_ratio':'30','time_base_s':'1/30000','first_pts_s':0.,'duration_s':.1,
                    'max_cfr_grid_error_s':0.,'fps_source_field':'r_frame_rate','validation':'fixture'}),'selected target')


def batches(path):
    from rerun.experimental import RrdReader
    return [c.to_record_batch() for c in RrdReader(path).stream().to_chunks()]


class ExportTests(unittest.TestCase):
    def test_portable_pair_actual_rrd_geometry_time_clears_and_camera(self):
        for object_id,size in [('vessel',(640,360)),('tool',(320,640))]:
            with self.subTest(object_id=object_id),tempfile.TemporaryDirectory() as td:
                root=Path(td).resolve();z,b=fixture(object_id,size)
                with patch('subprocess.Popen',side_effect=AssertionError('encoder/subprocess forbidden')):
                    doc=export(z,b,'case_b',object_id,'track_d',root/'internal/final',visibility=judgment(object_id),identity=identity(),mode="geometry_debug")
                self.assertEqual(sorted(p.name for p in (root/'internal/final').iterdir()),['annotation.json','world.rrd'])
                shutil.copytree(root/'internal/final',root/'portable');shutil.rmtree(root/'internal')
                doc=json.loads((root/'portable/annotation.json').read_text())
                self.assertEqual(doc['input_identity'],identity().document())
                self.assertEqual(doc['frames'][2]['object']['visibility']['state'],'occluded')
                self.assertTrue(doc['frames'][1]['object']['fitted_pose_valid'])
                self.assertTrue(doc['frames'][1]['hands']['left']['source_surface_valid'])
                self.assertFalse(doc['frames'][1]['hands']['left']['surface_valid'])
                self.assertEqual(doc['recording']['sha256'],sha(root/'portable/world.rrd'))
                self.assertEqual([f['time_s'] for f in doc['frames']],z['time_s'].tolist())
                self.assertEqual([f['decode_position'] for f in doc['frames']],[0,1,2])
                self.assertIsNone(doc['frames'][1]['object']['geometry'])
                chunks=batches(root/'portable/world.rrd')
                entities={}
                for batch in chunks:
                    meta=batch.schema.metadata or {};entity=meta.get(b'rerun:entity_path',b'').decode()
                    entities.setdefault(entity,[]).append(batch)
                self.assertIn('/world/camera_trajectory',entities)
                for entity,vertices,faces in [('/world/objects/'+object_id,z['object_vertices_canonical'],z['object_faces']),('/world/hands/left',z['left_vertices_world'][0],z['left_faces']),('/world/hands/right',z['right_vertices_world'][0],z['right_faces'])]:
                    rows=entities[entity];found_v=[];found_f=[];clears=[]
                    for batch in rows:
                        for field,column in zip(batch.schema,batch.columns):
                            if 'vertex_positions' in field.name:found_v.extend(v for v in column.to_pylist() if v is not None)
                            if 'triangle_indices' in field.name:found_f.extend(v for v in column.to_pylist() if v is not None)
                            if 'Clear' in field.name:clears.extend(column.to_pylist())
                    self.assertGreaterEqual(len(found_v),2,entity);self.assertTrue(found_f,entity);self.assertTrue(clears,entity)
                    np.testing.assert_allclose(found_v[0],vertices,atol=1e-6);np.testing.assert_array_equal(found_f[0],faces)
                # Every camera sample has exact time, translation and intrinsics.
                camera=entities['/world/camera']
                stamps=[];translations=[]
                for batch in camera:
                    for field,col in zip(batch.schema,batch.columns):
                        if field.name=='frame':stamps.extend(col.to_pylist())
                        if 'translation' in field.name:translations.extend(v for v in col.to_pylist() if v is not None)
                self.assertEqual(stamps,z['frame_idx'].tolist())
                np.testing.assert_allclose(np.array(translations).reshape(-1,3),z['T_world_camera'][:,:3,3])
                image=entities['/world/camera/image']
                self.assertTrue(any('image_from_camera' in name for batch in image for name in batch.schema.names))

    def test_no_copythrough_or_malformed_state(self):
        with tempfile.TemporaryDirectory() as td:
            z,b=fixture();bad=copy.deepcopy(z);bad['base_object_rotation'][:]=np.eye(3)
            with self.assertRaisesRegex(ValueError,'reset'):export(bad,b,'c','vessel','vessel',Path(td)/'bad',visibility=judgment(),identity=identity('c'),mode="geometry_debug")
            for mutate in [lambda d:d.update(frame_idx=np.array([7,7,103])),lambda d:d.update(time_s=np.array([.1,.1,1.])),lambda d:d.update(object_faces=np.array([[0,1,99]])),lambda d:d.update(intrinsics_render=np.zeros((3,4))),lambda d:d['object_rotation'].__setitem__(0,np.zeros((3,3)))]:
                bad=copy.deepcopy(z);mutate(bad)
                with self.assertRaises(ValueError):timeline(bad)
            bad=copy.deepcopy(b);bad['object_id']='other'
            with self.assertRaises(ValueError):export(z,bad,'c','vessel','vessel',Path(td)/'wrong',visibility=judgment(),identity=identity('c'),mode="geometry_debug")

    def test_visibility_rejects_missing_duplicate_incomplete_bad_identity_and_states(self):
        z,b=fixture()
        for mutate in (lambda j:j['frames'].pop(),lambda j:j['frames'].append(j['frames'][0]),
                       lambda j:j['frames'][0]['visibility'].pop('left'),
                       lambda j:j['frames'][0]['visibility']['object'].update(state='invented'),
                       lambda j:j['frames'][0]['visibility']['left'].pop('uncertainty'),
                       lambda j:j['frames'][0].update(frame_idx='7'),
                       lambda j:j.update(object_id='other')):
            value=judgment();mutate(value)
            with self.assertRaises(ValueError):validate_visibility(value,z['frame_idx'],'vessel')
        bad=dict(z);bad.pop('object_valid')
        with self.assertRaisesRegex(ValueError,'explicit object_valid'):timeline(bad)
        with tempfile.TemporaryDirectory() as td:
            for value in (None,{},'bad'):
                with self.assertRaises(ValueError):export(z,b,'case_b','vessel','track',Path(td)/'final',visibility=judgment(),identity=value)
            self.assertFalse((Path(td)/'final').exists())

    def test_production_fit_wrapper_derives_visibility_before_export(self):
        import main as fitter
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();z,b=fixture();out=root/'fit';out.mkdir()
            for key in ('object_valid','object_pose_valid','left_display_valid','right_display_valid','display_visibility_policy'):
                z.pop(key)
            prior=root/'prior.json';prior.write_text(json.dumps({'pose_rows':b['source_pose_rows']}))
            posterior=root/'posterior.json'
            posterior.write_text(json.dumps({'schema':'rgb_scene_refined_pose_v1','object_id':'vessel',
                'pose_rows':b['pose_rows'],'source_pose_rows':b['source_pose_rows'],
                'source_prior':{'path':str(prior),'sha256':sha(prior)}}))
            b['refined_posterior']={'path':str(posterior),'sha256':sha(posterior)}
            binding=root/'binding.json';binding.write_text(json.dumps(b))
            j=root/'judgment.json';j.write_text(json.dumps(judgment()))
            def generic(*args,**kwargs):
                np.savez_compressed(out/'demo_state.npz',**z)
                (out/'fit_report.json').write_text('{}')
            argv=['fit','--run-root',str(root),'--pose-report',str(binding),'--object-id','vessel',
                  '--interaction-judgments',str(j),'--output-dir',str(out)]
            problem=SimpleNamespace(pose_report=binding,object_id='vessel',interaction_judgments=j,output_dir=out)
            with patch.object(fitter,'solve',generic):fitter.fit_prepared(problem,root)
            with np.load(out/'demo_state.npz',allow_pickle=False) as archive:state=dict(archive)
            np.testing.assert_array_equal(state['object_valid'],[True,False,True])
            for key in ('object_rotation','object_translation_world','left_valid','left_vertices_world'):
                np.testing.assert_array_equal(state[key],z[key])
            doc=export(state,b,'case_b','vessel','track',root/'final',visibility=judgment(),identity=identity(),mode="geometry_debug")
            self.assertIsNone(doc['frames'][1]['object']['geometry'])
            self.assertIsNotNone(doc['frames'][2]['object']['geometry'])
            self.assertTrue(batches(root/'final/world.rrd'))
            incomplete=judgment();incomplete['frames'].pop();j.write_text(json.dumps(incomplete))
            with patch.object(fitter,'solve',side_effect=AssertionError('must reject before fitting')):
                with self.assertRaises(ValueError):fitter.fit_prepared(problem,root)

    def test_input_identity_reads_current_contract_and_rejects_tampering(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();raw=root/'raw.mp4';raw.write_bytes(b'temporary identity fixture')
            bundle=root/'bundle.json';bundle.write_text('{}')
            contract=root/'contract.json'
            c={'schema':'rgb_scene_input_v1','run_id':identity().run_id,'run_root':str(root),'case':'case_b',
               'input_video':str(raw),'input_video_sha256':sha(raw),'bundle_manifest_sha256':sha(bundle),
               'sensor_calibration_metadata':None,'sensor_calibration_sha256':None,
               'source_timing':identity().document()['source_timing']}
            contract.write_text(json.dumps(c))
            actual=input_identity(contract,root,bundle)
            self.assertEqual(actual.input_video_sha256,sha(raw));self.assertEqual(actual.input_contract_sha256,sha(contract))
            raw.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'identity changed'):input_identity(contract,root,bundle)
            for fields in ({'run_id':'bad'},{'input_video_sha256':''},{'source_timing_json':'{}'},
                           {'sensor_calibration_sha256':'not a digest'}):
                with self.assertRaises(ValueError):replace(actual,**fields)

import os


class RealManoOptimizerTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('EGO_MANO_MODELS'), 'set EGO_MANO_MODELS to licensed MANO assets')
    def test_real_mano_forward_fit_and_replay(self):
        from optimization.hand.mano import mano_models, TIP_VERTICES, JOINT_ORDER
        from optimization.problems import PreparedVisualFit
        from optimization.solve import solve
        from observations.object.shape import AUTHORITY
        import torch, trimesh, cv2
        model_path=Path(os.environ['EGO_MANO_MODELS'])
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
         root=Path(tmp).resolve(); n=3; ids=np.arange(n); source={}; bridge_v=[]; bridge_ids=[]; bridge_sides=[]
         models=mano_models(model_path)
         for si,side in enumerate(('left','right')):
          orient=np.zeros((n,3),np.float32);pose=np.zeros((n,45),np.float32);beta=np.zeros((n,10),np.float32)
          translation=np.tile([-.15 if si==0 else .15,0.,1.],(n,1)).astype('float32')
          with torch.no_grad():
           output=models[side](global_orient=torch.tensor(orient),hand_pose=torch.tensor(pose),betas=torch.tensor(beta))
           vertices=(output.vertices+torch.tensor(translation)[:,None]).numpy()
           joints=(torch.cat([output.joints[:,:16],output.vertices[:,TIP_VERTICES]],dim=1)[:,JOINT_ORDER]+torch.tensor(translation)[:,None]).numpy()
          for suffix,value in [('root_orient_axis_angle',orient),('hand_pose_axis_angle',pose),('betas',beta),('trans_world_m',translation),('vertices_world_m',vertices),('joints_world_m',joints),('valid',np.ones(n,bool)),('faces',models[side].faces)]:source[side+'_'+suffix]=value
          bridge_v.extend(vertices);bridge_ids.extend(ids);bridge_sides.extend([side]*n)
         source['frame_idx']=ids;source['time_s']=ids*.1
         np.savez(root/'hands.npz',**source);np.savez(root/'bridge.npz',frame_idx=bridge_ids,hand_side=bridge_sides,vertices_current_v18_world_from_hawor_projection_relift_m=bridge_v)
         frames=[{'frame_idx':int(i),'time_s':float(i*.1),'source_width':64,'source_height':64,'camera':{'intrinsics_fx_fy_cx_cy':[60,60,31.5,31.5],'T_world_camera_metric':np.eye(4).tolist()}} for i in ids]
         ann=root/'ann.json';ann.write_text(json.dumps({'frames':frames}))
         mesh=trimesh.creation.box(extents=[.12,.12,.12]);mp=root/'mesh.ply';mesh.export(mp)
         mask=np.zeros((64,64),np.uint8);mask[28:36,28:36]=255;cv2.imwrite(str(root/'mask.png'),mask)
         appearance=root/'appearance.json';appearance.write_text(json.dumps({'object_id':'object','metric_owned_masks_consumed':False,'source_annotations_sha256':sha(ann),'frames':[{'object_id':'object','frame_idx':int(i),'status':'available','visible':True,'source_size_wh':[64,64],'appearance_mask_path':str(root/'mask.png'),'appearance_mask_sha256':sha(root/'mask.png')} for i in ids]}))
         rep={'object_id':'object','appearance_mask_manifest':{'path':str(appearance),'sha256':sha(appearance)},'visual_mesh_receipt':{**AUTHORITY,'output_mesh_path':str(mp),'output_mesh_sha256':sha(mp)},'render_mesh_contract':{'required_mesh_path':str(mp),'required_mesh_sha256':sha(mp)},'pose_rows':[{'frame_idx':int(i),'time_s':float(i*.1),'rotation_world_from_completed_canonical_matrix':np.eye(3).tolist(),'translation_world_m':[0.,0.,1.]} for i in ids]}
         (root/'binding.json').write_text(json.dumps(rep))
         judgment={'object_id':'object','interaction_judgments':[{'target_entity_id':'object:object','hand_side':s,'frame_start':0,'frame_end':2,'contact_state':'no_contact'} for s in ('left','right')]}
         (root/'judgment.json').write_text(json.dumps(judgment))
         report=solve(PreparedVisualFit(ann,root/'binding.json',root/'bridge.npz',root/'hands.npz',model_path,root/'fit','object',appearance,root/'judgment.json',iterations=2))
         with np.load(root/'fit/demo_state.npz') as state:
          assert state['left_vertices_world'].shape==(3,778,3)
          assert np.isfinite(state['left_vertices_world']).all()
          assert len(report['trace'])==2 and report['annotation_ready'] is False
          assert not report['signed_intrusion_loss_enabled']
          print(json.dumps({'status':'passed','frames':3,'left_vertices':list(state['left_vertices_world'].shape),'iterations':2,'replay_errors':report['zero_state_max_vertex_error_m'],'last_loss':report['trace'][-1],'physical_ready':report['annotation_ready']}))


class AstraTransportTests(unittest.TestCase):
    def test_request_model_and_fallback_lock_and_response_validation(self):
        import io
        from unittest.mock import patch
        from utils.runtime import astra_request, ASTRA_MODEL
        class Response(io.BytesIO):
            headers={'x-litellm-model-group':'gpt-6-astra'}
        schema={'type':'object','properties':{'color':{'type':'string'}},'required':['color']}
        payload={'model':ASTRA_MODEL,'status':'completed','output':[{'type':'message','role':'assistant',
                 'content':[{'type':'output_text','text':'{"color":"red"}'}]}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);credential=path/'credential';credential.write_text('test-only-secret')
            credential.chmod(0o600);image=path/'evidence.png';image.write_bytes(b'test-image')
            def endpoint(request,timeout):
                sent=json.loads(request.data)
                self.assertEqual(sent['model'],'gpt-6-astra')
                self.assertIs(sent['disable_fallbacks'],True)
                self.assertIs(sent['store'],False)
                self.assertEqual(sent['input'][0]['content'][-1]['type'],'input_image')
                return Response(json.dumps(payload).encode())
            with patch.dict(os.environ,{'EGO_ASTRA_CREDENTIAL_FILE':str(credential)}),patch('urllib.request.urlopen',endpoint):
                result,receipt=astra_request('color?',schema,{},[image])
                self.assertEqual(result,{'color':'red'})
                self.assertEqual(receipt['response_model'],'gpt-6-astra')
                payload['model']='another-model'
                with self.assertRaisesRegex(RuntimeError,'mismatch'):astra_request('color?',schema,{},[image])
                payload['model']=ASTRA_MODEL;Response.headers={'x-litellm-model-group':'another-model'}
                with self.assertRaisesRegex(RuntimeError,'mismatch'):astra_request('color?',schema,{},[image])
                Response.headers={};payload['status']='incomplete'
                with self.assertRaisesRegex(RuntimeError,'did not complete'):astra_request('color?',schema,{},[image])

    def test_thirteen_distinct_judgments_have_independent_prompts_and_schemas(self):
        paths=sorted((Path(__file__).parent/'observations/judgments').glob('judge_*.py'))
        self.assertEqual(len(paths),13)
        modules=[importlib.import_module('observations.judgments.'+p.stem) for p in paths]
        self.assertEqual(len({m.PROMPT for m in modules}),13)
        self.assertEqual(len({json.dumps(m.SCHEMA,sort_keys=True) for m in modules}),13)
        self.assertTrue(all(callable(m.judge) for m in modules))



class CompleteRawExportTests(unittest.TestCase):
    def test_complete_export_requires_real_registered_rgb_and_rejects_foreign_frames(self):
        import cv2
        from types import SimpleNamespace
        from dataclasses import replace
        from observations.camera.calibration import build
        from utils.data import RawFrameSource, read_cfr_video_timing
        from utils.state import RunState, input_identity, validated_raw_frames
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'input').mkdir()
            raw=root/'input/raw.mp4'
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30.,(64,64))
            self.assertTrue(writer.isOpened())
            for channel in range(3):
                image=np.zeros((64,64,3),np.uint8);image[:,:,channel]=180;writer.write(image)
            writer.release()
            timing=read_cfr_video_timing(raw)
            bundle=root/'input/bundle.json';bundle.write_text('{}')
            contract={'schema':'rgb_scene_input_v1','run_id':identity().run_id,'run_root':str(root),
                'case':'case_b','input_video':str(raw),'input_video_sha256':sha(raw),
                'bundle_manifest_sha256':sha(bundle),'sensor_calibration_metadata':None,
                'sensor_calibration_sha256':None,'source_timing':timing.report()}
            (root/'input/runtime_input_contract.json').write_text(json.dumps(contract))
            manifest=root/'input/raw_frame_manifest/manifest.json'
            build(SimpleNamespace(video=raw,output_dir=manifest.parent,render_width=64,
                frame_start=None,frame_end=None,jpeg_quality=95))
            RunState(root).record('actual_decode',[(manifest,'raw_manifest')],frame_ids=[0,1,2])
            source=RawFrameSource(manifest,root)
            ident=input_identity(root/'input/runtime_input_contract.json',root,bundle)
            frames,_=validated_raw_frames(source,ident)
            z,b=fixture(size=(64,64));z['frame_idx']=np.arange(3);z['time_s']=np.arange(3)/30.
            for name in ('pose_rows','source_pose_rows'):
                for i,row in enumerate(b[name]):row['frame_idx']=i
            j=judgment()
            for i,row in enumerate(j['frames']):row['frame_idx']=i
            with self.assertRaisesRegex(ValueError,'raw-frame source'):
                export(z,b,'case_b','vessel','track',root/'absent',visibility=j,identity=ident)
            doc=export(z,b,'case_b','vessel','track',root/'complete',raw_source=source,visibility=j,identity=ident)
            self.assertFalse(doc['partial']);self.assertFalse(doc['annotation_ready'])
            self.assertFalse(doc['physical_contact_authority'])
            self.assertEqual(doc['raw_frame_provenance']['input_video_sha256'],sha(raw))
            first=Path(frames[0]['raw_frame_path']);first.write_bytes(b'foreign RGB with identical frame ID/time')
            with self.assertRaisesRegex(ValueError,'RGB bytes'):
                export(z,b,'case_b','vessel','track',root/'tampered',raw_source=source,visibility=j,identity=ident)


class TriangleRenderTests(unittest.TestCase):
    def test_rendered_overlay_uses_actual_triangles_and_clears_missing_geometry(self):
        from render import mesh_overlay
        state={'intrinsics_render':np.array([[40.,40.,31.5,31.5]]),'render_size_wh':np.array([64,64]),
            'T_world_camera':np.eye(4)[None],'object_valid':np.array([True]),
            'object_vertices_canonical':np.array([[-.2,-.2,0],[.2,-.2,0],[0,.2,0]]),
            'object_faces':np.array([[0,1,2]]),'object_rotation':np.eye(3)[None],
            'object_translation_world':np.array([[0,0,1.]]),
            'left_display_valid':np.array([False]),'right_display_valid':np.array([False])}
        image=np.zeros((64,64,3),np.uint8)
        rendered=mesh_overlay(state,0,image)
        self.assertGreater(rendered.sum(),0)
        self.assertGreater(rendered[31,31].sum(),0)
        state['object_valid'][0]=False
        np.testing.assert_array_equal(mesh_overlay(state,0,image),image)



class ColdControlFlowTests(unittest.TestCase):
    def test_visual_rejection_and_uncertainty_are_preserved_without_physical_promotion(self):
        from main import scene_review_summary
        mask={'identity_consistent':True,'intervals':[{'state':'usable'}]}
        trajectory={'trajectory_consistent':True,'flagged_intervals':[]}
        clean={'visual_result':'reviewed_usable','physical_authority_granted':False}
        result=scene_review_summary(mask,trajectory,[clean])
        self.assertTrue(result['accepted']);self.assertFalse(result['physical_authority_granted'])
        self.assertFalse(scene_review_summary(mask,trajectory,[])['accepted'])
        self.assertFalse(scene_review_summary(mask,{**trajectory,'trajectory_consistent':False},[clean])['accepted'])
        result=scene_review_summary({**mask,'intervals':[{'state':'uncertain'}]},trajectory,[clean])
        self.assertEqual(result['status'],'reviewed_with_defects')
        result=scene_review_summary(mask,trajectory,[{**clean,'visual_result':'rejected'}])
        self.assertEqual(result['status'],'rejected');self.assertFalse(result['accepted'])

    def test_gpu_launch_checks_exact_uuid_occupancy_and_memory_without_fallback(self):
        from utils.runtime import inspect_gpu, run_model
        from unittest.mock import patch
        devices='4, GPU-test-device-four, 16000\n5, GPU-test-device-five, 40000\n'
        with patch('subprocess.check_output',side_effect=[devices,'GPU-test-device-five, 42\n']):
            self.assertEqual(inspect_gpu(4)['uuid'],'GPU-test-device-four')
        with patch('subprocess.check_output',side_effect=[devices,'GPU-test-device-four, 42\n']):
            with self.assertRaisesRegex(RuntimeError,'active compute'):inspect_gpu(4)
        with patch('subprocess.check_output',return_value=devices):
            with self.assertRaisesRegex(RuntimeError,'UUID changed'):inspect_gpu(4,expected_uuid='other')
        with patch('subprocess.check_output',side_effect=[devices,'']):
            with self.assertRaisesRegex(RuntimeError,'free memory'):inspect_gpu(4,min_free_mib=20000)
        with tempfile.TemporaryDirectory() as directory:
            with patch('subprocess.check_output',side_effect=[devices,'']),patch('tempfile.gettempdir',return_value=directory),patch('subprocess.run') as run:
                result=run_model(['python','model'],cwd=directory,environment={},gpu=4)
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-test-device-four')
                self.assertEqual(result['index'],4)


"""CPU synthetic contracts for the source-named final annotation exporters."""
import argparse
import builtins
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from render import assemble_formal_layers as assemble
from render import render_formal_rrd as render
from render import load_package, sha256


class AnnotationLayersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        n = 3
        ids = np.array([4, 5, 6])
        T = np.repeat(np.eye(4)[None], n, axis=0)
        T[:, 0, 3] = np.arange(n) * .01
        K = np.tile([400., 410., 320., 240.], (n, 1))
        vertices = np.zeros((n, 778, 3), dtype=np.float32)
        vertices[:, :, 2] = 1
        vertices[:, 1, 0] = .04
        vertices[:, 2, 1] = .05
        joints = vertices[:, :21].copy()
        valid = np.array([True, False, True])
        vertices[1] = np.nan
        joints[1] = np.nan
        self.hawor = dict(frame_idx=ids, R_c2w=T[:, :3, :3], t_c2w=T[:, :3, 3],
                          left_vertices_world_m=vertices, left_joints_world_m=joints,
                          left_faces=np.array([[0, 1, 2]], dtype=np.int32), left_valid=valid)
        np.savez(self.run / "hawor.npz", **self.hawor)
        m = trimesh.Trimesh(vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
        m.export(self.run / "mesh.ply")
        # PLY export uses float32, matching the actual completed mesh contract.
        m = trimesh.load(self.run / "mesh.ply", force="mesh", process=False)
        self.annotations = {"frames": [{"frame_idx": int(fid), "time_s": i / 30.,
            "source_width": 640, "source_height": 480,
            "camera": {"T_world_camera_metric": T[i].tolist(), "intrinsics_fx_fy_cx_cy": K[i].tolist()},
            "hands": [{"hand_side": "left", "visibility_state": "visible" if valid[i] else "unresolved", "uncertainty": ["measurement_uncertainty"]}],
            "objects": [{"object_id": "thing", "visibility_state": "partially_visible"}]} for i, fid in enumerate(ids)]}
        self.annotations["v19_inputs"] = {"hawor_npz": str(self.run / "hawor.npz")}
        for i in (0, 2):
            self.annotations["frames"][i]["hands"][0]["metric_mano_state"] = {
                "vertices_reference": {"source_hawor_npz": str(self.run / "hawor.npz"),
                                       "source_frame_index": i,
                                       "source_vertices_world_array": "left_vertices_world_m",
                                       "source_joints_world_array": "left_joints_world_m"}}
        self.dump("annotations.json", self.annotations)
        self.dump("p15.json", {"annotation_ready": False, "pose_rows": []})
        self.dump("p16.json", {"constraint_rows": [{"original": True}]})
        self.dump("p18b.json", {"per_frame_states": [{"source_joints_preserved": True}]})
        self.dump("completion.json", {"outputs": {"completed_mesh": str(self.run / "mesh.ply")}})
        self.judgments = {"case": "synthetic", "interaction_judgments": [{"frame_start": 4, "frame_end": 6,
            "hand_side": "left", "contact_state": "unresolved", "occlusion_relation": "unresolved",
            "contact_support_uncertainty_m": .04}]}
        self.dump("p17.json", self.judgments)
        R = np.repeat(np.eye(3)[None], n, axis=0)
        t = np.tile([0., 0., 1.], (n, 1))
        self.core = {"method": "build_v19_rigid_render_state", "case": "synthetic", "object_id": "thing",
            "annotation_ready": False, "physical_state_quarantined": True,
            "inputs": {"annotations": str(self.run / "annotations.json"), "pose_report": str(self.run / "p15.json"),
                       "constraint_report": str(self.run / "p16.json"), "temporal_mano_state": str(self.run / "p18b.json"),
                       "completion_report": str(self.run / "completion.json"), "completed_mesh": str(self.run / "mesh.ply")},
            "annotation_backbone": {"frame_ids": ids.tolist()},
            "object_geometry": {"mesh_frame": "completed_canonical", "completed_mesh_path": str(self.run / "mesh.ply"),
                                "physical_surface": {"mesh": str(self.run / "mesh.ply"), "signed_geometry_ready": False}},
            "object_pose_trajectory": {"pose_rows": [{"frame_idx": int(ids[i]),
                "rotation_world_from_completed_canonical_matrix": R[i].tolist(), "translation_world_m": t[i].tolist()} for i in [0, 2]]},
            "mano_constraint_state": {"constraint_rows": [{"original": True}]},
            "temporal_mano_state": {"payload": {"per_frame_states": [{"source_joints_preserved": True}]}}}
        self.dump("core.json", self.core)
        self.demo = dict(frame_idx=ids, time_s=np.arange(n)/30., render_size_wh=np.array([320, 240]),
            intrinsics_render=np.c_[K[:, :2] / 2, (K[:, 2:] + .5) / 2 - .5], T_world_camera=T,
            object_vertices_canonical=np.asarray(m.vertices), object_faces=np.asarray(m.faces),
            base_object_rotation=R, base_object_translation_world=t, object_rotation=R,
            object_translation_world=t + [.03, 0, 0], left_valid=valid,
            left_vertices_world=vertices + [.02, 0, 0], left_joints_world=joints + [.02, 0, 0],
            left_base_vertices_world=vertices, left_base_joints_world=joints, left_faces=self.hawor["left_faces"])
        np.savez(self.run / "demo.npz", **self.demo)
        self.dump("demo.json", {"annotation_ready": True, "demo_only": True, "corrections": "test"})
        self.args = argparse.Namespace(render_state=self.run / "core.json", annotations=self.run / "annotations.json",
            hawor_npz=self.run / "hawor.npz", interaction_judgments=self.run / "p17.json",
            demo_state=self.run / "demo.npz", demo_report=self.run / "demo.json", output_dir=self.root / "final",
            run_root=self.run, allow_external_replay=False)

    def tearDown(self):
        self.tmp.cleanup()

    def dump(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def test_source_immutability_layer_separation_and_missing_hand(self):
        before = {str(p): sha256(p) for p in self.run.iterdir()}
        path = assemble(self.args)
        package, layers = load_package(path)
        self.assertEqual(package["metric_prediction"]["render_state"], self.core)
        self.assertEqual(package["metric_prediction"]["interaction_judgments"], self.judgments)
        self.assertFalse(package["demo_visual"]["annotation_ready"])
        self.assertFalse(package["metric_prediction"]["render_state"]["annotation_ready"])
        self.assertEqual(before, {str(p): sha256(p) for p in self.run.iterdir()})
        self.assertFalse(layers["prediction"]["right_valid"].any())
        self.assertNotIn("right_vertices_world", layers["prediction"])
        self.assertFalse(layers["prediction"]["object_valid"][1])
        self.assertTrue(layers["demo"]["object_valid"][1])
        self.assertAlmostEqual(layers["demo"]["left_vertices_world"][0, 0, 0] - layers["prediction"]["left_vertices_world"][0, 0, 0], .02)
        self.assertEqual(package["timeline"][1]["object"]["uncertainty"], ["missing_object_pose"])
        # Relocation and removal of every source prove renderer is package-only.
        shutil.rmtree(self.run)
        moved = self.root / "moved"
        shutil.move(self.args.output_dir, moved)
        load_package(moved / path.name)

    def test_actual_mesh_rrd_smoke(self):
        path = assemble(self.args)
        shutil.rmtree(self.run)
        target = render(path, self.root / "world.rrd")
        self.assertGreater(target.stat().st_size, 1000)
        # RRD carries actual mesh geometry components, not only labels.
        raw = target.read_bytes()
        self.assertIn(b"Mesh3D", raw)
        self.assertIn(b"/world/prediction", raw)
        self.assertIn(b"/world/demo", raw)

    def test_missing_components(self):
        del self.hawor["left_vertices_world_m"]
        np.savez(self.args.hawor_npz, **self.hawor)
        with self.assertRaisesRegex(ValueError, "required component"):
            assemble(self.args)

    def test_wrong_transform(self):
        self.annotations["frames"][0]["camera"]["T_world_camera_metric"][0][0] = 2
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "rigid rotation"):
            assemble(self.args)

    def test_timeline_camera_mesh_demo_mismatches(self):
        mutations = {"frame_idx": self.demo["frame_idx"][::-1],
                     "time_s": self.demo["time_s"] + .2,
                     "intrinsics_render": self.demo["intrinsics_render"] + 1,
                     "object_vertices_canonical": self.demo["object_vertices_canonical"] + .1,
                     "T_world_camera": self.demo["T_world_camera"] + .1,
                     "base_object_translation_world": self.demo["base_object_translation_world"] + .2}
        for key, value in mutations.items():
            with self.subTest(key=key):
                np.savez(self.args.demo_state, **{**self.demo, key: value})
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    assemble(self.args)

    def test_hash_missing_and_path_escape_rejected(self):
        path = assemble(self.args)
        original = json.loads(path.read_text())
        relative = original["metric_prediction"]["source_assets"]["hawor_npz"]
        asset = path.parent / relative
        raw = asset.read_bytes()
        asset.write_bytes(raw + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_package(path)
        asset.unlink()
        with self.assertRaises(FileNotFoundError):
            load_package(path)
        asset.write_bytes(raw)
        for bad in ("../escape.npz", "/tmp/escape.npz", "assets/../../escape.npz", "assets\\escape.npz"):
            with self.subTest(path=bad):
                data = json.loads(json.dumps(original))
                data["assets"][bad] = data["assets"].pop(relative)
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "path"):
                    load_package(path)

    def test_fresh_boundary_and_labeled_replay(self):
        external = self.root / "external.npz"
        shutil.copyfile(self.args.hawor_npz, external)
        self.args.hawor_npz = external
        with self.assertRaisesRegex(ValueError, "escapes run-root"):
            assemble(self.args)
        self.args.allow_external_replay = True
        path = assemble(self.args)
        self.assertEqual(load_package(path)[0]["run_mode"], "external_replay_not_fresh")

    def test_package_metadata_tampering(self):
        path = assemble(self.args)
        original = path.read_text()
        for field in ("frame_idx", "camera", "interaction_judgments", "authority"):
            with self.subTest(field=field):
                data = json.loads(original)
                if field == "frame_idx":
                    data["timeline"][0]["frame_idx"] += 1
                elif field == "camera":
                    data["timeline"][0]["camera"]["intrinsics_fx_fy_cx_cy"][0] += 1
                elif field == "interaction_judgments":
                    data["timeline"][0][field] = []
                else:
                    data["demo_visual"]["annotation_ready"] = True
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_package(path)

    def test_render_arrays_cannot_disagree_even_with_updated_hash(self):
        path = assemble(self.args)
        package = json.loads(path.read_text())
        relative = package["metric_prediction"]["render_arrays_asset"]
        asset = path.parent / relative
        with np.load(asset, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["intrinsics_render"][0, 0] += 10
        np.savez_compressed(asset, **arrays)
        package["assets"][relative]["sha256"] = sha256(asset)
        package["assets"][relative]["size_bytes"] = asset.stat().st_size
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "render array mismatch: intrinsics_render"):
            load_package(path)

    def test_demo_report_hash_and_source_validation(self):
        self.dump("demo.json", {"inputs": {"hawor_params": str(self.args.hawor_npz)},
                                "input_hashes": {"hawor_params": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            assemble(self.args)

    def test_explicit_missing_physical_components(self):
        self.core["temporal_mano_state"]["payload"] = None
        self.dump("core.json", self.core)
        with self.assertRaisesRegex(ValueError, "required component: P18"):
            assemble(self.args)

    def test_source_camera_and_validity_mismatch(self):
        self.hawor["R_c2w"][0, 0, 0] = 2
        np.savez(self.args.hawor_npz, **self.hawor)
        with self.assertRaisesRegex(ValueError, "HaWoR world camera rotation"):
            assemble(self.args)
        self.hawor["R_c2w"][0, 0, 0] = 1
        del self.hawor["left_valid"]
        np.savez(self.args.hawor_npz, **self.hawor)
        with self.assertRaisesRegex(ValueError, "required component: left_valid"):
            assemble(self.args)

    def test_declared_hawor_rejects_surface_and_invalid_row_substitution(self):
        for kind in ("vertices", "validity"):
            with self.subTest(kind=kind):
                replacement = {k: v.copy() for k, v in self.hawor.items()}
                if kind == "vertices":
                    replacement["left_vertices_world_m"][0, 300, 0] += .1
                else:
                    replacement["left_valid"][1] = True
                    replacement["left_vertices_world_m"][1] = 0
                    replacement["left_joints_world_m"][1] = 0
                self.args.hawor_npz = self.run / "replacement.npz"
                np.savez(self.args.hawor_npz, **replacement)
                with self.assertRaisesRegex(ValueError, "source archive mismatch"):
                    assemble(self.args)

    def test_per_hand_source_binding_without_global_reference(self):
        del self.annotations["v19_inputs"]
        self.dump("annotations.json", self.annotations)
        self.args.hawor_npz = self.run / "replacement.npz"
        replacement = {k: v.copy() for k, v in self.hawor.items()}
        replacement["left_valid"][1] = True
        replacement["left_vertices_world_m"][1] = 0
        replacement["left_joints_world_m"][1] = 0
        np.savez(self.args.hawor_npz, **replacement)
        with self.assertRaisesRegex(ValueError, "source archive mismatch"):
            assemble(self.args)

    def test_declared_source_array_and_row_binding(self):
        ref = self.annotations["frames"][0]["hands"][0]["metric_mano_state"]["vertices_reference"]
        ref["source_frame_index"] = 2
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "source row mismatch"):
            assemble(self.args)
        ref["source_frame_index"] = 0
        ref["source_vertices_world_array"] = "left_joints_world_m"
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "source vertices mismatch"):
            assemble(self.args)

    def test_hawor_binding_replayed_after_relocation_and_rejects_substitution(self):
        # Content-identical copies are legitimate, but both identities are saved.
        self.args.hawor_npz = self.run / "copy.npz"
        shutil.copyfile(self.run / "hawor.npz", self.args.hawor_npz)
        path = assemble(self.args)
        shutil.rmtree(self.run)
        moved = self.root / "moved"
        shutil.move(path.parent, moved)
        path = moved / path.name
        package, _ = load_package(path)
        relative = package["metric_prediction"]["source_assets"]["hawor_npz"]
        asset = moved / relative
        replacement = {k: v.copy() for k, v in self.hawor.items()}
        replacement["left_vertices_world_m"][0, 300, 0] += .1
        np.savez(asset, **replacement)
        package["assets"][relative].update(sha256=sha256(asset), size_bytes=asset.stat().st_size)
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "source archive mismatch"):
            load_package(path)
        package["metric_prediction"]["hawor_source_bindings"] = {}
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "source bindings mismatch"):
            load_package(path)

    def test_same_snapshot_hawor_identity_survives_manifest_rehash(self):
        path = assemble(self.args)
        package = json.loads(path.read_text())
        relative = package["metric_prediction"]["source_assets"]["hawor_npz"]
        replacement = {k: v.copy() for k, v in self.hawor.items()}
        replacement["left_vertices_world_m"][0, 300, 0] += .1
        asset = path.parent / relative
        np.savez(asset, **replacement)
        package["assets"][relative].update(sha256=sha256(asset), size_bytes=asset.stat().st_size)
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "source archive mismatch"):
            load_package(path)

    def test_prefixed_object_track_metadata_and_ambiguity(self):
        for frame in self.annotations["frames"]:
            frame["objects"][0].update(object_id="object:thing", track_id="producer_track", occluder="hand:left")
        self.judgments["interaction_judgments"][0]["target_entity_id"] = "producer_track"
        self.dump("p17.json", self.judgments)
        self.dump("annotations.json", self.annotations)
        path = assemble(self.args)
        package, _ = load_package(path)
        self.assertEqual(package["timeline"][0]["object"]["source_state"], self.annotations["frames"][0]["objects"][0])
        self.assertEqual(package["timeline"][0]["object"]["occluder"], "hand:left")
        self.assertEqual(package["timeline"][0]["interaction_judgments"], self.judgments["interaction_judgments"])
        self.args.output_dir = self.root / "ambiguous"
        self.annotations["frames"][0]["objects"].append({"object_id": "thing"})
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "ambiguous object/track"):
            assemble(self.args)

    def test_obj_external_material_reads_denied_in_assembly_and_replay(self):
        self.args.demo_state = self.args.demo_report = None
        external = self.root / "outside.mtl"
        external.write_text("newmtl forbidden\nKd 1 0 0\n")
        mesh_path = self.run / "mesh.obj"
        original_open, original_io_open = builtins.open, io.open
        attempted = []
        def guarded(opener):
            def open_file(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file).name == "outside.mtl":
                    attempted.append(str(file))
                    raise AssertionError("external material read")
                return opener(file, *args, **kwargs)
            return open_file
        for i, reference in enumerate((str(external), "../outside.mtl")):
            with self.subTest(reference=reference):
                mesh_path.write_text(f"mtllib {reference}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl forbidden\nf 1 2 3\n")
                self.core["object_geometry"]["completed_mesh_path"] = str(mesh_path)
                self.dump("core.json", self.core)
                self.args.output_dir = self.root / f"obj_final_{i}"
                with mock.patch("builtins.open", guarded(original_open)), mock.patch("io.open", guarded(original_io_open)):
                    path = assemble(self.args)
                    # In replay, ../outside.mtl would resolve here rather than
                    # relative to the deleted/original mesh source directory.
                    with original_io_open(path.parent / "outside.mtl", "w") as f:
                        f.write("newmtl forbidden\n")
                    load_package(path)
                self.assertEqual(attempted, [])

    def distinct_demo_mesh(self):
        generated = trimesh.Trimesh(
            vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1], [.1, .1, .1]],
            faces=[[0, 1, 2], [0, 2, 3], [1, 3, 4], [1, 2, 4]], process=False)
        path = self.run / "intact_generated.ply"
        generated.export(path)
        generated = trimesh.load(path, force="mesh", process=False)
        completion = {"outputs": {"completed_mesh": str(self.run / "mesh.ply")},
                      "render_mesh_contract": {"required_mesh_path": str(path),
                                               "required_mesh_sha256": sha256(path), "renderer_must_match": True}}
        self.dump("completion.json", completion)
        self.demo["object_vertices_canonical"] = np.asarray(generated.vertices)
        self.demo["object_faces"] = np.asarray(generated.faces)
        np.savez(self.args.demo_state, **self.demo)
        self.dump("demo.json", {"demo_only": True, "mesh": str(path), "mesh_sha256": sha256(path)})
        return path, completion

    def test_distinct_completion_bound_demo_mesh_relocated_rrd(self):
        _, completion = self.distinct_demo_mesh()
        before = {str(p): sha256(p) for p in self.run.iterdir()}
        path = assemble(self.args)
        self.assertEqual(before, {str(p): sha256(p) for p in self.run.iterdir()})
        shutil.rmtree(self.run)
        moved = self.root / "relocated"
        shutil.move(path.parent, moved)
        package, layers = load_package(moved / path.name)
        self.assertEqual(package["metric_prediction"]["render_state"], self.core)
        self.assertEqual(len(layers["prediction"]["object_vertices_canonical"]), 4)
        self.assertEqual(len(layers["demo"]["object_vertices_canonical"]), 5)
        self.assertEqual(len(layers["prediction"]["object_faces"]), 2)
        self.assertEqual(len(layers["demo"]["object_faces"]), 4)
        binding = package["demo_visual"]["canonical_mesh_binding"]
        self.assertEqual(binding["reference"], completion["render_mesh_contract"]["required_mesh_path"])
        target = render(moved / path.name, self.root / "distinct.rrd")
        self.assertGreater(target.stat().st_size, 1000)

    def test_rrd_canonical_mesh_logged_per_valid_segment_and_restored(self):
        import rerun as rr
        self.distinct_demo_mesh()
        path = assemble(self.args)
        _, layers = load_package(path)
        events = []
        current = None
        real_log, real_set_time = rr.log, rr.set_time
        def set_time(timeline, **kwargs):
            nonlocal current
            if timeline == "frame":
                current = kwargs["sequence"]
            return real_set_time(timeline, **kwargs)
        def log(entity, value, **kwargs):
            events.append((current, entity, value))
            return real_log(entity, value, **kwargs)
        with mock.patch.object(rr, "log", side_effect=log), mock.patch.object(rr, "set_time", side_effect=set_time):
            render(path, self.root / "segments.rrd")
        for layer, mesh_frames, transform_frames in (("prediction", [4, 6], [4, 6]), ("demo", [4], [4, 5, 6])):
            root = f"/world/{layer}/object"
            meshes = [(fid, value) for fid, entity, value in events if entity == root + "/mesh"]
            self.assertEqual([fid for fid, _ in meshes], mesh_frames)
            self.assertEqual([fid for fid, entity, value in events if entity == root and isinstance(value, rr.Transform3D)], transform_frames)
            for _, value in meshes:
                self.assertEqual(value.triangle_indices.as_arrow_array().to_pylist(), layers[layer]["object_faces"].tolist())
                np.testing.assert_allclose(value.vertex_positions.as_arrow_array().to_pylist(), layers[layer]["object_vertices_canonical"])
        self.assertEqual([fid for fid, entity, value in events if entity == "/world/prediction/object" and isinstance(value, rr.Clear)], [5])

    def test_distinct_demo_mesh_without_contract_rejected(self):
        self.distinct_demo_mesh()
        self.dump("completion.json", {"outputs": {"completed_mesh": str(self.run / "mesh.ply")}})
        with self.assertRaisesRegex(ValueError, "demo canonical mesh"):
            assemble(self.args)

    def test_distinct_demo_mesh_bad_hash_or_substitution_rejected(self):
        path, completion = self.distinct_demo_mesh()
        completion["render_mesh_contract"]["required_mesh_sha256"] = "0" * 64
        self.dump("completion.json", completion)
        with self.assertRaisesRegex(ValueError, "bound canonical mesh hash mismatch"):
            assemble(self.args)
        completion["render_mesh_contract"]["required_mesh_sha256"] = sha256(path)
        self.dump("completion.json", completion)
        np.savez(self.args.demo_state, **{**self.demo, "object_vertices_canonical": self.demo["object_vertices_canonical"] + .1})
        with self.assertRaisesRegex(ValueError, "demo canonical mesh"):
            assemble(self.args)
        np.savez(self.args.demo_state, **self.demo)
        self.dump("demo.json", {"mesh": str(self.run / "mesh.ply"), "mesh_sha256": sha256(self.run / "mesh.ply")})
        with self.assertRaisesRegex(ValueError, "demo report canonical mesh mismatch"):
            assemble(self.args)

    def test_distinct_demo_retains_base_pose_camera_mano_checks(self):
        self.distinct_demo_mesh()
        for key in ("base_object_translation_world", "T_world_camera", "left_base_vertices_world"):
            with self.subTest(key=key):
                np.savez(self.args.demo_state, **{**self.demo, key: self.demo[key] + .1})
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    assemble(self.args)

    def test_distinct_demo_replay_requires_original_contract_binding(self):
        self.distinct_demo_mesh()
        path = assemble(self.args)
        shutil.rmtree(self.run)
        original = path.read_text()
        package = json.loads(original)
        package["demo_visual"]["canonical_mesh_binding"]["asset"] = package["metric_prediction"]["canonical_mesh_asset"]
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "bound canonical mesh hash mismatch"):
            load_package(path)
        package = json.loads(original)
        package["demo_visual"]["canonical_mesh_binding"]["reference"] = "unbound_mesh.ply"
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "canonical mesh binding mismatch"):
            load_package(path)
        package = json.loads(original)
        relative = package["metric_prediction"]["source_assets"]["completion_report"]
        report_path = path.parent / relative
        report = json.loads(report_path.read_text())
        report["render_mesh_contract"]["required_mesh_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))
        package["assets"][relative].update(sha256=sha256(report_path), size_bytes=report_path.stat().st_size)
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "completion report binding hash mismatch"):
            load_package(path)

    def test_prediction_only(self):
        self.args.demo_state = self.args.demo_report = None
        path = assemble(self.args)
        self.assertEqual(list(load_package(path)[1]), ["prediction"])
        with self.assertRaisesRegex(ValueError, "demo layer"):
            render(path, self.root / "no.rrd", "demo")

if __name__ == '__main__':
    unittest.main()
