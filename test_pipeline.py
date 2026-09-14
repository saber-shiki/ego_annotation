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

if __name__=='__main__':unittest.main()

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
                    doc=export(z,b,'case_b',object_id,'track_d',root/'internal/final',visibility=judgment(object_id),identity=identity())
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
            with self.assertRaisesRegex(ValueError,'reset'):export(bad,b,'c','vessel','vessel',Path(td)/'bad',visibility=judgment(),identity=identity('c'))
            for mutate in [lambda d:d.update(frame_idx=np.array([7,7,103])),lambda d:d.update(time_s=np.array([.1,.1,1.])),lambda d:d.update(object_faces=np.array([[0,1,99]])),lambda d:d.update(intrinsics_render=np.zeros((3,4))),lambda d:d['object_rotation'].__setitem__(0,np.zeros((3,3)))]:
                bad=copy.deepcopy(z);mutate(bad)
                with self.assertRaises(ValueError):timeline(bad)
            bad=copy.deepcopy(b);bad['object_id']='other'
            with self.assertRaises(ValueError):export(z,bad,'c','vessel','vessel',Path(td)/'wrong',visibility=judgment(),identity=identity('c'))

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
            doc=export(state,b,'case_b','vessel','track',root/'final',visibility=judgment(),identity=identity())
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
