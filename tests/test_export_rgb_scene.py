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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_rgb_scene import export,timeline
from rgb_scene_visibility import apply_display_visibility, validate_visibility
from rgb_scene_identity import InputIdentity, input_identity
from rgb_scene_pose_io import RKEY,TKEY
from v19_runtime_contracts import sha


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
                root=Path(td);z,b=fixture(object_id,size)
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
        import fit_rgb_scene_visual as fitter
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);z,b=fixture();out=root/'fit';out.mkdir()
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
            with patch.object(sys,'argv',argv),patch.object(fitter.runpy,'run_path',generic):fitter.main()
            with np.load(out/'demo_state.npz',allow_pickle=False) as archive:state=dict(archive)
            np.testing.assert_array_equal(state['object_valid'],[True,False,True])
            for key in ('object_rotation','object_translation_world','left_valid','left_vertices_world'):
                np.testing.assert_array_equal(state[key],z[key])
            doc=export(state,b,'case_b','vessel','track',root/'final',visibility=judgment(),identity=identity())
            self.assertIsNone(doc['frames'][1]['object']['geometry'])
            self.assertIsNotNone(doc['frames'][2]['object']['geometry'])
            self.assertTrue(batches(root/'final/world.rrd'))
            incomplete=judgment();incomplete['frames'].pop();j.write_text(json.dumps(incomplete))
            with patch.object(sys,'argv',argv),patch.object(fitter.runpy,'run_path',side_effect=AssertionError('must reject before fitting')):
                with self.assertRaises(ValueError):fitter.main()

    def test_input_identity_reads_current_contract_and_rejects_tampering(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'raw.mp4';raw.write_bytes(b'temporary identity fixture')
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

if __name__=='__main__':unittest.main()
