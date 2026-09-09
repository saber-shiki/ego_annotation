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
       'object_valid':np.array([True,False,True]),'interaction_states':np.full((n,2),'unresolved')}
    rng=np.random.default_rng(5)
    for side in ('left','right'):
        z[side+'_valid']=np.array([True,False,True]);z[side+'_vertices_world']=rng.normal(size=(n,778,3))*.01+[0,0,1.]
        z[side+'_vertices_world'][1]=np.nan
        z[side+'_faces']=np.array([[i,i+1,i+2] for i in range(776)])
    binding={'schema':'rgb_scene_visual_refined_binding_v1','object_id':object_id,
        'canonical_gauge':{'name':'initial_completed_canonical','anchor_frame_idx':7},
        'source_prior':{'sha256':'a'*64},'refined_posterior':{'sha256':'b'*64},
        'source_pose_rows':[{'frame_idx':int(i),RKEY:np.eye(3).tolist(),TKEY:[0,0,1]} for i in ids],
        'pose_rows':[{'frame_idx':int(i),'time_s':float(t),RKEY:r.tolist(),TKEY:p.tolist(),'uncertainty':.7,'metric_observation':False} for i,t,r,p in zip(ids,times,rotations,translations)]}
    return z,binding


def batches(path):
    from rerun.experimental import RrdReader
    return [c.to_record_batch() for c in RrdReader(path).stream().to_chunks()]


class ExportTests(unittest.TestCase):
    def test_portable_pair_actual_rrd_geometry_time_clears_and_camera(self):
        for object_id,size in [('vessel',(640,360)),('tool',(320,640))]:
            with self.subTest(object_id=object_id),tempfile.TemporaryDirectory() as td:
                root=Path(td);z,b=fixture(object_id,size)
                with patch('subprocess.Popen',side_effect=AssertionError('encoder/subprocess forbidden')):
                    doc=export(z,b,'case_b',object_id,'track_d',root/'internal/final')
                self.assertEqual(sorted(p.name for p in (root/'internal/final').iterdir()),['annotation.json','world.rrd'])
                shutil.copytree(root/'internal/final',root/'portable');shutil.rmtree(root/'internal')
                doc=json.loads((root/'portable/annotation.json').read_text())
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
                    self.assertTrue(found_v,entity);self.assertTrue(found_f,entity);self.assertTrue(clears,entity)
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
            with self.assertRaisesRegex(ValueError,'reset'):export(bad,b,'c','vessel','vessel',Path(td)/'bad')
            for mutate in [lambda d:d.update(frame_idx=np.array([7,7,103])),lambda d:d.update(time_s=np.array([.1,.1,1.])),lambda d:d.update(object_faces=np.array([[0,1,99]])),lambda d:d.update(intrinsics_render=np.zeros((3,4))),lambda d:d['object_rotation'].__setitem__(0,np.zeros((3,3)))]:
                bad=copy.deepcopy(z);mutate(bad)
                with self.assertRaises(ValueError):timeline(bad)
            bad=copy.deepcopy(b);bad['object_id']='other'
            with self.assertRaises(ValueError):export(z,bad,'c','vessel','vessel',Path(td)/'wrong')

if __name__=='__main__':unittest.main()
