"""Synthetic closed-triangle tests; production always consumes native geometry."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import trimesh

from data.test_coarse_training import fixture
from data.immutable import freeze
from envs.test_object_only_coarse_env import fixture_config
from envs.generated_collision_coarse_env import GeneratedCollisionCoarseEnv
from utils.generated_collision import CONTRACT, GeneratedCollisionScene, intervals


def config(models):
    cfg = fixture_config(models)
    cfg.pop('object_only')
    cfg['generated_collision'] = {'contract':CONTRACT,'trust_complete_generated_mesh':True,'epsilon_m':.0001,'nsamples':5,'weight':300000.}
    return cfg


def closed_fixture(root):
    b,s,a,m=fixture(root)
    arrays=dict(b.load());objects=dict(arrays['objects']);key=next(iter(objects));obj=dict(objects[key])
    # Synthetic fixture's tetrahedron has inward winding. This creates a new
    # synthetic dataset with consistent outward winding; production has no repair.
    obj['faces']=np.array(obj['faces'])[:,[0,2,1]].copy()
    obj['pose_faces']=obj['faces'];obj['collision_faces']=obj['faces']
    objects[key]=freeze(obj);arrays['objects']=freeze(objects);b._cache=freeze(arrays)
    return b,m


class GeneratedCollisionTests(unittest.TestCase):
    def test_signed_distance_rigid_world_transform_and_outward_improvement(self):
        mesh=trimesh.creation.box(extents=[.02,.02,.02])
        scene=GeneratedCollisionScene(mesh.vertices,mesh.faces)
        faces=np.array([[0,1,2]],np.int64)
        v=np.array([[.009,0,0],[.009,.001,0],[.009,0,.001]])
        base=scene.query(v,faces,np.eye(3),np.zeros(3))
        moved=scene.query(v+[.002,0,0],faces,np.eye(3),np.zeros(3))
        self.assertGreater(base['depth_m'].min(),.0009)
        self.assertLess(moved['depth_m'].max(),0)
        from scipy.spatial.transform import Rotation
        R=Rotation.from_rotvec([.3,-.1,.2]).as_matrix();t=np.array([1.,2.,3.])
        transformed=scene.query(v@R.T+t,faces,R,t)
        np.testing.assert_allclose(base['depth_m'],transformed['depth_m'],atol=1e-7)
        with self.assertRaises(ValueError):GeneratedCollisionScene(mesh.vertices,mesh.faces[:-1])

    def test_actual_triangle_edge_crossing_with_no_inside_vertices(self):
        mesh=trimesh.creation.box(extents=[.02,.02,.02]);scene=GeneratedCollisionScene(mesh.vertices,mesh.faces)
        v=np.array([[-.02,0,0],[.02,0,0],[0,.02,.02]])
        row=scene.query(v,np.array([[0,1,2]]),np.eye(3),np.zeros(3),edges=True)
        self.assertTrue((row['vertex_depth_m']<0).all())
        self.assertGreater(row['edge_crossings'],0)
        self.assertEqual(intervals([1,2,3,5,6]),[[1,3],[5,6]])

    def test_full_action_probability_return_provenance_and_cache_invalidation(self):
        with TemporaryDirectory() as td:
            b,m=closed_fixture(Path(td));cfg=config(m)
            env=GeneratedCollisionCoarseEnv([b],cfg);obs,_=env.reset(seed=0)
            self.assertEqual(env.action_space.shape,(118,));self.assertEqual(obs.shape,(551,))
            source=env.state.episode.objects['object:0'];self.assertFalse(source['signed_geometry_ready']);self.assertFalse(np.any(source['signed_authority']))
            rewards=[]
            for i in range(3):_,r,done,_,info=env.step(np.zeros(118,np.float32));rewards.append(r)
            self.assertTrue(done);self.assertAlmostEqual(sum(rewards),-1e-5*info['objective'],places=9)
            self.assertAlmostEqual(info['original_formal_objective']+info['generated_collision_objective'],info['objective'],places=8)
            self.assertTrue(any(k.startswith('hypothesis/') for k in info['factor_ledger']['terms']))
            export=env.export_annotation(Path(td)/'zero.json')
            contract=export['document']['preprocessed_state_adapter'];self.assertFalse(contract['hands_frozen']);self.assertFalse(contract['source_signed_authority_promoted'])
            env.reset(seed=0);initial={s:env.state.hand_vertices[s][0].copy() for s in ('left','right')};layout=env.state.episode.layout
            a=np.zeros(118,np.float32)
            for side in ('left','right'):
                a[layout.block(f'hand.{side}.translation').start]=.2
                a[layout.block(f'hand.{side}.articulation').start]=.2
                a[layout.block(f'contact.{side}').slice]=.1
            _,_,_,_,info=env.step(a)
            np.testing.assert_array_equal(info['raw_action'],a)
            for s in initial:self.assertGreater(np.linalg.norm(env.state.hand_vertices[s][0]-initial[s]),0)
            self.assertGreater(np.abs(info['executed_delta']).sum(),0)
            from envs.dynamics import decode_candidate
            changed=env.state.copy();changed.position=0
            env.evaluator.objective(changed,1)
            a=np.zeros(118);a[layout.block('object.object:0.translation').start]=.002
            decode_candidate(changed,a,cfg)
            self.assertNotIn(0,changed.term_cache)
            env.evaluator.objective(changed,1)
            self.assertIn('generated_collision_hypothesis',changed.term_cache[0])
            self.assertEqual(len(env.evaluator._scenes),1)

    def test_operator_trust_is_explicit_and_hand_mask_forbidden(self):
        with TemporaryDirectory() as td:
            b,m=closed_fixture(Path(td));cfg=config(m)
            cfg['generated_collision']['trust_complete_generated_mesh']=False
            with self.assertRaises(ValueError):GeneratedCollisionCoarseEnv([b],cfg)
            cfg=config(m);cfg['hard_action_mask']=[True]*118
            with self.assertRaises(ValueError):GeneratedCollisionCoarseEnv([b],cfg)


if __name__=='__main__':unittest.main()
