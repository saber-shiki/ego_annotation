"""Mechanism tests: non-identity poses, Jacobians, frames and missing evidence."""
import importlib.util
import sys
import unittest
import tempfile
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('late_material_tests',ROOT/'scripts/fit_v20_late_window_prediction_only_se3.py')
late=importlib.util.module_from_spec(spec);sys.modules[spec.name]=late;spec.loader.exec_module(late)


def make_fixture():
    rs=Rotation.from_euler('xyz',[.2,-.3,.15]).as_matrix()
    rt=Rotation.from_euler('xyz',[-.12,.5,-.25]).as_matrix()
    ts=np.array([.15,.01,.6]);tt=np.array([.04,-.02,.72])
    q=np.array([[-.02,-.03,0],[.02,-.025,.01],[-.015,.03,-.01],[.035,.02,.03],[0,.015,.025]])
    xs=late.apply_pose(q,rs,ts);xt=late.apply_pose(q,rt,tt)
    rc=Rotation.from_euler('xyz',[.03,-.08,.04]).as_matrix();tc=np.array([.01,.02,-.01])
    camera=late.inverse_pose(xt,rc,tc);K=np.array([650.,630.,480.,470.]);uv=camera[:,:2]/camera[:,2,None]*K[:2]+K[2:]
    nodes=[late.PoseNode(120,rs,ts,xs,'synthetic',True,0),late.PoseNode(130,rt,tt,xt,'synthetic',True,0)]
    factor=late.RGBReprojectionFactor(120,130,0,1,xs,uv,K,rc,tc,np.ones(len(q)),1.,True)
    return nodes,factor,{120:(rs,ts),130:(rt,tt)}


class FactorDependencies(unittest.TestCase):
    def test_true_nontrivial_motion_is_zero_reprojection(self):
        nodes,factor,poses=make_fixture();cfg=late.SolverConfig(rgb_reprojection_weight_scale=1.)
        np.testing.assert_allclose(late.rgb_reprojection_residual(factor,poses,nodes,cfg),0,atol=1e-10)
        before=factor.source_world.copy()
        for idx in [120,130]:
            moved=dict(poses);r,t=moved[idx];moved[idx]=(r,t+np.array([.01,0,0]))
            self.assertGreater(np.linalg.norm(late.rgb_reprojection_residual(factor,moved,nodes,cfg)),1.)
        np.testing.assert_array_equal(before,factor.source_world)

    def test_pixel_jacobian_has_both_endpoint_blocks(self):
        nodes,factor,poses=make_fixture();cfg=late.SolverConfig(rgb_reprojection_weight_scale=1.)
        f=lambda x:late.rgb_reprojection_residual(factor,late.current_poses(nodes,x),nodes,cfg)
        x=np.zeros(12);eps=1e-6;J=np.column_stack([(f(x+np.eye(12)[k]*eps)-f(x-np.eye(12)[k]*eps))/(2*eps) for k in range(12)])
        self.assertTrue(np.all(np.linalg.norm(J,axis=0)>1e-4))
        full=late.pose_residual(x,nodes,[],[],120,cfg,[factor]);sp=late.pose_residual_sparsity(nodes,[],[],120,[factor])
        self.assertEqual(sp.shape,(len(full),len(x)))

    def test_low_conditioning_pixels_have_zero_rotation_jacobian(self):
        nodes,factor,poses=make_fixture();factor=replace(factor,rotation_eligible=False);cfg=late.SolverConfig(rgb_reprojection_weight_scale=1.)
        original=late.rgb_reprojection_residual(factor,poses,nodes,cfg)
        for idx in [120,130]:
            moved=dict(poses);r,t=moved[idx];moved[idx]=(Rotation.from_euler('xyz',[.2,.1,.3]).as_matrix()@r,t)
            np.testing.assert_array_equal(late.rgb_reprojection_residual(factor,moved,nodes,cfg),original)

    def test_translation_only_true_motion_rotation_jacobian_and_world_origin(self):
        nodes,_,poses=make_fixture();rs,ts=poses[120];rt,tt=poses[130];Rm=rt@rs.T;tm=tt-Rm@ts
        c=nodes[0].observed_world.mean(0)
        edge=late.RGBEdge(120,130,Rm,tm,1.,{'translation_only':True},Rm.copy(),c.copy())
        r,t=late.rgb_edge_residual(edge,poses);np.testing.assert_allclose(t,0,atol=1e-12)
        for idx in [120,130]:
            moved=dict(poses);R,T=moved[idx];moved[idx]=(Rotation.from_euler('y',1,degrees=True).as_matrix()@R,T)
            np.testing.assert_array_equal(late.rgb_edge_residual(edge,moved)[1],t)
        shift=np.array([12.,-3.,8.]);rebased={idx:(R,T+shift) for idx,(R,T) in poses.items()}
        edge2=replace(edge,translation_source_to_target_world_m=tm+(np.eye(3)-Rm)@shift,translation_source_support_world=c+shift)
        np.testing.assert_allclose(late.rgb_edge_residual(edge2,rebased)[1],t,atol=1e-12)
        moved=dict(poses);moved[130]=(rt,tt+np.array([.01,0,0]));self.assertAlmostEqual(np.linalg.norm(late.rgb_edge_residual(edge,moved)[1]),.01)

    def test_translation_only_missing_reference_fails_closed(self):
        nodes,_,poses=make_fixture()
        edge=late.RGBEdge(120,130,np.eye(3),np.zeros(3),1.,{'translation_only':True})
        with self.assertRaisesRegex(ValueError,'fixed orientation'):
            late.rgb_edge_residual(edge,poses)


class MaterialMechanism(unittest.TestCase):
    def test_metric_and_pixel_true_motion_and_jacobians(self):
        nodes,rp,poses=make_fixture()
        mf=late.material.MaterialFactor(120,130,0,1,rp.source_world,nodes[1].observed_world,
            rp.target_uv,np.block([[rp.target_camera_rotation,rp.target_camera_translation[:,None]],
                                  [np.array([[0.,0.,0.,1.]])]]),rp.target_intrinsics,np.ones(5,bool),np.ones(5),1.)
        cfg=late.SolverConfig(observed_factor_mode='material_tracks')
        f=lambda x:late.material.residual(mf,late.current_poses(nodes,x),image_weight=.25,metric_weight=1.,sigma_px=2.,sigma_m=.008)
        np.testing.assert_allclose(f(np.zeros(12)),0,atol=1e-10)
        eps=1e-6;x=np.zeros(12)
        J=np.column_stack([(f(x+np.eye(12)[k]*eps)-f(x-np.eye(12)[k]*eps))/(2*eps) for k in range(12)])
        self.assertTrue(np.all(np.linalg.norm(J,axis=0)>1e-3))
        result=late.pose_residual(x,nodes,[],[],120,cfg,[],[mf])
        sparsity=late.pose_residual_sparsity(nodes,[],[],120,[],[mf])
        self.assertEqual(sparsity.shape,(len(result),12))
        self.assertEqual(sparsity[12:42].nnz,30*12)

    def test_missing_target_depth_has_no_metric_residual(self):
        nodes,rp,poses=make_fixture()
        C=np.eye(4);C[:3,:3]=rp.target_camera_rotation;C[:3,3]=rp.target_camera_translation
        mf=late.material.MaterialFactor(120,130,0,1,rp.source_world,np.zeros((5,3)),rp.target_uv,C,rp.target_intrinsics,np.zeros(5,bool),np.ones(5),1.)
        uv,xyz,z=late.material.errors(mf,poses)
        np.testing.assert_array_equal(xyz,0)
        r=late.material.residual(mf,poses,image_weight=.25,metric_weight=1.,sigma_px=2.,sigma_m=.008).reshape(5,6)
        np.testing.assert_array_equal(r[:,2:5],0)
        self.assertEqual(late.material.metrics([mf],poses)['material_metric_count'],0)
        moved=dict(poses);R,t=moved[130];moved[130]=(R,t+[.01,0,0])
        self.assertGreater(np.linalg.norm(late.material.errors(mf,moved)[0]),1)

    def test_missing_metric_cannot_be_smuggled_through_bundle(self):
        nodes,rp,poses=make_fixture();nodes[1].metric_observation=False
        C=np.eye(4);C[:3,:3]=rp.target_camera_rotation;C[:3,3]=rp.target_camera_translation
        meta={'schema':'v20_fixed_world_material_rgbd_v1','object_id':'obj','gt_consumed':False,
              'source_frame':'fixed_world_observation','input_sha256':{'annotations':'test'}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'factors.npz'
            values=dict(metadata=np.asarray([json.dumps(meta)]),source_frame_idx=np.array([120]),target_frame_idx=np.array([130]),offsets=np.array([0,5]),
                source_world=rp.source_world,target_world=nodes[1].observed_world,target_uv=rp.target_uv,target_camera=np.array([C]),
                target_intrinsics=rp.target_intrinsics[None],weights=np.ones(5),edge_weight=np.ones(1),metric_valid=np.ones(5,bool),
                source_metric_eligible=np.ones(1,bool),target_metric_eligible=np.ones(1,bool))
            np.savez_compressed(path,**values)
            with self.assertRaisesRegex(ValueError,'fabricated metric'):
                late.material.load_factors(path,nodes,annotation_sha256='test',object_id='obj')
            values['metric_valid']=np.zeros(5,bool);values['target_metric_eligible']=np.zeros(1,bool)
            np.savez_compressed(path,**values)
            ff,diag=late.material.load_factors(path,nodes,annotation_sha256='test',object_id='obj')
            self.assertEqual(diag['metric_points'],0)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                late.material.load_factors(path,nodes,annotation_sha256='wrong',object_id='obj')

    def test_legacy_canonical_must_bind_producer_pose(self):
        nodes,rp,poses=make_fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'producer.json'
            report={'object_id':'obj','gt_consumed':False,'pose_rows':[
                {'frame_idx':i,'rotation_world_from_completed_canonical_matrix':R.tolist(),'translation_world_m':t.tolist()}
                for i,(R,t) in poses.items()]}
            p.write_text(json.dumps(report))
            meta={'object_id':'obj','initial_pose_report':str(p),'input_sha256':{'initial_pose_report':late.sha256_file(p)}}
            q=late.inverse_pose(rp.source_world,*poses[120])
            arrays={'metadata':np.array([json.dumps(meta)]),'source_frame_idx':np.array([120]),
                    'reprojection_evidence_offsets':np.array([0,len(q)]),'reprojection_canonical_points':q}
            world,_=late.fixed_rgb_source_world(arrays)
            np.testing.assert_allclose(world,rp.source_world,atol=1e-12)
            p.write_text(json.dumps({**report,'tampered':True}))
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                late.fixed_rgb_source_world(arrays)


class InputHardening(unittest.TestCase):
    def test_invalid_source_depth_mask(self):
        import build_v20_rgbd_material_factors as builder
        valid=builder.valid_source_depth_mask(np.array([.4,np.nan,0.,-.1,np.inf]),np.array([.4,.4,.4,.4,.4]),np.zeros(5))
        np.testing.assert_array_equal(valid,[True,False,False,False,False])
        with self.assertRaisesRegex(ValueError,'source depth'):
            builder.valid_source_depth_mask(np.array([.5]),np.array([.4]),np.ones(1))

    def test_world_points_also_require_producer_identity(self):
        data={'metadata':np.array([json.dumps({'gt_consumed':False,'reprojection_evidence_contract':{'source_frame':'fixed_world_observation'}})]),
              'reprojection_source_world':np.zeros((4,3))}
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
            late.fixed_rgb_source_world(data)


if __name__=='__main__':unittest.main()
