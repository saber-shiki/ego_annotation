"""Milestone-one V7 differential tests, no model execution/optimization/training.

Named pure functions are extracted from frozen git objects, never imported as
whole scripts. Optional real milk artifacts are read-only reference fixtures;
legacy canonical RGB points are decoded only inside this test, not admitted to a
production source bundle or relabeled as a new prediction.
"""
import ast
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from data.immutable import thaw
from data.interaction import load_interaction_evidence
from data.interaction_features import compile_interaction_timeline
from data.motion import prepare_motion
from observations.object import rgb_tracking, surface, shape
from envs.qualification import object_observation
from utils.data import PoseNode, RGBEdge, RGBReprojectionFactor
from utils.state import sha


class FrozenReference:
    def __init__(self, root):
        self.root=Path(root);self.sources={}

    def functions(self,path,names,namespace=None):
        source=subprocess.check_output(['git','-C',str(self.root),'show','67c40e6:'+path],text=True)
        definitions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in definitions}!=set(names):raise ValueError('missing frozen reference function')
        self.sources[path]={'commit':'67c40e6','sha256':hashlib.sha256(source.encode()).hexdigest(),'functions':sorted(names)}
        values={'np':np,'cv2':cv2,'math':math,'replace':replace,'Rotation':Rotation,'Any':Any}
        values.update(namespace or {})
        tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*definitions],type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree),path,'exec'),values)
        return values


@unittest.skipUnless(os.environ.get('EGO_V7_REFERENCE'),'requires explicit frozen V7 reference')
class V7ObservationKernelParityTests(unittest.TestCase):
    def setUp(self):self.ref=FrozenReference(os.environ['EGO_V7_REFERENCE'])

    def test_front_only_classification_is_identical(self):
        reference=self.ref.functions('scripts/rgb_scene_depth_order.py',['front_only'])['front_only']
        mask=np.ones((7,9),bool);depth=np.ones(mask.shape);hand=np.full(mask.shape,np.inf)
        hand[:2]=.98;hand[2:4]=1.02;hand[4:5]=1.;depth[0,0]=np.nan
        for tolerance in (0.,.005,.02):
            expected=reference(mask,hand,depth,tolerance_m=tolerance)
            actual=surface.front_only(mask,hand,depth,tolerance_m=tolerance)
            np.testing.assert_array_equal(actual[0],expected[0]);self.assertEqual(actual[1],expected[1])
            for key in expected[2]:np.testing.assert_array_equal(actual[2][key],expected[2][key])

    def test_native_depth_eligibility_including_fringe_and_absent_hands(self):
        ref=self.ref.functions('scripts/rgb_scene_metric_eligibility.py',['conservative_mask_on_depth','classify_native_depth'],
            {'CLASS_CODES':surface.CLASS_CODES})['classify_native_depth']
        raw=np.ones((12,16),bool);owned=raw.copy();owned[2:4,3:6]=False
        depth=np.ones((24,32));depth[1,1]=np.nan;hand=np.full(depth.shape,np.inf)
        hand[5:9,7:13]=.95;hand[10:12,15:20]=1.05;hand[13:15,15:20]=1.
        from envs.qualification.camera import resize_affine
        args={'owned_mask':owned,'raw_mask':raw,'hand_z':hand,'depth_z':depth,
            'A_depth_from_mask':resize_affine((16,12),(32,24)), 'tolerance_m':.005,'uncertainty_padding_mask_px':2}
        for complete,absent in [(True,False),(False,False),(False,True)]:
            args.update(complete_hand_geometry=complete,validated_hand_absence=absent)
            expected=ref(**args);actual=object_observation.classify_native_depth(**args)
            np.testing.assert_array_equal(expected[0],actual[0]);np.testing.assert_array_equal(expected[1],actual[1]);self.assertEqual(expected[2],actual[2])

    def test_robust_first_surface_filter_numerics_and_failure_reasons(self):
        ref=self.ref.functions('scripts/build_v19_visible_geometry_from_sam2_depth.py',
            ['numeric_summary','backprojected_extent','robust_first_surface_depth_ownership'])['robust_first_surface_depth_ownership']
        mask=np.zeros((64,80),bool);mask[4:60,5:75]=True
        yy,xx=np.indices(mask.shape);base=1.+xx*.0004+yy*.0002
        options=dict(enabled=True,mad_sigma=2.5,min_half_width_m=.03,min_retained_fraction=.9,
            fail_raw_to_robust_extent_ratio=2.,intrinsics=np.array([90.,90.,39.5,31.5]))
        for case in ('coherent','boundary_bleed','small_interior','large_interior','missing_confidence'):
            depth=base.copy();confidence=np.ones(mask.shape)
            if case=='boundary_bleed':depth[:,5:7]+=.3
            if case=='small_interior':depth[30,40]+=.3;confidence[30,40]=10.
            if case=='large_interior':depth[15:45,20:60]+=.5
            if case=='missing_confidence':confidence=None
            expected=ref(mask,depth,confidence=confidence,**options)
            actual=surface.robust_first_surface_depth_ownership(mask,depth,confidence=confidence,**options)
            np.testing.assert_array_equal(actual[0],expected[0])
            self.assertEqual(actual[1],expected[1],case)

    def test_native_front_depth_scale_kernel_not_a_new_scale_optimizer(self):
        ref=self.ref.functions('experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py',
            ['finite_vector','convex_projection_and_front_depth'])['convex_projection_and_front_depth']
        xy=np.linspace(-.25,.25,32);xx,yy=np.meshgrid(xy,xy)
        v=np.column_stack([xx.ravel(),yy.ravel(),np.ones(xx.size)])
        mask=np.zeros((80,100),bool);mask[12:68,20:80]=True;K=np.array([100.,100.,49.5,39.5])
        expected=ref(v,K,mask);actual=shape.convex_projection_and_front_depth(v,K,mask)
        self.assertEqual(actual,expected)
        scale,report=shape.scene_similarity(v,v*.7,K,mask)
        self.assertAlmostEqual(scale,.7/expected['owned_zbuffer_median_native_depth'])
        self.assertEqual(report['native_front_projection'],expected)

    def test_source_pnp_matches_v7_local_measurement_for_identical_matches(self):
        # V7 canonicalization uses a P14 pose; the measurement D,d itself must not
        # depend on it. New source-world evidence retains source samples directly.
        import torch
        from observations.test_rgb_source import source_fixture
        with tempfile.TemporaryDirectory() as tmp:
            fixture=source_fixture(Path(tmp));frames={r['frame_idx']:r for r in fixture['frames']}
            with np.load(fixture['inputs']['depth_npz'],allow_pickle=False) as z:
                depth=z['depth'];confidence=z['confidence'];K=z['intrinsics_fx_fy_cx_cy']
            images,native,masks,_=rgb_tracking.preflight_manifest_images(fixture['inputs']['raw_frame_manifest'],frames,[3,11],'object:synthetic',(80,60))
            common={'torch':torch,'resize_intrinsics':rgb_tracking.resize_intrinsics,'lift_source_world':rgb_tracking.lift_source_world,
                'rgb_source_eligibility':rgb_tracking.rgb_source_eligibility}
            ref=self.ref.functions('scripts/build_rgb_scene_edges.py',['feature_pair'],common)['feature_pair']
            args=SimpleNamespace(object_id='object:synthetic',min_confidence=.05,min_matches=8,
                pnp_iterations=1000,reprojection_threshold_px=2.5,min_inliers=30,min_inlier_fraction=.2,
                max_reprojection_median_px=1.5,max_edge_rotation_deg=180.,observation_mode='source-world',pnp_refinement='lm')
            count=len(fixture['world']);matched={'matches':torch.tensor(np.column_stack([np.arange(count)]*2)),
                'scores':torch.full((count,),.8)}
            module=ModuleType('lightglue.utils');module.rbd=lambda x:x
            poses={3:(Rotation.from_rotvec([.2,-.1,.1]).as_matrix(),np.array([.1,.03,.6]))}
            parameters=(3,11,fixture['features'],masks,frames,depth,confidence,{3:0,11:1},K,(80,60),
                lambda _:matched,torch.device('cpu'),args,poses,{3:np.ones(depth.shape[1:],bool)})
            with patch.dict(sys.modules,{'lightglue.utils':module}):
                cv2.setRNGSeed(42);old=ref(*parameters,target_evidence_size_wh=(80,60))
                cv2.setRNGSeed(42);new=rgb_tracking.feature_pair(*parameters,target_evidence_size_wh=(80,60),source_evidence_size_wh=(80,60))
            self.assertEqual(old[0]['status'],new[0]['status'])
            for a,b in zip(old[1],new[1]):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
            recovered=old[2]['canonical_points']@poses[3][0].T+poses[3][1]
            np.testing.assert_allclose(new[2]['source_world'],recovered,atol=1e-12,rtol=0)
            for key in ('target_uv','target_intrinsics','weights'):
                np.testing.assert_allclose(old[2][key],new[2][key],atol=1e-12,rtol=0)

    def test_motion_seed_and_independent_loop_on_same_numeric_inputs(self):
        reference=self.ref.functions('scripts/rgb_scene_rotation_chain.py',['local_rotation_evidence','observed_rotation_authority'])
        rng=np.random.default_rng(4);ids=[3,8,13,40];times=np.array([0.,.1,.2,.3])
        nodes=[PoseNode(fid,np.eye(3),np.zeros(3),rng.normal(size=(24,3))*.1,'synthetic',True,.1) for fid in ids]
        edges=[]
        diag={'inlier_fraction':.95,'reprojection_median_px':.05,'source_3d_conditioning':.03}
        for a,b in zip(ids[:-1],ids[1:]):edges.append(RGBEdge(a,b,Rotation.from_rotvec([0,.1,0]).as_matrix(),np.zeros(3),1.,diag))
        edges.append(RGBEdge(3,40,Rotation.from_rotvec([0,1.8,0]).as_matrix(),np.zeros(3),1.,diag))
        pixels=[RGBReprojectionFactor(e.source_frame_idx,e.target_frame_idx,ids.index(e.source_frame_idx),ids.index(e.target_frame_idx),
            np.ones((4,3)),np.zeros((4,2)),np.array([1.,1.,0,0]),np.eye(3),np.zeros(3),np.ones(4),1.) for e in edges]
        expected=reference['local_rotation_evidence'](nodes,times,edges,3,pixels)
        actual=rgb_tracking.local_rotation_evidence(nodes,times,edges,3,pixels)
        self.assertEqual(expected[3],actual[3]);self.assertEqual(actual[1][-1].weight,0.);self.assertEqual(actual[2][-1].edge_weight,0.)
        for a,b in zip(expected[0],actual[0]):np.testing.assert_array_equal(a.initial_rotation,b.initial_rotation)
        self.assertEqual(reference['observed_rotation_authority'](nodes,expected[3]),rgb_tracking.observed_rotation_authority(nodes,actual[3]))


@unittest.skipUnless(os.environ.get('EGO_V7_REFERENCE') and os.environ.get('EGO_V7_MILK_CANDIDATE'),
                     'requires explicit read-only V7 milk candidate')
class RealMilkReferenceTests(unittest.TestCase):
    def test_existing_milk_source_arrays_and_motion_qualification(self):
        root=Path(os.environ['EGO_V7_MILK_CANDIDATE']).resolve()
        files={
            'annotations':root/'measurements/object_geometry/visible_geometry/milk_carton/annotations_v19_visible_geometry.json',
            'edges':root/'measurements/pose_fits/milk_carton/scene_rgb/edges.npz',
            'edge_report':root/'measurements/pose_fits/milk_carton/scene_rgb/edges.json',
            'source_pose':root/'measurements/pose_fits/milk_carton/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json',
            'judgments':root/'state/agent_interaction_judgments/milk_carton.json'}
        before={name:sha(p) for name,p in files.items()}
        ann=json.loads(files['annotations'].read_text());rows=ann['frames'];ids=[r['frame_idx'] for r in rows]
        times=[r['time_s'] for r in rows];source=json.loads(files['source_pose'].read_text())
        poses={p['frame_idx']:p for p in source['pose_rows']};report=json.loads(files['edge_report'].read_text())
        accepted=[r for r in report['rows'] if r['status']=='accepted']
        with np.load(files['edges'],allow_pickle=False) as z:arrays=dict(z)
        edges=[]
        for i in np.flatnonzero(arrays['accepted']):
            a=int(arrays['source_frame_idx'][i]);b=int(arrays['target_frame_idx'][i]);p=poses[a]
            lo,hi=map(int,arrays['reprojection_evidence_offsets'][i:i+2]);R=np.asarray(p['rotation_world_from_completed_canonical_matrix']);t=np.asarray(p['translation_world_m'])
            world=arrays['reprojection_canonical_points'][lo:hi]@R.T+t
            diagnostic=next(r for r in accepted if (r['source_frame_idx'],r['target_frame_idx'])==(a,b))
            # Only test-consumed fields; missing independent source pixels are NOT
            # fabricated into a production measurement bundle.
            edges.append(SimpleNamespace(source_frame_idx=a,target_frame_idx=b,source_time_s=times[a],target_time_s=times[b],
                source_world=world,target_uv=arrays['reprojection_target_uv'][lo:hi],
                target_camera=arrays['reprojection_target_camera'][i],target_intrinsics=arrays['reprojection_target_intrinsics'][i],
                rotation_source_to_target_world=arrays['rotation_source_to_target_world'][i],
                translation_source_to_target_world_m=arrays['translation_source_to_target_world_m'][i],
                weights=arrays['reprojection_weights'][lo:hi],diagnostics=diagnostic))
        evidence=SimpleNamespace(object_id='milk_carton',edges=edges)
        prepared=prepare_motion(evidence,rows,'milk_carton',0)
        reference=FrozenReference(os.environ['EGO_V7_REFERENCE'])
        kernels=reference.functions('scripts/rgb_scene_rotation_chain.py',['local_rotation_evidence','observed_rotation_authority'])
        # Replace only the two pure helpers used inside prepare_motion; input,
        # local SE3 propagation and versioned-output assembly stay identical.
        with patch('data.motion.local_rotation_evidence',kernels['local_rotation_evidence']),\
             patch('data.motion.observed_rotation_authority',kernels['observed_rotation_authority']):
            frozen=prepare_motion(evidence,rows,'milk_carton',0)
        self.assertEqual(thaw(prepared.qualification),thaw(frozen.qualification))
        self.assertEqual(thaw(prepared.initialization),thaw(frozen.initialization))
        decisions=Counter(r['reason'] for r in prepared.qualification['edges'])
        self.assertEqual(len(edges),188);self.assertEqual(decisions['rejected_independent_local_cycle'],25)
        self.assertEqual(len(prepared.initialization['frames']),150)
        # Same declared P17 source to typed representation, without rejudging images.
        primary=reference.functions('scripts/v19_demo_timeline.py',['frame_ids','interaction_states','primary_contact_pads'],
            {'SIDES':('left','right'),'PRIMARY_DIGITS':('thumb','index','middle','ring','little')})
        judgments=json.loads(files['judgments'].read_text())
        typed=compile_interaction_timeline(load_interaction_evidence(judgments,ids,['milk_carton']),ids,'milk_carton')
        np.testing.assert_array_equal(typed.primary_visual_eligible,primary['primary_contact_pads'](judgments,ids,'milk_carton'))
        self.assertEqual(before,{name:sha(p) for name,p in files.items()})
        if os.environ.get('EGO_PARITY_REPORT'):
            destination=Path(os.environ['EGO_PARITY_REPORT'])
            with destination.open('x') as f:json.dump({'scope':'read_only_real_V7_intermediate_reference_not_new_v2_prediction',
                'input_hashes':{str(files[k]):v for k,v in before.items()},'frame_count':150,'RGB_accepted_measurements':188,
                'reprojection_points':sum(len(e.source_world) for e in edges),'loop_decisions':dict(decisions),
                'qualification_numeric_equal':True,'P17_primary_numeric_equal':True,'source_unchanged':True,
                'seed_comparison_scope':'same new SE3 propagation with old/new rotation qualification kernels; NOT equivalence to V7 P14/P15 trajectory',
                'V7_source_functions':reference.sources,'not_tested':['new LightGlue inference','new SAM3D inference','real RL raw-to-v2','policy integration']},f,indent=2)


if __name__=='__main__':unittest.main()
