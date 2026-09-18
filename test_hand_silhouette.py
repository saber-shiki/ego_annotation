"""CPU execution contracts for independent SAM2 hand silhouettes."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import torch

import main
from factors.hand.silhouette import (bidirectional_contour_residual,mask_contour,smooth_robust_squared,
    triangle_edge_points,visible_boundary_associations)
from observations.interaction.hand_masks import (half_pixel_affine,make_sam2_hand_prompt_payload,
    produce_sam2_hand_masks,transform_points,validate_hand_mask_receipt)
from observations.hand.hawor import FULL_K_RECTIFIED_MODE,aggregate_file_sha256
from optimization.problems import PreparedVisualFit
from optimization.solve import solve
from optimization.object.surface_fit import fit_surface
from test_surface_integration import consumer_fixture,hands_and_judgment
from utils.state import sha
from utils.geometry import camera,project,resized_K


class FakeOfficialSAM2:
    """Official predictor protocol double; masks, not producer requests, are emitted."""
    def __init__(self):self.objects=[];self.count=0;self.calls=[]
    def init_state(self,**kwargs):
        paths=list(Path(kwargs['video_path']).glob('*'));self.count=len(paths)
        image=cv2.imread(str(paths[0]));self.height,self.width=image.shape[:2]
        return {}
    def add_new_points_or_box(self,inference_state,frame_idx,obj_id,box,**kwargs):
        self.calls.append((frame_idx,obj_id,np.asarray(box).copy()))
        if obj_id not in self.objects:self.objects.append(obj_id)
        logits=[]
        for value in self.objects:
            m=torch.full((1,self.height,self.width),-1.)
            y0,y1=max(1,int(self.height*.3)),min(self.height-1,int(self.height*.75))
            # Keep two independent side masks at proportional full-frame locations.
            x0,x1=((int(self.width*.60),int(self.width*.84)) if value==1 else
                   (int(self.width*.12),int(self.width*.38)))
            m[:,y0:y1,x0:x1]=1.;logits.append(m)
        return frame_idx,list(self.objects),torch.stack(logits)


class OverlappingOfficialSAM2(FakeOfficialSAM2):
    def add_new_points_or_box(self,inference_state,frame_idx,obj_id,box,**kwargs):
        self.calls.append((frame_idx,obj_id,np.asarray(box).copy()))
        if obj_id not in self.objects:self.objects.append(obj_id)
        logits=[]
        for _value in self.objects:
            m=torch.full((1,self.height,self.width),-1.)
            m[:,int(self.height*.25):int(self.height*.8),int(self.width*.25):int(self.width*.75)]=1.
            logits.append(m)
        return frame_idx,list(self.objects),torch.stack(logits)


def raw_manifest(root,ids,size=(64,64),frame_paths=None):
    video=root/'source.mp4'
    if not video.exists():video.write_bytes(b'synthetic authoritative source video')
    rows=[]
    for n,i in enumerate(ids):
        path=(Path(frame_paths[n]) if frame_paths is not None else root/f'rgb_{i}.png')
        if frame_paths is None:cv2.imwrite(str(path),np.full((size[1],size[0],3),40+n,np.uint8))
        rows.append({'frame_idx':int(i),'raw_frame_path':str(path.resolve()),'raw_frame_sha256':sha(path),
            'source_width':size[0],'source_height':size[1],'manifest_width':size[0],'manifest_height':size[1]})
    path=root/'raw_manifest.json';path.write_text(json.dumps({'input_video':str(video.resolve()),
        'input_video_sha256':sha(video),'frames':rows}))
    return path


def detector_npz(root,ids,size=(64,64),intrinsics=None):
    path=root/'detectors.npz';n=len(ids);width,height=size
    source_K=np.asarray(intrinsics if intrinsics is not None else [100.,100.,(width-1)/2.,(height-1)/2.],np.float64)
    inference_K=np.asarray([source_K[0],source_K[1],width/2.,height/2.],np.float64)
    A=np.asarray([[1.,0.,inference_K[2]-source_K[2]],[0.,1.,inference_K[3]-source_K[3]],[0.,0.,1.]])
    rectified=root/'hawor_rectified';rectified.mkdir(exist_ok=True)
    for local in range(n):cv2.imwrite(str(rectified/f'{local:06d}.jpg'),np.full((height,width,3),50+local,np.uint8))
    jpgs=sorted(rectified.glob('*.jpg'));video=root/'source.mp4'
    contract={'status':'exact_source_to_centered_hawor_image_plane','mode':FULL_K_RECTIFIED_MODE,
        'source_size_wh':[width,height],'inference_frame_count':n,'source_video_sha256':sha(video),
        'source_intrinsics_fx_fy_cx_cy':source_K.tolist(),'hawor_inference_intrinsics_fx_fy_cx_cy':inference_K.tolist(),
        'A_hawor_inference_from_source':A.tolist(),'A_source_from_hawor_inference':np.linalg.inv(A).tolist(),
        'inference_extracted_frames':str(rectified),'inference_frames_aggregate_sha256':aggregate_file_sha256(jpgs),
        'inference_first_frame_sha256':sha(jpgs[0]),'inference_last_frame_sha256':sha(jpgs[-1])}
    contract_path=root/'hawor_camera_plane.json';contract_path.write_text(json.dumps(contract))
    np.savez(path,frame_idx=np.asarray(ids),left_det_box_xyxyscore=np.tile([20,10,42,44,.9],(n,1)),
        right_det_box_xyxyscore=np.tile([8,10,30,44,.8],(n,1)),left_track_id=np.asarray(['1.0']*n),
        right_track_id=np.asarray(['2.0']*n),left_detected_same_frame=np.ones(n,bool),right_detected_same_frame=np.ones(n,bool),
        camera_intrinsics_contract_mode=np.asarray([FULL_K_RECTIFIED_MODE]),
        camera_intrinsics_fx_fy_cx_cy=np.repeat(source_K[None],n,axis=0),
        hawor_inference_intrinsics_fx_fy_cx_cy=np.repeat(inference_K[None],n,axis=0),
        A_hawor_inference_from_source=A,A_source_from_hawor_inference=np.linalg.inv(A),
        camera_image_plane_contract_path=np.asarray([str(contract_path)]),camera_image_plane_contract_sha256=np.asarray([sha(contract_path)]),
        video_path=np.asarray([str(video.resolve())]),video_sha256=np.asarray([sha(video)]))
    return path


def produce(root,ids,size=(64,64),predictor_type=FakeOfficialSAM2):
    manifest=raw_manifest(root,ids,size);hawor=detector_npz(root,ids,size)
    checkpoint=root/'sam2.pt';checkpoint.write_bytes(b'fake checkpoint identity')
    source=root/'sam2_source';cfg=source/'configs/model.yaml';cfg.parent.mkdir(parents=True);cfg.write_text('model: fixture\n')
    predictor=predictor_type()
    receipt=produce_sam2_hand_masks(raw_frame_manifest=manifest,hawor_npz=hawor,output_dir=root/'masks',
        checkpoint=checkpoint,model_cfg='configs/model.yaml',sam2_source=source,device='cpu',
        predictor_factory=lambda *a,**k:predictor)
    return manifest,hawor,receipt,predictor


class HandMaskProducerTests(unittest.TestCase):
    def test_actual_grouped_producer_writes_full_side_timeline_and_string_tracks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,hawor,receipt,predictor=produce(root,[7,19,103],(80,48))
            doc=validate_hand_mask_receipt(receipt,frame_ids=[7,19,103],source_size_wh=(80,48),
                source_frame_manifest=manifest)
            self.assertEqual(len(doc['frames']),6);self.assertEqual(len(predictor.calls),6)
            self.assertEqual({r['detector_track_id'] for r in doc['frames']},{'1.0','2.0'})
            self.assertTrue(all(Path(r['mask_path']).is_file() for r in doc['frames']))
            self.assertEqual(doc['model_execution']['propagation_used_as_direct_observation'],False)

    def test_receipt_rejects_side_frame_source_hash_and_halfpixel_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,_,receipt,_=produce(root,[4,9])
            original=json.loads(receipt.read_text())
            mutations=[lambda d:d['frames'][0].update(hand_side='right'),
                lambda d:d['frames'][0].update(frame_idx=9),
                lambda d:d['frames'][0].update(raw_frame_sha256='b'*64),
                lambda d:d['coordinate_transform']['A_mask_from_source'].__setitem__(0,[1,0,3])]
            for n,mutate in enumerate(mutations):
                doc=copy.deepcopy(original);mutate(doc);bad=root/f'bad{n}.json';bad.write_text(json.dumps(doc))
                with self.subTest(n=n),self.assertRaises(ValueError):
                    validate_hand_mask_receipt(bad,frame_ids=[4,9],source_size_wh=(64,64),source_frame_manifest=manifest)

    def test_missing_detection_is_inactive_and_never_gets_propagated_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=raw_manifest(root,[0]);hawor=detector_npz(root,[0])
            with np.load(hawor,allow_pickle=False) as z:data=dict(z)
            data['left_detected_same_frame'][0]=False;np.savez(hawor,**data)
            checkpoint=root/'sam2.pt';checkpoint.write_bytes(b'x');source=root/'source';source.mkdir();(source/'cfg.yaml').write_text('x')
            receipt=produce_sam2_hand_masks(raw_frame_manifest=manifest,hawor_npz=hawor,output_dir=root/'out',
                checkpoint=checkpoint,model_cfg='cfg.yaml',sam2_source=source,device='cpu',
                predictor_factory=lambda *a,**k:FakeOfficialSAM2())
            rows={(r['frame_idx'],r['hand_side']):r for r in json.loads(receipt.read_text())['frames']}
            self.assertEqual(rows[0,'left']['status'],'unresolved');self.assertIsNone(rows[0,'left']['mask_path'])
            self.assertEqual(rows[0,'right']['status'],'visible')

    def test_missing_or_foreign_detector_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=raw_manifest(root,[0]);hawor=detector_npz(root,[0])
            with np.load(hawor,allow_pickle=False) as z:data=dict(z)
            data.pop('left_detected_same_frame');np.savez(hawor,**data)
            with self.assertRaisesRegex(ValueError,'detected_same_frame'):
                make_sam2_hand_prompt_payload(raw_frame_manifest=manifest,hawor_npz=hawor)
            hawor=detector_npz(root,[0])
            with np.load(hawor,allow_pickle=False) as z:data=dict(z)
            data['video_sha256']=np.asarray(['b'*64]);np.savez(hawor,**data)
            with self.assertRaisesRegex(ValueError,'source/camera contract'):
                make_sam2_hand_prompt_payload(raw_frame_manifest=manifest,hawor_npz=hawor)

    def test_predictor_input_hash_and_unconstrained_overlap_are_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,_,receipt,_=produce(root,[0],predictor_type=OverlappingOfficialSAM2)
            doc=json.loads(receipt.read_text())
            self.assertTrue(all(r['status']=='unresolved' and r['inactive_reason']=='ambiguous_side_overlap' for r in doc['frames']))
            self.assertFalse(doc['model_execution']['non_overlap_masks'])
            predictor_frame=Path(doc['predictor_inputs'][0]['path']);predictor_frame.write_bytes(b'tampered derived jpeg')
            with self.assertRaisesRegex(ValueError,'input frame hash mismatch'):
                validate_hand_mask_receipt(receipt,frame_ids=[0],source_size_wh=(64,64),source_frame_manifest=manifest)


class VisibleBoundaryTests(unittest.TestCase):
    def test_full_topology_tail_gradient_internal_edge_and_hidden_boundary(self):
        uv=torch.tensor([[5.,5.],[25.,5.],[25.,25.],[5.,25.]])
        faces=np.asarray([[0,1,2],[0,2,3]])
        points=triangle_edge_points(uv,faces,samples_per_edge=3)
        self.assertTrue(any(torch.allclose(p,torch.tensor([15.,15.])) for p in points)) # closing edge exists
        xyz=np.c_[uv.numpy()/20.,np.ones(4)]
        entities=[{'label':1,'vertices_camera':xyz,'vertices_uv':uv.numpy(),'faces':faces}]
        face_ids,bary,pixels=visible_boundary_associations(entities,1,(32,32),max_samples=1000)
        hits=(xyz[face_ids]*bary[:,:,None]).sum(1)[:,:2]*20
        self.assertTrue(np.all((np.min(hits,axis=1)<6)|(np.max(hits,axis=1)>24)))
        # A complete front object suppresses every hidden hand boundary association.
        front=xyz.copy();front[:,2]=.5
        hidden=[{'label':1,'vertices_camera':xyz,'vertices_uv':uv.numpy(),'faces':faces},
            {'label':0,'vertices_camera':front,'vertices_uv':uv.numpy(),'faces':faces}]
        self.assertEqual(len(visible_boundary_associations(hidden,1,(32,32))[0]),0)
        x=torch.tensor([15.],requires_grad=True);smooth_robust_squared(x).sum().backward()
        self.assertGreater(abs(float(x.grad)),0.)

    def test_inactive_missing_or_occluded_contour_has_zero_force(self):
        candidate=torch.tensor([[10.,10.]],requires_grad=True)
        loss=bidirectional_contour_residual(candidate,torch.zeros((0,2)))
        loss.backward()
        self.assertEqual(float(loss.detach()),0.);np.testing.assert_array_equal(candidate.grad.numpy(),np.zeros((1,2)))

    def test_candidate_cap_preserves_outer_hole_and_disconnected_island(self):
        outer=np.asarray([[5,5],[75,5],[75,75],[5,75]],np.float64)
        inner=np.asarray([[28,28],[52,28],[52,52],[28,52]],np.float64)
        island=np.asarray([[82,8],[88,8],[88,14],[82,14]],np.float64)
        uv=np.vstack((outer,inner,island));xyz=np.c_[uv/100.,np.ones(len(uv))]
        faces=np.asarray([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7],
                          [8,9,10],[8,10,11]])
        _,_,pixels=visible_boundary_associations([{'label':1,'vertices_camera':xyz,
            'vertices_uv':uv,'faces':faces}],1,(96,84),max_samples=320)
        self.assertLessEqual(len(pixels),320)
        self.assertTrue(np.any((pixels[:,0]<7)&(pixels[:,1]>20)))
        self.assertTrue(np.any((pixels[:,0]>27)&(pixels[:,0]<31)&(pixels[:,1]>30)))
        self.assertTrue(np.any(pixels[:,0]>80))

    def test_all_mask_islands_holes_and_nonsquare_affine(self):
        mask=np.zeros((30,60),np.uint8);mask[2:12,2:12]=1;mask[15:28,35:58]=1;mask[18:24,41:50]=0
        contour=mask_contour(mask,300)
        self.assertTrue(np.any(contour[:,0]<15));self.assertTrue(np.any(contour[:,0]>50))
        self.assertTrue(np.any((contour[:,0]>40)&(contour[:,0]<51)&(contour[:,1]>17)&(contour[:,1]<25)))
        A=half_pixel_affine((120,50),(60,30));p=np.asarray([[0.,0.],[119.,49.]])
        np.testing.assert_allclose(transform_points(transform_points(p,A),np.linalg.inv(A)),p,atol=1e-5)


class PreparedRawFitTests(unittest.TestCase):
    def test_actual_prepared_fit_consumes_raw_masks_preserves_object_and_pickle_false(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=consumer_fixture(root);candidate=fit_surface(p);candidate_path=p.output_dir/'pose_mesh.json'
            models,_=hands_and_judgment(root,p);ids=np.asarray([7,19,103])
            manifest=raw_manifest(root,ids,frame_paths=[root/'rgb.png']*3)
            detector=detector_npz(root,ids,intrinsics=[100.,100.,31.5,31.5])
            # Add real-shape detector/source-plane provenance to the exact HaWoR
            # archive consumed as the fit's metric MANO initializer.
            with np.load(root/'hands.npz',allow_pickle=False) as z:hands=dict(z)
            with np.load(detector,allow_pickle=False) as z:
                for key in z.files:
                    if key!='frame_idx':hands[key]=z[key]
            np.savez(root/'hands.npz',**hands)
            checkpoint=root/'sam2.pt';checkpoint.write_bytes(b'fake')
            source=root/'sam2';source.mkdir();(source/'cfg.yaml').write_text('fixture')
            receipt=produce_sam2_hand_masks(raw_frame_manifest=manifest,hawor_npz=root/'hands.npz',output_dir=root/'hand_masks',
                checkpoint=checkpoint,model_cfg='cfg.yaml',sam2_source=source,device='cpu',predictor_factory=lambda *a,**k:FakeOfficialSAM2())
            judgments=json.loads((root/'judgments.json').read_text())
            judgments['frames'][0]['visibility']['left'].update(state='occluded',occluder='object:thing')
            (root/'judgments.json').write_text(json.dumps(judgments))
            problem=PreparedVisualFit(p.annotations,candidate_path,root/'bridge.npz',root/'hands.npz',root/'fake_models',
                root/'raw_fit','thing',p.appearance,root/'judgments.json',iterations=30,object_profile='surface_no_box',
                hand_profile='raw_hand_silhouette',raw_hand_mask_manifest=receipt,raw_frame_manifest=manifest)
            foreign_hands=root/'foreign_hands.npz';foreign_hands.write_bytes((root/'hands.npz').read_bytes())
            with self.assertRaisesRegex(ValueError,'differs from visual-fit HaWoR initializer'):
                main.fit_prepared(replace(problem,hawor_params=foreign_hands,output_dir=root/'foreign_fit'),root)
            inactive=copy.deepcopy(judgments)
            for frame in inactive['frames']:
                for side in ('left','right'):
                    frame['visibility'][side].update(state='occluded',occluder='object:thing')
            inactive_path=root/'inactive_judgments.json';inactive_path.write_text(json.dumps(inactive))
            missing_problem=replace(problem,output_dir=root/'missing_fit',interaction_judgments=inactive_path)
            with open('/dev/null','w') as sink,patch('optimization.interaction.joint_fit.mano_models',return_value=models),contextlib.redirect_stdout(sink):
                report=main.fit_prepared(problem,root)
                missing_report=main.fit_prepared(missing_problem,root)
            with (np.load(root/'raw_fit/demo_state.npz',allow_pickle=False) as state,
                  np.load(root/'missing_fit/demo_state.npz',allow_pickle=False) as missing_state):
                self.assertEqual(state['raw_hand_mask_active'].shape,(3,2))
                self.assertFalse(bool(state['raw_hand_mask_active'][0,0]))
                self.assertTrue(bool(state['raw_hand_mask_active'][0,1]))
                np.testing.assert_array_equal(state['object_rotation'],state['base_object_rotation'])
                np.testing.assert_array_equal(state['object_translation_world'],state['base_object_translation_world'])
                np.testing.assert_array_equal(state['object_delta'],np.zeros((3,6)))
                self.assertTrue(all(state[k].dtype!=object for k in state.files))
                self.assertEqual(state['left_vertices_world'].shape,(3,778,3))
                self.assertGreater(len(state['left_faces']),0)
                rendered_state=dict(state)
                # The independent target is closer to the final visible MANO
                # boundary than to the initializer on the optimizer's fit plane.
                fit_K=resized_K([100.,100.,31.5,31.5],(64,64),(960,960))
                distances=[];missing_distances=[]
                for side in ('left','right'):
                    row=next(r for r in json.loads(receipt.read_text())['frames']
                        if r['frame_idx']==19 and r['hand_side']==side)
                    target=transform_points(mask_contour(cv2.imread(row['mask_path'],0)),half_pixel_affine((64,64),(960,960)))
                    faces=state[side+'_faces']
                    def boundary_distance(vertices):
                        vc=camera(vertices,state['T_world_camera'][1]);uv=project(vc,fit_K)
                        _,_,pixels=visible_boundary_associations([{'label':1,'vertices_camera':vc,
                            'vertices_uv':uv,'faces':faces}],1,(960,960))
                        pair=np.linalg.norm(pixels[:,None]-target[None],axis=2)
                        return float((pair.min(axis=1).mean()+pair.min(axis=0).mean())*.5)
                    distances.append((boundary_distance(state[side+'_base_vertices_world'][1]),
                                      boundary_distance(state[side+'_vertices_world'][1])))
                    missing_vertices=missing_state[side+'_vertices_world'][1]
                    missing_distances.append(boundary_distance(missing_vertices))
                self.assertLess(np.mean([v[1] for v in distances]),np.mean([v[0] for v in distances]))
                self.assertLess(np.mean([v[1] for v in distances]),np.mean(missing_distances))
                self.assertTrue(np.all(~missing_state['raw_hand_mask_active']))
                # Report deltas are on the 64px render plane; the factor is on
                # the fixed 960px fitting plane used by this optimizer.
                fit_plane_delta=max(m['joint_projection_delta_median_px'] for m in report['measurements'])*960/64
                self.assertGreater(fit_plane_delta,5.)
            import render
            self.assertTrue(render.mesh_overlay(rendered_state,0,np.zeros((64,64,3),np.uint8)).any())
            self.assertIn('appearance_mask_manifest',report['inputs'])
            self.assertGreater(max(row['raw_hand_silhouette'] for row in report['trace']),0.)
            self.assertTrue(all(row['raw_hand_silhouette']==0. for row in missing_report['trace']))
            self.assertGreater(report['trace'][-1]['image'],0.)

    def test_direct_legacy_solve_accepts_interval_only_judgments_without_external_mano(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=consumer_fixture(root);fit_surface(p);models,judgment=hands_and_judgment(root,p)
            interval_only={'object_id':p.object_id,'interaction_judgments':judgment['interaction_judgments']}
            interval_path=root/'interval_only_judgments.json';interval_path.write_text(json.dumps(interval_only))
            problem=PreparedVisualFit(p.annotations,p.output_dir/'pose_mesh.json',root/'bridge.npz',root/'hands.npz',
                root/'fake_models',root/'legacy_direct','thing',p.appearance,interval_path,iterations=1,
                object_profile='surface_no_box',hand_profile='legacy',run_root=root)
            with open('/dev/null','w') as sink,patch('optimization.interaction.joint_fit.mano_models',return_value=models),contextlib.redirect_stdout(sink):
                report=solve(problem)
            self.assertEqual(report['hand_profile'],'legacy')
            with np.load(problem.output_dir/'demo_state.npz',allow_pickle=False) as state:
                self.assertEqual(state['hand_profile'].item(),'legacy')
                self.assertEqual(state['left_vertices_world'].shape,(3,778,3))
                self.assertTrue(all(state[key].dtype!=object for key in state.files))
                self.assertNotIn('raw_hand_mask_manifest',state.files)

    def test_legacy_default_and_explicit_profile_are_bitwise_identical_and_pickle_free(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=consumer_fixture(root);fit_surface(p);candidate=p.output_dir/'pose_mesh.json'
            models,_=hands_and_judgment(root,p)
            common=dict(annotations=p.annotations,pose_report=candidate,mano_bridge=root/'bridge.npz',
                hawor_params=root/'hands.npz',mano_models=root/'fake_models',object_id='thing',
                appearance_mask_manifest=p.appearance,interaction_judgments=root/'judgments.json',
                iterations=2,object_profile='surface_no_box')
            a=PreparedVisualFit(output_dir=root/'legacy_default',**common)
            b=PreparedVisualFit(output_dir=root/'legacy_explicit',hand_profile='legacy',**common)
            with open('/dev/null','w') as sink,patch('optimization.interaction.joint_fit.mano_models',return_value=models),contextlib.redirect_stdout(sink):
                main.fit_prepared(a,root);main.fit_prepared(b,root)
            with np.load(a.output_dir/'demo_state.npz',allow_pickle=False) as za,np.load(b.output_dir/'demo_state.npz',allow_pickle=False) as zb:
                self.assertEqual(za.files,zb.files)
                for key in za.files:
                    if np.issubdtype(za[key].dtype,np.floating):
                        np.testing.assert_array_equal(np.isnan(za[key]),np.isnan(zb[key]))
                        np.testing.assert_array_equal(np.nan_to_num(za[key]),np.nan_to_num(zb[key]))
                    else:np.testing.assert_array_equal(za[key],zb[key])
                    self.assertNotEqual(za[key].dtype,object)
                self.assertNotIn('raw_hand_mask_manifest',za.files)


if __name__=='__main__':unittest.main()
