"""CPU execution contracts for independent SAM2 hand silhouettes."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import main
from factors.hand.silhouette import (bidirectional_contour_residual,mask_contour,smooth_robust_squared,
    triangle_edge_points,visible_boundary_associations)
from observations.interaction.hand_masks import (half_pixel_affine,produce_sam2_hand_masks,
    transform_points,validate_hand_mask_receipt)
from optimization.problems import PreparedVisualFit
from optimization.object.surface_fit import fit_surface
from test_surface_integration import consumer_fixture,hands_and_judgment
from utils.state import sha


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


def raw_manifest(root,ids,size=(64,64)):
    rows=[]
    for n,i in enumerate(ids):
        path=root/f'rgb_{i}.png';cv2.imwrite(str(path),np.full((size[1],size[0],3),40+n,np.uint8))
        rows.append({'frame_idx':int(i),'raw_frame_path':str(path.resolve()),'raw_frame_sha256':sha(path),
            'source_width':size[0],'source_height':size[1],'manifest_width':size[0],'manifest_height':size[1]})
    path=root/'raw_manifest.json';path.write_text(json.dumps({'input_video_sha256':'a'*64,'frames':rows}))
    return path


def detector_npz(root,ids):
    path=root/'detectors.npz';n=len(ids)
    np.savez(path,frame_idx=np.asarray(ids),left_det_box_xyxyscore=np.tile([20,10,42,44,.9],(n,1)),
        right_det_box_xyxyscore=np.tile([8,10,30,44,.8],(n,1)),left_track_id=np.asarray(['1.0']*n),
        right_track_id=np.asarray(['2.0']*n),left_detected_same_frame=np.ones(n,bool),right_detected_same_frame=np.ones(n,bool),
        camera_intrinsics_contract_mode=np.asarray(['fixture_source_rgb']),A_source_from_hawor_inference=np.eye(3))
    return path


def produce(root,ids,size=(64,64)):
    manifest=raw_manifest(root,ids,size);hawor=detector_npz(root,ids)
    checkpoint=root/'sam2.pt';checkpoint.write_bytes(b'fake checkpoint identity')
    source=root/'sam2_source';cfg=source/'configs/model.yaml';cfg.parent.mkdir(parents=True);cfg.write_text('model: fixture\n')
    predictor=FakeOfficialSAM2()
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
            # Add real-shape string detector rows to the same HaWoR archive consumed by fit.
            with np.load(root/'hands.npz',allow_pickle=False) as z:hands=dict(z)
            hands.update(left_det_box_xyxyscore=np.tile([20,15,42,50,.9],(3,1)),right_det_box_xyxyscore=np.tile([8,15,30,50,.8],(3,1)),
                left_track_id=np.asarray(['1.0']*3),right_track_id=np.asarray(['2.0']*3),left_detected_same_frame=np.ones(3,bool),right_detected_same_frame=np.ones(3,bool),
                camera_intrinsics_contract_mode=np.asarray(['fixture_source_rgb']),A_source_from_hawor_inference=np.eye(3))
            np.savez(root/'hands.npz',**hands)
            manifest=raw_manifest(root,ids);checkpoint=root/'sam2.pt';checkpoint.write_bytes(b'fake')
            source=root/'sam2';source.mkdir();(source/'cfg.yaml').write_text('fixture')
            receipt=produce_sam2_hand_masks(raw_frame_manifest=manifest,hawor_npz=root/'hands.npz',output_dir=root/'hand_masks',
                checkpoint=checkpoint,model_cfg='cfg.yaml',sam2_source=source,device='cpu',predictor_factory=lambda *a,**k:FakeOfficialSAM2())
            problem=PreparedVisualFit(p.annotations,candidate_path,root/'bridge.npz',root/'hands.npz',root/'fake_models',
                root/'raw_fit','thing',p.appearance,root/'judgments.json',iterations=30,object_profile='surface_no_box',
                hand_profile='raw_hand_silhouette',raw_hand_mask_manifest=receipt,raw_frame_manifest=manifest)
            with open('/dev/null','w') as sink,patch('optimization.interaction.joint_fit.mano_models',return_value=models),contextlib.redirect_stdout(sink):
                report=main.fit_prepared(problem,root)
            with np.load(root/'raw_fit/demo_state.npz',allow_pickle=False) as state:
                self.assertEqual(state['raw_hand_mask_active'].shape,(3,2))
                np.testing.assert_array_equal(state['object_rotation'],state['base_object_rotation'])
                np.testing.assert_array_equal(state['object_translation_world'],state['base_object_translation_world'])
                np.testing.assert_array_equal(state['object_delta'],np.zeros((3,6)))
                self.assertTrue(all(state[k].dtype!=object for k in state.files))
                self.assertEqual(state['left_vertices_world'].shape,(3,778,3))
                self.assertGreater(len(state['left_faces']),0)
                rendered_state=dict(state)
                # Report deltas are on the 64px render plane; the factor is on
                # the fixed 960px fitting plane used by this optimizer.
                fit_plane_delta=max(m['joint_projection_delta_median_px'] for m in report['measurements'])*960/64
                self.assertGreater(fit_plane_delta,5.)
            import render
            self.assertTrue(render.mesh_overlay(rendered_state,0,np.zeros((64,64,3),np.uint8)).any())
            self.assertIn('appearance_mask_manifest',report['inputs'])
            self.assertGreater(report['trace'][-1]['image'],0.)

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
