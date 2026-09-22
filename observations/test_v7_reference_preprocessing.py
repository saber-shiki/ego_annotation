"""Opt-in read-only differential checks against frozen V7 pure functions.

Extract only named function definitions from git show; never import/execute a V7
producer, module-level entry, optimizer, provider or model loader.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from observations.test_interaction_adapter import adapted_fixture
from data.interaction import load_interaction_evidence
from data.interaction_features import compile_interaction_timeline
from data.geometry import bind_visual_to_candidate, load_visual_geometry
from observations.test_visual_geometry import visual_fixture
from observations.object.visual_geometry import build_visual_geometry
from observations.common import pose_from_values
from utils.geometry import validate_transforms, resized_K
from utils.data import RKEY,TKEY


@unittest.skipUnless(os.environ.get('EGO_V7_REFERENCE'),'set EGO_V7_REFERENCE to frozen read-only checkout')
class V7ReferencePreprocessingTests(unittest.TestCase):
    def reference(self,filename,names,globals_extra=None):
        root=os.environ['EGO_V7_REFERENCE']
        source=subprocess.check_output(['git','-C',root,'show','67c40e6:scripts/'+filename],text=True)
        selected=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in selected},set(names))
        namespace={'np':np,'SIDES':('left','right'),'PRIMARY_DIGITS':('thumb','index','middle','ring','little'),
            'validate_transforms':validate_transforms,'pose_from_values':pose_from_values,'RKEY':RKEY,'TKEY':TKEY,
            'digest_json':lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()}
        namespace.update(globals_extra or {})
        exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),namespace)
        return namespace

    def test_pixel_center_transform_matches_frozen_v7(self):
        ref=self.reference('v20_demo_geometry.py',['resized_K'])['resized_K']
        for source,target in [((1408,1408),(960,960)),((1280,720),(960,540)),((80,60),(160,120))]:
            k=np.array([974.3,974.3,702.66,706.11])
            np.testing.assert_array_equal(resized_K(k,source,target),ref(k,source,target))

    def test_primary_visual_declarations_match_frozen_v7(self):
        ref=self.reference('v19_demo_timeline.py',['frame_ids','interaction_states','primary_contact_pads'])
        payload=adapted_fixture();ids=[2,7,11]
        expected=ref['primary_contact_pads'](payload,ids,'synthetic')
        actual=compile_interaction_timeline(load_interaction_evidence(payload,ids,['synthetic']),ids,'synthetic')
        np.testing.assert_array_equal(actual.primary_visual_eligible,expected)

    def test_full_mesh_canonical_transform_matches_frozen_v7(self):
        ref=self.reference('rgb_scene_shape_kernels.py',['_pose','canonical_gauge'])['canonical_gauge']
        from scipy.spatial.transform import Rotation
        with tempfile.TemporaryDirectory() as temp:
            args,frame,cv,faces=visual_fixture(Path(temp));path=build_visual_geometry(**args)
            obs=load_visual_geometry(path,frame_ids=[7],frames=[frame],object_id='synthetic',expected_inputs={})
            R=Rotation.from_rotvec([.1,.2,-.3]).as_matrix();t=np.array([.02,.03,.4])
            expected,_=ref(cv,obs.report['scene_similarity']['scene_similarity_scale'],obs.T_world_camera,{RKEY:R.tolist(),TKEY:t.tolist()})
            result=bind_visual_to_candidate(obs,R,t,candidate_revision='reference-test-only')
            np.testing.assert_allclose(result['vertices_canonical'],expected,atol=1e-6,rtol=0)
            np.testing.assert_array_equal(result['faces'],faces)


if __name__=='__main__':unittest.main()
