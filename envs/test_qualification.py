"""Authority and interval observation predicates at their actual boundaries."""
from pathlib import Path
import tempfile
import unittest
from envs.qualification.hand_observation import translation_support_gate_applies
from envs.qualification.object_geometry import load_signed_face_authority
from envs.qualification.object_trajectory import pose_graph_readiness

class FormalAuthorityTests(unittest.TestCase):
    def test_zero_support_is_not_translation_authority(self):
        self.assertTrue(translation_support_gate_applies(enabled=True,support_count=0,minimum_supported_vertices=0))
        self.assertFalse(translation_support_gate_applies(enabled=True,support_count=1,minimum_supported_vertices=0))
        self.assertFalse(translation_support_gate_applies(enabled=False,support_count=0,minimum_supported_vertices=0))

    def test_pose_rejection_quarantines_physical_hand_solve(self):
        value=pose_graph_readiness({'annotation_ready':False,'graph_support':{'sufficient':True}})
        self.assertTrue(value['explicitly_unready']);self.assertFalse(value['explicitly_ready'])

    def test_missing_required_local_face_authority_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'mesh.ply';p.write_bytes(b'not loaded for absent authority')
            result=load_signed_face_authority({'geometry_readiness':{'signed_face_authority_required':True,'signed_geometry_ready':True}},
                source_report_path=None,mesh_path=p,mesh_face_count=2)
            self.assertFalse(result['usable']);self.assertFalse(result['mask'].any())

if __name__=='__main__':unittest.main()
