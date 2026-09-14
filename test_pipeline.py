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
            root=Path(directory)
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
            metadata={'initial_pose_report':str(source),'annotations':str(annotations),
                      'depth_npz':str(depth),'input_sha256':{'initial_pose_report':sha(source),
                      'annotations':sha(annotations),'depth_npz':sha(depth)}}
            np.savez(edge,metadata=np.array([json.dumps(metadata)]),
                     source_frame_idx=np.array([],int), target_frame_idx=np.array([],int),
                     reprojection_evidence_offsets=np.array([0]),
                     reprojection_canonical_points=np.empty((0,3)))
            report.write_text(json.dumps({'rows':[]})); before=sha(source)
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
