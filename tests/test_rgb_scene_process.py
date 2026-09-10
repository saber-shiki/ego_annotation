"""Short CPU subprocess fixtures, readiness pipes, no polling or sleeps."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rgb_scene_process import run_owned, OwnedProcessFailure
from rgb_scene_runtime import run_command

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.env={**os.environ,'PYTHONPATH':str(SCRIPTS),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}

    def run_owned(self,code,timeout=1.):
        return run_owned('fixture',[sys.executable,'-c',code],cwd=self.root,env=self.env,log=self.root/'fixture.log',timeout=timeout,cleanup_timeout=.5)

    def test_success_nonzero_and_spawn_error_records(self):
        record=self.run_owned('print("success")')
        self.assertEqual(record['returncode'],0);self.assertEqual(record['cleanup']['status'],'stopped')
        (self.root/'fixture.log').unlink()
        with self.assertRaises(OwnedProcessFailure) as caught:self.run_owned('raise SystemExit(7)')
        self.assertEqual(caught.exception.record['returncode'],7)
        self.assertGreater(caught.exception.record['elapsed_s'],0)
        with self.assertRaises(OwnedProcessFailure) as caught:
            run_owned('missing',['/not/an/executable'],cwd=self.root,env=self.env,log=self.root/'missing.log',timeout=.5)
        self.assertIn('spawn/IO error',caught.exception.record['error'])
        self.assertEqual(caught.exception.record['cleanup']['status'],'not_spawned')

    def child_code(self):
        # Grandchild detaches its session; the owned tree/subreaper still finds it.
        return '''import os,signal,json,pathlib
r,w=os.pipe()
pid=os.fork()
if pid==0:
    os.close(r);os.setsid();os.write(w,b'ready');os.close(w);signal.pause()
else:
    os.close(w);assert os.read(r,5)==b'ready';os.close(r)
    pathlib.Path('ready.json').write_text(json.dumps({'child':os.getpid(),'grandchild':pid}))
    signal.pause()
'''

    def assert_stopped(self):
        ids=json.loads((self.root/'ready.json').read_text())
        for pid in ids.values():self.assertFalse(Path(f'/proc/{pid}').exists(),f'owned pid {pid} remains')

    def test_timeout_cleans_detached_grandchild(self):
        with self.assertRaises(OwnedProcessFailure) as caught:self.run_owned(self.child_code())
        record=caught.exception.record
        self.assertEqual(record['error'],'timeout');self.assertEqual(record['cleanup']['status'],'stopped')
        self.assertEqual(record['timeout_s'],1.)
        self.assert_stopped()

    def nested(self,inner_timeout):
        out=self.root/'geometry'
        code=f'''from pathlib import Path
from run_rgb_scene_reconstruction import recorded_step
import os,sys
out=Path({str(out)!r});out.mkdir()
recorded_step('P12_fixture',[sys.executable,'-c',{self.child_code()!r}],out=out,steps=[],repo=Path.cwd(),environment=dict(os.environ),timeout={inner_timeout})
'''
        return out,code

    def test_nested_inner_timeout_propagates_exact_phase_to_outer_blocker(self):
        out,code=self.nested(.5)
        with self.assertRaises(OwnedProcessFailure):
            run_command('scene_geometry',[sys.executable,'-c',code,'--output-dir',str(out)],[out/'result.json'],root=self.root,cwd=self.root,env=self.env,timeout=4.)
        inner=json.loads((out/'execution.json').read_text())['steps'][0]
        blocker=json.loads((self.root/'state/runtime_blockers/scene_geometry.json').read_text())
        self.assertEqual(inner['phase'],'P12_fixture');self.assertEqual(inner['error'],'timeout')
        self.assertEqual(inner['cleanup']['status'],'stopped')
        self.assertEqual(blocker['failed_inner_phase'],'P12_fixture')
        self.assertEqual(blocker['nested_execution']['steps'][0],inner)
        self.assert_stopped()

    def test_outer_expiry_allows_nested_cancellation_record_and_cleanup(self):
        out,code=self.nested(20.)
        with self.assertRaises(OwnedProcessFailure):
            run_command('scene_geometry',[sys.executable,'-c',code,'--output-dir',str(out)],[out/'result.json'],root=self.root,cwd=self.root,env=self.env,timeout=1.5)
        inner=json.loads((out/'execution.json').read_text())['steps'][0]
        self.assertEqual(inner['error'],'outer_cancelled')
        self.assertEqual(inner['cleanup']['status'],'stopped')
        self.assert_stopped()

    def test_timeout_does_not_signal_external_preexisting_process(self):
        import subprocess
        outsider=subprocess.Popen([sys.executable,'-c','import signal; print("ready",flush=True); signal.pause()'],stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(outsider.stdout.readline().strip(),'ready')
            with self.assertRaises(OwnedProcessFailure):self.run_owned(self.child_code(),timeout=.5)
            self.assertIsNone(outsider.poll())
            self.assert_stopped()
        finally:
            outsider.terminate();outsider.wait(timeout=2);outsider.stdout.close()

    def test_exhausted_inner_budget_never_spawns(self):
        with self.assertRaises(OwnedProcessFailure) as caught:self.run_owned('raise AssertionError("must not run")',timeout=0)
        self.assertEqual(caught.exception.record['error'],'deadline_exhausted_before_spawn')
        self.assertFalse((self.root/'fixture.log').exists())

if __name__=='__main__':unittest.main()
