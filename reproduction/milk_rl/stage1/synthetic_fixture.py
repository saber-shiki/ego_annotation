"""API-only synthetic adapter; no milk data, MANO, or domain-quality assertions."""
import importlib.util
from pathlib import Path
import numpy as np

_fixture_path = Path('/mnt/user-home/kupingxin/ego_annotation_rl_refactor_baseline/models/test_skrl_runtime.py')
_spec = importlib.util.spec_from_file_location('source_api_fixture', _fixture_path)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
FixtureDataset = _fixture.FixtureDataset
fixture_config = _fixture.fixture_config
write_index = _fixture.write_index
rows = _fixture.rows


class DiagnosticFixtureEnv(_fixture.FixtureAnnotationEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        raw = np.asarray(action).reshape(-1)
        executed = np.asarray(self.actions[-1]['executed'])
        info.update(bundle_id=self.current['bundle_id'], raw_action=raw, executed_action=executed,
                    executed_delta=executed * .01, gate_trace={'allowed_mask': [True] * 3,
                    'reasons': ['fixture_clip'] * 3}, objective=float(self.position),
                    factor_ledger={'end_position': self.position, 'objective': float(self.position),
                                   'terms': {'fixture/term': float(self.position)}})
        return obs, reward, terminated, truncated, info


def config():
    cfg = fixture_config()
    cfg['runtime']['dataset_class'] = 'synthetic_fixture:FixtureDataset'
    cfg['runtime']['environment_class'] = 'synthetic_fixture:DiagnosticFixtureEnv'
    return cfg
