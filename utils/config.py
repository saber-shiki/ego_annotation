"""Load project settings and external environment/asset declarations."""
from pathlib import Path
import json
import yaml

PROBLEMS = {'standalone_scene', 'prepared_visual'}


def load_config(path):
    path=Path(path)
    value=yaml.safe_load(path.read_text())
    if not isinstance(value,dict) or value.get('problem') not in PROBLEMS:
        raise ValueError('configuration requires an implemented named problem')
    agent=value.get('agent',{})
    if (agent.get('model')!='gpt-6-astra' or agent.get('provider')!='dexgem-responses'
            or agent.get('disable_fallbacks') is not True
            or agent.get('credential_environment')!='EGO_ASTRA_CREDENTIAL_FILE'):
        raise ValueError('Astra model/provider/no-fallback contract cannot be overridden')
    manifest=value.get('assets_manifest')
    if manifest is not None:
        manifest=Path(manifest).expanduser()
        if not manifest.is_absolute():manifest=path.resolve().parent/manifest
        value={**value,'assets':json.loads(manifest.read_text())}
    return value


def require_asset(assets, name):
    path=Path(assets[name]).expanduser().resolve()
    if not path.exists():raise FileNotFoundError(f'missing provisioned asset: {name}')
    return path
