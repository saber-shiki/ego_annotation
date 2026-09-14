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


def preflight_providers(assets, *, include_optional=False):
    """Verify each configured consumer in its own interpreter and package root.

    This does not download weights or instantiate GPU models. It catches a valid
    parent directory being mistaken for the actual HaWoR/LightGlue package root.
    """
    import os, subprocess
    consumers=[
        ('hawor','hawor_python','hawor_root','hawor.utils.process','hawor/utils/process.py'),
        ('unidepth','unidepth_python','unidepth_repo','unidepth.models','unidepth'),
        ('owlv2','owlv2_python',None,'transformers',None),
        ('sam2','unidepth_python','sam2_source','sam2.build_sam','sam2/build_sam.py'),
        ('rgb','rgb_python','lightglue_root','lightglue','lightglue'),
        ('sam3d','sam3d_python','sam3d_repo',None,'notebook/inference.py')]
    if include_optional:consumers.extend([
        ('wilor','wilor_prediction_python','wilor_prediction_root','wilor.models','wilor/models'),
        ('trellis','trellis_python','trellis_repo',None,'trellis/pipelines/trellis_image_to_3d.py')])
    results=[]
    for name,python_key,root_key,module,relative in consumers:
        interpreter=Path(os.path.abspath(Path(assets[python_key]).expanduser()))
        if not interpreter.is_file() or not os.access(interpreter,os.X_OK):
            raise ValueError(f'{name}: configured interpreter is not executable')
        root=Path(assets[root_key]).resolve() if root_key else Path.cwd()
        if not root.is_dir() or (relative and not (root/relative).exists()):
            raise ValueError(f'{name}: configured root lacks actual consumer package {relative}')
        snippet='import importlib,sys; sys.path.insert(0,sys.argv[1]); '
        snippet+=f'importlib.import_module({module!r}); print("consumer import ok")' if module else 'print("consumer file root ok")'
        result=subprocess.run([str(interpreter),'-c',snippet,str(root)],cwd=root,
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},text=True,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        if result.returncode:
            raise RuntimeError(f'{name}: actual consumer import failed: {result.stderr[-2000:]}')
        results.append({'provider':name,'interpreter':str(interpreter),'consumer_root':str(root),
                        'consumer_imported':module,'status':'ok'})
    return results


def recovery_asset_manifest(assets):
    """Bind the declared cold-path model/source trees and checkpoint bytes.

    Interpreter binaries are included; installed site-packages are provisioned
    environments, not recursively inventoried. Credentials and mutable caches
    never enter the manifest. No cross-code/config reuse is inferred from this.
    """
    from utils.state import sha
    keys = ('main_python','demo_python','rgb_python','owlv2_python','unidepth_python',
            'hawor_python','sam3d_python','owlv2_model','sam2_checkpoint','mano_models',
            'lightglue_root','sam3d_repo','sam3d_config','sam3d_moge_checkpoint',
            'sam3d_dinov2_repo','sam3d_dinov2_checkpoint','unidepth_repo','unidepth_model',
            'sam2_source','hawor_root')
    roots = {key: Path(assets[key]).expanduser().absolute() for key in keys}
    # SAM3D config resolves sibling checkpoint files through its checkpoint root.
    roots['sam3d_checkpoint_root'] = roots['sam3d_config'].parent
    for key, item in assets['lightglue_weights'].items():
        roots['lightglue_weights:'+key] = Path(item['path']).expanduser().absolute()
    ignored = {'.git', '__pycache__', '.pytest_cache'}
    content = {}
    hashed = {}
    def visit(path, ancestors):
        resolved = path.resolve(strict=True)
        if resolved.is_dir():
            if resolved in ancestors:
                raise ValueError(f'cyclic model asset directory: {path}')
            for child in sorted(path.iterdir()):
                if child.name not in ignored:
                    visit(child, ancestors | {resolved})
        elif resolved.is_file():
            if resolved not in hashed:
                hashed[resolved] = sha(resolved)
            content[str(path)] = {'resolved_path': str(resolved), 'sha256': hashed[resolved]}
        else:
            raise ValueError(f'unsupported model asset: {path}')
    for path in roots.values():
        visit(path, set())
    return {'schema': 'cold_declared_model_assets_v1', 'roots': {key:str(path) for key,path in roots.items()}, 'files': content}
