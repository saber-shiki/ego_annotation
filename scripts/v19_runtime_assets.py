"""Identity of external runtime assets, not a portable environment snapshot."""
from pathlib import Path
import os
import re
from v19_runtime_contracts import sha, load, interpreter

SOURCE_SUFFIXES = {'.py', '.so', '.yaml', '.yml', '.json', '.cu', '.cpp', '.h', '.hpp', '.sh'}
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', 'outputs', 'results', 'examples', 'example',
            'tests', 'docs', 'doc', '.cache', '_DATA', 'hot3d', 'weights', 'checkpoints'}


def tree_files(root, kind):
    root=Path(root)
    found=[]
    excluded=EXCLUDED if kind=='source' else {'.git','.cache','__pycache__'}
    for directory, dirs, files in os.walk(root,followlinks=True):
        dirs[:]=sorted(d for d in dirs if d not in excluded)
        for name in sorted(files):
            path=Path(directory)/name
            if path.is_file() and (kind!='source' or path.suffix in SOURCE_SUFFIXES):found.append(path)
    return sorted(found)


def source_files(root):
    return tree_files(root,'source')


def external_asset_identity(config):
    """Enumerate only consumed external source/model/interpreter roots.

    SAM2, WiLoR, MANO and LightGlue code have already been copied and remapped;
    their old originals are deliberately not referenced here.
    """
    files = set()
    trees = {}
    def add(path):
        path = Path(os.path.abspath(Path(path).expanduser()))
        if not path.is_file():
            raise ValueError(f'external asset missing: {path}')
        files.add(path)
    def tree(path, kind):
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f'external asset tree missing: {path}')
        members = tree_files(path,kind)
        trees[str(path)] = {'kind': kind, 'members': [str(p.relative_to(path)) for p in members]}
        for p in members: add(p)
    for key in ('sam2_checkpoint','sam3d_moge_checkpoint','sam3d_dinov2_checkpoint'):
        add(config[key])
    for key in ('unidepth_model','owlv2_model'):
        tree(config[key], 'model')
    tree(Path(config['unidepth_repo'])/'unidepth', 'source')
    tree(Path(config['sam3d_dinov2_repo'])/'dinov2', 'source')
    add(Path(config['sam3d_dinov2_repo'])/'hubconf.py')
    for sub in ('sam3d_objects','notebook'):
        tree(Path(config['sam3d_repo'])/sub, 'source')
    hawor = Path(config['hawor_root'])/'third_party/HaWoR'
    for sub in ('hawor','infiller','lib','scripts/scripts_test_video','thirdparty'):
        tree(hawor/sub, 'source')
    for name in ('_DATA/data/mano/MANO_RIGHT.pkl','_DATA/data_left/mano_left/MANO_LEFT.pkl',
                 '_DATA/data/mano_mean_params.npz','thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth'):
        add(hawor/name)
    for name in ('hawor/checkpoints/hawor.ckpt','hawor/checkpoints/infiller.pt','hawor/model_config.yaml'):
        add(hawor/'weights'/name)
    # Model-source helpers may use other packaged regression/SLAM weights.
    tree(hawor/'weights', 'model')
    pipeline = Path(config['sam3d_config'])
    add(pipeline)
    for line in pipeline.read_text().splitlines():
        if re.match(r'^\w+_(?:config|ckpt)_path:',line):
            add(pipeline.parent/line.split(':',1)[1].strip())
    for record in config['lightglue_weights'].values():add(record['path'])
    executables = {config[k] for k in config if k.endswith('_python')}
    executables.add(str(Path(config['hawor_root'])/'.venv_hawor/bin/python'))
    interpreters = {}
    for value in sorted(executables):
        path = interpreter(value)
        add(path)
        # Byte identity of executable plus venv binding (not all site-packages).
        cfg = path.parent.parent/'pyvenv.cfg'
        if cfg.exists():add(cfg)
        interpreters[str(path)] = {'resolved_executable':str(path.resolve()),
                                   'venv_config':str(cfg) if cfg.exists() else None}
    activate = Path(config['hawor_root'])/'.venv_hawor/bin/activate'
    add(activate)
    return {'files':{str(p):{'sha256':sha(p),'resolved_path':str(p.resolve())} for p in sorted(files)},
            'trees':trees,'interpreters':interpreters,
            'scope':'consumed source/checkpoints/configs and interpreter executable/venv identity; not all environment packages'}


def verify_external_assets(identity):
    # Loader-owned checkpoints still retain manifest target identity. Their
    # actual loaders own the expensive byte hash; this check performs no reads.
    for name,record in identity.get('resolved_targets',{}).items():
        path=Path(name)
        if not path.is_file() or str(path.resolve())!=record['resolved_path']:
            raise ValueError(f'external asset changed (resolved target): {path}')
    for root, record in identity['trees'].items():
        path = Path(root)
        members = tree_files(path,record['kind'])
        if [str(p.relative_to(path)) for p in members] != record['members']:
            raise ValueError(f'external asset tree membership changed: {root}')
    for name, record in identity['files'].items():
        path=Path(name)
        if not path.is_file() or str(path.resolve()) != record['resolved_path'] or sha(path) != record['sha256']:
            raise ValueError(f'external asset changed: {path}')


def runtime_asset_subset(identity, config, role):
    """Fixed entrypoint responsibilities, not a user-configurable skip cache.

    Full inventories remain mandatory at preparation/authorized launch. Runtime
    entrypoints rehash only what they consume. SAM3D's pinned MoGe/DINOv2 loader
    and the offline LightGlue loader verify those checkpoint bytes at load time.
    The geometry dispatcher consumes no model itself; its nested reconstruction
    entry owns the SAM3D checks. All selected bytes are hashed, never mtime-only.
    """
    python_keys = {
        'P01':('main_python',), 'P06':('main_python','owlv2_python'),
        'P07':('main_python',), 'P08':('main_python',), 'P09-propose':('main_python',),
        'P09':('main_python',), 'P11':('main_python',),
        'camera':('main_python','unidepth_python'),
        'geometry_dispatch':('main_python',), 'geometry':('main_python','sam3d_python'),
        'pose':('main_python','rgb_python'), 'pose_initializer':('main_python',),
        'rgb_loader':('rgb_python',), 'interaction':('main_python',),
        'physical-render':('main_python',), 'demo':('main_python','demo_python'),
        'export':('main_python','demo_python'),
    }
    if role not in python_keys:raise ValueError(f'unknown runtime asset role: {role}')
    files=set();trees=set();resolved_targets={}
    def loader_target(path):
        name=str(Path(os.path.abspath(Path(path).expanduser())))
        if name not in identity['files']:raise ValueError(f'loader asset not pinned: {name}')
        resolved_targets[name]=identity['files'][name]
    def file(path):
        name=str(Path(os.path.abspath(Path(path).expanduser())))
        if name not in identity['files']:raise ValueError(f'runtime asset not pinned: {name}')
        files.add(name)
    def tree(path):
        name=str(Path(path))
        if name not in identity['trees']:raise ValueError(f'runtime source tree not pinned: {name}')
        trees.add(name)
        for member in identity['trees'][name]['members']:file(Path(name)/member)
    def python(value):
        name=str(interpreter(value));file(name)
        record=identity['interpreters'].get(name)
        if record is None:raise ValueError(f'interpreter identity not pinned: {name}')
        if record['venv_config']:file(record['venv_config'])
    for key in python_keys[role]:python(config[key])
    if role=='P06':tree(config['owlv2_model'])
    if role=='P07':file(config['sam2_checkpoint'])
    if role=='camera':
        tree(config['unidepth_model']);tree(Path(config['unidepth_repo'])/'unidepth')
        hawor=Path(config['hawor_root'])/'third_party/HaWoR'
        for sub in ('hawor','infiller','lib','scripts/scripts_test_video','thirdparty','weights'):tree(hawor/sub)
        for name in ('_DATA/data/mano/MANO_RIGHT.pkl','_DATA/data_left/mano_left/MANO_LEFT.pkl',
                     '_DATA/data/mano_mean_params.npz','thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth'):file(hawor/name)
        python(Path(config['hawor_root'])/'.venv_hawor/bin/python')
        file(Path(config['hawor_root'])/'.venv_hawor/bin/activate')
    if role=='rgb_loader':
        for record in config['lightglue_weights'].values():loader_target(record['path'])
    if role=='geometry':
        for key in ('sam3d_moge_checkpoint','sam3d_dinov2_checkpoint'):loader_target(config[key])
        for sub in ('sam3d_objects','notebook'):tree(Path(config['sam3d_repo'])/sub)
        tree(Path(config['sam3d_dinov2_repo'])/'dinov2');file(Path(config['sam3d_dinov2_repo'])/'hubconf.py')
        pipeline=Path(config['sam3d_config']);file(pipeline)
        # Verify pipeline bytes before using its declared dependency list.
        verify_external_assets({'files':{str(pipeline):identity['files'][str(pipeline)]},'trees':{}})
        for line in pipeline.read_text().splitlines():
            if re.match(r'^\w+_(?:config|ckpt)_path:',line):file(pipeline.parent/line.split(':',1)[1].strip())
    return {'files':{name:identity['files'][name] for name in sorted(files)},
            'trees':{name:identity['trees'][name] for name in sorted(trees)},
            'resolved_targets':resolved_targets,'role':role}


def verify_bundle(bundle, *, asset_role=None):
    bundle=Path(bundle).resolve()
    manifest=load(bundle/'runtime/bundle_manifest.json')
    for name,digest in manifest['files'].items():
        path=bundle/name
        if not path.resolve().is_relative_to(bundle) or not path.is_file() or sha(path)!=digest:
            raise ValueError(f'bundle file missing/changed: {name}')
    identity=manifest['external_assets']
    if asset_role is not None:
        identity=runtime_asset_subset(identity,load(bundle/'runtime/runtime_config.json'),asset_role)
    verify_external_assets(identity)
    return manifest
