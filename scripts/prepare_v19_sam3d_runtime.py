#!/usr/bin/env python3
"""Developer-side preparation of a curated, offline RGB/SAM3D workspace.

This command copies code/assets, checks imports, and optionally copies raw input.
It never launches a runtime agent or invokes a prediction/model stage.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import uuid
from v19_runtime_contracts import interpreter, load, sha
from v20_prediction_contracts import find_positive_gt_flags
from v19_video_timing import read_cfr_video_timing
from v19_runtime_assets import external_asset_identity, verify_bundle

SPEC = 'runtime/rgb_sam3d_runtime_spec.md'
PROMPT = 'runtime/rgb_sam3d_system_prompt.md'
NATIVE_PROFILE = '.pi/agents/ego-annotation.rgb-sam3d-runtime.md'
DEMO_PROFILE = '.pi/agents/ego-annotation.rgb-sam3d-demo.md'
ADDITIVE_DEMO_HELPER = 'scripts/run_v19_additive_demo_branch.py'
NATIVE_PROVIDER = '/mnt/user-home/kupingxin/ego_annotation_worktrees/shellgate_native_write_transport_fix_20260909/extensions/shellgate.ts'
SCRIPT_RE = re.compile(r'scripts/[A-Za-z0-9_]+\.(?:py|sh)')
ASSET_KEYS = {
    'compute_host', 'gpu', 'main_python', 'demo_python', 'rgb_python', 'unidepth_python',
    'owlv2_python', 'hawor_root', 'unidepth_repo', 'unidepth_model', 'owlv2_model',
    'sam2_source', 'sam2_checkpoint', 'wilor_root', 'mano_models', 'lightglue_root',
    'lightglue_torch_home', 'sam3d_python', 'sam3d_repo', 'sam3d_config',
    'sam3d_moge_checkpoint', 'sam3d_dinov2_repo', 'sam3d_dinov2_checkpoint',
    'sam3d_min_free_mib', 'sam3d_environment', 'render_width',
}
# Only imported helpers of the late-window module are used by the fresh
# initializer. Its optional generated-validation CLI is not a runtime stage.
UNUSED_DYNAMIC = {
    'scripts/fit_v20_late_window_prediction_only_se3.py': {'v20_generated_first_hit_factors.py'},
}
EXPERIMENT_HELPERS = {
    'experiments/sam3d_p11_p12_branch/run_p12_parallel_geometry_priors.py',
    'experiments/sam3d_p11_p12_branch/build_p11_dual_geometry_inputs.py',
    'experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py',
}
TORCH_WEIGHTS = ('superpoint_v1.pth', 'superpoint_lightglue_v0-1_arxiv.pth')
FORBIDDEN_INSTRUCTIONS = ('AGENTS.md', '.memory/', 'Workbench', 'EPISTEMIC', 'ablation', 'evaluator', 'parent-session')


def validate_native_profile(path):
    text=Path(path).read_text()
    pieces=text.split('---',2)
    if len(pieces)!=3 or pieces[0].strip():raise ValueError('native profile frontmatter missing')
    fields={}
    for line in pieces[1].splitlines():
        if not line.strip():continue
        key,value=line.split(':',1)
        if key in fields:raise ValueError('duplicate native profile field')
        fields[key]=value.strip()
    expected={'name':'rgb-sam3d-runtime','package':'ego-annotation','systemPromptMode':'replace',
        'inheritProjectContext':'false','inheritGlobalContext':'false','inheritSkills':'false',
        'defaultContext':'fresh','subagentOnlyExtensions':NATIVE_PROVIDER}
    if any(fields.get(k)!=v for k,v in expected.items()):raise ValueError('native profile context/provider mismatch')
    if set(t.strip() for t in fields.get('tools','').split(',')) != {'read','write','bash','shellgate_connect','shellgate_input','contact_supervisor'}:
        raise ValueError('native profile tools mismatch')
    provider=Path(NATIVE_PROVIDER)
    return [provider,provider.with_name('shellgate-broker.py'),provider.parent.parent/'package.json',provider.parent.parent/'package-lock.json']


def closure(source):
    source = Path(source).resolve()
    available = {p.name: p.relative_to(source) for p in (source / 'scripts').glob('*') if p.suffix in {'.py', '.sh'}}
    available.update({Path(p).name: Path(p) for p in EXPERIMENT_HELPERS})
    modules = {Path(k).stem: v for k, v in available.items()}
    queue = [Path(p) for p in SCRIPT_RE.findall((source / SPEC).read_text())]
    selected = set()
    while queue:
        relative = queue.pop()
        if relative in selected:
            continue
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing/nonlocal runtime script: {relative}')
        selected.add(relative)
        text = path.read_text()
        queue.extend(Path(p) for p in SCRIPT_RE.findall(text))
        if path.suffix != '.py':
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(modules[a.name.split('.')[0]] for a in node.names if a.name.split('.')[0] in modules)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split('.')[0] in modules:
                    queue.append(modules[node.module.split('.')[0]])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Covers script('name.py'), Path/.../'name.py', with_name and
                # literal filenames used by importlib loaders (including P13).
                if node.value in available and node.value not in UNUSED_DYNAMIC.get(str(relative), set()):
                    queue.append(available[node.value])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'script':
                if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value not in available:
                    raise ValueError(f'unresolved dynamic script dependency in {relative}')
    return sorted(selected)


def validate_config(config):
    missing = ASSET_KEYS - config.keys()
    extra = config.keys() - ASSET_KEYS
    if missing or extra:
        raise ValueError(f'asset-only config keys: missing={sorted(missing)}, unexpected={sorted(extra)}')
    if not isinstance(config['gpu'],int) or config['gpu'] < 0 or not isinstance(config['render_width'],int) or config['render_width'] < 2:
        raise ValueError('invalid GPU/render width')
    if config['compute_host'] != socket.gethostname():
        raise ValueError('preflight must run on the declared compute host')
    for key, value in config.items():
        if key.endswith('_python'):
            interpreter(value)
        elif key not in {'compute_host', 'gpu', 'render_width', 'sam3d_min_free_mib', 'sam3d_environment'}:
            path = Path(value).expanduser()
            if not path.is_absolute() or not path.exists():
                raise ValueError(f'missing model-source/weight asset {key}: {value}')
            if path.suffix.lower() in {'.mp4', '.npz', '.ply', '.glb', '.json'}:
                raise ValueError(f'clip prediction is not a model asset: {key}')
    env = config['sam3d_environment']
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError('sam3d_environment must contain strings')
    if env.get('LIDRA_SKIP_INIT', '').lower() != 'true':
        raise ValueError('SAM3D lightweight import overlay requires LIDRA_SKIP_INIT=true')
    required = [
        Path(config['mano_models']) / f'MANO_{side}.pkl' for side in ('LEFT', 'RIGHT')
    ] + [
        Path(config['wilor_root']) / 'wilor/models/mano_wrapper.py',
        Path(config['wilor_root']) / 'mano_data/mano_mean_params.npz',
        Path(config['sam2_source']) / 'sam2/build_sam.py',
        Path(config['lightglue_root']) / 'lightglue/superpoint.py',
        Path(config['lightglue_root']) / 'lightglue/lightglue.py',
        Path(config['hawor_root']) / '.venv_hawor/bin/python',
        Path(config['hawor_root']) / 'third_party/HaWoR/weights/hawor/checkpoints/hawor.ckpt',
        Path(config['hawor_root']) / 'third_party/HaWoR/weights/hawor/checkpoints/infiller.pt',
        Path(config['hawor_root']) / 'third_party/HaWoR/weights/hawor/model_config.yaml',
    ]
    # SAM3D config's relative checkpoint/config references are resolved by its
    # own inference loader relative to pipeline.yaml, not the shell cwd.
    for line in Path(config['sam3d_config']).read_text().splitlines():
        if re.match(r'^\w+_(?:config|ckpt)_path:', line):
            required.append(Path(config['sam3d_config']).parent / line.split(':', 1)[1].strip())
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'missing required model asset: {path}')
    for key in ('unidepth_model', 'owlv2_model'):
        model = Path(config[key])
        if not (model / 'config.json').is_file() or not any(model.glob('*.safetensors')):
            raise ValueError(f'offline model config/weights missing: {model}')
    if not (Path(config['hawor_root']) / 'third_party/HaWoR/.git').exists():
        raise ValueError('HaWoR source checkout identity is missing')
    weights = {}
    for name in TORCH_WEIGHTS:
        path = Path(config['lightglue_torch_home']) / 'hub/checkpoints' / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'offline LightGlue weight missing: {path}')
        weights[name] = {'path': str(path), 'sha256': sha(path)}
    return weights


def copy_package(source, destination):
    # Code/config closure only; no checkout metadata, tests, prompts, demos,
    # documentation, downloaded clip data or cached interpreter products.
    allowed = {'.py', '.yaml', '.yml', '.json', '.cu', '.cpp', '.h', '.hpp', '.so'}
    for path in sorted(Path(source).rglob('*')):
        rel = path.relative_to(source)
        if any(p in {'.git', '__pycache__', 'tests', 'docs', 'demo', 'notebooks', '.memory'} for p in rel.parts):
            continue
        if path.is_file() and path.suffix in allowed:
            dest = Path(destination) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)


def import_preflight(bundle, config):
    # Imports only: never instantiate a model, call cuda(), or execute a stage.
    checks = [
        ('main', config['main_python'], ['numpy', 'scipy', 'torch', 'cv2', 'open3d', 'trimesh', 'smplx', 'rerun',
          'run_v19_sam3d_integration_stage', 'run_v19_sam3d_reconstruction', 'build_v19_rgb_pose_initializer',
          'build_v19_base_annotations', 'build_v19_visible_geometry_from_sam2_depth',
          'fit_v18_compact_rigid_object_pose', 'filter_v19_rigid_completion_multiview_support',
          'solve_v19_rigid_object_pose_graph', 'build_v18_mano_object_constraint_state',
          'build_v19_visible_contact_ownership_factor', 'solve_v18_joint_mano_interval_trajectory',
          'build_v19_mano_surface_hypothesis_state', 'build_v19_rigid_render_state',
          'render_v19_rigid_state_artifact', 'assemble_v19_annotation_layers',
          'render_v19_annotation_layers_rrd', 'render_v19_annotation_layers_video'], []),
        ('rgb', config['rgb_python'], ['torch', 'cv2', 'lightglue'], [config['lightglue_root']]),
        ('depth', config['unidepth_python'], ['torch', 'cv2', 'unidepth'], [config['unidepth_repo']]),
        ('detector_sam2', config['owlv2_python'], ['torch', 'transformers', 'sam2.build_sam'], [config['sam2_source']]),
        ('hawor', str(Path(config['hawor_root']) / '.venv_hawor/bin/python'), ['torch', 'cv2', 'smplx'], []),
        ('sam3d', config['sam3d_python'], ['torch', 'hydra', 'omegaconf', 'sam3d_objects', 'inference'],
         [config['sam3d_repo'], str(Path(config['sam3d_repo']) / 'notebook')]),
    ]
    results = []
    for name, python, modules, paths in checks:
        code = 'import sys,importlib; sys.path[:0]=' + repr([str(bundle / 'scripts'), *paths]) + '; '
        code += '; '.join(f'importlib.import_module({m!r})' for m in modules)
        code += '; print("imports_ok")'
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TORCH_HOME': config['lightglue_torch_home'],
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        env.pop('PYTHONPATH', None)
        if name == 'sam3d':
            env.update(config['sam3d_environment'])
        # Never allow optional overlays to expose GPUs or enable networking.
        env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        env.pop('PYTHONPATH',None)
        done = subprocess.run([str(interpreter(python)), '-c', code], cwd=bundle,
                              env=env, capture_output=True, text=True, timeout=90)
        results.append({'check': name, 'argv': [python, '-c', code], 'returncode': done.returncode,
                        'stdout': done.stdout, 'stderr': done.stderr, 'inference_executed': False})
    return results


def build(source, bundle, assets, report_dir, *, check_imports=True):
    source, bundle, report_dir = map(lambda p: Path(p).resolve(), (source, bundle, report_dir))
    if bundle.exists() or bundle.is_relative_to(source):
        raise ValueError('new isolated bundle directory required')
    for executable in ('ffprobe','ffmpeg'):
        if shutil.which(executable) is None:
            raise ValueError(f'required video executable missing: {executable}')
    config = load(assets)
    weights = validate_config(config)
    selected = closure(source)
    for instruction in (SPEC, PROMPT):
        text = (source / instruction).read_text()
        for token in FORBIDDEN_INSTRUCTIONS:
            if token.lower() in text.lower():
                raise ValueError(f'forbidden instruction context: {instruction}: {token}')
    native_assets=validate_native_profile(source/NATIVE_PROFILE)
    demo_profile=(source/DEMO_PROFILE).read_text()
    for required in ['name: rgb-sam3d-demo','package: ego-annotation','systemPromptMode: replace',
                     'inheritProjectContext: false','inheritGlobalContext: false','inheritSkills: false',
                     'defaultContext: fresh',f'subagentOnlyExtensions: {NATIVE_PROVIDER}']:
        if required not in demo_profile:raise ValueError('additive demo native profile mismatch: '+required)
    for path in native_assets:
        if not path.is_file():raise ValueError(f'native provider file missing: {path}')
    bundle.mkdir(parents=True)
    runtime_files=[*selected,Path(SPEC),Path(PROMPT),Path(NATIVE_PROFILE),Path(DEMO_PROFILE),Path(ADDITIVE_DEMO_HELPER)]
    if len(runtime_files)!=len(set(runtime_files)):raise ValueError('duplicate curated runtime file')
    for relative in runtime_files:
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    for source_key, package, destination in [
        ('sam2_source', 'sam2', 'third_party/sam2/sam2'),
        ('wilor_root', 'wilor', 'third_party/WiLoR/wilor'),
        ('lightglue_root', 'lightglue', 'third_party/LightGlue/lightglue'),
    ]:
        copy_package(Path(config[source_key]) / package, bundle / destination)
    mano = bundle / 'third_party/WiLoR/mano_data'
    mano.mkdir(parents=True)
    shutil.copyfile(Path(config['wilor_root']) / 'mano_data/mano_mean_params.npz', mano / 'mano_mean_params.npz')
    for side in ('LEFT', 'RIGHT'):
        shutil.copyfile(Path(config['mano_models']) / f'MANO_{side}.pkl', mano / f'MANO_{side}.pkl')
    for key, target in [('wilor_root', 'third_party/WiLoR'), ('sam2_source', 'third_party/sam2'), ('lightglue_root', 'third_party/LightGlue')]:
        original = Path(config[key])
        for name in ('LICENSE', 'license.txt', 'LICENSE_cctorch'):
            if (original / name).is_file():
                shutil.copyfile(original / name, bundle / target / name)
        config[key] = str(bundle / target)
    config['mano_models'] = str(mano)
    config['lightglue_weights'] = weights
    (bundle / 'runtime/runtime_config.json').write_text(json.dumps(config, indent=2) + '\n')
    for path in bundle.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    # Verify closure on the copied tree, without resolving back to development.
    if closure(bundle) != selected:
        raise ValueError('copied script closure differs')
    asset_identity=external_asset_identity(config)
    asset_identity['native_tool_files']=list(map(str,native_assets))
    for path in native_assets:
        asset_identity['files'][str(path)]={'sha256':sha(path),'resolved_path':str(path.resolve())}
    results = import_preflight(bundle, config) if check_imports else []
    files = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    manifest = {'schema': 'v19_rgb_sam3d_bundle_v1', 'scripts': list(map(str, [*selected,Path(ADDITIVE_DEMO_HELPER)])), 'files': files,
                'external_assets':asset_identity,
                'preflight_status': 'imports_checked' if results and all(r['returncode'] == 0 for r in results) else 'imports_unchecked_or_failed',
                'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()}
    (bundle / 'runtime/bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'status': 'prepared_imports_checked' if results and all(r['returncode'] == 0 for r in results) else 'blocked_or_imports_not_checked',
              'bundle': str(bundle), 'source': str(source), 'source_commit': manifest['source_commit'],
              'script_count': len(selected), 'file_count': len(files), 'imports': results,
              'external_asset_file_count':len(asset_identity['files']),
              'external_asset_bytes':sum(Path(p).stat().st_size for p in asset_identity['files']),
              'prediction_executed': False, 'gpu_live_check_executed': False,
              'native_profile':NATIVE_PROFILE,'native_launch_tool_probe':'pending_separate_supervisor_validation',
              'video_executables': {name:shutil.which(name) for name in ('ffprobe','ffmpeg')}}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    if check_imports and any(r['returncode'] for r in results):
        raise RuntimeError(f'import preflight blocked; inspect {report_dir / "preflight.json"}')
    return report


def prepare_run(bundle, run_root, video, case, sensor=None, sensor_source=None,
                sensor_authority='prediction_side_sensor_metadata', sensor_key=None):
    bundle, root, video = (Path(p).resolve() for p in (bundle, run_root, video))
    if root.exists() or root.is_relative_to(bundle):
        raise ValueError('fresh run directory outside the bundle required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', case):
        raise ValueError('unsafe case ID')
    if not video.is_file():
        raise ValueError('raw video missing')
    manifest = load(bundle / 'runtime/bundle_manifest.json')
    if manifest.get('preflight_status') != 'imports_checked':
        raise ValueError('bundle import preflight has not passed')
    verify_bundle(bundle)
    timing = read_cfr_video_timing(video)
    if sensor:
        payload = load(sensor)
        if find_positive_gt_flags(payload):
            raise ValueError('sensor metadata must be prediction-side')
        if sensor_source is None or sha(sensor_source) != sha(video):
            raise ValueError('sensor-associated source video identity mismatch')
    (root / 'input').mkdir(parents=True)
    shutil.copyfile(video, root / 'input/source.mp4')
    if sensor:
        shutil.copyfile(sensor, root / 'input/sensor_calibration.json')
    contract = {'run_id':str(uuid.uuid4()),'run_root':str(root),'case': case, 'input_video': str(root / 'input/source.mp4'), 'input_video_sha256': sha(video),
                'sensor_calibration_metadata': str(root / 'input/sensor_calibration.json') if sensor else None,
                'sensor_calibration_sha256':sha(sensor) if sensor else None,
                'sensor_calibration_authority': sensor_authority if sensor else None,
                'sensor_frame_intrinsics_key': sensor_key, 'source_timing':timing.report(), 'bundle_manifest_sha256': sha(bundle / 'runtime/bundle_manifest.json')}
    (root / 'input/runtime_input_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    for name in ('logs', 'state', 'measurements', 'renders'):
        (root / name).mkdir()
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bundle')
    for name in ('source-root', 'bundle-root', 'asset-paths', 'report-dir'):
        b.add_argument('--' + name, type=Path, required=True)
    r = sub.add_parser('prepare-run')
    for name in ('bundle-root', 'run-root', 'video'):
        r.add_argument('--' + name, type=Path, required=True)
    r.add_argument('--case', required=True)
    r.add_argument('--sensor-metadata', type=Path)
    r.add_argument('--sensor-source-video', type=Path)
    r.add_argument('--sensor-authority', choices=('dataset_sensor_calibration', 'prediction_side_sensor_metadata', 'explicit_user_calibration'), default='prediction_side_sensor_metadata')
    r.add_argument('--sensor-frame-intrinsics-key')
    a = parser.parse_args()
    if a.command == 'bundle':
        result = build(a.source_root, a.bundle_root, a.asset_paths, a.report_dir)
    else:
        result = prepare_run(a.bundle_root, a.run_root, a.video, a.case, a.sensor_metadata,
                             a.sensor_source_video, a.sensor_authority, a.sensor_frame_intrinsics_key)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
