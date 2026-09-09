"""Standalone camera/depth/HaWoR stage extracted from the tested camera producer."""
import os
from pathlib import Path
import hashlib, json
from v20_prediction_contracts import find_positive_gt_flags
from v19_video_timing import read_cfr_video_timing, validate_package_cfr_times
from v19_camera_contract import load_contract as load_camera_contract, plane_intrinsics
from v19_runtime_contracts import in_run, validate_references, validate_timeline, runtime_environment
from rgb_scene_runtime import run_command
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
_LINEAGE=None

def load(path):
    x=json.loads(Path(path).read_text())
    if not isinstance(x,dict):raise ValueError(f'expected JSON object: {path}')
    return x

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def file_in_run(value,root):
    p=in_run(value,root)
    if _LINEAGE is not None and p != root/'state/object_selection.json':
        _LINEAGE.require(p)
    if p.suffix == '.json':
        payload=load(p)
        if find_positive_gt_flags(payload):raise ValueError(f'non-prediction source: {p}')
        validate_references(payload,root)
    return p

def interpreter(value):
    # Do not resolve a virtualenv symlink and accidentally select system Python.
    p=Path(os.path.abspath(Path(value).expanduser()))
    if not p.is_file() or not os.access(p,os.X_OK):raise ValueError(f'interpreter unavailable: {p}')
    return p

def hawor_source_intrinsics(camera, frame_ids, source_size_wh):
    """Read the authoritative source plane in both sensor and RGB-fallback V2 contracts."""
    payload, normalized = load_camera_contract(camera, expected_frame_ids=frame_ids)
    K, _ = plane_intrinsics(payload, normalized, plane_name='source_rgb',
                            actual_size_wh=source_size_wh, allow_implicit_resize=False)
    if K.shape != (4,) or not np.isfinite(K).all() or np.any(K[:2] <= 0):
        raise ValueError('invalid source K')
    if not np.isclose(K[0], K[1], rtol=1e-6, atol=1e-6):
        raise ValueError('HaWoR requires square focal; cannot average supplied fx/fy')
    return K

def copy_verified_raw_video(source, destination, expected_sha256):
    """Verify the new raw copy itself before it enters the lineage cache.

    The first read can refresh write-time metadata on CIFS. The destination
    byte check is required copy verification, not a metadata-only acceptance.
    Later lineage checks and post-command byte checks remain unchanged.
    """
    import shutil
    destination=Path(destination)
    if destination.exists():raise ValueError('HaWoR sequence already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    copied_sha256=sha(destination)
    if copied_sha256 != expected_sha256:
        raise ValueError('HaWoR raw copy byte identity mismatch')
    return copied_sha256

def camera_stage(args, cfg, root):
    main_python=interpreter(cfg['main_python'])
    depth_python=interpreter(cfg['unidepth_python'])
    manifest_path=file_in_run(root/'input/raw_frame_manifest/manifest.json',root)
    manifest=load(manifest_path)
    contract=load(file_in_run(root/'input/runtime_input_contract.json',root))
    video=file_in_run(manifest['input_video'],root)
    if sha(video) != contract['input_video_sha256']:raise ValueError('input video changed')
    if not manifest.get('full_source_timeline'):raise ValueError('full raw-video timeline required')
    frames=manifest['frames'];width=frames[0]['source_width'];height=frames[0]['source_height']
    ids=validate_timeline(manifest,manifest)
    if ids != list(range(len(ids))):raise ValueError('raw manifest must index the full decoded video')
    timing=read_cfr_video_timing(video,expected_frame_count=len(ids))
    validate_package_cfr_times([f['time_s'] for f in sorted(frames,key=lambda f:f['frame_idx'])],timing.frame_rate)
    if timing.source_size_wh != (width,height):raise ValueError('source/manifest dimensions mismatch')
    env=runtime_environment(cfg,args.gpu)
    def run(key,argv,outputs,environment=None):
        run_command(key,argv,outputs,root=root,cwd=ROOT,env=environment or env,
                    lineage=_LINEAGE,extra_inputs=[camera,clip] if key=='P04_HaWoR' else [],
                    capture_roots=[handout,clip.with_suffix('')] if key=='P04_HaWoR' else [],
                    gpu=args.gpu if key in {'P03a_intrinsics_probe','P03_conditioned_depth','P04_HaWoR'} else None)
    probe=root/'measurements/depth_slam/unidepth_intrinsics_probe'
    final=root/'measurements/depth_slam/unidepth_full_frame'
    calibration=root/'state/calibration'
    camera=calibration/'v19_camera_calibration_contract.json'
    depth_name='unidepth_full_frame_depth_v3.npz'
    def depth_command(output):
        return [depth_python,ROOT/'scripts/run_unidepth_full_frame_v3.py','--manifest',manifest_path,
            '--output-dir',output,'--frame-start',0,'--frame-end',ids[-1],
            '--unidepth-repo',cfg['unidepth_repo'],'--model-id',cfg['unidepth_model'],
            '--remote-root',manifest_path.parent,'--local-root',manifest_path.parent,
            '--source-width',width,'--source-height',height]
    sensor=contract.get('sensor_calibration_metadata')
    if sensor:
        extra=['--sensor-calibration-contract',file_in_run(sensor,root),
            '--sensor-calibration-authority',contract['sensor_calibration_authority'],
            '--prediction-source-video',video,'--sensor-source-video',video]
        if contract.get('sensor_frame_intrinsics_key'):
            extra += ['--sensor-frame-intrinsics-key',contract['sensor_frame_intrinsics_key']]
    else:
        run('P03a_intrinsics_probe',depth_command(probe),[probe/depth_name])
        extra=['--unidepth-npz',probe/depth_name,'--fallback-reason','no supplied prediction-side calibration','--aggregation','median','--square-focal']
    run('P03b_camera',[main_python,ROOT/'scripts/resolve_v19_camera_contract.py','--case',contract['case'],
        '--raw-frame-manifest',manifest_path,'--output-dir',calibration,*extra],[camera])
    K=hawor_source_intrinsics(camera, ids, (width,height))
    run('P03_conditioned_depth',depth_command(final)+['--camera-contract',camera,
        '--camera-input-plane','manifest_rgb','--camera-output-plane','source_rgb'],[final/depth_name])
    bound=calibration/'depth_camera_contract/unidepth_full_frame_depth_camera_contract_v2.npz'
    run('P03c_depth_binding',[main_python,ROOT/'scripts/adapt_v19_depth_to_camera_contract.py',
        '--source-depth-npz',final/depth_name,'--camera-contract',camera,'--depth-plane','source_rgb',
        '--output-dir',bound.parent],[bound])
    handout=root/'measurements/hand_candidates/hawor_world'
    if os.environ.get('RGB_SCENE_VERIFIED_RESUME') == '1' and handout.exists():
        for name in ('hawor_world_hands.npz','qc_hawor_world_hands.json'):
            _LINEAGE.require(handout/name)
        return
    if handout.exists():raise ValueError('HaWoR output already exists')
    clip=root/'input/hawor_sequence'/f"{contract['case']}.mp4"
    copied_sha256=copy_verified_raw_video(video,clip,contract['input_video_sha256'])
    hawor_env={**env,'EGO_HAWOR_ROOT':cfg['hawor_root'],'EGO_HAWOR_CASE':contract['case'],
        'EGO_HAWOR_CLIP':str(clip),'EGO_HAWOR_CLIP_SHA256':copied_sha256,'EGO_HAWOR_OUTPUT_DIR':str(handout),
        'EGO_HAWOR_IMG_FOCAL':str(K[0]),'EGO_HAWOR_CAMERA_INTRINSICS_FX_FY_CX_CY':' '.join(map(str,K)),
        'EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH':'1'}
    run('P04_HaWoR',['bash',ROOT/'scripts/remote_run_hawor_export.sh'],
        [handout/'hawor_world_hands.npz',handout/'qc_hawor_world_hands.json'],hawor_env)
