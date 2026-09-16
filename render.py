from __future__ import annotations
'Complete triangle-scene JSON/RRD export and publication, preserving display authority.'
from utils.data import read_json
from utils.data import write_json_payload
from pathlib import Path
import json, re, uuid
import numpy as np
from utils.data import indexed, RKEY, TKEY, load, RawFrameSource
from utils.state import sha, InputIdentity
from observations.common import pose_from_values
from envs.qualification.output import mesh, timeline
from actions.filters.visibility import validate_visibility, apply_display_visibility, POLICY

def export(z, binding, case, object_id, track_id, output, *, raw_frames=None, raw_source=None, mode='complete', calibration=None, visibility=None, provenance=None, identity=None, visual_review=None):
    """All backing arrays are embedded in RRD or numeric JSON, never NPZ/PLY refs."""
    for value in (case, object_id, track_id):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', value):
            raise ValueError('unsafe scene identity')
    if binding.get('object_id') != object_id or binding.get('schema') != 'rgb_scene_visual_refined_binding_v1':
        raise ValueError('typed selected-object binding required')
    if output.exists():
        raise ValueError('final directory already exists')
    (ids, times, order, T, K, size, valid) = timeline(z)
    if not isinstance(identity, InputIdentity) or identity.case != case or binding.get('case', case) != case:
        raise ValueError('typed immutable input identity/case mismatch')
    if identity.document()['source_timing']['frame_count'] != len(ids):
        raise ValueError('input identity/full timeline mismatch')
    visibility = validate_visibility(visibility, ids, object_id)
    expected = apply_display_visibility(z, {'object_id': object_id, 'frames': [{'frame_idx': i, 'visibility': v} for (i, v) in visibility.items()]}, object_id)
    for key in ('object_valid', 'object_pose_valid', 'left_display_valid', 'right_display_valid', 'display_visibility_policy'):
        if key not in z or not np.array_equal(z[key], expected[key]):
            raise ValueError('fitted display validity/policy differs from judgment')
    poses = indexed(binding['pose_rows'])
    source = indexed(binding['source_pose_rows'])
    if set(poses) != set(ids.tolist()):
        raise ValueError('binding/full timeline mismatch')
    for (i, idx) in enumerate(ids):
        (r, t) = pose_from_values(poses[int(idx)][RKEY], poses[int(idx)][TKEY])
        if not np.allclose(z['base_object_rotation'][i], r, atol=1e-07) or not np.allclose(z['base_object_translation_world'][i], t, atol=1e-07):
            raise ValueError('fit reset the refined posterior to a stale source')
    if mode not in {'complete', 'geometry_debug'}:
        raise ValueError('unknown export mode')
    raw_provenance = None
    if mode == 'complete':
        from utils.state import validated_raw_frames
        (verified, raw_provenance) = validated_raw_frames(raw_source, identity)
        if raw_frames is not None and raw_frames != verified:
            raise ValueError('caller RGB differs from producer-bound raw frames')
        raw_frames = verified
    if raw_frames is not None and set(raw_frames) != set(ids.tolist()):
        raise ValueError('RGB/full timeline mismatch')
    if provenance is None:
        provenance = {}
    staging = output.with_name('.' + output.name + '.staging-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    import rerun as rr
    recording_id = str(uuid.uuid4())
    recording = rr.RecordingStream('rgb_scene_annotation', recording_id=recording_id)
    recording.save(str(staging / 'world.rrd'))
    recording.log('/scene/input_identity', rr.TextDocument(json.dumps(identity.document(), sort_keys=True)), static=True)
    recording.log('/world', rr.ViewCoordinates.RDF, static=True)
    recording.log('/scene/semantics', rr.TextDocument('Complete native reconstructed object and MANO surfaces. Camera-origin hand similarity and unsigned triangle proximity are visual hypotheses, not measured contact or signed collision authority.'), static=True)
    objpath = '/world/objects/' + object_id
    handpaths = {s: '/world/hands/' + s for s in ('left', 'right')}

    def ref(path, components):
        return {'file': 'world.rrd', 'recording_id': recording_id, 'entity_path': path, 'timeline': {'sequence': 'frame', 'duration': 'time_s'}, 'components': components}
    recording.log('/world/camera_trajectory', rr.LineStrips3D([T[order, :3, 3]]), static=True)
    frames = []
    mesh_present = False
    try:
        for (decode_position, i) in enumerate(order):
            idx = int(ids[i])
            stamp = float(times[i])
            recording.set_time('frame', sequence=idx)
            recording.set_time('time_s', duration=stamp)
            recording.set_time('decode_position', sequence=decode_position)
            if valid[i]:
                recording.log(objpath, rr.Transform3D(translation=z['object_translation_world'][i], mat3x3=z['object_rotation'][i]))
                if not mesh_present:
                    recording.log(objpath, rr.Mesh3D(vertex_positions=z['object_vertices_canonical'], triangle_indices=z['object_faces'], albedo_factor=[80, 188, 183, 255]))
                mesh_present = True
            else:
                recording.log(objpath, rr.Clear(recursive=True))
                mesh_present = False
            recording.log('/world/camera', rr.Transform3D(translation=T[i, :3, 3], mat3x3=T[i, :3, :3]))
            recording.log('/world/camera/image', rr.Pinhole(image_from_camera=[[K[i, 0], 0, K[i, 2]], [0, K[i, 1], K[i, 3]], [0, 0, 1]], resolution=size))
            if raw_frames is not None:
                import cv2
                raw = raw_frames[idx]
                if abs(float(raw['time_s']) - stamp) > 1e-06:
                    raise ValueError('RGB timestamp mismatch')
                image = cv2.imread(str(raw.get('raw_frame_path') or raw.get('rgb')))
                if image is None:
                    raise ValueError('source RGB unavailable')
                rgb = cv2.cvtColor(cv2.resize(image, tuple(map(int, size))), cv2.COLOR_BGR2RGB)
                recording.log('/world/camera/image', rr.Image(rgb).compress(jpeg_quality=90))
            handstates = {}
            for (si, side) in enumerate(('left', 'right')):
                hv = bool(z[side + '_display_valid'][i])
                path = handpaths[side]
                if hv:
                    recording.log(path, rr.Mesh3D(vertex_positions=z[side + '_vertices_world'][i], triangle_indices=z[side + '_faces'], albedo_factor=[235, 183, 158, 255]))
                else:
                    recording.log(path, rr.Clear(recursive=True))
                params = {}
                if bool(z[side + '_valid'][i]):
                    for key in ('hand_articulation', 'hand_root', 'hand_translation_camera', 'hand_log_scale'):
                        if key in z:
                            params[key] = np.asarray(z[key][i, si]).tolist()
                    for suffix in ('root_orient_axis_angle', 'hand_pose_axis_angle', 'betas', 'trans_world_m'):
                        key = side + '_source_' + suffix
                        if key in z:
                            params['source_' + suffix] = np.asarray(z[key][i]).tolist()
                joints_key = side + '_joints_world'
                joints_world = np.asarray(z[joints_key][i]).tolist() if joints_key in z and bool(z[side + '_valid'][i]) else None
                handstates[side] = {'id': side, 'surface_valid': hv, 'source_surface_valid': bool(z[side + '_valid'][i]), 'surface': ref(path, ['Mesh3D:vertex_positions', 'Mesh3D:triangle_indices']) if hv else None, 'visibility': visibility[idx][side], 'joints_world': joints_world, 'parameters': params, 'semantics': 'MANO LBS with fixed beta; world wrist-root rotation, then camera-origin exp(log_scale) similarity and camera translation', 'uncertainty': 'visual similarity changes metric depth/extent; occluded states are uncertain', 'interaction': str(z['interaction_states'][i, si]) if 'interaction_states' in z else 'unresolved'}
            frames.append({'frame_idx': idx, 'time_s': stamp, 'decode_position': decode_position, 'camera': {'T_world_camera': T[i].tolist(), 'intrinsics_fx_fy_cx_cy': K[i].tolist(), 'image_size_wh': size.tolist(), 'entity': ref('/world/camera', ['Transform3D']), 'uncertainty': 'prediction-side calibrated or estimated camera; see calibration provenance'}, 'object': {'id': object_id, 'track_id': track_id, 'source_prior': source.get(idx), 'refined_graph_pose': {RKEY: poses[idx][RKEY], TKEY: poses[idx][TKEY]}, 'visual_fitted_pose': {RKEY: np.asarray(z['object_rotation'][i]).tolist(), TKEY: np.asarray(z['object_translation_world'][i]).tolist()} if z['object_pose_valid'][i] else None, 'geometry': ref(objpath, ['Mesh3D:vertex_positions', 'Mesh3D:triangle_indices', 'Transform3D']) if valid[i] else None, 'fitted_pose_valid': bool(z['object_pose_valid'][i]), 'display_valid': bool(valid[i]), 'metric_observation': poses[idx].get('metric_observation', False), 'uncertainty': poses[idx].get('uncertainty', 1.0), 'visibility': visibility[idx]['object']}, 'hands': handstates})
        recording.flush()
    finally:
        recording.disconnect()
    document = {'schema': 'rgb_scene_annotation_v1' if mode == 'complete' else 'rgb_scene_geometry_debug_v1', 'partial': mode != 'complete', 'annotation_ready': False, 'physical_contact_authority': False, 'physical_acceptance': {'accepted': False, 'scope': 'standalone_visual_candidate'}, 'visual_acceptance': visual_review or {'status': 'pending_visual_review', 'accepted': False}, 'raw_frame_provenance': raw_provenance, 'scene_id': case, 'object_id': object_id, 'track_id': track_id, 'input_identity': identity.document(), 'display_visibility_policy': POLICY, 'units': 'meters', 'transform_convention': 'column vectors; world_from_camera and world_from_object; left SO3 plus additive world translation', 'canonical_gauge': binding['canonical_gauge'], 'shape_provenance': binding.get('visual_mesh_receipt', {}), 'calibration': calibration or {'source': 'estimated', 'uncertainty': 'unresolved'}, 'geometry_authority': 'native_reconstructed_complete_triangles; unsigned visual proximity only', 'hand_model_provenance': provenance, 'source_prior_sha256': binding['source_prior']['sha256'], 'refined_posterior_sha256': binding['refined_posterior']['sha256'], 'recording': {'file': 'world.rrd', 'sha256': sha(staging / 'world.rrd'), 'recording_id': recording_id, 'application_id': 'rgb_scene_annotation'}, 'frames': frames}

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for (k, v) in value.items() if not (isinstance(v, str) and v.startswith('/'))}
        if isinstance(value, list):
            return [portable(v) for v in value]
        return value
    for f in document['frames']:
        f['object']['source_prior'] = portable(f['object']['source_prior'])
    document['shape_provenance'] = portable(document['shape_provenance'])
    document['calibration'] = portable(document['calibration'])
    document['hand_model_provenance'] = portable(document['hand_model_provenance'])
    with (staging / 'annotation.json').open('x') as f:
        json.dump(document, f, indent=2, allow_nan=False)
    if output.exists():
        raise ValueError('final destination appeared during export')
    staging.rename(output)
    return document

def export_scene(state_path, binding_path, fit_report_path, run_root, bundle_manifest, output, *, visual_review=None):
    """Validate an actual fitted archive and publish complete JSON/RRD geometry."""
    from utils.state import in_run, input_identity, recovery_provenance
    from envs.qualification.object_geometry import validate_posterior
    from observations.object.shape import mesh_arrays
    root = Path(run_root).resolve()
    for path in (state_path, binding_path, fit_report_path):
        in_run(path, root)
    in_run(output, root, must_exist=False)
    binding = load(binding_path)
    fit = load(fit_report_path)
    reference = binding['refined_posterior']
    if sha(in_run(reference['path'], root)) != reference['sha256']:
        raise ValueError('refined state changed before export')
    validate_posterior(load(reference['path']), binding['object_id'], root)
    if fit['input_hashes']['pose_report'] != sha(binding_path):
        raise ValueError('fitter consumed another pose binding')
    mesh_contract = binding['render_mesh_contract']
    mesh_path = in_run(mesh_contract['required_mesh_path'], root)
    if sha(mesh_path) != mesh_contract['required_mesh_sha256'] or fit['mesh_sha256'] != sha(mesh_path):
        raise ValueError('fitter/export mesh identity mismatch')
    (vertices, faces) = mesh_arrays(mesh_path)
    with np.load(state_path, allow_pickle=False) as archive:
        state = dict(archive)
    if not np.array_equal(state['object_faces'], faces) or not np.allclose(state['object_vertices_canonical'], vertices, atol=1e-06, rtol=0):
        raise ValueError('fit changed complete canonical mesh topology or vertices')
    judgment_path = in_run(fit['interaction_judgments'], root)
    if sha(judgment_path) != fit['input_hashes']['interaction_judgments']:
        raise ValueError('interaction judgment changed since fit')
    identity = input_identity(root / 'input/runtime_input_contract.json', root, bundle_manifest)
    raw = indexed(load(root / 'input/raw_frame_manifest/manifest.json')['frames'])
    return export(state, binding, identity.case, binding['object_id'], binding.get('track_id', binding['object_id']), Path(output), raw_source=RawFrameSource(root / 'input/raw_frame_manifest/manifest.json', root), visibility=load(judgment_path), identity=identity, visual_review=visual_review, provenance={'recovery': recovery_provenance(root), 'fit_report_sha256': sha(fit_report_path), 'input_hashes': fit['input_hashes'], 'interaction_diagnostics': load(judgment_path).get('diagnostics', {}), 'timeline_evidence': load(judgment_path).get('timeline_evidence', {})})

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'binding', 'fit-report', 'run-root', 'bundle-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    return export_scene(args.state, args.binding, args.fit_report, args.run_root, args.bundle_manifest, args.output)
if __name__ == '__main__':
    main()

def mesh_overlay(state, position, image):
    """Render actual object/MANO triangles through the declared calibrated camera."""
    import cv2
    import open3d as o3d
    from utils.geometry import resized_K
    width = min(960, image.shape[1])
    height = round(image.shape[0] * width / image.shape[1])
    image = cv2.resize(image, (width, height))
    K = resized_K(state['intrinsics_render'][position], state['render_size_wh'], (width, height))
    T = state['T_world_camera'][position]
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    colors = {}

    def add(vertices, faces, color):
        vertices = (np.asarray(vertices) - T[:3, 3]) @ T[:3, :3]
        key = scene.add_triangles(o3d.core.Tensor(vertices.astype('float32')), o3d.core.Tensor(np.asarray(faces, dtype='uint32')))
        colors[key] = np.asarray(color, float)
    if state['object_valid'][position]:
        add(state['object_vertices_canonical'] @ state['object_rotation'][position].T + state['object_translation_world'][position], state['object_faces'], [183, 188, 80])
    for side in ('left', 'right'):
        if state[side + '_display_valid'][position]:
            add(state[side + '_vertices_world'][position], state[side + '_faces'], [158, 183, 235])
    if not colors:
        return image
    (y, x) = np.mgrid[:height, :width]
    rays = np.stack([np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), (x - K[2]) / K[0], (y - K[3]) / K[1], np.ones_like(x)], axis=-1).astype('float32')
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
    ids = hit['geometry_ids'].numpy()
    normals = hit['primitive_normals'].numpy()
    shading = 0.4 + 0.6 * np.abs(normals[:, :, 2])
    result = image.astype(float)
    for (key, color) in colors.items():
        selected = ids == key
        result[selected] = 0.35 * result[selected] + 0.65 * color * shading[selected, None]
    return np.clip(result, 0, 255).astype('uint8')

def render_video(state_path, raw_manifest_path, output):
    """Full-timeline MP4 driven by complete triangles, not 2-D proxy boxes."""
    import cv2
    output = Path(output)
    if output.exists():
        raise ValueError('video output must be new')
    with np.load(state_path, allow_pickle=False) as archive:
        state = dict(archive)
    (ids, times, order, *_) = timeline(state)
    raw = indexed(load(raw_manifest_path)['frames'])
    if set(raw) != set(map(int, ids)):
        raise ValueError('render raw/state timeline mismatch')
    fps = 1 / float(np.median(np.diff(times))) if len(times) > 1 else 30.0
    writer = None
    reviews = []
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for pos in order:
            row = raw[int(ids[pos])]
            image = cv2.imread(str(row['raw_frame_path']))
            if image is None:
                raise ValueError('render raw RGB missing')
            rendered = mesh_overlay(state, pos, image)
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('MP4 encoder did not open')
            writer.write(rendered)
            review = output.parent / 'review' / f'{int(ids[pos]):06d}.jpg'
            review.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(review), rendered):
                raise RuntimeError('failed to write rendered review frame')
            reviews.append({'frame_idx': int(ids[pos]), 'path': str(review), 'sha256': sha(review)})
    finally:
        if writer is not None:
            writer.release()
    return reviews

def pose_review_images(annotations, binding, hand_archive, frame_ids, output):
    """Project the actual refined full mesh and immutable source MANO for pose review."""
    import cv2
    from observations.object.shape import mesh_arrays
    from utils.mano import hand_validity
    frames = indexed(annotations['frames'])
    poses = indexed(binding['pose_rows'])
    ids = list(frames)
    with np.load(hand_archive, allow_pickle=False) as archive:
        hands = dict(archive)
    positions = {int(i): j for (j, i) in enumerate(hands['frame_idx'])}
    order = [positions[i] for i in ids]
    (vertices, faces) = mesh_arrays(binding['render_mesh_contract']['required_mesh_path'])
    state = {'T_world_camera': np.array([frames[i]['camera']['T_world_camera_metric'] for i in ids]), 'intrinsics_render': np.array([frames[i]['camera']['intrinsics_fx_fy_cx_cy'] for i in ids]), 'render_size_wh': np.array([frames[ids[0]]['source_width'], frames[ids[0]]['source_height']]), 'object_vertices_canonical': vertices, 'object_faces': faces, 'object_valid': np.ones(len(ids), bool), 'object_rotation': np.array([poses[i][RKEY] for i in ids]), 'object_translation_world': np.array([poses[i][TKEY] for i in ids])}
    for side in ('left', 'right'):
        state[side + '_display_valid'] = hand_validity(hands, side, len(ids))[order]
        if state[side + '_display_valid'].any():
            state[side + '_vertices_world'] = hands[side + '_vertices_world_m'][order]
            state[side + '_faces'] = hands[side + '_faces']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frame_ids:
        image = cv2.imread(frames[frame]['raw_frame_path'])
        if image is None:
            raise ValueError('missing raw pose-review image')
        path = output / f'{frame:06d}.jpg'
        if not cv2.imwrite(str(path), mesh_overlay(state, ids.index(frame), image)):
            raise RuntimeError('pose-review triangle rendering failed')
        images.append(path)
    return images
import argparse
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
ACCEPTED_RIGID_POSE_STATUSES = {'fit_to_visible_depth_samples', 'fit_to_visible_depth_archive_vertices', 'corrected_temporal_rigid_pose_graph', 'completed_temporal_rigid_pose_uncertain'}

def formal_state_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--object-label', default=None)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--pose-report', type=Path, required=True)
    parser.add_argument('--completion-report', type=Path, required=True)
    parser.add_argument('--physical-completion-report', type=Path, help='Separate accepted physical surface/readiness; never changes display geometry')
    parser.add_argument('--completed-mesh', type=Path, default=None)
    parser.add_argument('--constraint-report', type=Path, required=True)
    parser.add_argument('--temporal-mano-state', type=Path, default=None)
    parser.add_argument('--hidden-volume-validation', type=Path, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path-rewrite', action='append', default=[], metavar='OLD=NEW', help='Rewrite absolute path prefixes while building state, useful when testing remote-run artifacts locally.')
    parser.add_argument('--allow-missing-poses', action='store_true', help='Write explicit missing-pose uncertainty instead of failing when the rigid trajectory is not full-timeline.')
    return parser.parse_args(argv)

def parse_rewrites(items: list[str]) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []
    for item in items:
        if '=' not in item:
            raise RuntimeError(f'invalid --path-rewrite {item!r}; expected OLD=NEW')
        (old, new) = item.split('=', 1)
        if not old:
            raise RuntimeError(f'invalid --path-rewrite {item!r}; OLD is empty')
        rewrites.append((old.rstrip('/'), new.rstrip('/')))
    return rewrites

def rewrite_path(path: Path | str | None, rewrites: list[tuple[str, str]]) -> Path | None:
    if path is None:
        return None
    text = str(path)
    for (old, new) in rewrites:
        if text == old or text.startswith(old + '/'):
            text = new + text[len(old):]
            break
    return Path(text)

def completion_report_completed_mesh(path: Path, rewrites: list[tuple[str, str]]) -> Path:
    data = read_json(path)
    outputs = data.get('outputs') if isinstance(data, dict) else None
    if not isinstance(outputs, dict):
        raise RuntimeError(f'completion report {path} has no outputs object')
    value = outputs.get('completed_mesh_labeled') or outputs.get('completed_mesh')
    if not value:
        raise RuntimeError(f'completion report {path} has no completed mesh output')
    out = rewrite_path(str(value), rewrites)
    assert out is not None
    return out

def same_existing_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False

def validate_completed_mesh(completed_mesh: Path, expected_mesh: Path) -> None:
    if completed_mesh.name == 'trellis_mesh.ply' or any((part.startswith('trellis_') for part in completed_mesh.parts)):
        raise RuntimeError(f'render-state mesh contract broken: completed mesh must be the P13 completed-canonical mesh, not raw TRELLIS output ({completed_mesh})')
    if not same_existing_file(completed_mesh, expected_mesh) and completed_mesh.resolve(strict=False) != expected_mesh.resolve(strict=False):
        raise RuntimeError(f'render-state mesh contract broken: pose rows are in the P13 completed-canonical frame, but completed_mesh={completed_mesh} differs from completion outputs.completed_mesh_labeled={expected_mesh}')
    if not completed_mesh.exists() or completed_mesh.stat().st_size <= 0:
        raise RuntimeError(f'completed mesh is missing or empty: {completed_mesh}')

def accepted_pose_rows(pose_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in pose_data.get('pose_rows', []) if isinstance(pose_data.get('pose_rows'), list) else []:
        if not isinstance(raw, dict):
            continue
        if raw.get('status') not in ACCEPTED_RIGID_POSE_STATUSES:
            continue
        frame_idx = int(raw.get('frame_idx'))
        rot = np.asarray(raw.get('rotation_world_from_completed_canonical_matrix'), dtype=np.float64)
        trans = np.asarray(raw.get('translation_world_m'), dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,) or (not np.isfinite(rot).all()) or (not np.isfinite(trans).all()):
            raise RuntimeError(f'invalid rigid pose row for frame {frame_idx}')
        rows.append(dict(raw))
    rows.sort(key=lambda row: int(row['frame_idx']))
    return rows

def frame_ids_from_annotations(annotations: dict[str, Any]) -> list[int]:
    frames = annotations.get('frames')
    if not isinstance(frames, list) or not frames:
        raise RuntimeError('annotations contain no frames')
    out: list[int] = []
    for (pos, frame) in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f'annotation frame at position {pos} is not an object')
        out.append(int(frame.get('frame_idx', pos)))
    return out

def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f'{path} must contain a JSON object')
    return payload

def build_formal_render_state(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rewrites = parse_rewrites(list(args.path_rewrite or []))
    annotations_path = rewrite_path(args.annotations, rewrites)
    pose_report_path = rewrite_path(args.pose_report, rewrites)
    completion_report_path = rewrite_path(args.completion_report, rewrites)
    constraint_report_path = rewrite_path(args.constraint_report, rewrites)
    temporal_mano_path = rewrite_path(args.temporal_mano_state, rewrites)
    hidden_validation_path = rewrite_path(args.hidden_volume_validation, rewrites)
    if annotations_path is None or pose_report_path is None or completion_report_path is None or (constraint_report_path is None):
        raise RuntimeError('required input path unexpectedly resolved to None')
    annotations = read_json(annotations_path)
    pose_data = read_json(pose_report_path)
    completion_data = read_json(completion_report_path)
    constraint_data = read_json(constraint_report_path)
    if not isinstance(annotations, dict) or not isinstance(pose_data, dict) or (not isinstance(completion_data, dict)) or (not isinstance(constraint_data, dict)):
        raise RuntimeError('annotations, pose report, completion report, and constraint report must be JSON objects')
    graph_support = pose_data.get('graph_support') if isinstance(pose_data.get('graph_support'), dict) else {}
    pose_annotation_ready = pose_data.get('annotation_ready')
    graph_support_sufficient = graph_support.get('sufficient')
    pose_readiness_legacy_fields_missing = pose_annotation_ready is None and graph_support_sufficient is None
    pose_trajectory_quarantined = bool(pose_annotation_ready is False or graph_support_sufficient is False)
    expected_mesh = completion_report_completed_mesh(completion_report_path, rewrites)
    completion_outputs = completion_data.get('outputs') if isinstance(completion_data.get('outputs'), dict) else {}
    physical_report_path = rewrite_path(getattr(args, 'physical_completion_report', None), rewrites) or completion_report_path
    physical_data = read_json(physical_report_path)
    physical_outputs = physical_data.get('outputs') or {}
    completion_readiness = physical_data.get('geometry_readiness') or {}
    physical_surface = rewrite_path(physical_outputs.get('collision_eligible_mesh_labeled') or expected_mesh, rewrites)
    if physical_report_path != completion_report_path and 'collision_eligible_mesh_labeled' not in physical_outputs:
        raise ValueError('explicit physical report must declare its physical surface')
    if physical_surface is None:
        raise RuntimeError('completion physical surface could not be resolved')
    if not physical_surface.is_file() or physical_surface.stat().st_size <= 0:
        raise RuntimeError(f'completion physical surface is missing or empty: {physical_surface}')
    completed_mesh = rewrite_path(args.completed_mesh, rewrites) if args.completed_mesh is not None else expected_mesh
    if completed_mesh is None:
        raise RuntimeError('completed mesh could not be resolved')
    validate_completed_mesh(completed_mesh, expected_mesh)
    frame_ids = frame_ids_from_annotations(annotations)
    pose_rows = accepted_pose_rows(pose_data)
    pose_frame_ids = {int(row['frame_idx']) for row in pose_rows}
    missing_pose_frames = sorted(set(frame_ids).difference(pose_frame_ids))
    if missing_pose_frames and (not args.allow_missing_poses):
        preview = missing_pose_frames[:20]
        raise RuntimeError(f'rigid render state is not full-timeline: {len(missing_pose_frames)} annotation frames lack accepted rigid poses; first missing frames={preview}. P15 must complete uncertain poses before P19.')
    constraint_rows = constraint_data.get('constraint_rows') if isinstance(constraint_data.get('constraint_rows'), list) else []
    temporal_payload = load_optional_json(temporal_mano_path)
    hidden_payload = load_optional_json(hidden_validation_path)
    if pose_trajectory_quarantined and constraint_data.get('physical_constraint_quarantined') is not True:
        raise RuntimeError('P15 object pose is explicitly unready, but the supplied P16 constraint report is not quarantined; rerun P16 instead of embedding stale object-relative physical candidates')
    if pose_trajectory_quarantined and temporal_payload is not None and (temporal_payload.get('physical_constraint_quarantined') is not True):
        raise RuntimeError('P15 object pose is explicitly unready, but the supplied P18/P18b temporal state is not quarantined; rerun the source-only P18/P18b quarantine path before building render state')
    output = rewrite_path(args.output, rewrites)
    if output is None:
        raise RuntimeError('output path resolved to None')
    object_label = args.object_label or args.object_id
    state = {'status': 'ok_with_unready_object_pose_quarantine' if pose_trajectory_quarantined else 'ok', 'method': 'build_v19_rigid_render_state', 'annotation_ready': False, 'physical_state_quarantined': pose_trajectory_quarantined, 'claim_scope': 'explicit render-consumed state for a rigid object body, pose trajectory, MANO/contact uncertainty, and projection contract', 'case': str(args.case), 'object_id': str(args.object_id), 'object_label': str(object_label), 'created_unix_s': time.time(), 'inputs': {'annotations': str(annotations_path), 'pose_report': str(pose_report_path), 'completion_report': str(completion_report_path), 'physical_completion_report': str(physical_report_path), 'completed_mesh': str(completed_mesh), 'constraint_report': str(constraint_report_path), 'temporal_mano_state': str(temporal_mano_path) if temporal_mano_path is not None else None, 'hidden_volume_validation': str(hidden_validation_path) if hidden_validation_path is not None else None}, 'annotation_backbone': {'path': str(annotations_path), 'frame_count': len(frame_ids), 'frame_ids': frame_ids, 'raw_video': annotations.get('raw_video') if isinstance(annotations.get('raw_video'), dict) else None}, 'object_geometry': {'state': 'completed_canonical_rigid_mesh', 'completed_mesh_path': str(completed_mesh), 'completion_report_path': str(completion_report_path), 'completion_outputs': completion_outputs, 'mesh_frame': 'completed_canonical', 'raw_trellis_mesh_is_renderable': False, 'observation_surface_path': str(completed_mesh), 'physical_surface': {'mesh': str(physical_surface), 'completion_report': str(physical_report_path), 'completion_report_sha256': sha256(physical_report_path), 'source': completion_readiness.get('signed_geometry_source') or completion_readiness.get('collision_surface_source') or 'completion_report_collision_surface', 'signed_geometry_ready': completion_readiness.get('signed_geometry_ready') is True, 'signed_geometry_consumer_policy': completion_readiness.get('signed_geometry_consumer_policy'), 'generated_faces_collision_eligible': completion_readiness.get('generated_faces_collision_eligible') is True, 'generated_faces_contact_eligible': completion_readiness.get('generated_faces_contact_eligible') is True, 'generated_faces_signed_distance_eligible': completion_readiness.get('generated_faces_signed_distance_eligible') is True, 'geometry_readiness': completion_readiness}}, 'object_pose_trajectory': {'state': 'full_timeline_unresolved_sparse_support_hypothesis' if pose_trajectory_quarantined else 'full_timeline_rigid_pose_trajectory' if not missing_pose_frames else 'rigid_pose_trajectory_with_explicit_missing_frames', 'pose_report_path': str(pose_report_path), 'pose_report_status': pose_data.get('status'), 'annotation_ready': pose_annotation_ready if isinstance(pose_annotation_ready, bool) else None, 'graph_support': graph_support, 'graph_support_sufficient': graph_support_sufficient if isinstance(graph_support_sufficient, bool) else None, 'readiness_legacy_fields_missing': pose_readiness_legacy_fields_missing, 'physical_state_quarantined': pose_trajectory_quarantined, 'accepted_statuses': sorted(ACCEPTED_RIGID_POSE_STATUSES), 'pose_rows': pose_rows, 'frame_count_with_pose': len(pose_frame_ids), 'missing_pose_frames': missing_pose_frames, 'full_timeline_rigid_pose_completion': pose_data.get('full_timeline_rigid_pose_completion'), 'correction_summary': pose_data.get('correction_summary')}, 'mano_constraint_state': {'state': 'constraint_rows_quarantined_unready_object_pose' if pose_trajectory_quarantined else 'constraint_rows_embedded_for_render_labels', 'constraint_report_path': str(constraint_report_path), 'constraint_report_status': constraint_data.get('status'), 'physical_constraint_quarantined': bool(pose_trajectory_quarantined or constraint_data.get('physical_constraint_quarantined') is True), 'constraint_rows': constraint_rows}, 'temporal_mano_state': {'state': 'embedded_object_pose_quarantine' if pose_trajectory_quarantined and temporal_payload is not None else 'embedded' if temporal_payload is not None else 'not_supplied', 'path': str(temporal_mano_path) if temporal_mano_path is not None else None, 'status': temporal_payload.get('status') if isinstance(temporal_payload, dict) else None, 'physical_constraint_quarantined': bool(isinstance(temporal_payload, dict) and temporal_payload.get('physical_constraint_quarantined') is True), 'payload': temporal_payload}, 'hidden_volume_validation': {'state': 'embedded' if hidden_payload is not None else 'not_supplied', 'path': str(hidden_validation_path) if hidden_validation_path is not None else None, 'payload': hidden_payload}, 'projection_contract': {'intrinsics_coordinate_rule': "Use camera/hand fx_fy_cx_cy in the annotation frame's source coordinate system; scale fx,cx by rendered_width/source_width and fy,cy by rendered_height/source_height before projection.", 'source_size_fields': ['frame.source_width/source_height', 'raw_video.width/height'], 'render_size_source': 'actual decoded raw_frame_path image size', 'forbidden_projection_rule': 'Do not project a 1408x1408/source-size K directly onto a 960x960 rendered frame.'}, 'outputs': {'render_state_json': str(output)}, 'evidence': {'total_frames': len(frame_ids), 'pose_rows': len(pose_rows), 'constraint_rows': len(constraint_rows), 'temporal_mano_rows': len(temporal_payload.get('per_frame_states', [])) if isinstance(temporal_payload, dict) and isinstance(temporal_payload.get('per_frame_states'), list) else 0, 'missing_pose_frame_count': len(missing_pose_frames), 'pose_trajectory_annotation_ready': pose_annotation_ready if isinstance(pose_annotation_ready, bool) else None, 'graph_support_sufficient': graph_support_sufficient if isinstance(graph_support_sufficient, bool) else None, 'pose_readiness_legacy_fields_missing': pose_readiness_legacy_fields_missing, 'physical_state_quarantined': pose_trajectory_quarantined}, 'total_elapsed_s': time.time() - started}
    write_json_payload(output, state)
    return state

def formal_state_cli(argv=None) -> None:
    state = build_formal_render_state(formal_state_parse_args(argv))
    print(json.dumps({'status': state['status'], 'method': state['method'], 'output': state['outputs']['render_state_json'], 'evidence': state['evidence']}, indent=2))
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import numpy as np
SCHEMA = 'v19_annotation_layers_v1'
SIDES = ('left', 'right')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def layers_load_json(path):
    data = json.loads(Path(path).read_text())
    require(isinstance(data, dict), f'required JSON object: {path}')
    return data

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def array(data, key, shape=None):
    require(key in data, f'required component: {key}')
    a = np.asarray(data[key])
    require(shape is None or a.shape == shape, f'invalid shape: {key}: {a.shape}, expected {shape}')
    return a

def rigid(rotation, translation, label):
    require(np.isfinite(rotation).all() and np.isfinite(translation).all(), f'nonfinite transform: {label}')
    require(np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-05) and np.allclose(np.linalg.det(rotation), 1, atol=1e-05), f'invalid rigid rotation: {label}')

def same(a, b, label, atol=1e-05):
    require(np.shape(a) == np.shape(b) and np.allclose(a, b, atol=atol, rtol=0), f'mismatch: {label}')

def layers_mesh(vertices, faces, label):
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and (len(vertices) > 0) and np.isfinite(vertices).all(), f'invalid mesh vertices: {label}')
    require(faces.ndim == 2 and faces.shape[1] == 3 and (len(faces) > 0) and np.issubdtype(faces.dtype, np.integer) and (faces.min() >= 0) and (faces.max() < len(vertices)), f'invalid mesh faces: {label}')

def load_geometry_mesh(path):
    """Load only embedded geometry; no material, texture, or buffer dependency reads."""
    import trimesh
    path = Path(path)
    file_type = path.suffix.lower().lstrip('.')
    require(file_type in {'ply', 'obj', 'stl', 'glb'}, 'unsupported self-contained mesh format')
    m = trimesh.load(io.BytesIO(path.read_bytes()), file_type=file_type, resolver={}, allow_remote=False, force='mesh', process=False, skip_materials=True)
    require(isinstance(m, trimesh.Trimesh), 'required component: canonical triangle mesh')
    (v, f) = (np.asarray(m.vertices), np.asarray(m.faces))
    layers_mesh(v, f, 'canonical object')
    return {'object_vertices_canonical': v, 'object_faces': f}

def object_alias(value):
    text = str(value)
    return text.removeprefix('object:')

def resolve_object(frame, object_id):
    """Resolve established object:/track aliases, rejecting competing rows."""
    target = object_alias(object_id)
    matches = [row for row in frame.get('objects', []) if any((value is not None and object_alias(value) == target for value in (row.get('object_id'), row.get('track_id'))))]
    require(len(matches) <= 1, f'ambiguous object/track identity: {object_id}')
    return matches[0] if matches else {}

def object_target_matches(frame, object_id, target):
    if target in (None, ''):
        return True
    source = resolve_object(frame, object_id)
    if source:
        return resolve_object(frame, target) is source
    return object_alias(target) == object_alias(object_id)

def annotation_hawor_references(annotations):
    """Archive identity binds all rows, including invalid rows without joints."""
    references = set()
    declared = annotations.get('v19_inputs', {}).get('hawor_npz')
    if declared:
        references.add(declared)
    for frame in annotations.get('frames', []):
        for hand in frame.get('hands', []):
            ref = (hand.get('metric_mano_state') or {}).get('vertices_reference')
            if ref:
                require(ref.get('source_hawor_npz'), 'required component: vertices_reference.source_hawor_npz')
                references.add(ref['source_hawor_npz'])
    return sorted(references)

def validate_hawor_bindings(annotations, supplied_path, bindings, resolve_asset):
    require(set(bindings) == set(annotation_hawor_references(annotations)), 'annotation HaWoR source bindings mismatch')
    digest = sha256(supplied_path)
    for (reference, binding) in bindings.items():
        require(sha256(resolve_asset(binding['asset'])) == binding['sha256'] == digest, f'annotation HaWoR source archive mismatch: {reference}')

def hand_arrays(source, n, demo=False):
    out = {}
    for side in SIDES:
        suffix = 'world' if demo else 'world_m'
        (vk, jk) = (f'{side}_vertices_{suffix}', f'{side}_joints_{suffix}')
        valid_key = f'{side}_valid'
        if vk not in source and jk not in source:
            require(valid_key not in source or not np.any(source[valid_key]), f'missing valid hand geometry: {side}')
            out[valid_key] = np.zeros(n, dtype=bool)
            continue
        valid = array(source, valid_key, (n,))
        require(np.isin(valid, [0, 1]).all(), f'invalid hand validity: {side}')
        valid = valid.astype(bool)
        v = array(source, vk, (n, 778, 3))
        j = array(source, jk, (n, 21, 3))
        f = array(source, f'{side}_faces')
        require(np.isfinite(v[valid]).all() and np.isfinite(j[valid]).all(), f'nonfinite valid hand: {side}')
        layers_mesh(np.zeros((778, 3)), f, side)
        out.update({valid_key: valid, f'{side}_vertices_world': v, f'{side}_joints_world': j, f'{side}_faces': f})
    require(any((f'{s}_vertices_world' in out for s in SIDES)), 'required component: source MANO surfaces/joints')
    return out

class Snapshot:

    def __init__(self, output, run_root, allow_external):
        self.output = Path(output).resolve()
        self.run_root = Path(run_root).resolve(strict=True)
        self.allow_external = allow_external
        self.assets = {}
        self.by_source = {}

    def source(self, path, relative_to=None):
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (relative_to or self.run_root) / p
        p = p.resolve(strict=True)
        require(p.is_file(), f'required asset is not a file: {p}')
        require(self.allow_external or p.is_relative_to(self.run_root), f'fresh input escapes run-root: {p}')
        return p

    def copy(self, path, relative_to=None):
        p = self.source(path, relative_to)
        if str(p) in self.by_source:
            return self.by_source[str(p)]
        digest = sha256(p)
        rel = f'assets/{len(self.assets):04d}_{p.name}'
        target = self.output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        require(sha256(target) == digest == sha256(p), f'source changed during snapshot: {p}')
        self.assets[rel] = {'sha256': digest, 'size_bytes': target.stat().st_size, 'source': str(p)}
        self.by_source[str(p)] = rel
        return rel

    def generated(self, name, data):
        rel = f'assets/{name}'
        np.savez_compressed(self.output / rel, **data)
        p = self.output / rel
        self.assets[rel] = {'sha256': sha256(p), 'size_bytes': p.stat().st_size, 'source': 'derived_lossless_render_arrays'}
        return rel

def core_arrays(core, annotations, hawor, mesh_path, resolve_archive=None):
    frames = annotations.get('frames')
    require(isinstance(frames, list) and len(frames) > 0, 'required component: annotation frames')
    n = len(frames)
    ids = np.asarray([f['frame_idx'] for f in frames])
    require(np.issubdtype(ids.dtype, np.integer) and (np.diff(ids) > 0).all(), 'reordered/duplicate annotation frame IDs')
    same(ids, core['annotation_backbone']['frame_ids'], 'P19 timeline', atol=0)
    same(ids, array(hawor, 'frame_idx', (n,)), 'HaWoR timeline', atol=0)
    time = np.asarray([f['time_s'] for f in frames], dtype=float)
    require(np.isfinite(time).all() and (np.diff(time) > 0).all(), 'invalid timestamp timeline')
    (sizes, intrinsics, transforms) = ([], [], [])
    for f in frames:
        cam = f.get('camera', {})
        T = np.asarray(cam.get('T_world_camera_metric', cam.get('T_world_camera')))
        require(T.shape == (4, 4), 'required component: camera T_world_camera_metric')
        same(T[3], [0, 0, 0, 1], 'camera homogeneous row')
        rigid(T[:3, :3], T[:3, 3], 'camera')
        K = cam.get('intrinsics_fx_fy_cx_cy', cam.get('current_v18_camera_intrinsics_fx_fy_cx_cy'))
        K = np.asarray(K, dtype=float)
        require(K.shape == (4,) and np.isfinite(K).all() and (K[:2] > 0).all(), 'required component: camera intrinsics')
        raw = annotations.get('raw_video') or {}
        wh = [f.get('source_width') or raw.get('width'), f.get('source_height') or raw.get('height')]
        require(all((isinstance(x, (int, float)) and x > 0 and (int(x) == x) for x in wh)), 'required component: source image dimensions')
        sizes.append(wh)
        intrinsics.append(K)
        transforms.append(T)
    T = np.asarray(transforms)
    same(array(hawor, 'R_c2w', (n, 3, 3)), T[:, :3, :3], 'HaWoR world camera rotation')
    same(array(hawor, 't_c2w', (n, 3)), T[:, :3, 3], 'HaWoR world camera translation')
    require(core['object_geometry'].get('mesh_frame') == 'completed_canonical', 'required component: completed_canonical mesh convention')
    geometry = load_geometry_mesh(mesh_path)
    (v, faces) = (geometry['object_vertices_canonical'], geometry['object_faces'])
    (R, t) = (np.full((n, 3, 3), np.nan), np.full((n, 3), np.nan))
    valid = np.zeros(n, dtype=bool)
    index = {int(f): i for (i, f) in enumerate(ids)}
    for row in core['object_pose_trajectory']['pose_rows']:
        idx = row['frame_idx']
        require(idx in index and (not valid[index[idx]]), 'duplicate/out-of-timeline object pose')
        i = index[idx]
        R[i] = array(row, 'rotation_world_from_completed_canonical_matrix', (3, 3))
        t[i] = array(row, 'translation_world_m', (3,))
        rigid(R[i], t[i], f'object frame {idx}')
        valid[i] = True
    out = dict(frame_idx=ids, time_s=time, render_size_wh=np.asarray(sizes), intrinsics_render=np.asarray(intrinsics), T_world_camera=T, object_vertices_canonical=v, object_faces=faces, object_rotation=R, object_translation_world=t, object_valid=valid)
    out.update(hand_arrays(hawor, n))
    for (i, frame) in enumerate(frames):
        for hand in frame.get('hands', []):
            side = hand.get('hand_side')
            metric = hand.get('metric_mano_state') or {}
            reference = metric.get('vertices_reference')
            if reference and side in SIDES:
                pos = reference.get('source_frame_index')
                require(isinstance(pos, int) and 0 <= pos < n and (ids[pos] == ids[i]), 'annotation HaWoR source row mismatch')
                for kind in ('vertices', 'joints'):
                    declared = reference.get(f'source_{kind}_world_array')
                    require(declared in hawor, f'required annotation source array: {declared}')
                    actual = array(hawor, declared)[pos]
                    expected = out[f'{side}_{kind}_world'][i]
                    require(actual.shape == expected.shape and np.array_equal(actual, expected, equal_nan=True), f'annotation HaWoR source {kind} mismatch')
            joints = metric.get('joints_current_v18_world_m', metric.get('joints_world_m'))
            if side in SIDES and joints is not None:
                require(out[f'{side}_valid'][i], f'annotation/HaWoR validity mismatch: {side}')
                same(joints, out[f'{side}_joints_world'][i], f'annotation/HaWoR source joints: {side}')
    payload = (core.get('temporal_mano_state') or {}).get('payload') or {}
    acceptance = payload.get('full_mano_acceptance') or {}
    if acceptance.get('accepted') is True:
        require(acceptance.get('signed_object_surface_factor_active') is True and acceptance.get('signed_geometry_declared_ready') is True and (acceptance.get('physical_state_quarantined') is not True) and (core.get('physical_state_quarantined') is not True), 'accepted MANO cannot bypass signed/source quarantine')
        reference = payload.get('accepted_full_mano_vertices_world_archive')
        archive = Path(reference) if resolve_archive is None else resolve_archive(reference)
        require(sha256(archive) == payload.get('accepted_full_mano_vertices_world_archive_sha256'), 'accepted full MANO archive identity mismatch')
        data = load_npz(archive)
        rows = {(int(i), str(side)): pos for (pos, (i, side)) in enumerate(zip(data['frame_idx'], data['hand_side']))}
        require(len(rows) == len(data['frame_idx']), 'duplicate accepted MANO archive rows')
        for side in SIDES:
            if not out[side + '_valid'].any():
                continue
            out[side + '_source_vertices_world'] = out[side + '_vertices_world'].copy()
            out[side + '_source_joints_world'] = out[side + '_joints_world'].copy()
            out[side + '_physical_accepted'] = np.zeros(n, bool)
            for (i, fid) in enumerate(ids):
                if not out[side + '_valid'][i]:
                    continue
                require((int(fid), side) in rows, 'accepted MANO lacks a valid source row')
                position = rows[int(fid), side]
                vertices = data['vertices_world_m'][position]
                joints = data['joints_world_m'][position]
                require(vertices.shape == (778, 3) and joints.shape == (21, 3) and np.isfinite(vertices).all() and np.isfinite(joints).all(), 'invalid accepted MANO geometry')
                out[side + '_vertices_world'][i] = vertices
                out[side + '_joints_world'][i] = joints
                out[side + '_physical_accepted'][i] = True
    return out

def demo_mesh_contract(core, completion):
    if 'render_mesh_contract' not in completion:
        return (core['object_geometry']['completed_mesh_path'], None)
    contract = completion['render_mesh_contract']
    require(isinstance(contract, dict) and contract.get('required_mesh_path') and isinstance(contract.get('required_mesh_sha256'), str) and (len(contract['required_mesh_sha256']) == 64), 'required component: completion render_mesh_contract path/hash')
    return (contract['required_mesh_path'], contract['required_mesh_sha256'])

def demo_arrays(demo, prediction, canonical_geometry=None):
    n = len(prediction['frame_idx'])
    for key in ('frame_idx', 'time_s', 'T_world_camera'):
        same(array(demo, key), prediction[key], f'demo {key}')
    size = array(demo, 'render_size_wh')
    if size.shape == (2,):
        size = np.broadcast_to(size, (n, 2))
    require(size.shape == (n, 2) and np.isfinite(size).all() and (size > 0).all() and np.equal(size, np.floor(size)).all(), 'invalid demo render_size_wh')
    scale = size / prediction['render_size_wh']
    expected = prediction['intrinsics_render'].copy()
    expected[:, :2] *= scale
    expected[:, 2:] = (expected[:, 2:] + 0.5) * scale - 0.5
    same(array(demo, 'intrinsics_render', (n, 4)), expected, 'demo intrinsics/render size')
    geometry = prediction if canonical_geometry is None else canonical_geometry
    for key in ('object_vertices_canonical', 'object_faces'):
        same(array(demo, key), geometry[key], f'demo canonical mesh {key}', atol=0 if key == 'object_faces' else 1e-05)
    valid = prediction['object_valid']
    for key in ('object_rotation', 'object_translation_world'):
        base = array(demo, 'base_' + key, prediction[key].shape)
        same(base[valid], prediction[key][valid], f'demo base {key}')
    R = array(demo, 'object_rotation', (n, 3, 3))
    t = array(demo, 'object_translation_world', (n, 3))
    dv = np.asarray(demo.get('object_valid', np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1)))
    require(dv.shape == (n,) and np.isin(dv, [0, 1]).all(), 'invalid demo object validity')
    dv = dv.astype(bool)
    rigid(R[dv], t[dv], 'demo object')
    hands = hand_arrays(demo, n, demo=True)
    for side in SIDES:
        same(hands[f'{side}_valid'], prediction[f'{side}_valid'], f'demo {side} validity', atol=0)
        hv = hands[f'{side}_valid']
        if f'{side}_vertices_world' in hands:
            for kind in ('vertices', 'joints'):
                key = f'{side}_{kind}_world'
                same(array(demo, f'{side}_base_{kind}_world')[hv], prediction[key][hv], f'demo source {side} {kind}')
            same(hands[f'{side}_faces'], prediction[f'{side}_faces'], f'demo {side} topology', atol=0)
    return {**{k: prediction[k] for k in ('frame_idx', 'time_s', 'T_world_camera')}, **{k: geometry[k] for k in ('object_vertices_canonical', 'object_faces')}, 'render_size_wh': size, 'intrinsics_render': expected, 'object_rotation': R, 'object_translation_world': t, 'object_valid': dv, **hands}

def package_asset(root, relative):
    require(isinstance(relative, str) and '\\' not in relative, 'invalid relative asset path')
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and p.parts and (p.parts[0] == 'assets'), 'asset path traversal/absolute path')
    resolved = (root / relative).resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), 'asset escapes package or is missing')
    return resolved

def load_package(path):
    path = Path(path).resolve(strict=True)
    package = layers_load_json(path)
    require(package.get('schema') == SCHEMA, 'unsupported final package schema')
    root = path.parent
    assets = package['assets']
    for (relative, spec) in assets.items():
        p = package_asset(root, relative)
        require(p.stat().st_size == spec['size_bytes'] and sha256(p) == spec['sha256'], f'asset hash mismatch: {relative}')

    def asset(relative):
        require(relative in assets, f'unregistered asset: {relative}')
        return package_asset(root, relative)
    metric = package['metric_prediction']
    core = layers_load_json(asset(metric['source_assets']['render_state']))
    ann = layers_load_json(asset(metric['source_assets']['annotations']))
    require(core == metric['render_state'] and ann == metric['annotations'], 'verbatim source document mismatch')
    require(layers_load_json(asset(metric['source_assets']['interaction_judgments'])) == metric['interaction_judgments'], 'P17 source document mismatch')
    hawor_path = asset(metric['source_assets']['hawor_npz'])
    validate_hawor_bindings(ann, hawor_path, metric.get('hawor_source_bindings', {}), asset)
    expected = core_arrays(core, ann, load_npz(hawor_path), asset(metric['canonical_mesh_asset']), lambda reference: asset(package['source_asset_map'][str(Path(reference).resolve())]))
    pred = load_npz(asset(metric['render_arrays_asset']))
    compare_arrays(pred, expected, 'prediction')
    timeline = package['timeline']
    require(len(timeline) == len(pred['frame_idx']), 'package timeline length mismatch')
    judgments = metric['interaction_judgments']
    source_rows = judgments.get('interaction_judgments', judgments.get('judgments', judgments.get('segments')))
    for (i, row) in enumerate(timeline):
        fid = int(pred['frame_idx'][i])
        require(row['frame_idx'] == fid and row['time_s'] == pred['time_s'][i], 'package timeline/time mismatch')
        for (key, expected_key) in (('T_world_camera', 'T_world_camera'), ('intrinsics_fx_fy_cx_cy', 'intrinsics_render'), ('source_size_wh', 'render_size_wh')):
            same(row['camera'][key], pred[expected_key][i], f'package camera {key}')
        require(row['object']['pose_valid'] == bool(pred['object_valid'][i]), 'package object validity mismatch')
        for side in SIDES:
            require(row['hands'][side]['valid'] == bool(pred[f'{side}_valid'][i]), 'package hand validity mismatch')
            source_hand = next((h for h in ann['frames'][i].get('hands', []) if h.get('hand_side') == side), {})
            require(row['hands'][side]['source_state'] == source_hand, 'package source hand mismatch')
            for (field, default) in (('visibility_state', 'unresolved'), ('occluder', 'unresolved'), ('uncertainty', ['source_uncertainty_not_quantified'])):
                require(row['hands'][side][field] == source_hand.get(field, default), f'package hand {field} mismatch')
        source_object = resolve_object(ann['frames'][i], core['object_id'])
        require(row['object']['source_state'] == source_object, 'package source object mismatch')
        for field in ('visibility_state', 'occluder'):
            require(row['object'][field] == source_object.get(field, 'unresolved'), f'package object {field} mismatch')
        uncertainty = ['source_physical_uncertainty_preserved'] if pred['object_valid'][i] else ['missing_object_pose']
        require(row['object']['uncertainty'] == uncertainty, 'package object uncertainty mismatch')
        selected = []
        for judgment in source_rows:
            target = judgment.get('target_entity_id', judgment.get('object_id', judgment.get('track_id')))
            matches = object_target_matches(ann['frames'][i], core['object_id'], target)
            start = judgment.get('frame_start', judgment.get('start_frame', pred['frame_idx'][0]))
            end = judgment.get('frame_end', judgment.get('end_frame', pred['frame_idx'][-1]))
            if matches and int(start) <= fid <= int(end):
                selected.append(judgment)
        require(row['interaction_judgments'] == selected, 'package P17 timeline mismatch')
        require(row['interaction_uncertainty'] == ([] if selected else ['no_P17_judgment_for_frame']), 'package interaction uncertainty mismatch')
    layers = {'prediction': pred}
    demo = package.get('demo_visual')
    if demo is not None:
        require(demo.get('annotation_ready') is False and demo.get('collision_eligible') is False, 'demo physical authority promotion forbidden')
        require(layers_load_json(asset(demo['source_assets']['report'])) == demo['report'], 'demo report source mismatch')
        report_binding = metric['completion_report_binding']
        require(report_binding['reference'] == core['inputs']['completion_report'], 'completion report binding mismatch')
        completion_path = asset(metric['source_assets']['completion_report'])
        require(sha256(completion_path) == report_binding['sha256'], 'completion report binding hash mismatch')
        completion = layers_load_json(completion_path)
        (reference, digest) = demo_mesh_contract(core, completion)
        binding = demo['canonical_mesh_binding']
        require(binding['reference'] == reference, 'demo canonical mesh binding mismatch')
        mesh_path = asset(binding['asset'])
        expected_digest = digest or sha256(asset(metric['canonical_mesh_asset']))
        require(sha256(mesh_path) == expected_digest, 'demo bound canonical mesh hash mismatch')
        if demo['report'].get('mesh'):
            require(demo['report'].get('mesh_sha256') == expected_digest, 'demo report canonical mesh mismatch')
        geometry = load_geometry_mesh(mesh_path)
        source = load_npz(asset(demo['source_assets']['state']))
        expected_demo = demo_arrays(source, pred, geometry)
        actual = load_npz(asset(demo['render_arrays_asset']))
        compare_arrays(actual, expected_demo, 'demo')
        layers['demo'] = actual
    return (package, layers)

def compare_arrays(actual, expected, label):
    require(actual.keys() == expected.keys(), f'{label} render array keys mismatch')
    for key in expected:
        (a, b) = (actual[key], expected[key])
        require(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-07, equal_nan=True), f'{label} render array mismatch: {key}')
import argparse
import json
from pathlib import Path

def assemble_formal_layers(args):
    output = Path(args.output_dir).resolve()
    require(not output.exists(), f'output-dir must be new: {output}')
    require(not args.demo_report or args.demo_state, 'demo-report requires demo-state')
    require(not args.demo_state or args.demo_report, 'required component: demo-report with demo-state')
    snap = Snapshot(output, args.run_root, args.allow_external_replay)
    source_paths = {k: snap.source(getattr(args, k)) for k in ('render_state', 'annotations', 'hawor_npz', 'interaction_judgments')}
    core = layers_load_json(source_paths['render_state'])
    ann = layers_load_json(source_paths['annotations'])
    judgments = layers_load_json(source_paths['interaction_judgments'])
    require(core.get('method') == 'build_v19_rigid_render_state', 'required component: P19 rigid render state')
    require('mano_constraint_state' in core and 'constraint_rows' in core['mano_constraint_state'], 'required component: P16 constraints')
    require(core.get('temporal_mano_state', {}).get('payload') is not None, 'required component: P18/P18b temporal MANO payload')
    rows = judgments.get('interaction_judgments', judgments.get('judgments', judgments.get('segments')))
    require(isinstance(rows, list) and len(rows) > 0, 'required component: explicit P17 interaction_judgments/judgments/segments schema')
    require(judgments.get('case') in (None, core.get('case')), 'P17 case mismatch')
    require(judgments.get('status') in (None, 'ok'), 'P17 status is not ok')
    for row in rows:
        require(isinstance(row, dict), 'invalid P17 judgment row')
        require(row.get('hand_side', row.get('side')) in SIDES, 'required component: P17 hand side')
        require(row.get('contact_state') in {'likely_contact', 'possible_contact', 'no_contact', 'unresolved'}, 'required component: P17 contact state')
        require(row.get('occlusion_relation', row.get('visibility_relation')) in {'hand_in_front_of_object', 'object_in_front_of_hand', 'object_partially_occluded_by_hand', 'no_visible_occlusion', 'unresolved'}, 'required component: P17 occlusion relation')
    core_dir = source_paths['render_state'].parent
    annotation_source = snap.source(core['inputs']['annotations'], core_dir)
    require(annotation_source == source_paths['annotations'], 'P19 annotation source mismatch')
    canonical = snap.source(core['object_geometry']['completed_mesh_path'], core_dir)
    hawor_paths = {ref: snap.source(ref, source_paths['annotations'].parent) for ref in annotation_hawor_references(ann)}
    hawor_bindings = {ref: {'asset': p, 'sha256': sha256(p)} for (ref, p) in hawor_paths.items()}
    validate_hawor_bindings(ann, source_paths['hawor_npz'], hawor_bindings, lambda p: p)
    prediction = core_arrays(core, ann, load_npz(source_paths['hawor_npz']), canonical)
    source_paths['completion_report'] = snap.source(core['inputs']['completion_report'], core_dir)
    completion = layers_load_json(source_paths['completion_report'])
    demo_source = demo_report = demo = None
    demo_files = []
    if args.demo_state:
        demo_source = snap.source(args.demo_state)
        demo_report = snap.source(args.demo_report)
        report = layers_load_json(demo_report)
        (demo_mesh_reference, digest) = demo_mesh_contract(core, completion)
        demo_mesh = snap.source(demo_mesh_reference, source_paths['completion_report'].parent) if digest else canonical
        require(sha256(demo_mesh) == (digest or sha256(canonical)), 'demo bound canonical mesh hash mismatch')
        demo = demo_arrays(load_npz(demo_source), prediction, load_geometry_mesh(demo_mesh))
        for (key, digest) in report.get('input_hashes', {}).items():
            value = report.get('inputs', {}).get(key)
            if value is None:
                continue
            p = snap.source(value, demo_report.parent)
            require(sha256(p) == digest, f'demo report input hash mismatch: {key}')
            demo_files.append(p)
        if report.get('mesh'):
            p = snap.source(report['mesh'], demo_report.parent)
            require(report.get('mesh_sha256') == sha256(p), 'demo report mesh hash mismatch')
            require(sha256(p) == sha256(demo_mesh), 'demo report canonical mesh mismatch')
            demo_files.append(p)
        for (key, expected) in (('annotations', source_paths['annotations']), ('hawor_params', source_paths['hawor_npz'])):
            if key in report.get('inputs', {}):
                actual = snap.source(report['inputs'][key], demo_report.parent)
                require(sha256(actual) == sha256(expected), f'demo report source mismatch: {key}')
    core_files = [snap.source(v, core_dir) for (k, v) in core['inputs'].items() if v]
    physical = core['object_geometry'].get('physical_surface', {}).get('mesh')
    if physical:
        core_files.append(snap.source(physical, core_dir))

    def archives(value):
        if isinstance(value, dict):
            for (key, child) in value.items():
                if isinstance(child, str) and value.get(key + '_sha256'):
                    p = snap.source(child, core_dir)
                    require(sha256(p) == value[key + '_sha256'], f'core declared asset hash mismatch: {key}')
                yield from archives(child)
        elif isinstance(value, list):
            for child in value:
                yield from archives(child)
        elif isinstance(value, str) and value.lower().endswith(('.npz', '.ply', '.obj', '.glb', '.stl')):
            yield snap.source(value, core_dir)
    core_files.extend(archives(core))
    core_files.extend(archives(ann.get('v19_inputs', {})))
    core_files.extend(demo_files)
    output.mkdir(parents=True)
    sources = {k: snap.copy(p) for (k, p) in source_paths.items()}
    for p in core_files:
        snap.copy(p)
    mesh_asset = snap.copy(canonical)
    metric = {'render_state': core, 'annotations': ann, 'interaction_judgments': judgments, 'source_assets': sources, 'canonical_mesh_asset': mesh_asset, 'completion_report_binding': {'reference': core['inputs']['completion_report'], 'sha256': sha256(source_paths['completion_report'])}, 'hawor_source_bindings': {ref: {'asset': snap.copy(binding['asset']), 'sha256': binding['sha256']} for (ref, binding) in hawor_bindings.items()}, 'render_arrays_asset': snap.generated('prediction_render_arrays.npz', prediction), 'authority': 'source metric prediction with explicit gate-accepted full MANO when present; no demo promotion', 'full_mano_acceptance': (core.get('temporal_mano_state', {}).get('payload') or {}).get('full_mano_acceptance', {'accepted': False})}
    demo_block = None
    if demo is not None:
        demo_block = {'source_assets': {'state': snap.copy(demo_source), 'report': snap.copy(demo_report)}, 'report': layers_load_json(demo_report), 'canonical_mesh_binding': {'reference': demo_mesh_reference, 'asset': snap.copy(demo_mesh)}, 'render_arrays_asset': snap.generated('demo_render_arrays.npz', demo), 'authority': 'visual hypothesis only; not contact/collision/sign evidence', 'annotation_ready': False, 'collision_eligible': False}
    timeline = []
    for (i, frame) in enumerate(ann['frames']):
        fid = int(prediction['frame_idx'][i])
        frame_judgments = []
        for row in rows:
            require(isinstance(row, dict), 'invalid P17 judgment row')
            start = row.get('frame_start', row.get('start_frame', prediction['frame_idx'][0]))
            end = row.get('frame_end', row.get('end_frame', prediction['frame_idx'][-1]))
            require(int(end) >= int(start), 'inverted P17 interval')
            target_id = row.get('target_entity_id', row.get('object_id', row.get('track_id')))
            matches = object_target_matches(frame, core['object_id'], target_id)
            if matches and int(start) <= fid <= int(end):
                frame_judgments.append(row)
        hands = {}
        for side in SIDES:
            raw = next((h for h in frame.get('hands', []) if h.get('hand_side') == side), {})
            hands[side] = {'valid': bool(prediction[f'{side}_valid'][i]), 'visibility_state': raw.get('visibility_state', 'unresolved'), 'occluder': raw.get('occluder', 'unresolved'), 'source_state': raw, 'uncertainty': raw.get('uncertainty', ['source_uncertainty_not_quantified'])}
        obj = resolve_object(frame, core['object_id'])
        timeline.append({'frame_idx': fid, 'time_s': float(prediction['time_s'][i]), 'camera': {'T_world_camera': prediction['T_world_camera'][i].tolist(), 'intrinsics_fx_fy_cx_cy': prediction['intrinsics_render'][i].tolist(), 'source_size_wh': prediction['render_size_wh'][i].tolist()}, 'object': {'pose_valid': bool(prediction['object_valid'][i]), 'visibility_state': obj.get('visibility_state', 'unresolved'), 'occluder': obj.get('occluder', 'unresolved'), 'source_state': obj, 'uncertainty': ['source_physical_uncertainty_preserved'] if prediction['object_valid'][i] else ['missing_object_pose']}, 'hands': hands, 'interaction_judgments': frame_judgments, 'interaction_uncertainty': [] if frame_judgments else ['no_P17_judgment_for_frame']})
    package = {'schema': SCHEMA, 'run_mode': 'external_replay_not_fresh' if args.allow_external_replay else 'fresh_run_root_inputs', 'coordinate_contract': {'world': 'sequence_local_estimated_metric', 'units': 'm', 'camera_axes': 'RDF', 'object': 'V_world = V_completed_canonical @ R.T + t', 'hands': 'source MANO world surfaces; demo uses its own corrected surfaces'}, 'metric_prediction': metric, 'demo_visual': demo_block, 'timeline': timeline, 'assets': snap.assets, 'source_asset_map': snap.by_source}
    target = output / 'annotation_layers.json'
    target.write_text(json.dumps(package, indent=2, allow_nan=False) + '\n')
    load_package(target)
    return target

def formal_layers_cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('render-state', 'annotations', 'hawor-npz', 'interaction-judgments', 'output-dir', 'run-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--demo-state', type=Path)
    p.add_argument('--demo-report', type=Path)
    p.add_argument('--allow-external-replay', action='store_true')
    print(assemble_formal_layers(p.parse_args(argv)))
import subprocess
import numpy as np
import cv2

def shade(scene, K, size, faces, normals, labels=None, background=(239, 241, 245)):
    import open3d as o3d
    (w, h) = size
    (yy, xx) = np.mgrid[:h, :w]
    rays = np.stack(((xx - K[2]) / K[0], (yy - K[3]) / K[1], np.ones_like(xx)), axis=-1).astype('float32')
    q = np.concatenate((np.zeros_like(rays), rays), axis=-1)
    hit = scene.cast_rays(o3d.core.Tensor(q), nthreads=4)
    depth = hit['t_hit'].numpy()
    ids = hit['geometry_ids'].numpy()
    normal = hit['primitive_normals'].numpy()
    mask = np.isfinite(depth)
    face_ids = hit['primitive_ids'].numpy()
    bary = hit['primitive_uvs'].numpy()
    if labels is not None and len(labels) != len(faces):
        raise ValueError('semantic labels must match BVH geometry order')
    for geom in range(len(faces)):
        if labels is None and geom == 0 or (labels is not None and labels[geom] not in (1, 2)):
            continue
        valid = mask & (ids == geom)
        tris = faces[geom][face_ids[valid]]
        b = bary[valid]
        weights = np.column_stack([1 - b.sum(1), b])
        smooth = (normals[geom][tris] * weights[:, :, None]).sum(1)
        normal[valid] = smooth / np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-08)
    if labels is not None:
        raw_ids = ids.copy()
        for (geom, label) in enumerate(labels):
            ids[raw_ids == geom] = label
    colors = np.array([[80, 188, 183], [235, 183, 158], [242, 200, 168]], float)
    light = np.array([-0.35, -0.6, -1.0])
    light /= np.linalg.norm(light)
    n = normal.copy()
    facing = (n * rays).sum(-1) > 0
    n[facing] *= -1
    intensity = 0.45 + 0.5 * np.maximum(0, (n * light).sum(-1)) + 0.05 * np.maximum(0, -n[:, :, 2])
    im = np.empty((h, w, 3), np.uint8)
    im[:] = background
    for (i, c) in enumerate(colors):
        im[mask & (ids == i)] = np.clip(c * intensity[mask & (ids == i), None], 0, 255).astype('uint8')
    edges = cv2.Canny(mask.astype('uint8') * 255, 100, 200) > 0
    im[edges] = np.clip(im[edges].astype(float) * 0.75, 0, 255).astype('uint8')
    return (im, mask, ids, depth)

def encode_process(path, size, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{size[0]}x{size[1]}', '-framerate', str(fps), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p' if size[0] % 2 == 0 and size[1] % 2 == 0 else 'yuv444p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
import argparse
from pathlib import Path

def render_formal_rrd(package_path, output, layer='both', display_geometry=None):
    import rerun as rr
    import rerun.blueprint as rrb
    require(layer in ('prediction', 'demo', 'both'), 'invalid layer selection')
    (package, layers) = load_package(package_path)
    selected = list(layers) if layer == 'both' else [layer]
    require(all((s in layers for s in selected)), 'required component: demo layer not supplied')
    output = Path(output).resolve()
    require(not output.exists(), f'RRD output must be new: {output}')
    require(not output.is_relative_to(Path(package_path).resolve().parent / 'assets'), 'RRD may not modify snapshot assets')
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init('v19_annotation_layers', spawn=False)
    rr.save(str(output))
    rr.log('/world', rr.ViewCoordinates.RDF, static=True)
    rr.log('/world/authority', rr.TextDocument(package['run_mode'] + '\nPrediction: source metric state. Demo: visual hypothesis, not collision/contact authority.'), static=True)
    p = layers['prediction']
    if display_geometry is not None:
        name = 'display_' + display_geometry['backend']
        layers = {name: display_backend_state(p, display_geometry, package_path)}
        selected = [name]
        rr.log('/world/authority', rr.TextDocument('Shared physical pose/MANO; generated object geometry is display-only, no signed/contact authority'), static=True)
    rr.log('/world/camera_trajectory', rr.LineStrips3D([p['T_world_camera'][:, :3, 3]], colors=[150, 150, 150]), static=True)
    for (i, fid) in enumerate(p['frame_idx']):
        rr.set_time('frame', sequence=int(fid))
        rr.set_time('time_s', duration=float(p['time_s'][i]))
        T = p['T_world_camera'][i]
        K = p['intrinsics_render'][i]
        rr.log('/world/camera', rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log('/world/camera/image', rr.Pinhole(image_from_camera=[[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], resolution=p['render_size_wh'][i]))
        rr.log('/world/camera/timestamp', rr.Scalars(float(p['time_s'][i])))
        for name in selected:
            state = layers[name]
            root = '/world/' + name
            if state['object_valid'][i]:
                rr.log(root + '/object', rr.Transform3D(translation=state['object_translation_world'][i], mat3x3=state['object_rotation'][i]))
                if i == 0 or not state['object_valid'][i - 1]:
                    rr.log(root + '/object/mesh', rr.Mesh3D(vertex_positions=state['object_vertices_canonical'], triangle_indices=state['object_faces'], albedo_factor=[80, 170, 240, 255] if name == 'prediction' else [110, 220, 110, 255]))
                    if i > 0:
                        rr.log(root + '/object/status', rr.Clear(recursive=True))
            else:
                rr.log(root + '/object', rr.Clear(recursive=True))
                rr.log(root + '/object/status', rr.TextLog('missing object pose; unresolved'))
            for side in SIDES:
                entity = root + '/hands/' + side
                if state[f'{side}_valid'][i]:
                    rr.log(entity, rr.Mesh3D(vertex_positions=state[f'{side}_vertices_world'][i], triangle_indices=state[f'{side}_faces'], albedo_factor=[235, 183, 158, 255]))
                    rr.log(entity + '/joints', rr.Points3D(state[f'{side}_joints_world'][i], radii=0.002))
                else:
                    rr.log(entity, rr.Clear(recursive=True))
            if name == 'prediction':
                import json
                rr.log(root + '/interaction_state', rr.TextDocument(json.dumps(package['timeline'][i])))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*[rrb.Spatial3DView(origin='/world', name=name, contents=[f'+ /world/{name}/**', '+ /world/camera/**', '+ /world/camera_trajectory']) for name in selected])))
    rr.get_global_data_recording().flush()
    rr.disconnect()
    require(output.is_file() and output.stat().st_size > 0, 'RRD writer produced no data')
    return output

def formal_rrd_cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', choices=['prediction', 'demo', 'both'], default='both')
    a = p.parse_args(argv)
    print(render_formal_rrd(a.package, a.output, a.layer))
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from utils.geometry import camera
from utils.geometry import resized_K
from utils.data import read_cfr_video_timing
from utils.data import validate_package_cfr_times

def scene_for_layer(state, index):
    import open3d as o3d
    'Full triangle BVH; omit only explicitly invalid entities, never faces.'
    T = state['T_world_camera'][index]
    pairs = []
    if state['object_valid'][index]:
        world = state['object_vertices_canonical'] @ state['object_rotation'][index].T + state['object_translation_world'][index]
        pairs.append((world, state['object_faces'], 0))
    for (si, side) in enumerate(('left', 'right')):
        if state[side + '_valid'][index]:
            pairs.append((state[side + '_vertices_world'][index], state[side + '_faces'], si + 1))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    (faces, normals, labels) = ([], [], [])
    for (world, topology, label) in pairs:
        vertices = camera(world, T)
        scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(topology.astype(np.uint32)))
        triangle = vertices[topology]
        fn = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        vn = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(vn, topology[:, corner], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-08)
        faces.append(topology)
        normals.append(vn)
        labels.append(label)
    return (scene, faces, normals, labels)

def header(image, text):
    output = image.copy()
    scale = max(0.25, min(0.7, output.shape[1] / 1000))
    cv2.rectangle(output, (0, 0), (output.shape[1], max(18, int(45 * scale))), (25, 30, 40), -1)
    cv2.putText(output, text, (8, max(13, int(30 * scale))), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)
    return output

def render_formal_video(package_path, source_video, output_dir, width=640, expected_video_sha256=None, display_geometry=None):
    (package, layers) = load_package(package_path)
    out = Path(output_dir).resolve()
    require(not out.exists(), 'new video output directory required')
    require(width >= 16, 'render width must be >=16')
    require(not out.is_relative_to(Path(package_path).resolve().parent / 'assets'), 'video output must not alter snapshot assets')
    source_video = Path(source_video).resolve(strict=True)
    video_hash = sha256(source_video)
    if expected_video_sha256:
        require(video_hash == expected_video_sha256, 'original video hash mismatch')
    pred = layers['prediction']
    if display_geometry is not None:
        layers = {'display_' + display_geometry['backend']: display_backend_state(pred, display_geometry, package_path)}
    n = len(pred['frame_idx'])
    source_timing = read_cfr_video_timing(source_video, expected_frame_count=n)
    fps = source_timing.fps
    source_wh = np.asarray(source_timing.source_size_wh)
    require(np.all(pred['render_size_wh'] == source_wh), 'video/annotation source dimensions mismatch')
    times = pred['time_s']
    package_grid_error = validate_package_cfr_times(times, source_timing.frame_rate)
    video = cv2.VideoCapture(str(source_video))
    require(video.isOpened(), 'cannot open original RGB video')
    size = np.array([width, max(1, round(source_wh[1] * width / source_wh[0]))])
    out.mkdir(parents=True)
    writer = encode_process(out / 'annotation_layers.mp4', (int(size[0]) * (len(layers) + 1), int(size[1])), source_timing.frame_rate)
    rows = []
    selected_stills = set(np.linspace(0, n - 1, min(n, 10), dtype=int).tolist())
    try:
        for (index, frame_id) in enumerate(pred['frame_idx']):
            (ok, bgr) = video.read()
            require(ok, f'video decode failed at sequence index {index}')
            require(bgr.shape[:2] == (int(source_wh[1]), int(source_wh[0])), 'source video dimensions changed during decoding')
            rgb = cv2.cvtColor(cv2.resize(bgr, tuple(size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            panels = [header(rgb, 'Original RGB')]
            row = {'frame_idx': int(frame_id), 'time_s': float(times[index]), 'layers': {}}
            for (name, state) in layers.items():
                (scene, faces, normals, labels) = scene_for_layer(state, index)
                K = resized_K(state['intrinsics_render'][index], state['render_size_wh'][index], size)
                if labels:
                    (shaded, mask, geometry_ids, _) = shade(scene, K, tuple(size), faces, normals, labels)
                else:
                    (shaded, mask, geometry_ids) = (np.zeros_like(rgb), np.zeros(rgb.shape[:2], bool), np.full(rgb.shape[:2], -1))
                overlay = rgb.copy()
                overlay[mask] = (0.25 * rgb[mask] + 0.75 * shaded[mask]).astype(np.uint8)
                panels.append(header(overlay, 'Metric prediction (estimated)' if name == 'prediction' else 'Generated display (shared physical state)' if name.startswith('display_') else 'Demo visual (fitted hypothesis)'))
                row['layers'][name] = {'object_valid': bool(state['object_valid'][index]), 'left_valid': bool(state['left_valid'][index]), 'right_valid': bool(state['right_valid'][index]), 'pixels_by_object_left_right': [int(np.count_nonzero(mask & (geometry_ids == k))) for k in range(3)], 'triangle_count': sum((len(f) for f in faces)), 'faces_removed': 0}
            image = np.concatenate(panels, axis=1)
            writer.stdin.write(np.ascontiguousarray(image).tobytes())
            if index in selected_stills:
                cv2.imwrite(str(out / f'frame_{int(frame_id):06d}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append(row)
            if index % 25 == 0:
                print(f'rendered {index + 1}/{n}', flush=True)
        (extra_frame, _) = video.read()
        require(not extra_frame, 'video decoder returned frames beyond its presentation timeline')
    finally:
        video.release()
        writer.stdin.close()
        require(writer.wait() == 0, 'ffmpeg encoding failed')
    encoded_timing = read_cfr_video_timing(out / 'annotation_layers.mp4', expected_frame_count=n)
    require(encoded_timing.frame_rate == source_timing.frame_rate, 'encoder changed source CFR frame rate')
    for (row, source_pts, output_pts) in zip(rows, source_timing.presentation_times_s, encoded_timing.presentation_times_s):
        row['source_video_pts_s'] = source_pts
        row['output_video_pts_s'] = output_pts
    report = {'schema': 'v19_final_package_video_v1', 'package': str(Path(package_path).resolve()), 'package_sha256': sha256(package_path), 'source_video': str(source_video), 'source_video_sha256': video_hash, 'source_video_identity_bound': bool(expected_video_sha256), 'run_mode': package['run_mode'], 'frame_count': len(rows), 'fps': fps, 'duration_s': len(rows) / fps, 'panel_size_wh': size.tolist(), 'layers': list(layers), 'geometry_source': 'same normalized package arrays used by world RRD; no fitting or state mixing', 'rendering': 'full triangle first-hit BVH; no display face deletion', 'per_frame': rows, 'visual_acceptance': 'pending_manual_review'}
    report['timing'] = {'mapping': 'one-to-one CFR frame order; time-origin offsets explicit, no VFR resampling', 'package_time_origin_s': float(times[0]), 'package_max_cfr_grid_error_s': package_grid_error, 'source': source_timing.report(), 'encoded_output': encoded_timing.report()}
    (out / 'render_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return out / 'annotation_layers.mp4'

def formal_video_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'source-video', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--render-width', type=int, default=640)
    parser.add_argument('--source-video-sha256', required=True, help='Original authorized RGB bytes; not an inferred video identity')
    args = parser.parse_args(argv)
    print(render_formal_video(args.package, args.source_video, args.output_dir, args.render_width, args.source_video_sha256))

def display_backend_state(shared, geometry, package_path):
    """Only complete object geometry branches; camera, poses, MANO and validity stay shared."""
    require(geometry.get('backend') in {'sam3d', 'trellis'}, 'unknown retained geometry backend')
    require(geometry.get('package_sha256') == sha256(package_path), 'geometry belongs to another shared state')
    require(geometry.get('canonical_frame') == 'shared_completed_canonical_meters', 'backend mesh gauge is unbound')
    require(geometry.get('role') == 'generated_display_only', 'generated backend has no physical force authority')
    path = Path(geometry['mesh'])
    require(sha256(path) == geometry['mesh_sha256'], 'backend mesh bytes changed')
    mesh = load_geometry_mesh(path)
    return {**shared, 'object_vertices_canonical': mesh['object_vertices_canonical'], 'object_faces': mesh['object_faces']}

def render_dual_backend(package_path, source_video, geometries, output):
    """Publish two full display branches after one already-frozen shared physical solve."""
    output = Path(output)
    require(not output.exists(), 'dual output must be new')
    require({g['backend'] for g in geometries} == {'sam3d', 'trellis'} and len(geometries) == 2, 'dual display requires exactly SAM3D and TRELLIS')
    (package, states) = load_package(package_path)
    for geometry in geometries:
        display_backend_state(states['prediction'], geometry, package_path)
    output.mkdir(parents=True)
    shared_digest = hashlib.sha256()
    for (name, value) in sorted(states['prediction'].items()):
        if name not in {'object_vertices_canonical', 'object_faces'}:
            shared_digest.update(name.encode())
            shared_digest.update(np.ascontiguousarray(value).tobytes())
    results = []
    for geometry in geometries:
        directory = output / geometry['backend']
        directory.mkdir()
        render_formal_rrd(package_path, directory / 'world.rrd', layer='prediction', display_geometry=geometry)
        video = render_formal_video(package_path, source_video, directory / 'video', expected_video_sha256=sha256(source_video), display_geometry=geometry)
        results.append({'backend': geometry['backend'], 'geometry': geometry, 'rrd_sha256': sha256(directory / 'world.rrd'), 'video_sha256': sha256(video), 'shared_state_sha256': shared_digest.hexdigest()})
    manifest = {'schema': 'dual_backend_shared_state_display_v1', 'package_sha256': sha256(package_path), 'shared_state_sha256': shared_digest.hexdigest(), 'shared_physics_recomputed': False, 'generated_geometry_authority': 'display_only', 'branches': results, 'physical_acceptance': package['metric_prediction'].get('full_mano_acceptance', {'accepted': False})}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest
