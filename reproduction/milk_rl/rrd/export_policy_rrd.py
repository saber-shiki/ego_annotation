"""Read-only policy NPZ -> self-contained RRD, independent of legacy acceptance."""
from pathlib import Path
import hashlib
import io
import json
import sys
import uuid

SOURCE = Path('/mnt/user-home/kupingxin/ego_annotation_runtime/rl_milk_minfix_v1_20260918')
sys.path.insert(0, str(SOURCE))
import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
from envs.qualification.output import timeline
from utils.geometry import resized_K


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def export(state_path, raw_manifest, output, label):
    state_path, raw_manifest, output = map(Path, (state_path, raw_manifest, output))
    if output.exists():
        raise FileExistsError(output)
    source_hash, raw_hash = sha(state_path), sha(raw_manifest)
    with np.load(state_path, allow_pickle=False) as archive:
        z = dict(archive)
    ids, times, order, T, K, size, valid = timeline(z)
    object_valid = valid & z['object_pose_valid']
    raw_rows = json.loads(raw_manifest.read_text())['frames']
    raw = {int(row['frame_idx']): row for row in raw_rows}
    if len(raw) != len(raw_rows) or set(raw) != set(map(int, ids)):
        raise ValueError('raw/state frame identity mismatch')
    image_bytes, image_sizes = {}, {}
    for i, idx in enumerate(ids):
        row = raw[int(idx)]
        if abs(float(row['time_s']) - float(times[i])) > 1e-6:
            raise ValueError('raw/state time mismatch')
        data = Path(row['raw_frame_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['raw_frame_sha256']:
            raise ValueError('raw image hash mismatch')
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            image_sizes[int(idx)] = im.size
        if image_sizes[int(idx)] != (row['manifest_width'], row['manifest_height']):
            raise ValueError('RGB image plane mismatch')
        image_bytes[int(idx)] = data
    for side in ['left', 'right']:
        mask = z[f'{side}_display_valid']
        if mask.shape != ids.shape or mask.dtype != bool or np.any(mask & ~z[f'{side}_valid']):
            raise ValueError('invalid hand display mask')
    metadata = {
        'schema': 'ego.annotation.policy_npz_rrd_export.v1', 'label': label,
        'source_npz_sha256': source_hash, 'raw_manifest_sha256': raw_hash,
        'status': 'policy_candidate' if label != 'zero' else 'zero_action_candidate',
        'annotation_ready': False, 'physical_contact_authority': False,
        'physical_acceptance': False, 'units': 'meters',
        'transforms': 'parent-from-child; NPZ object/camera matrices map canonical/camera coordinates to world',
        'camera_axes': 'RDF', 'world_axes': 'source world unchanged; no gravity alignment asserted',
        'display': 'NPZ validity used exactly; false entries cleared; this is not independently inferred visibility',
        'image_plane': 'original saved JPEG embedded; K resized with project half-pixel convention',
        'geometry': 'complete NPZ object triangles and full MANO surfaces, no decimation or solver substitution',
        'rgb_role': 'separate source camera image; not generated mesh texture',
        'frame_ids': ids.tolist(), 'times_s': times.tolist(),
        'object_entity_ids': z['object_entity_ids'].tolist(),
        'rerun_version': rr.__version__,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + '.partial')
    if pending.exists():
        raise FileExistsError(pending)
    recording = rr.RecordingStream('milk_rl_' + label, recording_id=str(uuid.uuid4()))
    recording.save(str(pending))
    mesh_present = False
    mesh_emissions, clear_ids, image_count, hand_counts = [], [], 0, {'left': 0, 'right': 0}
    try:
        recording.log('/metadata', rr.TextDocument(json.dumps(metadata, indent=2)), static=True)
        recording.log('/world/camera_trajectory', rr.LineStrips3D([T[order, :3, 3]], colors=[120, 170, 250], radii=.0007), static=True)
        recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(name=label + ' / 3D candidate', origin='/world'),
            rrb.Spatial2DView(name='Source RGB / camera', origin='/world/camera/image', contents=['/world/**']),
        )))
        for i in order:
            idx = int(ids[i])
            recording.set_time('frame', sequence=idx)
            recording.set_time('time_s', duration=float(times[i]))
            if object_valid[i]:
                recording.log('/world/object', rr.Transform3D(translation=z['object_translation_world'][i], mat3x3=z['object_rotation'][i]))
                if not mesh_present:
                    recording.log('/world/object', rr.Mesh3D(vertex_positions=z['object_vertices_canonical'], triangle_indices=z['object_faces'], albedo_factor=[80, 188, 183, 255]))
                    mesh_emissions.append(idx)
                mesh_present = True
            else:
                recording.log('/world/object', rr.Clear(recursive=True))
                clear_ids.append(idx)
                mesh_present = False
            for side in ['left', 'right']:
                entity = '/world/hands/' + side
                if z[f'{side}_display_valid'][i]:
                    recording.log(entity, rr.Mesh3D(vertex_positions=z[f'{side}_vertices_world'][i], triangle_indices=z[f'{side}_faces'], albedo_factor=[235, 183, 158, 255]))
                    hand_counts[side] += 1
                else:
                    recording.log(entity, rr.Clear(recursive=True))
            recording.log('/world/camera', rr.Transform3D(translation=T[i, :3, 3], mat3x3=T[i, :3, :3]))
            k = resized_K(K[i], size, image_sizes[idx])
            recording.log('/world/camera/image', rr.Pinhole(image_from_camera=[[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], resolution=image_sizes[idx], camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=.15))
            recording.log('/world/camera/image', rr.EncodedImage(contents=image_bytes[idx], media_type='image/jpeg'))
            image_count += 1
        recording.flush()
    finally:
        recording.disconnect()
    if sha(state_path) != source_hash or sha(raw_manifest) != raw_hash:
        raise RuntimeError('source changed during export')
    pending.rename(output)
    receipt = {
        **metadata, 'source_npz': str(state_path), 'raw_manifest': str(raw_manifest),
        'output': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size,
        'frame_count': len(ids), 'rgb_count': image_count, 'hand_mesh_frames': hand_counts,
        'object_pose_frames': int(object_valid.sum()), 'object_clear_frame_ids': clear_ids,
        'object_mesh_emitted_frame_ids': mesh_emissions,
        'object_mesh_vertices': len(z['object_vertices_canonical']), 'object_mesh_faces': len(z['object_faces']),
        'mano_vertices_per_hand': int(z['left_vertices_world'].shape[1]),
        'mano_faces_per_hand': {s: len(z[s+'_faces']) for s in hand_counts},
        'inputs_unchanged': True,
    }
    with output.with_suffix('.receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    return receipt


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--raw-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    print(json.dumps(export(a.state, a.raw_manifest, a.output, a.label), indent=2))
