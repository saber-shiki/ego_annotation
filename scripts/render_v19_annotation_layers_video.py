#!/usr/bin/env python3
"""Render full-duration prediction/demo MP4 panels from the SAME final package as RRD.

Only the original RGB video is supplied separately. The package's normalized
world geometry is authoritative for all visible meshes; no refitting occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from v19_annotation_layers import load_package, require, sha256
from v20_demo_geometry import camera, resized_K
from render_v20_visual_contact_demo import encode_process, shade


def scene_for_layer(state, index):
    """Full triangle BVH; omit only explicitly invalid entities, never faces."""
    T = state['T_world_camera'][index]
    pairs = []
    if state['object_valid'][index]:
        world = state['object_vertices_canonical'] @ state['object_rotation'][index].T + state['object_translation_world'][index]
        pairs.append((world, state['object_faces'], 0))
    for si, side in enumerate(('left', 'right')):
        if state[side + '_valid'][index]:
            pairs.append((state[side + '_vertices_world'][index], state[side + '_faces'], si + 1))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    faces, normals, labels = [], [], []
    for world, topology, label in pairs:
        vertices = camera(world, T)
        scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(topology.astype(np.uint32)))
        triangle = vertices[topology]
        fn = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        vn = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(vn, topology[:, corner], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
        faces.append(topology)
        normals.append(vn)
        labels.append(label)
    return scene, faces, normals, labels


def header(image, text):
    output = image.copy()
    scale = max(0.25, min(0.7, output.shape[1] / 1000))
    cv2.rectangle(output, (0, 0), (output.shape[1], max(18, int(45 * scale))), (25, 30, 40), -1)
    cv2.putText(output, text, (8, max(13, int(30 * scale))), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)
    return output


def render(package_path, source_video, output_dir, width=640, expected_video_sha256=None):
    package, layers = load_package(package_path)
    out = Path(output_dir).resolve()
    require(not out.exists(), 'new video output directory required')
    require(width >= 16, 'render width must be >=16')
    require(not out.is_relative_to(Path(package_path).resolve().parent / 'assets'), 'video output must not alter snapshot assets')
    source_video = Path(source_video).resolve(strict=True)
    video_hash = sha256(source_video)
    if expected_video_sha256:
        require(video_hash == expected_video_sha256, 'original video hash mismatch')
    video = cv2.VideoCapture(str(source_video))
    require(video.isOpened(), 'cannot open original RGB video')
    pred = layers['prediction']
    n = len(pred['frame_idx'])
    fps = video.get(cv2.CAP_PROP_FPS)
    source_wh = np.array([int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    require(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == n, 'video/package frame count mismatch')
    require(np.isfinite(fps) and fps > 0, 'invalid source video FPS')
    require(np.all(pred['render_size_wh'] == source_wh), 'video/annotation source dimensions mismatch')
    # This writer emits constant-rate MP4. Never silently resample an unknown
    # variable-rate timeline or seek by arbitrary annotation frame IDs.
    times = pred['time_s']
    require(np.allclose(times - times[0], np.arange(n) / fps, atol=0.5 / fps, rtol=0), 'video/package constant-rate timeline mismatch')
    size = np.array([width, max(1, round(source_wh[1] * width / source_wh[0]))])
    out.mkdir(parents=True)
    writer = encode_process(out / 'annotation_layers.mp4', (int(size[0]) * (len(layers) + 1), int(size[1])), fps)
    rows = []
    selected_stills = set(np.linspace(0, n - 1, min(n, 10), dtype=int).tolist())
    try:
        for index, frame_id in enumerate(pred['frame_idx']):
            ok, bgr = video.read()
            require(ok, f'video decode failed at sequence index {index}')
            rgb = cv2.cvtColor(cv2.resize(bgr, tuple(size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            panels = [header(rgb, 'Original RGB')]
            row = {'frame_idx': int(frame_id), 'time_s': float(times[index]), 'layers': {}}
            for name, state in layers.items():
                scene, faces, normals, labels = scene_for_layer(state, index)
                K = resized_K(state['intrinsics_render'][index], state['render_size_wh'][index], size)
                if labels:
                    shaded, mask, geometry_ids, _ = shade(scene, K, tuple(size), faces, normals, labels)
                else:
                    shaded, mask, geometry_ids = np.zeros_like(rgb), np.zeros(rgb.shape[:2], bool), np.full(rgb.shape[:2], -1)
                overlay = rgb.copy()
                overlay[mask] = (0.25 * rgb[mask] + 0.75 * shaded[mask]).astype(np.uint8)
                panels.append(header(overlay, 'Metric prediction (estimated)' if name == 'prediction' else 'Demo visual (fitted hypothesis)'))
                row['layers'][name] = {'object_valid': bool(state['object_valid'][index]), 'left_valid': bool(state['left_valid'][index]), 'right_valid': bool(state['right_valid'][index]), 'pixels_by_object_left_right': [int(np.count_nonzero(mask & (geometry_ids == k))) for k in range(3)], 'triangle_count': sum(len(f) for f in faces), 'faces_removed': 0}
            image = np.concatenate(panels, axis=1)
            writer.stdin.write(np.ascontiguousarray(image).tobytes())
            if index in selected_stills:
                cv2.imwrite(str(out / f'frame_{int(frame_id):06d}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append(row)
            if index % 25 == 0:
                print(f'rendered {index + 1}/{n}', flush=True)
    finally:
        video.release()
        writer.stdin.close()
        require(writer.wait() == 0, 'ffmpeg encoding failed')
    report = {'schema': 'v19_final_package_video_v1', 'package': str(Path(package_path).resolve()), 'package_sha256': sha256(package_path), 'source_video': str(source_video), 'source_video_sha256': video_hash, 'source_video_identity_bound': bool(expected_video_sha256), 'run_mode': package['run_mode'], 'frame_count': len(rows), 'fps': fps, 'duration_s': len(rows) / fps, 'panel_size_wh': size.tolist(), 'layers': list(layers), 'geometry_source': 'same normalized package arrays used by world RRD; no fitting or state mixing', 'rendering': 'full triangle first-hit BVH; no display face deletion', 'per_frame': rows, 'visual_acceptance': 'pending_manual_review'}
    (out / 'render_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return out / 'annotation_layers.mp4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'source-video', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--render-width', type=int, default=640)
    parser.add_argument('--source-video-sha256', required=True, help='Original authorized RGB bytes; not an inferred video identity')
    args = parser.parse_args()
    print(render(args.package, args.source_video, args.output_dir, args.render_width, args.source_video_sha256))


if __name__ == '__main__':
    main()
