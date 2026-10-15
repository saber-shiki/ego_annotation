"""Actual complete mesh/MANO, common A/B/C raster and fixed-world rendering."""
from pathlib import Path
import subprocess
import cv2
import numpy as np

from preprocessing.geometry import TriangleMesh, camera_pose, resized_intrinsics
from preprocessing.render import paint, look_at
from preprocessing.mask_contract import boundary
from preprocessing.surface_pipeline import write_json


def render_comparison(inputs, states, output, width=320):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    names = ('A', 'B', 'C'); mesh = TriangleMesh(states['A']['vertices_canonical'], states['A']['faces'])
    wh = (width, round(width*inputs.wh[1]/inputs.wh[0])); fps = 1/float(np.median(np.diff(inputs.times)))
    if wh[1] % 2:
        wh = (wh[0], wh[1]+1)
    trajectories = np.concatenate([states[k]['object_translation_world'] for k in names])
    center = np.median(trajectories, axis=0); C0 = inputs.cameras[0]
    radius = max(.2, float(np.max(np.linalg.norm(trajectories-center, axis=1)))+.12)
    view = look_at(center+C0[:3, :3]@np.array([radius*.7, -radius*.25, -radius*2.5]), center, C0[:3, :3]@np.array([0., -1., 0.]))
    kv = np.array([width*1.05, width*1.05, (width-1)/2, (width-1)/2])
    files = [(output/'ABC_overlay.mp4', (3*width, wh[1]+40)), (output/'ABC_world.mp4', (3*width, width+40))]
    pipes = []
    for path, size in files:
        pipes.append(subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-threads', '1', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE))
    labels = {'A': 'A: frozen R6', 'B': 'B: observed motion/surface', 'C': 'C: B + soft envelope'}
    try:
        for i, fid in enumerate(inputs.ids):
            raw = cv2.imread(inputs.raw_paths[i])
            if raw is None:
                raise FileNotFoundError(inputs.raw_paths[i])
            rgb = cv2.resize(raw, wh, interpolation=cv2.INTER_AREA)
            k = resized_intrinsics(inputs.K[i], inputs.wh, wh); C = inputs.cameras[i]
            green = boundary(cv2.resize(inputs.masks[i].astype('uint8'), wh, interpolation=cv2.INTER_NEAREST_EXACT)>0)
            hands = []
            for side, color in [('left', (225, 150, 75)), ('right', (95, 185, 225))]:
                hand = inputs.hands[side]
                if not hand['valid'][i]:
                    continue
                faces = hand['faces'][i] if hand['faces'].ndim == 3 else hand['faces']
                hs = TriangleMesh(hand['vertices'][i], faces)
                _, hz, _ = hs.raster(k, C[:3, :3].T, -C[:3, :3].T@C[:3, 3], wh)
                _, wz, _ = hs.raster(kv, view[:3, :3].T, -view[:3, :3].T@view[:3, 3], (width, width))
                hands.append((hz, wz, color))
            overlays, worlds = [], []
            for name in names:
                state = states[name]; R, t = state['object_rotation'][i], state['object_translation_world'][i]
                rc, tc = camera_pose(R, t, C); _, z, _ = mesh.raster(k, rc, tc, wh)
                im = rgb.copy(); depth = np.full(wh[::-1], np.inf); paint(im, depth, z, (240, 160, 40), .55)
                rc, tc = camera_pose(R, t, view); _, z, _ = mesh.raster(kv, rc, tc, (width, width))
                world = np.full((width, width, 3), 242, np.uint8); wd = np.full((width, width), np.inf)
                paint(world, wd, z, (90, 180, 245), .95)
                for hz, wz, color in hands:
                    paint(im, depth, hz, color, .45); paint(world, wd, wz, color, .95)
                im[green] = (40, 235, 40)
                header = np.zeros((40, width, 3), np.uint8)
                cv2.putText(header, f'f{fid} {labels[name]}', (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .39, (255, 255, 255), 1)
                cv2.putText(header, 'full mesh + source MANO; no GT', (5, 33), cv2.FONT_HERSHEY_SIMPLEX, .33, (220, 220, 220), 1)
                overlays.append(np.vstack((header, im))); worlds.append(np.vstack((header, world)))
            overlay = np.concatenate(overlays, axis=1); world = np.concatenate(worlds, axis=1)
            pipes[0].stdin.write(overlay.tobytes()); pipes[1].stdin.write(world.tobytes())
            # All frames for timeline review; not a substitute for complete videos.
            cv2.imwrite(str(output/f'overlay_{int(fid):06d}.jpg'), overlay)
            if i % 30 == 0 or i in (inputs.anchor, len(inputs.ids)-1, 143, 148):
                cv2.imwrite(str(output/f'world_{int(fid):06d}.jpg'), world)
    finally:
        for pipe in pipes:
            if pipe.stdin:
                pipe.stdin.close()
        codes = [p.wait() for p in pipes]
    if any(codes):
        raise RuntimeError('FFmpeg render failed: '+str(codes))
    decoded = {}
    for path, _ in files:
        cap = cv2.VideoCapture(str(path)); count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
        decoded[path.name] = count; cap.release()
        if count != len(inputs.ids):
            raise RuntimeError('Full-video decode mismatch: '+str(path))
    write_json(output/'render.json', {'frames': len(inputs.ids), 'fps': fps, 'duration_s': len(inputs.ids)/fps,
               'decoded_frames': decoded, 'mesh_vertices': len(mesh.vertices), 'mesh_faces': len(mesh.faces),
               'source_MANO_unchanged': True, 'common_fixed_world_view': view.tolist(),
               'rasters': {'overlay_panel': wh, 'world_panel': [width, width]}})
