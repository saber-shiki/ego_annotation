#!/usr/bin/env python3
"""Actual triangle/z-buffer demo renderer for fitted MANO and intact SAM3D.

Main video: original vs image-aligned shaded reconstruction overlay.
Detail video: raw crop vs surface-only yawed 3D inspection (+20 degrees by default).
Use --detail-yaw-deg 0 for a camera-aligned geometry comparison.
"""
from pathlib import Path
import argparse, json, subprocess, time
import numpy as np
import cv2
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from v20_demo_geometry import camera, project, sha
from v19_demo_timeline import hand_validity, state_timeline


def scene_for_frame(z, idx, base=False, yaw_deg=0.0, valid=None):
    prefix = "base_" if base else ""
    V = z["object_vertices_canonical"]
    F = z["object_faces"]
    R = z[prefix + "object_rotation"][idx]
    t = z[prefix + "object_translation_world"][idx]
    T = z["T_world_camera"][idx]
    vertices = [camera(V @ R.T + t, T)]
    faces = [F]
    labels = [0]
    if valid is None:
        valid = {side: hand_validity(z, side, len(z["frame_idx"]), fitted=True) for side in ["left", "right"]}
    for si, side in enumerate(["left", "right"]):
        if not valid[side][idx]:
            continue
        labels.append(si + 1)
        key = side + ("_base_vertices_world" if base else "_vertices_world")
        vertices.append(camera(z[key][idx], T))
        faces.append(z[side + "_faces"])
    if yaw_deg:
        pivot = np.mean(vertices[0], axis=0)
        view_R = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
        vertices = [(v - pivot) @ view_R.T + pivot for v in vertices]
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    for v, f in zip(vertices, faces):
        scene.add_triangles(
            o3d.core.Tensor(v.astype("float32")), o3d.core.Tensor(f.astype("uint32"))
        )
    normals = []
    for v, f in zip(vertices, faces):
        triangle = v[f]
        fn = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        vn = np.zeros_like(v)
        for col in range(3):
            np.add.at(vn, f[:, col], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
        normals.append(vn)
    return scene, vertices, faces, normals, labels


def shade(scene, K, size, faces, normals, labels=None, background=(239, 241, 245)):
    w, h = size
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack(
        ((xx - K[2]) / K[0], (yy - K[3]) / K[1], np.ones_like(xx)), axis=-1
    ).astype("float32")
    q = np.concatenate((np.zeros_like(rays), rays), axis=-1)
    hit = scene.cast_rays(o3d.core.Tensor(q), nthreads=4)
    depth = hit["t_hit"].numpy()
    ids = hit["geometry_ids"].numpy()
    normal = hit["primitive_normals"].numpy()
    mask = np.isfinite(depth)
    face_ids = hit["primitive_ids"].numpy()
    bary = hit["primitive_uvs"].numpy()
    if labels is not None and len(labels) != len(faces):
        raise ValueError("semantic labels must match BVH geometry order")
    for geom in range(len(faces)):
        # Package scenes may have no valid object, so geometry zero can be a
        # hand. Smooth by semantic role, retaining the old index convention
        # only for legacy callers which do not provide labels.
        if (labels is None and geom == 0) or (labels is not None and labels[geom] not in (1, 2)):
            continue
        valid = mask & (ids == geom)
        tris = faces[geom][face_ids[valid]]
        b = bary[valid]
        weights = np.column_stack([1 - b.sum(1), b])
        smooth = (normals[geom][tris] * weights[:, :, None]).sum(1)
        normal[valid] = smooth / np.maximum(
            np.linalg.norm(smooth, axis=1, keepdims=True), 1e-8
        )
    if labels is not None:
        raw_ids = ids.copy()
        for geom, label in enumerate(labels):
            ids[raw_ids == geom] = label
    # Neutral stylization; no raw image is pasted into the mesh appearance.
    colors = np.array([[80, 188, 183], [235, 183, 158], [242, 200, 168]], float)
    light = np.array([-0.35, -0.6, -1.0])
    light /= np.linalg.norm(light)
    n = normal.copy()
    facing = (n * rays).sum(-1) > 0
    n[facing] *= -1
    intensity = (
        0.45
        + 0.5 * np.maximum(0, (n * light).sum(-1))
        + 0.05 * np.maximum(0, -n[:, :, 2])
    )
    im = np.empty((h, w, 3), np.uint8)
    im[:] = background
    for i, c in enumerate(colors):
        im[mask & (ids == i)] = np.clip(
            c * intensity[mask & (ids == i), None], 0, 255
        ).astype("uint8")
    # Fine silhouettes, not every triangle edge, keep contact detail legible.
    edges = cv2.Canny(mask.astype("uint8") * 255, 100, 200) > 0
    im[edges] = np.clip(im[edges].astype(float) * 0.75, 0, 255).astype("uint8")
    return im, mask, ids, depth


def encode_process(path, size, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{size[0]}x{size[1]}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p" if size[0] % 2 == 0 and size[1] % 2 == 0 else "yuv444p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def banner(img, left, right):
    im = img.copy()
    scale = min(1.0, im.shape[0] / 960)
    bar_height = max(1, round(49 * scale))
    cv2.rectangle(im, (0, 0), (im.shape[1], bar_height), (25, 30, 40), -1)
    cv2.putText(
        im,
        left,
        (max(1, round(20 * scale)), max(1, round(31 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67 * scale,
        (240, 240, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        im,
        right,
        (im.shape[1] // 2 + max(1, round(20 * scale)), max(1, round(31 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67 * scale,
        (240, 240, 245),
        1,
        cv2.LINE_AA,
    )
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--source-video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frames", default="all")
    p.add_argument("--base", action="store_true")
    p.add_argument("--detail-yaw-deg", type=float, default=20.0)
    p.add_argument("--render-width", type=int, help="Preserve state image aspect ratio")
    a = p.parse_args()
    out = a.output_dir
    if out.exists():
        raise ValueError("new output directory required")
    out.mkdir(parents=True)
    with np.load(a.state, allow_pickle=False) as archive:
        z = dict(archive)
    video = cv2.VideoCapture(str(a.source_video))
    fps = video.get(cv2.CAP_PROP_FPS)
    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ids, times, order, render_wh, intrinsics = state_timeline(z, fps, n, a.render_width)
    W, H = map(int, render_wh)
    if "source_size_wh" in z and not np.array_equal(z["source_size_wh"], [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))]):
        raise ValueError("source video dimensions disagree with state")
    id_to_index = {int(f): i for i, f in enumerate(ids)}
    video_position = {int(i): pos for pos, i in enumerate(order)}
    selected = order if a.frames == "all" else np.array([id_to_index[int(f)] for f in a.frames.split(",")])
    valid = {side: hand_validity(z, side, n, fitted=True) for side in ["left", "right"]}
    full = a.frames == "all"
    mp4 = encode_process(out / "interaction_demo.mp4", (2 * W, H), fps) if full else None
    detail = encode_process(out / "contact_detail.mp4", (2 * W, H), fps) if full else None
    scene_centers = []
    scene_sizes = []
    for i in range(n):
        pts = np.concatenate(
            [z[s + ("_base_vertices_world" if a.base else "_vertices_world")][i] for s in ["left", "right"] if valid[s][i]]
            + [
                z["object_vertices_canonical"][::max(1, len(z["object_vertices_canonical"]) // 5000)] @ z[("base_" if a.base else "") + "object_rotation"][i].T
                + z[("base_" if a.base else "") + "object_translation_world"][i]
            ]
        )
        uv = project(camera(pts, z["T_world_camera"][i]), intrinsics[i])
        lo = np.percentile(uv, 1, axis=0)
        hi = np.percentile(uv, 99, axis=0)
        scene_centers.append((lo + hi) / 2)
        scene_sizes.append(max((hi - lo)[0], (hi - lo)[1] * W / H) * 1.18)
    centers = np.empty((n, 2))
    sizes = np.empty(n)
    centers[order] = gaussian_filter1d(np.asarray(scene_centers)[order], 4, axis=0)
    sizes[order] = np.clip(gaussian_filter1d(np.asarray(scene_sizes)[order], 4), min(W, H) * 0.385, max(W, H) * 0.834)
    if full:
        rr.init("V20_visual_hand_object_demo", spawn=False)
        rr.save(str(out / "interaction_demo.rrd"))
        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        rr.log(
            "/metadata",
            rr.TextDocument(
                "Visual-fit demo: real SAM3D mesh and MANO surfaces. Contact is fitted hypothesis, not measured physical ground truth. All front faces retained."
            ),
            static=True,
        )
        rr.log(
            "/world/object",
            rr.Mesh3D(
                vertex_positions=z["object_vertices_canonical"].astype("float32"),
                triangle_indices=z["object_faces"].astype("int32"),
                albedo_factor=[80, 188, 183, 255],
            ),
            static=True,
        )
    stats = []
    start = time.monotonic()
    try:
        for idx in selected:
            idx = int(idx)
            frame_id = int(ids[idx])
            video.set(cv2.CAP_PROP_POS_FRAMES, video_position[idx])
            ok, bgr = video.read()
            if not ok:
                raise ValueError("source frame decode failure")
            rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)
            K = intrinsics[idx]
            scene, vertices, faces, normals, labels = scene_for_frame(z, idx, base=a.base, valid=valid)
            shaded, mask, gids, depth = shade(scene, K, (W, H), faces, normals, labels)
            overlay = rgb.copy()
            alpha = np.where(gids == 0, 0.55, 0.68)
            overlay[mask] = (
                rgb[mask] * (1 - alpha[mask, None]) + shaded[mask] * alpha[mask, None]
            ).astype("uint8")
            pair = banner(
                np.hstack((rgb, overlay)),
                f"Original video  |  frame {frame_id}",
                "Visual interaction fit  |  demo-only",
            )
            size = float(sizes[idx])
            cx, cy = centers[idx]
            x0 = cx - size / 2
            y0 = cy - size * H / W / 2
            ratio = W / size
            zoomK = np.r_[K[:2] * ratio, (K[2:] - [x0, y0] + 0.5) * ratio - 0.5]
            inspect_scene, _, inspect_faces, inspect_normals, inspect_labels = scene_for_frame(
                z, idx, base=a.base, yaw_deg=a.detail_yaw_deg, valid=valid
            )
            solid, _, _, _ = shade(
                inspect_scene, zoomK, (W, H), inspect_faces, inspect_normals, inspect_labels
            )
            affine = np.array(
                [
                    [ratio, 0, (-x0 + 0.5) * ratio - 0.5],
                    [0, ratio, (-y0 + 0.5) * ratio - 0.5],
                ],
                np.float32,
            )
            crop = cv2.warpAffine(
                rgb,
                affine,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(239, 241, 245),
            )
            closepair = banner(
                np.hstack((crop, solid)),
                "Original grasp detail",
                f"3D inspection ({a.detail_yaw_deg:+g} deg) | "
                + ("before fit" if a.base else "fitted contact hypothesis"),
            )
            if full:
                mp4.stdin.write(pair.tobytes())
                detail.stdin.write(closepair.tobytes())
                T = z["T_world_camera"][idx]
                rr.set_time("frame", sequence=frame_id)
                rr.set_time("time_s", duration=float(times[idx]))
                prefix = "base_" if a.base else ""
                R = z[prefix + "object_rotation"][idx]
                t = z[prefix + "object_translation_world"][idx]
                rr.log("/world/object", rr.Transform3D(translation=t, mat3x3=R))
                for si, side in enumerate(["left", "right"]):
                    if not valid[side][idx]:
                        rr.log("/world/hands/" + side, rr.Clear(recursive=True))
                        continue
                    key = side + (
                        "_base_vertices_world" if a.base else "_vertices_world"
                    )
                    rr.log(
                        "/world/hands/" + side,
                        rr.Mesh3D(
                            vertex_positions=z[key][idx],
                            triangle_indices=z[side + "_faces"].astype("int32"),
                            albedo_factor=[[235, 183, 158, 255], [242, 200, 168, 255]][
                                si
                            ],
                        ),
                    )
                rr.log(
                    "/world/camera",
                    rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
                )
                rr.log(
                    "/world/camera/image",
                    rr.Pinhole(
                        image_from_camera=[[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]],
                        resolution=[W, H],
                    ),
                )
                rr.log("/world/camera/image", rr.Image(rgb).compress(jpeg_quality=85))
                rr.log("/comparison/original", rr.Image(rgb).compress(jpeg_quality=85))
                rr.log(
                    "/comparison/overlay", rr.Image(overlay).compress(jpeg_quality=85)
                )
                rr.log(
                    "/comparison/detail", rr.Image(closepair).compress(jpeg_quality=85)
                )
            if not full or video_position[idx] in np.linspace(0, n - 1, min(n, 10), dtype=int):
                cv2.imwrite(
                    str(out / f"frame_{frame_id:03d}_overlay.png"),
                    cv2.cvtColor(pair, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(out / f"frame_{frame_id:03d}_detail.png"),
                    cv2.cvtColor(closepair, cv2.COLOR_RGB2BGR),
                )
            stats.append(
                {
                    "frame": frame_id,
                    "time_s": float(times[idx]),
                    "left_valid": bool(valid["left"][idx]),
                    "right_valid": bool(valid["right"][idx]),
                    "object_pixels": int(np.count_nonzero(gids == 0)),
                    "left_pixels": int(np.count_nonzero(gids == 1)),
                    "right_pixels": int(np.count_nonzero(gids == 2)),
                    "faces_removed": 0,
                }
            )
            if idx % 25 == 0 or idx == selected[-1]:
                print(
                    "render",
                    idx,
                    "elapsed",
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
    finally:
        video.release()
        for proc in [mp4, detail]:
            if proc is not None:
                proc.stdin.close()
                code = proc.wait()
                if code:
                    raise RuntimeError(f"ffmpeg failed {code}")
        if full:
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            origin="/world", name="Actual 3D hands + object"
                        ),
                        rrb.Vertical(
                            rrb.Spatial2DView(
                                origin="/comparison/original", name="Original"
                            ),
                            rrb.Spatial2DView(
                                origin="/comparison/overlay", name="Visual fit"
                            ),
                            rrb.Spatial2DView(
                                origin="/comparison/detail", name="Grasp detail"
                            ),
                        ),
                    ),
                    collapse_panels=False,
                )
            )
            rr.disconnect()
    (out / "render_report.json").write_text(
        json.dumps(
            {
                "schema": "v20_visual_contact_demo_render_v1",
                "demo_only": True,
                "annotation_ready": False,
                "gt_consumed": False,
                "source_video": str(a.source_video.resolve()),
                "source_video_sha256": sha(a.source_video),
                "state": str(a.state.resolve()),
                "state_sha256": sha(a.state),
                "full_duration": bool(full),
                "frame_count": len(selected),
                "fps": fps,
                "render_size_wh": [W, H],
                "missing_hand_policy": "unresolved; no geometry logged; rr.Clear",
                "elapsed_s": time.monotonic() - start,
                "object_triangle_count": len(z["object_faces"]),
                "mano_triangle_counts": {
                    s: len(z[s + "_faces"]) if s + "_faces" in z else 0 for s in ["left", "right"]
                },
                "rendering": "full original triangle BVH z-buffer; no point-cloud gate or front-face deletion",
                "base_mode": a.base,
                "detail_view_yaw_deg": a.detail_yaw_deg,
                "per_frame": stats,
            },
            indent=2,
        )
        + "\n"
    )
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
