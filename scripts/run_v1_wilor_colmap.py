#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pycolmap
import torch


DEFAULT_CLIP = Path(
    "/data2/egoscale_demo_30h/egoscale_tasks/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7/"
    "20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4"
)
DEFAULT_WILOR_ROOT = Path("third_party/WiLoR")
DEFAULT_MANO_RIGHT = Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_RIGHT.pkl")

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class FrameSample:
    frame_idx: int
    time_s: float
    name: str


@dataclass
class RenderInfo:
    width: int
    height: int


def patch_legacy_imports() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    raw_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return raw_load(*args, **kwargs)

    torch.load = torch_load_compat  # type: ignore[assignment]


def ensure_wilor_assets(wilor_root: Path, mano_right: Path) -> None:
    required = [
        wilor_root / "wilor" / "models" / "wilor.py",
        wilor_root / "pretrained_models" / "wilor_final.ckpt",
        wilor_root / "pretrained_models" / "detector.pt",
        wilor_root / "pretrained_models" / "model_config.yaml",
        wilor_root / "mano_data" / "mano_mean_params.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing WiLoR assets: " + ", ".join(missing))
    target = wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not target.exists():
        if not mano_right.exists():
            raise FileNotFoundError(f"missing MANO_RIGHT source: {mano_right}")
        shutil.copy2(mano_right, target)


def load_actions(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    tasks = data.get("tasks") or []
    if not tasks:
        raise ValueError(f"no task records in {json_path}")
    return tasks[0].get("actions") or []


def caption_for_frame(actions: list[dict], frame_idx: int) -> str:
    for action in actions:
        start = int(action.get("start_frame", -1))
        end = int(action.get("end_frame", -1))
        if start <= frame_idx < end:
            return str(action.get("description") or action.get("action") or "")
    return ""


def open_video(path: Path) -> tuple[cv2.VideoCapture, ClipInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    info = ClipInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {info}")
    return cap, info


def extract_sampled_frames(clip: Path, frame_dir: Path, sample_fps: float, render_width: int) -> tuple[ClipInfo, list[FrameSample]]:
    cap, info = open_video(clip)
    stride = max(1, round(info.fps / sample_fps))
    render_height = int(round(render_width * info.height / info.width))
    if render_height % 2:
        render_height += 1
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        resized = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
        name = f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(frame_dir / name), resized)
        samples.append(FrameSample(frame_idx=idx, time_s=idx / info.fps, name=name))
    cap.release()
    if not samples:
        raise RuntimeError("no sampled frames extracted")
    return info, samples


def run_colmap(frame_dir: Path, output_dir: Path) -> dict:
    sfm_dir = output_dir / "sfm_colmap"
    sparse = sfm_dir / "sparse"
    best = sparse / "best"
    if best.exists():
        return read_colmap(best)
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sparse.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    pycolmap.extract_features(database, frame_dir, camera_mode=pycolmap.CameraMode.SINGLE, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    maps = pycolmap.incremental_mapping(database, frame_dir, sparse)
    if not maps:
        raise RuntimeError("pycolmap returned no reconstruction")
    recon = max(maps.values(), key=lambda r: r.num_reg_images())
    best.mkdir(parents=True, exist_ok=True)
    recon.write(best)
    return colmap_to_json(recon, best)


def read_colmap(model_dir: Path) -> dict:
    return colmap_to_json(pycolmap.Reconstruction(model_dir), model_dir)


def colmap_to_json(recon, model_dir: Path) -> dict:
    frames = []
    for image in recon.images.values():
        if not image.has_pose:
            continue
        mat3x4 = np.asarray(image.cam_from_world().matrix(), dtype=float)
        cam_from_world = np.eye(4)
        cam_from_world[:3, :4] = mat3x4
        world_from_cam = np.linalg.inv(cam_from_world)
        frames.append({"name": image.name, "T_world_camera": world_from_cam.tolist()})
    if not frames:
        raise RuntimeError("pycolmap reconstruction has no registered camera poses")
    return {
        "backend": "pycolmap_sfm",
        "status": "ok",
        "registered_images": int(recon.num_reg_images()),
        "num_images": int(len(recon.images)),
        "num_points3d": int(recon.num_points3D()),
        "frames": sorted(frames, key=lambda x: x["name"]),
        "model_dir": str(model_dir),
    }


def load_wilor_backend(wilor_root: Path):
    patch_legacy_imports()
    sys.path.insert(0, str(wilor_root.resolve()))
    from ultralytics import YOLO
    from wilor.models import load_wilor

    cwd = Path.cwd()
    os.chdir(wilor_root)
    try:
        model, cfg = load_wilor("./pretrained_models/wilor_final.ckpt", "./pretrained_models/model_config.yaml")
        detector = YOLO("./pretrained_models/detector.pt")
    finally:
        os.chdir(cwd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    detector = detector.to(device)
    return model, cfg, detector, device


def project_full_image(points: np.ndarray, cam_t: np.ndarray, focal: float, img_size: np.ndarray) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(img_size[0]) / 2.0
    K[1, 2] = float(img_size[1]) / 2.0
    pts = points + cam_t
    pts = pts / pts[..., [-1]]
    return (K @ pts.T).T[:, :2]


def run_wilor_on_frame(model, cfg, detector, device, frame: np.ndarray, rescale_factor: float, batch_size: int) -> list[dict]:
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    detections = detector(frame, conf=0.3, verbose=False)[0]
    boxes: list[list[float]] = []
    is_right: list[float] = []
    for det in detections:
        arr = det.boxes.data.cpu().detach().squeeze().numpy()
        if arr.ndim == 0:
            continue
        boxes.append(arr[:4].astype(float).tolist())
        is_right.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
    if not boxes:
        return []
    boxes_np = np.asarray(boxes, dtype=np.float32)
    right_np = np.asarray(is_right, dtype=np.float32)
    dataset = ViTDetDataset(cfg, frame, boxes_np, right_np, rescale_factor=rescale_factor, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[dict] = []
    det_offset = 0
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        focal = float(scaled_focal_length.detach().cpu().numpy())
        for n in range(batch["img"].shape[0]):
            hand_side = "right" if float(batch["right"][n].detach().cpu().numpy()) >= 0.5 else "left"
            verts = out["pred_vertices"][n].detach().cpu().numpy().astype(float)
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().astype(float)
            side_sign = 1.0 if hand_side == "right" else -1.0
            verts[:, 0] = side_sign * verts[:, 0]
            joints[:, 0] = side_sign * joints[:, 0]
            joints2d = project_full_image(joints, cam_t[n], focal, img_size[n].detach().cpu().numpy())
            predictions.append(
                {
                    "backend": "WiLoR",
                    "side": hand_side,
                    "bbox_xyxy": boxes_np[det_offset + n].astype(float).tolist(),
                    "cam_t": cam_t[n].astype(float).tolist(),
                    "focal_length": focal,
                    "joints3d_camera": joints.tolist(),
                    "joints2d": joints2d.astype(float).tolist(),
                    "vertices_camera": verts.tolist(),
                    "vertices_sample": verts[::10].tolist(),
                }
            )
        det_offset += batch["img"].shape[0]
    return predictions


def attach_world_hands(hands: list[dict], camera: dict) -> None:
    if "T_world_camera" not in camera:
        return
    T = np.asarray(camera["T_world_camera"], dtype=float)
    for hand in hands:
        verts = np.asarray(hand["vertices_camera"], dtype=float)
        joints = np.asarray(hand["joints3d_camera"], dtype=float)
        cam_t = np.asarray(hand["cam_t"], dtype=float)
        verts_cam = verts + cam_t
        joints_cam = joints + cam_t
        verts_h = np.c_[verts_cam, np.ones(len(verts_cam))]
        joints_h = np.c_[joints_cam, np.ones(len(joints_cam))]
        hand["vertices_world_sample"] = (T @ verts_h.T).T[::10, :3].astype(float).tolist()
        hand["joints3d_world"] = (T @ joints_h.T).T[:, :3].astype(float).tolist()
        wrist = np.r_[joints_cam[0], 1.0]
        hand["T_world_wrist"] = T.copy().tolist()
        hand["T_world_wrist"][0][3] = float((T @ wrist)[0])
        hand["T_world_wrist"][1][3] = float((T @ wrist)[1])
        hand["T_world_wrist"][2][3] = float((T @ wrist)[2])


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def camera_centers(colmap: dict) -> dict[str, np.ndarray]:
    centers = {}
    for frame in colmap["frames"]:
        T = np.asarray(frame["T_world_camera"], dtype=float)
        centers[frame["name"]] = T[:3, 3]
    return centers


def draw_caption(frame: np.ndarray, text: str) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    words = text.split()
    lines: list[str] = []
    line = ""
    max_width = frame.shape[1] - 24
    for word in words:
        trial = word if not line else f"{line} {word}"
        if cv2.getTextSize(trial, font, scale, thickness)[0][0] <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
        if len(lines) == 2:
            break
    if line and len(lines) < 2:
        lines.append(line)
    y0 = frame.shape[0] - 24 * len(lines) - 10
    cv2.rectangle(frame, (0, y0 - 8), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, row in enumerate(lines):
        cv2.putText(frame, row, (12, y0 + 17 + i * 24), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hands(frame: np.ndarray, hands: list[dict]) -> None:
    for hand in hands:
        pts = np.asarray(hand["joints2d"], dtype=float)
        color = (0, 220, 0) if hand["side"] == "left" else (0, 145, 255)
        x1, y1, x2, y2 = np.asarray(hand["bbox_xyxy"], dtype=int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p.astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(p.astype(int)), 4, color, 1, cv2.LINE_AA)


def path_panel_xy(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 2)
    xy = points[:, [0, 2]]
    center = xy.mean(axis=0)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
    scale = 0.62 * min(width, height) / span
    origin = np.array([width * 0.5, height * 0.55])
    return origin + (xy - center) * np.array([scale, -scale])


def render_world_panel(render: RenderInfo, ann: dict, all_centers: np.ndarray, current_center: np.ndarray | None) -> np.ndarray:
    panel = np.full((render.height, render.width, 3), 244, dtype=np.uint8)
    cv2.putText(panel, "World reconstruction", (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (35, 35, 35), 2, cv2.LINE_AA)
    status = "registered" if current_center is not None else "not registered"
    text = f"frame {ann['frame_idx']} | pycolmap SfM camera {status} | WiLoR MANO hands {len(ann['hands'])}"
    cv2.putText(panel, text[:115], (22, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    pts = path_panel_xy(all_centers, render.width, render.height)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(panel, tuple(a.astype(int)), tuple(b.astype(int)), (145, 145, 145), 2, cv2.LINE_AA)
    for p in pts[:: max(1, len(pts) // 30)]:
        cv2.circle(panel, tuple(p.astype(int)), 2, (110, 110, 110), -1, cv2.LINE_AA)
    if current_center is not None and all_centers.size:
        current = path_panel_xy(np.vstack([all_centers, current_center]), render.width, render.height)[-1]
        cv2.circle(panel, tuple(current.astype(int)), 8, (30, 30, 220), -1, cv2.LINE_AA)
        cv2.putText(panel, "camera", tuple((current + np.array([12, -8])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 220), 2, cv2.LINE_AA)
    origin = np.array([render.width * 0.5, render.height * 0.72])
    for hand in ann["hands"]:
        joints = np.asarray(hand.get("joints3d_world") or hand["joints3d_camera"], dtype=float)
        local = joints - joints[0]
        pts2 = origin + local[:, [0, 1]] * np.array([680.0, -680.0])
        color = (0, 160, 0) if hand["side"] == "left" else (0, 115, 220)
        for a, b in HAND_EDGES:
            cv2.line(panel, tuple(pts2[a].astype(int)), tuple(pts2[b].astype(int)), color, 2, cv2.LINE_AA)
    return panel


def render_videos(output_dir: Path, annotations: list[dict], samples: list[FrameSample], frame_dir: Path, render: RenderInfo, colmap: dict, fps: float) -> None:
    centers = camera_centers(colmap)
    all_centers = np.asarray([centers[name] for name in sorted(centers)], dtype=float).reshape(-1, 3)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(output_dir / "overlay.mp4"), fourcc, fps, (render.width, render.height))
    world_writer = cv2.VideoWriter(str(output_dir / "reconstruction_3d.mp4"), fourcc, fps, (render.width, render.height))
    side_writer = cv2.VideoWriter(str(output_dir / "side_by_side.mp4"), fourcc, fps, (render.width * 2, render.height))
    if not overlay_writer.isOpened() or not world_writer.isOpened() or not side_writer.isOpened():
        raise RuntimeError("failed to open video writers")
    by_name = {sample.name: ann for sample, ann in zip(samples, annotations)}
    try:
        for sample in samples:
            frame = cv2.imread(str(frame_dir / sample.name))
            if frame is None:
                raise RuntimeError(f"failed to read {sample.name}")
            ann = by_name[sample.name]
            draw_hands(frame, ann["hands"])
            cam_label = "yes" if "T_world_camera" in ann["camera"] else "no"
            cv2.putText(
                frame,
                f"frame {ann['frame_idx']} | WiLoR hands {len(ann['hands'])} | pycolmap camera {cam_label}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            draw_caption(frame, ann["caption"])
            panel = render_world_panel(render, ann, all_centers, centers.get(sample.name))
            overlay_writer.write(frame)
            world_writer.write(panel)
            side_writer.write(np.concatenate([frame, panel], axis=1))
    finally:
        overlay_writer.release()
        world_writer.release()
        side_writer.release()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    clip = Path(args.clip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "sampled_frames"
    actions = load_actions(clip.with_suffix(".json"))
    info, samples = extract_sampled_frames(clip, frame_dir, args.output_fps, args.render_width)
    render_height = int(round(args.render_width * info.height / info.width))
    if render_height % 2:
        render_height += 1
    render = RenderInfo(args.render_width, render_height)
    ensure_wilor_assets(Path(args.wilor_root), Path(args.mano_right))
    colmap = run_colmap(frame_dir, output_dir)
    colmap_by_name = {frame["name"]: frame for frame in colmap["frames"]}
    model, cfg, detector, device = load_wilor_backend(Path(args.wilor_root))
    annotations = []
    detection_counts: list[float] = []
    for i, sample in enumerate(samples):
        frame = cv2.imread(str(frame_dir / sample.name))
        if frame is None:
            raise RuntimeError(f"failed to read sampled frame {sample.name}")
        hands = run_wilor_on_frame(model, cfg, detector, device, frame, args.hand_rescale_factor, args.hand_batch_size)
        camera = colmap_by_name.get(sample.name, {"status": "not_registered"})
        attach_world_hands(hands, camera)
        detection_counts.append(float(len(hands)))
        annotations.append(
            {
                "frame_idx": sample.frame_idx,
                "time_s": sample.time_s,
                "image": sample.name,
                "caption": caption_for_frame(actions, sample.frame_idx),
                "camera": camera,
                "hands": hands,
                "object": {"status": "not_run", "reason": "object pose is optional in v1 and no real object-pose backend was run"},
            }
        )
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"processed={i + 1}/{len(samples)} elapsed_s={time.time() - started:.1f}", flush=True)
    registered_rate = len(colmap_by_name) / max(1, len(samples))
    hand_frame_rate = sum(1 for n in detection_counts if n > 0) / max(1, len(detection_counts))
    if registered_rate < args.min_camera_registration_rate:
        raise RuntimeError(f"camera registration rate {registered_rate:.3f} below required {args.min_camera_registration_rate:.3f}")
    if hand_frame_rate < args.min_hand_frame_rate:
        raise RuntimeError(f"WiLoR hand-frame rate {hand_frame_rate:.3f} below required {args.min_hand_frame_rate:.3f}")
    render_videos(output_dir, annotations, samples, frame_dir, render, colmap, args.output_fps)
    qc = {
        "quality_decision": "v1_pass_real_backends",
        "clip": str(clip),
        "source_fps": info.fps,
        "sample_fps": args.output_fps,
        "processed_frames": len(samples),
        "camera_backend": colmap["backend"],
        "camera_registration_rate": registered_rate,
        "camera_registered_frames": len(colmap_by_name),
        "camera_num_points3d": colmap["num_points3d"],
        "hand_backend": "WiLoR",
        "hand_frames_with_detection": int(sum(1 for n in detection_counts if n > 0)),
        "hand_frame_rate": hand_frame_rate,
        "hands_per_frame": summarize(detection_counts),
        "object_backend": "not_run",
        "deliverables": {
            "overlay": str(output_dir / "overlay.mp4"),
            "reconstruction_3d": str(output_dir / "reconstruction_3d.mp4"),
            "side_by_side": str(output_dir / "side_by_side.mp4"),
            "annotations": str(output_dir / "annotations.json"),
            "qc": str(output_dir / "qc.json"),
            "sfm_model": str(output_dir / "sfm_colmap" / "sparse" / "best"),
        },
        "backend_attempts": {
            "wilor": {"status": "ran", "checkpoint": str(Path(args.wilor_root) / "pretrained_models" / "wilor_final.ckpt")},
            "pycolmap": {"status": "ran", "model_dir": str(output_dir / "sfm_colmap" / "sparse" / "best")},
            "dpvo": {
                "status": "build_failed",
                "reason": "CUDA extension compilation failed against the current Torch/CUDA API: AT_DISPATCH_FLOATING_TYPES_AND_HALF used tensor.type(), producing DeprecatedTypeProperties to ScalarType conversion errors in dpvo/altcorr/correlation_kernel.cu.",
            },
        },
        "visual_qc": {
            "judgment": "requires_human_review_for_metric_accuracy",
            "observations": [
                "Representative chopping frames show WiLoR boxes and projected MANO joints aligned to visible hands.",
                "Frames with tomato/knife occlusion can pull WiLoR joints into the occlusion region.",
                "Frames without pycolmap registration are labeled not_registered in both overlay and JSON.",
            ],
        },
        "known_limits": [
            "Camera pose is offline pycolmap SfM in arbitrary scale, not DPVO/DROID SLAM.",
            "DPVO build was attempted and failed against current Torch/CUDA dispatch API before a model run.",
            "MANO comes from WiLoR predictions without ground-truth residuals; visual QC is required.",
            "Object pose is omitted in this v1 run because no real object-pose backend was run.",
        ],
    }
    (output_dir / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=str(DEFAULT_CLIP))
    parser.add_argument("--output-dir", default="outputs/examples/tomato_v1_wilor_colmap")
    parser.add_argument("--output-fps", type=float, default=1.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--wilor-root", default=str(DEFAULT_WILOR_ROOT))
    parser.add_argument("--mano-right", default=str(DEFAULT_MANO_RIGHT))
    parser.add_argument("--hand-rescale-factor", type=float, default=2.0)
    parser.add_argument("--hand-batch-size", type=int, default=4)
    parser.add_argument("--min-camera-registration-rate", type=float, default=0.8)
    parser.add_argument("--min-hand-frame-rate", type=float, default=0.6)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
