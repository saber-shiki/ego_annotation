#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

FULL_K_RECTIFIED_MODE = (
    "explicit_source_full_pinhole_K_consumed_via_affine_centered_hawor_plane_and_slam"
)
LEGACY_CENTER_MODE = "legacy_square_focal_image_center"


def as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def file_info(path: Path, *, hash_file: bool = False) -> dict:
    info = {"path": str(path), "exists": path.exists(), "is_file": path.is_file() if path.exists() else False, "bytes": path.stat().st_size if path.exists() and path.is_file() else None}
    if hash_file and path.exists() and path.is_file():
        info["sha256"] = sha256(path)
    return info


def finite_camera_intrinsics(value: Any, *, label: str) -> np.ndarray:
    intrinsics = np.asarray(value, dtype=np.float64).reshape(-1)
    if (
        intrinsics.shape != (4,)
        or not np.isfinite(intrinsics).all()
        or np.any(intrinsics[:2] <= 0.0)
    ):
        raise RuntimeError(f"{label} must be finite [fx,fy,cx,cy], got {intrinsics}")
    return intrinsics


def requested_camera_intrinsics(args: argparse.Namespace) -> np.ndarray | None:
    raw = getattr(args, "camera_intrinsics", None)
    if raw is None:
        return None
    intrinsics = finite_camera_intrinsics(raw, label="--camera-intrinsics")
    fx, fy, _cx, _cy = intrinsics.tolist()
    # The released HaWoR video path has one square-focal input.  Supporting a
    # rectangular focal here would require changing the upstream model API, not
    # averaging metadata after inference.
    tolerance = max(1.0e-6, max(abs(fx), abs(fy)) * 1.0e-6)
    if abs(fx - fy) > tolerance:
        raise RuntimeError(
            "HaWoR video inference currently requires fx==fy; refusing to replace "
            f"the active camera K with a geometric-mean focal: {intrinsics.tolist()}"
        )
    if args.img_focal is not None and abs(float(args.img_focal) - fx) > tolerance:
        raise RuntimeError(
            f"--img_focal={args.img_focal} disagrees with --camera-intrinsics fx={fx}"
        )
    args.img_focal = float(fx)
    return intrinsics


def K_from_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = finite_camera_intrinsics(
        intrinsics, label="camera intrinsics"
    ).tolist()
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def centered_hawor_plane_contract(
    source_intrinsics: np.ndarray,
    *,
    source_size_wh: tuple[int, int],
    source_video: Path,
) -> dict[str, Any]:
    source = finite_camera_intrinsics(
        source_intrinsics, label="source camera intrinsics"
    )
    width, height = [int(value) for value in source_size_wh]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid source image size {source_size_wh}")
    fx, fy, cx, cy = source.tolist()
    if not (0.0 <= cx < float(width) and 0.0 <= cy < float(height)):
        raise RuntimeError(
            f"source principal point {(cx, cy)} lies outside {width}x{height}"
        )
    inference = np.asarray(
        [fx, fy, width / 2.0, height / 2.0], dtype=np.float64
    )
    A_inference_from_source = np.asarray(
        [
            [1.0, 0.0, inference[2] - cx],
            [0.0, 1.0, inference[3] - cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transformed_K = A_inference_from_source @ K_from_intrinsics(source)
    if not np.allclose(
        transformed_K, K_from_intrinsics(inference), atol=1.0e-9, rtol=0.0
    ):
        raise RuntimeError("centered HaWoR plane affine does not transform source K exactly")
    return {
        "status": "exact_source_to_centered_hawor_image_plane",
        "mode": FULL_K_RECTIFIED_MODE,
        "source_video": str(source_video),
        "source_video_sha256": sha256(source_video),
        "source_size_wh": [width, height],
        "inference_size_wh": [width, height],
        "source_intrinsics_fx_fy_cx_cy": source.tolist(),
        "hawor_inference_intrinsics_fx_fy_cx_cy": inference.tolist(),
        "A_hawor_inference_from_source": A_inference_from_source.tolist(),
        "A_source_from_hawor_inference": np.linalg.inv(
            A_inference_from_source
        ).tolist(),
        "pixel_center_convention": "integer_pixel_centers_opencv",
        "image_resampling": "cv2.warpAffine INTER_LINEAR BORDER_CONSTANT",
        "camera_axes_changed": False,
        "camera_pose_frame_changed": False,
        "claim_scope": (
            "The official source-plane K is converted by an explicit image translation to a same-size "
            "HaWoR inference plane whose principal point is exactly [width/2,height/2]. HaWoR left-hand "
            "flipping, projected hand masks, and masked SLAM consume that centered plane. The inverse "
            "affine binds MANO projections and exported detector boxes back to source RGB pixels."
        ),
    }


def aggregate_file_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def materialize_centered_hawor_input(
    *,
    source_video: Path,
    output_dir: Path,
    source_intrinsics: np.ndarray,
) -> tuple[Path, dict[str, Any]]:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video for HaWoR rectification: {source_video}")
    metadata_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if metadata_width <= 0 or metadata_height <= 0:
        capture.release()
        raise RuntimeError(f"invalid source video dimensions: {source_video}")
    contract = centered_hawor_plane_contract(
        source_intrinsics,
        source_size_wh=(metadata_width, metadata_height),
        source_video=source_video,
    )
    input_root = output_dir / "_hawor_full_K_rectified_input"
    alias = input_root / "active_source_K_centered_hawor_plane.mp4"
    sequence_root = alias.parent / alias.stem
    frame_dir = sequence_root / "extracted_images"
    if input_root.exists() and any(input_root.rglob("*")):
        capture.release()
        raise RuntimeError(f"refusing to reuse non-empty HaWoR rectified input root: {input_root}")
    frame_dir.mkdir(parents=True, exist_ok=False)
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_video, alias)
        alias_mode = "hardlink_source_container_not_decoded_by_hawor"
    except OSError:
        shutil.copy2(source_video, alias)
        alias_mode = "copied_source_container_not_decoded_by_hawor"
    A = np.asarray(contract["A_hawor_inference_from_source"], dtype=np.float64)
    output_frames: list[Path] = []
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (metadata_height, metadata_width):
                raise RuntimeError(
                    f"malformed source frame {decoded} while building centered HaWoR input"
                )
            rectified = cv2.warpAffine(
                frame,
                A[:2],
                (metadata_width, metadata_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            frame_path = frame_dir / f"{decoded:04d}.jpg"
            if not cv2.imwrite(
                str(frame_path), rectified, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"failed to write rectified HaWoR frame: {frame_path}")
            output_frames.append(frame_path)
            decoded += 1
    finally:
        capture.release()
    if decoded <= 0 or (metadata_frames > 0 and decoded != metadata_frames):
        raise RuntimeError(
            f"rectified HaWoR frame count {decoded} != source metadata {metadata_frames}"
        )
    contract.update(
        {
            "source_container_alias": str(alias),
            "source_container_alias_mode": alias_mode,
            "source_container_alias_sha256": sha256(alias),
            "inference_extracted_frames": str(frame_dir),
            "inference_frame_count": decoded,
            "source_metadata_frame_count": metadata_frames,
            "source_fps": metadata_fps,
            "inference_frames_aggregate_sha256": aggregate_file_sha256(output_frames),
            "inference_first_frame_sha256": sha256(output_frames[0]),
            "inference_last_frame_sha256": sha256(output_frames[-1]),
            "source_container_alias_is_not_the_inference_raster": True,
            "detect_track_video_must_reuse_prematerialized_extracted_frames": True,
        }
    )
    contract_path = input_root / "hawor_full_K_image_plane_contract.json"
    contract["path"] = str(contract_path)
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return alias, contract


def transform_boxes_to_source_plane(
    boxes: np.ndarray,
    A_source_from_inference: np.ndarray,
    source_size_wh: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64).copy()
    if values.shape[-1] < 4:
        raise RuntimeError(f"invalid detector box shape {values.shape}")
    x0, y0, x1, y1 = values[:4]
    corners = np.asarray(
        [[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]],
        dtype=np.float64,
    )
    transformed = corners @ np.asarray(A_source_from_inference, dtype=np.float64).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    width, height = source_size_wh
    values[0] = float(np.clip(np.min(transformed[:, 0]), 0.0, width - 1.0))
    values[1] = float(np.clip(np.min(transformed[:, 1]), 0.0, height - 1.0))
    values[2] = float(np.clip(np.max(transformed[:, 0]), 0.0, width - 1.0))
    values[3] = float(np.clip(np.max(transformed[:, 1]), 0.0, height - 1.0))
    return values.astype(np.float32)


def prepare_focal_cache_contract(
    seq_folder: Path,
    start_idx: int,
    end_idx: int,
    img_focal: float | None,
    *,
    camera_intrinsics: np.ndarray | None = None,
    camera_image_plane_contract: dict[str, Any] | None = None,
    force_refresh: bool,
) -> dict:
    """Prevent silent reuse of camera-dependent HaWoR cache products.

    HaWoR stores motion chunks, rendered hand masks, and SLAM outputs under a
    sequence folder keyed by the video pathname. Those artifacts depend on the
    source/inference camera-image-plane contract, although upstream historically
    keyed them by neither focal nor principal point. A V19 rerun with an explicit
    source K therefore requires an exact prior source-K/centered-plane match or
    invalidates every motion/mask/SLAM cache product.
    """
    if img_focal is None:
        return {"enabled": False, "reason": "img_focal_not_explicit"}
    focal = float(img_focal)
    requested_intrinsics = (
        finite_camera_intrinsics(camera_intrinsics, label="requested HaWoR camera intrinsics")
        if camera_intrinsics is not None
        else None
    )
    tracks_dir = seq_folder / f"tracks_{int(start_idx)}_{int(end_idx)}"
    contract_path = seq_folder / "v19_hawor_focal_cache_contract.json"
    focal_artifacts = [
        tracks_dir / "frame_chunks_all.npy",
        tracks_dir / "model_masks.npy",
        seq_folder / "SLAM" / f"hawor_slam_w_scale_{int(start_idx)}_{int(end_idx)}.npz",
    ]
    focal_dirs = [seq_folder / "cam_space"]
    existing = [str(p) for p in focal_artifacts if p.exists()] + [str(p) for p in focal_dirs if p.exists()]
    previous: dict | None = None
    if contract_path.exists():
        try:
            previous = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"invalid HaWoR focal cache contract {contract_path}: {exc}") from exc
    previous_focal = None
    previous_intrinsics = None
    previous_mode = None
    previous_plane_sha256 = None
    if isinstance(previous, dict) and previous.get("img_focal") is not None:
        previous_focal = float(previous["img_focal"])
    if isinstance(previous, dict) and previous.get("camera_intrinsics_fx_fy_cx_cy") is not None:
        previous_intrinsics = finite_camera_intrinsics(
            previous["camera_intrinsics_fx_fy_cx_cy"],
            label=f"previous HaWoR cache K in {contract_path}",
        )
        previous_mode = previous.get("camera_contract_mode")
        previous_plane = previous.get("camera_image_plane_contract")
        if isinstance(previous_plane, dict):
            previous_plane_sha256 = previous_plane.get("sha256")
    requested_mode = (
        str(camera_image_plane_contract.get("mode"))
        if isinstance(camera_image_plane_contract, dict)
        else LEGACY_CENTER_MODE
    )
    requested_plane_sha256 = (
        sha256(Path(str(camera_image_plane_contract["path"])))
        if isinstance(camera_image_plane_contract, dict)
        and camera_image_plane_contract.get("path")
        else None
    )
    focal_compatible = previous_focal is not None and abs(previous_focal - focal) <= max(1.0e-3, abs(focal) * 1.0e-6)
    if requested_intrinsics is not None:
        # A historical focal-only record is intentionally incompatible: it
        # cannot prove which principal point generated cached MANO masks/SLAM.
        compatible = bool(
            focal_compatible
            and previous_intrinsics is not None
            and np.allclose(previous_intrinsics, requested_intrinsics, atol=1.0e-6, rtol=0.0)
            and previous_mode == requested_mode
            and previous_plane_sha256 == requested_plane_sha256
        )
    else:
        compatible = bool(focal_compatible)
    removed: list[str] = []
    if existing and not compatible:
        if not force_refresh:
            raise RuntimeError(
                "HaWoR camera-dependent cache does not match the requested full camera contract. "
                f"seq_folder={seq_folder} requested_focal={focal} previous_focal={previous_focal} "
                f"requested_K={None if requested_intrinsics is None else requested_intrinsics.tolist()} "
                f"previous_K={None if previous_intrinsics is None else previous_intrinsics.tolist()} existing={existing[:6]}. "
                "Use a camera-contract-specific video path/sequence folder or pass --force-focal-cache-refresh."
            )
        for p in focal_artifacts:
            if p.exists():
                p.unlink()
                removed.append(str(p))
        for p in focal_dirs:
            if p.exists():
                shutil.rmtree(p)
                removed.append(str(p))
    payload = {
        "status": "ok",
        "img_focal": focal,
        "camera_intrinsics_fx_fy_cx_cy": (
            requested_intrinsics.tolist() if requested_intrinsics is not None else None
        ),
        "camera_contract_mode": requested_mode,
        "camera_image_plane_contract": (
            {
                "path": camera_image_plane_contract.get("path"),
                "sha256": (
                    sha256(Path(str(camera_image_plane_contract["path"])))
                    if camera_image_plane_contract.get("path")
                    else None
                ),
                "source_intrinsics_fx_fy_cx_cy": camera_image_plane_contract.get(
                    "source_intrinsics_fx_fy_cx_cy"
                ),
                "hawor_inference_intrinsics_fx_fy_cx_cy": camera_image_plane_contract.get(
                    "hawor_inference_intrinsics_fx_fy_cx_cy"
                ),
                "A_hawor_inference_from_source": camera_image_plane_contract.get(
                    "A_hawor_inference_from_source"
                ),
            }
            if isinstance(camera_image_plane_contract, dict)
            else None
        ),
        "seq_folder": str(seq_folder),
        "tracks_range": [int(start_idx), int(end_idx)],
        "focal_dependent_artifacts_seen_before_refresh": existing,
        "removed_for_force_refresh": removed,
        "compatible_previous_contract": bool(compatible),
        "force_refresh": bool(force_refresh),
        "claim_scope": "guards HaWoR motion, projected hand-mask, and SLAM cache reuse under the exact source-K to centered-inference-plane affine contract",
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "path": str(contract_path)}


def call_with_hawor_inference_intrinsics(
    function: Callable[..., Any],
    *,
    intrinsics: np.ndarray | None,
    hawor_video_module: Any,
    args: tuple[Any, ...],
) -> Any:
    """Call upstream HaWoR motion inference with the centered inference-plane K.

    Upstream exposes ``img_center`` to the HAWOR model but its video wrapper
    hard-codes image center. It also builds the projected hand mask with a
    center-principal-point renderer. Patch those two call boundaries only for
    this invocation and restore them immediately afterwards.
    """
    if intrinsics is None:
        return function(*args)
    fx, fy, cx, cy = finite_camera_intrinsics(
        intrinsics, label="HaWoR inference-plane intrinsics"
    ).tolist()
    original_inference = hawor_video_module.HAWOR.inference
    original_create_camera = hawor_video_module.Renderer.create_camera_from_cv

    def inference_with_active_center(self: Any, *call_args: Any, **kwargs: Any) -> Any:
        kwargs["img_focal"] = float(fx)
        kwargs["img_center"] = [float(cx), float(cy)]
        return original_inference(self, *call_args, **kwargs)

    def camera_with_active_K(
        self: Any,
        R: torch.Tensor,
        T: torch.Tensor,
        K: torch.Tensor | None = None,
        image_size: torch.Tensor | None = None,
    ) -> Any:
        if K is None:
            K = self.K.clone()
            K[..., 0, 0] = float(fx)
            K[..., 1, 1] = float(fy)
            K[..., 0, 2] = float(cx)
            K[..., 1, 2] = float(cy)
        return original_create_camera(self, R, T, K=K, image_size=image_size)

    hawor_video_module.HAWOR.inference = inference_with_active_center
    hawor_video_module.Renderer.create_camera_from_cv = camera_with_active_K
    try:
        return function(*args)
    finally:
        hawor_video_module.HAWOR.inference = original_inference
        hawor_video_module.Renderer.create_camera_from_cv = original_create_camera


def call_slam_with_hawor_inference_intrinsics(
    function: Callable[..., Any],
    *,
    intrinsics: np.ndarray | None,
    hawor_slam_module: Any,
    args: tuple[Any, ...],
) -> Any:
    if intrinsics is None:
        return function(*args)
    exact = finite_camera_intrinsics(intrinsics, label="HaWoR SLAM inference-plane intrinsics")
    original_est_calib = hawor_slam_module.est_calib
    hawor_slam_module.est_calib = lambda _images: exact.astype(float).tolist()
    try:
        return function(*args)
    finally:
        hawor_slam_module.est_calib = original_est_calib


def load_track_support(seq_folder: Path, start_idx: int, end_idx: int, frame_count: int) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Return same-frame HaWoR detection support by side.

    HaWoR's infiller can produce MANO rows where there is no same-frame detector
    box. The exported metric MANO arrays are still useful evidence, but V18 must
    not conflate infilled rows with detector-supported rows. This function keeps
    that provenance in the NPZ/QC contract.
    """
    tracks_path = seq_folder / f"tracks_{start_idx}_{end_idx}" / "model_tracks.npy"
    support: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        support[side] = {
            "detected_same_frame": np.zeros(frame_count, dtype=np.uint8),
            "det_box_xyxyscore": np.full((frame_count, 5), np.nan, dtype=np.float32),
            "track_id": np.full(frame_count, "", dtype="<U64"),
        }
    report = {
        "tracks_path": str(tracks_path),
        "tracks_file_exists": tracks_path.exists(),
        "side_handedness_mapping": {"left": 0, "right": 1},
        "records_read": 0,
        "records_used": 0,
    }
    if not tracks_path.exists():
        report["status"] = "tracks_file_missing_detection_support_unavailable"
        return support, report
    try:
        tracks_obj = np.load(tracks_path, allow_pickle=True)
        tracks = tracks_obj.item() if getattr(tracks_obj, "shape", None) == () else tracks_obj
    except Exception as exc:  # pragma: no cover - defensive runtime provenance
        report["status"] = "tracks_file_load_failed_detection_support_unavailable"
        report["error"] = repr(exc)
        return support, report
    if not isinstance(tracks, dict):
        report["status"] = "tracks_file_not_dict_detection_support_unavailable"
        report["tracks_object_type"] = str(type(tracks))
        return support, report
    for track_id, records in tracks.items():
        for rec in records:
            report["records_read"] += 1
            try:
                f = int(rec.get("frame", -1))
                if f < 0 or f >= frame_count:
                    continue
                handed_arr = np.asarray(rec.get("det_handedness"), dtype=np.float32).reshape(-1)
                if handed_arr.size == 0:
                    continue
                handed = int(round(float(handed_arr[0])))
                side = "left" if handed == 0 else "right" if handed == 1 else None
                if side is None:
                    continue
                box_arr = np.asarray(rec.get("det_box"), dtype=np.float32).reshape(-1)
                if box_arr.size < 5 or not np.isfinite(box_arr[:5]).all():
                    continue
                current = support[side]["det_box_xyxyscore"][f]
                if support[side]["detected_same_frame"][f] == 0 or float(box_arr[4]) > float(current[4]):
                    support[side]["detected_same_frame"][f] = 1
                    support[side]["det_box_xyxyscore"][f] = box_arr[:5]
                    support[side]["track_id"][f] = str(track_id)
                    report["records_used"] += 1
            except Exception:
                continue
    report["status"] = "ok"
    report["detected_same_frame_counts"] = {side: int(np.count_nonzero(support[side]["detected_same_frame"])) for side in support}
    return support, report


def run(args: argparse.Namespace) -> dict:
    hawor_root = args.hawor_root.resolve()
    video_path_obj = Path(args.video_path).expanduser()
    if not video_path_obj.is_absolute():
        video_path_obj = (Path.cwd() / video_path_obj).resolve()
    checkpoint_path = resolve_path(args.checkpoint, base=hawor_root)
    infiller_path = resolve_path(args.infiller_weight, base=hawor_root)
    model_config_path = resolve_path(args.model_config, base=hawor_root)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.video_path = str(video_path_obj)
    args.checkpoint = str(checkpoint_path)
    args.infiller_weight = str(infiller_path)
    args.model_config = str(model_config_path)
    sys.path.insert(0, str(hawor_root))
    os.chdir(hawor_root)

    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left  # type: ignore
    from lib.eval_utils.custom_utils import load_slam_cam  # type: ignore
    from scripts.scripts_test_video.detect_track_video import detect_track_video  # type: ignore
    from scripts.scripts_test_video import hawor_slam as hawor_slam_module  # type: ignore
    from scripts.scripts_test_video import hawor_video as hawor_video_module  # type: ignore

    requested_intrinsics = requested_camera_intrinsics(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_sha256 = sha256(video_path_obj) if video_path_obj.exists() and video_path_obj.is_file() else None
    if requested_intrinsics is not None:
        inference_video_path, camera_image_plane_contract = materialize_centered_hawor_input(
            source_video=video_path_obj,
            output_dir=args.output_dir,
            source_intrinsics=requested_intrinsics,
        )
        args.video_path = str(inference_video_path)
        source_intrinsics = requested_intrinsics
        inference_intrinsics = np.asarray(
            camera_image_plane_contract[
                "hawor_inference_intrinsics_fx_fy_cx_cy"
            ],
            dtype=np.float64,
        )
    else:
        inference_video_path = video_path_obj
        camera_image_plane_contract = None
        args.video_path = str(video_path_obj)
        source_intrinsics = None
        inference_intrinsics = None
    export_provenance = {
        "hawor_root": str(hawor_root),
        "source_video": file_info(video_path_obj, hash_file=True),
        "hawor_inference_video_alias": file_info(inference_video_path, hash_file=True),
        "camera_image_plane_contract": camera_image_plane_contract,
        "checkpoint": file_info(checkpoint_path, hash_file=True),
        "infiller_weight": file_info(infiller_path, hash_file=True),
        "model_config": file_info(model_config_path, hash_file=True),
    }
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    first_image = cv2.imread(str(imgfiles[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"cannot read first HaWoR inference frame: {imgfiles[0]}")
    image_height, image_width = first_image.shape[:2]
    if camera_image_plane_contract is not None:
        if [image_width, image_height] != camera_image_plane_contract["inference_size_wh"]:
            raise RuntimeError(
                "HaWoR inference frame size differs from the rectified image-plane contract"
            )
        aggregate = aggregate_file_sha256([Path(path) for path in imgfiles])
        if aggregate != camera_image_plane_contract["inference_frames_aggregate_sha256"]:
            raise RuntimeError(
                "HaWoR detect/track inference frames differ from the rectified image-plane contract"
            )
    else:
        inference_intrinsics = np.asarray(
            [
                float(args.img_focal),
                float(args.img_focal),
                image_width / 2.0,
                image_height / 2.0,
            ],
            dtype=np.float64,
        )
        source_intrinsics = inference_intrinsics.copy()
    assert source_intrinsics is not None and inference_intrinsics is not None
    focal_cache_contract = prepare_focal_cache_contract(
        Path(seq_folder),
        int(start_idx),
        int(end_idx),
        args.img_focal,
        camera_intrinsics=source_intrinsics,
        camera_image_plane_contract=camera_image_plane_contract,
        force_refresh=bool(args.force_focal_cache_refresh),
    )
    frame_chunks_all, img_focal = call_with_hawor_inference_intrinsics(
        hawor_video_module.hawor_motion_estimation,
        intrinsics=inference_intrinsics,
        hawor_video_module=hawor_video_module,
        args=(args, start_idx, end_idx, seq_folder),
    )
    if abs(float(img_focal) - float(inference_intrinsics[0])) > 1.0e-5:
        raise RuntimeError(
            f"HaWoR model focal {img_focal} differs from inference-plane K {inference_intrinsics.tolist()}"
        )
    slam_path = Path(seq_folder) / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if not slam_path.exists():
        call_slam_with_hawor_inference_intrinsics(
            hawor_slam_module.hawor_slam,
            intrinsics=inference_intrinsics,
            hawor_slam_module=hawor_slam_module,
            args=(args, start_idx, end_idx),
        )
    if not slam_path.exists():
        raise RuntimeError(f"HaWoR SLAM output missing: {slam_path}")
    with np.load(slam_path, allow_pickle=False) as slam_archive:
        slam_focal = float(np.asarray(slam_archive["img_focal"]).reshape(-1)[0])
        slam_center = np.asarray(slam_archive["img_center"], dtype=np.float64).reshape(-1)
    slam_intrinsics = np.asarray(
        [slam_focal, slam_focal, *slam_center[:2].tolist()], dtype=np.float64
    )
    if not np.allclose(slam_intrinsics, inference_intrinsics, atol=1.0e-5, rtol=0.0):
        raise RuntimeError(
            "HaWoR SLAM output does not use the centered inference-plane K: "
            f"slam={slam_intrinsics.tolist()} inference={inference_intrinsics.tolist()}"
        )

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_video_module.hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
    _, _, R_c2w, t_c2w = load_slam_cam(str(slam_path))

    faces = np.asarray(get_mano_faces(), dtype=np.int32)
    hands = {}
    hand_to_idx = {"left": 0, "right": 1}
    for side, idx in hand_to_idx.items():
        trans = pred_trans[idx : idx + 1]
        rot = pred_rot[idx : idx + 1]
        pose = pred_hand_pose[idx : idx + 1]
        betas = pred_betas[idx : idx + 1]
        if side == "left":
            mano = run_mano_left(trans, rot, pose, betas=betas)
            hand_faces = faces[:, [0, 2, 1]]
        else:
            mano = run_mano(trans, rot, pose, betas=betas)
            hand_faces = faces
        hands[side] = {
            "vertices_world_m": as_numpy(mano["vertices"][0]).astype(np.float32),
            "joints_world_m": as_numpy(mano["joints"][0]).astype(np.float32),
            "trans_world_m": as_numpy(trans[0]).astype(np.float32),
            "root_orient_axis_angle": as_numpy(rot[0]).astype(np.float32),
            "hand_pose_axis_angle": as_numpy(pose[0]).astype(np.float32),
            "betas": as_numpy(betas[0]).astype(np.float32),
            "valid": as_numpy(pred_valid[idx]).astype(np.uint8),
            "faces": hand_faces.astype(np.int32),
        }

    frame_idx = np.arange(len(imgfiles), dtype=np.int32)
    track_support, track_support_report = load_track_support(Path(seq_folder), int(start_idx), int(end_idx), len(frame_idx))
    if camera_image_plane_contract is not None:
        A_source_from_inference = np.asarray(
            camera_image_plane_contract["A_source_from_hawor_inference"],
            dtype=np.float64,
        )
        source_size_wh = tuple(
            int(value) for value in camera_image_plane_contract["source_size_wh"]
        )
        transformed_box_count = 0
        for side in ("left", "right"):
            boxes = track_support[side]["det_box_xyxyscore"]
            valid = track_support[side]["detected_same_frame"].astype(bool)
            for row_index in np.flatnonzero(valid):
                boxes[row_index] = transform_boxes_to_source_plane(
                    boxes[row_index], A_source_from_inference, source_size_wh
                )
                transformed_box_count += 1
        track_support_report["detector_box_output_plane"] = "source_rgb"
        track_support_report["detector_box_input_plane"] = "hawor_centered_inference"
        track_support_report["A_source_from_hawor_inference"] = (
            A_source_from_inference.tolist()
        )
        track_support_report["detector_boxes_transformed_to_source_count"] = (
            transformed_box_count
        )
    else:
        A_source_from_inference = np.eye(3, dtype=np.float64)
        track_support_report["detector_box_output_plane"] = (
            "legacy_hawor_source_equals_inference"
        )
    camera_contract_mode = (
        FULL_K_RECTIFIED_MODE
        if camera_image_plane_contract is not None
        else LEGACY_CENTER_MODE
    )
    A_inference_from_source = (
        np.asarray(
            camera_image_plane_contract["A_hawor_inference_from_source"],
            dtype=np.float64,
        )
        if camera_image_plane_contract is not None
        else np.eye(3, dtype=np.float64)
    )
    camera_image_plane_contract_path = (
        Path(str(camera_image_plane_contract["path"]))
        if camera_image_plane_contract is not None
        else None
    )
    out_npz = args.output_dir / "hawor_world_hands.npz"
    np.savez_compressed(
        out_npz,
        frame_idx=frame_idx,
        R_c2w=as_numpy(R_c2w).astype(np.float32),
        t_c2w=as_numpy(t_c2w).astype(np.float32),
        left_vertices_world_m=hands["left"]["vertices_world_m"],
        left_joints_world_m=hands["left"]["joints_world_m"],
        left_trans_world_m=hands["left"]["trans_world_m"],
        left_root_orient_axis_angle=hands["left"]["root_orient_axis_angle"],
        left_hand_pose_axis_angle=hands["left"]["hand_pose_axis_angle"],
        left_betas=hands["left"]["betas"],
        left_valid=hands["left"]["valid"],
        left_detected_same_frame=track_support["left"]["detected_same_frame"],
        left_det_box_xyxyscore=track_support["left"]["det_box_xyxyscore"],
        left_track_id=track_support["left"]["track_id"],
        left_faces=hands["left"]["faces"],
        right_vertices_world_m=hands["right"]["vertices_world_m"],
        right_joints_world_m=hands["right"]["joints_world_m"],
        right_trans_world_m=hands["right"]["trans_world_m"],
        right_root_orient_axis_angle=hands["right"]["root_orient_axis_angle"],
        right_hand_pose_axis_angle=hands["right"]["hand_pose_axis_angle"],
        right_betas=hands["right"]["betas"],
        right_valid=hands["right"]["valid"],
        right_detected_same_frame=track_support["right"]["detected_same_frame"],
        right_det_box_xyxyscore=track_support["right"]["det_box_xyxyscore"],
        right_track_id=track_support["right"]["track_id"],
        right_faces=hands["right"]["faces"],
        img_focal=np.asarray([float(img_focal)], dtype=np.float32),
        camera_intrinsics_fx_fy_cx_cy=np.repeat(
            source_intrinsics[None, :], len(frame_idx), axis=0
        ).astype(np.float64),
        hawor_inference_intrinsics_fx_fy_cx_cy=np.repeat(
            inference_intrinsics[None, :], len(frame_idx), axis=0
        ).astype(np.float64),
        camera_intrinsics_contract_mode=np.asarray([camera_contract_mode]),
        A_hawor_inference_from_source=A_inference_from_source.astype(np.float64),
        A_source_from_hawor_inference=A_source_from_inference.astype(np.float64),
        camera_image_plane_contract_path=np.asarray([
            str(camera_image_plane_contract_path)
            if camera_image_plane_contract_path is not None
            else ""
        ]),
        camera_image_plane_contract_sha256=np.asarray([
            sha256(camera_image_plane_contract_path)
            if camera_image_plane_contract_path is not None
            else ""
        ]),
        video_path=np.asarray([str(video_path_obj)]),
        hawor_inference_video_path=np.asarray([str(inference_video_path)]),
        video_sha256=np.asarray([video_sha256 or ""]),
        checkpoint_sha256=np.asarray([export_provenance["checkpoint"].get("sha256") or ""]),
        infiller_weight_sha256=np.asarray([export_provenance["infiller_weight"].get("sha256") or ""]),
        model_config_sha256=np.asarray([export_provenance["model_config"].get("sha256") or ""]),
        seq_folder=np.asarray([str(seq_folder)]),
        track_support_status=np.asarray([track_support_report.get("status", "unknown")]),
        track_support_path=np.asarray([track_support_report.get("tracks_path", "")]),
        focal_cache_contract_path=np.asarray([focal_cache_contract.get("path", "")]),
        focal_cache_contract_status=np.asarray([focal_cache_contract.get("status", "disabled")]),
    )
    valid_counts = {side: int(np.count_nonzero(hands[side]["valid"])) for side in hands}
    detected_counts = {side: int(np.count_nonzero(track_support[side]["detected_same_frame"])) for side in hands}
    qc = {
        "status": "ok",
        "video_path": str(video_path_obj),
        "video_sha256": video_sha256,
        "hawor_inference_video_path": str(inference_video_path),
        "export_provenance": export_provenance,
        "seq_folder": str(seq_folder),
        "output_npz": str(out_npz),
        "frames": int(len(frame_idx)),
        "img_focal": float(img_focal),
        "camera_intrinsics_fx_fy_cx_cy": source_intrinsics.tolist(),
        "hawor_inference_intrinsics_fx_fy_cx_cy": inference_intrinsics.tolist(),
        "camera_intrinsics_contract_mode": camera_contract_mode,
        "camera_image_plane_contract": camera_image_plane_contract,
        "slam_intrinsics_fx_fy_cx_cy": slam_intrinsics.tolist(),
        "camera_intrinsics_exactly_bound": bool(
            np.allclose(slam_intrinsics, inference_intrinsics, atol=1.0e-5, rtol=0.0)
            and np.allclose(
                A_inference_from_source @ K_from_intrinsics(source_intrinsics),
                K_from_intrinsics(inference_intrinsics),
                atol=1.0e-9,
                rtol=0.0,
            )
        ),
        "camera_intrinsics_binding_semantics": (
            "source official K -> exact affine centered HaWoR plane -> MANO/masks/SLAM; inverse affine returns projections/boxes to source pixels"
            if camera_image_plane_contract is not None
            else "legacy source image is the centered HaWoR inference plane"
        ),
        "valid_hand_frames": valid_counts,
        "detected_same_frame_hand_frames": detected_counts,
        "track_support": track_support_report,
        "focal_cache_contract": focal_cache_contract,
        "slam_path": str(slam_path),
    }
    (args.output_dir / "qc_hawor_world_hands.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    project_hawor = Path(__file__).resolve().parents[1] / ".runtime" / "hawor_work" / "third_party" / "HaWoR"
    parser.add_argument(
        "--hawor-root",
        type=Path,
        default=Path(os.environ.get("EGO_HAWOR_REPO", str(project_hawor))),
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--model_config", type=str, default="./weights/hawor/model_config.yaml")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument(
        "--camera-intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help=(
            "Exact source-image pinhole K. The exporter materializes a hash-bound same-size image translation "
            "whose K has principal point at image center; HaWoR regression, projected hand masks, and SLAM "
            "consume that centered inference plane, and the inverse affine binds outputs to source pixels. "
            "The current upstream video model requires FX==FY."
        ),
    )
    parser.add_argument("--force-focal-cache-refresh", action="store_true", help="Delete camera-dependent HaWoR motion/mask/SLAM cache artifacts when their exact full K differs. Prefer a fresh camera-contract-specific video path when possible.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
