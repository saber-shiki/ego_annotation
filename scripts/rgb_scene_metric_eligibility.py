"""Typed native-depth metric eligibility for standalone MANO depth ordering.

The appearance-owned SAM2 mask is intentionally separate.  This module only
loads a hash-bound native depth raster and intersects it immediately before P09
robust-depth filtering/lifting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = "rgb_scene_native_depth_metric_eligibility_v1"
RASTER_SCHEMA = "rgb_scene_native_depth_metric_eligibility_raster_v1"
CLASS_CODES = {
    "outside_owned_appearance": 0,
    "hand_clearly_front": 1,
    "hand_clearly_behind": 2,
    "near_depth_ambiguous": 3,
    "clear_no_hand_overlap_candidate": 4,
    "no_hit_uncertainty_fringe": 5,
    "unsupported_depth": 6,
    "invalid_mask_coordinate": 7,
    "unresolved_hand_geometry": 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"metric eligibility input missing: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256(path)}


def conservative_mask_on_depth(mask: np.ndarray, A_depth_from_mask: np.ndarray,
                               depth_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV-nearest mask samples and an explicit coordinate-valid guard.

    OpenCV can assign a border source sample to a destination center whose
    inverse-mapped continuous coordinate lies outside the declared source
    raster.  Such subpixel extrapolation remains appearance support but cannot
    become a metric candidate.
    """
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(A_depth_from_mask, dtype=np.float64)
    if (mask.ndim != 2 or len(depth_shape) != 2 or min(depth_shape) <= 0
            or affine.shape != (3, 3) or not np.isfinite(affine).all()
            or abs(float(np.linalg.det(affine))) < 1e-12):
        raise ValueError("invalid mask/depth affine contract")
    sampled = (cv2.resize(mask.astype(np.uint8), (depth_shape[1], depth_shape[0]),
                          interpolation=cv2.INTER_NEAREST_EXACT) > 0
               if mask.shape != depth_shape else mask.copy())
    inverse = np.linalg.inv(affine)
    yy, xx = np.mgrid[:depth_shape[0], :depth_shape[1]]
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ inverse.T
    finite = np.isfinite(points).all(axis=2) & (np.abs(points[..., 2]) > 1e-12)
    mx = np.divide(points[..., 0], points[..., 2], out=np.zeros_like(points[..., 0], dtype=float), where=finite)
    my = np.divide(points[..., 1], points[..., 2], out=np.zeros_like(points[..., 1], dtype=float), where=finite)
    coordinate_valid = (finite & (mx >= 0.0) & (mx <= mask.shape[1] - 1.0)
                        & (my >= 0.0) & (my <= mask.shape[0] - 1.0))
    return sampled, coordinate_valid


def classify_native_depth(*, owned_mask: np.ndarray, hand_z: np.ndarray,
                          depth_z: np.ndarray, A_depth_from_mask: np.ndarray,
                          complete_hand_geometry: bool, validated_hand_absence: bool,
                          tolerance_m: float, uncertainty_padding_mask_px: int,
                          raw_mask: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Classify at the exact depth raster and return robust-filter candidates.

    Clear no-overlap object body remains a candidate; no-hit pixels within the
    conservative projected-hand band do not.  Missing declared hand geometry
    makes the frame ineligible while preserving appearance elsewhere.
    """
    depth_z = np.asarray(depth_z, dtype=np.float64)
    hand_z = np.asarray(hand_z, dtype=np.float64)
    if (depth_z.ndim != 2 or hand_z.shape != depth_z.shape
            or not np.isfinite(tolerance_m) or tolerance_m < 0
            or type(uncertainty_padding_mask_px) is not int or uncertainty_padding_mask_px < 0):
        raise ValueError("invalid native depth-order inputs")
    owned_depth, coordinate_valid = conservative_mask_on_depth(
        owned_mask, A_depth_from_mask, depth_z.shape)
    raw_depth, raw_coordinate_valid = conservative_mask_on_depth(
        owned_mask if raw_mask is None else raw_mask, A_depth_from_mask, depth_z.shape)
    if not np.array_equal(coordinate_valid, raw_coordinate_valid):
        raise ValueError("raw/owned mask coordinate guards disagree")
    codes = np.full(depth_z.shape, CLASS_CODES["outside_owned_appearance"], dtype=np.uint8)
    support = np.isfinite(depth_z) & (depth_z > 0)
    active = raw_depth & coordinate_valid
    codes[raw_depth & ~coordinate_valid] = CLASS_CODES["invalid_mask_coordinate"]
    codes[active & ~support] = CLASS_CODES["unsupported_depth"]
    if not complete_hand_geometry and not validated_hand_absence:
        codes[active & support] = CLASS_CODES["unresolved_hand_geometry"]
        eligible = np.zeros(depth_z.shape, dtype=bool)
    else:
        hit = np.isfinite(hand_z) & (hand_z > 0)
        valid_hit = active & support & hit
        delta = np.zeros(depth_z.shape, dtype=np.float64)
        np.subtract(hand_z, depth_z, out=delta, where=valid_hit)
        front = valid_hit & (delta < -tolerance_m)
        behind = valid_hit & (delta > tolerance_m)
        near = valid_hit & ~front & ~behind
        codes[front] = CLASS_CODES["hand_clearly_front"]
        codes[behind] = CLASS_CODES["hand_clearly_behind"]
        codes[near] = CLASS_CODES["near_depth_ambiguous"]
        no_hit = active & support & ~hit
        if validated_hand_absence:
            fringe = np.zeros(depth_z.shape, dtype=bool)
        else:
            linear = np.asarray(A_depth_from_mask, dtype=np.float64)[:2, :2]
            depth_pad = int(np.ceil(float(uncertainty_padding_mask_px)
                                    * max(np.linalg.norm(linear[:, 0]), np.linalg.norm(linear[:, 1]))))
            if depth_pad:
                kernel = np.ones((2 * depth_pad + 1, 2 * depth_pad + 1), np.uint8)
                fringe = cv2.dilate(hit.astype(np.uint8), kernel) > 0
            else:
                fringe = hit.copy()
        codes[no_hit & fringe] = CLASS_CODES["no_hit_uncertainty_fringe"]
        codes[no_hit & ~fringe] = CLASS_CODES["clear_no_hand_overlap_candidate"]
        eligible = (behind | (no_hit & ~fringe)) & owned_depth
    counts = {name: int(np.count_nonzero(codes == code)) for name, code in CLASS_CODES.items()}
    counts["metric_candidate_eligible"] = int(np.count_nonzero(eligible))
    counts["appearance_owned_at_depth"] = int(np.count_nonzero(owned_depth))
    counts["raw_appearance_at_depth"] = int(np.count_nonzero(raw_depth))
    return codes, eligible, counts


class MetricEligibility:
    def __init__(self, manifest_path: Path, *, case: str, object_id: str, track_id: str,
                 sam2_track: Path, depth_npz: Path, base_annotations: Path | None,
                 calibration_contract: Path | None, frame_to_depth_i: dict[int, int],
                 depth: np.ndarray):
        self.path = Path(manifest_path).expanduser().resolve()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != SCHEMA:
            raise ValueError("wrong metric eligibility manifest schema")
        if value.get("class_codes") != CLASS_CODES or value.get("policy", {}).get("physical_authority") is not False:
            raise ValueError("metric eligibility class/authority contract mismatch")
        for key, expected in (("case", case), ("object_id", object_id), ("track_id", track_id)):
            if value.get(key) != expected:
                raise ValueError(f"metric eligibility identity mismatch: {key}")
        expected_inputs = {
            "owned_sam2_track": sam2_track,
            "depth_npz": depth_npz,
            "base_annotations": base_annotations,
            "calibration_contract": calibration_contract,
        }
        inputs = value.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("metric eligibility input identity missing")
        for name, row in inputs.items():
            if row is None:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError(f"malformed metric eligibility input: {name}")
            recorded_path = Path(row["path"]).expanduser().resolve()
            if (not recorded_path.is_file() or row.get("sha256") != sha256(recorded_path)
                    or row.get("bytes") != recorded_path.stat().st_size):
                raise ValueError(f"metric eligibility recorded input changed: {name}")
        for name, expected_path in expected_inputs.items():
            if expected_path is None:
                if inputs.get(name) is not None:
                    raise ValueError(f"metric eligibility unexpected input: {name}")
                continue
            expected_path = Path(expected_path).expanduser().resolve()
            row = inputs.get(name)
            if (not isinstance(row, dict) or row.get("path") != str(expected_path)
                    or row.get("sha256") != sha256(expected_path)
                    or row.get("bytes") != expected_path.stat().st_size):
                raise ValueError(f"metric eligibility input path/hash mismatch: {name}")
        rows = value.get("frames")
        if not isinstance(rows, list) or len(rows) != len(frame_to_depth_i):
            raise ValueError("metric eligibility frame coverage mismatch")
        self.rows = {}
        self.depth = np.asarray(depth)
        self.frame_to_depth_i = frame_to_depth_i
        for row in rows:
            if not isinstance(row, dict) or type(row.get("frame_idx")) is not int:
                raise ValueError("malformed metric eligibility frame")
            idx = row["frame_idx"]
            if idx in self.rows or idx not in frame_to_depth_i:
                raise ValueError("duplicate/foreign metric eligibility frame")
            if row.get("depth_array_sha256") != array_sha256(self.depth[frame_to_depth_i[idx]]):
                raise ValueError(f"metric eligibility depth row changed: {idx}")
            self.rows[idx] = row
        if set(self.rows) != set(frame_to_depth_i):
            raise ValueError("metric eligibility frame IDs mismatch")
        self.manifest = value

    def frame(self, frame_idx: int, *, owned_mask_path: Path,
              depth_shape: tuple[int, int], A_depth_from_mask: np.ndarray,
              K_depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        row = self.rows.get(frame_idx)
        if row is None:
            raise ValueError(f"metric eligibility lacks frame {frame_idx}")
        owned_mask_path = Path(owned_mask_path).expanduser().resolve()
        owned = row.get("owned_mask")
        if (not isinstance(owned, dict) or owned.get("path") != str(owned_mask_path)
                or owned.get("sha256") != sha256(owned_mask_path)):
            raise ValueError(f"metric eligibility owned mask changed: {frame_idx}")
        if (np.asarray(row.get("A_depth_from_mask"), dtype=float).shape != (3, 3)
                or not np.allclose(np.asarray(row["A_depth_from_mask"], dtype=float),
                                   np.asarray(A_depth_from_mask, dtype=float), atol=1e-9, rtol=0)
                or np.asarray(row.get("K_depth"), dtype=float).shape != (3, 3)
                or not np.allclose(np.asarray(row["K_depth"], dtype=float),
                                   np.asarray(K_depth, dtype=float), atol=1e-9, rtol=0)):
            raise ValueError(f"metric eligibility depth plane/affine mismatch: {frame_idx}")
        raster_path = Path(str(row.get("raster", {}).get("path", ""))).expanduser().resolve()
        raster = row.get("raster")
        if (not isinstance(raster, dict) or raster.get("sha256") != sha256(raster_path)
                or raster.get("bytes") != raster_path.stat().st_size):
            raise ValueError(f"metric eligibility raster changed: {frame_idx}")
        with np.load(raster_path, allow_pickle=False) as blob:
            if set(blob.files) != {"schema", "frame_idx", "class_code", "metric_candidate_eligible"}:
                raise ValueError("metric eligibility raster keys mismatch")
            if str(np.asarray(blob["schema"]).reshape(-1)[0]) != RASTER_SCHEMA:
                raise ValueError("metric eligibility raster schema mismatch")
            if int(np.asarray(blob["frame_idx"]).reshape(-1)[0]) != frame_idx:
                raise ValueError("metric eligibility raster frame mismatch")
            classes = np.asarray(blob["class_code"])
            eligible = np.asarray(blob["metric_candidate_eligible"])
        if (classes.shape != depth_shape or classes.dtype != np.uint8
                or eligible.shape != depth_shape or eligible.dtype != np.bool_
                or not set(np.unique(classes)).issubset(set(CLASS_CODES.values()))):
            raise ValueError("metric eligibility raster type/shape mismatch")
        owned_mask_image = cv2.imread(str(owned_mask_path), cv2.IMREAD_GRAYSCALE)
        if owned_mask_image is None:
            raise ValueError(f"metric eligibility owned mask unreadable: {frame_idx}")
        owned_depth, _ = conservative_mask_on_depth(
            owned_mask_image > 0, np.asarray(A_depth_from_mask, dtype=float), depth_shape)
        expected_eligible = (np.isin(classes, [CLASS_CODES["hand_clearly_behind"],
                                                 CLASS_CODES["clear_no_hand_overlap_candidate"]])
                             & owned_depth)
        if not np.array_equal(eligible, expected_eligible):
            raise ValueError("metric eligibility boolean disagrees with typed classes")
        if (row.get("class_code_array_sha256") != array_sha256(classes)
                or row.get("metric_candidate_array_sha256") != array_sha256(eligible)):
            raise ValueError("metric eligibility raster array hash mismatch")
        return eligible, {"manifest": str(self.path), "raster": str(raster_path),
                          "counts": row.get("counts"), "class_codes": CLASS_CODES}
