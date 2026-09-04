#!/usr/bin/env python3
"""Build prediction-side RGB/PnP relative pose factors for the experimental P14 graph.

The factors are deliberately separate from the formal P14/P15 state.  They use
only object-owned RGB masks, calibrated pinhole intrinsics, and source metric
surfel points.  A failed or low-quality optical-flow/PnP bridge is recorded as
rejected rather than silently becoming a pose observation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "fit_to_object_owned_rgb_calibrated_pnp",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj
    return None


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def relative_pose(R_source: np.ndarray, t_source: np.ndarray, R_target: np.ndarray, t_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return D such that R_target=D_R R_source and t_target=D_R t_source+D_t."""
    D_R = np.asarray(R_target, dtype=np.float64) @ np.asarray(R_source, dtype=np.float64).T
    D_t = np.asarray(t_target, dtype=np.float64) - D_R @ np.asarray(t_source, dtype=np.float64)
    return D_R, D_t


def edge_quality(rotation_conflict_deg: float, translation_conflict_m: float, reprojection_median_px: float, inlier_fraction: float) -> float:
    # RGB factors are soft consistency evidence, not ground truth.  A large
    # disagreement with the metric edge lowers their influence in the global
    # objective while retaining the diagnostic row.
    score = math.exp(-0.5 * (rotation_conflict_deg / 12.0) ** 2 - 0.5 * (translation_conflict_m / 0.025) ** 2)
    score *= float(np.clip(inlier_fraction, 0.25, 1.0))
    score *= float(np.clip(1.5 / max(reprojection_median_px, 0.25), 0.25, 1.0))
    return float(np.clip(score, 0.05, 1.0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--p14-script", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--pair-stride", type=int, default=1)
    p.add_argument("--max-edges", type=int, default=None)
    p.add_argument("--flow-max-seed-points", type=int, default=1200)
    p.add_argument("--flow-min-tracked-points", type=int, default=50)
    p.add_argument("--flow-min-pnp-inliers", type=int, default=40)
    p.add_argument("--flow-min-pnp-inlier-fraction", type=float, default=0.50)
    p.add_argument("--flow-max-reprojection-median-px", type=float, default=2.5)
    p.add_argument("--flow-max-reprojection-p95-px", type=float, default=4.0)
    args = p.parse_args()
    if args.pair_stride < 1:
        raise RuntimeError("--pair-stride must be >= 1")

    spec = importlib.util.spec_from_file_location("p14_pose_module", args.p14_script.expanduser().resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import P14 module: {args.p14_script}")
    p14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p14)

    annotations = load_json(args.annotations)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    pose_report = load_json(args.pose_report)
    pose_rows = {
        int(row["frame_idx"]): row
        for row in pose_report.get("pose_rows", [])
        if isinstance(row, dict)
        and str(row.get("status") or "") in POSE_STATUSES
        and (args.frame_start is None or int(row["frame_idx"]) >= args.frame_start)
        and (args.frame_end is None or int(row["frame_idx"]) <= args.frame_end)
    }
    objects = {}
    observed = {}
    for idx, frame in frames.items():
        if args.frame_start is not None and idx < args.frame_start:
            continue
        if args.frame_end is not None and idx > args.frame_end:
            continue
        obj = frame_object(frame, args.object_id)
        if obj is None:
            continue
        objects[idx] = obj
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        pts = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 50 and np.isfinite(pts).all() and idx in pose_rows:
            observed[idx] = pts
    direct = sorted(set(pose_rows) & set(observed))
    if len(direct) < 2:
        raise RuntimeError(f"only {len(direct)} usable direct RGB/PnP source frames")
    pairs = [(direct[i], direct[i + args.pair_stride]) for i in range(0, len(direct) - args.pair_stride, args.pair_stride)]
    if args.max_edges is not None:
        pairs = pairs[: int(args.max_edges)]

    R_rgb_all=[]; t_rgb_all=[]; R_metric_all=[]; t_metric_all=[]
    source_all=[]; target_all=[]; weight_all=[]; accepted_all=[]
    rows=[]; accepted_count=0
    for edge_no, (source_idx, target_idx) in enumerate(pairs, 1):
        source_pose=pose_rows[source_idx]; target_pose=pose_rows[target_idx]
        Rs=np.asarray(source_pose["rotation_world_from_completed_canonical_matrix"],dtype=np.float64);ts=np.asarray(source_pose["translation_world_m"],dtype=np.float64)
        Rt=np.asarray(target_pose["rotation_world_from_completed_canonical_matrix"],dtype=np.float64);tt=np.asarray(target_pose["translation_world_m"],dtype=np.float64)
        Rm,tm=relative_pose(Rs,ts,Rt,tt)
        base={"source_frame_idx":int(source_idx),"target_frame_idx":int(target_idx),"frame_gap":int(target_idx-source_idx),"metric_rotation":Rm.tolist(),"metric_translation_m":tm.tolist()}
        try:
            Rabs, tabs, bridge, observability = p14.estimate_optical_flow_pnp_bridge(
                source_idx=source_idx,target_idx=target_idx,frames=frames,objects=objects,
                observed_world=observed[source_idx],source_rotation=Rs,source_translation=ts,
                max_seed_points=int(args.flow_max_seed_points),min_tracked_points=int(args.flow_min_tracked_points),
                min_pnp_inliers=int(args.flow_min_pnp_inliers),min_inlier_fraction=float(args.flow_min_pnp_inlier_fraction),
                max_reprojection_median_px=float(args.flow_max_reprojection_median_px),max_reprojection_p95_px=float(args.flow_max_reprojection_p95_px),
            )
            Rrgb,trgb=relative_pose(Rs,ts,Rabs,tabs)
            rot_conflict=float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rrgb@Rm.T).as_rotvec())))
            trans_conflict=float(np.linalg.norm(trgb-tm))
            reproj=float(bridge["reprojection_error_px"]["median"]); inlier_frac=float(bridge["pnp_inlier_fraction"])
            weight=edge_quality(rot_conflict,trans_conflict,reproj,inlier_frac)
            accepted=True; accepted_count+=1
            base.update({"status":"accepted","rgb_rotation":Rrgb.tolist(),"rgb_translation_m":trgb.tolist(),"rotation_conflict_deg":rot_conflict,"translation_conflict_m":trans_conflict,"quality_weight":weight,"reprojection_median_px":reproj,"reprojection_p95_px":float(bridge["reprojection_error_px"]["p95"]),"pnp_inlier_fraction":inlier_frac,"tracked_count":int(bridge["target_track_count"]),"observability_score":float(observability.get("rotation_observability_score",0.0)) if isinstance(observability,dict) else None})
        except Exception as exc:
            accepted=False; weight=0.0; Rrgb=np.eye(3);trgb=np.zeros(3)
            base.update({"status":"rejected","reason":str(exc)[:1000],"quality_weight":0.0})
        source_all.append(source_idx);target_all.append(target_idx);R_rgb_all.append(Rrgb);t_rgb_all.append(trgb);R_metric_all.append(Rm);t_metric_all.append(tm);weight_all.append(weight);accepted_all.append(accepted);rows.append(base)
        print(f"[rgb-pnp] {edge_no}/{len(pairs)} {source_idx}->{target_idx} {base['status']}",flush=True)

    metadata={"schema":"v19_rgb_pnp_relative_pose_factors_v1","annotations":str(args.annotations.expanduser().resolve()),"pose_report":str(args.pose_report.expanduser().resolve()),"p14_script":str(args.p14_script.expanduser().resolve()),"object_id":args.object_id,"generated_geometry_consumed":False,"input_sha256":{"annotations":sha256_file(args.annotations),"pose_report":sha256_file(args.pose_report),"p14_script":sha256_file(args.p14_script)}}
    args.output_npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output_npz,metadata=np.asarray([json.dumps(metadata)]),source_frame_idx=np.asarray(source_all,dtype=np.int64),target_frame_idx=np.asarray(target_all,dtype=np.int64),rotation_rgb=np.asarray(R_rgb_all,dtype=np.float64),translation_rgb_m=np.asarray(t_rgb_all,dtype=np.float64),rotation_metric=np.asarray(R_metric_all,dtype=np.float64),translation_metric_m=np.asarray(t_metric_all,dtype=np.float64),quality_weight=np.asarray(weight_all,dtype=np.float64),accepted=np.asarray(accepted_all,dtype=bool))
    report={**metadata,"pair_count":len(pairs),"accepted_count":accepted_count,"rejected_count":len(pairs)-accepted_count,"accepted_fraction":accepted_count/max(1,len(pairs)),"rotation_conflict_deg":numeric_summary([r["rotation_conflict_deg"] for r in rows if r.get("status")=="accepted"]),"translation_conflict_m":numeric_summary([r["translation_conflict_m"] for r in rows if r.get("status")=="accepted"]),"quality_weight":numeric_summary(weight_all),"rows":rows,"outputs":{"npz":str(args.output_npz.expanduser().resolve()),"report":str(args.output_json.expanduser().resolve())}}
    args.output_json.parent.mkdir(parents=True,exist_ok=True);args.output_json.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps({k:report[k] for k in ['pair_count','accepted_count','rejected_count','accepted_fraction','rotation_conflict_deg','translation_conflict_m','quality_weight']},indent=2))


if __name__ == "__main__":
    main()
