#!/usr/bin/env python3
"""Build the canonical ego.hoi contact_frame_detail table for clip001850.

Consumes the kill-test rows produced by
`run_clip001850_masked_contact_killtests.py` (KT-1/KT-2/KT-3/KT-5) and applies
the contact-state decision policy from subagent 10 (`contact_state_policy`):

  * each frame's kill-test `route` becomes the canonical `contact_state` only
    after the policy admissibility checks. In particular, a bare one-vertex
    `contact_candidate_keyboard_masked` is demoted to
    `unresolved_incoherent_evidence` because the policy requires a compact,
    temporally persistent contact patch, not a single depth outlier;
  * provenance columns required by policy §5 are attached to every row:
    depth-source hash + barrier flags, keyboard mask hash / missing reason,
    contact surface + sign mesh hashes / watertight, free-space state,
    pose provenance, combined metric uncertainty sigma + source_gap_z,
    renderer-consumed mesh hash + renderer_consumes_contact_state flag,
    observed-masked / raw penetrating counts, survival fraction,
    penetrating vertex ids, temporal IoU.

If the kill-test NDJSON is missing the script reruns the kill-test first.

Outputs (under --out-dir, default /tmp/clip001850_contact_state_table):
  contact_frame_detail.ndjson   — one canonical row per frame/hand
  summary.json                  — schema, route counts, provenance hashes,
                                  sigma_clip basis, per-state frame lists,
                                  renderer-consumption manifest

No model inference, no GPU. CPU-only file hashing + JSON transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

RUN_ROOT_DEFAULT = Path(
    "/data2/ego_annotation_outputs/v19_runs/"
    "20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1"
)
KILL_DIR_DEFAULT = Path("/tmp/clip001850_masked_contact_killtests")
OUT_DIR_DEFAULT = Path("/tmp/clip001850_contact_state_table")

# Combined metric uncertainty for clip001850 (policy §4 / hand_metric_mechanisms.md
# Track J lateral floor 25-30 mm + 3.40 deg rotation; systematic, not averageable).
SIGMA_CLIP_M = 0.028
SIGMA_CLIP_RANGE_M = [0.025, 0.030]

SCHEMA = "ego.hoi.contact_frame_detail/0.1.0"

# Frames the review renderer consumes (KT-7 review artifact).
DEFAULT_RENDERER_CONSUMED_FRAMES = [32, 36]


# --------------------------------------------------------------------------- #
# File hashing
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str | None:
    """Streaming SHA-256 of a file. Returns None if the file is absent."""
    path = _remount(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _remount(p: str | Path) -> Path:
    """Map /mnt/truenas-user-home/... stored paths to the local /data2 mount."""
    p = Path(p)
    if p.exists():
        return p
    s = str(p)
    for prefix in ("/mnt/truenas-user-home", "/mnt/user-home"):
        if s.startswith(prefix):
            local = Path("/data2" + s[len(prefix):])
            if local.exists():
                return local
    return p


# --------------------------------------------------------------------------- #
# Provenance blocks
# --------------------------------------------------------------------------- #
def depth_source_block(run_root: Path, interval_params: dict[str, Any]) -> dict[str, Any]:
    npz = run_root / "measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz"
    return {
        "path": str(npz),
        "sha256": sha256_file(npz),
        # Barrier flags as actually configured in the interval solver run.
        # The dense observed-surface barrier queries the completed mesh
        # filtered to depth-supported observed faces (KT-1/subagent 07).
        "barrier_target": "completed_mesh_observed_face_filtered",
        "barrier_mask_gate": False,
        "barrier_hand_quarantine": bool(interval_params.get("hand_owned_object_depth_quarantine", False)),
        "barrier_eligibility": False,
        "barrier_weight": float(interval_params.get("dense_observed_penetration_weight", 0.0) or 0.0),
        "barrier_enabled": bool(interval_params.get("dense_observed_surface_barrier", False)),
    }


def contact_surface_mesh_block(run_root: Path) -> dict[str, Any]:
    ply = run_root / "measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply"
    return {
        "path": str(ply),
        "sha256": sha256_file(ply),
        "watertight": False,           # set from killtest summary below
        "face_count": None,
    }


def contact_sign_mesh_block(run_root: Path) -> dict[str, Any]:
    """Sign-mesh provenance. On clip001850 no observed-face watertight sign mesh
    was built and free space was never carved, so the sign channel is absent."""
    return {
        "sha256": None,
        "watertight": False,
        "missing_reason": (
            "no_observed_face_watertight_sign_mesh_built; "
            "free_space_not_evaluated; completed_mesh_non_watertight"
        ),
    }


def face_provenance_global(summary: dict[str, Any]) -> dict[str, Any]:
    geom = summary.get("geometry_epoch", {})
    fp = geom.get("face_provenance", {}) or {}
    counts = fp.get("face_label_counts", {}) or {}
    total = sum(int(v) for v in counts.values()) if counts else 0
    return {
        "observed_fraction": fp.get("observed_fraction"),
        "trellis_fraction": fp.get("trellis_fraction"),
        "unsupported_fraction": fp.get("unsupported_fraction"),
        "face_label_counts": counts,
        "total_faces": total,
    }


# --------------------------------------------------------------------------- #
# Row transform
# --------------------------------------------------------------------------- #
def _stat(row: dict[str, Any], path: list[str], key: str = "median") -> Any:
    cur: Any = row
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if isinstance(cur, dict):
        return cur.get(key)
    return cur


def derive_contact_state(kill_row: dict[str, Any], temporal_iou_mean: float | None) -> tuple[str, str, list[str]]:
    """Apply the policy gates from subagent 10/14 to a kill-test route.

    The kill-test script used a permissive presence rule for
    contact_candidate_keyboard_masked. The canonical table must require the
    stronger policy: an eligible contact patch must be more than a single vertex
    and must have temporal support. Otherwise the state is unresolved with the
    concrete incoherence reason preserved.
    """
    route = kill_row.get("route", "evidence_missing")
    reasons: list[str] = []
    if route != "contact_candidate_keyboard_masked":
        return route, f"killtest_route:{route}", reasons

    hq = kill_row.get("KT1_keyboard_masked_hand_quarantined_depth", {}) or {}
    kt2 = kill_row.get("KT2_vertex_coherence", {}) or {}
    pen_ids = list(kt2.get("penetrating_sample_ids") or hq.get("penetrating_sample_ids") or [])
    near_ids = list(kt2.get("near_band_sample_ids") or hq.get("near_band_sample_ids") or [])
    pen_count = int(hq.get("penetrating_vertex_count", 0) or 0)
    near_count = int(hq.get("near_band_vertex_count", 0) or 0)

    if pen_count <= 1:
        reasons.append("only one eligible keyboard-masked penetrating vertex")
    if temporal_iou_mean is None or float(temporal_iou_mean) < 0.5:
        reasons.append(f"window temporal IoU below persistent-patch requirement: {temporal_iou_mean}")
    if len(pen_ids) <= 1 and len(near_ids) <= 1:
        reasons.append("no compact multi-vertex near/penetrating set")

    if reasons:
        return "unresolved_incoherent_evidence", "demoted_contact_candidate:single_vertex_or_temporally_incoherent", reasons
    return route, "killtest_route:contact_candidate_keyboard_masked", reasons


def canonical_row(
    kill_row: dict[str, Any],
    *,
    case: str,
    run_root: Path,
    depth_src: dict[str, Any],
    contact_mesh: dict[str, Any],
    sign_mesh: dict[str, Any],
    face_prov: dict[str, Any],
    sigma_clip_m: float,
    sigma_clip_range_m: list[float],
    temporal_iou_mean: float | None,
    render_consumed_mesh_sha256: str | None,
    renderer_consumes_contact_state: bool,
) -> dict[str, Any]:
    frame = int(kill_row["frame_idx"])
    route = kill_row.get("route", "evidence_missing")
    contact_state, contact_basis, demotion_reasons = derive_contact_state(kill_row, temporal_iou_mean)
    hq = kill_row.get("KT1_keyboard_masked_hand_quarantined_depth", {}) or {}
    kb = kill_row.get("KT1_keyboard_masked_depth", {}) or {}
    full = kill_row.get("KT1_full_frame_depth", {}) or {}
    kt2 = kill_row.get("KT2_vertex_coherence", {}) or {}

    # Use keyboard-masked + hand-quarantined delta median as the signed gap.
    # Convention: positive = hand behind surface (penetrating); negative = hand
    # in front (gap). Null when no masked channel is available.
    signed_gap_m = _stat(kill_row, ["KT1_keyboard_masked_hand_quarantined_depth", "delta_summary_m"])
    if signed_gap_m is None:
        signed_gap_m = _stat(kill_row, ["KT1_keyboard_masked_depth", "delta_summary_m"])
    source_gap_z = (signed_gap_m / sigma_clip_m) if (signed_gap_m is not None and sigma_clip_m) else None

    # survival fraction = masked+HQ penetrating / raw full-frame penetrating
    raw_pen = int(full.get("penetrating_vertex_count", 0) or 0) if full.get("available") else 0
    masked_pen = int(hq.get("penetrating_vertex_count", 0) or 0) if hq.get("available") else (
        int(kb.get("penetrating_vertex_count", 0) or 0) if kb.get("available") else 0
    )
    survival_fraction = (masked_pen / raw_pen) if raw_pen > 0 else None

    # keyboard mask provenance
    mask_source = kill_row.get("keyboard_mask_source", "missing")
    mask_files_exist = bool(kill_row.get("mask_files_exist", False))
    mask_path = (
        run_root / "measurements/object_tracks/sam2_agent_points/keyboard/sam2/sam2_masks" / f"{frame:06d}.png"
    )
    if mask_files_exist and mask_path.exists():
        mask_sha = sha256_file(mask_path)
        mask_missing_reason = None
    elif mask_source.startswith("proxy_"):
        # temporal hold: hash the held mask file if resolvable, else null
        mask_sha = sha256_file(mask_path) if mask_path.exists() else None
        mask_missing_reason = f"proxy_temporal_hold:{mask_source}; mask_files_exist=false"
    else:
        mask_sha = None
        mask_missing_reason = "no_keyboard_mask_file_and_no_proxy"

    # pose provenance
    pose_src = kill_row.get("pose_source", "unknown")
    direct_visible = bool(kill_row.get("pose_direct_visible", False))
    gap_frames = int(kill_row.get("pose_gap_frames", 0) or 0)
    if direct_visible:
        pose_provenance = "direct_visible_measurement"
    elif gap_frames > 0 or pose_src in ("nearest_visible_pose_hold", "missing_initial_graph_pose"):
        pose_provenance = "interpolated_or_held"
    else:
        pose_provenance = pose_src

    return {
        "schema": SCHEMA,
        "case": case,
        "frame_idx": frame,
        "hand_side": kill_row.get("hand_side", "right"),
        # Canonical contact state after policy admissibility gates.
        "contact_state": contact_state,
        "raw_killtest_route": route,
        "contact_state_basis": contact_basis,
        "contact_state_demotion_reasons": demotion_reasons,
        "object_body_provenance": "geometry_epoch_contaminated",
        # ---- Provenance §5 ----
        "depth_source": depth_src,
        "keyboard_mask": {
            "source": mask_source,
            "path": str(mask_path),
            "sha256": mask_sha,
            "files_exist": mask_files_exist,
            "missing_reason": mask_missing_reason,
        },
        "contact_surface_mesh": contact_mesh,
        "contact_sign_mesh": sign_mesh,
        "free_space": {
            "evaluated": False,
            "rejected_state": "not_evaluated",
            "rejected_fraction": None,
        },
        "nearest_surface_face_provenance": "not_evaluated_per_query",
        "face_provenance_global": face_prov,
        "pose": {
            "source": pose_src,
            "direct_visible_measurement": direct_visible,
            "gap_frames": gap_frames,
            "provenance": pose_provenance,
            "report": "measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json",
        },
        "combined_metric_uncertainty_sigma_m": sigma_clip_m,
        "sigma_clip_range_m": sigma_clip_range_m,
        "sigma_clip_basis": (
            "hand_metric_mechanisms.md Track J lateral floor 25-30mm + "
            "3.40deg camera-frame rotation; systematic bias, not averageable"
        ),
        "source_gap": {
            "signed_gap_m": signed_gap_m,
            "sigma_clip_m": sigma_clip_m,
            "source_gap_z": source_gap_z,
            "computable": signed_gap_m is not None,
            "surface": "keyboard_masked_hand_quarantined_depth",
            "convention": "positive=hand_behind_surface(penetrating); negative=hand_in_front(gap)",
        },
        # ---- KT-1 observed masked statistics ----
        "observed_masked_penetrating_count": masked_pen,
        "observed_masked_near_count": int(hq.get("near_band_vertex_count", 0) or 0) if hq.get("available") else (
            int(kb.get("near_band_vertex_count", 0) or 0) if kb.get("available") else 0
        ),
        "raw_penetrating_count": raw_pen,
        "survival_fraction": survival_fraction,
        "penetrating_vertex_ids": kt2.get("penetrating_sample_ids", []),
        "near_band_vertex_ids": kt2.get("near_band_sample_ids", []),
        "temporal_iou": temporal_iou_mean,
        # ---- KT-7 render consumption ----
        "render_consumed_mesh_sha256": render_consumed_mesh_sha256,
        "renderer_consumes_contact_state": renderer_consumes_contact_state,
        # ---- Interval-solver published channel (for cross-solver comparison) ----
        "interval_solver_published": kill_row.get("interval_solver_published", {}),
        "interval_published_penetration_max_m": _stat(
            kill_row, ["interval_solver_published", "interval_full_observed_surface_penetration_m"], "max"
        ),
        # ---- Original kill-test evidence (verbatim, for the renderer / audit) ----
        "killtest_evidence": kill_row,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_killtest_rows(kill_dir: Path, run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nd = kill_dir / "contact_killtests_frame_detail.ndjson"
    sm = kill_dir / "summary.json"
    if not nd.exists() or not sm.exists():
        print(f"[contact_state] kill-test outputs missing, rerunning kill-test script...", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_clip001850_masked_contact_killtests.py"),
            "--out-dir", str(kill_dir),
        ]
        subprocess.check_call(cmd)
    rows = []
    for line in nd.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    summary = json.loads(sm.read_text())
    return rows, summary


def interval_params(run_root: Path) -> dict[str, Any]:
    p = run_root / (
        "measurements/mano_interval_correction/keyboard_0_149/"
        "hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/"
        "v18_joint_mano_interval_trajectory_state.json"
    )
    if not p.exists():
        return {}
    return (json.loads(p.read_text()) or {}).get("parameters", {}) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--kill-dir", type=Path, default=KILL_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument(
        "--renderer-consumed-frames", type=int, nargs="+",
        default=DEFAULT_RENDERER_CONSUMED_FRAMES,
        help="Frames whose contact_state the review renderer consumes "
             "(sets renderer_consumes_contact_state=true on those rows).",
    )
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    kill_rows, kill_summary = load_killtest_rows(args.kill_dir, args.run_root)
    print(f"[contact_state] loaded {len(kill_rows)} kill-test rows from {args.kill_dir}", flush=True)

    case = kill_summary.get("case", args.run_root.name)
    params = interval_params(args.run_root)

    # Shared provenance (file hashes computed once).
    depth_src = depth_source_block(args.run_root, params)
    contact_mesh = contact_surface_mesh_block(args.run_root)
    # fill watertight / face_count from killtest summary
    geom = kill_summary.get("geometry_epoch", {})
    contact_mesh["watertight"] = bool(geom.get("completed_mesh_watertight", False))
    contact_mesh["face_count"] = int(geom.get("completed_mesh_face_count", 0) or 0)
    contact_mesh["aabb_extents_m"] = geom.get("completed_mesh_aabb_extents_m")
    contact_mesh["extent_ratio_to_keyboard_prior"] = geom.get("completed_mesh_extent_ratio_to_keyboard_prior")
    sign_mesh = contact_sign_mesh_block(args.run_root)
    face_prov = face_provenance_global(kill_summary)
    temporal_iou_mean = (kill_summary.get("KT2_temporal_coherence", {}) or {}).get("mean_iou")
    render_consumed_mesh_sha256 = contact_mesh.get("sha256")

    consumed = set(args.renderer_consumed_frames)

    canon_rows = []
    for kr in kill_rows:
        cr = canonical_row(
            kr,
            case=case,
            run_root=args.run_root,
            depth_src=depth_src,
            contact_mesh=contact_mesh,
            sign_mesh=sign_mesh,
            face_prov=face_prov,
            sigma_clip_m=SIGMA_CLIP_M,
            sigma_clip_range_m=SIGMA_CLIP_RANGE_M,
            temporal_iou_mean=temporal_iou_mean,
            render_consumed_mesh_sha256=render_consumed_mesh_sha256,
            renderer_consumes_contact_state=(int(kr["frame_idx"]) in consumed),
        )
        canon_rows.append(cr)

    # ---- Write NDJSON ----
    nd_path = out / "contact_frame_detail.ndjson"
    with open(nd_path, "w") as fh:
        for r in canon_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[contact_state] wrote {nd_path} ({len(canon_rows)} rows)", flush=True)

    # ---- Summary ----
    from collections import Counter, defaultdict
    state_counts = dict(Counter(r["contact_state"] for r in canon_rows))
    state_frames: dict[str, list[int]] = defaultdict(list)
    for r in canon_rows:
        state_frames[r["contact_state"]].append(r["frame_idx"])

    summary = {
        "schema": SCHEMA,
        "case": case,
        "run_root": str(args.run_root),
        "killtest_source": {
            "ndjson": str(args.kill_dir / "contact_killtests_frame_detail.ndjson"),
            "summary": str(args.kill_dir / "summary.json"),
        },
        "window": kill_summary.get("window"),
        "policy_source": (
            ".memory/tasks/2026-07-06-research-track-execution/subagents/"
            "10_contact_state_policy_from_killtests.md"
        ),
        "contact_state_counts": state_counts,
        "contact_state_frames": dict(state_frames),
        "sigma_clip": {
            "combined_sigma_m": SIGMA_CLIP_M,
            "range_m": SIGMA_CLIP_RANGE_M,
            "basis": (
                "hand_metric_mechanisms.md Track J lateral floor 25-30mm + "
                "3.40deg rotation; systematic bias"
            ),
        },
        "provenance": {
            "depth_source_sha256": depth_src.get("sha256"),
            "depth_barrier": {
                "target": depth_src.get("barrier_target"),
                "mask_gate": depth_src.get("barrier_mask_gate"),
                "hand_quarantine": depth_src.get("barrier_hand_quarantine"),
                "eligibility": depth_src.get("barrier_eligibility"),
                "weight": depth_src.get("barrier_weight"),
                "enabled": depth_src.get("barrier_enabled"),
            },
            "contact_surface_mesh_sha256": contact_mesh.get("sha256"),
            "contact_surface_mesh_watertight": contact_mesh.get("watertight"),
            "contact_sign_mesh_sha256": sign_mesh.get("sha256"),
            "contact_sign_mesh_watertight": sign_mesh.get("watertight"),
            "contact_sign_mesh_missing_reason": sign_mesh.get("missing_reason"),
            "free_space_evaluated": False,
            "free_space_rejected_state": "not_evaluated",
            "face_provenance_global": face_prov,
        },
        "renderer_consumption": {
            "consumed_frames": sorted(consumed),
            "render_consumed_mesh_sha256": render_consumed_mesh_sha256,
            "renderer_consumes_contact_state": True,
            "note": (
                "The review renderer (render_clip001850_egohoi_contact_review.py) "
                "reads contact_state from this table for the listed frames. The "
                "production publish_v19_render_artifact.py does NOT yet consume "
                "this table (KT-7 consumer gap, policy §9)."
            ),
        },
        "required_state_check": {
            "f32_geometry_epoch_contaminated": any(
                r["frame_idx"] == 32 and r["contact_state"] == "geometry_epoch_contaminated"
                for r in canon_rows
            ),
            "f36_full_frame_depth_leak": any(
                r["frame_idx"] == 36 and r["contact_state"] == "full_frame_depth_leak"
                for r in canon_rows
            ),
            "f45_unresolved_incoherent_evidence": any(
                r["frame_idx"] == 45 and r["contact_state"] == "unresolved_incoherent_evidence"
                for r in canon_rows
            ),
            "pose_unresolved_for_proxy_or_interpolated": all(
                r["contact_state"] == "pose_unresolved"
                for r in canon_rows
                if r["pose"]["provenance"] == "interpolated_or_held"
                and not r["keyboard_mask"]["files_exist"]
            ),
        },
        "rows": [
            {
                "frame_idx": r["frame_idx"],
                "contact_state": r["contact_state"],
                "raw_killtest_route": r["raw_killtest_route"],
                "contact_state_demotion_reasons": r["contact_state_demotion_reasons"],
                "source_gap_z": r["source_gap"]["source_gap_z"],
                "observed_masked_penetrating_count": r["observed_masked_penetrating_count"],
                "raw_penetrating_count": r["raw_penetrating_count"],
                "survival_fraction": r["survival_fraction"],
                "pose_provenance": r["pose"]["provenance"],
                "mask_source": r["keyboard_mask"]["source"],
                "renderer_consumes_contact_state": r["renderer_consumes_contact_state"],
            }
            for r in canon_rows
        ],
    }
    sm_path = out / "summary.json"
    with open(sm_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[contact_state] wrote {sm_path}", flush=True)

    # ---- Console headline ----
    print("\n===== CONTACT STATE TABLE =====", flush=True)
    print(f"contact_state_counts: {state_counts}", flush=True)
    print(f"required checks: {summary['required_state_check']}", flush=True)
    print(f"depth_source_sha256: {depth_src.get('sha256')}", flush=True)
    print(f"contact_surface_mesh_sha256: {contact_mesh.get('sha256')}", flush=True)
    print(f"contact_surface_mesh_watertight: {contact_mesh.get('watertight')}", flush=True)
    print(f"free_space_evaluated: False (not_evaluated)", flush=True)
    for r in canon_rows:
        if r["frame_idx"] in (28, 32, 36, 45):
            sg = r["source_gap"]
            print(f"  f{r['frame_idx']:3d} state={r['contact_state']:34s} "
                  f"signed_gap_m={sg['signed_gap_m']:+.4f} z={sg['source_gap_z']} "
                  f"masked_pen={r['observed_masked_penetrating_count']} "
                  f"raw_pen={r['raw_penetrating_count']} "
                  f"survival={r['survival_fraction']} "
                  f"renderer_consumes={r['renderer_consumes_contact_state']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
