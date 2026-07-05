#!/usr/bin/env python3
"""
Package the clip001850 research slice into an `ego.hoi` 0.1.0 sidecar extension
that a downstream evaluator/render consumer can read.

Scope
-----
This is a packaging layer only. It does not run any model, GPU, or heavy
inference. It reads durable artifacts already produced under
`/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/`
and emits a sidecar package whose manifest points at the rendered full-duration
artifact and whose tables are sufficient to reproduce f32/f36/f45 contact state.

Output layout (under --out-dir, default
`/tmp/clip001850_ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/`):

    manifest.json
    tables/
      contact_frame_detail.ndjson          one row per frame (150 rows)
      geometry_epochs_body_provenance.ndjson  one row per geometry epoch / body
      render_consumption.ndjson            one row per full-duration render output
      graph_health.ndjson                  graph-health row(s) carried verbatim
      motion_coupling.ndjson               placeholder (channel not yet produced)

Manifest contract (what makes this a real consumer, not a schema container):

  - input_hashes: sha256 of every durable source file the package was built from
  - render_artifacts: paths + sha256 + frame_count + fps of v19_overlay/world/
    side_by_side full-duration mp4s
  - frame_count: 150 (full duration of the raw clip)
  - zero_confirmed_contact_frames: true; confirmed_contact count == 0
  - defaulted_frame_policy: how the 129 frames outside the kill-test window are
    assigned a documented default state (unresolved_evidence_incomplete)
  - decisive_frame_states: f32/f36/f45 contact states, byte-derivable from the
    contact_frame_detail table
  - reproduce_f32_f36_f45_from_table: explicit true/false gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_FAMILY = "ego.hoi"
SCHEMA_VERSION = "0.1.0"
NAMESPACE = "org.ego.hoi"
RUN_ID = "clip001850_research"
PRODUCER = "build_clip001850_ego_hoi_sidecar_package.py"
PRODUCER_VERSION = "r20-sidecar-package-1"

DEFAULT_ARTIFACT_ROOT = Path(
    "/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706"
)
DEFAULT_OUT_DIR = Path(
    "/tmp/clip001850_ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research"
)

FRAME_COUNT = 150
FPS = 30.0
DEFAULT_CONTACT_STATE = "unresolved_evidence_incomplete"
HAND_SIDE = "right"
WINDOW_LO = 28
WINDOW_HI = 48

# f32/f36/f45 are the decisive review frames whose contact_state must be
# byte-reproducible from the contact_frame_detail table. These are the
# canonical-named states from the durable artifact_manifest.json.
DECISIVE_FRAMES = {
    32: "geometry_epoch_contaminated",
    36: "full_frame_depth_leak",
    45: "unresolved_incoherent_evidence",
}


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Source artifacts
# --------------------------------------------------------------------------- #
class Sources:
    """Resolved durable source paths and their loaded contents + hashes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        # Canonical contact table
        self.contact_ndjson = root / "contact_state_table" / "contact_frame_detail.ndjson"
        self.contact_summary = root / "contact_state_table" / "summary.json"
        # Body repair
        self.body_repair_summary = root / "keyboard_body_repair" / "summary.json"
        self.body_face_prov = root / "keyboard_body_repair" / "face_provenance_freespace_summary.json"
        self.observed_body_ply = root / "keyboard_body_repair" / "repaired_observed_contact_body.ply"
        self.carved_body_ply = root / "keyboard_body_repair" / "repaired_carved_support_body.ply"
        # Hand-depth counterfactual
        self.hand_depth_summary = root / "hand_depth_bias_counterfactual" / "summary.json"
        self.hand_depth_per_frame = root / "hand_depth_bias_counterfactual" / "per_frame.ndjson"
        # Graph health
        self.graph_health_json = root / "egohoi_graph_health_provenance" / "ego_hoi" / "graph_health.json"
        self.graph_health_ndjson = root / "egohoi_graph_health_provenance" / "ego_hoi" / "graph_health.ndjson"
        self.graph_health_manifest = root / "egohoi_graph_health_provenance" / "ego_hoi" / "manifest.json"
        self.factor_graph_summary = root / "egohoi_graph_health_provenance" / "clip001850_factor_graph_summary.json"
        # Full-duration render
        self.fd_dir = root / "v19_contact_state_full_duration"
        self.fd_manifest = self.fd_dir / "manifest.json"
        self.fd_overlay = self.fd_dir / "v19_overlay.mp4"
        self.fd_world = self.fd_dir / "v19_world.mp4"
        self.fd_side_by_side = self.fd_dir / "v19_side_by_side.mp4"
        # Top-level artifact manifest
        self.artifact_manifest = root / "artifact_manifest.json"

        for p in self._required():
            if not p.exists():
                raise FileNotFoundError(f"required durable source missing: {p}")

        # Load contents used by table builders.
        self.contact_rows = load_ndjson(self.contact_ndjson)
        self.contact_summary_obj = load_json(self.contact_summary)
        self.body_repair_obj = load_json(self.body_repair_summary)
        self.body_face_prov_obj = load_json(self.body_face_prov)
        self.hand_depth_obj = load_json(self.hand_depth_summary)
        self.graph_health_rows = load_ndjson(self.graph_health_ndjson)
        self.fd_manifest_obj = load_json(self.fd_manifest)
        self.artifact_manifest_obj = load_json(self.artifact_manifest)
        try:
            self.hand_depth_per_frame_rows = load_ndjson(self.hand_depth_per_frame)
        except FileNotFoundError:
            self.hand_depth_per_frame_rows = []

    def _required(self) -> list[Path]:
        return [
            self.contact_ndjson,
            self.contact_summary,
            self.body_repair_summary,
            self.body_face_prov,
            self.observed_body_ply,
            self.carved_body_ply,
            self.hand_depth_summary,
            self.graph_health_ndjson,
            self.graph_health_manifest,
            self.fd_manifest,
            self.fd_overlay,
            self.fd_world,
            self.fd_side_by_side,
            self.artifact_manifest,
        ]

    def input_hashes(self) -> dict[str, str]:
        """sha256 of every durable source file the package is built from."""
        items: list[tuple[str, Path]] = [
            ("artifact_manifest.json", self.artifact_manifest),
            ("contact_frame_detail.ndjson", self.contact_ndjson),
            ("contact_state_summary.json", self.contact_summary),
            ("keyboard_body_repair_summary.json", self.body_repair_summary),
            ("face_provenance_freespace_summary.json", self.body_face_prov),
            ("repaired_observed_contact_body.ply", self.observed_body_ply),
            ("repaired_carved_support_body.ply", self.carved_body_ply),
            ("hand_depth_bias_counterfactual_summary.json", self.hand_depth_summary),
            ("hand_depth_bias_counterfactual_per_frame.ndjson", self.hand_depth_per_frame),
            ("graph_health.ndjson", self.graph_health_ndjson),
            ("graph_health_manifest.json", self.graph_health_manifest),
            ("v19_contact_state_full_duration_manifest.json", self.fd_manifest),
            ("v19_overlay.mp4", self.fd_overlay),
            ("v19_world.mp4", self.fd_world),
            ("v19_side_by_side.mp4", self.fd_side_by_side),
        ]
        if self.factor_graph_summary.exists():
            items.append(("clip001850_factor_graph_summary.json", self.factor_graph_summary))
        out: dict[str, str] = {}
        for role, path in items:
            if path.exists():
                out[role] = sha256_file(path)
            else:
                out[role] = "MISSING"
        return out


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def build_contact_frame_detail_table(src: Sources) -> list[dict[str, Any]]:
    """One row per frame (0..149). The 21 kill-test-window rows are lifted from
    the canonical table (carrying their measured evidence); the 129 uncovered
    frames carry the documented default state with source=default_policy_uncovered.
    """
    # Index canonical rows by frame_idx (right hand only; the slice is right-hand).
    canonical_by_frame: dict[int, dict[str, Any]] = {}
    for row in src.contact_rows:
        if row.get("hand_side") != HAND_SIDE:
            continue
        canonical_by_frame[int(row["frame_idx"])] = row

    rows: list[dict[str, Any]] = []
    for frame_idx in range(FRAME_COUNT):
        canonical = canonical_by_frame.get(frame_idx)
        if canonical is not None:
            # Carry the canonical contact_state and the measured evidence fields
            # that make the state reproducible. Keep the full row under
            # canonical_row for audit; surface the consumer-facing fields.
            row = {
                "schema": "ego.hoi.contact_frame_detail/0.1.0",
                "run_id": RUN_ID,
                "frame_idx": frame_idx,
                "hand_side": HAND_SIDE,
                "contact_state": canonical.get("contact_state"),
                "source": "contact_frame_detail",
                "object_body_provenance": canonical.get("object_body_provenance"),
                "contact_state_basis": canonical.get("contact_state_basis"),
                "signed_gap_m": (canonical.get("source_gap") or {}).get("signed_gap_m"),
                "source_gap_z": (canonical.get("source_gap") or {}).get("source_gap_z"),
                "observed_masked_penetrating_count": canonical.get("observed_masked_penetrating_count"),
                "observed_masked_near_count": canonical.get("observed_masked_near_count"),
                "raw_penetrating_count": canonical.get("raw_penetrating_count"),
                "survival_fraction": canonical.get("survival_fraction"),
                "interval_published_penetration_max_m": canonical.get("interval_published_penetration_max_m"),
                "render_consumed_mesh_sha256": canonical.get("render_consumed_mesh_sha256"),
                "combined_metric_uncertainty_sigma_m": canonical.get("combined_metric_uncertainty_sigma_m"),
                "sigma_clip_range_m": canonical.get("sigma_clip_range_m"),
                "pose_provenance": (canonical.get("pose") or {}).get("provenance"),
                "pose_source": (canonical.get("pose") or {}).get("source"),
                "renderer_consumes_contact_state": canonical.get("renderer_consumes_contact_state"),
                "contact_surface_mesh_sha256": (canonical.get("contact_surface_mesh") or {}).get("sha256"),
                "contact_surface_mesh_watertight": (canonical.get("contact_surface_mesh") or {}).get("watertight"),
                "face_provenance_global": canonical.get("face_provenance_global"),
            }
        else:
            # Documented default for frames outside the kill-test window.
            row = {
                "schema": "ego.hoi.contact_frame_detail/0.1.0",
                "run_id": RUN_ID,
                "frame_idx": frame_idx,
                "hand_side": HAND_SIDE,
                "contact_state": DEFAULT_CONTACT_STATE,
                "source": "default_policy_uncovered",
                "object_body_provenance": "geometry_epoch_contaminated",
                "contact_state_basis": "default: frame outside kill-test window (28-48); no admissible contact evidence evaluated",
                "signed_gap_m": None,
                "source_gap_z": None,
                "observed_masked_penetrating_count": None,
                "observed_masked_near_count": None,
                "raw_penetrating_count": None,
                "survival_fraction": None,
                "interval_published_penetration_max_m": None,
                "render_consumed_mesh_sha256": None,
                "combined_metric_uncertainty_sigma_m": None,
                "sigma_clip_range_m": None,
                "pose_provenance": None,
                "pose_source": None,
                "renderer_consumes_contact_state": False,
                "contact_surface_mesh_sha256": None,
                "contact_surface_mesh_watertight": None,
                "face_provenance_global": None,
            }
        rows.append(row)
    return rows


def build_geometry_epochs_body_provenance_table(src: Sources) -> list[dict[str, Any]]:
    """One row per geometry epoch / body variant, with face provenance, the
    consumer that uses it, and watertight / admissibility flags.
    """
    body = src.body_repair_obj.get("body_summaries", {})
    face_prov = src.body_face_prov_obj
    completed = body.get("completed_mesh", {})
    observed = body.get("observed_only_body", {})
    carved = body.get("carved_body", {})
    free_space = body.get("free_space_carve", {})

    # Cross-solver epoch ids from the graph-health row (solver / contact / render).
    gh_row = src.graph_health_rows[0] if src.graph_health_rows else {}
    csg = gh_row.get("cross_solver_geometry_consistency") or gh_row.get("mechanism_decision", {}).get("cross_solver_geometry") or {}

    rows: list[dict[str, Any]] = []

    # 1. Completed TRELLIS mesh (contact-ineligible epoch).
    completed_sha = (src.contact_rows[0].get("contact_surface_mesh") or {}).get("sha256") if src.contact_rows else None
    rows.append({
        "schema": "ego.hoi.geometry_epochs_body_provenance/0.1.0",
        "run_id": RUN_ID,
        "epoch_id": csg.get("contact_query_geometry_epoch_id") or "epoch_contact_e44915052d53",
        "body_variant": "completed_trellis_mesh",
        "role": "contact_query_geometry (interval solver + contact/NP signed query)",
        "sha256": completed_sha,
        "path": (src.contact_rows[0].get("contact_surface_mesh") or {}).get("path") if src.contact_rows else None,
        "face_count": completed.get("total_faces"),
        "vertex_count": None,
        "watertight": completed.get("watertight"),
        "aabb_extents_m": completed.get("aabb_extents_m"),
        "extent_ratio_to_keyboard_prior": completed.get("extent_ratio_to_keyboard_prior"),
        "face_label_counts": completed.get("face_label_counts"),
        "face_provenance_fractions": {
            "observed": face_prov.get("completed_mesh", {}).get("face_label_counts", {}).get("observed_depth_surface", 0) / max(1, completed.get("total_faces", 1)),
        },
        "admissible_for_contact": False,
        "admissibility_reason": "non-watertight, 95.4% TRELLIS-inferred hidden surface, free-space uncarved at query time; mesh penetration is not contact-admissible",
        "consumer": "interval_solver_signed_query (REFUTED by masked depth kill-tests)",
    })

    # 2. Observed-only body (render-consumed repaired body).
    observed_sha = sha256_file(src.observed_body_ply)
    rows.append({
        "schema": "ego.hoi.geometry_epochs_body_provenance/0.1.0",
        "run_id": RUN_ID,
        "epoch_id": "epoch_observed_body_" + observed_sha[:12],
        "body_variant": "observed_only_body",
        "role": "full-duration replacement render body (hatched/uncertain patch)",
        "sha256": observed_sha,
        "path": str(src.observed_body_ply),
        "face_count": observed.get("faces"),
        "vertex_count": observed.get("vertices"),
        "watertight": observed.get("watertight"),
        "aabb_extents_m": observed.get("aabb_extents_m"),
        "extent_ratio_to_keyboard_prior": None,
        "face_label_counts": {"observed_depth_surface": observed.get("faces")},
        "face_provenance_fractions": {"observed": 1.0},
        "admissible_for_contact": False,
        "admissibility_reason": "open singly-observed surface patch; unsigned distance only, no watertight sign mesh; observed distances 35.6-115.2 mm do not reach contact",
        "consumer": "v19_contact_state_full_duration render (render_consumed_mesh_sha256 in full-duration manifest)",
    })

    # 3. Carved support body.
    carved_sha = sha256_file(src.carved_body_ply)
    rows.append({
        "schema": "ego.hoi.geometry_epochs_body_provenance/0.1.0",
        "run_id": RUN_ID,
        "epoch_id": "epoch_carved_body_" + carved_sha[:12],
        "body_variant": "carved_support_body",
        "role": "free-space-carved support body (TRELLIS faces removed where observed depth violates)",
        "sha256": carved_sha,
        "path": str(src.carved_body_ply),
        "face_count": carved.get("faces"),
        "vertex_count": carved.get("vertices"),
        "watertight": carved.get("watertight"),
        "aabb_extents_m": carved.get("aabb_extents_m"),
        "extent_ratio_to_keyboard_prior": None,
        "face_label_counts": None,
        "face_provenance_fractions": None,
        "admissible_for_contact": False,
        "admissibility_reason": "still non-watertight (back/sides/bottom never observed); thin axis unchanged; not a contact body",
        "consumer": "research audit only (not rendered into full-duration artifact)",
    })

    # 4. Solver / render geometry epochs from graph-health cross-solver block.
    for role_key, epoch_key, source_key in [
        ("solver", "solver_geometry_epoch_id", "solver_geometry_source_family"),
        ("contact_query", "contact_query_geometry_epoch_id", "contact_query_geometry_source_family"),
        ("render", "render_geometry_epoch_id", "render_geometry_source_family"),
    ]:
        epoch_id = csg.get(epoch_key)
        if not epoch_id:
            continue
        rows.append({
            "schema": "ego.hoi.geometry_epochs_body_provenance/0.1.0",
            "run_id": RUN_ID,
            "epoch_id": epoch_id,
            "body_variant": f"{role_key}_geometry_epoch",
            "role": f"{role_key} stage geometry lineage (graph_health cross_solver_geometry)",
            "sha256": (csg.get(f"{role_key.replace('contact_query', 'contact_query')}_geometry_provenance") or {}).get("sign_mesh_sha256")
                      or (csg.get("contact_query_geometry_provenance") or {}).get("sign_mesh_sha256"),
            "path": None,
            "face_count": None,
            "vertex_count": None,
            "watertight": csg.get("watertight"),
            "aabb_extents_m": None,
            "extent_ratio_to_keyboard_prior": None,
            "face_label_counts": None,
            "face_provenance_fractions": None,
            "admissible_for_contact": False,
            "admissibility_reason": "cross_solver_geometry lineage record; geometry_epoch_consistent=False, geometry_source_consistent=False",
            "consumer": f"graph_health.{role_key}_stage",
            "geometry_source_family": csg.get(source_key),
            "mismatch_reasons": csg.get("mismatch_reasons"),
        })

    return rows


def build_render_consumption_table(src: Sources) -> list[dict[str, Any]]:
    """One row per full-duration render output, carrying the path, hash, the
    consumed contact table, the consumed mesh, and decisive frame states.
    """
    fd = src.fd_manifest_obj
    consumed_mesh_sha = fd.get("render_consumed_mesh_sha256")
    consumed_mesh_faces = fd.get("render_consumed_mesh_faces")
    contact_table_sha = (fd.get("inputs") or {}).get("contact_state_table_sha256")
    contact_table_path = (fd.get("inputs") or {}).get("contact_state_table")
    body_not_trellis = fd.get("body_not_trellis")
    decisive = fd.get("decisive_frame_states") or {}
    acceptance = fd.get("acceptance_gates") or {}
    counts = fd.get("contact_state_counts") or {}

    rows: list[dict[str, Any]] = []
    for role, path in [
        ("overlay", src.fd_overlay),
        ("world", src.fd_world),
        ("side_by_side", src.fd_side_by_side),
    ]:
        rows.append({
            "schema": "ego.hoi.render_consumption/0.1.0",
            "run_id": RUN_ID,
            "render_role": role,
            "path": str(path),
            "sha256": sha256_file(path),
            "frame_count": fd.get("frame_count", FRAME_COUNT),
            "fps": fd.get("fps", FPS),
            "duration_s": fd.get("duration_s"),
            "consumed_contact_state_table": contact_table_path,
            "consumed_contact_state_table_sha256": contact_table_sha,
            "consumed_observed_body_sha256": consumed_mesh_sha,
            "consumed_observed_body_faces": consumed_mesh_faces,
            "body_not_trellis": body_not_trellis,
            "default_contact_state": fd.get("default_contact_state"),
            "defaulted_frame_count": fd.get("defaulted_frame_count"),
            "contact_state_counts": counts,
            "zero_confirmed_contact_frames": fd.get("zero_confirmed_contact_frames"),
            "decisive_frame_states": decisive,
            "acceptance_gates": acceptance,
            "full_duration_manifest": str(src.fd_manifest),
            "full_duration_manifest_sha256": sha256_file(src.fd_manifest),
            "pixel_diff_vs_published": (fd.get("pixel_diff_vs_published") or {}).get(role),
        })
    return rows


def build_graph_health_table(src: Sources) -> list[dict[str, Any]]:
    """Carry the existing graph-health row(s) verbatim, annotated with run_id."""
    rows: list[dict[str, Any]] = []
    for row in src.graph_health_rows:
        out = dict(row)
        out["sidecar_run_id"] = RUN_ID
        out["source_path"] = str(src.graph_health_ndjson)
        rows.append(out)
    return rows


def build_motion_coupling_table(src: Sources) -> list[dict[str, Any]]:
    """Placeholder: the motion-coupling channel is not yet produced on this
    slice. We emit one explicit placeholder row stating the contract, so the
    downstream consumer knows the channel is absent rather than empty-by-query.
    """
    return [{
        "schema": "ego.hoi.motion_coupling/0.1.0",
        "run_id": RUN_ID,
        "status": "not_produced",
        "reason": (
            "Motion-coupling channel (object-motion onset time-locked to hand "
            "kinematics) is the named floor-independent route to promote a frame "
            "to confirmed_contact. It is not yet produced on this slice. Expected "
            "signal is weak: the keyboard barely moves under right-hand approach. "
            "GT-free ceiling for this slice is unresolved / contact_candidate at "
            "most. See EPISTEMIC.md next-artifact-changing route and subagent 16 "
            "eval summary promote-to-confirmed_contact contract."
        ),
        "contract": {
            "required_signal": "object motion onset vs hand velocity/acceleration, per frame",
            "admissible_promotion": "confirmed_contact only with a populated motion_coupling provenance column",
            "inadmissible": "signed-distance thresholds, gap-magnitude hand shifts, 2D-reprojection changes",
        },
        "frame_count": FRAME_COUNT,
        "frames_with_motion_coupling": 0,
    }]


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def build_manifest(
    src: Sources,
    contact_rows: list[dict[str, Any]],
    render_rows: list[dict[str, Any]],
    out_dir: Path,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    # Contact-state distribution across the full 150-frame table.
    state_counts: dict[str, int] = {}
    for r in contact_rows:
        state_counts[r["contact_state"]] = state_counts.get(r["contact_state"], 0) + 1
    confirmed_count = state_counts.get("confirmed_contact", 0)

    # f32/f36/f45 reproducibility check: the contact_frame_detail table must
    # yield the decisive states.
    by_frame = {int(r["frame_idx"]): r for r in contact_rows}
    reproduce_ok = True
    reproduced: dict[str, str] = {}
    for fidx, expected in DECISIVE_FRAMES.items():
        got = (by_frame.get(fidx) or {}).get("contact_state")
        reproduced[str(fidx)] = got or "MISSING"
        if got != expected:
            reproduce_ok = False

    fd = src.fd_manifest_obj
    render_artifacts = {
        role: {
            "path": str(getattr(src, f"fd_{('side_by_side' if role == 'side_by_side' else role)}")),
            "sha256": next((r["sha256"] for r in render_rows if r["render_role"] == role), None),
            "frame_count": fd.get("frame_count", FRAME_COUNT),
            "fps": fd.get("fps", FPS),
            "consumed_observed_body_sha256": fd.get("render_consumed_mesh_sha256"),
            "consumed_contact_state_table_sha256": (fd.get("inputs") or {}).get("contact_state_table_sha256"),
        }
        for role in ("overlay", "world", "side_by_side")
    }

    return {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "run_id": RUN_ID,
        "artifact_role": "clip001850 research-slice ego.hoi sidecar package",
        "provenance": {
            "producer": PRODUCER,
            "producer_version": PRODUCER_VERSION,
            "created_at_unix": time.time(),
            "durable_artifact_root": str(src.root),
            "source_run_root": src.contact_summary_obj.get("run_root"),
            "source_case": src.contact_summary_obj.get("case"),
        },
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "hand_side": HAND_SIDE,
        "kill_test_window": {"frame_lo": WINDOW_LO, "frame_hi": WINDOW_HI},
        "input_hashes": input_hashes,
        "input_hashes_sha256": sha256_json_canonical(input_hashes),
        "render_artifacts": render_artifacts,
        "render_artifacts_full_duration_manifest": str(src.fd_manifest),
        "render_consumed_mesh_sha256": fd.get("render_consumed_mesh_sha256"),
        "render_consumed_mesh_faces": fd.get("render_consumed_mesh_faces"),
        "body_not_trellis": fd.get("body_not_trellis"),
        "contact_state_counts_full_duration": state_counts,
        "zero_confirmed_contact_frames": confirmed_count == 0,
        "confirmed_contact_count": confirmed_count,
        "defaulted_frame_policy": {
            "default_contact_state": DEFAULT_CONTACT_STATE,
            "defaulted_frame_count": fd.get("defaulted_frame_count", FRAME_COUNT - (WINDOW_HI - WINDOW_LO + 1)),
            "policy": (
                "Frames inside the kill-test window (28-48) carry the canonical "
                "contact_state from the durable contact_frame_detail table. The "
                "remaining frames carry unresolved_evidence_incomplete: they were "
                "not evaluated by the masked-depth kill-tests and no admissible "
                "contact evidence exists for them. They are not dropped and not "
                "fabricated as contact."
            ),
            "source": "v19_contact_state_full_duration/manifest.json default_contact_state",
        },
        "decisive_frame_states": DECISIVE_FRAMES,
        "reproduce_f32_f36_f45_from_table": reproduce_ok,
        "reproduced_frame_states": reproduced,
        "tables": [
            {"name": "contact_frame_detail", "path": "tables/contact_frame_detail.ndjson",
             "format": "ndjson", "row_count": len(contact_rows)},
            {"name": "geometry_epochs_body_provenance", "path": "tables/geometry_epochs_body_provenance.ndjson",
             "format": "ndjson", "row_count": "see file"},
            {"name": "render_consumption", "path": "tables/render_consumption.ndjson",
             "format": "ndjson", "row_count": len(render_rows)},
            {"name": "graph_health", "path": "tables/graph_health.ndjson",
             "format": "ndjson", "row_count": "see file"},
            {"name": "motion_coupling", "path": "tables/motion_coupling.ndjson",
             "format": "ndjson", "row_count": 1, "status": "placeholder_not_produced"},
        ],
        "contact_verdict": {
            "summary": "zero admissible GT-free contact frames on clip001850 right-hand keyboard",
            "geometry_repair_decision": src.body_repair_obj.get("decision"),
            "hand_depth_counterfactual_verdict": src.hand_depth_obj.get("overall_verdict"),
            "promote_to_confirmed_contact_contract": (
                "floor-independent channel only (motion coupling or HOT3D GT/R8); "
                "signed-distance thresholds, gap-magnitude hand shifts, and "
                "2D-reprojection changes are inadmissible"
            ),
        },
        "out_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_clip001850_ego_hoi_sidecar_package.py",
        description=(
            "Package the clip001850 research slice into an ego.hoi 0.1.0 sidecar "
            "extension. Reads durable artifacts only; no model/GPU/heavy inference."
        ),
    )
    p.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="durable research artifact root (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output package directory (default: %(default)s)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Sources(args.artifact_root)
    out_dir = args.out_dir
    tables_dir = out_dir / "tables"

    contact_rows = build_contact_frame_detail_table(src)
    geom_rows = build_geometry_epochs_body_provenance_table(src)
    render_rows = build_render_consumption_table(src)
    graph_rows = build_graph_health_table(src)
    motion_rows = build_motion_coupling_table(src)

    write_ndjson(tables_dir / "contact_frame_detail.ndjson", contact_rows)
    write_ndjson(tables_dir / "geometry_epochs_body_provenance.ndjson", geom_rows)
    write_ndjson(tables_dir / "render_consumption.ndjson", render_rows)
    write_ndjson(tables_dir / "graph_health.ndjson", graph_rows)
    write_ndjson(tables_dir / "motion_coupling.ndjson", motion_rows)

    input_hashes = src.input_hashes()
    manifest = build_manifest(src, contact_rows, render_rows, out_dir, input_hashes)
    write_json(out_dir / "manifest.json", manifest)

    # Console summary.
    print(f"[ego.hoi sidecar] wrote {out_dir}/manifest.json")
    for t in manifest["tables"]:
        print(f"[ego.hoi sidecar]   tables/{t['path'].split('/')[-1]}")
    print(f"[ego.hoi sidecar] frame_count={manifest['frame_count']}")
    print(f"[ego.hoi sidecar] zero_confirmed_contact_frames={manifest['zero_confirmed_contact_frames']} "
          f"(confirmed_contact_count={manifest['confirmed_contact_count']})")
    print(f"[ego.hoi sidecar] reproduce_f32_f36_f45_from_table={manifest['reproduce_f32_f36_f45_from_table']}")
    print(f"[ego.hoi sidecar] contact_state_counts_full_duration={manifest['contact_state_counts_full_duration']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
