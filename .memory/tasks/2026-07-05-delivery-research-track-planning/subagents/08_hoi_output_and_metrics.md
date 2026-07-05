# Subagent 8 — HOI research extension: outputs, ontology, metrics

Research-track companion to `ego.delivery.output`. Defines how a hand–object interaction
(HOI) layer publishes its state, evidence, and metrics **without contaminating the delivery
base schema**. Every claim below is opt-in through the extension namespace; absence of the
extension MUST leave a delivery run byte-identical to v1.0.0.

## Scope and isolation contract

- **Extension namespace (reverse-DNS):** `org.ego.research.hoi`; **schema family:**
  `ego.research.hoi`; **initial version:** `0.1.0` (pre-1.0; breaking changes allowed until
  the first promotion closes the gate, §10).
- **Base coupling:** reads delivery substrates (frames, depth, masks, hand rows, intrinsics,
  camera trajectory) as **read-only pinned-version inputs**. MUST NOT add columns to, rename,
  or alter semantics of any `ego.delivery.output` table. Produces its own tables/streams/renders
  under the extension namespace; joins to delivery only via stable keys (`job_id`, `frame_index`,
  `hand_track_id`, `clip_id`).
- **Compute isolation:** A800/offline branches, own run root; never on the delivery serving
  path or the user workstation (runtime invariant).
- **Monotonicity to research v18+ base:** preserves metric-MANO hand state and visible-surface
  geometry as anchors; new HOI state is layered additional hypotheses with explicit uncertainty.

## File layout

Lives alongside delivery output, addressed only through `manifest.extensions[]`:

```text
{root}/delivery/{schema_version}/{job_id}/extensions/org.ego.research.hoi/0.1.0/
  hoi_manifest.json
  schemas/                                   # JSON Schema + Arrow snapshots
  tables/{object_instances,object_geometry_epochs,object_pose,rigidity_state,
          contact_state,occlusion_state,hand_corrections,hoi_validation_metrics}.parquet
  stream/{hoi_overlay_events,hoi_provenance_events}.ndjson
  assets/object_meshes/{object_id}/{mesh_v{n}.ply, face_provenance.parquet}  # watertight; observed|interpolated|prior-completed
  renders/{hoi_overlay,hoi_side_by_side,hoi_world}.mp4   # full-duration, same frame count as input
```

Delivery artifacts under `{root}/.../` are unchanged. `hoi_manifest.json` mirrors the delivery
manifest discipline: schema name/version, `extensions[]` lineage, substrate pins (delivery
version, model weights, config hashes), artifact index with sha256/byte_size/row_count, and a
`promotion_status` block (default `not_promoted`).

## Renderable artifact requirement (mandatory)

Every complete HOI run MUST emit full-duration `hoi_overlay.mp4` and `hoi_side_by_side.mp4`
with frame count and duration equal to the delivery input video. Short windows, contact
slices, debug clips, and selected-frame renders are QC artifacts only. The overlay is driven
by `hoi_overlay_events.ndjson` layers pointing back to HOI table rows, mirroring the delivery
overlay↔table contract. A run shipping tables without a full-duration, source-traceable render
is incomplete.

## Coordinate frames

Inherits delivery named frames (`image_px`, `camera_t`, `world_w0`, `mano_left/right`). Adds
`object_canonical_{object_id}` (per-instance canonical mesh frame, declared gauge with
provenance) and optional `hoi_world_{epoch}` (gravity-aligned/anchored world variant when a
drift-correction epoch is solved). Frame changes are 0.x-minor breaking events until 1.0.

## Ontology

### `object_instances.parquet` — one row per object instance per clip
Object identity, hypothesis source, and roster lineage. Join key `object_id` (stable per run).

| Column | Type | Notes |
|---|---|---|
| `object_id` | string | Stable instance id. |
| `label_openvocab` | string | Open-vocabulary noun from detector/VLM. |
| `hypothesis_sources` | list<enum> | `semantic_owlv2`, `kinematic_rigid_group`, `caption_noun`, `manual`. At least one required. |
| `track_state` | enum | `tracked`, `intermittent`, `lost`, `static_inferred`. |
| `first_frame` / `last_frame_exclusive` | int64 | Timeline span. |
| `delivery_clip_link` | list<string> | `clip_id`s where it appears; for caption consistency (delivery S5). |
| `provenance_refs` | list<string> | Into `hoi_provenance_events.ndjson`. |

### `object_geometry_epochs.parquet` — shape provenance per epoch
Replaces single-anchor TRELLIS completion with labeled, multi-frame-fused geometry epochs.

| Column | Type | Notes |
|---|---|---|
| `object_id`, `epoch_id` | string | `epoch_id` = `{start_frame}_{end_frame_exclusive}`. |
| `mesh_uri` | string | `assets/object_meshes/.../mesh_v{n}.ply`. |
| `watertight` | bool | Required true for signed-distance claims; else mark nonpenetration unresolved. |
| `observed_face_fraction` | float32 | Fraction labeled `observed` in `face_provenance.parquet`. |
| `prior_completed_face_fraction` | float32 | TRELLIS-filled, never-seen regions. |
| `fusion_method` | enum | `single_anchor_trellis`, `multiframe_tsdf`, `surfel`, `hybrid`. |
| `anchor_frames` | list<int64> | Frames used for prior conditioning. |
| `chamfer_visible_mm` | float32 nullable | When GT CAD exists (HOT3D). |
| `silhouette_iou_p50` | float32 nullable | All-frame projected IoU vs SAM2 mask (delivery R3 shared). |
| `uncertainty_band` | enum | `high` (<30% observed), `medium`, `low` (>70% observed). |

### `object_pose.parquet` — one row per `(frame_index, object_id)`
Pose trajectory from correspondence-first estimation (Sub7 §3.3); depth/silhouette are
secondary consistency channels, not the primary fit.

| Column | Type | Notes |
|---|---|---|
| `frame_index`, `object_id` | — | Join keys. |
| `T_camera_from_object` | fixed_size_list<float64,16> | Row-major. |
| `q_camera_from_object_xyzw` | fixed_size_list<float64,4> | Convenience. |
| `pose_status` | enum | `estimated_correspondence`, `estimated_depth_icp`, `interpolated`, `inferred_from_contact`, `unresolved`. |
| `primary_observation_channel` | enum | `texture_tracks`, `depth_icp`, `silhouette`, `hybrid`. Null-space directions (in-plane rot/trans of planar surfaces) MUST be marked `unresolved` unless texture tracks observe them (B1). |
| `translation_sigma_m`, `rotation_sigma_rad` | fixed_size_list<float32,3> | From fitted noise (B7). |
| `track_count`, `track_median_lifetime_frames` | int32 | Correspondence evidence density. |
| `residual_to_gt_mm` / `residual_to_gt_deg` | float32 nullable | GT only. |
| `provenance_refs` | list<string> | |

### `rigidity_state.parquet` — measured rigidity (Sub7 B5, E1)
Converts VLM rigid/deformable *declaration* into a *measurement*.

| Column | Type | Notes |
|---|---|---|
| `object_id`, `epoch_id` | string | |
| `rigidity_class` | enum | `rigid`, `articulated_multi_part`, `deformable`, `unresolved`. |
| `procrustes_residual_mm` | float32 | Best per-frame SE(3) residual over tracked point subset. |
| `pairwise_distance_drift_mm` | float32 | Mean pairwise-distance conservation error. |
| `track_subset_count` | int32 | Number of low-residual clusters (articulation evidence). |
| `deformation_event_frames` | list<int64> | Residual-burst frames (e.g. tomato peel). |
| `declared_vs_measured_agreement` | enum | `agree`, `disagree_promotes_to_deformable`, `disagree_other`, `unverified`. |

### `contact_state.parquet` — one row per `(frame_index, object_id, hand_track_id)` where hypothesis is live
Contact is a switch posterior (Sub7 §4, E7), never a hard nonpenetration=0 clamp.

| Column | Type | Notes |
|---|---|---|
| keys | — | frame/object/hand. |
| `contact_hypothesis` | enum | `contact`, `near`, `none`, `unresolved`. |
| `posterior_p_contact` / `posterior_p_near` | float32 | Switch mixture posterior. |
| `evidence_channels` | list<enum> | `learned_contact_map`, `depth_order_render_compare`, `object_motion_onset`, `visible_mask_overlap` (flagged structurally-empty under manipulation). |
| `min_signed_penetration_mm` | float32 nullable | Negative = gap; small positive expected at true contact (soft tissue). Undefined when `!watertight`. |
| `contact_sigma_mm` | float32 | Fitted per-channel (B7); replaces hand-set 30.48 mm constant. |
| `depth_order_hand_vs_object` | enum | `hand_in_front`, `object_in_front`, `ambiguous`, `unresolved`. |
| `provenance_refs` | list<string> | |

### `occlusion_state.parquet` — explicit visibility ownership
Satisfies the project occlusion invariant. One row per `(frame_index, object_id)` and per hand.

| Column | Type | Notes |
|---|---|---|
| keys | — | frame/object, `hand_track_id` nullable for hand rows. |
| `visibility_state` | enum | `visible`, `partially_visible`, `occluded_inferred`, `out_of_frame`, `unresolved`. |
| `primary_occluder_ids` | list<string> | Owning object/hand ids. |
| `depth_order_confidence` | float32 | |
| `inferred_from` | list<enum> | `temporal_stitch`, `depth_order`, `contact_kinematic`. |
| `uncertainty_band` | enum | high/medium/low. |

### `hand_corrections.parquet` — drift latent corrections (B4, E5)
The one designed coupling to delivery hand accuracy. Joins to delivery `hand_states`; never
mutates it. Promotion to delivery happens only through the gate (§10).

| Column | Type | Notes |
|---|---|---|
| `frame_index`, `hand_track_id` | — | |
| `drift_latent_id` | string | Per-clip spline block id. |
| `delta_t_camera_m`, `delta_q_camera_xyzw` | — | SE(3)-ish correction applied to delivery hand root. |
| `latent_dof` | int32 | Effective dof (~8–24). |
| `anchor_channels` | list<enum> | `cross_detector_reproj`, `size_consistency`, `static_scene_features`, `contact_events`. |
| `anchor_residual_px` / `anchor_residual_ratio` | float32 | GT-free anchor fit residual. |
| `applied_to_delivery` | bool | Always false in-research; flips only post-promotion. |
| `pre_correction_gt_mm` / `post_correction_gt_mm` | float32 nullable | GT-only diagnostic. |

## Streams

### `hoi_overlay_events.ndjson`
Same discipline as delivery `overlay_events.ndjson`: one object per frame, `layers[]` pointing
back to a HOI table row via `source_table` + `source_row_key`. Layer types add `object_contour_2d`,
`object_pose_axes_3d`, `contact_patch_2d` (uncertainty-styled), `rigidity_band`,
`drift_correction_arrow`. Uncertainty encoded via alpha/dash/label, never hidden. Every visible
HOI mark in `hoi_overlay.mp4` MUST have a layer; no layer drawn without a backing row.

### `hoi_provenance_events.ndjson`
Per-module/model/config event: track source (CoTracker/SpatialTrackerV2), contact prior
(DeepContact-class), fusion method, solver run with `liveness_asserted=true` (B10), gauge
declaration block, runtime/device class, run-root id.

## Metrics

### GT metrics (HOT3D fixed slice + lockbox)
| Metric | Unit | Threshold | Blind spot |
|---|---|---|---|
| `object_pose_translation_residual_gt_mm` | mm | green ≤20, yellow 20–40, red >40 (B1 baselines 38.85/77.1) | gauge/symmetry; novel-object regime |
| `object_pose_rotation_residual_gt_deg` | deg | green ≤5, yellow 5–10, red >10 | planar null-space, symmetry |
| `object_chamfer_visible_mm` | mm | green ≤15 | GT CAD mesh quality |
| `hand_wrist_residual_gt_mm` (post-correction) | mm | target ≤5, yellow 5–10 | GT-fit proves smoothness not transfer |
| `contact_auroc_vs_proximity` | score | green ≥0.8 | proximity ≠ patch truth; grasp-prior domain gap |
| `rigidity_classification_accuracy` | fraction | green ≥0.9 | small deformation events |

### GT-free metrics (shared currency with delivery, Sub5)
| Metric | Unit | Threshold | Blind spot / routing |
|---|---|---|---|
| `rigidity_procrustes_residual_mm` | mm | keyboard ≪ tomato-peel ≪ trash expected; keyboard green ≤15 | depth-noise floor (B16); route to depth substrate if all ≈ lifted noise |
| `object_silhouette_iou` (delivery R3 shared) | ratio | green >0.5, review 0.3–0.5 | deformable/transparent objects |
| `cross_detector_hand_reproj_px` (delivery H4 shared) | px | green ≤25, red >50 | cannot see depth |
| `projected_to_detected_size_ratio` (delivery H5 shared) | ratio | green 0.9–1.1 | foreshortening |
| `contact_evidence_support_fraction` | fraction | red if contact rows 0/N support → demote to `unresolved` | structural emptiness under occlusion |
| `solver_liveness_diff_count` | count | zero-diff run ⇒ flagged inert (B10) | does not judge correctness |
| `anchor_residual_correlation_with_gt_drift` | ratio | diagnostic; >0.5 ⇒ anchor has signal | requires GT for the correlation only |

Blind-spot routing rules mirror Sub5: hand metrics fail + camera static passes ⇒ hand source;
object IoU fails but hand good ⇒ separate object confidence; intrinsics sweep helps all layers
⇒ adapter fix; one band diverges ⇒ band-local, never global reweight.

## Uncertainty boundaries

1. Every estimated HOI physical field carries `*_status`, `confidence`, sigma fields, and
   quality flags. Missing values are null + explicit status, never zeros.
2. Non-watertight meshes ⇒ `min_signed_penetration_mm` is null and nonpenetration claims are
   `unresolved` (B11). Watertight-with-uncertainty-labels is the buildable path.
3. Contact under manipulation occlusion is structurally under-observed; the contact row must
   expose which channels produced it and mark `visible_mask_overlap` as non-evidential there.
4. Drift corrections are hypotheses anchored GT-free; they are not delivery hand state until
   promoted. `applied_to_delivery=false` always in-research.
5. Gauge freedoms (per-clip world similarity, object canonical frame, camera-convention
   rotation) are declared variables with provenance; "improvements" along an ungauged freedom
   are gauge motion, not progress.

## Promotion gate back to delivery

A HOI mechanism enters `ego.delivery.output` only with ALL of:

1. **GT win** on fixed slice + ≥2 lockbox clips, declared predictions met (not narrated after).
2. **GT-free self-consistency win** on delivery-regime clips, measured with the shared metric
   family (one implementation, versioned together — the designated bridge).
3. **Runtime within delivery budget** (≤2.45 GPU-h/video-h lane share; same-order-of-magnitude
   as input duration as default path).
4. **Regime-transfer evidence** (HOT3D↔egoscale; B15/B18).
5. **Implementation-class review**: solver liveness assert, freshness boundary, gauge
   declaration, mask ownership (B9/B10/B11/B13).

Until all five close, the mechanism ships nowhere and delivery milestones never depend on it.
Promotion is recorded in `hoi_manifest.promotion_status` and triggers a delivery minor-version
bump with the new field added under a delivery-owned namespace (still no base-field mutation).

## Validation protocols and datasets

- **Fixed slice:** HOT3D `001849/001850/001851` (existing evaluator, P37 constant-gauge
  template). Add ≥2 lockbox clips named before any tuning (B18).
- **Regime-transfer:** ≥2 egoscale delivery clips (trash/tomato/phone/window/scissors) scored
  GT-free only.
- **Multi-view probe (B19):** HOT3D synchronized second stream as additional graph channels —
  quantifies single-view information limit.
- **Contact GT scarcity (B17):** HOT3D proximity-derived contact for AUROC; EgoPressure as
  external patch/pressure reference before any contact factor ships.
- **Experiments** (Sub7 E1–E9) declare default-path vs offline-branch status, predicted
  discriminating outcomes, run root; negative results redirect within the chain, never hop to
  an unrelated mechanism.

## Residual risks

1. Novel-object egocentric RGB pose is open (BOP-H3 GigaPose 9.4 AP); pose claims inherit that
   ceiling until correspondence-first estimation is validated (E2/E9).
2. UniDepth metric noise floor 26–45 mm at hand pixels (B16) may swamp the rigidity statistic;
   E1 decides this empirically.
3. Grasp-trained contact priors may mispredict press/hover (B17); E6 gates any contact factor.
4. Three-clip benchmark overfits (B18); lockbox clips mandatory before tuning.
5. Offline teacher creep (HOLD/MagicHOI-class) must stay off the default path; teachers never
   ship as stages.
