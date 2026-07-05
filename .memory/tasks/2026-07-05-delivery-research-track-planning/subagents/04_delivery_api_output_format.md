# Subagent 4 — Delivery API output format proposal

## Scope boundary

This schema is for the **delivery track only**. It returns customer-consumable head/camera state, metric hand state, visible render/overlay data, semantic 2–3s video clips/captions, provenance, uncertainty, and validation metrics. It deliberately excludes HOI/factor-graph outputs: no object pose, contact ownership, nonpenetration, physical interaction factors, or reconstructed object geometry are required fields. If a future research pipeline produces those fields, they belong in an extension namespace and cannot be silently promoted into this delivery contract.

Design target: a direct customer API where the customer can submit videos, poll/receive completion, download stable artifacts, query frame/clip ranges, and reproduce the visible overlays from structured data without reverse-engineering internal pipeline state.

## Versioned contract

**Schema family:** `ego.delivery.output`

**Initial version:** `1.0.0`

**API base path:** `/v1/delivery`

**Canonical output encodings:**

- `manifest.json`: run-level source of truth and artifact index.
- `*.parquet`: typed analytics tables for frames, head/camera, hands, semantic clips, validation metrics.
- `*.ndjson`: appendable/streamable frame overlay events, caption events, provenance events, and errors.
- `*.mp4` / `*.webm`: customer-visible overlay/side-by-side renders.
- `schemas/`: JSON Schema and Arrow/Parquet schema snapshots for the exact version used.

The manifest is the only required entry point. Every table/render is discovered through `manifest.artifacts[]`, not by hard-coded paths.

## Coordinate frames and units

### Global conventions

- Distances: meters (`*_m`) unless a field explicitly ends in `_mm`.
- Angles: radians for stored rotations unless a field explicitly ends in `_deg`.
- Time: seconds from decoded video start (`*_sec`) plus original presentation timestamp where available (`pts_sec`).
- Image coordinates: pixels in the **original decoded video resolution**, origin at top-left pixel center, `x` right, `y` down.
- 3D camera coordinates: OpenCV pinhole convention, `x` right, `y` down, `z` forward, meters.
- Homogeneous transforms: `T_parent_from_child`, row-major 4x4 float64 array; `p_parent = T_parent_from_child @ [p_child_x, p_child_y, p_child_z, 1]`.
- Quaternions: `xyzw` order, unit normalized.
- Clip intervals: half-open `[start_frame, end_frame_exclusive)`.

### Named frames

| Frame id | Definition | Required? | Notes |
|---|---|---:|---|
| `image_px` | Original decoded frame pixel plane | yes | All overlay geometry is in this frame. |
| `camera_t` | Per-frame camera coordinate system at frame `t` | yes | Hand 3D outputs must be expressible here. |
| `world_w0` | World frame anchored to camera frame 0 | yes when camera trajectory is returned | Gauge frame for trajectories; not gravity-guaranteed. |
| `head_t` | Estimated head/rig frame at frame `t` | optional | Required only when head pose differs from camera pose. |
| `mano_left/right` | MANO hand root frame for each hand state | optional per row | Present when MANO parameters are produced. |

Frame changes are major-version changes unless represented as a new named frame alongside the old one.

## Artifact layout

Recommended object-store layout:

```text
{bucket_or_root}/delivery/{schema_version}/{job_id}/
  manifest.json
  input/
    input_metadata.json
    normalized_video_info.json
  schemas/
    manifest.schema.json
    frames.arrow.json
    head_camera.arrow.json
    hand_states.arrow.json
    semantic_clips.arrow.json
    validation_metrics.arrow.json
    overlay_events.schema.json
    provenance_events.schema.json
  tables/
    frames.parquet
    head_camera.parquet
    hand_states.parquet
    semantic_clips.parquet
    validation_metrics.parquet
  stream/
    overlay_events.ndjson
    caption_events.ndjson
  renders/
    overlay.mp4
    side_by_side.mp4
    overlay_lowres.mp4
    thumbnails/
      frame_{frame_index:06d}.jpg
  assets/
    hand_meshes/
      mano_topology.json
      vertices_{hand_track_id}.parquet        # optional if dense meshes are exported
  metrics/
    validation_summary.json
    calibration_summary.json
  provenance/
    provenance.ndjson
    errors.ndjson
```

Each artifact entry includes `logical_name`, `media_type`, `encoding`, `schema_ref`, `uri`, `byte_size`, `sha256`, `row_count` when applicable, and `created_at`.

## API endpoint shapes

### Submit one job

`POST /v1/delivery/jobs`

Request:

```json
{
  "request_id": "cust-req-20260705-00017",
  "input": {
    "video_uri": "s3://customer-bucket/inbox/clip_001.mp4",
    "video_sha256": "6f9c...",
    "content_type": "video/mp4"
  },
  "outputs": {
    "tables": ["frames", "head_camera", "hand_states", "semantic_clips", "validation_metrics"],
    "streams": ["overlay_events", "caption_events", "provenance", "errors"],
    "renders": ["overlay", "side_by_side", "overlay_lowres"],
    "dense_hand_meshes": false
  },
  "quality_profile": "delivery_default_v1",
  "callback": {
    "url": "https://customer.example.com/hooks/ego-delivery",
    "events": ["job.completed", "job.failed"]
  },
  "metadata": {
    "customer_video_id": "clip_001",
    "collection": "pilot_batch_a"
  }
}
```

Response:

```json
{
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "schema_name": "ego.delivery.output",
  "schema_version": "1.0.0",
  "status": "queued",
  "created_at": "2026-07-05T12:00:31Z",
  "status_url": "/v1/delivery/jobs/del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "manifest_url": null
}
```

### Submit a batch

`POST /v1/delivery/batches`

Batch requests wrap the single-job request without callbacks per item unless overridden. The response returns `batch_id`, item-level `job_id`s, accepted/rejected rows, and idempotency conflicts keyed by `request_id`.

### Poll job status

`GET /v1/delivery/jobs/{job_id}`

```json
{
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "status": "completed",
  "schema_version": "1.0.0",
  "progress": {
    "phase": "finalizing_artifacts",
    "frames_total": 1800,
    "frames_processed": 1800
  },
  "quality_summary": {
    "head_camera": {"status": "usable", "median_reprojection_error_px": 1.8},
    "left_hand": {"status": "usable", "median_reprojection_error_px": 7.4, "median_size_ratio_error": 0.08},
    "right_hand": {"status": "low_confidence", "median_reprojection_error_px": 18.6, "median_size_ratio_error": 0.21},
    "captions": {"status": "usable", "clip_count": 31, "median_clip_duration_sec": 2.4}
  },
  "manifest_url": "https://signed.example.com/delivery/1.0.0/del_.../manifest.json",
  "errors_url": "https://signed.example.com/delivery/1.0.0/del_.../provenance/errors.ndjson"
}
```

### Fetch manifest or range data

- `GET /v1/delivery/jobs/{job_id}/manifest`
- `GET /v1/delivery/jobs/{job_id}/artifacts?logical_name=hand_states`
- `GET /v1/delivery/jobs/{job_id}/frames?start_frame=300&end_frame_exclusive=420&include=hands,head_camera,overlay_events`
- `GET /v1/delivery/jobs/{job_id}/clips?start_sec=0&end_sec=60`

Range endpoints are convenience views over immutable artifacts. They must not return fields absent from the manifest schemas.

### Webhook event

```json
{
  "event_type": "job.completed",
  "event_id": "evt_01JZ2CB58R",
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "schema_name": "ego.delivery.output",
  "schema_version": "1.0.0",
  "status": "completed",
  "manifest_url": "https://signed.example.com/delivery/1.0.0/del_.../manifest.json",
  "created_at": "2026-07-05T12:22:13Z"
}
```

## Manifest schema

`manifest.json` is run-level JSON.

Required top-level fields:

```json
{
  "schema_name": "ego.delivery.output",
  "schema_version": "1.0.0",
  "job": {
    "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
    "request_id": "cust-req-20260705-00017",
    "status": "completed",
    "created_at": "2026-07-05T12:00:31Z",
    "completed_at": "2026-07-05T12:22:13Z",
    "producer": {"name": "ego-delivery-pipeline", "version": "delivery-2026.07.05"}
  },
  "input": {
    "customer_video_id": "clip_001",
    "source_uri_redacted": "s3://customer-bucket/inbox/clip_001.mp4",
    "video_sha256": "6f9c...",
    "width_px": 1920,
    "height_px": 1080,
    "frame_count": 1800,
    "duration_sec": 60.0,
    "nominal_fps": 30.0,
    "decode_policy": "constant_frame_index_with_original_pts"
  },
  "coordinate_frames": [
    {"frame_id": "image_px", "type": "image", "units": "px", "origin": "top_left_pixel_center", "x_axis": "right", "y_axis": "down"},
    {"frame_id": "camera_t", "type": "3d", "units": "m", "convention": "opencv_x_right_y_down_z_forward"},
    {"frame_id": "world_w0", "type": "3d", "units": "m", "definition": "camera_t at frame_index 0"}
  ],
  "artifacts": [],
  "quality_summary": {},
  "compatibility": {
    "min_reader_version": "1.0.0",
    "schema_semver": "1.0.0",
    "extensions": []
  }
}
```

## Parquet table schemas

### `tables/frames.parquet`

One row per decoded output frame. This table defines the authoritative frame index.

| Column | Type | Required | Description |
|---|---|---:|---|
| `job_id` | string | yes | Job id. |
| `video_id` | string | yes | Customer or generated video id. |
| `frame_index` | int64 | yes | 0-based contiguous output frame index. |
| `pts_sec` | float64 nullable | yes | Original container PTS if available. |
| `time_sec` | float64 | yes | Timeline time used by all tables. |
| `width_px` / `height_px` | int32 | yes | Decoded dimensions. |
| `decode_status` | enum | yes | `decoded`, `duplicated`, `dropped_recovered`, `decode_failed`. |
| `image_sha256` | string nullable | no | Optional per-frame hash for reproducibility. |
| `blur_score` | float32 nullable | no | Diagnostic only. |
| `exposure_score` | float32 nullable | no | Diagnostic only. |

### `tables/head_camera.parquet`

One row per frame with camera/head state and metrics. Head/camera metrics are kept separate from hand metrics.

| Column | Type | Required | Description |
|---|---|---:|---|
| `job_id` | string | yes | Job id. |
| `frame_index` | int64 | yes | Join key to `frames`. |
| `camera_state_status` | enum | yes | `estimated`, `interpolated`, `low_confidence`, `unresolved`. |
| `intrinsics_model` | enum | yes | `pinhole`, `pinhole_radial_tangential`, `unknown`. |
| `fx_px`, `fy_px`, `cx_px`, `cy_px` | float64 nullable | yes | Intrinsics in original image pixels. |
| `distortion_coeffs` | list<float64> nullable | no | OpenCV order when present. |
| `T_world_from_camera` | fixed_size_list<float64,16> nullable | yes | Row-major transform. Null if no trajectory. |
| `q_world_from_camera_xyzw` | fixed_size_list<float64,4> nullable | no | Convenience rotation. |
| `t_world_from_camera_m` | fixed_size_list<float64,3> nullable | no | Convenience translation. |
| `T_camera_from_head` | fixed_size_list<float64,16> nullable | no | Present when head frame is estimated separately. |
| `linear_velocity_world_mps` | fixed_size_list<float32,3> nullable | no | Smoothed velocity. |
| `angular_velocity_world_radps` | fixed_size_list<float32,3> nullable | no | Smoothed angular velocity. |
| `gravity_world` | fixed_size_list<float32,3> nullable | no | Unit vector if estimated; null if not reliable. |
| `tracking_confidence` | float32 | yes | 0–1 producer confidence. |
| `translation_sigma_m` | fixed_size_list<float32,3> nullable | yes | Per-axis uncertainty. |
| `rotation_sigma_rad` | fixed_size_list<float32,3> nullable | yes | Per-axis uncertainty. |
| `median_feature_reprojection_error_px` | float32 nullable | no | Camera-tracking residual. |
| `head_camera_quality_flags` | list<string> | yes | Example: `low_texture`, `fast_motion`, `intrinsics_uncertain`. |
| `provenance_refs` | list<string> | yes | IDs into `provenance.ndjson`. |

### `tables/hand_states.parquet`

One row per `(frame_index, hand_track_id)`. Expected tracks are `left_primary` and `right_primary`; additional tracks are allowed only when multiple people/hands are explicitly enabled by request.

| Column | Type | Required | Description |
|---|---|---:|---|
| `job_id` | string | yes | Job id. |
| `frame_index` | int64 | yes | Join key to `frames`. |
| `hand_track_id` | string | yes | Stable id, e.g. `left_primary`. |
| `side` | enum | yes | `left`, `right`, `unknown`. |
| `hand_state_status` | enum | yes | `visible`, `partially_visible`, `occluded_inferred`, `out_of_frame`, `not_detected`, `unresolved`. |
| `source_method` | enum | yes | `hawor`, `wilor`, `hybrid_temporal`, `interpolated`, `none`, `other`. |
| `bbox_xywh_px` | fixed_size_list<float32,4> nullable | no | Detector/render bbox in image pixels. |
| `root_t_camera_m` | fixed_size_list<float32,3> nullable | yes | MANO/root translation in `camera_t`. |
| `root_q_camera_xyzw` | fixed_size_list<float32,4> nullable | yes | Root orientation in `camera_t`. |
| `root_t_world_m` | fixed_size_list<float32,3> nullable | no | Present if world camera trajectory exists. |
| `joints_3d_camera_m` | fixed_size_list<float32,63> nullable | yes | 21 joints, xyz packed, MANO joint order declared in manifest. |
| `joints_2d_px` | fixed_size_list<float32,42> nullable | yes | 21 projected joints, xy packed. |
| `joint_visibility` | fixed_size_list<uint8,21> nullable | yes | 0 unknown, 1 visible, 2 occluded, 3 out-of-frame. |
| `mano_global_orient_axis_angle` | fixed_size_list<float32,3> nullable | no | MANO parameter when available. |
| `mano_hand_pose_axis_angle` | fixed_size_list<float32,45> nullable | no | MANO 15-joint pose when available. |
| `mano_betas` | fixed_size_list<float32,10> nullable | no | Shape params when available. |
| `mano_topology_ref` | string nullable | no | URI/ref for mesh topology if vertices exported. |
| `mesh_vertices_ref` | string nullable | no | Row group or asset ref for dense vertices, optional. |
| `confidence` | float32 | yes | 0–1 producer confidence for this row. |
| `translation_sigma_m` | fixed_size_list<float32,3> nullable | yes | Metric uncertainty. |
| `joint_sigma_m` | fixed_size_list<float32,21> nullable | no | Per-joint radial uncertainty. |
| `reprojection_error_px` | float32 nullable | no | Median joint reprojection residual. |
| `cross_detector_keypoint_delta_px` | float32 nullable | no | Independent 2D detector disagreement. |
| `projected_to_detected_size_ratio` | float32 nullable | no | Rendered bbox height / detected bbox height. |
| `temporal_jitter_mps` | float32 nullable | no | Local wrist/root speed diagnostic. |
| `quality_flags` | list<string> | yes | Example: `depth_uncertain`, `2d_3d_size_mismatch`, `source_switch`. |
| `provenance_refs` | list<string> | yes | IDs into `provenance.ndjson`. |

Null metric fields are allowed only when `hand_state_status` explains why. For example, `out_of_frame` may have null joints; `visible` must include joints or emit a row-level error.

### `tables/semantic_clips.parquet`

One row per semantic clip. The default segmentation target is 2–3 seconds. Longer rows are allowed only when the video evidence is static and `boundary_reason` explains the decision.

| Column | Type | Required | Description |
|---|---|---:|---|
| `job_id` | string | yes | Job id. |
| `clip_id` | string | yes | Stable id: `clip_{start_frame:06d}_{end_frame_exclusive:06d}_{ordinal}`. |
| `clip_index` | int32 | yes | 0-based order. |
| `start_frame` | int64 | yes | Inclusive. |
| `end_frame_exclusive` | int64 | yes | Exclusive. |
| `start_sec`, `end_sec`, `duration_sec` | float64 | yes | Timeline seconds. |
| `boundary_reason` | enum | yes | `semantic_change`, `time_window`, `scene_cut`, `low_confidence_split`, `static_extension`. |
| `caption_short` | string | yes | One-sentence customer-facing caption. |
| `caption_detailed` | string | yes | Fine-grained visible-action description. |
| `atomic_actions` | list<string> | yes | Verb phrases, e.g. `left hand reaches`, `right hand steadies`. |
| `visible_entities` | list<string> | yes | Open-vocabulary visible nouns; no object pose implied. |
| `active_hands` | list<string> | yes | `left`, `right`; based on hand visibility/state rows. |
| `caption_confidence` | float32 | yes | 0–1. |
| `boundary_uncertainty_sec` | float32 | yes | Temporal uncertainty. |
| `evidence_frame_indices` | list<int64> | yes | Representative frames used for caption. |
| `render_clip_uri` | string nullable | no | Optional short video for this segment. |
| `quality_flags` | list<string> | yes | Example: `motion_blur`, `ambiguous_action`, `hand_occluded`. |
| `provenance_refs` | list<string> | yes | IDs into `provenance.ndjson`. |

### `tables/validation_metrics.parquet`

Long-form metric table for dashboarding and customer SLAs. This prevents metric-schema churn from forcing new columns on every release.

| Column | Type | Required | Description |
|---|---|---:|---|
| `job_id` | string | yes | Job id. |
| `metric_scope` | enum | yes | `head_camera`, `left_hand`, `right_hand`, `caption`, `render`, `decode`. |
| `entity_id` | string | yes | `camera`, `left_primary`, `clip_...`, etc. |
| `frame_index` | int64 nullable | no | Present for frame metrics. |
| `clip_id` | string nullable | no | Present for clip metrics. |
| `metric_name` | string | yes | Stable snake_case name. |
| `metric_value` | float64 nullable | yes | Numeric value. |
| `metric_unit` | string | yes | `px`, `m`, `mm`, `rad`, `sec`, `ratio`, `count`, `score`. |
| `aggregation` | enum | yes | `per_frame`, `per_clip`, `median`, `p95`, `mean`, `count`, `fraction`. |
| `interpretation` | enum | yes | `lower_better`, `higher_better`, `target_range`, `diagnostic_only`. |
| `status` | enum | yes | `pass`, `warn`, `fail`, `not_applicable`, `diagnostic`. |
| `threshold_ref` | string nullable | no | Names the threshold/config when one is used. |
| `provenance_refs` | list<string> | yes | IDs into `provenance.ndjson`. |

Required metric names for v1:

- Head/camera: `median_feature_reprojection_error_px`, `p95_feature_reprojection_error_px`, `tracked_frame_fraction`, `intrinsics_confidence`, `trajectory_discontinuity_count`, `gravity_estimate_confidence` if gravity is returned.
- Hands per side: `median_hand_reprojection_error_px`, `p95_hand_reprojection_error_px`, `median_cross_detector_keypoint_delta_px`, `median_projected_to_detected_size_ratio_error`, `p95_wrist_speed_mps`, `visible_frame_fraction`, `occluded_inferred_frame_fraction`, `unresolved_frame_fraction`.
- Captions: `clip_count`, `median_clip_duration_sec`, `p95_clip_duration_sec`, `low_confidence_clip_fraction`, `caption_evidence_coverage_fraction`.
- Render: `overlay_frame_count`, `overlay_duration_sec`, `overlay_matches_input_frame_count`, `render_layer_missing_frame_count`.

## NDJSON stream schemas

### `stream/overlay_events.ndjson`

One JSON object per frame. This is the structured source for visible overlay/render data, independent of the rendered MP4.

```json
{
  "schema_name": "ego.delivery.overlay_event",
  "schema_version": "1.0.0",
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "frame_index": 345,
  "time_sec": 11.5,
  "layers": [
    {
      "layer_id": "left_hand_joints",
      "layer_type": "keypoints_2d",
      "source_table": "hand_states",
      "source_row_key": {"frame_index": 345, "hand_track_id": "left_primary"},
      "coordinate_frame": "image_px",
      "points_xy_px": [[812.4, 503.1], [820.2, 481.7]],
      "visibility": [1, 1],
      "style": {"color_rgba": [0, 255, 80, 210], "radius_px": 3},
      "confidence": 0.86,
      "quality_flags": []
    },
    {
      "layer_id": "left_hand_mesh_outline",
      "layer_type": "polyline_2d",
      "source_table": "hand_states",
      "source_row_key": {"frame_index": 345, "hand_track_id": "left_primary"},
      "coordinate_frame": "image_px",
      "polylines_xy_px": [[[790.0, 530.2], [798.1, 512.6], [815.4, 501.9]]],
      "style": {"color_rgba": [0, 255, 80, 120], "line_width_px": 2},
      "confidence": 0.74,
      "quality_flags": ["partial_visibility"]
    },
    {
      "layer_id": "semantic_clip_band",
      "layer_type": "text_label",
      "source_table": "semantic_clips",
      "source_row_key": {"clip_id": "clip_000330_000405_005"},
      "coordinate_frame": "image_px",
      "anchor_xy_px": [36, 996],
      "text": "Left hand reaches toward the work surface.",
      "style": {"color_rgba": [255, 255, 255, 230], "background_rgba": [0, 0, 0, 160]},
      "confidence": 0.81,
      "quality_flags": []
    }
  ],
  "provenance_refs": ["prov_000041", "prov_000077"]
}
```

Rules:

- If a visible rendered mark exists in `renders/overlay.mp4`, the corresponding source layer must exist here.
- If a layer is inferred/low-confidence, style must encode that uncertainty via alpha/dash/label and the row must include the quality flag. Uncertainty is represented, not hidden.
- Render layer data must use current source rows; stale upstream validity labels cannot suppress newer hand evidence.

### `stream/caption_events.ndjson`

Optional event stream for more detailed caption structure than the clip table.

```json
{
  "schema_name": "ego.delivery.caption_event",
  "schema_version": "1.0.0",
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "clip_id": "clip_000330_000405_005",
  "event_index": 2,
  "start_frame": 352,
  "end_frame_exclusive": 371,
  "event_type": "hand_motion",
  "subject": "left_hand",
  "predicate": "reaches_toward",
  "object_text": "work surface area",
  "text": "The left hand moves downward and forward toward the work surface.",
  "confidence": 0.79,
  "evidence_frame_indices": [352, 360, 370],
  "provenance_refs": ["prov_000088"]
}
```

Text may mention visible objects as language observations, but it must not imply metric object pose or HOI contact unless those claims are explicitly produced by a future extension.

### `provenance/provenance.ndjson`

One JSON object per module/model/input artifact/config event.

```json
{
  "schema_name": "ego.delivery.provenance_event",
  "schema_version": "1.0.0",
  "provenance_id": "prov_000077",
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "event_type": "model_inference",
  "module": "hand_state_estimator",
  "module_version": "hand-delivery-0.3.1",
  "model": {
    "name": "HaWoR/WiLoR hybrid",
    "weights_id": "hawor_2026q2+wilor_2026q2",
    "weights_sha256": "91b2..."
  },
  "inputs": [
    {"logical_name": "normalized_video", "sha256": "6f9c..."},
    {"logical_name": "camera_intrinsics", "sha256": "d02a..."}
  ],
  "outputs": [
    {"logical_name": "hand_states", "row_range": {"start": 0, "end_exclusive": 3600}}
  ],
  "config": {"quality_profile": "delivery_default_v1"},
  "started_at": "2026-07-05T12:05:00Z",
  "finished_at": "2026-07-05T12:08:42Z",
  "runtime": {"device_class": "server_gpu", "duration_sec": 222.4}
}
```

### `provenance/errors.ndjson`

Errors are structured and can be job-level, artifact-level, row-level, or metric-level.

```json
{
  "schema_name": "ego.delivery.error_event",
  "schema_version": "1.0.0",
  "error_id": "err_000013",
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "severity": "warning",
  "scope": "row",
  "artifact": "hand_states",
  "frame_index": 912,
  "entity_id": "right_primary",
  "code": "HAND_VISIBLE_WITH_SIZE_DEPTH_MISMATCH",
  "message": "Projected hand size is inconsistent with detector size; metric depth is uncertain for this frame.",
  "retryable": false,
  "customer_actionable": false,
  "provenance_refs": ["prov_000077"]
}
```

API error responses use the same code namespace:

```json
{
  "error": {
    "code": "UNSUPPORTED_VIDEO_CODEC",
    "message": "The submitted video codec is not supported by the delivery pipeline.",
    "retryable": false,
    "field": "input.video_uri",
    "details": {"detected_codec": "example_codec"}
  }
}
```

## Uncertainty and validity rules

1. Every estimated physical field has a status and uncertainty path: `*_status`, `confidence`, sigma fields where metric, and quality flags.
2. Missing values are encoded as nulls plus an explicit status/quality flag. A missing physical estimate must not be represented by zeros.
3. `confidence` is producer confidence, not accuracy. Accuracy-like claims must come from validation metrics or benchmark reports.
4. Visible overlays must be driven by source rows in this schema. A rendered hand, camera path, caption band, or uncertainty badge must have an overlay event pointing back to the table row or clip row that produced it.
5. Inferred/occluded hands are represented with `hand_state_status = occluded_inferred` and reduced-confidence render style. They are not drawn as certain visible hands.
6. `unresolved` means the delivery pipeline cannot produce a coherent state for that row; downstream clients should not infer absence from it.
7. Customer-facing thresholds in `validation_metrics` are diagnostic unless `status` is `pass/warn/fail` with a `threshold_ref`. The metric value remains valid even when no threshold is assigned.

## Validation summary JSON

`metrics/validation_summary.json` is a compact dashboard view over `validation_metrics.parquet`.

```json
{
  "schema_name": "ego.delivery.validation_summary",
  "schema_version": "1.0.0",
  "job_id": "del_01JZ2BQ5N8M9WJXJ4E6H1YTA9W",
  "input_frame_count": 1800,
  "overlay_frame_count": 1800,
  "frame_count_match": true,
  "head_camera": {
    "tracked_frame_fraction": 0.992,
    "median_feature_reprojection_error_px": 1.8,
    "p95_feature_reprojection_error_px": 5.9,
    "trajectory_discontinuity_count": 0,
    "status": "usable"
  },
  "hands": {
    "left_primary": {
      "visible_frame_fraction": 0.74,
      "median_hand_reprojection_error_px": 7.4,
      "median_projected_to_detected_size_ratio_error": 0.08,
      "median_cross_detector_keypoint_delta_px": 11.2,
      "unresolved_frame_fraction": 0.03,
      "status": "usable"
    },
    "right_primary": {
      "visible_frame_fraction": 0.69,
      "median_hand_reprojection_error_px": 18.6,
      "median_projected_to_detected_size_ratio_error": 0.21,
      "median_cross_detector_keypoint_delta_px": 26.4,
      "unresolved_frame_fraction": 0.08,
      "status": "low_confidence"
    }
  },
  "captions": {
    "clip_count": 31,
    "median_clip_duration_sec": 2.4,
    "low_confidence_clip_fraction": 0.10,
    "status": "usable"
  },
  "render": {
    "overlay_matches_input_frame_count": true,
    "render_layer_missing_frame_count": 0,
    "status": "usable"
  }
}
```

## Backward-compatibility rules

1. Semantic versioning governs both API and artifact schema.
   - Patch: documentation, threshold text, or producer bug fixes that do not change schema shape or field semantics.
   - Minor: additive optional fields, additive artifact logical names, additive enum values.
   - Major: required field changes, removals, renames, coordinate convention changes, unit changes, changed interval semantics, changed meaning of existing enum values.
2. Existing required fields in `1.x` cannot be removed or made nullable.
3. Existing nullable fields cannot change units or coordinate frames.
4. New enum values are allowed in minor versions; clients must treat unknown enum values as `unrecognized` and preserve the raw string.
5. Artifact `logical_name`s are stable within a major version. Physical URIs and file partitioning may change because the manifest is the source of truth.
6. Parquet column names and logical types are stable within a major version. New columns are nullable or repeated and appended with stable field metadata.
7. NDJSON events are forward-compatible by ignoring unknown keys, but required keys for an event schema remain required.
8. Deprecation requires at least two minor versions or 180 days, whichever is longer, and must be announced in `manifest.compatibility.deprecations`.
9. A major-version output may be produced in parallel with the previous major for migration; mixed-major artifacts inside one manifest are forbidden except under an explicit `extensions[]` entry.
10. Extensions use reverse-DNS namespaces, e.g. `org.example.research_hoi.contact_state`, and cannot alter base-field semantics.

## Implementation acceptance checklist for the schema

A delivery run satisfies this output contract when:

- `manifest.json` exists and all artifact hashes referenced by it resolve.
- `frames.parquet` has exactly the decoded output frame count and contiguous 0-based `frame_index` values.
- All frame-indexed tables join to `frames.parquet`; no out-of-range frames exist.
- `renders/overlay.mp4` and `renders/side_by_side.mp4` have the same frame count and duration as the normalized input unless the job failed before rendering.
- Every visible overlay mark has a corresponding `overlay_events.ndjson` layer with a source table row/clip row.
- Head/camera metrics are reported separately from left/right hand metrics.
- Hand rows distinguish visible, occluded inferred, absent/out-of-frame, and unresolved states.
- Semantic clips cover the requested timeline with half-open intervals and have evidence frames.
- Validation metrics include the required v1 metric names and units.
- Provenance rows identify model/config/input hashes for each output-producing module.
- Errors are explicit in `errors.ndjson`; silent fallback is not allowed.

## Concrete implementation note

The minimum implementable v1 can omit dense hand mesh vertex export by setting `outputs.dense_hand_meshes=false`, leaving `mesh_vertices_ref=null`, and still returning metric joints/MANO parameters plus 2D overlay layers. The render artifacts must still be driven by `hand_states` and `overlay_events`; a customer should be able to reproduce the visible hand keypoints/skeleton/caption bands without access to internal runtime logs.
