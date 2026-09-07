# V20 late-window run results

## Artifact archive

The first complete visual-review candidate (v16 = pose v14 + shape v15) is copied independently under:

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_depth_order_corrected_fullrun_20260903T101508Z/run/P0014_84ea2dcc_carton_milk_f2370_2519/deliverables/v20_late_window_se3_v16_visualization_archive_20260907/
```

The adjacent archive is:

```text
v20_late_window_se3_v16_visualization_archive_20260907.tar.gz
```

Archive SHA256:

```text
944ef525dae5b0e8d5a80782d17f3b83ee133375296b7f64d1f9f6fe41e64fcb
```

## Selected diagnostic candidate

The latest evidence-protected candidate is v25:

- pose: translation-only low-conditioned RGB v23;
- shape: pose-frozen v24 with strict signed-front diagnostics;
- render: full 150-frame v25 MP4/RRD.

The candidate is intentionally marked:

```text
prediction_candidate_incomplete_signed_front_validation
```

The strict visible-pose/render audit reports sustained negative segments:

```text
134–142 and 144–149
```

with frame 149 signed median approximately `-44 mm`. This is not a collision or signed-volume claim.

## Prediction-only pose comparison

Post-freeze HOT3D evaluation is in a separate report and was not consumed by candidate generation. The stable v23 pose has approximately:

- rotation median `10.147°`;
- rotation p90 `13.262°`;
- rotation p95 `14.716°`;
- frame-149 rotation error `19.849°`;
- frame-149 translation error `28.08 mm`;
- predicted late relative rotation `123.03°` versus GT diagnostic `136.54°`.

The low-conditioning policy preserves weak translation evidence while excluding those edges from rotation authority (`41` rotation-eligible edges, `11` translation-only edges).

## Optional RGB pixel-evidence experiment

Serialized source-canonical→target-UV evidence was added as an optional observed-only factor. It is disabled by default and must be explicitly weighted.

- high weight diagnostic v24 produced larger pose corrections but worsened translation and was not selected;
- weight `0.08` with generated validation (v26) worsened the late per-frame signed bias and was rejected;
- no generated mesh factor entered `pose_residual()` in any run.

## Rendering contract

The latest diagnostic render is full duration:

```text
1920×960, 30 FPS, 150 frames, 5.000 seconds
```

The renderer report confirms:

- exact pose/mesh binding;
- objective mesh preserved;
- no display-only front-face pruning;
- camera-frame P09 points converted to world once;
- overlay projection uses world-to-camera once.

Generated geometry remains `diagnostic_only` and cannot authorize collision, contact, SDF, sign, signed volume, or nonpenetration.

## Additional shape experiment

A shared bounded canonical registration rotation was tested after pose freeze (`shape_after_late_pose_v23_canonical_rotation_v28`). It did not remove the late signed-depth segment and reduced coverage/IoU relative to the anisotropic-only shape. It is not the selected candidate.
