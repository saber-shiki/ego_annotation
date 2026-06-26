# Pipeline V19 current epistemic state

## Current supported claim

For HOT3D `clip-001850`, the current accepted frozen prediction boundary is the runtime-owned OWLv2/SAM2 downstream branch, not either rejected click/box or stale v5 branch. The frozen manifest is `state/v19_prediction_freeze_manifest.json` with SHA256 `f0b563c7e2bb1ed5109e3a8dcbf39a9d0938ed1126b6e454c66121000be5832b` under run root `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.

The physical annotation claim supported by render consumption is bounded: OWLv2/SAM2 materially repaired the prior broad table/hand/arm support failure, and direct visible keyboard poses now register to the physical key field/rim substantially better in the canonical overlay/world/side-by-side videos. Full-duration canonical render hashes: overlay `9be7ef18de54a7f0924f87a9bcb9b463666b646f73ff2ed3c63607076f4b74cc`, world `4793492f59ba717043594a6e43a39a30ed6328070147d1498fbdfd59a4bc25ab`, side-by-side `864ea55ee0912bb90e88ed3a32c5d6dd81a2c4d6eb46ad562dac5ad540bddf56`. This does not prove contact, nonpenetration, or late temporal-completion frames; those remain uncertainty.

The original frozen P18 interval MANO state is not a 21-joint HOT3D accuracy improvement. Against HOT3D MANO on 300 hand/frame rows, original interval median wrist error worsens from `0.03359 m` to `0.03704 m`, median MPJPE worsens from `0.04719 m` to `0.05922 m`, while root-aligned MPJPE changes only from `0.02267 m` to `0.02297 m`. The regression is right-hand/global-translation dominated, not wrist-relative articulation dominated. Evidence: `evaluation/hot3d_mano3d_owlv2_interval/hot3d_runtime_hawor_vs_v19_interval_comparison.json`, SHA256 `67eeece6fb98cd91dcc4f8cf1fbfc439daa9d67e7f47cb60b7b491ad37c6edc0`.

Workbench item 7 autoresearch has now identified and tested a general support-gate mechanism. Decomposition JSON `hot3d_interval_mano_error_decomposition.json` (SHA256 `ba1352636948e8cbd62f70e2f793600690afd9b2def3f823e86d9cb6aae48b01`) shows the original right-hand interval corrections have mean MPJPE delta +`0.01960 m`, p90 +`0.05902 m`, root-aligned mean delta only +`0.000235 m`, median wrist displacement `0.02402 m`, and 74% of right wrist corrections move farther along the baseline error direction away from GT. The top worsened rows all have `visible_surface_depth_order_selected_vertex_count=0`.

A prediction-only support-gated ablation confirms the mechanism. `build_v19_interval_mano_translation_gate.py` preserved source HaWoR wrist/root translation whenever selected visible-surface support count was `<=0`, while keeping interval wrist-relative articulation. It gated all 300 rows because the frozen P18 state had zero selected support for each row. The support-gated state (SHA256 `dae754858f362331db288f99390a7442697113095d84d7ee3fb2852140a6ed37`) returns wrist error exactly to baseline, improves median MPJPE slightly relative to baseline (`0.04605 m` vs `0.04719 m`), and leaves root-aligned MPJPE identical to the original interval (`0.02297 m`). Evidence: `support_gated_baseline_wrist_zero_support/hot3d_baseline_vs_interval_vs_support_gated_comparison.json`, SHA256 `3cfe16a05de10a5eea136dd6130b08e9ea1066170987d2caf5fe54e5144e687c`.

Therefore the current best causal claim is: the original P18 failure is ungrounded global wrist/root translation under absent visible-surface support. Wrist-relative interval articulation is not the primary failure and may be weakly beneficial once root translation is protected. The support-gated ablation is an autoresearch result on the fixed clip, not held-out generalization.

## Rejected boundaries and mechanisms

- Wrong raw-mesh freeze `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c`: rejected because P18/P19 consumed raw P12 TRELLIS mesh while P15 poses were in P13 completed-canonical mesh coordinates.
- Mesh-frame-repaired freeze `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`: rejected because the object still visibly followed contaminated mask/surfel support.
- Positive-click-derived prompt boxes and explicit agent-authored box prompts: rejected because they still allowed SAM2 to select broad tabletop/hand/background support.
- P09 960/1408 intrinsics/depth-size suspicion: ruled out for the OWLv2 branch; the earlier cyan surfel mismatch was diagnostic display scaling.
- HOT3D evaluator coordinate/review failure: ruled out because 1408-frame evaluator review sheets project GT and predictions onto visible hands.
- Pure articulation-error explanation for P18: ruled down by decomposition and support-gated ablation. Root-aligned errors stayed effectively fixed while absolute errors followed wrist/root translation.

## Current causal model

Object registration and interval MANO are separate mechanisms. Object registration failed through segmentation support contamination; OWLv2 text-grounded boxes changed that upstream cause and produced a bounded usable keyboard render.

The interval MANO failure comes from allowing contact/temporal/object terms to move global hand translation without selected visible-surface support. In this run every P18 row had zero selected visible-surface support vertices, yet the optimizer moved the right wrist by centimetres in a direction that usually increased GT error. The support gate blocks exactly that unsupported degree of freedom while preserving any wrist-relative articulation correction.

The physically correct rule is not “never correct translation.” It is: global wrist/root translation requires direct support evidence such as selected visible-surface/depth-order vertices or another calibrated metric support factor. Without such evidence, contact remains an uncertainty annotation and articulation may be adjusted only relative to the observed wrist/root prior.

## Live uncertainties

1. Generalization: the support gate is confirmed on the fixed `clip-001850` interval but not yet tested on another HOT3D clip or a project representative. Prediction: on intervals with zero selected visible-surface support, the gate should prevent MPJPE regressions; on intervals with positive support, translation corrections may still be allowed and need separate evaluation.
2. Runtime integration: the solver/spec now contain `--gate-translation-with-visible-surface-support`, but the frozen runtime prediction has not been rerun through P18-P21 with the integrated gate. The ablation state under `evaluation/` is not the runtime prediction state.
3. Contact/nonpenetration: still uncertain. The gate prevents unsupported hand translation; it does not prove contact or signed nonpenetration.
4. Benchmark scope: current interval-correction evidence is one HOT3D clip. Workbench evaluation still needs broader fixed-slice/held-out evidence before any general claim.

## Next action

Commit the support-gate solver/spec/evaluator-autoresearch changes and then run a runtime-owned P18-P21 continuation using the integrated gate, without HOT3D GT, to produce gated full-duration renders and a new freeze boundary. After that, run the same evaluator/comparison on the new gated runtime state and then expand to the next fixed HOT3D clip if the runtime artifact remains visually coherent.
