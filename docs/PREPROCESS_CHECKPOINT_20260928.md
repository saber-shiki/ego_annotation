# Monocular preprocessing checkpoint — 2026-09-28

Independent branch: `feature/monocular-preprocess-reliability-20260928`.
Worktree: `/mnt/user-home/kupingxin/ego_annotation_preprocess_monocular`.
Base before this checkpoint: `790893c25e24d5bd91d68e56c30728a0604b80d3`.

This checkpoint preserves the accumulated experimental preprocessing package, tests, packaging integration, design/result documents, and archived external diagnosis/evaluation scripts. It does not promote these experiments into observation production or RL. The legacy route remains the default; fixed-UniDepth fitting is explicitly `--profile visible-surface`. Geometry-only relocalization remains experimental/opt-in.

Latest frozen prediction: `/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_visible_surface_20260928_jump_r2/`.
Matching code snapshot: `/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_visible_surface_20260928_jump_r2/`.
Review: `/mnt/user-home/kupingxin/ego_annotation_experiments/milk_jump_diagnosis_20260928/final_review/`.

Recorded verification of this numerical implementation: 193 regression tests, 189 passing and 4 skipped; 141 runtime-snapshot files matched; 750 unfiltered proposal poses identical to the prior baseline; 15 full videos / 2250 decoded frames. This records the preceding run, not a fresh test execution in this checkpoint step. Source depth/clouds, MANO and triangle topology were preserved. The first jump repair's BBQ regression is retained and documented; the latest patch is not a claim of universal pose or contact accuracy.

Source observations, all historical/final prediction outputs, model assets and large review media remain outside Git without relocation. `tools/preprocessing_review_archive/archive_manifest.json` binds copied script source to original paths/hashes. No source S3 packages, original dirty worktree, reward/RL code or trained weights are modified by this checkpoint. No merge or push is requested/performed.

Follow-up work in this branch will first characterize observation reliability and specify shared-mechanism changes, without silently revising the fixed-UniDepth contract or using GT/CAD as prediction input.
