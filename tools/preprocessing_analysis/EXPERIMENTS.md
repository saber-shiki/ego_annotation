# Observation/candidate experiment utilities

`preprocessing.observation_experiment` is an explicit experimental command. It does not change `--profile visible-surface` or legacy defaults. Inputs are a frozen source bundle plus a hash-bound frozen prediction used as a memoized initializer; no evaluator/GT reads occur there. New 2D tracks are extracted before any candidate is generated and keep their full train/holdout tracklet identity. New PnP fits never see the held-out tracklets, but initial baseline/keyframe geometry was estimated from the same RGB: this is NOT an independently acquired validation dataset.

Artifacts for each case:
- `tracklets.npz`: pure 2D observed tracks, persistent identity/reference/frame and conditional binding availability;
- `candidates.json`: fixed-population per-family scores, provenance, support and pose-Jacobian observability;
- `experiment.npz`: full-timeline four-slot world poses, candidate-valid masks, all selector indices and scale-refitted poses;
- `mechanism.mp4`: RGB / frozen baseline / stale-PnP / lifecycle-PnP / multi / shared-scale-refit;
- `leave_one.mp4`: multi / no_rgb / no_mask / no_surface / no_mano / raw_mask;
- `candidates.mp4`: all four slots. Invalid slots repeat baseline only for rendering, never oracle or selection;
- `scale_refit.json`, `translation_refit.json`: fixed-UniDepth refits;
- `completed.json`: source/baseline/code bindings and explicit limitations.

The current no_mano and raw_mask selectors are intentionally equivalent: MANO appears in the candidate scoring mask domain, not a separate independent energy term. Removing that mask-domain coupling cannot undo upstream MANO-owned-mask generation. Similarly, no_surface does not regenerate the initialized reference mesh/poses without depth. These are strictly **same-bank selection-score** ablations, not whole-pipeline modality-removal experiments.

`tools/preprocessing_analysis/evaluate_experiments.py` runs only after `batch.exit=0`. It verifies all source receipts, numerical frozen candidate0 identity and video decoding, then uses existing external evaluation helpers to compute all candidates under one common baseline anchor gauge. A per-frame GT oracle is a diagnostic lower bound inside this finite bank, not an executable or temporally consistent trajectory. It never writes oracle choices into prediction output.

The first five-case run is `/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_observation_experiments_20260928_r1/`. The corresponding frozen producer snapshot is `/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_observation_experiments_20260928_r1/`. Runtime uses the existing numerical environment, one CPU thread on the current server and CUDA disabled. No models/installations, source overwrites or RL changes.

See `docs/OBSERVATION_EXPERIMENT_CONTRACT_20260928.md` for the causal contract and limitations of the gap-spread experiment (not a calibrated covariance implementation).
