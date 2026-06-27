# Pipeline V19 current epistemic state

## Current supported claim

Workbench item 6 has met the bounded primary HOT3D gate on the fixed slice `clip-001849`, `clip-001850`, and `clip-001851`: each clip has a frozen prediction boundary with full-duration overlay/world/side-by-side renders, post-freeze HOT3D 21-joint MANO scoring, and evaluator review sheets consumed as hand-state evidence (OPS 2026-06-27T05:48, 2026-06-27T07:38, 2026-06-27T09:15).

The supported physical claim is bounded. V19 now produces a Pi-runtime-owned keyboard/MANO artifact where the keyboard mesh/pose is visibly registered to the physical keyboard on the three HOT3D clips, and the support-gated interval solver prevents unsupported global wrist/root translation. It does **not** prove contact closure, signed nonpenetration, full MANO surface accuracy, object-pose metrics, or general MANO improvement.

Three frozen support-gated runs:

- `clip-001850`: run root `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`; freeze SHA256 `06c8ff45a955421b341ceafb0548b27d2d791e9ea6495f1fc6d17ee4db67c594`. Against runtime HaWoR, wrist median error is unchanged, median joint MPJPE improves by `-0.001136926 m`, and median root-aligned MPJPE worsens by `+0.000304392 m` (OPS 2026-06-27T05:48; `docs/v19_hot3d_fixed_slice_v1_results.md`).
- `clip-001849`: run root `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001849_pinhole_a800_native_v1_supportgate`; freeze SHA256 `b7a82b2add62ca268f136982d76b5cbdc227cc7c52d4157b42d314242a3b3bea`. Against runtime HaWoR, wrist median error is unchanged, median joint MPJPE worsens by `+0.000851875 m`, and median root-aligned MPJPE improves by `-0.001110242 m` (OPS 2026-06-27T07:38; `docs/v19_hot3d_fixed_slice_v1_results.md`).
- `clip-001851`: run root `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001851_pinhole_a800_native_v1_supportgate`; freeze SHA256 `faacbf86eb890ebac022e97bfdf34b98175c7acd48487cab3ff472d12f7e605f`. Against runtime HaWoR, wrist median error is unchanged, median joint MPJPE improves by `-0.000999961 m`, and median root-aligned MPJPE worsens by `+0.000852540 m` (OPS 2026-06-27T09:15; `docs/v19_hot3d_fixed_slice_v1_results.md`).

Across all three clips, every P18 row has zero selected visible-surface depth-order support and the output translation gate is applied. The mechanism-level result is therefore a safety result: unsupported contact/temporal/object terms no longer move the global wrist/root. The remaining metric changes are wrist-relative articulation effects under uncertainty.

## Current causal model

Object registration and hand correction are separate mechanisms. OWLv2/SAM2 plus crop-conditioned TRELLIS and visible-depth pose fitting repaired the systematic keyboard-support contamination that earlier freezes exhibited. On `clip-001851`, P14/P15 fit all 150 keyboard poses with millimeter-scale observed-to-mesh residuals and zero temporal correction, so later MANO uncertainty is not explained by missing object-pose registration.

The original P18 MANO failure was systematic, not normal measurement noise: contact/temporal terms moved global hand translation even when there was no selected visible-surface depth-order support. The integrated output gate implements the physically narrower rule: when selected support count is zero, preserve the source HaWoR wrist/root translation and allow only wrist-relative articulation. The exact discriminator is row-level `output_translation_gate` plus `visible_surface_depth_order_selected_vertex_count` (OPS 2026-06-27T09:15).

Item-7 causal diagnosis has narrowed why all fixed-slice rows are zero-support. In the solver, selected visible-surface support is built by projecting MANO vertices into the visible/ownership-eligible object mask and reading finite metric depth there. Across all three clips, `finite_inside_count` is zero for all 900 hand/frame rows even though object-owned masks and contact-patch priors are active. Therefore selected support is not being sorted away after construction; the exact visible-object-mask overlap criterion does not find MANO/object surface overlap. This is physically plausible for keyboard interaction because the hand-owned pixels occlude the touched keys: support is adjacent or latent behind the hand, not visible under the MANO projection.

The current solver still optimizes a latent global translation before the output gate projects the wrist/root back to HaWoR. That latent unsupported translation could contaminate root/pose articulation through temporal/contact coupling. Workbench item 7 has launched a controlled `clip-001851` ablation that freezes global MANO translation inside optimization whenever selected visible-surface support is zero, then scores the resulting interval state after it is written under `evaluation/autoresearch/`. If articulation metrics change materially while wrist remains unchanged, output gating was insufficient because unsupported latent translation affected the articulation solution. If metrics are unchanged, the residual articulation behavior comes from root/pose/contact terms rather than latent translation.

## Rejected boundaries and mechanisms

- Wrong raw-mesh freeze `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c`: rejected because P18/P19 consumed raw P12 TRELLIS mesh while P15 poses were in P13 completed-canonical mesh coordinates.
- Mesh-frame-repaired freeze `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`: rejected because the object still visibly followed contaminated mask/surfel support.
- Positive-click-derived prompt boxes and explicit agent-authored box prompts: rejected because they still allowed SAM2 to select broad tabletop/hand/background support.
- P09 960/1408 intrinsics/depth-size suspicion: ruled out for the OWLv2 branch; the earlier cyan surfel mismatch was diagnostic display scaling.
- HOT3D evaluator coordinate/review failure: ruled out by evaluator review sheets showing GT and predictions projected onto visible hands.
- General MANO-accuracy-improvement claim: rejected by the fixed-slice metrics. Two clips improve median joint MPJPE and one worsens; root-aligned medians are also mixed. The supported claim is safety against unsupported wrist/root translation, not accuracy closure.
- Contact/nonpenetration closure: rejected for current HOT3D runs. Completed/sign meshes are non-watertight, P17 factors are broad possible-contact priors, and render/world views show hand-keyboard gaps.
- Visible-surface support being discarded after candidate selection: rejected by item-7 diagnostic. `finite_inside_count` is zero in all 900 rows, so positive candidates never exist under the exact-overlap construction.
- P19 verifier logic that expects `manifest.videos` or lets `Path("")` pass is invalid; the renderer writes `manifest.outputs.*`. The durable spec is patched for current/future launches.

## Live uncertainties

1. Positive-support MANO correction remains untested. All fixed-slice support-gated rows are zero-support, so the pipeline has not yet shown a physically supported global hand-translation correction.
2. The support criterion may be too exact for occluded contact. A general next mechanism is nearby/latent support from adjacent visible rigid surface plus uncertainty, not per-clip threshold tuning.
3. The current output gate may leave articulation contamination from latent unsupported translation inside optimization. The running item-7 ablation is designed to distinguish this.
4. Contact/nonpenetration remains unresolved because current keyboard meshes/sign fields are non-watertight and P17 does not establish stable contact anchors.
5. Full MANO surface metrics remain unavailable for interval candidates because P18 states store optimized joints and sampled vertices, not full optimized MANO vertices.
6. HOT3D scoring currently covers 21-joint MANO only. Object pose/contact/occlusion metrics remain unimplemented or unsupported by the evaluator path.
7. Runtime remains a design problem: support-gated P18 takes tens of minutes on 5-second clips. This does not falsify the physics, but it violates the target runtime scale.

## Next action

Continue Workbench item 7. Monitor the existing tmux session window `v19_hot3d_001850_a800_v5_owlv2_p07:clip001851_autoresearch` only when its result is needed. The running branch writes under `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001851_pinhole_a800_native_v1_supportgate/evaluation/autoresearch/in_solver_translation_freeze_v1/` and must not modify the frozen prediction state. When it exits, inspect the interval state, `optimizer_translation_support_gate`/`latent_optimizer_translation_world_m`, HOT3D comparisons, and review sheet. Then either run the same ablation on `clip-001849`/`clip-001850` if the mechanism changes articulation, or reject it and move to a nearby/latent visible-surface support mechanism.
