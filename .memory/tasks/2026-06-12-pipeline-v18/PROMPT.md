## Objective

Produce V18 as a real full-video physical annotation pipeline whose primary deliverable is improved metric MANO hand pose annotation for `trash_1050` and `task5_tomato_960`.

The goal is solved, mechanism-driven hand state over the full videos, with object/part/contact/occlusion/nonpenetration state serving as constraints or uncertainty evidence for that hand annotation. Final overlay/world/side-by-side videos and backing data must reproduce the hand state and show how object/contact/occlusion evidence changes, constrains, or invalidates MANO pose claims. Fine-grained contact activation/rejection is not an objective by itself; it is useful only when it improves, validates, or falsifies the metric MANO hand annotation. Reports, validators, labels, row counts, rendered glyphs, and runner success are evidence only when they directly constrain the delivered hand state or expose why the current evidence cannot improve it.

## Scientific judgment constraint

Every result must distinguish ordinary quantitative error from qualitative pipeline mistakes by visual/geometric judgment, not by a scalar threshold. Scalar residuals, counts, p95 values, masks, and validator outcomes are diagnostic cues that direct inspection; they do not decide whether an output is physically acceptable or whether a mechanism is wrong. A small residual can still be a qualitative failure if the rendered hand/object relation is physically impossible or produced by an invalid dataflow. A large residual can be ordinary measurement noise if the rendered trajectory and causal evidence remain coherent and the uncertainty is represented. Do not create thresholds to separate these categories. Inspect the rendered videos, world geometry, depth/mask ownership, temporal behavior, and mechanism provenance, then state the subjective physical judgment and the observations that support it.

## Workbench

V18 has reached **scoped bounded MANO closure under current evidence**: the current v5 full-video artifact is the delivered interval-MANO annotation, with Task5 support limits and Trash occlusion limits rendered explicitly. The workbench is no longer “build the solver” or “try another factor”; new work is justified only if final-artifact consumption exposes a specific systematic implementation/dataflow/physical defect in `H_t`, or if genuinely new independent evidence can tighten a current support/occlusion information limit. The reusable physical factor families must remain general; `task5_tomato_960` and `trash_1050` are fixtures for causal discrimination, not special-case algorithms.

### Current objective

Improve the delivered metric MANO hand trajectories by replacing invalid or ambiguous physical constraints with general, model-driven factors. Each factor must define its variables, observations, solver residuals/quarantines, and rendered consequence. Each workbench item must end in a solver rerun and rendered overlay/world/side-by-side consequence on at least one fixture, or in a causal revision that changes the reusable factor interface or the final artifact's uncertainty claim. A generic request for “more evidence” is not a workbench item.

### Current frontier facts

- Current combined interval-MANO frontier artifact: `/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/`. It exposes sanitized non-H-prime task5 support-bounded and trash occluded-translation-posterior full-video interval-MANO renders under standard overlay/world/side-by-side names, writes merged backing interval solver states, copies source interval state JSONs, freezes the Trash posterior report into the artifact, and regenerates physical-cause uncertainty classification. Relative to v4, Trash left 1004-1008 is repaired as an observation-invalid lid-occlusion boundary: the generic `hand_observation_visibility` factor directly zeroes those five rows, the interval solve changes those rows by up to `0.008654m` and propagates smaller temporal changes through later left rows, and the selected first-surface conflict remains broad (`784 -> 617`) rather than cleared. The cross-interval posterior now covers 58 zero-observation rows: 46 cannot clear selected first-surface conflicts inside the translation bound, 11 are clearable by bound but retain residual under feasible-start fallback after optimizer failure, and 1 clears only at a fallback representative point. This root, not `/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard` or the older v1/v2/v3/v4 artifact roots, is the current user-consumable scoped V18 MANO deliverable: bounded/latent annotation with explicit support and occlusion uncertainty, not solved contact/object pose/nonpenetration or known hidden-hand reconstruction.
- Task5 frontier render: `/data2/ego_annotation_outputs/v18_task5_joint_mano_surface_support_uncertain_sanitized_base_full_video_v1/task5_tomato_960/`. It renders 942 support-aware hand states over 471 unique frames from sanitized non-H-prime annotations; optimized H_t is exactly identical to the pre-sanitized support-bounded frontier, but the render no longer uses final-v7 H-prime fields for the original/context layer. It is the more honest task5 artifact because observed-surface support uncertainty is carried over every ownerq interval, not only 690-725. The all-interval rerun falsifies exact posed-tomato surface nonpenetration as a confident MANO correction under current evidence: hard surface-driven motion nearly vanishes once the tomato surface is treated with its independent support uncertainty, while rendered magenta marks carry the resulting bounded uncertainty. Task5 690-725 has also falsified direct visible-surface depth-order and hard/overconfident current-proximity contact-patch mechanisms. A later strict zero-row check against an external mesh-contact report was diagnostic-only and is not the V18 contact method; current annotation evidence now feeds a false-positive-tolerant 71-row latent contact factor instead of being discarded, and a follow-up solver can optimize explicit per-row contact probability `C_t`. These corrected contact mechanisms reach H_t and render coherently on 690-725, but they are not promoted as the task5 frontier because `C_t` stays essentially at its prior, visual improvement is absent, and residual effects remain mixed relative to support-bounded/fixed-contact renders. A later support-scaled contact attraction and simple MANO-to-patch-distance geometry likelihood for `C_t` moved interval H_t and made contact uncertainty render honestly in magenta, but still did not produce a visibly stronger annotation or meaningful contact posterior update under current object support. The persistent point-anchor form of `A_t` was then tested and rejected: all 71 regenerated rows are local visible-surface contact only, stable anchor residuals are disallowed, object-frame patch dispersion is centimetric, and the solver now fails loudly if asked to consume those rows as stable anchors. Dense visible-archive tomato pose support was also tested through the same solver/render path; it modestly tightened object support but worsened full/raw residual sums and produced no visible MANO improvement, so it is not promoted. Directional normal support was also tested: normal p95 is essentially the same as Euclidean p95 and the targeted 453-508 render trades much larger H_t motion for only tiny residual reduction, so it is not promoted. Task5 annotation-box visible ownership was then rerun over all current intervals with sanitized non-H-prime annotations, a repaired `surface_rows` parser, aligned-only hard ownership, exclusive ownership states, and a sanitized/no-ownership surface-support factor. The clean-provenance task5 baseline is exactly identical to the current task5 frontier H_t, proving the prior final-v7-derived support/ownership provenance was inert; the repaired ownership factor leaves only 13 hard pixels across five early left boundary rows, removes 925 projected faces from eligibility, and changes optimized H_t/vertex samples and max full/raw residual deltas by exactly zero across 942 states. It is therefore a factor-interface repair and falsification, not a promoted render; do not describe every residual statistic as identical because residual count/mean bookkeeping can change when eligible faces are removed. The reusable local object/contact-patch support state was then implemented for 690-725 using visible tomato mask/depth, current object pose/mesh normals, and temporal same-patch object-frame samples. A clean-room review found the first accepted v3 measurement was overbroad because it used band-near MANO vertices rather than the exact solver residual selector. The corrected exact-selector run now reproduces the solver's `contact_patch_targets_from_vertices` selection, including eligible observed faces, hand-owned-depth quarantine, surface-eligibility intersection, distance ordering, and the row `max_vertices` cap. Selector parity passes for all 71 rows against the solver-consumed contact-patch vertex ids; median exact patch vertex count is 96, matching the cap. The exact patch is still physically broad on the tomato (target-point p95 radial spread median ~`0.0367m`, max ~`0.0495m`), local depth samples still cover much of the visible tomato (median local/valid object-depth pixel ratio `0.953`), and local normal support remains centimetric and usually worse than global support (exact local median `0.011243m`, p95 `0.013203m`; global median `0.009287m`, p95 `0.011905m`). Only 2/71 rows consume exact local support, with max support reduction `0.000737m`; a same-code global-contact attribution solve is exactly identical to exact-local at optimized joints, vertex samples, translation, root, pose, and full/raw residual summaries. Therefore the observed contact-row H_t motion is not caused by measured local support. Task5 remains object-support-bounded uncertainty under current visible/depth/object-pose evidence; do not rerun local support unless a genuinely new independent support source or physically narrower contact-manifold mechanism is added.
- Task5 object fixture: one scale-sane TRELLIS completed mesh, treated as a rigid body with per-frame SE(3) pose from `/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/pose_fit_frame929prior_frame806scale_v1_from_tracked/v18_compact_rigid_object_pose_fit_report.json`. Hidden TRELLIS faces are not automatically trusted; solver constraints must be restricted to physically eligible faces.
- Trash frontier render: `/data2/ego_annotation_outputs/v18_trash_handobs_extend_1004_1008_posterior_full_video_v1/trash_1050/`, consolidated into v5. It applies the accepted transition-v2 physical trajectory family with a durable visual-boundary factor contract from sanitized non-H-prime annotations, an independent annotation-box visible-ownership repair over 871-930, the repaired occluded-hand camera-z posterior/feasible-stress layer over the hard occlusion span, and the additional generic observation-validity repair for left 1004-1008. It renders 564 optimized hand states over 282 unique frames; full-video frames outside solver states remain context/passthrough. Sanitization changes trash H_t numerically in some earlier/late intervals (max joint diff `0.017163m`, p95 `0.002912m` relative to the pre-sanitized render) but expanded review did not show visible degradation; this sanitized posterior render is the current frontier because older contract-repro/final-v7 roots consumed H-prime annotation fields. It preserves the signed-volume falsification from the mask-depth frontier, uses generic row-weighted `hand_depth_shift_prior`, zero-observation latent-occlusion factors for occluded `H_t`, coherent first-surface filtering for frames 995-1001, and hard-bound magenta camera-z posterior endpoints on zero-observation rows. The current artifact has 58 zero-weight hand-observation states, including left 936, 960, 972-1002, and 1004-1008 plus additional right-hand latent rows; 871-930 remains a bounded occlusion-shift branch with partial downweighting rather than zero-observation rows. The left transition into occlusion is modeled as a durable observation-invalid interval beginning at frame 972, based on RGB visibility: 970-971 retain visible hand/skin-edge evidence, while 972-1002 are latent under the lid and left 1004-1008 remains a visually continuous observation-invalid boundary before ordinary rendering returns at 1009. A selected-vertex camera-depth slack/posterior probe falsifies a clean hidden-depth uncertainty envelope: most zero-observation rows require more camera-z shift to clear selected visible-surface constraints than the solver translation bound permits. This is improved bounded/latent hand annotation, not contact/nonpenetration closure, known hidden-hand reconstruction, an optimized MAP trajectory, or a calibrated probability distribution.

### Artifact naming discipline

V18 output names are historically messy. Do not infer global progress from suffixes such as `v3`, `v5`, or `v7`. `v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard` is an old H-prime line and is not newer/better than the current MANO frontier. `v18_current_frontier_interval_mano_artifact_v5` is the current user-facing MANO artifact despite local suffixes elsewhere. Treat every path suffix as a local artifact counter unless this PROMPT explicitly names it as the current frontier.

### Reusable workbench items, in priority order

1. **Resolved for current evidence — Task5 local object/contact-patch support state.** A reusable local support variable for the manipulated object patch now exists and has been tested on the 690-725 Task5 contact interval from sanitized non-H-prime inputs. The accepted result is the exact-selector correction, not the earlier overbroad v3 run: the measurement reproduces the solver's contact-patch residual selector and exposes selector parity against solver-consumed vertex ids. It carries uncertainty from visible masks, metric depth, temporal same-patch evidence, and current object pose/mesh hypotheses into the existing `contact_patch` solver residual. Result: the exact solver patch itself is broad on the visible tomato, current local patch support is not physically tighter than frame-global object support, and exact-local support is solver-inert relative to same-code global contact. The correct Task5 claim is support-bounded uncertainty unless a new independent support source or physically narrower contact-manifold mechanism appears; another local-support threshold/radius/reweighting pass is not the next workbench item.

2. **Resolved for current evidence — Trash occluded-hand trajectory posterior and 1004-1008 observation-validity boundary.** The current single latent point-estimate shifts in the hard hidden-hand span have been replaced in the current frontier by a bounded uncertainty layer: a one-dimensional additional camera-z hard-bound feasible/energy stress-test over zero-observation rows, using sanitized per-vertex selected visible-surface depth-order deltas, current solved `H_t`, translation bounds, and temporal smoothness/acceleration. Clean-room review conditionally accepted consolidation under a narrow interpretation, and then accepted v5's left 1004-1008 extension as a systematic observation-validity repair. The repaired report/render preserves selected deltas, selected ids/depths, grid residual energies, representative feasible fallback state, and state/provenance validation; the renderer validates annotations, camera-z axes, selected evidence, grid energies, state fingerprints, and translation bounds before drawing. Result: the hard hidden-hand span and the left 1004-1008 boundary remain mostly saturated/conflicted, not narrowly reconstructable by this mechanism, and v5 visibly exposes broad magenta lower/upper hard-bound posterior skeletons on hard rows. This is not hidden articulation reconstruction, an optimized MAP trajectory through failed optimizer rows, a calibrated probability distribution, or proof that no free 3D/articulated hidden-hand trajectory exists.

3. **Resolved for current evidence — Frontier artifact consolidation after a real `H_t` or uncertainty change.** Item 2 first widened/scoped Trash hidden-hand uncertainty without changing the optimized point trajectory, creating v4, and then repaired the left 1004-1008 observation-validity boundary with a real interval `H_t`/uncertainty change, creating v5. Do not create another numbered artifact for unchanged hand pose, renamed fields, cleaner reports, or a local experiment counter.

### Success and failure standard

A workbench item succeeds when it changes the delivered MANO hand trajectory, its rendered physical uncertainty, or the causal explanation of why the trajectory cannot be further constrained from current evidence. A negative outcome is useful only when it revises the causal model and forces the next mechanism: for example, local patch support proving still too broad, or an occluded-hand posterior proving the hidden-hand feasible set is genuinely wide. Do **not** report progress from masks, face labels, validators, residual summaries, manifests, artifact counters, or commits unless the final MANO render/backing state consumes them and the physical interpretation changes.

### Active prohibitions

- No isolated-frame correction.
- No object-category, color, material, or action-phrase if/else logic for tomato, trash, lid, rim, or any named object family.
- No same-solver reweighting, extra active-set passes, or threshold tuning unless one of the reusable workbench inputs above changed first.
- No standalone diagnostics, validators, reports, or ledgers unless they directly produce a reusable solver input named above or verify a rendered MANO consequence.
- No hidden-volume nonpenetration proof from free-space-conflicted or unvalidated geometry.
- No contact/object/part work unless it changes MANO variables, solver constraint eligibility, or rendered physical uncertainty through a general factor record.
- No generic “collect more evidence” item; every item must name the variable, observation, intervention, predicted outcome, reusable interface change, and rendered consequence.

## Context

Project root: `/home/yiwen/ego_annotation`

Task memory root: `.memory/tasks/2026-06-12-pipeline-v18/`

Primary output root: `/data2/ego_annotation_outputs/`

Current final pipeline root: `/data2/ego_annotation_outputs/v18_full_pipeline/`

A800/GPU host: `yiwen@192.168.11.220`

Compute-placement invariant: do not run local GPU jobs, local model inference, or heavy local CPU inference on the user workstation unless the user explicitly authorizes that exact local run. SAM2, OWLv2, HaWoR, UniDepth, VLM/LLM batch inference, training, reconstruction, or any other fan/noise/thermal-heavy workload must run on the server/A800 or another non-local compute target. The local machine is for light repo edits, inspection, orchestration, and small non-heavy scripts only. Before launching any model or heavy job, make the compute target explicit in the command/session notes; accidental framework defaults such as local `cuda` are forbidden.

Write-heavy remote work must use truenas paths; `/mnt/user-home` on A800 is full or unsafe for large HaWoR intermediates.

Important current evidence roots:

- Complete-depth extension: `/data2/ego_annotation_outputs/v18_unidepth_extension/`
- Support-aware HaWoR exports: `/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/`
- Physical state schema: `/data2/ego_annotation_outputs/v18_physical_state_schema/`
- Part depth-fused reconstruction: `/data2/ego_annotation_outputs/v18_part_depth_fused_reconstruction/`
- Current final annotations: `/data2/ego_annotation_outputs/v18_full_pipeline/*/annotations_v18_full.json`

Important code areas:

- Final pipeline and graph: `scripts/run_v18_full_pipeline.py`
- Object reconstruction: `scripts/build_v18_depth_fused_reconstruction.py`
- Part reconstruction: `scripts/build_v18_part_depth_fused_reconstruction.py`
- Contact/nonpenetration: `scripts/build_v18_signed_nonpenetration_evidence.py`, `scripts/build_v18_triangle_nonpenetration_evidence.py`
- Occlusion: `scripts/build_v18_occlusion_depth_order_evidence.py`, `scripts/build_v18_occlusion_owner_graph.py`
- Part motion: `scripts/build_v18_articulation_fit_candidates.py`, `scripts/build_v18_part_se3_surface_residuals.py`

## Task specifications

### Inputs and outputs

- Process the full raw videos for `trash_1050` and `task5_tomato_960`.
- Final videos must match raw frame count and duration exactly.
- Final outputs must include overlay, world, side-by-side video, and backing JSON.
- Runtime for the default path must stay in the same order of magnitude as input duration.

### Current scientific gaps to close

**Task5 local support gap.** The current Task5 frontier is not blocked by missing ownership labels or by lack of a MANO optimizer. The defect is that the surface/contact evidence used to correct the hand is weaker than the correction being requested: hand/object discrepancies are millimetric, while independent tomato/object support remains roughly centimetric. Existing support-aware solves therefore collapse confident hand corrections back into bounded uncertainty. The local object/contact-patch support mechanism has now been implemented and corrected after clean-room review to measure the exact solver-consumed contact patch rather than an overbroad hand-adjacent patch. The exact-selector result supports the educated judgment that the available visible video/depth/object-pose evidence cannot support a tighter hand pose than the current bounded artifact: the selected patch is broad, centimetric-supported, and solver-inert relative to same-code global contact. Final V18 consolidation must expose that support-bounded Task5 uncertainty rather than forcing a local contact correction.

**Trash occluded-hand gap.** The hard hidden-hand span is still not reconstructed as a known pose, but the closure requirement for the available evidence is now met by a broad occluded-hand trajectory feasible/stress layer. Zero-observation rows are rendered as latent trajectory hypotheses constrained by visible-surface depth order, temporal smoothness, observation-validity state, and hard camera-z translation bounds; selected-vertex slack shows many rows cannot clear visible-surface constraints within the existing translation bound. The final artifact therefore renders broad uncertainty rather than pretending to know the hidden hand pose. A stronger claim would require genuinely new independent evidence or a higher-dimensional posterior with discriminating observations, not another point estimate.

**Versioning/name gap.** Existing V18 directories contain local counters (`v1`, `v2`, `v3`, `v5`, `v7`) from different artifact lines. They are not a global release ordering. The task spec treats only the explicitly named current frontier root as current; older H-prime/final-v7 roots are historical evidence and rejected deliverables.

### Hand state

- HaWoR metric MANO is the required physical hand source and the primary state to improve.
- WiLoR and RTMLib are supporting evidence, not replacements.
- Hand rows need metric 3D MANO state, provenance, camera/world-frame semantics, surface or reproducible MANO parameters, and uncertainty.
- Object/contact/occlusion/nonpenetration factors count as progress only when they update the final MANO hand state, tighten its uncertainty, or falsify a claimed hand-state correction.
- Inferred HaWoR rows may preserve temporal continuity, but physical claims require appropriate support and geometry.

### Object and part perception

- Object and part tracks must come from model-produced perception: VLM/object plan, OWLv2, SAM2, or equivalent documented model outputs.
- Do not encode case variation with category/color/action branches.
- Part-required objects must have part tracks and part geometry before part pose/contact claims.

### Geometry

- Object geometry means reconstructed object geometry when manipulated.
- Visible surfaces are measurements, not complete objects.
- Depth-fused meshes are candidates until they support object pose and physical constraints.
- For compact rigid object completion, the selected RGB hidden-surface prior is TRELLIS. TRELLIS output is not accepted geometry by itself: it must be metric-aligned to V18 object-owned depth fusion, overwritten by observed depth surfaces, checked against silhouette/depth/free-space evidence, and labeled as observed/inferred/rejected/uncertain before it can support object pose or MANO correction.
- The first execution target for the generic compact-rigid completion path is `task5_tomato_960/object:obj_tomato`; this is a target instance, not a category-specific method.
- Articulated objects require part geometry and part-relative motion.
- Deformable or unknown objects must remain physically uncertain unless modeled by a real mechanism.

### Corrected contact formulation

Contact is a latent physical relation and a means to improve or validate MANO hand pose annotation, not a delivered objective by itself and not measurement availability.

For each possible hand/object or hand/part interaction, V18 must distinguish:

- `C_{t,h,x}`: latent contact mode / contact phase state.
- `A_{t,h,x}`: latent contact patch, object-frame anchor, or contact manifold state where applicable.
- `H_{t,h}`: metric MANO hand state.
- `T_{t,x}`: object or part pose/state.
- Observations/factors: raw MANO-to-surface distance, 2D visual contact evidence, VLM/visual priors, mask/box association, depth order, contact ownership, temporal persistence, nonpenetration, and emitted pose-anchor measurements.

The following are observations or consequences, not contact state by themselves:

- raw surfaces being close;
- box/mask overlap;
- pair-contact proposal rows;
- accepted contact owner;
- manipulation episode labels;
- rendered contact lines;
- per-frame `contact_object_pose_anchor` or `contact_part_pose_anchor` emission;
- JSON support fields or labels.

A pose-anchor factor is an intermittent constraint handle generated by contact evidence. It must not be used as a per-frame existence gate for contact. Sustained physical contact may continue through missing or weak anchor observations, occlusion, mask instability, depth noise, or frames where object pose is already constrained by other factors.

Raw metric distance is a noisy likelihood/capture signal, not a hard final prerequisite. Annotation contact support, temporal mesh-distance ownership, visual priors, object-owned contact-patch depth, and proposal rows are observations on latent contact, not final truth and not automatic rejection triggers. The graph must be able to keep weak/conflicted candidates as low-weight or wide-deadband contact factors that can improve or bound noisy MANO hand state, and only secondarily object/part state when physically eligible. Final active contact is not progress unless it participates in a posterior explanation that changes or bounds `H_{t,h}` or falsifies a hand-state correction. If contact state is only a label while MANO joints/vertices/parameters remain unchanged, the principled improvement to the primary objective is zero.

Visual/VLM contact priors are allowed and encouraged as factors on the latent contact state. They may propose or weight contact from the 2D frame and action context, but they are not final contact labels. Final contact must still be checked against metric geometry, nonpenetration, occlusion/depth order, and temporal consistency.

Falsifiers:

- Active contact disappears only because a per-frame pose-anchor measurement is missing while visual manipulation, association, and physically plausible solved geometry remain continuous.
- Active contact appears from raw distance/proximity without independent association or plausible contact-state explanation.
- Active contact claims full object pose correction when only local contact evidence exists.
- Final rendered contact is driven by proposal rows, labels, or render code rather than graph-derived contact state.
- Contact labels change while final MANO joints/vertices/parameters and their uncertainty remain unchanged, yet the work is reported as progress.

### Factor graph and solver

- Variables must include hand state `H_{t,h}` as an optimized or explicitly corrected MANO state, plus object SE(3), part SE(3), articulation, latent contact state, contact patch/anchor state where applicable, nonpenetration, and occlusion ownership.
- The optimization objective must include hand-observation terms, contact/patch residual terms that can update or bound `H_{t,h}`, temporal hand regularization, nonpenetration/depth-order terms, and uncertainty terms. A formulation that keeps MANO fixed and only solves contact labels is insufficient for the corrected objective.
- Object/part state may be affected by geometry, temporal motion, contact, and nonpenetration factors when physically eligible, but that is secondary to improving or validating MANO hand annotation.
- A contact switch/label that does not participate in a latent contact-state explanation and does not constrain hand state is not enough for progress.
- Contact must use MANO hand surfaces and object/part/deformable geometry with uncertainty.
- Nonpenetration must use physically eligible object geometry; do not force rigid SDF logic onto deformable or articulated cases.

### Occlusion

- Occlusion must include visibility state, owner or unresolved state, depth-order evidence, temporal evidence, and uncertainty.
- Missing data alone is not occlusion reasoning.
- Do not silently fill occluded hands or objects as certain states.

### Verification

- Compile edited Python files.
- Run targeted checks after the mechanism is wired.
- Run the full pipeline only after final state consumes the change.
- Inspect the solved MANO hand state, timeline, and representative frames.
- Before/after evidence must show how MANO joints/vertices/parameters or uncertainty changed, or why the attempted contact/object/occlusion factor cannot legitimately change them.
- The primary evidence is improved/falsified hand-state annotation or concrete mechanism failure, not validator success or contact-count changes.

### V18 closure requirements

V18 can be called done only after the final full-video artifact makes a coherent physical MANO hand-state claim for both `task5_tomato_960` and `trash_1050`. Closure does not require pretending the sensors reveal hidden facts; it does require that every unresolved fact be represented as a specific hand-state uncertainty caused by an understood mechanism, not by stale dataflow, missing solver coupling, or an unexamined proxy.

Before claiming V18 done, the following concrete work must be true in the final overlay/world/side-by-side videos and backing state:

1. **Full-video interval MANO state is the delivered object.** Both videos have raw-duration overlay/world/side-by-side renders and backing interval MANO states from sanitized non-H-prime inputs. The render must distinguish optimized/claimed `H_t`, bounded/uncertain `H_t`, and context-only frames. Old H-prime/final-v7 artifacts are historical evidence only, never the closure artifact.

2. **Visible-hand spans are physically coherent.** Where a hand is visible, the rendered MANO surface must follow the visible hand evidence in image and depth well enough that a knowledgeable viewer would not see the hand floating through, missing, or contradicting the visible interaction. Any remaining visible mismatch must be explained as a bounded measurement/model error with a physical cause, not hidden behind a label or validator.

3. **Task5 local support gap is resolved.** The final Task5 artifact must either:
   - include a local object/contact-patch support state at the actual tomato interaction patch, estimated from model-produced masks, metric depth, temporal surface evidence, and current object pose/mesh hypotheses, with uncertainty tight and causal enough to justify any MANO correction it applies; or
   - explicitly render Task5 as support-bounded because the local patch remains too uncertain to constrain `H_t` beyond the current artifact. In that case the final state must show why the hand cannot be corrected more strongly from available video/depth evidence, rather than leaving an unexamined contact/nonpenetration claim.

4. **Trash occluded-hand gap is resolved.** The final Trash artifact must replace single hidden-hand point-estimate shifts with an occluded-hand trajectory posterior or feasible set through the hard occlusion span. This posterior must be grounded in temporal MANO dynamics, visible entry/exit evidence, occluder depth-order, visible-surface constraints, and observation-validity state. If the feasible set is broad, the artifact must visibly show broad hidden-hand uncertainty; if it is narrow, the artifact must show the mechanism that narrows it. A certain hidden-hand pose without this posterior is not closure.

5. **Object/contact/occlusion factors explain `H_t`, not themselves.** Contact, object pose, nonpenetration, ownership, and occlusion states count toward closure only when they constrain, bound, or explain the MANO hand trajectory. A contact label, object row, mask, face state, or render glyph that does not affect or justify `H_t` is not closure evidence.

6. **Remaining uncertainty is classified by cause.** Every important unresolved interval is classified as one of: normal measurement noise carried by uncertainty; a physical information limit from the available sensors; or an implementation/dataflow defect that must still be fixed. Closure is forbidden while an implementation defect remains in the final artifact. Closure is allowed with information limits only if the rendered hand-state uncertainty makes those limits visible and the backing state explains their cause.

7. **The final artifact has been consumed as an annotation.** Representative easy and hard frames from both videos must be inspected in overlay and world views: visible hand spans, contact/manipulation spans, Task5 local tomato interaction spans, Trash visible-to-hidden transition, and Trash hard occlusion. The closure claim must describe what the final artifact asserts for those spans and why those assertions are physically supported or bounded.

If any of these concrete requirements is not satisfied, V18 is not done. The next action must target the unmet physical requirement directly, not add validators, counters, manifests, or another artifact version name.

## Constraints

Unattended operating mode: continue in the same turn until the deliverables are finished, a true user-only decision is required, or the next action is high-risk/irreversible. Keep internal Deliverables/Completed/Next actions/Parked user decisions state; surface it only when yielding is necessary. Use strongest available verification and run clean-room adversarial review before any necessary yield.

Do not replace the hard mechanism with an easier artifact. A report, validator, manifest, support label, rejection label, gate, row count, render glyph, status field, or clean checklist is not progress unless it is attached to a mechanism that changed the final physical state or exposed a concrete mechanism failure.

Do not use proxy object pose. Centroids, boxes, masks, PCA-only axes, category primitives, visual patches, and rendered labels are not reconstructed object geometry or object pose.

Do not treat false-claim protection as refusal to solve. Guardrails are mandatory, but they must constrain the solver rather than replace it.

Do not reject noisy measurements by default. Carry uncertainty, downweight, smooth, or expose it unless the measurement is mechanically invalid.

Do not claim V18 closure while unresolved object geometry, object pose, object-contact coupling, latent contact state, part pose, or occlusion ownership makes the rendered MANO hand annotation physically incoherent or unsupported. If a component cannot be inferred from the available sensors, closure may only state that as an explicit information limit and render the corresponding hand-state uncertainty; do not hide it behind a checklist or gate.

Do not use BundleSDF, NeRF, or per-instance test-time neural-field optimization in the default path.

Do not use `sleep`, polling loops, or idle waits.

Do not stage broad paths or unrelated changes.
