# Pipeline V19 current epistemic state

## Current supported claim

Workbench items 3, 4, and 5 are complete for the corrected HOT3D keyboard artifact lineage. Workbench item 6 is active and has produced a causal chain on clip001849:

- Metric MANO must remain the WiLoR visible root-relative geometry on the HaWoR metric wrist trajectory for this clip. Contact-biased MANO/object fitting corrupts HOT3D hand metrics and is rejected.
- Contact-surface hypotheses must remain separate from accepted metric MANO. They can be rendered as uncertain surface/source-gap posteriors, not accepted contact, nonpenetration, or MANO correction.
- The remaining large hand/object gap is not primarily a target-selection artifact and not primarily a hand-root error. It is a coupled object-pose/geometry/camera/contact-truth problem.
- A translation-only object-pose stabilization from the support-reviewed anchor partially improves clip001849 object trajectory/source-gap metrics while preserving metric MANO exactly. P43 is accepted only as a clip001849 uncertain object/hand-separation posterior. P46 falsifies support-weighted stationary translation as a default cross-clip object-pose repair because it worsened clip001851 HOT3D object residuals while improving only hand/object proximity proxies.

Current full-duration artifact: P43 published the P42 support-weighted stationary object posterior and is the current Workbench-6 artifact for clip001849. It is a coherent negative/uncertain-contact annotation, not contact closure. The overlay/world/side-by-side videos are full-duration and state-driven; the banner states `rows 224 | source gap 106.3mm | normal 95.4mm | contact compat~0.002 | gap z 3.5 | joint shift 0.0px | metric MANO preserved | contact uncertain`. Visual review confirms the keyboard projection remains readable, metric MANO remains separate, and world-view links still show centimeter-scale hand/object separation. P43 supersedes P41 because it improves source-gap/normal/tangent metrics and has no visible artifact-level regression. See OPS 2026-07-03T06:38:57.

Recent autoresearch result: P42 support-weighted geometric-median stationary translation over visible-depth object pose observations improved clip001849 source gap (`114.7 -> 106.3 mm`), normal gap (`103.8 -> 95.4 mm`), tangent gap (`42.0 -> 37.6 mm`), and compatibility score (`0.000838 -> 0.002299`), with exact `0.0 m` MANO deltas. It did not improve object median translation residual and it worsened observed-to-mesh support (`17.95 -> 22.01 mm`) while improving mesh-to-observed support (`77.03 -> 71.33 mm`). P43 showed the visible-support tradeoff is acceptable as an uncertain clip001849 posterior. P46 then tested the same mechanism on clip001851 and rejected it as a generalized repair: source gap improved (`24.24 -> 19.02 mm`) but HOT3D object residuals worsened (`38.85 -> 55.75 mm` median; `94.53 -> 173.85 mm` p90) and rows dropped (`300 -> 266`). See OPS 2026-07-03T06:08:39, 2026-07-03T06:38:57, 2026-07-03T07:14:54, and 2026-07-03T07:25:00.

## Hand/MANO mechanism

HaWoR alone is not an acceptable hand foundation for clip001849: same-projection HOT3D GT review shows visible 2D/3D errors, and the failure is not explained by a simple focal/center mismatch. WiLoR is the better visible-hand candidate, but raw WiLoR metric translation is rejected. The supported hybrid is WiLoR root-relative visible MANO geometry on HaWoR metric wrist trajectory.

Direct hand-owned mask/depth evidence has not produced a safe MANO correction. On fixed-slice clip001851, HaWoR-projected SAM2 hand masks plus object-mask subtraction produced sparse right-hand surface support (`97/150` right masks, `1/150` left masks). Scale-preserving promotion accepted zero rows. A scale-relaxed right-hand branch required median scale `0.80`, median translation delta `63.65 mm`, and median base-joint reprojection shift `103.66 px`; HOT3D scoring worsened median wrist by `+6.69 mm`, joint MPJPE by `+17.54 mm`, and root-aligned MPJPE by `+31.70 mm` versus the support-gated candidate. Therefore scale-relaxed mask/depth refit is rejected as metric MANO correction, and future hand correction must use stronger hand-owned surface evidence that remains scale-preserving. See OPS 2026-07-03T07:00:00.

Raw V19/UniDepth hand-pixel depth is also rejected as a wrist/root replacement for clip001849. It worsens wrist/root error, while only tightly gated variants give negligible full-state benefit. See OPS 2026-07-02 depth-root entries.

Contact-coupled MANO correction is systematically invalid as metric hand state. It reduced surface-normal residuals by moving the hand with ~10 cm rigid transforms and worsened HOT3D wrist/MPJPE by ~8–11 cm on matched rows. Split-state repair preserves metric MANO exactly and stores the contact solution only as an uncertain surface posterior. See OPS 2026-07-03T03:20:00 and 2026-07-03T03:48:00.

P38 directly tested whether the remaining P35 source gap was caused by the source hand estimate. Replacing the selected V19 hand vertices with HOT3D GT MANO vertices against the same V19 object targets left the median gap essentially unchanged (`122.8 mm` V19 source gap vs `117.5 mm` GT-hand gap; selected hand-to-GT shift median `30.6 mm`). Therefore another MANO-root/contact-biased correction attacks the wrong variable. See OPS 2026-07-03T05:10:00.

## Object geometry and pose mechanism

The pruned keyboard mesh is better than the earlier slab but remains broad/solid and non-watertight. It can support visible surface/posterior visualization; it cannot support signed nonpenetration. The accepted pruning mechanism removes TRELLIS-completed geometry that projects outside SAM ownership and in front of observed depth. See OPS around the multiframe depth/SAM pruning entries and P34 inspection.

P37 introduced object-pose trajectory attribution against HOT3D object id `28` (`keyboard`). Because V19 completed-canonical and HOT3D/BOP object frames differ, raw object origins cannot be compared. The evaluator fits one constant transform between object frames and measures camera-coordinate residuals over time. Original V19 object trajectory had time-varying residuals (`77.1 mm` median translation, `4.93 deg` median rotation on 120 direct rows), so object/camera pose inconsistency is real and systematic. See OPS 2026-07-03T05:02:00.

Full static anchor pose was rejected. Holding both anchor translation and rotation fixed reduced object-origin translation residual (`19.7 mm` median over all visible frames) but worsened rotation residual (`11.85 deg` median, p90 `52.84 deg`) and worsened contact/source gap (`132.6 mm` median). Mechanism: per-frame rotations compensate camera/object orientation effects; full static pose breaks that compensation. See OPS 2026-07-03T05:22:00.

Support-weighted stationary translation is the current best rendered clip001849 object/hand-separation posterior, not a general object-pose intervention. Single-anchor translation stabilization improved P35 but left source gap `114.7 mm`; support-weighted geomedian translation over visible-depth pose observations reduced clip001849 to `106.3 mm` while preserving per-frame rotations and exact metric MANO equality on matched HOT3D rows. This remains partial progress: P42/P43 still have source gap z median `3.5`, low compatibility `0.002`, and many rows over `3σ`; object translation residual median did not improve. P43 renders the state honestly as uncertain contact/non-contact evidence. P46 shows the same mechanism can overfit contact/source-gap proxies on clip001851: it reduces retained-row source gap but worsens object residuals and support coverage. See OPS 2026-07-03T06:08:39, 2026-07-03T06:38:57, and 2026-07-03T07:14:54.

## Contact and target-selection mechanism

Global nearest-neighbor target selection was not the dominant failure. P35 localpatch changed object targets by `16.1 mm` median while source vertices were identical, but source gap did not improve (`117.6 mm` global vs `122.8 mm` localpatch); tangent residual worsened. Visual render showed 2D-local targets but long world-view links. See OPS 2026-07-03T04:39:00.

The Gaussian score is a contact compatibility residual, not a calibrated contact probability. Under the stated combined sigma `30.48 mm`, P35 localpatch source gap median `122.8 mm` corresponds to gap z `4.03` and compatibility `0.000298`. See OPS 2026-07-03T04:18:00 and 2026-07-03T04:28:00.

Visible object-owned surfels did not close the gap. P36 produced fewer rows (`79`), skipped `161` candidates for too few visible-support vertices, and still had median gap `126.7 mm`, z `4.16`, compatibility `0.000177`. This rules out hidden/completed-mesh target ambiguity as the primary remaining cause. See OPS 2026-07-03T04:45:00.

No current branch supports accepted contact ownership or signed nonpenetration. Rendered cyan/yellow/orange posterior/source-gap marks must be read as uncertain object-surface/source-gap diagnostics. The artifact should explicitly show contact uncertainty.

## Rejected mechanisms and claims

- HaWoR-only hand state as accepted clip001849 foundation.
- Raw WiLoR camera translation as metric V19 hand translation.
- Raw V19/UniDepth hand-pixel depth as a HaWoR wrist replacement on clip001849.
- Any contact-biased or mask/depth-driven MANO scale/translation/refit that requires scale change or large reprojection displacement, including large scales such as `1.35` and clip001851 scale-relaxed right-hand refit around `0.80`, as a metric hand correction.
- Promoting point-to-plane/contact-fitted MANO joints into metric hand state.
- Treating a source-gap link, row count, compatibility score, or JSON field as accepted contact.
- Treating localpatch target selection or visible-surfels targets as contact closure.
- Treating full static anchor pose as an accepted object-pose correction; it breaks rotation and worsens source gap.
- Claiming signed nonpenetration with the current non-watertight keyboard mesh.

## Live uncertainties

1. Renderer runtime: P43 raw rasterization completed but default ffmpeg encoding stalled and required P43b single-thread recovery. P45 face-budget probe is testing whether lower face budgets can preserve visual readability while reducing the CPU rasterization/encoding burden.
2. Object/camera source of residual: P44 rules out camera/head trajectory drift as the dominant residual mechanism. After one fixed V19-world to HOT3D-world transform, camera residuals are millimetric (`3.74 mm` median, `10.89 mm` p90) and uncoupled from object residuals (`r=-0.036`), while object residuals remain centimeter-scale (`33.57 mm` median, `156.23 mm` p90). The live mechanism is object pose/shape/partial planar support, not camera repair.
3. Contact truth: HOT3D GT hand to V19 object targets remains far, so the visible interaction may not contain true physical contact at the rendered target rows, or V19 object geometry/pose is still wrong. Current evidence cannot claim contact.
4. Generalization beyond keyboard HOT3D slices: split-state and attribution tools are mechanically general, but support-weighted stationary object translation is not a default/general repair. P46 falsified it on clip001851 by worsening object trajectory residuals and dropping support rows despite improving hand/object source-gap proxies.
5. Runtime: P41 still took `1053.6 s` for a 5-second clip, and P43 needed single-thread encoder recovery. P45 supports 3500-face presentation caps and single-thread frame-count-bounded ffmpeg as the next renderer default. The patch compiles under the A800 runtime env, produced a one-frame state-driven smoke render with `mesh_face_budget=3500` and `world_face_budget=3500`, and was synced to `/mnt/user-home/yiwen/ego_annotation_runtime/v19_bundle_a800/scripts/render_v19_rigid_state_artifact.py`; git commit remains pending.

## Next action

Commit the validated renderer runtime patch and task-memory updates with explicit staging. The next physical V19 intervention should target object shape/partial planar support/pose observability, or build a watertight thin keyboard sign mesh if signed nonpenetration is required. Do not render/promote P46; do not continue stationary-translation variants, anchor/radius sweeps, contact-biased MANO correction, scale-relaxed hand mask/depth refit, or camera repair.
