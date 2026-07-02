# Pipeline V19 current epistemic state

## Current supported claim

Workbench items 3, 4, and 5 are complete for the corrected HOT3D keyboard artifact lineage. Workbench item 6 is active and has produced a causal chain on clip001849:

- Metric MANO must remain the WiLoR visible root-relative geometry on the HaWoR metric wrist trajectory for this clip. Contact-biased MANO/object fitting corrupts HOT3D hand metrics and is rejected.
- Contact-surface hypotheses must remain separate from accepted metric MANO. They can be rendered as uncertain surface/source-gap posteriors, not accepted contact, nonpenetration, or MANO correction.
- The remaining large hand/object gap is not primarily a target-selection artifact and not primarily a hand-root error. It is a coupled object-pose/geometry/camera/contact-truth problem.
- A translation-only object-pose stabilization from the support-reviewed anchor partially improves object trajectory and source-gap metrics while preserving metric MANO exactly. It is accepted for a full render attempt, with strict uncertainty: contact remains unsupported.

Current full-duration artifact: P41 published the P40 translation-stabilized render and is the current Workbench-6 artifact for clip001849. It is a coherent negative/uncertain-contact annotation, not contact closure. The overlay/world/side-by-side videos are full-duration and state-driven; the banner states `rows 235 | source gap 114.7mm | normal 103.8mm | contact compat~0.001 | gap z 3.8 | joint shift 0.0px | metric MANO preserved | contact uncertain`. Visual review confirms the keyboard projection is readable, metric MANO remains separate, and world-view links still show centimeter-scale hand/object separation. See OPS 2026-07-03T05:55:00.

Live autoresearch branch: P42 tested a support-weighted geometric-median stationary translation posterior over visible-depth object pose observations, preserving per-frame rotations and metric MANO. It uses no hand/contact/GT in prediction state. It improved source gap (`114.7 -> 106.3 mm`), normal gap (`103.8 -> 95.4 mm`), tangent gap (`42.0 -> 37.6 mm`), and compatibility score (`0.000838 -> 0.002299`), with exact `0.0 m` MANO deltas. It did not improve object median translation residual and it worsened observed-to-mesh support (`17.95 -> 22.01 mm`) while improving mesh-to-observed support (`77.03 -> 71.33 mm`). P43 is rendering this state to decide whether the support tradeoff is visually acceptable. See OPS 2026-07-03T06:08:39.

## Hand/MANO mechanism

HaWoR alone is not an acceptable hand foundation for clip001849: same-projection HOT3D GT review shows visible 2D/3D errors, and the failure is not explained by a simple focal/center mismatch. WiLoR is the better visible-hand candidate, but raw WiLoR metric translation is rejected. The supported hybrid is WiLoR root-relative visible MANO geometry on HaWoR metric wrist trajectory.

Raw V19/UniDepth hand-pixel depth is also rejected as a wrist/root replacement for clip001849. It worsens wrist/root error, while only tightly gated variants give negligible full-state benefit. See OPS 2026-07-02 depth-root entries.

Contact-coupled MANO correction is systematically invalid as metric hand state. It reduced surface-normal residuals by moving the hand with ~10 cm rigid transforms and worsened HOT3D wrist/MPJPE by ~8–11 cm on matched rows. Split-state repair preserves metric MANO exactly and stores the contact solution only as an uncertain surface posterior. See OPS 2026-07-03T03:20:00 and 2026-07-03T03:48:00.

P38 directly tested whether the remaining P35 source gap was caused by the source hand estimate. Replacing the selected V19 hand vertices with HOT3D GT MANO vertices against the same V19 object targets left the median gap essentially unchanged (`122.8 mm` V19 source gap vs `117.5 mm` GT-hand gap; selected hand-to-GT shift median `30.6 mm`). Therefore another MANO-root/contact-biased correction attacks the wrong variable. See OPS 2026-07-03T05:10:00.

## Object geometry and pose mechanism

The pruned keyboard mesh is better than the earlier slab but remains broad/solid and non-watertight. It can support visible surface/posterior visualization; it cannot support signed nonpenetration. The accepted pruning mechanism removes TRELLIS-completed geometry that projects outside SAM ownership and in front of observed depth. See OPS around the multiframe depth/SAM pruning entries and P34 inspection.

P37 introduced object-pose trajectory attribution against HOT3D object id `28` (`keyboard`). Because V19 completed-canonical and HOT3D/BOP object frames differ, raw object origins cannot be compared. The evaluator fits one constant transform between object frames and measures camera-coordinate residuals over time. Original V19 object trajectory had time-varying residuals (`77.1 mm` median translation, `4.93 deg` median rotation on 120 direct rows), so object/camera pose inconsistency is real and systematic. See OPS 2026-07-03T05:02:00.

Full static anchor pose was rejected. Holding both anchor translation and rotation fixed reduced object-origin translation residual (`19.7 mm` median over all visible frames) but worsened rotation residual (`11.85 deg` median, p90 `52.84 deg`) and worsened contact/source gap (`132.6 mm` median). Mechanism: per-frame rotations compensate camera/object orientation effects; full static pose breaks that compensation. See OPS 2026-07-03T05:22:00.

Translation-only stabilization is the current best rendered object-pose intervention. Holding anchor translation while preserving per-frame rotations improved object trajectory relative to original and avoided the full-static rotation failure: translation residual `33.5 mm` median, rotation residual `3.85 deg` median, source gap `114.7 mm` median versus P35 `122.8 mm`. MANO matched-row deltas remain exactly `0.0 m` on 235 rows. P41 rendered this full-duration and the user-facing artifact is readable and honest, but it remains partial progress: p90 translation residual remains `159.7 mm`, source gap z median remains `3.76`, `179/235` rows exceed `3σ`, and the world view still shows long source-gap links. See OPS 2026-07-03T05:31:00 and 2026-07-03T05:55:00.

## Contact and target-selection mechanism

Global nearest-neighbor target selection was not the dominant failure. P35 localpatch changed object targets by `16.1 mm` median while source vertices were identical, but source gap did not improve (`117.6 mm` global vs `122.8 mm` localpatch); tangent residual worsened. Visual render showed 2D-local targets but long world-view links. See OPS 2026-07-03T04:39:00.

The Gaussian score is a contact compatibility residual, not a calibrated contact probability. Under the stated combined sigma `30.48 mm`, P35 localpatch source gap median `122.8 mm` corresponds to gap z `4.03` and compatibility `0.000298`. See OPS 2026-07-03T04:18:00 and 2026-07-03T04:28:00.

Visible object-owned surfels did not close the gap. P36 produced fewer rows (`79`), skipped `161` candidates for too few visible-support vertices, and still had median gap `126.7 mm`, z `4.16`, compatibility `0.000177`. This rules out hidden/completed-mesh target ambiguity as the primary remaining cause. See OPS 2026-07-03T04:45:00.

No current branch supports accepted contact ownership or signed nonpenetration. Rendered cyan/yellow/orange posterior/source-gap marks must be read as uncertain object-surface/source-gap diagnostics. The artifact should explicitly show contact uncertainty.

## Rejected mechanisms and claims

- HaWoR-only hand state as accepted clip001849 foundation.
- Raw WiLoR camera translation as metric V19 hand translation.
- Raw V19/UniDepth hand-pixel depth as a HaWoR wrist replacement on clip001849.
- Any contact-biased MANO scale/translation/refit, including large scales such as `1.35`, as a metric hand correction.
- Promoting point-to-plane/contact-fitted MANO joints into metric hand state.
- Treating a source-gap link, row count, compatibility score, or JSON field as accepted contact.
- Treating localpatch target selection or visible-surfels targets as contact closure.
- Treating full static anchor pose as an accepted object-pose correction; it breaks rotation and worsens source gap.
- Claiming signed nonpenetration with the current non-watertight keyboard mesh.

## Live uncertainties

1. P43 visual result: P42 improved source-gap physics but introduced a small visible-support residual tradeoff; the full render must decide whether the object overlay/world view remains sane. If P43 visibly worsens keyboard alignment, reject P42 despite better source gap and keep P41.
2. Object/camera source of residual: translation stabilization helps, but high p90 object residual remains. The deeper mechanism may be camera trajectory drift, partial planar pose underconstraint, object mesh shape error, or their combination.
3. Contact truth: HOT3D GT hand to V19 object targets remains far, so the visible interaction may not contain true physical contact at the rendered target rows, or V19 object geometry/pose is still wrong. Current evidence cannot claim contact.
4. Generalization beyond keyboard HOT3D slices: split-state and attribution tools are mechanically general, but the current strong evidence is clip001849 keyboard-specific.
5. Runtime: P41 still took `1053.6 s` for a 5-second clip, so renderer runtime remains a design blocker.

## Next action

Inspect P43 full render/publish when complete. Accept it as the current artifact only if the overlay/world/side-by-side remain physically readable and do not introduce a first-glance object misalignment. If P43 is worse visually, reject P42 as a state-level tradeoff and keep P41. Do not run anchor/radius sweeps; the mechanism being tested is robust stationary translation, not parameter search.
