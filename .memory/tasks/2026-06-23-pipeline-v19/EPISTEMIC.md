# Pipeline V19 current epistemic state

## Current supported claim

For HOT3D `clip-001850`, the accepted prediction boundary is the runtime-owned OWLv2/SAM2 branch with integrated support-gated P18 interval MANO, frozen after regenerated full-duration renders. Freeze manifest SHA256: `06c8ff45a955421b341ceafb0548b27d2d791e9ea6495f1fc6d17ee4db67c594` under `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1` (OPS 2026-06-27T05:48:12+00:00). The physical claim is bounded: keyboard registration is materially repaired in direct frames, and support gating prevents zero-support contact/temporal terms from moving global hand translation, but contact, signed nonpenetration, exact late-frame object completion, and full object-pose accuracy are not proven.

For HOT3D `clip-001849`, the accepted prediction boundary is now the frozen crop-conditioned OWLv2/SAM2 support-gated run under `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001849_pinhole_a800_native_v1_supportgate`. Freeze manifest SHA256: `b7a82b2add62ca268f136982d76b5cbdc227cc7c52d4157b42d314242a3b3bea`; `hot3d_scoring_run=false` at freeze. P18 interval state SHA256: `3bf7f2e2d7a01e4bfc81d1d83a6fe08576f480ec01963111860131afe850e40c`; canonical render SHA256s are overlay `305cb7cac74e2b20a83df533bafc974c8724f4276e2fb4c3cc6b81e8d2d73bc2`, world `06304084146582778f9ecc50527170892823958823e5af3f00deed28664bb357`, and side-by-side `fb162c3c71ba4e53aa38c2b984f7aa34abcf67e7482aaddcdb431185b8e61e48` (OPS 2026-06-27T07:38:10+08:00 entry to be read with immutable details).

`clip-001849` physical render consumption supports a bounded rigid-keyboard artifact, not contact closure. The rendered green completed keyboard state follows the physical keyboard/key grid across sampled frames 30/75/120/149, but the completed mesh is broad/side-biased and the world view shows the orange/cyan MANO hypotheses separated from the green keyboard by large gaps. P20 labels summarize median contact-patch gaps around left `79.6 mm` and right `70.6 mm`; P18 active-set closure is false on the left and true on the right only under the uncertain factor model. Therefore P21 correctly accepts the rigid keyboard registration and support-gated interval MANO as uncertainty, not as coordinate-level MANO/contact/nonpenetration closure.

The current HOT3D 21-joint MANO quantitative claim is two-clip and mechanism-bounded. On `clip-001850`, support gating exactly restores wrist/root translation to HaWoR baseline and slightly improves median joint MPJPE (`0.047187152 m` baseline to `0.046050225 m`) while slightly worsening root-aligned median MPJPE (`0.022665785 m` to `0.022970178 m`). On held-out `clip-001849`, comparing against the runtime HaWoR baseline, the gate exactly preserves wrist error (`0.020158342317334703 m` median for both baseline and support-gated runtime), but the interval articulation worsens absolute joint MPJPE (`0.02971920653135254 m` to `0.03057108183250891 m`, +`0.000851875 m`) while improving root-aligned MPJPE (`0.022308057211138405 m` to `0.021197814917388345 m`, -`0.001110242 m`). The evaluator review sheet projects GT and predicted joints onto visible hands, so this is not a gross camera/review artifact.

## Current causal model

Object registration and MANO interval correction are separate mechanisms. The object-front-end failure mode was systematic segmentation-support contamination: earlier prompt routes allowed table/hand/arm support to define the keyboard, whereas OWLv2 text-grounded boxes plus SAM2 localized the support enough to fit/render a keyboard state. `clip-001849` exposed a second systematic object-geometry bug: P12 originally consumed the raw full frame instead of the object-isolated RGBA crop, contaminating TRELLIS with scene context. The crop resolver repair restored the intended per-instance mesh-prior mechanism.

The original P18 MANO failure was systematic, not normal measurement noise: contact/temporal/object terms moved the global wrist/root translation even when selected visible-surface support count was zero. The support gate implements the physically narrower rule that global wrist/root translation may change only with selected visible-surface depth-order support; otherwise the source HaWoR wrist/root anchor is preserved while wrist-relative articulation can still change.

`clip-001849` confirms that support gating is doing the intended wrist/root job but does not guarantee MPJPE improvement. All 300 P18 rows had `output_translation_gate`; selected visible-surface support was zero for all rows; wrist error versus the runtime HaWoR baseline is exactly unchanged. The remaining metric change is wrist-relative articulation: it improves root-aligned MPJPE but worsens absolute joint MPJPE. The mechanism-level implication is that the interval solver currently provides an articulation hypothesis under uncertainty, not a general hand-accuracy improvement.

Runtime is still a concern. `clip-001849` P18 took roughly 21 minutes for a 5-second clip and used 240 active contact-patch rows versus 196 on `clip-001850`. This is explainable by the current LBFGS/active-set design but violates the intended runtime scale; it is not evidence against the support-gate physics, but it is evidence that the default interval solver is too slow for broader deployment.

## Rejected boundaries and mechanisms

- Wrong raw-mesh freeze `daa3978f641691ef148720a75a9ec3bea35a22e55b6c01eb6a0eea0ed85f2a3c`: rejected because P18/P19 consumed raw P12 TRELLIS mesh while P15 poses were in P13 completed-canonical mesh coordinates.
- Mesh-frame-repaired freeze `2d999fbc4f51e375b4a9eb7277943bba74ce542cf0c613ece3479c888a267080`: rejected because the object still visibly followed contaminated mask/surfel support.
- Positive-click-derived prompt boxes and explicit agent-authored box prompts: rejected because they still allowed SAM2 to select broad tabletop/hand/background support.
- P09 960/1408 intrinsics/depth-size suspicion: ruled out for the OWLv2 branch; the earlier cyan surfel mismatch was diagnostic display scaling.
- HOT3D evaluator coordinate/review failure: ruled out by evaluator review sheets showing GT and predictions on visible hands.
- Pure articulation-error explanation for the original `clip-001850` P18 regression: ruled down by decomposition and support-gated runtime scoring. Root-aligned errors stayed effectively fixed while absolute errors followed wrist/root translation.
- `clip-001849` old P12 blocker is superseded negative evidence, not an active blocker. It records the raw-frame conditioning failure; crop-conditioned P12 completed and downstream P13-P21 artifacts were frozen.
- P19 verifier logic that expects `manifest.videos` or lets `Path("")` pass is invalid; the renderer writes `manifest.outputs.*`. The durable spec is patched for future launches.

## Live uncertainties

1. Broader HOT3D slice: Workbench item 6 fixed slice remains incomplete because `clip-001851` has baseline/pinhole inputs but not a V19 support-gated runtime freeze/evaluation.
2. Generalization of MANO benefit: support gating generalizes as a wrist/root safety mechanism across `clip-001850` and `clip-001849`, but MANO accuracy benefit does not monotonically generalize. The current interval articulation can improve root-aligned metrics while worsening absolute joint MPJPE.
3. Positive-support behavior: both scored support-gated freezes have zero selected visible-surface support rows, so the pipeline has not evaluated whether physically supported global translation corrections improve or harm MANO metrics.
4. Contact/nonpenetration: still unresolved. `clip-001849` P16 reports non-watertight completed/sign meshes, P17 rows are uncertain factor inputs, P18/P20 labels retain large contact gaps, and P21 explicitly keeps contact/nonpenetration unaccepted.
5. Full MANO surface metrics: unavailable for P18 interval states because the solver stores optimized joints and sampled vertices, not full optimized MANO vertices.
6. Object metrics: current HOT3D scoring covers 21-joint MANO only. Object pose/contact/occlusion metrics remain unimplemented or unsupported by the current evaluator path.

## Next action

Preserve the `clip-001849` evidence checkpoint, then continue Workbench item 6 on fixed `clip-001851` using the repaired runtime spec. Do not tune thresholds on `clip-001850`/`clip-001849` before running the remaining fixed slice. For `clip-001851`, predict that the support gate should again protect zero-support wrist/root translation; if absolute MPJPE still worsens while wrist is preserved and root-aligned metrics improve, the interval solver should be treated as an articulation hypothesis rather than a MANO-accuracy improvement.
