# Current RGB scene: restore effective rotation and primary-pad fitting

## Authorized objective and baseline

Owner requested modifying the current generic pipeline according to the old-demo audit and actually comparing the result. This is a mechanism patch/experiment within the standalone pipeline, not a new top-level pipeline version. Production base is clean commit `14a6d1378483b73cb0e1c51337c134217b754f09`. Implement only in this isolated worktree; do not modify frozen runtimes, the source run prefix, old demo, v13 final, model assets or prior candidates. Local commits only, no merge/push.

Success is a complete new JSON/RRD result whose actual rendered object rotation and hand–object fitting improve, not only a successful command or test. Output remains JSON/RRD-only; bounded diagnostic full-triangle stills for comparison are allowed, no default MP4 export.

## Causal account and discriminating predictions

The frozen v13 loses correct late rotation: first-to-last observed prior56.44°, graph posterior27.04°, final29.99° versus old demo112.85°. Current short RGB chains contain the large motion, while long PnP edges conflict by64–160°. Old demo first built an adjacent RGB orientation chain (~120°) and then conditioned cross-frame consistency (249→234 retained); its75→90 bad edge was excluded. Current graph instead starts from the weak observed-surface prior and admits all accepted PnP edges at nonzero weights.

Both hand fitters already use full MANO,31 keyframes,5 pads and240 steps. The old primary thumb had an image-support gate without the12mm pre-fit triangle-gap cutoff; the new code imposes12mm on all digits. On identical v13 diagnostics this alone suppresses22 left/8 right visually-supported thumb keyframes. Old image-only primary rule would allow28/29; current allows6/21. Do not presume that restoring gates alone fixes poor object pose.

Predictions: a repaired current-RGB initialization/consistency layer should remove the wrong f80–90 turn and preserve substantially more late rotation on this clip without reading old poses. A supported primary-pad objective must be active and have gradient for a finite 20–40mm initial visual gap; absent/no-contact/unsupported-primary evidence must NOT cause automatic attraction. The complete new fit should close visible pad gaps without distorting the raw-image projection or relying on signed generated normals. If rotation remains wrong, inspect current short-chain reliability or observed/RGB residual coupling; if rotation improves but gaps do not, inspect primary evidence/gate/reachability and fit gradients. Do not hop to an unrelated dense-contact model before testing these two mechanisms.

Old-demo audit/evidence (development/evaluation only):
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/comparisons/new_v13_vs_old_demo_20260911/old_demo_reuse/REUSE_ASSESSMENT.md`
and adjacent `implementation_evidence.json`; rotation evidence under `../rotation_cause/`. No old/reference output is permitted as a runtime fitting target or prediction input.

## A. Current-RGB rotation mechanism

- Reuse mathematically correct PnP/world relative motion: `D=Rt@Rs.T`, `d=tt-D@ts`. No extra canonical conversion or sign reversal.
- Build a deterministic local temporal RGB orientation initialization anchored at the actual selected metric anchor. Only short-time/local edges may build the initializer; use timestamps and timeline positions, not numeric frame-ID distance. Support reversed edges and nearest valid-source gap bridges. No hard-coded frame0,75,90,149 or input length. Disconnected/poor support retains source/latent uncertainty instead of fabricated measurements or a pipeline readiness failure.
- Retain original source pose rows and hashes immutably. Record new RGB seed and its provenance distinctly from the source prior; do not relabel the source prediction. Translations may start from current source priors; canonical anchor gauge must remain correct.
- Use conditioning-aware consistency against independently established local paths for cross-frame rotation/reprojection constraints. Old6–20° rule is a starting policy, not a guarantee; avoid circularly using a long edge to validate itself. Poor/disconnected initialization must not let a wrong ICP prior automatically reject correct large motion. Contradictory-edge evidence must not still leak through its reprojection factors.
- Preserve full SO(3)/SE(3), source depth eligibility and refreshed observed support. Weak observed geometry should not silently override well-supported large RGB rotation. Prefer reusing the old notion of rotation-vs-translation observability rather than fixing frame110–149 or turning off all geometry globally. Keep changes minimal and document any needed policy adjustment.
- Record seeds, rejected/downweighted edge reasons, local support and SO3/SE3 snapshots/bounded diagnostics needed to trace real effect. Do not add a new elaborate recovery framework.

## B. Evidence-selected primary contact pads

- Keep actual MANO LBS, fixed beta, correct joint/left-hand conventions, complete object triangles, unsigned closest-surface pad objective, image projection/mesh silhouette priors, temporal terms and existing bounded root/articulation/camera-origin similarity.
- Add a compact optional per-hand interaction-interval declaration for primary contact digits/pads, with actual current image rationale. Use anatomical finger IDs, not object category conditions. Runtime selects based on images, never assumes two thumbs contact everywhere. No-contact/unresolved/out-of-frame/invalid hand must not activate expanded reach. Missing primary designation retains conservative behavior for compatibility.
- Visually-supported primary pads may start farther than12mm, within a documented finite reachable range consistent with bounded similarity/root changes. Do not require a correction already to have succeeded before providing a gradient. Secondary pads keep conservative weak proximity. Generated surfaces remain visual hypotheses; do not promote metric-ineligible pixels or surface normals to measured contact/collision/SDF/nonpenetration authority.
- Runtime spec must explain the exact new judgment field/schema; authoring directory creation is allowed for named new judgment destinations. Preserve append-only events/create-only output rules. Do not permit scripts that secretly supply hand-selected case outputs.
- Record primary activation and fitted support separately from intended `likely_contact` judgment. Do not claim labels prove contact. No dense surface replacement or manual vertex/pose edits.

## Targeted checks before real execution

Use `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python -m unittest discover`; no pytest/install/download. Focused tests must cover: >90° cumulative rotation with several wrong long loops, nonzero selected anchor/reversed local edges, nonconsecutive frame IDs/time/rate invariance, absent/degraded support with uncertainty, no mutation of source prior, rejected rotation evidence not leaking via reprojection. Contact tests: explicit primary with20–40mm finite gap gets unsigned attraction gradient; secondary/noncontact/unresolved/no designation remains conservative; invalid/missing states are not promoted; actual MANO/mesh and output time contracts retained. Reuse relevant existing tests. Full suite is not required; document any existing unrelated failure honestly.

## Runtime and output

Build a new frozen bundle only from clean committed implementation. Proposed bundle root:
`/mnt/user-home/kupingxin/ego_annotation_runtime/rgb_scene_chain_contact_gpu5_20260911`
Asset config:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/runtime_asset_paths_gpu5_exclusive.json`

Source run:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/cold_raw_rgb_v1`
New create-only candidate ID: `rgb_chain_contact_repair_v1`.
Native cache:
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/frame0_sam3d_control_20260910`.
Use generic prepare-candidate ranking-first and strict native receipt import (no new neural SAM3D). SourceP01–P08 may be reused; execute candidate-local P09/P11/P13/pose/shape/MANO/export through the guarded native stage entry. Do not import v13 suffix outputs as a fresh prediction.

Native execution must use configured `ego-annotation.rgb-scene-runtime` via subagent and curated dedicated ShellGate target, same pinned environments/GPU5 exclusive UUID`GPU-e4ff72b8-fff7-21d0-2872-6aeecb56a91e`, tools serial, heavy bash timeout1200. GPU5 currently has another user's compute process; do not kill or switch it. Recheck at execution time; if still occupied, stop before producer launch and report the resource blocker, no polling/sleep.

Prepare launch commands from parsed absolute argv (shlex), not wrapped dispatch strings or substring replacement; check cwd, actual file existence, exact runtime path, command SHA and no self-recursion. Every startup read is an explicit absolute path. Bundle has no input directory. Use bounded JSON summaries. Do not re-run successful commands or overwrite a failed candidate. Same-protocol tool/infrastructure issues must stop for parent disposition.

After real export, evaluate only frozen outputs. Compare complete RRD world object/MANO transforms, unsigned triangle/pad gaps (not point-cloud distance mislabeled as contact), RGB projection, full-triangle same-camera stills with common styling, full timeline/visibility and final pair identity/portability. Baselines: v13 final and exact old demo. Keep baseline/reference data away from runtime agent. Either deliver demonstrated improvement with uncertainties, or report concrete regression/failed mechanism; do not certify quality from counts or cost decrease alone.
