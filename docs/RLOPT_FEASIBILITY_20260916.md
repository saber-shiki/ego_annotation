# RL-opt integrated feasibility checkpoint — 2026-09-16

## Identity and verdict

- Upstream merged: `8b17958a0bccfba4b2951ce50fc9c8bc60a24c31`.
- Current tested prediction source: `9ef06b3a5e5deaba1105ce87b8cd5bfc4ee67ef6`.
- Development branch: `docs/refactor-rl-opt-baseline-20260915`.
- Production source closure: `/mnt/user-home/kupingxin/ego_annotation_runtime/rlopt_repairs_20260915/build_9ef06b3` (65 production Python files; tests/evaluation/docs/task-memory excluded).
- Latest run: `/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/rlopt_fixed_9ef06b3_20260916/milk_full`.

**End-to-end feasibility has not passed.** The latest run completed29 stages,
including real SAM3D complete-mesh generation, then rejected its metric bridge.
Final annotation JSON, RRD and MP4 do not exist. No physical annotation or
accuracy claim follows from the software tests or intermediate files.

## Repairs actually validated

| Mechanism | Commit in integration branch | Evidence |
|---|---|---|
| UniDepth RGB/K preprocessing | `c223333` | Real A800 same-frame/same-K/same-weights ray error2.049129°→0.006845°; actual conditioning K captured; original1° threshold retained |
| Astra pre-request connection recovery | `25f28ee` | Bad DNS address TLS timeout→another address of the same hostname; strict certificates/SNI/Host; real single POST succeeds, no POST replay |
| SAM2 unprompted prefix | `0618e91` | Missing0..26 before first prompt27 inferred with real reverse propagation;150/150identity checks pass; original123forward mask files byte-identical; prefix overlays inspected |
| Bounded rigidity review | `9ef06b3` | All15sheets/150frame facts retained in one request;9.60MB→2.693MB using existing full-sheet encoding ladder and3MB cap; real Astra rigid judgment returned with uncertainties |

Latest integrated CPU suite:220tests,218pass,2skip. Licensed MANO and actual
OWLv2 tokenizer tests were enabled. Skips are the optional PyTorch3D interpreter
fixture and an optional historical diagnostic. Test success is not model quality.
Independent review IDs: `eca16f96` (camera/network), `3a94dae1` (SAM2),
`ec9d123e` (rigidity); each returned no source-proven blocking findings.

## Latest genuine failure

The latest full150-frame/5-second milk run selected anchor48, completed rigidity
review, and generated a316,936-vertex/633,908-triangle SAM3D mesh. It stopped at
`metric_geometry` with `SAM3D observed-front geometry quality failed`:

- observed surface extent diagonal:0.280186m;
- sampled observed→generated median:0.026076m, fraction0.093068 versus allowed0.05;
- sampled P95:0.054474m, fraction0.194422 versus allowed0.18.

Exact triangle nearest-point measurement gives median0.026053m/P950.054445m.
Therefore this is not a40,000-surface-sample numerical artifact. The input mask
visually follows the correct held carton; no obvious object identity substitution
explains the failure.

## Camera-domain mismatch contributes materially

A standalone diagnostic reran only the same provisioned MoGe model on the exact
SAM3D input RGB with `force_projection=False`, matching its depth provider. It
did not regenerate the object mesh or alter prediction artifacts.

On the960×960input plane:

| Camera | fx=fy, pixels | Principal point | Same mesh triangle silhouette IoU |
|---|---:|---|---:|
| Pipeline constant UniDepth camera |257.1213|482.7812,481.3750|0.297814|
| Recomputed SAM3D-internal MoGe camera |440.0238|479.5,479.5|0.607639|

Focal ratio is1.71135. Under pipeline K the native mesh projects too small/high;
using the independently recomputed native K substantially improves alignment,
although IoU0.608 still leaves residual native pose/shape/occlusion uncertainty.
This is strong evidence of incompatible camera assumptions, **not proof that
camera mismatch alone explains all error**. Original native-run MoGe intrinsics
were not exported; the diagnostic is a same-input/weights recomputation, not a
claim of bit-identical historical intermediate state.

`bind_sam3d_metric` reinterprets the native mesh under the pipeline K and then
multiplies all camera-space coordinates by one scalar depth ratio. Such
camera-origin scaling leaves x/z and y/z unchanged, so it cannot repair a
projection mismatch. Tightening/loosening quality thresholds or resampling the
same surface does not solve that coupling.

## Next design decision (not implemented)

Before another full run, amend the native-geometry input contract rather than
rerolling SAM3D or loosening the gate. Recommended candidate: feed the current
shared metric camera/depth pointmap into the supported SAM3D pointmap path, with
explicit camera conventions and actual effective intrinsics recorded/verified.
Its internal pointmap-intrinsics inference also needs checking: merely passing
points must not silently restore a centered incompatible K. SAM3D must remain the
complete-mesh producer; generated faces remain display/unsigned hypotheses,
never signed collision or pose evidence.

This changes the existing default internal-MoGe conditioning contract and needs
an explicit design amendment and paired mechanism experiment. Do not import
old observations into a new production run or override the physical acceptance
flags. A diagnostic may read frozen artifacts but must remain separate and must
not acquire production receipts.

## Runtime feasibility also remains unmet

Latest run wall time was about25min11s for a5-second clip, and it stopped before
pose/visual fitting/render. Stage timings include nativeSAM3D204.6s,
ownership174.3s, conditioned depth129.6s, HaWoR127.1s and intrinsics probe117.9s.
Asset hashing, API calls and other stages add further overhead. This does not
meet the project's input-duration-order runtime objective; no performance
success is claimed.

## Durable diagnostic evidence

Under `/mnt/user-home/kupingxin/ego_annotation_runtime/rlopt_repairs_20260915/`:

- `VALIDATION_25f28ee.md`, `camera_live_25f28ee/result.json`;
- `SAM2_VALIDATION_0618e91.md`, `sam2_live_0618e91/prefix_review.jpg`;
- `rigidity_live_9ef06b3/result.json` and `request/` exact audit;
- `geometry_bridge_diagnostic/report.json` and `geometry.npz`;
- `geometry_bridge_diagnostic/camera_projection_comparison.json` / `.jpg`;
- `moge_native_camera_diagnostic/result.json` / `moge_geometry.npz`;
- `timings_9ef06b3.json` and `milk_full_9ef06b3.log` / `.exit`.

All prior failed runs are retained. At this checkpoint no owned prediction/model
job remains active; the next full run is paused on the camera-conditioning design
decision. No changes have been pushed to the remote repository.
