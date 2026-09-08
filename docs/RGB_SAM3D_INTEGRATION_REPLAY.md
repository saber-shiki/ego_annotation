# RGB / SAM3D integration replay checkpoint

This is an **integration replay**, not a fresh RGB-to-SAM3D prediction run and not a replacement for the published milk demo or V16 archive.

## What was actually executed

On `dexrobot-Standard-PC-Q35-ICH9-2009`, physical GPU5 / logical `cuda:0`, a broker-managed window under `ego_annotation:79` executed:

1. Original P19 state assembly from frozen P15/P16/P18b inputs, with a separate completion-report copy carrying the verified native-canonical SAM3D render-mesh binding. Original pose rows, P16 rows and P18b payload were checked unchanged.
2. The generalized real-MANO/triangle contact demo fitter, using those source poses and original HaWoR/MANO plus the supplied P17 judgments. This is a newly computed demo state, not the prior demo NPZ.
3. `assemble_v19_annotation_layers.py`, preserving physical state and demo state separately.
4. `render_v19_annotation_layers_rrd.py` and `render_v19_annotation_layers_video.py`, both consuming the same final package.

All four fit/export phases exited 0. Recorded phase durations: fit **74.64 s**, assembly **37.97 s**, RRD **12.56 s**, MP4 **97.90 s**. These are replay timings, not end-to-end prediction performance.

Artifacts:

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/milk_replay_v1/
  final/annotation_layers.json
  final/assets/
  final/world.rrd
  final/video/annotation_layers.mp4
  final/video/frame_*.jpg
  integration_contact_sheet.jpg
  fit/demo_state.npz
  fit/fit_report.json
  REPLAY_PROVENANCE.json
  ACTUAL_EXPORT_CHECKS.json
  execution.jsonl
```

Developer replay setup/run/readback scripts are retained in the adjacent `reproducibility/` directory. They are not part of the prediction runtime bundle.

## Actual render and geometry evidence

- MP4: **150 decoded frames**, **1920×640**, **30 FPS**, **5.000 s**. Panels are original RGB / estimated metric prediction / fitted demo hypothesis.
- Read back the actual RRD with Rerun 0.37 `experimental.RrdReader` Arrow chunks, not merely binary component names or writer-call counts.
- Demo mesh: **238,302 vertices / 476,616 faces**, exactly matching the bound full SAM3D canonical mesh. No display face deletion.
- Prediction mesh: the unchanged observed physical support surface, **3,596 vertices / 6,904 faces**. It is deliberately not relabeled as the complete generated object or a signed body.
- Both layers contain all **150** left/right MANO meshes, object transforms and common camera transforms. Exported hand vertices equal package arrays; maximum absolute readback error was **0** for this float32 source/export.
- Decoded world rotations, translations, timestamps and camera trajectory agree with the final package. Static object geometry is emitted once, while transforms remain per-frame.
- Original source hashes were rechecked unchanged.

The catalog-based RRD reader was unavailable because `datafusion` is not installed. No installation was performed; direct RRD Arrow-chunk readback succeeded using the already installed SDK. **RRD GUI playback was not separately inspected.**

## Visual inspection and limits

Inspected the ten full-sequence contact-sheet samples and full-size frames 99 and 149. Actual shaded object and hand triangle geometry is visible in both layers; the demo gives an intact, smoother object outline while the source physical layer retains its jagged partial support geometry. Camera/world alignment is not doubled. The wrist cutoff, approximate object top/cap geometry and local finger/image mismatch remain visible. No texture was pasted into the mesh; the raw image is visible only through the explicitly translucent overlay.

This checks the integrated render/state mechanism, not metric accuracy, true contact, nonpenetration, or fresh reconstruction. It does not establish superiority over the separately published V16-seeded visual demo. P16/P17/P18/P18b evidence and uncertainty remain source data rather than being inferred from demo fit.

## Code validation at this checkpoint

- 77 standard-library unittest cases passed, including the new dynamic timeline, layered source/mesh binding, SAM3D adapter stubs, RGB initializer and actual short MP4 tests.
- 22 late-window and 12 keyframe function tests passed via a direct focused runner: **111 focused tests total**.
- The first ad-hoc function loader omitted registration in `sys.modules` and failed in a test-local dataclass; the corrected loader passed all 34 function tests. No product/test code was changed to bypass that error.
- `pytest` remains unavailable; this is not a full-repository pytest claim.
- Demo and state-export lanes received independent scoped review after fixing all reported P1/P2 findings. The parent fresh-stage dispatcher/runtime closure still needs its own contract review and execution.

## Remaining work

The SAM3D-only runner, fresh RGB initializer, stage dispatcher and runtime spec are implemented/drafted, but **the current-revision cold pipeline has not run**. The remaining integration blocker is to close the curated runtime/script/config/asset launch contract and verify actual producer/consumer CLIs for the fresh stage graph. No historical video predictions may be admitted as fresh inputs. Reusable model weights are allowed. No push or merge has been authorized for this integration branch.

Artifact hashes:

- MP4: `871a38d0793ea9014b155a4c87dfc23eaacda94d39238401adc1fd09e2701dab`
- RRD: `c62d661f10ab1c0e4d28b4c60eb778ad8fbd59431d6e20d0c8faa59f66052b47`
- Contact sheet: `c1f32462ef76f2096f292a245aa1dbe0dc904bfb1a99ff2250c9286e3e06fe30`
