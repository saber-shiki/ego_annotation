# First cold RGB attempt: camera consumer failure and correction

Run: `0188576c-cec8-4349-9205-d86709bd8188`, case `rgb_case_20260909_v1`.

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/cold_raw_rgb_v1
```

This was a native fresh-context execution from curated v3 on `dexrobot-Standard-PC-Q35-ICH9-2009`, physical GPU5 / logical cuda:0. Only original RGB bytes and the prepared input contract existed before launch. No sensor calibration or old clip prediction was supplied.

## Actual execution, not a completed annotation

P00/P01/P02 completed. P01 extracted the full150-frame/30FPS/5s timeline. Actual command records and model logs show:

| Subphase | Exit | Recorded duration |
|---|---:|---:|
| P03a unconditioned intrinsics probe |0|119.51s|
| P03b source-neutral camera resolver |0|1.46s|
| P03 conditioned depth |0|139.29s|
| P03c depth/camera binding |0|71.95s|

Before P04 HaWoR was invoked, the camera dispatcher failed:

```text
KeyError: 'source_plane_intrinsics_fx_fy_cx_cy'
```

No MANO inference, object reconstruction, physical interaction stages, final annotation MP4/RRD or visual acceptance followed. The run and its newly computed depth/calibration outputs remain frozen as a failed attempt, not reused to masquerade as another fresh run.

## Cause and smallest dataflow correction

The real RGB fallback emits the valid `v19_camera_image_transform_contract_v2` schema. Its authoritative source plane is in `image_planes.source_rgb`; unlike the sensor branch, it does not emit the optional top-level `source_plane_intrinsics_fx_fy_cx_cy` convenience field. Earlier dispatcher tests fabricated that field and therefore missed the real interface failure.

P04 now obtains K through the existing `v19_camera_contract.load_contract` / `plane_intrinsics` API, requiring the current frame IDs, actual source dimensions, named source_rgb plane and no implicit resize. This uses the calibrated image affine, not blindly the top-level calibration-plane K. Source values, square-focal model limitation and physical authority are unchanged. Validation now happens immediately after P03b, before spending the final depth pass on an unsupported camera contract.

The actual failed-run contract gives:

`[1248.5732649270853,1248.5732649270853,716.0255208333333,721.5922566731771]`.

The corrected reader returned these values and matched all150 bound-depth intrinsics rows. The camera JSON SHA remained `c51c93a34c2d8d1fed6b14c4c6418015504c7c828663723f9faa05acefa02058`. This read-only check did not rerun a model or change a calibration value.

## Execution-control findings

The agent used a200s tool wait for a camera group whose completed subcommands alone took over330s. Tool timeout did not terminate the model process: parent inspection observed the live P03c adapter after that timeout. The agent correctly did not relaunch or shut down the active broker.

The transcript also shows P02 executing `set -euo pipefail` in the persistent managed shell. A subsequent nonzero camera pipeline can therefore terminate that shell before its broker status marker. The broker/shell/socket later disappeared; the next log-read FileNotFoundError was consistent with the missing transport, while the real camera log remained present.

The runtime spec/profile now isolate shell options in subshells, specify an explicit1200s bash-tool budget for heavy stage groups, and prohibit competing managed reads/commands, relaunch or shutdown while a timed-out job may still run. Actual broker-generated transaction tests show the old persistent-errexit sequence exiting before status/follow-up, whereas a subshell reports exit7 and keeps the shell alive. No ShellGate provider code or protocol was changed by this correction.

Camera setup exceptions between subcommands now also persist a camera-stage blocker directly, rather than depending on later agent file access to record the failure.

## Regression evidence and limits

The dispatcher fixture now runs the **actual camera resolver** for both RGB fallback and sensor branches. Tests cover missing convenience alias, full K sent to HaWoR, actual sensor-affine source K differing from calibration K, frame/size rejection, square-focal refusal without averaging, and setup-blocker persistence. The old dispatcher loaded in memory fails this real-producer fixture with the same KeyError; no repository files were reverted for the negative control.

The focused runtime suite passes22tests and the combined unittest suite132tests. These tests include synthetic model stubs around the real camera resolver; they are not a new full cold prediction. Separate evidence is retained under `cold_v1_failure/` in the integration output directory.

Next: independent review, new immutable runtime candidate and fresh raw run, then execute the full native pipeline again. Do not edit this failed run, fabricate the missing field, relabel source readiness, or use a CLI/foreground-agent fallback.
