# Egocentric Locomanipulation Annotation Pipeline v1

## Scope

Pipeline v1 targets RGB-only egocentric clips from `/data2/egoscale_demo_30h`. Each task folder contains one MP4 and one JSON action annotation. The v1 output is an inspectable annotation package per clip:

- video overlay with hand detections, hand keypoints, quality flags, and semantic caption
- 3D animation with head-camera trajectory, hand landmarks or MANO mesh when available, and optional object track
- side-by-side video plus 3D reconstruction with caption timeline
- per-frame annotation JSON and QC report

The RGB-only package does not contain depth, IMU, calibration, or ground truth. Metric 5 mm accuracy cannot be claimed globally from these inputs. v1 treats 5 mm as a refinement target for visible, locally constrained contacts after calibration and scale anchoring; the deliverable must expose uncertainty and gap intervals.

## Dataset Observations

The package has 1,757 MP4/JSON pairs, about 99 GB total. Sampled videos are 1920x1080, H.264, 30 fps. The JSON is action-level semantic metadata and camera-motion labels. Metadata loading must use `utf-8-sig`, because some JSON files include a UTF-8 BOM.

Frame-weighted camera-motion labels in the delivered metadata are approximately:

- small: 20.05 h, 84.3%
- large: 3.41 h, 14.3%
- medium: 0.33 h, 1.4%

The hard cases are split across two mechanisms. Tabletop tasks have useful scene texture and stable head pose, but hands, tools, yarn, cloth, and food frequently occlude fingers. Floor and mopping tasks have wider camera motion, blur, specular or low-texture surfaces, and long foreground tools that dominate the image.

## Coordinate Contract

All modules should emit data into one clip-local coordinate system:

- `world`: right-handed, meters, initialized from the first accepted camera frame
- `T_world_camera`: camera-to-world transform per frame
- `T_camera_world`: inverse transform for rendering and reprojection checks
- `T_world_wrist_left`, `T_world_wrist_right`: MANO wrist/root transforms when available
- `T_world_object_<id>`: rigid object pose when object tracking is enabled
- 2D image coordinates in pixels, with original video resolution recorded

Every numeric stream must include confidence and status. Missing measurements are represented as gaps with reasons. Interpolation is allowed only as a marked prediction interval.

## Modules

### 1. Preprocessing and QC

Inputs:

- MP4
- action JSON
- optional camera intrinsics and distortion

Operations:

- read metadata with `encoding="utf-8-sig"`
- sample representative frames and compute blur/exposure/visibility diagnostics
- extract hand/object candidate intervals
- build a caption timeline from existing action segments

Outputs:

- `metadata.json`
- `frames/` or decoded stream cache when needed
- `qc.json` with decode status, frame count, fps, resolution, and metadata consistency

### 2. Head Camera Localization

Preferred backends:

- MASt3R-SLAM for dense correspondence and difficult RGB-only scenes
- DROID-SLAM for strong learned visual SLAM when installation and GPU memory permit
- DPVO as a lighter learned VO baseline
- COLMAP only as an offline SfM diagnostic, not as the primary production path

Required behavior:

- use calibrated intrinsics when available
- run at lower resolution for pose, then render overlays at original resolution
- expose tracking loss and relocalization intervals
- align monocular scale using explicit priors only: hand anthropometry, floor/table plane, known tool/object size, AprilTag/calibration clips, or external measurements

Output:

- per-frame `T_world_camera`
- `scale_status`: `metric`, `scale_prior`, or `relative`
- SLAM QC: tracked frame ratio, lost intervals, reprojection or correspondence residuals, loop/scale drift indicators

### 3. Hand Pose and MANO

Preferred backends:

- HaMeR for MANO hand mesh recovery from RGB
- WiLoR for end-to-end 3D hand localization and reconstruction
- MediaPipe or similar hand detector only as a fast proposal/QC backend, not as the final MANO source

Required behavior:

- detect left/right hands and preserve identity through time
- output 2D keypoints, MANO pose/shape/camera parameters when available, and per-frame confidence
- smooth only over visible, consistent intervals
- mark occlusions, truncations, and low-confidence predictions

Output:

- per-frame hand keypoints
- MANO vertices, faces, pose, shape, wrist transforms when available
- hand QC: detection rate, left/right identity switches, high-occlusion intervals, reprojection quality

### 4. Semantic Captioning

v1 uses the existing JSON action segmentation as the primary caption source. A vision LLM can refine captions by sampling frames in each action interval and asking for visible events only. The captioner must preserve the existing timing and must not invent off-screen objects.

Output:

- action timeline with start/end frames, action label, visible-object notes, and final caption text

### 5. Optional Object Pose

Object pose is deferred unless the object is rigid, visible, and identifiable. Preferred backends:

- SAM2 or XMem for object mask tracking
- FoundationPose or MegaPose when CAD/reference models exist
- BundleSDF when the clip provides enough views for object reconstruction and tracking

Output:

- object masks
- `T_world_object_<id>`
- object QC: mask stability, rigid-fit residual, contact consistency, lost intervals

### 6. Physical Consistency

The refinement stage should optimize a sliding window with:

- camera pose residuals from SLAM/VO
- 2D reprojection residuals for hands and objects
- MANO pose and shape priors
- temporal velocity and acceleration penalties
- hand-object contact terms only where contact is detected
- soft non-penetration terms using object or coarse scene signed-distance fields
- rigid object constancy
- scale priors from explicit measurements

The optimizer should fail loudly when constraints disagree. A visually plausible overlay is not enough, because overlays can remain plausible even when the world trajectory is wrong.

## Single-Clip Example Policy

The first runnable v1 example should process one challenging but tractable clip:

`/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_7/20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4`

This clip is a home-kitchen tomato preparation task with both small and large camera-motion labels. It has visible hands, a manipulated food object, and enough head motion to expose localization uncertainty.

The example run is acceptable only if it exercises real backends for the modules it claims. A run that only decodes video, draws generic keypoints, or writes files is a baseline overlay, not a pipeline proof.

The example run is acceptable only if it produces:

- `overlay.mp4`
- `reconstruction_3d.mp4`
- `side_by_side.mp4`
- `annotations.json`
- `qc.json`

For the current local environment, MANO assets are available under `/data/dex_home/yiwen/mano_assets/mano/models/`. Template MANO availability is not per-frame MANO reconstruction. A minimum real-backend example must either run HaMeR/WiLoR or fit MANO parameters against observed evidence and report the fit residuals.

The current v1 runner is `scripts/run_v1_wilor_colmap.py`. It uses:

- WiLoR for per-frame MANO vertices, joints, 2D projections, hand side, and camera-relative hand translations.
- pycolmap for offline SfM camera poses in arbitrary-scale world coordinates.
- EgoScale JSON action segments for semantic captions.
- No object-pose backend. Object pose is marked `not_run` in the output JSON.

This runner should be reported as `camera_backend=pycolmap_sfm`, not as SLAM. DPVO was attempted as the SLAM/VO backend, but its CUDA extension failed to compile against the current Torch/CUDA API because the kernels call `AT_DISPATCH_FLOATING_TYPES_AND_HALF` with `tensor.type()`. The v1 output therefore answers the deliverable format with real hand reconstruction and real camera pose, while leaving the SLAM-specific risk unresolved.

## Quality Checks

A v1 clip package should be rejected or marked partial when:

- video decode fails or frame count is inconsistent
- hand detection rate is too low for the requested deliverable
- hand identity flips are visible
- head pose is relative-scale only while the report claims metric accuracy
- SLAM loses tracking without a marked gap
- predicted frames are rendered as measured annotations
- caption mentions non-visible objects or actions
- side-by-side output is blank, out of sync, or text occludes the view

The QC report should make failure modes visible instead of smoothing them away.
