# V19 metric camera contract + visible geometry（方案 A，前两步）

## 范围与结论边界

本分支以 additive experiment 完成方案 A 的前两步：

1. 建立 source-neutral、sensor-first 的 camera/image-transform contract；无传感器标定时保留原 UniDepth robust aggregation 作为显式 fallback。
2. 将现有 metric-depth raster 绑定到同一 camera contract，并在新目录重建 HOT3D clip-001851 keyboard 的 150-frame visible geometry、anchor visible surface 和 prediction-only A/B render。

本轮**没有**重跑或修复：HaWoR/MANO inference、object pose graph、SAM3D/TRELLIS、contact、collision、nonpenetration、P14/P15。旧 canonical run、冻结 SAM3D baseline 及 v1–v8 诊断目录均未覆盖。

最终 additive artifact root：

`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2/experiments/v19_metric_camera_contract_visible_geometry_v9`

sentinel：`.../experiments/v19_metric_camera_contract_visible_geometry_v9_exit.json`

结果：`status=ok`、`exit_code=0`、`elapsed_s=110`。

## 通用机制

### Camera/image planes

`scripts/v19_camera_contract.py` 定义 schema `v19_camera_image_transform_contract_v2`：

- 原生 K 明确位于 `calibration` plane；
- 每个 plane 满足 `K_plane = A_plane_from_calibration @ K_calibration`；
- planes：`calibration`、`source_rgb`、`manifest_rgb`、`sam2_mask`、`render`；
- resize 使用 OpenCV integer-pixel-center affine：`x_t = s(x_s + 0.5) - 0.5`。

`scripts/resolve_v19_camera_contract.py`：

- sensor-first，支持 top-level K 或逐帧 `[fx,fy,cx,cy]`；
- 严格校验 frame timeline、fixed-K tolerance、尺寸与 affine；
- sensor-first 接受 K 前必须证明 prediction input video 与 sensor-metadata source video SHA256 完全一致；有逐帧 sensor rows 时还严格核对 frame IDs 与 `time_s`；只有相同 row count 不够；
- 若 top-level sensor contract 没有逐帧 rows，timeline provenance明确标为 unavailable，不虚报 exact；
- supplied-but-missing/malformed/wrong-source sensor metadata 默认失败，不 silent fallback；
- 无 sensor metadata 时显式复用旧 `build_v19_calibration_contract.py` aggregation functions；旧 builder 未删除。

### Depth binding

`scripts/adapt_v19_depth_to_camera_contract.py`：

- depth values、dtype、shape、frame IDs、source size完全不变；
- 原 estimated K保留为 `source_estimated_intrinsics_fx_fy_cx_cy`；
- active K写入 `intrinsics_fx_fy_cx_cy`；
- 内嵌 camera-contract SHA256、named depth plane、K与 `A_depth_from_calibration`。

P09 builder 对 V2 fail closed：depth archive必须由 adapter绑定，且 contract hash、plane、150行K、affine必须与命令行 contract精确一致。传入未绑定或绑定到另一 contract 的 archive会失败；CPU test覆盖该路径。

### Mask/depth sampling

P09显式消费 `source_rgb` depth plane 和 `sam2_mask` plane：

- 960×960 mask → 1408×1408 depth使用 `cv2.INTER_NEAREST_EXACT`；
- 离散 sampler 与V2 half-pixel affine一致；
- actual raster与declared plane不一致时默认失败；implicit resize只能显式override；
- object-owned mask area/bbox从实际owned mask重算；raw SAM2 metadata单独保存。

## HOT3D sensor provenance

Camera metadata来自 launcher-provided prediction-side manifest：

`/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_benchmarks/hot3d_clips/v19_inputs_pinhole/clip-001851/input/raw_frame_manifest/manifest_for_eval.json`

- schema：`v19_hot3d_pinhole_raw_frame_manifest_v1`
- clip/stream：`clip-001851` / `214-1`
- camera adapter：`fisheye624_to_pinhole_v1`
- 150 rows，字段 `hot3d_pinhole_fx_fy_cx_cy`
- metadata SHA256：`c8fd1e1663eb8c263a6a3deab2db0d75d7c7942ba441001cbfb2198e668d42bd`
- 不含 hand/object/pose/contact/CAD字段；`evaluator_gt_consumed=false`

Source identity硬检查：

- prediction input：`/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/hot3d_clip001851_keyboard_pinhole/input.mp4`
- sensor metadata source：`.../clip-001851/input/clip-001851_214_1_pinhole.mp4`
- 两者 SHA256均为 `7a9baf0553e5dcfb4411b6cfabbe3a734f815ee4c5b014bdd965b9e547ec2310`
- 150/150 frame IDs及`time_s` exact，max delta `0.0 s`

当前run的manifest JPEG是同一1408视频经P01缩到960并重新编码，因此不应要求JPEG byte equality；V2显式建模1408 source plane到960 manifest plane。代表帧resize后内容相关系数约0.998，但最终authority依赖更强的源视频hash与timeline exact checks。

runtime spec禁止自行搜索 evaluator/benchmark root；metadata与source-video path必须由launcher提供。

## Camera contracts

Sensor-first：

- authority：`prediction_side_sensor_metadata`
- calibration/source raster：1408×1408
- K：`[609.85009765625,609.85009765625,707.4874877929688,702.32177734375]`
- `fallback.used=false`

Estimated fallback control：

- authority：`estimated_from_rgb_depth_model`
- K：`[537.0213035603387,537.0213035603387,722.2881469726562,720.8964233398438]`
- `fallback.used=true`
- 证明无sensor metadata时仍保留原接口，而非硬编码HOT3D K。

## 真实150-frame结果

### Depth invariants

全部通过：

- depth array/hash exact：true
- dtype：`float16`
- shape：`[150,1408,1408]`
- frame IDs/source size exact：true
- source estimated K preservation：true
- output K = contract `source_rgb` K：true
- P09 depth-contract hash/plane/K/affine binding：exact

Adapter只改camera metadata；没有以sensor K重新运行depth network，也不声称depth values自动更准确。

### Camera/MANO inheritance：无silent relabel

为保持additive geometry-only control，v9继承旧HaWoR camera poses与完整MANO arrays；它们没有在sensor K下重跑：

- 150/150 camera poses与旧base exact；
- 300-row full MANO bridge所有arrays exact；
- inherited HaWoR input为1408×1408，上游用`img_center=[width/2,height/2]`；
- source HaWoR K：`[537.0213012695312,537.0213012695312,704.0,704.0]`；
- active V19 K：`[609.85009765625,609.85009765625,707.4874877929688,702.32177734375]`；
- delta：`[72.8287964,72.8287964,3.4874878,-1.6782227] px`。

因此：

- `active_contract_reinference_required=true`；
- `builder_reestimated_hawor_camera_or_mano=false`；
- 150 camera rows、300 hand rows及300 hand uncertainty均携带mismatch；
- hand payload分别保存source HaWoR K与active V19 K，没有把inherited MANO静默标成sensor-K inference。

这明确阻止了“MANO/object已共同metric”或contact-ready声明。

### Visible geometry

- 150/150 rows使用sensor K；
- 150/150 mask/depth affine与contract一致；
- 150/150使用`INTER_NEAREST_EXACT`；
- anchor frame：109；
- anchor centroid world：`[0.1835396501,-0.1112362313,0.4470900474] m`；
- anchor extent：`[0.0835496261,0.4562746183,0.1874766516] m`；
- 相对旧estimated-K geometry，centroid delta median/p95/max：`45.58/64.63/67.19 mm`；
- raw/object-owned mask bytes不变；geometry变化来自K/ray contract；
- anchor：1495 downsampled points、Poisson 2375 vertices/4576 faces、hull 72 vertices/140 faces；仅visible surface，不是hidden/completed/collision geometry。

### Prediction-only reprojection render

旧/new visible 3D都经同一sensor K投回同一960 plane，并与同一prediction-owned mask比较；不是GT evaluator。

- 旧estimated-K 3D：inside-support median/p10 `0.55757/0.21162`
- 新sensor-K 3D：`1.00000/1.00000`
- new better：150/150
- frame 0/50/109/149 review可见旧红点系统偏移，新黄点回到prediction support。

视频：`.../visible_geometry_camera_contract_ab/old_estimated_k_vs_sensor_k_visible_geometry_reprojection.mp4`

- MPEG-4/yuv420p，1920×1030，150 frames，30 FPS，5.000 s
- ffmpeg full decode通过

## 诊断版本历史

- v1：早期schema，缺最终`intrinsics_coordinate_plane`要求。
- v2：camera/depth/base成功，pre-P09 Unicode checker bug导致P09未启动。
- v3：完整run后发现legacy `INTER_NEAREST`与half-pixel affine不一致。
- v4：切换`INTER_NEAREST_EXACT`后发现inherited HaWoR mismatch尚未表达。
- v5/v6：加入HaWoR mismatch、source/active K分离。
- v7：加入depth archive与contract hash/plane/K/affine闭环。
- v8：加入exact source-video SHA256与150-frame timeline binding。
- v9：最终run；修正top-level-only sensor contract的timeline provenance，真实HOT3D机制数值与v8一致。本冻结包只引用v9。

v1–v8 NAS目录保留为不可变诊断，不属于正式输出。

## 严格解释边界

本结果证明旧visible geometry链存在真实K/image-plane mismatch；source video、camera contract、depth metadata、mask raster transform与visible backprojection现在闭环，并产生可见reprojection改善。

它**不证明**：

- UniDepth values已按sensor K重新推理；
- inherited HaWoR camera/MANO与sensor K一致（v9明确记录不一致）；
- object trajectory或P15绿色层已修复；
- SAM3D/TRELLIS backend优劣变化；
- contact/collision/nonpenetration可用；
- visible Poisson/hull是完整或collision surface。

下一阶段应先在active contract下重跑/对齐HaWoR，重构per-pixel ownership与robust object pose graph，再做observed-only first-hit metric audit；不能把本轮提升为canonical backend/contact结论。

## 验证

- `py_compile`：通过
- CPU self-tests：13/13通过
- sensor source-video/timeline exact：通过
- sensor-first real run：通过
- estimated fallback real control：通过
- JSON/NPZ invariants：通过
- malformed/wrong-source/mismatched-depth fail-closed tests：通过
- `git diff --check`：通过
- MP4 ffprobe/full decode：通过
- representative frame/anchor review：通过

机器摘要：`reports/v19_metric_camera_visible_geometry_v9_summary.json`

NAS大artifacts与全部实质输入由`external_artifacts.sha256`锁定（809 files）；Git只保留代码、runbook、报告与review图。
