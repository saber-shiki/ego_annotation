# Ego Annotation 可获取观测与输出目录说明

## 1. 文档目的

本文整理 `ego_annotation` 项目能够生成的观测、推断状态、文件格式、坐标语义和实际存储目录，并以当前已经完成的本机 V19 真实运行作为实例。

这里严格区分三类数据：

1. **直接输入观测**：原始 RGB、帧号、时间戳等；
2. **模型测量观测**：单目深度、相机轨迹、MANO 候选、检测框、Mask、可见三维表面等；
3. **推断物理状态**：完整物体 Mesh、物体位姿、遮挡所有权、接触候选、非穿透状态等。

检测框、关键点、Mask、深度图或 JSON 行本身不是最终物理状态。最终状态应保留来源、坐标系和不确定性，并能够驱动渲染结果。

当前执行合同与部署记录可进一步参见：

- [`docs/ego_annotation_pipeline_optimization_and_losses.md`](ego_annotation_pipeline_optimization_and_losses.md)：当前 V19 输入处理、优化问题、目标函数、loss wiring 和实际优化结果
- [`runtime/v19_runtime_spec.md`](../runtime/v19_runtime_spec.md)
- [`docs/local_deployment.md`](local_deployment.md)
- [`docs/pipeline_v16.md`](pipeline_v16.md)
- [`docs/pipeline_v17.md`](pipeline_v17.md)
- [`docs/pipeline_v18.md`](pipeline_v18.md)

---

## 2. 当前目录和运行实例

本文使用以下路径别名：

```bash
REPO=/mnt/user-home/kupingxin/ego_annotation

OUT=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs

RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2

CASE=hot3d_clip001851_keyboard_pinhole

OBJECT=keyboard
```

当前真实输入视频：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/hot3d_clip001851_keyboard_pinhole/input.mp4
```

视频规格：

| 属性 | 值 |
|---|---:|
| 分辨率 | 1408×1408 |
| 帧数 | 150 |
| FPS | 30 |
| 时长 | 5 秒 |

重要目录说明：

- `$REPO/data/` 当前基本为空；大文件和运行结果不存放在 Git 仓库内。
- `/home/kupingxin/ego_annotation` 和 `/mnt/user-home/kupingxin/ego_annotation` 指向同一个目录。
- 当前权威完整运行是上述 `_v2`；`_v1` 在 P03 阶段失败。
- `.runtime/paths.env` 中的 `EGO_RUNTIME_RUN_ROOT` 仍指向 `_v1`，使用时不能直接照搬。
- 文档中的 `/data2/ego_annotation_outputs/...` 多为历史机器路径；当前部署优先使用 `/mnt/truenas-user-home/kupingxin/...`。
- `/mnt/user-home/kupingxin/ego_annotation_runtime/` 保存隔离 runtime bundle 和 session，不是观测结果目录。

---

## 3. 总体观测类别

当前 V19 状态本体覆盖以下类别：

| 类别 | 主要内容 |
|---|---|
| RGB 与时间轴 | 原始视频、逐帧 RGB、帧号、时间戳、FPS、分辨率 |
| 相机与深度 | 相机内参、头部/相机轨迹、度量深度、尺度来源 |
| 手部状态 | 左右手 MANO 顶点、关节、姿态、形状、全局变换、可见性和不确定性 |
| 物体二维观测 | 对象计划、开放词汇检测框、SAM2 Mask 和轨迹 |
| 物体三维观测 | Mask+Depth 可见表面、三维点、表面 Mesh |
| 物体物理状态 | 物理分支、完整 Mesh、刚体位姿或其他分支状态 |
| 可见性与遮挡 | 可见/部分可见/遮挡/出框/未解，遮挡者所有权 |
| 接触 | 接触、可能接触、非接触、接触 patch、距离和不确定性 |
| 非穿透 | 手物最近距离、穿透候选、signed-distance 可用性和不确定性 |
| 渲染与审计 | Overlay、世界视图、side-by-side、QC、来源和运行日志 |

---

## 4. RGB 帧、时间轴和视频元数据

### 4.1 可获取内容

每帧可得到：

- `frame_idx`
- `time_s`
- RGB 文件路径
- 原始视频尺寸
- runtime 图像尺寸
- FPS、总帧数和总时长

### 4.2 输出目录

```text
$RUN/input/raw_frame_manifest/
```

主要文件：

```text
$RUN/input/raw_frame_manifest/manifest.json
$RUN/input/raw_frame_manifest/v19_raw_frame_manifest_report.json
$RUN/input/raw_frame_manifest/rgb/000000.jpg
...
$RUN/input/raw_frame_manifest/rgb/000149.jpg
```

当前 runtime RGB 为 960×960，共 150 张；原视频尺寸仍为 1408×1408。

生产脚本：

```text
$REPO/scripts/build_v19_raw_frame_manifest.py
```

---

## 5. 单目度量深度

### 5.1 可获取内容

每帧可得到：

- 度量深度图；
- UniDepth 预测焦距；
- `[fx, fy, cx, cy]`；
- 深度中位数、p05、p95；
- 有限值、图像尺寸和运行 QC。

### 5.2 核心目录

```text
$RUN/measurements/depth_slam/unidepth_full_frame/
```

核心 NPZ：

```text
$RUN/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz
```

主要数组：

```text
frame_idx                    (150,)
depth                        (150, 1408, 1408), float16
source_size                  (2,)
focal_px                     (150,)
intrinsics_fx_fy_cx_cy       (150, 4)
```

单位约定：

- NPZ 中的 `depth` 单位为米；
- `depth/*.png` 为 `uint16` 毫米深度。

逐帧深度 PNG：

```text
$RUN/measurements/depth_slam/unidepth_full_frame/depth/000000.png
...
$RUN/measurements/depth_slam/unidepth_full_frame/depth/000149.png
```

可视化：

```text
$RUN/measurements/depth_slam/unidepth_full_frame/stills/frame_000000.png
...
```

QC：

```text
$RUN/measurements/depth_slam/unidepth_full_frame/qc_unidepth_full_frame_v3.json
```

当前深度约覆盖 0.194 m～4.891 m。它来自单目模型，不是 RGB-D 传感器真值。

生产脚本：

```text
$REPO/scripts/run_unidepth_full_frame_v3.py
```

---

## 6. 相机内参

### 6.1 可获取内容

- 相机模型；
- 固定或逐帧内参；
- 焦距与主点；
- 水平/垂直 FOV；
- 内参来源和聚合方式。

### 6.2 输出目录

```text
$RUN/state/calibration/
```

主要文件：

```text
$RUN/state/calibration/v19_camera_calibration_contract.json
$RUN/state/calibration/v19_camera_calibration_intrinsics.npz
```

当前选定固定内参：

```text
fx = 537.0213
fy = 537.0213
cx = 722.2881
cy = 720.8964
```

当前内参来自 UniDepth 150 帧预测内参的中位数聚合，不是数据集真实标定。

生产脚本：

```text
$REPO/scripts/build_v19_calibration_contract.py
```

---

## 7. 相机/头部位姿轨迹

### 7.1 可获取内容

每帧可得到：

- `R_c2w`：camera-to-world 旋转；
- `t_c2w`：camera-to-world 平移；
- `T_world_camera_metric`；
- 相机在世界坐标中的位置；
- HaWoR SLAM 中间轨迹、视差和尺度。

### 7.2 核心位置

相机轨迹数组保存在：

```text
$RUN/measurements/hand_candidates/hawor_world/hawor_world_hands.npz
```

数组：

```text
R_c2w        (150, 3, 3)
t_c2w        (150, 3)
```

逐帧统一 annotation 中的相机状态：

```text
$RUN/state/base_annotations/annotations_v19_base.json
```

相关字段：

```text
frames[t].camera.T_world_camera_metric
frames[t].camera.position_world_m
frames[t].camera.intrinsics_fx_fy_cx_cy
frames[t].camera.v19_camera_pose_source
```

HaWoR/SLAM 中间结果：

```text
$RUN/input/hawor_sequence/$CASE/SLAM/hawor_slam_w_scale_0_150.npz
```

包含：

```text
tstamp
disps
traj
img_focal
img_center
scale
```

当前运行没有独立的最终 `camera_trajectory.npz`；推荐从 HaWoR NPZ 或逐帧 annotation 读取相机状态。

---

## 8. 双手 3D MANO

### 8.1 可获取内容

左右手逐帧可以得到：

- 778 个 MANO 顶点；
- 21 个 3D 关节；
- MANO 三角面拓扑；
- 根节点旋转；
- 45 维手部姿态轴角；
- 10 维 shape betas；
- 世界坐标平移；
- 左右手身份；
- 2D 检测框及分数；
- track id；
- 同帧是否实际检测；
- 是否由时间模型补全；
- 相机/世界坐标语义和来源。

### 8.2 原始 HaWoR 候选

目录：

```text
$RUN/measurements/hand_candidates/hawor_world/
```

主要文件：

```text
$RUN/measurements/hand_candidates/hawor_world/hawor_world_hands.npz
$RUN/measurements/hand_candidates/hawor_world/qc_hawor_world_hands.json
```

主要数组：

```text
left_vertices_world_m       (150, 778, 3)
left_joints_world_m         (150, 21, 3)
left_trans_world_m          (150, 3)
left_root_orient_axis_angle (150, 3)
left_hand_pose_axis_angle   (150, 45)
left_betas                  (150, 10)
left_valid                  (150,)
left_detected_same_frame    (150,)
left_det_box_xyxyscore      (150, 5)

right_vertices_world_m      (150, 778, 3)
right_joints_world_m        (150, 21, 3)
...
```

当前覆盖：

| 手侧 | 有效轨迹 | 同帧实际检测 |
|---|---:|---:|
| 左手 | 150/150 | 136/150 |
| 右手 | 150/150 | 127/150 |

`valid=150/150` 包含时间补全，不能解释为 150 帧都存在原始可见检测。

生产脚本：

```text
$REPO/scripts/remote_run_hawor_export.sh
$REPO/scripts/export_hawor_world.py
```

### 8.3 相机坐标与世界坐标桥接

```text
$RUN/state/base_annotations/v19_mano_bridge_from_hawor_world.npz
```

包含：

```text
vertices_current_v18_world_from_hawor_projection_relift_m
joints_current_v18_world_from_hawor_projection_relift_m
vertices_current_v18_camera_m
joints_current_v18_camera_m
vertices_hawor_camera_m
```

总计 300 行，即 150 帧×左右两手。

### 8.4 当前 canonical MANO 状态

推荐下游使用：

```text
$RUN/measurements/mano_interval_correction/keyboard_0_149_surface_hypothesis_metric_mano/$CASE/v18_joint_mano_interval_trajectory_state.json
```

虽然文件名仍是 `v18_*`，它是当前 V19 P18b 输出。当前 300 行全部采用：

```text
joint_state_policy = hawor_npz_metric_mano_preserved
```

其含义是保留 HaWoR metric MANO joints/root，并把接触优化结果作为独立的不确定表面假设，而不是无证据地修改整只手的根位姿。

### 8.5 HaWoR 内部中间目录

```text
$RUN/input/hawor_sequence/$CASE/
```

主要内容：

```text
extracted_images/
cam_space/0/
cam_space/1/
tracks_0_150/model_tracks.npy
tracks_0_150/model_masks.npy
tracks_0_150/model_boxes.npy
tracks_0_150/frame_chunks_all.npy
SLAM/hawor_slam_w_scale_0_150.npz
world_space_res.pth
```

这些是中间测量，不应代替 canonical MANO 状态。

---

## 9. 物体语义、对象列表和物理分支

### 9.1 可获取内容

- 物体 ID 和自然语言描述；
- 目标对象及排除对象；
- 预期可见区间和活跃区间；
- 开放词汇检测 prompt；
- 刚体、关节、可变形等物理分支概率；
- anchor frame；
- 分支决定依据和不确定性。

### 9.2 对象语义计划

```text
$RUN/measurements/object_candidates/object_plan_agent.json
```

当前对象：

```text
object_id = keyboard
track_id  = keyboard
```

该文件包含：

```text
physical_branch_hypotheses
evidence_frames
expected_visible_intervals
detector_text_prompts
detector_prompt_frames
detector_active_interval
excluded_entities
```

### 9.3 物理分支决定

```text
$RUN/state/physical_branch_decisions/keyboard.json
```

当前决定：

```text
decision = rigid
status   = accepted_with_uncertainty
```

### 9.4 Anchor 决定

```text
$RUN/state/anchor_decisions/keyboard.json
```

当前选中：

```text
anchor frame = 109
```

Anchor 候选和 review：

```text
$RUN/measurements/object_geometry/anchor_candidates/keyboard/
```

---

## 10. OWLv2 物体检测框

### 10.1 可获取内容

- 文本 prompt；
- 每个候选 bbox；
- 置信分数；
- 对应文本标签；
- 最终选中的检测框；
- 提示图像尺寸和坐标系。

### 10.2 输出目录

```text
$RUN/measurements/object_candidates/object_box_prompts_owlv2/keyboard/
```

主要文件：

```text
object_point_prompts_vlm.json
v19_owlv2_object_box_prompt_report.json
```

当前 prompt：

```text
keyboard.
computer keyboard.
```

提示框坐标系：

```text
manifest_pixels_960x960
```

检测框 review：

```text
$RUN/renders/review_frames/P06_owlv2_object_boxes/
```

生产脚本：

```text
$REPO/scripts/build_v19_owlv2_object_box_prompts.py
```

检测框仅用于提示 SAM2，不是物体几何或物体位姿。

---

## 11. SAM2 物体 Mask 和时序跟踪

### 11.1 可获取内容

每帧可得到：

- 二值物体 Mask；
- 可见标志；
- bbox；
- Mask 中心；
- Mask 面积；
- Mask 文件路径；
- 完整时间轨迹。

### 11.2 输出目录

```text
$RUN/measurements/object_tracks/sam2_owlv2_box_points/
```

轨迹 JSON：

```text
$RUN/measurements/object_tracks/sam2_owlv2_box_points/keyboard/sam2/sam2_track.json
```

逐帧 Mask：

```text
$RUN/measurements/object_tracks/sam2_owlv2_box_points/keyboard/sam2/sam2_masks/000000.png
...
$RUN/measurements/object_tracks/sam2_owlv2_box_points/keyboard/sam2/sam2_masks/000149.png
```

Mask 分辨率为 960×960。

QC：

```text
$RUN/measurements/object_tracks/sam2_owlv2_box_points/keyboard/sam2/qc_sam2_vlm_points_track.json
$RUN/measurements/object_tracks/sam2_owlv2_box_points/qc_sam2_multiobject_points.json
```

Overlay 视频：

```text
$RUN/measurements/object_tracks/sam2_owlv2_box_points/sam2_multiobject_overlay.mp4
```

Review：

```text
$RUN/renders/review_frames/P07_sam2_overlay/
```

当前键盘轨迹为 150/150 帧有 Mask。

生产脚本：

```text
$REPO/scripts/run_sam2_vlm_points_multiobject.py
```

---

## 12. 手部剔除后的物体所有权 Mask

为了避免手覆盖键盘时把手像素反投影成物体表面，项目会生成“物体拥有”的 Mask。

推荐目录：

```text
$RUN/measurements/object_geometry/visible_geometry/keyboard/object_owned_masks/
```

逐帧文件：

```text
000000_keyboard_object_owned_mask.png
...
000149_keyboard_object_owned_mask.png
```

它近似表示：

```text
SAM2 object mask - hand-owned/hand-overlap region
```

适用于：

- 深度反投影；
- 可见物体表面构建；
- Mesh conditioning；
- 接触和遮挡所有权判断。

Anchor 阶段也保存一份：

```text
$RUN/measurements/object_geometry/anchor_candidates/keyboard/object_owned_masks/
```

---

## 13. 物体可见三维表面

### 13.1 可获取内容

每帧对象可得到：

- Mask 内深度统计；
- 可见物体表面点；
- 世界坐标三维点；
- 顶点数量；
- 相机位姿和内参来源；
- 是否适合作为刚体位姿观测；
- 初始物体中心和位姿。

### 13.2 核心目录

```text
$RUN/measurements/object_geometry/visible_geometry/keyboard/
```

主要文件：

```text
annotations_v19_visible_geometry.json
v19_visible_geometry_adapter_report.json
v19_visible_geometry_depth_fused_report.json
v19_visible_mask_report.json
anchor_candidate_proposals.json
```

`annotations_v19_visible_geometry.json` 中的主要对象字段包括：

```text
depth_m
visible_geometry_candidate
world_vertices_sample_m
reconstructed_geometry_pose
rigid_pose_observation_eligible
rigid_pose_observation_reason
```

当前有 143 帧被标为 rigid-pose eligible，7 帧因 Mask 范围或所有权问题不适合直接作为刚体位姿观测。这是冻结 keyboard run 的测量结果；其当时的 P14 没有消费该 gate。实现提交 `b7b97e6` 的 contract 为：explicit false 默认 hard reject，missing field 仅作 legacy compatibility，历史复现 override 必须显式写入 report。

生产脚本：

```text
$REPO/scripts/build_v19_visible_geometry_from_sam2_depth.py
```

### 13.3 Anchor 可见表面 PLY

```text
$RUN/measurements/object_geometry/visible_geometry/keyboard/anchor_visible_surface_mesh/keyboard/
```

包含：

```text
frame_000109_keyboard_anchor_visible_points_canonical.ply
frame_000109_keyboard_anchor_poisson_visible_mesh.ply
frame_000109_keyboard_anchor_convex_hull_visible_candidate.ply
```

这些文件表示可见深度表面，不提供可信的隐藏背面。

---

## 14. 完整物体几何和 Mesh

### 14.1 刚体证据包

```text
$RUN/measurements/geometry_completion/rigid_evidence/$CASE/keyboard/evidence_bundle/
```

主要内容：

```text
evidence_bundle_report.json
crops/
```

其中保存 anchor RGB/RGBA crop、object-owned Mask、可见深度表面和 TRELLIS conditioning image。

### 14.2 原始 TRELLIS 几何先验

```text
$RUN/measurements/geometry_completion/trellis_keyboard_seed42/
```

文件：

```text
trellis_mesh.ply
trellis_gaussian.ply
qc_trellis_shape_v3.json
```

`trellis_mesh.ply` 是生成先验，不能直接作为最终物体 Mesh。

### 14.3 适配后的 canonical 完整 Mesh

最终应使用：

```text
$RUN/measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply
```

完整目录：

```text
$RUN/measurements/geometry_completion/compact_keyboard_seed42/
```

主要内容：

```text
keyboard_observed_depth_surface_labeled.ply
keyboard_trellis_aligned_all_candidate_labeled.ply
keyboard_compact_rigid_completed_mesh_labeled.ply
observed_depth_surface_face_labels.json
trellis_candidate_face_labels.json
completed_mesh_face_labels.json
v18_compact_rigid_trellis_completion_report.json
```

当前最终 Mesh：

| 属性 | 值 |
|---|---:|
| 顶点 | 12,822 |
| 面 | 22,758 |
| watertight | 否 |
| 几何性质 | 粗略键盘刚体，不是精确 CAD |

生产脚本：

```text
$REPO/scripts/build_v18_compact_rigid_evidence_bundle.py
$REPO/scripts/remote_run_trellis_shape_v3.py
$REPO/scripts/build_v18_compact_rigid_trellis_completion.py
```

---

## 15. 物体 SE(3) 位姿轨迹

### 15.1 可获取内容

每帧可得到：

- `rotation_world_from_completed_canonical_matrix`；
- `translation_world_m`；
- visible-surface ICP 残差；
- mesh-to-observation 残差；
- 位姿来源；
- 时间图修正量；
- 插值/外推状态和不确定性。

### 15.2 可见帧位姿拟合

```text
$RUN/measurements/pose_fits/keyboard_visible_pose_fit/v18_compact_rigid_object_pose_fit_report.json
```

### 15.3 完整时间轴位姿

推荐使用：

```text
$RUN/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json
```

兼容副本：

```text
$RUN/measurements/pose_fits/keyboard_rigid_pose_graph/v18_compact_rigid_object_pose_fit_report.json
```

当前结果：

- 150 个 object pose rows；
- 全部状态为 `corrected_temporal_rigid_pose_graph`；
- observed-to-mesh median 残差约 3.84 mm；
- temporal graph 的平移和旋转修正量全部为 0。

因此，冻结 keyboard 状态名虽然包含 `corrected`，实际 150 帧均来自直接可见表面拟合，没有发生非零时间图修正。该 run 还保留当时 eligibility 未接线的行为，不能用当前代码反向改写。

实现提交 `b7b97e6` 的 P15 report 还会保存：

```text
annotation_ready
graph_support.sufficient
graph_support.configured_min_graph_frames
graph_support.actual_graph_frames
pose_observation_eligibility_policy
```

若 trusted direct rows 少于 minimum，full-timeline interpolation/nearest hold 仍可作为 renderer diagnostic rows，但所有 row 都必须 `annotation_ready=false`、`graph_support_sufficient=false`。下游 P16/P18 只能 quarantine object-relative physical factors；row count 不再等于可信轨迹 coverage。

生产脚本：

```text
$REPO/scripts/fit_v18_compact_rigid_object_pose.py
$REPO/scripts/solve_v19_rigid_object_pose_graph.py
```

---

## 16. 可见性与遮挡

### 16.1 规范状态

手和物体的规范可见性状态：

```text
visible
partially_visible
occluded
out_of_frame
unresolved
```

规范遮挡关系：

```text
hand_in_front_of_object
object_in_front_of_hand
object_partially_occluded_by_hand
no_visible_occlusion
unresolved
```

### 16.2 当前运行输出

Agent 区间判断：

```text
$RUN/state/agent_interaction_judgments/keyboard_0_149.json
```

遮挡与所有权 factor：

```text
$RUN/measurements/contact_visibility_factors/keyboard_0_149/$CASE/v19_visible_contact_ownership_factor_report.json
```

逐手侧所有权 Mask：

```text
$RUN/measurements/contact_visibility_factors/keyboard_0_149/$CASE/ownership_masks/left/
$RUN/measurements/contact_visibility_factors/keyboard_0_149/$CASE/ownership_masks/right/
```

有效帧通常包含：

```text
<frame>_projected_mano_hand_support.png
<frame>_visible_object_owned.png
<frame>_non_object_owned.png
<frame>_constraint_eligible_entity.png
```

这些分别表示：

- 投影的 MANO 手支持区域；
- 可见物体拥有区域；
- 非物体拥有区域；
- 可用于约束的对象区域。

### 16.3 当前局限

当前 V19 运行没有单独物化一个统一的：

```text
state/visibility_occlusion/full_timeline.json
```

遮挡状态目前分散在 base annotations、agent judgment、factor report 和 ownership Mask 中。因此当前可以获得遮挡证据和所有权因子，但还不是完全统一的五状态最终表。

---

## 17. 接触观测与接触候选

### 17.1 可获取内容

每只手、每个物体和每个区间可以得到：

- `likely_contact`；
- `possible_contact`；
- `no_contact`；
- `unresolved`；
- 接触先验概率；
- 接触支持不确定度；
- 2D 邻接/重叠像素；
- 接触 patch 候选；
- 接触点、法向和距离；
- 约束权重和激活状态。

### 17.2 Agent 接触先验

```text
$RUN/state/agent_interaction_judgments/keyboard_0_149.json
```

当前包含 5 个区间判断：

| 手侧 | 帧区间 | 状态 |
|---|---:|---|
| 左手 | 0–149 | `likely_contact` |
| 右手 | 0–93 | `likely_contact` |
| 右手 | 94–119 | `possible_contact` |
| 右手 | 120–144 | `likely_contact` |
| 右手 | 145–149 | `no_contact` |

这些是视觉/语义先验，不是度量接触 GT。

### 17.3 接触与所有权 factor

```text
$RUN/measurements/contact_visibility_factors/keyboard_0_149/$CASE/v19_visible_contact_ownership_factor_report.json
```

主要内容：

```text
factor_rows
ownership_rows
contact_patch_prior_probability
contact_patch_support_uncertainty_m
image_contact_counts
local_patch_sample_count
weight
agent_contact_state
agent_occlusion_relation
```

生产脚本：

```text
$REPO/scripts/build_v19_visible_contact_ownership_factor.py
```

### 17.4 MANO/contact 优化状态

原始 P18 状态：

```text
$RUN/measurements/mano_interval_correction/keyboard_0_149/$CASE/v18_joint_mano_interval_trajectory_state.json
```

当前 canonical 接触表面假设：

```text
$RUN/measurements/mano_interval_correction/keyboard_0_149_surface_hypothesis_metric_mano/$CASE/v18_joint_mano_interval_trajectory_state.json
```

其中可以读取：

```text
optimized_joints_world_m
optimized_vertices_world_sample_m
contact_surface_vertices_world_sample_m
contact_patch_vertex_ids
contact_patch_prior_probability
contact_patch_posterior_probability
contact_patch_weight
visible_surface_depth_order_*
```

当前结论是：存在接触语义先验和接触表面候选，但 metric MANO–keyboard 接触没有闭合，不能当作真实接触标签。

---

## 18. 手物距离与非穿透

### 18.1 可获取内容

每帧、每只手可以得到：

- 最近物体表面的无符号距离；
- near-surface 顶点数量和比例；
- signed-distance 查询状态；
- 穿透顶点数量；
- 穿透深度统计；
- 候选平移修正；
- 修正前后的 2D 一致性；
- 表面 Mesh 和 sign Mesh 是否 watertight。

### 18.2 输出目录

```text
$RUN/measurements/contact_nonpenetration/keyboard_mano_object_constraint/
```

核心文件：

```text
$RUN/measurements/contact_nonpenetration/keyboard_mano_object_constraint/v18_mano_object_constraint_state.json
```

共 300 行，即 150 帧×两只手。

主要字段：

```text
nearest_surface_unsigned_m
near_surface_vertex_count
penetrating_vertex_count
penetration_depth_m
signed_distance_m
candidate_translation_world_m
candidate_application_state
completed_surface_mesh_watertight
sign_mesh_watertight
```

当前统计：

- 左手每帧最近表面距离中位数的时间中位数约 0.145 m；
- 右手约 0.104 m；
- completed/sign Mesh 都不是 watertight；
- 没有有效 signed volume；
- 没有施加坐标修正。

所以“未检测到穿透”不能解释为“已证明不穿透”；当前正确状态是 `nonpenetration unresolved`。

生产脚本：

```text
$REPO/scripts/build_v18_mano_object_constraint_state.py
```

---

## 19. 统一逐帧 annotation 和最终状态

### 19.1 Base annotation

```text
$RUN/state/base_annotations/annotations_v19_base.json
```

每帧包含：

```text
frame_idx
time_s
raw_frame_path
camera
hands
objects
contact_hypotheses
frame_summary
```

它主要整合相机、MANO 和物体 Mask，还不包含最终 Mesh pose/contact。

### 19.2 带可见几何的 annotation

```text
$RUN/measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json
```

它包含：

- 相机；
- 双手 MANO；
- 物体 Mask；
- 物体可见三维表面；
- 初始物体 pose；
- 表面有效性。

这是很多后续 solver 的主要 annotation backbone。

### 19.3 Renderer-consumed state

当前最接近“一体化最终状态”的文件是：

```text
$RUN/state/render_state/keyboard_rigid_render_state.json
```

它嵌入或引用：

- 150 帧 annotation backbone；
- completed object Mesh；
- 150 个刚体 pose rows；
- 300 个 MANO/object constraint rows；
- 300 个 temporal MANO rows；
- 投影坐标合同；
- 渲染证据和不确定性。

如果下游只想读取一个统一入口，应优先读取该文件。

---

## 20. 最终渲染

### 20.1 Canonical 视频

```text
$RUN/renders/v19_overlay.mp4
$RUN/renders/v19_world.mp4
$RUN/renders/v19_side_by_side.mp4
```

当前规格：

| 视频 | 分辨率 | 帧数 | FPS | 时长 |
|---|---:|---:|---:|---:|
| Overlay | 960×1038 | 150 | 30 | 5 秒 |
| World | 1280×798 | 150 | 30 | 5 秒 |
| Side-by-side | 1920×618 | 150 | 30 | 5 秒 |

物体专用原始渲染分支：

```text
$RUN/renders/keyboard_rigid_state_runtime/$CASE/
```

包含：

```text
v19_overlay_keyboard.mp4
v19_world_keyboard.mp4
v19_side_by_side_keyboard.mp4
v19_rigid_state_render_manifest.json
overlay_frames/
world_frames/
```

Canonical 发布副本：

```text
$RUN/renders/v19_published_runtime/
```

视觉检查图片：

```text
$RUN/renders/review_frames/P19b_visual_consumption/
$RUN/renders/review_frames/P21_canonical_visual_consumption/
```

生产脚本：

```text
$REPO/scripts/build_v19_rigid_render_state.py
$REPO/scripts/render_v19_rigid_state_artifact.py
$REPO/scripts/publish_v19_render_artifact.py
```

---

## 21. QC、审计、来源和不确定性

最终结构和有限值审计：

```text
$RUN/FINAL_RUNTIME_AUDIT.json
```

Agent 物理证据与最终判断：

```text
$RUN/state/v19_agent_evidence.md
```

执行日志：

```text
$RUN/logs/harness_events.jsonl
```

运行输入合同：

```text
$RUN/input/runtime_input_contract.json
```

历史 blocker：

```text
$RUN/state/runtime_blockers/
```

当前 `state/runtime_blockers/P20.json` 是已经修复的历史发布故障记录，不表示当前完整运行仍失败。

---

## 22. 当前运行目录树

```text
$RUN/
├── input/
│   ├── runtime_input_contract.json
│   ├── raw_frame_manifest/
│   │   ├── manifest.json
│   │   └── rgb/
│   └── hawor_sequence/
│       └── hot3d_clip001851_keyboard_pinhole/
│           ├── extracted_images/
│           ├── tracks_0_150/
│           ├── cam_space/
│           └── SLAM/
│
├── measurements/
│   ├── depth_slam/
│   │   └── unidepth_full_frame/
│   ├── hand_candidates/
│   │   └── hawor_world/
│   ├── object_candidates/
│   │   └── object_box_prompts_owlv2/keyboard/
│   ├── object_tracks/
│   │   └── sam2_owlv2_box_points/keyboard/sam2/
│   ├── object_geometry/
│   │   ├── anchor_candidates/keyboard/
│   │   └── visible_geometry/keyboard/
│   ├── geometry_completion/
│   │   ├── rigid_evidence/.../keyboard/
│   │   ├── trellis_keyboard_seed42/
│   │   └── compact_keyboard_seed42/
│   ├── pose_fits/
│   │   ├── keyboard_visible_pose_fit/
│   │   └── keyboard_rigid_pose_graph/
│   ├── contact_visibility_factors/
│   │   └── keyboard_0_149/<case>/
│   ├── contact_nonpenetration/
│   │   └── keyboard_mano_object_constraint/
│   └── mano_interval_correction/
│       ├── keyboard_0_149/
│       └── keyboard_0_149_surface_hypothesis_metric_mano/
│
├── state/
│   ├── calibration/
│   ├── base_annotations/
│   ├── anchor_decisions/
│   ├── physical_branch_decisions/
│   ├── agent_interaction_judgments/
│   ├── render_state/
│   ├── runtime_blockers/
│   ├── v19_physical_state.json
│   ├── v19_uncertainty_state.json
│   └── v19_agent_evidence.md
│
├── renders/
│   ├── v19_overlay.mp4
│   ├── v19_world.mp4
│   ├── v19_side_by_side.mp4
│   ├── keyboard_rigid_state_runtime/
│   ├── v19_published_runtime/
│   └── review_frames/
│
├── logs/
│   └── harness_events.jsonl
│
├── experiments/
│   └── sam3d_vs_trellis_20260803/
│
└── FINAL_RUNTIME_AUDIT.json
```

---

## 23. 推荐的下游读取集合

如果要把结果接入其他模型，推荐使用以下最小集合：

| 信息 | 推荐文件 |
|---|---|
| 帧时间轴/RGB | `$RUN/input/raw_frame_manifest/manifest.json` |
| 度量深度 | `$RUN/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz` |
| 相机内参 | `$RUN/state/calibration/v19_camera_calibration_intrinsics.npz` |
| 相机轨迹 | `hawor_world_hands.npz` 中的 `R_c2w/t_c2w` |
| 双手原始 MANO | `$RUN/measurements/hand_candidates/hawor_world/hawor_world_hands.npz` |
| 双手 canonical 状态 | `.../keyboard_0_149_surface_hypothesis_metric_mano/$CASE/v18_joint_mano_interval_trajectory_state.json` |
| 原始物体 Mask | `$RUN/measurements/object_tracks/sam2_owlv2_box_points/keyboard/sam2/sam2_masks/` |
| 物体拥有 Mask | `$RUN/measurements/object_geometry/visible_geometry/keyboard/object_owned_masks/` |
| 可见三维表面 | `$RUN/measurements/object_geometry/visible_geometry/keyboard/annotations_v19_visible_geometry.json` |
| 完整物体 Mesh | `$RUN/measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply` |
| 物体位姿 | `$RUN/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json` |
| 遮挡/接触所有权 | `$RUN/measurements/contact_visibility_factors/keyboard_0_149/$CASE/v19_visible_contact_ownership_factor_report.json` |
| 手物距离/非穿透 | `$RUN/measurements/contact_nonpenetration/keyboard_mano_object_constraint/v18_mano_object_constraint_state.json` |
| 一体化状态 | `$RUN/state/render_state/keyboard_rigid_render_state.json` |
| 最终视频 | `$RUN/renders/v19_{overlay,world,side_by_side}.mp4` |

---

## 24. 坐标、单位和分辨率约定

### 24.1 帧与时间

- `frame_idx` 为原始视频帧号，当前范围 0～149；
- 时间通常由 `frame_idx / fps` 得到；
- 当前 FPS 为 30。

### 24.2 三维单位

- 深度 NPZ：米；
- 深度 PNG：毫米；
- MANO 顶点、关节和平移：米；
- 物体位姿平移：世界坐标米；
- 轴角和旋转向量：弧度。

### 24.3 变换语义

- `R_c2w/t_c2w`：camera-to-world；
- `T_world_camera_metric`：相机到世界的 4×4 度量变换；
- 物体 pose 通常为 `completed_canonical -> world`。

### 24.4 两套主要图像栅格

当前存在：

- 原视频、深度和 source-coordinate 内参：1408×1408；
- runtime RGB、SAM2 Mask 和 object-owned Mask：960×960。

把 source-coordinate 内参投影到 960 图像时，必须按实际尺寸缩放：

```text
scale_x = rendered_width  / source_width
scale_y = rendered_height / source_height

fx' = fx * scale_x
cx' = cx * scale_x
fy' = fy * scale_y
cy' = cy * scale_y
```

不能把 1408 坐标系的 K 直接用于 960 图像。

---

## 25. 必须避免的错误读取方式

### 25.1 不要把启动状态当最终状态

以下文件仍主要是 P00 启动时的 unresolved 状态：

```text
$RUN/state/v19_physical_state.json
$RUN/state/v19_uncertainty_state.json
```

当前真正驱动最终渲染的是：

```text
$RUN/state/render_state/keyboard_rigid_render_state.json
```

### 25.2 不要把原始 TRELLIS Mesh 当最终 Mesh

不要直接使用：

```text
$RUN/measurements/geometry_completion/trellis_keyboard_seed42/trellis_mesh.ply
```

应使用：

```text
$RUN/measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply
```

### 25.3 不要把原始 P18 状态当 canonical MANO

原始优化状态：

```text
$RUN/measurements/mano_interval_correction/keyboard_0_149/...
```

推荐 canonical 状态：

```text
$RUN/measurements/mano_interval_correction/keyboard_0_149_surface_hypothesis_metric_mano/...
```

### 25.4 不要把接触先验当 GT

Agent judgment、2D 邻接和 contact factor 都是带不确定性的测量或先验。当前 metric MANO 和物体表面仍有明显间距，不能把 `likely_contact` 直接当成真实三维接触标签。

### 25.5 不要把零穿透行当成非穿透证明

当前 Mesh 非 watertight，没有有效 signed volume。零穿透候选只说明当前查询没有生成可用修正，不代表已经证明无穿透。

---

## 26. 当前结果的可信范围

| 观测/状态 | 当前状态 |
|---|---|
| RGB/时间轴 | 完整，150/150 |
| 深度 | 完整，但为单目预测 |
| 相机内参 | 完整，但为 UniDepth 聚合估计 |
| 相机轨迹 | 完整，但来自 HaWoR/SLAM 假设 |
| 左右手 MANO | 150×2 行，含补全；视觉上仍存在偏移 |
| 物体 Mask | 150/150，当前键盘 identity 基本稳定 |
| 可见物体表面 | 可用，但依赖预测深度和 Mask |
| 完整物体 Mesh | 可渲染，但粗糙、非 watertight、不是 CAD |
| 物体刚体位姿 | 150/150；时间图实际为零修正 |
| 遮挡所有权 | 有 projected-hand/ownership Mask 和关系先验 |
| Metric 接触 | 未闭合，只能视为不确定候选 |
| Signed nonpenetration | 未闭合 |
| GT/benchmark 评估 | 当前 run 没有 `evaluation/`，不是 GT |

当前可接受的物理结论是：

> 完整时长的键盘刚体 Mesh-pose 注释，加上明确标为不确定的 metric MANO 假设。

不能声称：

- Mesh 是精确 CAD；
- temporal pose graph 进行了非零修正；
- 手和键盘的 metric contact 已闭合；
- nonpenetration 已被证明；
- 当前预测是 GT。

---

## 27. 仓库中的扩展观测能力

以下能力存在于整个仓库或历史版本中，但没有进入当前 canonical V19 v2 输出：

| 扩展观测 | 主要代码 |
|---|---|
| CoTracker 2D/3D 材质点轨迹 | `scripts/run_cotracker_object_tracks_v5.py` |
| 稀疏对应和刚体运动因子 | `scripts/build_cotracker_sparse_correspondence_edges_v5.py`、`fit_cotracker_pairwise_rigid_factors_v6.py` |
| 可变形物体逐帧表面运动 | `scripts/fit_dynamic_surface_graph_v5.py` |
| 多对象 roster/角色 | `scripts/build_v17_multi_object_timeline.py` |
| 物体材质轨迹 | `scripts/run_v17_object_material_tracks.py` |
| 物体 part Mask/part pose | `scripts/build_v18_part_*` |
| 关节物体状态 | `scripts/build_v18_articulation_fit_candidates.py` |
| 独立可见性/遮挡图 | `scripts/build_v18_visibility_occlusion_state.py`、`build_v18_occlusion_*` |
| Sliding/sticking 接触动力学 | `scripts/solve_contact_dynamics_factor_graph_v12.py` |
| 接触模式切换 | `scripts/solve_contact_mode_dynamics_factor_graph_v13.py` |
| Contact handoff | `scripts/solve_contact_handoff_factor_graph_v14.py` |
| Contact switch | `scripts/solve_contact_switch_surface_factor_graph_v15.py` |
| SAM3D/TRELLIS/Hunyuan/Mesh4D 几何候选 | `scripts/remote_run_*shape*.py`、`remote_run_mesh4d_sequence_v7.py` |

当前额外的 SAM3D/TRELLIS 对比实验位于：

```text
$RUN/experiments/sam3d_vs_trellis_20260803/
```

该目录包含替代 Mesh、z-buffer、pose replay 和 ablation，但没有晋级 canonical V19。

---

## 28. 当前不能直接提供的传感观测

当前项目没有真实输出：

- 触觉或压力阵列；
- 力和力矩；
- 物体质量、惯量；
- 关节扭矩；
- 眼动/gaze；
- IMU 原始数据；
- 全身人体姿态；
- 音频；
- 真实 RGB-D；
- 精确 CAD；
- GT 接触标签；
- GT 物体位姿。

特别注意：

```text
scripts/run_egoforce_export_v3.py
```

其中的 `EgoForce` 是手姿态相关模型分支名称，不代表项目获得了真实力传感器或六维力/力矩观测。

---

## 29. 总结

对于一个完整 V19 rigid-object 运行，最有价值的机器可读观测组合是：

```text
RGB/时间轴
+ 深度与内参
+ 相机轨迹
+ 双手 MANO
+ 物体 Mask/所有权 Mask
+ 可见三维表面
+ 完整 Mesh
+ 每帧物体 SE(3)
+ 遮挡/接触/非穿透不确定性
+ renderer-consumed state
```

其中，RGB、时间轴和文件结构最接近直接观测；深度、相机、MANO 和 Mask 是模型测量；Mesh、pose、contact 和 nonpenetration 是带来源和不确定性的推断状态。下游系统应始终保留这一层级区别。

---

## 30. Ego-Exo4D partial-GT benchmark 补充

为了避免只根据 V19 自身 report 判断准确性，另选 validation take：

```text
take_uid:      a89215f1-92fe-4207-9293-62ce108171da
take:          georgiatech_bike_07_10
camera:        aria06_214-1
target:        yellow bicycle tire lever_0
source frames: 2040–2189
```

完整 benchmark adapter、运行结果和 evaluator 位于：

- [`experiments/egoexo4d_rigid_benchmark/README.md`](../experiments/egoexo4d_rigid_benchmark/README.md)
- [`experiments/egoexo4d_rigid_benchmark/RESULTS_V19_V1_ZH.md`](../experiments/egoexo4d_rigid_benchmark/RESULTS_V19_V1_ZH.md)

### 30.1 可用真值

| GT family | Coverage | 可支持的 claim |
|---|---:|---|
| raw-view visible object Mask | 5/150 frames | P06/P07 sparse visible segmentation |
| sparse named 3D hand joints | 49/150 frames、916 filtered joints | camera-space absolute/root-relative hand error |
| rectified-view 2D hand joints | 同上 | official 512 view reprojection，不是 raw RGB pixel error |
| camera extrinsics | 150/150 frames | aligned ATE、RPE、rotation error |
| object CAD/6DoF/contact/nonpenetration | 0 | 不评测，不制造 scalar |

### 30.2 Raw 448 RGB 与 rectified 512 pose view 必须分开

Ego-Exo4D ego-pose K：

```text
f=150, cx=cy=255.5
```

属于官方 `512×512` undistorted linear camera，不属于 NAS 中的 raw distorted 448×448 MP4。官方 Ego4d 代码在 rectification 前还会把 extracted Aria RGB 旋回 90°。

固定 camera-vector adapter：

```text
X_extracted = [[0,-1,0],
               [1, 0,0],
               [0, 0,1]] @ X_annotation
```

先前将 K/UV 直接乘 `960/448` 的 v1 adapter 已被标记 invalid；旧的 `588.742 mm` hand MPJPE 和 `417.799 px` reprojection 不能报告。relation masks 原本位于 raw 1408 grid，因此 object-Mask 结果不受该修正影响。

当前 NAS 没有对应 take 的 no-image-stream VRS calibration，所以 prediction 侧不再发布 rectified K 作为 raw RGB K。

### 30.3 完整 V19 运行

运行：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/
20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_kupingxin_v1
```

launcher `exit_code=0`，但 P19/P21 最终状态是：

```text
completed with an exposed physical failure
```

独立 GT headline：

```text
object visible-Mask mean IoU:       0.6522
HaWoR absolute MPJPE:             192.956 mm
HaWoR root-relative MPJPE:         35.559 mm
P18 raw absolute improvement:       0.516 mm
camera SE3 ATE RMSE:               58.588 mm
camera 1-frame translation RPE:     7.055 mm
camera 1-frame rotation RPE:        0.345 deg
```

Object Mask 的 mean recall 为 `0.9855`，但 mean precision 仅 `0.6587`，说明目标位置基本正确但 support 偏厚。

### 30.4 新运行暴露的关键数据流

P09 只接受 6 个 rigid-pose rows，却将 27 个 rows 标成 mask extent/hand-background leakage：

```text
P09 metric rows:       33
P09 eligible rows:      6
P09 ineligible rows:   27
P14 consumed rows:     33
```

P14 eligible rows 的 observed→mesh median-of-medians从 `3.956 mm` 降到 `2.006 mm`；27 个 ineligible rows 则从 `51.783 mm` 恶化到 `63.425 mm`。这说明冻结 V19 v1 的 eligibility flag 没有进入 optimizer。

独立 `b7b97e6` mechanism ablation 已修复该 wiring，而不修改冻结 run：

```text
P14 trusted fits:                  6
P14 explicit-false rejects:       27
P14 no usable pose/sample:       117
P15 trusted direct:                6
P15 completion:                  144
  nearest hold / interpolation: 141 / 3
P15 min_graph_frames:              8
P15 annotation_ready:          false
```

`6+144=150` 只是 full-timeline hypothesis rows，不是 150 个观测。6 个 direct rows 只位于 local frame `115,116,119,120,121,123`。P16 将 physical correction candidate quarantine；P18 跳过 object-relative optimization；selected-frame P19 mechanism QC 用橙色 `POSE UNREADY` artifact 暴露 sparse hold 的可见失败。该 QC 不是新的 full-duration V19 run。修复提升的是数据流正确性和不确定性传播，不是 object 6DoF accuracy。

P18 raw 相对 HaWoR 只改善：

```text
absolute MPJPE:       0.516 mm
root-relative MPJPE:  0.325 mm
wrist error:          0.000 mm
```

P18b 与 HaWoR baseline 完全一致。该结果支持继续把 P18 raw 视为 uncertain surface hypothesis，而不是 canonical metric MANO。

### 30.5 顶层 state 注意事项

新运行结束后：

```text
state/v19_physical_state.json
state/v19_uncertainty_state.json
```

仍保留 P00 的 `initial_unresolved` 内容。实际下游状态应读取：

```text
state/render_state/tire_lever_rigid_render_state.json
state/v19_agent_evidence.md
logs/harness_events.jsonl
```

这不是把 renderer state 升格为 GT，而是指出当前 top-level state 没有被 phase graph 收尾更新。

### 30.6 Runtime footprint

```text
run size: 540 MB
file count: 1702
P00→P21: 3652 s
video duration: 5.0 s
ratio: 730.4× realtime
```

因此该 run 是完整、可审计的离线实验，但不满足项目对 V18+ default runtime 与输入时长同一数量级的要求。

### 30.7 Eligibility 修复 bundle 和 preflight

实现提交 `b7b97e6` 的隔离 runtime：

```text
/mnt/user-home/kupingxin/ego_annotation_runtime/v19_bundle_a800_b7b97e6_pose_gate_local3/
```

Manifest 记录 source revision `b7b97e617e56a48fcfcdfe327d83fe66017f2fe5`、27 个 scripts、97 个 hashed files；全部 bundle integrity/path-isolation checks 通过。

对应 preflight：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/
runtime_preflight_egoexo4d_tire_lever_pose_gate_b7b97e6_raw_v2.json
```

状态为 `ready_for_runtime_agent_launch`、`failed_checks=[]`，但预留 run root 保持未创建。没有 VRS calibration 时，同字节 raw RGB 的新运行只能是 mechanism ablation，不能称为 sensor-calibrated rerun。

### 30.8 当前输出 vs available GT 故障分类

新增 evaluator：

```text
experiments/egoexo4d_rigid_benchmark/compare_current_outputs_to_gt.py
```

输出：

```text
$BENCH/evaluation_current_output_vs_gt_pose_gate_b7b97e6/
```

其中重新评价了 sparse visible Mask、冻结/fixed completed-Mesh projection、HaWoR/P18/P18b/fixed quarantine hand joints 和 camera trajectory。Projected completed-Mesh mean IoU 为 frozen `0.1508`、fixed `0.0651`；fixed 在 GT frame 0/30/60/90 上为零 overlap，只在 frame 120 与 frozen 同为 `0.3253`。这证明 sparse nearest hold 不能作为可信轨迹，但不提供 object SE(3) GT。

人类可读结论和 bug/design/observability taxonomy：

```text
experiments/egoexo4d_rigid_benchmark/
CURRENT_OUTPUT_VS_GT_AND_FAILURE_TAXONOMY_ZH.md
```

### 30.9 下一阶段 multi-clip regression suite

完整问题清单、因果修正顺序和选片报告：

```text
experiments/egoexo4d_rigid_benchmark/
NEXT_PIPELINE_CORRECTION_AND_MULTICLIP_PLAN_ZH.md
```

类别无关 scanner 在 317 个同时有 relations、hand 和 camera annotation 的 train/val takes 中，找到 1202 个满足五张非空 1 Hz Mask、完整 camera 和 raw video contract 的 target-track/stream windows；rigidity 和 manipulation suitability 再由 raw RGB + evaluation-only Mask review 判断。

最终冻结 12 cases：6 development、1 个已消费 tire-lever reference、5 locked internal holdout；使用 12 个不同 take、participant 和 capture，development/holdout 无 participant/capture overlap。Prediction/evaluation roots：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/
  egoexo4d_v19_rigid_multiclip_v1/

/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_v19_rigid_multiclip_v1/
```

每个 prediction case 使用 opaque `exo_rigid_001 ... exo_rigid_012` ID，且仅含 `input.mp4` 和 `PREDICTION_INPUT_MANIFEST.json`；relation track、take UID/name 和 role 不发布到 prediction side。12/12 为 960×960、150 frames、30 fps，coordinate-contract self-test 和 source endpoint alignment 均通过。所有 12 cases 本地 VRS 仍缺失，且尚未启动任何 suite runtime。该 suite 是 P13/temporal observation/P15–P16/P18 修正的 guardrail，不是物理结果改善。

### 30.10 新分支 P13/P14b geometry evidence ablation

这些目录是 `research/v19-multiclip-pipeline-corrections` 上的 mechanism replay，不是冻结 V19 run，也不是 prediction/evaluation GT：

```text
$BENCH/ablations/p13_single_view_hidden_quarantine_v3/
$BENCH/ablations/p14_from_p13_hidden_quarantine_v3/
$BENCH/ablations/p14b_multiview_hidden_support_v6_depth_grid_intrinsics/
$BENCH/ablations/p15_from_p14b_multiview_hidden_support_v4_depth_grid_intrinsics/
$BENCH/ablations/p16_from_p14b_multiview_support_v4_depth_grid_intrinsics/
$BENCH/ablations/p16_signed_geometry_inactive_ready_pose_regression_v2/
$BENCH/ablations/p18_multiview_physical_surface_quarantine_v4_depth_grid_intrinsics/
$BENCH/ablations/p18_signed_inactive_oneframe_diagnostic_v4_depth_grid_intrinsics/
```

其中 P13 pose hypothesis 与冻结 completed Mesh 的 vertices/faces/colors 精确保留；P14 6 个 eligible fits 和 residual 精确保留。P14b 的 direct observations 仍只有 frames `115,116,119,120,121,123`，span `[115,123]`，15° object-canonical diagnostic viewpoint bins 只有 1，达到阈值的 exact pose-row pair 为 0。投影 UniDepth raster 时使用 NPZ row K；frame 115 的 depth K 为 `[337.2726,350.7081,487.7248,487.7248]`，而 annotation-camera K 为 `[347.2999,347.2999,487.7248,487.7248]`，两者被显式记录而不静默混用。Visible support 还要求 canonical-mesh camera-ray first hit：observed faces 只对 measured observed surface 做 visibility，generated faces 对完整 pose hypothesis 做 visibility。496 个 measured collision faces 中，495 个至少一次获得 visible-depth support、477 个重复支持、11 个出现过 support/free-space conflict，但 0 个在至少两帧出现 visible free-space contradiction。151,335 个 generated hidden faces 中，44,801 个只有 same-view repeated support、0 个具有 separated-view support、17,365 个纯 free-space contradicted、23,206 个在不同帧出现 support/free-space conflict、65,963 个其余 unsupported/self-occluded；collision surface 保持 496 faces、非 watertight，`signed_geometry_ready=false`。

P15 继续输出 6 direct + 144 unresolved completion，不能因 P14b row/face 数增加 readiness。P16 和 P18 明确记录 collision-surface path；P16 signed query/correction 为 inactive，P18 signed active-set/dense barrier 为 inactive。额外的 P16 ready-pose synthetic-top-level regression 将 P15 readiness 临时设为 true，signed factor 仍因 geometry readiness false 而保持 inactive，证明两个 gate 相互独立。P18 one-frame diagnostic 只验证 consumer wiring 和 factor inactivity，不是 full-duration runtime、性能结果或 annotation-ready hand 改善。

独立 synthetic regression：

```text
experiments/egoexo4d_rigid_benchmark/regression_geometry_evidence_contract.py
```

它不读取 NAS/GT，覆盖 single-view default/override、same-view negative、distinct-view positive、P16/P18 surface selection 和 legacy-unknown readiness。

### 30.11 Geometry-gate isolated runtime bundle

从 source revision `da87a1a5c4b6cc2ed3d2caa3c5d832742ae3ed57` 构建：

```text
/mnt/user-home/kupingxin/ego_annotation_runtime/
  v19_bundle_a800_da87a1a_geometry_gate_local4/
```

Manifest 记录 28 个 scripts、98 个 hashed files、path isolation `ok`；`filter_v19_rigid_completion_multiview_support.py` 已进入 runtime closure。WiLoR 来源是既有 `b7b97e6` curated bundle 的 non-Git tree，revision `fcb911312a38fa8badd30d9656a167485d61b8f9` 通过 parent manifest 显式继承，而不是把 tree hash 冒充 Git revision。

对应 prediction-only raw-v2 preflight：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/
  runtime_preflight_egoexo4d_tire_lever_geometry_gate_da87a1a_raw_v2.json
```

结果为 `ready_for_runtime_agent_launch`、`failed_checks=[]`、28/28 script CLI contracts 通过；input SHA-256 仍为 `37de09c1193bc5c56e23a4c9ea49caa1d38623cb4f6ea5e92dc03a78d9f29ec4`。预留 fresh run root：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/
  20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_geometry_gate_da87a1a_kupingxin_v3/
```

该 root 保持不存在；没有用相同 RGB 启动新的 full run，也没有把 staged replay 描述为 sensor-calibrated rerun。
