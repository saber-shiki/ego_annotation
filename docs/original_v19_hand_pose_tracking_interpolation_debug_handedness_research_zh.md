# 原始 V19 Ego Annotation Research 分支：手部姿态跟踪、插帧、除错与左右手校对方法调研

> 文档类型：代码与历史产物只读调研报告  
> 调研对象：`ego_annotation` 原始 V19 research 基线及其所调用的 HaWoR 手部实现  
> 基线分支：`yiwen_research` / `origin/yiwen_research`  
> 基线提交：`0c8e6a9ff1925caa5fa2665de116c404b5d39eee`  
> 文档日期：2026-08-24  
> 说明：本文梳理的是原始方法及其 research 分支中已经出现的诊断/修复机制，不把后续 SAM3D、HOT3D release/fix 分支的变化静默当成原始默认能力。

---

## 摘要

原始 V19 并没有一个单体的“手部跟踪—插帧—去噪—左右手校正”模块。手部状态实际由四层机制组合产生：

1. **HaWoR/YOLO 检测与跟踪层**：检测手框，读取 handedness 类别和 tracker ID，再把原始 tracklet 按多数 handedness 合并为 left/right 两条 side stream；
2. **HaWoR 时序 MANO 与相机层**：在连续检测片段上估计 MANO 平移、根旋转、手指姿态和 shape，同时利用 MANO 手 mask 做 masked DROID-SLAM，并用 Metric3D 为相机轨迹赋予 metric scale；
3. **双手 Transformer infiller 层**：对无检测或无 MANO 的时间段，以线性插值和旋转 Slerp 为初值，再由 120 帧时域的双手 Transformer 补全完整 MANO 参数；
4. **V19 外围 provenance、诊断与防误修层**：显式区分同帧检测支持与 infill，提供短边界补帧、2D identity continuity、左右手双 hypothesis、MANO convention replay、WiLoR/HaWoR hybrid、P18 输出平移门控和 metric-MANO/contact-surface 状态拆分。

最关键的结论是：

> 原始 V19 擅长生成完整、连续、可渲染的左右手 metric MANO 时间轴，但其默认跟踪本质上是 **side-centric**，并不是严格的物理 hand identity tracking；Transformer infiller 能补齐时间轴，却不能把长遮挡区间自动变成可靠观测；接触优化也不能自动被当作更准确的 metric 手状态。

因此，原始 research 分支后期最有效的思路并不是无条件平滑或强行闭合接触，而是：

- 保留较可信的 HaWoR metric wrist/root；
- 显式区分 observed、infilled、boundary-filled 和 hybrid 来源；
- 用 WiLoR 等可见手模型只替换 root-relative geometry；
- 在缺少直接可见表面支持时阻止接触优化篡改全局手根；
- 将 metric MANO 与不确定 contact-surface posterior 分离。

---

## 1. 调研范围与证据边界

### 1.1 原始分支定义

本文将“原始 V19 research 分支”定义为：

```text
yiwen_research
origin/yiwen_research
commit 0c8e6a9ff1925caa5fa2665de116c404b5d39eee
```

后续本地分支包括但不限于：

```text
research/v19-multiclip-pipeline-corrections
research/sam3d-p11-p15-keyboard-freeze-20260811
research/v19-metric-camera-contract-visible-geometry
release/hot3d-sam3d-trellis-dual-backend-20260817
```

它们与原始基线具有继承关系，但本文只有在说明历史演化、评价结果或机制边界时才引用，不能据此反推所有能力都是原始 runtime 的默认步骤。

### 1.2 代码来源

V19 仓库中的直接证据主要包括：

- `runtime/v19_runtime_spec.md`
- `scripts/export_hawor_world.py`
- `scripts/build_v19_base_annotations.py`
- `scripts/repair_hawor_boundary_invalid_frames.py`
- `scripts/associate_measured_hand_tracks_v3.py`
- `scripts/diagnose_mano_side_hypotheses_v3.py`
- `scripts/probe_v18_mano_left_replay_conventions.py`
- `scripts/build_v19_wilor_hawor_hybrid_hand_npz.py`
- `scripts/solve_v18_joint_mano_interval_trajectory.py`
- `scripts/build_v19_mano_surface_hypothesis_state.py`
- `docs/v19_evaluation_orchestration.md`
- `.memory/tasks/2026-06-23-pipeline-v19/OPS.md`

V19 的 P04 导出脚本并不内嵌 HaWoR 实现，而是在运行时动态导入外部 HaWoR checkout。本文还只读检查了本机现存 HaWoR checkout：

```text
/mnt/user-home/kupingxin/ego_annotation/.runtime/hawor_work/third_party/HaWoR
```

该 checkout 的 HEAD 为：

```text
66c7d4108d58a716deccd192cb7645170cdc7bd7
```

因此，涉及 HaWoR 内部 detector/tracker、chunk、infiller 和 filling preprocess 的描述，是对 **V19 导出器实际调用接口与本机现存 HaWoR 实现** 的联合还原。V19 git 基线固定了模型、权重、导出字段和调用顺序，但没有把外部 HaWoR 源码快照完整 vendor 到自身提交中。

### 1.3 能力层级

本文严格区分：

- **默认 runtime 主路径**：正式 `runtime/v19_runtime_spec.md` 会执行的步骤；
- **可选修复工具**：仓库中存在，但需额外调用；
- **research/diagnostic 工具**：用于定位错误或做假设比较，默认不进入 canonical annotation；
- **历史实验但源码缺失**：文档与运行记录存在，当前提交无法逐行复现。

---

## 2. 总体手部数据流

原始 V19 手部主链可概括为：

```text
输入视频
  │
  ├─ 30 FPS 抽帧
  │
  ├─ YOLO hand detector + Ultralytics tracker
  │      ├─ bbox
  │      ├─ detector score
  │      ├─ handedness class: 0=left, 1=right
  │      └─ tracker id
  │
  ├─ 每个 tracker tracklet 做 handedness 多数投票
  │      ├─ 所有 majority-left tracklets 合并为 left stream
  │      └─ 所有 majority-right tracklets 合并为 right stream
  │
  ├─ 连续检测片段上的 HaWoR temporal MANO inference
  │      ├─ camera-space translation
  │      ├─ root orientation
  │      ├─ 15-joint hand pose
  │      └─ 10-dim betas
  │
  ├─ MANO replay 并栅格化动态手 mask
  │
  ├─ hand-masked DROID-SLAM
  │
  ├─ Metric3D depth 对 SLAM 轨迹赋 metric scale
  │
  ├─ camera-space MANO → world-space metric MANO
  │
  ├─ 缺失时间段：
  │      linear/Slerp seed + 120-frame two-hand Transformer infiller
  │
  ├─ 完整 MANO replay
  │      ├─ 778 vertices
  │      ├─ 21 joints
  │      └─ left/right face winding
  │
  ├─ P04 导出 hawor_world_hands.npz
  │      ├─ valid
  │      ├─ detected_same_frame
  │      ├─ detector box / score
  │      ├─ raw track id
  │      └─ focal / checkpoint / cache provenance
  │
  ├─ P08 构建 V19 base annotations 和 MANO bridge
  │
  ├─ P18 手-物 interval optimization
  │      └─ 缺少可见表面支持时，输出端保留 HaWoR wrist/root
  │
  └─ P18b metric-MANO / contact-surface split
         ├─ canonical metric joints 保留 P04 source
         └─ contact optimization 只保留为不确定 surface posterior
```

核心 P04 入口：

```text
scripts/remote_run_hawor_export.sh
    └─ scripts/export_hawor_world.py
         ├─ detect_track_video()
         ├─ hawor_motion_estimation()
         ├─ hawor_slam()
         ├─ hawor_infiller()
         └─ MANO replay + NPZ/QC export
```

---

## 3. 手部检测与轨迹跟踪

## 3.1 Detector 与 tracker

HaWoR 使用 YOLO 手检测权重：

```python
hand_det_model = YOLO('./weights/external/detector.pt')
results = hand_det_model.track(
    img_cv2,
    conf=thresh,
    persist=True,
    verbose=False,
)
```

V19 的 `detect_track_video()` 调用阈值为：

```text
thresh = 0.2
```

每个 detection 读取：

```text
boxes.xyxy
boxes.conf
boxes.cls    # handedness
boxes.id     # tracker id
```

### Tracker 可复现性限制

代码没有显式传入：

```text
tracker=botsort.yaml
tracker=bytetrack.yaml
...
```

因此精确的 tracker backend 取决于当时安装的 Ultralytics 默认配置。V19 固定了 detector 权重和调用方式，却没有把 tracker yaml 写进 run contract 或导出 provenance。

这意味着跨环境重跑时，即使模型和阈值相同，也可能因为 Ultralytics 版本/默认 tracker 变化而出现不同的 track ID 和关联结果。

## 3.2 每帧只保留一个 left 与一个 right

HaWoR detector path 每帧设置：

```python
find_right = False
find_left = False
```

只接受返回结果中遇到的第一条 left-class detection 和第一条 right-class detection。代码没有在这里显式按 confidence 排序后再选最高分，因此选择依赖 YOLO result 的返回顺序。

若 tracker 没有返回 ID，则使用固定伪 ID：

```text
left  → 5000
right → 10000
```

所以所有无 tracker ID 的同侧 detection 会被归入同一伪轨迹。

## 3.3 从 tracker tracklet 到 left/right side stream

HaWoR 并不直接把每个 tracker ID 一直保留到最终 MANO state，而是对每条 tracklet 统计 handedness：

```python
if is_right.sum() / len(is_right) < 0.5:
    left_trk.extend(trk)
else:
    right_trk.extend(trk)
```

随后：

1. 所有 majority-left tracklet 合并到 `left_trk`；
2. 所有 majority-right tracklet 合并到 `right_trk`；
3. 两侧分别按 frame 排序；
4. 最终逻辑 index 固定为：
   ```text
   0 = left
   1 = right
   ```

因此最终轨迹本质上是：

```text
anatomical side stream
```

而不是：

```text
stable physical hand identity stream
```

### 容易失败的情形

- tracker ID 断裂后，同侧多个 tracklet 被直接串联；
- 某一物理手在遮挡前后 handedness 多数类别不同；
- 两手交叉或高度重叠；
- 单手被临时判成另一侧；
- side classifier 错误与 tracker ID 变化同时发生。

## 3.4 连续片段切分

HaWoR 按 frame index 连续性切分 chunk：

```python
step = frame[1:] - frame[:-1]
breaks = np.where(step != 1)[0]
```

每个连续 chunk 独立进入 HaWoR temporal MANO inference。中间缺失的时间段不直接当作检测帧，而是留给后面的 infiller。

## 3.5 BBox interpolation 的真实作用

代码中存在：

```python
interpolate_bboxes(...)
```

它会对首尾非零框之间的全零 bbox row 做线性插值。但是正常 real-video detector path 只保存实际 detection row，不会为漏检帧自动插入 zero-box placeholder。

所以在正常 V19/HaWoR 路径中：

> bbox interpolation 通常不是补 detector 掉帧的主机制；真正的时间缺口主要由 chunk split 和后续 MANO infiller 处理。

只有上游 track 数据结构本身含有 zero-box 占位时，bbox interpolation 才会直接补框。

---

## 4. 连续片段上的 MANO 姿态估计

## 4.1 时序 MANO 输出

对每个 left/right 连续 chunk，HaWoR 一次性输入该段图像和 bbox：

```python
results = model.inference(
    img_ck,
    boxes_ck,
    img_focal=img_focal,
    img_center=img_center,
    do_flip=do_flip,
)
```

主要输出为：

```text
init_trans         [1, T, 3]
init_root_orient   [1, T, 1, 3, 3]
init_hand_pose     [1, T, 15, 3, 3]
init_betas         [1, T, 10]
```

因此这不是单纯的逐帧 WiLoR/HaMeR 结果再做后处理，而是 HaWoR 自身的时序 hand-motion inference。

## 4.2 左手镜像和姿态 convention

HaWoR 主要按右手 canonical convention 推理。左手路径使用：

```text
do_flip = True
```

得到姿态后，将 axis-angle 的 y/z 分量翻转：

```python
init_root[..., 1] *= -1
init_root[..., 2] *= -1
init_hand_pose[..., 1] *= -1
init_hand_pose[..., 2] *= -1
```

随后：

- left 使用 `run_mano_left()`；
- right 使用 `run_mano()`；
- left mesh faces 反转 winding：
  ```python
  faces[:, [0, 2, 1]]
  ```

在左手 MANO replay 中还应用：

```python
mano.shapedirs[:, 0, :] *= -1
```

这是 MANO_LEFT shapedirs 镜像 convention 的已知修复。

---

## 5. 相机轨迹与 metric world 手状态

## 5.1 用 MANO 栅格化动态手 mask

HaWoR 将每个连续片段上的 MANO mesh 栅格化为手部 mask，并把左右手 mask 做并集：

```text
model_masks[t] = left_mano_mask ∪ right_mano_mask
```

这些 mask 进入后续 DROID-SLAM，用来屏蔽动态手像素。

因此相机和手之间存在如下因果链：

```text
初始 detector/hand pose
  → MANO projection mask
  → masked SLAM
  → world camera trajectory
  → world MANO state
```

若初始 handedness、bbox 或 MANO projection 明显错误，错误 mask 也可能屏蔽错误区域并影响相机估计。

## 5.2 Masked DROID-SLAM

相机轨迹使用：

```python
droid, traj = run_slam(imgfiles, masks=masks, calib=calib)
```

相机内参由输入 focal 和估计中心组成。输出包括：

```text
traj
keyframe timestamp
disparity
```

## 5.3 Metric3D scale

HaWoR 在 SLAM keyframe 上运行 Metric3D：

```text
pred_depth = Metric3D(image, calibration)
```

然后比较：

```text
SLAM depth = 1 / disparity
Metric3D depth
```

对每个 keyframe 估计 scale，最终使用 median scale：

```python
median_s = np.median(scales_)
```

## 5.4 Camera-space MANO 转 world-space MANO

连续片段结果通过 SLAM 的：

```text
R_c2w[t]
t_c2w[t]
```

变换到 world frame，最终保存：

```text
vertices_world_m
joints_world_m
trans_world_m
root_orient_axis_angle
hand_pose_axis_angle
betas
```

因此 P04 的手状态是可重放的 metric 3D MANO，不只是 2D hand boxes 或 keypoints。

---

## 6. 手部插帧与 Transformer Infiller

## 6.1 插帧对象

Infiller 补全的是完整 MANO 参数，而不只是 bbox 或 wrist 轨迹：

```text
trans       3 dimensions / hand
root rot    6D representation
hand pose   15 × 6D
betas       10 dimensions / hand
```

左右手被拼接到同一个时序表示中。

## 6.2 第一级：线性与 Slerp 初值

在 `filling_preprocess()` 中先构造合理初值：

| 参数 | 初值方法 |
|---|---|
| global translation | 逐维线性插值 |
| betas | 逐维线性插值 |
| root orientation | axis-angle 经 `scipy Rotation/Slerp` 插值 |
| 15 个 hand joint rotations | 每关节 Slerp |

中间缺失帧使用插值；边界外推采用最近有效值 hold：

```text
缺失在首个有效帧之前 → 复制首个有效旋转
缺失在末个有效帧之后 → 复制末个有效旋转
```

线性/Slerp 只用于构造 Transformer 输入 seed，不是最终补帧输出。

## 6.3 Canonicalization

每个 filling window 内，左右手各自根据窗口首时刻的：

```text
root orientation
wrist/root location
```

转换到 canonical hand frame。

这样 Transformer 学习的是相对、规范化后的双手运动，而不是任意 SLAM world gauge 下的绝对坐标。网络输出后，再从 canonical frame 变回 world frame。

## 6.4 双手联合 Transformer

Infiller 配置为：

```text
horizon = 120 frames
frame rate = 30 FPS
时域长度约 4 秒
nhead = 8
d_model = 384
hidden dim = 2048
layers = 8
dropout = 0.05
```

输入同时包含左右手，因此它可以利用：

- 另一只手是否可见；
- 双手相对运动；
- 相邻时刻 root/pose；
- 双手的共同运动模式。

这不是“左右手分别独立线性插值”，而是 learned two-hand temporal completion。

## 6.5 Masked attention

代码根据 `seq_valid` 构造：

```text
data_mask
attention mask
```

只把双手同时有有效观测的位置作为完整有效 token，缺失位置由 Transformer 预测。

短于 120 帧的 window 会复制最后一个输入 token 做 padding，并把 padding 标为 valid，最后只取原始长度的输出。

## 6.6 缺失值替换

Transformer 输出后，只替换 `~seq_valid` 对应的：

```text
trans
rot
hand_pose
betas
```

已观测参数保持原 HaWoR 结果。

## 6.7 处理顺序的细节

外层循环顺序为：

```python
for idx in [1, 0]:
```

看似先右后左，但每次 filling window 都包含双手，并且会同时替换该 window 中双手所有缺失项，随后把整个双手窗口标成 valid。

所以：

- infiller 是双手联合补全；
- 某一侧的缺失 run 可能顺带补全另一侧；
- 后处理另一侧时，前一次 learned output 已经可能被当作 valid context；
- 最终结果可能受缺失段和 window 划分顺序影响。

## 6.8 `valid` 不等于同帧观测

Infiller 完成后，代码执行：

```python
pred_valid[:, filling_net_start:filling_net_end] = 1
```

所以最终 `left_valid/right_valid` 更准确的语义是：

```text
该帧存在一个可导出的 MANO state
```

而不是：

```text
该帧有同帧 detector/图像观测支持
```

这也是 V19 必须额外导出 `detected_same_frame` 的原因。

---

## 7. P04 导出与观测来源标注

## 7.1 导出数组

`hawor_world_hands.npz` 包含：

```text
frame_idx
R_c2w
t_c2w

left/right_vertices_world_m
left/right_joints_world_m
left/right_trans_world_m
left/right_root_orient_axis_angle
left/right_hand_pose_axis_angle
left/right_betas
left/right_valid
left/right_faces

left/right_detected_same_frame
left/right_det_box_xyxyscore
left/right_track_id

img_focal
video/checkpoint/infiller/config hash
sequence/cache provenance
```

## 7.2 同帧检测支持恢复

V19 重新读取：

```text
tracks_<start>_<end>/model_tracks.npy
```

并按每个 frame/side 选择最高 confidence detection，形成：

```text
detected_same_frame
best det_box_xyxyscore
track_id
```

此步骤不会重新运行 MANO，也不会把 infilled row 变成 observed row；它只补充 provenance。

## 7.3 P08 中的基础置信度语义

`build_v19_base_annotations.py` 中：

```text
同帧 detection 支持 → confidence 0.65
无同帧 detection     → confidence 0.45
```

同时：

```text
valid HaWoR → hawor_valid_world_mano
invalid     → hawor_invalid_or_not_visible
```

这些数值是规则化层级，不是校准后的统计概率。

## 7.4 支持 side 可能与生成 stream 不完全一致

一个容易忽略的细节是：

- HaWoR MANO stream 按 **整条 tracklet 的多数 handedness** 归入 left/right；
- `export_hawor_world.py` 的同帧支持按 **该 detection 当前帧原始 handedness** 归入 left/right。

如果一条 majority-left tracklet 某一帧被 detector 短暂判成 right，则可能出现：

```text
该帧的 MANO 被写入 left stream
同帧 detector support 被记入 right
```

因此 `detected_same_frame` 是 detector-side provenance，不能严格证明它就是生成该侧 MANO chunk 的观测来源。

---

## 8. 边界帧补全

仓库提供：

```text
scripts/repair_hawor_boundary_invalid_frames.py
```

它不是 P04 默认 runtime 步骤，而是显式的短边界修复工具。

## 8.1 可修复范围

只允许：

- 所有 invalid 都在序列开头，或
- 所有 invalid 都在序列结尾；
- gap 长度不超过 `max_gap`，默认 1。

不满足条件时状态为：

```text
blocked_invalid_frames_not_fillable_boundary_gap
```

不会悄悄填内部或长缺口。

## 8.2 Camera-local hold

对最近有效源帧：

1. world vertices/joints 转到源帧 camera-local；
2. `trans_world_m` 转 camera-local；
3. root orientation 变成 camera-local orientation；
4. 在目标帧保持 camera-local hand state；
5. 用目标帧 `R_c2w/t_c2w` 重新转回 world。

可写为：

\[
X^{cam}_{src}=R_{src}^{\top}(X^{world}_{src}-t_{src})
\]

\[
X^{world}_{dst}=R_{dst}X^{cam}_{src}+t_{dst}
\]

同时：

- hand pose、betas 复制源帧；
- `detected_same_frame=0`；
- bbox 设为 NaN；
- track ID 清空；
- 写入：
  ```text
  temporal_boundary_fill_camera_local_from_frame_<src>
  ```

这比直接复制 world-space vertices 更合理，因为头戴相机在运动；但它仍然只是短时外推，不能声称为观测。

## 8.3 Padding rerun

历史上 trash clip 尾帧 invalid 曾先用 camera-local hold 修复，后来采用更干净的方法：

1. 给原视频尾部加一个 padding frame；
2. 重新运行 HaWoR；
3. 保留原始时间轴；
4. 丢弃 padding frame。

这使原最后一帧变成模型序列内部帧，而不是 synthetic boundary hold。对短边界 failure，这通常比事后复制更可信。

---

## 9. 低支持区间的时序除错研究

原始 research 历史中记录了：

```text
scripts/repair_v19_mano_support_temporal.py
```

但该脚本不在当前 `yiwen_research` tree，也无法从现存 git object 恢复。因此以下内容来自运行命令、实验报告和评价记录，不能等同于当前可逐行复现的源码分析。

## 9.1 异常区间检测信号

它仅使用 prediction-side 信号，不读取 HOT3D hand GT：

- `detected_same_frame`；
- detector score；
- bbox area ratio；
- hand-pose magnitude 的 robust z-score；
- 时间连续性；
- 小缺口 closing；
- 前后 dilation；
- 长异常 run 的额外后向 dilation；
- 合法 anchor 的最大时间距离。

文档中的代表参数：

```text
min_score                         0.45
min_area_ratio                    0.35
pose_norm_z                       3.0
pose_norm_mad_floor               0.25
fill_gap_frames                   3
pre_dilate_frames                 4
post_dilate_frames                4
long_run_min_frames               5
long_run_post_dilate_frames       22
min_interval_frames               3
max_anchor_gap_frames             55
min_raw_bad_frames_per_interval   12
```

## 9.2 Full-surface interpolation

早期版本在前后好 anchor 间插值整只手，包括：

```text
wrist/root
vertices
joints
articulation
```

结果表明：

- 能改善部分 wrist-subtracted articulation；
- 容易把非目标区间也纳入修复；
- 改动绝对 wrist/root；
- 某些 clip 上绝对 MPJPE 恶化。

因此 broad full-surface interpolation 被否决。

## 9.3 Wrist-relative interpolation

后续版本改成：

1. 保留当前帧 HaWoR wrist/root；
2. 只在 anchor 之间插值 wrist-relative joints/vertices/articulation；
3. 将插值后的局部几何重新附着到当前帧 wrist。

其核心形式为：

\[
\tilde{J}_t = w^{HaWoR}_t + \operatorname{Interp}
\left(J_a-w_a, J_b-w_b; t\right)
\]

这样避免纯 temporal prior 直接修改 metric root。

## 9.4 历史评价结果

固定三 clip aggregate：

| 指标 | baseline | wrist-relative repair |
|---|---:|---:|
| overall MPJPE | 32.86 mm | 32.57 mm |
| left MPJPE | 42.15 mm | 40.56 mm |
| left vertex-centroid | 37.43 mm | 35.73 mm |
| left wrist-subtracted median | 34.61 mm | 34.69 mm |
| wrist | 不变 | 不变 |

目标区间例子：

```text
clip001850 left frame 62–115
MPJPE:             106.2 → 94.5 mm
wrist-subtracted:   81.9 → 60.7 mm
```

结论：

> Wrist-relative temporal repair 可作为弱 articulation prior，但不能单靠时间连续性可靠恢复遮挡手。下一步需要 hand-owned mask/depth、可见表面或多视图证据。

---

## 10. 左右手判定与校对

## 10.1 默认 P04 左右手策略

默认路径由三层组成：

1. detector handedness：
   ```text
   0 = left
   1 = right
   ```
2. tracker tracklet 内做多数投票；
3. 固定 MANO convention：
   ```text
   left  → image/pose flip + run_mano_left + reversed faces
   right → run_mano
   ```

这是：

```text
detector handedness
+ tracklet-level majority
+ MANO side convention
```

而不是一个显式的全序列左右手联合优化。

## 10.2 默认路径缺少的机制

默认 P04 不具备：

- 全视频 left/right assignment 的 dynamic programming；
- side-switch penalty；
- 基于肩膀/身体拓扑的 anatomical constraint；
- 基于 3D chirality 的 hard gate；
- 两手交叉时的全局 identity reassignment；
- 使用多模型共识做 side correction。

因此默认左右手可靠性主要取决于 detector handedness 和 tracklet majority。

## 10.3 2D keypoint identity continuity

`associate_measured_hand_tracks_v3.py` 不把 side 当 identity，而是按 21 个 2D keypoints 的时间连续性关联轨迹。

关联度量包括：

```text
median keypoint displacement
p95 keypoint displacement
hand center displacement
time gap
```

默认阈值：

```text
max_gap_frames                  3
max_center_delta_px             120
max_median_keypoint_delta_px    120
max_p95_keypoint_delta_px       220
```

代价为：

\[
C = e_{median} + 0.25e_{center} + 0.02e_{p95}
\]

采用贪心最近轨迹关联。

脚本明确指出：

> side label 只保留为 metadata，因为 ego-view occlusion 下 side label 可能翻转。

但它标记为：

```text
diagnostic_only = true
annotation_ready = false
```

所以没有接入默认 P04 runtime。

## 10.4 左右 MANO 双 hypothesis

`diagnose_mano_side_hypotheses_v3.py` 对每个检测分别构造：

```text
candidate_side = left
candidate_side = right
```

每个 hypothesis 重新生成 MANO canonical geometry，并联合拟合：

- 2D keypoint reprojection；
- metric depth；
- hand bone length prior；
- rotation prior；
- scale prior。

选择分数近似为：

\[
S =
\frac{e_{reproj}}{\sigma_{reproj}}
+
\frac{|e_{depth}|}{\sigma_{depth}}
+
\frac{|L_{bone}-L_0|}{\sigma_{bone}}
+
\frac{\|\Delta R\|}{\sigma_R}
\]

同时报告：

- 3D palm chirality determinant；
- 2D palm cross-product；
- 两者符号是否一致；
- candidate 与原 stored source 的 joint/vertex gap。

### 一个重要限制

Chirality consistency 只被记录，并没有进入 `best_rows_by_score()` 的最终 selection score。因此当前实现不能被描述为“用 chirality 强制校正左右手”。

### 时间一致性限制

该脚本逐帧选择，没有 side-switch penalty 或全序列 DP，所以在证据接近时可能出现 side 抖动。

如果使用 `--output-annotations`，它会真正覆盖：

```python
hand["side"] = selected_side
```

但默认仍属于 diagnostic 工具。

## 10.5 两手唯一分配

`select_v17_hamer_hand_repair_candidates.py` 在同帧有两个检测组、每组都具备 left/right hypothesis 时，会比较：

```text
检测 A=left,  检测 B=right
检测 A=right, 检测 B=left
```

对两种双射分别汇总：

```text
median reprojection residual
p95 reprojection residual
confidence
```

选择总代价较小的分配。

这是 frame-level unique side assignment，比每只手独立选 side 更强，但仍没有跨时间全局约束。

## 10.6 MANO 左右模型 convention replay

`probe_v18_mano_left_replay_conventions.py` 校对的是实现 convention，而不是图像中手的 anatomical identity。

它测试多种组合：

```text
MANO_RIGHT.pkl + is_rhand=True
MANO_RIGHT.pkl + is_rhand=False
MANO_LEFT.pkl  + is_rhand=False
MANO_LEFT.pkl  + is_rhand=True
MANO_LEFT.pkl  + is_rhand=False + HaWoR shapedirs-x fix
```

每种 convention 都用保存的 HaWoR：

```text
root_orient
hand_pose
betas
translation
```

重放 vertices/joints，再与 source NPZ 比较。

只有 frame-median 和 p95 都达到近零阈值，才视为可用于后续 optimizer。脚本明确拒绝：

```text
plausible mirrored geometry
```

也就是“看起来像左手”但无法精确重放 source surface 的 convention。

---

## 11. WiLoR/HaWoR Hybrid 手部校正

原始 research 分支后期发现：

- HaWoR 的 metric wrist/root trajectory 相对更有价值；
- WiLoR 的 visible root-relative hand geometry 往往更贴合图像；
- WiLoR standalone camera translation 不能直接当作 V19 metric root。

因此实现：

```text
scripts/build_v19_wilor_hawor_hybrid_hand_npz.py
```

## 11.1 `wilor_metricfit`

该策略：

1. 保留 WiLoR local MANO geometry；
2. 在 V19 intrinsics 下只拟合 camera translation；
3. 使投影接近 WiLoR 自己的 2D joints；
4. reprojection 过大则回退 HaWoR。

结果：

- 严格 `4 px` median gate 下全部或几乎全部 row 被拒绝；
- 放宽后 clip001849 的 median wrist error 约 `142.94 mm`；
- joint MPJPE 约 `130.29 mm`；
- right wrist 约 `221.08 mm`。

原因是 WiLoR 的 crop/canonical focal/camera convention 不能通过 translation-only refit 直接变成 V19 metric camera state。

## 11.2 `hawor_wrist_aligned`

更有效的策略为：

1. 取 HaWoR world wrist 作为 metric root；
2. 取 WiLoR root-relative joints/vertices；
3. 经 HaWoR camera rotation 放回 world；
4. 没有合法 WiLoR visible row 时 fallback 到原始 HaWoR/infiller。

公式：

\[
J_t^{world}
=
w_t^{HaWoR,world}
+
R_{c2w,t}
\left(
J_t^{WiLoR,cam}-w_t^{WiLoR,cam}
\right)
\]

vertices 同理。

## 11.3 结果与边界

clip001849：

| 指标 | HaWoR baseline | hybrid |
|---|---:|---:|
| wrist error | 20.16 mm | 20.16 mm |
| joint MPJPE | 29.72 mm | 26.95 mm |
| vertex-centroid | 25.47 mm | 23.93 mm |
| wrist-subtracted MPJPE | 22.31 mm | 26.85 mm |

因此它支持：

```text
WiLoR visible root-relative geometry
+ HaWoR metric wrist trajectory
```

但不支持：

```text
WiLoR standalone metric translation
```

也不能声称对纯 articulation 一定改善，因为 wrist-subtracted metric 反而退化。

## 11.4 Provenance

Hybrid NPZ 写入：

```text
left/right_hybrid_source
left/right_wilor_fit_reprojection_median_px
left/right_wilor_fit_reprojection_p90_px
left/right_wilor_fitted_translation_camera_m
hybrid_policy
```

P08 会继续把这些 source tag 传播到：

```text
visibility_state
metric_mano_state.source
support_state
uncertainty
hand_geometry_source
```

从而避免 hybrid row 被错误标成 plain HaWoR。

---

## 12. P18 手部轨迹优化

P18 不是 detector 掉帧插值器，而是手-物 interaction interval optimizer。

主要变量与因素包括：

```text
每帧全局 translation
root orientation
finger articulation
latent contact state
MANO temporal smoothness
reprojection / visibility
visible-surface depth order
contact patch
nonpenetration / surface constraints
```

左右手分别调用 `build_rows(args, side)` 和 optimizer，因此 temporal smoothness 是 side-local，不存在左右手被同一个平滑项错误耦合的问题。

## 12.1 接触优化的系统性风险

历史实验发现，优化器可以通过整体移动手来减小接触 residual，而不是测量出更准确的手。

一个典型结果：

```text
contact normal gap → 约 10.19 mm
optimized translation norm median → 约 116.44 mm
wrist shift median → 约 121.23 mm
HOT3D wrist error regression → 约 +108.71 mm
joint MPJPE regression → 约 +83.18 mm
root-aligned regression → 只有约 +2.23 mm
```

说明主要错误是全局 rigid hand transform，而不是局部 articulation。

## 12.2 Output translation gate

P18 提供：

```text
--gate-translation-with-visible-surface-support
--translation-gate-min-visible-surface-depth-vertices 0
```

若某 row 的直接 visible-surface depth-order support 数不高于阈值，则输出时：

```python
gate_shift = source_hawor_wrist - optimized_wrist
state_joints += gate_shift
state_vertices += gate_shift
state_translation += gate_shift
```

即：

\[
J_t^{out}
=
J_t^{opt}
+
\left(w_t^{HaWoR}-w_t^{opt}\right)
\]

它保留：

```text
optimized wrist-relative articulation
```

同时恢复：

```text
source HaWoR wrist/root translation
```

这是一个防误修安全边界，不是正向测量。

## 12.3 In-solver translation freeze

另有实验参数：

```text
--freeze-translation-without-visible-surface-support
```

它在 optimizer 内直接冻结无支持 row 的 global translation。

clip001851 实验中：

- wrist 保持不变；
- 对比 output-gated candidate，joint MPJPE 恶化约 `+6.67 mm`；
- wrist-subtracted MPJPE 恶化约 `+6.03 mm`。

解释是：optimizer 内部 translation 仍可作为 slack，完全冻结会让 articulation/root 被迫吸收 residual。原始研究因此保留输出端 gate，否决更严格的 in-solver freeze 作为默认改进。

---

## 13. P18b：Metric MANO 与 Contact Surface 拆分

正式 runtime 使用：

```text
scripts/build_v19_mano_surface_hypothesis_state.py
```

默认原则是：

> 局部 surface-normal contact factor 只证明“可能存在一个不确定接触表面”，不能证明整只手的 metric root 和 21 joints 都应该移动。

因此 P18b：

- 将 `optimized_joints_world_m` 恢复为 P04 metric source；
- 保留 P18 contact surface samples；
- 保留 correspondence / posterior；
- 将 metric-failed candidate 作为 demoted state/provenance；
- 将 contact 标记为 unresolved/uncertain。

canonical P19/P20 默认消费 split 后的状态，而不是 raw P18 skeleton。

这一步是原始 V19 “除错”体系中最重要的结构性设计：

```text
metric hand state
≠
contact surface hypothesis
```

---

## 14. 其他关键除错机制

## 14.1 Focal cache contract

HaWoR 上游缓存路径按视频 sequence 命名，但以下结果都依赖 focal：

```text
frame_chunks_all.npy
model_masks.npy
cam_space/
SLAM/hawor_slam_w_scale_*.npz
```

V19 增加 focal cache contract：

1. 记录请求的 `img_focal`；
2. 检查旧 contract；
3. focal 不兼容时拒绝静默复用；
4. `--force-focal-cache-refresh` 时删除 focal-dependent artifacts；
5. runtime 默认：
   ```text
   EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH=1
   ```

历史实验证明，真正重新计算后 focal 从约 600 改到约 1083，会使相机系手深度从约 `0.30 m` 变到 `0.49–0.52 m`。此前“focal 修改无效”是缓存复用，不是模型对 focal 不敏感。

## 14.2 Fisheye 到 pinhole 适配

HaWoR 使用 pinhole 假设，而 HOT3D Aria 原图是 fisheye。经过 pinhole adapter 后固定三 clip aggregate：

| 指标 | raw fisheye | pinhole |
|---|---:|---:|
| wrist median | 78.93 mm | 25.81 mm |
| joint MPJPE | 71.66 mm | 32.86 mm |
| wrist-subtracted MPJPE | 29.70 mm | 22.95 mm |
| vertex-centroid | 70.04 mm | 28.53 mm |

这表明相机模型/input contract 是绝对 3D 手位置的一类系统误差，不能把所有偏差都归因于 MANO 网络。

## 14.3 Same-frame 与 infilled 分层评估

固定三 clip、900 个 hand-frame row：

```text
same-frame supported   861 / 900
infilled/unsupported    39 / 900
```

| row 类型 | joint MPJPE | wrist-subtracted MPJPE |
|---|---:|---:|
| same-frame detector-supported | 31.59 mm | 22.87 mm |
| infilled / not-same-frame | 106.07 mm | 72.76 mm |

结论：

> Infiller 可以补齐连续状态，但在遮挡/低支持区间不能被当作与观测帧同等可信，误差可能达到 10 cm 级。

## 14.4 MANO joint order 校对

早期 evaluator 直接比较 HOT3D raw 21 joints 与 HaWoR 21 joints，曾产生约 `67 mm` 的假 root-aligned MPJPE。

HaWoR/WiLoR 使用 OpenPose 风格重排：

```python
[0,13,14,15,16,
 1,2,3,17,
 4,5,6,18,
 10,11,12,19,
 7,8,9,20]
```

修正 GT joint order 后，clip001849 变为：

```text
wrist error               19.93 mm
joint MPJPE               29.68 mm
wrist-subtracted MPJPE    22.37 mm
```

这是 evaluator convention bug，不是模型突然变好。

## 14.5 Projection scaling contract

历史 review 中还出现过：

```text
source coordinates: 1408 × 1408
render image:        960 × 960
```

却直接使用 source intrinsics 投到 render image，没有乘：

```text
scale_x = 960 / 1408
scale_y = 960 / 1408
```

导致可视化明显偏移。

因此手部投影必须同时绑定：

```text
source image size
render image size
intrinsics coordinate frame
camera model
joint order
T_world_camera / T_camera_world convention
```

否则“手姿态看起来错”可能是 renderer contract 错，而不是 MANO state 错。

## 14.6 Source hash 与运行 provenance

P04 导出还记录：

```text
video SHA256
HaWoR checkpoint SHA256
infiller weight SHA256
model config SHA256
sequence folder
SLAM path
focal cache contract
```

这些信息不能证明手正确，但能定位：

- 是否跑了错误视频；
- 是否复用了旧 cache；
- 是否更换模型/权重；
- 是否 focal 不一致；
- 是否 source artifact 不同。

---

## 15. 方法能力矩阵

| 能力 | 默认 P04 | 可选/诊断 | 可靠性判断 |
|---|---:|---:|---|
| 2D hand detector | 是 | — | 同帧观测来源，但不等于姿态正确 |
| Tracker ID | 是 | — | backend 未显式固定 |
| Side stream | 是 | — | tracklet 多数 handedness，不等于稳定 identity |
| 连续片段 temporal MANO | 是 | — | 比逐帧独立估计更连续 |
| MANO-aware dynamic mask | 是 | — | 用于 masked SLAM |
| Metric camera/world hand | 是 | — | 依赖 focal、SLAM 和 Metric3D scale |
| 双手 Transformer infill | 是 | — | 补全时间轴，但不能视为同帧观测 |
| Same-frame provenance | 是 | 可后处理补充 | 是区分 observed/inferred 的关键 |
| Camera-local boundary fill | 否 | 是 | 仅适合极短首尾 gap |
| 2D identity continuity | 否 | 是 | diagnostic，side-independent |
| Left/right dual hypothesis | 否 | 是 | 逐帧 residual selection，无全局 DP |
| Chirality hard gate | 否 | 否 | 只记录，未进入 score |
| MANO convention replay | 否 | 是 | 能防止错误 left-model/mirror convention |
| WiLoR root-relative hybrid | 否 | 是 | 可改善可见 geometry，root 仍依赖 HaWoR |
| Low-support wrist-relative repair | 否 | 历史 research | 弱正向，当前源码缺失 |
| P18 hand-object optimization | 后续阶段 | — | 可产生大幅错误全局手移动 |
| P18 output translation gate | 是 | — | 防止无支持的 root shift |
| Metric/contact state split | 是 | — | canonical 状态的关键安全机制 |

---

## 16. 方法优点

### 16.1 输出是真正的 metric 3D MANO

每帧可获得：

- 778 vertices；
- 21 joints；
- MANO root orientation；
- 15-joint hand pose；
- betas；
- camera/world frame；
- faces 与可重放参数。

它明显强于只有 bbox、2D skeleton 或 image-space smoothing 的方案。

### 16.2 时序建模完整

- 连续片段由 HaWoR temporal model 估计；
- 缺失片段由双手 Transformer infiller 补全；
- rotation 使用 Slerp seed，而不是 axis-angle 逐维线性插值；
- infiller 在 canonical frame 中学习双手相对运动。

### 16.3 手和相机联合考虑

MANO mask 用于 masked SLAM，避免动态手直接污染静态场景相机估计；相机轨迹又用于把 MANO 放入 metric world。

### 16.4 Provenance 逐渐完善

后期能区分：

```text
same-frame detector-supported
HaWoR valid but inferred
boundary-filled
WiLoR-visible hybrid
HaWoR fallback
P18 raw candidate
metric-MANO preserved output
```

### 16.5 防误修机制合理

- 先分解 absolute wrist 与 wrist-relative articulation；
- 无 direct support 时恢复 source wrist；
- metric MANO 与 contact surface posterior 分离；
- 对 MANO left/right convention 要求 near-zero replay。

---

## 17. 主要局限与风险

### 17.1 跟踪是 side-centric，不是 identity-centric

所有 majority-left tracklets 被合并，所有 majority-right tracklets 被合并。它适合“每侧最多一只手”的 ego 场景，但不等于严格追踪同一物理手 identity。

### 17.2 Tracker backend 未固定

未显式指定 tracker yaml，运行环境变化会影响可复现性。

### 17.3 每帧 left/right detection 选择依赖返回顺序

代码不是显式最高 confidence 选择，且 fallback 伪 ID 会合并所有同侧无 ID detection。

### 17.4 BBox interpolation 不是普通漏检的主要补帧机制

正常漏检主要直接进入 MANO infiller，而不是由图像证据驱动的 bbox tracking/interpolation。

### 17.5 `valid=1` 可能只是 learned infill

如果下游只读取 `valid`，会把 inferred row 当作 observed row。必须结合 `detected_same_frame` 和 source tag。

### 17.6 Infiller 长遮挡误差明显

固定 HOT3D slice 中 infilled/unsupported row 的 MPJPE 约 `106 mm`，不能直接支撑 contact、nonpenetration 或毫米级手物关系。

### 17.7 默认 side 校对缺少全序列优化

双 side hypothesis 和 2D identity continuity 存在，但默认 P04 不调用；诊断选择也是逐帧或贪心，没有全序列 side-switch penalty。

### 17.8 Chirality 没进入最终 side score

当前 chirality 只能辅助人工分析，不能阻止 score 相近时选错 side。

### 17.9 同帧 bbox 重叠是过弱的质量指标

后续 HOT3D review 表明：bbox 可以覆盖正确手，但 2D joint 和 3D MANO 仍有显著错误。box IoU 不能替代 hand pose/surface quality。

### 17.10 Metric root 来源仍受相机和深度系统误差影响

- Fisheye/pinhole contract 会造成大偏差；
- focal cache 会让重跑无效；
- Metric3D scale 并非手部直接深度观测；
- UniDepth-at-hand 也不足以稳定替代 HaWoR wrist。

### 17.11 P18 接触项容易优化错变量

局部接触 residual 可能通过移动整只手被满足。没有 output gate 和 split-state 时，metric hand 会被接触目标污染。

### 17.12 部分历史 research 机制不可复现

`repair_v19_mano_support_temporal.py` 有文档和结果，但源码不在当前基线，不能作为完整可复现实装交付。

---

## 18. 对方法的最终理解

原始 V19 手部 pipeline 最准确的概括是：

> **HaWoR side-based temporal MANO + MANO-masked metric SLAM + learned two-hand infill**，外加 V19 对 same-frame support、camera/intrinsics contract、MANO convention、side hypotheses、hybrid geometry 和 hand-object optimizer 的 provenance 与安全门控。

它能构造完整、连续、可渲染的左右手 metric MANO 状态，但不能把以下概念混为一谈：

```text
valid              ≠ observed
side stream         ≠ physical identity
bbox on hand        ≠ MANO pose correct
infilled state      ≠ visible hand measurement
contact residual小  ≠ metric hand更准
2D overlap          ≠ 3D contact
plausible mirror    ≠ correct left MANO convention
```

最值得继承的设计原则是：

1. **保留真实同帧支持和所有 provenance**；
2. **绝不让 infill 与 observation 共用同一置信语义**；
3. **metric root 与 root-relative geometry 分开评估和替换**；
4. **左右手 identity 应独立于不稳定的 framewise side label**；
5. **任何 MANO convention 必须能精确 replay source surface**；
6. **接触优化必须有 direct geometric support，并与 canonical metric hand 隔离**；
7. **相机模型、内参、图像尺度和 cache contract 必须先正确，才讨论手模型误差**。

---

## 19. 推荐的代码阅读顺序

### 19.1 正式 runtime 语义

```text
runtime/v19_runtime_spec.md
```

重点：P04、P18、P18b。

### 19.2 HaWoR 导出和 provenance

```text
scripts/remote_run_hawor_export.sh
scripts/export_hawor_world.py
```

重点：focal cache、same-frame support、NPZ 字段。

### 19.3 HaWoR detector/tracker

```text
HaWoR/lib/pipeline/tools.py
HaWoR/scripts/scripts_test_video/detect_track_video.py
```

重点：每帧左右 detection、tracker ID、chunk 数据结构。

### 19.4 HaWoR temporal MANO 与 infiller

```text
HaWoR/scripts/scripts_test_video/hawor_video.py
HaWoR/lib/eval_utils/filling_utils.py
```

重点：tracklet majority、left flip、chunk、canonicalization、linear/Slerp seed、双手 Transformer。

### 19.5 Camera/metric scale

```text
HaWoR/scripts/scripts_test_video/hawor_slam.py
HaWoR/lib/eval_utils/custom_utils.py
```

### 19.6 边界补帧

```text
scripts/repair_hawor_boundary_invalid_frames.py
```

### 19.7 Identity 与左右手诊断

```text
scripts/associate_measured_hand_tracks_v3.py
scripts/diagnose_mano_side_hypotheses_v3.py
scripts/select_v17_hamer_hand_repair_candidates.py
scripts/probe_v18_mano_left_replay_conventions.py
```

### 19.8 Hybrid 手状态

```text
scripts/build_v19_wilor_hawor_hybrid_hand_npz.py
```

### 19.9 P18 防误修与状态拆分

```text
scripts/solve_v18_joint_mano_interval_trajectory.py
scripts/build_v19_mano_surface_hypothesis_state.py
```

### 19.10 历史评价与机制结论

```text
docs/v19_evaluation_orchestration.md
docs/v19_hot3d_fixed_slice_v1_results.md
.memory/project/hand_metric_mechanisms.md
.memory/tasks/2026-06-23-pipeline-v19/OPS.md
```

---

## 20. 关键源文件索引

| 主题 | 文件 |
|---|---|
| V19 正式 P04/P18/P18b | `runtime/v19_runtime_spec.md` |
| HaWoR 远程入口 | `scripts/remote_run_hawor_export.sh` |
| HaWoR world MANO 导出 | `scripts/export_hawor_world.py` |
| 后补同帧 track support | `scripts/augment_hawor_track_support.py` |
| 边界 invalid 修复 | `scripts/repair_hawor_boundary_invalid_frames.py` |
| V19 base annotation/bridge | `scripts/build_v19_base_annotations.py` |
| 2D identity continuity | `scripts/associate_measured_hand_tracks_v3.py` |
| 左右 MANO 双 hypothesis | `scripts/diagnose_mano_side_hypotheses_v3.py` |
| HaMeR 两手唯一 side 分配 | `scripts/select_v17_hamer_hand_repair_candidates.py` |
| 左 MANO convention replay | `scripts/probe_v18_mano_left_replay_conventions.py` |
| WiLoR/HaWoR hybrid | `scripts/build_v19_wilor_hawor_hybrid_hand_npz.py` |
| P18 trajectory optimizer | `scripts/solve_v18_joint_mano_interval_trajectory.py` |
| Metric/contact state split | `scripts/build_v19_mano_surface_hypothesis_state.py` |
| HOT3D MANO 评价 | `scripts/evaluate_v19_hot3d_hawor_mano3d.py` |
| 历史修复命令 | `docs/v19_evaluation_orchestration.md` |
| 固定 slice 结果 | `docs/v19_hot3d_fixed_slice_v1_results.md` |

---

## 21. 调研结论摘要表

| 问题 | 原始方法 | 结论 |
|---|---|---|
| 手如何跟踪？ | YOLO detector + Ultralytics tracker，tracklet 多数 handedness 后合并为左右 side stream | 有时序检测关联，但不是严格物理 hand identity tracking |
| 手姿态如何估计？ | 连续 bbox chunk 上运行 HaWoR temporal MANO | 输出完整 metric MANO 参数、关节和表面 |
| 缺帧如何插值？ | linear/Slerp seed + 120-frame 双手 Transformer infiller | 不是简单线性插值；能补全但低支持区误差大 |
| 首尾一帧坏了怎么办？ | camera-local hold，或 padding 后重跑并裁剪 | padding rerun 更可信；hold 必须标为 inferred |
| 如何知道是检测还是补出来的？ | `detected_same_frame`、bbox、track ID、state source | 必须与 `valid` 分开读取 |
| 左右手怎么定？ | detector class + tracklet majority + MANO side convention | 默认无全序列 side 校对，遮挡/交叉时有风险 |
| 是否有左右手校对工具？ | 2D identity continuity、双 side MANO fit、两手唯一分配、MANO replay probe | 多为 diagnostic，未统一进入 P04 默认路径 |
| 如何去除异常姿态？ | provenance 分层、历史 low-support interval detection、wrist-relative interpolation、WiLoR hybrid | 只靠时序修复是弱正向，不能恢复长遮挡真值 |
| 接触优化能修手吗？ | P18 联合 translation/root/articulation/contact/temporal factors | 容易错误移动整只手，不能自动晋升为 metric MANO |
| 如何防止接触误修？ | visible-support output gate + P18b metric/contact split | 是原始 V19 最重要的安全机制之一 |
| 最大工程坑是什么？ | tracker 未固定、focal cache、fisheye/pinhole、joint order、render scale | 这些 contract bug 可伪装成手模型错误 |

---

## 22. 复现注意事项

若要重新复现原始方法，至少应固定并记录：

1. V19 commit：`0c8e6a9...`；
2. 外部 HaWoR commit；
3. Ultralytics 版本和显式 tracker yaml；
4. detector checkpoint hash；
5. HaWoR checkpoint、infiller 和 model config hash；
6. MANO_LEFT/MANO_RIGHT 资产 hash 与 shapedirs convention；
7. 输入视频 hash、解码 FPS 与 frame count；
8. source camera model、intrinsics 和畸变处理；
9. focal cache 是否彻底刷新；
10. `valid`、`detected_same_frame`、`boundary_fill`、`hybrid_source` 的分别统计；
11. source/render image size 与 projection `scale_xy`；
12. MANO 21-joint ordering；
13. P18 raw state、translation-gated state和 P18b split state分别保存；
14. 左右手 overlay、world view 和时间轴 review。

如果缺少这些记录，即便输出文件名相同，也不能保证跟原始 V19 手部结果具有同一语义。
