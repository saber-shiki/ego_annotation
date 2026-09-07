# V20 Late-Window SE(3) 改进执行路线（2026-09-07）

## 目的与当前判断

目标是修复 frame 120 以后、重点 137–149 的 generated visible surface 与 P09 visible first-hit 的前后错位，同时保持 prediction-only、真实 6DoF 运动、frame 143 的不确定性标记，以及 generated mesh 的 authority 边界。

当前判断不是“已经修复完成”：v16 是目前最好的真实可视化 diagnostic candidate，但 shape-validation 在 frame 149 仍约有 -44 mm 的 signed depth median（负值表示 generated first hit 在 observed surface 前方），late relative rotation 仍低估约 13.5°。因此后续工作必须优先改善证据与状态判定，而不是为了视频外观盲目平移 mesh。

## 已完成阶段

### 阶段 0：坐标契约与可视化可信度

- `camera_vertices_sample_m` 明确视为 camera-frame。
- Rerun world view 只执行一次 camera→world。
- 2D overlay 对 world points 只执行一次 world→camera。
- 禁止 display-only front-face pruning。
- 已通过 renderer path/SHA256 mesh contract。

### 阶段 1：prediction-only late-window pose

- window `110–149`，anchor `120`，逐帧完整 SE(3) 状态。
- frame 143 是显式 latent node；没有 P09 metric 时不作为 interpolation source，并标记 uncertain provenance。
- RGB adjacent/multihop world source→target edges；低 3D conditioning edge 降权或拒绝。
- P09 observed point factors 每个 outer pass 重建。
- translation 使用 additive 参数化，避免旧实现中的 rotation/translation coupling。
- generated mesh 不进入 `pose_residual()`。
- current candidate 与 immutable initial 同时 gate，拒绝状态不能泄漏。
- 当前较稳健候选为 pose v14。

### 阶段 2：pose-frozen shared shape

- 固定 v14 pose，只优化 shared low-dimensional anisotropic shape。
- shape v15 接受一次更新：depth median、coverage、silhouette 均相对其 initial state 改善，并通过 fixed-initial gates。
- 生成绑定候选 v16。

### 阶段 3：完整可视化

- 已完成 150/150 帧、1920×960、30 FPS、5 秒 MP4。
- 已完成对应 RRD、render JSON、代表帧和 montage。
- v16 已独立复制到 `deliverables/v20_late_window_se3_v16_visualization_archive_20260907/`，并生成带 SHA256 的 tar.gz。

## 正确的后续步骤

### 步骤 A：先修 validation 语义，不改变 pose objective

1. 对 generated first-hit validation 同时保存：
   - per-frame signed median、front fraction、front bias；
   - late-window 连续 segment 的长度、起止帧、segment median/p90；
   - 全局 common-hit 指标。
2. 不允许全局 median 掩盖 frame 137–149 的连续负 bias。
3. 将 signed-front 结果明确标为 `visible_pose/render evidence gate`，不能解释为 collision、contact、SDF、signed-volume 或 nonpenetration authority。
4. 若 per-frame/segment gate 失败，最终状态应为 `diagnostic_incomplete` / `uncertain`，不能写成 annotation-ready。
5. 该 gate 不加入 `pose_residual()`，也不用于替代 RGB/P09 旋转证据。

### 步骤 B：只用 observed/PnP/temporal evidence 做有限 pose 实验

1. 检查 RGB edge 的 accepted/rejected、cycle consistency、conditioning、multihop 覆盖和 frame 129–149 的证据密度。
2. 若存在足够的非平面 RGB evidence，再尝试一个小范围 observed-only 变体：
   - 保持 rotation 全 6DoF，不设小累计 rotation cap；
   - x/y 使用 observed feature/mask reprojection 或 P09 correspondence；
   - z 使用 P09 first-hit depth；
   - 保留 velocity/acceleration/continuity 和 immutable initial gates；
   - 不能使用 GT 或 generated mesh 生成、选 edge 或初始化。
3. 每个变体先只做 pose + post-solve validation，不立即 shape/render。
4. 用以下条件筛选：
   - RGB/P09 residual 和 cycle consistency 没有恶化；
   - frame 143 continuity 合理且 uncertain provenance 保留；
   - outside-window rows byte-level 保持；
   - generated validation 的 per-frame/segment bias 有实质改善，而不是只改善 aggregate median；
   - 没有 translation/rotation clipping。
5. 如果放宽 translation 或增强 P09 后仍无法改善（已有 v17 实验显示末帧约 -47 mm、最大 correction 约 35.2 mm），停止继续盲目调参，并将问题归类为 shape/depth mismatch 或观测不可辨识。

### 步骤 C：只有 pose candidate 通过才做 shape

1. 固定选中的 pose，不能在 shape stage 再优化逐帧 pose。
2. 每轮重建 true first-hit correspondence。
3. 同时比较 current-step 与 immutable initial：per-frame、segment、global depth，coverage，silhouette，以及 signed-front diagnostics。
4. 若 signed-front 仍形成连续 late segment，shape report 必须显示失败/不完整，而不是把 candidate 冻结为正式结果。

### 步骤 D：独立渲染与人工检查

1. exact pose/shape/mesh path + SHA256 binding。
2. 保留完整 mesh faces，不通过删除前方面来掩盖错位。
3. 检查 frames `120,130,135,140,142,143,144,149`。
4. 对 frame 143 显式显示 uncertain 标记。
5. 真实检查 MP4/RRD，而不是只看 JSON 字段或 row count。

### 步骤 E：post-freeze evaluation 与 review

1. 候选冻结后才运行 HOT3D evaluator；GT 只写入独立 evaluation report。
2. 比较 v4、base k5、v14 和最终候选的 rotation/translation tail。
3. fresh reviewer 检查：authority、provenance、坐标、gate、mesh identity、状态语义和真实渲染。
4. 修复所有 P0/P1 后，运行 direct regression、py_compile、diff check。
5. 仅 commit 当前 clean tree；未经明确授权不 push/merge。

## 明确停止条件

以下情况不能通过调参掩盖：

- late segment 的 signed-front bias 仍持续为负；
- RGB/PnP 为近似平面、conditioning 不足，无法辨识大幅旋转；
- P09 partial visible surface 与 generated shape 的深度定义无法相互验证；
- candidate 需要固定 camera-z 后移、删除 front faces 或使用 GT 才能通过 gate。

此时正确输出是：最佳 diagnostic visualization + uncertain/incomplete status + residual risk，而不是 annotation-ready 声明。

### 新增 observed-only 证据保护

- `source_3d_conditioning < min_rgb_rotation_conditioning` 的 edge 仍可保留 translation evidence，但其 rotation residual 被显式置零，并在 diagnostics 中记录原因。
- 这不是人为压平运动：它只避免近似平面 PnP 在不可辨识时冒充旋转证据；真正的大幅旋转必须由有足够 3D conditioning 的 RGB evidence 支持。

### 证据实验结果（本轮）

- 将 accepted RGB edge 的 source-canonical→target-UV correspondence 以 flattened NPZ + offsets 序列化，并实现可选 observed-only pixel residual；该 residual 默认关闭，只有显式 weight 才进入 pose objective。
- low-conditioning edge 的 pixel residual 与 PnP rotation residual 都冻结到 immutable orientation，但可继续提供 translation evidence。
- `rgb_reprojection_weight_scale=0.25` 的无 generated-gate diagnostic pose 会产生较大的约 3.3° correction 和约 44 mm translation correction；post-freeze GT 仅作诊断时 rotation median/p90改善，但 translation 恶化，且不满足 generated validation gate，因此不选用。
- `rgb_reprojection_weight_scale=0.08` 在 generated gate 下只允许很小更新，但 late per-frame signed bias 仍恶化到约 -65 mm；不选用。
- 结论：新增像素证据契约已实现并测试，但当前 v19/v16 仍是更稳健的可视化候选；不能用该实验结果为未观测旋转虚构证据。

### Shape-stage canonical-registration experiment

A shared bounded canonical mesh-registration rotation was added as an explicit optional shape-stage parameter (not a per-frame pose variable). The v28 run reduced aggregate depth median slightly but reduced coverage/IoU and left the same sustained late negative segment. It is therefore retained as an auditable optional mechanism but is not selected over the v16/v25 diagnostic render.
