# 当前 V20 分支：可见表面对齐目标、目标函数与后半段深度误差分析

- 案例：`P0014_84ea2dcc_carton_milk_f2370_2519`
- 分支：`research/milk-pose-repair-v20-complete-20260906`
- 基线：`36e87ecdad656939ced4ac0b8c5ca4198848fd2e`
- 本文性质：对当前分支实际代码和实际 K5 global/local run 的语义审计
- formal P15/P16–P18/D18/D19：未修改

## 1. 结论先行

如果“重建表面”和 P09 可见表面是同一个物理表面，并且这个表面对齐项真正进入 pose solver，那么位姿误差和 camera-ray 深度误差应该被同时约束，不能出现后半段深度误差越来越大的现象。

但是，当前分支并没有实现这个强约束。当前 active solver 优化的是：

```text
observed-only surface / P09 first-hit / image factors
```

而后半段被发现出现深度前穿的对象是：

```text
SAM3D generated completion mesh
```

generated completion 在当前 pose solver 中明确没有被使用。因此当前现象不是 active objective 内部自相矛盾，而是：

> 当前优化目标与最终检查的 generated SAM3D ray-depth 不是同一个目标。

当前分支已经完成 observed-only pose repair 与 display-only SAM3D depth ordering，但还没有完成 generated completion 与 P09 visible surface 的联合 active surface alignment。

## 2. 当前分支中的三类表面

### 2.1 P09 visible first-hit surface

每帧的 P09 visible surface 来自：

- corrected SAM2 object mask；
- front-only MANO/object ownership；
- UniDepth metric depth；
- official camera calibration；
- camera-ray first-hit ownership。

记为：

```text
O_i
```

它是当前视角的可见表面，不是完整 closed mesh，不包含当前视角不可见的背面或被手遮挡区域。

### 2.2 Observed surface mesh

文件：

```text
collision_eligible_observed_surface.ply
```

在当前 V20 local refine 中，它实际作为：

```text
pose_mesh_observed_surface
```

它是 observed-only 的表面 proxy，不是 SAM3D generated completion。它有两个用途：

1. 构建 P15 first-hit/silhouette factors；
2. 在 V20 中计算 `surface_before`、`surface_after` 和 candidate surface gate。

### 2.3 SAM3D generated completion

文件：

```text
generated_aligned_raw_render_prior_intact.ply
```

它约有：

```text
238,302 vertices
476,616 faces
```

但在整个 V19/V20 代码中保持：

```text
render_only_completion_hypothesis
```

它没有进入：

- pose solver；
- active 3D point factor；
- collision；
- contact；
- SDF；
- nonpenetration；
- formal D19。

## 3. 当前实际运行的 K5 global/local objective

实际 K5 local refine report：

```text
experiments/pose_repair_v20_keyframe_20260904/
global_orientation_lg_full/local_refine_k5_outer2/
v20_keyframe_global_pose_report.json
```

其中 active factor 统计为：

```json
{
  "keyframe_point_factor_group_count": 0,
  "keyframe_relative_edges": 0,
  "rgb_absolute_or_relative_edges": 0,
  "image_factor_keyframe_count": 30,
  "image_depth_factor_count": 4548,
  "image_silhouette_factor_count": 4538,
  "generated_geometry_consumed": false
}
```

因此当前 K5 local refine 中：

```text
3D point-to-plane factor: 关闭
3D point-to-point factor: 关闭
keyframe relative factor: 0
RGB factor: 0
```

它从 prediction-only LightGlue global orientation chain 的 pose 开始，真正 active 的主要项是：

```text
pose prior
image first-hit depth
image silhouette
correction velocity
correction acceleration
absolute motion acceleration
anchor gauge
```

## 4. 当前 active 目标函数

当前 selected run 的目标可以写成：

```text
E_total =
    E_pose_prior
  + E_image_depth
  + E_image_silhouette
  + E_correction_velocity
  + E_correction_acceleration
  + E_motion_acceleration
  + E_anchor_gauge
```

代码位置：

```text
scripts/fit_v20_keyframe_observed_pose_graph.py
scripts/fit_v20_keyframe_periodic_graph.py
```

### 4.1 Pose correction prior

每个 keyframe 使用左乘 SE(3) correction：

```text
T_i = DeltaT_i * T_i^0
```

其中：

```text
DeltaT_i = (DeltaR_i, Deltat_i)
```

残差近似为：

```text
r_pose_i = [
    Deltat_i / sigma_pose_translation,
    log(DeltaR_i) / sigma_pose_rotation
]
```

### 4.2 Image first-hit depth residual

P15 factor builder 首先使用 observed surface mesh 做一次 first-hit rasterization：

```text
M_obs -> camera z-buffer -> canonical point map
```

随后在 P09 observed point 的像素位置取得：

```text
p_canonical
z_P09
```

当前 pose 下重新变换：

```text
p_world = p_canonical @ R_i.T + t_i
p_camera = (p_world - t_camera) @ R_camera
```

depth residual 为：

```text
r_depth =
    clip(
        p_camera.z - z_P09,
        -max_image_first_hit_residual_m,
        +max_image_first_hit_residual_m
    )
    / sigma_image_first_hit_m
```

同时乘以：

```text
sqrt(image_first_hit_weight)
* sqrt(depth_weight)
```

重要的是：这里的 canonical point 来自 observed surface mesh 的 first-hit raster，不是 generated SAM3D completion。

对应代码：

```text
fit_v20_keyframe_observed_pose_graph.py:367-391
```

### 4.3 Image silhouette residual

canonical silhouette support point 投影到当前 frame：

```text
u_pred = project(K, p_camera)
```

与冻结的 target UV 比较：

```text
r_silhouette =
    clip(u_pred - u_target, max_pixel_residual)
    / sigma_image_silhouette_px
```

这是稀疏 2D point-to-target residual，不是完整 differentiable silhouette signed-distance field。

对应代码：

```text
fit_v20_keyframe_observed_pose_graph.py:392-411
```

### 4.4 Temporal residual

包括：

```text
correction velocity
correction acceleration
absolute motion acceleration
```

这些项约束时间连续性，不是 surface alignment 本身。

### 4.5 V20 linearized Gauss–Newton

当前 K5 local refine 使用：

```text
optimizer_mode = linearized_gn
```

每次 GN iteration：

1. 对 residual 做 sparse finite-difference Jacobian；
2. 使用 robust soft-L1 权重；
3. 通过 sparse LSMR 求 step；
4. 使用 trust box 限制 per-node rotation/translation correction；
5. 只有 candidate cost 下降时才接受。

## 5. V20 中存在但当前 selected run 没有启用的 3D surface factors

代码中确实实现了 observed-only 3D point factors：

```text
build_dynamic_point_factors()
```

位置：

```text
fit_v20_keyframe_observed_pose_graph.py:205-240
```

流程为：

1. 使用当前 pose 把每帧 P09 world points 逆变换到 canonical：

```text
p_canonical_i = (p_world_i - t_i) @ R_i
```

2. 估计 kNN PCA normals；
3. anchor/target 之间建立 mutual-nearest correspondences；
4. trim correspondence；
5. 建立 point-to-plane 与 point-to-point residual。

point-to-plane residual：

```text
d_j = n_target_j dot (
    T_source(p_source_j)
    - T_target(p_target_j)
)
```

point-to-point residual：

```text
r_j =
    T_source(p_source_j)
    - T_target(p_target_j)
```

对应目标项：

```text
E_3D =
    lambda_plane * sum_j w_j * rho(d_j / sigma_plane)^2
  + lambda_point * sum_j w_j * rho(||r_j|| / sigma_point)^2
```

但当前 selected K5 global/local run 中：

```text
anchor_point_factor_scale = 0
local_point_factor_scale = 0
```

report 明确记录：

```text
"skipped": "both point-factor scales are zero"
```

所以这些 3D factors 在当前最终 pose candidate 中不是 active objective，只保留为 evidence/gating artifact。

## 6. `surface_before/after` 不是 active surface residual

`mesh_surface_metrics_from_poses()` 计算的是：

```text
observed P09 world points
    vs
pose_mesh sampled points
```

nearest-neighbor 统计：

```text
observed_to_mesh = median(
    NN(observed_world_points, transformed_pose_mesh)
)
```

以及反向：

```text
mesh_to_observed = median(
    NN(transformed_pose_mesh, observed_world_points)
)
```

在当前 selected run 中，`pose_mesh` 是：

```text
collision_eligible_observed_surface.ply
```

因此 `surface_before/after` 反映的是 observed-only mesh metric，不是 generated SAM3D first-hit metric。

当前 report：

```text
surface_median_degradation_m = -0.0016019125 m
```

这表示 observed-only surface metric 约改善 `1.60 mm`，不能推出 generated SAM3D completion 的 ray-depth 也改善。

## 7. “两个表面对齐应该同时解决位姿和深度”何时成立

假设：

```text
M_c
```

是 canonical 中的真实物体表面，

```text
O_i
```

是 frame `i` 中同一表面的 metric observation，并且对应关系有效，那么：

```text
T_i(M_c) ~= O_i
```

的 point-to-plane/point-to-point 优化确实会同时约束：

- rotation；
- translation；
- visible surface position；
- camera-ray first-hit depth；
- projected silhouette。

在 geometry、correspondence 和 depth scale 都正确时：

```text
3D surface residual -> 0
```

应当导致：

```text
camera-ray depth residual -> 0
```

因此，若 generated SAM3D surface 真正与 P09 visible surface 是同一物理表面，并且进入 active solver，后半段 depth error 不应持续增大。

## 8. 当前分支为什么不满足这个前提

### 8.1 Active objective 对齐的是 observed surface，不是 generated SAM3D

当前 active P15 factors 使用：

```text
collision_eligible_observed_surface.ply
```

depth audit 检查的却是：

```text
generated_aligned_raw_render_prior_intact.ply
```

这两个 geometry 不是同一个 mesh。

因此可以出现：

```text
M_obs 与 P09 对齐
M_gen 与 P09 不对齐
```

同一个 V20 pose 下，当前诊断观察到：

```text
frame 149:
observed surface mesh median Delta-z ~= -1.5 mm
generated SAM3D median Delta-z ~= -69 mm
```

这说明后半段主要问题不是简单的整体 pose translation，而是 generated completion 与 observed surface 的 canonical/visible-side geometry binding mismatch。

### 8.2 P09 是 partial first-hit surface

P09 只提供当前视角的 first-hit surface，不提供：

- 背面；
- 被手遮挡的部分；
- 当前视角不可见的侧面；
- 完整 closed surface。

因此 P09 可以约束当前可见 patch，但不能直接验证 generated completion 的隐藏几何。

当物体旋转时，generated mesh 的不同侧面会成为 first hit。某一侧的 completion mismatch 会转化成明显的 camera-ray depth error，即使 observed visible patch 的 pose 仍然合理。

### 8.3 UniDepth 具有 frame-dependent bias

P09 depth 来自 prediction-side UniDepth，不是严格 metric ground truth，可能存在：

- frame-dependent scale bias；
- frame-dependent offset；
- 局部深度噪声；
- 物体边缘深度不稳定；
- residual hand/object ownership error。

confidence/boundary weighting 可以降低影响，但不能自动消除所有 per-frame scale/offset bias。

### 8.4 Image factors 是 frozen correspondences

当前 P15 factor contract 是：

```text
strict v2
frozen local first-hit and silhouette correspondences
```

流程是：

1. factor build 阶段 rasterize observed surface；
2. 取得 canonical points；
3. 后续 GN 中只重新计算这些点的 pose projection；
4. 不会在每次 pose update 后重新做完整 GPU rasterization；
5. 不会动态重新选择当前 pose 下的 first-hit triangle；
6. 不会完整更新 ray-to-surface correspondence。

因此当前实际目标更接近：

```text
sum_k rho(
    z(T_i * p_k^frozen) - z_P09,k
)
```

而不是完整的：

```text
sum_u rho(
    z_render(T_i * M, u) - z_P09(u)
)
```

当物体旋转较大、可见面切换或 pose 离开局部线性区域时，两者不再等价。

### 8.5 只优化 keyframes，非 keyframes 使用插值

K5 版本大约优化 30 个 keyframe nodes。非 keyframe 使用显式 left-SE(3) correction interpolation，不逐帧独立优化 P09 depth residual。

因此即使 keyframe residual 下降，keyframe 之间仍可能出现：

- ray first-hit surface crossing；
- depth residual 增长；
- generated mesh 与 P09 前后关系变化。

## 9. 当前 active objective 的数值表现

当前 K5 local refine 的 image first-hit normalized RMS：

```text
outer 0: 0.17479 -> 0.10172
outer 1: 0.10172 -> 0.09132
```

observed-only surface metric：

```text
surface_median_degradation_m = -0.0016019125 m
```

这证明当前 solver 定义的目标在改善：

```text
P09 / observed surface / image factors
```

但没有证明：

```text
generated SAM3D first-hit vs P09
```

也就是说：

```text
active objective decreases
while generated mesh depth error increases
```

并不矛盾，因为两者不是同一个 residual。

## 10. 对“当前分段逻辑不应该让深度越来越大”的准确回答

### 10.1 如果深度误差指 active P09/observed-only image depth

原则上不应该系统性变大。

accepted candidate 必须满足：

```text
candidate_cost < before_cost
```

并且还经过：

- observed-only surface degradation gate；
- mask IoU gate；
- mask centroid gate；
- cumulative pose correction gate；
- temporal regularization。

当前 active image-depth residual 确实下降了。

### 10.2 如果深度误差指 generated SAM3D vs P09

当前分支没有保证它不变大，因为 generated SAM3D 没有进入 pose objective。

因此后半段 generated ray-depth 误差增大，在当前代码语义下是可能的，但这也明确暴露出：

> 如果目标是让完整 SAM3D reconstructed surface 与 P09 visible surface 深度一致，当前 branch 的 objective 还不完整。

## 11. 真正联合解决位姿和深度所需的目标

若要让 generated completion 也进入 visible-surface alignment，需要定义：

```text
M_gen^c
```

为 generated mesh canonical geometry，

```text
B
```

为 generated-to-observed canonical binding，

```text
T_i
```

为每帧 pose。

对每个 frame 动态 rasterize generated mesh first hit：

```text
z_gen_i(u) =
FirstHitDepth(
    T_i * B * M_gen^c,
    camera_i,
    u
)
```

然后在 P09 valid support rays/pixels 上加入：

```text
E_depth_gen =
    sum_i sum_u
    w_i(u) * rho(
        z_gen_i(u) - z_P09_i(u)
    )
```

silhouette 项：

```text
E_sil_gen =
    sum_i
    rho(
        S_render(T_i * B * M_gen^c) - S_P09_i
    )
```

可见 3D point-to-plane 项：

```text
E_3D_visible =
    sum_i sum_j
    w_ij * rho(
        n_ij dot (
            T_i * B * m_j - o_ij
        )
    )
```

完整目标可以是：

```text
E =
    lambda_depth    * E_depth_gen
  + lambda_sil      * E_sil_gen
  + lambda_3D       * E_3D_visible
  + lambda_obs      * E_obs_surface
  + lambda_temporal * E_temporal
  + lambda_binding  * E_binding_prior
```

但此设计必须额外处理两个问题。

### 11.1 Binding gauge

如果同时优化 `B` 和所有 `T_i`，会出现 binding/pose gauge ambiguity：

```text
T_i * B
```

可以由不同的 `T_i` 与 `B` 组合产生相似结果。

需要至少采用一种约束：

- 固定 frame 0 binding；
- 对 `B` 添加 rotation/translation prior；
- 先估计并冻结 `B`；
- 或只优化 per-frame pose，不优化 binding。

### 11.2 Dynamic first-hit correspondence

不能继续只使用一次性的 frozen local correspondences。每个 outer pass 至少需要：

1. 根据当前 pose rasterize generated mesh；
2. 重新取得每条 P09 ray 的 current generated first hit；
3. 根据 visibility change 更新 valid factors；
4. 更新 silhouette boundary support；
5. 重新进行 GN；
6. 对全部 frame 做最终 ray-depth validation，而不仅是 keyframes。

### 11.3 UniDepth bias model

因为 P09 depth 不是严格 ground truth，可以加入 frame-wise bias：

```text
z_corrected_P09_i(u) = a_i * z_P09_i(u) + b_i
```

并添加：

```text
E_bias =
    lambda_a * (a_i - 1)^2
  + lambda_b * b_i^2
```

或者使用局部 relative depth/normalized depth，避免把 UniDepth bias 错误解释为 object translation。

## 12. 最终判断

当前分支的准确语义是：

```text
P09 visible surface
    -> observed-only pose/image target

observed surface mesh
    -> factor source and surface evaluation mesh

generated SAM3D completion
    -> render-only mesh and post-hoc depth-order diagnostic
```

因此当前分支没有完成：

```text
joint generated-surface / P09 visible-surface pose-depth reconstruction
```

当前 depth-order clipping 只能称为：

```text
post-hoc display correction
```

不能称为：

```text
pose + depth joint reconstruction
```

如果要彻底解决后半段误差，下一版必须把 validated generated/visible completion surface 的动态 first-hit depth residual 真正加入 solver，并在每个 outer pass 刷新 correspondence，同时保留 observed-only 与 generated-completion 两套明确分层的 residual。