# HOT3D 牛奶盒案例：V19/V20 prediction-only pose repair 完整改动总结

- 案例：`P0014_84ea2dcc_carton_milk_f2370_2519`
- 代码分支：`research/milk-pose-repair-v20-complete-20260906`
- 当前实现基线：`36e87ecdad656939ced4ac0b8c5ca4198848fd2e`
- 当前完整提交：见本分支 HEAD
- 代码 worktree：`/mnt/user-home/kupingxin/ego_annotation_worktrees/milk_pose_repair_v20_keyframe_20260904`
- 代码性质：prediction-only diagnostic / visualization experiment

## 1. 设计边界

本分支的全部 V19/V20 改动都被设计为隔离实验，不替换 formal annotation pipeline。

明确不修改：

- formal P15、P16–P18、D18、D19；
- `case_result_manifest.json`、`SUITE_DONE.json`；
- formal pose/collision/contact/SDF authority；
- P13 generated SAM3D source mesh；
- MANO、P09 visible surface 和正式 RRD 的权威状态。

SAM3D/TRELLIS generated completion 在所有 V19/V20 实验中都保持：

```text
render_only_completion_hypothesis
```

generated faces 不作为 pose、collision、contact、signed distance、nonpenetration 或 formal annotation authority。

HOT3D GT 只允许用于 post-hoc evaluation、误差曲线和诊断；不进入 solver、初始化、candidate generation 或 physical authority。

## 2. P09 ownership 与输入合同修复

此前 P09 使用 depth-blind MANO subtraction，导致大量位于物体后方的 MANO first-hit 被误删。相关修复和审计包括：

- 采用 P09 depth-order ownership；
- 使用 5 mm 深度序容差；
- corrected run 使用 `front_only` masks 与 `--no-exclude-hand-bboxes`；
- 保留位于物体后方的手投影；
- anchor 改为 frame 0；
- 146 个 accepted metric frames；
- 无 metric frames：`22, 25, 31, 143`；
- 每个 metric frame 使用 2,500 个 visible first-hit points；
- owned-mask 删除率中位约 `0.077%`。

fresh UniDepth 也在 corrected root 内重跑并复核：

- source depth SHA256：`c7a6171e19c72597ed0c09114b9e7adb28d1c42296219b8a37756367d7673b78`；
- canonical P03c SHA256：`2e06a981b9614bdc61f835a8521f3d4821f997bad36eb29d183f611f219af1d9`；
- 与旧数组逐元素一致，`max_abs=0.0`。

## 3. V19 pose repair 代码

V19 实验加入了：

- 显式 `1408 -> 960 -> 256` 图像平面合同；
- strict factor contract v2；
- projected MANO triangle silhouette unknown support；
- hand-unknown observed pixels 不生成 missing-coverage factor；
- per-factor ownership/confidence/boundary-depth 权重；
- 145 条 RGB/PnP relative edges；
- observed-only global P14 diagnostic；
- P15 first-hit depth 与 silhouette factors；
- absolute RGB/PnP target-pose factor，避免错误跨 source baseline 合成运动；
- 禁用 relative scale 时真正关闭 factor，不再被 minimum weight floor 重新激活。

V19 final global+image diagnostic 指标：

- factor frames：`146`；
- depth factors：`25,726`；
- outside silhouette：`18,660`；
- missing coverage：`1,375`；
- total silhouette：`20,035`；
- total residual RMS：`0.0778080 -> 0.0742890`；
- image RMS：`0.0793362 -> 0.0735665`；
- surface median degradation：`-0.000318 m`；
- max rotation step：`9.67 deg`；
- max translation step：`8.51 mm`。

正式 P15 zero correction 被保留为结构性退化诊断：没有 image/physical target 时，zero correction 是目标函数的精确最优，不被伪装成有效优化结果。

## 4. 坐标与 P14/P15 根因审计

坐标合同已通过 round-trip audit：

```text
p_world = p_camera @ R_world_camera.T + t_world_camera
p_camera = (p_world - t_world_camera) @ R_world_camera
p_world = p_canonical @ R_world_from_canonical.T + t_world
```

审计结果：

- MANO camera->world roundtrip 最大误差：`4.21e-8 m`；
- P09 projection roundtrip 最大误差：`4.60e-5 px`；
- `V_world = V_canonical @ R.T + t` 正确；
- 未发现基础 `R/R.T`、inverse、world/camera 或重复 transform 错误。

Formal P14 根因：

- 145 条 adjacent trimmed mutual-nearest observed-surfel ICP 开放链；
- local residual median `0.650 mm`，最大 `1.394 mm`；
- edge rotation median `1.052 deg`，最大 `10.181 deg`；
- 缺少 RGB consistency 与 global loop closure；
- unary anchor ICP 只能在已有局部盆地内工作，不能恢复缺失的大范围方向分支。

Formal P15 根因：

- `image_factor_npz = null`；
- `constraint_report = null`；
- optimizer `nfev=1`、`cost=0`；
- all corrections exactly zero；
- `zero_correction_is_structural_objective_minimum=true`。

## 5. V20 keyframe/global pose graph

V20 停止了全 146-node sparse finite-difference `scipy.least_squares` 路径：

- 876 个 SE(3) correction variables；
- 约 2 万 residuals；
- 运行不可接受，已 fail-closed 停止。

替代实现：

- keyframe-only graph；
- K=5 约 30 个优化节点；
- K=10 约 16 个优化节点；
- 非 keyframe 显式左乘 SE(3) correction interpolation；
- outer pass 可刷新 3D correspondence；
- image factors strict 校验后冻结并 nonlinear relinearize；
- sparse linearized Gauss–Newton；
- rotation information eigenvalues/condition；
- covariance、point-to-plane ICP、point-to-point Umeyama cross-check；
- canonical correction magnitude 与方法一致性 fail-closed；
- surface、cumulative rotation/translation、mask IoU/centroid gates。

3D candidate 因与 image factors 冲突被 fail-closed。当前选定的 K5/K10 global/local runs 将 3D point-factor optimization scale 设为 0，3D edges 仅保留为 evidence/gating artifact。

## 6. Prediction-only global orientation recovery

新增 SuperPoint + LightGlue global orientation initializer：

- object-owned mask 内匹配；
- source-frame UniDepth metric lift；
- calibrated target-frame PnP；
- 不读取 HOT3D GT、CAD 或 generated mesh；
- 182 条 feature edges；
- 177 accepted、5 rejected；
- 150/150 帧 chain coverage。

旋转端点：

```text
HOT3D GT reference-only: 134.373370 deg
formal P14:               58.478542 deg
LightGlue chain:         120.450789 deg
global+local K5:          120.748635 deg
global+local K10:         120.750582 deg
```

GT-aligned reference-only evaluation：

```text
formal median:       13.5389 deg
K5 global+local:     10.2347 deg, p90 13.4312 deg
K10 global+local:    10.2129 deg, p90 13.0750 deg
```

当前 global/local 结果仍为 diagnostic-only，不能替换 formal P14/P15。

## 7. V20 Rerun 与视频可视化

分支新增/保留的主要 V20 构建脚本：

```text
scripts/build_v20_keyframe_edges.py
scripts/fit_v20_keyframe_observed_pose_graph.py
scripts/fit_v20_keyframe_periodic_graph.py
scripts/build_v20_global_rgb_orientation_edges.py
scripts/build_v20_diagnostic_rrd.py
scripts/build_v20_pose_delta_rrd.py
scripts/build_v20_hand_object_interaction_rrd.py
scripts/build_v20_focused_interaction_rrd.py
scripts/build_v20_depth_ordered_focused_rrd.py
scripts/build_v20_support_aware_focused_rrd.py
```

可视化分层：

- raw SAM3D generated mesh：完整 render-only mesh；
- observed-only P09 surface：prediction-side first-hit points；
- MANO：prediction-side hand geometry；
- camera/original RGB；
- depth-ordered display mesh：独立 display copy，不修改 raw mesh。

原始 SAM3D mesh：

```text
238,302 vertices
476,616 faces
```

focused/global-refined RRD 与 MP4 均使用 150 帧时间轴，显式处理无 metric frames。

## 8. Depth-order 与 translation/canonical binding 诊断

初始 camera-ray audit 发现 generated mesh first-hit 在后半段严重位于 P09 前方，例如：

- frame 84：median `+4.3 mm`，front fraction 约 `12%`；
- frame 92：median `+10.0 mm`，front fraction 约 `18%`；
- frame 140：median `-49 mm`，front fraction `100%`；
- frame 149：median `-69 mm`，front fraction `100%`。

随后完成 prediction-only translation/canonical-binding diagnostic：

```text
scripts/diagnose_v20_depth_order.py
scripts/diagnose_v20_translation_canonical_binding.py
```

测试结论：

- observed surface mesh 在当前 V20 pose 下通常仍接近 P09；
- generated completion 才出现后半段几十毫米级前穿；
- translation-only correction 在部分中间帧有帮助，但全时段不稳定；
- static generated-to-observed canonical ICP 约 `7.33 deg`、`8.02 mm`，p90 residual 约 `40.88 mm`；
- translation-only 和 static binding candidate 均 fail-closed，不写回 pose；
- 不能把问题简化为水平位移或 camera-Z 平移单因果；
- 更可能是 rotation/lateral motion 暴露了 generated completion 与 observed surface 的 canonical/visible-side binding mismatch，另叠加部分 UniDepth/pose difficult segments。

完整诊断报告位于 corrected output 外部实验目录：

```text
experiments/pose_repair_v20_keyframe_20260904/translation_canonical_binding_full/translation_canonical_binding_diagnostic.json
experiments/pose_repair_v20_keyframe_20260904/translation_canonical_binding_full/TRANSLATION_CANONICAL_BINDING_DIAGNOSTIC.md
```

## 9. Visible-support-aware display gate

最终新增：

```text
scripts/build_v20_support_aware_focused_rrd.py
```

它不是物理 mesh 修复，而是更保守的 display-only depth compositor：

1. 保留 P09 原始 first-hit rays；
2. 在 P09 object-owned mask 内增加 dense rays；
3. 新增 ray 深度只能从最近已有 P09 point 复制；
4. 不合成深度，不改 pose；
5. 仅删除 generated display face 在 P09 depth 前方超过 3 mm 的部分；
6. mask/support 之外不作任意 depth push；
7. raw mesh 和 clipped display mesh 在同一 RRD 中同时保留。

完整产物外部路径：

```text
experiments/pose_repair_v20_keyframe_20260904/support_aware_focused_v20_k5_full/
```

主要文件：

```text
v20_k5_support_aware_sam3d_mano_surface_camera.rrd
v20_k5_support_aware_vs_original_side_by_side.mp4
v20_k5_support_aware_sam3d_mano_surface_camera.json
V20_SUPPORT_AWARE_DISPLAY_REPORT.md
```

RRD 已通过 `rerun rrd verify`。该 RRD 的 raw layer 与 support-aware layer 仍然只是可视化层。

## 10. 测试与验证

已完成：

- V20 scripts `py_compile`；
- 五个 V20 semantic/sparsity tests 手动执行通过：
  - `test_left_se3_correction_roundtrip_and_interpolation`；
  - `test_canonical_registration_factor_has_zero_residual_for_true_relation`；
  - `test_rgb_absolute_factor_does_not_remove_physical_motion`；
  - `test_disabled_relative_scales_are_really_disabled`；
  - `test_image_residual_sparsity_matches_flattened_rows`；
- `pytest` 未安装，因此完整 pytest 命令不可用：
  `No module named pytest`；
- raw/depth-ordered/support-aware RRD 均通过 `rerun rrd verify`；
- 150 帧视频编码成功；
- Web Viewer localhost 与 LAN HTTP 均返回 200。

## 11. 当前限制与后续工作

当前仍明确禁止：

- 把 clipped mesh 当作完整几何；
- 用 generated completion 做 pose/collision/contact/SDF authority；
- 把 per-frame ray push 或 triangle deletion 写入 formal state；
- 用 HOT3D GT 进入 solver 或候选生成；
- 用 K5/K10 diagnostic pose 替换 formal P14/P15。

如果要继续做物理级修复，需要重新生成或验证 visible-support-aware completion geometry，或者建立通过可见表面支持、旋转、尺度、拓扑与长基线一致性 gates 的新 completion binding。当前 display clipping 只能作为可视化辅助。
