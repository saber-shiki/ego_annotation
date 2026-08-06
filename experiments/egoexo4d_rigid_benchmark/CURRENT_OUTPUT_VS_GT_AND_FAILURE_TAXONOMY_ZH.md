# 当前 V19 输出 vs Ego-Exo4D 可用真值，以及故障分类

## 0. 结论先行

本报告比较两条状态：

1. **冻结 V19 v1**：完整 P00–P21 运行，source revision `b18cecd`；
2. **pose-gate fixed ablation**：提交 `b7b97e6` 从冻结 P09/P13 exact artifacts 重放 P14–P19，不重跑上游模型，也不覆盖冻结 run。

冻结 runtime input 与 corrected raw-v2 benchmark video 字节一致：

```text
SHA256 37de09c1193bc5c56e23a4c9ea49caa1d38623cb4f6ea5e92dc03a78d9f29ec4
```

结论：

- 冻结运行存在确定的软件/数据流 bug，不只是“模型效果不好”；最重要的是 P09 已拒绝的 27 个 rows 仍进入 P14/P15。
- `b7b97e6` 修复了 eligibility、低支撑 readiness 和 downstream quarantine，但没有把物理结果修好；它把原先被污染观测掩盖的 **不可观测状态** 正确暴露出来。
- 可见目标分割尚可，mean IoU `0.6522`；但 completed-Mesh+pose 投影与 sparse visible Mask GT 很差：冻结 v1 mean IoU `0.1508`，fixed sparse-hold trajectory 仅 `0.0651`。
- fixed 结果更差不表示应该恢复 27 个 rejected rows。它表示 6 个可信 pose observations 只集中在 frame `115–123`，其余 nearest hold 没有轨迹依据，因此正确状态是 `annotation_ready=false`。
- 手的主要问题是 metric translation/camera，而不是只有 articulation：absolute MPJPE `192.956 mm`，wrist `186.295 mm`，root-relative MPJPE `35.559 mm`。
- 当前 shard 无 object CAD/SE(3)/contact/nonpenetration GT；这些状态不能通过本报告制造一个“总体物理准确率”。

---

## 1. 输入、输出和真值边界

### 1.1 冻结 run

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/
20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_kupingxin_v1
```

### 1.2 Fixed ablation

```text
$BENCH/ablations/p14_eligibility_default_v2/
$BENCH/ablations/p15_from_p14_eligibility_quarantine_v3/
$BENCH/ablations/p16_from_p15_eligibility_quarantine_v3/
$BENCH/ablations/p18_unready_pose_quarantine_v3/
$BENCH/ablations/p18b_unready_pose_quarantine_v3/
```

其中：

```text
$BENCH=/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189
```

### 1.3 可用 GT

| GT family | Coverage | 可评价内容 |
|---|---:|---|
| visible object Mask | 5/150 frames | raw 960 grid 上的 visible segmentation / projected silhouette diagnostic |
| named hand joints | 49/150 annotated frames，916 filtered joints | 3D annotation-camera error、root-relative error、rectified-512 reprojection |
| camera trajectory | 150/150 | SE(3)/Sim(3) ATE、orientation gauge、gauge-invariant RPE |

明确缺失：

- object CAD / metric Mesh；
- object per-frame SE(3)；
- MANO parameters / complete surface；
- metric contact patch/force；
- signed nonpenetration GT；
- dense metric depth；
- 本 take 的 `aria06_noimagestreams.vrs` calibration。

因此下文的 projected-Mesh Mask 指标是 **geometry + pose + runtime estimated camera + visible occlusion 的联合诊断**，不是 object 6DoF GT。

---

## 2. 可见目标分割：P06/P07 相对有效，但不等于物理状态

5 个 released visible-mask frames 的冻结 SAM2 track：

| 指标 | Mean |
|---|---:|
| IoU | 0.652174 |
| precision | 0.658740 |
| recall | 0.985495 |
| bbox IoU | 0.901385 |
| centroid error | 4.751 px |
| boundary F1 | 0.936850 |

解释：

- 高 recall 表示黄色撬胎棒大体被覆盖；
- precision 和 IoU 明显较低，说明 Mask 偏厚/混入邻近 support；
- 这只能评价 visible segmentation，不能证明 completed geometry、metric depth 或 6DoF pose。

---

## 3. Completed Mesh + object pose 投影 vs sparse video Mask GT

### 3.1 评价方法

对同一个 P13 completed Mesh：

1. 分别使用冻结 P15 和 fixed P15 的每帧 `R,t`；
2. 使用 runtime annotation 中实际消费的 camera trajectory 和 estimated raw-view pinhole；
3. 将全部 151,831 faces 投影到 raw 960 grid，生成 binary silhouette；
4. 与 5 个 released visible masks 比较。

由于没有 VRS distortion calibration，且 completed Mesh 是 amodal、GT 是 visible mask，这个指标只用于发现明显投影冲突。

### 3.2 Aggregate

| State | mean IoU | mean precision | mean recall | bbox IoU | centroid error | boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| frozen v1 | 0.150834 | 0.517248 | 0.259751 | 0.238295 | 20.108 px | 0.357720 |
| pose-gate fixed | 0.065066 | 0.070616 | 0.161081 | 0.160117 | 97.786 px | 0.167729 |

两者都远低于 SAM2 visible segmentation 的 `0.6522`。这说明错误主要发生在从 visible Mask/depth 提升为 completed geometry + metric pose 的物理分支，而不是仅仅“没有找到黄色目标”。

### 3.3 Per-frame IoU

| Local frame | frozen v1 | pose-gate fixed | Fixed pose 来源 |
|---:|---:|---:|---|
| 0 | 0.084135 | 0.000000 | nearest hold from sparse cluster |
| 30 | 0.205142 | 0.000000 | nearest hold from sparse cluster |
| 60 | 0.134893 | 0.000000 | nearest hold from sparse cluster |
| 90 | 0.004670 | 0.000000 | nearest hold from sparse cluster |
| 120 | 0.325328 | 0.325328 | trusted direct interval 附近 |

Fixed 在 frame 0/30/60/90 完全不与 GT 相交，frame 120 与冻结状态相同。可视化中 fixed 橙色 Mesh 落在手/撬胎棒之外，直接证实 early-timeline nearest hold 不是可信轨迹。

这不是 eligibility fix 的回归理由：

- 冻结 v1 的部分重叠依赖 27 个已被 P09 判断为 hand/background leakage 的 pseudo-measurements；
- 恢复它们会提高某些 2D overlap，却重新引入错误 observation；
- 正确处理是增加跨时间轴的 trusted observations，而不是把污染 row 当物体 pose GT。

### 3.4 冻结与 fixed state 差异不是 accuracy metric

150 个 common frames：

| Delta | Mean | Median | P90 | Max |
|---|---:|---:|---:|---:|
| translation | 181.183 mm | 137.896 mm | 406.218 mm | 603.356 mm |
| rotation | 36.436° | 42.683° | 65.761° | 87.023° |

变化很大，但没有 object SE(3) GT，不能根据“改动较小/较大”判断谁更准确。Sparse Mask 只能说明两条轨迹都不够好，fixed 的长距离 hold 尤其不可用。

---

## 4. P14/P15：确认的软件 bug 与修复后的真实 observability

### 4.1 Eligibility 数据流

```text
P09 metric rows:                    33
P09 explicit eligible:               6
P09 explicit ineligible:            27

frozen P14 fits:                    33
fixed P14 trusted fits:              6
fixed rejected rows:                27
fixed missing initial pose rows:   117
```

Residual：

```text
frozen all-row final median:       63.038 mm
fixed eligible-only final median:   2.006 mm
```

`63.038 → 2.006 mm` 表明污染 measurements 被删除；它不代表完整轨迹精度提高，因为 only-six trusted rows 的时间 coverage 极差。

### 4.2 P15

冻结 P15：

```text
graph frames: 33
annotation_ready: true
nonpenetration target frames: 0
nfev: 1
cost: 0
correction: 全零
completion: 117 = 91 interpolation + 26 nearest hold
```

Fixed P15：

```text
trusted graph frames: 6
configured minimum: 8
trusted frame span: 115–123
annotation_ready: false
completion: 144 = 3 interpolation + 141 nearest hold
status: completed_uncertain_insufficient_trusted_pose_graph_support
```

这里可以确定：

- P09→P14 eligibility 断裂是 implementation bug；
- fixed 后不足 minimum 是数据/observability 事实，不是 crash；
- completion row 数量不能当 measurement coverage。

---

## 5. 手部 GT：Fixed quarantine 是正确回退，但没有改善手精度

| State | absolute MPJPE | root-relative MPJPE | wrist error | rectified reprojection |
|---|---:|---:|---:|---:|
| HaWoR baseline | 192.956 mm | 35.559 mm | 186.295 mm | 30.139 px |
| frozen P18 raw | 192.440 mm | 35.234 mm | 186.295 mm | 30.058 px |
| frozen P18b canonical | 192.956 mm | 35.559 mm | 186.295 mm | 30.139 px |
| fixed P18 quarantine | 192.956 mm | 35.559 mm | 186.295 mm | 30.139 px |
| fixed P18b quarantine | 192.956 mm | 35.559 mm | 186.295 mm | 30.139 px |

Fixed P18/P18b 的 300/300 rows 与 HaWoR joints 在 `1e-9 m` 内完全相同。

解释：

- frozen P18 raw 只改善 absolute `0.516 mm`、root-relative `0.325 mm`、reprojection `0.081 px`，wrist 改善 `0`；
- 相对于约 `193 mm` absolute error，这个变化没有实际意义；
- root-relative `35.6 mm` 远低于 wrist `186.3 mm`，说明 articulation 比 metric hand translation 好，但仍不是高精度；
- fixed quarantine 没有“修好手”，它只是正确拒绝用 unready object trajectory 驱动手部物理优化。

---

## 6. Camera GT：短期相对运动尚可，metric/global trajectory 不够准

| 指标 | 结果 |
|---|---:|
| SE(3) ATE RMSE | 58.588 mm |
| Sim(3) diagnostic ATE RMSE | 51.848 mm |
| Sim(3) fitted scale | 1.180235 |
| orientation-gauge mean | 7.795° |
| 1-frame translation RPE | 7.055 mm |
| 1-frame rotation RPE | 0.345° |
| 30-frame translation RPE | 130.770 mm |
| 30-frame rotation RPE | 6.591° |

这表示：

- 相邻帧 rotation 较平滑；
- 约 1 秒窗口存在明显 drift/long-tail；
- fitted Sim(3) scale `1.18` 暴露 metric scale mismatch，不能用 Sim(3) 把它掩盖成 metric success；
- camera 和 hand 的绝对 translation error 会继续污染 object world pose 与 contact。

---

## 7. Geometry/contact/nonpenetration 的机制证据

### 7.1 P13 hidden prior 主导

```text
completed faces:                 151831
observed-depth faces:               496
hidden TRELLIS-prior faces:      151335
silhouette rejected faces:            0
planar-slab rejected faces:            0
watertight: false
```

也就是说约 `99.67%` completed faces 来自 hidden prior，而两个 acceptance gate 对本例均没有排除任何 TRELLIS face。结合投影 silhouette 很差，这说明 P13 acceptance policy 没有提供有效的 physical certificate。

### 7.2 P15/P16 顺序

P15 在 P16 之前执行，因此 frozen P15：

```text
nonpenetration_target_frame_count = 0
nfev = 1
cost = 0
```

P16 随后才生成 measurement，而且 completed/sign Mesh 都非 watertight：

```text
measured pairs = 300
candidate corrections = 0
```

这是 phase-graph architecture 问题：如果希望 nonpenetration 影响 object pose，需要可信 sign geometry 后的第二个 P15 pass，或重新排列 phase；单纯调 loss weight 没有输入因子可调。

### 7.3 P18 solve/gate 不一致

冻结 P18：

- left raw translation 最大 `315.646 mm`；
- right raw translation 最大 `33.948 mm`；
- 两侧 150/150 rows 都触发 output translation gate；
- selected visible depth-order support 全为 0；
- left active set 六个 pass 后仍未闭合；
- contact rows 没有 stable object-frame anchor，active contact residual count 为 0。

这不是 metric contact solve。更合理的 formulation 是在缺少 depth-order/object anchor 时，在优化内冻结不可观测 translation，而不是先允许大幅移动再全部 gate 回去。

---

## 8. 哪些是 bug，哪些不是

### 8.1 已确认的 implementation/dataflow bugs

| ID | Bug | Severity | 状态 | 判断依据 |
|---|---|---|---|---|
| BUG-01 | P09 eligibility 没传到 P14/P15 | critical | `b7b97e6` 已修复 | 27 个 explicit-false rows 在冻结 P14 被拟合 |
| BUG-02 | 低支撑 trajectory 没有稳定的 unresolved completion/readiness contract | high | `b7b97e6` 已修复 | 6<8 现在输出 `annotation_ready=false`，而不是 crash 或被 completion 数量掩盖 |
| BUG-03 | 下游可继续消费 unready object trajectory | high | `b7b97e6` 已修复 | P16/P18/P18b/render 现在 quarantine 并拒绝 stale state 混接 |
| BUG-04 | selected render 输出 index 与 source frame 无 manifest 映射 | low | `b7b97e6` 已修复 | manifest 现在保存 source IDs 和 index mapping |
| BUG-05 | 顶层 `v19_physical_state.json`/`v19_uncertainty_state.json` 停在 P00 | medium | 未修复 | renderer 实际消费后期 state，顶层状态仍 stale |

这些属于 bug，因为同一 cached inputs 只改代码/contract 就能得到更正确的拒绝、状态和 provenance，不需要新传感器。

### 8.2 Pipeline 设计/架构问题：可修，但不是简单 coding typo

| ID | 问题 | 为什么不是“固有不可解” |
|---|---|---|
| DESIGN-01 | P15 在 P16 前，无 required feedback pass | 可以重排 phase 或增加显式第二 pass |
| DESIGN-02 | P13 hidden prior 几乎完全主导，gate 不具判别力 | 可以重构 triangle visibility/free-space/slab acceptance |
| DESIGN-03 | partial-view NN ICP residual 被过度解释为 full object pose | 可以加入 multi-frame silhouette/depth、robust ownership、observability gate |
| DESIGN-04 | P18 soft solve 后再全量 translation gate | 可以在优化变量、bounds/trust region 内冻结不可观测 gauge |
| DESIGN-05 | contact prior 没有 stable object-frame anchor | 可以构造局部 patch transport、depth order 和 persistent anchor |
| DESIGN-06 | 5 秒视频运行 3652 秒，即 `730.4×` realtime | 可以 profiling、batching、vectorization；不能用删帧伪造性能 |

这些是当前 pipeline formulation 的问题。代码按当前设计可能“正常运行”，但设计本身不足以支持最终物理 claim。

### 8.3 当前输入/GT 条件下的固有限制

| ID | 限制 | 不能靠当前代码凭空解决的原因 |
|---|---|---|
| LIMIT-01 | 无 object CAD/SE(3) GT | 无法直接评价 canonical geometry 或 6DoF accuracy |
| LIMIT-02 | 无 contact/nonpenetration/MANO-surface GT | narration、Mask overlap、非 watertight predicted Mesh 都不是 signed GT |
| LIMIT-03 | 缺 Aria VRS calibration | 无法精确 raw distorted RGB→official rectified camera |
| LIMIT-04 | 单目 scale/depth/occlusion ambiguity | 遮挡下 metric hand/object translation 不唯一，需要 calibration、multi-view 或可信 depth |
| LIMIT-05 | released GT 稀疏 | object masks 5/150、hand GT 49/150，无法 dense score 全时间轴 |

这里的“固有”是 **相对于当前输入和本地 released labels**。换成有 calibration、多视角、metric depth、CAD/pose/contact GT 的数据后，并非理论上永远不可解决。

### 8.4 暂时只能列为 suspected bug，不能先下结论

1. **P18 source-grid/factor-mask grid 路径不统一风险**：审计发现 projection 与 Mask lookup contract 分散，且 frozen P18 depth-order selected vertices 为 0；但 0 也可能来自真实 support 缺失。必须做 synthetic grid regression 和 real before/after remeasurement 才能确认。
2. **P13 两个 filter 均拒绝 0 face**：这证明本例 policy 无判别力，但尚不能单凭结果断言是实现公式错误，还是 threshold/input prior 本身不合适。

### 8.5 单独的 evaluator bug，不应混为 prediction pipeline bug

旧 v1 benchmark 曾把 released rectified-512 K/UV 当作 raw-448 K/UV。该 coordinate-contract bug 已 invalidated，旧 hand/reprojection 数字不得使用。本报告只使用 corrected v2 contract；这不表示 runtime 曾获得 released raw sensor calibration。

---

## 9. 最终判断

### Pipeline 本身是否有问题？

**有。** 至少 eligibility wiring、低支撑状态语义、downstream trust propagation 和顶层 provenance 是明确 pipeline bugs，其中前三项已在 `b7b97e6` 修复。

### 修复后 pipeline 是否已经正确完成物理标注？

**没有。** 修复后的正确结果是：

```text
object trajectory: unresolved
annotation_ready: false
P16 physical factors: quarantined
P18 optimization: skipped
metric contact/nonpenetration: unsupported
```

它从“错误地看起来完成”变成“正确地报告无法完成”。

### 当前最大的非 bug 阻塞是什么？

1. 可信 object pose rows 只有 6 帧且集中在 115–123；
2. completed geometry 是 hidden-prior-dominated non-watertight sheet；
3. camera/hand metric translation 有明显误差；
4. 缺 VRS、object pose/CAD/contact GT。

因此下一步优先级应是：

1. 修 P13 triangle-level acceptance，并用 projected sparse-mask diagnostic 做 regression；
2. 增加跨时间轴的 trusted object observations，而不是恢复 rejected rows；
3. 重构 P15↔P16 feedback pass；
4. 统一 P18 source-grid/mask-grid helper并做 targeted kill test；
5. 获取 `aria06_noimagestreams.vrs`；
6. 若要评价完整物理状态，另取得 object CAD/SE(3)、MANO surface、contact/nonpenetration GT。

---

## 10. 产物和复现

评价输出：

```text
$BENCH/evaluation_current_output_vs_gt_pose_gate_b7b97e6/
├── current_output_vs_gt_evaluation.json
├── failure_taxonomy.json
├── summary.json
├── object_mask_comparison.jpg
├── projected_mesh_vs_sparse_gt.jpg
└── projected_mesh_masks/
```

Evaluator：

```text
experiments/egoexo4d_rigid_benchmark/compare_current_outputs_to_gt.py
```

复现命令：

```bash
cd /mnt/user-home/kupingxin/ego_annotation
B=/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_kupingxin_v1
CASE=egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189

.venv/bin/python experiments/egoexo4d_rigid_benchmark/compare_current_outputs_to_gt.py \
  --run-root "$RUN" \
  --ground-truth-dir "$B/ground_truth_v2_rectified_camera" \
  --fixed-p14-report "$B/ablations/p14_eligibility_default_v2/v18_compact_rigid_object_pose_fit_report.json" \
  --fixed-p15-report "$B/ablations/p15_from_p14_eligibility_quarantine_v3/v19_rigid_object_pose_graph_report.json" \
  --fixed-p16-report "$B/ablations/p16_from_p15_eligibility_quarantine_v3/v18_mano_object_constraint_state.json" \
  --fixed-p18-state "$B/ablations/p18_unready_pose_quarantine_v3/$CASE/v18_joint_mano_interval_trajectory_state.json" \
  --fixed-p18b-state "$B/ablations/p18b_unready_pose_quarantine_v3/v18_joint_mano_interval_trajectory_state.json" \
  --output-dir "$B/evaluation_current_output_vs_gt_pose_gate_b7b97e6" \
  --object-id tire_lever \
  --replace
```

关键可视化：

- green：released visible Mask GT；
- magenta：冻结 v1 projected completed Mesh；
- orange：pose-gate fixed projected completed Mesh；
- white：GT 与 prediction intersection。

所有 quantitative conclusions 都来自上述 JSON、released GT 或 frozen/fixed runtime reports；没有为缺失的 object pose/contact/nonpenetration GT 构造替代真值。
