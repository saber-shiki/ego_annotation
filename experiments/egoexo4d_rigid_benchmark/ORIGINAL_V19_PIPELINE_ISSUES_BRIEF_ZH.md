# 原始 V19 Pipeline 问题简报

## 1. 范围与结论

本报告只总结冻结的原始 V19 执行，不把后续修复后的行为反写为原始结果。当前 correction branch 已开始实现其中部分修正；这不改变原始运行的审计结论。

```text
bundle: v19_bundle_a800_b18cecd_local2
run:    20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_kupingxin_v1
input:  150 frames / 30 fps / 5 s
```

Launcher 返回 `exit_code=0`，说明 P00–P21 程序执行完毕，但 runtime agent 的最终物理判断是：

```text
completed with an exposed physical failure
```

因此原始 V19 的准确定位是：

> 能产生 Mask、depth、hand、camera、geometry/pose hypotheses、uncertainty 和完整视频 render 的 measurement/proposal pipeline，但不能把本次输出视为 annotation-ready 的手物理状态。

---

## 2. 主要问题

| 类别 | 问题 | 关键证据 | 后果 |
|---|---|---|---|
| 数据流 bug | P09 eligibility 没有进入 P14/P15 | 33 个 metric rows 中只有 6 个 eligible，原始 P14 仍拟合全部 33 个 | 27 个 hand/background leakage rows 污染 object pose |
| Geometry 设计 | P13 hidden prior 主导 completed Mesh | 151,831 faces 中 151,335 来自 hidden prior，仅 496 来自 observed depth；两个 gate 均拒绝 0 faces | 得到非 watertight、局部薄片状错误 Mesh |
| Pose graph 设计 | P15 是 inert optimization | `nfev=1`、`cost=0`、所有 correction 为 0；33 direct +117 completion | full timeline 主要是传播 hypothesis，不是新物理观测 |
| Phase wiring | P15 在 P16 前执行 | P15 nonpenetration target count 为 0 | P16 诊断无法反向约束 object pose |
| Readiness bug | completion 数量掩盖 direct support 质量 | 原始 33 rows 中只有 6 个可信；可信帧集中在 115–123 | 稀疏局部支撑被误写成完整 trajectory |
| Contact/physics | contact 与 nonpenetration 没闭合 | Mesh/sign Mesh 非 watertight；P16 correction count 0；P18 contact weight 0、depth-order support 0 | 不能声称 metric contact 或 signed nonpenetration |
| P18 设计 | 先产生不可观测大平移，再由 output gate 回退 | left/right max raw correction 315.646/33.948 mm；300/300 rows 被 gate | optimizer raw state 不可晋级 canonical MANO |
| State provenance | 顶层 physical/uncertainty state 停在 P00 | renderer 实际消费后期 render state | 顶层 JSON 不能代表最终执行状态 |
| Runtime | 默认执行远慢于输入时长 | 5 s 视频运行 3652 s，约 730.4× realtime | 不满足 V18+ runtime invariant |

---

## 3. 独立 GT 对照

Ego-Exo4D 当前只支持 visible Mask、named hand joints 和 camera trajectory 的 partial-GT 评价，不提供 object CAD/SE(3)、MANO surface、contact 或 nonpenetration GT。

```text
SAM2 visible-mask mean IoU:          0.6522
原始 completed-Mesh projection IoU:  0.1508
HaWoR absolute MPJPE:              192.956 mm
HaWoR root-relative MPJPE:          35.559 mm
mean wrist error:                  186.295 mm
camera SE3 ATE RMSE:                58.588 mm
```

解释：

- 目标大体找对，但 Mask 偏厚；
- completed geometry + pose 与可见目标轮廓明显不一致；
- hand articulation 比 absolute metric translation 好；
- camera 存在 metric error 和长时 drift；
- P18 raw 对 hand GT 只改善约 `0.516 mm` absolute MPJPE，wrist error 不变，不能证明物理优化有效。

---

## 4. 哪些属于 Bug，哪些不是

### 已确认的软件/数据流 Bug

1. P09 explicit-false rows 被 P14/P15 继续消费；
2. 低支撑 completion/readiness 语义不完整；
3. unready object state 可继续进入 P16/P18/render；
4. selected render 缺少 source-frame mapping；
5. 顶层 runtime state 与 renderer-consumed state 不一致。

其中前四项已在后续 `b7b97e6` 修复或 quarantine；修复的含义是“正确报告 unresolved”，不是 object accuracy 已提升。

### Pipeline 设计问题

1. P13 缺少真正有效的 multi-view hidden-face support/free-space gate；
2. object observations 缺少跨时间和跨视角覆盖；
3. P15/P16 phase graph 没有形成 physical second pass；
4. P18 没有按 observability 限制自由变量；
5. contact prior 没有 stable object-frame metric anchor；
6. runtime 和 renderer 实现过慢。

### 当前数据条件的限制

1. 本地缺 Aria VRS calibration；
2. 单目 RGB 存在 scale/depth/occlusion/symmetry ambiguity；
3. released object Mask 通常只有 1 Hz；
4. 没有 object pose、CAD、contact 和 nonpenetration GT。

这些限制不能通过降低 threshold、增加 nearest hold 或继续堆 soft loss 消除。

---

## 5. 针对原始问题的建议顺序

以下是由原始 failure 推导的优先级；其中部分工作已在后续 correction branch 开始实施。

1. **P13 geometry**：分离 observed surface、hidden completion 和 collision/sign eligibility；无跨帧支持的 hidden faces 只保留为 uncertain hypothesis。
2. **Temporal observations**：增加跨时间、跨 viewpoint 的 direct object observations；completion rows 永不计为 direct support。
3. **P15/P16 wiring**：使用 `P15a observation pose → P16 constraints → P15b bounded refinement → independent remeasurement`。
4. **P18 observability**：统一 source/runtime/factor grids；无 depth-order/stable anchor 时在 optimizer 内冻结 global translation。
5. **验证与 runtime**：先在 development clips 做 staged/exact-state replay，再做 full-duration run；最后才运行 locked holdout 和优化性能。

禁止的捷径：

- 不恢复 27 个 rejected rows；
- 不降低 `min_graph_frames`；
- 不把 nearest hold 当新观测；
- 不把 projected Mask IoU 当 object SE(3) GT；
- 不把 non-watertight Mesh 当 signed collision surface；
- 不用更大的联合 optimizer 或 RL 掩盖 observability 问题。

---

## 6. 当前判断

原始 pipeline 不是“完全没有作用”：其 segmentation、measurement provenance、uncertainty 和失败 render 都有价值。但原始 tire-lever 输出在 object geometry、pose、hand-object consistency、contact 和 nonpenetration 上失败。

后续正确方向不是让失败结果重新变成 `annotation_ready=true`，而是：

```text
先修 geometry/observation/wiring
→ 在证据不足时明确 unresolved
→ 证据充分时再允许 bounded physical refinement
→ 用多视频 development + locked holdout 检查是否真正改善
```
