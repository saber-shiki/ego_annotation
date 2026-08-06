# V19 下一阶段：问题清单、修正顺序与 Ego-Exo4D 多视频测试计划

## 0. 决策与当前状态

本轮决策不是“继续找视频”或“直接改 pipeline”二选一，而是：

1. 冻结一个小型、多样化、prediction/evaluation 隔离的多视频 suite；
2. 立即修改已经有因果证据的 pipeline 机制；
3. 先用 development cases 和 exact-state replay 验证机制；
4. 机制和 suite-level acceptance logic 冻结后，再运行 locked internal holdout；
5. 不用更多缺少 object SE(3)/contact GT 的视频制造“完整物理准确率”。

当前 suite 准备属于 **benchmark guardrail**，不是物理 annotation 机制改善。它减少的是“只在一个 tire-lever clip 上修改并过拟合”的风险；P13 geometry、跨时间 object observations、P15/P16 wiring 和 P18 observability 仍需真实修改和渲染验证。

截至本文件记录时：

```text
suite cases:                       12
development:                        6
consumed development reference:     1
locked internal holdout:             5
unique takes/participants/captures: 12 / 12 / 12
development↔holdout participant overlap: 0
development↔holdout capture overlap:     0
local VRS available:                0 / 12
runtime runs launched for suite:    0
```

“locked internal holdout”不等于 externally administered blind test：curator 已查看 raw RGB 和 sparse Mask 来确认目标身份、刚性和 interval。后续 runtime prediction 仍然严格只接收 RGB 和 target text，不能接收 relation track、Mask、hand GT、camera GT 或 narration。

---

## 1. 已有证据及其边界

主要证据：

- HOT3D keyboard `_v2` 完整 V19 审计；
- Ego-Exo4D tire-lever 冻结 V19 v1 完整盲运行；
- `b7b97e6` P14–P19 exact-state pose-gate/quarantine ablation；
- Ego-Exo4D released partial GT：visible Mask、named hand joints、camera trajectory；
- projected completed-Mesh vs sparse visible Mask diagnostic；
- P13/P14/P15/P16/P18 toy problems 和 synthetic regressions。

当前可以量化：

- 2-D visible object Mask；
- released named 3-D hand joints；
- rectified-view hand reprojection；
- camera ATE/RPE；
- projected completed-Mesh silhouette 的联合 failure diagnostic；
- readiness、abstention、quarantine 和 runtime 行为。

当前不能量化：

- object CAD/metric completed geometry；
- object per-frame SE(3)；
- MANO surface；
- metric contact patch/force；
- signed nonpenetration；
- raw RGB 精确 reprojection（本地缺 VRS calibration）。

因此更多 Ego-Exo4D 视频可以回答“错误是否普遍、系统是否正确 abstain、Mask/hand/camera 是否退化”，但不能单独证明完整物理标注正确。

---

## 2. 遇到的问题

### 2.1 已确认 implementation/dataflow bugs

| ID | 问题 | 严重度 | 状态 | 证据/后果 |
|---|---|---:|---|---|
| BUG-01 | P09 explicit eligibility 没传到 P14/P15 | critical | `b7b97e6` 已修复 | 冻结 P14 拟合 27 个 P09 已拒绝 rows |
| BUG-02 | 低支撑 P15 没有稳定 unresolved completion/readiness contract | high | 已修复 | 现在 6<8 时保留完整 uncertain hypothesis，但 `annotation_ready=false` |
| BUG-03 | P16/P18/P18b/render 可混入 unready/stale object-relative state | high | 已修复 | 现在 quarantine 并拒绝 stale artifact 拼接 |
| BUG-04 | selected-frame render output index 没有 source-frame mapping | low | 已修复 | manifest 现在记录 source frame IDs |
| BUG-05 | 顶层 physical/uncertainty state 仍可能停在 P00 | medium | 未修复 | renderer-consumed state 与用户首先看到的顶层 state 可能不一致 |

已修复 bug 的收益是数据流正确和 false-ready 降低，不是 object trajectory accuracy 提升。不得为了提高 projected IoU 恢复 rejected rows。

### 2.2 Pipeline design / architecture problems

#### DESIGN-01：P13 hidden prior 几乎完全主导 completion

Tire-lever：

```text
completed faces:              151831
observed-depth faces:            496
hidden-prior faces:           151335
hidden-prior fraction:         99.67%
silhouette-rejected faces:         0
planar/slab-rejected faces:        0
mesh watertight:               false
```

当前系统将“生成模型补出的 hidden surface”过早升格为 collision/contact 可消费的 object body。局部 anchor ICP residual 下降不能证明 hidden geometry 正确。

#### DESIGN-02：跨时间轴可信 object observations 不足

Tire-lever fixed P14 只有：

```text
trusted frames = [115,116,119,120,121,123]
trusted count = 6
configured minimum = 8
trusted span = 8 local frames
```

其余 144 rows 是 completion，含 141 nearest holds 和 3 interpolations。它们不是新观测。

#### DESIGN-03：P15/P16 phase 顺序断开 nonpenetration coupling

冻结执行：

```text
P15 pose graph
→ P16 MANO/object nonpenetration diagnostic
```

P15 因而没有 P16 constraint target：

```text
nonpenetration target count = 0
nfev = 1
cost = 0
all pose corrections = 0
```

需要先建立可信 sign geometry，再做 P16 constraint construction 和 bounded P15 second pass；不能仅把当前 non-watertight sign hypothesis 接回 P15。

#### DESIGN-04：P18 先允许不可观测大移动，再依赖 output gate

Tire-lever frozen P18 raw：

```text
left max translation correction:  315.646 mm
right max translation correction:  33.948 mm
left/right output-gated rows:      150/150, 150/150
```

无 depth-order/stable object anchor 时，translation 应在 optimizer variable set 内冻结或 bounded，而不是先生成大移动后全量回退。

#### DESIGN-05：contact semantic prior 没有变成 metric anchor

- P17 `likely_contact` 是 semantic/visibility prior；
- P18 `contact_patch_weight=0`；
- selected depth-order vertices 为 0；
- stable object-frame contact anchor rows 为 0；
- non-watertight Mesh 不能提供可信 sign。

因此 contact/nonpenetration 没闭合不是“再调一个 loss weight”可以解决的。

#### DESIGN-06：Runtime 不满足默认设计 invariant

```text
input duration: 5 s
P00→P21:       3652 s
ratio:          730.4× realtime
P19a→P19b:     1271 s
```

在 correctness 修复前不应靠删 full timeline 或 selected-frame render 制造速度；但通过后必须 profiling、batching、vectorization 和 renderer acceleration。

### 2.3 需要 targeted test 才能确认的 suspected bugs

| ID | 风险 | 所需测试 |
|---|---|---|
| SUSPECT-01 | P18 source-grid / factor-mask-grid lookup contract 不统一 | synthetic source→runtime→mask/depth round trip，覆盖 crop、resize、border |
| SUSPECT-02 | P13 silhouette/slab 两个 filter 对所有 hidden faces 都接受 | synthetic visible/free-space contradiction fixture + real before/after triangle support replay |

“结果差”本身不足以把它们写成 implementation bug；必须先区分公式错误、输入太弱和 factor 未真正耦合。

### 2.4 当前输入和 GT 的信息限制

1. NAS 没有 object CAD/metric Mesh 或 per-frame object SE(3) GT；
2. 没有 MANO surface、contact 或 signed nonpenetration GT；
3. 12/12 suite cases 的 metadata 指向 no-image-stream VRS，但本地文件均不存在；
4. monocular RGB 存在 scale、depth、occlusion 和 symmetry ambiguity；
5. object visible Mask 通常仅 1 Hz，hand GT 也不是每帧完整；
6. visible Mask 只能评价可见轮廓，不能认证 amodal hidden geometry。

这些限制应转化为 uncertainty/abstention 或通过新增 sensor/GT 解决，不能被更多 JSON rows、nearest hold 或生成 prior 消除。

---

## 3. 修正顺序与因果实验

### Priority 0：顶层 state 一致性和可重放 instrumentation

先修 BUG-05：

- P21 原子发布最终 canonical physical/uncertainty state；
- 记录 renderer 实际消费 artifact 的 path、revision、SHA256；
- 顶层 state 与 render state 不一致时明确失败；
- 不从 row 数量或文件存在推断物理成功。

这一步成本低，但只修 provenance/consumer correctness，不算 geometry 改善。

### Priority 1：P13 observed surface 与 completion hypothesis 分离

目标状态类型：

```text
observed_surface
completion_hypothesis
triangle_support_state
collision_eligible_surface
signed_geometry_state
```

每个 generated face 至少记录：

- 可见/不可见 frame IDs；
- silhouette consistency；
- camera-ray free-space contradiction；
- visible-depth support；
- measured-surface distance；
- 是否可进入 collision/nonpenetration。

默认规则：hidden completion 没有独立跨帧支持时仍可作为橙色 uncertain hypothesis 渲染，但不能进入 signed collision/contact。

#### 当前实现状态（新分支 mechanism ablation）

已实现并通过 synthetic + tire-lever replay：

- P13 输出 `pose_hypothesis_mesh_labeled` 与 `collision_eligible_mesh_labeled`；legacy `completed_mesh_labeled` 只保留 pose/render hypothesis 语义；
- 默认 collision surface 只含 measured observed faces；`--promote-single-view-hidden-prior-to-collision` 是显式历史诊断 override，仍不产生 sign readiness；
- P14/P15 保持 pose-hypothesis canonical frame，P16/P18 改为消费 collision surface；
- P16/P18 除实际 watertight 外，还要求显式 `geometry_readiness.signed_geometry_ready=true`，否则 signed correction/active-set/dense barrier inactive；
- P14b 记录 per-face visible-depth support、camera-ray first-hit self-visibility、free-space contradiction、same-view repetition、diagnostic viewpoint bins 和 exact separated-view supporting-frame pairs，并输出可视化 QC 和 compressed face-state NPZ；
- standalone regression：`regression_geometry_evidence_contract.py`，覆盖默认 quarantine、历史 override、canonical geometry 精确保留、same-view 不晋级、distinct-view synthetic 晋级、P16/P18 consumer selection 和 legacy-unknown readiness。

Tire-lever 当前结果（depth-grid K + first-hit visibility contract 修正后）：pose hypothesis 151,831 faces；collision 496 observed faces，其中 495 个在 P14 direct frames 至少一次获得 visible-depth support、477 个重复支持、11 个出现过 support/free-space conflict，但 0 个在至少两帧出现 visible free-space contradiction。151,335 hidden faces 中 44,801 只有 same-view repeated support、0 个获得 ≥15° exact separated-view support、17,365 个纯 free-space contradicted、23,206 个在不同帧出现 support/free-space conflict、65,963 个其余 unsupported/self-occluded，因此 hidden collision promotion 仍为 0，sign readiness 仍为 false。该结果暴露证据不足和预测测量冲突，不是通过降低门槛制造 geometry。P14b 投影 UniDepth raster 时使用 NPZ row 的 `intrinsics_fx_fy_cx_cy`；annotation-camera scalar-focal K 只允许在 depth K 缺失时作为显式 fallback。

#### 判别预测

- 如果 hidden prior 是主要故障：held-out-frame silhouette/free-space consistency 改善，unsupported hidden faces 被隔离；
- 如果几乎所有 hidden faces 都被移除：说明 completion measurement 本身不够，不应降低 gate；下一步应增加多视角 evidence；
- 如果 geometry gate 生效但 projection 仍错：camera/object pose 是后续主导项；
- 如果 filter 数值变化但 render 不变：说明 renderer/consumer dataflow 仍断开。

### Priority 2：增加分布在时间轴上的 object observations

Keyframe proposal 必须考虑：

- target Mask area 和 boundary ownership；
- valid metric depth support；
- hand/background contamination；
- camera baseline/viewpoint novelty；
- 与已有 trusted frames 的时间距离；
- occlusion state。

P15 support 不能只计数，还要报告：

```text
trusted direct count
trusted source-frame span
occupied temporal bins
maximum unobserved gap
viewpoint/baseline diversity
```

Interpolation/nearest hold 永远不计为 direct support。低支撑 measurement 继续进入 uncertainty graph 和 failure render，但不得变成 annotation-ready trajectory。

当前只完成了 **support/readiness evaluator**，尚未完成新的 object keyframe acquisition：P14b 已报告 tire-lever 的 direct count=6、span=[115,123]、span fraction=0.05369、occupied bins=2/5、maximum gap=115 frames（0.7667 timeline）、15° viewpoint bins=1、pairwise angular max=11.394°。因此 44,801 个 first-hit-visible repeated-support hidden faces全部归为 same-view-only。下一步仍需让 P09/P14 在时间轴和 viewpoint 上实际产生新的可信 direct observations，不能把这个 evaluator 本身描述为新增观测。

### Priority 3：P15/P16 block-coordinate second pass

建议 phase graph：

```text
P14 trusted observation fits
→ P15a observation-only pose graph
→ P13b/P16 geometry sign-readiness and physical constraints
→ P15b bounded physical pose refinement
→ independent silhouette/depth/hand remeasurement
→ accept or rollback
```

只有 watertight 或局部 sign validity 有明确 provenance 时才激活 nonpenetration。P15b 不得一次性放开 camera/object/MANO 的 gauge。

### Priority 4：P18 grid contract 和 variable observability

- 建立唯一 projection/grid helper；
- synthetic round-trip regression；
- 无 stable object anchor/depth-order 时直接从 variable set 冻结 global translation；
- 每个 active-set pass 输出 per-family loss、gradient norm、closure count、termination reason；
- gate 前后都做独立 hand/camera/geometry 重测；
- object pose unready 时继续 source-only quarantine。

当前不引入 RL。RL 最多在上述机制稳定后控制 scheduling、evidence request、trust region 和 rollback；不能直接预测连续物理状态。

### Priority 5：保持完整机制做 runtime profiling

正确性通过后再优化：

- batch model inference；
- vectorized projection/mask/depth lookup；
- mesh rasterization profiling；
- 避免重复 decode/hash/serialization；
- full-duration renderer acceleration。

不能把 selected-frame QC 称作 full-duration runtime。

---

## 4. 多视频选择标准

### 4.1 机器可验证 hard contract

每个 case 必须满足：

1. raw Aria RGB source MP4 本地存在；
2. 150 帧、30 fps、5 秒 source interval；
3. local frame `0,30,60,90,120` 对应 5 张非空 raw-grid visible Mask；
4. 150/150 world→annotation-camera extrinsics；
5. hand GT coverage 被记录，不要求所有 case 都高覆盖；
6. prediction input 与 evaluation GT 位于物理分离 root；
7. prediction case 目录只能有 `input.mp4` 和 `PREDICTION_INPUT_MANIFEST.json`；
8. prediction manifest 不发布 relation track、Mask、hand GT、camera GT、narration或 rectified K；
9. source take、participant、capture、split 和 SHA256 可追溯；
10. 缺失 VRS 必须显式保留，不能把 rectified K 用作 raw-view K。

### 4.2 Curator visual/physical contract

机器条件不能判断目标是否是合适刚体。Curator 额外检查：

- target identity 在五张 Mask 上一致；
- 目标在 interval 内近似 rigid；
- 不是包装袋、纸张、食材、软管等显著 deformable object；
- 至少 development/holdout 总体包含：大物体、小物体、细长工具、反光物体、低纹理/对称物体、强遮挡和预期 unresolved negative controls；
- target 确实被操作或与手物任务直接相关，而不是远处静态背景物体；
- 不因为某个类别“看起来容易”加入 category-specific pipeline branch。

Mask area、centroid range 和 hand coverage 只是 selection diagnostics，不是 object pose GT。

### 4.3 Split 和过拟合控制

- 6 个 development cases：可用于机制调试；
- 1 个 consumed reference：tire-lever 已经被查看和调试，不再宣称 blind；
- 5 个 locked internal holdout：冻结 case、target、interval 和 suite-level metrics 后才运行；
- development 与 holdout 无 participant/capture overlap；
- 每个 case 使用不同 take、participant 和 capture；
- holdout 不允许逐 case 改 target prompt、threshold 或 branch。

---

## 5. 已选择 suite

Config：

```text
experiments/egoexo4d_rigid_benchmark/multiclip_rigid_suite_v1.json
```

### 5.1 Development 与 consumed reference

| Case | Target | Stratum | Mask mean area | Centroid range / grid diagonal | Hand GT frames / ≥15-joint frames |
|---|---|---|---:|---:|---:|
| tire-lever reference, val, f2040 | yellow tire lever | small/slender、强遮挡、已知低 pose support | 0.169% | 0.100 | 49 / 34 |
| iiith_cooking_117_2, train, f5550 | wooden spatula | elongated、持续手持、pan contact | 0.490% | 0.094 | 50 / 50 |
| iiith_cooking_123_4, train, f540 | black-handle knife | 薄、细长、chopping | 0.262% | 0.190 | 47 / 42 |
| fair_bike_06_8, train, f1290 | socket wrench | tiny、reflective、强遮挡 | 0.272% | 0.241 | 48 / 42 |
| cmu_bike17_3, train, f30 | bicycle wheel | large、rotational symmetry、双手 | 3.307% | 0.314 | 50 / 50 |
| iiith_cooking_79_2, train, f2310 | steel spatula | reflective、入画、较大运动 | 0.388% | 0.483 | 48 / 44 |
| georgiatech_covid_05_4, train, f480 | solution tube | tiny cylinder、低像素、hand→table | 0.076% | 0.347 | 50 / 50 |

### 5.2 Locked internal holdout

| Case | Target | Stratum | Mask mean area | Centroid range / grid diagonal | Hand GT frames / ≥15-joint frames |
|---|---|---|---:|---:|---:|
| iiith_cooking_59_2, val, f4140 | wooden spatula | development analogue、不同 participant/capture | 0.687% | 0.227 | 44 / 43 |
| indiana_bike_10_8, val, f390 | wrench | tiny、hub clutter、强遮挡、不同 site | 0.043% | 0.210 | 41 / 41 |
| iiith_cooking_120_4, val, f1680 | frying pan | large、低纹理、visibility transition | 1.549% | 0.523 | 44 / 37 |
| iiith_cooking_126_2, val, f2250 | blue-handle knife | negative control、离开视野、低 hand GT | 0.129% | 0.333 | 18 / 17（仅 2/5 temporal bins） |
| georgiatech_bike_14_6, val, f180 | wheel hub | small symmetric part、assembly clutter | 0.139% | 0.290 | 48 / 48 |

上述数值来自五张 sparse visible Mask 和 released hand GT，只用于覆盖/难度描述。

### 5.3 Suite aggregate

```text
universities:
  iiith         6
  georgiatech   3
  cmu           1
  fair          1
  indiana       1

tasks:
  Cooking an Omelet                         6
  Fix a Flat Tire - Replace a Bike Tube     2
  Install a Wheel                           3
  Covid-19 Rapid Antigen Test               1
```

所有 12 cases 都有五张非空 Mask 和 150/150 camera extrinsics。11/11 新选 cases 可由类别无关 scanner 的 deterministic best-window contract 重新得到；tire-lever 例外保留历史冻结 f2040 interval，而不是改成 scanner 当前偏好的高 hand-coverage window。Blue-handle knife 被有意保留为低 temporal support negative control；它的目标不是逼 pipeline 输出 pose，而是验证系统能否继续产生 uncertainty render 且不 false-ready。

### 5.4 Selection review 中明确拒绝或修正的候选

| 候选 | 决定 | 原因 |
|---|---|---|
| COVID test kit/package | 不作为 rigid positive | 开盒/包装 flap 改变形状，track 可能合并 package 与 device |
| static spoon / pepper container / tea container | 拒绝 | 手在操作其他对象，target 只是背景可见，不是该 interval 的 hand-object target |
| pot gripper | 拒绝 | 铰链工具且早期严重出画；不能稳定当单 rigid body |
| yellow knife in `iiith_cooking_59_2` | 拒绝 | 目标躺在台面，实际手持的是 wooden spatula；改选 wooden spatula |
| frying pan f1650 | interval 改为 f1680 | f1650 relation row 解码为 0 foreground；f1680–1800 五张 Mask 均非空且保留 visibility transition |
| 多 Aria stream target | 固定到 target 对应 stream | 不允许仅按 take 选择错误 wearer/camera；suite audit 同时验证 camera payload mapping |

这些决定属于 benchmark curation，不会成为 object-category-specific runtime branch。

---

## 6. 目录、隔离和运行身份

Selection scanner：

```text
experiments/egoexo4d_rigid_benchmark/scan_multiclip_candidates.py

/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_v19_rigid_multiclip_v1_candidate_scan.json
```

Scan 结果：train `1030` + val `172` = `1202` candidate target-track/stream windows。

Suite auditor/preparer：

```text
experiments/egoexo4d_rigid_benchmark/prepare_multiclip_suite.py
```

复现 candidate scan：

```bash
cd /mnt/user-home/kupingxin/ego_annotation
.venv/bin/python experiments/egoexo4d_rigid_benchmark/scan_multiclip_candidates.py \
  --output /mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_v19_rigid_multiclip_v1_candidate_scan.json
```

重新准备 suite（`--replace` 会删除并重建两个 suite roots）：

```bash
cd /mnt/user-home/kupingxin/ego_annotation
.venv/bin/python experiments/egoexo4d_rigid_benchmark/prepare_multiclip_suite.py \
  --prediction-root /mnt/truenas-user-home/kupingxin/ego_annotation_inputs/egoexo4d_v19_rigid_multiclip_v1 \
  --benchmark-root /mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_v19_rigid_multiclip_v1 \
  --output-manifest /mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_v19_rigid_multiclip_v1/SUITE_PREPARATION_MANIFEST.json \
  --prepare \
  --replace \
  --render-selection-review
```

Prediction root：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/
  egoexo4d_v19_rigid_multiclip_v1/
```

Evaluation-only benchmark root：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_v19_rigid_multiclip_v1/
```

权威 preparation manifest：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_v19_rigid_multiclip_v1/SUITE_PREPARATION_MANIFEST.json
```

12 张 curator review sheets：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_v19_rigid_multiclip_v1/_selection_review_evaluation_only/
```

这些图片含 released Mask overlay，绝不能进入 runtime workspace。

Integrated validation：

```text
prepared cases:                              12 / 12
prediction directories physically isolated: 12 / 12
coordinate-contract self-tests:              12 / 12 pass
endpoint source-frame alignment:             12 / 12 pass
max endpoint prepared↔source MAE:             1.178 BGR levels
exo_rigid_001 video SHA256:                   37de09c1193bc5c56e23a4c9ea49caa1d38623cb4f6ea5e92dc03a78d9f29ec4
runtime runs launched:                        0
```

每个 prediction case 必须严格是：

```text
exo_rigid_001 ... exo_rigid_012
```

Prediction IDs 故意使用不含 split、role、take 或 target 的 opaque aliases；可读的 take/interval/evaluation label 只保存在 suite/GT side。每个 opaque case directory 严格是：

```text
<case_id>/
├── input.mp4
└── PREDICTION_INPUT_MANIFEST.json
```

每个 evaluation case 独立保存：

```text
<case_id>/ground_truth_v2_rectified_camera/
├── BENCHMARK_MANIFEST.json
├── GROUND_TRUTH_AVAILABILITY.json
├── camera_pose_gt.json
├── hand_pose_gt.json
├── object_visible_mask_gt.json
├── object_visible_masks_1408/
├── object_visible_masks_960/
└── ground_truth_preview.jpg
```

Runtime agent 只能收到某个 prediction case directory、fresh run root、case ID、runtime bundle/spec 和 target hint；不能看到 suite config、selection report 或 benchmark root。

---

## 7. 下一步执行协议

### Phase A：只在 development 上建立 staged baseline

先运行 development 的 measurement/geometry/pose stages，重点到 P14：

```text
P03 depth/intrinsics
P04 hand
P06/P07 object track
P09 visible metric observations + eligibility
P13 observed/completed geometry
P14 eligible-only pose fit
```

这类 staged run 必须标为 diagnostic，不冒充完整 V19。它用于回答：

- trusted rows 少且集中是否跨对象普遍；
- P13 zero-rejection 是否普遍；
- 哪些物体有跨时间 viewpoint support；
- 小物体与大/对称物体的 failure mode 是否不同。

### Phase B：优先 exact-state correction replay

机制修改先重放：

1. tire-lever frozen artifacts；
2. keyboard `_v2` frozen artifacts；
3. wooden spatula、bicycle wheel、solution tube 三个 development strata。

每次修改必须有 causal prediction、geometry/render evidence 和 rollback 条件，不能顺序猜 threshold。

### Phase C：development full P00–P21

只有以下机制通过后，先完整运行 2 个 development cases：

- P13 support-aware geometry；
- P15 support quantity + temporal coverage；
- P15/P16 second-pass coupling 或明确 inactive；
- P18 grid regression 和 variable freeze；
- 顶层 state 与 renderer state 一致。

完整运行必须保留 150 帧/full-duration render，不用 selected frames 伪造 runtime。

### Phase D：锁定 holdout

在查看任何 holdout prediction output 前冻结：

- source revision/runtime bundle；
- target hints；
- per-family metrics；
- readiness/abstention logic；
- no per-case retry/tuning policy。

然后一次性运行 5 个 holdout。出现基础设施 crash 可以在不改变 prediction mechanism 的前提下重启；感知/geometry 失败不能逐 case 改 prompt 或 threshold。

---

## 8. Go / no-go 标准

这些是 annotation-ready 与 full-run 资源分配标准，不是阻止 noisy measurements 继续进入 uncertainty graph/render 的全局 stop gate。

### 必须满足的 correctness invariants

- explicit-false observation consumption count = 0；
- interpolation/nearest hold 不计 direct support；
- P15 unready 必须传播到 P16/P18/P18b/render；
- raw/rectified grid contract regression 通过；
- final top-level state 与 renderer-consumed state revision/hash 一致；
- prediction bundle 无 GT sidecar。

### Geometry/pose annotation-ready 条件

- direct trusted observations 同时满足数量、时间 span、temporal bins 和 viewpoint diversity；
- P13 hidden faces 有独立 multi-frame support，否则保持 hypothesis；
- collision-eligible geometry 有 sign validity provenance；
- P15 不是仅 completion rows 造成“支持充分”；
- 若 active physical factors 为 0，必须 unresolved，不能由 `nfev=1,cost=0` 推断成功；
- correction 后 independent silhouette/depth/hand checks 不退化。

开发阶段可以测试 provisional coverage threshold，但必须在 development 冻结后再看 holdout；不能根据 holdout 调 threshold。

### P18 条件

- 无 depth-order/stable object-frame anchor 时 global translation 不进入自由变量；
- per-family loss/gradient/closure 有记录；
- gate 前后独立 hand GT 重测；
- raw optimizer state 不自动晋级 canonical P18b。

### Suite-level 报告

必须 macro-average 并同时报告：

```text
ready coverage
unresolved/abstention rate
false-ready evidence count
accuracy conditional on ready
all-case Mask/hand/camera metrics
runtime distribution
```

不能只 pool 全部 pixels/joints，也不能因 fixed pipeline 更常 abstain 就把 coverage loss 写成 accuracy gain。

---

## 9. 完整物理 GT 的并行工作

该 suite 主要解决 breadth 和 regression，不解决完整物理 certification。仍需寻找或制作至少 3–5 个具有以下数据的 clips：

```text
calibrated raw cameras
object CAD/scanned metric mesh
per-frame object SE(3)
metric MANO surface or reproducible MANO parameters
contact labels
signed penetration/nonpenetration GT
```

如果公开数据不具备这些字段，小规模受控采集比继续增加无 object pose GT 的 Ego-Exo4D clips 更有价值。

---

## 10. 明确禁止的捷径

1. 不降低 `min_graph_frames` 来让 tire-lever 变 ready；
2. 不恢复 27 个 explicit-false rows；
3. 不把 141 nearest holds 写成 141 个观测；
4. 不把 projected Mask IoU 当 object SE(3) GT；
5. 不把 semantic likely-contact 当 metric contact；
6. 不让 non-watertight Mesh 提供全局 signed nonpenetration；
7. 不先放开 camera/object/MANO/contact 一次性联合 MAP；
8. 不用 RL 代替 calibration、geometry 或 observability；
9. 不因更多视频都失败就把 failure frequency 当物理准确率；
10. 不把 staged/selected-frame run 冒充 full-duration V19。

下一项真正的机制工作应从 **P13 triangle support/hidden completion quarantine** 开始，同时为 development cases建立 P00–P14 staged artifacts；随后才是跨时间 object observations 和 P15/P16 second pass。
