# Ego-Exo4D tire-lever：V19 完整盲运行与 partial-GT 结果

## 0. 结论

这次 V19 phase graph 已完整运行，launcher 返回：

```text
exit_code=0
```

但这不是“物理标注成功”。runtime agent 在 P19/P21 的最终判断是：

```text
completed with an exposed physical failure
```

综合独立 GT、runtime 内部残差和最终 render，可得：

1. **目标身份基本找对**：黄色撬胎棒 track 覆盖 150/150 帧，且 agent 主动修复了 frame 10 和 frame 120 的错误 support。
2. **可见 Mask 有效但明显偏厚/过分割**：5 张 sparse GT 上 mean IoU `0.6522`、mean recall `0.9855`、mean precision `0.6587`。
3. **HaWoR 手部 articulation 尚可但 metric translation 很差**：root-relative MPJPE `35.56 mm`，absolute MPJPE `192.96 mm`，mean wrist error `186.30 mm`。
4. **P18 raw 对独立 hand GT 仅有极小改善**：absolute MPJPE 改善 `0.516 mm`，root-relative 改善 `0.325 mm`，wrist error 完全不变；P18b canonical 与 HaWoR baseline 完全相同。
5. **camera 短时旋转较平滑，长时 drift 明显**：1-frame local-camera translation RPE mean `7.05 mm`、rotation RPE mean `0.345°`；30-frame分别升到 `130.77 mm` 和 `6.59°`。
6. **object metric geometry/pose 失败**：只有 6/33 个 metric visible rows 被 P09 自己判定为 rigid-pose eligible，但 P14/P15 仍消费了全部 33 行；最终 Mesh 是局部、锯齿状、近似薄片的错误 completion，且 world render 与双手不一致。
7. **contact/nonpenetration 没有闭合**：Mesh 非 watertight；P15 没有 nonpenetration target；P18 depth-order support 为 0；P17 的 6 个 contact rows 没有 stable pose anchor，P18 几何 contact weight 实际为 0。
8. 当前 NAS 没有 object CAD/6DoF/contact/nonpenetration GT，因此不能为这些 family 制造误差数字；但最终 render 和内部机制已经足以拒绝“成功物理标注”的结论。
9. **Runtime 也不达标**：P00→P21 用时 `3652 s` 处理 `5 s` 视频，约实时的 `730×`；这不符合仓库对 V18+ default runtime“与输入时长同一数量级”的 invariant。

---

## 1. 运行身份与隔离

### Runtime

```text
bundle:
/mnt/user-home/kupingxin/ego_annotation_runtime/v19_bundle_a800_b18cecd_local2

source revision:
b18cecd2c90932282eb3b50ecef7298bb7546498

run root:
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/
  20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_kupingxin_v1

run size: 540 MB
files:    1702
```

### Input

```text
frames:   150
size:     960 × 960
fps:      30
length:   5.0 s
SHA256:   37de09c1193bc5c56e23a4c9ea49caa1d38623cb4f6ea5e92dc03a78d9f29ec4
```

### Runtime budget

根据 durable harness events：

```text
P00: 2026-08-06T10:45:01Z
P21: 2026-08-06T11:45:53Z
elapsed: 3652 s = 60 min 52 s
input duration: 5.0 s
runtime / video duration: 730.4×
```

其中：

```text
P17→P18: 545 s
P19a→P19b render: 1271 s
```

这次运行满足“完整 phase graph + full-duration render”的实验要求，但不满足项目的 default-runtime invariant；不能把它描述为可部署实时或近实时 pipeline。

Runtime 只收到：

```text
yellow rigid plastic bicycle tire lever
```

以及排除对象文字；没有收到 GT hand、GT Mask、camera extrinsics 或 action annotation。

对 runtime session 和 run root 的路径搜索没有发现 evaluation-only GT 目录引用。修正后的 v2 benchmark adapter 重新生成了相同字节的视频；evaluator 验证两条路径的 SHA256 完全相同。

### Preflight

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/
  runtime_preflight_egoexo4d_tire_lever_v1.json
```

状态：

```text
ready_for_runtime_agent_launch
failed_checks = []
```

修正 camera contract 后的 raw-RGB-only v2 bundle 也通过独立 preflight：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/
runtime_preflight_egoexo4d_tire_lever_raw_distorted_v2.json

status = ready_for_runtime_agent_launch
files  = input.mp4 + PREDICTION_INPUT_MANIFEST.json
```

没有立即重复启动 v2：它与已完成 run 的 RGB SHA256 完全相同，且当前仍缺 VRS raw distortion calibration；在没有新增观测的情况下重跑只会重复同一 monocular calibration 条件。v2 fresh run root 保持未创建，待取得 VRS 后再使用。

---

## 2. 必须说明的 camera-coordinate 修正

最初 adapter 曾错误地认为：

```text
Ego-Exo4D hand UV/K 位于 raw 448×448 RGB grid
```

这个假设已经被撤销。官方代码明确说明：

- ego-pose K `f=150, cx=cy=255.5` 属于 **512×512 undistorted linear camera**；
- raw extracted Aria image 在 undistortion 前先旋回 90°；
- rectification 还需要 VRS 中的 raw `camera-rgb` distortion calibration。

官方代码来源：

```text
facebookresearch/Ego4d
commit 4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb

ego4d/internal/human_pose/undistort_to_halo.py
ego4d/internal/human_pose/utils.py::aria_original_to_extracted
```

固定 camera-vector adapter 为：

```text
X_extracted = A @ X_annotation

A = [[0, -1, 0],
     [1,  0, 0],
     [0,  0, 1]]
```

它来自官方 image orientation contract，不是用本次 GT 拟合出来的。

因此：

- 旧 v1 hand MPJPE `588.742 mm` 和 raw-grid reprojection `417.799 px` 已明确标记 `invalidated_do_not_report`；
- object Mask 指标不受影响，因为 relation masks 原本就在 raw 1408 grid；
- 3D hand/camera comparison 使用上述固定轴变换；
- hand 2D 只在官方 rectified 512 view 中评测；
- 没有 VRS distortion calibration 时，不报告 raw 960 RGB hand reprojection。

修正后的 GT：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189/
  ground_truth_v2_rectified_camera
```

修正后的 evaluator report：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189/
  evaluation_v2_rectified_camera_contract/
  egoexo4d_partial_gt_evaluation.json
```

Coordinate adapter synthetic self-test：

```text
exact synthetic hand absolute error: 9.2e-7 mm
exact synthetic hand root-relative:  1.3e-13 mm
exact synthetic camera SE3 ATE RMSE:  5.8e-14 mm
synthetic 3D PCK@50 mm:              1.0
```

synthetic rectified reprojection仍有 `3.45 px` mean，是 released 2D annotation 与 released 3D reprojection本身的差异，不是 adapter error。

---

## 3. GT 支持范围

### 可独立量化

| Family | Coverage | Grid / units |
|---|---:|---|
| visible object Mask | 5/150 frames | raw 1408，resize 到 raw 960 |
| hand 3D named joints | 49/150 frames | world meter；评测时转 annotation camera |
| hand 2D named joints | 49/150 frames | rectified 512 view |
| camera trajectory | 150/150 frames | world→annotation-camera SE(3) |

Hand quality filter：

```text
num_views_for_3d >= 2
released rectified reprojection error <= 20 px
eligible named joints = 916
matched joints = 916
```

### 不存在 released GT

- MANO parameters / vertices；
- dense metric depth；
- object CAD / metric Mesh / dimensions；
- object per-frame 6DoF；
- object 3D surface points；
- metric contact point/patch/force；
- SDF / signed nonpenetration；
- per-pixel occlusion owner。

所以没有任何“overall physical accuracy”分数。

---

## 4. Object visible-Mask 指标

GT local frames：

```text
0, 30, 60, 90, 120
```

### Aggregate

| Metric | mean | median | p90 | max |
|---|---:|---:|---:|---:|
| Mask IoU | 0.6522 | 0.5973 | 0.7663 | 0.7752 |
| precision | 0.6587 | 0.6039 | 0.7754 | 0.7832 |
| recall | 0.9855 | 0.9822 | 0.9915 | 0.9946 |
| bbox IoU | 0.9014 | 0.9095 | 0.9310 | 0.9431 |
| centroid error px | 4.751 | 5.228 | 7.771 | 9.446 |
| boundary F1 @ 5 px | 0.9369 | 0.9722 | 0.9801 | 0.9848 |

### Per frame

| frame | IoU | precision | recall | bbox IoU | centroid px | boundary F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7752 | 0.7832 | 0.9869 | 0.8794 | 2.035 | 0.9381 |
| 30 | 0.7529 | 0.7636 | 0.9817 | 0.9431 | 5.259 | 0.9848 |
| 60 | 0.5541 | 0.5598 | 0.9821 | 0.9095 | 9.446 | 0.8161 |
| 90 | 0.5973 | 0.6039 | 0.9822 | 0.9128 | 1.787 | 0.9722 |
| 120 | 0.5814 | 0.5832 | 0.9946 | 0.8621 | 5.228 | 0.9730 |

解释：

- 目标位置和 bbox 基本正确；
- recall 接近 1，但 precision 只有约 0.66；
- 主要错误不是漏掉撬胎棒，而是把附近黄色/手边缘 support 包得过宽；
- frame 60 最差，IoU `0.5541`。

这些数字只评价 P06/P07 visible segmentation，不评价 complete geometry 或 object pose。

---

## 5. Hand partial-GT 指标

### Aggregate comparison

| State | absolute MPJPE mm | root-relative MPJPE mm | wrist error mm | rectified reproj px | 3D PCK@20 | 3D PCK@50 |
|---|---:|---:|---:|---:|---:|---:|
| P04 HaWoR | 192.956 | 35.559 | 186.295 | 30.139 | 0.0000 | 0.0011 |
| P18 raw | 192.440 | 35.234 | 186.295 | 30.058 | 0.0000 | 0.0011 |
| P18b canonical | 192.956 | 35.559 | 186.295 | 30.139 | 0.0000 | 0.0011 |

P18 raw 相对 HaWoR：

```text
absolute MPJPE:      -0.516 mm
root-relative MPJPE: -0.325 mm
wrist error:          0.000 mm
rectified reproj:    -0.081 px
```

这是极小变化，不能称为有意义的 metric hand correction。

P18b 与 HaWoR 完全一致，符合：

```text
metric_mano_preserved_from_hawor
```

### HaWoR distribution

```text
absolute joint error:
  mean   192.956 mm
  median 163.206 mm
  p95    484.327 mm
  max    616.554 mm

root-relative joint error:
  mean    35.559 mm
  median  32.025 mm
  p95     80.163 mm
  max    174.234 mm

wrist error:
  mean   186.295 mm
  median 134.161 mm
```

By side：

| side | absolute mean mm | root-relative mean mm | wrist mean mm | rectified reproj mean px |
|---|---:|---:|---:|---:|
| left | 206.272 | 36.007 | 189.460 | 35.701 |
| right | 184.413 | 35.187 | 182.381 | 26.571 |

### Signed camera-axis bias

全部 916 个关节：

| annotation-camera axis | mean signed mm | median signed mm | mean absolute mm |
|---|---:|---:|---:|
| x | +142.041 | +127.720 | 142.150 |
| y | -12.698 | -16.885 | 52.710 |
| z | +99.629 | +79.821 | 100.025 |

这表明主要问题不是单纯 finger articulation：整个手在 camera x/z 上存在很大的 systematic translation bias。raw distorted RGB 上使用 monocular-estimated pinhole K 是一个重要来源，但这些数字只证明最终 hand state 与 released metric GT 不一致，不能单独把全部误差归因于某一个模型。

---

## 6. Camera trajectory partial-GT 指标

### Coverage / scale

```text
common frames:             150
GT trajectory length:      0.939603 m
HaWoR trajectory length:   0.816809 m
prediction / GT:           0.869313
```

### Position alignment

| Alignment | scale | center ATE RMSE | center median | center max |
|---|---:|---:|---:|---:|
| SE(3), metric scale preserved | 1.0 | 58.588 mm | 25.660 mm | 183.070 mm |
| Sim(3), diagnostic only | 1.180235 | 51.848 mm | 30.053 mm | 118.396 mm |

Sim(3) scale只能诊断 scale drift，不能作为 metric prediction 接受值。

### Orientation gauge

使用所有 camera orientations 拟合单一 SO(3) world gauge 后：

```text
mean:    7.795°
median:  6.023°
p95:    23.966°
max:    24.554°
```

该 orientation-fit gauge 对应的 center RMSE 为 `225.407 mm`。也就是说，position-fit world gauge 与 orientation-fit world gauge 明显不一致，不能只挑其中一个较小数字声称整体 camera 已对齐。

### Gauge-invariant RPE

| delta | pairs | local-camera translation mean | median | p95 | relative rotation mean | median | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 frame | 149 | 7.055 mm | 3.389 mm | 23.822 mm | 0.345° | 0.222° | 1.059° |
| 30 frames | 120 | 130.770 mm | 31.043 mm | 509.170 mm | 6.591° | 2.066° | 23.900° |

结论：短时 rotation continuity 尚可，但 1 秒尺度出现很强的长尾 drift。

---

## 7. V19 runtime 内部阶段审计

### P03/P03b：depth 与 calibration

```text
UniDepth frames:            150
focal median:               347.342 px
focal p05/p95:              337.092 / 388.407 px
full-frame depth median:    1.2714 m
runtime constant K:
  fx=fy=347.299875
  cx=cy=487.724762
```

这个 K 是 raw distorted RGB 上的 UniDepth robust-video constant hypothesis，不是 released raw sensor calibration。

本地 shard 没有 VRS calibration；released `f=150,c=255.5` 是 rectified 512 K，不能覆盖到 runtime raw RGB。

### P04：HaWoR

```text
valid output rows:             left 150, right 150
detected in same frame:        left 101, right 136
propagated/infilled rows:       left 49,  right 14
img_focal:                      347.299875 px
```

独立 GT 已表明 full-timeline valid 不等于 metric accurate。

### P06/P07：detector + tracker

Final prompts：

```text
yellow bicycle tire lever.
yellow plastic tire lever.
```

Final prompt frames：

```text
0, 5, 15, 20, 30, 45, 60, 75, 90, 105, 110, 135, 149
```

Agent 做了两次必要 repair：

1. frame 120 的初始 OWLv2 box 选到右侧 repair stand；
2. frame 10 的初始 SAM2 support 包含黑色 bicycle brake handle。

修复后 track 仍有过分割，但 sparse GT 显示目标 identity 和 bbox 大体正确。

### P09：visible metric geometry

```text
SAM2 visible rows:                 150
metric visible surface rows:        33
P09 rigid-pose eligible rows:         6
P09 rigid-pose ineligible rows:      27
selected anchor:                    115
anchor visible points:               60
anchor depth median:             0.532 m
anchor depth p05-p95 spread:      0.078 m
anchor visible-support diagonal:  0.110 m
```

Eligible frames：

```text
115, 116, 119, 120, 121, 123
```

其余 27 个 metric rows 被 P09 自己标记：

```text
systematic_mask_extent_inconsistent_with_selected_anchor_rigid_object_probable_hand_background_leakage
```

### P12/P13：TRELLIS 与 metric adaptation

TRELLIS raw：

```text
vertices: 89074
faces:    177790
model extent: [0.8336, 1.0008, 0.0574]
```

P13 internal alignment：

```text
scale: 0.0898406
observed→TRELLIS median: 3.164 mm → 2.988 mm
completed vertices/faces: 77761 / 151831
```

这只是对 frame-115 partial visible surfels 的内部 fit，不是 GT shape error。

两个 intended rejection gate 实际都没有删掉 TRELLIS face：

```text
silhouette free-space rejected: 0
planar-slab rejected:           0
```

最终 accepted hidden prior face 达 `151335`，而 observed-depth faces 只有 `496`。因此 accepted body 几乎完全由单帧 TRELLIS hidden prior 决定。

### P14：per-frame visible pose fitting

```text
fit rows:       33
missing rows:  117
sample count: 6000
iterations:      4
```

关键数据流错误：P14 消费了全部 33 个 metric rows，包括 P09 已拒绝的 27 行。

按 P09 eligibility 重新分组：

| Rows | count | initial observed→mesh median-of-medians | final |
|---|---:|---:|---:|
| eligible | 6 | 3.956 mm | 2.006 mm |
| ineligible | 27 | 51.783 mm | 63.425 mm |
| all | 33 | 47.745 mm | 63.038 mm |

P14 report 的 all-row final：

```text
median 63.038 mm
p90   132.089 mm
max   245.009 mm
```

所以本次 P14 对污染 rows 的 aggregate fit 反而变差。不能把 33/33 optimizer completion 当成可靠 object pose。

### P15：rigid pose graph

```text
graph frames:                 33
nonpenetration target frames:  0
nfev:                          1
cost:                          0
translation corrections:      all zero
rotation corrections:         all zero
```

Full timeline：

```text
direct rows:        33
interpolated rows:  91
nearest holds:      26
```

P15 只填满时间轴，没有通过 active factor 改善 pose。

### P16：nonpenetration

```text
measured hand/frame pairs: 300
completed Mesh watertight: false
sign Mesh watertight:      false
candidate corrections:       0
```

零 correction 和零 penetration count 不是无穿透证明。

### P17：agent/contact/ownership factor

```text
ownership rows:             12
contact rows:               12
active contact rows:         6
active frames: 115,116,119,120,121,123
active side: right
contact image pixels:     9332
non-object-owned pixels:  2384
```

这些 rows 来自 agent `likely_contact` 加 image adjacency，只是 prior/factor，不是 metric contact GT。

### P18：joint MANO interval optimizer

本次实际配置：

```text
contact_patch_weight = 0.0
visibility-weighted hand observation = false
visible-surface track factor = false
visible-object-mask gate = false
surface-eligibility factor = false
visible-depth-order term requested = true
selected depth-order vertices = 0 on both sides
```

#### Left

```text
corrected frames: 45
initial observed-surface penetration max:       172.388 mm
full post-solver penetration max:                39.314 mm
active-constraint residual max:                 122.358 mm
active-set closed: false
raw translation correction max:                315.646 mm
output translation gate applied rows:          150/150
```

#### Right

```text
corrected frames: 30
initial observed-surface penetration max:         4.760 mm
full post-solver penetration max:                10.526 mm
active-constraint residual max:                  10.526 mm
active-set closed: false
raw translation correction max:                 33.948 mm
output translation gate applied rows:          150/150
```

Right contact factor：

```text
active rows:                         6
prior probability median:         0.88
posterior probability median:     0.88
posterior - prior median:        -4.77e-9
initial patch distance median:    43.16 mm
final |normal gap| median:        14.13 mm
support uncertainty:              65.00 mm
stable pose-anchor residual rows:      0
object-frame centroid p95 dispersion: 68.43 mm
```

由于 contact geometry weight 为 0、无 stable anchor、uncertainty 大于目标尺度，posterior 基本只是 prior 重放。

External hand GT 也确认：P18 raw 的 aggregate improvement 小于 1 mm，wrist 完全不变。

### P18b

```text
rows: 300
metric joint shift: 0 px
contact distance rows: 0
visible-surface distance rows: 0
```

P18b 正确地拒绝把 raw surface hypothesis 升格为 canonical MANO，但也没有修正 HaWoR 的 metric hand translation error。

---

## 8. Render-based physical rejection

P19b 生成了 150-frame overlay/world/side-by-side videos。Agent 的具体现象：

- green body 是 jagged planar partial sheet，不是完整细长撬胎棒；
- overlay 中只覆盖黄色工具的一小块或发生 offset；
- world view 中 object body 与 controlling right hand 分离；
- 某些 frame 中与错误 hand/world location 相交；
- contact sample 和 signed nonpenetration 都不成立。

代表帧：

```text
frame 60:
  green object 在 world view 中位于双手上方，明显不与任一手一致

frame 115:
  green sheet 与大量 hand vertices/skeleton 混杂，仅对应局部 support
```

因此 P20 发布了 diagnostic render，但 P19c presentation rerender 被主动跳过：styling 不能修复 geometry/pose failure。

Canonical videos：

```text
$RUN/renders/v19_overlay.mp4
$RUN/renders/v19_world.mp4
$RUN/renders/v19_side_by_side.mp4
```

---

## 9. 可报告与不可报告的结论

| Claim | Status | 依据 |
|---|---|---|
| 找到黄色撬胎棒目标 | 部分支持 | detector repairs、track、5-frame GT Mask |
| visible segmentation 高 recall | 支持 | mean recall 0.9855 |
| visible segmentation 高精度 | 不支持 | mean IoU 0.6522、precision 0.6587 |
| camera short-term rotation smooth | 部分支持 | 1-frame RPE 0.345° mean |
| camera globally metric accurate | 不支持 | SE3 ATE 58.6 mm、1 s long-tail drift、scale diagnostic 1.18 |
| hand articulation reasonable | 部分支持 | root-relative MPJPE 35.6 mm |
| hand metric translation accurate | 明确不支持 | wrist 186.3 mm、absolute MPJPE 193.0 mm |
| P18 明显改善 hand GT | 不支持 | 仅改善 0.516/0.325 mm，wrist 0 mm |
| object geometry accurate | 不可量化且视觉拒绝 | 无 GT CAD；render 为 partial sheet |
| object 6DoF accurate | 不可量化且内部不可靠 | 无 GT pose；27 个 ineligible rows 被消费；P15 inert |
| metric contact established | 不支持 | 无 GT；weight 0；无 anchor；65 mm uncertainty |
| nonpenetration established | 不支持 | non-watertight；无 P15 target；P16 zero candidate |
| 完整高精度物理标注 | 明确不支持 | 上述多项共同失败 |
| default runtime 与输入同一数量级 | 明确不支持 | 3652 s / 5 s = 730.4× |

---

## 10. Runtime contract/audit 异常

### 10.1 P09 eligibility 没有传到 P14

这是本次最直接的 correctness bug：

```text
P09 eligible = 6
P14 consumed = 33
P14 consumed ineligible = 27
```

应先修复它，再讨论更复杂的联合 MAP 或 RL。

### 10.2 P15 没有 active correction

P15 没有收到 P16 target，并在零 correction 处 `nfev=1/cost=0`。full timeline completion 不能被解释为优化成功。

### 10.3 Raw Aria calibration 不闭合

当前 NAS 只有 raw downscaled RGB，没有该 take 的 VRS/no-image-stream VRS calibration。因而：

- 不能生成官方 512 rectified prediction clip；
- 不能把 rectified K 当 raw K；
- runtime 只能使用 monocular pinhole approximation。

要补齐，需下载：

```text
takes/georgiatech_bike_07_10/aria06_noimagestreams.vrs
```

然后用 `projectaria_tools` 和官方 `undistort_to_halo.py` contract 生成真正 rectified clip。

### 10.4 顶层 state 文件保持 P00 初值

运行结束后：

```text
state/v19_physical_state.json
state/v19_uncertainty_state.json
```

仍是 `initial_unresolved`/“awaiting P03/P04”等 P00 内容。当前实际可消费状态应以：

```text
state/render_state/tire_lever_rigid_render_state.json
state/v19_agent_evidence.md
logs/harness_events.jsonl
```

为准。顶层 state 不更新是一个 provenance/usability 缺口。

当前 spec 不要求 `FINAL_RUNTIME_AUDIT.json`，本次也没有生成该文件。

---

## 11. 下一步优先级

1. **修 P09→P14 eligibility wiring**：P14/P15 默认拒绝 ineligible rows。
2. **取得 Aria no-image-stream VRS calibration**，生成官方 rectified 512/960 clip；不要继续混用 raw RGB 与 rectified K。
3. **对 object evidence 做 multi-frame fusion**，而不是从 60 个 anchor points/64×64 partial crop决定全部 hidden body。
4. **P13 acceptance gate 必须真的删除 unsupported faces**；本次 silhouette/slab gate 均删除 0 face。
5. **P15 放到可信 nonpenetration constraint 之后再跑第二 pass**，并验证 residual 非零。
6. **P18 没有 depth-order support 时冻结 translation/root correction**；不要先产生 316 mm raw move 再全部 gate 回去。
7. **contact factor 必须有 stable object-frame anchor 和 nonzero audited weight**。
8. **用独立 hand/camera GT 做回归测试**，要求每个新优化 phase 同时报告 baseline、raw、canonical 和 gate 后指标。
9. 若要完整 object/contact benchmark，另采 CAD、object SE(3)、MANO surface 和 contact/nonpenetration GT；当前 shard 无法补造。
10. **单独优化 runtime**：P19 render 当前占 `1271 s`，P18 占 `545 s`；在保持同一物理机制和 full-duration artifact 的前提下做 profiling、batching/vectorization，不能用少帧 debug render 冒充 full pipeline 加速。
