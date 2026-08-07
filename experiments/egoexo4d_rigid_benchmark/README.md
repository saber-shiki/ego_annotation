# Ego-Exo4D rigid hand-object partial benchmark

本目录用于：

1. 审计 `/mnt/nas-106/ego4d` 中实际存在的 Ego-Exo4D v2 真值；
2. 构造 prediction/evaluation 物理隔离的刚体手物 clip；
3. 对 V19 的目标可见 Mask、named hand joints 和 camera trajectory 做 partial-GT 评测；
4. 对缺少 GT 的 object geometry/6DoF/contact/nonpenetration 明确输出 `not_evaluated`。

完整运行结果与当前输出/GT 故障分类见：

- [`RESULTS_V19_V1_ZH.md`](RESULTS_V19_V1_ZH.md)
- [`ORIGINAL_V19_PIPELINE_ISSUES_BRIEF_ZH.md`](ORIGINAL_V19_PIPELINE_ISSUES_BRIEF_ZH.md)
- [`CURRENT_OUTPUT_VS_GT_AND_FAILURE_TAXONOMY_ZH.md`](CURRENT_OUTPUT_VS_GT_AND_FAILURE_TAXONOMY_ZH.md)
- [`NEXT_PIPELINE_CORRECTION_AND_MULTICLIP_PLAN_ZH.md`](NEXT_PIPELINE_CORRECTION_AND_MULTICLIP_PLAN_ZH.md)

---

## 1. 文件

```text
prepare_benchmark.py
    构造 raw Aria RGB prediction bundle 和独立 evaluation-only GT。

evaluate_benchmark.py
    评测 sparse visible Mask、3D/rectified-2D hand joints 和 camera trajectory。

self_test_coordinate_contract.py
    用 exact synthetic 3D round trip 验证固定 camera-axis adapter 和 evaluator 数学实现。

compare_current_outputs_to_gt.py
    重算 available GT，并比较冻结/fixed projected Mesh、hand、camera 和 failure taxonomy。

scan_multiclip_candidates.py
    类别无关扫描五张非空 1 Hz Mask、完整 camera、raw MP4 和 hand coverage 的候选 windows。

multiclip_rigid_suite_v1.json
    12-case development/locked-holdout curated suite identity、target hint、rigidity assumption 和 strata。

prepare_multiclip_suite.py
    审计并准备 12-case prediction/evaluation 隔离 bundle，验证坐标 contract 和 source-frame alignment。

regression_pose_eligibility.py
    无 pytest 依赖的三帧 synthetic P09→P14→P15 eligibility/support regression。

filter_v19_rigid_completion_multiview_support.py（位于 `scripts/`）
    以 explicit-eligible direct P14 poses 重测 hidden faces 的 mask/depth、free-space、temporal coverage 和 object-canonical viewpoint support；physical promotion 使用 exact separated-view pair，same-view repetition不晋级。

regression_geometry_evidence_contract.py
    无 pytest/NAS 依赖的 P13 pose/collision/sign split、P14b multi-view promotion 和 P16/P18 consumer-contract regression。

RESULTS_V19_V1_ZH.md
    本次 V19 完整盲运行、内部阶段审计和 partial-GT 结果。

ORIGINAL_V19_PIPELINE_ISSUES_BRIEF_ZH.md
    冻结原始 V19 的主要 bug、设计问题、数据限制和修正优先级简报。

CURRENT_OUTPUT_VS_GT_AND_FAILURE_TAXONOMY_ZH.md
    当前输出 vs available GT，以及 confirmed bug / pipeline design / observability limit 分类。

NEXT_PIPELINE_CORRECTION_AND_MULTICLIP_PLAN_ZH.md
    下一阶段问题清单、因果修正顺序、12-case suite、split policy 和 go/no-go protocol。
```

---

## 2. 数据审计结论

NAS root：

```text
/mnt/nas-106/ego4d
```

观察到：

```text
takes:                    5035
frame-aligned MP4:       45905
video footprint:          409 GB
annotations footprint:     11 GB
```

主要 annotation：

| Annotation | train | val | test |
|---|---:|---:|---:|
| camera pose JSON | 2253 | 586 | 780 |
| body pose JSON | 864 | 218 | — |
| hand pose JSON | 286 | 65 | — |
| relations takes | 1157 | 267 | 366 |

当前 shard 支持：

- Ego-Exo4D world 中的 sparse 3D hand joints；
- 官方 512×512 rectified view 中的 2D hand joints；
- world→annotation-camera extrinsics；
- raw Aria RGB grid 上的 sparse visible object masks；
- object identity、keystep 和 atomic descriptions。

当前 shard 不支持：

- object CAD / metric Mesh / dimensions；
- object per-frame 6DoF；
- MANO parameters / complete vertices；
- dense metric depth；
- metric contact points/patches/force；
- SDF / signed nonpenetration GT。

---

## 3. 选定 clip

```text
take_uid:      a89215f1-92fe-4207-9293-62ce108171da
take_name:     georgiatech_bike_07_10
split:         val
task:          Fix a Flat Tire - Replace a Bike Tube
camera stream: aria06_214-1
target track:  yellow bicycle tire lever_0
source frames: 2040 ... 2189 inclusive
local frames:  0 ... 149
fps:           30
length:        5.0 s
```

目标是单一刚性塑料撬胎棒；右手在 interval 中持续操作它，并在后段用它撬轮胎胎唇。

---

## 4. 关键 camera contract

### 4.1 Raw RGB

NAS 视频：

```text
frame_aligned_videos/downscaled/448/aria06_214-1.mp4
```

是 **raw distorted Aria RGB 的 extracted orientation**。Benchmark 仅把它从 448×448 isotropic resize 到 960×960；这不会产生新图像信息。

### 4.2 Ego-pose K 不是 raw RGB K

Ego-pose annotation 中：

```text
K = [[150, 0, 255.5],
     [0, 150, 255.5],
     [0,   0,     1]]
```

对应官方生成的：

```text
512 × 512 undistorted linear camera
```

而不是 raw 448 grid。

官方实现：

```text
facebookresearch/Ego4d
commit 4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb

ego4d/internal/human_pose/undistort_to_halo.py
ego4d/internal/human_pose/utils.py::aria_original_to_extracted
```

官方流程先将 extracted image 旋回 90°，再用 VRS device calibration 从 raw camera-rgb rectification 到 512 linear camera。

固定 camera-vector adapter：

```text
X_extracted = A @ X_annotation

A = [[0, -1, 0],
     [1,  0, 0],
     [0,  0, 1]]
```

该 adapter 来自官方坐标定义，不是从 benchmark GT 拟合。

### 4.3 本地缺口

当前 NAS 没有：

```text
takes/georgiatech_bike_07_10/aria06_noimagestreams.vrs
```

因此不能精确复现 raw→rectified pixel mapping，也不能把 rectified K 发布成 prediction-side raw RGB calibration。

---

## 5. Leakage-safe 目录

### Prediction input v2

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/
  egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_raw_distorted_v2/
├── input.mp4
└── PREDICTION_INPUT_MANIFEST.json
```

预测侧只含：

- raw distorted RGB；
- target text hint；
- camera model unavailable 的 provenance。

不含：

- rectified K；
- camera extrinsics；
- hand GT；
- object Mask GT；
- narration GT。

视频：

```text
960 × 960
150 frames
30 fps
SHA256 37de09c1193bc5c56e23a4c9ea49caa1d38623cb4f6ea5e92dc03a78d9f29ec4
```

### Evaluation-only GT v2

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189/
  ground_truth_v2_rectified_camera/
├── BENCHMARK_MANIFEST.json
├── GROUND_TRUTH_AVAILABILITY.json
├── camera_pose_gt.json
├── hand_pose_gt.json
├── object_visible_mask_gt.json
├── object_visible_masks_1408/
├── object_visible_masks_960/
└── ground_truth_preview.jpg
```

### Legacy v1 invalidation

旧 adapter 曾把 rectified 512 K/UV 错当成 raw 448 K/UV。旧目录保留用于审计，但写有：

```text
ground_truth/INVALIDATED_COORDINATE_CONTRACT.json
evaluation_v1_estimated_intrinsics/INVALIDATED.json
```

旧 hand MPJPE/reprojection 数字不得报告。旧 object Mask result 仍有效。

---

## 6. GT coverage

### Hand

```text
clip frames:                    150
annotated hand frames:           49
raw named joint rows:           938
eligible rows after filter:     916
matched prediction rows:        916
```

默认 filter：

```text
num_views_for_3d >= 2
released 3D→2D error <= 20 px on rectified 512 view
```

### Object visible Mask

```text
local frames:  0, 30, 60, 90, 120
source frames: 2040, 2070, 2100, 2130, 2160
coverage:      5 / 150
```

### Camera

```text
150 / 150 frames have world→annotation-camera extrinsics
```

---

## 7. Prepare

```bash
cd /mnt/user-home/kupingxin/ego_annotation

.venv/bin/python \
  experiments/egoexo4d_rigid_benchmark/prepare_benchmark.py \
  --input-dir \
    /mnt/truenas-user-home/kupingxin/ego_annotation_inputs/egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_raw_distorted_v2 \
  --ground-truth-dir \
    /mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189/ground_truth_v2_rectified_camera \
  --rigidity-assumption \
    "Single rigid plastic tire lever over the selected interval; no articulated parts are visible." \
  --replace
```

脚本会：

1. 截取并 resize 150 帧 raw RGB；
2. 解码 Ego-Exo4D LZ-string + compressed COCO RLE masks；
3. 保存 raw 1408 和 raw 960 masks；
4. 抽取 sparse 3D/rectified-2D hand joints；
5. 抽取 camera extrinsics；
6. 写入官方 camera-axis adapter provenance；
7. 不把 rectified K 冒充 prediction-side raw K；
8. 不把 evaluation-only relation track 写入 prediction manifest；
9. 生成 mask-only raw-view preview。

---

## 8. Evaluate

本次 run：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/
  20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_kupingxin_v1
```

命令：

```bash
cd /mnt/user-home/kupingxin/ego_annotation

.venv/bin/python \
  experiments/egoexo4d_rigid_benchmark/evaluate_benchmark.py \
  --run-root \
    /mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260806_egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189_kupingxin_v1 \
  --ground-truth-dir \
    /mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189/ground_truth_v2_rectified_camera \
  --output-dir \
    /mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189/evaluation_v2_rectified_camera_contract \
  --object-id tire_lever
```

Evaluator 同时输出：

```text
egoexo4d_partial_gt_evaluation.json
summary.json
SUPPORTED_AND_UNSUPPORTED_CLAIMS.json
object_mask_comparison.jpg
```

`SUPPORTED_AND_UNSUPPORTED_CLAIMS.json` 明确禁止将 visible Mask、narration/contact prior 或 non-watertight zero count 误写成完整物理 GT。

### Object

- sparse visible Mask IoU；
- precision/recall；
- bbox IoU；
- centroid error；
- boundary F1。

缺失 prediction mask 按 empty prediction 计分。

### Hand

分别评测：

- P04 HaWoR；
- P18 raw；
- P18b canonical。

指标：

- camera-space absolute MPJPE；
- wrist error；
- wrist-relative MPJPE；
- signed x/y/z camera-axis bias；
- rectified-512 reprojection error；
- 3D/2D PCK；
- P18/P18b 相对 HaWoR delta。

### Camera

- SE(3) metric-scale ATE；
- Sim(3) scale diagnostic；
- orientation-gauge error；
- gauge-invariant local-camera translation RPE；
- relative rotation RPE；
- position/orientation gauge conflict。

### 明确不评测

```text
object canonical geometry
object 6DoF
metric contact
nonpenetration
dense depth
```

### Coordinate-contract self-test

```bash
.venv/bin/python \
  experiments/egoexo4d_rigid_benchmark/self_test_coordinate_contract.py \
  --ground-truth-dir \
    /mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/egoexo4d_georgiatech_bike_07_10_tire_lever_f2040_2189/ground_truth_v2_rectified_camera
```

当前状态为 `pass`；exact synthetic hand/camera error 约为数值舍入量级。released rectified 2D annotation 本身相对 released 3D projection 有少量误差，因此 synthetic 3D prediction 的 2D reprojection不会强制为零。

---

## 9. 当前 headline result

```text
object Mask mean IoU:                0.6522
HaWoR absolute MPJPE:              192.956 mm
HaWoR root-relative MPJPE:          35.559 mm
P18 raw absolute improvement:        0.516 mm
camera SE3 ATE RMSE:                58.588 mm
camera 1-frame translation RPE:      7.055 mm
camera 1-frame rotation RPE:         0.345 deg
P00→P21 runtime:                   3652 s for 5 s video (730.4×)
full physical GT available:          false
```

这些数字的详细 coverage、distribution、P09–P18 runtime mechanism 和 render failure 见 [`RESULTS_V19_V1_ZH.md`](RESULTS_V19_V1_ZH.md)。

---

## 10. P09→P15 eligibility 修复回归

冻结的 V19 v1 run 不做原地修改。实现提交 `b7b97e6` 的 mechanism ablation 将 P09 的 explicit eligibility 接入 P14/P15 后得到：

```text
P14 frozen v1 fits:                    33
P14 fixed trusted fits:                 6
P14 fixed explicit-false rejected:     27
P14 fixed rows without pose/sample:   117
frozen all-row final median:        63.038 mm
fixed eligible-only final median:    2.006 mm

P15 trusted direct rows:                 6
P15 configured minimum:                  8
P15 completion rows:                   144
  nearest holds:                       141
  interpolations:                        3
P15 annotation_ready:                false
```

这不是 object-trajectory accuracy 改善声明。6 个可信 rows 只集中在 local frame `115–123`；selected-frame state render 显示早期时间轴的 nearest hold 仍明显错误。因此修复的收益是：

1. 污染观测不再进入 P14/P15；
2. 33-row aggregate residual 不再掩盖 trusted subset；
3. sparse completion 不再升格为 annotation-ready trajectory；
4. P16 将 object-relative correction rows quarantine；
5. P18 在加载 MANO model/optimizer 前生成 source-only quarantine state；
6. P18b 清空 object-relative surface samples，P19a 拒绝与新 P15 混用的 stale non-quarantined P16/P18 state；
7. P19 路径仍支持完整时长 failure artifact，但 object hypothesis 使用橙色并显式标注 `POSE UNREADY`。本次 ablation 只做 selected-frame mechanism QC，不冒充新的 full-duration V19 run。

Standalone regression 不依赖 pytest：

```bash
cd /mnt/user-home/kupingxin/ego_annotation

.venv/bin/python \
  experiments/egoexo4d_rigid_benchmark/regression_pose_eligibility.py \
  --python .venv/bin/python
```

它构造三帧 cuboid fixture：frame 0 explicit true、frame 1 explicit false、frame 2 legacy unspecified，并断言：

- P14 default 仅拟合 `0,2`；
- P14 historical override 才拟合 `0,1,2`；
- P15 对低于 minimum 的 full-timeline completion 输出 `annotation_ready=false`；
- 单个 direct observation 只能生成 explicit nearest-hold hypotheses，不会因 Slerp key 不足崩溃；
- 禁用 completion 时低支撑 hard fail；
- 即使 P14 override，P15 default 仍二次拒绝 explicit false；
- 只有 P14 和 P15 均显式 override 时才复现三行图。

Geometry evidence regression 同样不依赖 pytest、NAS 或 benchmark GT：

```bash
.venv/bin/python \
  experiments/egoexo4d_rigid_benchmark/regression_geometry_evidence_contract.py
```

它构造 partial observed plane + hidden box 和两帧 synthetic camera/depth fixture，并断言：P13 默认隔离 single-view hidden faces；历史 override 必须显式且不产生 sign readiness；pose canonical geometry 精确保留；P14 pose/completion canonical binding mismatch hard fail；depth-grid projection 优先使用 depth NPZ K；camera-ray first-hit 会排除 self-occluded generated support；相同 viewpoint 的重复 depth support 不晋级；跨 viewpoint 且 temporal coverage 足够时只有 visible hidden faces 才可晋级；exact pair test 不受 greedy-bin boundary 影响；P16/P18 选择 collision surface 而非 pose hypothesis；external sign readiness 必须绑定 exact mesh；legacy 缺 readiness 时保持 unknown/inactive。

---

## 11. `b7b97e6` isolated runtime bundle

已从提交 `b7b97e617e56a48fcfcdfe327d83fe66017f2fe5` 构建新的隔离 bundle：

```text
/mnt/user-home/kupingxin/ego_annotation_runtime/
  v19_bundle_a800_b7b97e6_pose_gate_local3/
```

Bundle manifest：

```text
status:          curated_runtime_bundle_built
source_revision: b7b97e617e56a48fcfcdfe327d83fe66017f2fe5
scripts:         27
manifest files:  97
path isolation:  ok
```

Raw-v2 prediction input 的新 preflight：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/
  runtime_preflight_egoexo4d_tire_lever_pose_gate_b7b97e6_raw_v2.json

status:        ready_for_runtime_agent_launch
failed_checks: []
input SHA256:  37de09c1193bc5c56e23a4c9ea49caa1d38623cb4f6ea5e92dc03a78d9f29ec4
```

预留的 run root 仍未创建。当前没有启动第二次 full run：输入 RGB 与冻结 v1 字节完全相同，且仍缺 `aria06_noimagestreams.vrs`，所以它不能被称为 sensor-calibrated rerun。P14–P19 的 exact-state mechanism replay 已在独立 ablation 中完成；若以后启动该 fresh root，只能标为 `b7b97e6` mechanism-ablation run，不能据此增加 camera/geometry GT claim。

Geometry-evidence 修正后的 isolated bundle：

```text
/mnt/user-home/kupingxin/ego_annotation_runtime/
  v19_bundle_a800_da87a1a_geometry_gate_local4/
source revision: da87a1a5c4b6cc2ed3d2caa3c5d832742ae3ed57
scripts/files:   28 / 98
```

其 raw-v2 preflight 为 `runtime_preflight_egoexo4d_tire_lever_geometry_gate_da87a1a_raw_v2.json`，状态 `ready_for_runtime_agent_launch`、`failed_checks=[]`；预留 `...geometry_gate_da87a1a_kupingxin_v3` run root 仍未创建。它只冻结新的 phase/CLI/provenance contract，不是新 full run。

---

## 12. 当前输出 vs available GT

新增输出：

```text
$BENCH/evaluation_current_output_vs_gt_pose_gate_b7b97e6/
├── current_output_vs_gt_evaluation.json
├── failure_taxonomy.json
├── summary.json
├── object_mask_comparison.jpg
├── projected_mesh_vs_sparse_gt.jpg
└── projected_mesh_masks/
```

Headline：

```text
SAM2 visible-mask mean IoU:           0.6522
frozen projected-Mesh mean IoU:       0.1508
fixed projected-Mesh mean IoU:        0.0651
fixed P15 trusted/min frames:          6 / 8
fixed P15 annotation_ready:            false
fixed P18b absolute MPJPE:           192.956 mm
fixed P18b exactly equals HaWoR:       true
camera SE3 ATE RMSE:                  58.588 mm
```

完整解释见 [`CURRENT_OUTPUT_VS_GT_AND_FAILURE_TAXONOMY_ZH.md`](CURRENT_OUTPUT_VS_GT_AND_FAILURE_TAXONOMY_ZH.md)。Projected-Mesh 指标同时受 estimated raw-view camera、错误 completed geometry、pose 和 visible occlusion 影响；它是 failure diagnostic，不是 object SE(3) GT。

---

## 13. 下一阶段 12-case multi-clip suite

问题、建议、因果修正顺序和完整选片依据：

- [`NEXT_PIPELINE_CORRECTION_AND_MULTICLIP_PLAN_ZH.md`](NEXT_PIPELINE_CORRECTION_AND_MULTICLIP_PLAN_ZH.md)

类别无关 metadata/partial-GT scan：

```text
train hand+camera+relations overlap takes: 258
val hand+camera+relations overlap takes:    59
train candidate track-stream windows:     1030
val candidate track-stream windows:        172
total candidates:                         1202
```

机器 scan 只要求五张非空 1 Hz visible Mask、完整 camera、raw MP4，并记录 hand coverage；它不能判断刚性或真实手物操作。Curator 随后检查 raw RGB + evaluation-only masks，并冻结：

```text
development:                         6
consumed development reference:      1
locked internal holdout:              5
unique takes/participants/captures: 12 / 12 / 12
dev↔holdout participant overlap:      0
dev↔holdout capture overlap:          0
local VRS available:                  0 / 12
runtime runs launched:                0
```

Prediction root：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/
  egoexo4d_v19_rigid_multiclip_v1/
```

Evaluation-only root：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_benchmarks/
  egoexo4d_v19_rigid_multiclip_v1/
```

Prediction case IDs 使用不含 split/role/take/target 的 opaque aliases `exo_rigid_001 ... exo_rigid_012`。每个 prediction case 恰有 `input.mp4 + PREDICTION_INPUT_MANIFEST.json`；relation track、take UID/name、Mask、hand/camera GT、narration和 rectified K 均不发布到 prediction side。12/12 视频为 960×960、150 帧、30 fps；12/12 coordinate-contract self-tests 和 endpoint source-frame alignment checks 通过。

该 suite 是下一步 P13/temporal-observation/P15–P16/P18 修正的 regression guardrail，不是新的物理成功结果。Holdout 在 revision、target hints、suite-level metrics 和 no-per-case-tuning policy 冻结前不得运行。
