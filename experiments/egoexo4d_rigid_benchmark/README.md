# Ego-Exo4D rigid hand-object partial benchmark

本目录用于：

1. 审计 `/mnt/nas-106/ego4d` 中实际存在的 Ego-Exo4D v2 真值；
2. 构造 prediction/evaluation 物理隔离的刚体手物 clip；
3. 对 V19 的目标可见 Mask、named hand joints 和 camera trajectory 做 partial-GT 评测；
4. 对缺少 GT 的 object geometry/6DoF/contact/nonpenetration 明确输出 `not_evaluated`。

完整运行结果见：

- [`RESULTS_V19_V1_ZH.md`](RESULTS_V19_V1_ZH.md)

---

## 1. 文件

```text
prepare_benchmark.py
    构造 raw Aria RGB prediction bundle 和独立 evaluation-only GT。

evaluate_benchmark.py
    评测 sparse visible Mask、3D/rectified-2D hand joints 和 camera trajectory。

self_test_coordinate_contract.py
    用 exact synthetic 3D round trip 验证固定 camera-axis adapter 和 evaluator 数学实现。

RESULTS_V19_V1_ZH.md
    本次 V19 完整盲运行、内部阶段审计和 partial-GT 结果。
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
8. 生成 mask-only raw-view preview。

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
