# ego annotation 研究管线版本树（含 depth-order front-only mask 修正）

- 日期：2026-09-03
- 仓库：`~/ego_annotation`（`origin = DexGEM-Lab/ego_annotation`，`personal = saber-shiki/ego_annotation`）
- 范围：主仓 + 6 个 worktree 的分支谱系、每条线留下的机制、以及本轮 depth-order front-only mask 修正链
- 性质：只读整理 + 把未提交的 depth-order 研究线正式并入版本树
- 配套文档：`EGO_ANNOTATION_PIPELINE_MODULE_RELATIONS_20260902_ZH.md`（模块关系/参数/门控）、
  `docs/version_delivery_audit.md`（v1–v19 交付判决）

---

## 0. 怎么读这棵树

三件事必须分开读，否则会把"跑通了"当成"物理正确"：

1. **测量（measurement）**：depth / mask / MANO / surfel，带 source + uncertainty。
2. **优化（optimizer）**：P13 Sim(3)、P14 ICP、P15 pose graph、P18 MANO——`status: ok` 只代表求解器退出，不代表几何收敛。
3. **门控（gate / quarantine）**：证据不足时**显式拒绝**，这是设计意图而非 bug。

本文所有数值都来自已冻结的报告；报告没记录的量一律标"未记录"。

分支命名前缀的语义：

| 前缀 | 含义 |
|---|---|
| `research/` | 机制探索，允许留下否定结论 |
| `feature/` | 新后端 / 新阶段 |
| `fix/` `repair/` | 修正已发布链路上的合同缺陷 |
| `release/` | 对外可复现的冻结点 |
| `archive/` | 冻结当时的工作状态，供回溯 |
| `local/` | 机器本地部署，不可移植 |

---

## 1. 版本树总览

```
yiwen_research 0c8e6a9  (grafted 基线, origin)
│
└─ b18cecd  deploy: isolated local V19 A800 runtime
   │        └── local/kupingxin-v19-a800-deployment (c8940c9)
   │
   ├─ ca254f7  audit V19 optimization + Ego-Exo4D benchmark
   ├─ b7b97e6  gate rigid poses / quarantine sparse support      ← 证据门控起点
   ├─ a6b5a71  gate hidden geometry with multiview evidence
   ├─ …  research/v19-multiclip-pipeline-corrections (64dc656)
   ├─ 7d5720e  freeze SAM3D P11–P15 geometry experiment
   ├─ d92fe1e  research/sam3d-p11-p15-keyboard-freeze-20260811   ← 分叉点 A
   │  │
   │  ├─ da926ec  archive/pre-da3-working-state-20260824  ★ 当前 HEAD + runtime bundle 来源
   │  │            (+5472/−294：HOT3D VRS adapter、oracle depth、depth-source A/B、
   │  │             triangle-visibility prior、P03c official-K adapter)
   │  │
   │  └─ 905aab4  research/v19-metric-camera-contract-visible-geometry
   │     │
   │     ├─ d65e19c … 6d5471a  release/hot3d-sam3d-trellis-dual-backend-20260817 (origin+personal)
   │     └─ 468c503 … e26b507  release 分支 tip                  ← 分叉点 B（四条线共同基座）
   │        │
   │        ├─ d3925d6  repair/hot3d-upstream-mask-depth-trace-20260818   (+1338)
   │        │
   │        ├─ 88a3604 → be8dae3  feature/hot3d-shared-p18-reintegration-20260818
   │        │  └─ 496c84e → 7930280  feature/hot3d-milk-shared-signed-geometry-20260818
   │        │     └─ 3513be2 → 5a7c195  fix/hot3d-local-signed-authority-p18-support-20260819
   │        │        │
   │        │        └─ 4947d5b … 36e87ec  research/sam3d-native-ghost-lite-alignment-20260826
   │        │           │        （SAM3D native pose 合同 → first-hit Sim(3) → 跨视频 A/B 否决）
   │        │           │
   │        │           └─ ★ 本轮新增：depth-order front-only mask 修正链 S1–S4
   │        │
   │        └─ 2231512 … a2eb282  feature/hot3d-da3-pose-conditioned-depth-20260824  (DA3 后端，negative result)
```

`archive/pre-da3-working-state-20260824` 与 `release/…-20260817` **不是祖先关系**：
两者在 `d92fe1e` 之后各走一边（archive 领先 22 / 落后 1）。
运行时 bundle 取自 archive 侧，而 HOT3D/SAM3D 研究都在 release 侧——
这就是"当前 checkout ≠ 冻结运行行为"的结构性原因。

### 1.1 worktree ↔ 分支映射

| worktree | 分支 | HEAD | 工作区状态 |
|---|---|---|---|
| `ego_annotation`（主） | `archive/pre-da3-working-state-20260824` | `da926ec` | 脏：`filter_v19_rigid_completion_multiview_support.py`、未跟踪 `render_p04_handedness_overlay.py` |
| `sam3d_native_ghost_lite` | `research/sam3d-native-ghost-lite-alignment-20260826` | `36e87ec` | ★ depth-order 研究线（本轮提交） |
| `sam3d_native_ghost_lite_clean_20260903` | detached `36e87ec` | — | 干净对照，用于确认脏改动的归属 |
| `hot3d_da3_pose_conditioned` | `feature/hot3d-da3-pose-conditioned-depth-20260824` | `a2eb282` | 干净 |
| `hot3d_backend_suite_v1` | `fix/hot3d-local-signed-authority-p18-support-20260819` | `5a7c195` | 干净 |
| `hot3d_upstream_trace_finalize_v1` | `repair/hot3d-upstream-mask-depth-trace-20260818` | `d3925d6` | 干净 |
| `v19_metric_camera_contract` | `research/v19-metric-camera-contract-visible-geometry` | `905aab4` | 干净 |

### 1.2 远端同步状态

`personal` 已有 8 个分支。**未推送**的是：

- `research/sam3d-native-ghost-lite-alignment-20260826`（本轮推送）
- `research/v19-multiclip-pipeline-corrections`
- `feature/hot3d-shared-p18-reintegration-20260818`、`feature/hot3d-milk-shared-signed-geometry-20260818`（中间态，已被 `5a7c195` 包含）
- `local/kupingxin-v19-a800-deployment`（机器本地，不应推送）

---

## 2. 每条分支留下了什么机制

| 分支 | tip | 机制贡献 | 判决 |
|---|---|---|---|
| `local/kupingxin-v19-a800-deployment` | `c8940c9` | 隔离 A800 runtime bundle + preflight（manifest hash / prompt 隔离 / CLI self-test） | 保留为运行载体 |
| `research/v19-multiclip-pipeline-corrections` | `64dc656` | rigid pose eligibility 硬门、稀疏支撑 quarantine、隐藏几何 multiview 门控 | **已成为主线不变量** |
| `research/sam3d-p11-p15-keyboard-freeze-20260811` | `d92fe1e` | 冻结 keyboard run；暴露 P15 `nfev=1, cost=0` 退化与 P18 栅格 bug | 诊断基线 |
| `research/v19-metric-camera-contract-visible-geometry` | `905aab4` | source-neutral 相机几何合同（K 链 P03b→P03c→P04→P08） | 已成为合同 |
| `release/hot3d-sam3d-trellis-dual-backend-20260817` | `e26b507` | SAM3D vs TRELLIS 公平 A/B（唯一分叉变量 = generated prior）、P11 anchor 原子绑定 | 冻结发布点（3:2 SAM3D 胜） |
| `repair/hot3d-upstream-mask-depth-trace-20260818` | `d3925d6` | 上游 mask/depth/pose 追溯器（+1338 行） | 诊断工具 |
| `fix/hot3d-local-signed-authority-p18-support-20260819` | `5a7c195` | 逐面 local signed authority（schema v1→v2）+ P18 support gating | **fail-closed 生效**：milk 30,002 面全信 → 5,605 面 observed；平移静默撤销 300/300 → 0 行 |
| `feature/hot3d-da3-pose-conditioned-depth-20260824` | `a2eb282` | DA3 pose-conditioned depth 后端；phase 改为 P03b→P04→P03d→P03c | **negative result**：overlap 相对中位 0.622 / 单帧 2.081，超 fail-closed 门（0.10/0.25） |
| `archive/pre-da3-working-state-20260824` | `da926ec` | HOT3D VRS slice adapter、oracle object depth、depth-source A/B、triangle-visibility gated prior | runtime bundle 来源 |
| `research/sam3d-native-ghost-lite-alignment-20260826` | `36e87ec` + S1–S4 | 见 §3、§4 | 主线研究前沿 |

---

## 3. SAM3D × GHOST-lite 线（`36e87ec` 之前的已提交部分）

| commit | 内容 | 关键数字 |
|---|---|---|
| `4947d5b` | SAM3D native pose 合同审计 | projection IoU：raw local `0.041` / pytorch3d `0.006` / **native OpenCV `0.702`** |
| `1dd2d17` | first-hit Sim(3) 取代 depth-centred median scale | frame 92 前表面残差 `−25.7 mm → ≈0`；scale `0.41112`（被否 `0.3826`）。测量面从"穿过 mesh 厚度 48.5%"变为"距前边界 −5.7%" |
| `6ad3e11` | 导出 object-hand-surface GLB / PLY | frame 92 联合场景 |
| `e50ad57` | 全片修正物体视频 | 150 帧 |
| `c926e81` | source K → full-video review 平面映射 | 修正前 centroid median `349 px` → `42.4 px` |
| `36e87ec` | **冻结跨视频 A/B 否决 shared Sim(3)** | can：proxy 预测 `0.682→0.702` 改善，全拓扑中位 IoU 反而下降、`105/150` 帧变差；proxy vs 交付 delta 中位 `+0.011` vs `−0.024`、`5/16` 帧翻号（相关性 0.66） |

`36e87ec` 的结论直接指向下一轮：**必须做每帧 SE(3) + 图像因子，而不是一个全局 7-DoF 校正。**
这正是 §4 的起点。

---

## 4. ★ 本轮并入：depth-order front-only mask 修正链

牛奶盒案例 `P0014_84ea2dcc_carton_milk_f2370_2519`（150 帧，anchor 92），
运行 `20260819T122452Z_hot3d_milk_local_authority_local29_v1`。

```
36e87ec  (shared Sim(3) 被否决)
   │
   ├─ S0  三实验排除 depth 假设            diagnose_mask_depth_alignment_three_experiments.py
   │       └ 结论：不是 depth 栅格/数值问题
   │
   ├─ S1  hand ownership depth-order 诊断  diagnose_hand_depth_order_ownership.py
   │       └ 结论：78.29% 被删像素其实是「手在物体后方」
   │
   ├─ S2  front-only 反事实 mask           build_depth_order_counterfactual_masks.py
   │       │                               evaluate_depth_order_counterfactual_visible_geometry.py
   │       │                               compare_depth_order_ghost_alignment.py
   │       │                               compare_depth_order_full_video_ab.py
   │       └ 结论：仅换评价 mask，IoU 0.574 → 0.785
   │
   ├─ S3  重建 P14/P15 + GHOST 重跑        build_depth_order_counterfactual_pose_reference.py
   │       └ 结论：修尾不修中；centroid max 38.6 → 27.0 px
   │
   └─ S4  P15 接入 first-hit/silhouette    build_p15_first_hit_silhouette_factors.py
           图像因子                         compare_p15_image_term_pose_shifts.py
           + scripts/solve_v19_rigid_object_pose_graph.py (+294)
           └ 结论：P15 不再退化；full-video IoU 0.7912、centroid max 19.31 px
```

辅助：`diagnose_temporal_lag_mesh_mask.py` 排除"时序滞后"假设。

### S0 — 先排除 depth 假设

`diagnose_mask_depth_alignment_three_experiments.py`（1107 行，CPU-only、prediction-read-only）：

1. 同帧 `depth → 3D → pixel` 闭环误差 **0 px**；`camera→world→camera` 误差 `9.34e-08 m`。
2. 不经 SAM3D 时，anchor Poisson 观测表面对 **raw SAM2 mask** IoU `0.8738` / centroid `3.09 px`；
   对 **object-owned mask** 却是 IoU `0.7680` / centroid `21.07 px`。
3. mm 级轴向 depth residual **不可能**产生几十 px 的横向错位；错误 K 曾产生 `349 px` 固定错位，但已修复。

→ 差异来源被锁定在 raw mask 与 owned mask 之间，也就是 **hand ownership subtraction**。

### S1 — 根因：ownership 没有前后关系判断

当前 V19 实现等价于：

```text
object_owned_mask = raw_SAM2_mask − dilate(projected_full_MANO_silhouette, 4 px)
```

即**无条件**删除所有 MANO 投影，不判断手是否真的在物体前面。

`diagnose_hand_depth_order_ownership.py` 逐像素栅格化每只手的 first-hit 深度并与同像素物体深度比较
（判据 `±5 mm`），146 帧全量诊断，ownership 复现 `146/146` exact：

| 被删像素类别 | 数量 | 占比 |
|---|---:|---:|
| 手在物体**后方**（不该删） | 1,451,344 | **78.29%** |
| 手在物体前方（应删） | 12,359 | 0.67% |
| 近接触 / 不确定 | 5,661 | 0.31% |
| 无 MANO first-hit（padding/raster） | 384,362 | 20.73% |

raw mask 被删比例中位数 `29.2%`，尾帧接近 `50%`。
独立 MANO vertex probe（不经 CPU triangle rasterizer）交叉验证：
depth 比 MANO vertex 更近的行中位比例 `70.76%`，`depth_z − hand_z` 中位 `−28.42 mm`。

### S2 — front-only 反事实 mask：评价目标修正

`build_depth_order_counterfactual_masks.py` 产出两条 SAM2-track 兼容流：

- **`front_only`**：只删 `hand_z < object_z − 5 mm`；
- **`front_plus_ambiguous`**：额外删近接触/不确定像素。

behind-object、padding-only、no-hit 一律**保留**为物体支撑；缺少 visible-geometry 行的帧原样复制并标记 unprocessed。

结果（`front_only`）：retained raw fraction 中位 `0.99923`，每帧中位恢复 `13,630.5 px`、p95 `20,980 px`。

同一 depth/K/相机轨迹/anchor 重建 visible geometry：

- counterfactual robust depth retained fraction 中位 `0.98354`；
- baseline surfels 落入 counterfactual mask 比例中位 `1.0`；
- counterfactual surfels 落入 baseline owned mask 比例中位 `0.7002`（说明恢复区确实来自被误扣的支撑）；
- rigid-eligible frames 两组共同行上均 `144/144`。

**关键结果（同一个旧 GHOST mesh，未重优化，只换评价 mask）**：

| mesh | target mask | IoU median | centroid median | boundary median |
|---|---|---:|---:|---:|
| baseline GHOST | baseline owned | 0.5741 | 43.29 px | 16.98 px |
| baseline GHOST | **depth-order front-only** | **0.7854** | **9.30 px** | **9.99 px** |

即：**大部分"表观错位"是评价目标被污染，不是几何错误。旧评价显著低估了 GHOST。**

在 corrected mask 上重跑 GHOST 只有微小收益（IoU `+0.00365` 中位、131/150 帧改善），
但 centroid 反而差 `+2.15 px` —— 不能宣称几何收敛。

### S3 — 重建 P14/P15：修尾不修中

`build_depth_order_counterfactual_pose_reference.py` 把 front-only visible geometry 绑到与 baseline 相同的
P14/P15 输入合同（observed-only，生成面不进 pose body），再跑原有 P14 surfel-chain fit 与 P15 pose graph。

新旧 P15 轨迹差异：translation 中位 `19.37 mm` / p95 `32.27 mm` / max `40.52 mm`；
rotation 中位 `2.75°` / max `12.09°`；anchor 92 仅 `10.37 mm`、旋转 0；尾帧 146/149 约 `40 mm` / `10–11°`。

→ mask 修复**不只是改评价**，它确实改变了时序位姿证据链。

Full-video A/B（同一 corrected mask，仅轨迹不同）：

| branch | IoU median | centroid median | centroid max |
|---|---:|---:|---:|
| 旧 mesh / 旧 P15 | 0.7854 | 9.30 px | 38.56 px |
| 重跑 GHOST / rebuilt P15 | 0.7847 | 14.14 px | **27.01 px** |

分窗口——典型的"尾部修复、前中段轻微退化"：

| frames | ΔIoU median | Δcentroid median | improved |
|---|---:|---:|---|
| 0–83 | −0.0022 | +8.39 px | IoU 37/84 |
| 84–127 | −0.0038 | +1.04 px | IoU 13/44 |
| 128–149 | **+0.0547** | **−9.22 px** | IoU 18/22 |
| 132–149 | **+0.0736** | **−10.75 px** | IoU 18/18 |

frame 146：IoU `0.694→0.816`、centroid `38.3→7.48 px`；frame 149：`0.696→0.795`、`38.6→9.58 px`。

### S4 — P15 接入 first-hit / silhouette 图像因子

这是对 §3 `36e87ec` 结论的直接回应，也是对 P15 `nfev=1, cost=0` 退化的修复。

**两层结构**（不是逐帧可微渲染，也不是把 SAM3D 生成面塞进 P15）：

1. `build_p15_first_hit_silhouette_factors.py`：用 **observed-only** pose-hypothesis mesh 在低分辨率栅格渲染一次，
   冻结三类局部对应；
2. `scripts/solve_v19_rigid_object_pose_graph.py`（+294 行）：把冻结的 canonical 点在当前逐帧 SE(3) 下重投影，作为 graph residual。

三类 factor：

- `first-hit depth`：observed surfel 像素上的 first-hit canonical 点 → camera z vs observed depth；
- `rendered → silhouette`：渲染到 known background 的像素 → 最近 target/unknown 像素的 2D 残差；
- `observed → rendered`：无 mesh 覆盖的 observed surfel → 最近 rendered canonical 点的 2D 覆盖残差。

**手投影区域是 explicit unknown support**，渲染到手区域不会被当成 known background。

新增 CLI 与默认值：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--image-factor-npz` | `None` | 不传则**完全保持旧行为**（`nfev=1, cost=0`） |
| `--image-first-hit-weight` / `--image-silhouette-weight` | 1.0 / 1.0 | 两类图像项权重 |
| `--sigma-image-first-hit-m` | 0.008 | 深度残差尺度 |
| `--sigma-image-silhouette-px` | 4.0 | 轮廓残差尺度 |
| `--max-image-first-hit-residual-m` | 0.030 | 截断 |
| `--max-image-silhouette-residual-px` | 16.0 | 截断 |

**安全边界**：factor NPZ 与 `annotations` / `pose_report` / `completed_mesh` / `object_id` 做逐项路径绑定，
不匹配直接 `RuntimeError`；SAM3D 生成隐藏面不进 P15；collision/contact/sign authority 不变。

iter2 factor：146 帧、first-hit depth factors `25,080`、silhouette factors `20,874`、
frozen-pose silhouette IoU 中位 `0.7469`、observed first-hit coverage 中位 `0.9401`。

P15 结果：`status = corrected_pose_graph_surface_preserved_image_factors_improved`，
optimizer **success `nfev=10`**（不再退化），graph RMS `0.07708 → 0.07081`，
first-hit depth 中位 `−3.84 → −2.65 mm`，outside silhouette 中位 `3.47 → 2.17 px`，
observed-surface 退化 `0.249 mm`（低于 3 mm 门）。
相对无图像项的 front-only P15：translation 修正中位 `3.52 mm` / max `9.16 mm`，rotation 中位 `0.38°` / max `4.45°`。

GHOST 采样帧：

| branch | IoU median | coverage median | depth median |
|---|---:|---:|---:|
| corrected mask + 旧 P15 | 0.7460 | 0.848 | −1.24 mm |
| corrected mask + rebuilt P15 | 0.7417 | 0.861 | −1.89 mm |
| corrected mask + **image-factor P15** | **0.7535** | **0.865** | **−1.49 mm** |

全分辨率 first-hit audit：abs median mean `3.64 → 2.70 mm`，coverage median `0.806 → 0.810`。

Full-video：

| branch | IoU median | centroid median | centroid max |
|---|---:|---:|---:|
| 旧 GHOST + corrected mask | 0.7854 | **9.30 px** | 38.56 px |
| rebuilt P15 GHOST | 0.7847 | 14.14 px | 27.01 px |
| **image-factor P15 GHOST** | **0.7912** | 11.57 px | **19.31 px** |

分窗口仍有未闭合区：`0–83` 改善（centroid 82/84）、`128–149` 改善（IoU 15/22）、
但 **`84–127` 退化**（IoU 2/44、centroid 0/44）。

### S5 — 时序滞后假设被排除

`diagnose_temporal_lag_mesh_mask.py`：mesh/mask 的最佳整数 lag 在三种配置下**均为 `lag = 0`**
（旧 GHOST/旧 P15、image-factor P15、P14 低平滑）。
P14 平移时序正则化的 `tau` 仅 `0.0091 s`（`0.27` 帧），降低平滑后进一步降到 `0.0047 s`。

→ 残余错位**不是**时序滞后，是逐帧 pose / canonical shape 误差。

### 本轮最终归因

1. 不是 depth raster、也不是 depth 数值 —— 闭环为 0 px。
2. **主因：hand ownership subtraction 缺少 depth-order 判断**，78.29% 的删除像素其实是后方手。
3. 修正 mask 后，观测表面与 mask 立即恢复到 raw-mask 水平（IoU `0.574 → 0.785`）。
4. 修正后的 visible support 会改变 P14/P15 轨迹，新轨迹显著修复视频后段漂移。
5. 图像因子让 P15 从退化问题变成真正可优化的问题，并在 full-video 上改善 IoU 与 centroid max。
6. 但 `84–127` 窗口仍退化，且 GHOST 优化仍是 `optimizer_incomplete` —— **尚未闭合**。

### 建议的实现修复（尚未并入主线）

1. 把 P09 的 hand ownership 从"投影 silhouette 无条件删除"改为 **per-pixel depth-order ownership**：
   `hand_z < object_z − tol` 才 hand-owned；`hand_z > object_z + tol` 保留 object support；中间标 ambiguous。
2. 4 px padding 只能作为 review/uncertainty band，**不能删除** object mask。
3. 用 depth-order mask 重新生成 P14/P15 的 completion binding 与 pose trajectory。
4. 对跨视频 spatula / bottle / can / mug 复跑同一诊断——**牛奶盒结论不可直接泛化**（can 已在 §3 出现退化）。
5. 下一步若要修 `84–127`：逐帧 residual confidence gating、单独审计该窗口的 mask/hand-unknown/depth 冲突、
   或把 differentiable first-hit renderer 接入局部 Gauss-Newton，而不是只冻结一次对应关系。

---

## 5. 跨版本的不变量（每条线都必须遵守）

1. **生成面永不获得物理权威**：`pose / contact / collision / signed_distance_eligible = false`。
   P13 之后 mesh 分三种身份：`pose_hypothesis_mesh`（只做 pose/渲染）、
   `collision_eligible_mesh`（只含 measured observed 面）、`geometry_readiness`（独立判定）。
2. **四个 readiness 变量不可混用**：P09 `rigid_pose_observation_eligible`、
   P15 `graph_support.sufficient`、P13/P14b `signed_geometry_ready`、顶层 `annotation_ready`。
3. **手关节权威永远是 P04 HaWoR metric MANO**；P18 跑过 ≠ P18 结果被接受（P18b 强制 joint shift = 0）。
4. **栅格合同必须显式**：source（1920×1456 或 1408×1408）/ manifest（960）/ mask（960）三套并存，
   K 只能经显式仿射跨栅格；主分支 P18 的 1408↔960 缺口未修前，depth-order 与 ownership quarantine 都是无效项。
5. **A/B 只允许一个分叉变量**，其余上游全部共享同一 sha256 绑定产物。
6. **proxy 不等于交付审计**：can 案例已证明代理目标可以预测改善而交付指标退化（相关性 0.66、5/16 帧翻号）。

---

## 6. 未闭合问题（按优先级）

| # | 问题 | 状态 | 证据 |
|---|---|---|---|
| 1 | P15 默认接线退化（`nfev=1, cost=0`） | **本轮 S4 已给出机制解**，但仅牛奶盒验证 | keyboard / fill_kettle 均复现退化 |
| 2 | hand ownership 缺 depth-order | **本轮已定位并验证反事实**，未并入 P09 主线 | 78.29% 后方手被误删 |
| 3 | P18 栅格合同 1408↔960 | sam3d worktree 已用显式仿射修好，未回合主线 | keyboard run `selected_depth_order_vertex_count = 0` |
| 4 | `signed_geometry_ready` 几乎恒 false | 设计上的正确拒绝，但根因未解 | fill_kettle 10,279 面重复 free-space 矛盾 |
| 5 | full-video `84–127` 窗口退化 | 新发现，未解释 | image-factor P15：IoU 2/44、centroid 0/44 |
| 6 | GHOST 仍 `optimizer_incomplete` | 达 nfev budget，非平滑收敛 | 所有 GHOST run |
| 7 | 跨视频泛化未验证 | 只做了牛奶盒 | can 在 shared Sim(3) 下已退化 |

---

## 7. 产物根目录

```
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/
  hot3d_pinhole_rgbd_selection_v1/backend_tests/
    20260819T122452Z_hot3d_milk_local_authority_local29_v1/runs/
      P0014_84ea2dcc_carton_milk_f2370_2519/experiments/sam3d_native_ghost_lite_20260826/
        mask_depth_alignment_three_experiments_v1/    ← S0
        hand_depth_order_ownership_audit_v1/          ← S1
        depth_order_counterfactual_masks_v1/          ← S2/S3/S4
          front_only/  front_plus_ambiguous/
          p15_front_only_pose_graph/                  ← S3
          p15_image_terms/                            ← S4 (iter1/iter2 + factor NPZ)
          ghost_front_only_*_p15_first_hit_depth_authority_v1/
          temporal_lag_audit/                         ← S5
          DEPTH_ORDER_COUNTERFACTUAL_FINAL_ZH.md
          DEPTH_ORDER_REPOSED_P15_FINAL_ZH.md
          P15_IMAGE_FACTORS_FINAL_ZH.md
        p15_first_hit_depth_authority_bounded_v4/     ← §3 推荐配置
  sam3d_native_ghost_lite_cross_video_benchmark_20260828/   ← §3 跨视频 A/B
```

计算环境：A800，factor/GHOST/full-video 用 GPU5，P14/P15/诊断为 CPU。
脚本环境：`sam3d-objects` conda env（pytorch3d + trimesh + open3d）；渲染用 `ego_annotation/.venv/bin/python`。
