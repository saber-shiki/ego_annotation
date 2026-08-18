# HOT3D SAM3D / TRELLIS 公平双后端标注管线

## 1. 文档定位

本文档整理经过多轮根因分析、代码修复、fresh prediction run 和视频视觉检查后，能够完成 HOT3D 针孔 RGB 视频双后端标注任务的当前管线。

发布分支：

```text
release/hot3d-sam3d-trellis-dual-backend-20260817
```

本分支的代码目标是：

1. 从原始针孔 RGB 视频和 prediction-side sensor calibration 开始执行；
2. 预测完整时长的 metric MANO、object mask、observed metric surface 和 object trajectory；
3. 公平比较 SAM3D Objects 与 TRELLIS 的单图生成几何；
4. 保证两后端只在 generated render geometry 上分叉；
5. 生成每后端四个完整时长视频、mesh/state/report 和最终 manifest；
6. 对多 case 结果执行逐视频解码、SHA256 和共享状态审计。

> 本文档描述 prediction 与 post-prediction aggregation。模型权重、数据集视频、运行输出、私有路径和大型 mesh/video 不提交到 Git。

---

## 2. 已验证范围与版本说明

### 2.1 代码提交链

相对于源中已有的 metric-camera / SAM3D P11–P15 基线 `905aab48`，本管线包含以下主要提交：

| Commit | 作用 |
|---|---|
| `d65e19c` | 新增 HOT3D five-case SAM3D/TRELLIS runtime suite |
| `4c5fc26` | 保留相机内参精度，避免 float32 静默改写 |
| `fb325df` | 恢复并 preflight 冻结的 SAM3D launch contract |
| `f83b20d` | 修复几何尺度、native pose bridge 和 readiness |
| `241f92c` | 将更新后的 TRELLIS runner 纳入 runtime bundle |
| `4c5af20` | 对稀疏 UniDepth interior holes 做显式 quarantine |
| `354e008` | observed-only pose、atomic evidence 和 SAM3D offline contract |
| `0dc49f2` | immutable bundle self-test closure |
| `9ca7851` | TRELLIS DINOv2 离线 source/checkpoint fail-closed binding |
| `8625ad1` | 修复 preflight path-isolation self-match |
| `39a780e` | anchor/front-quality 修复与唯一正式 D18 runner |
| `a92d227` | 支持从 read-only immutable base 构建新 bundle |
| `6ce5fc6` | bounded sparse underobservable rotation-tail uncertainty |
| `6fc458e` | 五例 post-prediction collection finalizer |
| `ec64364` | CIFS-safe collection links 与 inclusive `0.20` audit policy |

### 2.2 生产验证版本

最终 Spatula fresh prediction 使用的 immutable runtime bundle：

```text
v19_bundle_a800_6ce5fc6_hot3d_observed_pose_local22
```

其 `RUNTIME_BUNDLE_MANIFEST.json` SHA256：

```text
5229a5c33dd2c898866c847d50a180f7c8d454fe7a6e47292b32935a99c5f377
```

该 bundle 的 prediction source revision 是 `6ce5fc6`。后续 `6fc458e` 和 `ec64364` 只增加/修复 post-prediction 五例聚合，不改变已经冻结的 Spatula prediction artifacts。

历史五例结果是迭代修复过程中分别由 local20、local21、local22 等 immutable bundle 产生的，并非五例全部在同一个最终 bundle 下重新计算：

| Case | 有效 D19 来源 |
|---|---|
| milk | 独立 milk D19 suite |
| soup | local20 fresh suite |
| mug | local21 fresh recovery suite |
| BBQ | local21 fresh recovery suite |
| spatula | local22 fresh suite |

因此：

- 这些结果证明各 case 的完整任务已经成功完成并经过统一 post-prediction 审计；
- 若实验要求“五例严格使用同一个 source revision 重新预测”，应从本发布分支构建一个新的 immutable bundle，并为五例创建五个全新的 run root；
- 不应复制历史 prediction artifacts 到新 run root。

---

## 3. 核心公平性合同

每个 case 的两后端共享：

- 同一输入 RGB 视频；
- 同一 prediction-side sensor camera contract；
- 同一 sensor-conditioned UniDepth；
- 同一 HaWoR/MANO prediction；
- 同一 OWLv2 → SAM2 object track；
- 同一 MANO-subtracted object-owned mask；
- 同一 P09 selected anchor；
- 同一 P11 atomic evidence；
- 同一 observed metric surface；
- 同一 observed-only object trajectory；
- 同一 unsigned hand-object state；
- 同一 renderer、camera、MANO 和视频输出合同。

唯一允许的 branch variable 是：

```text
single-image generated render geometry prior and its integration
```

原生 conditioning 按各模型官方接口保留：

- **SAM3D Objects**：完整 scene RGB + object-owned binary mask，`pointmap=None`；
- **TRELLIS**：由同一 object-owned mask 构造的 object-isolated RGBA crop。

这不是给两个模型强行输入相同 tensor，而是在共享证据下使用各自原生、公开的 conditioning contract。

### 3.1 Generated geometry 的硬限制

所有 generated faces 均为 render-only：

```text
generated_faces_pose_eligible       = false
generated_faces_collision_eligible  = false
generated_faces_contact_eligible    = false
```

它们不得提供：

- object pose correspondence；
- contact evidence；
- collision evidence；
- signed distance；
- penetration / nonpenetration 结论。

物理上唯一可用的 object surface 是 prediction-side observed metric surface；由于它是 partial、unsigned、通常 non-watertight，signed contact 和 nonpenetration 仍然 unresolved。

---

## 4. 数据来源与禁止项

### 4.1 Prediction 允许使用

- 原始 RGB 视频；
- prediction-side sensor calibration contract 中的官方 K；
- 离线固定模型和标准 MANO template/model assets；
- semantic target hint，仅用于帮助 runtime agent 确认目标身份。

### 4.2 Prediction 禁止使用

- HOT3D GT object pose；
- HOT3D GT MANO；
- GT foreground depth；
- CAD model；
- GT object mask；
- sibling run 的 mask、pose、surface、mesh、anchor 或 diagnostic artifacts；
- evaluator targets 或 post-hoc reference labels。

Final manifest 必须满足：

```json
{
  "released_reference_labels_consumed_by_prediction": false
}
```

### 4.3 本项目 MANO 的来源

本管线中的每帧 MANO pose/shape/vertices 是 **HaWoR 从 RGB 视频预测得到的**，不是 HOT3D 数据集自带 GT MANO。

P04 实际执行：

1. hand detection / tracking；
2. HaWoR MANO regression；
3. DROID-SLAM camera motion；
4. Metric3D scale estimation；
5. temporal infiller；
6. 导出 metric world-space MANO NPZ。

标准 MANO 文件仅提供模型参数化和拓扑。相机 K 来自 prediction-side sensor contract；两者都不等于使用 GT hand pose。

输出示例：

```text
{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz
{RUN_ROOT}/measurements/hand_candidates/hawor_world/qc_hawor_world_hands.json
```

HaWoR 直接检测缺失但被 temporal infiller 补全的帧应视为低置信连续性假设，不能冒充同帧直接观测。

---

## 5. 完整数据流

```text
P00  startup / input isolation
P01  raw frame manifest
P02  base annotation skeleton
P03b resolve prediction-side sensor camera contract
P03  sensor-conditioned UniDepth
P03c depth-camera ray binding validation
P04  active source-K → centered HaWoR inference plane → MANO/masked SLAM prediction
P05  raw visual inspection
P06  OWLv2 target detection
P07  SAM2 full-timeline segmentation + visual inspection
P08  object plan / mask support
P09  owned RGB/depth visible geometry + fresh anchor review
P10  evidence preparation
P11  atomic shared evidence bundle
D11  native dual conditioning materialization
D12  SAM3D and TRELLIS raw generated priors
D13  controlled metric adaptation + SAM3D dual mesh
D14  observed-only completion reference and direct pose fit
D15  shared observed-only 150-frame pose graph
D16  unsigned MANO/object state
D16b shared P17 ownership/contact prior + P18 MANO candidate + P18b metric-MANO split
D17  shared P18b source state + geometry-only branch states
D18  exact dual-backend full-duration renders
D19  stable per-case publication and SUITE_DONE
POST  optional multi-case collection finalization
```

`runtime/hot3d_dual_backend_runtime_spec.md` 是 D11–D19 的权威命令文档；`runtime/v19_runtime_spec.md` 仅为 common P00–P11 提供权威流程。

---

## 6. 关键修改内容

### 6.0 Shared P18 安全回接（本分支 fresh-run 合同）

本分支在 D15 与 D17 之间恢复一次共享的 P17→P18→P18b：

```text
D15 observed-only pose authority
  -> D16 unsigned measurement
  -> shared P17 visual interaction/ownership prior
  -> shared P18 unsigned MANO candidate (object translation frozen)
  -> shared P18b metric-MANO-preserved uncertain surface samples
  -> D17 geometry-only SAM3D/TRELLIS split
```

安全约束：

- P04 必须用显式 affine 将 active source K 绑定到 centered HaWoR inference plane，并用逆 affine回绑 source pixels；历史 center-K archive fail closed；
- P17/P18 mask 使用 P09 exact source→mask affine，depth lookup 仍在 source plane；
- P18 只消费 observed-only physical surface，SAM3D/TRELLIS generated faces 不进入求解；
- D15 是唯一 object pose authority，P18 `optimize_object_translation=false` 且所有 delta 为零；
- P18b 不覆盖 metric MANO joints/root，只保留黄色 uncertain surface samples；
- D18 必须实际显示这些 samples，D19 必须发布其 provenance；
- observed surface 仍为 partial/unsigned/non-watertight，不能宣称 signed contact/nonpenetration。

完整设计见 `docs/hot3d_shared_p18_reintegration_design_zh.md`。

### 6.1 Sensor-first camera / depth contract

旧问题包括：先生成 depth，再仅替换 K metadata；或把 UniDepth camera-head K 当作 active geometry K。这会造成 byte-identical depth raster 被错误声称已适配新相机。

当前实现：

1. P03 前必须先执行 P03b；
2. 将 exact sensor K/rays 传入 UniDepth；
3. UniDepth camera-head K 只作诊断；
4. 将 metric radius 显式投影到 exact output-contract rays；
5. 输出 camera-z meters；
6. P03c 验证 depth、K 和 rays 的 byte/hash binding。

关键代码：

- `scripts/run_unidepth_full_frame_v3.py`
- `scripts/run_unidepth_metric_source_v3.py`
- `scripts/adapt_v19_depth_to_camera_contract.py`
- `experiments/v19_metric_camera_contract/self_test.py`

相机内参不再被不必要地降为 float32，官方 K 的精度保持到输出合同。

### 6.2 MANO silhouette ownership 替代 hand bbox ownership

旧实现曾允许 hand bbox 直接拥有像素，容易删除大块真实 object surface。

当前实现：

- bbox 只用于诊断；
- pixel ownership 必须来自完整 HaWoR/MANO triangle silhouette 投影；
- MANO 缺失或 malformed 时 fail closed；
- object-owned appearance mask 必须在 projected-MANO subtraction 之后生成。

这保证“手附近”不等于“手拥有”。

### 6.3 Per-component、confidence-aware depth ownership

旧问题是 SAM2 mask 内的背景或错误深度长尾进入 metric point cloud，导致尺度、centroid、extent 和 pose 全部污染。

当前 first-surface ownership：

- 按 connected component 独立处理；
- 使用 local-depth connected growth；
- 使用 UniDepth confidence / predicted error；
- 普通 boundary quarantine band 为 10 px；
- 更深的 interior rejection 仅在固定 sparse/confidence 合同下允许；
- rejected depth 永远不能恢复为 metric pose evidence；
- appearance support 与 metric-depth eligibility 分离保存。

关键代码：

- `scripts/build_v19_visible_geometry_from_sam2_depth.py`
- `experiments/v19_metric_camera_contract/self_test.py`

### 6.4 Orientation-invariant extent gate

旧 extent gate 可能依赖某个 anchor 的 camera-axis AABB，物体旋转后会产生不公平长宽变化。

当前 P09 extent consistency 使用：

- sorted extents；
- visible population median；
- raw/accepted extent tail 只作污染诊断。

### 6.5 Atomic P11 anchor binding

P11 必须将同一 selected frame 的以下内容绑定为一个不可拆分 evidence row：

- RGB；
- object-owned mask；
- camera；
- metric surfels；
- centroid；
- canonical point cloud / observed mesh。

`selected_anchor_atomic_binding` 要求：

```text
selected_frame_idx == visible_geometry_frame_idx == canonical_surface_frame_idx
selected_vs_canonical_centroid_error_m <= 1e-6
validated == true
```

P14/P15 同时 byte-bind `candidate_evidence_report_sha256`。任何 mixed-frame 或 tampered completion/pose report 都 fail closed。

关键代码：

- `scripts/build_v18_compact_rigid_evidence_bundle.py`
- `scripts/build_v19_observed_only_completion_reference.py`
- `scripts/fit_v18_compact_rigid_object_pose.py`
- `scripts/solve_v19_rigid_object_pose_graph.py`

### 6.6 SAM3D native pose 和 metric bridge

SAM3D official native pose contract 已按上游实现验证：

```text
q = [w, x, y, z]
p_pytorch3d = (p_raw * scale_xyz) @ R(q) + t
p_opencv = p_pytorch3d @ diag(-1, -1, 1)
```

当前 runner 显式导出：

- raw local mesh；
- native PyTorch3D-camera mesh；
- native OpenCV-camera mesh；
- quaternion / translation / scale provenance。

D13 bridge：

1. 保留 SAM3D native pose；
2. 对整个 camera-origin scene similarity 做 sensor metric alignment；
3. 使用 identity canonical alignment；
4. 检查 native projection overlap；
5. 检查 observed-front median/P95 residual、coverage 和 extent；
6. 保留完整 generated topology 作为灰色 render underlay；
7. 绿色 observed surface 独立保持物理权威。

关键代码：

- `scripts/remote_run_sam3d_objects_mesh_v7.py`
- `experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py`
- `experiments/sam3d_p11_p12_branch/build_p13_dual_mesh_geometry_prior.py`

禁止回退到 generated-mesh RMS/PCA permutation/ICP 来驱动 object pose。

### 6.7 D13 observed-front quality

SAM3D projection IoU 是必要但非充分条件。D13 同时验证：

- observed-front median / extent；
- P95 / extent；
- native projection IoU；
- coverage；
- extent consistency。

默认 strict P95/extent 为 `0.15`。命名的 conditional P95 tail 只允许在不改变 median gate、projection gate 和物理 eligibility 的情况下延伸到 `0.18`，并必须传播 uncertainty。

### 6.8 Observed-only object pose

P14/P15 不再使用 generated mesh ICP。

直接 pose evidence 仅来自：

1. adjacent accepted observed metric-surface registration；
2. 在 strict-depth gap 中通过全部检查的 RGB bridge：
   - accepted source metric surfels；
   - MANO-subtracted owned RGB optical flow；
   - exact-camera PnP；
   - track/inlier/reprojection/positive-depth gates；
   - 邻近 metric-pose consistency。

Rejected depth 不能通过 RGB bridge 重新获得 metric eligibility。

关键代码：

- `scripts/fit_v18_compact_rigid_object_pose.py`
- `scripts/solve_v19_rigid_object_pose_graph.py`

默认 temporal readiness：

| 项 | 当前门限 |
|---|---:|
| direct pose fraction | `>= 0.80` |
| max direct gap | `<= 10` |
| completed pose fraction | `<= 0.20`（inclusive） |
| nearest hold fraction | `<= 0.05` |
| max rotation step | `<= 15°` |
| max translation step | `<= 0.05 m` |
| rotation observability score | `>= 0.02` |
| observable fraction | `>= 0.80` |

本次最终审计按用户确认采用 `completed_pose_fraction <= 0.20`；因此 `30/150 = 0.20` 是合格边界。

Optimizer success、interpolation、nearest hold 或 zero correction 都不能替代 direct evidence。

### 6.9 Bounded conditional sparse rotation tail

Strict `15°` 门限没有全局提高。可选的：

```text
conditional_sparse_underobservable_rotation_tail
```

必须显式启用：

```bash
--allow-sparse-conditional-rotation-tail \
--conditional-max-rotation-step-deg 18 \
--conditional-max-rotation-step-count 2 \
--conditional-max-rotation-step-fraction 0.015 \
--conditional-max-rotation-step-translation-m 0.020 \
--conditional-max-endpoint-rotation-observability-score 0.030
```

同时要求：

- transition 相邻；
- 两端均是 eligible direct observed-metric rows；
- 不允许 generated geometry pose evidence；
- 不允许修改、裁剪或用平滑掩盖原始 SE(3)；
- destination rows、D17、D18 和 D19 必须传播 uncertainty；
- 超过任一 ceiling 时 fail closed。

若 fresh run 本身 strict pass，则 conditional 开关保持未触发，不应人为制造 warning。例如最终 fresh Spatula 最大 rotation step 为 `13.4743°`，所以其正式 D19 是 strict mode，conditional frames 为空。

### 6.10 Offline model binding

SAM3D/TRELLIS 的 DINOv2 和 SAM3D MoGe 必须离线且 hash-bound：

| Asset | SHA256 |
|---|---|
| DINOv2 checkpoint | `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` |
| DINOv2 `hubconf.py` | `c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64` |
| MoGe `model.pt` | `da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f` |

任何 undeclared Hugging Face resolution、缺失 checkpoint 或 hash mismatch 都是 preflight/D12 blocker。

### 6.11 Immutable runtime bundle

Runtime agent 不直接使用开发 worktree，而使用由 clean committed source 构建的 isolated bundle。

Bundle builder：

- 拒绝 dirty source；
- byte-bind copied files 到 source revision；
- 从 read-only base 复制时只修改 destination owner-write；
- 删除 cache；
- 写出完整 `RUNTIME_BUNDLE_MANIFEST.json`；
- 记录每个文件 SHA256；
- 最终 bundle 应重新锁为只读。

Preflight 会验证：

- manifest missing/hash/undeclared files；
- prompt/path isolation；
- 输入视频 dimensions/FPS/frame count/hash；
- fresh run root；
- SAM2、OWLv2、UniDepth、MANO、DINOv2、MoGe hash；
- 各 interpreter imports；
- 全部 bundled CLI `--help`；
- camera、dual-backend 和 shared-P18 三套 CPU self-tests；
- preflight 前后 bundle byte-identical。

### 6.12 唯一 D18 runner

正式 D18 只能运行：

```text
scripts/run_hot3d_dual_backend_d18_renders.py
```

它从 D17 state 推导 shared pose / warning rows，依次运行两个 backend，并验证每后端：

- `camera_overlay.mp4`
- `world_view.mp4`
- `side_world_view.mp4`
- `side_by_side.mp4`
- 150 frames；
- 30 FPS；
- 相同 shared state；
- 同一 shared P18b payload value hash；
- 有 P18b samples 时 renderer 实际消费并画入 full-duration frames；
- generated collision/contact eligibility 为 false。

直接调用底层 renderer 的诊断输出不得冒充正式 D18。

### 6.13 D19 与五例 collection

Per-case D19：

- 验证两个 render manifests；
- 验证并发布 shared P17、raw P18、P18b 和 interaction judgment；
- 验证 P18 object translation 始终为零；
- 发布稳定视频名；
- 发布 geometry/state/reports；
- 写 `case_result_manifest.json`；
- 最后写 `SUITE_DONE.json`。

五例 collection finalizer：

- 只做 post-prediction aggregation；
- 不写入 case run root；
- 实际逐帧解码全部视频；
- 验证两个 backend shared state 的 canonical value hashes；
- 验证 observed surface hash 相同；
- 生成统一 geometry 和 hand-object state 报告；
- 生成 `ARTIFACT_SHA256_INDEX.json` 和 `SHA256SUMS.txt`。

TrueNAS CIFS `nounix` 上不能安全假设“临时 symlink + `os.replace`”仍是 symlink；当前 finalizer 使用经过 capability probe 和 exact-target resolution 的直接相对 symlink，并对非 symlink fail closed。`--repair-empty-link-stubs` 仅用于一次性修复旧实现产生的已确认 0-byte stubs。

---

## 7. 环境与模型准备

代码仓库不包含以下大型资产：

- HOT3D clips；
- UniDepth model；
- HaWoR checkpoint/infiller/SLAM assets；
- MANO_LEFT / MANO_RIGHT model files；
- OWLv2 model；
- SAM2 checkpoint；
- TRELLIS repo/model；
- SAM3D Objects repo/config/checkpoints；
- DINOv2/MoGe checkpoints。

典型解释器：

```text
MAIN_PYTHON      主 ego_annotation venv
UNIDEPTH_PYTHON  UniDepth 独立环境
HAWOR_PYTHON     HaWoR 独立环境
TRELLIS_PYTHON   TRELLIS 独立环境
SAM3D_PYTHON     由 sam3d-objects/activate.sh 解析
```

不要让 runtime agent 在 prediction run 内安装依赖、修复环境或创建虚拟环境。环境准备与 prediction execution 必须分离。

---

## 8. 从 clean branch 构建 bundle

以下使用占位变量；所有路径必须替换为当前机器上的真实绝对路径。

```bash
set -euo pipefail

SOURCE_ROOT=/ABS/PATH/ego_annotation_worktree
BASE_BUNDLE=/ABS/PATH/validated_v19_base_bundle
BUNDLE_ROOT=/ABS/PATH/new_hot3d_dual_backend_bundle
MAIN_PYTHON=/ABS/PATH/ego_annotation/.venv/bin/python

cd "$SOURCE_ROOT"
test -z "$(git status --porcelain=v1)"

"$MAIN_PYTHON" scripts/build_hot3d_dual_backend_runtime_bundle.py \
  --base-bundle "$BASE_BUNDLE" \
  --source-root "$SOURCE_ROOT" \
  --bundle-root "$BUNDLE_ROOT"
```

构建后至少检查：

```bash
python3 - "$BUNDLE_ROOT/RUNTIME_BUNDLE_MANIFEST.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["source_worktree_dirty"] is False
assert m["offline_model_assets"]["network_resolution_allowed"] is False
print(m["source_revision"], m["file_count"])
PY
```

完成 preflight 后再递归锁为只读；不要修改 base bundle。

---

## 9. Case preflight

完整参数以 `scripts/preflight_local_v19_runtime.py --help` 为准。典型形式：

```bash
"$MAIN_PYTHON" "$BUNDLE_ROOT/scripts/preflight_local_v19_runtime.py" \
  --bundle "$BUNDLE_ROOT" \
  --input-video "$INPUT_VIDEO" \
  --run-root "$FRESH_RUN_ROOT" \
  --output "$PREFLIGHT_REPORT" \
  --main-python "$MAIN_PYTHON" \
  --unidepth-python "$UNIDEPTH_PYTHON" \
  --unidepth-repo "$UNIDEPTH_REPO" \
  --unidepth-model "$UNIDEPTH_MODEL" \
  --hawor-python "$HAWOR_PYTHON" \
  --hawor-repo "$HAWOR_REPO" \
  --hawor-asset-root "$HAWOR_ASSET_ROOT" \
  --trellis-python "$TRELLIS_PYTHON" \
  --trellis-model "$TRELLIS_MODEL" \
  --torch-home "$TORCH_HOME" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --owlv2-model "$OWLV2_MODEL" \
  --sam3d-python "$SAM3D_PYTHON" \
  --sam3d-repo "$SAM3D_REPO" \
  --sam3d-config "$SAM3D_CONFIG" \
  --sam3d-activation "$SAM3D_ACTIVATION" \
  --sam3d-moge-checkpoint "$SAM3D_MOGE_CHECKPOINT" \
  --expected-input-sha256 "$INPUT_SHA256" \
  --expected-width 1408 \
  --expected-height 1408 \
  --expected-fps 30 \
  --expected-frame-count 150 \
  --allow-input-sidecar v19_camera_calibration_contract.json
```

只有报告状态为：

```text
ready_for_runtime_agent_launch
```

才可启动。

---

## 10. Suite manifest 与启动

可复制模板：

```text
configs/hot3d_dual_backend_suite_manifest.example.json
```

每个 case 至少绑定：

- `case_id`
- `object_id`
- `window`
- 独占 `gpu_id`
- `input_video`
- `sensor_calibration_metadata`
- fresh `run_root`
- 已通过的 `preflight_report`
- `log`
- `agent_done`
- `target_hint`
- `target_exclusions`

启动：

```bash
MAIN_PYTHON=/ABS/PATH/ego_annotation/.venv/bin/python
"$MAIN_PYTHON" scripts/launch_hot3d_dual_backend_suite.py \
  --suite-manifest /ABS/PATH/suite_manifest.json
```

监控器由 launcher 在 dedicated tmux session 中启动；也可读取：

```text
{SUITE_ROOT}/progress/progress.json
{SUITE_ROOT}/progress/progress.md
{SUITE_ROOT}/progress/live_progress.log
{SUITE_ROOT}/launch_state.json
```

不要复用旧 run root。每个 GPU assignment 必须唯一，并在 GPU-heavy command 前做一次 live safety probe。

---

## 11. D18、D19 与 collection 命令

### 11.0 Shared D16b P17/P18/P18b

正式命令以 `runtime/hot3d_dual_backend_runtime_spec.md` 为准：先由 runtime agent 基于 prediction-side 图像写完整左右手 interaction judgment，再运行：

```bash
"$MAIN_PYTHON" scripts/run_hot3d_shared_p17_p18_tail.py \
  --case "$CASE_ID" \
  --object-id "$OBJECT_ID" \
  --annotations "$ANNOTATIONS" \
  --pose-report "$POSE_GRAPH" \
  --completion-report "$OBSERVED_COMPLETION" \
  --depth-npz "$DEPTH_NPZ" \
  --hawor-npz "$HAWOR_NPZ" \
  --interaction-judgment "$INTERACTION_JUDGMENT" \
  --wilor-root third_party/WiLoR \
  --wilor-mano-left third_party/WiLoR/mano_data/MANO_LEFT.pkl \
  --wilor-mano-right third_party/WiLoR/mano_data/MANO_RIGHT.pkl \
  --output-root "$EXP_ROOT/P16b_shared_p17_p18_tail" \
  --start-frame 0 --end-frame 149 --sides left right --device cuda
```

该 runner 在加载 MANO/GPU optimizer 前验证 source-K↔centered-HaWoR inference-plane 的 camera/MANO affine合同；执行后验证 exact mask affine、observed-only surface、D15 pose binding、零 object delta 和 P18b metric-MANO preservation。

### 11.1 D18

```bash
"$MAIN_PYTHON" scripts/run_hot3d_dual_backend_d18_renders.py \
  --run-root "$RUN_ROOT" \
  --anchor-frame "$ANCHOR_FRAME" \
  --expected-frame-count 150 \
  --expected-fps 30 \
  --generated-face-budget 12000 \
  --replace
```

必须使用 image-read 实际检查：

- anchor review；
- full timeline representative frames；
- bridge/低置信 transition 邻域（若有）；
- 两 backend pose 连续性；
- MANO handedness 和明显 projection drift。

### 11.2 D19

```bash
"$MAIN_PYTHON" scripts/finalize_hot3d_dual_backend_case.py \
  --case "$CASE_ID" \
  --object-id "$OBJECT_ID" \
  --run-root "$RUN_ROOT" \
  --sam3d-render-manifest "$SAM3D_RENDER_MANIFEST" \
  --trellis-render-manifest "$TRELLIS_RENDER_MANIFEST" \
  --controlled-report "$CONTROLLED_REPORT" \
  --dual-report "$DUAL_REPORT" \
  --state-adapter-report "$STATE_ADAPTER_REPORT" \
  --shared-tail-report "$SHARED_TAIL_REPORT" \
  --output-dir "$RUN_ROOT/final_results" \
  --expected-frame-count 150 \
  --expected-fps 30
```

### 11.3 多 case collection

```bash
"$MAIN_PYTHON" scripts/finalize_hot3d_fivecase_collection.py \
  --collection-root "$COLLECTION_ROOT" \
  --case milk="$MILK_RUN_ROOT" \
  --case soup="$SOUP_RUN_ROOT" \
  --case mug="$MUG_RUN_ROOT" \
  --case bbq="$BBQ_RUN_ROOT" \
  --case spatula="$SPATULA_RUN_ROOT"
```

正常新 collection 不应使用 `--repair-empty-link-stubs`。

---

## 12. 稳定输出合同

每个 D19 case：

```text
{RUN_ROOT}/SUITE_DONE.json
{RUN_ROOT}/final_results/case_result_manifest.json

{RUN_ROOT}/final_results/sam3d/videos/camera_overlay.mp4
{RUN_ROOT}/final_results/sam3d/videos/world_view.mp4
{RUN_ROOT}/final_results/sam3d/videos/side_world_view.mp4
{RUN_ROOT}/final_results/sam3d/videos/side_by_side.mp4

{RUN_ROOT}/final_results/trellis/videos/camera_overlay.mp4
{RUN_ROOT}/final_results/trellis/videos/world_view.mp4
{RUN_ROOT}/final_results/trellis/videos/side_world_view.mp4
{RUN_ROOT}/final_results/trellis/videos/side_by_side.mp4
```

此外每后端发布：

```text
geometry/
state/render_state.json
reports/render_manifest.json
backend_result.json
```

五例 collection：

```text
collection_manifest.json
COLLECTION_DONE.json
ARTIFACT_SHA256_INDEX.json
SHA256SUMS.txt
BACKEND_GEOMETRY_COMPARISON.json
BACKEND_GEOMETRY_COMPARISON_ZH.md
SHARED_HAND_OBJECT_STATE_REPORT.json
SHARED_HAND_OBJECT_STATE_REPORT_ZH.md
README_ZH.md
runs/<case>                 # validated relative symlink
final_results/<case>        # validated relative symlink
```

---

## 13. 已完成五例结果摘要

统一 collection 已完成：

- 5 cases；
- 10 backend branches；
- 40 final videos；
- 每视频 150 frames / 30 FPS / 5 s；
- collection finalizer 完整解码 40 个视频；
- 独立复核共解码 6000 frames；
- 132 条 `SHA256SUMS.txt` 校验通过。

| Case | direct | completed | max rotation | SAM3D observed→generated median | TRELLIS median | visible median winner |
|---|---:|---:|---:|---:|---:|---|
| milk | 148 | 2 | 14.625° | 2.527 mm | 6.656 mm | SAM3D |
| soup | 139 | 11 | 8.001° | 3.373 mm | 7.677 mm | SAM3D |
| mug | 128 | 22 | 10.692° | 7.170 mm | 3.496 mm | TRELLIS |
| BBQ | 144 | 6 | 13.363° | 3.956 mm | 5.187 mm | SAM3D |
| spatula | 120 | 30 | 13.474° | 2.650 mm | 1.650 mm | TRELLIS |

Visible-fit median 胜数：SAM3D 3，TRELLIS 2。该指标只衡量 generated prior 对 prediction-side visible observed surface 的支持，不是 hidden-shape GT accuracy。

Fresh Spatula：

- selected anchor frame：149；
- direct fraction：0.80；
- completed fraction：0.20，inclusive pass；
- max direct gap：8；
- nearest hold：0；
- observable fraction：0.90；
- max rotation step：13.474°；
- strict temporal readiness；
- generated pose evidence：false；
- nonpenetration target count：0。

---

## 14. 测试与发布检查

CPU 回归：

```bash
MAIN_PYTHON=/ABS/PATH/ego_annotation/.venv/bin/python

PYTHONDONTWRITEBYTECODE=1 "$MAIN_PYTHON" \
  experiments/sam3d_p11_p12_branch/self_test.py

PYTHONDONTWRITEBYTECODE=1 "$MAIN_PYTHON" \
  experiments/v19_metric_camera_contract/self_test.py
```

当前验证结果：

```text
SAM3D branch tests: 20/20
camera contract tests: 20/20
```

Git 发布前至少执行：

```bash
python3 -m py_compile \
  scripts/*.py \
  experiments/sam3d_p11_p12_branch/*.py \
  experiments/v19_metric_camera_contract/*.py

git diff --check
git status --short
```

同时检查：

- 没有 checkpoint、视频、mesh、NPZ、运行输出进入 commit；
- 没有 credential/token/private key；
- 新增 blob 大小合理；
- worktree clean；
- release branch 指向预期 commit。

---

## 15. 已知限制

1. **MANO 非 GT**：HaWoR/MANO 是 RGB prediction，遮挡和 infiller 帧有不确定性。
2. **HaWoR camera contract**：新 fresh run 要求 P04 将 active source K 通过显式 affine 转到 centered inference plane，并让 MANO regression、hand-mask projection 和 HaWoR SLAM 共同消费该 centered K，再用逆 affine 回绑 source pixels。历史五例仍是 image-center archive，不能直接晋级到 shared P17/P18；必须从 P04 重跑。
3. **Observed surface partial**：不能提供全物体 signed SDF。
4. **Generated hidden shape 非证据**：完整后侧形状是 backend-dependent render hypothesis。
5. **Contact/nonpenetration unresolved**：unsigned proximity 不能解释为 signed contact 或无穿透。
6. **Monocular symmetry**：即使 surface residual 很小，rotation 仍可能因对称/低观测而不确定。
7. **RGB-PnP bridge 较低置信**：它是受严格 calibrated gates 约束的 direct bridge，但仍应在最终 uncertainty 中保留。
8. **历史五例版本不同**：统一 collection 是对旧 D16/D17 成功 D19 artifacts 的 post-prediction 聚合，不包含本分支新增的 shared P17/P18/P18b；不等价于新 source-K/centered-HaWoR/shared-P18 bundle 的五例全量重跑。
9. **路径需本机适配**：runtime spec 中的 A800/model 路径是已验证部署的绝对路径；迁移到新机器需要重新 provision、hash check、preflight 和 immutable bundle build。

---

## 16. 常见故障与排查

### 16.1 K/rays 不一致

症状：UniDepth 返回 K/rays 与 supplied K 差异过大。

处理：

- 检查 P03b sensor contract；
- 检查 input/output image plane；
- 不允许仅改 metadata；
- 重新执行 sensor-conditioned P03。

### 16.2 Anchor 投影 IoU 很低

不要降低 projection gate。应检查：

- SAM3D frame-conditioned native pose 是否错误；
- anchor mask 是否连通且 target-owned；
- conditioning coherence；
- P11 frame/centroid atomic binding。

### 16.3 P15 rotation jump

先区分：

- 错误 pose evidence；
- symmetry / low observability；
- RGB bridge failure；
- 少量 bounded underobservable transitions。

Conditional tier 不是全局放宽；不符合全部 ceilings 时必须 blocker。

### 16.4 D18 找不到 renderer

只运行：

```text
scripts/run_hot3d_dual_backend_d18_renders.py
```

不要猜测或重建 renderer filename。

### 16.5 Collection link 变成 0-byte 文件

这是 CIFS `nounix` 上 symlink rename 的已知问题。当前 finalizer 已改为直接相对 symlink + exact resolution validation。仅修复旧 stubs 时显式使用：

```bash
--repair-empty-link-stubs
```

该选项不会允许覆盖非空普通文件。

---

## 17. Review 建议

代码审查建议按以下顺序：

1. `runtime/hot3d_dual_backend_runtime_spec.md`
2. `configs/hot3d_dual_backend_agent_system_prompt.md`
3. `scripts/build_hot3d_dual_backend_runtime_bundle.py`
4. `scripts/preflight_local_v19_runtime.py`
5. `scripts/build_v19_visible_geometry_from_sam2_depth.py`
6. `scripts/build_v18_compact_rigid_evidence_bundle.py`
7. `scripts/remote_run_sam3d_objects_mesh_v7.py`
8. `experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py`
9. `scripts/fit_v18_compact_rigid_object_pose.py`
10. `scripts/solve_v19_rigid_object_pose_graph.py`
11. `scripts/run_hot3d_shared_p17_p18_tail.py`
12. `scripts/solve_v18_joint_mano_interval_trajectory.py`
13. `experiments/sam3d_p11_p12_branch/render_p14_p15_layered_state.py`
14. `scripts/run_hot3d_dual_backend_d18_renders.py`
15. `scripts/finalize_hot3d_dual_backend_case.py`
16. `scripts/finalize_hot3d_fivecase_collection.py`
17. 三套 `self_test.py`

审查时应始终坚持：manifest/gate 是机制证据的记录，不是对视觉与物理机制本身的替代。
