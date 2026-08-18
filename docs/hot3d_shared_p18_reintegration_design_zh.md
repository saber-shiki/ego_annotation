# HOT3D 双后端共享 P18 回接设计（安全 patch）

## 1. 目标与非目标

本 patch 在现有 HOT3D SAM3D/TRELLIS 双后端管线的 D15 与 D17 之间恢复一次共享的 canonical P17、P18、P18b：

```text
P04 shared source-K → centered-HaWoR-plane MANO reinference
  -> D15 shared observed-only object pose
  -> D16 shared unsigned MANO/object measurement
  -> P17 shared visible ownership/contact prior
  -> P18 shared interval MANO candidate
  -> P18b shared metric-MANO-preserved surface hypothesis
  -> D17 clone shared state
       -> SAM3D render geometry
       -> TRELLIS render geometry
  -> D18 full-duration renders
```

目标是使 P18/P18b 真实进入最终渲染数据流，同时保持几何后端是唯一分支变量。

本 patch **不**把 SAM3D 或 TRELLIS generated faces 提升为 pose、contact、collision 或 signed-distance 证据；也不宣称 partial/non-watertight observed surface 支持 signed contact/nonpenetration。

## 2. 输入合同

每个 case 必须从同一 fresh run root 读取：

- P09 visible-geometry annotations；
- P03 active-camera depth NPZ；
- P04 HaWoR world MANO archive；
- D14 observed-only completion report 和 physical surface；
- D15 observed-only full-timeline pose graph；
- agent 基于 prediction-side 图像证据写入的 P17 interaction judgment。

P17 judgment 不能由 hand/object distance 自动生成；它仍是视觉 contact/occlusion prior。

## 3. 单一位姿权威

`D15 pose_rows` 是唯一的逐帧 object-pose authority：

```text
X_observed_world(t) = R_D15(t) X_observed_canonical + t_D15(t)
X_SAM3D_world(t)    = R_D15(t) X_SAM3D_canonical    + t_D15(t)
X_TRELLIS_world(t)  = R_D15(t) X_TRELLIS_canonical  + t_D15(t)
```

约束：

1. SAM3D native `q/t/scale` 只在 D13 烘焙一次；D17/D18 不得再次应用。
2. 正式共享 P18 必须使用 `--no-optimize-object-translation`。
3. P18/P18b 每行 `optimized_object_translation_world_m` 必须为零。
4. P18、SAM3D state、TRELLIS state 必须绑定同一 D15 report SHA256 和同一 `object_pose_trajectory` value SHA256。

## 4. active-K MANO 对齐

HOT3D official active K 的 principal point 与 full-image center 相差约 1–2 px。历史 HaWoR video wrapper虽然把 `img_center` 暴露给底层模型，却在 wrapper 中硬编码 image center；仅传 focal 或替换 JSON K 都不算对齐。

本 patch 要求 P04 将完整 active source-plane `[fx,fy,cx,cy]` 通过一个显式、hash-bound 的同尺寸图像平移 affine 转为 center-principal-point HaWoR inference plane，并在同一次 fresh HaWoR 执行中把该 inference-plane K 绑定到：

- HaWoR per-frame MANO regression 的 centered `img_focal` 和 `img_center`；
- 生成 SLAM hand masks 的投影相机；
- HaWoR masked DROID-SLAM 和 metric-depth scale 的 K；
- 最终 NPZ 中逐帧保存 source K、inference K、`A_hawor_inference_from_source`、逆 affine、rectified frame hashes，并将 detector boxes 逆变换回 source plane。

HaWoR 当前 video model 仍只支持一个 square focal，因此 `fx != fy` 时必须 fail closed，不能用几何平均值伪装；当前 HOT3D active K 满足 `fx==fy`。

所有 HaWoR motion/mask/SLAM cache 必须绑定 source K、centered inference K 和双向 image affine。历史仅记录 focal 的 cache 在 source-K/centered-plane 模式下视为不兼容并强制清除或换新 sequence root。P08 必须重新 hash image-plane contract 文件和全部 rectified input frames，验证 NPZ source/inference K、双向 affine、source video hash 和 150 帧 timeline；只有这套闭环成立且 source K 与 active source-plane K 四项相等时，才允许 `active_contract_reinference_required=false`。

已有 center-K 五例不能通过后验小平移或 metadata rewrite 晋级；它们必须从 P04 开始 fresh rerun，或者继续被 shared P17/P18 tail fail closed 拒绝。

## 5. source/mask/depth 图像平面

P17 ownership masks 位于 SAM2 mask plane（当前为 960×960），depth 与 active K 位于 source/depth plane（当前为 1408×1408）。所有投影必须显式维护：

```text
uv_source = project(K_active, X_camera)
uv_mask   = A_mask_from_source uv_source
```

- mask membership 只用 `uv_mask`；
- depth lookup 只用 `uv_source`；
- `A_mask_from_source` 来自 P09 `mask_depth_transform_contract` 的显式逆变换；
- 禁止按 `width/(2*cx)` 猜测 resize。

P17 factor rows必须携带 exact image-plane transform；P18 必须验证 mask raster size 与 transform contract。

## 6. 物理表面隔离

共享 P18 的 pose hypothesis 和 physical surface 都必须解析到 D14/P13 的 prediction-side observed-only mesh。必须满足：

- D15 pose canonical mesh geometry hash == shared observed physical mesh geometry hash；
- completion report `collision_eligible_mesh_labeled` == shared observed mesh；
- generated faces 的 pose/contact/collision/signed-distance eligibility 全为 false；
- SAM3D/TRELLIS mesh 路径或 hash 不得出现在 P17/P18 physical inputs 中。

当前 observed surface partial、unsigned、non-watertight，因此 P18 signed object-surface factor保持 inactive。

## 7. P18b 和最终可见产物

P18b 默认保留 fresh source-K/centered-plane P04 的 metric MANO joints/root；P18 的 optimized sampled vertices作为单独的 uncertain visible-surface/contact hypothesis。

D18 layered renderer 必须：

- 继续用 fresh source-K/centered-plane P04 的 full 778 source metric MANO 绘制手体；
- 显式读取 `render_state.temporal_mano_state.payload`；
- 在 camera/world/side views 绘制 P18b uncertain surface samples；
- manifest 记录 consumed row count、value hash、每帧可见 sample count；
- 不把这些 samples 标为 accepted contact 或 collision。

若 temporal state 被写入 JSON 但未改变最终渲染内容，视为接线失败。

## 8. A/B 公平性与审计

D17 clone 后必须逐 value hash 相等：

- annotation backbone；
- D15 object trajectory；
- D16 constraint state；
- P18b temporal MANO state；
- hidden-volume state；
- projection contract。

允许不同的只有 generated render geometry 与相应 provenance/label。D18 wrapper在渲染前后再次验证这些合同。

## 9. 完整时长输出

每个 backend 仍输出与原视频相同的 150 帧、30 FPS：camera overlay、world、side-world、side-by-side。D19 finalizer发布 P17/P18/P18b 和 source-K/centered-plane P04/D15 handoff provenance，使发布包能证明最终视频实际消费的共享状态。

## 10. 失败模式

以下情况必须写 blocker 并停止 shared P18 tail，不得回退成伪物理状态：

- P04 未从 active source K 生成并 hash-bind centered HaWoR inference plane，未在该平面同时重跑 regression/mask projection/SLAM，或 source↔inference affine binding 不闭合；
- 缺少结构化 P17 judgment；
- source/mask affine 缺失、raster size 不匹配；
- D15 不是 ready full timeline；
- physical mesh 不是 observed-only exact mesh；
- P18 object translation 非零；
- 两个 backend 的共享 state hashes 不一致；
- renderer 没有实际消费 P18b rows。

噪声 contact prior 或弱 unsigned proximity 不应阻止渲染，但必须继续标为 uncertain。

## 11. 验证计划

1. 单元/合同测试：source-K→centered-HaWoR affine及逆变换、rectified-frame篡改检测、source→mask affine、P18 object-delta rejection、branch hash equality、temporal renderer consumption。
2. 对已有五例做 handoff dry-run，确认它们因历史 center-K archive 被明确拒绝；对一个 fresh source-K/centered-plane rerun确认 300 hand rows、150 pose rows和 explicit affine 全覆盖。
3. P18 是 GPU/优化任务，只能在 A800 runtime workspace执行；本地不启动重型求解。
4. 首个 A800 case 完成后检查 anchor 和 contact interval 的四路视频，确认 P18b samples可见且两后端手/相机/pose完全一致。
