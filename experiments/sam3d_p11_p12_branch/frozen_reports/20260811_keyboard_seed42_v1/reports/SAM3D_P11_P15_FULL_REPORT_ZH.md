# SAM3D Objects 替换 TRELLIS 几何先验：P11–P15 全流程总结

## 1. 结论摘要

- 当前证据支持把 **SAM3D Objects 继续作为 P12 完整形状/render prior 的首选实验候选**；尚不支持替换物体时序、MANO、contact、collision 或 signed nonpenetration。
- object-owned mask 的新 SAM3D 结果保持了旧结果的主体形状，同时减少组件并改善 P13 tail/hidden retention；但仅有 keyboard、单 anchor、seed 42。
- P13 legacy observed-band face deletion 是约 11 mm seam 和拓扑破坏的直接来源。Topology-preserving dual mesh 保留原始 275,208 faces、0 boundary edges 和 watertight 状态。
- 150 帧 P15 可视结果把完整 SAM prior、共享 observed surface、完整 778-vertex MANO 和同一 observed-only trajectory/camera 放在同一 artifact 中；TRELLIS 的隐藏形状在世界/侧视图中仍明显偏离 keyboard，SAM 两路更接近完整矩形主体。
- 当前最硬的物理 blocker 已从“没有完整手物视频”转为 **MANO/object/camera/trajectory 的度量冲突**：共享 observed surface 上，仅 6.00% 的 frame-side hand rows 有任一完整 MANO vertex 位于 5 mm 内；因此不能把生成网格的近距离当成 contact。

## 2. 不变量与实验边界

- 所有工作位于 additive experiment；canonical V19 与冻结 TRELLIS 产物未改。
- SAM3D 原生输入：full RGB + prediction-side binary object-owned mask；`pointmap=None`；无外部 crop/rotation/rectification。
- P14/P15 三路只有 geometry source/integration 不同；annotation/camera、150-frame observed-only pose、MANO payload、temporal MANO payload、projection contract 的 value hash 完全相同。
- 三路共享 2,394 vertices / 4,598 faces 的 observed-only physical surface；generated faces 的 collision/contact eligibility=false，signed geometry=false。
- 这里的 painter 是可视 review，不是 metric z-buffer evaluator；旧 1408→960 K 合同仍需在 official HOT3D pinhole/P03c 上重做。

## 3. P11：公平输入合同与 mask provenance

- anchor frame：109；object-owned mask：63,861 px。
- 旧 raw SAM2 mask：67,917 px；XOR 4,056 px；IoU 0.940280。
- ownership mask 与 visible-geometry candidate byte-identical；来源是 prediction-side OWLv2→SAM2→hand ownership subtraction，无 HOT3D GT mask/CAD/pose leakage。
- TRELLIS 保留 object-isolated RGBA crop；SAM3D 保留 full-scene native contract。公平性来自同 anchor/owned mask，而不是强迫相同 crop。

## 4. P12：SAM3D 原生生成与 mask A/B

- 新 raw mesh：137,598 vertices / 275,208 faces；extents=[0.9994843006134033, 0.38356591761112213, 0.10400628671050072]；watertight=True。
- raw mesh SHA256：`a3f6c3b545169ce7971659c2671afa638dc45cbafd1c16f68bd6e0239e1c9e79`；seed=42；native uniform scale=0.511941。
- old/new normalized symmetric surface mean distance：0.004466 longest-axis units。
- silhouette IoU：top=0.9884，long-side=0.9179，end=0.8924。
- components：4→2；native rotation delta=6.35°；scale ratio=0.9489。

## 5. P13：共同下游 controlled A/B

| Candidate | obs→gen median (mm) | obs→gen p95 (mm) | gen→obs p95 (mm) | free-space reject | retained hidden | hidden faces |
|---|---:|---:|---:|---:|---:|---:|
| trellis_frozen | 2.99 | 9.96 | 99.06 | 33.27% | 6.60% | 18,160 |
| sam3d_old_raw_sam2_mask | 3.41 | 9.38 | 28.90 | 3.88% | 43.43% | 115,973 |
| sam3d_new_object_owned_mask | 3.51 | 8.40 | 28.01 | 4.03% | 45.55% | 125,352 |

单一 obs→gen median 会偏向 TRELLIS，但它同时有高 free-space reject、极低 hidden retention 和错误的完整主轴形状；因此不能用单一 nearest residual 选 backend。

### Legacy seam

| Candidate | symmetric median (mm) | symmetric p95 (mm) | observed→generated median (mm) | observed boundary within 10 mm |
|---|---:|---:|---:|---:|
| trellis_frozen | 8.87 | 25.95 | 6.84 | 62.77% |
| sam3d_old_raw_sam2_mask | 11.29 | 15.01 | 10.61 | 40.96% |
| sam3d_new_object_owned_mask | 11.31 | 14.04 | 9.93 | 50.53% |

Legacy observed band=12.119 mm；new SAM legacy seam median≈11.31 mm，与该删除带一致。

### Topology-preserving dual mesh

- intact aligned SAM：137,598 vertices / 275,208 faces，boundary edges=0，components=2，watertight=True。
- legacy cut：68,175 vertices / 129,950 faces，boundary edges=6,318，components=137，watertight=False。
- dual mesh 不填洞、不 stitch、不形变、不删 generated faces；observed surface 独立覆盖，collision 保持 observed-only。

## 6. P14/P15：150-frame object + full MANO + camera

- 三路均完成 150 frames / 5.000 s；所有输出通过 ffprobe frame/duration/codec 检查及完整 decode。
- 每个 frame/side 使用完整 778 MANO vertices + 1,538 faces；共 233,400 个 vertex-surface queries；64-vertex inline sample reproduction max error=0.0e+00 m。
- camera overlay 中共享 observed green layer一致；geometry 差异主要出现在 observed silhouette 外缘、隐藏底面与世界/侧视图。
- frame 50/109 等代表帧中：SAM dual 与 SAM legacy 均维持 keyboard-like 完整长方体；dual 保留更多连续 purple underlay；TRELLIS 在 X-Z/Y-Z 中表现为明显 wedge/偏轴完整先验。
- 完整 MANO 在 2D overlay 大体贴合可见手，但 world/side 中与 object 的大距离暴露了现有 hand/object/camera/trajectory 度量冲突。

### 完整 MANO unsigned proximity（仅 observed surface 可作物理 proximity）

- observed surface：all-vertex median=114.89 mm；per-row minimum median=54.34 mm；任一 vertex within 5/10 mm 的 row fraction=6.00%/9.67%。
- intact SAM 对同一 MANO 比 legacy-cut 更近的 vertex fraction=59.57%；median delta=-0.66 mm。该结果只说明 face deletion 删除了邻近 render surface，**不是 contact 改善**。
- observed surface 非 watertight，不能给 sign；generated mesh 即使 watertight 也没有 metric/collision validation。

## 7. 当前判断

1. **继续保留 SAM3D object-owned-mask + topology-preserving dual mesh 作为实验 render prior 主候选。**
2. **TRELLIS 暂时保留 frozen fallback，不改默认 canonical backend。** 当前单样本视觉/hidden-tail 证据偏向 SAM3D，但不足以正式推广。
3. **不复用 SAM3D native pose 代替 P15 trajectory。** 当前视频使用 observed-only trajectory 是正确隔离；native quaternion/scale 仍需独立 reprojection/GT convention 验证。
4. **contact/collision/signed state 继续禁用。** 当前完整 MANO proximity 明确显示度量冲突，生成表面的 near-zero distance不能升级为物理事实。

## 8. 还需要做的工作（优先级）

### P0：正式替换前必做
- 在 official HOT3D pinhole K/P03c camera contract 上重复 P11–P15，消除旧 1408→960 K 合同的外推风险。
- 解决/重新估计 MANO–object–camera–trajectory 的共同 metric state；用完整 MANO、可见深度和 mask reprojection 同时验证，而不是让 generated mesh 吸收冲突。
- 对三路固定预测做真正 perspective-correct metric z-buffer evaluator：held-out silhouette、first-hit depth median/p95、free-space contradiction、hand/object depth order。
- 扩展到多 object、多 anchor、多 seed；至少报告均值、方差和失败类型，不能以 keyboard/seed42 决定默认 backend。

### P1：模型与 pose 归因
- 独立验证 SAM3D native quaternion/axis/translation/scale convention；HOT3D CAD/pose 如可用，只能在预测冻结后作为 evaluator。
- 做 generator-only 共同下游 A/B 与 SAM3D-native capability 分离报告；legacy `trellis_*` 兼容字段不得进入 source attribution。
- 评估多 anchor 或多视角支持是否能稳定完整 prior，而不是 seed/anchor 挑优。

### P2：物理状态与生产接入
- 独立构建 conservative collision proxy，并验证 watertight、metric front surface、free-space 与多视角一致性；通过前 generated mesh 永远 render-only。
- 在可信 proxy 后，才重做完整 778 MANO contact/nonpenetration；继续保留 unsigned/signed 语义隔离。
- 只有多样本视频与 evaluator 均改善后，才把 source-neutral geometry contract 接到 runtime spec 并考虑修改默认 backend。

## 9. 关键产物

- 三路 camera overlay A/B：`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2/experiments/sam3d_p11_p12_native_owned_mask_v1/p14_p15_layered_geometry_ab_v1/full_video_three_branch_ab/p15_three_branch_camera_overlay_ab.mp4`
- 三路 overlay/world/side A/B：`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2/experiments/sam3d_p11_p12_native_owned_mask_v1/p14_p15_layered_geometry_ab_v1/full_video_three_branch_ab/p15_three_branch_overlay_world_side_ab.mp4`
- 多帧 QC sheet：`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2/experiments/sam3d_p11_p12_native_owned_mask_v1/p14_p15_layered_geometry_ab_v1/final_qc/p15_three_branch_multiframe_qc.jpg`
- 完整 MANO unsigned report：`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2/experiments/sam3d_p11_p12_native_owned_mask_v1/p14_p15_layered_geometry_ab_v1/full_mano_unsigned_distance/p15_full_mano_unsigned_surface_distance_report.json`
- 最终 machine-readable report：`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2/experiments/sam3d_p11_p12_native_owned_mask_v1/p14_p15_layered_geometry_ab_v1/final_qc/sam3d_p11_p15_final_evidence_report.json`

本报告中的视觉判断必须结合上述视频/QC sheet；JSON/schema 仅是 backing evidence。
