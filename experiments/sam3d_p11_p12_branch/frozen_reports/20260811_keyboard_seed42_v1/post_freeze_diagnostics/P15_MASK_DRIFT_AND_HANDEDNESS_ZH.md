# P15 视频后冻结诊断：键盘漂移与 MANO 左右手

## 诊断边界

- 基线预测已先冻结于 commit `7d5720e2422a3a5c825d28aa7209a1d8268275a6`。
- 本诊断随后才读取 HOT3D evaluator-only sidecar；GT 未进入 SAM2、SAM3D、TRELLIS、物体 pose、MANO 或 P15 render 的预测过程。
- 本诊断不修改 P11–P15 基线输出，也不作 contact、collision 或 signed nonpenetration 声明。
- 机器可读主报告：NAS `post_freeze_diagnostics_v1/p15_mask_drift_handedness_audit.json`。
- 代表帧可视复核：`post_freeze_diagnostics/review/p15_mask_drift_review.jpg`（raw SAM2 / object-owned / green observed mesh 三列）。

## 1. 键盘 mask 是否仍在漂移？

### 先区分视频里的三个不同量

1. **raw SAM2 mask**：逐帧 tracking raster mask。
2. **object-owned mask**：raw SAM2 mask 再减去 HaWoR 手部检测矩形（每边另加 12 px padding）。
3. **P15 绿色层**：固定 observed metric mesh 经逐帧 object pose 和 camera 投影后的表面；它不是逐帧 mask。

因此，“绿色层漂了”不能直接归因于 SAM2 tracker。

### 结果

- raw SAM2 对 official HOT3D modal keyboard mask，150 帧：
  - IoU median `0.97055`，p10/p90 `0.96111 / 0.98504`；
  - centroid distance median `2.27 px`，p90 `3.04 px`；
  - 最差 frame 149 IoU `0.89591`，是尾帧边界/出画面误差，而不是贯穿全片的明显 tracker drift。
- object-owned mask 对 GT：
  - IoU median 仅 `0.33355`；
  - 它保留 raw SAM2 面积的 median `34.75%`，最低 `12.85%`。
  - 原因是**整块 hand bbox subtraction 过度保守**，并非 SAM2 时序漂移。
- P15 绿色 observed-mesh 对 GT：
  - IoU median `0.62297`，p10 `0.39874`；
  - centroid distance median `35.84 px`，p90 `115.57 px`。
  - 这与肉眼看到的明显漂移一致，但漂移对象是 mesh/pose/camera 投影链。
- 固定 canonical transform 后，当前 object trajectory 对 HOT3D 的残差：
  - translation median/p95 `91.4 / 173.5 mm`；
  - rotation median/p95 `8.14° / 13.41°`。
- 当前 HaWoR camera 与 official camera 在去除恒定 world-frame convention 后的 frame-0-relative motion mismatch：
  - translation median/p95 `14.1 / 123.0 mm`；
  - rotation median/p95 `0.56° / 3.24°`。
- 当前 K（1408 坐标）为约 `(537.02, 537.02, 722.29, 720.90)`；official pinhole K 为约 `(609.85, 609.85, 707.49, 702.32)`。但只把冻结 pose/camera 的 K 换成 official K，mesh-vs-GT IoU median 反而为 `0.47031`，所以不能把问题简化成单独换 K；必须在 official camera contract 上重建 depth/visible geometry/object trajectory。

### 结论

**raw SAM2 键盘 mask 在这 150 帧中没有用户所见程度的明显漂移；P15 绿色层确实明显漂移，但主要属于 observed mesh + object trajectory + camera/canonical projection 链。** 另有一个独立问题：object-owned mask 因矩形级 hand subtraction 被严重切残。

还发现 annotations 的语义不一致：150/150 帧 `mask_path` 已指向 object-owned mask，但外层 `area_px`/bbox 仍保留 source-track/raw 空间的值；不能把这些字段当作 owned-mask 元数据。

## 2. World 视图中的左右手是否反了？

### 数据与 renderer 链检查

- bridge 每帧严格按 `left`、`right` 两行排列；300 行 reference 的 frame/side mismatch 为 `0`。
- bridge 的完整 778 vertices 与对应 HaWoR `left_*`/`right_*` source arrays 逐值一致，左右两侧 max abs error 均为 `0.0 m`。
- world→camera 恢复完整 778 vertices 后，与 bridge camera vertices 的误差 median `2.09e-8 m`、max `3.32e-8 m`；不是外参方向反用。
- renderer 按 `hand_side` 读取同侧 bridge row，并使用同侧 `left_faces`/`right_faces`；读入时还显式检查 bridge 的 frame 和 side。

### 独立 handedness 证据

- HOT3D hand boxes：
  - 原标签 same-side IoU median `0.57923`；
  - 全交换 IoU median `0.0`；
  - 113 个两手均可比较的帧中，原标签 assignment **113/113 全部胜出**。
- HOT3D MANO 3D（300 frame-side rows）：
  - 原标签 wrist / MPJPE / root-aligned MPJPE median：`33.6 / 54.6 / 22.2 mm`；
  - 左右完全交换反事实：`243.1 / 234.0 / 84.7 mm`；
  - wrist、MPJPE、vertex-centroid 三项原标签均为 `300/300` rows 更好；root-aligned 为 `280/300` 更好，聚合结果仍强烈反对交换。

### 颜色易产生误读

代码里的颜色 tuple 是 **OpenCV BGR**，不是 RGB：

- `left = (245, 175, 55)`：显示时约为蓝/青；
- `right = (45, 145, 255)`：显示时约为橙/红。

再加上 world view 是外部观察视角，画面左右不等于佩戴者自身左右，所以容易产生“反了”的视觉印象。

### 结论

**左右手没有交换。** 当前证据同时排除了 bridge row 交换、source array 交换、renderer face-side 交换和 world→camera 方向反用。视觉上的“反了”来自外部 world-view 观察方向及 BGR legend/颜色认知，而不是 MANO 左右标签错误。

## 后续修复方向

1. raw SAM2 保留作为 tracker evidence，不因绿色 mesh 漂移而重跑/调参。
2. 将 hand ownership 从整矩形 subtraction 改为预测侧 hand silhouette/depth-order ownership，并同步重建 owned-mask 的 `area_px`/bbox 元数据。
3. 在 official HOT3D pinhole K/P03c 上重建 depth、visible geometry、camera/object trajectory，再做 first-hit z-buffer observed-only control。
4. 若 observed-only control 仍失败，先修 pose/camera/scale；不要继续把差异归因于 SAM3D vs TRELLIS。
5. world render 的 legend 明写 `LEFT (blue/cyan, OpenCV BGR)` / `RIGHT (orange/red)`，并标注视角是 external world observer。
