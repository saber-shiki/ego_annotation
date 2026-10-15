# HOT3D CAD 与 MoGe / UniDepth 尺度归因

2026-09-27。仅作 evaluation-only 诊断：HOT3D CAD/GT 深度及位姿没有进入 v2 source、模型输入、RL candidate 或训练。本报告修订 2026-09-23 初稿的错误坐标比较。

**外推限制（同日四视频复核）**：见 `CROSS_VIDEO_MOGE_UNIDEPTH_DIAGNOSIS.md`。四个 S3 视频确认两路观测未统一，但不支持普遍 native 偏小或默认放大 1.14 倍；汤罐是直接反例，另有显著 UniDepth metric 深度偏差与 camera XY 位置误差。本报告的 milk 单例结论不能替代该跨视频结论。

## 初稿更正

初稿把 CAD **object-local** AABB 与 P09 **camera/world** AABB、P13 **canonical** AABB 和 native **camera** AABB 按长度排序后直接相除，称“P09 横截面在 SAM3D 之前偏大 41% / 51%，native 不是统一偏小”。**该因果结论错误，已经撤回。** 排序后的 AABB 不是旋转不变量，AABB 对角线也随坐标旋转而变。原始数值仍在外部 `cad_gt_scale_comparison.json`，只能作为未对齐坐标系的原始测量，不能据其 `raw_ratio_to_gt_sorted` 等字段判断物理尺寸。

用 HOT3D GT 可见深度在 frame 0 反投影，在相同 camera 坐标逐轴比较（mm）：

| 几何 | camera X | camera Y | camera Z |
|---|---:|---:|---:|
| GT 可见表面 | 102.4 | 192.9 | 110.0 |
| P09 可见 surfels | 102.1 | 194.2 | 109.1 |

SAM2 源 mask 与 GT 可见 mask IoU 为 `0.939`。同像素 UniDepth 相对 GT 深度中位偏差为 `+7.22 mm`、约 `+2.5%`。因此**没有证据说 frame 0 P09 横截面大 40%–50%**；在可见表面 AABB 上它反而很接近 GT。

将**完整** CAD GLB 用官方 GT pose 转到同一 frame 0 camera 坐标，再与 native metric-camera PLY 比较（mm）：

| 完整几何 | camera X | camera Y | camera Z |
|---|---:|---:|---:|
| HOT3D CAD GT | 102.83 | 201.74 | 121.53 |
| SAM3D native | 86.25 | 172.26 | 106.85 |
| native / CAD | 0.839 | 0.854 | 0.879 |

在**这个共同坐标系、该帧相同轴向**下，native 三轴近似统一偏小约 `12%–16%`。GT 后验 AABB 最小二乘倍率约 `1.166`，仅作诊断，不能作为预测输入。P13 是 canonical mesh，未经 pose/frame binding 的尺寸不能直接宣称等于 CAD 的任何 camera/local 轴向。将当前 v2 P13 按 receipt 的 anchor centroid 和 frame 0 相机姿态明确变到 camera 后，其完整三角面按 sensor K 投影的面积为 `88,513 px`、与源 mask IoU `0.926`。这比 native 的 `0.704` 贴合，但 P13 不是单纯原样 native：completion 用 `7,026` 个 P09 观测表面三角面替换了 `210,607` 个 native 观测区候选面，只保留 `263,266` 个生成的隐藏面作 render/pose hypothesis，隐藏面不获得 collision 权威。因而 P13 与 native 的差异还包含明显的前表面拓扑和几何角色变化。

## 两套模型的实际数据流

`main.py` 先得到 sensor K，运行带相机条件的 UniDepth；其 `960×960` 输入 K 为 `[664.326, 664.326, 478.928, 481.280]`，输出转换到 `1408×1408` source RGB 平面的 K 为 `[974.345, 974.345, 702.661, 706.110]`。UniDepth 的 metric radius 沿**输出平面的精确 sensor rays**转换为 camera-Z，再绑定给 P09。绑定验证了相机契约 hash、仿射、K、图像平面，且 `metadata_only_ray_relabel_override=false`。模型的 camera-head 自估 K 仅用于诊断，不能当活动 K；报告中活动 ray 最大角差约 `0.90°`。这保证了内部输出契约的相机一致性，**不保证其 metric depth 准确**。

SAM3D 接收同一帧的 RGB 与 mask，但调用 `Inference(image, mask, seed)` 时 `pointmap=None`。其内部 `compute_pointmap()` 独立运行 MoGe `infer(..., force_projection=False)`，利用 MoGe pointmap 和其自估相机生成 SAM3D pose/shape；**没有注入 UniDepth depth、sensor K 或 camera-conditioned pointmap**。MoGe 的 native scene units 不等于米。随后 `scene_similarity()` 用 P09 的单帧中位 camera-Z 除以 native mesh 投影的中位前表面深度，对完整 native camera-origin pose 乘同一个比例，r2 为 `0.32766085`。这只绑定锚点前表面深度，并没有统一两套相机射线、形状或多帧 metric 深度。

用正式 MoGe checkpoint 和 SAM3D 的同一 `960×960` 输入 RGB 做离线 MoGe-only 重放，自估 `fx=fy=855.311 px`、`cx=cy=480 px`，水平 FOV `58.60°`；sensor/UniDepth 为 `fx=fy=664.326 px`、FOV `71.70°`。焦距相差约 `28.8%`。把 MoGe 深度中位数诊断性地归一到 UniDepth 后，同 mask 像素的深度差第 5/50/95 分位约 `-9.18/+2.14/+12.02 mm`，Pearson 相关约 `0.620`。这是不同尺度与射线模型下的离线比较，不应视为已经逐像素物理对齐。正式 run 未保存 MoGe 的 pointmap/K receipt，因此此重放需要另行审计，而不能称作正式 source 测量。

## 受控完整三角面投影

固定 frame 0 的**同一完整 native mesh 与 pose**，只改变投影 K；另外单独做绕 mesh 中心 `1.14×` visual-scale 的反事实。mask、RGB、物体 pose 和 mesh topology 均不变。完整 CAD 用官方 GT pose 与真实 sensor K 仅作评价：

| 投影条件 | 面积 | 源 mask IoU | GT 可见 mask IoU |
|---|---:|---:|---:|
| native + sensor K（当前真实渲染） | 64,692 px | 0.704 | 0.726 |
| v2 P13 + source anchor/frame 0 + sensor K | 88,513 px | 0.926 | 未计算 |
| native + MoGe K（仅相机反事实） | 107,029 px | 0.764 | 0.767 |
| native 绕中心 `1.14×` + sensor K（仅尺寸反事实） | 84,987 px | 0.803 | 0.848 |
| CAD 完整 mesh + GT pose + sensor K | 85,208 px | 0.911 | 0.960 |

MoGe K 会把同一 mesh 在图上投得**过大**，也不能用它替换真实相机 K。sensor K 下的 native 确实太小；两套 K 不统一对投影尺寸的影响已被固定 mesh 的反事实证明，**但不等于证明 MoGe 内部 K 差异是生成 mesh 物理尺寸偏小的唯一原因**：生成形状、旋转、深度、遮挡也参与。绕 mesh 中心放大改善该帧 mask 且三个方向更接近 CAD，但 `1.14` 只是无 GT appearance 的粗略诊断值，不能偷偷写成物理真值。camera-origin 的 `lambda` 乘在 translation 和 scale 上，**无法改变投影面积**；visual scale 须明确 pivot，并只用于 render prior。

[Frame 0 完整三角面 K 对照图](/mnt/truenas-user-home/kupingxin/rl_v2_a800_execution_20260921/moge_vs_sensor_k_full_triangle_frame0.png) 显示真实 RGB 上的 GT 可见区域、sensor-K native、MoGe-K native 三者的覆盖，不是最终产品渲染。

## 独立的时序偏差

在相同 GT target 可见像素上，UniDepth 相对 GT 的深度中位误差分别约为 frame `0:+7.2 mm`、`60:+10.8 mm`、`94:+21.9 mm`、`120:+36.1 mm`、`149:+39.2 mm`；这是随时间变化的 metric 深度偏差，**与 frame 0 固定 native mesh 尺寸问题不同**。frame 94 P09 有明显远尾部，已因 geometry extent 被标为不具备 rigid pose eligibility。单一全片 mesh scale 无法解决这些逐帧深度误差。

## 处理边界与下一步

1. 不以 GT CAD 或后验 `1.166` 校准 predictor；不改 GT、r2、历史 P13、V7 或 runtime snapshot。
2. 为 SAM3D 添加只读审计输出：每次保存其内部 MoGe 的图像平面、K、pointmap 摘要和输入 hash。只有在独立的新 run 才可能采用 sensor K 条件化的点图，并与原生成结果在同一 renderer 下做受控比较；不能把内部 K 简单替换为 sensor K 去重标已有点图。
3. 在不使用 GT 的多帧 source pose + appearance 下，检验有界、统一、绕明确 pivot 的完整 native visual scale；同时审计残余 silhouette、前表面 depth、时序一致性，边界/冲突要标记 unresolved。它始终不是 signed/collision/contact authority。
4. 对 UniDepth 后续帧的 metric 深度漂移单独处理，不应通过逐帧 mesh scale 吸收。

证据：

- [外部机器可读初版 AABB 清单](/mnt/truenas-user-home/kupingxin/rl_v2_a800_execution_20260921/cad_gt_scale_comparison.json)：数值原样保留，**跨坐标 `raw_ratio_to_gt_sorted` 等解释无效**。
- [HOT3D CAD/GT 报告](/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/clips/P0014_84ea2dcc_carton_milk_f2370_2519/depth_gt/depth_export_report.json)：CAD SHA256 `e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc`。
- [v2 视觉观测 receipt](/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2/measurements/visual_geometry_observations/milk_carton/visual_geometry_observation.json)，[UniDepth 相机/深度报告](/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2/measurements/depth_slam/unidepth_full_frame/qc_unidepth_full_frame_v3.json)。

仅完成归因和反事实视觉检验，未对正式 mesh/source 施加尺度修正，未启动 PPO。
