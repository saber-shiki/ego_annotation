# 鱼眼去畸变是否导致 UniDepth 深度偏远？

日期：2026-09-27。基于五个 HOT3D pinhole 输入的生成代码、四个 S3 冻结观测包和本次评价专用对照。**没有改变正式输入、source bundle、S3 或 GT，没有运行训练/策略/轨迹优化。**

**同日后续更新：原生鱼眼直接A/B已完成，见 `RAW_FISHEYE_PINHOLE_AB_RESULTS.md`。** 五个锚点在同朝向fresh输入下，针孔分支相对鱼眼预测更远约16–33mm；发现并在隔离进程校正内部resize/像素中心细节后，差异仍然存在。本页保留此前几何核查和针孔裁切实验的记录，其“原生鱼眼A/B尚未执行”的段落是当时状态，不再代表最新进展。

## 原生鱼眼A/B之前的判断

**不能排除间接影响，但当前没有证据支持“去畸变的几何实现把深度系统放大”这一直接归因。** 还须分开两种假设：

1. **几何/单位错误**：图像去畸变后仍用鱼眼 K、90°旋转未同步相机、resize 后 K 不变、把径向距离当 camera-Z、错误重映射 GT 深度等。
2. **模型对输入分布的敏感性**：即使几何正确，虚拟相机 FOV、裁切、重采样、可见上下文变化，也可能影响模型学习得到的 metric 深度。

本次几何检查未发现第一类明显错误。正确更新 K 的针孔裁切对照证明第二类影响存在，但在所测范围内只有毫米级变化，未消除原有约60–100 mm的偏远。**这仍不是原生鱼眼与去畸变针孔的直接 A/B，不能据此完全排除原始去畸变带来的域变化。**

## 1. 实际去畸变链

五个输入统一使用：

```text
原始 Aria RGB，官方 online FISHEYE624 参数
  → destination pixel 按固定 pinhole K 反投影成射线
  → FISHEYE624 正投影查找原始 RGB 像素
  → cv2.remap(..., INTER_LINEAR)
  → upright-cw 90° 图像旋转 + 同步 camera rotation/K
  → 1408×1408 pinhole RGB，f≈974–975 px
  → 预测端 960×960 RGB，明确半像素仿射，f≈664–665 px
```

`selection_config.json` 的 `pinhole_focal_scale=1.6` 是生成虚拟针孔视场的选择，水平 FOV 约71.7°；**不是把深度乘1.6**。同一 camera center、保持光轴、只绕光轴旋转的此流程不会因像素重映射而改变同一点的 camera-Z；图像尺寸改变必须同步 K，此处已做。

相关冻结代码与记录：

- `.../hot3d_pinhole_rgbd_selection_v1/tools/reference_build_v19_hot3d_vrs_slice_adapter.py`：`build_maps`、固定 K、`np.rot90` 和 camera rotation。
- `.../hot3d_pinhole_rgbd_selection_v1/tools/build_rgb_adapters.sh`：`--focal-scale 1.6 --orientation upright-cw`。
- 各 clip 的 `state/calibration/v19_camera_calibration_contract.json`、`input/raw_frame_manifest/manifest.json`。
- `observations/camera/unidepth.py`：`infer(image, camera=K)`，metric radius沿精确输出射线转camera-Z，再输出到 source plane。
- 当前固定 UniDepth checkout `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` 的 decoder：有输入camera rays时使用该rays，不使用camera-head自估rays替代。

GT目标深度由CAD/GT姿态**直接在目标 pinhole K 下渲染 camera-Z**，不是把原始鱼眼 depth 当RGB简单remap。该GT仍是模型化动态前景的评价参照，不是完整传感器深度真值。

## 2. 数值几何核查

五个锚点（红铲43、白杯48、BBQ0、汤罐111、milk0），各抽取65×65=4,225条射线：

- 冻结 NumPy adapter 的映射，与固定 UniDepth `Fisheye624.project` 的独立实现对比，最大差 **0.0000811 px**。
- `CW90`图像像素变换与旋转后的 K/相机射线对应，最大误差约 **5.7e-14 px**。
- 该90°旋转精确保留 camera-Z。
- 五段源报告及所测map均有 `warp_valid_fraction=1.0`。

限制：这是两个 FISHEYE624 投影实现的数学对照，**不是 Aria SDK 全流程图像解码重放，也没有把完整预测正确性归因于几何测试通过**。当前已知模型环境未安装 Aria VRS SDK，本次未新装或修改生产环境。

## 3. 正确更新 K 的针孔裁切实验

预测端实验只读四个 S3 包的正式锚点RGB、同一 UniDepth checkpoint和frozen wrapper，在A800 GPU0上运行，无GT输入。每张图做额外 `zoom=1.0 / 1.15 / 1.30` 的中心裁切放大，图像输出仍为960×960，同时令 `K_new=A_crop×K_old`。

评价端在三种视场**共同保留的同一批原始物理像素/射线**比较：source owned mask与GT target可见区相交，先腐蚀5像素以减少边缘混合。每条射线用对应新像素采样模型metric radius，再沿相同原始射线还原camera-Z。不同zoom的归一化射线最大差小于 `1.7e-16`。

基线重放与S3正式depth在相同像素的绝对差中位数为 `0.058–0.081 mm`，95分位不高于 `0.222 mm`，与float16储存精度相容。说明本次对照没有实质更换模型或输入契约。

同一组物理像素上的 `prediction camera-Z − GT camera-Z` 中位数，单位mm：

| 视频锚点 | 共同像素数 | 原视场1.0 | 裁切1.15 | 裁切1.30 |
|---|---:|---:|---:|---:|
| 红铲 f43 | 11,819 | +95.9 | +88.7 | +89.1 |
| 白杯 f48 | 49,967 | +58.6 | +56.2 | +64.2 |
| BBQ酱瓶 f0 | 37,306 | +64.2 | +67.5 | +65.5 |
| 汤罐 f111 | 100,445 | +63.8 | +63.0 | +58.6 |

这四个锚点在额外改变15%–30%虚拟焦距并保持射线契约正确时，深度偏远仍明显存在；方向并不统一，也没有出现深度随焦距倍率线性放大的现象。**所测范围不支持“固定焦距缩放/裁切错误是大部分偏远的唯一原因”。** 模型对裁切及上下文仍有数毫米敏感性，因此不能说去畸变绝无间接影响。

这里测的是**已经去畸变图像上的更窄FOV**，没有增加原针孔图外的真实内容；不能冒充更宽FOV或原生鱼眼分支的结果。每视频仅一个锚点，不构成跨600帧的裁切敏感性实验。

## 4. 完整因果判别所需下一实验

要直接判断原始去畸变是否使模型偏远，应使用同一VRS时间戳、同一checkpoint：

- A：原生鱼眼 RGB + 完整 FISHEYE624 camera 参数；
- B：当前去畸变 pinhole RGB + 正确目标 pinhole K；
- C：从同一原始鱼眼重新生成不同目标FOV的针孔图，不通过黑边补图伪造宽视场。

将三者输出的点/radius按各自相机模型统一到共同camera坐标和共同物理射线，再作GT-only评价。**不能把原鱼眼图直接送进只带3×3 pinhole K的分支**，那会把实验变成另一个相机模型错误。

结果解释：若A明显改善而B/C保持偏远，才加强“rectification/表示域”假设；若A/B/C都偏远，说明更接近模型在近距离手持/遮挡场景上的metric尺度偏差或共有相机/评价问题；若偏差随FOV改变且在重采样对齐后可重复，则可归因于FOV/上下文敏感性的一部分。不要提前把任何一种结果写成结论。

## 证据与来源

评价目录：

```text
/mnt/truenas-user-home/kupingxin/rl_v2_cross_video_scale_audit_20260927/rectification_hypothesis/
```

- `geometry_checks.json`：五个锚点独立投影/旋转检查。
- `replay.json`、每case的`zoom_*.npz/jpg`：四锚点×三裁切的预测冻结结果、输入和输出hash。
- `crop_sensitivity_results.json`：同射线GT评价与S3基线重放误差。
- 上级脚本 `rectification_crop_replay.py`、`rectification_crop_evaluate.py`、`check_rectification_geometry.py`。

官方资料已核查：Project Aria Image Utilities说明先定义目标pinhole calibration再undistort，顺时针旋转后同步更新calibration；UniDepth固定commit README明确3×3 camera输入按pinhole解释，鱼眼须提供`Fisheye624`对象。

```text
https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/advanced_code_snippets/image_utilities
https://github.com/lpiccinelli-eth/UniDepth/tree/8d8cfe4c7ee15297099983607febf0d4f32eb3d6
```

本次只完成可用现有环境中的几何检查及针孔crop控制，**尚未完成原生鱼眼A/B，未证明去畸变是主因，也未彻底排除间接域影响**。
