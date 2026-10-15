# 四个 S3 视频：MoGe / UniDepth 与 native 尺度归因复核

日期：2026-09-27。**本报告是冻结产物的评价，不是新的预测或尺度修复。** GT 只进入评价脚本；S3、原 source、模型/runtime snapshot、V7 均未改写，没有运行 PPO、policy 或 trajectory optimizer。

## 结论

四组继续支持**MoGe 与 UniDepth 没有统一成同一相机条件、同一逐像素几何观测**这一工程结论；但**不支持把 milk 的“native 偏小”泛化为所有视频，也不支持默认统一放大 1.14 倍**。

- 红铲：三轴变化方向不一致，深度/姿态/形状问题显著，不是一个 scalar scale 能解决。
- 白杯：同 camera frame 的 native AABB 比 CAD 大 12%–16%，但投影位置偏下、偏左；增加尺度虽提高部分 silhouette IoU，不能据此称物理尺寸改善。
- BBQ 酱瓶：MoGe 焦距仅高 2.4%，native camera AABB 不整体偏小；平移对轮廓的改善远大于改 K。校正单锚点深度后仍有剩余尺寸/形状问题。
- 汤罐：MoGe 焦距比 sensor 小 3.6%，native 轮廓已很好，但米制 AABB 大 34%–45%；1.14 倍放大明显破坏轮廓。它是“看起来贴合不等于米制尺寸正确”的反例。

**新的共同问题是：UniDepth 的米制前表面深度普遍偏远，且误差随时间变化；它会通过 scene similarity 同时放大 native 尺寸和 camera translation。投影基本不变，因此 silhouette 本身无法发现该米制偏差。** 这不等于所有几何误差都归因于 UniDepth，更不等于把任一路点图替换为另一条就能修复。

## 1. 确认使用的是 S3 的本批 v2 产物

存储前缀：`s3://lance-datasets/ego_annotation/intermediate_artifacts/packages/`。

| case（去掉共同前缀 hot3d_ 和后缀 _rl_v2_low） | appearance anchor | P13 metric anchor | source SHA256 前缀 |
|---|---:|---:|---|
| P0002_016222d1_spatula_red_f3285_3434 | 43 | 149 | 2ccd3ab43627 |
| P0012_73e66984_mug_white_f2145_2294 | 48 | 29 | 196a0a39be2f |
| P0012_ca1f6626_bottle_bbq_f1515_1664 | 0 | 138 | 6cbe591bb52e |
| P0015_7e95628d_can_soup_f1770_1919 | 111 | 106 | c4cbe3aa4f22 |

四包使用相同 producer code/config 身份，输入视频各自独立。白杯、红铲成功 S3 package 上传于 9 月 24 日；9 月 23 日的 `batch_final.json` 尚将其列为不完整，不能以旧汇总代替实际 S3 package。

每组从 S3 GET 172 个所需文件并校验内容 SHA256，同时只读比对原目录，共 688 个文件。评估副本保留原 JSON 字节、hash 和绝对路径；**这是部分评价输入的 byte copy，不是可迁址的完整 observation bundle**。没有调用 restore 覆盖原目录。

评价范围：四个 appearance 锚点的完整三角面投影及几何，四段各 150 帧、合计 600 帧逐像素深度；**没有声称完成 600 帧 native mesh 重放或新的 SAM3D 生成消融**。

## 2. 相机与模型观测确实未完全统一

四个 UniDepth 都用 `provided_pinhole_intrinsics`，输出为 `camera_z_m_from_metric_radius_on_exact_contract_rays`；source K、输出图像平面与 HOT3D pinhole K 一致，没有发现 mm/m、输出尺寸或换 K 重标深度的错误。报告中模型活动 ray 相对输入 K 的最大角差约 0.90°，最终 metric radius 显式沿输出契约精确射线转换 camera-Z。该契约不保证预测深度准确。

SAM3D 的 `pointmap=None` 使其独立运行 MoGe，没有输入 UniDepth 的 depth 或 sensor K。离线 MoGe-only 重放使用各包正式 checkpoint 与**同字节**锚点 RGB，结果另存，不修改生产 receipt。四组与 milk 的比较（960×960 图像；MoGe 的归一化主点按整数像素中心使用 `c*size - 0.5` 转换）：

| 视频 | sensor fx / px | MoGe fx / px | MoGe 相对 sensor |
|---|---:|---:|---:|
| 红铲 | 663.89 | 694.93 | +4.7% |
| 白杯 | 664.84 | 758.03 | +14.0% |
| BBQ 酱瓶 | 664.38 | 680.66 | +2.4% |
| 汤罐 | 664.76 | 640.75 | −3.6% |
| milk，前次评价 | 664.33 | 855.31 | +28.8% |

MoGe pointmap 按 sensor K 重投影回自身像素，目标区域误差中位数分别约 4.3 / 10.4 / 4.3 / 10.8 px。它不是同一组精确 sensor rays 上的深度。此误差还包含 `force_projection=False` 的非严格投影分量，不能全部记在焦距上。

## 3. 真实完整三角面：反驳通用放大修复

所有 native 比较保持各自同一 appearance anchor、源 mesh、RGB、mask，逐三角形 union rasterization；不使用凸包，不通过多 polygon even/odd 抵消制造轮廓。所有 mesh 顶点均明确检查 camera Z > 0.02 m；遇到穿近裁剪面则失败，而非静默丢面。native 面数分别为 90,100、624,436、347,460、1,267,564。GT 完整 CAD 用该帧 GT pose 只作 reference；物体轮廓单独渲染，没有声称手部 z-buffer 遮挡已处理。

以下均为对同帧 source appearance mask 的 IoU：

| 视频 | 正确 sensor K 原始 native | 仅换 MoGe K | 仅绕中心放大1.14× | 仅平移，尺寸/旋转/Z不变 | GT CAD 对照 |
|---|---:|---:|---:|---:|---:|
| 红铲 | 0.469 | 0.464 | 0.464 | 0.641 | 0.729 |
| 白杯 | 0.406 | 0.417 | 0.466 | 0.686 | 0.772 |
| BBQ 酱瓶 | 0.397 | 0.445 | 0.463 | 0.736 | 0.903 |
| 汤罐 | 0.910 | 0.874 | 0.738 | 0.921 | 0.956 |

平移反事实完全来自 source mask 与原 native silhouette 中心差，在 native median Z 下计算一次 camera XY 平移；不使用 GT，不优化轨迹，不变更 mesh topology。四组 camera XYZ 平移分别约 `(2.9,−20.7,0)`、`(12.1,−25.9,0)`、`(20.0,−4.4,0)`、`(1.9,1.0,0)` mm。它只是定位误差的敏感度测试，不是正式姿态修复。

红铲、白杯、酱瓶的平移收益明显大于只换 K/放大；它说明位置偏差是重要残差。红铲尚有形状/方向差异，白杯有手部遮挡、杯把与姿态差异，不能把中心对齐当作完整恢复。汤罐原轮廓已贴合，放大反而破坏轮廓。

P13 构造均含 P09 observed surface 和 native hidden prior 的混合，且四组 metric anchor **都与 appearance anchor 不同**；本次只在 P13 自己的 metric anchor 做记录，IoU 分别约 0.763 / 0.788 / 0.749 / 0.850，不能把这些与上表不同时刻的 native 数字作同帧胜负比较。

## 4. 同 camera 轴系的尺寸与米制偏差

表中是完整 native 与完整 CAD 在同一相机坐标轴下的 AABB 比值，**不是旋转不变的物体 intrinsic dimensions**；不同估计姿态/生成形状仍会影响它们。绝不跨 object-local、world、camera AABB 直接比较。

| 视频 | native / CAD camera XYZ AABB | 是否像 milk 一样整体小 |
|---|---|---|
| 红铲 | 1.062 / 1.363 / 0.728 | 否，方向混合 |
| 白杯 | 1.163 / 1.151 / 1.119 | 否，三个轴均大 |
| BBQ 酱瓶 | 1.060 / 1.060 / 1.220 | 否，三个轴均大 |
| 汤罐 | 1.452 / 1.338 / 1.397 | 否，三个轴均大 |

### 直接检查 bridge 的深度来源

将每个 appearance anchor 的 P09 retained camera surfel 投回原像素，仅在存在 GT target 可见 depth 的对应像素比较。这里用的是**比值：预测深度中位数 / GT 深度中位数**，不是“逐像素比值的中位数”；与全视频统计口径分开记录。

| 视频 | 匹配 P09 像素数 | 预测 median Z | GT median Z | 预测/GT |
|---|---:|---:|---:|---:|
| 红铲 f43 | 1,194 / 1,418 | 0.58789 m | 0.489 m | 1.202 |
| 白杯 f48 | 2,401 / 2,500 | 0.33960 m | 0.279 m | 1.217 |
| BBQ 酱瓶 f0 | 2,275 / 2,300 | 0.31592 m | 0.252 m | 1.254 |
| 汤罐 f111 | 2,459 / 2,500 | 0.26489 m | 0.201 m | 1.318 |

代码公式是 `lambda = median(P09 camera-Z) / median(native front depth)`。单锚点预测深度偏远可直接传递到 mesh 的米制尺寸与 camera-origin translation。用相同采样像素的 GT/预测中位深度比作为**仅评价**的 sensitivity，整体缩放 camera scene，四组完整三角面投影均逐像素不变。汤罐的 native/CAD AABB 从 `1.452/1.338/1.397` 降至约 `1.102/1.016/1.060`；白杯降至 `0.955/0.945/0.919`。这表明深度尺度能解释大量米制偏差，但剩余形状、方向和匹配采样差异仍在；BBQ 此敏感度后变为约 `0.845/0.846/0.973`，说明还叠加了其他机制，不能说全部误差由深度解释。

**没有输出 GT 修正后的 mesh，也没有把这些倍率写入 source。**

### 完整时间轴深度偏差

每帧采用 `source owned mask ∩ GT target visible depth`、正且有限的预测 depth，先算像素深度差中位数，再对 150 个帧中位数统计。结果单位 mm：

| 视频 | 150帧中位误差的中位数 | 同一值在5像素边界腐蚀后 | 帧中位误差 P05–P95 |
|---|---:|---:|---:|
| 红铲 | +109.0 | +108.8 | +42.8～+211.4 |
| 白杯 | +55.5 | +54.7 | +27.2～+111.7 |
| BBQ 酱瓶 | +53.9 | +53.7 | +17.8～+76.3 |
| 汤罐 | +53.3 | +53.2 | +41.6～+84.3 |

内部像素仍有类似偏差，不只是掩膜边界混入背景。时间轴存在明显波动，并非每个视频都像 milk 一样末段单调变差。GT 是官方 CAD/GT pose 渲染的目标可见前表面，不是直接深度传感器；未知遮挡、CAD 误差及同步限制仍应保留。

## 5. 对后续修复的约束

1. **支持并保留**：按同一 RGB plane/sensor K 审计 MoGe、UniDepth 和 native 输出，保存 MoGe K/pointmap provenance；不能只靠一个 depth ratio 就称它们统一。
2. **否定推广**：milk 的 +28.8% focal mismatch、native 偏小、1.14 visual scale 收益不是跨四视频通用规律。不得全局硬编码放大。
3. **优先分离变量**：source-mask residual 中的 camera XY/姿态偏差，与 camera-origin metric scene scale 和生成形状分别评价。良好 silhouette 不能验证米制尺度。
4. **不能盲目替换**：UniDepth 本身有显著 metric bias，所以直接以 UniDepth pointmap 取代 MoGe，不构成已证明的修复。sensor-conditioned MoGe/外部点图的生成阶段对照需在新的隔离 run 上执行，不带 GT 输入。
5. 本次没有做新的 SAM3D 条件化生成，故**仍未识别 MoGe 的 K 差异对生成 mesh 的独立因果贡献**；固定 mesh 只换 K 的检验是渲染敏感度，不是生成因果消融。

## 6. 可复现产物与保护检查

评估根目录：

```text
/mnt/truenas-user-home/kupingxin/rl_v2_cross_video_scale_audit_20260927/
```

- `inventory.json`：远端 manifest key、SHA、上传时间、original_root；四份 parsed S3 manifest 和 GET/SHA 记录在各 case 下。
- `fetch_evaluation_inputs.py`：TLS 验证开启、只 GET、不 PUT；没有打印或保存凭据。
- `replay_moge.py`、各 case 的 `moge_replay.npz/json`：A800 GPU0 的四张只读离线重放，正式 checkpoint/image/code hash。
- `evaluate.py`、各 case 的 `evaluation.json`：完整三角面和全部 150 帧深度指标。
- `sensitivity.py`、各 case 的 `sensitivity.json`：translation-only 与 depth-scale sensitivity。
- `cross_video_mechanism_comparison.jpg`：四行分别红铲/白杯/酱瓶/汤罐；三列分别原 native / 仅平移 / 1.14倍。蓝色填充=mesh，黄线=source mask，绿线=GT完整CAD轮廓。
- `depth_bias_all_600_frames.png`：600帧深度偏差，所有数字来自实际冻结数组。
- 各 case 的 `anchor_full_triangle_comparison.jpg`、`translation_only_comparison.jpg` 已检查，`anchor_rendered_silhouettes.npz` 可复核面积/IoU。
- `test_evaluation.py` 的6项数学/栅格测试通过，包含重复三角面 union、near-plane拒绝、camera-origin缩放不变、pivot缩放变化、AABB旋转非不变性；它们不代替几何质量判断。
- `final_verification.json`：688 个下载副本及原文件 hash 与 S3 清单一致，四个 source manifest hash 不变，600 帧有实际有效像素；MoGe 重放 hash 已验证。

未修改生产源码、原产物、GT、S3 或别人的缓存模块工作；没有提交/推送本次诊断。
