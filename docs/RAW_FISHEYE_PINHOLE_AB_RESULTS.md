# 原生鱼眼 vs 去畸变针孔：UniDepth 直接 A/B

日期：2026-09-27。评价专用，不修改生产代码/观测/S3/GT。输入来自同一原始 VRS 时间戳，使用同一 UniDepth checkpoint，GT 只在预测结果冻结后由评价进程加载。

## 结论更新

**直接 A/B 支持：在被测的五个锚点，当前“鱼眼转针孔”的输入表示分支会把 UniDepth 的深度预测向远处推，是偏远的一个影响因素。** 该分支同时改变投影表示、FOV、目标像素尺度、上下文和重采样，不能将结果缩窄成“去畸变公式错误”或“插值操作本身”导致。

- 在相同 upright 朝向、相同新提取/生成图像条件下，针孔相对鱼眼的**逐像素配对深度增量中位数**为 `+26.0 / +32.4 / +28.1 / +15.9 / +23.1 mm`（红铲/白杯/BBQ/汤罐/milk）。
- 修正本次发现的模型内部resize/像素中心转换后，该增量仍为 `+26.7 / +32.8 / +28.8 / +16.1 / +23.1 mm`，因此不是该小几何实现问题造成的伪差异。
- 90°图像朝向对模型也有明显影响，必须与去畸变分开。原始鱼眼/native朝向不能直接与针孔/upright朝向作为只有一个变量的A/B。
- 鱼眼中位偏差较小**不代表所有像素更好**：白杯、汤罐等有严重长尾。不能据五个锚点直接切换生产默认，更不能按每个视频的GT挑选最佳分支。
- 对既有S3编码输入，收益不能直接照搬：milk既有针孔源已经约 `+7.2 mm`，并不比鱼眼upright的 `+8.2 mm`差。

这更新了 `FISHEYE_RECTIFICATION_DEPTH_HYPOTHESIS.md` 中“尚未进行原生鱼眼A/B”的状态。原先的针孔crop试验无法捕获完整表示域差异；本次直接A/B补上了这项证据。

## 1. 实验设计和边界

五个锚点：红铲 f43、白杯 f48、BBQ酱瓶 f0、汤罐 f111、milk f0。它们对应前次实际appearance-native mesh的锚点，不按GT结果重新选择。

每组五种输入，第一轮复现既有模型推理行为：

| 变体 | 图像 | 模型相机 |
|---|---|---|
| A_fisheye_native | VRS原始鱼眼，原朝向 | 完整16参数FISHEYE624 |
| A_fisheye_upright | 相同鱼眼RGB，顺时针90° | 同步旋转后的完整FISHEYE624 |
| B_pinhole_native | 从该VRS帧按当前map生成针孔，原朝向 | 对应pinhole K |
| B_pinhole_upright_fresh | 上述针孔旋转正立 | 对应旋转后的pinhole K |
| B_pinhole_upright_published | 原正式包的同字节RGB | 正式pinhole K |

所有模型输入为960×960。fresh图像统一通过area resize，保存为PNG，不额外引入视频/JPEG编码；published控制保留原字节。A和B的**完整FOV不同**，这项A/B评价的是实际输入表示路径，不是单独的插值核。

第二轮在隔离进程内修正内部resize比例和像素中心转换，再跑相同25个输入。合计50次UniDepth-only预测；没有重跑SAM2、MANO、SAM3D、Astra、RL或优化器。

输入提取使用新建的独立工具环境：

```text
/mnt/user-home/kupingxin/ego_annotation_tools/.venv_aria_eval_1_5_1/
```

该环境仅用于Aria SDK解码/校准，不改生产模型环境。所有模型推理在当前A800服务器GPU0，checkpoint路径和SHA256写入推理清单；网络模型下载关闭。

## 2. 几何、时间戳和同射线比较

- 五帧 `capture_timestamp_ns` 均逐个匹配原manifest的device timestamp。
- 官方Aria SDK的FISHEYE624投影与冻结adapter map比对，最大误差小于 `0.0001 px`。
- 鱼眼旋转不是只改图像或K：主点、切向系数、thin-prism向量均随camera XY基变换；与官方SDK验证的CW90等价误差不高于 `1.2e-13 px`。
- 不给鱼眼RGB只传3×3 pinhole K；使用显式 `fx,fy,cx,cy,k1..k6,t0,t1,s0..s3` 共16参数。
- 评价选定同一个upright pinhole平面上的 `source owned mask ∩ GT visible target`，腐蚀5像素作为主要统计；所有变体使用同一组保留像素。
- 由冻结的去畸变map把这些物理射线映射到原鱼眼像素，再按每组图像朝向/resize映射到输入像素。
- 采样模型输出的metric radius，除以**同一条物理射线**的范数得到共同camera-Z。只绕光轴的90°旋转不改变camera-Z，避免把坐标方向差异当深度误差。
- GT为官方CAD/pose渲染目标前表面，不是直接深度传感器；只用于评价。共同内部像素数分别为11,819、49,967、37,306、100,445、162,086。

## 3. 原行为A/B结果

以下为共同内部像素的 `prediction camera-Z − GT camera-Z` 中位数，单位mm，正数表示偏远：

| 视频 | 鱼眼/native | 鱼眼/upright | fresh针孔/native | fresh针孔/upright | 正式包针孔/upright |
|---|---:|---:|---:|---:|---:|
| 红铲 f43 | +26.6 | +76.2 | +66.4 | +103.7 | +95.9 |
| 白杯 f48 | +12.2 | +28.6 | +59.7 | +61.9 | +58.6 |
| BBQ f0 | −19.4 | +39.3 | +47.1 | +66.0 | +64.2 |
| 汤罐 f111 | +2.1 | +54.0 | +24.9 | +70.6 | +63.8 |
| milk f0 | −41.3 | +8.2 | +34.9 | +29.7 | +7.2 |

严格控制朝向时，`fresh针孔 − 鱼眼` 的配对像素深度差中位数：

| 视频 | native朝向差值 | upright朝向差值 |
|---|---:|---:|
| 红铲 | +37.8 | +26.0 |
| 白杯 | +47.5 | +32.4 |
| BBQ | +65.0 | +28.1 |
| 汤罐 | +22.2 | +15.9 |
| milk | +77.0 | +23.1 |

这些是**逐像素配对差的中位数**，不等于上一表两列中位数相减。五个锚点的方向一致，支持完整去畸变/视场表示分支对偏远的贡献。不能用 `B_upright − A_native` 作为去畸变的独立效应，因为它同时改变朝向。

### 朝向效应

保持投影表示不变，`upright − native` 的配对像素差在鱼眼上分别约 `+51.6/+17.5/+56.6/+50.5/+47.8 mm`；在针孔上分别约 `+38.8/+4.7/+18.6/+43.8/−5.5 mm`。模型不是旋转等变的，且朝向与投影表示有相互作用。不能把模型在native侧向图上局部较准，直接推论为所有视频应永久侧向输入。

### 已发布输入差异

`published针孔 − fresh针孔` 的配对像素中位差约 `−7.4/−2.7/−1.8/−6.3/−22.7 mm`。它包含视频/JPEG编解码和历史导出链的字节差异，**不将其归因于某一个codec**。因此既有milk包表现好，并不意味着所有新生成针孔图会有同样深度。

正式包基线在本次原行为重放下，与原深度数组的绝对差中位数约 `0.059–0.089 mm`，证明确实复现了原模型行为，而不是比较了另一个checkpoint。

## 4. 额外发现：内部resize与像素中心问题

固定UniDepth源码 `unidepth/models/unidepthv2/unidepthv2.py::get_resize_factor` 在960输入、`pixels_max=600000`时：

```text
返回给camera.resize的ratio = sqrt(600000 / 960²) = 0.80687153
实际图像尺寸 = 向上取整到14倍数 = 784×784
实际图像ratio = 784/960 = 0.81666667
相对差 = 1.21396%
```

另有输入契约整数像素中心（0..W−1）与模型 `coords_grid` 半整数中心（0.5..W−0.5）的转换。它们解释了先前非零活动射线误差，而不是Aria去畸变公式出错。

本次只在独立推理进程中：

1. 对本组方形输入使用实际rounded图像ratio；
2. 输入模型前将主点从整数中心换成半整数中心（+0.5）；
3. 保持相同checkpoint、RGB、所有图像数组和后续评价，不修改安装源码。

校正后，目标区域模型ray与共同精确物理ray的误差中位数降至约 `6.5e-6～2.6e-5°`，全图fp32角度检查最大约 `0.040°`。该进程补丁仅覆盖本次方形等比例输入，**不是可直接推广到任意长宽比的完整生产修复**。

| 视频 | 原行为 upright B−A | 射线校正后 upright B−A |
|---|---:|---:|
| 红铲 | +26.0 mm | +26.7 mm |
| 白杯 | +32.4 mm | +32.8 mm |
| BBQ | +28.1 mm | +28.8 mm |
| 汤罐 | +15.9 mm | +16.1 mm |
| milk | +23.1 mm | +23.1 mm |

**A/B差异在严格射线校正后仍存在。** 此实现细节应作为独立问题修复，但它不是此次大部分深度偏远的主因。校正后的结果不能冒称原S3重放，因为推理机制已经有意识改变。

## 5. 不可只看中位数：鱼眼长尾

原行为、upright控制的平均绝对深度误差（MAE）：

| 视频 | 鱼眼upright MAE | fresh针孔upright MAE | 已发布针孔 MAE |
|---|---:|---:|---:|
| 红铲 | 83.2 mm | 103.5 mm | 96.0 mm |
| 白杯 | 59.1 mm | 63.0 mm | 60.0 mm |
| BBQ | 37.4 mm | 66.5 mm | 64.5 mm |
| 汤罐 | 71.1 mm | 71.8 mm | 65.2 mm |
| milk | 8.4 mm | 29.9 mm | 7.5 mm |

白杯、汤罐鱼眼/upright的绝对误差99分位分别约 **1.239 m / 0.595 m**；fresh针孔约 **0.110 m / 0.100 m**。图上的鱼眼多数目标像素偏差更小，但细边、遮挡邻近和复杂局部仍可能出现很大的深度尾部。本次保存了全部像素，不以剔除坏点来美化指标。

所以不能直接宣布“鱼眼整体更好”，不能仅凭均值/中位数选择默认深度源，也不能按GT挑分支。

## 6. 后续工程含义

- 可以把“当前去畸变/视场表示分支影响UniDepth米制深度”提升为**有直接A/B支持的原因之一**，而不是继续仅列为猜测。
- 不应将原因表述为“去畸变几何公式错了”或“应该给深度除以1.6”。官方投影/旋转检查正确，模型输入表示敏感性才是此次A/B检测到的效应。
- 先对独立相机resize/像素中心问题写生产侧修复及回归；另外在更多帧上评估raw-native/raw-upright/pinhole的长尾与时序一致性，才决定新版本是否调整深度来源。
- 若后续采用鱼眼深度，也必须映射到统一的source pinhole物理射线与camera/world契约后给P09使用，不能直接塞入不同像素平面。
- 当前仍只有5个预先选定锚点，**没有完成5段全视频/750帧A/B**；也没有把新的深度接入完整mesh/MANO/RL管线，不能声称最终标注质量已修复。

## 7. 产物与复现

```text
/mnt/truenas-user-home/kupingxin/rl_v2_raw_fisheye_ab_20260927/
```

- `extract_inputs.py`、`input_manifest.json`：VRS时间戳、完整相机参数、官方SDK数值检查与25组输入hash；不含GT物体/手部pose或depth。
- `infer_ab.py`、`inference_manifest.json`：25次原行为预测、checkpoint/代码/输出hash。
- `infer_raycorrected.py`、`inference_raycorrected_manifest.json`：进程内几何校正的25次独立复测。
- `evaluate_ab.py`、`evaluate_raycorrected.py`：冻结预测之后，同物理射线、同GT目标前表面评价。
- `ab_summary.json`、`ab_raycorrected_summary.json`、`tail_error_statistics.json`：中位、均值、配对差及长尾。
- 各case的 `common_ray_samples.npz`、`common_raycorrected_samples.npz` 保存共同像素及实际预测/GT数值，未导出GT修正的生产网格。
- `ab_overview.png`：朝向和相机分支对照；各case `ab_QC.png` 为真实输入与共同针孔ROI误差热图，含MAE和坏像素比例。热图色阶在±150mm饱和，超范围尾部在JSON中完整保留。
- `verification.json`：全部输入/预测hash、5个原source hash、安装模型源码及checkpoint未变的核验。

正式source、GT、S3、runtime snapshot、V7未修改；未提交/推送任何生产行为修复。
