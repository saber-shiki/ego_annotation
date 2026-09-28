# 前处理继续修改：当前默认、失败实验及验证

日期：2026-09-27。工作区 `/mnt/user-home/kupingxin/ego_annotation_preprocess_monocular`，分支 `feature/monocular-coarse-preprocess-20260927`。

## 结论

**本轮没有取得显著的GT精度突破。** 当前默认保留原R2局部鲁棒跟踪，并加入经过共同留出集验证的跨关键帧描述子约束；它避免了本轮早期候选选择逻辑造成的大幅退化，但仍不满足“RL只负责接触微调”的粗位姿精度要求。几何多假设分支真实实现并跑完五例，但白杯出现“轮廓更好、物理姿态更错”，已从默认路径撤回，保留为显式实验选项。

“默认”只指新独立命令 `python -m preprocessing`，不表示原始观测链已经切换到本模块。原主分支、S3、source、MANO和RL未改；未提交、合并或推送。

## 保留的实质修改

1. `preprocessing/relocalization.py`：互相匹配的内纹理RootSIFT关键帧库，保留appearance anchor及长基线，不再让最近几个帧挤占全部参考。
2. 多个PnP候选从同一鲁棒训练支持产生；当前位姿与候选都在相同的留出像素身份上评分，不以各自筛出的少量内点中位数自证正确。
3. 独立几何复核使用固定遮挡评价域、同一可见轮廓及source表面；不拿“优化后分数”替没有返回的优化位姿背书。
4. 接受重定位后替换对应帧旧的3D绑定，加入真正被联合求解器消费的source/target双端观测行。当前五例没有出现接受位姿却缺少后端对应行的情况。
5. 恢复R2已有RANSAC局部跟踪，删除R3–R6的不公平候选评分、全点精化污染及25°/60mm校正幅度硬限制。该限制并不等于真实相邻帧运动限制。
6. 补充候选输入不变性、外点、伪低中位误差、关键帧多样性、旧绑定替换、端到端描述子/JSON、低纹理三角面重定位及默认关闭失败实验的回归测试。

## 真正尝试但未通过的几何路径

`preprocessing/shape_relocalization.py` 使用完整真实网格、24个proper SO(3)朝向、多帧轮廓/表面评分及离散时间路径；不是包围盒、点云或质心替代物体pose。合成非对称mesh的90°错误可恢复，不能把这个单测扩展成五例真实精度通过。

白杯在所测关键帧只有3–9个内纹理特征，属于确有不足的纹理观察，不应靠降低PnP门槛伪造6DoF。几何路径在R9产生预测侧轮廓改善，但独立GT评价反驳了其姿态质量：

| 白杯 | R2 | 几何实验R9 |
|---|---:|---:|
| source-mask IoU中位数 | 0.697 | 0.737 |
| 轨迹旋转误差中位数 | 110.73° | 141.94° |
| 轨迹平移误差中位数 | 68.75mm | 85.76mm |

因此该分支默认关闭。只有明确传入 `--experimental-geometry-relocalization` 才能重现此实验，不能用于宣称已达标或自动向RL提供合格起点。没有按GT逐case挑选结果或回填GT旋转/尺度。

## 最新默认R10与R2的公平对照

以下均为前次相同共同有效帧集上的单锚点对齐轨迹误差中位数，**不是绝对初始pose误差，也没有消除对象对称性**。外部JSON同时保存全部GT质量合格帧统计；不通过删除失败帧改善数字。

| 视频 | 旋转 R2→当前 | 平移 R2→当前 | 实际接受的重定位帧 | 新增双端观测行 |
|---|---:|---:|---:|---:|
| milk | 4.68° → 4.68° | 11.99 → 11.99mm | 0 | 0 |
| 汤罐 | 9.03° → 8.79° | 152.11 → 152.36mm | 11 | 7436 |
| BBQ瓶 | 45.21° → 45.20° | 44.56 → 44.28mm | 5 | 278 |
| 白杯 | 110.73° → 110.73° | 68.75 → 68.75mm | 0 | 0 |
| 红铲 | 94.43° → 94.43° | 231.17 → 231.17mm | 0 | 0 |

观测行包括已一致时加入的loop约束，每对匹配有source/target两行，不是把7436行称为7436条独立轨迹边。milk/白杯/红铲当前没有通过验证的新重定位，保留R2结果；汤罐/BBQ的数值改变很小，不构成实质精度提升。

## 验证范围

- 当前源码与R10冻结快照逐文件一致。完整回归174项：170通过、4跳过；预处理专项31项全部通过。
- 五例每段150帧；10个overlay/world视频共1500帧完整解码，源MANO顶点/关节/面数组逐项未变。
- source manifest hash保持不变；预测读取记录未包含GT/CAD/评估目录。GT只由独立评价脚本在预测完成后读取。
- 全程服务器CPU，不启动新观测模型、不新增相机输入、不修改训练或reward。
- 已查看R2与当前默认的同帧mesh/MANO对照；默认结果没有显著可见改善，这与量化结果一致。
- R3/R3b等早期烟测中的OpenCV返回值及JSON数组序列化错误已修复并保留失败记录；有些probe仅为单case/不渲染，不冒充完整五例验证。

## 路径与复现

最新默认输出：
```text
/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_coarse_20260927_r10_default/
```
冻结代码：
```text
/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_coarse_20260927_r10_default/
```
外部评价与完整性验证：
```text
/mnt/user-home/kupingxin/ego_annotation_experiments/monocular_coarse_evaluation_20260927/monocular_coarse_20260927_r10_default
```

各case下有 `coarse_scene.{json,npz}`、`relocalization.json`、`tracking.json`、`solver.json`、`shape_relocalization.json`，以及两个完整视频。所有先前尝试保留在各自新目录，没有覆盖source或旧结果。

```bash
cd /mnt/user-home/kupingxin/ego_annotation_preprocess_monocular
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -m preprocessing \
  --source-observations /ABS/FROZEN_SOURCE/source_observations.json \
  --output-dir /ABS/NEW_OUTPUT
```

默认执行经过验证的描述子重定位；`--no-relocalization`是局部跟踪消融；`--experimental-geometry-relocalization`是默认禁用的几何实验。这两个开关不能同时使用。

## 尚未解决

大角度错误和厘米/分米级位移仍在，不能让RL扩大动作来接管这些问题。当前结果说明：修复候选控制和增加少量可靠跨帧匹配不足以弥补弱纹理、局部遮挡、生成形状误差及单目深度尺度偏差。形状轮廓的正确性还不能稳定消除材料朝向歧义；仅凭当前预测侧一致性不能宣布物理精度已收敛。下一次机制必须针对这些实际可观测性/建模问题，而不是继续调候选阈值。
