# 固定UniDepth可见表面的SAM3D尺度配准：实现与五例结果

日期：2026-09-27。独立工作区 `ego_annotation_preprocess_monocular`；代码和输出均未合并或推送。

## 结论

按用户确认的参考关系实现并运行了新 `visible-surface` profile：**UniDepth原始深度与合格去遮挡可见表面固定不动，让SAM3D完整网格拟合一个共享统一尺度和独立平移。** 不改MoGe、不重跑生成模型、不把局部可见点云包围盒当作完整物体尺寸。

五例均跑通。相对于同一批UniDepth可见表面，多个锚点的对应表面误差显著下降，投影覆盖增加；红铲仍有约26mm的锚点残差，说明其问题不只是一个尺寸倍率。此结果证明了配准操作可用，不证明整个标注已经达到真实毫米级精度。

## 具体实现

- `preprocessing/visible_surface.py`：实际三角面第一命中/可见面对应；统一正尺度与独立平移；命中区域用点到平面残差，未命中区域保留可见面及覆盖约束；多帧共享尺度。
- `preprocessing/surface_pipeline.py`：同帧锚点配准、分布式关键帧共享尺度、冻结尺度后的全时间轴平移配准及完整导出。
- `preprocessing/inputs.py`：同帧camera/K核对，实际加载去遮挡P09 masks，记录输入哈希。
- `preprocessing/__main__.py`：新增明确的 `--profile visible-surface`；legacy仅保留对照，不偷偷改变其语义。
- `preprocessing/render.py`：此profile的before是“配准前RGB位姿提案”，after是“固定UniDepth参考后的对齐”；实际网格/MANO完整重放。

第一轮测试发现，冻结点到点对应会错误锁住平面内XY位移，已改成命中表面的点到平面约束。第一轮真实运行还显示，先改模板尺寸再重新绑定跟踪点可能改变PnP旋转分支。最新实现先从原模板计算既有RGB旋转提案，然后在整个配准中冻结R，只拟合共享s和t；运行时与独立验证均确认所有旋转矩阵逐值不变。

新尺度相对于已含旧scene-similarity的源visual mesh，因此只应用一次残余尺度；manifest分别记录原scene scale、新residual scale和combined length scale，位移另行求解。

## 最新共享尺度与锚点效果

误差为最终输出在同一锚点、**配准前后都命中的同一批观测射线**上的3D距离中位数。参考是固定UniDepth/P09，不是GT深度。覆盖率单独列出，避免用更少的命中点假装误差更好。before/after旋转、K和相机相同，改变的是尺度与位移。

| 对象 | 共享尺度倍率 | 同射线3D误差 前→后 | 可见点命中覆盖 前→后 |
|---|---:|---:|---:|
| milk | 1.1828 | 16.06 → 3.92 mm | 81.8% → 96.9% |
| 汤罐 | 0.9678 | 11.52 → 5.36 mm | 93.6% → 94.9% |
| BBQ瓶 | 1.1816 | 24.31 → 3.07 mm | 79.0% → 94.1% |
| 白杯 | 0.9482 | 38.24 → 3.94 mm | 65.4% → 79.2% |
| 红铲 | 1.2070 | 38.54 → 25.67 mm | 62.2% → 77.0% |

这些不是统一乘1.14或按类别写死的倍率，均来自该视频固定可见表面的拟合。倍数小于1表示模型缩小；等比例应用于同一网格全部帧，未使用逐帧大小变化。

## 全时间轴固定参考诊断

每帧在固定的合格可见样本集合上计算距离，再对帧中位数统计。此指标对命中射线使用前表面对应，对未命中射线使用可见面回退，因此需与覆盖率一起解释，不能冒充精确的全模型Chamfer或GT姿态误差。

| 对象 | 有效表面帧数 | 帧中位距离的中位数 前→后 |
|---|---:|---:|
| milk | 148 | 46.70 → 5.88 mm |
| 汤罐 | 135 | 28.55 → 3.01 mm |
| BBQ瓶 | 143 | 74.39 → 20.34 mm |
| 白杯 | 136 | 41.59 → 22.82 mm |
| 红铲 | 108 | 84.81 → 19.82 mm |

锚点改善不代表全片所有帧已经贴合：BBQ、白杯和红铲的全片残差仍较大，受原旋转提案、形状差异与遮挡影响。当前模块不通过改写深度、强制非均匀拉伸或删除失败帧掩盖这些问题。

## CAD / GT pose补充评价

仅在预测完成后独立读取CAD与GT pose。没有实测GT depth输入；下表也不是深度标定目标。轨迹指标采用一次固定锚点对齐、与旧基准相同可比帧集，未消除对象对称性，不是绝对初始pose误差。

| 对象 | 旋转误差中位数 | 轨迹平移误差 配准前提案→配准后 |
|---|---:|---:|
| milk | 7.96°（配准未改变） | 19.90 → 22.00 mm |
| 汤罐 | 6.33°（配准未改变） | 163.25 → 123.59 mm |
| BBQ瓶 | 48.76°（配准未改变） | 41.12 → 49.83 mm |
| 白杯 | 110.45°（配准未改变） | 70.92 → 93.46 mm |
| 红铲 | 64.96°（配准未改变） | 321.77 → 155.28 mm |

GT位姿误差并非全部下降。这不应被隐藏，也不应反向使用GT修改这次约定的UniDepth参考。该模块解决的是生成网格与指定观测的尺寸/位置一致性；原旋转错误、不可见形状和手物深度口径仍需分别处理，不能据此宣布RL已经只需毫米级微调。

## 保护与测试

- 完整回归182项：178通过、4跳过；新增8项实际三角面测试全部通过。包括独立scale/t恢复、遮挡不缩小整物体、禁止隐藏背面冒充前表面、多帧单尺度及固定数据路由。
- 所有源深度NPZ哈希保持不变；配准输入点云与重新读取原annotations所得点云哈希一致；深度scale为全1、bias为全0。
- 输出R与 `alignment_initial_rotation` 每帧逐值相同；顶点等于源顶点乘唯一s，面索引完全保留。
- MANO顶点/关节/面数组未改。代码快照与工作区已核对，五例共10个150帧视频完整解码通过（1500帧）。
- 无CAD/GT预测输入、无额外相机、无MoGe/UniDepth模型重跑、无RL/reward修改、无S3覆盖。

## 文件与命令

运行输出：
```text
/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_visible_surface_20260927_r2/
```
独立评价与核验：
```text
/mnt/user-home/kupingxin/ego_annotation_experiments/monocular_coarse_evaluation_20260927/monocular_visible_surface_20260927_r2
```

关键文件：`anchor_alignment.json`、`shared_scale_alignment.json`、`timeline_surface_alignment.json`、`visible_surface_metrics.json`、`coarse_scene.{json,npz}`、`renders/before_after_overlay.mp4`、`renders/coarse_world.mp4`。评价目录包含`fixed_visible_reference.json`、GT pose对照、`verification.json`和五例锚点图。

```bash
cd /mnt/user-home/kupingxin/ego_annotation_preprocess_monocular
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -m preprocessing \
  --profile visible-surface --registration-width 192 \
  --scale-keyframes 5 --max-nfev 24 \
  --source-observations /ABS/FROZEN_RUN/source_observations.json \
  --output-dir /ABS/NEW_OUTPUT
```

输出目录必须是新的。旧profile仍可通过`--profile legacy`复现，但它不属于当前固定UniDepth的操作。运行包不覆盖原始source。
