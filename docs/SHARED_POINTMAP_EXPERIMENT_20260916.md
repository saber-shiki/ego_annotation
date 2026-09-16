# Shared pointmap / exact-K 单帧三臂实验（2026-09-16）

> **后续选择**：用户已选择只共享 pointmap，不新增独立 exact-K 模型干预。
> [可见表面空间包络整合设计](POINTMAP_ENVELOPE_INTEGRATION_DESIGN_ZH.md) 记录了
> 两次 native 调用的接线、旧包络机制、历史负结果和候选优化/末端渲染接缝；尚未默认部署。

## 结论与边界

共享 metric pointmap 在 milk anchor48上能显著改善 SAM3D 与当前观测的几何自一致性；
实验局部 exact-K 适配器能保留正确的相机合同，并通过未改动的 metric bridge。
**不是独立GT精度、跨帧/多案例可行性或完整标注成功。默认预测代码尚未改成该路径。**

基线为 `9ef06b3` 冻结运行的 anchor48。三个条件在同一已加载的 SAM3D 实例上，
使用相同RGB/mask字节、权重/config、每组seed42，在A800GPU5运行：

- A：原生内部 MoGe 点图；
- B：共享 metric depth/K 点图，调用原生 public pointmap 接口；
- C：与B逐元素相同的点图，实验局部 `compute_pointmap` 适配器显式绑定已知K。

源RGB是960×960；共享点图为米制H×W×3 PyTorch3D camera coordinates。
原点图重投影误差最大0.0000544px，五项适配器CPU合同测试通过。

## K 字段与实际生成计算的区别

B的原生K推导返回负焦距：normalized `fx=fy=-0.0011693`，主点强制0.5。
C的已知K为：

```text
[[0.2678346634, 0, 0.5034179688],
 [0, 0.2678346634, 0.5019530654],
 [0, 0, 1]]
```

C不重估焦距、不z-shift、不重采样/改变点图，不修改共享provider文件。
实测 B/C 的生成条件张量、pose decoder 的 scene scale/shift和输出pose完全相同。
当前 `with_layout_postprocess=false` 路径仅读取点图字典的 `pointmap/pts_color`，
**未读取独立 `intrinsics` 字段**。因此：

- 几何改善应归因于共享点图条件化，不能归功于仅修改K元数据；
- exact-K修复是必要的接口正确性保证，尤其后续启用K消费者时；
- B/C网格拓扑/顶点不逐字节一致，差异来源未单独归因。不能声称完全相同网格。

## 同口径指标

统一pipeline K、1939个观测surfel、0.5–99.5百分位extent diagonal0.28018648m。
下表的距离来自**实际未改动的**`bind_sam3d_metric`，使用canonical frame的40000
sample/seed1313；三角面IoU另用共同480×480 full-triangle first-hit raster计算。

| 指标 | A 内部MoGe | B 共享点图 | C 共享点图+exact K |
|---|---:|---:|---:|
| bridge统一尺度 | 0.472413 | 0.995807 | 0.995805 |
| 三角面 silhouette IoU | 0.297814 | 0.801858 | 0.801858 |
| sampled median | 26.065mm | 3.103mm | 3.116mm |
| sampled P95 | 54.508mm | 19.503mm | 19.526mm |
| median / extent | 0.09303 | 0.01108 | 0.01112 |
| P95 / extent | 0.19454 | 0.06961 | 0.06969 |
| 原bridge结果 | rejected | strict passed | strict passed |

阈值未变：median/extent≤0.05、strict P95/extent≤0.15；B/C不依赖conditional tier。
B/C即使scale=1也strict通过（C的sampled median3.527mm/P9520.707mm）。
生成面pose/contact/collision/sign eligibility仍false，`annotation_ready=false`。

精确三角面最近距离C为median2.780mm/P9519.488mm，仅作独立数值复核。
观测surfel rays的first-hit覆盖率A36.5%、B/C95.6%；C命中集合内的z误差
median4.24mm/P9526.60mm。仍有残差；各组命中集合不同，不称同样本GT比较。

**口径更正**：上一次实验的0.7873是凸包IoU，不应与基线三角面0.2978混比。
统一三角渲染后，旧pointmap结果IoU为0.801858，与本次B/C一致。

## 视觉和时间

已查看同源相机的完整三角覆盖、同窗口放大和带观测surfel的额外3D视角。
B/C修正了A明显偏小/偏上的投影，但局部边界、遮挡和隐藏表面仍未验证。

- 一次模型加载145.25s；
- A forward15.22s，B8.55s，C8.52s；
- 峰值allocated GPU memory约17.4–17.5GiB。

此前约153s的单帧耗时包含模型加载。此次forward时间是同模型实例实测，A是第一组，
有warmup/cache顺序影响，不能将全部差值归因于省略MoGe，也不是全管线时延。

## 限制与推荐下一步

- 同一单目depth既用于条件化又用于几何评估，只证明自一致性，不是holdout/GT精度。
- 一帧一对象一seed；跨视角/遮挡/材质/其它视频未经验证。
- 当前禁用layout后优化；启用后会消费K，必须重新验证。
- 未运行此candidate的后续完整pose/visual-fit/final render。
- 推荐先做命名的shared-pointmap实验profile及输入/provenance合同，然后预先固定少量
  多case/anchor重复验证；不静默替换默认MoGe路径，不导入旧production receipts。

## 证据与复现

实验目录：
`/mnt/user-home/kupingxin/ego_annotation_runtime/rlopt_shared_pointmap_adapter_20260916_anchor48/`

- `PLAN.md`, `adapter.py`, `test_adapter.py`；
- `run_three_arms.py`, `launch_model.py`, `model_run.log`, `model_run.exit`；
- `model_run/completed.json`、每组完整mesh和`conditioning.npz`；
- `evaluation/metrics.json`, `ABC_full_projection.jpg`, `ABC_projection_detail.jpg`, `ABC_novel_view.jpg`；
- `actual_bridge/result.json`：真实bridge调用的A拒绝/B和C接受结果；
- `RESULT_ZH.md`：详细实验说明。

独立只读审查 `bd3c940c-487d-4045-9dd8-ed6a4820071b`：未发现使上述有限结论失效的问题，
OK for reporting these narrow results；审查者未执行模型或指标代码。
所有新产物为standalone诊断；默认源码/共享模型/旧运行未改，未推送远端。
