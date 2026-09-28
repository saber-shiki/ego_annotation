# 单目粗位姿预处理：独立实现与两轮验证

日期：2026-09-27。分支 `feature/monocular-coarse-preprocess-20260927`，起点 `790893c`。

本页保留R1/R2的历史结果。后续重定位修复见 `MONOCULAR_COARSE_RELOCALIZATION_REPAIR.md`、`MONOCULAR_LOW_TEXTURE_GEOMETRY_DESIGN.md`；不能将旧表格当作新运行结果。

## 结论与交付级别

**这是已运行的实验实现，不是精度验收通过的正式粗位姿模块。** 五例全视频均完成、图像贴合普遍改善，但白杯/红铲/BBQ仍有大角度误差，汤罐和红铲仍有严重平移问题；未达到“RL只需毫米级接触微调”的前提。没有合并到默认观测入口或接入旧PPO。不能用下面的IoU提升替代物理精度。

输入仅为原有单目RGB及其冻结预测观测/相机标定元数据；没有CAD、GT pose/depth或额外相机输入。GT仅由工作树外的独立评价脚本在候选完成后读取。原source、原MANO、S3和原dirty工作区未被本次修改。

## 已实际实现

- `inputs.py`：hash绑定v2观测、统一K/图像平面、固定native canonical frame、source缺测资格保留、拒绝评价目录输入。
- `tracking.py`：实际三角面绑定的masked forward/backward LK、PnP刚体候选、持久材料点和关键帧补充；不是把R=I加平滑。
- `factors.py` / `optimizer.py`：多帧RGB重投影、独立旋转因子、可见轮廓、MANO遮挡、前表面深度、观测点到真实三角面的三维配准、稀疏鲁棒SE3联合求解。
- `metric_alignment.py`：从单目跨帧对应点的刚体弦长估计相对深度尺度，固定一个source gauge；不宣称独立绝对米制标定。
- `geometry.py`：真实triangle ray/closest查询、精确相机变换、共享体积保持低秩形状变换。该形状模块保留完整native拓扑，不声称恢复遗漏孔洞或未知背面。
- `render.py`：完整150帧before/after原视频叠加与固定world视图，均使用真实native及源MANO三角面。
- 独立 `python -m preprocessing` / 安装后的 `ego-preprocess` 入口；原main/训练/reward不变。

## R1暴露的问题与R2的因果修复

R1允许自由逐帧additive depth bias，图像变好时仍能把对象移离三维观测；射线不命中后还会失去深度因子。R2没有通过截断t或删除坏帧掩盖问题，而是取消生产pass中的自由bias，使用刚体长度的相对尺度和观测点到三角面的三维约束。R1产物和失败证据完整保留。

R2降低了一些严重表面误差尾部，但并未全面修复。红铲旋转等指标还出现退化，当前方案不应作为正式初始化器。R3–R6后续尝试也有回归。纠正先前记录：R5的25°/60mm限制实际上比较的是候选对当前估计的校正，不是相邻帧真实运动，不能据此宣称物理约束已满足。该规则已删除；当前实现改用共同留出集与真实几何验证，并保留原有鲁棒局部跟踪。见新的重定位修复设计。

## 五例实测（R2）

下表旋转/平移是与前次同一共同有效帧集上的**单锚点对齐后轨迹误差**；不是绝对初始CAD pose，旋转未消除对称性。surface是未经GT对齐的真实camera-space双向采样点到三角面平均距离的跨帧中位数，包含形状/尺度/位姿误差。所有帧和全QA集合的统计都在外部JSON/CSV中，未只保存好帧。

| 视频 | source mask IoU 前→后 | 旋转误差中位数 | 平移误差中位数 | 原始表面差异中位数 |
|---|---:|---:|---:|---:|
| milk | 0.796 → 0.913 | 4.7° | 12.0 mm | 10.4 mm |
| 汤罐 | 0.742 → 0.910 | 9.0° | 152.1 mm | 47.2 mm |
| BBQ瓶 | 0.324 → 0.784 | 45.2° | 44.6 mm | 19.5 mm |
| 白杯 | 0.466 → 0.697 | 110.7° | 68.8 mm | 23.6 mm |
| 红铲 | 0.170 → 0.558 | 94.4° | 231.2 mm | 52.5 mm |

| 视频 | 原RGB seed旋转→R2 | 原RGB seed平移→R2 | R2旋转P95 / 平移P95 |
|---|---:|---:|---:|
| milk | 6.0° → 4.7° | 19.7 → 12.0 mm | 13.3° / 40.0 mm |
| 汤罐 | 44.5° → 9.0° | 151.5 → 152.1 mm | 33.0° / 175.3 mm |
| BBQ瓶 | 40.2° → 45.2° | 53.8 → 44.6 mm | 149.2° / 79.3 mm |
| 白杯 | 131.3° → 110.7° | 51.5 → 68.8 mm | 175.1° / 151.4 mm |
| 红铲 | 94.1° → 94.4° | 78.1 → 231.2 mm | 107.0° / 390.5 mm |

含义：

- milk旋转及相对seed的平移有所改善，但平移P95仍约40mm，不可宣称达到厘米以内全片粗定位。
- 汤罐旋转改善明显，但位移/尺度问题没有解决，不能用轮廓贴合通过来掩盖。
- BBQ仍有错误朝向和尾部失败；白杯仍有严重朝向歧义/跟踪问题；红铲仍受错误形状、稀疏支持及姿态失配影响。它们是未解决的算法/观测耦合问题，不能全部归咎于“单目不可观测”。
- 绝对尺度仍继承有偏的单目source。新的相对尺度一致性不提供独立的绝对真值。

## 验证与保护

- 新增15项判别性数值测试全部通过：真实triangle查询、相机往返、65°真实旋转PnP、像素驱动的旋转/位移恢复、稀疏Jacobian依赖、刚体长度相对scale、3D重新锚定、hash/input边界。
- 统一环境完整回归158项：154通过、4跳过。最初在不含gymnasium的数值环境运行全套时有1项环境缺依赖；随后改用已存在统一环境通过，没有安装或修改依赖。
- R1、R2各五例运行退出0。R2十个完整视频共1500帧已全部解码，每段150帧/5秒。
- R2 source manifest hash、运行代码快照与当前Python代码一致；MANO vertices/joints/faces逐数组未变；R为合法SO3，共享形状det=1。
- 已查看五例f100 before/after及milk的world视图，所有视频/源几何保留可复核。视觉检查不取代上表失败指标。
- 全程服务器CPU，无新观测模型、无GPU推理、无RL训练、无GT校正倍率回填。

## 路径

工作区：`/mnt/user-home/kupingxin/ego_annotation_preprocess_monocular`

R2冻结代码：`/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_coarse_20260927_r2`

R2运行：`/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_coarse_20260927_r2`

外部评价：`/mnt/user-home/kupingxin/ego_annotation_experiments/monocular_coarse_evaluation_20260927`

每个case目录包含 `coarse_scene.{json,npz}`、bootstrap、tracking、solver、relative_depth_calibration、image_diagnostics，以及 `renders/before_after_overlay.mp4` 和 `renders/coarse_world.mp4`。

## 复现

```bash
cd /mnt/user-home/kupingxin/ego_annotation_preprocess_monocular
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -m preprocessing \
  --source-observations /ABS/FROZEN_RUN/source_observations.json \
  --output-dir /ABS/NEW_OUTPUT
```

输出目录必须新建且不在source目录内。精确本次批量复现命令在R2运行根目录的 `run.sh`；勿覆盖已完成目录。

## 下一步尚需修复

优先在同一单目协议内补强遮挡后的多假设重定位、长程稳定材料对应与共享形状约束，并区分轮廓运动和材料点运动；不能通过放大RL动作范围兜底。需要新的、预测端可用的尺度/形状先验或更准确的单目观测来降低绝对metric偏差，不能使用评价CAD/GT直接纠正这五例。

当前代码/实验结果保留在独立分支，未合并、未提交或推送；没有把“能运行/图像更好”宣称为物理精度已达标。
