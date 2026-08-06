# 五个 Toy 输出逐图阅读指南（中文）

> 建议第一次阅读时，不要先看总 loss。按本文顺序看图：先认清“变量是什么”，再看“哪条线是观测、哪条线是优化结果”，最后才看数值。

运行 toy 后，本文中的图片位于仓库根目录 `outputs/toy_optimization_lab/`。如果 Markdown 预览不能加载被 `.gitignore` 的图片，也可以打开 `outputs/toy_optimization_lab/index.html`。

---

## 0. 先理解所有图共用的词

| 图中词语 | 实际含义 | 是否能在真实 pipeline 中直接获得 |
|---|---|---:|
| `hidden truth` / `true` | toy 生成数据时保存的真实参数，只用来评价 optimizer | 否；真实视频通常没有 |
| `observation` | 加了噪声、缺失和 outlier 的测量，类似 P14 pose 或 visible surfel | 是，但有误差 |
| `canonical model` | 物体自身坐标中的模板 | 是一个估计，不一定是真实 CAD |
| `initial` / `base` | 优化前、所有 correction 为零的状态 | 是 |
| `raw` | optimizer 直接返回的状态 | 是，但不一定被最终接受 |
| `gated` / `post-gate` | 经过 output gate 或 acceptance policy 后真正保存的状态 | 是，应重新测 residual |
| `residual` | 一个因子没有满足的程度 | 是 |
| `loss term` | residual 经过平方、权重和归一化后的目标函数贡献 | 是，但不同项单位/权重不同 |
| `success=true` | 数值求解器按自己的停止条件结束 | 不等于物理结果正确 |

三个重要阅读规则：

1. **不同 toy 的 cost 不能横向比较**：每个 toy 的量纲、样本数和权重不同。
2. **一个 loss 降低不代表所有 loss 都降低**：优化是在做 trade-off。
3. **raw optimizer result 不等于 canonical result**：只要后面还有 gate、clamp、state split，就必须重测。

---

# 1. `sim2`：点云/刚体配准到底在做什么

对应真实 pipeline：P13 TRELLIS metric alignment 和 P14 per-frame pose fit。

## 1.1 `sim2_alignment.png`

![sim2 alignment](../../outputs/toy_optimization_lab/sim2/sim2_alignment.png)

### 左图：`Canonical toy keyboard`

蓝色点是物体自己的 canonical model：

- 外框类似 keyboard chassis；
- 中间横线类似 key rows；
- 右侧竖线类似 numeric keypad；
- 右上有一个小的不对称突起，用于消除“旋转 180° 看起来仍一样”的歧义。

这张图还没有 camera/world pose。可以理解为：

```text
模型知道物体长什么样，但不知道它在世界中的 scale、rotation 和 translation。
```

### 中图：`Discrete initialization`

三类点：

- 蓝色 `partial/noisy observation`：模拟深度相机看到的点；有噪声、缺点和少量 outlier；
- 橙色 `PCA candidate init`：PCA 轴候选搜索后，模型第一次放到 observation 上的位置；
- 浅色 `hidden truth`：完整模型按真实变换放到 world 后的位置，仅供 toy 评价。

注意几个离群的蓝点，例如远离 keyboard 外框的散点。它们模拟：

- mask 泄漏；
- depth outlier；
- 手/桌面被误当成物体；
- partial observation 的错误点。

橙色已经和大多数蓝色点接近，说明 PCA candidate 给了 ICP 一个不错的初值。ICP 通常只做局部优化；如果这里旋转错了 90° 或 180°，后续可能陷入错误局部极小值。

### 右图：`Final alignment`

- 蓝色：observation；
- 橙色：8 轮 ICP 后的模型；
- 浅色：hidden truth。

橙色和主体蓝点比中图更重合，说明 scale/rotation/translation 被进一步调整。

但远处 outlier 仍没有被解释。这不是绘图错误，而是在展示：

> 一个全局刚体变换不可能同时贴合正确 keyboard 和错误 outlier；优化器会主要照顾数量更多的主体点。

## 1.2 `sim2_trace.png`

![sim2 trace](../../outputs/toy_optimization_lab/sim2/sim2_trace.png)

横轴：ICP iteration，1 到 8。

纵轴：每个 observed point 到最近 model point 的距离。

三条线：

- 橙色 `median`：典型点的误差，50% 点比它小；
- 蓝色 `mean`：平均误差，比 median 更受 outlier 影响；
- 绿色 `p95`：95% 点比它小，描述较坏的尾部，但忽略最坏 5%。

读图：

- iteration 1→4 下降明显；
- iteration 6 后几乎不动，表示 correspondence 和 transform 已稳定；
- p95 仍明显大于 median，表示 tail error 仍存在；
- 图里没有画 `max`，因为 max 被单个 outlier 主导，约 0.51，甚至略有变差。

JSON 中的关键值：

```text
initial median: 0.01808
final median:   0.01425
initial mean:   0.02180
final mean:     0.01806
final max:      0.51163
```

### 这个 toy 想说明什么

- ICP 的确改善了多数点；
- 但单向 nearest-neighbor objective 不会自动解决 outlier；
- `median` 好看不能证明 hidden mesh 全部正确；
- 真实 P13/P14 还必须看 bidirectional distance、mask identity、visibility 和 render。

---

# 2. `pose_graph`：为什么 P15 没有 target 就完全不动

对应真实 pipeline：P15 correction-field pose graph。

## 2.1 `pose_graph.png`

![pose graph](../../outputs/toy_optimization_lab/pose_graph/pose_graph.png)

图有三行，横轴都是 frame 0–59。

### 第一行：物体 y translation

- 黑色虚线 `hidden true y`：toy 的真实轨迹，optimizer 看不到；
- 灰色 `P14-like observation y`：带噪声的逐帧 pose observation；
- 蓝色 `active graph corrected y`：加入 soft pressure 后的结果；
- 红色半透明区域：pressure factor 生效的 frame 21–37。

在红色区域里，蓝线通常高于灰线，说明 pressure 希望给 object pose 一个正 y correction。

图上蓝线有时更接近黑色 truth，有时更远。原因是 pressure target 是人为构造的 factor，不是 truth。

### 第二行：真正被优化的 correction field

这是最关键的一行：

- 蓝色 `no target: correction`：整条为 0；
- 绿色虚线 `target correction`：希望 frame 21–37 增加 0.075；
- 橙色 `with target: correction`：optimizer 最后只增加约 0.018，并平滑延伸到区间边缘。

为什么橙线没有到绿色 0.075？因为目标函数同时要求：

```text
接近 pressure target
+ correction 不要太大
+ correction 随时间平滑
+ correction 加速度不要太大
```

最优解是折中，而不是完全满足某一项。

为什么无 target 的蓝线严格为 0？因为这时所有项都是：

```text
correction 要接近 0
correction 的一阶差分要接近 0
correction 的二阶差分要接近 0
```

`correction=0` 同时把所有 residual 变成 0，所以：

```text
nfev=1, cost=0
```

这正是当前真实 P15 的情况。

### 第三行：rotation

- 黑色虚线：true rotation；
- 灰色：observation；
- 橙色：corrected。

pressure 只作用于 translation y，没有 rotation target，所以 rotation correction 仍是 0；橙线覆盖在灰线上。

这说明 factor graph 不会“自动顺便修好”没有连接到 residual 的变量。

## 2.2 JSON 数值怎么读

```text
inert_case:
  nfev=1
  cost=0
  max_correction=0
```

这是严格的零目标情况。

active case：

```text
target residual before = 0.0750
target residual after  = 0.0590
```

说明 correction 向 target 走了约 0.016，但没有完全到达。

```text
hidden truth RMSE before = 0.01549
hidden truth RMSE after  = 0.01751
```

真实误差略微变差。不是 optimizer 算错，而是它没有访问 hidden truth，只能优化声明的 pressure/prior/smoothness。

### 这个 toy 想说明什么

- `optimizer success` 只说明数值求解完成；
- 没有 nonzero factor 时，零 correction 是正确最优解；
- 一个错误/过强的 factor 可以降低自己的 residual，却让真实结果变差；
- 必须用独立 surface/render evidence 做 post-check。

---

# 3. `halfspace`：非穿透约束的可行域是什么

对应真实 pipeline：P16 least-norm rigid escape translation。

## 3.1 `halfspace_escape.png`

![halfspace escape](../../outputs/toy_optimization_lab/halfspace/halfspace_escape.png)

图有四个 panel。

## 3.2 左上：`Feasible local escape`

这是几何空间：

- 黑色圆边界：object surface；
- 灰色圆内部：object interior；
- 红点：位于 interior 的 hand points；
- 橙色小箭头：每个点到最近 surface point 的 outward escape direction；
- 蓝色箭头：一个共同 rigid translation `t`；
- 绿色叉：所有红点加上同一个 `t` 后的位置。

因为红点都在圆的右边，outward normals 大致朝右，所以存在一个共同向右平移，把所有点推出 surface。

注意：P16 的变量只有一个 rigid translation。每个点不能各走各的方向。

## 3.3 右上：`Feasible halfspace intersection`

这是 **translation 参数空间**，不是 object 几何空间。

每一个位置 `(t_x,t_y)` 代表“把整只手平移多少”。

- 红色区域：至少违反一个 halfspace；
- 绿色区域：满足所有 `n_i^T t >= d_i`；
- 黑色加号：零 translation；
- 灰色同心圆：`1/2 ||t||²` 的等值线，越靠近原点 objective 越小；
- 蓝点：绿色可行域中离原点最近的点。

因此蓝点就是：


a) 满足全部 escape constraints；

b) 在满足约束的解中移动最少。

报告中的：

```text
translation = [0.2345, 0.0247]
translation norm = 0.2358
min slack = -5.55e-17
```

`-5.55e-17` 是 floating-point 数值误差，可以视为 0，说明最紧的约束刚好 active。

## 3.4 左下：`Conflicting rigid escape directions`

红点同时位于：

- 圆右侧：要求向右移；
- 圆左侧：要求向左移；
- 圆上侧：要求向上移。

左右两个约束近似为：

```text
t_x >= 0.22
-t_x >= 0.22
```

不存在一个共同 rigid translation。

## 3.5 右下：`Empty/infeasible intersection`

整个图都是红色，没有绿色区域，表示可行域为空。

SLSQP 仍会返回一个最后尝试的数值向量，但：

```text
success = false
min slack = -0.2202
```

说明约束最多还违反约 0.22。此时不能使用返回向量作为物理 correction。

### 这个 toy 想说明什么

- 局部 point normals 一致时，rigid translation repair 可能合理；
- normals 冲突时，问题可能真正 infeasible；
- infeasible 可能意味着需要 articulation，而不是 rigid translation；
- 也可能意味着 signed distance、watertight geometry 或 hand/object frame 有错；
- 真实 P16 当前连 watertight sign 前提都不满足，所以比 toy 更不确定。

---

# 4. `mano`：多个 loss 如何折中，以及 gate 为什么会破坏结果

对应真实 pipeline：P18 interval MANO optimization + output translation gate。

## 4.1 `mano_trajectory.png`

![mano trajectory](../../outputs/toy_optimization_lab/mano/mano_trajectory.png)

横向两列、纵向三行，共六个 panel。

### 左上：`Raw LBFGS hypothesis`

- 黑色横线 `y=0`：object top surface；
- 灰色区域 `y<0`：object interior；
- 浅色虚线 hand：base/source hand；
- 深色实线 hand：raw LBFGS result；
- `f8/f16/f29/f41`：四个代表 frame。

f16、f29 位于 contact interval。base hand 的 finger 深入灰色区域，而 raw optimizer 通过：

- root 向上平移；
- root angle 改变；
- articulation 改变；

把 fingertip 拉到 `y≈0`。

### 右上：`After no-support wrist gate`

这个 toy 假设每一帧都没有独立 visible-surface support，因此模拟 P18 output gate：

```text
把 wrist/root 重新平移回 source wrist
但保留 optimizer 得到的相对 articulation
```

结果是 f16/f29 的 finger 又进入灰色 object interior。

这说明 gate 不是“无害地改一个 JSON 字段”，而是改变了真实几何状态。

### 左中：`Raw optimized global translation`

- 蓝线：raw optimizer 的 root y translation correction；
- 绿色背景：contact interval；
- 红色虚线：soft translation bound ±0.085。

contact interval 内，optimizer 大约向上平移 0.05–0.08，用 global root motion 同时减少 penetration 和 contact gap。

蓝线接近 bound，但没有越过，因此 `soft_bounds` loss 最终为 0。soft bound 只在超过阈值后才惩罚。

### 右中：`Contact objective versus saved state`

三条 fingertip 高度：

- 蓝色 `base`：contact interval 内约为 -0.07 到 -0.10，明显穿透；
- 橙色 `raw`：被拉到 0 附近，满足 contact deadband；
- 绿色 `post-gate`：恢复 wrist 后约为 -0.05，再次穿透。

浅绿色带 `[-0.015,+0.015]` 是 contact deadband。进入该带时 contact residual 为 0。

raw 橙线进入 deadband，不代表整只手都正确；只代表 fingertip normal gap 这一项满足。

### 左下：`Re-measure after every state transform`

纵轴是每帧最大 penetration，使用对数/近零混合刻度：

- 蓝色 base：约 0.07–0.10；
- 橙色 raw solver：下降到约 0.003–0.009；
- 绿色 after gate：回升到约 0.05–0.09。

这是整个 toy 最重要的 panel：

> 如果只保存 raw solver 的 penetration report，会声称问题基本解决；但真正输出的 gated state 并没有解决。

### 右下：`Latent contact switch`

- 虚线：prior；
- 实线：posterior。

两条线几乎完全重合：

```text
contact interval: 0.88
其它 frame:       0.05
```

说明 contact logit 虽然是 optimizer variable，但当前 loss 下它几乎没有从几何数据更新，只复制了 prior。

真实 P18 也有类似现象：posterior≈prior。

## 4.2 `mano_losses.png`

![mano losses](../../outputs/toy_optimization_lab/mano/mano_losses.png)

### 左图：loss decomposition

纵轴是对数刻度下的 **加权后 loss contribution**。

蓝柱是 initial，橙柱是 final raw。

初始时所有 delta 都为 0，所以这些项为 0：

```text
translation_prior
root_prior
pose_prior
temporal
self_projection_hinge
self_depth_hinge
soft_bounds
```

初始非零的主要是：

```text
nonpenetration = 5.546
contact_patch  = 2.702
```

优化后：

- `contact_patch` 降为 0；
- `nonpenetration` 降为约 0.059；
- 但 translation/root/pose/temporal/self-hinge 从 0 变成正值。

这就是 trade-off：

```text
为了减少 penetration/contact，必须移动 hand；
移动 hand 又会支付 observation prior、temporal 和 hinge 代价。
```

最终不是每一项都最小，而是加权总和较小。

`contact_state_prior` 几乎为 0，表示 posterior 没离开 prior。

### 右图：LBFGS closure trace

横轴不是严格的 optimizer iteration，而是 closure evaluation 次数。Strong-Wolfe line search 在一个 iteration 内会多次试探，所以：

```text
max_iter = 140
closure evaluations = 175
```

曲线：

- 开头从约 8.25 快速下降；
- 中间在约 0.6 附近平台；
- 后续继续下降到约 0.486；
- 单次 closure 不保证严格单调，因为 line search 会评估被拒绝的 trial step。

## 4.3 JSON 的 raw/gate 数值

```text
total raw loss: 8.248 -> 0.486
raw penetration max: 0.00859
raw contact gap median: 0.00158
```

但 gate 后：

```text
penetration max: 0.09108
contact gap median: 0.05984
```

### 这个 toy 想说明什么

- 总 loss 下降是不同因子之间的折中；
- optimizer 可以主要使用 global translation，而不是 articulation；
- soft bounds 不是 hard constraints；
- contact posterior 可能只是 prior 的复制；
- raw solver metrics 必须在 gate/P18b/canonical state 上重算；
- acceptance policy 应根据 post-gate metrics 决定是否升格结果。

---

# 5. `joint_gauge`：大联合优化为什么会有不可观测方向

对应问题：把 camera、object、hand、contact 放进一个大 MAP/factor graph。

## 5.1 `joint_gauge.png`

![joint gauge](../../outputs/toy_optimization_lab/joint_gauge/joint_gauge.png)

### 左图：`Loss along global gauge direction`

横轴表示同时对 camera、object、hand 加共同 world translation。

蓝线 `relative-only factors` 几乎完全水平，energy variation 只有约 `3.9e-19`。原因是观测只有：

```text
object - camera
hand - camera
hand - object
```

三者共同平移后，这些差值都不变。

橙线加入 camera gauge anchor 后，在 0 附近有唯一最低点；共同平移会增加 anchor residual。

注意纵轴画的是 `energy above minimum`，对数/近零刻度。蓝线约 `1e-11` 是浮点数值噪声，可视为严格平坦。

### 中图：`One near-zero mode is a gauge`

横轴：从小到大排序的 singular-value index。

纵轴：residual Jacobian/design matrix 的 singular value。

直观理解：

- singular value 大：沿对应变量方向移动时，residual 变化明显，变量可观测；
- singular value 很小：移动很多，residual 也几乎不变，变量弱可观测；
- singular value≈0：存在完全不可观测的 gauge。

relative-only 蓝点中有一个约 `2.2e-16`；加入 anchor 后最小值约 `1.9e-2`。

这说明 gauge fixing 不是“给 optimizer 一个更好的初值”，而是让问题从 rank-deficient 变成有唯一参考系。

### 右图：`Anchored joint MAP estimate`

黑色三条线是 toy truth：

- camera；
- object；
- hand。

彩色三条线是 anchored MAP estimate。

为什么加了 anchor 后，彩色曲线仍没有完全追上黑色 truth？

因为：

1. 只锚定了 camera frame 0，而不是每帧 camera ground truth；
2. relative observation 主要约束三者之间的差；
3. temporal prior 希望轨迹平滑、变化小；
4. 真正的 camera 大幅运动和 temporal zero-velocity prior 冲突；
5. MAP 选择了“相对关系正确但 world motion 更平”的折中。

因此：

> 去掉 gauge 只解决唯一性，不保证 measurement/prior model 正确。

## 5.2 JSON 数值

```text
variables: 72 = 24 frames × (camera + object + hand)
relative-only matrix: 141 × 72
smallest singular value: 2.18e-16
```

加入一个 camera anchor row：

```text
anchored matrix: 142 × 72
smallest singular value: 1.88e-2
```

### 这个 toy 想说明什么

- 大联合问题可以是 sparse 的，变量数量未必不可承受；
- 主要危险是 rank deficiency/gauge；
- 多加 relative loss 不会消除共同平移 gauge；
- anchor 消除 gauge 后，错误 temporal prior 仍会导致偏差；
- 应先检查 Jacobian rank/observability，再调 optimizer 或 loss weight。

---

# 6. 五个 toy 串起来看

按 pipeline 顺序，可以把它们理解为：

```text
sim2
  先把物体模型和观测对齐
  但 correspondence/outlier 可能有问题
       ↓
pose_graph
  在逐帧 pose 上优化 correction field
  没有非零 factor 就不会动
       ↓
halfspace
  构造非穿透修正
  但 constraints 可能不可行
       ↓
mano
  同时折中手的 observation、temporal、penetration 和 contact
  raw result 还可能被 output gate 改坏
       ↓
joint_gauge
  如果把所有模块合起来
  首先会遇到 gauge 和不可观测性，而不只是“优化器够不够强”
```

---

# 7. 每次看真实 optimizer report 时应问的 10 个问题

1. optimizer 真正放开的变量是什么？
2. 哪些变量虽然存在字段，但 `requires_grad=false` 或没有接 residual？
3. 初值处的 residual 是否已经为 0？
4. loss 是单向还是双向？
5. `success=true` 的停止条件是什么？
6. soft threshold 是否被误当成 hard constraint？
7. 是否存在 gauge 或 near-zero Hessian/Jacobian mode？
8. contact posterior 是否真正偏离 prior？
9. optimizer 后是否还有 gate/clamp/state split？
10. canonical saved state 是否重新计算了全部 residual？

如果这 10 个问题没有回答，仅看一个 `cost` 或 `status=ok` 通常不足以判断物理优化是否成功。
