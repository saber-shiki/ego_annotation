# V19 Optimization Toy Lab：从模块化优化到联合 MAP 与 RL

## 1. 目的

这个目录把当前 V19 pipeline 里最重要的优化机制压缩成低维、可视化、可重复的 toy problems。目标不是复刻真实 MANO/Mesh 数值，而是让以下问题一眼可见：

- 优化变量到底是什么；
- 每个 residual 在拉哪个方向；
- loss 下降是否等于物理结果变好；
- soft penalty、hard constraint 和 output gate 有什么区别；
- 为什么没有外部 factor 时 P15 会原地不动；
- 为什么局部非穿透 halfspaces 可能不可行；
- 为什么把 camera、object、hand、contact 合成大问题会出现 gauge freedom；
- RL 如果加入，适合控制什么，不适合替代什么。

所有 toy 都是 synthetic 2-D/低维问题，不读取 V19 run，也不会修改 prediction state。

如果对图中的英文标题、颜色和曲线含义不熟悉，先阅读：

- [`WALKTHROUGH_ZH.md`](WALKTHROUGH_ZH.md)：逐图、逐 panel、逐字段解释五个 toy 的输出

---

## 2. 运行

在仓库根目录执行：

```bash
/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python \
  experiments/toy_optimization_lab/run_toys.py \
  --problems all \
  --output-dir outputs/toy_optimization_lab \
  --seed 1907
```

只运行一个或多个 toy：

```bash
# 只看 P15-like pose graph
/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python \
  experiments/toy_optimization_lab/run_toys.py \
  --problems pose_graph \
  --output-dir outputs/toy_optimization_lab

# 看 P16 和 P18
/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python \
  experiments/toy_optimization_lab/run_toys.py \
  --problems halfspace,mano \
  --output-dir outputs/toy_optimization_lab
```

`outputs/` 已在 `.gitignore` 中，不会把生成的 PNG/JSON 加入版本控制。运行结束后可直接打开：

```text
outputs/toy_optimization_lab/index.html
```

它会集中展示所有图，并链接每个 JSON report。

---

## 3. Toy 与真实模块的对应关系

| Toy | 对应真实模块 | 变量 | 主要目标/约束 | 重点观察 |
|---|---|---|---|---|
| `sim2` | P13/P14 | 2-D scale、rotation、translation | PCA candidate score；NN+Umeyama ICP | 单向 NN loss 下降、outlier max 不一定下降 |
| `pose_graph` | P15 | 每帧 translation/rotation correction | zero prior、target、velocity、acceleration、`soft_l1` | 无外部 target 时 `cost=0,nfev=1` |
| `halfspace` | P16 | 一个 rigid translation | `min 1/2||t||², Nt>=d` | feasible 与 incompatible normals 的区别 |
| `mano` | P18 | root translation、root angle、articulation、contact logit | prior、temporal、penetration、self-shift、contact、soft bounds | raw optimizer 与 post-gate saved state 可以完全不同 |
| `joint_gauge` | 联合 pipeline MAP | camera/object/hand 全时序 state | relative image、contact、temporal factors | common translation gauge 和 near-zero singular value |

---

## 4. 每个 toy 会生成什么

### 4.1 `sim2`：P13/P14-like 点云配准

输出：

```text
outputs/toy_optimization_lab/sim2/
├── sim2_alignment.png
├── sim2_trace.png
└── sim2_report.json
```

建模：

1. 生成一个带 numeric keypad 的不对称 keyboard point set；
2. 用未知 Sim(2) 变换生成 partial/noisy observation，并加入 outliers；
3. 枚举 PCA axis/sign candidates，使用：

   ```text
   median(observed -> model NN distance) + 0.25 * p90
   ```

4. 固定 nearest-neighbor correspondence 后用 Umeyama 闭式更新；
5. 迭代八轮。

默认 seed 的典型结果：

- mean NN distance 约 `0.0218 -> 0.0181`；
- median 约 `0.0181 -> 0.0142`；
- rotation error 约 `0.0056 rad`；
- 但 max outlier distance仍约 `0.51`。

直观结论：**optimizer 优化的是声明的 one-way correspondence objective；outlier tail、bidirectional support 和 hidden truth 必须另测。**

### 4.2 `pose_graph`：为什么本次 P15 完全不动

输出：

```text
outputs/toy_optimization_lab/pose_graph/
├── pose_graph.png
└── pose_graph_report.json
```

同时运行两组：

1. **inert case**：只有以 correction=0 为中心的 measurement prior、velocity 和 acceleration；
2. **active case**：额外在一个 interval 注入 nonzero soft pressure target。

默认结果：

```text
inert:  nfev=1, cost=0, max correction=0
active: nfev=5, nonzero correction
```

active pressure 的 residual 会下降，但 synthetic hidden-truth RMSE 可能反而升高，因为 pressure 是人为声明的 factor，不是 ground truth。

直观结论：

- 没有非零 factor 时，零 correction 不是 solver 坏了，而是正确最优解；
- 添加一个 factor 只保证它自己的 residual 参与 trade-off，不保证真实轨迹改善；
- P15 的 surface post-check 与 objective 必须分开看。

### 4.3 `halfspace`：P16 最小范数逃逸

输出：

```text
outputs/toy_optimization_lab/halfspace/
├── halfspace_escape.png
└── halfspace_report.json
```

上半部分：penetrating points 集中在 object 一侧，所有 outward normals 大致一致，存在一个 least-norm translation。

下半部分：points 同时位于圆的左右两侧，约束近似为：

```text
t_x >= 0.22
-t_x >= 0.22
```

可行域为空，SLSQP 报告 failure。

直观结论：

- optimizer failure 可以是刚体 translation 模型能力不足；
- 也可能是 sign/normal support 错误；
- 不能因为需要一个 H-prime 就强行使用 fallback correction；
- 真实 P16 还必须先满足 watertight/local-sign validity。

### 4.4 `mano`：P18 多 loss 冲突和 output gate

输出：

```text
outputs/toy_optimization_lab/mano/
├── mano_trajectory.png
├── mano_losses.png
└── mano_report.json
```

这个 toy 使用一个可微 3-link planar hand，变量为：

```text
translation_delta[t,2]
root_delta[t]
pose_delta[t,2]
contact_logit[t]
```

loss 包括：

- translation/root/pose prior；
- velocity 和 acceleration；
- dense nonpenetration barrier；
- self-projection hinge；
- self-depth hinge；
- soft parameter bounds；
- contact-state prior/temporal；
- contact deadband point-to-plane term。

使用 `torch.optim.LBFGS(strong_wolfe)`。默认结果会显示：

- raw total loss 显著下降；
- raw hand 通过 global translation 和 articulation 把 fingertip 拉到 contact surface；
- contact posterior 几乎等于 prior；
- 随后模拟“没有 visible support，恢复 source wrist”的 output gate；
- gate 后 penetration/contact gap 会重新变差。

默认 seed 的一组实际 toy 数值：

```text
raw loss:                         8.248 -> 0.486
raw penetration max:             0.0086
raw active contact gap median:   0.0016
post-gate penetration max:       0.0911
post-gate active contact gap:    0.0598
```

直观结论：**raw optimizer 的 before/after metrics 不能代表经过 post-processing/gate 的 canonical state；每次 state transform 后都必须重新测量所有物理 residual。**

### 4.5 `joint_gauge`：把全 pipeline 合成大问题时的 gauge

输出：

```text
outputs/toy_optimization_lab/joint_gauge/
├── joint_gauge.png
└── joint_gauge_report.json
```

一维 joint MAP 中同时优化：

- camera trajectory `C_t`；
- object trajectory `O_t`；
- hand trajectory `H_t`。

观测只有相对量：

```text
O_t - C_t       object image/depth-like observation
H_t - C_t       hand image/depth-like observation
H_t - O_t       contact-like relation
```

如果对三者共同加一个常量 `alpha`，所有相对 residual 不变：

```text
C_t <- C_t + alpha
O_t <- O_t + alpha
H_t <- H_t + alpha
```

默认结果：

```text
relative-only smallest singular value: ~2.2e-16
energy variation along common shift:   ~3.9e-19
```

加入 camera gauge anchor 后 smallest singular value 变为约 `1.9e-2`。

直观结论：

- 联合问题的主要困难不一定是变量多，而是不可观测方向；
- 多加 loss 不等于多加信息；
- camera/scale/world-frame anchor 必须显式定义；
- temporal prior 可以消除数值抖动，也可能把真实运动偏平，必须以观测不确定性定权。

---

## 5. 把完整 pipeline 写成一个大优化问题是否可行

### 5.1 数学上可以写成 MAP/factor graph

可以把连续 state 写成：

```text
X = {
  camera intrinsics/trajectory,
  left/right MANO trajectory,
  object canonical geometry,
  object SE(3) trajectory,
  contact switches/patches,
  visibility/ownership switches
}
```

形式上求：

```text
min_X,z
    E_rgb_or_feature
  + E_depth
  + E_hand_observation
  + E_object_mask_surface
  + E_rigid_pose
  + E_temporal
  + E_contact
  + E_nonpenetration
  + E_prior
```

其中 `z` 包括：

- object identity/branch；
- contact mode；
- visibility/ownership；
- correspondence；
- outlier switch。

这就是一个带离散 latent variables 的 sparse MAP 问题。

### 5.2 规模本身不是最大障碍

150 帧、双手和一个 rigid object 的核心连续变量通常只有万级：

- camera：约 `6T`；
- object pose：约 `6T`；
- 双手 root/pose：约 `2 × 51T`；
- contact switches：约 `2T`。

利用 temporal sparsity 后，Gauss-Newton/Levenberg-Marquardt、sparse least squares 或 differentiable optimization 在计算上并非绝对不可做。

真正困难的是：

1. **gauge**：camera、object、hand 可以共同平移/缩放；
2. **不可观测性**：单目 contact、hidden thickness、遮挡后手指没有独立证据；
3. **离散模式**：contact/no-contact、谁遮挡谁、物体 branch；
4. **不兼容测量**：UniDepth、HaWoR、mask、completion 可能不在同一坐标/尺度；
5. **非光滑 correspondence**：nearest face、mask ownership、topology change；
6. **模型误差**：非 watertight Mesh、错误 hand side、错误 object mask；
7. **loss scale**：px、m、rad、probability 不能靠随意大权重直接相加。

### 5.3 不建议“一次性 end-to-end 全变量同时放开”

更合理的是分层 block-coordinate / factor-graph architecture：

```text
measurement extraction
        ↓
calibration/camera gauge solve
        ↓
object canonical geometry + rigid pose solve
        ↓
hand observation solve
        ↓
contact/visibility discrete mode inference
        ↓
local joint refinement
        ↓
canonical acceptance + remeasurement
```

每层都：

1. 固定上一层可靠 state；
2. 只放开当前可观测变量；
3. 输出 residual、Jacobian rank、uncertainty；
4. 失败时回退或保留 competing hypotheses；
5. 最后可做一轮小范围 joint refinement，而不是从随机初值全局联立。

推荐形式：

- continuous inner loop：sparse MAP / differentiable solver；
- discrete outer loop：HMM/Viterbi、switchable factors、EM、beam search；
- sliding window：限制内存和错误传播；
- robust M-estimator：处理 outliers；
- explicit gauge fixing：camera/world/scale anchor；
- uncertainty-calibrated weights：尽量用 `1/sigma²`，而不是人工数量级。

---

## 6. 能不能用 RL 求解

### 6.1 不建议 RL 直接替代连续几何求解器

如果目标是对一个给定视频求 `R,t,MANO pose`，这个问题有：

- 可微 forward model；
- 可计算 residual；
- 稀疏时序结构；
- 约束几何。

这正是 Gauss-Newton、LBFGS、SQP、factor graph 擅长的问题。让 PPO/SAC 直接输出上万个连续状态通常会：

- 样本效率远低于梯度求解；
- reward 极难标定；
- 容易通过移动 camera/object/hand 一起 reward hacking；
- 对真实视频 domain gap 敏感；
- 很难保留可解释 residual 和不确定性；
- 对单个 sequence 而言，RL 退化成昂贵的 black-box optimizer。

因此“先理解 loss，再把整个 optimizer 换成 RL”通常不是最优路线。

### 6.2 RL 更适合外层序列决策

RL 有价值的地方是 **决定接下来做什么**，而不是直接代替 inner geometry solver。适合的 action 包括：

1. 选择下一张 anchor/evidence frame；
2. 决定在哪些 frame 重新跑 detector/SAM2；
3. 选择 active-set 中下一批 constraints；
4. 调整 trust-region/step-size/loss schedule；
5. 开关 contact mode 或选择 competing branch；
6. 决定是否请求人工/agent review；
7. active perception 场景中控制相机/手去获得更有信息的视角。

这时 environment 才真正具有 sequential decision 特征。

### 6.3 一个推荐的 RL toy 定义

不要让 policy 直接输出所有 `H_t,O_t,C_t`。可以定义 optimizer-controller MDP：

#### State

```text
s_k = [
  each loss-family value,
  gradient norms,
  pairwise gradient cosine,
  active constraint counts,
  Jacobian smallest singular values,
  2D/depth/contact residual quantiles,
  current uncertainty,
  previous accept/reject state
]
```

#### Action

```text
a_k ∈ {
  run object-pose block,
  run hand block,
  activate/deactivate contact factor,
  increase/decrease one weight,
  shrink/expand trust region,
  add one observation frame,
  reject and return to previous checkpoint
}
```

#### Reward

reward 不能只等于 training loss 下降。建议：

```text
reward =
  held-out observation improvement
  - constraint violation
  - canonical-state movement cost
  - uncertainty under-reporting penalty
  - render inconsistency
  - compute cost
```

必须使用 policy 未直接优化的 held-out evidence，否则 policy 很容易关闭难满足的 factor 或移动 gauge 来刷分。

#### Episode

每个 episode 是一个随机 synthetic scene：

- 不同 noise/outlier；
- 不同 contact mode；
- 不同 camera gauge；
- 部分 mask/depth 缺失；
- 部分错误 measurement。

先在 toy simulator 学 policy，再测试能否迁移到受控真实数据。

### 6.4 在 RL 前应比较的更简单 baseline

在投入 PPO/SAC 前，至少比较：

1. fixed schedule；
2. residual-greedy schedule；
3. grid/random search loss weights；
4. Bayesian optimization；
5. EM/switchable constraints；
6. learned optimizer / unrolled optimization；
7. imitation learning：模仿一个能访问 synthetic ground truth 的 oracle solver controller；
8. 最后才是 RL。

对于 loss weight，优先尝试：

- measurement uncertainty `w=1/sigma²`；
- robust scale estimation；
- GradNorm/gradient normalization；
- augmented Lagrangian/dual variables；
- constrained optimization。

它们通常比 RL 更稳定、更可解释。

---

## 7. 推荐实验顺序

### Stage A：先看单模块

依次看：

```text
sim2 -> pose_graph -> halfspace -> mano
```

对每个 toy 改一个参数并观察：

- outlier count；
- target weight；
- temporal sigma；
- halfspace normal 分布；
- contact deadband；
- nonpenetration/contact/observation weight；
- soft bound。

### Stage B：做 loss sensitivity/Pareto sweep

对 P18-like toy 扫：

```text
contact_weight
penetration_weight
observation_weight
translation_bound
contact_deadband
```

画：

```text
contact gap vs 2D shift
penetration vs translation norm
raw residual vs post-gate residual
```

不要先找“总 loss 最小”的单点，要先看 Pareto front 和不可兼得区域。

### Stage C：理解 joint problem

运行 `joint_gauge`，然后依次加入：

1. camera anchor；
2. metric depth/scale anchor；
3. object pose prior；
4. contact switch；
5. outlier measurement。

每次记录 Jacobian/Hessian smallest singular values，确认新增 factor 是真正增加信息，还是只增加权重。

### Stage D：再做 optimizer controller

先实现 deterministic controller：

```text
if rank deficient: add/fix gauge anchor
if contact residual conflicts with observation: lower contact confidence
if active set does not close: shrink trust region or reject
if post-gate residual regresses: reject canonical promotion
```

如果 deterministic controller 在 synthetic distribution 上稳定，再将同样 state/action 包装为 RL environment。

---

## 8. 最终建议

最推荐的技术路线是：

> **神经网络负责测量和 proposal；稀疏 MAP/可微优化负责连续物理 state；switchable factors/EM 负责离散 contact/ownership；RL 只负责外层调度、主动选证据或 optimizer control。**

不要一开始把完整 V19 当成一个无结构的大 policy，也不要把所有 loss 简单求和后一次性解完。先用这些 toy 找出：

- 哪些变量有独立观测；
- 哪些 residual 真正有梯度；
- 哪些方向是 gauge；
- 哪些 constraint 不可行；
- 哪些 post-processing 会破坏 optimizer 的结果；
- 哪些离散状态需要单独推断。

这些问题搞清楚之后，再决定 joint optimization、learned optimizer 或 RL 的边界，会比直接训练一个 end-to-end policy 更可靠。
