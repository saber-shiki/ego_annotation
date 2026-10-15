# Milk reward 消融：为什么最终变化看起来没有规律？

日期：2026-10-12。此文是原消融报告的**解释修正和机制诊断补充**，不是新增训练结果。

## 结论先行

用户的质疑成立：原来的两 seed、256-step 消融，不能可靠识别“某个 reward 对最终几何有什么因果作用”。原报告把最终轨迹变化称为表面深度／物体平滑的“显著影响”，容易被误读为这些项的物理职责已经得到验证；**该因果解释应撤回**。数值结果、视频和碰撞结果仍保留，不改写。

本轮不再新增训练，而把三层问题分开：

1. **Factor 本身是否响应目标几何变量？** 固定真实几何，只改一个已声明的物理变量，检查完整 ledger 响应。
2. **该响应是否进入可辨认的 PPO 学习信号？** 固定同一批动作／观测／critic，用各组真实 reward 重算 GAE 和 actor 梯度。
3. **最终学习轨迹是否稳定改善标注？** 原26条轨迹仍没有跨 seed 稳定优于 Zero，且全部150帧仍检出穿入／交叉。本轮没有制造新的标注改善。

已识别的具体失败机制：

- Full 的初始 advantage 几乎由像素大惩罚决定，移除碰撞等弱项几乎不改变它。
- 移除像素项后，其他 reward 太小，**未训练的随机 critic 的 TD／bootstrap 残差主导首次更新**。这不是“其他约束学得更好”。
- 自适应观测和 value 统计在更新开始时改变输入／值坐标，参数尚未更新就产生大范围 value clipping；这些变化混入 optimizer 的效果。
- 生成碰撞项的正样本数归一化仍有非单调问题，未在本轮修改。

这比“随机种子不同所以波动很正常”更具体，但**仍不等于已证明全部最终失稳都由某一个问题导致**。

## 运行边界与重建证据

- 服务器：`dexrobot-Standard-PC-Q35-ICH9-2009`；已配置环境 `/mnt/user-home/kupingxin/ego_skrl/.venv`；CPU、单线程；未安装包。
- 两个 seed 的初始 actor／critic hash 与原实验一致。
- 原始首128帧执行动作、frame index 和 float32 reward 全部逐元素一致。
- 从初始模型 RNG 重建的128个原始 Gaussian 动作也逐元素一致。
- 采样时原始 Gaussian 动作与 stored log probability 的最大重算误差为 **0**。
- 原始 PPO 的更新方法被截停，optimizer.step 被显式禁止。后续只 forward／backward 求梯度，模型参数 hash 不变。**本轮 optimizer 更新总数为0**。
- 没有感知重跑、额外训练、旧策略 warm start、改权重／归一化／阈值、替换 baseline 或推广 checkpoint。

源码依据是本机已安装 `skrl 2.1.0` 和冻结实验配置，不是按其他版本推测。

## 1. 最明确的学习信号问题：像素项移除后，随机 critic 接管了首次更新

GAE 使用的 TD 残差为：

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

critic 尚未学习时，V 是随机初始化网络的输出，不是可信的几何质量。GAE 又将 advantage 标准化到接近单位方差；原本很弱的信号也能被放大成非零 actor 更新。

真实首128帧中：

| 指标 | seed0 | seed1 |
|---|---:|---:|
| Full 单步 reward 标准差 | 0.886022 | 0.949566 |
| 去 hand_pixel 后 reward 标准差 | 0.000195 | 0.000191 |
| 初始 critic V 标准差 | 0.114824 | 0.151147 |
| Full advantage 与“reward-only、V=0”的方向余弦 | 0.999632 | 0.999404 |
| 去 hand_pixel 的 advantage 与“reward=0、原随机 critic”的方向余弦 | **0.999837** | **0.999930** |
| 去 hand_pixel 的 advantage 与“保留 reward、V=0”的方向余弦 | 0.077562 | 0.050318 |
| 去 hand_pixel 首 minibatch actor 梯度与“reward=0、原随机 critic”的方向余弦 | **0.999880** | **0.999948** |

方向余弦接近1表示几乎同向，**不是方差贡献百分比或统计显著性**。reward 与 V 标准差也不能单独替代 GAE 分解；这里的结论依据是实际 GAE 和梯度隔离结果。

因此，原结果中“去 hand_pixel 后完整 J 降低”，不能解释成没有像素项时其他约束学得更好。至少首次更新主要沿随机 critic 给出的方向走。后续 critic 状态、优化和归一化还会变化，本轮未将首更新结论外推为所有后续更新的唯一解释。

## 2. 弱项在初始学习信号中几乎不可辨认

同一 seed 下，各组首128帧动作和 critic 相同，只有被删除的 reward 不同。对真实 reward 重算 GAE：

| 去除项 | 与 Full 的归一化 advantage 差 RMS：s0 / s1 | 首 minibatch actor 梯度差 norm：s0 / s1 |
|---|---:|---:|
| generated_collision | 1.39e-6 / 1.03e-6 | 0.000156 / 0.000101 |
| hand_surface_depth | 0.000440 / 0.000359 | 0.052390 / 0.027325 |
| object_temporal | 2.49e-5 / 1.82e-5 | 0.002073 / 0.001645 |
| hand_pixel | 1.365292 / 1.442911 | 153.523666 / 137.924133 |

Full actor 梯度 norm 为86.21／115.44，且用的是同一原始首 minibatch。

特别是 object_temporal：首次 advantage 和梯度几乎没变，最终完整 J 却相对 Full 改变很大。它说明短训练轨迹对微小学习信号变化非常敏感；**不是直接证明该平滑项在稳定控制物体或双手**。hand_surface_depth 同理。

contact／hand_joint_depth／hand_bounds 的原有严格负对照仍成立：两 seed 最终模型、scaler、动作、几何都相同。它们在该配置／轨迹下没有可检出的训练效果，不能推广为一般无用。

## 3. 参数未更新，value clipping 已大量触发

已安装 PPO 的实际调用顺序：

```text
采集 rollout：obs_scaler(obs, train=False)
GAE：使用当时的 raw-unit V、reward 和 bootstrap V
存 old values：value_scaler(values, train=True)
存 returns：value_scaler(returns, train=True)   # 统计又变一次
第0 epoch，每个 minibatch：obs_scaler(batch, train=True)
预测 value：当前网络输出
value clipping：old_value + clamp(prediction - old_value, -.2, .2)
```

存在两个独立的坐标转换问题：

- old values 在 count=129 的 value 统计下编码；returns 在 count=257 的统计下编码。相同 raw V 在两套统计中的最大差为 **3.03／3.13**，远超过 value clip=.2。
- 观测统计也在 minibatch 中变化，fixed critic 的输入已不是采集时的输入。网络 head 没有伴随 value 统计变化做保留 raw V 的坐标重映射。

因此，“old V 与预测 V 的差”混入了统计／输入坐标变化，不再只反映 critic 参数学习。

为避免把多个因素混为一个原因，对同一真实首 minibatch 做了2×2**无 optimizer 的反事实检查**：

| 统计处理 | seed0 value clip 饱和比例 | seed1 |
|---|---:|---:|
| 已安装更新顺序 | **68.75%** | **68.75%** |
| 只冻结 observation 统计 | 93.75% | 75.00% |
| 只冻结 value 统计 | 68.75% | 56.25% |
| observation／value 均固定在采样时统计 | **0%** | **0%** |

两套统计都固定时，未更新参数的 predicted V 与 stored V 平均差仅约1.7e-7／2.1e-7。这证明上述统计转换足以在优化前引入 clipping，但也说明**只改一个 scaler 不一定立即改善 clipping**，不能把所有68.75%归给单一 value 分支。

本机实现是对 predicted V 本身 clamp，超出区间时该样本的 value 分支导数为0。actor 与 critic 是独立网络，不是共享 encoder；所以这不会直接将 actor 梯度也置0，但会影响 critic 学习，并经 GAE 和联合 gradient clip 影响后续训练链路。联合 clipping 使用 actor＋critic 参数的总 norm；没有证明它单独解释最终差异，Adam 也不能简单当成 SGD 的线性缩放。

此反事实仍沿用原 RunningStandardScaler 的 clip_threshold=5，不是调好尺度的可用训练配置；**0% clipping 是一致性诊断，不是性能改善证据**。

## 4. 观测统计变化本身也会改变策略

必须保留两个看似相反、但均成立的事实：

1. 初始 actor 的 mean head 全零，因此首次更新前改变观测统计仍不改变 mean／log probability；两 seed 的首 minibatch raw log-prob 差都为0。**不能指控原始动作／likelihood 存储错误。**
2. 即使初始 mean 不变，actor 的特征和梯度会改变。原始更新 vs 固定采样统计的首 minibatch actor 梯度方向余弦为 **0.507／0.442**。

对最终 Full actor 的参数保持不变，只改变 normalizer 做敏感性测试：

- 初始统计 vs 最终统计，精确条件 Gaussian KL 均值5.275／5.407。
- 在最终统计 count=257 的附近，仅用同一16条已记录观测更新一次统计，条件 KL 均值0.001045／0.001338。
- 完全不变的 normalizer 负对照：mean 差、log-prob 差、条件 KL 全为0。

这些是**固定模型／固定观测的 normalizer 敏感性**，不是缺失的历史中间 checkpoint KL，也不是第二个真实 rollout 的更新重现。初始－最终统计差较大，不能将其数值直接当作某次历史 PPO 的违规幅度。自适应 normalizer 是复合策略的一部分，它改变时是实际策略变化；概率比并不因为此事自动“算错”，但不能把该变化都归于 optimizer 参数更新。

最小一致性修正应把预处理器也纳入策略／value 状态，明确其冻结区间和单位变换，而不是继续将 scaler 漂移与网络学习混为一谈。

## 5. 固定几何测试：reward 不是全无规律，但有冲突和尺度失衡

使用同一 Zero R6 的真实 metric MANO 和完整 SAM3D 三角网格，固定历史、相机、物体、其余动作，只改一帧的手平移；没有 proxy mesh，也没有新标注输出。

以下数字是**完整 prefix family cost 的变化**，不是逐帧 reward，也不跨不同 prefix 长度直接比较大小。

### frame120：左手沿相机 Z 平移

| 平移 | surface-depth cost 变化 | generated-collision cost 变化 |
|---|---:|---:|
| -2mm，朝相机 | +1.723584 | +0.002489 |
| -1mm | +0.841784 | +0.000528 |
| 0 | 0 | 0 |
| +1mm，远离相机 | -0.801794 | +0.026854 |
| +2mm | -1.563585 | +0.032970 |

可见深度顺序项能明确响应相机深度；但往它偏好的方向走，并不保证相对当前生成实体的穿透成本下降。**P09 source first-surface depth 与生成实体不是同一个约束**，不能用其中一个代替另一个。

### frame120：左手沿相机 X 平移

- ±2mm 时 hand_pixel cost 仍为0，落在12px deadband内；不是 reward 未接通。
- ±10mm 时 hand_pixel cost 增加约 **106375／106377**。
- 同时 generated-collision cost 只变化约 +0.024385／-0.024798。

这显示 deadband 和跨项尺度的具体效果，不能只看最终总 J 断言像素保持项好或坏。

### frame0：不同平移方向

- 右手 world-Y +0.5／+1／+2mm，生成碰撞 cost 分别降低0.005563／0.005829／0.006038。
- 左手 world-Z +0.5／+1／+2mm，碰撞 cost 变化为+0.011030／+0.004611／-0.014168，不是单调下降。

后一曲线同时受真实几何接触集切换和原正样本平均影响。已有完整 audit 证明：平均 barrier 降低可以伴随总穿入平方和／crossing数上升。该归一化必须另行修正，不能仅凭这条局部曲线断言是光滑或噪声。

本次只测试了这些明确的手部平移，不声称已经为全部12组、所有关节／物体变量、全部150帧建立响应图。

## 6. 修正后的实验判断与下一步

原来的消融应重新标为：

> **不稳定训练系统的短预算敏感性实验，不是 reward 的有效性排序。**

可保留结论：pixel 尺度主导、弱项难以进入初始 learning signal、部分项当前没触发、训练不稳定、穿模未解决。

应撤回结论：从终点 J 大降直接推断 surface-depth／object-temporal 在按对应物理职责产生稳定作用；从 drop_pixel 低 J 推断余下约束学得更好；把改变 seed 的结果当作 reward 方向证明。

下一次实施必须按以下顺序，而不是再跑完整26组：

1. **先修同一个机制：采样－更新一致性。** actor 输入和 critic 值坐标在整个 rollout／更新中有明确固定的 contract；old V、target、head prediction 用同一单位。采用固定、训练侧定义的预处理或明确的保值坐标转换；不能只把 old V 再编码一次而忽略 head／输入变化。
2. **单独处理初始 critic 的信号主导。** 与随机 critic 对照，测试零输出初始化或明确的首 rollout reward-only advantage；不以修改全部 reward 权重掩盖这一问题。这是第二个独立干预，不能同一次训练一起改再归因。
3. **再单独修碰撞归一化。** 固定采样集／固定分母等方案须用同一几何上的深度、总平方残差和 crossing 同时验证，而非只让平均 J 好看。
4. 然后用小而明确的训练对照检查更新方向、完整150帧渲染和碰撞指标；链路表现可解释后才做 reward 消融。保持双手可训练和118D，不倒退为冻结手部。

这里写的是由证据约束的后续方案，**没有自动实施默认配置修改，也没有启动新训练**。原 Zero R6 继续保留；不提升任何策略为新 baseline。

## 文件与复现

诊断根目录：

```text
/mnt/user-home/kupingxin/ego_annotation_experiments/ppo_invariant_diagnostic_milk_20261012_r1
```

- `seed_0/diagnostic.json`、`seed_1/diagnostic.json`：重放／统计／GAE证据。
- `seed_*/first128_replay_tensors.npz`：实际重放采集的 tensors。
- `gradient_followup_r2/analysis.json`：首 minibatch 梯度、2×2 scaler 隔离和 endpoint normalizer 敏感性；r1保留。
- `fixed_pose_reward_response.json`：固定真实几何的24条受控响应。
- `diagnostic_summary.png`：已实际打开检查的诊断图；不是新的标注视频。
- `regression_r2.log`：诊断／消融审查回归。
- `delivery_verification.json`：冻结代码、原checkpoint和原rollout绑定的读取检查；不是质量评价。

源码：

```text
tools/rl_experiments/diagnose_ppo_update_invariants.py
tools/rl_experiments/analyze_ppo_replay_tensors.py
tools/rl_experiments/probe_fixed_pose_reward_response.py
tools/rl_experiments/test_diagnose_ppo_update_invariants.py
```

运行使用上述已配置 Python，`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。所有脚本只写新的诊断输出目录。首128重放脚本拦截 PPO 更新；梯度脚本不创建 optimizer；受控几何脚本不训练策略。原训练输入／结果目录不写入。

本轮改进的是**失败原因的可辨认性**，不是物理标注。原26条视频的质量判断和穿模结论不变。
