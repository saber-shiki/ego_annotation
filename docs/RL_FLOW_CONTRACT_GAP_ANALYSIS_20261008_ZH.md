# RL 流程图对照审计与实施计划（2026-10-08）

## 1. 审计范围与基线

本文件对照图片：

- `/mnt/truenas-user-home/kupingxin/rl流程图-new.jpg`

代码审计基线：

- 仓库：`ego_annotation`
- 独立实现分支：`feature/rl-flow-contract-20261008`
- 基线提交：`6c4fbab`（`docs(preprocessing): trace SAM2 mask to pose consistency contract`）
- 独立 worktree：`/mnt/user-home/kupingxin/ego_annotation_rl_flow_contract`

### 为什么基线是 6c4fbab

最初曾从 `790893c` 创建 worktree，因为它是 RL observation-alignment 分支的干净提交。但复核后确认这不足以代表“最新提交”。比较结果如下：

```text
git diff --name-status 790893c..6c4fbab -- train.py rollout.py configs actions envs rewards models data observations
# 无输出
```

`6c4fbab` 自身只新增了 `docs/SAM2_MASK_POSE_AUDIT_20260928.md`，没有修改 `train.py`、策略模型、动作空间、AnnotationEnv、reward evaluator 或 RL safety logic。因此该提交**没有补齐策略前向、安全层、奖励评估器的 RL gap**。它补充的是上游 SAM2 mask-to-pose 观测审计，应该作为后续证据资格/不确定性输入，而不是被误认为已接入 RL 训练。

原始 `ego_annotation_preprocess_monocular` 工作树另有未提交的 preprocessing 改动；这些改动不属于 `6c4fbab`，本分支不偷偷纳入。后续若要消费这些改动，必须先形成明确提交并重新审计 RL adapter 契约。

## 2. 流程图要求摘要

### 2.1 策略前向

流程图要求：

```text
s_t = {O_t, X_t, h_(t-1), 证据资格与不确定性}
O_t = RGB、深度、相机、手、mask、表面、纹理/形状候选
X_t = 当前物体轨迹、完整几何、左右手状态及未决部分
h_(t-1) = 前序状态、记忆、支持度、残差、阶段记录、失败诊断

a_t = pi_theta(s_t)
```

动作需要覆盖：

```text
每手：关节姿态45 + 腕部旋转3 + 相机坐标平移3 + log scale1 = 52
物体：Delta R3 + Delta t3 + 全片共享 object scale1
```

当前实现：

- `models/policy.py` 使用 skrl Gaussian PPO residual policy；
- MLP 为 `256 -> 256`，ELU + LayerNorm；
- mean head 默认零初始化；
- policy 输出 raw residual action，环境后续执行过滤；
- 当前 action layout 是 118 维，缺手部 scale 和 object shared scale；
- 当前 observation 是 flattened numeric vector，不是完整 RGB/Depth encoder；
- 配置虽然有 `causal_history_frames: 8`，但 `envs/observation.py` 没有真正拼接 8 帧历史摘要。

结论：**PPO 前向骨架存在，但动作契约、历史摘要和完整证据输入不满足流程图。**

### 2.2 安全层（硬约束）

流程图列出的 G1-G9 包括：

- G1：视频、帧、对象、K、坐标、MANO、阶段契约一致；成功证据才可复用；
- G2：像素归属、二维资格、深度相近/无效/手缘未知状态；
- G3：P09 与锚点门控；深度连续性、异常值、尺度；帧排名/看图；当前只接受确认刚体对象；
- G4：生成面只能用于视觉候选，不能自动作为物体姿态/接触/碰撞/符号距离证据；
- G5：P14 初始支持度、RGB 桥、最大间隔；
- G6：PnP、正深度、RGB 独立检查及边缘重要性；
- G7：逐指吸附激活、初始间隙、primary digit、手/物体可见性；
- G8：动作物理边界；
- G9：逐帧状态可行性、显示门、out-of-frame/遮挡与原始有效性分离。

G8 的流程图硬边界为：

```text
手 scale:       0.85 ~ 1.18
手平移:         每轴 +/-25 mm
关节增量:       +/-0.35 rad
腕部增量:       +/-0.15 rad
物体旋转:       每轴 +/-0.08 rad
物体相机平移:   x/y +/-15 mm, z +/-35 mm
```

当前实现：

- `envs/dynamics.py:125-145` 有实体存在性、对象 pose validity、contact patch 和可选 translation support mask；
- `actions/scaling.py` 只做单位换算，不做硬限幅；明确注释为无 tanh/clipping；
- formal reward 中的 bound 是 soft penalty，不是安全层；
- G7 工具存在于 `actions/filters/contact.py`，但没有接入逐指 action mask；
- 当前 active contact 可能被用于屏蔽整个 hand block，粒度过粗；
- 物体动作配置为 world translation，没有实现流程图的 camera-frame G8 bounds；
- G1-G6 的大量证据 contract 在上游存在，但没有统一进入 RL `safe_action` 决策；
- visibility 状态在上游有 schema，但 RL 每步没有完整使用 visible/partial/occluded/out-of-frame/unresolved 和 occluder ownership。

结论：**当前是基础 semantic filter，不是 G1-G9 完整硬安全层；尤其 G7、G8、G9 未闭环。**

### 2.3 奖励评估器

流程图要求状态更新：

```text
X' = Apply(X, a_safe)
```

然后计算：

```text
r_t = - (L_pose + L_prior + L_time + L_depth + L_shape + L_hand + L_pad)
```

主要奖励含义：

- `L_pose`：物体相对运动、RGB/PnP/重投影；
- `L_prior`：初始观测先验和不确定性加权；
- `L_time`：手和物体修正的速度/加速度连续性；
- `L_depth`：投影后深度可行性和前后顺序；
- `L_shape`：完整物体外观、轮廓/IoU；
- `L_hand`：手投影和 MANO 参数先验；
- `L_pad`：指腹到物体三角面的无符号视觉接近，仅在 G7 激活时使用。

当前实现已经具备：

- `AnnotationEnv.step()` 先 filter action，再 decode MANO/object，之后进入 evaluator；
- `RewardEvaluator` 使用 prefix objective；
- `gamma=1` 时，逐步 reward 累计等于最终 objective 的负值；
- 不在 reward evaluator 中调用 Adam；
- formal hand、contact、visible depth order、object observed point-to-plane/point-to-point、temporal terms 已实现；
- visual fixed-scale profile 中有 joint/surface projection、silhouette 和 unsigned pad 代码。

但以下 gap 仍存在：

- 当前默认 `formal_annotation` 不包含完整 `L_shape`/visual silhouette 组合；
- RGB edge、RGB reprojection、first-hit/silhouette image rows 只有在 factor rows 提供时才激活；
- signed nonpenetration 依赖 watertight、signed-ready 和 local face authority，缺证据时 inactive；
- source pose sigma prior 和 nonpenetration target 缺字段时 inactive；
- `visual_joint_fixed_scale` 没有手部 log scale；
- `L_pad` 不在默认 formal profile 中；
- factor 缺失通常只记录 inactive，不阻止声明为目标 profile 的训练；
- temporal 主要依据 frame index，尚需严格检查 `time_s` 语义。

最新可核实的 1024-step milk 训练中，`joint_pixel_deadband` 最终约占 objective 的 99.97%，且训练 checkpoint 没有优于零动作基线。真实 ledger 还显示 RGB factors、object sigma priors、nonpenetration target、signed terms 均未激活。

结论：**prefix reward accounting 结构是正确的，但当前默认 reward 不是流程图的完整 scene-v7 objective，且关键 factor 缺失会静默退化。**

## 3. 满足度矩阵

| 模块 | 当前状态 | 结论 |
|---|---|---|
| PPO/Gaussian policy 前向 | skrl PPO + residual MLP | 部分满足 |
| `Apply(X, a_safe)` 顺序 | filter -> decode -> reward | 满足 |
| raw/executed trace | 已导出 | 满足基础审计 |
| 手部/object scale action | 未实现 | 缺失 |
| causal history `h_(t-1)` | 只有上一动作/当前 residual | 缺失 |
| G1-G6 evidence contract | 上游分散存在 | 未形成 RL 闭环 |
| G7 逐指吸附 | helper 存在，未接入 policy mask | 缺失 |
| G8 硬边界 | 当前无界 raw action + soft reward bounds | 缺失 |
| G9 visibility/validity | 基础 valid flags，语义未完整接入 | 部分满足 |
| `L_pose` | observed geometry 已生效，RGB 可选 | 部分满足 |
| `L_prior` | hand 生效，object sigma 可选 | 部分满足 |
| `L_time` | hand/object velocity+acceleration | 基本满足 |
| `L_depth` | hand depth order 生效，image first-hit 可选 | 部分满足 |
| `L_shape` | default profile 未覆盖 | 缺失 |
| `L_hand` | projection + MANO priors，无 scale | 部分满足 |
| `L_pad` | visual profile 可选，default inactive | 部分满足 |
| v2 observation trainable adapter | `preprocessing_only` 被拒绝 | 缺失 |

## 4. 按优先级的实施计划

### P0-A：冻结动作/安全契约并实现 SafeActionLayer

新增独立安全层，不继续把所有行为塞进 `filter_action()`：

```text
a_raw = pi_theta(s_t)
a_safe, safety_trace = SafeActionLayer.project(X_t, a_raw, evidence_t)
X_(t+1) = Apply(X_t, a_safe)
```

实现内容：

1. 明确 action layout version；
2. 增加 hand log-scale 和 shared object-scale 的动作契约；
3. 明确 contact action 是否保留；如果保留，按 finger/hand 与普通 pose action 分开；
4. 实现 G8 硬边界，并明确 world/camera 坐标变换；
5. 将 entity validity mask、contact activation mask、physical bound projection mask 分离；
6. 接入 G7 primary digit / likely-contact / visible support；
7. 接入 G9 visibility/occlusion/out-of-frame gating；
8. 记录 raw action、safe action、per-dimension mask、constraint id、violation magnitude 和 reason；
9. 保证 MANO replay、object transform、render candidate 使用同一个 `a_safe`。

### P0-B：修正策略分布与安全投影的一致性

当前 raw Gaussian 的 log-prob 与 projected safe action 不严格一致。后续选择并实现一种方式：

- bounded/squashed Gaussian，并使用正确 log-prob correction；或
- 将 active mask 作为 policy input，并对 active dimensions 使用一致的 probability accounting。

在完成前，不宣称是严格 constrained-policy PPO。

### P0-C：实现真正 causal history

在 `AnnotationState` 增加固定长度 history buffer，至少记录：

- raw/safe action 摘要；
- gate mask/reasons；
- 当前 residual magnitude；
- factor-family objective/reward delta；
- evidence support/uncertainty；
- visibility state；
- stage/failure code。

`pack_observation()` 只加入固定维度归一化摘要，不放完整 ledger，不泄漏未来帧。

### P1-A：新增明确的 `scene_v7_rl` reward profile

将 reward family 命名为：

```text
L_pose + L_prior + L_time + L_depth + L_shape + L_hand + L_pad
```

各 family 必须有 activation report。对 `scene_v7_rl` 配置：

- 必需 factor 缺失时拒绝训练，不能静默退化；
- 非必需因子可以以 explicit inactive 状态继续；
- 保持生成 mesh 只能是 visual display prior，不能作为 physical pose/contact/sign authority；
- `L_pad` 只消费 G7 激活的指腹 rows；
- 加入手 scale/object shared scale 的 prior 和 temporal term；
- temporal terms 使用真实 `time_s` 或明确的 frame-time conversion。

### P1-B：实现显式 v2 -> trainable RL adapter

不删除 `preprocessing_only` guard，不直接把 v2 当旧 v1。

新增版本化 adapter，例如：

```text
ego.annotation.observation_bundle.v3_rl
```

adapter 必须输出并绑定：

- observation/action layout version；
- reward profile；
- G1-G9 evidence qualification；
- full timeline validity；
- visibility/occluder uncertainty；
- factor rows 和 activation provenance；
- mesh role/geometry authority；
- source/model/config hashes。

只有 adapter 输出满足 RL contract 后，`EpisodeDataset` 才能加载。

### P2：受控验证与训练门禁

先运行：

1. safe layer arbitrary-action bound tests；
2. G7/G9 state transition tests；
3. reward finite-difference direction tests；
4. zero-action vs one-step controlled-action tests；
5. short PPO smoke；
6. complete episode rollout；
7. factor activation contract audit。

只有单因素动作能改善对应 factor、reward 符号正确、zero baseline 不被无故破坏后，才增加训练步数。

## 5. 不应做的修改

- 不要仅删除 v2 `preprocessing_only` guard；
- 不要把 soft reward bound 叫作 hard safety；
- 不要把 generated shape mesh 当 signed collision truth；
- 不要把 contact patch proximity 当 signed contact force；
- 不要通过删除最大 pixel penalty 来制造 objective 改善；
- 不要在 safety projection 后继续假装 raw Gaussian log-prob 完全对应执行策略；
- 不要在没有 factor activation evidence 时声明完整 scene-v7 reward 已训练。

## 6. 当前分支实施状态

本分支从 `6c4fbab` 干净提交创建。已落地的机制性修改：

- `actions/safety.py`：G8 物理硬投影；手 root/articulation/translation、物体旋转、物体相机坐标平移、contact logit 及可选 hand/object scale bounds；保留 per-dimension constraint id/violation/projection trace；
- `envs/dynamics.py`：将 semantic gate、G9 `out_of_frame` gate 和 hard projection 接入 `AnnotationEnv -> Apply(X, a_safe)`；可选 hand log-scale/object shared log-scale 会实际作用于 MANO/物体几何；
- `envs/observation.py` / `envs/state.py`：固定宽度 causal history（可配置帧数，默认配置为 8）和可选 scale state/observation；
- `rewards/profiles.py` / `rewards/evaluator.py`：新增 fail-closed 的 `scene_v7_rl` profile，实际计算 shape/pad rows，并在自然 episode 末尾检查 pose/prior/time/depth/shape/hand/pad family 是否都激活；
- `data/v2_rl_adapter.py`：增加 v2->v3_rl 的显式 adapter contract inspector；它不会删除 `preprocessing_only` guard，也不会伪造 trainable promotion；
- `configs/ppo_scene_v7.yaml`：提供启用 scale action、causal history 和 `scene_v7_rl` 的新配置，不替换 legacy `ppo_annotation.yaml`。

验证结果：

```text
python3 -m unittest actions.test_safety envs.test_observation_history data.test_v2_rl_adapter rewards.test_reward_contract -v
# 11 tests passed
/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -m unittest \
  data.test_interaction envs.test_annotation_env envs.test_v2_consumption \
  rewards.test_evaluator models.test_skrl_runtime -v
# 39 tests passed (skrl emitted only its gymnasium.experimental warning)
/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -m py_compile \
  actions/*.py envs/*.py rewards/*.py data/*.py models/*.py train.py rollout.py render.py
# passed
```

系统 `python3` 仍缺少 RL 依赖，但仓库的 `/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python` 已用于上述环境回归；不能把系统解释器导入失败当作代码回归结论。

尚未完成、不能宣称已解决的机制：

1. raw Gaussian policy 的 log-prob 与 hard projected action 仍未做严格 constrained-distribution correction；
2. G7 仍是 contact block 级别的 gating，尚未把 45 维 articulation 拆成逐指 action block；
3. v2->v3_rl adapter 只有显式 contract/拒绝接口，还没有真正把 v2 bundle 转成可训练 EpisodeBundle；
4. scene-v7 的真实数据仍需提供 shape/pad factor rows、RGB/image factors 和完整 evidence qualification 后，才能通过 fail-closed reward contract；
5. 还没有在新契约上执行真实训练或宣称视觉质量改善。
