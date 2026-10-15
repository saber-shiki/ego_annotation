# RL 分支恢复 V7 标注能力：实施计划

## 用户授权与执行顺序

用户于 2026-09-17 批准按照下面五个阶段完善 `case/rl-refactor`。这是持续实施任务，不是仅交付审计报告；后续会话必须先读取本文件及 `.memory/tasks/2026-09-17-rl-v7-parity/README.md` 再继续。不得跳过缺失机制直接扩大 PPO 训练，也不得把 CPU 测试或参数更新当作标注质量完成。

起点：`823a906da71a03bfbb185fa5045d982f2684e7c8`。
V7 指 `STANDALONE_RGB_SCENE_V7_BASELINE.md` 的 standalone RGB scene，不是早期 `pipeline_v7.md`；冻结数值源码为 `fa9a84cbfb6836b92e35a6316073a6855e6b4d22`。开发侧已只读核对发布 bundle/candidate/final JSON/RRD 哈希。

| 顺序 | 阶段 | 必须实现的机制 | 状态 |
|---|---|---|---|
| 1 | 接通真实证据链 | judgment→P17 类型适配；独立 RGB 测量；appearance-native 全网格；typed pads、visibility；不可变数据装载 | 待实施 |
| 2 | 恢复物体轨迹能力 | 离线可回访候选修正、测量 seed、固定 gauge、连续时间 SO(3)/SE(3) 目标、PPO 接口 | 待实施 |
| 3 | 恢复完整手物拟合 | candidate canonical binding、全片物体 scale、V7 MANO camera-origin similarity、逐指 unsigned pads 与完整 visual 目标 | 待实施 |
| 4 | 完整交付闭环 | policy scene adapter、visibility clears、portable JSON/RRD、完整参数与动作/因子追踪 | 待实施 |
| 5 | 训练与泛化验证 | 明确授权服务器上的真实多视频训练、消融、整片渲染比较与吞吐验证 | 待实施；需要明确运行目标 |

阶段 2/3 的机制实现必须尽早接几何/可视化检查，不等训练结束才看结果。阶段顺序表示依赖；不能把一个小修复称为完成整个阶段。

## 范围、不变量和运行边界

- 首先恢复 V7：单选刚体、完整物体三角形、双手 MANO、全时间轴 camera/K/RGB、visibility/interaction/uncertainty、最终 `annotation.json` 与 `world.rrd`。MP4 是可选 QC，不替代最终文件对。
- V7 为 unsigned visual hypothesis，未认证接触、碰撞、SDF 或非穿透。保留 formal/signed 能力为独立 profile；generated display faces 不获得 signed authority。
- 不以质心、box、sphere、稀疏骨架代替重建 mesh 或完整 MANO。P09 identity+centroid 仅是源初始化，不能冒充物体运动估计。
- 不调用 legacy solver 来伪装 RL 更新。训练/rollout 不依赖 legacy 优化结果。共享 decoder、residual、registration、pure gauge helpers 放在 legacy 之外。
- 摄像机/深度/HaWoR/OWLv2/SAM2/SAM3D/Astra 作为预计算观察生产者；不在 PPO step 内启动大模型。Astra-only、不新增 fallback。
- raw evidence 不可变；测量派生 seed、资格判断、RL candidate、最终展示是不同角色。策略不能改写证据、隐藏困难帧、关闭观测或赋予物理 authority 来降低代价。
- 同一 cwd 只有一个写入者；不改其他工作区、既有运行、冻结 baseline、模型环境。未经授权不提交/推送。
- 本地只允许轻量代码工作和有界 CPU 合成测试：`CUDA_VISIBLE_DEVICES=''`、单线程数值库、无模型权重加载/无训练。真实模型推理、重建、PPO 训练和重 CPU 任务需要明确服务器/环境/GPU授权，不能使用默认 cuda。
- 初始环境探查：`ego_annotation/.venv/bin/python` 有 numpy/scipy/cv2/trimesh/torch/open3d 等但没有 gymnasium/skrl，rerun 为 0.37.0；`ego_skrl/.venv/bin/python` 有 skrl 2.1.0/gymnasium/torch，但缺域依赖。不得修改这两个既有环境或用混杂 site-packages 假装已配置统一 runtime。

## 阶段 1：观测与数据完整性

根阻塞：当前 contact judgment 严格 schema 不含 P17 loader 必需的 occlusion_relation/metric uncertainty；RGB 与 appearance-native 在 legacy 参考支路，RL 消费者拿不到；visibility 和 primary digit 判断没有传至 episode 数据。

### 1A — interaction adapter / typed evidence

- 在 observations 中显式合并独立 contact、pads、occlusion、depth reliability、visibility，精确按真实 frame ID、hand、target 连接，不把 frame ID 当连续数组位置。
- 显式映射两个 occlusion 词表；保留原判断与 provenance。缺失/冲突不得补成确定可见/接触。
- 文本 uncertainty 与米制 uncertainty 分开。米制值来自明确几何估计，或记录为保守配置 prior，不能谎称 VLM 测量。
- 编译 typed per-digit declarations 与五态 visibility，加载进入只读 EpisodeBundle。此阶段不宣称完成 primary reachable geometry 或 visual fitting。
- 使用当前严格 judgment schema 的合成响应测试真实 assembly→P17 loader，而不是手加历史字段；覆盖 no_contact、unresolved、重叠、foreign entity、非连续 ID。

### 1B — RGB 测量生产与装载

- 新 source observation 模式保存固定 source_world、target_uv、双端 camera/K/time、相对世界 motion、inliers/quality/support、失效原因，绑定 NPZ/JSON/hash。
- 解除测量格式必须提供 P14 optimized pose 的依赖；保留旧参考 CLI 兼容。不得接入包含 least_squares 的 `initialize()` 并称作 observation-only。
- 弱/无边是显式不确定测量结果；格式/坐标/hash 错误才是机械失败。
- 接入 cold main stage/recovery contract、manifest 和 EpisodeBundle。只读数据可供后续 seed/qualification，不在本阶段引入求解轨迹。

### 1C — appearance-native / 完整观测包

- 按现有独立 appearance decision 条件启动现有 SAM3D worker；保存 native mesh/pose、全部 faces、同帧相机与 retained metric support。不能通过旧 refined binding 获取输入。
- display mesh、pose hypothesis、observed/collision surface 分开。支持精确同输入/同模型 receipt 的复用；不同 mask/anchor 不允许伪复用。
- 新扩展需版本化，并使 v1 prepared bundles 显式兼容或报清楚能力缺失，不能悄悄升级旧 bundle。
- full frame ID/time 从声明时间轴读入，不用 arange 强行补洞。哈希和角色检查覆盖实际新消费者使用的内容。
- 验证 source→loader 全链；重 worker 仅在测试边界 mock。CPU mocks 不证明真实 cold inference。

## 阶段 2：离线 object learned refinement

- 新增明确 offline refinement 环境，保留现有 one-pass causal 环境供消融。episode=视频；step=候选更新，允许有限轮重叠窗口/节点回访。源完整观察可看，不把 offline 称为 online。
- 源观测、local RGB seed、候选轨迹分开；使用纯测量传播/独立 cycle decisions，固定 selected anchor。无 metric 但有实体/时序支持的节点为不确定 latent hypothesis，而非缺测即全动作屏蔽。
- 将 V7 RGB relative motion、双端 reprojection、near-depth、conditioned observed point/plane、SO(3) 秒单位时序项抽成共享纯目标，不搬入 least-squares loop。被拒 loop 的 pixel 因子也失效。
- actor 必须能看到 RGB edges/quality、时间差、候选 residual、geometry conditioning、边界状态，不仅看质心和上一动作。
- 对应采样身份固定；candidate-dependent associations 显式刷新；不能因远离观测导致对应消失而奖励逃逸。
- 阶段内采用固定目标改进奖励 `r=scale*(J_before-J_after)`、gamma=1；版本化 profile/观测/动作/解码。动作原始采样与执行变换分别保存。

## 阶段 3：完整 visual refinement

- 提取 V7 无 optimizer MANO decoder：fixed beta + source articulation residual → wrist-pivot root → camera-origin exp(log_scale) + camera translation → world。每手 52 个几何变量。
- 当前 formal decoder 与 V7 decoder 不假定数值相等，先并存并版本化；每个有效帧验证顶点、ordered joints、零/非零修正。
- appearance native 用阶段 A 候选 anchor pose 做纯 canonical binding；native metric similarity 与后续 vertex-mean-pivot image scale 分开。
- 全片共享一个 object scale；不允许每帧 shape 任意变动。完整 native faces 保持，candidate decoder/export 使用同一几何。
- primary/secondary 每指 12 顶点 pad，内部最近 3 样本独立计价；冻结 anatomical membership/evidence，动态刷新 visual closest triangle；formal stable anchor 仍是独立 factor。
- primary expanded reach 使用 current evidence、visibility、有限距离和一组共享 similarity/translation；不能每个指尖独立平移。V7 第一版不用可学 contact logit 关闭视觉约束。
- 新完整 visual profile 明确 960-width 图像平面、归约、尺度 prior/temporal、V7 bounds；不简单叠加 formal 与 visual priors。新归一化是独立实验，不静默改变 V7 目标。

## 阶段 4：完整可复现输出

- SourceObservationBundle / CandidateSceneState / AnnotationPackage 分开；policy 不伪造 refined_graph 输出或 legacy acceptance。
- source valid、candidate available、visibility、display valid 分开；out_of_frame clear，supported occluded 保留 uncertain，missing MANO 不制造确定手。
- 复用真正的 RRD triangle writer，接 policy adapter；最终文件对独立可读，不依赖外部 NPZ/PLY。
- 保存完整 frame/time/camera/RGB、source/seed/policy 区分、MANO 参数/scale、mesh 身份、contact/visibility/uncertainty。
- 内部逐步 raw/executed action、update indices、filter/reward/activation trace 必须落盘；refinement_step_count 不等于 video_frame_count。

## 阶段 5：真实验证与发布条件

- 先在明确授权的统一服务器 runtime 做完整域 smoke，再训 object，再训 visual，再做完整组合。
- 数据按原始录制/源视频 group 切分，不能只靠裁剪后 bytes；先划 split 再切窗口。固定同观察比较源零修正、V7 数值参考、RL 和消融；另做 raw-to-final 比较。
- 测每步 MANO/mesh/reward 成本，缓存静态几何与局部因子后决定并行度。需要 checkpoints/preprocessors/seeds/resume 完整性。
- 机制测试覆盖大角度、多错误 loops、重投影双端、cheirality、time/ID、scale 投影、primary 20–40 mm 支持、共享 reachability、visibility/reappearance、移走最终文件对仍能读取。
- 完整时间轴看 object rotation/translation/shape、双手形态与 scale、逐指 unsigned gap、投影、遮挡及 runtime。GT 只在预测冻结后独立评估。
- 未验证真实质量时保留 policy_candidate/annotation_ready=false。测试数、PPO 更新、schema 完整不代表 V7 质量等价。

## 更新规则

实施者在任务记录中写实际修改、准确命令与结果、未测项目和下一根阻塞；不要覆盖既有失败证据。保持表格状态与真实验证层级一致。

### 2026-09-17 实施记录 — 仅 1A

1A 已在工作树实现并通过 40 项有界 CPU 合成/契约回归，等待独立代码审查：当前严格 judgment schema 经 observations-owned adapter 接入真实 P17 loader；保留独立判断/provenance/diagnostic 角色；0.06 m 为显式配置 prior 而非 VLM 米制测量；EpisodeBundle 装载不可变逐指声明与五态 visibility，保留原 factors 和旧 v1 兼容；显式 sparse frame IDs 不补造中间帧。不可变数据新增的序列化兼容修复仅限现有边界。准确命令、日志和失败恢复记录见任务 README。

这是数据/证据链修复，不是渲染质量或阶段 1 完成。1B、1C 及阶段 2–5 仍待实施。当前域解释器缺 gymnasium/skrl，真实 policy export 集成只静态检查，未伪造统一 runtime 就绪；未执行模型、真实 MANO、训练或整片视频。
