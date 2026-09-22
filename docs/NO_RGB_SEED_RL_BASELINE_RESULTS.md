# 无 RGB seed 的 RL 完整网格基准与 V7 对齐审计

日期：2026-09-22。工作区：`/mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment`，基线`823a906`，未启动训练。

## 结论先行

当前 v2 RL 消费基准已经改为：

```text
完整 native visual mesh
        +
逐帧 source visible-geometry pose / P09 source initializer
        +
全部 RGB source measurements（accepted edges、source-world points、双端UV、质量、rejected诊断）
        ↓
RL未来候选状态的起点
```

逐段 RGB 传播出的 R4 seed 仍然保留，但只能通过显式模式
`initialization='rgb_seed_comparison'`使用。默认模式是
`initialization='source_full_native'`；默认候选不读取或应用 RGB seed 的 R/t。

这不是训练完成，也不是旧 `AnnotationEnv` 已接通。`V2ConsumptionSession`仍是
非训练验证/候选适配器，契约`trainable=false`，没有policy observation/action/reward
layout，也没有加载旧checkpoint或构建PPO。

## 本轮实现

### 代码边界

- `data/motion.py`
  - 新增`source_pose_initialization()`：复制当帧 source pose rows；缺行保持`available=false`。
  - 当 source pose row缺失但该帧有显式可见表面 centroid时，使用`R=I,t=source visible centroid`并标明`source_visible_geometry_centroid_initializer`；这仍是source/P09初始化，不是RGB seed。
  - `MotionPreparation`拆成`source_initialization`和`rgb_seed_comparison`。
  - RGB local forest、PnP边、loop qualification、reprojection weights仍全部计算；传播seed只作为comparison。
- `data/bundle.py`
  - v2 observation readback同时保留源帧行和两个初始化视图；不修改source。
- `envs/v2_consumption.py`
  - `ego.annotation.v2_rl_source_baseline_candidate.v1`。
  - 默认`source_full_native`，绑定完整native mesh到source anchor candidate。
  - 显式`rgb_seed_comparison`可复现旧seed对照。
  - RGB边和点继续通过当前candidate计算双端残差；测量源不因默认不使用seed而删除。
- `observations/inspect.py`
  - 默认几何绑定使用source initialization。
  - `source_pose_initialization.json`是默认候选基准；`measurement_seed_comparison.json`是显式对照；历史`measurement_seed.json`保留为comparison文件名，不再表示默认RL状态。
- 旧`data/build_index.py`、`data/dataset.py`、`envs/annotation_env.py`、旧actor/action/reward/checkpoint guard没有被绕过。

### 真实 r2 实测

源包：

`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2/source_observations.json`

- 完整native mesh：239,430 vertices / 478,880 faces。
- 默认source-baseline candidate available：149/150；source pose缺失帧：143。
- metric observation eligible：148/150；候选available和metric eligible保持分离。
- RGB source evidence：181 accepted edges、20,394 evidence points；全部保留。
- 默认reset/render没有读取RGB seed的R/t；seed对照单独实例化后与历史seed scene数组逐项相同。
- 默认候选与此前无seed完整native场景`v2_source_full_native_scene.npz`的关键数组逐项精确相同：camera、K、mesh、R/t、valid。
- 完整source readback：`complete_with_measurement_gaps`，唯一新增初始化警告是source pose缺失frame143；不是把该帧补成可信位姿。
- 默认控制验证：frame60/94实际更新完整mesh、手和RGB残差；148帧未控制状态精确保持；frame143物体控制明确拒绝；reset恢复；source manifest不变。
- 没有执行PPO、策略推理、优化器或checkpoint加载。

## 真实产物

根目录：

`/mnt/truenas-user-home/kupingxin/rl_v2_a800_execution_20260921/no_rgb_seed_rl_baseline/`

- `source_full_native_no_rgb_seed_h264.mp4`：**默认RL起点**，完整native mesh + source pose，150帧、960²、30FPS、5s。
- `rgb_seed_comparison_only_h264.mp4`：同一完整native mesh + 显式RGB seed，仅作对照。
- `source_full_native_no_rgb_seed_candidate/`：默认完整candidate JSON/NPZ/receipt。
- `rgb_seed_comparison_only_candidate/`：对照candidate JSON/NPZ/receipt。
- `results.json`：模式、hash、181边/20394点、149 available、未训练证据。
- `controlled_source_baseline/controlled_source_baseline_h264.mp4`：默认source-baseline的frame60/94受控消费检查视频。
- `controlled_source_baseline/results.json`：控制、未修改帧、缺测帧拒绝、reset及hash结果。

新的真实独立读取目录：

`/mnt/truenas-user-home/kupingxin/rl_v2_a800_execution_20260921/real_v2_inspection_source_baseline/`

其中：

- `inspection.json`：真实source-baseline前处理读取结果；
- `source_pose_initialization.json`：默认无seed初始化；
- `measurement_seed_comparison.json`：保留的seed对照；
- `arrays/complete_visual_geometry.npz`：完整native mesh实际读取。

## 与 V7 的逐项关系

符号：

- **一致**：同一语义/同一数学核在相同输入条件下对齐；不等于实际模型输出逐字节相同。
- **部分一致**：字段/局部核对齐，但输入、判断或下游阶段不同。
- **不一致/未接入**：当前明确没有复现V7阶段，不能宣称等价。

| 观测/前处理项目 | 当前 v2 RL | 与 V7 的结论 |
|---|---|---|
| 原视频、150帧时间轴、frame IDs | 同一真实视频，v2显式保存稀疏/时间连接 | **一致的时间轴契约**；真实V7输出仍有各自运行身份 |
| 相机/T与K坐标变换 | supplied HOT3D sensor K，明确source/960 render平面变换 | **局部契约一致**；历史V7实际候选使用过估计K约1248.57px，当前v2 K约974.34px，故历史数值产物不相同 |
| MANO source | source HaWoR world MANO、真实metric vertices/joints、双decoder replay | **源语义部分一致**；V7后续还有visual fitting，当前不拟合source MANO |
| MANO投影/front-only/depth ordering | 当前使用源MANO、depth、owned mask和前表面规则 | **数学核对齐**；实际mask/depth/模型判断不保证与V7每行相同 |
| raw/owned object masks | v2明确区分raw mask与depth-order owned mask | **角色一致/数值不保证一致**；v2不导入V7的后验mask答案 |
| P09 visible metric surface | 149 visible，148 metric-eligible | **流程语义对齐**；历史V7为132/132，实际覆盖不同 |
| metric eligibility/fail-closed | 缺资格时保留RGB/appearance，拒绝伪造metric | **策略原则一致**；逐帧资格结果不同 |
| RGB特征/PnP局部测量 | source-world固定点、目标UV、K、PnP/RANSAC+local LM、双端投影 | **局部测量核对齐**；同输入数值差分通过；实际producer输入不同会产生不同边 |
| RGB证据身份 | 保留181 accepted边、20394点、rejected/quality/loop diagnostics | **证据角色对齐**；v2不读取V7 P14 pose作为source输入 |
| 局部forest与独立loop | local temporal tree、conditioning、nonlocal loop reject | **资格/核对齐**；不是V7全局pose后验 |
| 默认物体候选位姿 | 完整native mesh + source/P09 pose；不使用RGB传播 | **刻意不等同V7 P14/P15**；这是本轮RL职责设计 |
| RGB propagated seed | 仍生成并保存为`rgb_seed_comparison` | **只作对照**；不得再称默认RL source或最终pose |
| P14/P15/global SO3/SE3 trajectory | 当前默认没有 | **不一致/未接入**；不能声称V7 refined trajectory已恢复 |
| 完整native visual mesh | 239430/478880，appearance-native R3，绑定明确 | **角色相近但不是同一产物**；V7历史native/visual拓扑和绑定运行不同 |
| object scale | v2只做single-view metric scene similarity，不做21档/共享尺度搜索 | **不一致**；V7有后续共享visual scale/fit（历史选过0.86），v2不把它当source观测 |
| appearance contour/distance | v2保存逐帧appearance预处理 | **前处理角色部分一致**；V7后续把appearance用于visual fit，v2目前不优化 |
| P17 typed interaction | v2保留contact/visibility/primary/occluder typed arrays | **schema/角色更明确，但语义判断不保证相同**；不把P17当物理GT |
| first-surface / contact membership | v2独立查询实际source surface，检查时不准备旧legacy contact patch | **角色原则一致但消费不同**；V7下游有自己的formal/P14/P15接法 |
| collision/signed authority | 完整visual mesh不获collision/signed权威 | **原则一致**；不把generated visual mesh冒充物理表面 |
| observation→policy | v2仍`preprocessing_only`，旧dataset/index拒绝 | **不一致/未接入**；旧480/118 actor没有自动消费v2 |
| candidate action/reward | 本轮只有验证适配器受控物理单位更新；无reviewed policy layout/reward | **不一致/未接入**；不能称为V7 final fit或RL训练结果 |
| 输出阶段 | source、证据、完整native网格、候选baseline/对照 | **前处理产物已实际运行**；V7完整后端最终能力不在本轮声明范围 |

## 因此“现在是否和 V7 一样？”

准确回答是：

1. **观测契约、时间轴、图像平面、front-only/depth eligibility、局部PnP/RGB证据格式和部分P17语义核，已按同一原则/局部数学核对齐。** 这些有代码差分测试和真实v2 readback支持。
2. **实际真实观测数值不一样**：K来源、mask、depth、SAM2/HaWoR判断、模型运行身份和资格帧覆盖不同，不能要求150帧输出与V7逐值相等。
3. **物体候选位姿现在故意不一样**：默认不使用RGB传播seed，更不导入V7 P14/P15；RL将从完整native mesh + source baseline开始，RGB边作为证据供未来候选更新。
4. **V7后续全局轨迹修正、共享尺度搜索、visual fitting、完整候选/最终优化尚未接入当前RL**。因此不能说“RL的观测和前处理已经整体等于V7”，只能说核心source/evidence前处理已做了可审计的局部对齐，后端职责仍明确分离。

## 测试与边界

本轮回归：**240项运行，235通过，5项跳过**。真实source-baseline读取、默认/seed comparison两个模式、默认受控消费、完整H264视频均完成；没有实际训练。

旧r2生产包、旧seed读回包、旧RL run、V7只读目录和原dirty workspace未改。源码工作区未commit/push。
