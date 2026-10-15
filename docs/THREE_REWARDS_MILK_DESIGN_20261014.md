# Milk 三组最小 reward 调试设计（2026-10-14）

## 因果目的和边界

原消融的严格问题是 learning signal 被像素大惩罚、随机初始critic和变化的归一化统计混淆，最终穿模没有消除。本实验将职责减为三组，以归一、固定分母的几何成本提供可以受控检验的方向，同时使用明确固定的 PPO 输入/value 单位。

**不是原12组的单因素消融**：reward、preprocessor、critic初始化、探索幅度和PPO预算作为一个最小调试系统重新定义。成功也不能将贡献独立归给某项。

分支 `feature/milk-three-rewards-20261014`；工作树 `/mnt/user-home/kupingxin/ego_annotation_rl_three_rewards`。从原实验未提交改动复制继承，源工作树不动，不提交/merge/push。

### 输入

同一150帧 Milk R6 coarse_scene.v1，source v2，551D因果观测，118D完整动作，双手根姿态/关节/平移、物体姿态及接触logits可训练。原metric MANO betas和native SAM3D的239430顶点/478880三角面不变。无CAD/GT/新感知/旧checkpoint。

无独立GT，所以“保持交互不偏离GT”只能改为**保持R6参考几何/图像/交互**。SAM3D inside/outside是用户认可的生成实体假设，不提升source signed authority。软reward不构成安全或准确性保证。

## 三组成本

所有frame项使用固定全片frame_count作分母，prefix只累计已执行帧；不存在prefix增长改变旧项均值的隐藏影响。单位尺度和weights固定，训练前写入config。

### 1. reference_alignment

每帧按存在的entity等权平均：

- 每手：完整MANO表面相对参考的mean squared displacement / (5mm)^2。
- 手的可见joint：smooth squared pixel displacement / (5px)^2，按冻结可见度归一，无12px deadband或百万尺度权重。
- 每物：完整native mesh的mean squared world displacement / (5mm)^2。用全部native顶点均值/二阶矩进行精确二次型计算，不简化/抽样mesh。
- 每手物对：完整MANO表面在当前object坐标中对参考object坐标的mean squared displacement / (5mm)^2，保持相对交互。
- 接触logits以冻结参考prior作轻量平方保持，避免118D中的logits自由漂移；它不能降低penetration。

### 2. nonpenetration_contact

对每个有效手物pair，查询全部MANO顶点和所有triangle centroid在完整object实体中的signed depth。

`mean_all(max(depth - 0.1mm, 0)^2)/(0.5mm)^2 + 0.1*(max_depth_residual/0.5mm)^2`。

分母是**固定完整采样集大小**，不是positive sample count。max-depth项关注最深点；两个项均不受contact logits控制。此离散表面采样不数学保证所有三角相交被惩罚，独立audit继续检查实际hand-edge/object-triangle crossing。

对已有冻结contact patch，实际MANO vertex与当前object变换后的native-anchor距离若比R6参考距离多2mm，则平方惩罚 / (5mm)^2。冻结source prior作权重，不使用policy contact概率；不强迫不可靠P09/生成接触绝对距离成为0。source patch不等于真接触。

### 3. temporal_continuity

真实time_s、只跨有效连续frame：完整hand surface和完整object surface的**修正位移**速度/(30mm/s)、修正加速度/(1m/s²)的平方平均。固定全片edge/triple分母，已执行prefix仅添加可评价edge/triple。whole-native object项仍由完整顶点二阶矩精确计算。

其含义是避免新增修正引入跳变、保留参考真实运动，**不是保证消除参考原有抖动**。最终同时报告实际轨迹和修正轨迹的速度/加速度，不以修正平滑偷换实际平滑。

初始组weights：reference=1，nonpenetration_contact=5，temporal=0.25。前置扰动检查如发现reward方向/单位/连接错误则停止训练并修实现，不做看到训练结果后调权重。

## 回报和PPO

训练回报 `r=-[(Jprefix-Jzero_prefix)-(Jprev-Jzero_prev)]`，gamma=1，完整episode return为 `-(Jfinal-Jzero_final)`。Jzero是同一冻结参考，不是policy输出或额外观测输入。独立评价同时报告raw三组J、zero-relative J、原formal完整J、碰撞audit。

显式runtime contract `fixed_asinh_raw_value_zero_critic.v1`：

- observation固定asinh变换，无学习统计、无局部batch更新；value不归一，stored old values/returns/prediction单位相同。
- actor/critic独立模型，mean和value head均初始化为0。原actor Gaussian raw sample与log_prob严格配对；environment deterministic gate不改变likelihood数据。
- rollout150：完整episode，避免128帧预算切断；GAE lambda=1以明确MC credit。
- initial log_std=-2.5，范围[-5,-2]；原physical scales .05rad/.01m不变；118D两手不冻结。
- epochs2、minibatches5、lr1e-4、clip0.1、value_clip0.2、grad norm0.5、KL阈值0.01，无entropy奖励。
- 专门检查optimizer前ratio=1、Vpred=Vold及value clipping未误触发；运行后记录actual optimizer步数和exact Gaussian KL（固定preprocessor下）。

本契约只在本实验config启用；旧train默认和实验快照不修改。

## 前置预测与受控检查

- 固定mesh/相机，手或物体参考扰动应增加reference cost；共同平移手/物应保持相对交互部分但改变绝对保持项。
- 外移穿入手应存在降低collision方向；改变contact logits不能改变collision；固定分母不能靠增多inside点稀释cost。
- 相邻常量修正比alternating修正时序成本小，反向瞬间跳变产生大成本。
- zero mean/value初始状态，无optimizer时log-prob、ratio、value units精确/数值容差一致；无reward时初始advantage为0。
- 用真实frame0/120和真实完整mesh检查上述变化，若PPO仍恶化则最小reward体系在此短预算中无可重复效果，不加预算掩盖结果。

## 执行预算

服务器 `dexrobot-Standard-PC-Q35-ICH9-2009`，已配置 `/mnt/user-home/kupingxin/ego_skrl/.venv/bin/python`。数值执行单线程CPU/禁CUDA，最多2个独立seed进程，canonical tmux `ego_annotation` 的独立window。immutable snapshot + fresh run root。

- 代码/受控检查约30–60分钟；如机制错误先停止。
- **600steps × seed0/1，总1200环境steps；每150steps独立完整150帧评价**。fresh policy，无自动续训/挑选checkpoint扩预算。
- complete native/MANO Zero/seed0/seed1 RGB和world视频，全部150帧/30fps/5秒。
- 完整碰撞audit、原formal成本、三组ledger、source/camera/timeline identities和实际视觉审查，预计运行/渲染20–40分钟。

时间估计不是标注改善承诺。保留失败输出，原R6、旧政策、源工作树/快照不变；不推广default。
