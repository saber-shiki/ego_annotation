# Milk 直接非穿透求解结果（2026-10-14）

## 结论：已完成真实几何尝试，但“消除穿模”没有实现

本轮确实关闭视频对齐、不训练PPO，直接修改双手的MANO参数及物体SE(3)，执行全部150帧。完整三角交叉减少29.24%，采样内部深度平方总量减少44.67%，最深点减少27.06%；但双手同时无碰撞的帧仍为0/150，间距和实际运动副作用恶化。输出只保留为失败实验，不替换R6。

**更正可实现性表述：** 上轮“可以实现”指可以实施直接几何约束求解，不是已经证明当前输入在小修正范围内有可找到的零穿透解。本轮局部求解失败不能证明全局不可行，尤其当前整手退出模式过于保守、QP多数未收敛。不能将原因归结成“网格不能解决”，也不能把深度下降叫作已消除穿模。

## 1. 实现的真实机制

- 完整native物体239430顶点/478880三角面，真实metric MANO，固定betas、尺度、相机和time_s，没有网格修补、简化体、CAD/GT或新感知。SAM3D是生成实体假设，未提升源signed/contact权威。
- 原118D布局保留；两手各根旋转3、关节45、平移3和物体6参数，共108个几何自由度参与直接求解。10个contact logits保留源prior，它们不能关闭碰撞。没有冻结两手，也没有使用旧checkpoint。
- 全部手面与全部native面的SAT相交检测：保守广相位候选（cKDTree）只加速，实际判断使用真实三角顶点、面法线/边叉乘/共面轴；包含物体边穿过手面的相交。旧手边检查会漏掉的合成反例已测试。相交容差1e-8m，另查全部MANO顶点和面中心inside depth。
- 真正MANO批量中心差分Jacobian，与既有decode_candidate回放最大误差约1.41e-7m，方向导数检查约5.95e-8m。对真实交线使用手面重心坐标，不任意推顶点。
- 顺序线性化：LP先最小化最大slack，QP在该线性可行集合中择小完整表面修正和轻量前帧连续性。每步以完整几何重检，不能用小reward冒充可行；若存在slack，则仍明确为未解决。
- 严格参数球以保守逐轴界实现：手平移≤8mm、根旋转≤5°、每关节≤8°；物体平移≤4mm、旋转≤5°；每步逐轴trust≤1.5mm/1.5°，每帧最多12外迭代。完整手表面移动并不只等于平移参数，累计旋转可使指端位移>8mm。

## 2. 全序列结果与副作用

导出后独立进程使用11-ray inside查询及全部三角SAT重检；窄相位仍是同一SAT实现，不冒称独立第三方几何oracle。无碰撞定义：手面/物面无相交且最大采样inside≤0.01mm。 检查范围为手—生成奶盒几何；未认证MANO腕口闭合体、手自交、手手碰撞或真实力学。

|指标|Zero R6|直接求解|
|---|---:|---:|
|双手同时无碰撞帧|0/150|0/150|
|左手无碰撞帧|5/150|11/150|
|右手无碰撞帧|0/150|8/150|
|全部相交三角对总数|307665|217705|
|全片最深采样inside（mm）|1.879061|1.370540|
|采样inside²总和（m²，无0.1mm hinge）|0.005302822|0.002934308|

三角对数不是旧实验的交叉手边数，inside²定义也与旧0.1mm hinge不同，不能直接跨定义拼接排名。若一只手某帧无碰撞但另一只仍碰撞，整帧仍未解决；两手无碰撞帧没有重合。

### 不能称为最小修正或合理抓握

|指标|左手|右手|
|---|---:|---:|
|平均MANO表面修正mm|5.160|1.849|
|最大MANO表面修正mm|26.263|20.729|
|冻结patch平均间距增加mm|8.714|3.566|
|冻结patch最大间距增加mm|32.968|29.847|
|实际加速度p95，Zero→求解（mm/s²）|9376.56→16770.07|12767.50→14514.87|

冻结patch不是真接触，源patch有误差；但增量揭示了手物关系变差，不能用“无视频对齐”掩盖该副作用。实际物体完整表面加速度p95亦从17041.02增至17610.97mm/s²。物体最大平移/旋转达到4mm/5°界限。

视觉审查已直接打开完整150帧RGB/world的6张序列图、最大手表面修正的frame42/142全尺寸图、frame143和实际native Viewer frame0。可看到指部和手物关系改变，红色相交区域部分减少但仍在；没有全片脱离很远的简单逃逸，但接触间距和时序副作用足以拒绝推广。RGB错位是允许的目标放松，**真正失败是穿透没有清除，同时几何抓握关系恶化**。

## 3. 失败机制，不能误判成物理不可行

### 已修的实现错误（日志保留）

- r1把小native面局部平面扩展到整张手面的远端顶点，虚假slack7–11mm；修为真实交线重心点及其Jacobian。
- r2最近出口法线在跨薄物体两侧的手面上相互反对；r3改为整手实际相交面法线主轴±模式，用完整native rays决定进入/退出，尝试一致侧。
- 审查r1继承的RRD工具要求3个候选，拒绝2列Zero/直接结果。仅修复新分支的导出器，并在fresh final_review_r2重新审查；数值快照和预测没有重跑/改写。

### 本轮仍存在的关键缺陷

整只手采用单个退出模式，对于不同手指/掌面碰撞并不合适。某条ray的远端出口不等于手面最小避碰方向；把它变为切平面可能过度保守。事后固定frame0/120检查：

|帧|最大采样深度mm|所选模式最大约束投影mm|单trust LP最小max slack mm|全参数界下局部LP slack mm|
|---:|---:|---:|---:|---:|
|0|1.5296|64.954|47.361|22.020|
|120|1.2995|34.597|21.805|7.102|

**这些数值不是证明手真的必须移动35–65mm。** 它们只说明当前选定的整手退出分支及零姿态Jacobian产生了难以满足的线性集合；换分支、局部连通patch或非线性姿态可能存在小修正。不能据此自动放宽范围把手推远，也不能据此宣称输入全局不可行。

全片359个QP尝试仅18次报告收敛；其余虽可满足带slack线性集合，但没有证明最小表面修正。minimax slack中位4.536mm、p95 53.584mm。150次非线性候选步骤被拒绝，每帧最终都停滞；有些步骤最大深度下降但三角交叉数反而上升。允许这一merit下降有助于局部探索，却不构成单调消除相交的保证。

本次没有PPO，所以本次失败不能归因于PPO采样或更新：该局部约束构造/求解本身仍失败。这不重建旧实验的因果过程，不否定直接几何方法，也不证明小修正解不存在。

**下一次必须针对这个耦合错误，而不是再次堆reward/扩PPO：** 按实际相交连通patch拆分退出模式，分别处理指部/掌面，避免整手共用一条跨远端物面的ray；在少数真实帧用完整三角回检验证模式可行，再恢复全序列。应改进可行QP的数值条件及可行解时序择优，不能继续把LP带slack顶点解当成最小修正。这个下一次干预本轮未执行；本轮固定一次全序列预算未扩展。

## 4. 隔离、预算和复现

- 分支`feature/milk-direct-nonpenetration-20261014`；工作树`/mnt/user-home/kupingxin/ego_annotation_nonpenetration_milk`，无commit/merge/push。原三reward树60份继承文件hash不变。
- 冻结执行代码`/mnt/user-home/kupingxin/ego_annotation_runtime/direct_nonpenetration_milk_20261014_r1`，240份Python/YAML均不变。
- 结果根`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_direct_nonpenetration_r6_20261014_r1`。CPU单线程/禁CUDA，host=`dexrobot-Standard-PC-Q35-ICH9-2009`；无安装新包、感知或PPO训练。
- 执行17:59:51–18:20:27（+08:00），20分36秒；其中全序列求解943.34秒。唯一一次150帧，最多12次/帧；没有运行中改参数、放宽修正界或自动重跑。
- 回归211项：207通过、4skip；审查工具12项通过，2列RRD修复后5项通过。MANO replay/Jacobian检查及hard参数界全通过，但测试不是目标完成证据。
- 两视频各150帧/30fps/5s，累计解码300帧；RRD全读回核对300物体poses、600个MANO mesh帧及全native面，官方verify成功。无签名几何/真实接触或零连续碰撞认证。

## 5. 实际交付路径

- 完整视频与图像入口：`/mnt/user-home/kupingxin/ego_annotation_experiments/direct_nonpenetration_milk_20261014_r1/final_review_r2/index.html`
- RGB：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_direct_nonpenetration_r6_20261014_r1/comparison/renders/zero_best_last_overlay.mp4`
- world：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_direct_nonpenetration_r6_20261014_r1/comparison/renders/zero_best_last_world.mp4`
- 完整native穿透高亮：`/mnt/user-home/kupingxin/ego_annotation_experiments/direct_nonpenetration_milk_20261014_r1/final_review_r2/zero_direct_complete_collision.rrd`
- 全帧三角表：`/mnt/user-home/kupingxin/ego_annotation_experiments/direct_nonpenetration_milk_20261014_r1/final_review_r2/all_frames.csv`；物理/修正/时序/gap：`analysis.json`；直接视觉判断：`visual_judgment.json`
- 全部求解过程：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_direct_nonpenetration_r6_20261014_r1/solve_trace.json`；导出后完整相交证据：`zero/complete_triangle_audit.json`和`solved/complete_triangle_audit.json`
- 事后局部可行性（非全局证明）：`/mnt/user-home/kupingxin/ego_annotation_experiments/direct_nonpenetration_milk_20261014_r1/posthoc_local_feasibility.json`；启动命令`launch.sh`，原失败控制日志`real_preflight_r1/r2/r3.log`。

视频实际两列：左Zero R6，右直接求解；文件名继承zero_best_last并不表示PPO或三个不同候选。完整预测和失败证据已经交付，**消除穿模的目标仍未完成**。保留R6及旧实验，不推广此输出。
