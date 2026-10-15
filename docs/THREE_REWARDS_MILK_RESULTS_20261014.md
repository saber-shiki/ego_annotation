# Milk R6 三组 reward 有限调试结果（2026-10-14）

## 结论：代码／有限实验完成，但标注没有可确认的整体改善

新系统确实使用了三组几何成本，双手没有冻结，完整118D动作保持可训练；参考偏移远小于旧实验。但两个seed在600步的三组总成本均比Zero差，穿透仍覆盖150/150帧，实际轨迹抖动没有明显降低。唯一低于Zero的是seed1@150（约0.884%），另一seed未重复，随后回退；不推广任何策略，Zero R6保留。

**没有独立GT。** reference_alignment保持的是有误差的R6几何、投影和相对交互，不是证明接近真实姿态。SAM3D闭实体只作为生成碰撞假设，没有提升源signed/contact权威。此实验是同一视频、短预算的联合重新设计，不是单因素消融、收敛或泛化证据。

## 1. 隔离实现与运行

- 分支：`feature/milk-three-rewards-20261014`；工作树：`/mnt/user-home/kupingxin/ego_annotation_rl_three_rewards`。基于`6c4fbab`；未commit、merge或push。原工作树46份继承文件hash不变。
- 冻结执行代码：`/mnt/user-home/kupingxin/ego_annotation_runtime/rl_three_rewards_milk_20261014_r1`。231份Python/YAML文件hash复核不变。
- 结果根：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_three_rewards_r6_600x2_20261014_r1`。固定config SHA256：`76803d63a500a8f98dcfaff7103e5c0b0aee33a18d6cadf10ae8948c327c8952`。
- 服务器`dexrobot-Standard-PC-Q35-ICH9-2009`，既有ego_skrl Python；CPU单线程／禁CUDA，无安装新包。seed0/1各600环境步，分别在150/300/450/600完整评价150帧；总1200步，没有暖启动、续训或结果后调参。
- 执行阶段15:13:24–15:27:04（+08:00），共13分40秒（前置检查、并行训练／评价、渲染）；两seed含评价分别580.71／583.21秒。每seed只有4次实际Adam更新，不是40次：KL early-stop跳过了多数minibatch。
- 几何：原metric MANO，native物体239430顶点／478880三角面，固定尺度、相机和时间轴，没有primitive、简化mesh、CAD、GT或感知重跑。观测551D。

## 2. 三组公式及其真正含义

总成本为`J3 = 1*reference + 5*nonpenetration_contact + 0.25*temporal`，小为好。帧项分母固定为150，边／三帧项固定为149／148；prefix不改变旧项分母。

1. **reference_alignment**：完整MANO、完整native物体表面对R6位移平方（5mm尺度）；可见joint对R6投影平方（5px尺度，无12px死区）；MANO在当前物体坐标相对参考的平方位移；轻量contact-logit参考保持。完整物体位移由所有native顶点的精确二阶矩二次型计算，等于逐顶点求和，不是质心代理。
2. **nonpenetration_contact**：所有MANO顶点及所有三角形中心对当前完整native物体的inside depth，`mean_all((max(d-0.1mm,0)/0.5mm)^2) + 0.1*max(max(d-0.1mm,0)/0.5mm)^2`。分母不是穿入点数，contact logits不能减小此项。冻结接触patch的真实MANO顶点到native-anchor距离若比R6多2mm，按5mm尺度惩罚；使用冻结prior，不强迫有误差的源间距归零。实际手边／物体三角交叉另作独立评价，未直接写入训练成本。
3. **temporal_continuity**：完整手／物表面**修正量**的速度（30mm/s尺度）和加速度（1m/s²尺度）平方。使用time_s和有效连续帧，不跨缺失帧。这只抑制新增修正跳变，**不等于消除R6原有抖动**。

回报为`r_t=-[(Jprefix-Jzero_prefix)-(Jprev-Jzero_prev)]`，gamma=lambda=1，reward_scale=1；完整return等于`-(Jfinal-Jzero_final)`。Zero三组总成本`2.8651853068`，几乎全为已有穿透成本。

固定PPO坐标：asinh观测变换无running统计、value原单位、critic末层初始0、actor mean初始0；原始Gaussian采样与其log_prob严格配对。log_std=-2.5，lr=1e-4，ratio_clip=.1，KL阈值=.01，150 rollout／2epochs／5minibatches；双手与物体原动作尺度未减维或冻结。旧默认契约及旧快照不变。

## 3. 所有checkpoint，不只挑选最好

|seed|环境步|J3|相对Zero ΔJ3|参考项|物理项|时序项|原formal J（评价用）|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0|150|2.900480|+0.035295|0.008199|2.892136|0.000145|2038.434067|
|0|300|2.985436|+0.120251|0.022964|2.962118|0.000354|2030.308674|
|0|450|3.011006|+0.145821|0.044806|2.965584|0.000617|2028.249754|
|0|600|3.036238|+0.171053|0.066270|2.969065|0.000904|2024.818470|
|1|150|2.839870|-0.025315|0.004803|2.834899|0.000169|2040.704719|
|1|300|2.867444|+0.002259|0.009759|2.857243|0.000442|2032.988551|
|1|450|2.891771|+0.026585|0.014276|2.876724|0.000771|2049.535168|
|1|600|2.912989|+0.047804|0.019829|2.892041|0.001120|2065.056857|

表中的三项已经乘组权重。两个seed的best-trained都在150步：seed0仍比Zero差，seed1随后三个checkpoint回退。原formal Zero为2067.838616；部分formal分数下降和SAM2轮廓IoU微升不能覆盖穿透、时序及新总目标的负面结果。formal和IoU都不是GT，也不是新训练目标。

## 4. 独立物理与运动证据

|候选|最深采样穿透mm|深度hinge²总和m²|实际交叉边总数|受影响帧|
|---|---:|---:|---:|---:|
|Zero|1.879061|0.004097724|26237|150|
|seed0@150|1.985053|0.004102386|26087|150|
|seed1@150|1.917590|0.004069656|26280|150|
|seed0@600|1.974470|0.004089610|26234|150|
|seed1@600|2.001495|0.004055097|26456|150|

最终seed0：残差总量−0.198%、交叉边−0.011%，最深点+5.078%；seed1：残差总量−1.040%、交叉边+0.835%，最深点+6.516%。小幅总量下降并不是消除穿模；带最深点项的物理组成本最终均高于Zero。

**参考保持：** 最终seed0左／右MANO表面平均修正1.172／0.383mm，投影偏移p95为2.699／1.566px；seed1为0.476／0.402mm和1.209／1.539px。物体最大修正分别0.572mm／0.147°和0.749mm／0.196°。视频中多数变化很小，两手实际都在变化，不是冻结双手制造稳定。

**接触间距：** 源patch只是限定接触参考，不是真接触。Zero左／右patch平均距离5.951／7.551mm，最大34.681／26.680mm；新成本只限制相对源间距增加，不会修复这些原始绝对gap。最终seed0左手gap增量p95=1.896mm，最大2.206mm，389/23680个vertex-frame超过2mm；右手最大增加0.985mm。seed1左／右最大增加1.324／0.809mm，无超过2mm样本。seed0的接触分离未完全抑制，源gap也不能被宣称正确。

**实际运动没有明显改善：**

|候选|左手实际加速度p95 mm/s²|右手实际加速度p95 mm/s²|完整物体表面实际加速度p95 mm/s²|
|---|---:|---:|---:|
|Zero|9376.56|12767.50|17041.02|
|seed0@600|9414.00|12796.92|17053.93|
|seed1@600|9355.85|12762.32|17082.05|

手指标为所有顶点／有效时间样本的p95；物体指标为每个三帧窗口上全部native顶点RMS加速度的p95，定义不同，不跨类型直接比较。物体完整表面运动已独立与显式逐顶点计算核对，不用质心速度替代。最终手修正速度p95约1.28–1.58mm/s；物体修正表面速度p95约1.10／1.76mm/s，虽然修正很小，实际运动中的原有大加速度仍在。

**实际视觉审查：** 四个完整视频均150帧／30fps／5s，解码600帧；已逐段打开best／last全部12张全帧RGB/world序列图，额外查看120／143／149全尺寸图。整体紧贴Zero，未见旧实验的大幅偏移，但没有可确认的接触／穿透改善。实际native Viewer在frame0看到两手指部红色穿透区仍存在；Viewer截图不是PNG伪造的几何替代。RRD完整readback核对450物体poses、900个MANO mesh帧、native全顶点／三角形及纳秒时间轴；官方verify成功。这些检查保证可视化真实连接，不作为改善证明。

## 5. 已确定的失败机制与仍不能推出的结论

### 坐标一致性确实修复，但不足以修复PPO更新

八次更新前raw log_prob最大误差≤3.052e−5、stored/predicted value最大误差≤1.49e−8、value clip fraction=0；fixed preprocessor全程不变。因此这次不能再将更新KL解释为running scaler漂移或原随机初始critic。

然而每个150步更新只执行1次Adam：seed0更新后真实联合Gaussian KL依次1.894／1.009／0.578／0.364，seed1为1.959／1.079／0.856／0.727，远超0.01。post-update clip fraction为92%–98%。已核对skrl2.1.0源码：KL在每个minibatch执行前检测，超过后break，不回滚上一已执行Adam步；所以阈值不能保证第一步不越界。

首个checkpoint固定评价观察上的初始zero-mean→新actor KL分解为1.895／1.962；左右articulation两块合计约1.434／1.508。这是118D联合概率的各维贡献求和，不是降低维数的理由，也不是历史训练KL的替代。raw mean RMS只有0.0147／0.0150，但相对std约0.082仍可累积成大的联合KL。

### 奖励组数少，不代表学习信号平衡

首轮critic为0、gamma=lambda=1，对原始完整150步reward-to-go分组：

|seed|参考return-to-go std|非穿透/接触 std|修正时间 std|
|---:|---:|---:|---:|
|0|0.066930|0.018152|1.217328|
|1|0.067590|0.009514|1.199490|

相邻独立探索动作带来很大的修正加速度成本，时间项在首轮reward-to-go中比物理项大约67–126倍（标准差比，不是因果百分比）。时间项首轮总惩罚约4.08／4.13，参考约0.23，中心化物理项约0.0687／0.0010。分组拆解把<1e−8的Zero参考/时序roundoff留在物理残差中；总reward严格相加一致，不影响量级判断。此结论局限首轮，不宣称重建后续所有GAE／梯度。

因此，这次没有可靠测试出“PPO主要沿穿透改善方向学习”的状态。受控几何方向存在，并不等于118D联合探索／更新能选到该方向。三项公式有效、参数／坐标一致，但它们的实际学习信号仍不平衡，单步策略更新也未被约束住。

### 受控检查与时间顺序的诚实边界

训练前完成frame0的30个完整几何动作检查：两手／物体参考扰动有效，两手都存在降低物理组方向，contact logits不能规避穿透。设计还写了frame120；它没有在训练前执行，现已作为**事后无optimizer固定参考补证**保存，不能倒称preflight。

frame120的16个事后扰动：左X+2mm物理项−0.000636、右Y+2mm−0.000861；但单帧跳变时序项约+0.01542，足以抵消局部改善。共同手物X+2mm保持相对交互至浮点误差、穿透基本不变；减contact logits20保持几何时物理项变化0。它证明连接和时序代价，不证明策略已经学会相干多帧修正。

**下一步的因果指向（本轮未执行）：** 不应先加训练步数或再次堆叠reward。首先用冻结首轮数据做无新增rollout的一次更新回放，测各动作块KL并验证可接受步长／拒绝越界更新机制；同时让修正探索在时间上相干，避免时序项淹没物理信号。若要再训练，应另声明预算并保留本轮失败，不自动续训。不能据此宣称三组方法无效，也不能宣称已稳定收敛。

## 6. 测试、视频和复现路径

- 回归205项：201通过／4skip；另外完整native-motion独立数学回归2项通过。最初错误contact-only测试的失败日志保留，修正后机制suite通过；它不曾启动真实Milk训练。`git diff --check`／AST通过。
- 8个checkpoint／24个评价导出、4个视频、R6／source／native输入hash已复核；独立评价没有修改训练RNG或环境状态。hash和测试不证明改善。

视频中实际三列：左Zero，中seed0，右seed1；文件名沿用`zero_best_last_*`，并不是中列best、右列last。`comparisons/best`两seed都取150步；`comparisons/last`取600步。

- 最终完整RGB：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_three_rewards_r6_600x2_20261014_r1/comparisons/last/renders/zero_best_last_overlay.mp4`
- 最终完整world：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_three_rewards_r6_600x2_20261014_r1/comparisons/last/renders/zero_best_last_world.mp4`
- 最好训练候选完整RGB/world：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_three_rewards_r6_600x2_20261014_r1/comparisons/best/renders/`
- 交互native三列＋红色穿透：`/mnt/user-home/kupingxin/ego_annotation_experiments/three_rewards_milk_20261014_r1/final_review/zero_seed0_seed1_last_collision.rrd`
- 入口：`/mnt/user-home/kupingxin/ego_annotation_experiments/three_rewards_milk_20261014_r1/final_review/index.html`；指标图`summary.png`；完整数值`all_checkpoints.csv`／`analysis.json`；全表面运动和间距`motion_contact_audit.json`；实际视觉判断`visual_judgment.json`。
- 前置动作`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_three_rewards_r6_600x2_20261014_r1/controlled_actions.json`；事后frame120`/mnt/user-home/kupingxin/ego_annotation_experiments/three_rewards_milk_20261014_r1/final_review/frame120_posthoc_probes.json`；PPO`seed_*/updates.json`；原始采样动作`seed_*/training_trace.json`。
- 原始设计`/mnt/user-home/kupingxin/ego_annotation_rl_three_rewards/docs/THREE_REWARDS_MILK_DESIGN_20261014.md`；训练命令及CPU环境`/mnt/user-home/kupingxin/ego_annotation_experiments/three_rewards_milk_20261014_r1/launch.sh`；冻结snapshot为本轮复现代码，新开发审查脚本副本在`review_source_copy/`。

**最终状态：有限实验与完整失败证据已交付；穿透消除、真实接触准确性和实际时序平滑的目标仍未解决。原R6与所有旧实验保持，不替换基线。**
