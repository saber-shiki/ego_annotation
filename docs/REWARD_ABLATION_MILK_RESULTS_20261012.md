# Milk reward 消融：最终分析与视觉审查

日期：2026-10-12。**这是单视频、256-step、两个 seed 的影响筛查，不是收敛实验、统计显著性检验或泛化验证。**

## 最终结论

1. 当前训练中，**手部像素 reward 的数值尺度主导随机 rollout**；可见表面深度顺序和物体平滑项也会显著改变学习轨迹。
2. Reward 的语义职责和当前 PPO 的实际影响并不等同：去掉物体 reward 也会改变手部动作。统一回报、GAE、critic、归一化和联合网络更新带来跨动作块影响；本次没有隔离每一种中介机制。
3. **生成碰撞项已接通，但短训练中的实际策略影响极弱**。接触项在当前尺度下没有可检出的参数/几何影响；关节深度和动作软边界在这些轨迹上没有触发。
4. **没有任何配置跨两个 seed 稳定超过 Zero R6，也没有消除穿模。** 所有26个候选仍在150/150帧检测到至少一只手相对生成网格的穿入/交叉。
5. 不建议直接删除 hand_pixel、hand_surface_depth 或 object_temporal。删除后完整J下降暴露的是当前学习动态/尺度问题，不证明对应物理约束无用。

保持完整118D、双手可训练、SAM3D真实三角网格与原始观测不变。没有重新训练、替换基线、merge或push。

## 实验与独立复核范围

- 13种配置：Full + 12种单功能组去除，每种seed0/1；26个fresh policy。
- 每种256steps，两个128-step rollout，每次64个Adam更新；共128个optimizer更新/策略。
- 同一seed各变体初始actor/critic hash相同；**前128个执行动作逐元素完全一致**。
- 原PPO参数保持：初始log_std=-1、epochs/minibatches8/8、lr3e-4、clip.2、gamma1、GAE.95、reward_scale1e-5、无KL停止。
- 评价一律恢复所有factor，使用同一个 **J_full = J_original_formal + J_generated_collision**。训练选中J、删除成本、完整J独立分账；逐项与实际ledger/return复核。
- 完整150帧deterministic rollout；原生239430顶点/478880三角面物体；左右MANO拓扑/有效性/相机K/world timeline一致。
- 独立检查26个checkpoint及其provenance、78个rollout artifacts、219份冻结源码/config hash；配置差异只允许drop_group，均通过。
- 重新解码26个完整视频，共3900个视频帧；实际审查13组全时序RGB/world总览，并重点检查同seed Full-versus-drop 的0/30/60/86/120/149帧大图。
- 两个重点比较导出完整碰撞标记RRD；每个含3候选、450物体姿态及900手mesh帧，实际读回比对，并在Rerun原生Viewer检查全场景和隐藏物体后的红色穿入区域。
- 针对hand_pixel/seed0和hand_surface_depth/seed1重新做完整150帧三角几何碰撞查询，结果与冻结audit一致。

原训练运行退出0、耗时47分55秒。独立审查没有把视频解码或hash成功等同于标注改善。

## 统一完整目标：全部结果

**Zero R6 J_full = 2101.587120**，其中原始formal J2067.838616、新碰撞J33.748504。

| 去掉的组（Full不去掉） | seed0 J_full | seed1 J_full | 相对同seed Full 的ΔJ：s0 / s1 |
|---|---:|---:|---:|
| Full | 36551.85 | 2364.97 | 0 / 0 |
| 生成碰撞 generated_collision | 36551.23 | 2361.99 | −0.61 / −2.97 |
| 手部像素 hand_pixel | 2350.66 | 2167.43 | −34201.19 / −197.54 |
| 可见表面深度顺序 hand_surface_depth | 2781.77 | 2101.23 | −33770.08 / −263.74 |
| 手部姿态先验 hand_priors | 36551.05 | 2362.72 | −0.80 / −2.25 |
| 手部平滑 hand_temporal | 36577.65 | 2332.03 | +25.80 / −32.94 |
| 物体观测几何 object_geometry | 36522.45 | 2349.80 | −29.40 / −15.16 |
| RGB旋转 object_rgb_rotation | 36544.53 | 2464.41 | −7.32 / +99.44 |
| RGB重投影 object_rgb_reprojection | 15924.71 | 2476.33 | −20627.14 / +111.37 |
| 物体平滑 object_temporal | 8305.92 | 2181.16 | −28245.93 / −183.80 |
| 接触 contact | 36551.85 | 2364.97 | 完全相同 |
| 关节深度 hand_joint_depth | 36551.85 | 2364.97 | 完全相同 |
| 手部软边界 hand_bounds | 36551.85 | 2364.97 | 完全相同 |

Full在两seed之间的差为 **34186.88 J**；许多弱项的消融差异远小于seed效应。不能只用平均值或挑seed1来宣称改善。

唯一数值低于Zero的候选是去掉hand_surface_depth/seed1：2101.230214，下降0.356906（约**0.017%**）。另一seed未重复。其原生网格穿模仍覆盖150帧；不推广为成功策略。

## 每组reward主要影响什么？

### 1. 手部像素保持（hand_pixel）——影响最强，但呈现不稳定学习而非有效锚定

定义：可见关节相对源MANO投影的位移超过12px后，施加平方hinge。不是GT监督。

前128帧相同随机动作下，此项累计最大成本为：

- seed0约**7802614.46**；对应full前缀约7805388.78。
- seed1约**8533002.96**；对应full前缀约8535685.18。

移除此项可令单步实际reward差达到4.78/5.32，而移除碰撞项的单步差只有约4.04e-6/3.43e-6。初始探索log_std−1在118维中产生了很大投影扰动，reward的信噪与尺度不平衡直接可见。

最终移除此项反而使其未被删除的完整评价成本下降：

- s0：像素成本34312.33→11.85；关节shift p95 L/R9.07/13.02→7.29/9.13px。
- s1：像素成本229.76→0；p95 L/R5.99/8.69→6.86/5.99px。

左手p95变化并非两seed都改善；超过deadband的尾部才决定像素成本。不能将其概括为“删除像素约束必然更准”。

**其他影响**：s0同seed手mesh最大差6.44/9.87mm、物体最大平移差5.47mm；s0物体IoU中位数0.848723→0.813993，s1也0.828018→0.825373。s0残差加速度增大，s1部分下降；碰撞残差也是s0改善、s1恶化。

**视觉判断**：姿态与手—物间距有小但可见的变化，没有mesh丢失；s0物体轮廓相对原RGB/mask并不更好，world/红色穿入图不能支持消除碰撞。结论是应修正训练尺度/更新而保留合理2D锚定，不能直接删它。

### 2. 独立可见表面深度顺序（hand_surface_depth）——影响手深度和统一回报

定义：对冻结P09 first-surface支持点，约束手部顶点不得进入错误的相机深度前方；10mm margin，单侧相机-Z不等式。它不等于当前物体mesh的全局内外检测。

去掉后完整J在两seed都下降，但下降主要包含像素成本变化，未表明完全按预期纠正手深度：

- s0 surface成本2073.53→2015.47，pixel成本34312.33→600.05。
- s1 surface成本1970.51→1939.93，pixel成本229.76→0.44。
- s1右手p95投影8.69→6.02px，左手5.99→6.23px；关节深度位移p95 L/R1.61/1.91→2.32/2.13mm。

s0手部残差加速度L增加、R下降；s1两手下降。s0碰撞平方sum相对Full略降，s1反而升0.60%左右。s1相对Zero的sum下降1.84%、最大深度1.879→1.864mm，但交叉/穿入帧没有减少。

**视觉判断**：s1非常接近Zero，轮廓无清楚的大幅改进；隐藏物体后的真实MANO红色区域仍存在。仅0.017%的J改善不能覆盖不可重复性。此项需核对first-surface ownership/坐标与缩放，而不是靠删除替代几何解决。

### 3. 物体平滑（object_temporal）——改变物体，也显著改变双手

定义：物体位姿修正的velocity/acceleration约束（不是原视频物体真实运动速度的ground truth）。

移除后两seed完整J下降，但主要收益同样来自手部pixel成本（s0 34312→6076；s1 230→67），不是“物体更不平滑”这一简单预期。相对Full，物体修正加速度p95反而从2238.6→2234.2和1802.2→1759.9mm/s²；旋转残差rate也下降。右手残差加速度两seed下降，左手一升一降。

s0同seed最大手mesh差4.64/5.07mm、物体平移差3.54mm；s0IoU下降0.848723→0.824768，s1IoU上升0.828018→0.831416。碰撞sum与交叉数也是不同seed相反趋势。

**视觉判断**：world构型有小幅变化，没有漂飞，但轮廓和穿模不提供稳定收益。该组对统一学习路径有强影响，当前PPO未学出可解释的“该项→该部件”的单独控制规律。

### 4. RGB重投影（object_rgb_reprojection）——保留自己的图像一致性，但产生跨手部影响

定义：独立source-world RGB点，在候选物体SE3的两端重新投影的误差；不是训练视频silhouette IoU。

移除后，其完整评价成本在两seed均略升：s0 53.0331→53.0442；s1 51.6820→51.7520，符合约束被拿掉的局部预期。但是完整J的变化方向不一致：s0−20627、s1+111，主要通过pixel尾部放大。s0最大MANO差1.55/0.96mm，s1约0.40/0.43mm。

两个seed的碰撞sum和crossing edges相对Full略降，但未消除任何全视频穿模。物体IoU变化s0下降、s1上升。

**视觉判断**：s0手部局部叠加有变化，s1差异较细微；沒有可泛化的视觉/接触改善证据。保留RGB evidence，不因一个seed的J下降就删掉。

### 5. RGB旋转（object_rgb_rotation）——短训练中直接姿态影响偏弱

移除后RGB旋转自己的完整成本变化很小（约19.68/19.39），s0完整J略降，s1升约99.44。s0几何差亚微米量级，s1最大MANO差约0.41/0.44mm。s1投影p95双手都略恶化；碰撞/物体IoU有微小变化，方向不足以形成稳定结论。

**视觉判断**：保持完整姿态轨迹，肉眼差异微小，没有发现与明确旋转修正对应的稳定收益。当前预算不能证明此证据无用。

### 6. 物体观测几何（object_geometry）——深度点的相对一致性，而非碰撞

包含point-to-plane和point-to-point。移除后完整J−29.40/−15.16，自己的成本也只轻微改变（52.927→52.916；54.358→54.313）。s0最大mesh差不足0.001mm；s1最大MANO差小于0.10mm、物体差小于0.06mm。与Full几乎重合，仍150帧穿模。

该项不是手物排斥，不应期待它直接移走手指。此消融只显示它在当前短PPO配置下对物体运动影响弱。

### 7. 手部姿态先验（hand_priors）——目前没有成为实质动作锚点

移除后完整J−0.80/−2.25。当前deterministic候选该组成本仅约0.015–0.018；s0最大几何差<0.0002mm，s1约0.015mm。影像/world不可分辨的近似重合，不应把微小数值J下降解释成更好的手姿态。

### 8. 手部平滑（hand_temporal）——存在信号，但方向不稳定

移除后J变化+25.80/−32.94。s0最大mesh差约0.0035mm；s1约0.215mm。s0 residual acceleration轻微增加，s1反而下降，且pixel/collision方向也不一致。不能说它已经稳定减抖，也不能说移除它可稳健减抖。

**注意**：报告的速度/加速度是相对Zero的修正轨迹，不是原始手运动的全部速度/加速度。只有两个seed和两个PPO update，不足以评估长期平滑学习。

### 9. 生成物体非穿透（generated_collision）——真实接通但影响太弱

移除后完整J仅−0.61/−2.97；相对Full，碰撞平方sum却略升，两seed一致但量级极小：约+6.2e-9 /+1.08e-7m²。s0最大MANO差约0.00019mm；s1约0.018mm。仍150帧检出穿模。

存在自己的reward信号：前128帧去掉此项，所有128个reward都变化，并非未接入。只是这类单步变化比像素项小约百万倍，当前策略变化极小。

此前正样本数归一化不单调的问题仍保留，没有在消融中偷偷修复。碰撞自身J在移除后反而下降，不代表排斥量下降。继续使用固定样本总平方深度、边交叉等独立指标。

### 10. 接触（contact）——对当前学习无可检出影响

含normal deadband以及接触概率prior/continuity；geometry likelihood仍未启用。64mm死区导致normal proximity在所选patch上成本0，概率项总成本约0.00026–0.00030。

移除后两个seed最终**policy、value、observation scaler、value scaler均逐tensor hash完全相同**；后128个执行动作、全帧手mesh/物体变换均完全一致。前128个float32reward有9/13个极小差别（最大约3e-8/9e-9），但没有形成参数变化。不能说本系统已靠该reward学会抓握或接触，也不能推断未来合理尺度下它仍无效。

### 11. 关节深度（hand_joint_depth）——轨迹未越过死区

35mm相机深度变化deadband在随机训练及最终deterministic候选均未触发，移除成本沿训练trace为0。两个seed参数、轨迹、几何均与Full完全相同。这是未被触发的保护项，不是已被验证的无用项。

### 12. 手部软边界（hand_bounds）——同样未触发

默认translation45mm/root.30rad/articulation.45rad软限制在当前256steps轨迹上没有正残差；移除成本0。模型和结果与Full完全相同。防止大动作的语义仍有效，只是这轮没触发。

## 视觉审查总结与局限

- 13组均保留完整物体与两手MANO，未发现丢mesh、空帧、负相机深度导致的爆炸或断裂式姿态跳转。所有有效关节在相机前方。
- millimeter级手/物修正仍在overlay/world可见，对hand_pixel、hand_surface_depth、object_temporal尤其应看同seed比较而不是seed0/seed1比较。
- 部分完整J大幅下降的候选，实际物体轮廓IoU下降、某只手残差平滑性变差，或碰撞指标在另一seed反向；没有统一的视觉改善。
- 所有候选相对原始生成几何的穿模检测均覆盖150帧，约1.9–2.0mm的局部穿入没有消除。Rerun隐藏物体后仍可看到掌指表面的红色交叉/inside区域。
- 密集总览用于全时序连续性/几何完整性检查，不能用低分辨率视频肉眼认证0.1mm无碰撞；因此用实际网格查询、同seed详细图和原生Rerun补充。也没有从视频断言contact force、真实signed几何或真实物体3D准确率。

## 哪些因果结论可以说，哪些不可以？

可以说：本配置中手部像素项强烈改变reward/后续策略；接触项/关节深度/边界出现精确负对照；生成碰撞项的短期几何影响很弱；某些物体项通过统一回报改变双手；小量数值赢家不可重复。

不可以说：去掉像素/深度/平滑约束普遍更好；两个seed证明统计显著性；某reward无用；PPO发散的唯一原因已经找到；被移除项的低训练J是真实质量提升。

一个可证实的下步是：保持两手解冻和全部真实几何，先对像素/深度/collision的物理单位和reduction做一致化、保留独立原J评价；修正collision正样本均值不单调，再逐项验证带尺度的策略更新/normalizer一致性。需要独立实验才能区分exploration、critic/GAE和normalizer作用，不能一次改所有项再宣称归因。本次没有实施新参数或训练。

## 交付位置

训练/视频根：

`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_reward_ablation_r6_256x2_20261012_r1`

每组完整视频：`comparisons/<组名>/renders/zero_best_last_overlay.mp4` 和 `zero_best_last_world.mp4`。

**这些历史generic文件名的面板实际是 Zero / 本组seed0 / 本组seed1，不是best/last。** 以画面标签为准。seed两个候选均256steps。

独立最终审查根：

`/mnt/user-home/kupingxin/ego_annotation_experiments/reward_ablation_milk_20261012_r1/final_review_r2`

- `analysis.json`：完整指标、paired差值、训练trace影响与模型同一性。
- `all_runs.csv`、`paired_deltas.csv`：26候选全部数值与配对差值，不筛掉失败结果。
- `verification.json`：checkpoint/export/source identity、分账/return复核。
- `video_verification.json`：26视频完整解码、hash与panel身份。
- `visual/<组名>/`：全时序总览、0–74/75–149详细sheet、同seed六帧配对。
- `collision_requery_verification.json`：两重点候选全150帧重新碰撞查询一致。
- `rrd/zero_full_drop_hand_pixel_seed0.rrd`：Zero / Fullseed0 / drophand_pixelseed0，红色穿模标记。
- `rrd/zero_full_drop_hand_surface_depth_seed1.rrd`：Zero / Fullseed1 / drophand_surface_depthseed1。
- `*_viewer.png`、`*_hands.png`：Rerun实际渲染审查，不是图像生成。
- `review_regression.log`：12个审查/消融/RRD回归通过；原204测试200通过4跳过。

两个RRD通过官方 `rerun rrd verify`，自含完整几何，不需要外部NPZ/MANO资产即可打开。初始观察视角相同，允许交互旋转与隐藏物体；默认都是真实mesh，而非视频billboard。

最初只读审查脚本使用错误provenance字段名导致KeyError，在`final_review.log`保留；修正到`training_config_sha256`后在新`final_review_r2`完成。训练artifact/冻结运行源码没有被修复或覆盖。
