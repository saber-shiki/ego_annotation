# 原始 RL 与 v2：物体大小/位姿回归排查

日期：2026-09-22。仅只读比较既有预测；未重新推理、未训练、未修改源run、未修复或改写RGB seed。

## 结论与对先前解释的纠正

用户指出的画面变化真实存在。此前只展示完整native mesh＋RGB seed、没有同时展示原RL口径的源初始化，导致比较不充分。**“读取/受控更新通过”不能替代与旧RL的几何质量回归比较。**

1. 原始RL确实已有不用RGB seed的三维零动作视频。之前只找到v2视频和SAM2 mask就称没有这种三维视频，遗漏了历史RL产物。
2. 新旧RL使用同一原视频、完全相同的150帧K和camera transforms。历史V7内参不同的解释不适用于这次RL对RL比较。
3. v2仍保留原P09初始化和P13 mesh；loader没有将它们覆写为seed。变化发生在新增验证消费者选择了另一套mesh和R4轨迹。
4. 同run控制变量证明：**网格选择造成首帧即存在的大小差；RGB seed又增加明显的位姿/投影退化。** 不能将其视为质量验收通过。
5. 当前检查未发现米/厘米倍数、相机转置或native尺度重复应用错误；不等于宣布全部实现无bug。RGB seed的测量偏差与未融合源约束仍未修复。

## 对照身份

- 旧完整观测：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/hot3d_milk_823a906_minfix_v1_20260918`
- 旧零动作：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_rl_smoke_minfix_v1/zero/candidate.npz`
- 旧视频：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_rl_smoke_minfix_v1/renders/zero/overlay.mp4`
- 新观测：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2`

严格未修复的823a906首轮运行曾中断；这里用的是此前完成的minfix v1源观测，不称其为完全无补丁重跑，也没有用训练后的候选冒充源观测。已逐数组验证旧零动作物体mesh、R/t、valid、camera/K与旧源状态精确一致。

## 四组完整视频（150帧）

| 位置 | 内容 | 分离的变量 |
|---|---|---|
| A 左上 | 历史RL零动作：P13 pose mesh＋逐帧P09初始化 | 历史基准 |
| B 右上 | 新r2 P13 pose mesh＋新r2逐帧P09初始化 | 与A比较源前端变化 |
| C 左下 | 新r2完整native mesh＋与B完全相同的P09初始化 | 与B比较mesh选择 |
| D 右下 | 已交付的新r2完整native mesh＋RGB seed | 与C比较轨迹选择 |

A/D复用各自冻结完整视频，B/C以同一未修改`render.mesh_overlay`重新渲染；所有几何使用实际三角面。B/C只读源MANO，不套用后续typed visibility隐藏，这可能与A的旧display不同；物体定量投影诊断单独渲染物体，排除手部遮挡设置干扰。

P09初始化是`R=I, t=当帧可见表面质心`，**不是每帧独立测得完整物体6DoF**。但这套已有初始化可以驱动固定重建网格，前次回答将其与“没有逐帧完整重建”混为一谈是不准确的。

## 数值与视觉证据

### 源前端没有产生厘米级回归

- raw视频hash相同：`43513ed7534f8d32ede3ec13ad8c5b92d41487705064081491ab7831ee0e1b59`。
- K及全部150帧camera transforms逐值相同；fx/fy=974.3447265625px。
- 148个共同有效帧：旧新源平移距离中位0.552mm、p90 1.193mm、最大2.700mm。
- 旧源没有pose的帧为141/143，新源只缺143；没有用seed填补B/C的143帧。
- 旧零动作与新源P13显示在0/60/94/149等检查帧基本一致。

### 大小差来自几何角色切换，而不是同一网格被错误缩放

下表为锚点world-aligned canonical的轴对齐包围盒跨度，**不是物体真实宽高深测量**：

| 几何 | X/Y/Z跨度cm |
|---|---|
| 旧P13 pose mesh（A） | 10.2285 / 19.3524 / 11.1139 |
| 新P13 pose mesh（B） | 10.2285 / 19.3524 / 11.1128 |
| 旧run保存的完整native mesh | 8.6248 / 17.2249 / 10.6819 |
| 新完整native mesh（C/D） | 8.6243 / 17.2244 / 10.6815 |

旧P13由7027个accepted observed surface faces和263318个生成面组合；它并非完整native原网格。新P13为7026＋263266面。新完整native保留478880面，但其观测区域没有替换成旧P13的观测表面，所以首帧投影即不同。

旧native scene scale=0.327670529，新=0.327660851，差约0.003%。新visual与同run native metric bridge转换后顶点最大差2.13e−8m，canonical/world往返差6.94e−18m。无需通过硬编码放大网格“修复”不存在的单位转换错误。

特别注意：旧completion report自身声明的`render_mesh_contract.required_mesh_path`指向intact native，但旧RL实际source/zero消费的是`reconstructed_mesh`（P13 pose mesh）。这是历史角色/显示口径不一致，不能据此把P13网格宣称为全拓扑native，也不能因为新显示遵循完整native就忽略实际图像贴合变差。

### RGB seed确实造成新增位姿退化

在新run内部，C与D使用同一完整native mesh、相同camera/K：

- seed与P09初始化平移差：中位24.91mm、p90 39.39mm、最大58.23mm；这是两种初始化之差，不是GT误差。
- frame149：平移差53.40mm，旋转相对源I为116.48°；D明显偏到右侧，C仍覆盖物体主体。
- frame60：D投影明显变窄，与C相比多出位姿影响。
- seed与source各有不确定性，不能把I+centroid升级为真实姿态；这里的结论是本次seed视觉对齐总体退化，而非所有帧或所有旋转分量均错误。

### 同一148帧、同一owned mask的物体三角面轮廓诊断

| 组 | IoU中位数 |
|---|---:|
| A 原RL source | 0.7668 |
| B 新v2 source/P13 | 0.7667 |
| C 新v2 source/完整native | 0.7104 |
| D 新v2 seed/完整native | 0.4518 |

480px真实三角面raycast，无手遮挡。mask同样来自预测，不是GT；完整网格可能投到被遮挡区域，因此IoU不是物理姿态精度。但它与实际检查帧一起支持差异归因。frame149分别约0.810/0.813/0.679/0.309。

## 修正了什么、没有修正什么

已完成：恢复明确的无seed三维源状态展示，并给出同run单变量比较；纠正“只能显示mask/可见表面”的过窄解释。无seed读取已在禁止打开`rgb_source.*`、`measurement_seed.json`、`source_seed_scene.npz`等文件时重新执行，源scene逐数组复现。

未完成：RGB seed算法和完整native形状的图像贴合仍未优化。本次没有把旧P13换回来后称seed已经修复，也没有手调scale或复制V7答案。

下一步若修复seed，应在相同冻结观测下联合使用已有相对RGB测量、双端重投影、逐帧可见表面/轮廓约束校正候选，并和B/C作图像及几何对照。不能直接把逐帧可见表面质心当作旋转后物体原点强行替换，也不能删除RGB信息后宣称完整运动能力恢复。

## 文件

根目录：

`/mnt/truenas-user-home/kupingxin/rl_v2_a800_execution_20260921/original_rl_comparison/`

- **`original_vs_v2_four_way_h264.mp4`**：四格对照，1440²、150帧、30FPS、5s。
- **`v2_source_pose_mesh_no_rgb_seed_h264.mp4`**：B，960²、150帧，与原RL同显示口径。
- **`v2_source_full_native_no_rgb_seed_h264.mp4`**：C，完整native但不使用RGB seed。
- `four_way_000000.jpg` / `000060.jpg` / `000094.jpg` / `000149.jpg`等：同帧图。
- `comparison.json`：全帧几何/投影诊断、输入hash、视频完整解码结果。
- `verification.json`：禁止seed文件读取的源状态复现及148帧同集合统计。
- `compare_sources.py`、`DESIGN.md`、`render.log`、`render.exit`：复现脚本、事前因果设计和执行证据。

三份H264完整解码通过，源文件hash保持不变。生产算法、旧工作区、V7及冻结快照均未修改；没有commit/push。
