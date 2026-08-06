# ego_annotation V19 输入处理、优化问题与 Loss 汇总

## 1. 文档范围和结论先行

本文继续沿着 V19 runtime 的实际 phase graph 梳理输入数据如何被处理，并把其中的机制分成四类：

1. **连续数值优化**：有明确变量、残差/约束和求解器，例如 P13 的相似变换 ICP、P14 的逐帧 ICP、P15 的时序 pose graph、P16 的半空间平移修正、P18 的可微 MANO 轨迹优化；
2. **闭式解、鲁棒统计或离散搜索**：有明确的估计准则，但不一定调用通用连续优化器，例如 median、PCA 轴枚举、Umeyama/Kabsch、候选 frame 评分；
3. **硬门控、启发式和 agent 判断**：会改变哪些观测可以进入下游，但不是 loss，也不能被写成“优化目标已经证明了物理状态”；
4. **预训练模型推理**：UniDepth、HaWoR、OWLv2、SAM2、TRELLIS 在本 runtime 中只执行推理/采样。它们的训练 loss 或模型内部推理目标没有在当前仓库 runtime wrapper 中暴露，不能根据模型名称臆造为本 pipeline 的 runtime loss。

### 1.1 权威运行和可复现范围

本文的主数值结果来自已经存在的权威冻结运行，不把报告缺失的 scalar cost 猜出来。后文显式标成 `eligibility/support ablation` 的数字来自独立 benchmark 输出目录，不回写冻结 run：

```bash
REPO=/mnt/user-home/kupingxin/ego_annotation
OUT=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
CASE=hot3d_clip001851_keyboard_pinhole
OBJECT=keyboard
```

输入是 1408×1408、150 帧、30 FPS、5 秒的 egocentric RGB 视频。该运行的 post-run audit 记录：

- runtime bundle：`/mnt/user-home/kupingxin/ego_annotation_runtime/v19_bundle_a800_0c8e6a9_local1`；
- bundle source revision：`0c8e6a9ff1925caa5fa2665de116c404b5d39eee`；
- 当前仓库检查分支：`local/kupingxin-v19-a800-deployment`；本次文档初始审计时仓库 HEAD 为 `b18cecd2c90932282eb3b50ecef7298bb7546498`，之后 benchmark/docs 提交为 `ca254f7`，eligibility/support quarantine 实现提交为 `b7b97e6`；
- P13/P14/P15/P16/P17/P18/P18b 的冻结执行行为以 runtime bundle 为准。实现提交 `b7b97e6` 中的 P14/P15/P16/P18/P18b/render consumer 已加入 eligibility/support quarantine，因此不能用当前源码反向声称冻结 run 当时执行过这些 gate；`export_hawor_world.py` 的默认路径部署差异不改变冻结调用逻辑；
- `b7b97e6` 已另建 `/mnt/user-home/kupingxin/ego_annotation_runtime/v19_bundle_a800_b7b97e6_pose_gate_local3`，raw-v2 preflight 通过；由于 RGB 未变且仍缺 VRS calibration，没有把它启动或描述成 calibrated rerun。

结果解释以以下文件为准：

- 最终物体 Mesh：
  `$RUN/measurements/geometry_completion/compact_keyboard_seed42/keyboard_compact_rigid_completed_mesh_labeled.ply`
- 最终物体 pose graph：
  `$RUN/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
- P18 原始 MANO 优化状态：
  `$RUN/measurements/mano_interval_correction/keyboard_0_149/hot3d_clip001851_keyboard_pinhole/v18_joint_mano_interval_trajectory_state.json`
- P18b 后的 canonical metric MANO 状态：
  `$RUN/measurements/mano_interval_correction/keyboard_0_149_surface_hypothesis_metric_mano/hot3d_clip001851_keyboard_pinhole/v18_joint_mano_interval_trajectory_state.json`
- renderer boundary：
  `$RUN/state/render_state/keyboard_rigid_render_state.json`
- 全部 phase 事件和最终审计：
  `$RUN/logs/harness_events.jsonl`、`$RUN/FINAL_RUNTIME_AUDIT.json`

### 1.2 最重要的实际结论

1. **P13 和 P14 确实改变了几何/pose 估计**：P13 将 TRELLIS 模型对齐到 metric visible surfels，P14 对 150 个可见帧执行了四轮逐帧 ICP。
2. **P15 在本次运行是数学上退化的零修正问题**：没有传入 P16 constraint report，所有残差在零 correction 处为零；SciPy `least_squares` 以 `nfev=1, cost=0` 返回，150 行 pose 被标成 corrected，但数值完全没有变化。
3. **P16 没有执行有效的 signed nonpenetration 求解**：P13 completed mesh 和作为 sign mesh 的同一 mesh 都不是 watertight，故 300 个 hand/frame 行没有可信 signed penetration correction；“零 penetration”不是非穿透证明。
4. **P18 原始求解器确实运行了，但没有形成可接受的 metric MANO state**：左手部分表面残差下降，右手仍有很大的 observed-surface violation；若干软 hinge 上界被超过；contact latent posterior 基本等于 prior；所有行都没有 visible-surface depth-order support，因此 output gate 重置了全局平移。
5. **P18b 按设计拒绝把 P18 的优化关节/root 升格为 canonical metric MANO**：最终 300 行保留 HaWoR metric joints/root，P18 的 160 点表面样本只作为 uncertain visible-surface proximity hypothesis。
6. **接触不是当前运行的 metric 结果**：P17 只是把 agent 的视觉判断和投影邻近关系变成 uncertainty-weighted factor；P18 的 deadband 为约 54–69 mm，且没有稳定 object-frame contact anchor，不能解释成毫米级接触。

### 1.3 当前 V19 与历史 V17/V18 optimizer 的边界

仓库里还保留了大量 `solve_v17_*`、`optimize_*` 和若干 standalone V18 optimizer。它们不是本次 `_v2` runtime phase graph 的调用路径。例如：

- `build_v18_temporal_mano_translation_interval_state.py` 自带的 `scipy.optimize.minimize`；
- `build_v18_temporal_mano_articulated_interval_state.py` 自带的旧版 `torch.optim.LBFGS`；
- 其它 V17 factor-graph、contact-depth、surface transport 实验脚本。

当前 `solve_v18_joint_mano_interval_trajectory.py` 会从部分 V18 模块导入 MANO replay、depth classification 和数据桥接函数，但没有调用这些模块的 standalone optimizer entrypoint；P18 的实际 loss 是该脚本自己的 `optimize_rows` closure。下文的“所有 loss”指 **当前 V19 `_v2` 实际 phase graph 和其实际调用链中可审计的目标**，不把未执行的历史实验目标冒充为本次运行结果。若要整理完整的 V3–V18 历史分支，需要另建按版本和 artifact 的 loss inventory。

---

## 2. 统一记号、坐标和“loss”边界

### 2.1 坐标和变量

| 记号 | 含义 | 当前约定 |
|---|---|---|
| `I_t` | 原始 RGB 帧 | source grid 1408×1408 |
| `D_t(u,v)` | UniDepth 深度 | 米；archive 为 1408×1408，dtype `float16` |
| `K_t` | 每帧或视频级内参 `[fx,fy,cx,cy]` | 当前下游使用 robust constant `K` |
| `T_world_camera,t` | camera 到 world 的位姿 | 点从 camera 到 world 的旋转/平移语义 |
| `H_t` | MANO hand state | 21 joints、778 vertices、root/pose/translation |
| `O_t` | 物体 state | completed canonical Mesh 加 `SE(3)` pose |
| `x_i` | canonical Mesh/TRELLIS 点 | P13 后为 metric canonical frame |
| `y_i` | visible metric object surfel | world 坐标 |
| `u_t` | MANO 全局 translation correction | P18 中的 world 平移增量 |
| `\delta r_t` | root orientation 增量 | axis-angle，弧度 |
| `\delta p_t` | 15 个 MANO hand-pose 增量 | 每个 axis-angle，弧度 |
| `c_t` | P18 contact latent probability | sigmoid 后在 `[0,1]` |

深度 lifting 的基本关系为：

$$
\mathbf{x}_{camera}=D_t(u,v)K^{-1}[u,v,1]^T,
\qquad
\mathbf{x}_{world}=R_{world\leftarrow camera,t}\mathbf{x}_{camera}+t_{world\leftarrow camera,t}
$$

仓库中的 row-vector 实现会把转置写法放在矩阵乘法一侧；阅读时应以字段名和函数实现为准，不要仅凭公式中的左右乘法判断方向。

### 2.2 四种机制不能混写

| 机制 | 是否是 loss | 是否能改变变量 | 当前典型例子 |
|---|---:|---:|---|
| 连续残差最小二乘 | 是 | 是 | P15、P18 |
| 闭式最小二乘/ICP | 是一个局部目标的求解 | 是 | P13、P14 |
| median/argmax/候选评分 | 通常不是连续 loss | 选择候选 | P03b、P06、P09 |
| mask/extent/watertight/2D 门控 | 不是 loss | 删除或隔离证据 | P09、P13、P16、P17 |
| agent 语义判断 | 不是几何 loss | 写 prior/uncertainty | P05、P09、P17 |
| neural inference/sampling | runtime wrapper 未暴露目标 | 产生测量/先验 | UniDepth、HaWoR、SAM2、TRELLIS |

在下文中，“目标函数”只在源码明确构造了 objective/residual 或有明确闭式准则时使用；“硬门控”会单独标记为 **非 loss**。

---

## 3. 从输入到最终 state 的完整处理流

| Phase | 输入如何被处理 | 优化/选择机制 | 输出和本次运行结果 |
|---|---|---|---|
| P00 | 建立输入、run root、未解析 state | 无 | 输入状态全部 unresolved |
| P01 | 解码视频并建立 frame manifest；另生成 960 review frame | 无 | 150 帧，1408 source / 960 review |
| P02 | 检查 A800/GPU | 离散 GPU 选择，不是物理优化 | GPU 0，A800-SXM4-40GB |
| P03 | 每帧 RGB 送入 UniDepth，resize depth 回 source grid | 模型推理；runtime 无显式 loss | 150×1408×1408 metric-depth archive |
| P03b | 汇总每帧 UniDepth K | coordinate-wise median + square-focal 约束 | `K=[537.0213,537.0213,722.2881,720.8964]` |
| P04 | HaWoR detector/track、motion、SLAM、infiller、MANO replay/export | 外部模型/SLAM 内部目标未在 wrapper 暴露 | 左右各 150 帧、778 vertices/21 joints |
| P05 | agent 从 raw frame 确认 object roster 和 rigid hypothesis | 语义判断，不是几何 loss | `keyboard` body only，rigid |
| P06 | OWLv2 文本 prompt → 候选 boxes；每帧取最高分 box | threshold + argmax 离散选择 | 11/11 prompt frames 有 box |
| P07 | SAM2 box/point prompt → video propagation；输出 mask | SAM2 inference；非 overlap 是约束/配置，不是下游 loss | keyboard mask 150/150 帧 |
| P08 | 把 camera、HaWoR、object mask 统一到 base annotations | schema/坐标桥接 | 仍不是 final physical state |
| P09 | hand bbox 从 object mask 中扣除；depth 反投影；采样 visible surfels | candidate score 只提议 anchor；agent 最终选 | anchor frame 109；150 visible metric rows；143 rows extent-eligible |
| P10 | 从 visible evidence 确认 rigid branch | branch gate | 启用 completion→pose→correction→render |
| P11 | 用 selected anchor 生成 object-isolated TRELLIS conditioning crop | 无 | frame 109 object crop |
| P12 | TRELLIS image-to-3D sparse/SLAT sampling | 预训练生成采样；wrapper 无显式 runtime loss | raw mesh 137,674 vertices / 275,328 faces |
| P13 | TRELLIS → metric visible surfel 的 Sim(3) 对齐；合并/过滤面 | PCA 离散枚举 + Umeyama + 6 轮 ICP；silhouette/slab 为硬门控 | accepted mesh 12,822 vertices / 22,758 faces |
| P14 | 每个 frame 用 visible surfels 对 canonical mesh 做 ICP | nearest-neighbor + Kabsch/Umeyama 闭式迭代 4 次 | 150/150 帧拟合 |
| P15 | 对 P14 pose correction field 建时间图 | SciPy sparse `least_squares`, `soft_l1` | 本次零修正；150 direct rows |
| P16 | MANO vertices 对 completed mesh 做 broadphase/signed 检查 | 仅 watertight 时才调用 SLSQP 半空间解 | mesh 非 watertight；无 correction |
| P17 | 投影 MANO 与 object mask；接收 agent contact/occlusion judgments | prior/ownership heuristic，不是 contact solve | 286 ownership/contact rows，281 active |
| P18 | 每个 hand side 独立优化 150-frame MANO interval | CUDA PyTorch LBFGS + active set | raw candidate 生成，但不是 canonical metric MANO |
| P18b | 保留 HaWoR metric joints；携带 P18 vertices 作为 uncertain surface hypothesis | acceptance/split gate，无优化 | 300 canonical rows，joint shift=0 |
| P19 | 从 render state 栅格化 completed mesh faces 和 MANO/uncertainty | z-buffer/rasterization，不是 state optimization | 150 帧 branch render |
| P20 | 发布 canonical mp4 | 文件复制/原子替换 | 3 个非空 canonical video |
| P21 | agent 消费 render 并记录机制失败 | visual acceptance，不是 loss | rigid body 可用；contact/nonpenetration unresolved |

---

## 4. 前端测量：哪些有准则，哪些没有 runtime loss

### 4.1 P01/P02：解码和资源选择

P01 只读取视频元数据、逐帧 RGB 和 frame index；不存在待优化物理变量。P02 以当前 GPU memory/utilization 选择 GPU 0，这个 `argmin` 只是在资源层面选设备，不能写入物理 objective。

### 4.2 P03 UniDepth：模型推理，不是本仓库可见的 depth loss

`scripts/run_unidepth_full_frame_v3.py` 对每帧执行：

1. 读取 manifest 指向的 RGB；
2. 调用 `infer_unidepth(model, image, device)`；
3. 将 raw depth resize 到 `source_width × source_height`；
4. 读取模型返回的 intrinsics，并按 resize 比例缩放 `fx,fy,cx,cy`；
5. 检查正值/finite 像素，写 PNG review 和 NPZ。

wrapper 没有 `optimizer.step()`、`loss.backward()` 或显式 depth residual。UniDepth 的训练损失属于模型训练阶段，不属于本次 V19 runtime 可审计的 loss。当前结果：

| 量 | 结果 |
|---|---:|
| archive shape | `[150,1408,1408]` |
| positive fraction | 1.0 |
| 每帧 focal median | 536.9555 px |
| 每帧 focal p05–p95 | 501.0723–591.0709 px |
| 全帧 depth median 的 median | 0.96453 m |
| depth median p05–p95 | 0.92347–1.15294 m |

这些是模型测量统计，不是 metric depth 的 ground truth 误差。

### 4.3 P03b 标定 contract：median 是一个鲁棒 L1 估计器

脚本 `build_v19_calibration_contract.py::aggregate_intrinsics` 有两种模式：`median` 和 `trimmed_mean`。本次调用是 `--aggregation median --square-focal`。

对每个内参分量独立求中位数，可以等价写成：

$$
\hat k_j = \arg\min_{k_j}\sum_{t=1}^{150}|k_{t,j}-k_j|,
\qquad j\in\{f_x,f_y,c_x,c_y\}.
$$

之后 `square_focal` 强制：

$$
\hat f=\sqrt{\hat f_x\hat f_y},\qquad
\hat f_x=\hat f_y=\hat f.
$$

这不是一个通过 `scipy.optimize` 求解的连续问题，而是闭式鲁棒统计聚合加一个模型约束。当前输出：

```text
fx = fy = 537.0213035603387 px
cx = 722.2881469726562 px
cy = 720.8964233398438 px
horizontal/vertical FOV = 105.326184 degrees
used frames = 150/150
```

注意：`cx,cy` 没有被 `center_principal_point` 重置到 704；它们保留了 UniDepth 的 robust aggregate。下游必须记住该 K 是“constant pinhole hypothesis derived from UniDepth”，而非数据集标定。

### 4.4 P04 HaWoR：多个外部处理环节，但本仓库没有可审计的统一 loss

`export_hawor_world.py` 实际调用外部 HaWoR 的：

- detector/track video；
- hand motion estimation；
- focal-dependent cache contract；
- HaWoR SLAM；
- learned infiller；
- left/right MANO replay 和 world-space export。

输出中的 SLAM 文件包含 `traj`、`disps`、`img_focal=537.0213` 和 `scale=0.691579`。但 `export_hawor_world.py` 本身没有把 SLAM 的 bundle adjustment residual、infiller training loss 或 MANO fitting loss写入 V19 report；外部 HaWoR 实现也不在当前 runtime script 的显式目标函数中。因此本文只能准确记录：

- 左右手 `valid_hand_frames` 均为 150；
- same-frame detector support：left 136、right 127；
- P04 输出的是 metric MANO **候选测量**，不是已被本 pipeline loss 闭合的 contact state；
- same-frame detector 缺失的 infilled rows 仍带有 provenance，不能被当作同等强度观测。

### 4.5 P05/P06/P07：选择和分割不是物理优化

#### P05

agent 写 object plan，给出文本 prompt、evidence frames、rigid hypothesis 和 uncertainty。这里没有像素级坐标 loss；agent 语义不能替代 detector/mask。

#### P06 OWLv2

`scripts/build_v19_owlv2_object_box_prompts.py`：

- threshold=`0.03`；
- 对每个 prompt frame 取 processor 返回 detection 中 score 最高者；
- box 只用于 SAM2 prompt，不是物体 mask、3D geometry 或 pose。

可把选框写作离散规则：

$$
q_t=\arg\max_{q\in\mathcal Q_t}\operatorname{score}_{OWL}(q),
\quad \operatorname{score}_{OWL}(q)\ge 0.03.
$$

这不是可微 loss。当前 11 个 prompt frames 全部有 box，score 范围约为 0.514–0.751。

#### P07 SAM2

SAM2 使用 OWLv2 box prompt（pad ratio 0.18、minimum pad 24 px），在 960×960 video grid 中传播 mask logits；代码以 `logit > 0` 形成 binary mask，开启 `non_overlap_masks=True`。这属于已训练视频分割器的 inference 和对象间硬竞争配置，runtime wrapper 没有 segmentation loss。

当前 keyboard track：

- source video：1408×1408；SAM2 grid：960×960；
- active interval `[0,149]`；
- visible mask rows：150/150；
- prompt contract reports：11，其中 10 个满足，frame 149 仍有边界/部分出框不确定性；
- 视觉 self-check 认为身份保留，没有转移到手、桌面、鼠标、手机。

---

## 5. P09：Mask+Depth 可见表面、anchor proposal 和 Poisson 重建

### 5.1 输入如何变成 visible metric surfels

对每个有 SAM2 mask 和 depth 的 frame，P09：

1. 读取 960 mask，并在 source/depth 尺寸之间进行显式 resize；
2. 从 base annotation 的同帧 hand bbox 中扣除 hand-owned 区域，在 mask grid 中使用 `pad=12 px`；
3. 使用 calibration contract K，而不是直接混用每帧 UniDepth K；
4. 保留 `0.05 m <= depth <= 4.0 m` 的 mask 内像素；
5. pixel stride=`4`、最多采样 `2500` 个点；
6. 反投影到 camera/world；
7. 计算 centroid、extent、depth p05/p95，并写入 `visible_geometry_candidate`。

扣除 hand bbox 是 **hard ownership gate**，不是一个惩罚项；被扣掉的像素只能进入 occlusion/uncertainty 解释，不能继续作为 object surface loss 的观测。

### 5.2 Anchor proposal score：启发式排序，不是接受 loss

`build_anchor_candidate_proposals` 对每个可见 frame 计算：

- mask area 归一化项 `area_score`；
- visible depth point count 项 `point_score`；
- hand removal clean 项 `1 - removed_fraction`；
- border 项；
- world extent 相对于全体 median extent 的一致性项；
- connected-component 项；
- depth spread 稳定项。

确切的 proposal score 是：

$$
S_t=0.24A_t+0.16P_t+0.18C_t+0.16B_t+0.14E_t+0.07G_t+0.05D_t.
$$

`S_t` 只用于生成 ranked review sheet。runtime contract 要求 agent 查看 review image 后另写 anchor decision；最终 anchor 不是 `argmax(S_t)` 的自动物理结论。

当前选中的 frame 109 的证据：

- object-owned mask：63,861 px；
- hand-owned removal：4,056 px，6.0%；
- sampled depth points：2,500；
- world extent diagonal：约 0.5598 m；
- rank 1，但 decision 明确写了视觉/几何 rationale，而不是只依据 score。

### 5.3 Anchor 可见 Mesh：Poisson 是隐式重建，wrapper 没有暴露标量能量

anchor 点先 voxel downsample，再估计 normals，调用 Open3D `create_from_point_cloud_poisson`；当前：

- input points：2,500；
- downsampled points：1,421；
- voxel size：0.006997 m；
- Poisson depth：7；
- output：2,527 vertices、4,882 faces。

从 Poisson reconstruction 的数学含义，可以把它概念化为根据 oriented points 求一个隐式 indicator field，使其梯度接近 oriented normal field，例如：

$$
\min_\chi \int \|\nabla\chi(x)-V(x)\|^2\,dx,
$$

再从隐式场提取等值面。**但是当前脚本没有构造或记录这个连续目标的数值、权重和迭代收敛信息**；因此在 pipeline audit 中应把它记成 library reconstruction，而不是声称有一个可复现的 Poisson loss 曲线。

### 5.4 冻结 runtime 的 P09 结果和 eligibility 数据流差异

P09 的 extent gate 标记：

- eligible：143/150；
- ineligible frames：`[0,18,19,20,55,60,69]`；
- 原因是 extent 与 selected anchor 不一致，疑似 mask/background leakage。

冻结 bundle 中的 `fit_v18_compact_rigid_object_pose.py` 只检查 visible samples 和初始 pose，**没有读取 `rigid_pose_observation_eligible` 字段并跳过这 7 帧**。因此该次冻结 P14/P15 实际仍使用了 150 帧。这个历史差异必须在下游解释中保留，不能用后续修复回写冻结结果。

实现提交 `b7b97e6` 已将 explicit false 接成 P14 hard gate，并保留两个可审计例外：缺失字段仅作为 legacy compatibility；`--include-ineligible-rigid-pose-observations` 仅作为显式历史复现 override。P15 还会独立二次拒绝仍携带 explicit false 的 fitted row。

---

## 6. P11–P13：完整 Mesh 的生成、metric 对齐和硬过滤

### 6.1 P11/P12 的处理

P11 只使用 agent 已选的 frame 109，生成 object-isolated RGBA crop。P12 将这个 crop 送入 TRELLIS，不使用 raw full frame 或 binary mask 作为 shape input。

当前 TRELLIS wrapper 的参数是：

- seed=42；
- sparse structure sampler：12 steps、CFG 7.5；
- SLAT sampler：12 steps、CFG 3.0；
- raw output：137,674 vertices、275,328 faces。

这些 steps/CFG 是生成采样超参数，不等于一个在当前运行中最小化的显式 geometric loss。TRELLIS 的训练/扩散 denoising objective 不在 V19 report 中。

### 6.2 P13 的变量和初始化

输入是：

- anchor visible fused points `Y={y_i}`，单位米；
- raw TRELLIS mesh/sample `X={x_i}`，模型单位；
- anchor centroid 和 evidence-frame camera/mask。

P13 先用 observed/model PCA basis 枚举 proper rotation candidates，并用 observed RMS radius / model RMS radius 初始化 scale。离散 rotation candidates 来自轴 permutation 和 sign combinations；这一步是有限候选搜索，不是梯度优化。

### 6.3 P13 的候选准则和 Sim(3) 目标

对每个 rotation candidate，计算 transformed TRELLIS surface 到 observed point cloud 的 nearest-neighbor statistics。候选 score 是：

$$
S(R)=\operatorname{median}(d_{Y\rightarrow X})+0.25\operatorname{p90}(d_{Y\rightarrow X}).
$$

选出 candidate 后，P13 做最多 6 轮 ICP。固定当前 nearest-neighbor correspondence 时，Umeyama 求解的核心问题可写成：

$$
\min_{s>0,R\in SO(3),t\in\mathbb R^3}
\sum_i\left\|sR x_{c(i)}+t-y_i\right\|_2^2.
$$

Umeyama 在每一轮给出闭式 `s,R,t`；下一轮重新用 transformed model surface 建 KD-tree。该目标是 **单向 observed-to-model correspondence 的 point-to-point similarity fit**，最终的双向 distance 只是诊断，不是另一个被联合优化的 loss。

### 6.4 P13 的硬门控不是 loss

P13 对 face 做三种语义处理：

1. **observed-band overwrite**：距离 visible depth surfel 小于 `observed_band_m` 的 TRELLIS face 不再作为 hidden completion；
2. **silhouette/free-space filter**：将 canonical TRELLIS face center 投影到 evidence frame，K 从 source 1408 grid 缩放到 mask 960 grid，mask dilation=16 px；投影落到 object-owned silhouette 外的 hidden face 被拒绝；
3. **planar-slab support filter**：若 observed points 的最小/最大 PCA eigenvalue ratio ≤0.04，则只保留 observed support plane 附近的 hidden faces。当前 slab half-width 被 clamp 在 0.018–0.055 m。

它们是 binary keep/reject，不是给 hidden face 加一个可调权重的 soft loss。

### 6.5 当前 P13 数值结果

当前 observed band 是 `voxel_size × sqrt(3)=0.0121192 m`，用于把一个 voxel diagonal 范围内的 face 归入 observed support；它是分类带宽，不是拟合 residual 的 sigma。

| 指标 | 当前结果 |
|---|---:|
| Sim(3) scale | 0.40529294 |
| observed→TRELLIS initial median / p95 | 3.0616 mm / 11.5795 mm |
| observed→TRELLIS final median / p95 | 2.9917 mm / 9.9582 mm |
| final TRELLIS→observed median / p95 | 7.0245 mm / 99.0630 mm |
| observed band | 0.0121192 m |
| silhouette input/kept/rejected faces | 275,328 / 183,571 / 91,757 |
| planar slab ratio | 0.001636，触发 filter |
| planar slab kept/rejected | 275,328 / 0 |
| accepted observed-depth faces | 4,598 |
| excluded unsupported observed Poisson faces | 284 |
| accepted hidden TRELLIS faces | 18,160 |
| completed Mesh | 12,822 vertices / 22,758 faces |

P13 的 6 轮 ICP refinement trace（observed→transformed TRELLIS nearest distance）是：

| 轮次 | mean | median | p95 |
|---:|---:|---:|---:|
| 1 | 4.2213 mm | 2.9202 mm | 11.6471 mm |
| 2 | 4.1317 mm | 2.9805 mm | 10.9773 mm |
| 3 | 4.0789 mm | 2.9634 mm | 10.7866 mm |
| 4 | 4.0440 mm | 2.9967 mm | 10.4318 mm |
| 5 | 3.9975 mm | 2.9820 mm | 9.8970 mm |
| 6 | 3.9446 mm | 2.9917 mm | 9.9582 mm |

median 并非每轮严格单调，但 mean/p95 总体下降；这只是 correspondence 更新后的 surface diagnostic，不是一个被单独记录的 optimizer cost。

`completed_mesh_labeled` 只由 `observed_depth_surface` 和 `trellis_inferred_hidden_surface` 组成。它不是 raw `trellis_mesh.ply`，也不是 watertight body；audit 的 `watertight=false` 必须继续传播到 P16/contact state。

---

## 7. P14：逐帧 completed-mesh pose fit

### 7.1 建模

每帧从 completed canonical mesh 采样 6,000 个 surface points，使用已有的 P09 centroid pose 作为初值。对当前 frame 的 visible metric points `Y_t`：

1. 将 canonical samples 按当前 `R_t,t_t` 变换到 world；
2. 对每个 observed point 找最近 model sample `c_t(j)`；
3. 以 matched pairs 执行 rigid Umeyama/Kabsch；
4. 重复 4 次。

变量是每帧独立的 `R_t∈SO(3), t_t∈R^3`；没有 temporal coupling，也没有将相邻帧速度作为 loss。

### 7.2 目标函数

固定 correspondence 时，闭式解对应：

$$
\min_{R_t\in SO(3),t_t}
\sum_{j\in Y_t}
\left\|R_t x_{c_t(j)}+t_t-y_{t,j}\right\|_2^2.
$$

代码实际用 nearest-neighbor 后的 Kabsch/Umeyama 闭式更新，没有显式 robust kernel、visibility weight 或 residual trimming。输出的 `observed_to_mesh_*` 与 `mesh_to_observed_*` 是评估统计；前者更接近 fit update 的方向，后者用于暴露 hidden mesh/partial-view mismatch。

### 7.3 冻结 keyboard 结果

由 P14 report 的 150 个 pose rows 汇总：

| 指标 | initial | final |
|---|---:|---:|
| observed→mesh median | 7.0451 mm | 3.8354 mm |
| observed→mesh p90 | 12.8029 mm | 5.0162 mm |
| observed→mesh max | 19.6562 mm | 7.1873 mm |
| final mesh→observed median | — | 25.8009 mm |
| final mesh→observed p90 | — | 58.4130 mm |

P14 因而确实产生了每帧 pose 改变。但冻结 bundle 没有尊重 P09 的 7 个 `rigid_pose_observation_eligible=false` 标记；这些数值是 bug-preserving execution record，不是 `b7b97e6` eligibility gate 的行为。

在独立 tire-lever ablation 中，`b7b97e6` 只拟合 6 个 explicit-eligible rows、拒绝 27 个 explicit-false rows；eligible-only final median 为 `2.006 mm`，而冻结 all-row aggregate final median 为 `63.038 mm`。这只说明污染 measurement 被删除，不证明 full-timeline pose accurate。

---

## 8. P15：时序刚体 pose graph

### 8.1 变量化

对每个可见 pose observation `i`，定义 correction field：

$$
R_i=\exp(\Delta r_i)R_i^{obs},
\qquad
t_i=t_i^{obs}+\Delta t_i.
$$

注意它只平滑 correction field `Δr,Δt`，不直接平滑物体真实运动；这样理论上可以保留 P14 观测到的 rigid motion。

### 8.2 残差和目标函数

`residual_vector` 由以下残差拼接而成：

#### (a) pose measurement prior

$$
r_{t,i}=\frac{\Delta t_i}{\sigma_{t,i}},
\qquad
r_{r,i}=\frac{\Delta r_i}{\sigma_{r,i}}.
$$

`pose_row_sigma` 使用 P14 的 `max(median_m,p95_m)`，再按 visible sample count 收紧，并裁剪到：

$$
 s_n=\sqrt{\frac{\min(n_{visible},400)}{100}},
\qquad
 \sigma_t=\operatorname{clip}\left(\frac{\max(\operatorname{median},\operatorname{p95})}{s_n},0.004,0.045\right),
$$

$$
 \sigma_r=\operatorname{clip}\left(\frac{\sigma_t}{r_{object}},0.035,0.35\right).
$$

当前实现的范围是：

- translation sigma：0.004–0.045 m；
- rotation sigma：0.035–0.35 rad。

#### (b) 可选 nonpenetration pressure

如果 constraint report 中的 row 状态属于允许集合，并且有 candidate translation `c_i`，代码使用 `-c_i` 作为 object correction 的软目标，并截断到 0.015 m：

$$
 r_{np,i}=\sqrt{w_i}\frac{\Delta t_i-q_i}{0.030}.
$$

这不是把手部 candidate 当作 object ground truth，而是 bounded soft pressure。

#### (c) correction field 一阶平滑

对相邻 observation，按 frame gap `g_i` 缩放：

$$
 r_{v_t,i}=\frac{\Delta t_i-\Delta t_{i-1}}
 {0.010\sqrt{g_i}},
\qquad
 r_{v_r,i}=\frac{\Delta r_i-\Delta r_{i-1}}
 {0.080\sqrt{g_i}}.
$$

#### (d) correction field 二阶差分/加速度先验

$$
 r_{a_t,i}=\frac{v_{t,i+1}-v_{t,i-1}}
 {0.006\sqrt{\max(g_i,g_{i+1})}},
$$

其中 `v_t` 是按各自 frame gap 归一化的一阶差分；rotation 同理，尺度为 0.050 rad。

整体是 `scipy.optimize.least_squares` 的 sparse Jacobian 问题，调用配置：

```text
loss = soft_l1
f_scale = 1.0
x_scale = jac
max_nfev = 80
```

SciPy `soft_l1` 的 cost 可概念化为：

$$
C(x)=\frac12\sum_k \rho(f_k(x)^2),
\qquad
\rho(z)=2(\sqrt{1+z}-1),
$$

这里 `f_scale=1`。**surface nearest-neighbor error 没有进入 P15 residual vector**；它只在 solve 后作为 acceptance check。

### 8.3 当前 P15 的真正输入和结果

runtime 的 P15 command 没有传 `--constraint-report`，而 P16 又在 phase order 中位于 P15 之后。因此本次：

- `nonpenetration_target_frame_count=0`；
- 所有 correction 变量初值 `x0=0`；
- measurement prior、速度平滑和加速度 prior 在 `x0=0` 都为零；
- `least_squares` 立即以 `gtol` 停止：`nfev=1, cost=0, residual_rms_before=0, residual_rms_after=0`；
- translation/rotation correction 的 median、p90、p95、max 全部为 0；
- 150 行都是 `corrected_temporal_rigid_pose_graph`，没有 interpolation rows。

P15 report 的 before/after surface metrics完全相同：

```text
observed→mesh median = 4.828408 mm
observed→mesh p90    = 5.794702 mm
mesh→observed median = 25.691243 mm
surface degradation  = 0
```

因此 `status=corrected_pose_graph_surface_preserved_no_nonpenetration_gain` 的准确解释是：surface 被保留，但没有任何 nonpenetration gain；不是“pose graph 找到了非零时序修正”。

### 8.4 P15 的结构性注意事项

- P15 的 pose prior 是“对 P14 pose 的 correction prior”，不是从 RGB 重新拟合 pose；零 correction 本来就是最优。
- P15 的 surface metrics 使用独立 2,500 sample/seed，和 P14 的 6,000 sample/seed 不应直接当成同一 loss 曲线。
- P16 结果若要影响 P15，必须显式重排 phase 或再运行一个带 constraint report 的 pose graph；当前一次 pass 没有这条数据流。
- full-timeline completion 在本次冻结 keyboard run 没有真正插值，因为 150 帧都有 direct pose；`completed_row_count=0`。

`b7b97e6` 不再把 completion row count 当作 support。若 trusted graph frames 少于默认 8，P15 仍可为 failure render 建立 full timeline，但顶层与每行都标 `annotation_ready=false`/`graph_support_sufficient=false`。真实 tire-lever 修复 ablation 是 6 direct +144 completion（141 nearest holds、3 interpolations），因此没有晋级为物理 trajectory。

---

## 9. P16：MANO/object 非穿透 candidate 的半空间问题

### 9.1 触发条件和几何建模

P16 对 completed Mesh 建 Open3D `RaycastingScene`，先做：

- surface AABB broadphase；
- 到 surface samples 的 unsigned distance；
- 若提供 sign mesh 且 mesh watertight，才进行 signed distance。

对穿透点 `p_i` 和最近表面点 `q_i`，定义局部 outward escape normal：

$$
 n_i=\frac{q_i-p_i}{\|q_i-p_i\|}.
$$

一阶 signed-distance escape 约束是：

$$
 n_i^T t\ge d_i,
$$

其中 `d_i` 是正的 penetration depth，`t` 是 rigid hand translation candidate。代码用 SLSQP 求：

$$
\min_t \frac12\|t\|_2^2
\quad\text{s.t.}\quad
N t\ge d.
$$

初始点 `x0` 是按 penetration depth 加权的 displacement average。随后还有一个保守 amplification gate：

$$
\|t\|\le\sqrt{N_{penetrating}}\,d_{max},
$$

并要求 candidate correction 后投影关节落入同帧 detector bbox 的比例不下降。后者是 2D compatibility gate，不是 loss。

### 9.2 当前 P16 没有闭合 signed problem

当前 report：

- completed surface mesh `watertight=false`；
- sign mesh 同样 `watertight=false`；
- measured pair count：300；
- candidate correction count：0；
- left/right `frames_with_any_penetration=0`。

状态计数是：

```text
no_penetration_no_coordinate_change_needed:       201 rows
uncertainty_only_nonwatertight_mesh_no_signed_correction: 99 rows
```

这里的 201 行主要表示没有进入需要 signed correction 的 broadphase/overlap 分支；99 行明确指出 mesh 不可用于 signed correction。故 P16 的 `status=ok` 只是 measurement artifact 成功写出，不能解读为 `nonpenetration=true`。

---

## 10. P17：visible ownership/contact factor —— 先验生成，不是接触优化

### 10.1 输入处理

`scripts/build_v19_visible_contact_ownership_factor.py` 对每个 P09 eligible frame/hand side：

1. 读取 object-owned mask；
2. 将 MANO joints、sampled vertices 投影到 mask grid；
3. 用 hand radius=10 px、line radius=6 px 画 projected hand support；
4. ownership dilation=8 px；
5. 根据 agent 的 occlusion judgment 决定 object pixels 是否进入 `non_object_owned`；
6. 计算 hand/object overlap 和 18 px image band 内邻近关系；
7. 生成 `visible_ownership` 和 `contact_patch` rows。

这些 rows 的 contract 明确写着：

- projected hand overlap 是 hand-owned/uncertain first-surface evidence；
- contact patch 是 latent hypothesis；
- 没有 persistent object-frame anchor；
- 不接受 contact ownership 或 pose correction 为事实。

### 10.2 image prior 和 agent prior 的启发式公式

代码先构造 image adjacency prior：

$$
 p_{img}=\operatorname{clip}_{[0.05,0.65]}\left(
 0.10+0.35\,p_{near}+0.20\min(1,4p_{overlap})+0.10p_{near\_fraction}
 \right).
$$

再依据 agent state 合并：

- `likely_contact`：`max(p_img,p_agent)`；
- `possible_contact`：`max(min(p_img,0.65),p_agent)`；
- `no_contact`：`min(p_img,p_agent)`；
- 其他：使用 agent prior。

P17 的 active row weight 为：

$$
 w_{row}=25000\cdot\max(0.65,p_{contact})\cdot m_{agent},
$$

但这是 factor 的候选权重；P18 在 `--optimize-contact-state` 时会读取 row 的 `contact_patch_base_weight`，实际 contact residual 使用 base weight 25,000（见 P18 章节）。

### 10.3 当前 P17 结果

agent judgments 共 5 段：

- left likely contact `[0,149]`，prior 0.90，uncertainty 0.05 m；
- right likely contact `[0,93]`，prior 0.85，uncertainty 0.055 m；
- right possible `[94,119]`，prior 0.65，uncertainty 0.065 m；
- right likely `[120,144]`，prior 0.80，uncertainty 0.055 m；
- right no contact `[145,149]`，prior 0.05。

因为 P09 的 7 个 ineligible frames 被 P17 跳过：

- ownership rows：286；
- contact rows：286；
- active contact rows：281；
- inactive no-contact rows：5；
- active rows 的 hand/object mask ownership 仍是 uncertainty measurement，不是 metric contact。

---

## 11. P18：联合 MANO interval trajectory 的完整 loss inventory

P18 实际读取的是 P15 pose report、P13 completed mesh、UniDepth depth 和 P17 generic factor report；没有把 P16 `v18_mano_object_constraint_state.json` 作为一个 factor report 传入。因此 P18 的 observed-surface barrier 是自己从 mesh/depth 重新构造的局部一阶约束，不是 P16 SLSQP candidate 的后续 refinement。

### 11.1 P18 的变量

对一个 side 的 150 帧 interval，变量包括：

- `trans_delta[t] ∈ R^3`：全局 MANO translation correction；
- `root_delta[t] ∈ R^3`：root orientation axis-angle increment；
- `pose_delta[t] ∈ R^(15×3)`：MANO articulation increments；
- `object_trans_delta[t] ∈ R^3`：可选，本次关闭；
- `contact_logit[t]`：可选，本次开启，经过 sigmoid 得到 `C_t`。

本次每个 side 的基础优化维度是 `150 × (3+3+45) = 7650`，另有最多 150 个 contact logits；object translation 没有加入 optimizer parameter list。

### 11.2 P18 的 MANO forward model 和 zero-surface mode

先用 WiLoR/MANO model replay 原始参数，检查 replay vertices/joints 与输入 source 的 median error。只有 replay pass 才进入 LBFGS。

root/pose 的更新是左乘局部 rotation：

$$
R^{root}_t=\exp(\delta r_t)R^{root,0}_t,
\qquad
R^{pose}_{t,j}=\exp(\delta p_{t,j})R^{pose,0}_{t,j}.
$$

本次 `zero_surface_mode=bridge_delta`。若 `V_t^0` 是 current V18 bridge surface，`V_t^{raw}(\delta)` 是 MANO 在新 root/pose 下的 raw output，`S_t,Q_t,b_t` 是 raw→current 的 similarity bridge，则代码的模型可写成：

$$
V_t(\delta,u)=V_t^0
 +S_t\,\bigl(V_t^{raw}(\delta)-V_t^{raw}(0)\bigr)Q_t^T
 +u_t,
$$

joints 使用同样的 delta mapping。它优化的是“当前 bridge 附近的 MANO 局部形变”，不是重新估计 shape、camera 或物体 geometry。

### 11.3 总目标

把代码里的各项写成：

$$
L=L_{contact-state}+L_{obs-prior}+L_{temporal}+L_{object-prior}
 +L_{bounds}+L_{surface}+L_{2D}+L_{depth}
 +L_{surface-order}+L_{contact-patch}.
$$

下表逐项给出源码实际形式、当前权重和当前是否有效。`ReLU(x)` 记作 `max(0,x)`。

#### 11.3.1 接触 latent state 项

定义：

$$
C_t=\sigma(\ell_t),
\qquad p_t=\text{P17 contact prior}.
$$

**Prior deviation**：

$$
L_{C,prior}
=\operatorname{mean}_{t\in A}\left[
 w_t(0.01)^2(C_t-p_t)^2
\right],
$$

其中 `w_t` 为 contact patch base weight，当前为25,000，故每行 prior strength 为 `2.5`。

**Temporal contact state smoothness**：相邻同 side active rows：

$$
L_{C,temp}
=\operatorname{mean}_{(t,t+1)}
\left[\frac{w_t+w_{t+1}}2
 (C_{t+1}-C_t)^2\right].
$$

**Geometry likelihood** 本来可以加：

$$
\operatorname{mean}[w_t(0.01)^2(C_t-p^{geom}_t)^2],
$$

但本次 `contact_state_geometry_likelihood=false`，没有进入 loss。

#### 11.3.2 MANO translation/root/pose 的零状态先验

设 `m_t` 是 hand-observation multiplier，`w^joint_{t,j}` 是 pose visibility weight。当前两者全部为1。

$$
L_{trans-prior}\propto
2000\cdot\operatorname{mean}_t\|m_tu_t\|^2,
$$

$$
L_{root-prior}\propto
150\cdot\operatorname{mean}_t\|m_t\delta r_t\|^2,
$$

$$
L_{pose-prior}\propto
75\cdot\operatorname{mean}_{t,j}
 w^{joint}_{t,j}\|\delta p_{t,j}\|^2.
$$

实现的 denominator 对 translation/pose 使用 3 个分量归一化；root 使用 `sum(m)*9` 的 code-level normalization，应以源码为准，不要只按名字猜权重的绝对量级。

#### 11.3.3 时间平滑和加速度

对 `x_t` 分别调用 `temporal_terms`：`u_t`、`δr_t`、`δp_t`：

$$
L_{vel}(x)=5000\,\operatorname{mean}_t\|x_t-x_{t-1}\|^2,
$$

$$
L_{acc}(x)=10000\,\operatorname{mean}_t
\|x_{t+1}-2x_t+x_{t-1}\|^2.
$$

P18 rows 本次是完整连续 frame 0–149，代码没有像 P15 那样显式按 frame gap 缩放；对于缺帧的其它运行，这一点会改变平滑的物理含义。

#### 11.3.4 可选 object translation prior

若开启 `optimize_object_translation`，代码会加入：

$$
L_{obj-prior}=4000\operatorname{mean}_t\|o_t\|^2
 +L_{vel}(o_t;8000).
$$

同时有 object bound hinge，最大 object translation 默认 0.015 m。本次 `optimize_object_translation=false`，所以 `o_t=0` 且这部分没有可移动的 object variable。

#### 11.3.5 可选 hand-ray shift prior

如果传入 hand-depth repair factor，代码会约束：

$$
L_{ray}=\frac1N\sum_t w_t
\left\|u_t-h^{ray}_t\right\|^2.
$$

本次没有 `hand_depth_repair_graph`，`use_hand_ray_shift_prior=false`，active count=0。

#### 11.3.6 软边界 hinge

代码没有把这些设置成 hard bound，而是加入 soft squared hinge：

$$
L_{bound}=3000\operatorname{mean}_t\left[
 \operatorname{ReLU}(\|u_t\|-0.045)^2
 +\operatorname{ReLU}(\|o_t\|-0.015)^2
 +\operatorname{ReLU}(\|\delta r_t\|-0.30)^2
 +\operatorname{mean}_j\operatorname{ReLU}(\|\delta p_{t,j}\|-0.45)^2
\right].
$$

因此报告中的 correction 超过这些数值不代表 solver 违反了一个硬约束；它只说明 hinge 惩罚没有压住数据/其它 loss 的冲突。

#### 11.3.7 active observed-surface penetration barrier

P18 先把 completed mesh faces 按 object projected depth 与 UniDepth 分类，只有 `observed_supported_strict` faces 可以成为 trusted observed surface；free-space/conflicting hidden volume 不应直接施力。

对当前 reference vertex、局部表面法向 `n_i` 和线性化 required displacement `d_i`，active residual 是：

$$
 r^{active}_{t,i}=
\operatorname{ReLU}\left(
 d_{t,i}-\sigma^{surface}_t
 -n_{t,i}^T\left(V_{t,i}-V^{ref}_{t,i}-o_t\right)
\right),
$$

$$
 L_{active}=300000\operatorname{mean}_{t,i}(r^{active}_{t,i})^2.
$$

它是局部一阶 tangent-plane barrier，不是精确的全局 signed-distance constrained solve。

#### 11.3.8 dense observed-surface barrier

当 `dense_observed_surface_barrier=true`，对所有 nearest face 属于 observed-supported 的 MANO vertices 重建 tangent-plane residual，而不只取当前已经穿透的 vertices：

$$
L_{dense}=300000\cdot
\frac{\sum_{t,i}(r^{dense}_{t,i})^2}
 {\max(1,\#\{r^{dense}_{t,i}>0\})}.
$$

当前 active：

- left dense constraint count median 354.5，max 778；
- right median 75，max 778。

该 barrier 仍依赖 non-watertight completed surface 的局部 sign/closest-point 几何，不能自动获得 watertight nonpenetration 语义。

#### 11.3.9 2D self-reprojection hinge

如果有 projection，代码使用当前 MANO 自己的 `base_uv` 作为 reference，不是独立 detector keypoints：

$$
 r^{2D}_{t,j}=\operatorname{ReLU}
 \left(\left\|\pi(J_{t,j})-\pi(J^0_{t,j})\right\|_2-12\right),
$$

$$
 L_{2D}=800\operatorname{mean}_{t,j}w_{t,j}r^{2D}_{t,j}{}^2.
$$

所以它只限制优化不要把 MANO 投影移动太远，并不把手拉向新的 RGB keypoint measurement。

#### 11.3.10 camera-depth self-consistency hinge

同样以当前 MANO joint depth 为 baseline：

$$
 r^{z}_{t,j}=\operatorname{ReLU}
 \left(|z(J_{t,j})-z(J^0_{t,j})|-0.035\right),
$$

$$
 L_{depth}=20000\operatorname{mean}_{t,j}w_{t,j}r^z_{t,j}{}^2.
$$

它不是直接对 UniDepth `D_t(u,v)` 拟合；UniDepth 只在 object-surface classification 和可选 depth-order 中提供 support。

#### 11.3.11 visible first-surface depth-order barrier

若 MANO vertex 投影在 visible-surface mask 内且能取到 metric depth `z_surface`，则：

$$
 r^{order}_{t,i}=\operatorname{ReLU}
 \left(z^{surface}_{t,i}-0.010-z(V_{t,i})\right),
$$

$$
 L_{order}=20000\operatorname{mean}_{t,i}(r^{order}_{t,i})^2.
$$

本次 CLI 开启了 `visible_surface_depth_order_term`，但 report 显示左右两侧 `selected_visible_surface_depth_order_vertex_count` 都是 count=0；因此这项在实际 run 中没有有效 residual。

造成该结果的一个已确认实现风险是栅格坐标契约：

- P17 生成 ownership masks 的尺寸是 960×960，并在投影时主动把 source K 缩放到 mask grid；
- P18 的 `project_world` 调用 `build_v18_mano_object_constraint_state.py::project`，返回 source-K 的 UV；
- P18 的 `mask_membership` 直接把该 UV 当作 960 mask 的 index，没有做 1408→960 缩放；
- annotation 中 K 的 source size 是 1408×1408，ownership mask 实际是 960×960。

因此 P18 的 mask-based depth-order 和 ownership face quarantine 可能因为 source/runtime grid mismatch 而失去 support。当前结果中的 selected count=0、visible ownership quarantined face count=0 与该风险一致；需要修复/复测后才能把这项 loss 视为有效观测。

#### 11.3.12 contact patch 的 deadband point-to-plane residual

P17 active row 触发后，P18 从当前 MANO vertices 找到距离 observed strict object face 不超过 row `band_m` 的 vertices，取最近 face point `q_i` 和 face normal `n_i`；本次 row `band_m=0.18 m`，最多 160 vertices。

实际 residual mode 是 `deadband_tube`：

$$
 r^{contact}_{t,i}=
 \operatorname{ReLU}\left(
 \left|n_{t,i}^T(V_{t,i}-q_{t,i}-o_t)\right|
 -\left(m_{t}+\sigma^{patch}_{t}\right)
 \right),
$$

$$
 L_{contact}=25000\,C_t\operatorname{mean}_i(r^{contact}_{t,i})^2.
$$

本次：

- target margin `m_t=0.004 m`；
- left support uncertainty 通常 `0.05 m`，deadband 约 `0.054 m`；
- right uncertainty `0.055–0.065 m`，deadband 约 `0.059–0.069 m`；
- 只约束 normal gap，不约束 tangent gap；
- `require_contact_patch_pose_anchor=false`，但每行都标记 `no_stable_pose_anchor`。

因此满足该 residual 并不等于 hand surface 距离 object surface 只有几毫米；一个 5–7 cm 的法向 tube 仍可能获得零 contact residual。

### 11.4 P18 solver mechanics：LBFGS + active-set，而不是一次精确约束解

P18 的可微优化器是：

```text
torch.optim.LBFGS
first pass:  lr=0.35, max_iter=120, strong_wolfe
later pass:  lr=0.25, max_iter=120, strong_wolfe
active_set_iterations=6
```

每个 active-set pass：

1. 执行 LBFGS closure；
2. 重新对完整 observed surface 测量 penetration；
3. 把新发现的 violating vertices 加入 active constraint set；
4. 重新构造 dense barrier；
5. 最多六轮，新增数量为零才算 `active_set_closed=true`。

P18 report 没有保存每轮 LBFGS 的 scalar `loss`、gradient norm、line-search status、actual iteration count 或 closure 的各项分解。因此只能使用 residual diagnostics 和 delta statistics 评价，不能写“总 loss 从 A 降到 B”。

### 11.5 当前 P18 运行结果

#### 左手

| 诊断 | 初始 | solver 后（注意是 output gate 前的 raw hypothesis） |
|---|---:|---:|
| per-frame observed-surface max 的 median | 0 | 0 |
| p95 | 0.136 mm | 0 |
| 全部 per-frame max | 14.857 mm | 3.664 mm |
| final active linear residual max | — | 7.813 mm |
| corrected frame count | — | 56/150 |
| raw translation delta max | — | 55.414 mm |
| raw root delta max | — | 0.0154 rad |
| raw max pose-joint delta max | — | 0.0208 rad |
| active-set added counts | — | `[7,3,4,1,5,0]`，closed=true |

#### 右手

| 诊断 | 初始 | solver 后（output gate 前的 raw hypothesis） |
|---|---:|---:|
| per-frame observed-surface max 的 median | 81.728 mm | 0 |
| p90 | 160.434 mm | 102.609 mm |
| p95 | 184.410 mm | 135.109 mm |
| 全部 per-frame max | 222.312 mm | 181.908 mm |
| final active linear residual median / max | — | 49.304 mm / 188.115 mm |
| corrected frame count | — | 100/150 |
| raw translation delta max | — | 140.361 mm |
| raw root delta max | — | 0.2499 rad |
| raw max pose-joint delta max | — | 0.1018 rad |
| active-set added counts | — | `[381,109,37,11,6,3]`，closed=false |

右手的 median 变为 0 不能掩盖 p90/p95 和 max 仍很大；这更像 observed-face support 被部分移开或变成无 support，而不是整个 MANO/object 几何冲突被解决。

两侧其它重要结果：

- left `visible_joint_shift_max_px` p90≈12.003、max≈12.075；right median≈12.420、max≈15.749；12 px 是 soft hinge threshold，不是 hard bound；
- left depth shift max 的 p90≈20.021 mm、max≈30.623 mm；right p90≈57.265 mm、max≈103.406 mm，明显超过 35 mm soft threshold；
- `object_translation_delta_norm_m` 两侧全为0，因为 object translation optimization 关闭；
- contact patch active rows：left 143、right 138；有 3D patch target 的 rows 只有 left96、right77；
- contact posterior 与 prior 基本没有变化：left prior/posterior median 0.9/0.89999998，right 0.85/0.8499986；geometry likelihood 关闭；
- contact final absolute normal gap median：left 28.499 mm、right 33.619 mm，但对应 deadband 是约54–69 mm，且没有 tangent residual；P18 记录的初始 Euclidean patch distance median 已是 left 68.437 mm、right 84.601 mm，不能把 solver 后的 normal-gap summary 当成“真实接触距离下降”；
- object-frame contact centroid dispersion p95：left101.649 mm、right251.513 mm，均超过各自 support uncertainty p95，不能支持 persistent anchor；
- visible ownership mask 虽有 non-object-owned pixels，但 P18 报告的 `visible_ownership_quarantined_face_count` 两侧全为0，故该 hard-surface quarantine 在本次没有实际删除 face force。

### 11.6 P18 output translation gate 和报告语义

CLI 开启：

```text
--gate-translation-with-visible-surface-support
--translation-gate-min-visible-surface-depth-vertices 0
```

但 selected visible-surface depth-order support 是 0，且 `freeze_translation_without_visible_surface_support=false`。因此：

- solver 内部仍让全局 `trans_delta` 参与优化；
- 每一行 output gate 都把优化后的 wrist 平移回 source HaWoR wrist；
- 两侧 `output_translation_gate_applied_count=150`；
- 相对 wrist articulation/root orientation 的 raw hypothesis 可以保留，但 global translation 不被接受。

还有一个报告语义问题：P18 在计算 `full_observed_surface_penetration_after_solver_m`、2D/depth shift 和 correction counts 时使用的是 `hyp_vertices`，即 gate 前的 raw optimizer output；它没有在 output gate 后再次用最终保存的 `state_vertices_world` 重测。因此这些字段不能直接当作 canonical saved state 的 post-gate penetration metric。

---

## 12. P18b：为什么 raw P18 state 没有成为最终 metric MANO

`build_v19_mano_surface_hypothesis_state.py` 不是另一个 optimizer，而是一个 **state split / acceptance gate**：

1. 从 HaWoR NPZ 读取每个 `(frame,side)` 的 metric 21 joints；
2. 将 P18 的 `optimized_joints_world_m` 保存为 `surface_fit_joints_world_m`，仅作诊断；
3. 将 `optimized_joints_world_m` 覆盖回 HaWoR source joints；
4. 保留 P18 的 `optimized_vertices_world_sample_m` 为 `contact_surface_vertices_world_sample_m`；
5. 明确写入 `uncertain_visible_surface_proximity_not_contact_ownership`。

当前 canonical P18b：

- rows：300，left/right 各150；
- `joint_state_policy=hawor_npz_metric_mano_preserved`：300/300；
- metric joint shift：0；
- 每行仍携带 160 个 P18 surface sample，但它们是 uncertain hypothesis；
- contact distance/normal/tangent 的 accepted summary 均为空（count=0）；
- claim scope 明确“不接受 contact ownership 或 nonpenetration”。

所以“P18 optimizer 运行过”与“P18 优化结果被最终 annotation 接受”是两件不同的事。当前最终 renderer 使用的是 P18b metric-preserved joints。

P19c 的 presentation rerender 是 render-only 分支，不重新运行任何 inference/optimizer，也不修改 prediction state；本次 `P19c` 没有运行，canonical artifact 使用 P19b diagnostic branch。

---

## 13. 按“优化问题—建模方式—目标函数—优化结果”的总汇总

下表把整个当前 V19 runtime 中可审计的优化/选择问题集中列出。表中“优化结果”只写报告实际支持的结果；没有 scalar loss 的地方明确写“未记录”。

| 优化问题 | 建模方式 | 目标函数 | 优化结果 |
|---|---|---|---|
| P03b 视频级内参估计 | constant pinhole；每个 K 分量从 150 个 UniDepth estimates 聚合，再施加 square focal | 每分量 `argmin Σ|k_t-k|`；随后 `fx=fy=sqrt(fx fy)` | `K=[537.0213,537.0213,722.2881,720.8964]`；这是 robust hypothesis，不是 GT calibration |
| P06 文本框选择 | OWLv2 text-grounded candidate set；每个 prompt frame 选一个 box | 离散 `argmax detector score`，阈值 0.03 | 11/11 帧有 box；仅是 SAM2 prompt |
| P09 anchor proposal | 对 mask area、depth support、hand removal、border、extent、component、depth stability 加权 | `S=0.24A+0.16P+0.18C+0.16B+0.14E+0.07G+0.05D` | rank1 frame109 后仍由 agent 视觉选择；不是自动物理接受 |
| P09 anchor visible surface | mask/depth lifting；voxel downsample；Poisson surface | library Poisson implicit reconstruction；wrapper 未记录可审计 scalar energy | frame109，1,421 downsampled points，2,527/4,882 visible mesh |
| P13 TRELLIS 初始对齐 | PCA axis/sign/permutation finite candidates + global Sim(3) | candidate `median(d_obs→model)+0.25 p90(d_obs→model)` | scale0.40529；initial median3.0616mm |
| P13 Sim(3) ICP | 固定 NN correspondence 后 Umeyama 闭式求 `s,R,t`，共6轮 | `min Σ||sRx_c+t-y||²` | final obs→model median2.9917mm、p95 9.9582mm |
| P13 hidden geometry acceptance | silhouette projection 和 planar support slab | binary keep/reject；无 soft loss | 91,757 silhouette faces rejected；planar filter kept all；最终 accepted body 22,758 faces |
| P14 每帧 object pose | canonical mesh 6,000 surface samples；observed point→nearest model point；独立 SE(3) | `min Σ||R x_c+t-y||²`，Kabsch/Umeyama，4 iterations | 150/150 fit；median 7.0451→3.8354mm |
| P15 correction-field pose graph | 每帧 `Δr,Δt`；pose prior、optional pressure、一阶/二阶 correction smoothness | sparse least squares + `soft_l1`；surface 只作 post-check | 本次 target=0，`nfev=1,cost=0`；所有 correction=0，无 nonpenetration gain |
| P16 hand escape | 穿透点局部 outward normals；rigid translation halfspaces | `min 1/2||t||² s.t. Nt≥d`，SLSQP；再做 bound/2D gate | sign mesh nonwatertight；solver 未有效触发；candidate correction=0，结果 unresolved |
| P17 ownership | projected MANO support 与 object mask 的 overlap/near band；agent 控制 quarantine | 没有 optimizer；image prior 和 agent prior 的启发式组合 | 286 rows；ownership/contact 仍是 uncertainty factor |
| P18 MANO state | 每 side 150 帧；root translation/orientation/articulation；MANO differentiable replay | 见第11节的 `L_prior+L_temporal+L_surface+L_2D+L_depth+L_contact`；CUDA LBFGS | left 局部下降且 active set closed；right 残差大且未 closed；scalar loss 未记录 |
| P18 contact switch | `C_t=sigmoid(logit_t)`；prior、temporal continuity、可选 geometry likelihood | `w(0.01)^2(C-p)^2` + temporal；geometry term 本次关闭 | posterior≈prior；没有证据更新出独立 contact state |
| P18 output translation gate | 没有 visible depth-order support 时保持 source wrist | 不是 loss；对 optimizer output 做 re-anchoring | 300/300 rows gate applied；global translation candidate 不被信任 |
| P18b canonical acceptance | source HaWoR joints 与 P18 surface samples 分离 | 不是 optimizer；state policy gate | 300 rows 保留 metric MANO；surface samples 仅 uncertain proximity hypothesis |
| P19 rasterization | completed mesh + accepted pose + MANO state 投影/深度排序 | z-buffer/min-depth pixel selection，不是物理 state loss | 150 帧 body pixels 全部存在；render 通过 visual sanity |

---

## 14. 当前运行中“声明了但没有实际激活”的 loss/factor

为了避免从 argparse 默认值误读实际结果，以下机制在代码中存在，但本次 run report 明确显示未使用或没有有效 support：

| 机制 | 代码默认/可选行为 | 本次状态 |
|---|---|---|
| P15 nonpenetration target residual | 由 constraint report 提供 bounded `Δt` target | P15 command 未传 report；0 rows |
| P18 object translation | 每帧 `o_t` + prior + smoothness | `false`，delta 全0 |
| P18 hand-ray shift prior | 约束 translation 到 hand-depth repair vector | 无 graph，active count0 |
| P18 visibility-weighted joint observation | 根据 depth 前后/遮挡降低 joint/pose prior | `false`，所有 weights=1 |
| P18 hand-owned object-depth quarantine | hand 可能在 object first surface 前方时删 face | `false` |
| P18 surface-eligibility factor | 用 face-state NPZ intersect/replace trusted faces | 未提供 |
| P18 visible-surface-track factor | 提供 960 mask + metric first-surface depth | 未提供 |
| P18 visible depth-order residual | 命令开启 | selected vertices=0；没有有效 residual |
| P18 contact geometry likelihood | 用 geometry target 更新 `C_t` | `false` |
| P18 in-solver translation freeze | 无 visible support 时优化中冻结 translation | `false`；只做 output gate |
| P18 persistent contact pose anchor | 将 local patch 升格成 stable object-frame `A_t` | `require=false`，且每行 blocker 表示无 anchor |
| P16 exact signed repair | watertight sign mesh 上运行 | mesh 非 watertight，未形成可信 signed solve |

---

## 15. 非 loss 的硬门控、报告和最终接受规则

这些规则同样会影响最终输出，但不应被误写成 loss：

1. **P09 mask ownership**：hand bbox 区域从 object visible support 删除；
2. **P09 extent eligibility**：explicit false 在 `b7b97e6` 中是 P14/P15 默认 hard rejection；冻结运行仍保留当时未接线的历史结果；
3. **P13 observed-band overwrite**：TRELLIS 近 observed surfels 的 face 不作为 hidden completion；
4. **P13 silhouette free-space**：投影到 object-owned silhouette 外的 hidden face 丢弃；
5. **P13 planar slab**：仅在 PCA ratio≤0.04 时生效；
6. **P15 surface-preservation acceptance**：aggregate observed→mesh median degradation 不得超过 3 mm，当前 degradation=0；这只是 post-check，不是 residual；
7. **P16 watertight gate**：没有 watertight sign mesh 时只能输出 uncertainty；
8. **P17 agent occlusion quarantine**：决定哪些 object pixels 不能作为 hard visible surface；
9. **P18 active set**：重新发现 violation 后增加 tangent constraints；不是把所有 constraints 一次性写成 exact equality；
10. **P18 output translation gate**：没有 visible support 时恢复 source wrist；不重新求解 objective；
11. **P18b metric-state split**：不允许 geometry-derived surface hypothesis 覆盖 metric MANO joints/root；
12. **P19/P21 visual audit**：检查 render 中的 body 是否跟随 keyboard silhouette，并保留 `INTERNAL MANO UNCERTAIN`/`contact uncertain`。

---

## 16. 当前实现中最值得优先修复的 loss/数据流问题

### 16.1 修复 P18 的 1408↔960 mask projection contract

这是最直接的实现问题：P17 已缩放 projection，P18 的 `project_world`/`mask_membership` 没有缩放。建议：

- 在 factor row 中保存 `mask_width/mask_height` 和 source size；
- 所有 P18 face/sample/joint projection 统一调用一个带 `source_size→mask_size` 的 projection helper；
- depth lookup、mask lookup、ownership face support 使用同一 grid；
- 对修复前后的 selected vertex count、quarantined face count 做 before/after report。

在此修复前，P18 的 visible depth-order 和 ownership quarantine loss 不应被当作有效 metric evidence。

### 16.2 P14 eligibility hard gate 与 P15 support gate（已实现，仍缺观测）

实现提交 `b7b97e6` 中的 `fit_v18_compact_rigid_object_pose.py` 已明确：

- explicit `rigid_pose_observation_eligible=false` 默认拒绝；
- missing field 仅作为单独计数的 legacy compatibility；
- historical override 必须显式传 `--include-ineligible-rigid-pose-observations`；
- P15 对 override row 再做一次默认 hard rejection。

真实 tire-lever ablation 从 33 个冻结 fits 变为 6 个 trusted fits；P15 默认 minimum 为 8，因此 6 direct +144 interpolation/nearest-hold rows必须输出：

```text
status = completed_uncertain_insufficient_trusted_pose_graph_support
annotation_ready = false
graph_support.sufficient = false
```

P16 将 object-relative corrections quarantine；P18 默认跳过 object-relative optimizer并生成 source-only hand state；P19 只把 object trajectory 画成橙色 unresolved hypothesis。剩余问题不是继续修改 eligibility gate，而是取得跨时间轴的更多可信 object measurements。

### 16.3 重新设计 P15/P16 的 phase wiring

若要让 nonpenetration candidate 影响 object pose：

- 先构造 P16 measurement，再运行带 `--constraint-report` 的 P15；或
- 保留当前顺序但增加第二个明确命名的 pose correction pass。

同时必须先获得 watertight 或有明确局部 sign validity 的 geometry；否则把 P16 candidate 接入 P15 只会传递错误的 sign hypothesis。

### 16.4 给 P18 记录真正的 objective diagnostics

每个 side/active-set pass 至少应保存：

- `loss_total_before/after`；
- 每个 loss family 的 contribution；
- LBFGS closure count、gradient norm、termination reason；
- active constraint count 和新增 constraint 的 residual distribution；
- output gate 前后重新测量的 full observed penetration、2D shift、depth shift；
- canonical accepted state 的同一组 metrics。

当前只有 residual summaries，没有 scalar loss 曲线，无法做严格的 objective-level ablation。

### 16.5 软 hinge 不能替代 trust region/hard feasibility

P18 的 `max_translation=0.045 m`、`visible_shift=12 px`、`depth_shift=0.035 m` 都是 squared hinge。右手结果已显示 translation 140 mm、depth shift 103 mm 仍可输出。若这些是物理可信域，应使用：

- explicit optimizer bounds；
- trust-region constrained solver；或
- solve 后硬拒绝/回退，并记录 rejected state。

### 16.6 contact patch 应先解决 observability，再调权重

当前 contact patch 有：

- 54–69 mm deadband；
- 只约束 normal，不约束 tangent；
- 没有 stable object-frame anchor；
- P17 prior 主要来自 agent semantic judgment；
- P18 posterior 没有从 prior 移动。

因此下一步不是简单增大 `contact_patch_weight`，而是先获得独立的 hand/object depth ordering、局部可信 surface、稳定 object-frame patch transport 和 watertight/local-sign support。否则 optimizer 只会把不可观测的全局 hand translation 或 articulation 吸收掉。

### 16.7 P13/P14 需要更稳健的 partial-view objective

当前 P13/P14 的 observed-to-model NN objective 会受：

- partial view；
- depth slope；
- hand leakage；
- hidden completion overshoot；
- sample density不均；

影响。可考虑 point-to-plane、trimmed bidirectional Chamfer、visibility-aware weights 或多帧 joint canonical refinement，但每个新增项都必须保留 source/coordinate/uncertainty，并用 render/QC 验证，不应直接把 hidden prior 当 observation。

---

## 17. 结果文件和审计读取顺序

若要复核本文的每个 objective/result，建议按以下顺序读取：

1. `$RUN/input/raw_frame_manifest/manifest.json`
2. `$RUN/measurements/depth_slam/unidepth_full_frame/qc_unidepth_full_frame_v3.json`
3. `$RUN/state/calibration/v19_camera_calibration_contract.json`
4. `$RUN/measurements/hand_candidates/hawor_world/qc_hawor_world_hands.json`
5. `$RUN/measurements/object_geometry/anchor_candidates/keyboard/anchor_candidate_proposals.json`
6. `$RUN/state/anchor_decisions/keyboard.json`
7. `$RUN/measurements/object_geometry/visible_geometry/keyboard/v19_visible_geometry_adapter_report.json`
8. `$RUN/measurements/geometry_completion/trellis_keyboard_seed42/qc_trellis_shape_v3.json`
9. `$RUN/measurements/geometry_completion/compact_keyboard_seed42/v18_compact_rigid_trellis_completion_report.json`
10. `$RUN/measurements/pose_fits/keyboard_visible_pose_fit/v18_compact_rigid_object_pose_fit_report.json`
11. `$RUN/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
12. `$RUN/measurements/contact_nonpenetration/keyboard_mano_object_constraint/v18_mano_object_constraint_state.json`
13. `$RUN/measurements/contact_visibility_factors/keyboard_0_149/$CASE/v19_visible_contact_ownership_factor_report.json`
14. `$RUN/measurements/mano_interval_correction/keyboard_0_149/$CASE/v18_joint_mano_interval_trajectory_state.json`
15. `$RUN/measurements/mano_interval_correction/keyboard_0_149_surface_hypothesis_metric_mano/$CASE/v18_joint_mano_interval_trajectory_state.json`
16. `$RUN/state/render_state/keyboard_rigid_render_state.json`
17. `$RUN/logs/harness_events.jsonl`、`$RUN/state/v19_agent_evidence.md`、`$RUN/FINAL_RUNTIME_AUDIT.json`

推荐先读 P18 raw report 的 `parameters` 和 `intervals`，再读 P18b；不要只读最终 JSON 的 `status` 字段。

---

## 18. 最终物理解释边界

当前 V19 run 可以支持的声明是：

- 有一个由 OWLv2→SAM2 identity-preserving evidence 支持的 keyboard object track；
- 有一个由 UniDepth/HaWoR 相机桥接形成的 metric hypothesis；
- 有一个经过 visible depth alignment、silhouette/free-space 和 planar-support filtering 的 coarse completed keyboard Mesh；
- 有 150 帧 completed-mesh rigid pose rows，P14 提供实际逐帧 fit，P15 没有额外非零 correction；
- 有 agent contact/occlusion priors、projected ownership masks 和 MANO/object distance/penetration diagnostics；
- 有 full-duration face-rasterized render，且 render 中 completed rigid body 沿 keyboard silhouette 保持。

当前不能支持的声明是：

- exact CAD geometry 或确定的 hidden keyboard thickness；
- metric MANO 与 keyboard 的已证明接触；
- signed nonpenetration 已证明；
- P18 raw optimized joints 是经过 metric evaluation 验证的最终手状态；
- P15 已从 nonpenetration pressure 学到 pose correction；
- P17 JSON 的 `likely_contact` 等于 contact ground truth。

因此当前 pipeline 的准确摘要是：

> **一个可渲染、可追溯的 coarse rigid keyboard mesh + full-timeline pose hypothesis，叠加 source-preserved metric MANO 和显式 unresolved contact/nonpenetration evidence；而不是已闭合的手物理接触优化结果。**

与观测字段/目录的更完整说明见：

- [`docs/ego_annotation_observations_and_directories.md`](ego_annotation_observations_and_directories.md)
- [`experiments/toy_optimization_lab/README.md`](../experiments/toy_optimization_lab/README.md)：P13/P14/P15/P16/P18 和联合 gauge 的低维可视化 toy，以及 RL 边界
- [`runtime/v19_runtime_spec.md`](../runtime/v19_runtime_spec.md)

---

## 19. Ego-Exo4D partial-GT 对 optimizer 的外部检验

完整材料：

- [`experiments/egoexo4d_rigid_benchmark/README.md`](../experiments/egoexo4d_rigid_benchmark/README.md)
- [`experiments/egoexo4d_rigid_benchmark/RESULTS_V19_V1_ZH.md`](../experiments/egoexo4d_rigid_benchmark/RESULTS_V19_V1_ZH.md)

新运行对象为 validation split 中的黄色刚性塑料撬胎棒，150 帧、30 fps。预测侧只消费 raw RGB 和文字 target hint；GT hand、Mask 和 camera extrinsics 在运行后才进入 evaluator。

### 19.1 先修正 camera contract

Ego-pose K/UV 属于官方 512×512 undistorted linear camera，不属于 raw 448/960 RGB。评测使用官方 `aria_original_to_extracted` 对应的固定 90° camera-axis adapter；该 adapter 不从 GT 拟合。

旧的 raw-grid hand error 已标记 invalid。没有 no-image-stream VRS calibration 时，不能用 rectified K 覆盖 runtime raw RGB calibration，也不能报告 raw RGB hand reprojection。

### 19.2 P13/P14：局部 objective 可以下降，但错误 rows 主导总体

P13 在单个 anchor partial surface 上得到：

```text
scale = 0.0898406
observed→TRELLIS median = 3.164 mm → 2.988 mm
```

但这只是内部 anchor fit。P13 的 silhouette 和 planar-slab gate 都拒绝 0 faces，最终 accepted body 中：

```text
observed-depth faces:       496
TRELLIS hidden-prior faces: 151335
```

P09/P14 eligibility audit：

```text
metric rows:       33
eligible rows:      6
ineligible rows:   27
P14 fit rows:      33
```

按 P09 eligibility 分组：

| group | initial observed→mesh median-of-medians | final |
|---|---:|---:|
| 6 eligible rows | 3.956 mm | 2.006 mm |
| 27 ineligible rows | 51.783 mm | 63.425 mm |
| all 33 rows | 47.745 mm | 63.038 mm |

这直接说明：优化器在可信局部 support 上有效，但 pipeline 没有把 hard eligibility 送到 P14，导致 aggregate objective/physical state 被污染 rows 主导。增加更复杂 loss 之前，必须先修 correctness wiring。

### 19.3 P15：依旧是 inert problem

```text
graph rows: 33
nonpenetration targets: 0
nfev: 1
cost: 0
all corrections: 0
```

117 个缺失 pose 被插值/nearest hold 填满并不等于优化获得了额外物理信息。

### 19.4 P16/P17/P18：contact factor 没有成为可信物理约束

P16：

```text
completed Mesh watertight = false
sign Mesh watertight = false
candidate correction count = 0
```

P17 产生 12 rows，其中 6 个 right-hand rows 是 `likely_contact`。但 P18 实际：

```text
contact_patch_weight = 0.0
stable contact-anchor residual rows = 0
visible-depth-order selected vertices = 0
visible-surface track factor = false
visibility-weighted hand observation = false
surface-eligibility factor = false
```

Right active contact rows 的 posterior median仍为 `0.88`，与 prior `0.88` 几乎完全相同；final normal gap median `14.13 mm`，support uncertainty `65 mm`，object-frame contact centroid p95 dispersion `68.43 mm`。

因此不能把它解释为 metric contact solve。

### 19.5 P18 external hand GT

916 个 filtered named joints：

| state | absolute MPJPE | root-relative MPJPE | wrist error |
|---|---:|---:|---:|
| HaWoR | 192.956 mm | 35.559 mm | 186.295 mm |
| P18 raw | 192.440 mm | 35.234 mm | 186.295 mm |
| P18b | 192.956 mm | 35.559 mm | 186.295 mm |

P18 raw 只改善 `0.516 mm / 0.325 mm`，wrist 完全不变；P18b 正确回退到 source-preserved metric MANO。

Left raw optimizer 曾产生最大 `315.646 mm` translation correction，right 最大 `33.948 mm`，但两侧 150/150 rows 都经过 output translation gate。该现象再次说明 soft hinge/solve 后 gate 不能替代可观测的 bounded trust-region problem。

### 19.6 Camera external GT

```text
SE3 center ATE RMSE:       58.588 mm
Sim3 diagnostic scale:      1.1802
1-frame translation RPE:    7.055 mm
1-frame rotation RPE:       0.345 deg
30-frame translation RPE: 130.770 mm
30-frame rotation RPE:      6.591 deg
```

短时 rotation continuity 尚可，但 1 秒尺度有显著长尾 drift。position-fit 与 orientation-fit world gauge 不能同时获得小误差，也说明联合 MAP 必须显式固定 camera gauge，并区分 absolute 与 relative acceptance。

### 19.7 对联合优化与 RL 的含义

本次 independent GT 没有支持“一次性放开 camera/object/MANO/contact”的做法。优先顺序仍应是：

1. 修 P09 eligibility、raw/rectified camera contract 和 factor wiring；
2. 用 hand/camera/Mask GT 建立每个 block 的 accept/reject baseline；
3. 固定 camera gauge，只允许可信 block 做局部 joint refinement；
4. 无 depth-order/stable anchor 时冻结 hand/object translation；
5. gate 后必须重测外部指标并 rollback regression；
6. RL 若使用，只控制 block scheduling、证据请求、trust region 和 rollback，不直接输出完整连续物理状态。

新运行的结论与 keyboard `_v2` 审计一致：当前 V19 是有价值的 measurement/proposal/uncertainty pipeline，但不是普遍高精度、强鲁棒的手物物理 GT 系统。

### 19.8 Runtime invariant 也失败

Harness event 的 P00→P21：

```text
input duration = 5.0 s
runtime = 3652 s
ratio = 730.4× realtime
P17→P18 = 545 s
P19a→P19b = 1271 s
```

完整 render 和失败机制确实被保留，但该运行不符合 V18+ default runtime 应与输入时长同一数量级的要求。优化方向应是同机制下的 profiling、batching、vectorization 和 renderer 加速；不能通过删掉 full timeline、只渲染 selected frames 或跳过物理 block 来制造速度数字。
