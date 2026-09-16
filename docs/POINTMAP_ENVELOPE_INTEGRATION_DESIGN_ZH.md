# Pointmap-only SAM3D 与可见表面空间包络的整合设计

日期：2026-09-16。分析基于开发源码 `0835af9`（最后实际 fullrun 源码 `9ef06b3`）。

**状态：用户已选择 pointmap-only 方向；本文是代码核查后的整合方案与有界回放结果，不是已接入默认生产链的实现。**
本轮没有修改 `main.py`、模型 worker、factor 或 solver；没有新模型推理、全视频优化或重跑旧预测。

## 1. 决策摘要

1. SAM3D 条件输入采用当前共享 metric camera-z/K 反投影得到的 **pointmap only**。
   不增加独立 exact-K 条件分支，不开启原来关闭的 layout postoptimization。
   K 仍用于点图构造、投影验证和下游渲染；不能不管其来源，或把 provider 的异常推导 K 当有效相机。
2. 用户提及的历史“框”已定位：它是**观测表面定向的粗空间包络**，不是检测框、完整重建物体、真实接触体积或标定置信区间。
3. 推荐整合旧模块的 **可见三角第一交点 + 轮廓 + 全时间轴 SE(3) + 一个共享尺度**，
   包络只作为命名 profile 中的弱空间先验和诊断。不能把全部收益归于框。
4. 主接缝应在 **最终 appearance mesh 的 `shape_binding` 之后**，不是只对第一轮 native mesh 做 anchor 修补。
   必须同时处理后续 `visual_fit` 的 object_delta，否则前面的表面/框约束可能被撤销。
5. 先保留 `surface_no_box` 和 `surface_weak_box` 两个相同求解器的对照。
   旧权重0.15可作为历史复现设置，**不是已验证最优值或必须默认启用的依据**。

## 2. 历史模块在哪里

这些实现位于本机实验目录，未被整理分支迁入 active grouped source。
不是 `migration-map.json` 已保证数学等价的现有模块，也不应恢复旧 runpy/script dispatcher。

| 原文件/函数 | 实际机制 | 拟议新 owner |
|---|---|---|
| `milk_surface_socket_20260914/probe.py::{local_normals,extract_planes,plane_labels}` | 自由角度平面和法向支持；不按类别套盒 | `observations/object/envelope.py` 的 surface-frame builder |
| `hot3d_envelope_ablation_20260915/stage_cases.py::descriptive_frame` | 主可见平面法向 + 最长切线方向；明确 PCA fallback | 同上 |
| `milk_surface_socket_20260914/build_spatial_envelopes.py::direct_bounds` | 1%/99%观测分位边界 + 缺失范围/隐藏厚度先验 | 同上 |
| 同文件 `outside_residual_m` | 表面坐标下框外 residual | `factors/object/envelope.py` |
| 同文件 `conditional_translation_interval` | 固定 R 时完整 mesh 顶点的平移可行区间 | `optimization/common/correspondences.py` 或 object 诊断工具 |
| `object_only_fit.py::Problem.raycast/correspondences` | 完整三角 BVH first-hit；每轮外循环刷新 | `optimization/object/surface_fit.py` |
| `hot3d_envelope_ablation_20260915/solver_kernel.py::fit_problem` | 多因子对象单独优化；box_weight消融 | 同上，命名问题显式 dispatch |
| `hot3d_envelope_ablation_20260915/run_arm.py::CaseProblem` | 修复像素中心变换、显式train-mask、同输入分臂 | 新的 typed observation loader/测试夹具，不复制历史路径 |

绝对历史目录：

```text
/mnt/user-home/kupingxin/ego_annotation_experiments/milk_surface_socket_20260914/
/mnt/user-home/kupingxin/ego_annotation_experiments/hot3d_envelope_ablation_20260915/
```

相关历史文档为 `SPATIAL_RANGE_DESIGN.md`、`OBJECT_ONLY_FIT_DESIGN.md`、
跨例 `DESIGN.md`、`README_ZH.md` 和 `gt_evaluation/README_ZH.md`。

## 3. “生成框限制位姿”的确切含义

### 3.1 表面 frame S，不是物体 canonical frame O

历史代码从直接可见点云中估计主平面；Z轴朝向相机，Y轴为主切向长轴，X轴补成右手坐标系。
无稳定平面时使用显式标记的全点云 PCA 描述 frame；不是证明物体有该平面。
`probe.py::extract_planes` 的历史设置包括：局部24邻点法向、3mm支持带、25°法向容差、
最多3个自由角度平面、至少80个支持点。当前 probe 原样使用，不作为通用最优阈值。

存储的 `T_world_surface = T_world_camera @ T_camera_surface` 是 **surface → world**。
不要把它误读为 world→surface，也不要把 S 的轴当作确定物体朝向。
世界点的局部坐标为（列向量公式）：

```text
q_S = R_WS^(-1) (p_W - t_WS)
```

历史 row-vector 实现为 `(p_W - t_WS) @ inv(R_WS).T`。
它保留源变换微小的数值非正交性；新合同应验证变换，不偷偷补偿数据或强制物理轴对齐。

### 3.2 直接框由观测和固定形状尺寸先验构造

`direct_bounds` 不接收逐帧候选 object R/t：

- 局部点云每轴1%/99%分位；
- 固定生成 mesh 的 PCA 长/中/短尺寸只作为一次性 shape-size prior；
- X/Y名义长度取 `[mid,long]`，每侧可能缺失长度为 `max(0,nominal-observed_span)`；
- 每侧再加 `8mm + 10% nominal_size` 的启发式余量；
- 向后（-Z）至少保留可见深度范围，并以 `1.1 × mid` 提供隐藏厚度 prior；
- 向前（+Z）以观测上分位 +8mm限制。

框的中心/朝向/可见范围来自观测，但隐藏厚度和缺失extent来自生成形状。
所以它**不是纯独立测量**；同一深度同时用于 shared pointmap、表面loss和框，不得作为三份独立证据反复抬高置信度。

优化前必须冻结 `S/lower/upper` 和 size prior。不能随着候选 mesh scale/R/t 更新框，使框永远包住候选。
也不能把部分可见点的薄 AABB 当作完整物体上界，强行把隐藏面压扁。

### 3.3 缺测帧不产生新的框测量

历史缺测帧：

```text
S_i = T_i_reference @ inverse(T_donor_reference) @ S_donor
extra_padding = 6mm + 3mm × |i-donor|
```

这种传播依赖旧 object pose，属于 temporal/display hypothesis，不是新观测。
**历史 solver 的缺测框损失权重为零**；迁移时必须保持，以避免循环自证。
直接点云存在但平面退化应携带不确定性，不能和“完全没观测”混写。

### 3.4 框是软 residual，不是硬接受门槛

对世界中的 candidate complete-mesh surface sample 转到 S frame：

```text
r = min(q_S - lower, 0) + max(q_S - upper, 0)
L_box = mean rho(r / 0.015m), rho(x)=2(sqrt(1+x²)-1)
```

历史 solver 对每个direct frame的实际mesh面积样本和三个分量取均值，再按direct帧数平均。
符号或绝对值写法在平方对称robust核里等价；归一化必须保持可测试。
框内为零，**框内没有吸引点、没有接触法向、没有真实表面对应**。

固定候选旋转时，可用全部顶点精确计算 conditional translation interval：

```text
a_k = inv(R_WS) @ (R_candidate @ v_k)
t_low  = lower - min_k(a_k)
t_high = upper - max_k(a_k)
```

它约束的是当前 canonical 原点的 translation，不一定是物体几何中心。
区间非空只说明该 R/shape 可装入这个**近似**框；不能据此证明正确姿态。
有共享尺度时必须使用实际 `pivot+s*(v-pivot)` 后的全顶点，而不是忘记pivot项。
可以作为诊断或有界初始化候选，不能硬剪切 pose 后称因子已优化。

## 4. 历史拟合不只是一个框loss

旧 object-only solver 变量：

- 全时间轴逐帧左SO(3)旋转增量及世界平移；
- **一个**全视频共享 log-scale，绕固定canonical顶点均值pivot；
- 固定完整mesh拓扑、camera、MANO，不做vertex morphing或逐帧scale。

旧协议：12 outer ×25 Adam inner；前2轮冻结rotation，先拟合translation/scale。
每个outer刷新的是 **first-hit和轮廓对应**，不是重新生成包络。
source ray的z分量为1，逆相似变换后保留ray参数，因此`RaycastingScene.t_hit`仍是camera-z；miss不造深度目标。

历史总目标：

```text
L = L_depth + 0.35 L_ray + L_silhouette + λ_box L_box
  + 0.05 L_pose_prior + 0.15 L_scale_prior
  + 0.30 L_correction_velocity + 0.10 L_actual_motion_acceleration
```

- `L_depth`：完整三角第一交点camera-z，sigma8mm；
- `L_ray/L_silhouette`：参考960宽平面的8px尺度，双向凸轮廓只作图像因子；最终评价实际全三角轮廓；
- `λ_box`：无框0，有框0.15；box sigma15mm；
- 192固定训练P09点，最多512个不同索引评估点；同传感源，非独立GT；
- scale范围 `[0.8,1.35]`；rotation/worldtranslation各有component clamp；
- 旧temporal按frame/row顺序差分，**当前timestamp-based管线不能默默照搬为任意fps等价**。

还必须修复可移植性：旧 `Problem.__init__`、`CaseProblem` 含绝对路径、150-frame假设、
历史state/RRD输入和自带入口；底层kernel可迁移，旧运行脚本不应直接import进生产。
原milk版本Kfit只比例缩放；跨例 `run_arm.CaseProblem` 才采用正确的half-pixel主点变换。
迁移应以明确像素变换合同为准，不回退到旧bug。

## 5. 已有证据不能略过

### 5.1 三案例“有框−无框”增量很小

已核对 `hot3d_envelope_ablation_20260915/summary.json` 的同输入/参数配对结果：

| 案例 | 配对深度MAE均值变化（有框−无框） | 平移差中位数 | 旋转差中位数 |
|---|---:|---:|---:|
| BBQ瓶 | -0.00484mm | 0.0105mm | 0.0206° |
| 汤罐 | -0.00048mm | 0.0162mm | 0.0228° |
| milk | +0.00015mm | 0.0159mm | 0.0205° |

负深度变化有利于框，但这些差别极小。不能把“表面/轮廓优化后变好”当成框的因果收益。
不要为了让框显得有效事后加强权重或逐例缩小框。

### 5.2 后验GT评估发现观测拟合可使绝对几何变差

历史 **有框整套优化**相对原v7最终的相机系表面面积中心平均误差：
BBQ55.45→57.01mm、汤罐39.44→62.93mm、milk36.04→78.24mm。
这些不是标准GT原点translation error，含形状中心差异；也不是box-only因果归因。
报告同时发现P09相对GT可见深度偏深约55.5/51.7/73.0mm，milk原预测K约28%偏差。

正确含义：**降低共享深度残差不能自动证明真实metric精度提升**。
上述是历史事后评估证据，只能用来限定claim、提醒观测偏差；不能拿这些GT值给新点图/框调尺度或补偿。
新的采集/源码不同，不能直接声称当时误差等同当前误差。

## 6. 本轮在当前anchor上的原函数回放

没有新推理和优化。使用当前共享pointmap三臂实验的anchor48 surfels/mesh，
调用历史原函数（带源码SHA），一个冻结框用于全部A/B/C。
size prior只取shared-pointmap B mesh的固定PCA尺寸，**未随任何候选位姿重建**。

- 自由角度平面：2个；主平面834点；切线长短比2.56；
- 固定size prior约 `[257.15,139.70,89.56] mm`；
- 包络在S轴的尺寸约 `[225.76,373.64,175.92] mm`；
- 当前观测点100%在框内。

| 候选 | 框内面积样本比例 | 全顶点框内比例 | 原box损失（未乘λ） |
|---|---:|---:|---:|
| A 内部MoGe mesh | 71.535% | 72.090% | 0.033631 |
| B shared pointmap mesh | 100% | 100% | 0 |
| C shared pointmap+exactK实验mesh | 100% | 100% | 0 |

把B人为沿S各轴平移±100mm，boxloss约0.087–1.011，说明它能响应明显跑偏。
但**当前B已在框内，框没有非零修正梯度**；此anchor上不能期待仅加框进一步改善pose。
不表示其它帧或坏初始化一定无用，也不是框约束优化成功的试验。
当前新管线仍未产出完整 refined pose/appearance binding；这个anchor回放不能证明
多帧solver已接通，不能直接复用旧v7轨迹冒充当前候选。

证据：

```text
/mnt/user-home/kupingxin/ego_annotation_runtime/rlopt_envelope_integration_20260916/
  PROBE_PLAN.md
  probe_current_anchor.py
  probe/report.json
  probe/box_and_observations.npz
  probe/envelope_scatter_diagnostic.png
```

图为明确标记的mesh顶点子集散点QC，不是全三角渲染或最终标注。
本次CPU数值计算约2.53s，没有执行历史多帧solver。
另用历史原函数跑了5项CPU数学合同检查：surface/world逆变换与residual符号、
固定R平移区间包含全部顶点、空区间不伪造、direct框构造不受候选评估影响、
框内没有吸引梯度，均通过（`test_historical_contracts.py` / `historical_contract_tests.log`）。

## 7. 推荐的生产整合位置（拟议）

```text
raw → camera/depth/MANO/mask/ownership → retained metric observations
                              │
                              ├→ metric-anchor pointmap → SAM3D metric_native
                              │                         → original metric bridge/quality
                              └→ observed/RGB pose → refined_pose

appearance frame decision → corresponding-frame shared pointmap
                          → SAM3D appearance_native
                          → canonical shape binding (this profile: no separate image-only scale search)
                          → frozen surface frames + envelope artifact
                          → ObjectSurfaceEnvelope candidate solve
                          → candidate pose review
                          → object-frozen visual hand fit (initial comparison profile)
                          → final actual mesh/pose/MANO render + inspection
```

### 7.1 Pointmap-only必须覆盖两次native调用

现代码 `observations/object/sam3d.py::run_one_case` 仍只传image/mask/seed，
`observations/object/shape.py::sam3d_candidate_record` 的 `external_pointmap=None`。
`main.py::annotate.native_job` 同时用于 **metric_native** 和 **appearance_native**。
整合时两处都必须从**各自选定帧**得到current-run点图，不能只修第一轮而让最终appearance网格回到内部MoGe。

建议引入命名 `shared_metric_pointmap` worker profile（名称待实现），有显式pointmap路径/receipt：

- 精确来源depth/frame/RGB/相机/像素plane/hash、米制及PyTorch3D变换；
- 禁止点图shape不一致时静默resize，显式生成符合plane的新观测；
- 校验正深度/finite/缺失策略、pointmap投影和原始K一致；
- 记录实际points→normalizer→pose条件和源depth correlation；
- K作为构造/验证数据保留，不新增exactK-only模型干预，不使用负焦距推导作为active camera；
- `with_layout_postprocess=false`固定，若未来启用则是单独设计/验证任务；
- 新profile及参数必须进入planned stage contracts；修改代码/config后新run，不cross-code resume。

### 7.2 为什么接在shape_binding后

1. `optimization/object/shape.py::bind` 会换成appearance-native网格并做独立image uniform scale搜索；
   对第一轮metric-native mesh做过的框拟合不会自动约束最终网格。
2. `optimization/interaction/joint_fit.py::fit_visual` 后续还会优化object delta（camera-z可到±35mm），
   其现有loss没有object metric first-surface depth或envelope。前面拟合好的对象可能再次偏离观测。
3. 当前 `StandaloneScene` 优化观测/RGB pose且固定anchor，不包含全mesh的共享scale；
   `FormalObjectTrajectory` 也不是旧object-only求解器，不能只改名字就声称迁移完成。

因此完整网格候选求解放到 **最终shape绑定后、pose_review之前**，最能约束实际要渲染的物体。
初期提供一个**命名 object-frozen visual tail**用于实验：手仍拟合，物体geometry/pose逐值保持候选。
这只用于新实验profile，旧joint路径保留，不静默删掉原有能力。
之后若希望hand/object joint refinement，需在同一个joint目标中持续消费相同深度/包络因子，
并重算correspondence，而不是靠小delta bound代替机制或事后把物体强行投回框。

## 8. 模块与数据合同（拟议，不是现有API）

```text
observations/object/pointmap.py         current-run shared metric pointmap producer
observations/object/envelope.py         observed Sframe + frozen coarse envelopes
gates/object/observation.py             retained metric input authority (reuse strict predicate)
gates/object/envelope.py                schema/frame/transform/hash/source-role validation
factors/object/envelope.py              robust outside-box residual; no acceptance authority
optimization/common/correspondences.py  full-mesh first-hit correspondence refresh
optimization/object/surface_fit.py      per-frame SE3 + one shared canonical scale solver
optimization/problems.py                ObjectSurfaceEnvelope + named profiles
optimization/solve.py                   explicit dispatch
render.py                              source/candidate/refined/render arrays separately bound
```

`ObjectSurfaceEnvelope`建议字段：annotations、refined_posterior、mesh_binding、
envelope_report、appearance、timestamp timeline、object_id/run_root、profile、output。
命名profile先只定义 `surface_no_box` / `surface_weak_box`；完整组成固定，
数值参数不能作为任意Boolean开关绕过review。

`observed_surface_envelope_v1`建议每帧包含：

- `frame_idx/time_s/object_id/run_id`、point/depth/camera/shape-prior receipt/hash；
- `T_world_surface`、lower/upper、构造quantiles和margins；
- `source_kind=direct_observation|transported_temporal_prior`；
- 支持点数、plane/tangent conditioning、normal sign约定、fallback和不确定性；
- `per_frame_candidate_pose_used_for_construction=false`（direct必须）；
- transported donor、gap及源pose hash，`optimization_weight=0`；
- `generated_shape_size_prior=true`、`calibrated_confidence_region=false`、
  `collision_authority=false`。

输出采用**独立候选binding/schema**，不覆写 `refined_pose.json`：

- 原observed/RGB source及posterior路径/hash和row逐值保留；
- mesh-conditioned候选R/t、唯一scale/pivot、完整mesh hash另存；
- `generated_mesh_used_as_residual_model=true` 与 `physical_contact_authority=false` 明确区分；
- 不继承旧posterior的“generated geometry未参与pose”声明给新候选；
- original native raw mesh、native pose、bridge/gauge转换、scaled display mesh分开追踪。

**不能只修改 `binding['pose_rows']` 让现有loader接受**：`main.fit_prepared` 目前检查它与
`refined_posterior`逐值相同，`render.export/export_scene`还校验base pose、schema和mesh。
需要显式扩展candidate绑定验证，在final JSON中分列observed/RGB refined、surface candidate和final visual pose；
不是删除这些保护。最终视频必须读取真实candidate数组，而不是只加字段或画盒子。

## 9. 变量/尺度的三个层次必须分清

1. native SAM3D scene scale到metric：camera-origin整体相似变换，translation和object scale一起变；
2. 由选定anchor/pose转换到共享canonical gauge：刚体换基，不改变世界顶点；
3. surface候选全片共享scale：绕固定canonical pivot，不是camera-origin scaling，也不是逐帧scale。

新profile暂不额外做 `estimate_scale` 的image-only先验搜索；保留raw binding，
统一由surface solver控制一个scale。其它旧profile维持原behavior，以便明确消融。
样本关联可以outer refresh；观测框、shape size prior、source pose及K不能随inner/outer变化。

## 10. 最小可验证实施顺序

### M1：pointmap-only完整数据接线

两次native都消费各自anchor的共享pointmap；新增CLI/命名profile和input receipts，
不更新default宣称；补数据plane/来源/不重复轴翻转/模型真实consumer测试。

### M2：迁移surface-frame/envelope机制，先diagnostic

从历史纯函数抽取到grouped owner，去掉绝对路径和150-frame假设；
候选R/t变化不能改变direct boxes，退化和无观测不同处理。导出可选envelope线框和残差，
但完整SAM3D/MANO层必须保留。此步不宣称pose被改善。

### M3：命名surface候选优化 + object-frozen尾部

迁移第一可见交点与全时间轴变量，不照搬已有frame-basedtemporal到非均匀时间；
深度不确定性/相关性显式，已有RGB source作为独立软先验。
完整mesh与同一候选输出必须真正驱动最终render，保留旧jointbaseline作为对照。

### M4：预注册三臂对照，不逐例调框

- shared-pointmap baseline（冻结最终mesh/RGB-refined pose）；
- surface_no_box；
- 同样surface solver + weak_box，仅lambda不同。

后两组相同mesh、训练/持出rays、初始化、time/window、手状态及尾部策略。
评价共同命中rays的depth、misscoverage、实际全三角silhouette、尺度/轨迹、完整视频；
不能只以boxinside或训练loss判断成功。存在有授权GT时，预测冻结之后再独立评价。

若要验证框独特价值，可另做预注册坏初始化/稀疏观测压力测试（固定扰动幅度/方向/集合），
与正常初始化结果分开，不事后选择“框有效的帧”。

### 关键回归

- source/camera/canonical/S变换、scale-pivot与half-pixel投影；
- direct box不依赖候选R/t/scale、missing/donor box不施加独立观测力；
- 正常inside residual零，受控outlier产生相应gradient；
- 全顶点conditional interval只对固定R/scale有效，空区间不能伪装物理拒绝；
- first-hit遮挡/空命中不造目标、完整mesh不删面迎合loss；
- 任意frame IDs和timestamps、非150帧、支持断层与对称旋转不确定性；
- `fit_prepared`/final renderer使用指定candidate且sourceposterior仍不变；
- hand-only/frozenobject profile物体数组严格保持，完整时长/visibility clear继续有效；
- 生成几何与框没有获得signed/contact/collision权限。

## 11. 当前建议

**能整合，且源码接缝清楚；应整合为共享点图之后的“表面主导、弱包络辅助”的命名候选优化，而不是让盒子决定物体位姿。**
当前单anchor的新点图结果本来就在框内，历史多case加框增量又很小，
因此优先把pointmap接线、真正的首交表面约束及末端消费连通，
包络保留诊断和低权重防跑偏作用；暂不作为默认硬门限或强化metric真值。
