# 里程碑一：v2可行性与V7对齐验证（阶段结果，未通过真实cold验收）

日期：2026-09-21。修改/测试工作区：

`/mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment`

分支：`feature/rl-observation-v7-alignment-20260921`；起点`823a906`。V7只读参考commit为`67c40e6`。

## 当前结论

1. **已有观测前处理代码的同输入数值一致性验证进一步通过。** 共10项直接V7对照检查（本轮新增7项，其中1项读取既有真实milk中间数据）。
2. **v2真实raw-video冷流程尚未验证成功，当前运行配置也未就绪。** 旧资产文件的`rgb_python`不存在，两个现有候选解释器分别缺torchvision或kornia。不能直接使用旧milk配置启动新v2。
3. **不能把这些数值对照写成“已经实现完整V7 baseline对齐”。** 新P14/P15替代状态、动态对应、策略输入与导出仍不在此轮验证闭环中；真实perception/reconstruction输出还需单独验证。
4. 未启动模型、API、重几何、PPO或真实raw-to-v2运行。里程碑二和三尚未开始；不跳过里程碑一的真实数据门槛。

## 1. 已完成的同输入对照

参考函数取自冻结Git对象，仅提取命名纯函数；不导入/执行整个V7脚本，不启动V7模型worker或优化器。代码：

- `observations/test_v7_reference_preprocessing.py`（已有3项）
- `observations/test_v7_observation_parity.py`（本轮新增7项）

| 检查 | 输入 | 对照结果 | 不能据此证明的事 |
|---|---|---|---|
| half-pixel K resize | 多组源/目标尺寸 | 与冻结V7一致 | 模型真的用了此K仍需真实producer证据 |
| front-only MANO/depth归属 | 合成front/behind/near/no-hit/invalid depth | mask、分类与计数一致 | 预测手/深度本身准确 |
| native-depth eligibility | 合成不同分辨率、fringe、缺手、确认无手 | code raster、eligibility和计数一致 | 任意真实场景的可靠度 |
| robust first-surface | 连续斜面、边缘bleeding、小/大内部异常、缺confidence | retained mask和全部diagnostic/failure理由一致 | 真视频所保留表面一定正确 |
| native front-depth投影 | 小型原生坐标网格 | 投影/深度统计一致；metric similarity复用同定义 | hidden shape真实、image-scale求解等价 |
| 相同特征匹配的PnP | 固定匹配、相机、depth与RANSAC seed | local LM的相对R/t、目标UV/K/权重一致；新固定world点等于旧canonical解码world点 | 真实SuperPoint/LightGlue已验证；完整P14/P15等价 |
| 局部forest与独立loop | 合成非连续ID、多条短边和错误长边 | seed旋转/loop决定/conditioning一致；拒绝的长边pixel权重同步归零 | 新candidate状态已使用这些证据 |
| primary声明/visibility | 固定P17声明 | 新typed可见primary资格与V7一致 | 语义判断准确或最终display已变化 |
| native/canonical变换 | 相同完整网格与明确candidate R/t | 数值/拓扑一致 | 已接入环境mesh或学习scale |
| 真实V7 milk中间观测 | 150帧source annotations、既有RGB边与P17 | 同输入资格与typed primary数值一致，源hash未变 | 新RL真实cold run完成 |

PnP差分测试的比较对象是**同输入匹配下的局部测量函数**。V7旧canonical表示只在测试内解码为world，用于比较；没有将旧P14/P15文件导入正式RL源包。

## 2. 真实milk参考回放的具体证据

只读目录：

`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910/cold_raw_rgb_v1/candidates/rgb_chain_contact_repair_v7`

读取以下现存文件，并核验读取前后hash不变：

- 当前candidate P09 annotations；
- `scene_rgb/edges.npz`、`edges.json`；
- P14 source pose报告，仅用于测试内解码历史canonical points；
- 当前candidate P17判断文件。

实际规模与结果：

- 150帧；
- 188条producer接受的RGB测量；
- 21,292个重投影支持点；
- local-tree 149条；
- independent-cycle拒绝25条，consistent-cycle14条；
- 新旧纯kernel的qualification结果相同；
- 同一P17声明编译出的primary-visibility资格相同；
- 读取的源文件未变化。

特别说明：新SE(3) seed的平移传播是本分支新增的初始化语义。此测试仅证明替换新旧rotation qualification核后，新seed组装结果相同，**不证明它等于V7 P14/P15的全局优化轨迹**。

机器记录：`real_v7_milk_parity_final.json`。

## 3. 真实v2执行前的实际阻塞

所查旧配置：

`/mnt/user-home/kupingxin/ego_annotation_runtime/rl_milk_minfix_v1_20260918/assets.json`

### 3.1 声明解释器已不存在

```text
rgb_python = /mnt/user-home/kupingxin/GHOST/.venv/bin/python
```

路径检查：不存在。`preflight_providers()`会拒绝此声明；不能忽略这一错误再把RGB stage标为完成。

### 3.2 两个现有候选环境也未通过LightGlue真实import

本次仅在`CUDA_VISIBLE_DEVICES=''`、单线程、offline、禁止pyc写入条件下import包；未加载模型权重。

| 候选解释器 | 真实import结果 |
|---|---|
| `ego_skrl/.venv/bin/python` | `ModuleNotFoundError: No module named 'torchvision'` |
| `ego_annotation/.venv/bin/python` | `ModuleNotFoundError: No module named 'kornia'` |

没有修改旧assets、没有安装依赖、没有将两个环境的site-packages拼接、没有悄悄选择替代解释器。`rgb_environment_import_candidates.json`保存完整命令对应解释器、返回码与错误尾部。

需要后续明确：

1. 实际计算host、GPU/UUID及使用预算；
2. 使用哪个已预配置RGB环境，或是否允许新建独立RGB环境；
3. 新assets与新不可变runtime快照，不能修改旧run身份；
4. preflight通过后再执行fresh milk v2，不能沿用旧代码run做跨版本resume。

现有代码中`main.py`、provider选择和stage参数已变，旧run重用若要实现，必须有显式导入与依赖证明；当前不能直接复制前缀并重签成新生产结果。

## 4. 剩余对齐项的明确分级

| 项目 | 本轮判定 |
|---|---|
| 既有几何核的same-input数值语义 | 上表覆盖部分已验证 |
| 独立source-world RGB格式、typed P17、complete visual角色 | 已实现并有合成集成验证 |
| 新LightGlue实际特征提取/匹配、真实SAM3D appearance输入 | 尚未实际执行验证 |
| 新v2从raw入口完整产出并通过inspection | 尚未执行；环境与计算授权阻塞 |
| P17更丰富源MANO/appearance/跨batch上下文 | 尚未实现；现接口不能称为完整上下文等价 |
| 独立局部ICP测量族 | 尚未新增；不能用RGB seed冒充该机制 |
| 所有有效帧licensed MANO LBS重放 | 未执行；目前source-array/bridge和纯变换检查不等于此项 |
| 逐点P09新增provenance archive | 未新增；现有P09来源与RGB matched支持仍保留 |
| P14/P15优化结果 | 故意不进入正式观测；不以复制优化答案完成对齐 |
| dynamic nearest triangle/outline、scale/52维MANO decoder、history | 后续消费/状态层工作，当前不是已完成项 |
| old actor/default reward/final render真正使用新v2 | 尚未接入，v2保护仍保留 |

所以当前不能说“除训练外全部与V7对齐”；准确表述是：**观测侧主体接口和若干关键纯处理核已实现并验证，真实producer和若干前处理/后端语义仍有明确缺口。**

## 5. 三个里程碑的后续执行门槛

### 里程碑一：真实v2与V7对齐验证

已完成A：代码/合成同输入差分；增加真实V7中间数据的只读纯kernel对照。

待完成B：配置独立、有效的模型环境 → fresh不可变snapshot → 真实milk raw-to-v2 → inspection覆盖全片 → 查看真实几何/投影/可见性输出。

两类比较必须分开：

- V7-source回放：固定旧K/depth/MANO等输入，检查数值语义。
- 新RL-sensor运行：保留合法sensor-K，不为接近历史结果强行替换为V7估计K；此时不得要求重现相同的147/132个metric帧或188条RGB边。

只有真实运行和处理证据足以说明各阶段可用后，才进入里程碑二；信息本身不确定可以继续，但缺模块、错坐标、未接通消费者不能靠warning放过。

### 里程碑二：不运行PPO的环境消费闭环

先补齐显式v2 episode/state adapter，再测试：

- source/seed/candidate分离；
- 完整visual mesh的canonical binding与独立collision角色；
- typed P17进入真实资格与display消费者；
- 零动作、受控ΔR/Δt产生合理几何/残差变化；
- 全150帧导出、无源数据改写。

当前旧环境没有这些适配，不能仅取消dataset拒绝就执行并声称通过。本阶段不需要修改PPO算法或启动训练，但涉及环境消费者代码，必须按里程碑明确实施与验收。

### 里程碑三：策略接口与学习验证

输入layout/动作与decoder/选定目标profile/normalizer/checkpoint兼容首先做合成和受控测试。旧checkpoint不直接视为新接口兼容。随后在明确训练资源与步数预算内做同观测学习，再考虑多视频泛化。

在前两步未闭环时不扩大PPO，也不靠统一大loss或删去困难帧制造改进。

## 6. 测试结果、文件和限制

本轮差分首次遇到AST测试夹具缺行号、随后缺`torch`namespace；修复的是测试提取环境，不是放宽数值阈值或改变生产核。失败日志保留。

最终总回归：**215项运行，211项通过，4项按真实模型/资产条件跳过**，约79.6秒。生产推理与训练次数均为0。

证据根目录：

`/mnt/truenas-user-home/kupingxin/rl_v2_milestone1_20260921/`

主要文件：

- `runtime_path_check.json`：旧资产解释器路径的可用性；
- `rgb_environment_import_candidates.json`：候选解释器真实LightGlue import失败；
- `real_v7_milk_parity_final.json`：既有真实V7 milk同输入核对；
- `parity_verified.log`：10项V7差分测试；
- `full_regression.log`：最终215项总回归；
- `before_worktree.tar.gz`、`before.json`：本轮前保存的工作区状态。

本轮仅扩展测试及报告，未修环境、未改训练模块、未改V7、未提交/推送。
