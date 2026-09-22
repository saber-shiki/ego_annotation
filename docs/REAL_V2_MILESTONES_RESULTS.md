# 真实 milk v2 与三个里程碑：执行结果

> **后续回归检查补充（2026-09-22）**：与旧RL真实零动作视频的四组对照已确认，
> 原P09源初始化/P13网格基本保留，但完整native网格选择和RGB seed使已交付seed视频的
> 物体大小/位姿显著改变，图像贴合总体退化。此前读回/消费链通过不等于几何质量通过。
> 现已补齐两种无seed三维视频；seed算法未修复。
> 见 [ORIGINAL_RL_SOURCE_COMPARISON.md](ORIGINAL_RL_SOURCE_COMPARISON.md)。

日期：2026-09-22。独立分支：`feature/rl-observation-v7-alignment-20260921`。开发目录：

`/mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment`

## 一、结论与验收层级

| 项目 | 结果 | 严格含义 |
|---|---|---|
| 真实raw→v2生产 | **通过** | r2退出0，96阶段全部成功，150帧/29artifact，非合成 |
| 完整v2读取/前处理 | **通过** | 所有声明模态实际解码、处理；源hash未变；不是只检查JSON存在 |
| V7前处理纯核同输入对照 | **覆盖项通过** | 10项已验证；不能扩展成整条V7后端等价 |
| 全时间轴源MANO重放 | **通过** | 两条既有decoder、每手150帧vertices/ordered joints，最大误差低于1e−4m |
| 完整seed场景QC | **已输出，仍有可见偏移** | 150帧、完整mesh/MANO；R4 seed不是优化或policy结果 |
| 里程碑二受控消费链 | **验证专用适配器通过** | v2→seed/full mesh→受控更新→动态测量→全帧导出；不是旧AnnotationEnv/PPO factory已升级 |
| 里程碑三策略兼容性 | **未通过，blocked_not_trainable** | index/dataset拒绝v2；旧checkpoint只有v1、480/118契约；新策略layout/decoder/reward未定义 |
| 训练/学习效果 | **未执行** | 不取消保护、不伪造reward、不将检查器当可训练环境 |
| 默认吞吐目标 | **未达到** | 5s视频冷观测生产约59m16s，不在视频时长同一量级 |

这次真实运行证明前端/前处理可执行，但**没有证明最终标注质量达到V7，也没有证明新观测已被原actor利用**。

## 二、运行与产物路径

### 真实预测源

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2/
  source_observations.json
  input/
  measurements/
  state/recovery/
  logs/
```

运行快照：

`/mnt/user-home/kupingxin/ego_annotation_runtime/rl_observation_v2_milk_20260921_r2`

独立RGB环境：

`/mnt/user-home/kupingxin/ego_annotation_envs/rgb_v2_20260921`

运行控制/全部验收根目录：

`/mnt/truenas-user-home/kupingxin/rl_v2_a800_execution_20260921/`

### 重点查看

| 路径（相对验收根目录） | 内容 |
|---|---|
| `prediction_r2/observe.exit`、`observe.log` | 实际退出0和运行日志 |
| `prediction_r2/parent_completion_check.json` | 独立核对96 start/success、snapshot未变及bundle身份 |
| `real_v2_inspection_r2/inspection.json` | 完整真实观测读取/前处理结果 |
| `real_v2_inspection_r2/consumer_matrix.csv`、`artifact_consumers.csv` | 每组/每项真实消费及尚未接入下游 |
| `real_v2_inspection_r2/review/` | 源图像、mask/depth、MANO/pad、RGB匹配QC |
| `real_source_seed_readback_r2/readback.json` | 全帧licensed MANO重放、seed几何统计 |
| `real_source_seed_readback_r2/source_seed_full_overlay.mp4` | **150帧源MANO+独立RGB seed检查视频，不是policy** |
| `milestone2_real_consumption/results.json` | 真实v2受控消费链测试结果 |
| `milestone2_real_consumption/zero/candidate.{json,npz}` | 零修正候选全片状态 |
| `milestone2_real_consumption/controlled/candidate.{json,npz}` | 三帧受控修正候选全片状态 |
| `milestone2_real_consumption/controlled_full_overlay.mp4` | **150帧受控修正检查视频，不是PPO结果** |
| `milestone2_real_consumption/zero_vs_controlled_*.jpg` | 60/94/143三帧实际几何变化对照 |
| `milestone3_compatibility.json` | 策略接入就绪度的阻塞证据 |
| `milestones_full_regression.log` | 239项测试，234通过、5跳过 |
| `real_stage_durations.json` | 各stage实测耗时，非纯模型算时 |

旧r1及其两次网络失败、恢复记录保持不变。r2未导入r1前缀；r2自己的appearance-native满足严格同帧同输入条件后复用了同run native，避免第二次SAM3D推理。

## 三、真实v2结果

- 输入视频SHA256：`43513ed7534f8d32ede3ec13ad8c5b92d41487705064081491ab7831ee0e1b59`。
- bundle SHA256：`f86e3dd8c758e811de90b4a957188a2967ce3f56af839c532d21aa29ee41921d`。
- 开始：2026-09-21 23:43:11；结束：2026-09-22 00:42:27，约59m16s。
- GPU5：`GPU-e4ff72b8-fff7-21d0-2872-6aeecb56a91e`；结束后无本任务遗留GPU进程。
- 独立runtime执行者运行生产；后续检查只读预测结果，无GT输入、无优化答案回灌。

### 新旧观测并列（不是控制变量精度排名）

| 内容 | 新RL v2 r2 | 研究V7历史发布 |
|---|---:|---:|
| 原视频 | 同一150帧 | 相同 |
| K来源/focal | supplied sensor /974.3447px | estimated UniDepth /1248.5733px |
| P09有visible metric表面 |149帧|132帧|
| P09 metric-eligible |148帧|132帧|
| RGB producer accepted边 |181|188|
| 重投影证据点 |20,394|21,292|
| 独立loop拒绝 |18|25|
| 局部tree边 |149|149|
| 完整visual topology |239,430顶点/478,880面|238,288顶点/476,588面|
| 最终优化 |没有，只有measurement seed|历史SO3/SE3及visual fitting|

新r2：94帧有表面但不metric-eligible；143帧robust first-surface失败，保留RGB/appearance，不制造该帧metric测量。其余RGB路径能为这些帧提供uncertain seed，所以seed覆盖150帧。不是通过放宽metric gate制造150个直接观测。

本次P17：300个frame/hand声明，450个visibility条目；双手均为partially_visible150帧，object132帧partially_visible/18帧visible；全部contact声明likely_contact。与旧minfix曾有12个hand out_of_frame不同，这是本次独立模型判断，不是原判断的修改。细分语义准确性仍需视觉判断，不能由schema通过保证。

### 完整读取检查

- 150个depth/confidence帧；300个raw/owned masks；150个native eligibility raster。
- 148帧retained metric支持实际重放；raw可见表面149帧与资格148帧分开。
- 150帧appearance轮廓/距离图前处理。
- 181条RGB测量的20,394个点重新计算投影。
- 完整visual mesh与pose/collision独立角色，canonical/world往返最大约6.94e−18m。
- P17 contact_patch296行、visible_ownership296行；实际first-surface query63,221个样本。
- 2,479个实际读取文件前后hash不变。
- 检查器的`complete_preprocessing_readback`表示读取/处理完整，**不表示全部局部测量合格或无质量缺陷**。缺P09资格帧、RGB rejected边和source-hand偏差见本报告；不得用`warnings=[]`概括成全片物理正确。

## 四、全帧MANO重放与几何质量边界

使用真实已配置MANO资产，分别按V7既有source decoder和RL既有source decoder对每只手的全部150个有效帧重放；不是只验证首尾，不执行拟合。

| 最大误差m | 左手 | 右手 |
|---|---:|---:|
| V7 vertices |9.996e−8|1.164e−7|
| V7 ordered joints |8.462e−8|1.024e−7|
| RL vertices |1.673e−7|1.382e−7|
| RL ordered joints |1.661e−7|1.374e−7|

均低于1e−4m。这证明参数/拓扑/关节排序与源几何一致，不证明源手姿态准确。

全片seed视频采用478,880个实际物体三角面与左右各完整MANO面；150帧解码成功、30FPS、5s。已查看0/60/90/143/149等代表帧和独立检查图：

- 源MANO投影有明显手形/位置偏差；
- seed物体在中段与尾段有图像位置/姿态偏移；
- 未使用V7 posterior、shape-scale搜索或手部优化来隐藏这些偏差；
- source-only seed不是最终物体运动估计。

seed逐帧旋转跳变：中位0.949°、p90约5.073°、最大14.058°（目标frame89）。unsigned full-generated-surface pad诊断（全5指、每指源12点取最近3点均值）中位左34.37mm/右21.64mm；它不是接触精度，未与V7不同定义的gap指标混比。

**因此允许作为带噪输入进入消费验证，但不声明V7最终拟合能力已恢复。**

## 五、里程碑二：验证适配器的真实消费测试

新增模块：`envs/v2_consumption.py`。

明确边界：这是环境侧`V2ConsumptionSession`验证适配器，具有reset、显式受控更新、fresh measurement与完整export；**不是原`AnnotationEnv`替换，不具有PPO action/reward契约，也未改train.py/rollout.py**。

**2026-09-22更新：**适配器默认模式已改为`source_full_native`：完整native mesh绑定到不使用RGB传播的source/P09初始化；历史R4 RGB传播轨迹保留为显式`rgb_seed_comparison`。本节下方历史三帧controlled实验明确使用seed comparison模式，不能当作默认RL起点。真实默认模式和V7逐项审计见`docs/NO_RGB_SEED_RL_BASELINE_RESULTS.md`。

实际接通：

1. v2 loader的源数据与新motion seed；
2. 完整visual mesh绑定到明确seed revision，binding不随每步偷偷更新；
3. source/candidate/display validity分离，typed visibility驱动该适配器的display；
4. fixed anatomical pad membership与动态closest triangle查询；
5. 当前source/target candidate共同决定固定RGB源点的重投影；
6. JSON/NPZ/receipt及完整三角面渲染。

### 实测

- 150次零控制后，R/t、完整手顶点与seed精确一致。
- zero candidate与前述source-seed readback的所有关键几何/有效性数组逐项精确一致，因此有hash记录地复用zero检查视频，不重新渲染。
- 在frame60/94/143分别注入world物体旋转`[0,0.02,0]rad`、物体平移`[0.005,0,0]m`、左手平移`[0,0.003,0]m`。
- 物体顶点最大移动约6.30/6.10/5.77mm；左手源投影相对变化的中位数约5.13/4.69/5.24px。
- RGB重投影和pad最近面目标都变化，原固定pad vertex IDs不变。
- frame94/143虽然无直接metric资格，seed可用，因此没有被错误屏蔽。
- 147个未控制帧的几何精确保持；reset还原seed；源manifest/hash未变。
- 导出150帧、object candidate available150、metric-eligible148；不能将两种bool合并。
- `controlled_full_overlay.mp4`实际完整解码150帧。

不将控制方向解释为改善：例如某条edge在60帧的误差下降、94/143帧上升，正好证明测量确实耦合当前状态；本实验没有自动挑选更好动作，没有计算reward。

合成测试还覆盖了out_of_frame clear不改source MANO、缺手不制造几何、固定anchor拒绝更新、非法控制原子拒绝。

### 里程碑二仍未完成的生产接入

原dataset/AnnotationEnv/action layout/model factory没有接通此适配器；每手52维visual decoder、全片共享object scale、动态轮廓目标和历史编码仍需后续实现。当前证明的是“可独立验证的新消费者闭环”，不是“旧策略已经读到了v2”。

## 六、里程碑三：兼容性测试阻塞，未训练

命令：

```bash
python -m observations.check_policy_compatibility \
  --source-observations /mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2/source_observations.json \
  --checkpoint-provenance /mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_learning_stage1_1024_v1/checkpoints/step_0256/agent.provenance.json \
  --output /ABS/NEW_REPORT.json
```

实测退出码2，`blocked_not_trainable`。不是故意报错后宣称通过学习，而是发现条件不足并保留阻塞：

- official index与direct dataset index均拒绝前处理专用v2；
- 旧checkpoint仅声明v1、`AnnotationEnv`、480维observation、118维action；
- controlled adapter没有policy observation/action layout与reviewed reward profile；
- 完整visual手decoder、共享object scale未接入；
- 原actor packing不会因新字段存在而自动消费；
- 不把小幅physical-unit注入接口冒充PPO动作空间。

未加载checkpoint、未构建PPO、未执行训练更新。虽然算力获准持续使用，**缺少目标/状态/动作契约不是扩大运行预算能解决的问题**。

下一步必须在明确的新v2 profile中真正实现episode/state、feature/action/decoder、动态因子与checkpoint兼容，再做同观测训练；不应该删除v2 guard后直接训练旧actor。

## 七、测试、源码与保存

- 新增`envs/test_v2_consumption.py`共6项：5项消费/几何边界 + 1项策略兼容拒绝检查。
- 最终总回归239项运行，**234通过、5项按真实依赖/可选fixture跳过**。
- 原PPO、train/rollout、旧动作/奖励/AnnotationEnv、旧renderer未修改；新增的是独立验证适配器/检查入口，不能将测试模块数量当作物理能力提升。
- r2生产源码/资产snapshot仍字节不变；后续验证代码在开发工作区及外部检查脚本中，不回写预测run。
- 未提交、未推送；原dirty工作区与V7代码保持不变。

本轮增加的代码：

- `envs/v2_consumption.py`
- `envs/test_v2_consumption.py`
- `observations/check_policy_compatibility.py`
- 本文。

外部检查脚本：`verify_real_source_seed.py`、`test_real_v2_consumption.py`，均位于验收根目录，只生成QC/验证candidate，不伪装runtime预测输出。

## 八、下一步优先级

1. 保留当前真实v2包，后续环境/接口测试直接复用，**不要重跑感知**。
2. 定义并实现可训练v2 profile：先source/seed/candidate和完整mesh/visibility，再observation encoding和动作decoder。
3. 给当前候选动态RGB/pad/轮廓测量一个明确、经同状态核对的objective，不偷换旧formal objective。
4. 先零动作/controlled regressions、checkpoint拒绝/重建契约，再小规模有界训练。
5. 单列runtime吞吐优化：本次judgment阶段合计约1881s、所有成功stage合计约3395s；API判断、ownership与IO开销明显，不能称作实时默认管线。

如仍严格要求“不涉及训练模块”，则本轮可停在前端真实验收和环境侧验证适配器；第三里程碑不能在这个范围内宣称完成。
