# 通用几何候选分支与后续 RGB 场景接续设计

## 用户目标与本轮边界

用户授权：采用当前run新生成的frame0 SAM3D结果，继续后续全流程；并要求“选候选→生成SAM3D→继续后段”成为可重复测试其他视频的正式管线能力。不得写成milk/frame0专用替换脚本，也不得把frame0 PLY直接放到anchor145姿态里。

本轮机制目标：一个明确、可重放的**候选分支**，使用同一套初始姿态→全局RGB/SO3/SE3→形状/MANO→JSON/RRD生产者；保留既有分支作为对照。不是新顶层版本，不是第四/第五个针对失败位置的恢复特例。

唯一writer工作树：`standalone_rgb_scene_candidate_pipeline_20260911`，分支`pipeline/standalone-rgb-scene-candidate-20260911`。生产基线9e8df13，已cherry-pick排名优先anchor规范ecd05a0。源码任务只允许代码/文档/CPU测试/本地提交；真实run、已部署bundle、控制产物只读。父级随后构建curatedruntime并执行。不要进行GPU/模型、install、push/merge或修改旧产物。

## 1. 固定不变的机制

- 原preferred分组和七项评分保留；默认最高排名经实际图像检查后使用，明确缺陷才否决。支持用户显式候选帧覆盖，但不绕过实际当前输入检查/来源。
- 动态case/object/track、真实frame IDs/timestamps/分辨率；不写死0/145/150/milk。保留当前单个选中rigid object的范围，不添加无请求的多对象系统。
- 原始RGB/相机/深度/HaWoR/MANO/P08、raw SAM2、depth-order ownership/eligibility是只读来源；完整真实MANO与三角mesh不能替换成代理。
- P09/P11的selected/canonical/metric anchor必须同帧。本次需在候选新命名空间重新建立P09 anchor0和P11证据，不能对145完成报告只改frame字段。无需重算P01-P08或已正确的ownership。
- 原PnP世界运动公式保持；全局后验必须真正驱动形状/MANO，不fallback到原P14。物理authority仍false，generated mesh只用于视觉假设/unsigned triangle proximity。
- 最终默认只有JSON和RRD。静态诊断图可供查看；不默认MP4。

## 2. 可复用接口与候选生命周期

推荐在现有prepare/runtime入口增加candidate模式，或者一个被明确收录的独立candidate-stage入口。不得暴露随意执行任意脚本的入口。

```text
已有/本次生成的当前视频前缀与P09候选列表
  → prepare-candidate(run, candidate_id, frame=auto|explicit, native_mode=new|reuse)
  → 当前候选RGB/mask检查与分支选择决定
  → 候选P09/P11同帧metric/canonical证据
  → native生成 或严格校验复用已有native结果
  → P13 metric bridge与完整render mesh
  → anchor binding/initial-pose
  → current-video RGB edges + global SO3/fullSE3
  → raw appearance/shape conditioning与refined canonical binding
  → native当前交互/visibility判断
  → 完整MANO视觉拟合
  → 候选 final/annotation.json + final/world.rrd
```

- 参数不能隐藏在开发脚本固定路径里。文档应给出任意视频A/B运行、默认排名选择及候选override/reuse的实际命令。
- 一个candidate_id是一个create-only分支。重复实验用新ID；同ID成功阶段可显式验证后续跑，失败/partial不得自动重试或覆盖。别实现任意rollback框架。
- 可在原run内建立候选命名空间，集中一个`CandidatePaths`/上下文处理各生产者路径。源run的全局`state/object_selection.json`/anchor145不得更改。
- 简单命名空间方案优先：所有新增mutable文件归属该候选，必要沿用低层函数对appearance输出位于`measurements/demo_visual/<object>`的限制，集中路径映射而非散落字符串补丁。
- **不要复制/改写整个源run并宣称fresh，也不要伪装出生bundle身份。** 原birth identity + 独立candidate ID/执行bundle/源snapshot/选择身份并存。
- 新候选模型执行仍用curated native subagent+ShellGate，parent只准备合同/环境/启动；runtime仅预测与真实模型辅助判断。无需每次新建恢复revision代码。

## 3. 生成与复用SAM3D的共同合同

`new`：同frame RGB和实际所选mask，显式seed、模型/config/环境身份，运行现有pinned原生SAM3D。记录完整raw/native-camera meshes和QC。

`reuse`：必须有真实成功模型receipt与输出SHA、输入RGB/mask SHA、源run/frame/模型/seed/config绑定。缓存命中只有这些相容且全部bytes重验才有效；不能仅因某个PLY存在而复用，不复制旧demo/其他clip的预测。

当前frame0控制结果位于原run之外的独立控制目录。允许一个明确的输入import/materialize stage将已验证完整native结果复制进候选范围并写**派生QC/路径映射**，记录原QC和原模型回执hash；这是cache reuse，不是重新推理。不得覆盖原QC或伪造returncode。派生QC路径变化必须显式声明，image/mask改为候选同字节输入仅在SHA相同且frame一致时允许。

目前控制wrapper结尾print失败，但真实模型returncode0/cleanupstopped、QC和7输出已保存，13来源输入哈希未变。校验复用时分开“模型阶段成功”和“wrapper打印失败”，将真实failure记录和父级post-freeze disposition一并保留，不把失败擦掉。其他无真实成功模型receipt的失败不得通过此例外。

Metric bridge继续调用已有P13数值，重放完整raw-local→PyTorch3D-camera→OpenCV-camera，使用**所选帧**当前retained depth/camera求scene similarity，再建立该候选canonical；不把nativeunits当米。

后续appearance/shape若使用与第一次native完全相同的输入/模型/seed，可合法复用这份native候选；不必为同字节条件再跑一次。若rawappearance与第一次conditioning不同则不能硬称cache命中，应按同一个候选接口新生成或明确选择已授权输入。

## 4. 来源与执行的最小实现原则

- 新候选使用一份compact candidate合同和独立append-only事件流。prepare侧将可用的已成功source receipts/当前模型版本/必要输入集合冻结；runtime验证必要输入与实际stage角色，source不可变，不每stage重复扫描所有无关权重。
- 允许新增standalone candidate lineage/context，读取已验证源snapshot和本候选事件；避免递归套用旧P09 recovery的1/2/3事件长度特例。这些旧合同仍是源历史证据，不是候选workflow的stage调度器。
- 保留原producer代码与输出数据类型，尽量复用现有`run_command`/`run_owned`、pose/shape/fit/export，不整体复制上千行renderer/geometry。若需要给低层standalone consumer传显式selection/judgment/output上下文，增加默认关闭的可选参数，旧路径不变。
- 所有当前计算产物path/hash应精确绑定候选。不能把一个candidate的P09/姿态/mesh拿给另一个仅因为object ID相同。
- create-only输出和进程预算保持；工具1200s、内层留cleanup余量。geometry实际consumer检查SAM3D资产，CPU阶段不再hash无关模型。
- 最终JSON/RRD身份需包含原birth及candidate/reused-native来源。两文件移离cache后完整几何、相机、MANO、时间、visibility仍可播放。

## 5. 已知需要接线的位置

- `run_rgb_scene_stage.py` 的路径调度和native-shape复用；`run_rgb_scene_frontend.py` 的P09/P11路径与候选frame。
- `prepare_rgb_scene_runtime.py` 的bundle closure/候选准备；native spec/profile和README加入genericcandidate模式。
- `rgb_scene_runtime.py`/`rgb_scene_lineage_base.py`：现有judgment路径为全局硬编码，需要可选candidate path context，或干净独立CandidateLineage；不放宽普通source校验。
- `run_rgb_scene_reconstruction.py`：现有只支持new SAM3D，增加明确cache/import逻辑或小wrapper复用P11native/P13；不得新run替代原生mesh。
- `bind_v19_completion_anchor.py`要求P09/P11三种anchor同帧，保持该条件，解决方法是重做候选P09/P11，而不是删检查。
- `bind_rgb_scene_shape.py`要求QC的image/mask路径与appearance匹配；为已验证cache增加显式derivedQC mapping，不伪造原来源。
- `rgb_scene_depth_order_appearance.py`输出范围有限制；用集中candidate paths或严格显式可选candidate context。
- `export_rgb_scene.py`和`rgb_scene_identity.py`当前读全局selection/judgment、recovery身份；需要候选可选接口，默认不变。

## 6. 测试必须碰真实边界

做有界CPU synthetic集成，不重复只检查字符串/手写事件的旧问题。

1. 动态frame（如7/42）、不同对象ID/非方形输入，默认原排序第一名、显式candidate覆盖、两个候选互不改写。
2. 实际P09/P11 producer以合成输入生成同帧anchor证据，接实际P13或数值绑定；不得靠mock掉调度/验证声称流程可用。
3. native cache import正例（完整三角、相同输入/config/seed）和frame/RGB/mask/model/mesh修改拒绝；原receipt不变，新mapping明示来源。
4. 连续完成至少两到三个阶段后追加回执，下一命令仍通过；实际嵌套producer进入时不会把其尚未写出的输出当source先验。
5. frame0/145类不同gauge禁止直接换mesh；nativecamera/world/canonical roundtrip、完整face索引、refinedrows不被重置。
6. 实际fit/export消费候选selection/visibility/identity；模型fitter可在CPUfixture替换为既有测试fixture，但真实export必须写出并读回JSON/RRD。
7. fresh/default无candidate路径回归；模型资产role和mutableinputhash不得全关。

先做这些必要接线和focusedtests，清洁本地提交、一次针对新增真实路径的独立审查。之后父级马上跑当前frame0后段；不把大量无关通用硬化作为此次产物的前置条件。

## 7. 当前样本的只读开发证据（不硬编码进实现或runtime）

```text
S=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910
R=$S/cold_raw_rgb_v1
原生frame0结果=$S/frame0_sam3d_control_20260910
源runtime=/mnt/user-home/kupingxin/ego_annotation_runtime/rgb_scene_p11_sam3d_smoke_9e8df13_gpu5_20260910
```

- 原run birth UUID `ded84007-0cd4-4717-b16a-2865e62eaea1`；150帧、object milk_carton、track milk_carton_track只是本次数据。
- 原anchor145/P09/P10/P11/geometry全已完成；之后初始姿态/优化/拟合/导出尚未执行。
- 当前frame0原生mesh238288v/476588f，SHA `3b7764f821c18e23a44ee942b46aa2e7f90e8a858c3a6a47328a34306e8321bd`；完整native QC/receipt在控制目录。
- `execution_receipt.json` SHA `b428af7def578e35a5f478888b8bac2f712a7db489a72feac67af84c7a4d7360`；内模型return0/198.605秒；`failure.json`是收尾print KeyError，不得删。
- 父级独立处置：控制目录`parent_verification/verified_model_and_wrapper_disposition.json`。
- 原native框架代码已修print为76c8fd9，但此控制产物仍归原执行b32b8aa，不重stamp。
- 当前frame0 mask与rawappearance恰同字节，可用于这次后续合法复用；通用代码必须显式比较而非假定。
- 旧demo和三组比较只供父级冻结后评估，不能进入native producer context。

## 8. 成功定义

**源码完成**：其他视频可经相同候选接口选帧/new或reuse生成完整SAM3D、使用正确同帧metric/canonical绑定，并沿同一调用链走到候选JSON/RRD。文档有真实可用命令与当前身份限制，focusedintegration能证明接线。

**本次任务完成**：native用该通用入口完成当前frame0候选的后续pose/refinement/shape/MANO/JSONRRD，并由父级验证可播放、完整时间轴、旋转/几何和uncertainty。仅文档/manifest/代码测试不算此用户任务完成。
