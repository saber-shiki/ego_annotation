# V2观测/前处理独立验收入口

> 2026-09-22更新：本文记录的是检查入口开发与合成验收。真实r2随后已完成并通过完整读取；请以
> [REAL_V2_MILESTONES_RESULTS.md](REAL_V2_MILESTONES_RESULTS.md) 查看最新真实运行结果与策略接口阻塞。

## 1. 本轮范围与真实结论

依据用户确认的四项要求：

1. v2各项观测可以完整读取；
2. 对应前处理确实消费内容；
3. 输出不依赖PPO的独立检查结果；
4. 明确每项信息尚未接入的下游消费者。

本轮实现了 `python -m observations.inspect`，并增加 `EpisodeBundle.load_observations()`。它们不调用训练dataset、PPO、policy、感知worker、Astra或MANO权重，不更新候选状态或源数据。

**代码与合成接口验收已完成；真实milk的新v2包验收尚未完成。** 当前既有milk是v1，检查入口明确拒绝将其静默升级。没有通过导入旧P15/拟合结果、改写manifest或伪造producer receipt来凑出真实v2运行。

独立分支：`feature/rl-observation-v7-alignment-20260921`。工作区：

`/mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment`

## 2. 检查入口

```bash
cd /mnt/user-home/kupingxin/ego_annotation_rl_observation_alignment
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/user-home/kupingxin/ego_skrl/.venv/bin/python -B -m observations.inspect \
  --source-observations /ABS/REAL_V2_RUN/source_observations.json \
  --output-dir /ABS/NEW_INSPECTION_DIRECTORY
```

以上真实输入路径是说明用占位符，本轮没有运行真实新v2视频。

- 输出目录必须不存在，并且必须位于源run之外。
- 可用 `--review-frames 0 30 60` 指定QC图帧；**数值读取/前处理仍覆盖全部声明帧**，不是只检查选中帧。
- 默认最多选择6个时间位置，再加入完整visual mesh的appearance anchor。
- 该入口是本轮完整单物体v2的验收profile：要求全部29类观测artifact和四个v2 capability。合法但不齐全的v2包返回 `incomplete_missing_capabilities`，不把缺功能补零后称为完整。
- 退出码：0=读取/前处理完整（允许测量缺口）；2=缺capability/artifact；1=损坏、契约错误或运行失败。
- 测量为空不等于缺capability：有合法记录但无RGB边时仍消费其诊断，报告`measured_empty_rgb_edges`，不生成零运动。

`load_observations()` 与旧 `.load()` 使用独立缓存。它跳过旧contact patch的最近三角面membership准备，但仍读取P17 factor及独立first-surface查询。旧`.load()`行为保持兼容。新增检查不是旧训练消费者的自动适配。

## 3. 每组内容实际做了什么

| 组 | 真实读取/前处理 | 独立检查产物 | 尚未接入消费者 |
|---|---|---|---|
| 输入与metadata | 所有绑定artifact哈希、JSON解析；原视频完整解码及帧数/尺寸；选定实体/track核对 | artifact_inventory.csv | metadata不意味着新actor输入 |
| camera/time | 显式帧/时间连接，camera contract验证，source K与annotation一致性，fit平面K | camera_timeline.npz | v2 episode适配；旧v1路径仍使用自己的对应字段 |
| RGB帧 | 每张声明图像解码、尺寸/time/hash检查 | rgb_frames.csv、input检查图 | 图像/纹理encoder |
| depth/confidence | 两次流式逐帧读取；验证数量、K/平面，float32数值表示与P09一致 | depth_frames.csv、depth检查图 | 更完整的深度/可靠度编码 |
| masks/metric资格/P09 | 全raw/owned mask解码；每张eligibility NPZ分类/hash/affine验证；实际重放robust retained depth；投影P09点核对depth | mask_frames.csv、eligibility_frames.csv、surface_frames.csv、p09_*.npz | 非metric候选支持与完整点级质量编码 |
| 完整源手 | 读取完整source vertices/joints/faces/参数；source→camera/fit投影；每指12顶点；全部有效bridge/source顶点比较 | left/right_source_preprocessing.npz、三角线框/pad检查图 | 52维visual decoder、新pad特征；未执行licensed MANO LBS |
| appearance | 每帧调用懒加载前处理，验证raw mask副本，实际计算960宽mask、160点轮廓、distance transform | appearance_frames.csv、抽样appearance_*.npz与轮廓检查图 | silhouette目标/特征、动态outline对应 |
| typed interaction | 原始judgment字节与汇总配对；contact/pad/visibility/occluder/unknown编译 | interaction_timeline.npz、声明JSON、interaction_frames.csv | EpisodeStatic/actor、primary activation与真正display-valid/clear |
| RGB/PnP | 全边固定源点经过测量motion→目标相机重新投影；保存原生像素残差和matcher-plane误差；消费loop权重 | rgb_edges.csv、rgb_edge_*.npz、匹配/投影检查图 | motion tokens、candidate两端重投影因子 |
| measurement seed | 执行局部树/cycle/conditioning，保存独立R4 seed和不可用帧 | motion_qualification.json、measurement_seed.json | 旧环境还使用P09 centroid；未更改初始化/动作资格 |
| 完整visual geometry | full native replay、角色/拓扑/K/anchor核验；源camera→world；有seed时执行显式canonical binding及往返 | complete_visual_geometry.npz、visual_geometry_check.json、anchor全三角面QC | 旧env/render仍用pose mesh；未接共享scale/candidate状态 |
| P17 formal辅助 | factor行和依赖mask/NPZ读取；source first-surface查询结果输出 | factor_rows.csv、first_surface_queries.json | 新v2消费适配；旧contact patch准备明确跳过 |

`consumer_matrix.csv`以组为单位；`artifact_consumers.csv`逐个列出29项artifact的role、读取组、旧路径与尚未连接的消费者。每行明确 `v2_policy_connected=false`，不宣称已经改动actor/渲染。

## 4. 读取完整性与边界

- 所有源包绑定文件在读取前通过原有hash契约验证。
- 本轮进一步解码图像、流式读取depth/confidence、加载完整mesh/手几何、读取逐帧eligibility与appearance，不以“文件存在”代替内容消费。
- `source_reads.json`记录本次实际读取的源文件，完成时重新哈希，防止检查意外改写或源文件并发变化。
- 已有receipt会验证；仅有路径但没有独立producer hash的传递依赖，记录本次读取前后hash，**不伪造历史生产者证明**。
- 模型资产仍按bundle已有身份验证；本轮不加载权重或运行模型。
- frame/time原始PTS的严格CFR生产验证仍属于上游camera/raw contract；本入口额外检查视频解码数量/尺寸和已声明time连接，不重新声称独立认证所有原始PTS。
- 缺depth、不确定visibility和无RGB边保留为显式warning，不被填成确定观测。
- 完整图像/栅格都处理；只保存所选appearance distance栅格作为复核样本，其余帧保留计算统计，避免重复存整片大数组。
- depth流读取使用与P09一致的float32消费者表示，再计算eligibility数组hash；原archive字节hash单独保留。首轮测试曾暴露dtype不同造成hash不匹配，修正转换而非绕过hash。

## 5. 检查图的含义

`inspection_render.py`直接使用已读取/计算的数组：

- 源RGB/raw mask/owned mask/depth四联图；
- 完整source手三角线框、实际pad顶点、appearance轮廓；
- appearance anchor上全部可投影visual triangles的覆盖；
- RGB source/target对应与实际重投影点。

没有用box、质心或abstract panel冒充完整3D几何。图仍是**前处理QC，不是最终annotation**：

- 不进行物理occlusion z-buffer或非穿透判断；
- 跨近裁剪面的triangle没有执行精确clipping，计数中显式报告不可投影面；
- 非appearance anchor帧不擅自用最终轨迹来摆放完整visual mesh；
- RGB检查图最多展示前6条边/每边最多32点，但数值文件消费全部edge/points；
- source几何显示不应用policy display规则，尤其不因为`out_of_frame`判断而丢掉需要检查的源手。

合成夹具里的778顶点手表面是用于坐标与拓扑的弯曲数值网格，**不是licensed MANO模型输出，也不是人体形态质量证据**。所有持久化夹具和检查图明确标为SYNTHETIC。

## 6. 本次实际运行与结果

证据根目录：

`/mnt/truenas-user-home/kupingxin/rl_v2_observation_acceptance_20260921/`

### 最终合成v2包与独立检查

- 输入：`synthetic_v2_source_final/source_observations.json`
- 输出：`synthetic_v2_inspection_final/`
- 入口报告：`synthetic_v2_inspection_final/README.md`
- 机器报告：`synthetic_v2_inspection_final/inspection.json`
- 尚未接入消费者：同目录 `consumer_matrix.csv`、`artifact_consumers.csv`

实际结果：

- 29项artifact，11组检查（输入metadata与camera/RGB分组分别记录）。
- 2个非连续frame IDs：3、11，2张RGB均完整解码；目标11故意缺depth但有RGB测量支持。
- depth/confidence实际解码1帧，raw/owned mask共4张，eligibility raster1张，P09 retained support实际重放1帧。
- RGB测量1条、30个固定source-world点；全部重新投影并保存误差。
- 完整synthetic visual mesh：625顶点、1152个三角面；全部可投影，face topology保持。
- canonical→world往返最大误差约 `2.78e-17 m`。
- 2个contact_patch和2个visible_ownership行被读取；独立first-surface查询实际1556个样本；旧contact patch最近面membership未执行。
- 47个实际读取源文件完成前后hash检查，无改写。
- 输出39个检查文件，其中4张QC图；已阅读几何与RGB匹配图，看到源三角网格/pad、完整visual表面与投影对应由数据驱动。
- 状态为 `complete_with_measurement_gaps`：缺depth和一项unknown visibility是夹具刻意包含的测量缺口，不伪造成完整测量。

早期 `synthetic_v2_source/`、`synthetic_v2_inspection/` 保留作开发记录；最终目录有`_final`后缀。早期手表面退化为线状，后改成非退化三角网格；早期空factor fixture后补为实际first-surface查询。不要把早期空因子报告作为该机制的最终验证。

### 既有真实milk的版本边界

命令输入仍是此前完整milk的 `source_observations.json`，schema为v1。检查器返回exit1：

`inspection requires explicit v2; no silent promotion of v1`

结果保存在 `existing_milk_v1_check/failure.json` 和 `existing_milk_check.log`。这只确认没有误用旧包，**不算真实milk v2验收成功**。

### 测试

- 新增 `observations.test_inspection`：9项通过。
- 总回归：运行208项，**204通过，4项按真实模型/资产条件跳过**；约69.4秒。
- 总回归日志：`full_regression.log`。
- 训练器、PPO、policy/value、动作、奖励、env更新与renderer源码未改；观测检查测试将相关模块置为不可导入，仍可运行完整检查。
- 没有运行PPO测试、模型推理、Astra或新真实milk producer。

精确总回归由既有199项列表加 `observations.test_inspection` 组成，仍以 `CUDA_VISIBLE_DEVICES=''` 和数值库单线程执行；真实MANO/tokenizer/image-target optional开关unset。

## 7. 修改文件

新增：

- `observations/inspect.py`：独立CLI、完整消费调度、状态报告、消费者矩阵；
- `observations/inspection_io.py`：绑定文件只读访问、depth/confidence流式解码；
- `observations/inspection_render.py`：源几何与纹理对应QC；
- `observations/inspection_fixture.py`：明确synthetic的测试夹具构建器；
- `observations/test_inspection.py`：消费、CLI、缺能力、损坏、未知和源变更测试；
- 本文。

调整：

- `data/bundle.py`：新增 `load_observations()`，跳过旧solver contact-membership准备；原`.load()`保留。
- `observations/object/surface.py::MetricEligibility`：接受已流式计算的depth-row hashes，仍执行原输入/分类/affine/raster校验，不削弱资格检查。
- README/实施报告：新增入口与限制说明。

## 8. 四项验收的结论

| 用户要求 | 当前结论 |
|---|---|
| v2各项观测完整读取 | 完整profile检查已实现，29类数据有独立读取记录；已在非连续ID合成v2上运行 |
| 对应前处理实际消费 | 真实函数消费数组/图像并输出值；有源变化→typed输出变化、P09重放、first-surface、RGB投影和geometry往返证据 |
| 独立结果、不依赖PPO | 新CLI可独立运行，输出数值/图像/诊断；相关训练模块禁导入测试通过 |
| 每项未接入消费者明确 | 两级CSV列出旧路径、新观测消费及pending消费者；无下游生效的虚假声明 |

下一步真实数据验收的必要条件是：获得真实新版v2观测包（包含此前milk不存在的独立RGB源测量与完整appearance-native观测契约），在明确授权的计算环境生成或严格验证复用，再运行本入口。不能只改旧v1 schema数字完成此步骤。
