# V19 本地 4×RTX 4090 复现运行报告（2026-07-30）

## 1. 结论摘要

本日完成了 V19 两条互斥物理分支的本地复现检查，但两条分支的最终状态不同：

| 分支 | 输入案例 | 最终状态 | 结论 |
|---|---|---|---|
| 刚体 | `20260105_1142_Recf97b_P0_Sf476d9_task_7`，白色陶瓷碗 | P00–P21 全部执行完成，wrapper `rc=0`，三路完整视频已发布 | 执行与渲染合同通过，但物理认证未通过；几何、部分位姿、接触和非穿透仍有明确失败或不确定性 |
| 非刚体 | `20251120_1535_Reca14d_P7_S3f2173_task_4`，两块清洁布 | P00–P06 完成；P07 修复重跑超时并记录 blocker | 尚未形成完整 P10d/P19d/P20/P21 结果，不能宣称非刚体分支端到端成功 |

因此，当前可以确认：

1. 刚体分支已经真实走通到完整时长 overlay/world/side-by-side 视频，并完成 P21 视觉消费；
2. 刚体视频中的可见标记确实来自 mesh、全时序 object pose、metric MANO 和交互状态，而不是方框、质心、mask 或静态占位图；
3. 刚体结果仍是带显式失败和不确定性的物理假设，不是完整物理认证通过；
4. 非刚体 P10d/P19d 已通过真实数据小范围 smoke test，但正式 task4 run7 仍停在 P07；
5. V19 双分支整体认证尚未完成。

## 2. 运行合同与边界

本次运行遵循以下边界：

- Pi 是唯一运行时智能 harness；P00–P21 是阶段编号，不是独立 agent；
- Python 脚本仅承担测量、优化、渲染和导出；
- 未下载安装依赖、未新建环境；
- 未使用 GT、人工任务标注或语义 action JSON 作为预测输入；
- 刚体和非刚体分别使用独立输入、独立 fresh run root；
- 没有把清洁布强行送入静态 mesh + SE(3) 刚体路径；
- 没有把刚体碗用于验证 P10d/P19d；
- 没有在缺少 mixed-state compositor 的情况下发布混合刚体/非刚体 canonical 视频；
- NAS 视频先复制到本地，避免 HaWoR 在 NAS 源目录生成旁路产物；
- 运行期间使用冻结 bundle，未在执行中途修改 bundle。

## 3. 环境与预检

### 3.1 计算环境

- Compute profile：`local_4x4090`
- GPU：4 × NVIDIA GeForce RTX 4090
- Pi provider/model：`dexgem-responses / gpt-5.6-sol`
- Thinking level：`max`
- 刚体 run 初始 GPU：物理 GPU 3
- P12 用户授权后的 phase-level GPU：物理 GPU 2

### 3.2 预检结果

刚体 run1 和 task4 run7 的 preflight 均为 `ok`，无 failure，只有两项已知 warning：

1. core 环境无 `pip`；依赖状态由已有 uv/provisioning 记录覆盖；
2. shared OWLv2/TRELLIS 环境的 `pip check` 不干净，但所需 import 实际成功。

这两项没有阻止测量或推理启动。

## 4. 输入与 provenance

### 4.1 刚体案例

- Case：`20260105_1142_Recf97b_P0_Sf476d9_task_7`
- 本地输入：`/mnt/user-home/kupingxin/ego_annotation_local/inputs/20260105_1142_Recf97b_P0_Sf476d9_task_7/input/raw_video.mp4`
- 编码：H.264
- 分辨率：1920×1080
- FPS：30
- 帧数：1,830
- 时长：61 秒
- 文件大小：54,427,467 bytes
- SHA-256：`eeaa8db73bea6ad41c90ad61e890325f66d8db75dd48848282aee05d89185057`
- Fresh run root：`/mnt/user-home/kupingxin/ego_annotation_local/outputs/v19_repro_20260105_1142_Recf97b_P0_Sf476d9_task_7_run1`

NAS MP4 与本地副本 SHA-256 一致。相邻的任务/action 语义 JSON 未作为预测输入。

### 4.2 非刚体案例

- Case：`20251120_1535_Reca14d_P7_S3f2173_task_4`
- 本地输入：`/mnt/user-home/kupingxin/ego_annotation_local/inputs/20251120_1535_Reca14d_P7_S3f2173_task_4/input/raw_video.mp4`
- 编码：H.264
- 分辨率：1920×1080
- FPS：30
- 帧数：3,030
- 时长：101 秒
- 文件大小：99,881,521 bytes
- SHA-256：`a8a190cac4b253f0b13d42261327f33eaa7be40ff6f41ec23d7ddeb917e34e19`
- Fresh run root：`/mnt/user-home/kupingxin/ego_annotation_local/outputs/v19_repro_20251120_1535_Reca14d_P7_S3f2173_task_4_run7`

## 5. 本日关键实现修复

### 5.1 非刚体 contact/nonpenetration archive 对齐

修复了 `scripts/build_v19_deformable_surface_state.py::append_object_arrays()`：

- 分离 `contact_correction_world_m` 与 `nonpenetration_correction_world_m`；
- 不再把 contact correction 同时写入两个 archive 字段；
- 新增 terminal offset 与 `contact_offsets[-1]` 行数一致性断言。

验证时使用不同的 synthetic correction 向量，两字段均保持 `(1, 3)` 形状且内容不同。

开发工作树和冻结 bundle 中该脚本 SHA-256 均为：

`4795706173cd016f659a6bc242edb6fd11e68c5079b7466183a4cd0fa2b15549`

### 5.2 稳定 TRELLIS 路径

P12 固定使用：

```text
--spconv-algo native
--decode-mode staged_mesh_only
```

该路径规避了历史 legacy CUDA OOM 和 auto-spconv SIGFPE，并保留真实 TRELLIS mesh 推理。

### 5.3 P20 视频兼容性修复

初始 canonical 视频由 OpenCV `mp4v` 编码。文件本身可被 FFmpeg 完整解码，但 MPEG-4 Part 2 / `mp4v` 不被许多浏览器、VS Code preview 和系统播放器支持。

本日采取了两层修复：

1. 对当前结果追加三份 H.264/AVC、`avc1`、`yuv420p`、faststart 兼容视频，不覆盖原始 canonical 文件；
2. 在本地开发工作树中把 `scripts/publish_v19_render_artifact.py` 改为 FFmpeg raw-BGR pipe → `libx264`，并在 canonical publication 前执行精确 ffprobe gate。

开发工作树 publisher SHA-256：

`f2ffba6febfa6d8e61decc9f4a8bdb8846ec7aebacbdb915d64aca87931d08d3`

冻结 runtime bundle 未被修改。

## 6. 刚体 run1 阶段结果

### 6.1 P00–P10：测量、跟踪与分支

| 阶段 | 结果 |
|---|---|
| P00/P01 | 验证 fresh root；提取 1,830 个 960×540 manifest frame |
| P02 | 本地 RTX 4090 host probe 通过 |
| P03 | UniDepth 生成完整 1,830 帧 metric depth；NPZ 约 1.84 GB，float16 NPY 约 7.59 GB |
| P03b | 固定内参：`fx=fy=1145.9494180952997`，`cx=967.88671875`，`cy=536.40234375` |
| P04 | HaWoR metric world MANO：左手 1,830/1,830 有效，右手 1,813/1,830 有效；DROID capacity 1024 |
| P05 | 选择单一目标 `white_ceramic_bowl`，rigid 概率 0.98；邻近白盘、空碗、筷子、锅具和食物不并入目标 body |
| P06 | 修复 OWLv2 相似白色餐具 instance switching；最终 prompt frame 为 30、60、120、1200、1260、1710 |
| P07 | SAM2 完成 0–1829；1,185 个 visible-mask frame，其余帧显式不可见 |
| P08 | 生成 base annotations、HaWoR→MANO bridge 和 base physical state |
| P09 | 视觉选择 anchor frame 1437；anchor extent 约 0.200×0.271×0.240 m |
| P10 | rigid branch，confidence 0.98 |

### 6.2 P11–P13：刚体几何

P11 生成 anchor evidence bundle 和 object-isolated RGBA crop。

P12 第一次按“16,000 MiB free 且 utilization ≤20%”策略未找到合格 GPU，记录历史 blocker。用户授权将 utilization 改为 advisory 后，立即重新 probe：

- 选择物理 GPU 2；
- free memory：21,061 MiB；
- utilization：28%；
- `CUDA_VISIBLE_DEVICES=2`；
- TRELLIS runtime 约 92.8 秒；
- 峰值 reserved memory 约 6,986 MiB；
- raw TRELLIS mesh：38,954 vertices / 77,904 faces；
- mesh SHA-256：`5ec1d4ccf1446cad670ca4d1caf90a5ead463fed0fcbb869cefc91b2909d83ff`。

P13 metric adaptation：

- observed→TRELLIS mean：约 5.90 mm；
- median：约 4.00 mm；
- p95：约 14.37 mm；
- accepted completed mesh：30,442 vertices / 58,052 faces；
- observed depth-supported faces：6,587；
- accepted hidden TRELLIS-prior faces：51,465；
- excluded unsupported observed faces：241。

### 6.3 P14–P18b：位姿与交互

- P14：visible-frame object pose fitting 完成；
- P15：full-timeline rigid pose graph 完成；
  - corrected direct rows：1,141；
  - explicitly uncertain completion rows：689；
  - nearest-visible hold：431；
  - interpolation：258；
- P16：MANO/object constraint state 完成；
- P17：1020–1155 contact/visibility ownership factor 完成；
- P18：CPU interval optimization 正常完成；
- P18b：canonical interval state 保留 source metric HaWoR MANO，共 272 个 hand/frame row，median joint shift 为 0 px；contact 仍为 uncertain。

两份 nonpenetration 相关 mesh 都不是 watertight；3,643 个 hand/object measured pair 上报告的零穿透顶点不能作为 signed-volume nonpenetration 证明。

### 6.4 P19–P21：完整渲染与发布

- P19a：生成约 40.1 MB rigid render state；
- P19b：完成 diagnostic full-duration overlay/world/side-by-side；
- P19c：完成 presentation rerender；
- P20：发布 canonical 名称；
- P21：消费 0–1829 的代表帧和三路完整视频，追加最终物理评估；
- 历史 P12 blocker 被正式写为 resolved；
- wrapper 正常退出，最终 exit 内容为 `rc=0`。

## 7. 刚体视频与结果身份

### 7.1 推荐播放的 H.264 兼容视频

| 输出 | 分辨率 | 帧数/FPS/时长 | 大小 | SHA-256 |
|---|---:|---:|---:|---|
| `renders/v19_overlay_h264.mp4` | 960×618 | 1,830 / 30 / 61 s | 42,500,955 bytes | `6bf21025e5061dfde496421a89087d6ee052a09a01c489347ae72b6bdf88464c` |
| `renders/v19_world_h264.mp4` | 1280×798 | 1,830 / 30 / 61 s | 38,899,391 bytes | `f5e940425b5e82b2ce5d3573b1331c5dccabc396457657d118b9147da888b82b` |
| `renders/v19_side_by_side_h264.mp4` | 1920×618 | 1,830 / 30 / 61 s | 61,856,928 bytes | `18edba09f3af74216f6737890174cb72a5b33c0e9c704f157fc9483d2e4122be` |

三份兼容视频均满足：

- H.264 High@4.1；
- MP4 tag `avc1`；
- `yuv420p`；
- `moov` 位于 `mdat` 前，faststart=true；
- ffprobe counted frames=1,830；
- FFmpeg `-xerror` 全量解码 `rc=0`，错误输出 0 bytes。

### 7.2 机器可读结果

本地结果文件：

- `state/runtime_result.json`
  - status：`completed_with_unresolved_physical_failures`
  - command exit：0
  - active blocker：null
  - SHA-256：`757f57b36160f7df4a6a5528012b6a6ab91c1ba93abeb1213e65ab8ed602d292`
- `state/video_compatibility_repair.json`
  - SHA-256：`201af5927dc04f7a7160f0cfd9af2e48c28fe87c8ae95d0c016bbea6fabb86da`
- wrapper exit：`rc=0`

大体积视频、measurement archive 和机器本地 JSON 未提交到 Git；本报告记录其本地相对位置、大小和内容身份。

## 8. 刚体最终物理评估

### 8.1 已工作的机制

1. UniDepth、calibration 和 HaWoR 在完整时间线上提供统一 metric camera/MANO convention；
2. renderer 从 58,052 个 accepted mesh face 栅格化刚体，而不是使用 point cloud、mask、box 或 centroid 替代；
3. source→render intrinsics 使用正确的 `[0.5, 0.5]` 缩放；
4. frame 1200、1437、1500 等 direct-pose frame 能把 body 放到被跟踪的目标碗上；
5. overlay、world、side-by-side 消费同一 render state；
6. P18b 没有把未验证 contact displacement 提升为 canonical MANO，而是保留 source metric HaWoR MANO；
7. canonical presentation 明确显示 `uncertain` 和 `contact uncertain`，没有把未解决状态伪装成确定接触。

### 8.2 失败或未解决的机制

1. hidden completion 外观更像封闭 dome/slab，而不是被视觉证明的中空陶瓷碗；
2. completed/sign mesh 非 watertight，signed-volume nonpenetration 未被证明；
3. frame 30、1020–1050、1710 存在明显 pose/extent overpaint 或偏移；
4. 689 个无直接观测 completion row 中，frame 900、1829 等 refrigerator/object-absent 视图可出现物理上无支持的位置；
5. contact closure 和 exact contact ownership 未解决；右手主要通过筷子与碗/内容物发生 tool-mediated relation，不接受为直接手-碗接触；
6. 相似白色餐具在 grounded/direct evidence interval 之外仍存在 identity uncertainty。

因此应区分：

```text
phase execution contract: passed
canonical render publication contract: passed
physical certification: not passed
```

## 9. 非刚体 task4 run7 状态

### 9.1 已完成

- P00–P06 完成；
- P03 处理全部 3,030 帧；
- P04：左手 3,030/3,030，右手 3,029/3,030 valid；
- P06 修复 gray-rag query 最初误选 green cloth 的问题；
- 第一轮 P07 SAM2 推理曾完成全部 3,030 帧；
- 视觉 self-check 发现 green cloth 的 frame 2500、2750 prompt 选择 bedding/support 或非目标区域，因此拒绝第一轮结果；
- P06 删除无效 green-cloth prompt 后，P07 从 clean output root 重跑。

### 9.2 当前 blocker

clean P07 repair rerun 在 7,200 秒 execution limit 到达前未形成完整 required artifact contract。最后观察到传播约到 source frame 2170/3029。

缺少：

- `qc_sam2_multiobject_points.json`；
- `green_cleaning_cloth/sam2/sam2_track.json`；
- `gray_striped_cleaning_cloth/sam2/sam2_track.json`；
- `sam2_multiobject_overlay.mp4`。

Blocker：

`state/runtime_blockers/P07.json`

下游策略保持正确：P08 及以后阶段没有从 partial/rejected mask 继续。截至本报告冻结时，task4 的 detached wrapper/Pi 仍保留用于后续恢复与审计，但没有活动的 SAM2 子进程，也没有生成 wrapper exit 文件。

### 9.3 P10d/P19d 机制 smoke test

使用历史 run6 的真实 measurement evidence 做了三个独立 smoke：

- frame 918：green cloth 296 vertices / 445 faces，64 个对齐 contact row；
- frame 3013：gray cloth 270 vertices / 396 faces；缺失 green cloth 保持 unresolved；
- frame 917–919：green cloth 716 vertices / 1,054 faces，192 个对齐 contact row 和 KLT factor；gray cloth 有 temporal-acceleration evidence。

三个 smoke 均生成实际 deformable triangle-surface overlay/world/side-by-side 视频，证明 P10d/P19d 实现链可以消费真实深度、mask、HaWoR MANO、material track 和 contact/ownership state。它们不是 task4 full-duration certification 的替代品。

## 10. 验证与测试

本日执行并通过：

- `python -m compileall -q scripts`；
- `python -m py_compile scripts/publish_v19_render_artifact.py`；
- `git diff --check`；
- deformable archive distinct-vector synthetic alignment test；
- P10d/P19d 三组真实数据 smoke test；
- 刚体 input FFmpeg full decode；
- P19b/P19c/P20 三路 1,830-frame timeline 检查；
- 三份 H.264 compatibility output 的 ffprobe count-frames、codec/tag/pixel-format、faststart、SHA-256 和 `-xerror` full decode；
- publisher 六帧 synthetic end-to-end smoke；
- canonical real-copy publication 检查；
- odd `yuv420p` output dimension rejection test；
- `runtime_result.json` 中所有记录的主要 provenance path/size/SHA-256 交叉验证。

## 11. 当前认证状态

### 已达到

- 刚体分支 fresh local prediction 完整执行；
- 刚体全时长三路 state-driven 视频存在且可播放；
- wrapper `rc=0` 和机器可读 runtime result 存在；
- renderer/projection/direct-pose 的有效边界已被视频证据确认；
- 未解决物理变量在视频和报告中显式保留。

### 尚未达到

- 刚体 hidden bowl topology 的可信物理几何；
- 全部 held-frame pose/extent 对齐；
- watertight signed-volume nonpenetration；
- accepted contact closure/ownership；
- task4 clean P07 full completion；
- task4 P08、P09、P10、P10d、P19d、P20、P21；
- 双分支 V19 end-to-end certification。

## 12. 后续操作

1. 为 task4 P07 使用 durable job handle 和足够的 execution allowance，重新运行同一 named command；
2. 完成后重新执行强制 mask/overlay 视觉 self-check，禁止恢复被拒绝的第一轮 mask；
3. 继续 task4 P08、P09、P10、P10d、P19d、P20、P21；
4. 对刚体 P13 hidden geometry 增加中空 vessel topology/free-space consistency 约束；
5. 修复 held-frame object pose/canonical-origin/extent 对接，并优先检查 frame 30、1020–1050、1710；
6. 在 watertight sign geometry 和可靠 ownership 存在前，不把零穿透顶点提升为 nonpenetration 成功；
7. 在两条分支都完成完整视频和最终视觉 gate 后，再声明 V19 branch-complete readiness。

## 13. 仓库提交边界

本报告面向 `yiwen_research` 分支。Git 提交只包含这份中文运行报告，不包含：

- 本地大体积运行产物；
- 视频、depth archive、MANO archive 或 mesh；
- 当前工作树中已有的 staged/modified/untracked V19 实现文件；
- task4 的 96 MB 本地目录。

这样可以避免把未整理代码、其他 staged 变更或机器专属产物混入报告提交。相关实现变更应在后续独立 review 后分批提交。
