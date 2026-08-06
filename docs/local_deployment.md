# 当前用户 A800 本地部署与 V19 实跑记录

本文记录 `yiwen_research` 在当前账号下的隔离部署、复现入口和一次完整真实视频运行。合成 smoke、安装成功和预检只用于验证基础设施；真实物理结果单独列出。

## 1. 布局与隔离边界

- 仓库：`/mnt/user-home/kupingxin/ego_annotation`
- 本地部署分支：`local/kupingxin-v19-a800-deployment`
- 上游基线提交：`0c8e6a9ff1925caa5fa2665de116c404b5d39eee`
- 主环境：`/mnt/user-home/kupingxin/ego_annotation/.venv`（Python 3.11）
- UniDepth 环境：`/var/tmp/kupingxin/ego_annotation_envs/unidepth_sam2`
- HaWoR 环境根：`/var/tmp/kupingxin/ego_annotation_envs/hawor_work`
- TRELLIS 环境根：`/var/tmp/kupingxin/ego_annotation_envs/trellis_work`
- 稳定入口：仓库下 `.runtime/` 中的符号链接和 `paths.env`
- 模型：`/mnt/truenas-user-home/kupingxin/ego_annotation_models`
- 输入：`/mnt/truenas-user-home/kupingxin/ego_annotation_inputs`
- 输出：`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs`

`.runtime/`、虚拟环境、授权 MANO 文件、模型和运行输出均不进入 Git。NAS/CIFS 用于大文件，不用于安装 venv、Git 工作树或依赖 POSIX metadata 的操作。

运行只消费隔离输入中的 `input.mp4` 和 provenance；不会把旧预测、GT 或评估 sidecar 复制到输入目录。MANO 是许可资产，只能在用户明确授权的范围内复制和使用。

## 2. 主环境、SAM2 与 OWLv2

```bash
cd /mnt/user-home/kupingxin/ego_annotation
python3 -m pip install --user uv==0.12.1
RUN_GPU_SMOKE=0 scripts/setup_local_runtime.sh
source .venv/bin/activate
source .runtime/paths.env
```

若需要 GPU smoke，先用 `nvidia-smi` 选择不影响其他用户的卡，再执行：

```bash
GPU_ID=<空闲卡号> RUN_GPU_SMOKE=1 scripts/setup_local_runtime.sh
```

固定项：

- Python 3.11，PyTorch `2.12.0+cu130`，NumPy `1.26.4`；
- SAM2 提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`；
- SAM2 checkpoint SHA-256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`；
- OWLv2 revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`；
- OWLv2 权重 SHA-256 `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`；
- Open3D `0.19.0`、`iopath` 和 Open3D 元数据所需的 `ipywidgets` 已进入 `uv.lock`。

SAM2 可能提示未编译可选 `_C` connected-components 扩展。系统 `nvcc` 为 CUDA 12.6，而主环境 Torch wheel 为 CUDA 13.0；不要强行混编。官方回退会跳过可选孔洞后处理，真实 P07 已完成 150 帧传播。

## 3. 独立模型环境

### UniDepth

```bash
EGO_UNIDEPTH_ROOT=/var/tmp/kupingxin/ego_annotation_envs/unidepth_work \
EGO_MODEL_ENV=/var/tmp/kupingxin/ego_annotation_envs/unidepth_sam2 \
EGO_UNIDEPTH_MODEL_DIR=/mnt/truenas-user-home/kupingxin/ego_annotation_models/unidepth-v2-vitl14-52b349b5 \
UV_CACHE_DIR=/var/tmp/kupingxin/uv-cache UV_LINK_MODE=copy \
CUDA_VISIBLE_DEVICES=<空闲卡号> bash scripts/remote_setup_unidepth.sh
```

固定源码 `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`，模型 revision `52b349b514bd8b47642f67ac78cb7b5dc5c51dd9`，权重 SHA-256 `ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6`。该环境使用 Python 3.10、Torch `2.4.1+cu121` 和 NumPy `2.2.6`，不能与主环境合并。

### HaWoR / DROID / Metric3D

```bash
UV_LINK_MODE=copy scripts/setup_local_hawor_runtime.sh
```

固定 HaWoR 提交 `66c7d4108d58a716deccd192cb7645170cdc7bd7`，Python 3.10、Torch `2.6.0+cu126`。DROID 扩展导入时必须先 `import torch`，否则可能出现 `libc10.so` 查找失败。MANO 必须从当前用户授权模型根提供，脚本不含其他用户视频或环境的隐式默认路径。

### TRELLIS

```bash
UV_LINK_MODE=copy scripts/setup_local_trellis_runtime.sh
```

固定 TRELLIS 提交 `442aa1e1afb9014e80681d3bf604e8d728a86ee7`，模型 revision `25e0d31ffbebe4b5a97464dd851910efc3002d96`。真实运行使用规范 12+12 sampling steps；4+4 的极低步数曾产生 pathological sparse latent 并触发 spconv int32 限制，不能作为部署成功证据。

`/var/tmp` 可能被系统策略清理。稳定链接失效时重跑对应 setup；不要把可执行 venv 直接放到 CIFS/NAS。

## 4. 授权资产和输入

授权 MANO provenance：

`/mnt/truenas-user-home/kupingxin/ego_annotation_models/mano/AUTHORIZED_PROVENANCE.json`

当前副本：

- `MANO_LEFT.pkl`：3,821,391 bytes，SHA-256 `c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30`；
- `MANO_RIGHT.pkl`：3,821,356 bytes，SHA-256 `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`。

隔离输入：

`/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/hot3d_clip001851_keyboard_pinhole/input.mp4`

其 SHA-256 为 `7a9baf0553e5dcfb4411b6cfabbe3a734f815ee4c5b014bdd965b9e547ec2310`，规格为 1408×1408、150 帧、30 FPS、5 秒。目录中只允许 `input.mp4` 和 `INPUT_PROVENANCE.json`。

## 5. 隔离 runtime bundle、预检和启动

构建未来运行使用的 clean bundle：

```bash
python scripts/build_local_v19_runtime_bundle.py \
  --source-root /mnt/user-home/kupingxin/ego_annotation \
  --bundle-root /mnt/user-home/kupingxin/ego_annotation_runtime/v19_bundle_a800_local_deployment \
  --wilor-source /var/tmp/kupingxin/ego_annotation_sources/WiLoR \
  --mano-left /mnt/truenas-user-home/kupingxin/ego_annotation_models/mano/MANO_LEFT.pkl \
  --mano-right /mnt/truenas-user-home/kupingxin/ego_annotation_models/mano/MANO_RIGHT.pkl \
  --python /mnt/user-home/kupingxin/ego_annotation/.venv/bin/python \
  --replace
```

构建器仅复制 runtime spec 所需的脚本闭包、SAM2、WiLoR 和授权 MANO，并拒绝含其他用户绝对路径的 bundle。`--wilor-source` 若是 Git checkout，manifest 记录 `git rev-parse HEAD`；若是从既有 bundle 复制的 curated non-Git tree，应通过 `--wilor-source-revision <parent-manifest-revision>` 显式继承 immutable provenance，否则构建器记录 deterministic `tree-sha256:`，不能伪造 Git revision。运行前必须用 `scripts/preflight_local_v19_runtime.py` 检查 bundle hash、prompt/路径隔离、输入 hash、模型 hash、四个解释器导入、脚本 CLI 和 fresh run root。

当前最新 pose-eligibility/support-quarantine bundle（source commit `b7b97e6`）：

```text
/mnt/user-home/kupingxin/ego_annotation_runtime/v19_bundle_a800_b7b97e6_pose_gate_local3
```

其 Ego-Exo4D raw-v2 preflight 位于：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/runtime_preflight_egoexo4d_tire_lever_pose_gate_b7b97e6_raw_v2.json
```

Preflight 已通过，但 fresh run root 未创建，也未启动同 RGB 的第二次 full run；该 bundle 只准备 future mechanism-ablation 或取得 VRS calibration 后的新实验。

启动器参数依次为输入、fresh run root、case id、对应 preflight report：

```bash
scripts/launch_local_v19_runtime_agent.sh \
  /mnt/truenas-user-home/kupingxin/ego_annotation_inputs/hot3d_clip001851_keyboard_pinhole/input.mp4 \
  /mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/<fresh-run-id> \
  hot3d_clip001851_keyboard_pinhole \
  /mnt/truenas-user-home/kupingxin/ego_annotation_outputs/<matching-preflight-report>.json
```

启动器会校验 preflight 中的 bundle、输入和 run root 与本次命令逐项一致；run root 已存在时拒绝覆盖。每个 GPU-heavy phase 前仍须重新检查 `nvidia-smi`，不能假定 P02 选中的卡一直空闲。

## 6. 已完成的真实 V19 运行

权威运行根：

`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2`

P00–P21 已闭合，P20 恢复 sentinel 为 `exit_code=0`。目标为 **keyboard body only**，排除手、袖子、桌面、鼠标、手机及其他物体。关键结果：

- UniDepth：150×1408×1408 metric-depth archive，焦距汇总 `537.0213035603387 px`；
- HaWoR：左右手各 150 帧 metric MANO 候选；
- OWLv2/SAM2：键盘 identity 150/150 帧；
- anchor：frame 109；
- TRELLIS raw completion：137,674 顶点、275,328 面；
- 适配后的键盘 mesh：12,822 顶点、22,758 面；
- 150 个对象 pose rows；visible observed-to-mesh median residual 约 3.84 mm；
- 300 个 canonical hand rows 均保留 `hawor_npz_metric_mano_preserved`。

P03 曾因 960×960 review RGB 与 1408×1408 source-coordinate depth 直接混合而失败。修复只 resize 可视化色彩图，不改变 depth archive 或 intrinsics。P20 曾因 CIFS 拒绝 `copy2 → copystat → utime` 而失败；修复为同目录临时普通文件的字节复制、size parity 和原子替换，不复制 POSIX metadata。历史 blocker 均保留，resolution 通过 append-only event 记录。

## 7. Canonical 视频和审计

- `renders/v19_overlay.mp4`：9,847,322 bytes，960×1038，SHA-256 `2511c6d635334b860c4a29e0e1c4235d0a829e45979dc936cf18423b8bd91bd8`；
- `renders/v19_world.mp4`：8,922,247 bytes，1280×798，SHA-256 `f2944ec9d4eecb2c0ed50600f33ba7c6154b05d52a2e2f849c0a9b0f7cecd9be`；
- `renders/v19_side_by_side.mp4`：7,843,068 bytes，1920×618，SHA-256 `509d08b4b8f29b493672ca258913f9667e557dff4d6f8b5cd7f778e0561048ae`。

三者均为非符号链接、非空普通文件，150 帧、30 FPS、5 秒；额外高度来自发布 banner。P21 contact sheets 位于 `renders/review_frames/P21_canonical_visual_consumption/`。

结构/有限值/解码审计：

`FINAL_RUNTIME_AUDIT.json`

agent 视觉与物理证据：

`state/v19_agent_evidence.md`

## 8. 必须保留的科学边界

可接受结论是：**完整时长的键盘刚体 mesh-pose 注释，外加被明确标为不确定的 metric MANO 假设**。

不能声称：

- TRELLIS completion 是精确 CAD；其键帽细节、隐藏背面和厚度未被验证；
- `corrected_temporal_rigid_pose_graph` 名称代表发生了实际修正；P15 报告零 pose delta、无 nonpenetration gain；
- 手和键盘的 metric contact 已闭合；overlay/world 中 MANO 与可见手存在可变偏移，canonical contact patch 权重为零；
- nonpenetration 已证明；completed/sign mesh 非 watertight，没有有效 signed distance；
- 这些预测是 GT 或评估结果。

合成 smoke 仍保存在 `deployment_smoke_sam2/`，只表示部署链路可执行，不属于上述真实物理结果。
