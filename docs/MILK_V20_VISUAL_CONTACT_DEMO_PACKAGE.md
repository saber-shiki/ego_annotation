# Milk V20 Visual Contact Demo (2026-09-08)

本分支是 **2026-09-08 实际产出 `milk_hand_object_demo.rrd` 演示**所用的管线代码快照（worktree `milk_v20_visual_contact_demo_clean_20260908`，分支 `demo-milk-v20-visual-contact-20260908`）。

演示产物（不在本仓库，仅记录溯源）：

- `/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/demos/milk_visual_contact_20260908/`
  - `milk_hand_object_demo.mp4` — 原视频 vs 平滑表面叠加，150 帧 / 30FPS / 5s
  - `milk_contact_detail.mp4` — 局部 + 20° 3D 接触检查视角
  - `milk_hand_object_demo.rrd` — 世界系 SAM3D 物体 mesh + 左右 MANO + 相机，可在 Rerun 交互查看
  - `state/`、`reproducibility/`（设计文档、拟合报告、独立复核、渲染报告）

## 范围声明（demo-only）

这是 **离线视觉拟合 demo**，不是严格度量标注：`annotation_ready=false`，未消费 GT，未修改任何 formal P14/P15/D19 状态或旧 v16 归档。接触数值是拟合参考几何中的距离，非传感器确认接触。

## 本分支包含的关键修改

### 手物接触优化（hand–object contact）
- `5ef8933` **demo: fit and render full-video MANO object visual contact** — 核心 demo：真实 SAM3D 三角 mesh + 完整 778 顶点/1538 面 MANO LBS 表面重放源 HaWoR 参数；修复 left shapedirs-X 与 flat-mean 约定；在 31 个关键帧上优化有界的关节/根/相机原点相似度修正 + 物体 SE(3)，插值到全 150 帧。接触对应来自真实手部表面片与最近/射线命中物体三角形，非质心或 2D 标记。代表性帧拇指 pad 间隙从 ~6–42mm 改善到亚毫米级。
- `f6af8a3` **first-hit and silhouette image factors for the P15 SE(3) graph** — 新增 `scripts/build_p15_first_hit_silhouette_factors.py` + `scripts/compare_p15_image_term_pose_shifts.py`，并扩展 `solve_v19_rigid_object_pose_graph.py`。
- `1dd2d17` **fix: align SAM3D visible surface with true first hits** — SAM3D 可见表面对齐到真实 first-hit。
- `aa36439` / `5d42c67` / `f22876a` / `c49e028` — prediction-only late-window SE(3) 修复管线、SAM3D MANO 相机位姿 demo 模式、RRD 记录 MANO 世界关节/骨架、跨 pose 端点传输固定 RGB-D 材质观测。

### Mask 错误扣除修复（mask mis-subtraction）
- `3cad519` **research: exclude hand-unknown pixels from silhouette coverage factors** — 从 silhouette 覆盖率因子中排除手部未知像素，避免把被手遮挡/未观测区域错误计入物体 silhouette 覆盖惩罚。

### Demo 基础设施
- `scripts/fit_v20_visual_contact_demo.py`、`scripts/render_v20_visual_contact_demo.py`、`scripts/v20_demo_geometry.py`
- `docs/V20_VISUAL_CONTACT_DEMO_DESIGN.md`、`docs/V20_VISUAL_CONTACT_DEMO_RESULTS.md`
- 测试：`tests/test_v20_visual_contact_demo.py`、`tests/test_v20_demo_runner.py`（14 项全部通过）

## 验证

```bash
python -m unittest discover -s tests -p "test_v20_*demo*.py" -v
# Ran 14 tests — OK（相机往返/halfpixel 内参、相机原点尺度投影不变性、
# 单位旋转处非零梯度、真实三角形接触/共享场景遮挡、torch/NumPy 导出变换一致性）
```

## 溯源

- code commit: `bf2d2ed146518b6e566d7fd4da5f8d032d20a8b4`
- 基线: 包含 `research/milk-v20-generated-first-hit-clean-20260906`（f017927）及更早的 milk V20 pose-repair 工作。
- 本分支初始为本地 demo 提交，2026-09-08 推送到 `origin`（DexGEM-Lab/ego_annotation）。
