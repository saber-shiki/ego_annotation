# HOT3D Milk shared signed-geometry recovery design

## 目的

初始实现位于 `feature/hot3d-milk-shared-signed-geometry-20260818`；局部 authority
修复位于独立分支 `fix/hot3d-local-signed-authority-p18-support-20260819`。目标是用
Milk fresh run 实际尝试恢复 signed object geometry，并让通过验证的物理体进入一次共享 P18。
本实验不把 SAM3D 或 TRELLIS 生成面提升为物理面；signed proxy 必须由 prediction-side
object-owned mask、active-K metric depth、D15 observed-only pose 和 direct P14
observations 独立构建。

## 因果假设

当前 P18 没有使用完整优化 MANO 的直接原因不是优化器缺失，而是 observed mesh
partial/non-watertight，无法定义 inside/outside。若用多帧 direct observed poses 将
mask/depth 约束变换到同一个 D15 canonical frame，并对 voxel 进行 silhouette +
free-space carving，则 Milk 的长方体/厚实体应能形成一个保守、winding-consistent
volume。若该体通过 first-hit depth、mask ownership、视角覆盖和拓扑检查，signed
P18 应能产生可复现的 full-778 MANO candidate；若失败，必须保留具体的 voxel/视角/
free-space 失败，而不能把 mesh flag 改成 ready。

## 物理输入与禁止输入

允许：

- fresh P09 object-owned masks 及其 exact source→mask affine；
- fresh P03 active-K depth NPZ；
- fresh P14 direct visible pose rows 和 D15 full timeline pose authority；
- fresh P04 MANO silhouette，仅用于把手遮挡区域标成 unknown；
- fresh D14 observed-only mesh，仅用于 canonical bounds/pose-frame binding。

禁止：

- SAM3D raw/aligned/generated mesh；
- TRELLIS raw/completed/generated mesh；
- evaluator/reference depth、CAD、GT pose、历史 prediction artifact；
- 用生成 mesh 的 silhouette 或 pose 修正 D15。

## Shared sign proxy

新脚本 `scripts/build_hot3d_shared_signed_geometry.py`：

1. 只选择 P14 `fit_to_*` 且 `rigid_pose_observation_eligible=true` 的 direct rows；
2. 以 D15 canonical observed mesh 和 fresh observed surfels 建立 bounded voxel grid；
3. 对每个 direct view：
   - source-K 投影到 source depth grid；
   - exact source→mask affine 查询 object-owned/raw mask；
   - projected MANO/ownership holes 作为 unknown，不作为 free space；
   - observed depth 前方作 free-space carve；
   - owned mask 内、depth 后方的 voxel 获得 volume support；
4. 跨视角保留无 free-space contradiction 且有分布式 silhouette support 的 voxel，
   用小尺度 closing/fill-holes 得到 volume，Marching Cubes 生成 sign proxy；
5. 对完整 proxy face set（不能只抽样）用 Open3D first-hit ray 检查 front surface、
   mask/depth support、hand-unknown adjacency、held-out free-space contradiction 与
   face-center 到 observed surfel 的距离；
6. 输出 mesh/hash-bound per-face authority，至少区分 direct first-hit-supported、
   silhouette-only completion、hand-unknown adjacent、free-space-risk 和 unsupported
   topology closure；
7. watertight proxy 只提供 inside/outside topology。只有 nearest face 的
   `signed_distance_eligible=true` 时，D16/P18 才能产生 signed force；
8. `signed_geometry_ready=true` 还要求 MANO interaction-near 区域具有局部 authority，
   且不能存在超过上界的 unauthorized closure penetration。全局 support fraction、
   watertight 或 winding consistent 都不能单独晋升 signed geometry。

输出同时保留：

- `pose_hypothesis_mesh_labeled`：D14 observed canonical mesh，供 D15 pose frame；
- `collision_eligible_mesh_labeled` / `signed_geometry_mesh`：shared proxy，供 P18
  signed query；
- collision face labels、candidate/selected mesh 各自的 mesh-bound face-authority NPZ、
  voxel support NPZ、QC report/mesh。即使 candidate 未晋升，其逐面 provenance 仍保留。

## P18 promotion

P18 继续使用 `--no-optimize-object-translation`。当 completion report 明确绑定
shared signed proxy 且 readiness 为 true 时，closed proxy 可用于计算 sign，但 signed
factor 只可作用于 per-face authority 明确允许的局部物理 faces；object delta 仍必须
逐行精确为零。

P17/P18 还必须区分两种 mask/depth 语义：

- `constraint_eligible_entity` 用于排除 hand-owned object pixels，不能再拿它查询 MANO
  顶点是否具有 translation support；
- translation gate 使用独立的 P09 accepted first-surface 邻域 raster。它只说明 MANO
  附近有真实物体 first-hit evidence，可允许 root translation；它本身不是“手必须在
  物体后面”的单向 depth-order residual，也不得读取 hand pixel 下的 raw depth。

没有独立 translation support 的 row 在 solver 内冻结 translation，并在输出端 fail
closed。报告同时保存 raw optimizer candidate 与 published candidate；任一 output
translation gate 生效都会阻止 full-MANO 晋升。

求解器额外输出 full optimized MANO vertices archive。P18b 只有在 signed report、
full timeline、zero object delta、active-set closure、authorized/unauthorized topology
penetration、bounded 2D/depth/temporal residual 全部通过时，才把
P18 optimized full-778 MANO 作为 accepted render state；否则继续使用 source metric
MANO，并将 P18 samples 标为 uncertain yellow diagnostics。

## A/B 公平性

D15、camera、P17、P18/P18b、shared signed proxy 和 accepted MANO 在 SAM3D/TRELLIS
分叉前只构建一次。两个 backend 仍只允许改变 generated render geometry；生成面
保持 `pose/contact/collision/signed_distance_eligible=false`。

## 预期结果与判别

- signed proxy 通过：重跑 Milk 的两 backend 视频，蓝/橙色完整 MANO 应来自 shared
  accepted P18 full archive，另保留 signed penetration/contact diagnostics；
- proxy 拓扑通过但 free-space/first-hit 失败：保持 unsigned，记录具体 contradiction
  和渲染对比；
- voxel 为空或视角覆盖不足：不降低门槛，不发布 signed claim，下一步应改善 direct
  observation acquisition，而不是改 readiness metadata。

## 输出

实验仍必须产出完整 150 帧、30 FPS 的 SAM3D/TRELLIS camera/world/side/side-by-side
视频；D19 manifest 绑定 shared sign proxy、P18/P18b hash、accepted MANO source 和
所有输入 SHA256。