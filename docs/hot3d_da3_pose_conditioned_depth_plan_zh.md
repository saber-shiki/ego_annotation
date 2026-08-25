# HOT3D 官方 K + 固定 HaWoR 轨迹的 DA3 多视图深度方案

## 目标与唯一干预变量

本分支从 `e26b507` 的 HOT3D SAM3D/TRELLIS 双后端 release 基线建立。目标是将外部
sensor-conditioned UniDepth 深度替换为 Depth Anything 3 的 pose-conditioned 多视图 metric
depth，同时冻结：

- 官方 prediction-side pinhole K；
- HaWoR 输出的 prediction-side metric camera trajectory；
- HaWoR/MANO、OWLv2、SAM2、object-owned mask 和 anchor 选择合同；
- SAM3D Objects 的 full RGB + binary object-owned mask 输入；
- SAM3D 内部 MoGe、配置、runner、seed 与 `pointmap=None`；
- TRELLIS 和双后端后续几何/渲染合同。

Checkpoint：`depth-anything/DA3NESTED-GIANT-LARGE-1.1`。License 为 CC BY-NC 4.0，
仅用于非商业研究/评估。

## 为什么 phase order 必须变化

DA3 第二种方案需要已知 K 和 camera poses：

- P03b 提供官方 K；
- P04 HaWoR 才产生当前 prediction-side `R_c2w/t_c2w`；
- 因而 DA3 不能继续占用原 P03 的时序位置。

本分支顺序为：

```text
P03b official camera/image-plane contract
  -> P04 HaWoR metric MANO + camera trajectory
  -> P03d DA3 pose-conditioned overlapping windows
  -> P03c exact depth-camera binding
  -> P05-P11 and unchanged SAM3D/TRELLIS tail
```

不得使用 HOT3D released camera pose；DA3 返回的 camera 只用于检查是否保留 fixed input，
不能成为新的 camera authority。

## 深度与栅格合同

`scripts/run_da3_pose_conditioned_full_frame_v1.py`：

1. 读取官方 V19 camera contract 的 `manifest_rgb` K；
2. 将 HaWoR `R_c2w/t_c2w` 组装成 OpenCV c2w 并求逆为 DA3 所需 w2c；
3. 以默认 16 帧窗口、4 帧 overlap 运行 DA3 Nested；
4. 传入固定 K/W2C、`align_to_input_ext_scale=True`、`use_ray_pose=False`；
5. 根据 DA3 返回的 processed-grid K，将 camera-z depth 逆映射到 exact `source_rgb` rays；
6. overlap 采用位置 taper × DA3 raw confidence 融合，并记录窗口间 depth disagreement；
7. 输出 provider-neutral P09 字段：`frame_idx/depth/confidence/source_size/intrinsics`；
8. 保留 `depth_provider/model/pose_conditioning/window/confidence` provenance。

DA3 `depth_conf` 是 higher-is-better confidence。P09 当前 first-surface ownership 需要
higher-is-worse error proxy，因此 P03d 写：

```text
error_proxy = 1 / max(raw_DA3_confidence, epsilon)
```

这只保持单调方向，不声称是经过标定的米制误差。

## 因果预测

1. 若 P09 surfel/CAD、P14/P15 pose residual 和最终 green observed-surface drift 同时改善，
   支持“单帧深度跨视角不一致是主导机制”。
2. 若静态背景改善，但手或被操作物体边界恶化，说明 DA3 静态多视图假设污染动态前景；
   下一步应做动态区域单帧 metric branch / 静态区域 multi-view 的 ownership split，而不是降低 gate。
3. 若 raw depth/CAD residual 改善但 P15 不变，说明 downsteam pose coupling/solver 是主导机制。
4. 若 overlap disagreement 大或 DA3 不保留 fixed input W2C/K，说明实现没有测试预定机制，应修正
   camera/window contract，而不是把结果解释成 DA3 模型能力失败。

## Provisioning 与 immutable bundle

Prediction runtime 不安装依赖、不下载权重。parent/deployment 阶段先固定：

```text
DA3 source commit: 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
DA3 model revision: b2359bdf726fb44ef62acca04d629dcf158053e7
```

环境必须能在离线模式下 import `depth_anything_3.api.DepthAnything3`；model 目录必须有
`DA3_RUNTIME_MANIFEST.json`、`config.json` 和完整 `model.safetensors`。bundle builder 对这些
文件逐个 SHA256，preflight 再校验 source commit、model revision、license 和文件 hash。

构建 bundle 时额外传：

```bash
--da3-repo /ABS/PATH/Depth-Anything-3 \
--da3-model /ABS/PATH/DA3NESTED-GIANT-LARGE-1.1-b2359bdf
```

case preflight 额外传：

```bash
--da3-python /ABS/PATH/py312-da3/bin/python \
--da3-repo /ABS/PATH/Depth-Anything-3 \
--da3-model /ABS/PATH/DA3NESTED-GIANT-LARGE-1.1-b2359bdf
```

任何 source dirty、revision mismatch、权重缺失/hash mismatch、环境 import 失败都必须阻断
launch，不能在 prediction run 内修环境或回退到未声明模型。

## Overlap fail-closed gate

多窗口输出只有在重复帧通过一致性 gate 后才能写正式 depth archive。默认要求：

```text
all overlap rows median(relative |z_prev-z_new|/z_prev): median <= 0.10
all overlap rows median(relative |z_prev-z_new|/z_prev): max <= 0.25
median best-scalar-aligned relative residual <= 0.10
```

该 gate 同时区分整体 scale 漂移与不能由一个 scalar 解释的 geometry/context 变化。失败时只写
`qc_da3_pose_conditioned_overlap_failure_v1.json`，不写可供 P09/P14/P18 使用的 depth archive；
provider-neutral adapter 还会二次拒绝 `overlap_consistency_passed=false`。

在 milk 150 帧技术验证中，原始 overlap relative median 为约 `0.622`、单帧最大约 `2.081`，
所以该配置必须被视为 negative result，不能靠 confidence blend 伪装成一致输出。

## 评估边界

Prediction run 不消费 GT。冻结 prediction 后，evaluation 才可使用 HOT3D CAD/pose/depth sidecar，
并至少比较：

- UniDepth + official K baseline；
- DA3 + official K + fixed HaWoR trajectory；
- evaluator-only oracle depth ceiling。

评估包括 raw depth、P09 surfel→CAD、P14/P15 translation/rotation、direct-pose coverage、
动态 foreground/hand-boundary 分层、最终完整视频漂移和 runtime/VRAM。
