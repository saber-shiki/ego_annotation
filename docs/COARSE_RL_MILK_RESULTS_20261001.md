# 最新 milk R6 预处理接入 PPO：1024-step 结果

日期：2026-10-01。任务：将最新 milk 预处理产物接入现有训练后端并查看效果。

## 结论

**显式训练接口已接通并真实跑完；当前配置没有获得整体目标改善。** 最佳已训练 checkpoint 是 256 steps，但仍劣于 R6 零动作。256 的物体轮廓诊断有小幅改善；1024 的轮廓和训练目标退化。保留 R6 零动作作为基线，不以任何本轮 checkpoint 替换默认结果，不盲目延长训练。

这是单视频自身优化实验，不是泛化、收敛、GT 精度或物理接触认证。R6 原本是实验性 mask 修正，本次没有将其提升为生产默认。旧 v1/v2 保护未删除。

## 实际接入的机制

- 输入：`/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_mask_alignment_20260929_r6/milk_observation_v2_20260921_r2/coarse_scene.json` 及 SHA 绑定 NPZ、原始 v2 源 bundle。
- R6 的完整 150 帧物体 SE(3)、239430 顶点/478880 三角面、固定统一尺度作为动作残差初值。相机/K、源 MANO/参数、原始深度/P09 和源姿态/metric 有效性保持不变；补出的位姿不变成 metric 观测。
- 新 `CoarseTrainingDataset` / `CoarseAnnotationEnv`：551 维观测、118 维动作，包含 typed 可见性/接触/指垫、原始/推断状态、因果 RGB 摘要及物体修正历史。不是 8 帧序列编码器。
- 独立源 RGB 旋转/双端重投影接入现有数学核；未启用 noisy RGB translation edge。按完整新 mesh 重绑定 unsigned 接触 prior，绝不赋予生成面 signed authority。
- 旧 480 维 checkpoint 被兼容性检查拒绝；从新策略训练，未加载旧权重。
- 原 `formal_annotation` 因子比例、reward_scale=1e-5、PPO 超参数、118 维 decoder 不变。没有重跑感知、Astra 请求或安装包，没有 GT/CAD/历史 RL 候选输入。

## 运行与检查

服务器：`dexrobot-Standard-PC-Q35-ICH9-2009`（A800），现成 ego_skrl Python，CPU-only、单线程。seed=0，split=all，连续 1024 steps，每 256 steps 独立进程执行完整 150 帧确定性 rollout，冻结 preprocessor，不改变训练 RNG。

零动作完整回报符合 gamma=1 的 telescope 契约；R6 物体平移未变，源 MANO 零动作 replay 最大误差左 1.67e-7 m / 右 1.38e-7 m。手/物体/接触的受控动作和 RGB 两端都确实耦合 reward。

R1 在训练前停止：用一帧目标检查物体平移，但没有 source sigma prior 时，NN/RGB/时间相对项需要至少两帧，第一帧对象目标本来就不响应。修正为三帧 prefix 机械检查，在新 R2 快照/新输出重跑；没有修改奖励或超参数。保留 R1 零动作、trace 和 failure_analysis。

## 学习结果

最终 ledger J 越低越好，所有数字使用本轮同一目标；不能直接和旧输入/旧目标的绝对值做受控比较。

| steps | J | hand/joint_pixel_deadband |
|---|---:|---:|
| R6 零动作 | 2067.838616 | 0 |
| 256 | 36518.639344 | 34311.733832 |
| 512 | 54584.290577 | 52234.822 |
| 768 | 505273.055236 | 502802.201 |
| 1024 | 370758.518351 | 368349.946 |

1024 时该投影项占总目标约 99.35%。这是加权惩罚，不是 37 万像素或物理位移。

实际几何诊断：
- 256：物体最大位置修正 3.568 mm、最大旋转修正 .670°；左右手平均顶点修正 2.417/3.713 mm，关节投影偏移 p95 9.075/13.021 原图 px。
- 1024：物体最大位置修正 10.944 mm、最大旋转修正 2.145°；左右手平均顶点修正 4.667/3.799 mm，关节投影偏移 p95 15.200/14.581 原图 px。
- 所有 checkpoint 的有效关节 camera-Z 非正计数为 0：本次大投影惩罚不是简单的点落到相机背后/近零深度爆炸。
- 256 物体 point-to-plane/point-to-point 相对零动作略降，但 RGB、时间修正及手部约束损失增加。不能把局部项下降说成整体改善。

原始 SAM2 对完整 mesh silhouette 的同 256px raster IoU 中位数：

| 候选 | raw IoU median |
|---|---:|
| R6 零动作 | .832334 |
| 256 | .848723 |
| 1024 | .772536 |

这是 visible SAM2 vs full silhouette 的视觉诊断，非 amodal/GT/3D 精度；不可与此前 192/320px 指标混用。256 的轮廓有局部收益，不能抵消手部目标恶化。末期轮廓也变差。

## 渲染与视觉 QC

完整 RGB/固定-world 对比视频均为 150 帧、30fps/5秒，真实完整 native triangles 与 full MANO 驱动。父级独立重新解码两视频、复核四个 checkpoint/所有 rollout artifact SHA 和冻结 preprocessors。

审阅了覆盖 0–149 的三张连续 timeline 接触图，完整尺寸 0/30/60/90/120/143/148/149 对比图，以及固定-world 图。整体动作仍连续、物体和双手完整表面保留；256 的少量轮廓修正可见，但没有明确整体标注改善，1024 出现更大的手部/物体偏移。timeline 接触图是缩小图，不是逐帧物理精度认证。

## 代码与产物

工作区：`/mnt/user-home/kupingxin/ego_annotation_rl_preprocessed_milk`，分支 `feature/rl-preprocessed-milk-20261001`。原 dirty 预处理分支未修改。无 merge/push。机制入口：`data/coarse_training.py`、`envs/coarse_annotation_env.py`、`configs/ppo_coarse_annotation.yaml`；只给原 AnnotationEnv 增加观测 hook 和 adapter export provenance。

成功冻结快照：`/mnt/user-home/kupingxin/ego_annotation_runtime/rl_coarse_milk_20261001_r2`。

训练根：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_coarse_r6_learning1024_20261001_r2`
- `summary.json`、`controlled_actions.json`、`mechanism_preflight.json`、`old_checkpoint_rejection.json`
- `checkpoints/step_{0256,0512,0768,1024}/`：agent/provenance、完整 rollout、最终分项
- `renders/zero_best_last_overlay.mp4`：左 R6 零动作、中 256、右 1024
- `renders/zero_best_last_world.mp4`：同一固定-world 视角
- `renders/review_sheet.jpg`、`renders/render_measurements.json`
- `completed.json`：exit 0，source/code 运行中保持不变

独立审阅根：`/mnt/user-home/kupingxin/ego_annotation_experiments/coarse_rl_milk_20261001_r2`
- `final_geometry_review/geometry_diagnostics.json`
- `final_visual_review/timeline_*.jpg`、`world_*.jpg`、`verification.json`
- `final_regression.log`：180 项运行、176 通过、4 跳过
- `launch.sh`、`launch_contract.json`、`launch.log`、`exit_code`

## 下一因果问题

最新预处理已经能够驱动现有 PPO 的真实物理状态和观测/reward；前端接入并没有解决后端学习信号问题。下一步应在冻结本轮输入和目标的前提下，分解投影 deadband 的权重/归约、探索动作、优势/value 估计和 PPO 更新方向之间的关系。根因尚未证明；不删除投影因子制造改善，也不直接放大到更多 steps。
