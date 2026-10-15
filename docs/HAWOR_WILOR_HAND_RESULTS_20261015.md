# RL HaWoR / WiLoR 手部误差对照（2026-10-15）

## 结论与范围

**本次 milk 实测不支持把 HaWoR 的完整米制时序手部状态直接换成 WiLoR。WiLoR 的原生 2D 更准，但绝对 3D 和时间稳定性更差；只修相机接口不能解决绝对深度。适合研究作为额外图像/姿态观测，不应直接替换默认 RL 手部来源。**

这是组件替换对照，不是全管线新版、RL 重训或生产接入完成。原始 HaWoR、默认 R6、现有 RL 训练及其他工作树均未修改。新工作树：

```text
/home/kupingxin/ego_annotation_hawor_wilor_hand_eval
experiment/rl-hawor-wilor-hand-eval-20261015
base: 4d8bb65
```

仅当前 RL 使用的 P0014_84ea2dcc milk，原帧2370–2519，150帧 / 30Hz / 5秒。不是独立留出集，也不足以推出所有视频上某模型更好。两模型左右手各150有效帧，WiLoR无同侧重复；主表的共同覆盖和各自覆盖完全相同，没有只挑容易帧。

## 方法

- 使用冻结的 `milk_observation_v2_20260921_r2` RGB、HaWoR archive、完整源K及相机轨迹。WiLoR不读GT、GT框、GT手形、GT轨迹或人工手大小。最高detector score决定同侧候选，不填补缺测。
- 固定两分支相机轨迹。源K为 `[974.3447265625,974.3447265625,702.6607055664062,706.1102294921875]`，1408×1408图像。
- WiLoR在A800服务器 `dexrobot-Standard-PC-Q35-ICH9-2009` 的显式GPU6运行。独立快照、完整150帧、真实778顶点MANO及可重放rotation-matrix参数。反射左手的面绕序反转，明确保存反射矩阵，不把反射伪称SO(3)。最大参数/表面回放误差 `3.65e-8m`。
- 预测冻结后，独立 evaluator读取精确时间戳官方MANO及hand/headset/QA masks；150帧全部合格。官方MANO FK与GT缓存逐顶点最大偏差左右均0。
- 2D主指标为official19关节对官方MANO投影的平均欧氏误差，单位源像素，**不是独立人工2D标注**。本例所有被评分19点均在图内且正深度。
- 相机系MPJPE不拟合尺度、平移或旋转。世界指标只用一次首帧相机SE3、scale1；PA21允许逐帧Sim3，仅为诊断。腕相对MPJPE只去平移，不去旋转/尺度。加速度误差是完整关节世界位置二阶差分×30²，m/s²，只用连续有效三帧。

## 两种接入方式，均保留

旧 `observations.hand.wilor` 的 `joints3d_camera` / `vertices_camera` 实际未加 `cam_t`，是MANO局部几何，不能直接拿来对比绝对相机3D。此外其native full-image focal是27500px，不能当成本视频真实K。

1. **WiLoR direct-K**：保持模型预测的手形/姿态/弱相机scale-offset，只把full-image camera translation转换到真实square focal和principal point；不缩放手的解剖几何。它没有让native弱透视投影与近距离真实透视自动等价。
2. **WiLoR perspective**：先冻结以上模型预测；随后在新的预测目录中，保持所有MANO形状、方向、关节参数不变，只求每帧每手3DOF平移，让真实K下的透视投影拟合该网络自己的native 21点2D。无GT、无尺寸倍率、无旋转拟合、无时序平滑。300/300手帧数值求解成功；与自身2D的平均残差左43.78→23.99px、右43.40→25.96px，仍有不相容残差。

第二步不是用GT校准，也不新增独立测量。它是识别“相机接口还是手部状态本身”的必要对照。GT结果没有回流修改这两次预测。

## 定量结果（均值；左 / 右）

|指标，越低越好|HaWoR|WiLoR direct-K|WiLoR perspective|
|---|---:|---:|---:|
|真实K下19点3D投影误差 /px|25.18 /31.38|47.42 /45.19|**16.97 /17.97**|
|原生网络19点2D误差 /px|25.18 /31.38¹|20.73 /20.19|20.73 /20.19²|
|相机系MPJPE19 /mm|**17.32 /14.75**|39.56 /40.33|45.06 /66.31|
|世界系MPJPE19 /mm|**20.25 /12.96**|44.19 /44.51|49.49 /70.69|
|腕相对相机MPJPE19 /mm|11.27 /**11.48**|**7.97** /14.83|**7.97** /14.83|
|PA-MPJPE21 /mm|6.11 /**5.39**|**4.75** /5.43|**4.75** /5.43|
|完整posed778顶点相机MPVPE /mm|**16.91 /14.58**|39.58 /41.41|45.36 /68.01|
|腕点相机Z有符号偏差 /mm³|+7.80 /−13.75|+39.27 /+47.16|+43.01 /+71.59|
|世界关节加速度误差 /m/s²|**3.58 /4.12**|5.73 /6.00|6.01 /6.85|

¹ HaWoR这一行是同一米制MANO源K投影，并非额外独立2D detector。  
² 两种WiLoR接入方式使用相同native网络2D，不能算两份独立观测。  
³ 正值表示比GT更远，不能只比较有符号均值大小。MPVPE是完整**已摆姿态**对应顶点误差，不是官方rest-shape MPVPE。

双手等权汇总：

- HaWoR：真实K投影28.28px，相机MPJPE16.03mm，加速度误差3.85m/s²。
- WiLoR原生2D：20.46px，比HaWoR源K投影低约27.6%。
- WiLoR direct-K：相机MPJPE39.94mm，约为基线2.49倍。
- WiLoR perspective：真实K投影17.47px，低约38.2%；但相机MPJPE55.69mm，约为基线3.47倍，加速度误差6.43m/s²，高约67%。
- 腕相对平均误差11.38→11.40mm，基本没有整体收益；左手改善、右手退化，不能用较小PA误差宣称完整3D更准。

这些帧高度相关，以上只是同一片段描述性均值，不是150个独立样本的统计显著性结论。

## 机制解释及替换建议

1. **“2D更贴合”不等于“米制3D更准”。** WiLoR给出有用的image-space手部测量；从弱透视转到真实近距离透视，固定姿态/形状和根平移并不能同时满足全部关节。仅拟合平移确实改善图像投影，却把根深度推得更远。这个结果支持尺度/形状/方向/深度的单目耦合问题，而不是换一个focal字段就解决了。
2. **时序不能被单帧回归直接替代。** 两种WiLoR都比现有HaWoR更抖；本实验没有通过平滑隐藏误差，也没有测试全遮挡恢复能力，因为本例两手全程都有检测。
3. **保留现有米制时序来源，研究WiLoR作为额外软观测。** 可以把native2D和部分局部articulation用于相机一致的时序MANO求解，携带置信度/visibility，联合估计根深度与共享shape，保留HaWoR作为另一带不确定性的假设。WiLoR同源2D/3D不能被重复计为独立证据，不能靠加大contact权重把手拉向同样有误差的物体。
4. **若目标是彻底移除HaWoR，当前实验还没有证明可行。** 两种手部对照都固定使用了原prediction-side相机骨架，其中包含HaWoR/SLAM结果；WiLoR本身不替代相机轨迹、度量锚点或时间infilling。
5. 生产接入还需将反射右MANO参数严格转为RL原有left/right FK语义，处理每帧β与共享shape合同，重建source/bridge/manifest及ownership、手物遮挡和contact因子，并在更多不同视频/遮挡的留出样本及完整RL消费链上重测。因此本分支**不提供伪装成HaWoR archive的默认替换开关**，不删除现有guard，也未重训RL。

WiLoR本次网络+输出耗时93.15秒，含加载/首帧31.77秒，其后149帧61.38秒，约2.43fps；只反映该GPU6、FP32/batch2、单CPU线程实现。三平移适配额外1.42秒。HaWoR用了缓存，未重跑计时，故没有公平端到端速度比较，也不能承诺换WiLoR提速。

## 视觉检查

已完成Astra-only冻结后图像检查，审阅完整视频对应的六帧RGB/世界对照sheet，原始回复和精确请求审计保存在实验根的`astra_visual_review.json`与`.request/`。审阅未发现明确左右手互换、整手缺失或退化几何；f90/f120的世界视角手间布局及f149的手指相对重叠没有显示WiLoR稳定优势。这是采样视觉QC，不取代全部帧的时间误差，也没有证明接触/遮挡或毫米精度。完整视频供继续逐帧检查。

## 代码、复现及产物

新增：

- `observations/hand/wilor_calibrated.py`：显式K/坐标/缺测语义，完整MANO回放。
- `observations/hand/wilor_perspective.py`：只依赖预测native2D的透视平移适配。
- `hot3d_eval/compare_hand_backends.py`：只读共同覆盖/自身覆盖、native2D与真实K投影分开评分。
- `hot3d_eval/render_hand_backends.py`：完整时序原生三角面RGB/同一fixed-world视图。
- `hot3d_eval/review_hand_backends.py`：冻结后Astra-only视觉QC，不反向改预测。
- `observations/hand/test_wilor_calibrated.py`：9项closed-form/缺帧/相机转换/透视回收测试。

原有一个测试在SciPy1.17.1下把`from_euler('z',[四个角])`解释为错误末维；只将测试fixture改为显式(N,1)，未改变相机算法。原始失败日志保留。修正后分组回归**179项，175通过/4skip**，无失败。

完整原始运行记录及最终指标根：

```text
/home/kupingxin/ego_annotation_experiments/hawor_wilor_hand_eval_20261015/
  baseline/metrics.json
  final_comparison/metrics.json
  final_comparison/frame_metrics.json
  final_comparison/comparison.npz
  perspective_renders/mano_overlay.mp4
  perspective_renders/mano_fixed_world.mp4
  renders/                         # direct-K 对照完整视频，未覆盖
  evaluate_baseline.sh
  evaluate_perspective_and_render.sh
  tests_r2.log
```

两个对照的每段视频都150帧、30fps、5秒，逐帧解码核对。所有视频由真实778顶点/1538三角面MANO驱动，不是盒子/骨架；fixed-world三列是同一观察相机，不各自居中/重对齐。RGB overlay是完整表面投影，**不代表已正确处理真实物体遮挡**，没有contact/无穿透认证。六帧contact sheet仅为采样QC，不是全视频的替代。

预测目录（均只读冻结）：

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_hand_runs/wilor_milk_20261015_r1/
/mnt/truenas-user-home/kupingxin/ego_annotation_hand_runs/wilor_milk_perspective_20261015_r1/
```

每目录含`wilor_hands.npz`及`.completed.json`；后者记录真实输入/权重/output hash、完整参数回放、覆盖和原始trace。冻结worker分别在`ego_annotation_runtime/wilor_hand_measurement_20261015_r1`和`wilor_hand_perspective_20261015_r1`。WiLoR源码revision `fcb911312a38fa8badd30d9656a167485d61b8f9`，public checkpoint SHA256 `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2`。无模型、视频、GT、凭据或大数组进入Git。

执行命令保留在上述本机目录，依赖已经授权的MANO/HOT3D文件和已配置环境；这些资产不随源码发布。当前工作树改动未commit、merge或push。
