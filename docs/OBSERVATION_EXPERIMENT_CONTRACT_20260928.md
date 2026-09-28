# 五例机制实验合同 — 2026-09-28

基线为 `monocular_visible_surface_20260928_jump_r2`；本轮是实验调查，不替换默认管线。当前服务器单CPU线程，禁用CUDA，无模型推理。完整源输入、原始UniDepth、相机、完整网格和MANO均固定，实验输出使用新根目录。

## 需要区分的故障

1. 材料点绑定生命周期错误：短时2D跟踪像素可用，但旧3D绑定不可见/错误。
2. 候选缺失与候选选错：生成的姿态集是否覆盖更好的解，观测评分能否识别。
3. 观测族误导或无信息：RGB、mask、第一表面、MANO投影域对同一候选集的作用。
4. 深度缺口与旋转混淆：保留当前RGB运动的桥接是否在长gap产生未约束分歧。
5. 尺度吸收姿态错误：固定候选后重新求唯一s/t，尺度是否明显变化。

## 有界实现

- 显式读取同一source bundle与其冻结baseline scene；baseline仅是memoized prediction初始化，校验source/hash，不读取GT或旧评价。
- 预先提取2D tracklet：每8帧及appearance anchor选关键帧，纯2D角点/双向LK向两侧跟踪最多8帧。2D轨迹不因任何候选改变；同一tracklet的全部观测永远属于同一训练/留出split，禁止候选各自重选验证内点。邻近关键帧可能重新检测同一材料点，因此该holdout只能称“tracklet留出”，不是保证完全独立的材料身份留出；另用mask/表面约束交叉检查。
- 绑定显式存参考帧、native triangle first-hit point、训练/留出状态；基线关键帧给出一个条件绑定假设，非观测真值。比较原绑定与当前帧重新做第一表面自可见性/像素一致性的lifecycle支持。解除绑定不删除2D记录，也不按候选删除评分人口。
- 候选包含冻结baseline、共同训练像素上的stale-PnP、lifecycle-PnP，以及不同参考绑定/PnP解的竞争候选。近似共面不强行压成平面后假称真实mesh点；严格共面才能用IPPE。最大4个输出候选，统一配置；报告生成数量/去重/覆盖及无候选状态。
- 固定候选集评分族：留出RGB重投影（不剔除未解释点）、完整mesh的fixed-domain mask、原始可见表面第一命中/coverage、MANO遮挡域与raw mask对照。各族按固定样本数归一化，预注册尺度RGB=8px、surface=20mm、mask=1-IoU。不能靠少命中点得低分。
- `stale`与`lifecycle`输出用于实验1；`multi`在共同候选上选解用于实验2；`no_rgb/no_mask/no_surface/no_mano/raw_mask`改变同一评分集，不重新生成候选。这是**候选选择层**留一，不是整个上游pipeline留一。no_mano仅取消下游MANO投影域，并不能取消已依赖MANO的上游owned masks。
- 可观测性：留出/训练像素Jacobian，旋转单位rad、平移按物体rms半径归一化，报告奇异值、弱方向及条件性。候选spread在gap逐帧报告，不能冒充标定的贝叶斯协方差。
- 实验4本轮用可计算的候选分歧/无深度帧分层诊断，**不声称已经实现gap covariance估计器**。短/长gap的实际位姿视频保留。
- 实验5对multi选出的R再求共享s及完整平移timeline，复用现有固定UniDepth backend。完整mesh topology不变，仅一个scale。不是调深度。

## 可辨识结论与限制

如果lifecycle减小自遮挡矛盾且最终画面/事后GT改善，支持绑定机制；若候选全部错，只能说这个生成器未覆盖好解，不能直接证明网格形状必错。若posthoc oracle候选明显好而预测选择差，支持评分/观测解释不足。若leave-one改变了选择且画面变好，定位对应观测族的条件冲突；不能仅凭GT调该族权重。若几乎不改变，需要检查候选多样性是否先不足。

所有候选/选择/尺度结果完成并冻结后才允许独立GT评价。oracle只输出诊断指标，不写回预测。五例必须全部150帧完整render，用同一配置；失败/退化保留，不能按对象挑最好的模式叫通用成功。输出包括完整mesh+源MANO的baseline/lifecycle/multi/scale-refit视频、候选网格与留一网格、逐帧分数/支持/候选/选择数组与source receipts。当前milk结果不覆盖。
