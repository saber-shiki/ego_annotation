# 无 RGB seed 完整网格初始化：本轮实现契约

用户决定：RGB seed只作对照；RL候选从无RGB seed的完整native网格起点修正，RGB测量不删；不启动训练。

## 具体机制缺口

V2ConsumptionSession硬编码motion.initialization并自动绑定完整网格，缺乏源初始化的默认候选路径。bundle loader也无条件构建R4传播轨迹。旧AnnotationEnv仍是v1/P13/480维体系，不能删除guard后冒充v2接通。

## 修改边界

1. 分开R2 motion qualification与可选R4 RGB传播：loader默认仅准备资格，显式comparison选项才构造seed。局部旋转forest仍用于独立loop判断，但不提供候选R/t。
2. 新建共享、版本化完整native候选初始化器：输入只限逐帧source pose、timeline、camera和visual geometry，不读RGB seed。source有效帧精确保留R/t，绑定在明确appearance-anchor source pose，不搜尺度。
3. 缺source pose有两种显式策略：默认`unresolved`精确复现无seed源基准；`previous_source_hold`仅在调用者明确请求时提供弱latent初值，另有flags/来源，不提升metric/source有效性。验证适配器提供显式缺测候选激活API，以保证后续策略能补缺测而不是永久冻结。
4. V2ConsumptionSession默认source_full_native，旧RGB seed仅`rgb_seed_comparison`。版本升级；保持完整geometry和source/latent/display区分。源到候选的RGB双端投影每次使用当前R/t，原始所有accepted边/点及rejected诊断全部保留；增加相对旋转和平移残差，不用seed作监督目标。
5. inspection默认输出source基准绑定，不再自动导出seed。显式`--include-rgb-seed-comparison`可保留历史对照。
6. 防止绕过dataset、直接向旧build_episode传v2导致静默旧P13消费；保持旧训练未适配的明确错误。新初始化器/证据接口用于下一阶段RL接入，不伪称当前480/118 actor或reward完整适配。

## 判别性验证

- 默认模式在禁止构造comparison seed的情况下仍可load/reset/render；扰动comparison seed不改变默认R/t/mesh。
- 有效帧zero control与旧无seed完整native基准逐数组相同；缺测143默认不补齐。
- 完整mesh顶点/面、camera/K固定，object旋转平移动作真实改变网格和RGB残差；mask/source evidence不变。
- 全部RGB181边、20394点、producer5拒绝及qualification18拒绝在真实包中不丢失；被降权的测量仍可读取，未强制当成可靠监督。
- 显式比较路径复现历史seed；不偷偷融合/替换新默认。
- 合成测试覆盖sparse IDs、缺测、source/latent区分、固定anchor、绑定、immutable arrays及旧训练guard。
- 真实150帧H264 zero/control/comparison渲染和代表帧观察；只做CPU读取/数值/渲染，不运行预测模型/PPO。

## V7对齐验收

逐项分别检查源预测机制、前处理核/坐标、证据保留与下游接入。新的no-RGB-seed初始化是明确设计差异，不称与V7 P14/P15相同；最终posterior/shape scale/visual fit不作为源观测导入。完整P17上下文、独立ICP等现有缺口需如实列出，代码测试不能代替全链质量等价。
