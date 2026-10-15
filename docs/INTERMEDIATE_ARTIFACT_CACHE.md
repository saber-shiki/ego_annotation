# 可复用观测 / 预处理中间产物（HOT3D 5 秒组）

> 2026-09-23 23:42 队列更新：用户要求 milk 不再运行，直接复用旧完整 S3 包。当前 mug 不受影响，结束后首先断点重试 spatula。活动控制器为 `finalize_no_milk.py`，窗口 `ego_annotation:hot3d-no-milk`。旧的 milk 续跑计划已取消。

## 存储位置与边界

- **持久库**：`s3://lance-datasets/ego_annotation/intermediate_artifacts/`
- S3 endpoint：`https://192.168.10.222:9000`。
- **运行工作区**（SMB，不是 S3 挂载）：`/mnt/truenas-user-home/kupingxin/ego_annotation_intermediate_artifacts/`。
- 已有完整 milk v2 run 不改写、不重打 producer receipt；其原始观测和既有前处理已保留。
- 这里只做观测和预处理，不执行训练/PPO/轨迹优化，不把 QC 图或 source MANO 当成最终标注质量证明。

## S3 布局

```text
objects/sha256/<前2位>/<SHA256>                # 去重后的真实文件字节
packages/<case>/observations/<SHA256>.json    # 完整观测包清单
packages/<case>/preprocessing/<SHA256>.json   # 独立前处理包清单
catalogs/hot3d_5s_20260923/<SHA256>.json       # 批次结果/可用包索引
README.md
```

每个 package 清单逐文件记录相对路径、大小、SHA256 和对象 key。相同文件跨
run 共用同一内容对象；不同输入/产物字节自动落到不同 key，不覆盖旧结果。
每个文件上传后通过 **GET 实际读回并重新计算 SHA256**；已有对象也读回核验，
不能用 multipart ETag 冒充 SHA256。只有所有文件验证完成才发布 package 清单。
因此失败的上传可以留下未被引用的内容对象，但不会出现伪造的完整包。

保留内容包含原始视频/相机元数据、RGB帧、深度、HaWoR/MANO、分割和跟踪、
SAM3D原生及metric几何、RGB/PnP、Astra原始判断和收据、typed interaction/P17、
恢复journal、code/model身份，以及独立前处理NPZ/CSV/QC文件。

## 复用，而不是重新跑模型

已有本地工作集时，直接加载冻结的源包：

```python
from data.bundle import EpisodeBundle
bundle = EpisodeBundle.open('/ABS/ORIGINAL_RUN/source_observations.json')
observations = bundle.load_observations()
```

修改RL、reward、可视化或前处理代码后，不要重新调用 `main.py observe`；
消费者使用这个源包。要重新导出新版前处理，使用新的输出目录：

```bash
python -m observations.inspect \
  --source-observations /ABS/ORIGINAL_RUN/source_observations.json \
  --output-dir /ABS/NEW_PREPROCESSING_REVISION
```

已经产生的 `preprocessing/arrays/*.npz` 等也可以直接消费，不必每次重新计算。
输入视频、相机标定、producer模型/权重、prompt/schema、reasoning effort或依赖
观测改变，相关产物必须视作新版本，不能偷偷沿用旧结果。

同一冻结版本的失败尝试使用原runtime目录执行 `main.py resume`，由现有
StageRecovery核验输入、代码、配置、execution contract和逐stage收据，跳过
成功阶段。这不是允许在变更代码后重写旧journal。跨版本整包消费与同版本
stage恢复是两个不同机制。

## 上传与恢复工具

新增源码：`observations/s3_cache.py`（可选依赖 `.[storage]`）和
`observations/artifact_cache.py`（保留共享目录上的整run备份）。
生产模型环境没有安装/修改这些S3依赖；独立S3工具环境为：
`/home/kupingxin/ego_annotation_tools/.venv_s3/`。

```bash
# credentials.json 位于 ~/.config/ego-artifact-s3/，权限0600；绝不可上传。
python -m observations.s3_cache publish \
  --root /ABS/ORIGINAL_RUN --case CASE --kind observations \
  --receipt /ABS/NEW_UPLOAD_RECEIPT.json

python -m observations.s3_cache publish \
  --root /ABS/PREPROCESSING --case CASE --kind preprocessing \
  --receipt /ABS/NEW_PREPROCESSING_UPLOAD_RECEIPT.json

python -m observations.s3_cache restore \
  --manifest-key ego_annotation/intermediate_artifacts/packages/CASE/observations/HASH.json \
  --destination /ABS/ORIGINAL_RUN
```

**恢复限制**：现有producer JSON内有绝对路径与字节哈希绑定，不能简单替换
根目录然后重打哈希。观测包必须恢复到manifest中的 `original_root`，且目标
必须不存在；工具拒绝覆盖和任意路径迁移。已有该路径时直接复用。独立前处理
包可以下载到新目录；其中的source引用仍指向原观测包。

自签证书以明确CA文件固定并验证IP SAN，不关闭TLS验证。初次从用户提供
endpoint获取的证书SHA256为：
`72a5668348c1de2fb53308e9fe2891e6b64dcf4bf01a996c88cd0ced65d25129`。
凭据内容从未写入代码、报告或上传产物。

## 本批次运行

五个输入都已通过ffprobe确认150帧、30FPS、5.000000秒：

- `hot3d_P0002_016222d1_spatula_red_f3285_3434`
- `hot3d_P0012_73e66984_mug_white_f2145_2294`
- `hot3d_P0012_ca1f6626_bottle_bbq_f1515_1664`
- `hot3d_P0014_84ea2dcc_carton_milk_f2370_2519`
- `hot3d_P0015_7e95628d_can_soup_f1770_1919`

运行快照：
`/mnt/user-home/kupingxin/ego_annotation_runtime/rl_observation_hot3d_5s_batch_low_20260923/`

每个快照基于已跑通的milk v2 r2 producer，只有Astra transport加入固定的
`reasoning: {effort: low}`，同时在planned inputs和request provenance记录该值。
仍使用`gpt-6-astra`，没有换模型或fallback。实际新判断收据中已核对`low`。
并未测量相对高思考的token节省比例，因此不报告一个虚构的节省数字。

A800服务器：`dexrobot-Standard-PC-Q35-ICH9-2009`，GPU5，UUID
`GPU-e4ff72b8-fff7-21d0-2872-6aeecb56a91e`。不占用其他用户的GPU进程；首次
spatula/mug在GPU占用检查处退出，后续从原收据恢复。milk新尝试的Astra读取
超时保留原日志并从断点续跑；旧完整milk v2包仍可独立复用。

实际状态以工作区的 `s3_receipts/*.json`、`logs/*/*.exit` 和
`manifests/batch_final.json` 为准，不以文件存在或此文档表示全批次完成。
最后一个文件由依赖作业在所有case结束后生成；`complete=false`必须保留失败，
不能把它读成成功。运行控制窗口位于现有`ego_annotation` tmux会话。

## 已执行验证

- 本地保留工具：3项单测；S3工具：4项单测。
- 已上传对象逐个GET/SHA256读回。
- BBQ前处理从S3恢复到独立目录，327个文件，285个NPZ / 1708个数组实际解码，
  无模型调用、无pickle数组加载。
- BBQ全150帧前处理完成但R4 measurement seed在7帧未解析，保留warning而不是
  补零伪造运动。已查看首/末帧QC：物体轮廓跟随瓶子，source MANO存在偏差，
  appearance anchor完整mesh投影也有偏差。这些是待下游优化的观测，不是最终解。
- 原milk已有完整前处理；新视频的全部观测/前处理完成情况见批次结果，不在这里
  提前声明还在运行的case成功。
