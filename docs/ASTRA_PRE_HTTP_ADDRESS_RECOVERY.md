# Astra网络阻塞的定位与传输修复（2026-09-21）

## 真实失败，未改写为成功

运行 `milk_observation_v2_20260921_r1`：

- 首次：raw_frames后，judge_object出现TLS handshake timeout，exit1。
- 一次同代码resume：推进至anchor_candidates，共15个成功stage；judge_pose_anchor的TLS握手被对端reset，exit1。
- r1代码/资产字节未变化，没有遗留GPU计算进程，原日志和产物保留。

证据目录：`/mnt/truenas-user-home/kupingxin/rl_v2_a800_execution_20260921/`。

## 可区分的根因证据

对同一固定域名`litellm.dex-gem.ai`的DNS结果逐地址做无凭证TLS/health诊断：

- IPv4 `104.21.42.134`：TLS握手超时；
- IPv4 `172.67.206.25`：TLSv1.3、正常证书校验，HTTP200；
- IPv6：当前服务器网络不可达。

这不能证明所有后续网络错误均来自同一原因，但明确说明“域名健康检查一次成功”不足以保证每次urllib解析后的地址可用。标准HTTPS连接在TCP已连接但TLS失败时，不会替请求尝试另一DNS地址。

## 修改内容

新增`utils/astra_transport.py`，由`utils/runtime.py`调用：

- 只对固定origin执行DNS-address连接尝试，保留同一域名、Host/SNI、端口443、系统证书验证。
- 单地址TCP+TLS预算15s，总连接预算45s；连接成功后仍使用原HTTP socket timeout120s。
- 仅在HTTP headers/body发送前选择可连接的地址。
- 最近成功地址仅当仍在当前DNS结果中时优先，不硬编码IP。
- 证书验证错误立即失败；不使用不验证TLS的context。
- 不重试已发送的POST、HTTP错误或response读取错误；不替换模型、API endpoint、提示词、图片或生成参数。
- 明确拒绝redirect；环境配置了代理时仍走标准代理连接，不绕过代理改成直连。
- transport policy加入planned request身份；代理和证书路径环境变量以hash纳入stage environment。
- 只记录连接地址/耗时/失败类型，无credential、token或请求内容。

这一修改需要**新快照/新run**。没有更改r1快照并继续冒充same-code resume。

## 验证

- 新增8项无网络单测：失败地址后尝试同origin其他地址、DNS内成功地址优先、证书失败不重试、拒绝不安全context、代理不绕过、连接预算、已发送HTTP失败不重新POST、origin/端口/redirect约束。
- 27项传输及判断相关测试：26通过、1可选真实诊断跳过。
- 扩展全回归：233项运行，228通过、5按真实依赖/可选输入跳过。
- 新传输的无凭证health实测记录了A地址失败后B地址TLS成功及HTTP200。
- r2快照的health检查中，一次默认Python User-Agent的GET返回HTTP403；使用与生产原有相同的`ego-annotation/1.0`请求头后返回HTTP200。保留两次日志，不把第一次改写为通过；生产User-Agent未修改。

相关文件：

- `network_tls_diagnostic_2.json`
- `new_transport_health.json`
- `address_transport_tests.log`
- `transport_fix_full_regression.log`
- `r2_transport_health.log`
- `r2_transport_health_production_headers.json`

## 新执行边界

- 快照：`/mnt/user-home/kupingxin/ego_annotation_runtime/rl_observation_v2_milk_20260921_r2`
- 新run：`/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2`
- 同一原视频/预测侧标定/模型资产/GPU5 UUID。
- 新源码哈希与运行身份；当前没有跨代码前缀导入机制，因此不复制r1的成功阶段并伪造新producer receipt。
- runtime执行者只执行预测，不修改代码、环境或补造判断；后续完成后仍需独立v2检查。

该修复消除了已观察到的一个连接级单点失败，不保证Astra服务永远可用，也不构成v2或最终标注成功声明。
