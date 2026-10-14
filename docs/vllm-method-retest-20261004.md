# 原生引擎按旧 vLLM 方法复测（2026-10-04）

按用户要求，当前原生引擎使用之前 vLLM 的固定短提示词及计时方式重新测速。预热 128 输出 token，然后单并发三轮，每轮输入“写个网站网页”、实际输入 16 token、实际输出 1024 token，temperature=0、seed=0、关闭思考、固定 MTP4。

| 轮次 | Decode tok/s | TTFT ms | 端到端 tok/s |
|---|---:|---:|---:|
| 1 | 172.42 | 44.19 | 171.14 |
| 2 | 172.76 | 44.97 | 171.45 |
| 3 | 172.01 | 45.24 | 170.71 |
| 均值 | **172.40** | **44.80** | **171.10** |

旧 vLLM 适配版在 2026-09-21 相同短提示词、MTP4、无思考及 1024 输出下三轮 Decode 均值为 162.83tok/s，本次观察高约 5.88%。这是与历史结果的比较，并非今天同时启动两套引擎的 A/B；本次运行的是原生引擎，没有重新测试 vLLM 本身。旧记录见 `.artifacts/q38-reference-ablation-v2-20260921/mtp1-graph1/benchmark/summary.json`。

## 方法和验证

沿用旧方式：输出 token 数 /（最后一个非空 SSE 内容片段时间 − 第一个非空片段时间）。三轮的 N−1 口径均值为 172.23tok/s；端到端含首字等待和流收尾，均值为 171.10tok/s。客户端在 Z4 使用 Python urllib 和本机 loopback，与旧参考测试相同，不使用本次 llm_speedtest 的 Mac 转发计时。完整启动校验和加载不计入请求计时。

API usage 每轮均为 16 输入、1024 输出、1040 总 token，并交叉核对 executor 发布计数与剩余 token 元数据。三轮输出内容分别与此前已合格的原生输出 SHA256 完全一致；均因输出预算到达而结束，这不是新增完整网页质量测试。

配置未改：双 CMP170HX，每卡 200W，现有 W4/W8，PLE RAM 预算 64GiB，mtp4-native-fast，durability=off。官方启动器重新完整校验 354 个唯一段、212367104472 字节，运行二进制 SHA256 7199f47ebf5d8dabf6b05611c14e55d9d66e7caf5e3dc8f4c95f2026f169613b；没有重编译。

每轮 MTP 草稿接受率为 82.54%；三轮测量区间没有 major fault 增长、process swap、系统 swap-in 增长、executor failure/rollback、Graph fallback 或 CUDA allocation failure。前后 boot ID 均为 2de99344-dfd6-4e89-8747-823eef433193，没有重启。模型和专用遥测已停止，两卡显存均 0MiB，仍限制 200W；挖矿和其他 AI 服务没有启动。

本次确认固定短提示词的 172tok/s 可复现；[4K 随机写作的 73.60tok/s](llm-speedtest-retry4k-20261004.md)也是真实结果。负载和计时边界不同，不能把短提示词成绩推广为通用速度，也不代表此前偶发重启已经解决。

原始证据：`/Users/icy/Homelab/.artifacts/q38-native-vllm-method-retest-20261004`，服务器副本 `/home/icy/q38-native-vllm-method-retest-20261004`。保留启动器日志、完整 manifest、请求/SSE、前后 metrics、三轮结果、原生输出一致性检查、历史 vLLM 摘要、1 秒 GPU 遥测、前后启动状态及 `results.csv`。未上传云历史或公网排行榜。
