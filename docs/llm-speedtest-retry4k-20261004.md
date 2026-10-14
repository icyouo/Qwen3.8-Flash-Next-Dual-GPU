# llm_speedtest 失败的 4K 请求单次复测（2026-10-04）

仅重放上次尝试 B 中导致整机重启的 4096 输入请求，本次成功，没有重启。没有重跑 128/32K 档或完整矩阵，也没有修改引擎、重新编译或调整硬件参数。一次通过不能证明之前的偶发重启已解决。

| 实际输入 token | 实际输出 token | Prefill tok/s | Decode tok/s | TTFT ms |
|---:|---:|---:|---:|---:|
| 4096 | 457 | 866.85 | 73.60 | 4749.16 |

输出上限为 512，正常提前结束；计数来自 API usage。上游工具为 [gengchaogit/llm_speedtest](https://github.com/gengchaogit/llm_speedtest)，固定提交 eb19940368f0d811ef7b14f77f61d1c7e8676c7a，使用相同 Python `/ws/test` 测速函数。单并发、greedy、temperature=0、top_p=1、no-thinking；双 CMP170HX，每卡 200W，现有 W4/W8，RAM PLE 预算 64GiB，固定 MTP4，mtp4-native-fast，durability=off。

Prefill 为实际输入 token /（首字等待减平均网络延迟 24.01ms）；Decode 为实际输出 token /（第一个非空 SSE 内容到流结束）。本次对应 4725.15ms 和 6209.60ms，经 Mac 本地适配器与 SSH 转发，不能把 Prefill 当作纯 GPU kernel 吞吐。SSE 内容事件间隔均值 31.84ms、标准差 8.90ms；MTP 成批发布，事件间隔不是逐 token 计算时间。没有 token 估算或计时边界回退。

## 重放范围和状态

冻结的请求来自上一轮 `.artifacts/q38-llm-speedtest-20261004/provider/003/effective-request.json`。本次 `provider/002/effective-request.json` 与其 JSON 内容完全一致，包括模型、完整 messages 和采样/流式参数；重放文件 SHA256 为 0ab973a62d41ca9d1d65098e1e1164c737a910c8fe83165b9936a2cb3562373a。包装层只在测量 4K 时返回冻结提示词，保留原有测速函数；代理仅添加 enable_thinking=False，原始 SSE 和 usage 不改写。

本次使用全新 native-state，并执行上游默认短提示词、8 输出 token 预热。此前失败 B 在同一进程中还先执行过 128 输入/512 输出测试；本次没有这条测试。因此重放的是相同请求，不是相同的完整前置执行历史，不能排除与缓存、分配或请求顺序有关的故障。

开始、请求完成和清理后的 boot ID 均为 2de99344-dfd6-4e89-8747-823eef433193。二进制 SHA256 7199f47ebf5d8dabf6b05611c14e55d9d66e7caf5e3dc8f4c95f2026f169613b 未变化。

## 统计和收尾

前后 metrics 区间包含预热与测量：发布 465 token（8+457），草稿 789、接受 265，接受率 33.59%，198 次 speculative transaction，最大草稿宽度 4。executor failure/rollback、CUDA allocation failure、Graph fallback、cgroup OOM 没有增长，process swap=0。冷启动区间 major fault 增加 197，不能声称全程无缺页，也不能把这个接受率直接标为仅 4K 请求的接受率。

1 秒 GPU 遥测整个记录窗口（含加载、预热、测量和释放）各 677 条，GPU0/1 采样最大功耗 154.18/151.57W、最高温度 50/47°C。采样不能排除供电瞬态。该窗口内保存的内核日志未找到 Xid、OOM、panic 或 MCE 记录；本次单次成功不改变此前两次重启原因未明的结论。

测试进程、专用转发和遥测已退出；清理确认两卡显存均 0MiB，上限仍 200W，AI 和挖矿没有启动。原始数据位于 `/Users/icy/Homelab/.artifacts/q38-llm-speedtest-retry4k-20261004`，服务器副本 `/home/icy/q38-llm-speedtest-retry4k-20261004`。包含 config、provenance、冻结请求、原始 SSE、WebSocket、前后 metrics、boot ID、遥测、内核日志、清理状态，以及可导入原工具的 `retry4k.csv`。未上传公网排行榜或云历史。

此次随机写作负载 Decode 为 73.60tok/s，仍未达到 160tok/s；此前 172–175tok/s 的短固定负载成绩不能替代这个结果。历史三条成功结果及两次重启记录保留在[上一轮报告](llm-speedtest-20261004.md)。
