# llm_speedtest 原生引擎实测（2026-10-04，因重启中断）

使用用户指定的 [gengchaogit/llm_speedtest](https://github.com/gengchaogit/llm_speedtest)，固定提交 eb19940368f0d811ef7b14f77f61d1c7e8676c7a，通过网页相同的 Python 后端 /ws/test 执行。测速函数、提示词生成和前端源码未修改。三条真实usage结果完成，但两次整机重启使三轮矩阵未能完成；不提供三轮均值或稳定吞吐结论。

## 有效结果

| 尝试 | 实际输入 token | 实际输出 token | Prefill tok/s | TTFT ms | Decode tok/s | MTP 草稿接受率 |
|---|---:|---:|---:|---:|---:|---:|
| A | 128 | 224 | 853.52 | 167.43 | 98.31 | 33.1% |
| A | 4096 | 512 | 863.35 | 4763.09 | 80.09 | 39.7% |
| B | 128 | 512 | 814.74 | 169.37 | 96.00 | 34.3% |

单并发、最多输出512 token（允许EOS提前结束），temperature=0/top_p=1/no-thinking。Z4双64GiB CMP170HX，SM80，driver610.43.02/CUDA13.3，每卡200W，现有W4/W8，RAM PLE预算64GiB，mtp4-native-fast固定MTP4，durability=off。二进制SHA256 7199f47ebf5d8dabf6b05611c14e55d9d66e7caf5e3dc8f4c95f2026f169613b没有重编译。

Prefill是上游真实输入token /（客户端首字等待 − 模型目录请求平均网络延迟），含tokenizer、排队、输入计算和首字，不能当作纯GPU kernel吞吐。Decode是真实输出token /（首个非空SSE内容到流结束）；经Mac本地代理和SSH转发。ITL按SSE内容事件统计，MTP成批发布时不是逐token计算时间。所有有效结果使用API usage，无本地估算或异常回退。工具默认随机英文词加哲学短文任务，实际输出长度如表。

128档接受率差值含该次8-token预热，4K档仅含该请求。此前固定网页/代码负载接受率约70.7%/79.0%，明显高于这里约33–40%；4K档在更高接受率下仍比128档慢，接受率不能单独解释差距，长上下文的验证成本还需单独测量。此前172.81/174.70tok/s来自另一负载及首尾内容事件计时，不能推广成通用速度。这次随机写作负载未达到160tok/s，后续应在相同输入、实际输出和计时口径下与参考引擎对比。

## 两次重启

A：128和4096成功，32768请求只收到HTTP头，没有首个SSE事件；随后失联。旧boot f259c6ecb3454f21840372356f5d249d，日志最后一条13:14:24北京时间；新boot 8a5d28a207a34aca9d37a75b06abaaf1开始约13:15:56。

B：重启后限定128/4096，各三轮计划；预热及128输入/512输出成功，下一条4096输入在首字之前中断。新boot 2de99344dfd64e898747823eef433193开始约13:28:30。最后保存的5秒间隔遥测（13:27:23）为GPU0/1约158.10/161.47W，49/45°C，200W限制，显存57934/58294MiB；采样无法捕获所有供电瞬态。问题并不限于32K。

两次没有正常关机日志；事故附近未找到OOM、Xid、panic或MCE记录。root读取pstore及systemd-pstore归档均无转储，var/crash只有更早的wireplumber崩溃和空kdump锁。旧boot早上07:33:10的Xid31不在本次事故窗口，不能作为原因。现有证据无法区分供电、驱动/PCIe/P2P、主机硬件复位或其他原因；不认定为功耗或过热。未继续重跑，当前两卡已空闲，仍限制200W，AI和挖矿没有启动。

A的两条成功请求没有executor failure/rollback、Graph fallback、CUDA allocation failure或major fault增长，process swap=0。B的成功128请求没有executor failure/rollback及process swap，但包含冷启动预热的统计区间增加178次major fault；不能声称这次全程无page fault。重启使后续完整统计无法取得。

## 修复和保存

测速前发现原生流式API忽略stream_options.include_usage，上游因此静默估算token。已修复tools/q38_sidecar.py：选项开启时在DONE前发送choices为空的usage事件，以完整tokenized history和已发布token IDs计数，普通流和工具调用流都支持。未改权重、数学、MTP参数、内容片段或测速公式。14项sidecar HTTP/协议测试通过，包括普通流/工具流计数、与非流式计数一致、可选usage及非法选项；Mac/Z4源码同步。

Mac证据：/Users/icy/Homelab/.artifacts/q38-llm-speedtest-20261004。valid-results.json仅有三条合格结果；attempt-A.csv/attempt-B.csv可在原工具导入对比图；long-attempt和根目录保留各次WebSocket日志、请求、原始SSE、metrics；reboot/reboot-2保存系统日志，telemetry.csv保存第二次遥测。preflight失败记录完全排除；未上传公网排行榜或云历史。Z4证据在/home/icy/q38-llm-speedtest-20261004。

完整模型正确性与此前短提示词成绩仍见[优化报告](native-followup-optimization-20261004.md)，本次没有增加长上下文或生产稳定性资格。

## 后续单次复测

用户随后限定仅复测失败请求。尝试 B 的相同 4K 请求单次重放成功：实际 4096 输入/457 输出，Prefill 866.85tok/s、Decode 73.60tok/s、TTFT 4749.16ms，boot ID 未变化。使用全新状态和短预热，未重放此前 B 的 128 输入/512 输出前置测试；因此只能确认相同请求这一次通过，不能认定偶发重启已解决。详细证据和独立 CSV 见[4K 复测报告](llm-speedtest-retry4k-20261004.md)。
