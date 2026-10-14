# 固定 MTP4 路由优化与整模型实测

2026-10-04。双 CMP170HX，现有 W4/W8，P2P，RAM PLE，MTP 固定为 4，单请求。
本轮对照是已经完成集成的 `mtp4-candidate`，新配置为 `mtp4-route-candidate`。

## 结果

每项预热一次，再计时两次。使用同一新二进制，只有两项路由开关不同；正式
HTTP 测速关闭完整 logits 诊断。以下为两次 decode 吞吐的算术平均：

| 请求 | 对照 tok/s | 路由优化 tok/s | 变化 |
|---|---:|---:|---:|
| 网页，16 输入 / 1024 输出，非思考 | 71.98 | 74.34 | +3.27% |
| 代码，24 输入 / 512 输出，非思考 | 72.31 | 74.50 | +3.02% |
| 8296 输入 / 512 输出，非思考 | 46.48 | 47.70 | +2.63% |

吞吐定义沿用此前 HTTP 测量：完成 token 数除以首末非空 SSE 内容事件的间隔。
计数来自最终已提交 token 数减输入 token 数。原始证据同时保留 N−1 和端到端
口径。对照/新版首字延迟分别为网页 61.39/59.72 ms、代码 74.32/72.16 ms、
8K 输入 11.025/10.879 s。首字延迟包含输入处理，不等同于独立 prefill 算子耗时。

所有计时样本的进程 VmSwap 为零，系统换入页数不变，进程 major faults 不增长；
两个 stage 的图 fallback 均为零。各项两次样本均保留。网页新版两次为
74.77/73.91 tok/s，代码为 74.50/74.50，8K 为 47.21/48.20。样本规模有限，
这里只报告这三类请求的本机收益。本轮没有测思考模式或真实 262K 整模型请求。

## 改动

`cuda/q38_moe_prefill.cu` 将专家计数、任务计数和活跃专家计数的串行前缀扫描，
改为整数 warp 扫描和跨 warp 前缀汇总。稳定路由顺序、专家任务切片、越界
状态和错误检查保持原合约。数值矩阵算子及其舍入路径沿用原实现。

运行时不读取路由计划的 `route_hash` / `plan_hash`，新配置省略这两项诊断
计算，并在 header 中明确设置 `kQ38RoutePlanHashesOmitted`。API 默认继续计算
诊断哈希；这项开关不关闭模型分片校验、session identity、传输边界检查或事务检查。

显式配置展开为 `Q38_CUDA_MOE_ROUTE_PARALLEL=1` 与
`Q38_CUDA_MOE_ROUTE_HASH_SKIP=1`。其他选项与 `mtp4-candidate` 相同。
原来的两个 MTP4 配置明确将这两项设为零，避免继承环境变量误启用。
配置及二进制 SHA256 继续进入 session identity。

路由算子计时中，MTP4 常用 5 行计划从 41.25 μs 降至 4.81 μs，约 8.57 倍。
这是局部算子的收益；整模型 HTTP 提升如上表。单行 MoE、矩阵计算、通信和
其他执行开销仍在。最初尝试的 gate/up 融合在 W8 等路径出现退化，已经撤回，
`cuda/q38_moe.cu` 没有纳入本轮修改。

## 验证

- 路由 fixture 覆盖 1/2/5/16/128/512 行、256/512 专家、重复和集中路由、
  未归属及非法 ID、变化输入与图回放，共 252 项。保留哈希时整个计划逐字节
  一致；省略哈希时仅 header 的哈希和对应标记不同，其余字节一致。
- 现有整套 CUDA kernel tests 与三个固定配置测试通过。
- 五类整模型输入（数学、叙述、代码、512、8192 token）固定 MTP4 的 token
  一致，续跑的完整 BF16 logits 一致。
- 两种上下文、拒绝位置 0–4 的十个前缀验证，发布边界和 MTP 续跑一致；
  与上项共检查 45 个完整 logits 探针。
- 九个自然结束的回答通过数学、JSON、Unicode 和代码执行检查，并逐 token
  一致；取消草稿、取消 append、超时后的 reset 续跑通过。
- 三项 HTTP 请求两次计时的内容、输入长度和输出长度分别一致。

整模型验证使用 RAM PLE。本轮没有追加 SSD 整包测试。原有 SSD 入口仍保留。

## 使用

服务器正式源码目录为 `/home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU`。
本轮二进制安装在 `build/native-mtp4-route/q38-cuda-runtime`，保留
`build/native-mtp4/q38-cuda-runtime`。在两张卡可用时启动：

```sh
cd /home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 .venv/bin/python tools/q38_tp_launch.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp-root /home/icy/q38-native-opt-mtp-20260921/artifact \
  --runtime build/native-mtp4-route/q38-cuda-runtime \
  --state-dir /home/icy/q38-tp2-mtp4-route-serving \
  --execution-profile mtp4-route-candidate \
  --ple-storage ram --ple-ram-budget-gib 64 \
  --segment-hash-workers 4 --port 30001
```

回退时使用旧二进制和 `--execution-profile mtp4-candidate`，重新由 launcher
生成身份文件。不要复用其他配置的 session identity。测试完毕已退出临时
runtime / sidecar，没有持续占用 GPU，也没有恢复挖矿或挖矿定时器。

## 证据

Mac：Homelab 根目录 `.artifacts/q38-native-moe-fused-20261004/`；服务器镜像：
`/home/icy/q38-native-moe-fused-20261004/`。目录名称沿用最初的融合实验。

`summary.json` 汇总结果；`whole-model/` 保存两种配置、身份、二进制 SHA、
命令、完整探针、质量与取消结果、原始 SSE 和计时统计；`route.jsonl` 保存
路由算子结果。`routing-changes.patch`、`routing-source.tar.gz` 及原源码归档
用于审阅与回退；`run-route-qualification.py` / `summarize-route.py` 可复查流程。

硬件记录为驱动 610.43.02、内核 7.0.0-31-generic、CUDA 13.3.73、SM80、每卡
200W。完整校验覆盖 348 个唯一模型分片；没有下载或替换量化权重。
