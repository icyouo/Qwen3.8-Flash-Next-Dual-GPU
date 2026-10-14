# 原生固定 MTP4 集成实现

2026-09-21。本轮集成实现已通过最终验收。MTP 固定为 4。

## 配置与范围

双 CMP170HX、现有 W4/W8、P2P、MTP4。RAM PLE 为主要运行配置，SSD PLE
保留同输出路径。没有改成 NVFP4，没有缩小词表，没有启用自适应 MTP 宽度。
`mtp4-candidate` 固定配置包含 GPU 图、设备边界传递、PLE staging、既有融合
算子、输出头批量规约和 KV-only 提交；`mtp4-baseline` 保留对照组合。

## 实现

- 每个 rank 记录 GPU 图片段，在第一次配对时校验通信类型、顺序、大小和 group。
  最终候选使用两个 rank-local graph；每个 collective 独享 ready/read-done
  序号，采用本地 system-release 发布、对端 system-acquire 等待。单个信号
  节点轮询，避免所有 CTA 同时争用 PCIe。双方读完后才覆写输入，沿用原来的
  BF16 舍入、FP32 求和和 token-major gather 顺序。
- 每个图片段只保留一次主机 rank 配对；PLE 主机查表、QSA 动态部分、事务
  发布仍在图外。GPU 等待有界，硬件/对端失败按 CUDA 故障终止，不发布部分结果。
- TP 草稿也使用图记录器。cuBLAS 工作区提前分配；MTP 小矩阵 scratch 在
  捕获前预热，扩容保留旧 allocation，直到引用它的图销毁。
- TP stage 边界使用本 rank 的设备 snapshot/lease，保留 checksum、generation
  和消费事件。PP 原有路径保留。
- PLE 统一规划已知 token 的行号，复用 pinned staging。copy stream 等待前次
  消费，主流等待本次数据；RAM owner 共用一份完整表，SSD 后端沿用同一合约。
- 取消仍先禁止生成；只有明确取消/超时、完整 drain 且成功 reset 后可继续。
  硬件、rollback、reset 失败不获得恢复权限。
- 显式配置清除继承的 Q38 调优变量，将展开配置和二进制 SHA 纳入 identity。
  HTTP 的配置默认值为 MTP4；手工模式继续保留原来的运行行为。

## 已发现并修正的问题

最初的多设备联合 DAG 通过数值检查，却使网页吞吐从约 68.6 降到 49 tok/s。
它不能作为最终优化版上线。使用相同整包配置的针对性对照：主机配对约 71.8，
联合 DAG 约 49.0，rank-local GPU 信号图约 71.9 tok/s，均为同一网页请求、
固定 MTP4、无进程换页的单次定位结果。后两者的差异说明不能只凭图节点减少
推断速度提升；71.8 与 71.9 的差别也不构成显著收益。

真实 cuBLAS 捕获暴露了分配节点不能作为子图克隆的问题，已通过源图所有权
转移及固定工作区修正。随后发现 MTP 首次捕获还可能分配小矩阵 scratch，已
提前预热并补齐扩容生存期。最新 SSD 模型检查中图 fallback 为零。

一次验收脚本误把 SSE 的 `remaining_tokens`（剩余上下文容量）当作剩余输出
预算。原始事件、时间和请求预算完整保留；计数已按“最终 committed_tokens
减 prompt_tokens”修正。负数旧日志不是有效吞吐结果。

部署烟测还发现 tokenizer 把结束标记直接显示到 HTTP 内容；输出解码现已在
首个 EOS / `<|im_end|>` / `<|endoftext|>` 处截断，保留推理和工具标签。
模型 token、计数、事务及计算路径不变；普通和累积流式解码测试通过。

## 验证边界

已通过模块级 changing-input、queued replay、双 group、cuBLAS M=5、边界
slot 复用与错误恢复检查。首版整包通过五类固定 MTP4 token/full-logit 探针、
十个拒绝前缀续跑、九个完整回答和取消/超时后重置续跑。SSD 与 RAM 的 token
及完整 BF16 探针一致。262144 上限有 CPU 尾部事务检查及 GPU RPC 越界拒绝；
这不等于完成了真实模型 262K 全长度质量评测。随机采样回退目前由既有 CPU
事务测试覆盖，不宣称完成 GPU 随机采样质量评测。

最终 rank-local 图已通过相同模型正确性检查、取消/超时恢复及 SSD/RAM 对照；
最终性能测试的图 fallback 为零。进程 VmSwap 全部为零，加载峰值 RSS 约
67.93 GiB。吞吐只汇总系统换入页也未增长的计时样本：

| HTTP 请求 | 原生基线 tok/s | 本轮 tok/s | 变化 |
|---|---:|---:|---:|
| 网页，非思考，1024 输出 | 68.56 | 71.80 | +4.72% |
| 网页，思考，2048 输出 | 38.87 | 41.31 | +6.27% |
| 代码，512 输出 | 68.39 | 71.81 | +5.00% |
| 8296 输入，512 输出 | 45.97 | 48.34 | +5.14% |

每项预热一次、计时两次；剔除个别系统换入 1 页的样本后，基线各项保留
1/1/1/2 次，本轮保留 2/1/2/2 次。以上为 decode 吞吐，定义为输出 token 数
除以首末内容事件间隔；原始证据同时保存 N−1 口径和端到端吞吐。短请求首字
约 62/123/75 ms，8K 输入约 11.21 s。输出上限测试与完整回答质量检查分开，
不能用到达输出上限代替自然结束的回答质量。


此前原生/参考库数字来自不同 workload，不能直接算作严格公平的倍数；参考库
约 163 tok/s 的网页数字仍是追赶目标，本次尚未达到。

## 运行入口

在目标机器的引擎目录使用 `tools/q38_tp_launch.py`。组合配置通过
`--execution-profile mtp4-candidate` 选择，PLE 通过 `--ple-storage ram|ssd`
选择；RAM 使用 `--ple-ram-budget-gib 64`。完整权重校验可用
`--segment-hash-workers 4` 并行完成。构建默认固定为 CUDA 13.3 / g++-14 / SM80。

目标机器启动命令：

```sh
cd /home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 .venv/bin/python tools/q38_tp_launch.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp-root /home/icy/q38-native-opt-mtp-20260921/artifact \
  --runtime build/native-mtp4/q38-cuda-runtime \
  --state-dir /home/icy/q38-tp2-mtp4-serving \
  --execution-profile mtp4-candidate \
  --ple-storage ram --ple-ram-budget-gib 64 \
  --segment-hash-workers 4 --port 30001
```

SSD 模式把存储参数改为 `--ple-storage ssd --ple-cache-gib 8`，停止当前实例后
再启动。两者加载相同 PLE 数据，不引入 GDS。固定配置下 HTTP 请求省略
`q38_mode` / `q38_mtp_width` 时默认使用 MTP4。

回退原服务使用保留的 `build/p2p-ram/q38-cuda-runtime`，TP artifact 改回
`/home/icy/q38-tp2-lowcomm`、state-dir 改回 `/home/icy/q38-tp2-serving`，并使用
`--execution-profile manual --tp-transport p2p`。完整原命令和源码备份记录在
部署证据 `deployment/deployment.json` / `original-source.tar.gz` 中。


原始证据位于 Homelab 根目录 `.artifacts/q38-native-mtp4-integrated-20260921/`；
服务器镜像为 `/home/icy/q38-native-mtp4-integrated-20260921/`。源码基线、候选
归档、修改补丁、展开配置、二进制 SHA、原始 SSE 和恢复日志均保留。

## 内存顺序参考

- [NVIDIA PTX memory consistency](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [NVIDIA GPU-initiated communication design](https://developer.nvidia.com/blog/a-practical-guide-to-gpu-initiated-communication-for-molecular-dynamics-at-scale/)

参考机制不替代本机 CMP170HX PCIe/P2P 的实测。该实现不依赖跨 GPU 原子加法。
