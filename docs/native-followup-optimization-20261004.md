# 固定 MTP4 的逐项优化（2026-10-04，已完成）

六项改动均已实现、逐项验证和测速。保留四项到 `mtp4-native-fast`，正式启动
入口独立复测网页 **172.81 tok/s**、代码 **174.70 tok/s**，比上一版正式结果
分别提升 **6.62% / 7.77%**。Mac 和 Z4 原生仓库已同步，服务器二进制已替换。

条件保持 HP Z4 双 64 GiB CMP170HX、SM80、每卡 200W、原有 W4/W8、RAM PLE
（51,840,248,832 字节，预算 64 GiB）和固定 MTP4。单并发、greedy、关闭
thinking、durability=off；测速关闭完整 logits 和 CUDA event 诊断。

## 正式入口结果

每项先预热 128 token，再测三轮。请求省略 `q38_mode` / `q38_mtp_width`，
由正式配置默认启用 MTP4。启动器重新校验全部 354 个独立 segment、
212,367,104,472 字节的哈希。

| 请求 | 三轮 decode tok/s | 均值 | 上一版均值 | 提升 |
|---|---|---:|---:|---:|
| 网页，16 输入 / 1024 输出 | 172.97 / 173.15 / 172.32 | 172.81 | 162.09 | 6.62% |
| 代码，24 输入 / 512 输出 | 173.80 / 174.47 / 175.83 | 174.70 | 162.10 | 7.77% |

吞吐沿用已提交输出 token 数 / 首尾非空 SSE 内容事件间隔，不计草稿 token。
更保守的 N−1 decode 均值为网页 **172.64**、代码 **174.36**；包含 TTFT 的
端到端均值为 **171.53 / 171.54 tok/s**。六次 SSE 内容与上一版冻结输出逐字
一致。每轮均无 major fault 增长、进程 swap、CUDA Graph 回退、CUDA 分配失败
或 executor failure/rollback。

## 逐项实验与取舍

每个候选都在完整模型正确性门禁通过后，用独立、无诊断进程测速。下表中的
单项候选通常叠加 shared overlap；它们不是纯算子时间，也不是可相加的收益。
源码/二进制经历迭代，表中同时保留早期及最终二进制的对照，避免把编译版本
或不同轮次的差异全部归因于开关。

| 改动 | 网页 tok/s | 代码 tok/s | 最终处理 |
|---|---:|---:|---|
| 初始二进制关闭 shared overlap | 162.46 | 163.90 | 同二进制对照 |
| ① shared expert / routed P2P 重叠 | 169.42 | 171.17 | 保留；同二进制约 +4.3% / +4.4% |
| 最终二进制只开 shared overlap | 170.11 | 174.28 | 后续实验共同对照 |
| ② 直接 BF16 top10 归约，原 5 CTA | 165.44 | 166.21 | 放弃此调度 |
| ② 改为每 token 10 个 hidden tile CTA | 171.68 | 172.68 | 保留到组合配置 |
| ③ full-K Marlin 持久化调度 | 166.67 | 167.34 | 关闭；没有稳定收益 |
| ③ Marlin N128 / 256-thread tile | 170.40 | 171.27 | 关闭；没有稳定收益 |
| ④ MoE combine + HC residual 融合 | 169.95 | 170.32 | 保留到组合配置 |
| ⑤ GDN conv + prepare 融合 | 169.99 | 169.35 | 关闭；没有稳定收益 |
| ⑥ MTP normalized embedding cache + broadcast | 172.05 | 172.04 | 保留到组合配置 |
| ①②④⑥ 组合，候选入口 | 172.97 | 173.58 | 全部门禁通过 |
| ①②④⑥ 组合，正式入口独立复测 | 172.81 | 174.70 | 已安装 |

②④⑥ 单项增益与输入相关，不能声称每项都独立加速所有请求。组合候选相对
最终二进制只开 overlap，网页 +1.68%，代码 −0.40%；正式入口复测代码达到
174.70。保留配置依据组合结果和完整等价门禁，不能用单次峰值作稳定吞吐。

### 实现与数值边界

1. Graph overlap 的两条分支依赖同一先行片段；下一消费者依赖 P2P commit
   与 shared 分支共同完成。支持 paired/rank-local 图；eager 路径按原顺序
   执行。沿用 generation fence、read-done、独立 scratch、超时和 drain/reset。
   捕获回调同步执行，独立分支禁止访问 collective buffer 或嵌套 collective。
2. 直接归约读取 Marlin BF16 专家输出，在寄存器转 FP32，按原 top10 槽位顺序
   加权，省掉完整路由 FP32 中间数组；未拥有的尾部清零。每 token 展开为十个
   hidden tile block，避免最初五个 CTA 的并行度不足。大批量保留原路径。
3. 两种 Marlin 调度候选都保留代码和独立开关，默认关闭；没有改权重或 scale。
4. MoE combine 与 HC finish 融合显式保留 MoE block、gate 与 residual 的
   BF16 舍入边界，目标图动作的 finish 消费归约结果及 shared 分支结果。
5. GDN 融合保留原 conv BF16 舍入、norm 求和顺序、canonical 状态及全部前缀
   快照；两个 rank 的 eager/graph 都通过等价检查，但测速没有支持默认启用。
6. MTP 缓存提前计算不可变的 embedding RMSNorm，草稿阶段直接 gather；
   投影仍用原始 GEMV。broadcast 与 embedding addition 融合保持相同 BF16
   舍入，支持原位写回。新增约 **1.18 GiB/卡** 的常驻表；bulk prefill 沿用
   原路径。缓存没有重新量化或近似投影，拒绝/取消后不需要重建不可变表。

## 验证

C++ runtime/rank-worker 检查和 **88 项 Python 测试**通过。每项与最终组合均
检查五类输入（含 512/8192 token 上下文）的生成 token、248320 维完整 BF16
logits、十个强制拒绝前缀、九项完整数学/JSON/Unicode/可执行代码回答，及
草稿取消、append 取消、deadline 后的 reset/继续生成恢复。

GPU fixture 覆盖图分叉延迟、连续排队、捕获失败、descriptor mismatch、
drain/reset；W4/W8 的路由重排、变化的 owned prefix 和 poisoned 尾部；
两种 Marlin 调度的 eager/graph 完整 BF16 输出；HC 舍入；两个 rank 的 GDN
状态/快照；MTP BF16/W4/W8 embedding 缓存、gather 和原位 broadcast addition。
CUDA 重建及 fixture 日志已保存，不以成功编译代替实际 GPU 或模型检查。

所有数值等价均相对于此前已验证的 **Marlin scaled-BF16 专家基准**，不意味
着与原 FP32-scale MoE 后端或未量化官方权重逐位一致。这是短上下文单请求
decode 证据；8192 输入短生成的正确性已检查，没有新增 256K 性能、长输出
或持续负载可靠性结论。本轮没有重新采集算子占比，旧 profile 的占比不能
直接当作新版瓶颈数据。

## 配置、启动与回退

本次新增的固定策略：

```text
Q38_TP_MOE_SHARED_OVERLAP=1
Q38_CUDA_MARLIN_DIRECT_REDUCE=1
Q38_CUDA_MOE_HC_FINISH_FUSED=1
Q38_CUDA_MTP_INPUT_CACHE=1
Q38_CUDA_MARLIN_PERSISTENT=0
Q38_CUDA_MARLIN_WIDE_N=0
Q38_CUDA_GDN_CONV_PREPARE_FUSED=0
```

启动器会清理环境中的其他 `Q38_*` 变量，以 `tools/q38_tp_profiles.py` 内的
固定策略执行；修改 shell 环境不会覆盖它。原有其他 profile 保留。

Z4 二进制：
`/home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU/build/native-mtp4-fast/q38-cuda-runtime`。
SHA256：`7199f47ebf5d8dabf6b05611c14e55d9d66e7caf5e3dc8f4c95f2026f169613b`。

```bash
cd /home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 .venv/bin/python tools/q38_tp_launch.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp-root /home/icy/q38-native-target160-20261004/gdn-qkvz-artifact \
  --runtime build/native-mtp4-fast/q38-cuda-runtime \
  --execution-profile mtp4-native-fast \
  --ple-storage ram --ple-ram-budget-gib 64 \
  --state-dir /home/icy/q38-native-followup-session --port 30001
```

使用新的、尚不存在的 state 目录；不要复用旧二进制身份的 session。
测试进程已退出，两卡显存已释放，200W 限制仍在，挖矿保持停用。

旧二进制保存在同目录 `q38-cuda-runtime.162-tps-20261004`，SHA256 为
`422eb7ff36cf05cd808d17165a1d541363b2fa1824a462564b5c454ed21842a0`。
同目录的 `q38_tp_profiles.162-tps-20261004.py` 保存旧策略，其内容已与上一版
正式启动器 manifest 的策略逐项核对。需要回退时，先停服务，恢复旧二进制
及该策略文件到 `tools/q38_tp_profiles.py`，并使用新 state 目录。

源码重建沿用 `make -B cuda-runtime CUDA_COMPAT=0`；不要在完整模型加载时
同时进行 CPU/CUDA 编译。新增 `mtp-input-test` 已接入 `native-fast-tests`。
基础布局转换及完整启动说明见[上一阶段报告](native-fast-optimization-20261004.md)。

## 原始证据

Mac：`/Users/icy/Homelab/.artifacts/q38-native-followup-20261004/`；
Z4：`/home/icy/q38-native-target160-20261004/`。其中：

- `followup-*/full/`：每个候选的 token、BF16 logits、冻结 oracle 比较、质量
  回答、拒绝前缀及取消恢复；最终组合为 `followup-final-combined/full/`。
- `followup-*/http/`：请求、原始 SSE、首尾时间、TTFT、N/N−1/端到端吞吐，
  各轮前后的内存、接受率和 executor metrics。
- `followup-production-confirmed/`：正式启动命令、完整 segment 校验、默认
  MTP4、六次 SSE 内容一致性、二进制 SHA、性能汇总及完成标记。
- `native-followup*.log`：真实 CUDA 重建和 GPU fixture；`cpu-tests.log` 与
  `final-profile-tests.log`：CPU 检查及最终 88 项 Python 测试。
- `FINAL.json`：当前汇总；`rollback-policy.json` 和旧策略模块：回退依据。

运行脚本为原 artifact 根目录的 `run-native-followup.py` 和
`run-followup-production-confirm.py`，后者使用正式仓库和默认 HTTP 请求。
上一版冻结 oracle、原始对照和脚本仍保存在 Mac
`.artifacts/q38-native-target160-20261004/`，不修改历史数据。
