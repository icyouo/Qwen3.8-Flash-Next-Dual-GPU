# 双 CMP170HX 固定 MTP4 原生引擎优化

本文保留最初 162 tok/s 阶段的历史测量及数值合同。随后六项逐项优化已完成，
当前正式 `mtp4-native-fast` 为网页 172.81 / 代码 174.70 tok/s；当前二进制、
配置、验证与回退方式见[后续报告](native-followup-optimization-20261004.md)。

2026-10-04，HP Z4（192.168.1.121），双 64 GiB CMP170HX，SM80，驱动
610.43.02，CUDA 13.3，每卡 200W。权重保持现有 W4/W8，PLE 完整常驻
RAM（51,840,248,832 字节），固定 MTP4、单并发、greedy、关闭 thinking。

正式可选配置为 `--execution-profile mtp4-native-fast`，保留旧配置。
完整模型门禁由 `native-peer-gdn16-tested` 验证；下表为正式启动入口
`native-production-confirmed` 的独立三轮复测，默认请求使用 MTP4：

| 请求 | 三轮 decode tok/s | 均值 |
|---|---|---:|
| 网页，16 输入 / 1024 输出 | 161.03 / 161.04 / 164.19 | 162.09 |
| 代码，24 输入 / 512 输出 | 163.09 / 162.83 / 160.37 | 162.10 |

计时为已提交输出 token 数 / 首尾非空 SSE 内容事件间隔，不计草稿 token。
同时保留 N−1 和包括 TTFT 的端到端数据；网页 N−1 均值 161.93，端到端
160.94；代码 N−1 为 161.78，端到端为 159.43。每项先预热 128 token，正式测速关闭完整 logits 和 CUDA event 诊断。
完整 JSON、环境策略和二进制 SHA 均保存在同名证据目录。

实现内容：

- 固定版本的 Marlin MoE Tensor Core 内核直接编入 C++/CUDA 运行时；无需
  安装 vLLM 或 PyTorch。紧凑路由计划、有界暂存和未拥有专家尾部初始化融合
  减少小批量 MoE 的空工作。
- 无损重排输出头词表行及 GDN QKV/Z；每卡计算自身拥有的词表行、24 个 GDN
  value head、12 个 QSA query head。B/A 控制投影保持完整，稳定 argmax 保留
  全词表与最小 token ID 的 tie break；需要诊断时仍可取得完整 logits。
- 2–7 行 dense 投影使用容量检查过的 8 行 workspace 视图，保留 cuBLAS
  的归约形状。专用小批量输出头复用权重，HC mixture 使用融合 Tensor Core
  路径，HC finish 显式保留 BF16 激活舍入。
- 目标和 MTP 草稿采用有界整段 CUDA Graph；QSA 位置来自设备值，dense/sparse
  变体独立。投机验证、前缀提交、拒绝恢复和取消恢复沿用 canonical 状态机。
- 小型 rank-local P2P 图集合通信合并为一个 CTA，先等待双方 generation ready，
  将结果写入 scratch，等待双方读取完成，再提交到输出。大于 256 KiB 的负载
  保留原路径；非对齐负载保留标量访问。等待仍有超时并按原合同使设备失败。
- GDN 状态更新采用已验证的
  16 列分区；输出头维持 8 warp，QSA value 采用 8 元素 staging。

数值边界：原始 W4/W8 量化数据及 scale 未重新量化。Marlin 的专家解量化权重
采用 scaled-BF16，这与此前 FP32 scale 的寄存器 MoE 后端不是同一浮点数值策略。
后续布局、图和融合优化与已验证的 Marlin 完整模型基准逐位比较；这些证据不能
推导为与未量化官方模型的所有输出完全一致。

验证包括五类 tokenizer 输入（含 512 和 8192 token 上下文）的生成 token 和
248320 维完整 BF16 logits，十个强制拒绝前缀，九项数学/JSON/Unicode/可执行
代码检查，以及草稿取消、append 取消和 deadline 后的 reset/继续生成恢复。
P2P fixture 覆盖 FP32/BF16、1/5/7 行、对齐和非对齐、原地和非原地 gather、
延迟生产者、所有排队中间结果及大负载 fallback。测速期间无 swap、major fault
增长、CUDA Graph fallback、CUDA 分配失败或 executor failure。

这是短上下文 HTTP decode 性能证据。8192 输入的短生成正确性检查通过，但新的
长输出/长上下文性能矩阵、256K 发布门禁和持续负载可靠性仍待独立验证。

## 启动与复测

该阶段 Z4 上测试的二进制路径：
`/home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU/build/native-mtp4-fast/q38-cuda-runtime`。
SHA256：`422eb7ff36cf05cd808d17165a1d541363b2fa1824a462564b5c454ed21842a0`。
该阶段二进制现已另存为同目录 `q38-cuda-runtime.162-tps-20261004`；上述路径
当前安装的是后续报告中的新版。源码已同步至 Mac 和 Z4 的原生仓库。

Z4 已有经过校验的布局，可以直接使用；`state-dir` 必须使用新目录，启动器会
绑定当前源码、二进制、固定策略及四份 rank index 生成独立身份：

```bash
cd /home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 .venv/bin/python tools/q38_tp_launch.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp-root /home/icy/q38-native-target160-20261004/gdn-qkvz-artifact \
  --runtime build/native-mtp4-fast/q38-cuda-runtime \
  --execution-profile mtp4-native-fast \
  --ple-storage ram --ple-ram-budget-gib 64 \
  --state-dir /home/icy/q38-native-fast-session --port 30001
```

默认 HTTP 使用 MTP4，可省略请求中的 `q38_mode` / `q38_mtp_width`。
启动器剥离环境中其他 `Q38_*` 调参变量，以固定策略执行。
本次速度来自 RAM PLE；保留的 `--ple-storage ssd` 不共享这项测速结论。

需要从基础 TP 布局创建新的无损 artifact 时，在同一文件系统上依次执行；
输出目录必须不存在：

```bash
.venv/bin/python tools/q38_tp_vocab_rows.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp /home/icy/q38-native-opt-mtp-20260921/artifact \
  --output /home/icy/q38-vocabulary-row-artifact
.venv/bin/python tools/q38_tp_gdn_packed.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp /home/icy/q38-vocabulary-row-artifact \
  --output /home/icy/q38-gdn-qkvz-artifact
```

工具校验原始量化数据和 scale 的哈希、布局及四份 TP2 记录，新增文件中只
重排行和 scale，不改变数值；未改变的 segments 使用 hard link。Marlin 在
加载时重新打包其设备布局。

Linux SM80 构建命令：`make -B cuda-runtime CUDA_COMPAT=0`，需 CUDA 13.3、
cuBLAS 和 g++-14。已固定版本的第三方内核源码、Apache-2.0 LICENSE 和 NOTICE
随仓库包含在 `cuda/vendor/marlin/`；不会调用外部 vLLM repo。

CPU 检查：`make -B test`，C++ 运行时/rank worker 和 87 项 Python 检查通过。
新路径的 GPU fixture 目标为 `make native-fast-tests CUDA_COMPAT=0`。
另外 `Q38_TP_PEER_FUSED=1 make te-graph-test CUDA_COMPAT=0` 验证了 capture
失败、两组独立图、连续 replay、descriptor mismatch 和 drain/reset/重新建图。
生产配置的 GDN 分区校验使用
`Q38_CUDA_GDN_PARTITION_COLUMNS=16 make gdn-local-test CUDA_COMPAT=0`。
重新构建或更改数值策略后需重新运行完整模型门禁。

## 原始证据

Homelab 的 `.artifacts/q38-native-target160-20261004/` 与服务器
`/home/icy/q38-native-target160-20261004/` 保存：

- `native-peer-gdn16-tested/full/`：token、完整 logits、拒绝前缀、完整质量回答、
  取消/deadline 恢复，及冻结的二进制 SHA 和策略。
- `native-peer-gdn16-tested/http/`：全部请求、SSE 事件、首尾时刻、已提交 token
  数、TTFT、N/N−1/端到端吞吐及每轮前后的完整 metrics。
- `native-production-confirmed/`：正式启动器复测，校验所有 segment 哈希、默认
  MTP4 和候选版本逐字一致的 SSE 内容；执行脚本为同目录上层的
  `run-native-production-confirm.py`。
- `native-fast-final-cpu-tests.log`、`peer-fused-test.log`、
  `native-fast-te-graph-tests.log`、`native-fast-gdn16-layout-tests.log`，以及之前
  保存的 HC finish、Marlin tail、QSA/MTP Graph 和 argmax fixture 日志。

正式启动器复测已完成：354 个独立 segment、212,367,104,472 字节通过哈希
校验；六次请求均使用默认 MTP4，SSE 内容与候选版逐字相同，两种请求的
N−1 decode 均值均超过 160。结果为
`native-production-confirmed/performance-summary.json`，完成标记为
`native-production-confirmed/completed.json`。测试进程已退出，GPU 显存释放；
挖矿保持停用。
