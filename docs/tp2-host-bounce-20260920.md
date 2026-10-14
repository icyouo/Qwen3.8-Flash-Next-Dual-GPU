# Z4 双 CMP 170HX：PP → TP2/EP 修复与验证

> 后续同日更新：移除 WX4100 并启用 Headless Boot 后，双卡 64 GiB BAR1/P2P 已打通；引擎新增可选 P2P 和 PLE RAM 模式。见 [后续实测与命令](p2p-ram-20260920.md)。本文保留此前 host-bounce 修复阶段的记录。

日期：2026-09-20。目标：`hp-z4-server`（`hp-z4-g4`），双 64 GiB CMP 170HX。

本轮把已有 TP smoke 接入了原生事务执行器及 HTTP/SSE 服务，并修复了影响正确性的同步和数据布局问题。当前是可运行的实验性 eager TP2/EP 路径，固定 256 步实测 32.43 tok/s。尚未证明 TP 比 PP 快，也没有通过完整模型质量或 256K 发布门禁。原来的 PP artifact、默认入口和 `build/q38-cuda-runtime` 保留。

## 已确认的环境与运行位置

| 项目 | 路径 / 配置 |
|---|---|
| SSH | `ssh hp-z4-server` |
| 远端源码 | `/home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU` |
| 本地源码 | `/Users/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU` |
| 驱动 / 编译 | driver 610.43.02；CUDA 13.3 `/usr/local/cuda`；g++-14；SM80 |
| 原 PP artifact | `/home/icy/models/Qwen3.8-Flash-Next-q38-cut25` |
| 原 TP artifact | `/home/icy/q38-tp2-local` |
| 本轮低通信 TP variant | `/home/icy/q38-tp2-lowcomm` |
| TP runtime | `build/te-repair/q38-cuda-runtime` |
| 服务状态 | `/home/icy/q38-tp2-serving` |
| API | 目标机器上的 `http://127.0.0.1:30001/v1` |

驱动仓库为 `/Users/icy/Code/cmpunlocker`，目标机也有其副本。本轮没有修改驱动或切换内核模块；没有启用 P2P。数据面仍走 pinned host bounce。验证期间双卡占用约 51,208 MiB/卡；查询最近两小时内核日志没有发现 `NVRM: Xid`。

## 根因与修复

1. **Collective 与生产计算的 stream 不一致。** 后端使用 nonblocking CUDA stream，旧 bridge 在默认 stream 拷贝，可能读取尚未算完的 tensor。现在显式传递生产 stream，按顺序完成 D2H、规约/拼接、H2D，并恢复调用线程原有 device。CUDA 的默认 stream 不保证和 nonblocking stream 自动同步，参见 [NVIDIA stream synchronization 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)。
2. **每次通信反复分配和释放内存。** 改为按 collective group 复用 pinned 缓冲和 device scratch；小规约直接在 CPU 上处理，避免额外 kernel 和整段 GPU scratch 往返。
3. **BF16 规约截断及 MoE 重复舍入。** BF16 求和改为 round-to-nearest-even；专家分片保留 FP32 partial，先跨 rank 求和，再统一转 BF16。PP 的原有输出路径保留。
4. **路由和 prefill 布局。** 专家 ID 重映射改为 GPU kernel。多 token 的列分片输入按 token-major stride 搬运，避免把多行当作连续半向量。两个 rank 的 expert-grouped packed route 顺序不同，不能直接相加；TP 暂时使用逐 token MoE 路径，避免错误规约和 scratch 越界。PP grouped prefill 仍使用原路径。
5. **服务的唯一状态所有者。** 新的 `--tp2-root` 在一个 `DualStageExecutor` 下创建四个 rank/stage 后端。两卡都计算每个逻辑 stage 的分片；stage0/1 只是兼容现有层分段和事务边界，已不再表示两张不同的卡。发布 token、sampling、commit 均由单一协调者拥有。配对执行会检查 rank 输出和 stage0 boundary 一致性。
6. **失败处理。** collective 检查 group/rank/尺寸/操作，超时或 rank 错误会唤醒另一 rank 并使执行器停止接受继续计算。当前原型在执行错误或取消后需要重启，不宣称可透明恢复。

核心代码：`cuda/cuda_te.cu`、`cuda/cuda_backend.cu`、`cuda/cuda_tp_backend.cc`、`cuda/q38_moe.cu`。

## 减少通信的 artifact 方案

`tools/q38_tp_replicated_controls.py` 从已经校验的 PP 源 tensor 创建新的 TP index；底层 segment 使用硬链接，不重新量化、不覆盖原 artifact。所替换 tensor 的 data/scale SHA-256 会先校验，输出 manifest 记录每个 index SHA 和新增 resident bytes。

| variant | 复制到每张卡的内容 | 每卡新增权重 |
|---|---|---:|
| `q38-tp2-controls` | HC、小 GDN a/b、shared-expert gate/up | 723,996,160 bytes |
| `q38-tp2-lowcomm` | 上述内容 + GDN QKV/Z、QSA Q | 1,682,460,160 bytes |

保留 attention/output 的 tensor 分片和 routed-expert 分片。低通信版本在短文本 smoke 中每 token 为 0 次 AllGather、97 次 AllReduce。当前仍会同步规约，不是 NVIDIA Transformer Engine 的集成，也不依赖 NCCL。

创建新 variant（输出目录必须尚不存在，源和输出须处于同一文件系统）：

```bash
.venv/bin/python tools/q38_tp_replicated_controls.py \
  --source /home/icy/models/Qwen3.8-Flash-Next-q38-cut25 \
  --tp /home/icy/q38-tp2-local \
  --output /home/icy/q38-tp2-lowcomm-new \
  --profile attention-inputs
```

`q38_tp_launch.py` 使用四个 TP index 及当前 runtime/kernels/tokenizer 等输入生成独立 session identity，不能沿用 PP identity。完整校验覆盖 357 个唯一 segment、210,836,224,440 bytes；按 inode 去重硬链接。验收所用 identity 为 `5254427814120297663`。源码变化后应重新生成，不应硬编码该数值。

## 验证结果与解释

| 检查 | 结果 |
|---|---|
| Python 全套 | 66 tests 通过，含 artifact 哈希损坏和 provenance 拒绝测试 |
| CPU runtime | `q38 runtime tests: ok` |
| CUDA kernel suite | 通过，含 W4/W8 expert FP32 partial 及互补 expert mask 重构 |
| 双卡 collective regression | 通过：延迟 producer stream、in-place 多行 gather、BF16 RNE、device 恢复、200 次双 group 规约、peer failure、双卡 optimized top-k |
| 中文解释、17×23、Python 函数 | HTTP 200，输出连贯；算术输出 391 |
| 最终后台服务 HTTP / SSE | 完整校验后重启通过；17×23 输出 391；2+3 流式输出 5，并收到 `[DONE]`；零失败、零回滚 |
| 初始 1,024 tokens + 8 output | 4.064 s；实际计算 1,031 tokens |
| 两次各追加 128 tokens + 8 output | 0.843 / 0.845 s；各实际计算 136 tokens；incremental hit，无前缀重放 |
| 原生 RPC 256 append + 256 output | 32.4333 tok/s；ITL p50 30.834 ms、p95 31.736 ms；append 1.159 s |
| 状态一致性 | 256 步后 canonical=512，target/stage0/stage1=511；保留最后一个生成 token 待下轮消费是预期语义 |

固定步数测试使用重复合成 token、greedy、plain eager、单并发、durability=off，不包含模型加载，不代表自然语言质量。native RPC benchmark 的首个输出可直接取自 prefill logits；p50/p95 更适合描述稳定逐 token 延迟。

性能节点应分别解释，不能把不同输入和代码阶段当作严格 A/B：

| 节点 | Decode |
|---|---:|
| 修复前旧 TP smoke，64 步且文本严重重复 | 6.3443 tok/s |
| stream/内存池修复后，开启完整 logit 诊断 | 20.8952 tok/s |
| 复制 controls 的中间版，短文本 64 步 | 27.7522 tok/s |
| 复制 attention inputs 的中间版，42 步至 EOS | 35.8703 tok/s |
| 最终 FP32 expert reduction + 原生 TP 服务，固定 256 步 | **32.4333 tok/s** |
| 同机此前 PP 固定 256 步基线，初始会话长度不同 | 38.5426 tok/s |

尚未取得 TP 相对 PP 的性能胜出。当前 97 次/token 的 host-bounce 同步仍是优化方向；不能直接把 `collective_seconds` 当作纯 PCIe 耗时，因为它包含等待生产 kernel 完成的时间。

后续原始证据复查还确认：上述 PP 256 步输出全为 token 42，MTP 和 graph 均未启用；新的 P2P TP 测试输出包含 124 种 token。两者的 token 历史、专家访问及缓存负载不构成严格同条件比较，不能从该表单独量化 TP 相对 PP 的退化幅度。

第一轮修复后的 PP/TP logit 对比，首步 cosine 为 0.999976、最大绝对差 0.125，两个 TP rank 的 logit 逐 bit 相同；但 PP/TP 在后续接近的候选分数处出现 greedy token 分歧。此数据来自 FP32 MoE 规约加入前的中间版本，不是最终版本的质量门禁。后续不同生成历史上的 logit 不应继续作为同输入 parity 对比。完整 teacher-forced / 官方模型 golden parity 仍需完成。

## 构建与运行

在目标机源码目录：

```bash
make -j2 cuda-runtime te-decode-smoke \
  BUILD_DIR=build/te-repair CUDA_HOME=/usr/local/cuda CUDA_COMPAT=0

LD_LIBRARY_PATH=/usr/local/cuda/lib64 .venv/bin/python tools/q38_tp_launch.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp-root /home/icy/q38-tp2-lowcomm \
  --state-dir /home/icy/q38-tp2-serving \
  --port 30001
```

独立测试命令：`make test python-test`；GPU regression 使用 `make te-test cuda-test BUILD_DIR=build/te-repair CUDA_HOME=/usr/local/cuda CUDA_COMPAT=0`（需设置 `LD_LIBRARY_PATH=/usr/local/cuda/lib64`）。

默认启动会完整哈希校验，所以 HTTP 端口不会立刻出现。`--prepare-only` 只校验并准备 identity。`--skip-segment-hashes` 仅用于诊断，不作为常规启动建议。状态目录使用 launcher lock，重复启动会拒绝。

本轮交付的后台进程日志和 PID 分别在 `~/q38-tp2-serving/service.log`、`launcher.pid`。这是 `nohup` 进程，没有注册系统开机自启。需停止时，确认 PID 对应 `q38_tp_launch.py` 后向该 PID 发送 SIGTERM，launcher 会清理 runtime 和 sidecar。

从本机访问：

```bash
ssh -N -L 30001:127.0.0.1:30001 hp-z4-server
```

然后请求 `http://127.0.0.1:30001/v1/chat/completions`，模型名为 `Qwen3.8-Flash-Next`。可使用 `stream: true`；短回答建议 `enable_thinking: false`。

不要在运行中的 HTTP 服务之外再用另一个 RPC Client 修改同一个 executor。它们的请求序号和 canonical history 不共享；本轮独立 RPC 测速后触发了 stale-writer 拒绝，随后已重启服务。底层 RPC 性能测试应使用独立实例或停止 sidecar，测试完成后重启完整服务。

## 当前限制和下一道门禁

- TP prefill 的 routed MoE 暂为逐 token fallback，不能套用 README 中 PP grouped-MMQ 的吞吐。
- TP 只验证 eager plain；MTP、CUDA graph、compact sampling、durable recovery 不在本轮支持范围。错误或取消后重启。
- 262,144 的配置和预分配成功，不等于通过严格 262,080 + 64 的长上下文验证。
- 自然语言检查是 smoke，完整 teacher-forced logits、路由和长文本质量对照仍待验证。现有 sidecar 的普通文本解码还可能包含 `<|im_end|>` 标记。
- 持续性能优化需要按层测量计算/等待/拷贝，优先减少 97 次同步点，再处理 TP expert-grouped prefill；驱动/P2P 的变更应独立验证。

本地原始证据：`/Users/icy/Homelab/.artifacts/q38-z4-tp2-repair-20260920/`。其中 `before/` 为修改前文件，`first-pass/` 为中间版 logit 对比，`final-evidence/` 包含 HTTP、增量和最终 native RPC 测试。历史 `tools/te_hostbounce/RESULT_TE_RUNTIME.md` 记录的是本轮之前的 artifact smoke 阶段。
