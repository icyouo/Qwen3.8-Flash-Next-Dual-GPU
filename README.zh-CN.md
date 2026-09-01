[English](README.md) | 简体中文

# Qwen3.8-Flash-Next-Dual-GPU

一个为双 Ampere GPU 定制、完全独立的 **Qwen3.8-Flash-Next** 长上下文推理运行时。
当前验证硬件为 2 × 64 GiB CMP 170HX（SM80）；不要求 CUDA P2P，即使每张卡只有
PCIe 2.0 x4 也可以运行。

本项目自己负责完整推理路径：官方权重转换、W4/W8 混合量化、双 stage 执行、
QSA/GDN/PLE 状态、SSD-PLE、CUDA kernels、事务与恢复，以及 HTTP/SSE 服务。它不是
vLLM、SGLang、llama.cpp 或 DS4 的 fork，运行时也不依赖它们。`transformers` 只是可选
的官方 tokenizer/chat template codec，不参与模型执行和状态所有权。

> **当前状态：research preview。** 双卡真实 CUDA 执行、自定义 artifact、事务化服务、
> batch-1 decode 和 grouped-MMQ prefill 已通过验证；可选的 width-one MTP 流水也已经通过
> 32K/128K/256K 方向性性能与状态一致性测试。完整 tokenizer/logit golden parity、严格
> 262,080 + 64、覆盖 accept/reject 的 MTP 质量门禁及长时间故障测试仍未完成。当前数据
> 证明运行时机制和性能，不代表最终模型质量，也不代表 256K 已达到生产发布标准。

## 实测结果

以下 q38 数据均来自 Ubuntu 主机 `p3-ultra`：2 × 64 GiB CMP 170HX、driver
610.43.02、CUDA 13.1、`sm_80`、已校验的 cut-25
`Q38_AMPERE_QUANT_POLICY_V5` artifact、batch size 1。除非表格明确标注 MTP，否则均关闭
MTP。

### Prefill

新的 prefill 数据面把 top-10-of-512 MoE 的不规则小任务转换成确定性的 expert-grouped
矩阵任务，再让两张 GPU 以 512-token slab 流水执行。

| Prefill 路径 | Boundary chunk | 4,096-token 总耗时 | Prefill |
|---|---:|---:|---:|
| 旧 route-wise atomic kernel | 4,096；内部 tile 32 | 54.66 s | 74.93 tok/s |
| Grouped MMQ；内部 tile 32 | 4,096 | 22.36 s | 183.15 tok/s |
| Grouped MMQ；512-token slab；双 stage 串行 | 4,096 | 8.62 s | 475.15 tok/s |
| **Grouped MMQ；512-token slab；双 stage 流水** | **512** | **5.02 s** | **815.78 tok/s** |
| 旧 DS4 参考 | 不适用 | 4,102 tokens 用时 5.27 s | 777.9 tok/s |

最后一行是同一机器上的历史数据，并非相同 artifact 的严格 A/B。最终 q38 路径相对最初
native prefill 提升 **10.9×**，达到了此前 DS4 的性能级别。输入为重复的合成 token，
因此它是真实 CUDA 与状态机性能测试，不是文本质量结论。

### Decode

Prefill 和 decode 有意使用两套不同 kernel family。完成流水 4K prefill 后继续生成 64
tokens，实测 **27.89 tok/s**、ITL p50 35.01 ms。

| 运行 | Context + output | Stage 0 | Stage 1 + head | ITL p50 | Decode |
|---|---:|---:|---:|---:|---:|
| q38 高吞吐（`durability=off`） | 8,195 + 32 | 18.89 ms | 19.21 ms | 38.65 ms | **25.81 tok/s** |
| q38 严格持久化 | 8,195 + 32 | 19.08 ms | 19.21 ms | 43.87 ms | **22.84 tok/s** |
| q38 短上下文基线 | 8 + 64 | 13.25 ms | 13.10 ms | 31.32 ms | **31.76 tok/s** |
| Decode 优化前的 native runtime | 约 8K | 40.87 ms | 40.92 ms | 87.03 ms | 约 11.4 tok/s |
| 旧 DS4 参考 | 约 8K | 约 25.5 ms | 约 26.0 ms | 约 51.5 ms | 约 19.4 tok/s |

8K 高吞吐模式下，两段 GPU 计算合计 38.10 ms，而端到端 p50 是 38.65 ms。严格模式更慢，
是因为每个成功的变更 RPC 都要等待 `fdatasync`；所以任何性能数据都必须同时标明
durability mode。

#### 长上下文 exact-QSA R2

精确并行 selector 与 tiled attention 消除了此前的长上下文 decode 崩塌，同时没有修改
512-block 选择预算或 tie 语义。

| Context | 优化前 decode | Exact-QSA R2 | 加速 | ITL p50，优化前 → R2 |
|---:|---:|---:|---:|---:|
| 32,768 | 20.11 tok/s | **27.84 tok/s** | **1.38×** | 47.88 → 32.79 ms |
| 131,072 | 10.71 tok/s | **26.03 tok/s** | **2.43×** | 91.58 → 36.33 ms |
| 262,080 | 6.21 tok/s | **22.67 tok/s** | **3.65×** | 159.19 → 42.61 ms |

条件：双 64 GiB CMP 170HX、单并发、`durability=off`、关闭 MTP、使用官方 tokenizer
处理源码语料、排除模型启动；每组先单独生成一个 seed token 计算 TTFT，再严格测量后续
五次真实 GPU decode。五个样本只用于方向性吞吐基线，不用于解释 p95/p99。128K 与
256K fixture 会在 69,579 个唯一语料 token 后重复，因此不代表最差 PLE locality，也不
构成模型质量 parity 结论。

#### 可选 width-one MTP

首版优化 MTP lane 会保留 draft 的 QSA row，把两个 target row 拆成两个单 token
microbatch，并让下一行的 stage 0 与当前行的 stage 1 流水执行。它只在
`--enable-mtp --mtp-width 1` 下启用；普通 `decode_one()` 和 width 大于 1 的行为不变。

| Context | 新二进制 plain decode | MTP width 1 | MTP 有效 ITL | 相对 plain 提升 |
|---:|---:|---:|---:|---:|
| 32,768 | 28.84 tok/s | **35.88 tok/s** | 27.87 ms | **24.4%** |
| 131,072 | 26.05 tok/s | **32.91 tok/s** | 30.38 ms | **26.3%** |
| 262,080 | 22.61 tok/s | **28.88 tok/s** | 34.62 ms | **27.7%** |

条件与上面的 exact-QSA 测试一致：单并发、`durability=off`、一个 seed token、五次被测
transaction。每档 fixture 的 draft 都恰好 5/5 接受；这个小型确定性样本只能证明
retained-row fast path 和吞吐有效，不能代表一般 acceptance rate 或模型质量。相同新二进制
的 plain mode 在 128K/256K 相对旧基线波动不超过 0.3%，32K 快 3.6%，通过了 1% 无回退
门禁。

原始证据、准确命令、已知限制和未完成门禁统一记录在 [READINESS.md](READINESS.md)。

## 架构

```text
OpenAI-compatible 或 token-native client
                    │
                    ▼
       q38_sidecar.py — HTTP / SSE / cancel
       tokenizer 与 chat template 为可选 codec
                    │  ExecutorRPC V1 / Unix socket
                    ▼
            单 native process、单语义 writer
                    │
        ┌───────────┴───────────────────────────┐
        │                                       │
 GPU0 / stage 0                         GPU1 / stage 1
 layers 0..24 + PLE                     layers 25..47 + LM head
        │                                       ▲
        └── BF16 4H via pinned-host ring ───────┘
              不要求 NCCL，也不要求 P2P
```

模型按连续层切分。单个 token 必须先经过 stage 0，再进入 stage 1，所以单会话 decode 在
两卡之间仍是串行的；stage boundary payload 很小，实测 PCIe 带宽不是 decode 瓶颈。

### 专用 prefill lane

Qwen3.8-Flash-Next 的每个 token 都会路由到 512 个 experts 中的 10 个。旧路径逐 route
发起大量小任务、重复加载 scale，并通过 FP32 atomic 累加结果。优化后的路径改为：

1. 在 GPU 上构建确定性的 expert-major route plan；
2. 按 expert 打包 assignment，执行直接 W4/W8-A16 Tensor Core MMQ；
3. 把 router weight 折叠进中间 activation；
4. 每个 assignment 单独写出 FP32 结果；
5. 按固定顺序归约每个 token 的十条 route，不再使用 atomic。

请求会被切成 512-token slabs。三个 pinned-host boundary buffers 按
`free → GPU0 D2H → ready → GPU1 H2D/compute → free` 循环。当 GPU1 处理 slab *n*
时，GPU0 可以同时生成 slab *n + 1*；第三个槽位用来保证 buffer ownership 并吸收传输
时序抖动。这是“两张 GPU + 三个缓冲区”，不是三卡流水。

```text
时间 ─────────────────────────────────────────────────────────────►
GPU0   slab 0     slab 1     slab 2     slab 3     ...
GPU1              slab 0     slab 1     slab 2     ...
ring      A           B          C          A
```

`grouped` 是默认 prefill 路径；`Q38_CUDA_PREFILL_MOE=legacy` 仅用于诊断回退，
`Q38_CUDA_PROFILE_PREFILL=1` 用于 CUDA-event profile。Grouped 路径属于新的 numerical
identity，不能复用旧算术路径创建的 session 或 READY identity。

### 独立 decode lane

Batch-1 decode 继续使用专用 GEMV/MoE/top-k kernels，不会为了复用 prefill 而把单行补成
矩阵。QSA 为全部 value dimensions 复用一次 FP32 probability 计算；512-expert router
使用确定性 stable top-k。长上下文 QSA 同样保持精确：历史不超过 512 个 block 时直接
跳过打分；更长历史使用全 GPU 并行的四头 score 扫描、四轮 byte-wide 并行 radix，以及
稳定的升序 gather。Attention 的 score 与 value 计算按每个 head 四个 tile 展开，让 24 个
head 最多占用 96 个 CTA，不再把工作集中在 24 个 SM 上。这些修改保留原来的 FP32 归约
顺序和 threshold tie 规则，并不是近似最近邻检索。诊断回退开关如下：

```sh
Q38_CUDA_DECODE_GEMV=scalar
Q38_CUDA_DECODE_MOE=scalar
Q38_CUDA_DECODE_TOPK=scalar
Q38_CUDA_PROFILE_DECODE=1
```

### Retained-draft MTP lane

Width-one MTP 在 target 验证后不会再次执行 draft token。Backend 会把 provisional QSA row
保留在 target epoch 下；target 接受 draft 时，commit 直接消费这行。两个 target row 以单
token chunk 进入既有 stage scheduler，从而让 GPU0 的 row `n+1` 与 GPU1 的 row `n`
重叠。Reject、cancel、rollback 和 partial replay 都会先丢弃或重建 provisional state，之后
才允许发布 canonical state。

当前 draft generation 仍发生在两行 target pipeline 之前；进一步让 draft 本身与 target
verification 重叠属于后续优化。MTP 默认关闭，plain scheduler 完全不进入 retained-state
路径。

## 模型 artifact 与内存分布

唯一 source of truth 是固定 commit 的官方 BF16 checkpoint：

```text
repository      Qwen/Qwen3.8-Flash-Next
source commit   de4b8e4d43b917e7706784d8bb445c9af86a3540
policy          Q38_AMPERE_QUANT_POLICY_V5
stage split     GPU0: layers 0..24；GPU1: layers 25..47 + LM head/MTP
context target  262,144 tokens
```

这个 artifact **不是 NVFP4、AWQ、GPTQ 或 GGUF**。它由固定版本的官方 checkpoint 直接
编译成面向 Ampere、版本化、content-addressed 且按 stage ownership 布局的格式。

| Tensor 类别 | 存储格式 | 运行方式 |
|---|---|---|
| Layers 2–45 routed experts | symmetric W4，group 128 | Prefill 使用 W4A16 grouped MMQ；decode 使用 batch-1 W4 kernel |
| Layers 0/1/46/47 routed experts | symmetric W8，group 128 | W8A16；边缘 experts 保留更高精度 |
| Embedding、LM head、attention/GDN projections、shared experts 等 always-active matrices | symmetric W8，group 128 | BF16 activation + BF16 group scale |
| MTP matrices 与 experts | symmetric W8，group 128 | 仅在 `--enable-mtp` 时加载 |
| PLE embedding table | row-scaled FP8 E4M3FN | 约 47.68 GiB 常驻 SSD；每行一个 BF16 scale |
| Router、norm、HC、convolution 等关键控制 tensors | 保留 BF16/F32/I64 | 不做一刀切量化 |
| Main QSA K/V 与 compressed index | BF16 | 256K capacity baseline |
| GDN recurrent/accumulator state | FP32 | 避免长序列循环误差累积 |
| Vision tensors | 跳过 | 当前 runtime 仅支持文本 |

只有 PLE table 常驻 SSD。默认 host PLE cache 是 8 GiB，且有硬上限；完整 PLE 不会复制
进内存或显存。读取路径是 `io_uring READ_FIXED + O_DIRECT → registered pinned host
buffer → async H2D`，当前实现**没有使用 GPUDirect Storage**。各 stage 的权重与活跃模型
状态常驻对应 GPU。

[tools/q38_quant_policy.py](tools/q38_quant_policy.py) 中的规则是量化策略唯一真源。策略
digest、source commit、tensor hashes、stage cut、runtime hash 和 state layout 都属于
artifact/session identity，任何不一致都会 fail closed。

## 运行时保证

- Append、decode、MTP 共用一个 semantic writer 与 commit 顺序。
- Chunked append 对整个请求保持原子性，并要求双 stage acknowledge；失败完整回滚。
- 只允许 suffix continuation：不会静默重算或替换已有 prefix；发生 fork 必须 cold rebuild。
- QSA/GDN/PLE/MTP/RNG provisional state 只在 commit 后发布。
- 无 P2P 的 pinned-host transport，包含 position 与 payload integrity 校验。
- 支持 cancellation、deadline、request-ID idempotency 和 stop-token-aware commit。
- CUDA/device/transport fatal error 会在尝试 rollback 后使 executor 失效，不会在不确定状态
  上继续执行。
- Artifact 构建可流式、可恢复，source/output 均有 hash，READY 最后原子发布。
- 持久化策略显式分为 `strict` 与 `off`，不能把两种模式的数据混在一起比较。

严格容量目标是 262,080-token prompt 加 64 个已提交输出，最终长度精确等于 262,144。
这是尚待完成的 release gate，不是已经完成的发布声明。

## 在 Ubuntu 上构建与测试

CUDA 编译与运行均应在 Ubuntu 完成。macOS 可以用来编辑源码，但不是受支持的 runtime
host。

```sh
make clean
make -j2 build/q38_runtime_tests build/q38-runtime
./build/q38_runtime_tests
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
make -j1 cuda-check cuda-runtime cuda-test cuda-bench
```

`make verify PYTHON=.venv/bin/python` 会运行 CPU、Python、CUDA compile 与 runtime build
门禁；`cuda-test` 需要 GPU。Grouped-MMQ 已通过真实 SM80 fixtures；验证机目前没有安装
`compute-sanitizer`，因此不能把它写成已通过门禁。

## 准备模型

仓库不包含模型权重。

### 1. 固定官方 source manifest

```sh
python3 tools/q38_hf_fetch.py \
  --repo Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --expected-commit de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --output /data/models/Qwen3.8-Flash-Next-official-source \
  --jobs 4 \
  --manifest-only
```

### 2. 流式编译 Q38 artifact

```sh
.venv/bin/python tools/q38_prepare_artifact.py \
  --source /data/models/Qwen3.8-Flash-Next-official-source \
  --metadata /data/models/Qwen3.8-Flash-Next-official-metadata \
  --output /data/models/Qwen3.8-Flash-Next-q38-cut25 \
  --session-hash 0x380025 \
  --cut 25 \
  --jobs 1 \
  --stream-from-manifest \
  --prune-source-shards
```

每个 source shard 都会先校验，再以受限内存完成转换并重新 hash，发布成功后才允许删除；
流程可以断点恢复。磁盘空间充足时可去掉 `--prune-source-shards` 以保留官方 BF16 source。
只有两个 stage、PLE layout、tensor census 和 identity 全部通过后，才会出现 `READY.json`。

## 启动服务

默认使用 strict durability：

```sh
.venv/bin/python tools/q38_launch.py \
  --ready /data/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --runtime build/q38-cuda-runtime \
  --socket /tmp/q38-executor.sock \
  --snapshot /var/lib/q38/session.q38j \
  --durability strict \
  --host 127.0.0.1 \
  --port 30000
```

启动器会在执行 CUDA 前检查 source commit、quantization policy、全部 artifact segments、
runtime identity、stage plan、state layout 与 context/sampling contracts。生产 PLE 必须成功
建立 `io_uring + O_DIRECT` lane，不会静默回退到 buffered I/O。

Benchmark 或客户端能够重放完整 canonical token history 时，可以只关闭 crash recovery：

```sh
.venv/bin/python tools/q38_launch.py \
  --ready /data/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --runtime build/q38-cuda-runtime \
  --durability off
```

`durability=off` 只会移除 snapshot journal 及其 crash-rebuild 保证，不会弱化进程存活期间
的 transaction、rollback 或 committed-token 语义。`--dry-run` 可只验证 artifact 而不
启动；MTP 默认关闭，只有 plain lane 的正确性与显存门禁通过后才应使用 `--enable-mtp`。

## API

创建唯一 live session：

```sh
curl -sS -X POST http://127.0.0.1:30000/v1/q38/sessions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"deep-1"}'
```

追加 token IDs 并流式接收已提交输出：

```sh
curl -sS -N -X POST \
  http://127.0.0.1:30000/v1/q38/sessions/deep-1/execute \
  -H 'Content-Type: application/json' \
  -d '{"append_token_ids":[1,2,3],"max_new_tokens":64,"stream":true}'
```

如果提供 `full_token_ids`，它必须是 server canonical prefix 的精确 extension。运维接口：

```text
POST /v1/q38/cancel
GET  /v1/q38/sessions/{id}
GET  /v1/q38/metrics
```

配置官方 tokenizer 后，sidecar 还会提供 `POST /v1/chat/completions`。Codec 不进入
ExecutorRPC ABI，也不拥有模型状态。

## 验证与路线图

接下来的发布顺序为：

1. 发布 grouped-MMQ numerical identity，并从 cold state 启动；
2. 固定 tokenizer 生成的 golden prompts，与可信官方 BF16 reference 对比 logits/tokens；
3. 分别运行全新 32K、128K 和严格 262,080 + 64 门禁，并记录完整 GPU、host、transport
   与 PLE metrics；
4. 证明 near-256K suffix continuation 不会重放旧 prefix；
5. 在每档 context 验证 rollback、cancel、duplicate request、crash recovery 与 fault injection；
6. 扩展 opt-in width-one MTP 的 accept/reject、cancel、parity 与长生成门禁，再优化更宽
   draft width；
7. 完成长时间稳定性与 failure-injection soak。

完整实现/证据边界见 [READINESS.md](READINESS.md)，详细架构与发布门禁见
[runtime architecture document](docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md)。

## License

运行时代码使用 [MIT License](LICENSE)。模型权重不属于本许可范围；下载、转换及使用模型
仍须遵守上游仓库许可。
