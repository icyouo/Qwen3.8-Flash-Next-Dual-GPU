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
> batch-1 decode 和 grouped-MMQ prefill 已通过验证；可选的 width-one MTP 流水已经通过
> 32K/128K/256K 方向性性能与状态一致性测试。有界 width-N retained-draft 支持 1..64，
> 初始运维上限为 4；它的在线前缀 width-4 路径已经通过首轮 32K 真实 GPU token parity
> 及混合 accept/reject 性能测试。Width 2/3 与更完整的 width-4 矩阵仍待验证。完整
> tokenizer/logit golden parity、严格
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
| **当前版：+ GDN 状态驻留 + QSA 分块复用** | **512** | **2.74 s** | **1,497.04 tok/s** |
| 旧 DS4 参考 | 不适用 | 4,102 tokens 用时 5.27 s | 777.9 tok/s |

最后一行是同一机器上的历史数据，并非相同 artifact 的严格 A/B。当前 q38 路径相对最初
native prefill 提升 **20.0×**，超过了此前 DS4 的性能级别。输入为重复的合成 token，
因此它是真实 CUDA 与状态机性能测试，不是文本质量结论。

最新版普通模式（关闭 MTP）使用同一个 runtime binary 连续测量 prefill 与 decode。所有行
都关闭 profiling；Decode 是相应冷启动 prefill 完成后紧接着执行的五步真实 GPU 计算。

| 实际上下文 | Append 时间 | Prefill | Decode | ITL p50 |
|---:|---:|---:|---:|---:|
| 4,096 | 2.736 s | **1,497.04 tok/s** | **29.75 tok/s** | 31.60 ms |
| 32,768 | 19.313 s | **1,696.65 tok/s** | **29.65 tok/s** | 31.74 ms |
| 262,080 | 174.363 s | **1,503.07 tok/s** | **26.82 tok/s** | 34.24 ms |

相较上一版已发布 prefill checkpoint，三档分别提升 12.8%、17.0% 和 13.3%。另一次开启
逐 kernel CUDA event 的 32K 细粒度 profile 为 1,668.40--1,668.72 tok/s。本次 prefill
修改没有改变 decode kernels。

同一 build 还通过了五轮 coding 风格增量测试。首轮冷启动 prefill 4K，随后每一轮都复用
在线 recurrent/QSA 状态并追加一段新的 1,024-token suffix。多计算的一个 token 是会话
frontier 上一轮已生成的 continuation。

| 轮次 | 新增用户 tokens | 实际新计算 | Prefill / suffix | Decode |
|---:|---:|---:|---:|---:|
| 0，冷启动 | 4,096 | 4,096 | **1,497.04 tok/s** | 29.75 tok/s |
| 1，增量 | 1,024 | 1,025 | **1,182.22 tok/s** | 30.69 tok/s |
| 2，增量 | 1,024 | 1,025 | **1,167.95 tok/s** | 30.84 tok/s |
| 3，增量 | 1,024 | 1,025 | **1,166.96 tok/s** | 30.56 tok/s |
| 4，增量 | 1,024 | 1,025 | **1,159.85 tok/s** | 30.70 tok/s |

测试条件：单并发、`durability=off`、关闭 MTP，每轮测五步 decode。这是实时会话的
incremental hit，不是 radix cache 或 prompt cache 模拟。

### Decode

Prefill 和 decode 有意使用两套不同 kernel family。当前同一 build、同一轮测试的数据见
上面的联合表。下表仅保留为早期 decode 优化节点的历史运行模式对照，不代表当前版本
checkpoint。

| 历史运行 | Context + output | Stage 0 | Stage 1 + head | ITL p50 | Decode |
|---|---:|---:|---:|---:|---:|
| q38 高吞吐（`durability=off`） | 8,195 + 32 | 18.89 ms | 19.21 ms | 38.65 ms | **25.81 tok/s** |
| q38 严格持久化 | 8,195 + 32 | 19.08 ms | 19.21 ms | 43.87 ms | **22.84 tok/s** |
| q38 短上下文基线 | 8 + 64 | 13.25 ms | 13.10 ms | 31.32 ms | **31.76 tok/s** |
| Decode 优化前的 native runtime | 约 8K | 40.87 ms | 40.92 ms | 87.03 ms | 约 11.4 tok/s |
| 旧 DS4 参考 | 约 8K | 约 25.5 ms | 约 26.0 ms | 约 51.5 ms | 约 19.4 tok/s |

8K 高吞吐模式下，两段 GPU 计算合计 38.10 ms，而端到端 p50 是 38.65 ms。严格模式更慢，
是因为每个成功的变更 RPC 都要等待 `fdatasync`；所以任何性能数据都必须同时标明
durability mode。

#### 历史长上下文 exact-QSA 优化

精确并行 selector 与 tiled attention 消除了此前的长上下文 decode 崩塌，同时没有修改
512-block 选择预算或 tie 语义。下表记录的是该优化节点；当前 binary 的发布基线以上面的
Prefill/Decode 联合表为准。

| Context | 优化前 decode | Exact-QSA R2 | 加速 | ITL p50，优化前 → R2 |
|---:|---:|---:|---:|---:|
| 32,768 | 20.11 tok/s | **27.84 tok/s** | **1.38×** | 47.88 → 32.79 ms |
| 131,072 | 10.71 tok/s | **26.03 tok/s** | **2.43×** | 91.58 → 36.33 ms |
| 262,080（当前 ballot gather） | 6.21 tok/s | **27.79 tok/s** | **4.47×** | 159.19 → 33.91 ms |

条件：双 64 GiB CMP 170HX、单并发、`durability=off`、关闭 MTP、使用官方 tokenizer
处理源码语料、排除模型启动；每组先单独生成一个 seed token 计算 TTFT，再严格测量后续
五次真实 GPU decode。五个样本只用于方向性吞吐基线，不用于解释 p95/p99。128K 与
256K fixture 会在 69,579 个唯一语料 token 后重复，因此不代表最差 PLE locality，也不
构成模型质量 parity 结论。

#### 实验性 Piecewise CUDA Graph decode

`--enable-piecewise-decode-graph` 只加速普通的单 token decode。每个 stage 会把固定形状的
embedding/GDN/MoE/PLE-GPU/head 工作捕获为 7 段静态 graph，6 个依赖当前位置的
exact-QSA 层仍在各段之间 eager 执行。PLE 读取与 H2D staging、跨卡传输、sampling、
cancel，以及 transaction commit/rollback 都不会进入 graph。

CUDA graph node 会保留原始 state pointer，而 GDN/PLE transaction 会在两个 working bank
之间交替。因此运行时针对每个 bank pair 最多惰性建立一套 graph，不会把一套 graph
错误地重放到另一组 bank。捕获失败时，该 stage 会关闭 graph lane 并回退到原有 eager
decode。此功能默认关闭；在真实 GPU 上通过 token/logit parity、rollback、显存与 latency
A/B 门禁前，不宣称任何加速。可通过 `cuda_graph_captures`、`cuda_graph_replays`、
`cuda_graph_fallbacks` 与 `cuda_graph_nodes` 指标确认实际行为。

严格比较 eager/graph 时，用 `--enable-logit-diagnostics` 启动待测 executor。这个纯诊断
开关只保留最近一次已提交 append/decode 事务的原始 BF16 model-head 输出；默认关闭，
因此普通 greedy 服务不会增加整段 logits 的 D2H 拷贝。分别在 eager 与 graph 的相同已提交
步骤抓取，再逐 BF16 bit 比较：

```bash
python3 tools/q38_logit_parity.py capture \
  --socket /tmp/q38-eager.sock --session-hash 368 --output eager-step
python3 tools/q38_logit_parity.py capture \
  --socket /tmp/q38-graph.sock --session-hash 368 --output graph-step
python3 tools/q38_logit_parity.py compare eager-step.json graph-step.json \
  --output parity.json
```

抓取会生成一份精简 JSON 清单和原样 little-endian `.bf16` 文件，并同时校验运行时 FNV-1a
与工具侧 SHA-256。只有 session/epoch/frontier/事务类型一致、选中 token 一致且每个 BF16
元素完全相等才会通过。延迟基准测试不要打开这个诊断开关。

#### 已实测的 width-one MTP 基线

首版实测 MTP lane 会保留一个 draft QSA row，把两个 target row 拆成两个单 token
microbatch，并让下一行的 stage 0 与当前行的 stage 1 流水执行。此后运行时已将该路径
通用化为有界 width N；下表仍然只是 width-one 实测数据，不代表更宽模式的推算结果。

| Context | Plain decode 基线 | MTP width 1 | MTP 有效 ITL | 相对 plain 提升 |
|---:|---:|---:|---:|---:|
| 32,768 | 28.84 tok/s | **40.44 tok/s** | 24.73 ms | **40.2%** |
| 131,072 | 26.05 tok/s | **35.05 tok/s** | 28.53 ms | **34.5%** |
| 262,080 | 22.61 tok/s | **30.62 tok/s** | 32.66 ms | **35.4%** |

条件与上面的 exact-QSA 测试一致：单并发、`durability=off`、一个 seed token、五次被测
transaction。每档 fixture 的 draft 都恰好 5/5 接受；这个小型确定性样本只能证明
retained-row fast path 和吞吐有效，不能代表一般 acceptance rate 或模型质量。Plain 数据
来自前一版 retained-row revision：128K/256K 相对旧基线波动不超过 0.3%，32K 快 3.6%。
后续 overlap revision 只修改非空 retained-draft request 分支；runtime tests 验证普通 decode
不会进入该分支，因此有意跳过了重复且耗时很长的完整 256K plain 重跑。

原始证据、准确命令、已知限制和未完成门禁统一记录在 [READINESS.md](READINESS.md)。

#### 在线前缀 width-four 首轮结果

更宽的 MTP 路径现在会逐行在线校验，并在第一个 mismatch 立即停止。GPU0 最多只领先一行，
用一个可选 GDN/PLE checkpoint 保护前缀；GPU1 则在 GPU0 计算下一行时，用真实 target HC
在线修正每个已接受的深层 MTP row。Commit 直接发布已校验前缀，不再校验 rejected suffix，
也不再重放已接受的 target/MTP rows。

在冻结的 32,768-token coding fixture 上，五次 width-4 transaction（不含 seed）分别发布
`5, 4, 4, 2, 5` 个 token，覆盖两次全接收和三个提前停止深度。总计接受 15/20 个 draft，
达到 **48.65 tok/s**；backend commit 总耗时 **1.19 ms**，而旧 retained-draft trace 为
**124.17 ms**。发布的 21 个 token 与普通 target decode 的相同长度前缀逐 token 完全一致，
并且全程零 failure、零 rollback；同轮 plain reference 为 28.67 tok/s。

旧 trace 的 acceptance 数量不同，所以这是一组方向性的端到端结果，不是固定工作量 kernel
A/B。它验证了 32K 修复路径，但不能替代剩余的 128K/256K、logit golden、cancellation、
长生成以及 width-2/3 对比门禁。

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

GDN prefill recurrent kernel 为每个 CTA 分配一块 128x32 state-column tile。每个 thread
在整个 512-token slab 内把四个 FP32 state cell 驻留在寄存器中：只加载一次、让每个
normalized key 同时复用于两个 recurrent phase，最后只写回一次。末个 slab 的 GDN
recurrent 耗时在 stage 0 从 69.47 ms 降至 34.95 ms，在 stage 1 从 62.35 ms 降至
30.78 ms。

Grouped QSA 把 score key tile 调为 32 个位置，使 score CTA 的 shared memory 从约
38 KiB 降至 22 KiB。Value kernel 则一次暂存 64 个 selected position 及其全部 12 个
共享 query head 的 score，让 256 个 value-dimension thread 复用；末个 slab 的
`qsa_value` 从 22.17 ms 降至约 13.15 ms。两项修改都保留原有 FP32 运算顺序。全词表
parity 门禁逐一比较了 248,320 个 BF16 logits，结果零 mismatch，最终 token 也完全一致。

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

Width N 下，GPU1 会递归运行 checkpoint 中经过 multi-step 训练的同一个 MTP layer，同时
GPU0 开始 target row 0；N+1 个 target row 随后以单 token chunk 进入既有 scheduler，使
GPU0 的 row `n+1` 与 GPU1 的 row `n` 重叠。校验会在第一个 mismatch 停止；GPU0 在每次
lookahead 前保存当前 GDN/PLE prefix，提前停止时只恢复这一个 checkpoint，而 QSA 直接提交
较短的 append-only extent。第一个 retained MTP QSA row 使用 canonical target HC，可以
直接提交；后续 draft row 使用的是 MTP 预测 HC，因此每个已接受的深层 row 都会在 GPU0
计算 lookahead 时，用对应真实 target HC 在线修正，commit 不再串行修复前缀。Reject、
stop token、cancel、rollback 与 context 尾部裁剪仍保持请求原子性。MTP 默认关闭，plain
scheduler 完全不进入 retained-state 路径。

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
- 精确 prefix continuation 会复用 resident GPU state，只计算 suffix；新聊天或历史 fork
  会原子 cold rebuild 可变会话状态，同时保留权重与非语义 PLE/matrix cache。
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
启动；MTP 默认关闭，只有 plain lane 的正确性与显存门禁通过后才应使用
`--enable-mtp --mtp-max-draft 4`。Token-native 请求通过
`{"mode":"mtp","mtp_width":4}` 选择宽度，OpenAI-compatible 请求使用
`{"q38_mode":"mtp","q38_mtp_width":4}`；启动参数只是能力上限，每个请求仍显式选择。
Piecewise CUDA Graph decode 同样默认关闭，只能通过
`--enable-piecewise-decode-graph` 显式启用；完成 [READINESS.md](READINESS.md) 中的 graph
专项门禁前，不应把它用于性能结论。

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

当前运行时只有一个 GPU 常驻会话槽。`full_token_ids` 若是当前 canonical prefix 的精确
extension，就复用 GPU 状态并只计算 suffix；若它来自新聊天或历史分叉，executor 会原子
清除 QSA/GDN/PLE/MTP、sampler 与 token ledger 的可变会话状态，再对新历史执行 cold
prefill，不再返回 HTTP 409。权重、有限容量的 PLE host cache 与 CUDA prefill matrix
cache 都会保留。

响应通过 `q38.cache_status` 区分 `cold_start`、`incremental_hit` 与
`cold_rebuild`，同时给出 `cold_rebuild` 和 `reset_ns`。向
`/v1/chat/completions` 传入不同的 `session_id` 会显式驱逐当前 resident logical
session。`GET /v1/q38/metrics` 会返回该单槽策略及命中/重建计数。

这属于语义正确的单槽会话/cache 管理，并不宣称多个会话的 KV 同时驻留。返回已被驱逐的
聊天时，客户端仍需重放完整历史并 cold prefill；多槽 KV、LRU state swap 与 radix prefix
cache 是后续扩展。运维接口：

```text
POST /v1/q38/cancel
GET  /v1/q38/sessions/{id}
GET  /v1/q38/metrics
```

配置官方 tokenizer 后，sidecar 还会提供 `GET /v1/models`、
`GET /v1/models/{id}` 和 `POST /v1/chat/completions`。Codec 不进入 ExecutorRPC ABI，
也不拥有模型状态。Chat completions 支持 OpenAI function `tools`、`tool_choice`、历史
assistant `tool_calls` 和后续 `role: tool`；Qwen 官方 XML tool syntax 会转换成 OpenAI
`tool_calls`。Tool 模式的流式响应会先缓冲到完整调用通过结构校验，再输出
`delta.tool_calls`。

输出长度依次识别 `max_completion_tokens`、`max_tokens`、`max_new_tokens` 与
`max_output_tokens`。全部缺省时，输出预算等于 prompt 之后剩余的完整模型上下文；EOS、
stop token 或 tool completion 仍可提前结束。显式请求超过剩余上下文会返回 HTTP 422。
可通过 `--default-max-tokens` 与 `--max-output-tokens` 设置更小的服务端默认值或硬上限；
两者默认为 0，表示不施加低于模型上下文的额外限制。

## 验证与路线图

接下来的发布顺序为：

1. 发布 grouped-MMQ numerical identity，并从 cold state 启动；
2. 固定 tokenizer 生成的 golden prompts，与可信官方 BF16 reference 对比 logits/tokens；
3. 分别运行全新 32K、128K 和严格 262,080 + 64 门禁，并记录完整 GPU、host、transport
   与 PLE metrics；
4. 证明 near-256K suffix continuation 不会重放旧 prefix；
5. 在每档 context 验证 rollback、cancel、duplicate request、crash recovery 与 fault injection；
6. 完成 width 2/3 对比，以及 width-4 剩余的 accept-depth、cancel、token/logit parity、
   长生成、128K/256K 与吞吐门禁，再选择生产默认宽度；
7. 完成长时间稳定性与 failure-injection soak。

完整实现/证据边界见 [READINESS.md](READINESS.md)，详细架构与发布门禁见
[runtime architecture document](docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md)。

## License

运行时代码使用 [MIT License](LICENSE)。模型权重不属于本许可范围；下载、转换及使用模型
仍须遵守上游仓库许可。
