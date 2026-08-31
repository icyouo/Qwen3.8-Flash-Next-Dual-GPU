# q38-runtime

面向 **Qwen3.8-Flash-Next** 和双 64 GiB CMP 170HX（SM80）的专用长上下文推理
运行时。目标是在两张没有 GPU P2P、单卡仅 PCIe 2.0 x4 的 Ampere 卡上完整容纳模型，
并让一个会话可以持续追加到 256K context，而不重新计算旧 prefix。

本项目不是 vLLM、SGLang 或 llama.cpp 的 fork。模型数据面、离线量化、双卡调度、
KV/循环状态、事务、SSD-PLE、artifact 格式及 CUDA kernels 均由本仓库实现。HTTP、
tokenizer、chat template 和 SSE 是可替换的控制面；SGLang 不是运行时依赖。

> **项目状态：research preview。** 双卡真机已建立可重复的 8K decode 性能基线，
> 但 tokenizer golden parity、32K/128K/严格 256K、MTP 和长时间稳定性门禁尚未全部完成。
> 当前结果不能作为模型质量或完整 256K 发布声明。权重不随本仓库分发，使用者须遵守
> 上游模型许可。

```text
OpenAI / deep-session client
            │
            ▼
SGLang or q38_sidecar.py
tokenize / template / SSE / cancel
            │  ExecutorRPC V1 over Unix socket
            ▼
single native process
GPU0 layers 0..24 + SSD-PLE ── BF16 4H host ring ── GPU1 layers 25..47 + LM/MTP
```

严格产品门禁是一个 262,080-token prompt 加 64 个 committed generation tokens，最终
总长度正好 262,144。后续请求只能追加 exact suffix，旧 prefix 不会重新 prefill。

## 当前基线

以下数据在 Ubuntu `p3-ultra` 上实测，使用两张 64 GiB CMP 170HX、驱动
610.43.02、CUDA 13.1、`sm_80`、cut-25、自有 mixed W4/W8 artifact、batch 1、
plain decode（MTP 关闭）。两个 stage 对单 token 串行执行；表中的 stage 数值为各自
p50，并不表示两卡能够把同一个 token 并行算完。

| 运行 | 上下文与输出 | stage 0 | stage 1 + head | ITL p50 | 实测 decode |
|---|---:|---:|---:|---:|---:|
| q38 当前基线 | 8,195 + 32 | 19.08 ms | 19.21 ms | 43.87 ms | **22.84 tok/s** |
| q38 当前基线 | 8 + 64 | 13.25 ms | 13.10 ms | 31.32 ms | **31.76 tok/s** |
| q38 优化前 | 约 8K | 40.87 ms | 40.92 ms | 87.03 ms | 约 11.4 tok/s |
| 旧 DS4 参考 | 约 8K | 25.5 ms（0–23） | 26.0 ms（24–47 + head） | 约 51.5 ms | 约 19.4 tok/s |

当前 8K stage 合计为 38.29 ms，对应纯 GPU stage 上限约 26.1 tok/s；端到端 p50
为 43.87 ms。其差值包含 RPC、事务提交以及启用 snapshot journal 时每个成功写请求的
`fdatasync`。DS4 行仅是同一机器上的历史工程参考，并非相同 artifact、量化和服务路径的
严格等价 A/B。

这些基准使用合成 token 验证真实 CUDA 状态机、transport 和性能，不代表文本质量。
原始构建、fixture、profile、fallback A/B 和端到端 JSON 见
[`READINESS.md`](READINESS.md) 记录的 evidence 目录。

## 模型与量化格式

artifact 必须由固定 commit 的官方 BF16 checkpoint 生成：

```text
model          Qwen/Qwen3.8-Flash-Next
source commit  de4b8e4d43b917e7706784d8bb445c9af86a3540
policy         Q38_AMPERE_QUANT_POLICY_V5
stage split    GPU0: layers 0..24; GPU1: layers 25..47 + LM head/MTP
context limit  262,144 tokens
```

这不是 NVFP4、AWQ、GPTQ 或 GGUF。compiler 从官方 BF16/F32/I64 tensor 直接生成
Q38 的 stage-owned artifact；量化权重在运行时不会先整体反量化，也不会 repack 成另一种
框架格式。

| 张量类别 | artifact 格式 | 运行精度/说明 |
|---|---|---|
| 第 2–45 层 routed expert gate/up/down | symmetric **W4A16 group-128** | batch-1 专用 W4 GEMV/MoE kernel，BF16 activation |
| 第 0/1/46/47 层 routed experts | symmetric **W8A16 group-128** | 质量更敏感的边缘 experts |
| always-active matrices，包括 embedding、LM head、attention/GDN projections、shared expert | symmetric **W8A16 group-128** | BF16 activation；每组 scale 以 BF16 保存 |
| MTP matrices 与 routed experts | symmetric **W8A16 group-128** | MTP 默认不加载，需显式启用 |
| SSD-PLE embedding table | row-scaled **FP8 E4M3FN** | 每行一个 BF16 scale；表位于 SSD，不整表常驻 GPU/DRAM |
| router、norm、gate、hyper-connection、卷积与其他关键控制张量 | **preserve** | 保留上游 BF16/F32/I64 dtype |
| main QSA K/V 与 compressed index | **BF16** | 256K 容量基线；尚未启用 INT8 KV |
| GDN recurrent/累加状态 | **FP32** | 避免长序列状态误差累积 |
| vision tensors | **skip** | 当前是 text-only runtime |

具体匹配规则以 [`tools/q38_quant_policy.py`](tools/q38_quant_policy.py) 为唯一事实来源；
policy digest 会进入 artifact/session identity，策略不匹配时启动器 fail closed。

## 已实现的关键合同

- 单 semantic writer，append/decode/MTP 统一 prepare → execute → acknowledge → commit；
- 跨 chunk 的请求级原子 append、双 stage rollback、取消/超时与 request-id 幂等；
- prefill 的 stage0/stage1 流水，小 decode ring 与三槽大 prefill ring；
- BF16 QSA KV、GDN/PLE/MTP provisional state，以及 stop-token-aware MTP commit；
- 无 P2P 的 pinned D2H/handoff/H2D boundary，GPU0 payload checksum 在 GPU1
  消费前对 pinned-host activation 复验；
- official BF16 → Ampere W4A16/W8A16/FP8-PLE 的流式、可恢复 artifact compiler；
- 4 KiB registered-buffer `io_uring READ_FIXED + O_DIRECT` PLE 路径和硬容量 cache；
- 事务化 RNG/penalties、append-only cold-rebuild snapshot journal；
- `SessionIdentityV1`、`MetricsSchemaV1`、直接 token 严格门禁和 HTTP/SSE adapter。

生产 CUDA backend 的执行异常会让 executor fail closed；cancel/deadline 仍走完整事务
rollback。这样 OOM、illegal access、Xid 或 transport corruption 后不会在残留 device
state 上继续服务。

架构与性能门槛见
[`docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md`](docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md)。
代码证据、未声明项和首轮真机顺序见 [`READINESS.md`](READINESS.md)。

## Ubuntu 构建与验证

生产 CMP 170HX 在当前驱动下报告为 GA100（compute capability 8.0），
默认目标因此固定为 `sm_80`；其他开发卡可用 `CUDA_ARCH=...` 覆盖：

```sh
make clean
make -j2 build/q38_runtime_tests build/q38-runtime
./build/q38_runtime_tests
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
make -j1 cuda-check cuda-runtime
```

GPU 接通后再运行 kernel fixture：

```sh
make -j1 cuda-test
make -j1 cuda-bench
```

batch-1 decode 直接消费生产 artifact 的 BF16 activation 与 group-128 W4/W8
布局。通用矩阵路径使用 warp-per-row GEMV；routed MoE 另有 top-10 gate/up、SiLU
和 fused down/reduce 路径；512-expert router 使用稳定 tie-break 的并行 bitonic top-k。
QSA softmax 概率只在 FP32 score scratch 中计算一次，再由 256 个 value 维度复用。
这些路径都不会把单行 decode 填充成 16 行 WMMA。真机 A/B 或紧急回滚可分别设置
`Q38_CUDA_DECODE_GEMV=scalar`、`Q38_CUDA_DECODE_MOE=scalar` 和
`Q38_CUDA_DECODE_TOPK=scalar`；默认启用优化路径。`cuda-bench` 同时输出生产形状的
GEMV 新旧对比和 routed-MoE 新旧对比。设置 `Q38_CUDA_PROFILE_DECODE=1` 可输出按
hyper/GDN/QSA/MoE 子路径聚合的 CUDA-event decode profile；默认完全不创建 event。

`make verify PYTHON=.venv/bin/python` 可执行 CPU、Python 和 CUDA 编译门禁。所有测试与
运行时都应在 Ubuntu 主机执行；macOS 工作区只用于编辑/同步源码。

## 1. 下载固定的 official BF16 source

```sh
python3 tools/q38_hf_fetch.py \
  --repo Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --expected-commit de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --output /data/models/Qwen3.8-Flash-Next-official-source \
  --jobs 4 \
  --manifest-only
```

生产 commit 固定为 `de4b8e4d43b917e7706784d8bb445c9af86a3540`。下载器使用
解析后的 commit URL（不会继续追随 `main`）、`.part` 文件、LFS size/SHA-256 和原子
rename。这里先只发布固定 manifest；下一步按 shard 流式拉取和转换，避免源权重与最终
artifact 同时占满本机 468 GB 根盘。

## 2. 生成生产 artifact

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

每个 shard 都按“固定 commit 下载与 LFS 校验 → 有界量化 → fragment/segment SHA-256
复验 → 删除该 source shard”的顺序执行；中断后即使源 shard 已删除，也能根据 official
tensor index、fetch manifest 和已验证 fragment 精确续跑。`--prune-source-shards` 是显式的
磁盘回收开关；磁盘足够且希望保留 BF16 源时可省略。最终只有在两个 stage 的精确 tensor
census、PLE layout、identity 全部通过后才原子发布 `READY.json`。

默认精度策略：

- routed experts：symmetric group-128 W4A16；
- always-active 适合项：group-128 W8A16；
- PLE table：row-scaled FP8 E4M3FN；
- router/norm/critical state、QSA KV 等按合同保留 BF16/FP32。

## 3. 一键 fail-closed 启动

```sh
.venv/bin/python tools/q38_launch.py \
  --ready /data/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --runtime build/q38-cuda-runtime \
  --socket /tmp/q38-executor.sock \
  --snapshot /var/lib/q38/session.q38j \
  --host 127.0.0.1 \
  --port 30000
```

启动器在执行 CUDA 前验证 pinned repo/commit/cut、顶层 digest、全部 segment SHA-256、
identity checksum 及其 8 类输入 digest、context/sampling/parser/stop-token 合同。生产 PLE
强制 `io_uring + O_DIRECT`；不能建立 direct lane 时启动失败，不静默退回 page cache。

`GET /v1/q38/metrics` 使用固定的 `q38.metrics.v1` ABI，同时报告每 stage 的
weight arena/上传/host-only、QSA/GDN/PLE/MTP state、workspace、prefill cache、
boundary/workspace pinned bytes、CUDA tracked current/peak/free/total，以及 executor
进程 RSS/peak/anon/file/swap、fault、context switch、MemAvailable、swap 与 cgroup
current/peak/max/OOM。PLE 另外报告 cache current/cap、requested rows、每批去重后的
page requests、useful/physical bytes（可直接计算 read amplification）、read count/batch/QD、
io_uring/direct-I/O 状态、P50/P95/P99 与 FP8 row-scale 常驻字节。原始 `cudaMalloc` 路径没有 caching allocator，因此
`cuda_allocator_retries` 和 `cuda_graph_held_bytes` 在当前实现中应保持 0；prefill
优化因显存不足回退时由 `cuda_allocation_failures` 单独计数。

启动默认是设计中的 plain/M1 lane：MTP 权重不会上传、draft QSA cache 与 target-HC
workspace 也不会分配，省下的 artifact 字节通过 `weight_excluded_bytes` 报告。plain
正确性与显存门禁通过后，给启动器增加 `--enable-mtp` 才会加载官方单层 MTP；该
checkpoint 首轮只开放 1 个 draft token（即 width-2 target verify），更宽请求会 fail
closed，不会假装执行 width-3。

先检查而不启动：

```sh
.venv/bin/python tools/q38_launch.py --ready /path/READY.json --dry-run
```

## 4. Deep-session / SGLang adapter

SGLang 应调用 token-native endpoint；这样其 tokenizer、chat template 和 parser 可以升级，
executor ABI 不变。创建唯一 live session：

```sh
curl -sS -X POST http://127.0.0.1:30000/v1/q38/sessions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"deep-1"}'
```

追加已 tokenized 的 delta 并生成：

```sh
curl -sS -N -X POST \
  http://127.0.0.1:30000/v1/q38/sessions/deep-1/execute \
  -H 'Content-Type: application/json' \
  -d '{"append_token_ids":[1,2,3],"max_new_tokens":64,"stream":true}'
```

使用 `full_token_ids` 时，sidecar 会验证它是服务器 canonical tokens 的精确扩展；分叉返回
`cold_rebuild_required`，不会覆盖 live state。响应只流出 executor 已 commit 的 token event。
取消、状态和指标入口分别是：

```text
POST /v1/q38/cancel
GET  /v1/q38/sessions/{id}
GET  /v1/q38/metrics
```

若安装了 `transformers` 并由启动器加载 official tokenizer，也提供
`POST /v1/chat/completions`。该兼容路径会单独报告 tokenize 和 exact-prefix 验证耗时。

## 5. 严格 262K 门禁

executor 已启动后：

```sh
.venv/bin/python tools/q38_strict_gate.py \
  --socket /tmp/q38-executor.sock \
  --session-hash 0x380025 \
  --output out/strict-262080-plus-64.json
```

门禁不仅检查长度，还要求：

- `canonical=262144`，`target=stage0=stage1=262143`；
- exactly one append、63 plain decode transactions、64 published tokens；
- evaluated/state-committed token census 精确一致；
- failure、rollback、cancel、deadline 均为零；
- 输出 prompt/output digest、TTFT/ITL P50/P95/P99、完整版本化 metrics 和主机/GPU 元数据。

合成 tokens 只验证 transport/state。模型 correctness 门禁必须传入 tokenizer 生成的
`--tokens-json`，并结合 marker/sentinel reference fixture；不能用合成结果声称模型质量通过。

## 发布边界

CPU/mock 的严格门禁证明协议、事务和 262K 容量路径；它不代表真实模型速度或数值正确性。
真实发布仍必须在两张 170HX 与最终 artifact 上依次通过 8K、32K、128K、262K+64、
near-256K suffix-only、reference parity、显存/主存门槛和 12 小时 fault-injection soak。

## License

运行时代码以 [MIT License](LICENSE) 发布。模型权重不属于本许可证；下载、转换和使用
Qwen3.8-Flash-Next 时仍须遵守上游模型仓库的许可条款。
