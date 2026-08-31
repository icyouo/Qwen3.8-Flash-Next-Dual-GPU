# q38-runtime

Qwen3.8-Flash-Next 专用运行时，目标机器是两张无 P2P 的 64 GiB CMP 170HX。它不是
vLLM/SGLang fork：模型数据面、双卡调度、状态事务、PLE、artifact 和 CUDA kernels
都在本仓库；SGLang 或参考 sidecar 只作为可替换的 API/tokenizer 前门。

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
[`../docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md`](../docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md)。
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
```

`make verify PYTHON=.venv/bin/python` 可执行 CPU、Python 和 CUDA 编译门禁。所有测试与
运行时都应在 Ubuntu 主机执行；macOS 工作区只用于编辑/同步源码。

## 1. 下载固定的 official BF16 source

```sh
python3 tools/q38_hf_fetch.py \
  --repo Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --expected-commit de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --output /home/icy/models/Qwen3.8-Flash-Next-official-source \
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
  --source /home/icy/models/Qwen3.8-Flash-Next-official-source \
  --metadata /home/icy/models/Qwen3.8-Flash-Next-official-metadata \
  --output /home/icy/models/Qwen3.8-Flash-Next-q38-cut25 \
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
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --runtime build/q38-cuda-runtime \
  --socket /tmp/q38-executor.sock \
  --snapshot /home/icy/q38-state/session.q38j \
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
