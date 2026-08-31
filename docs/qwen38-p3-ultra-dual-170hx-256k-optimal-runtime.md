# Qwen3.8-Flash-Next 双 CMP 170HX / 262K 最优运行时架构

**状态：** ARCHITECTURE ACCEPTED / IMPLEMENTATION GO / RELEASE CONDITIONAL  
**日期：** 2026-08-30  
**硬件：** 2 × CMP 170HX（SM80，64 GiB/卡），约 61 GiB 可用主存，无 P2P 假设  
**模型：** Qwen3.8-Flash-Next，单个可持续追加的 262,144-token text session  
**严格门禁：** 262,080 prompt + 至少 64 decode；后续 turn 必须 suffix-only continuation  
**替代文档：** 本文取代 `qwen38-p3-ultra-dual-170hx-256k-design-reviewed.md` 作为最终实现路线。

## 1. 最终结论

最优产品架构是：

```text
OpenAI-compatible client
          │
          ▼
SGLang control plane（可替换）
API / tokenizer / template / parser / streaming / cancel
          │  ExecutorRPC v1：请求、token、txn、commit event
          ▼
自研 Qwen3.8 native executor（真正的运行时）
单进程、双 GPU worker、统一状态事务与内存规划
      │                                  │
      ├─ GPU0：连续前半层 + SSD-PLE      ├─ GPU1：连续后半层 + LM/MTP
      └──────── 一次 host-staged 4H boundary ────────┘
```

这里的核心不是 SGLang fork。**模型数据面、调度、状态、双卡传输、PLE、artifact 和 hot kernels 都由我们自研。** SGLang 只提供成熟且低性能敏感的服务入口；如果它不合适，可换成 vLLM frontend 或 Rust/C++ gateway，而无需改 executor。

从数据面看，这是独立 model executor；从完整产品看，这是成熟控制面与专用 executor 的混合架构。

## 2. 独立复核结论

第二轮 Pro 推荐的主路线成立，但采用以下本地限定：

1. **SGLang 是可替换 sidecar/control plane，不是 runtime base。** 首版不把私有 scheduler hook 写进核心 ABI；优先使用 UDS 或共享内存 ring 的稳定 adapter。
2. **连续双阶段保留，generic PP2 撤销。** 每次 target pass 只跨一次边界，但不继承 rank、NCCL、通用 PP scheduler 或两进程状态模型。
3. **单 native process 是目标 executor 合同。** 每卡独立 worker/thread 和 CUDA context；任一卡或 epoch 出错，整个 executor fail closed。
4. **24/24 只是首个 profile seed。** 23/25、24/24、25/23、26/22 必须由最终 artifact 字节账本、stage profile 和显存余量共同裁决。
5. **4 KiB PLE page、8 GiB host cache 和性能数字都是 baseline/目标，不是写死的协议。** 真实 trace 要比较 512 B、1 KiB、4 KiB 读单元和 4/8/12 GiB cache。
6. **性能目标不是预测完成值。** 在真实双 170HX trace 前，只能作为工程门禁。

## 3. 为什么不是深 fork

### 3.1 实测瓶颈指向 executor 与 kernels

当前 ds4 在约 8K 上为 19.4 tok/s，即约 51.5 ms/token：两阶段各约 26 ms，GPU SM 平均仅约 44%–46%，memory busy 约 5%，PCIe 低于 80 MB/s。现有单 token 只传很小 activation，链路和 stage 失衡都不是主要瓶颈。

真正限制是 48 层中大量 batch-1、top-10-of-512 MoE、shared expert、HC、GDN/QSA 与 mixed-quant 小 kernel，launch 多、occupancy 低。这个问题必须在 artifact、scheduler、graph、融合和 exact-shape kernels 上解决；换一个通用 serving scheduler 不会自动消失。

当前 distributed MTP 已把 short decode 从 24.9 提到 33.6 tok/s，约 10K 从 19.1 提到 30.9 tok/s，并达到约 1.78–1.92 committed tokens/step。这证明批量 target verify、事务化 rollback 和自适应 MTP 是主优化路径之一。

### 3.2 深 fork 最终仍等于重做数据面

若基于 vLLM 或 SGLang 深 fork，仍需重写：

- QSA、GDN、PLE、MTP、RNG 的统一 commit/rollback；
- 无 P2P 的单机双卡传输；
- bounded SSD-PLE；
- 离线 mixed artifact 与精确 load peak；
- suffix-only 262K state；
- 单会话 batch-1/MTP hot path。

把这些改动散落在 generic scheduler、block manager、model runner 和 PP worker 内，只会增加状态审计与升级冲突。既然核心数据面必然自研，就应把边界切干净。

## 4. 固定的系统合同

实现前冻结以下版本化接口：

- `ModelArtifactManifestV1`
- `StagePlanV1`
- `ExecutorRPCV1`
- `SessionIdentityV1`
- `SessionTxnV1`
- `StageBoundaryFrameV1`
- `PLEStoreV1`
- `StateSnapshotV1`
- `KernelTensorLayoutV1`
- `MetricsSchemaV1`

稳定合同描述语义和所有权，不写死 24/24、chunk=4096、BF16-only compressed QSA、固定 cache size 或 MTP K。

## 5. 组件边界

| 类别 | 内容 |
|---|---|
| 直接复用 | SGLang API/tokenizer/template/parser/detokenizer；CUDA/cuBLASLt；CUTLASS；通过 ABI 包装的 Marlin、FlashAttention/FlashInfer 适用算子；safetensors；Linux `io_uring` |
| 必须自研 | artifact compiler、stage planner、单进程双卡 executor、四类 execution lane、无 P2P transport、统一状态事务、SSD-PLE provider、durable snapshot、端到端 trace |
| 按结果选择 | MoE/GDN/QSA/PLE/HC/LM-head kernels：可靠上游实现与自研实现使用同一 fixture 和 microbenchmark 竞赛 |
| 仅作 oracle/reference | vLLM/SGLang Qwen model、state、MTP、PLE 和当前 ds4；不继承其生产 scheduler/cache/process ABI |

原则：自研 kernel 若没有至少 5%–10% 的端到端或集成收益，就 vendor 更可靠的实现；目标是最优 runtime，不是追求“自研率”。

## 6. 执行与状态设计

### 6.1 双阶段执行

- 物理拓扑：两段连续 decoder layers，每个 target pass 一次跨卡。
- 逻辑 BF16 boundary：`4 × 2560 × 2 = 20,480 B/token`，即 20 KiB/token。
- 当前 ds4 观测约 40 KiB/token 是 FP32 boundary；最终 BF16 传输必须做数值 parity 验证。
- plain decode 的两 stage 对单序列仍然串行；加速来自更快的 stage kernels、消除 Python/process 开销和 MTP/verify batching，不能把双卡算力简单相加。
- prefill chunk 和 MTP verify 可做跨 stage pipeline；small decode ring 与 large prefill ring 分离。

### 6.2 单一语义 writer

每个 session 维护：

- `T`：对客户端可见的 canonical token frontier；
- `C`：target committed frontier；
- `C0/C1`：两 stage frontier，commit 时必须等于 `C`；
- `D`：MTP draft committed frontier；
- `txn_epoch`：单调递增事务号。

每次 append/decode 都经过 prepare → execute → decide → dual-stage acknowledge → commit → publish。只有 commit 后的 token 才能进入 detokenizer。失败时释放 provisional QSA pages、恢复 GDN/PLE/MTP/RNG slot；任一 stage 不允许单独继续。

### 6.3 262K state

- Main QSA K/V 保留 BF16：12 层总计约 6.0 GiB；24/24 seed 时每卡约 3.0 GiB。
- QSA compressed index BF16 baseline，总计约 192 MiB；只有真实 256K profile 和质量门禁通过后试 INT8。
- GDN committed state 全模型约 110 MiB，MTP 使用 bounded candidate ring。
- PLE recurrent state bounded；47.6839 GiB FP8 PLE table 是 SSD 一级存储，不进入匿名主存或 GPU 常驻。
- 旧 prefix 不重新执行；suffix 只追加 QSA pages、更新 recurrent state，并可读取历史 index/K/V。

## 7. Artifact、精度与内存基线

离线 compiler 生成 stage-owned、可直接执行的 artifact，避免启动时反量化/repack 和 source+destination 双份峰值。

- Routed experts：SM80 上 Marlin W4A16；先正确转换现有 group-16 NVFP4，再从可信 BF16 source 评估 group-64/128。
- Always-active：按收益逐个引入 INT8，顺序优先 LM head、GDN projections、attention projections、shared expert、MTP。
- 保持 BF16/FP32：router/selection、norm/gate/critical scales、GDN recurrent FP32、main QSA K/V BF16、PLE FP8 source。
- 最终 aggregate weights 目标约 66–69 GiB，需由真实 tensor ledger 替换估算。
- Release 目标：steady ≤52 GiB/卡、controlled peak ≤54 GiB/卡、任意 peak reserve ≥8 GiB。
- Host steady `MemAvailable ≥16 GiB`、peak ≥10 GiB、swap-in=0；PLE cache/pinned/file cache 全部 hard-bounded。

## 8. SSD-PLE

Production baseline 是 4 KiB arithmetic page（25 × 160 B rows + 96 B padding），但 artifact/provider ABI 必须允许更换读单元。

- 8 GiB ordinary-DRAM cache 是初始目标，不整块 pin；pinned I/O pool 仅 256–512 MiB。
- prefill 使用 hash/read/gather、H2D 和 GPU0 stage pipeline；4096 是 chunk seed，内部仍用更小 token slabs。
- decode 在 GPU1 sample 后立即发出 token event，PLE I/O 与 GPU0 在 PLE consumer 之前的计算重叠。
- 必须测真实随机 I/O tail。2.7 GB/s sequential 不能证明 16-way random 4 KiB P99。

## 9. 性能目标

这些是完成 custom executor 后的工程门禁，不是承诺：

| 场景 | Release floor | 工程目标 |
|---|---:|---:|
| 262,080 cold prefill | 220 tok/s | 350 tok/s |
| short plain decode | 26 tok/s | 36 tok/s |
| 256K plain decode | 16 tok/s | 25 tok/s |
| short MTP | 34 tok/s | 52 tok/s |
| 256K MTP | 22 tok/s | 35 tok/s |
| near-256K + 1K suffix TTFT | ≤10 s | 4–6 s |

同时必须报告 TTFT、ITL P50/P95/P99、stage time、kernel breakdown、transport non-overlap、PLE hit/miss/IOPS/tail、MTP accepted histogram、显存/RSS/pinned/MemAvailable、power/clock/throttle。只报 tok/s 不算通过。

## 10. 实施顺序

### Phase A：四个高信息量实验

1. **Artifact ledger**：真实 checkpoint headers、最终 dtype/layout/owner/load peak，扫描 23/25、24/24、25/23、26/22。
2. **Transport matrix**：20 KiB 及 K/2048/4096/8192 payload，pinned D2H→handoff→H2D 的 P50/P95/P99 与热稳态。
3. **PLE trace replay**：真实 prompt page IDs、512 B/1 KiB/4 KiB、不同 QD 和 4/8/12 GiB cache。
4. **Kernel shootout + transaction simulator**：冻结 fixture 对比 reference/vendor/custom，并先证明 rollback/frontier/crash consistency。

### Phase B：可抛弃 vertical slice

`Q38-8K-DUAL-STAGE-LAB`：无 HTTP、直接 token IDs、一个 native process、两个 GPU workers、BF16 QSA、MTP/INT8/durability 关闭、512 MiB–1 GiB PLE cache、2048/4096 chunks。先证明 exact forward、state ownership、host boundary、100 次 append/rollback 和 forced failure。

### Phase C–F：产品化

1. 冻结 executor/artifact/state ABI，升窗 32K → 128K → strict 262K；
2. 引入最终 mixed artifact 与胜出的 SM80 kernels；
3. 加入 adaptive MTP 和 append-only durable snapshots；
4. 最后接入 SGLang control plane，要求吞吐回归 ≤2%，且它只消费 committed token events。

## 11. Release gates

- strict `262,080 + 64` 成功，后续 turn 不重算旧 prefix；
- tokenizer/template/special-token、PLE hash/row、8K/32K/128K/near-262K marker retrieval 通过；
- QSA group boundary、GDN/PLE state、MTP accept 0..K、cancel、crash recovery 通过；
- 1000-turn growing-prefix 与 12-hour soak 无 frontier divergence、Xid、swap 或 unbounded growth；
- 满足第 7、9 节 memory/performance floor；
- control plane 与 executor 的版本、artifact hash、stage plan、state/PLE layout 都进入 session identity。

## 12. 证据与责任边界

本结论由本地实测基线、第一轮可行性审查、第二轮 greenfield 路线审查和本地独立复核共同形成。

- Pro conversation: <https://chatgpt.com/c/6a941734-1ad4-83e9-b4f3-050a2e214abd>
- 第二轮 ZIP SHA-256: `c944cd6ea2709f015bcccac0d96fcfc02afab64331c656554209ec6950b11589`
- 完整 Pro 方案：`.artifacts/gpt-pro/qwen38-170hx-256k-optimal-route/extracted/OPTIMAL_ROUTE.md`
- 路线比较：`.artifacts/gpt-pro/qwen38-170hx-256k-optimal-route/extracted/ROUTE_COMPARISON.md`
- 本地验证：`.artifacts/gpt-pro/qwen38-170hx-256k-optimal-route/VALIDATION.md`

最终责任结论：**我们要自研的是整个 Qwen3.8 数据面；SGLang 只是前门。** 先以直接 token 的 vertical slice 证明 executor，再接 serving。这样既不被当前 upstream 可实施性限制，也不会为了重造 API 层牺牲模型运行时的性能和正确性。
