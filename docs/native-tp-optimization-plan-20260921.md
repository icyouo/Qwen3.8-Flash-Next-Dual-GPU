# 原生双 170HX 优化方案

> 最新执行范围以[固定 MTP4 集中改造计划](native-mtp4-consolidated-plan-20260921.md)
> 为准：固定 MTP4、集中实现一份候选、统一验收。下文宽度扫描、自适应调优和
> 大规模逐项消融属于历史方案，不再执行；历史证据保留。

目标：保持现有 W4/W8 权重及模型语义，用原生 C++/CUDA 补齐高吞吐执行能力，
再针对 SM80、双卡 PCIe 和本模型的形状细调。参考引擎仅用于机制分析与性能对照。
方案不要求 NVFP4、GDS、BIOS/驱动变更，也不复制参考推理实现。

## 已知基线与归因边界

同机、同量化产物、P2P、RAM PLE、单请求 greedy：原生 TP 非思考 30.757 tok/s，
参考适配版 173.094 tok/s；思考分别 29.003 和 76.247 tok/s。原始请求、SSE、计时
定义和限制见 [实测记录](reference-own-w4w8-20260920.md)。这些是吞吐结果，不是
整模型输出等价结论。参考长生成三次不同，尚未逐项关闭 MTP/Graph 做贡献分解。

原生 routed experts **已经按专家分区**：本地 remap 后计算 FP32 partial，再规约。
当前缺口不是从零实现 EP。TP 每 token 97 次 AllReduce；paired stage 每次操作创建/
回收 peer 线程；P2P 每次 collective 还等待两条 GPU stream。TP grouped MoE 被禁用。
现有 PP MTP/Graph 可复用组件，但 PP retained MTP 本身逐行验证，TP 不应仅放开开关。

## 完成标准

1. 当前功能：SSD/RAM 两种 PLE、plain greedy/采样、取消/重置、上下文上限均保留。
2. 无额外有损量化、无 top-N 近似候选过滤、无忽略拒绝后缀的投机提交。
3. 每一步有可回退开关；同二进制 A/B 优先；记录源文件/二进制/权重身份及实际开关。
4. 纯调度改动要求相同输入 raw BF16 logits 和 token 序列逐位相等。计算顺序变化要
   单列误差、top-1 分歧和状态等价结果，不能悄悄降低原有正确性门槛。
   影响算术的内核选择必须进入运行身份/manifest，旧的 session/cache 不跨身份复用。
5. 先让原生与参考具有相当执行能力，再以同工作负载追平/超过参考。173 tok/s 是
   已测的对照目标，不是承诺每种提示、长上下文或思考均达到的最低速度。

## 性能预算与里程碑

按相同计时口径，173.09 tok/s 对应平均每个发布 token 约 5.78 ms，当前原生约
32.51 ms。投机解码应使用 `吞吐 ≈ 每轮平均发布数 / 每轮总耗时` 评估；若平均
每轮发布 5 个 token，追平该对照需要 draft+verify+commit 总耗时约 28.9 ms。
接受率低的思考任务必须独立评估，不能套用 5 个 token 的假设。

| 里程碑 | 交付物 | 通过条件 |
|---|---|---|
| M0 测量可信 | 固定输入、版本与分项基线，参考消融结果 | 相同开关可复现，profile 与正常计时分开 |
| M1 调度优化 | 常驻 worker、异步 P2P、独立开关 | raw logits/token 逐位一致，重复生成有净收益 |
| M2 多行计算 | TP grouped MoE、小 M 算子、多行 target API | 因果状态与逐行对照通过，分别报告 prefill/verify 收益 |
| M3 图执行 | plain/verify 指定形状的分段图 | changing-input/bank/reset 正确，实际 replay 计数非零 |
| M4 TP MTP | 宽度 1/2/4/6、精确前缀提交和回滚 | 所有接受长度与拒绝位置通过，生成吞吐提升 |
| M5 性能收敛 | 形状选择、融合、通信/查表重叠 | 同工作负载追平参考作为目标，并单列质量/长上下文结果 |

M1 与 M3 的 plain decode 部分可先推进；M4 的高吞吐验收依赖 M2/M3。每一步先
保留单项结果，再测组合，避免某个回退路径把另一个改动的收益掩盖。没有实测前不
给每个内核分配虚构的提速百分比或保证完成日期。

## P0：可重复测量与性能拆分

- 固定版本、prompt/token IDs、chat template、采样、输出数、上下文、预热次数。
  保留非思考 1024 和思考 2048 的原对照，增加数学、代码、普通中文三类提示。
  同时保留原生 PP plain/MTP 对照，不能只和功能尚未迁齐的 TP eager 比较。
- 测试 16/512/8K/32K 输入及后续长上下文；先单请求，稳定后再单独评估并发。
  1024/2048 是性能预算，质量任务必须另外运行到完成，区分 reasoning/content。
- 输出 TTFT、prefill tok/s、decode tok/s、总耗时、ITL p50/p95、固定窗口吞吐、
  峰值显存/RSS、PLE 实际磁盘读取、draft/verify/commit 时间、接受长度直方图。
- 原生复用已有 CUDA event profile 与 TE 计数，区分 CPU rendezvous、GPU compute、
  collective、PLE/H2D、状态提交；profile 有额外开销，性能数据用关闭 profile 的运行。
- 参考适配版测 MTP off/on × Graph off/on，保持现有 W4/W8/RAM；这些运行占用同一
  对 GPU，须顺序切换并恢复服务。下载继续，但记录背景 I/O 与内存，不伪称完全受控。

## P1：先消除重复调度和 CPU 等待

**常驻 peer worker。** `cuda_tp_backend.cc` 每 logical stage 保留一个 peer worker，
沿用一个 coordinator、两 rank 结果检查、异常唤醒和 fail-closed。对照旧线程路径，
覆盖重复调用、单侧/双侧异常、等待中的 peer、析构和 reset；避免为每个 token 的
execute/commit 等动作创建 OS 线程。

**stream-ordered P2P。** 保留同步模式；新增显式模式，使 producer ready → 两端
读取到独立 scratch → 两端 read-done → 写回全部在 GPU stream 上建立依赖，CPU
完成成对入队后继续提交。增加上一代输出完成事件，保护跨 stream scratch 复用；
取消、reset、析构和向 CPU 发布仍有完成边界。验证延迟 producer、快慢 rank、
in-place gather、不同 group、跨 stream、scratch 扩容和成千上万次连续 generation。
CUDA event 可复用的依据是 wait 绑定调用时最近一次 record，后续 record 不改变
已提交的等待；实现必须保持“先提交旧代 wait，再记录新代 event”的顺序。
参见 [NVIDIA CUDA Runtime event 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)。

**减少中间主机搬运。** 审计 TP logical stage0→stage1 的 HC boundary。后续采用
各 rank 的设备 lease 传递，避免回主机再传回同一 GPU；lease 必须覆盖 producer、
consumer 与 ring generation，不能只传裸指针。此项以 boundary profile 为收益依据。

## P2：真正的多 token 执行

**TP grouped MoE。** 在 route planner 中显式处理不属于本 rank 的 -1 routes，
本地 grouped gate/up/down 后，按原 token/top-k slot 顺序归并为 `[tokens, hidden]`
FP32 partial，再跨卡规约、沿用现有 BF16 舍入边界。禁止直接规约各卡不同排列的
packed route buffer。覆盖空专家、某 rank 无路由、偏斜负载、重复专家、tile 尾部。

**小 batch 专用算子。** 为 M=1/2/3/5/7/8/16/32 测量 W4/W8 GEMV/GEMM 路径，
按实际 K/N（含 shared expert K=640）选实现。当前 dense prefill GEMM 的 M≥8
阈值会让 MTP6 的 7 行验证错失路径；必须实现正确 padding/掩码或小 M 内核，而非
简单改阈值后假设性能、布局均成立。比较直接量化计算与有限缓存的解量化 GEMM，
控制缓存显存，不能把完整模型解量化后占满显存。

**多行 target API。** 一次输入 N+1 行并保留各行 logits/argmax 与目标 HC。PLE 哈希、
short-conv、GDN、QSA 必须与逐行因果执行一致。先独立验证多行与逐行，再用于投机。

## P3：TP 图执行

从 GPU-only 固定片段开始，collective 与动态 QSA 边界明确分开；旧 PP graph 含
CPU collective 的执行体不能直接在 TP 捕获。按 rank、M、GDN/PLE 工作 bank 和
其他真实指针身份缓存图，先 M=1，再 2/3/5/7 等验证形状。输入 token/position、
PLE 行和动态元数据在 replay 前更新；预热、分配、CPU 查表不进入 capture。

要求跨请求 changing-input、bank 切换、rollback/reset、context 尾部、取消、capture
失败与 teardown 验证；图实例先销毁，底层 allocation 后释放。记录 capture/replay/
fallback/显存，避免静默回退后宣称启用了 Graph。是否捕获通信留到分段图已验证后。

## P4：TP MTP 与批量验证

1. 审计 MTP artifact 的实际分片，补齐 replicated controls 与 LM head 布局；不能
   把主模型的分片假设套到所有 MTP 张量。两 rank draft token/状态必须一致。
2. 先 greedy，宽度 1→2→4→6；一次草稿后做 N+1 行 target 验证，计算最长接受前缀。
3. GDN/PLE 是递归状态，整 batch 最终 bank 不能直接用于部分接受。比较逐前缀
   checkpoint 与拒绝时从事务起点重算接受前缀；先保证语义，再用实测选择内存/延迟
   方案。QSA 的有效长度、PLE 历史、目标 HC 和 MTP deeper-row reconciliation 同步提交。
   当前 GDN 完整 36 层每份 recurrent+conv bank 约 110.81 MiB/rank，若额外保留
   6 个前缀约需 664.88 MiB/rank（按现有 FP32 recurrent/BF16 conv 布局计算），
   因此值得优先评估前缀快照，不能默认拒绝后整段重算最划算。该估算不包含 QSA、
   PLE、HC、图缓存与工作区，实际额外分配和内存带宽仍须测量。
4. 验证接受长度 0..N、首/中/尾拒绝、全部接受、EOS、取消、context 尾部、两 rank
   失败。提交前不能向外发布 draft；拒绝后必须恢复到对应的 canonical 状态。
5. sampled MTP 只有在 target/draft 概率及拒绝采样经过验证后启用；此前沿用目标模型
   普通采样。自适应宽度依据最近 draft/verify 成本与接受率，而不是固定用最大 N。

## P5：原生细调

- 当前执行基准按用户要求固定为 MTP4。取消额外的宽度扫描、自适应 GPU 测速和
  为 MTP3 新增图形状的工作；保留已有 MTP6 历史证据，不将其作为默认优化目标。
  后续融合、同步、MoE 候选均在固定 MTP4 下做开关对照和必要的状态一致性检查。
- 根据 profile 选择 HC norm/mix、PLE projection/norm/residual、gate/up 和 shared
  expert combine 的融合；已有候选先验证，不重复造开关。控制寄存器与 occupancy。
- MoE 小 M 的专家聚集率通常低，不能假定大 M prefill 内核适用于 7 行验证。分别
  测 grouped GEMM、route-major 小矩阵、W4/W8 vectorized load 和共享权重读复用。
  进一步测 tile 大小、warp 数、共享内存双缓冲和寄存器占用；选择结果按 GPU/量化
  类型/M/N/K 固定记录，热路径不做在线搜索，避免把 benchmark 最优单点当成全局最优。
- PLE 多行提前 hash/查表、批量 pinned staging、H2D 双缓冲；RAM 表仍只拥有一份。
  owner 查表后广播与两 rank 小 H2D 做 A/B，不能假定多一个广播一定更快。
- LM head 当前规约整个词表；评估 vocab 分片 + 全局精确 argmax（稳定 tie-break），
  保持 sampled 模式所需的完整分布语义。它不是 top-N 近似筛选，且需要新 artifact view。
- 在双卡 Gen2 x4 的实际带宽约束下测通信/计算重叠、CPU 亲和性和线程数量；不把
  理论 PCIe 带宽或 C++ 语言本身当成吞吐保证。
- 启动校验/加载单独优化。本轮恢复时 launcher 的完整 segment 校验持续数分钟，
  CPU 接近单核满载；先保留 inode 去重，再测受控并行 SHA256 与加载流水，限制
  I/O 并发和内存。更进一步可评估校验与上传共享读取，但必须保留完整验证及失败
  退出语义。冷启动、热缓存启动、RAM PLE preload 分别计时，不混入 decode tok/s。

## P6：验收与推广

每个候选依次过 CPU/独立 CUDA fixtures → 实权重算子 → 同输入模型 logits/状态 →
重复完整生成 → 性能对照。负收益保留诊断但不默认启用；性能差异接近噪声时增加
交错重复。先验证简单问题，再跑多轮、数学、代码执行和长上下文任务。

记录 raw token/logit 校验、输出完成状态、功能回归及误差边界；不能仅用“能说中文”
作为质量门槛。验证后再切换 30001，保留旧二进制、旧启动命令与状态目录用于回退。

## 实施状态

- 参考库固定 MTP4 消融已完成：关闭 Graph 时非思考 29.0795、思考 19.2979 tok/s；
  开启 Graph 时非思考三次均值 162.8255、思考 93.6676 tok/s。同 W4/W8、RAM PLE，
  非思考 1024、思考 2048 token 均达到长度上限，不能据此声明完整回答质量通过。
  四模式汇总见 Homelab 根目录 `.artifacts/q38-reference-ablation-v2-20260921/comparison.json`。

- 参考引擎同 W4/W8、RAM PLE 消融：Graph/MTP 均关闭时非思考均值 7.7116 tok/s；
  仅启用 Graph、MTP 关闭时 78.2997 tok/s，思考 78.0777 tok/s。三次非思考
  重复及双 rank 图捕获日志已保存于 Homelab 根目录
  `.artifacts/q38-reference-ablation-v2-20260921/mtp0-graph1/`。后续 MTP 固定为4。
  此证据提高图覆盖、CPU 调度和 collective 边界优化的优先级，不直接证明
  原生耗时中 CPU 与 GPU 的具体比例；仍需针对原生执行边界定位。
- 固定 MTP4 融合对照已完成：HC、PLE、投影及组合四项在所测 16 组生成配对、
  40 组拒绝续跑配对中均与基线一致，210 份完整 BF16 快照通过审计。性能对照
  作废：基线无进程 swap，候选约 6.8–7.6 GiB 被换出且测速中有换入，不能将
  表面 4%–10% 降速归因于融合。暂不推广，先释放闲置 NVFP4 tmpfs 占用；保留
  正确性结论。证据：Homelab 根目录
  `.artifacts/q38-native-opt-fusion-20260921/audit-summary.json`。继续优先验证
  输出头同步和批量 MoE；后续测速必须记录并检查驻留情况。
- 方案与基础代码审计：已完成。
- P1 常驻 worker / stream-ordered P2P：已实现并通过首轮真机对照。256-token、
  diagnostics 开启时 32.756→36.990 tok/s（+12.9%），token 与抽取的全词表 logits
  一致；保持 opt-in，长上下文/故障矩阵和关闭诊断的性能复核仍待完成。
  详见[代码、方法与实测边界](native-tp-p1-20260921.md)。
- P2 TP grouped MoE：已接入独立开关，处理各 rank 的空路由、token-major FP32
  partial，并保持 TP 的 post-down 路由加权位置。增加 2..7 行补零 dense GEMM；
  独立 CUDA fixtures 已通过。实权重短测 7 行 prefill 约快 2.48 倍、512 行约快
  3.05 倍，但 logits 数值有差异，精度及完整质量验收仍未完成，暂不默认启用。见
  [P2 实施记录](native-tp-p2-20260921.md)。
- P2 多行目标验证：已增加外部候选 API/RPC、独立批量开关及 CPU 边界测试。
  真机三组各 54 个案例的 token 和提交位置一致；逐行组后续 logits 逐位一致，
  两个批量组各 51 个案例存在数值差异。7 行全接受验证约 177.65→57.15 ms，
  不含草稿、不能等同端到端吞吐。自动 TP MTP 尚未交付。见
  [目标验证记录](native-tp-target-verification-20260921.md)。
- P4 前缀快照：已实现独立开关，直接保存批量计算的 GDN/PLE 前缀状态，部分接受
  恢复快照。算子逐字节测试及模型短测通过：两组各 54 个接受案例、27 个后缀变化
  案例；宽度 6 拒绝验证约 84.53→58.96 ms，每卡额外 665.93 MiB。仍需 GPU
  取消/长上下文及质量验收，尚未推广。见[快照记录](native-tp-prefix-snapshots-20260921.md)。
- P3 单行 TP Graph：GPU 图片段与 CPU collective 按序回放。基础 CUDA 三种通信
  路径/两组共 768 轮通过；真机各 14 案例、1135 token、115 份完整 logits 快照
  一致且零 fallback，含 8192-token 上下文与快照恢复。预热网页生成约
  37.04→39.85 tok/s（+7.6%）。GPU 取消等仍待验证，未默认启用。
  见[Graph 记录](native-tp-graph-20260921.md)。
- P3 多行目标 Graph：M=2/3/5/7、交替状态 bank 的真机对照各 108 案例，
  跨模式 token、完整 tail logits、提交位置一致，各 54 拒绝后缀扰动通过且零 fallback。
  六候选验证中位耗时 59.47→53.48 ms；不含自动草稿，不代表端到端 tok/s。
  见[多行 Graph 记录](native-tp-batch-graph-20260921.md)。
- P5 启动校验：已实现有界并行完整 SHA256；独立真机 4 线程校验 357 个 segment、
  210.84 GB 用时 109.41 秒。缓存状态未受控，不声明相对提速；生产仍用原串行校验。
- P4 自动 TP MTP：独立开关、完整 MTP 权重视图、双 rank 草稿一致性及批量验证已接通。
  首轮三提示×五模式短测完成；代码 MTP6 65.26 tok/s 且 token 一致，但数学和中文
  描述均与普通解码分歧，不能视为质量通过。自适应宽度、分歧归因与 GPU 生命周期
  验收待完成。见[MTP 记录](native-tp-mtp-20260921.md)。
- P0 补充消融、P3 图生命周期/P4 MTP 验收及其余 P5/P6：待实施/验证；不得描述为已交付能力。
- P6 首轮完整短回答：9 个数学/结构化输出/代码执行案例各重复两次，普通目标与
  固定 MTP6 各 18/18 通过，配对 raw token 完全一致。初始检查器错误处理带换行的
  结束标记，已修复并对保留的原始输出重评，原 verdict 未覆盖。证据在
  `.artifacts/q38-native-opt-quality-20260921/reevaluation-summary.json`（Homelab 根目录）。
  此小样本通过不解除更长生成的数值分歧、长上下文及完整质量验收门槛。
- P5 新实验：MTP 提交修复阶段仅保留 QSA 缓存写入，跳过未使用的注意力输出和
  MoE；开关 `Q38_TP_MTP_KV_ONLY_COMMIT=1` 默认关闭，仅 TP2 生效。GPU 同二进制
  A/B 已完成 20 组生成、14 组拒绝后续跑，token 和三步后续解码完整 logits
  配对一致，独立核验 204 份 BF16 快照。后续审计发现候选存在约 7.6 GiB swap，
  性能对照受污染，暂不能判断净收益，仍默认关闭。证据在 Homelab 根目录
  `.artifacts/q38-native-opt-mtp-kv-retry-20260921/audit-summary.json`。该比较仅证明
  此优化与既有批量路径在所测案例一致，不解除批量路径与普通逐 token 解码的
  数值差异及完整质量门槛。
