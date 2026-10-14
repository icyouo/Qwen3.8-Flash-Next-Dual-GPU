# 双 CMP 170HX P2P 实测（2026-10-11）

P2P 可以直接读写，当前链路的大块双向传输约 **1.40 GB/s / 方向**。
原生 bulk 归约已接近同形状 NCCL 直接 P2P 的耗时。此前的直接 peer-read
归约在双向同时读取时吞吐较低；改成 peer copy 后本卡归约的收益可以复现。
这些是通信微基准，整模型剩余性能差距还需要分段 profiling。

以下主表使用 **27B 服务停止后的五轮中位数**。最初完整扫描时
`huihui-w4-vision-video` 仍驻留 GPU 0，前后请求与输入/输出 token 计数增量
均为 0。此前整模型测试结束后曾恢复这个容器；用户指出状态后，本轮已
再次停止，并在停止状态补测 prefill 对应的两个大块形状。最终容器
`running=false`，两卡显存各 306 MiB，保持停止。

## 硬件与实际路径

- hp-z4-g4，双 CMP 170HX 64 GiB，CUDA sm_80，驱动 610.43.02，CUDA 13.3，
  两卡均维持 200 W。
- 当前协商链路 **PCIe Gen2 ×4**；查询的最大代数为 2、最大宽度为 16。
  最大宽度不等于当前工作宽度，当前数据不能证明仅改配置就能达到 ×16。
- GPU bus ID 为 `15:00.0` 和 `21:00.0`。`nvidia-smi topo -m` 标记为
  `NODE`，即同一 NUMA 节点内跨 PCIe Host Bridge，没有 NVLink。
- `topo -p2p r/w` 两方向均显示 OK；本次另外执行了真实 peer 读、写与复制，
  并校验完整数据、边界和源数据保留，结论不只来自 capability flag。
- Gen2 ×4 按 8b/10b 编码计算的单向上限是 2.0 GB/s，尚未扣除 TLP 等
  协议开销；约 1.4 GB/s 是本机实测值，不能混用为同一口径。

## 纯传输

5 MiB（5,242,880 字节）每方向，五轮中位数，GB/s 使用十进制。
双向列是同时传输时的**每方向**速率，不是两方向合计。

| 方法 | GPU 0 → 1 | GPU 1 → 0 | 双向同时，每方向 |
|---|---:|---:|---:|
| `cudaMemcpyPeerAsync` | 1.431 | 1.431 | 1.396 |
| SM 直接 peer read，标量 | 1.445 | 1.445 | 1.135 |
| SM 直接 peer read，16 字节向量 | 1.445 | 1.445 | 1.157 |
| SM 直接 peer write，16 字节向量 | 1.441 | 1.440 | 1.400 |

双向 copy 的 CPU wall 合计吞吐为 2.791 GB/s；这不表示每方向 2.791 GB/s。
同卡 5 MiB copy 约 812–829 GB/s，说明大块 peer copy 与本卡 copy
是不同量级。向量化 peer read 仅部分改善双向退化，没有达到 copy/write
的双向速率。

与 prefill 对应的两种 payload：

| 每方向字节数 | 双向 peer copy | 双向标量 peer read | 双向向量 peer write |
|---|---:|---:|---:|
| 2.5 MiB / BF16 `512 × 2560` | 1.884 ms | 2.316 ms | 1.875 ms |
| 5 MiB / FP32 `512 × 2560` | 3.756 ms | 4.620 ms | 3.744 ms |

此表为 CUDA event 区间；原始数据同时保存 CPU wall 区间。连续提交
小包时 event 区间也可能包含 CPU 提交间隙，不能把它一概解释成纯总线耗时。

## 真实原生 bridge 与 NCCL 对照

直接链接当前 `cuda/cuda_te.cu`，每次先生成各 rank 输入再归约，包含
producer、rank rendezvous、通信、本卡归约和最终同步，不含分配、初始化
communicator 和校验。所有实现使用相同输入、形状和 rank 线程组织。

单 group，512 token，五轮 CPU wall 中位数：

| 方法 | BF16，2.5 MiB/rank | FP32，5 MiB/rank |
|---|---:|---:|
| 原生 host bounce | 3.292 ms | 6.495 ms |
| 原生 scalar P2P | 2.498 ms | 4.974 ms |
| 原生 bulk P2P | **1.933 ms** | **3.804 ms** |
| NCCL 默认，日志确认 SHM/direct | 2.765 ms | 5.327 ms |
| NCCL `NCCL_P2P_LEVEL=SYS`，确认 P2P/direct pointer | **1.938 ms** | **3.813 ms** |

原生 bulk 相对 scalar 延迟分别减少 22.60%、23.52%。两个大块归约仅比
纯双向 copy 的 wall 区间多约 48 µs，当前形状下已经接近实测传输速率；
这不是对 CPU、producer、reduce 各自耗时的独立分解。

两个 group 同时运行，用于观察两个 stage 的通信争用。表中时间是两组
完成一轮的最大 wall 区间，不能再把两组时间相加。

| 方法 | BF16，每 group 2.5 MiB/rank | FP32，每 group 5 MiB/rank |
|---|---:|---:|
| 原生 host bounce | 3.964 ms | 7.799 ms |
| 原生 scalar P2P | 4.867 ms | 9.910 ms |
| 原生 bulk P2P | 3.898 ms | 7.736 ms |
| NCCL 默认 SHM | 5.178 ms | 10.179 ms |
| NCCL 直接 P2P | 3.795 ms | 7.541 ms |

bulk 在两个 group 上比 NCCL 直接 P2P 慢约 2.7%，可以继续检查调度开销；
并发两组没有让同一 PCIe 路径获得两倍总吞吐。

NCCL 库是从现有 27B 容器只读提取的 **2.30.7+cuda13.3**，不是对历史
Flash-Next vLLM 镜像或实际 all-reduce 路线的确认。默认测试日志选择
SHM/direct；仅对诊断进程设置 `NCCL_P2P_LEVEL=SYS` 后选择
P2P/direct pointer。这个参数允许指定拓扑距离以内使用 P2P，定义见
[NVIDIA 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level)。
本轮没有向服务配置写入此参数，也不能据此声称历史 vLLM 走 SHM。

完整扫描还覆盖 1/5/128 token。小包原生 bulk 不跨 256 KiB 门槛，仍用
scalar 路径。初始扫描单 group、1 token 的原生 eager 约 79–81 µs，
NCCL 直接 P2P 约 34–36 µs；这提示 eager 的线程/提交开销值得拆分。
这组小包结果来自 27B 驻留但无请求的阶段，未在停服务后重测，且不代表
当前 decode CUDA Graph 路径的实际收益。

## 对 4K prefill 的意义

按当前 48 层、hidden 2560、512 token chunk，以及每层一次 BF16 attention
输出归约、一次 FP32 routed-MoE 输出归约计算，4096 token 的这两类数据为：

```text
每 rank 接收量 = 48 × 4096 × 2560 × (2 + 4)
              = 3,019,898,880 字节（3.02 GB / 2.8125 GiB）
```

这是根据代码和形状计算的估算，尚未由整模型 trace 核对，不包含其他
gather/head 通信。若把单 group 两个归约的中位数按 48 层、8 chunk
串行相加，约为 2.20 秒；stage/chunk 重叠、CPU 等待与实际执行路径会改变
它在 TTFT 中的占比，不能直接称为实测通信壁时。

上一轮完整 4K API A/B 的 prefill 中位数从 864.58 提升至 990.04 tok/s，
TTFT 从 4751.24 降至 4143.70 ms；本轮通信微基准解释了部分收益，不是新的
整模型 A/B。详见 [整模型结果](native-vllm-parity-20261011.md)。

下一步应记录实际 collective 次数、字节数与等待区间，检查通信与 kernel
重叠，并继续拆分 GEMM/MoE/attention。FP32 routed 输出占上述估算字节数
的 2/3；改变其数值精度需要另行通过数值门禁，本轮未更改。
此前 cache-hit 差异已排除，KV 布局或读写开销仍需单独测量。

## 校验、复现与证据

新增 `cuda/q38_p2p_bench.cu`、`cuda/q38_collective_bench.cu` 和两个 Makefile
target。生产 `cuda_te.cu`、runtime/profile 未改，已安装候选 runtime 的
SHA256 仍为 `ad7f2651dbc0594182c0ee8b08e34b871c63d2afbf540a78c64de66e0e5d0dd9`。
两个诊断源和 Makefile 已同步 Z4 正式源码；覆盖前核对原 Makefile hash，
旧 Makefile 备份在实验目录 `Makefile.before-bench`。

- 原始完整扫描：630 个 raw、320 个 native/NCCL 默认、80 个 NCCL P2P
  timed batch；停服务补测：180 个 raw、80 个 native/NCCL 默认、20 个 NCCL
  P2P batch。合计 **1310 个 batch**，每个条件五轮，全部通过校验。
- raw 使用独立不可变 source/output，双向无 in-place 读写竞争。每个 timed
  batch 后检查完整整数 payload、两端 guard、未参与 GPU sentinel、全部
  source 保留；校验在计时外。
- collective 在每次归约前重置输入，避免反复累加溢出。每个 timed batch
  后检查两 rank 的完整输出和 guard。BF16 按 RNE oracle，FP32 逐位。
  输入是固定有限值分布，不能替代任意浮点输入或整模型 logits 门禁。
- full scan raw 大小为 4 KiB/64 KiB/256 KiB/2.5 MiB/5 MiB/16 MiB/64 MiB。
  `--prefill-only` 仅筛选 2.5/5 MiB payload 或 collective 512 token。
- 本地保留两阶段实时 GPU/内存/内核记录，boot ID 一致；观察窗口中没有
  新 Xid/AER/OOM，OOM counter 增量为 0。所有诊断与采集进程已退出。
  没有重启，也没有修改驱动、功率或生产 NCCL 参数。

在 Z4 正式源码目录可编译/运行无需 NCCL 的版本：

```bash
make p2p-bench
make -B build/q38_collective_bench
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 ./build/q38_collective_bench
```

可选 NCCL 对照需要提供现有 `nccl.h` 与 `libnccl.so.2`，不要求修改服务：

```bash
NCCL_REF=/home/icy/q38-p2p-diagnostic-20261011/nccl
make -B build/q38_collective_bench \
  P2P_BENCH_CPPFLAGS="-DQ38_P2P_BENCH_NCCL -I$NCCL_REF" \
  P2P_BENCH_LIBS="-L$NCCL_REF -l:libnccl.so.2"
LD_LIBRARY_PATH="$NCCL_REF:/usr/local/cuda-13.3/lib64" \
  ./build/q38_collective_bench --prefill-only
NCCL_P2P_LEVEL=SYS LD_LIBRARY_PATH="$NCCL_REF:/usr/local/cuda-13.3/lib64" \
  ./build/q38_collective_bench --nccl-only --prefill-only
./build/q38_p2p_bench --prefill-only
```

修改编译开关后使用 `-B`，避免 Make 复用另一组 flags 的旧二进制。
本轮验证了 Makefile 的 raw、带 NCCL collective 和不带 NCCL collective 构建。

本地证据：`build/p2p-diagnostic-20261011/`；服务器证据：
`/home/icy/q38-p2p-diagnostic-20261011/`。`summary.json` 保存每条件五轮的
中位数/min/max，`*-stopped.jsonl` 是主表来源；其余 JSONL、NCCL 日志、
metrics、源文件/二进制 hash 和两阶段 telemetry 保留原始记录。可运行
`python3 build/p2p-diagnostic-20261011/summarize.py` 重新核对结果。
