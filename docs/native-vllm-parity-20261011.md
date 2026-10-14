# 原生运行时追平 vLLM：真实负载性能第一批

最新精确传输候选采用关闭诊断的两轮 A/B，每版本六次：4K TTFT 3.506→3.346s、Prefill +4.77%；对照同日 vLLM 2.777s，仍有 20.51% TTFT 差距。见[最新验收与启动说明](prefill-exact-transport-20261011.md)。

后续同轮 vLLM/原生耗时分布与第二项无损通信优化见
[prefill 耗时分布](prefill-distribution-20261011.md)：原生本机 HTTP 中位数
TTFT 3.95→3.45s，输入/TTFT 1037→1188 tok/s；vLLM 同轮 2.78s。
计时边界与本页的历史 llm_speedtest 不同。27B 此后保持停止，NAS 已规范到
`/mnt/nas`，旧路径保留兼容。

2026-10-11。优先处理 prefill、TTFT 和 decode，沿用双 CMP170HX、每卡 200W、
现有 W4/W8、RAM PLE 和固定 MTP4。

## 当前基线与验收目标

| 冻结 4K 写作请求的历史测量 | 原生 longinput | vLLM 适配版 |
|---|---:|---:|
| 实际输入 token | 4096 | 4096 |
| 实际输出 token | 457 | 336 |
| Prefill tok/s | 873.51 | 1485.51 |
| Decode tok/s | 83.59 | 88.56 |
| TTFT ms | 4711.10 | 2782.92 |

来源：[原生最终确认](native-longinput-optimization-20261004.md)与相邻
`Qwen3.8-Flash-Next-VLLM/docs/llm-speedtest-20261004.md`。两者回答长度不同，
预热状态也不完全相同。这是确定优化方向的历史参考，不是本次配对 A/B。
短网页已有约 177–179 tok/s，不能用它证明真实 4K 负载已经追平。

正式验收需要冻结相同 messages、greedy/no-thinking/MTP4、权重、功率和
计时边界。原生候选与原生基线须先通过 token、完整 BF16 logits、拒绝前缀、
完整质量回答及取消恢复检查，再在关闭诊断的独立进程中各测至少三次 4K
请求，并检查短网页回归。与 vLLM 的新配对需分别报告真实输出长度、接受率、
TTFT、N/N−1 decode 和端到端吞吐，不把不同输出当作等长推理比较。

## 本次完整模型配对结果

原始 llm_speedtest `/ws/test` 各三轮，客户端位于 Mac，通过本地包装及 SSH
转发访问 Z4。测量函数、SSE 和 token 公式未修改；包装冻结随机 prompt，
添加 `enable_thinking=false`。两边均使用 incremental tokenizer decode、
stream buffer=2、固定 MTP4 和每卡 200W；加载完成后关闭 logits/event 诊断。
每轮先执行原工具的 8 输出 token 短提示词预热。

| 原生版本，三轮中位数 | 已安装 longinput | 新 prefill 候选 | 变化 |
|---|---:|---:|---:|
| 输入 / 输出 token | 4096 / 457 | 4096 / 457 | 一致 |
| Prefill tok/s | 864.58 | **990.04** | **+14.51%** |
| TTFT ms | 4751.24 | **4143.70** | **−12.79%** |
| Decode N tok/s | 85.83 | 85.25 | −0.68% |
| Decode N−1 tok/s | 85.65 | 85.06 | −0.68% |
| 输出 / 客户端端到端秒，tok/s | 45.42 | **48.28** | **+6.30%** |

Prefill 三轮分别为基线 866.27 / 862.22 / 864.58，候选
985.98 / 1058.74 / 990.04 tok/s。候选有轮间波动，主表使用中位数；decode
范围分别为 82.93–86.43 与 83.54–85.77 tok/s，没有确认 decode 收益。
Prefill 仍为输入 /（首个非空内容 SSE 时间减平均目录请求延迟），不是纯
GPU kernel 吞吐；N−1 decode 使用相同 decode 时间扣除首 token。

六次有效请求、API usage 和完整最终文本一致，reasoning 为空，均无 token
警告或计时边界回退；六次都为 cold rebuild，prefix=0、suffix=4096，未
复用 terminal prefix。完整文本 SHA256：
`676c591a4cf676b89a230a53e41d9a7b09726348b0afcc67df602f52521b6e9c`。
包含三次短预热的草稿接受率为基线 33.36%、候选 33.54%；MTP 宽度仍为 4。
两边 failures/rollbacks/Graph fallback/CUDA allocation failure 均无增量，
运行时 process swap=0。

短网页 16 输入 / 1024 输出的独立回归为 176.89→177.13 tok/s，内容与
此前 qualified 输出完全一致。真实 API cancel、取消后的下一请求，以及
消费者断线后的下一请求恢复均通过。

**尚未追平 vLLM。** 990.04 tok/s 仍低于 vLLM 历史 1485.51 tok/s；该
历史请求输出 336 token，本轮没有重跑 vLLM，不能把历史差值宣称为新的
配对提升或最终验收。下一批仍需分离剩余 GDN/attention 输出、FP32 MoE
归约和 QSA 输出投影/通信的关键路径；旧粗粒度 profile 含通信等待，不能
把双 rank 的并行计时相加归因。

## 保留的第一项候选：大块 eager P2P 归约

`cuda/cuda_te.cu` 的原 eager P2P 归约由 SM 从另一张卡直接读取输入、写入
scratch，待双方读完后复制回输出。新增可选路径对至少 256 KiB 的归约输入
用 `cudaMemcpyPeerAsync` 复制远端输入至本卡 scratch，再在本卡归约。

双方 producer-ready 和 read-done 事件保持不变；必须完成双方读取才能覆盖
任一输入。后续 scratch 复用仍等待 output-done，增长、reset 和 transport
切换仍 drain。两卡都按 rank0 + rank1 的次序计算，保留 FP32 加法和 BF16
RNE。AllGather、较小归约、paired/device Graph 路径均沿用现有实现。

显式候选 profile 为 `mtp4-native-prefill-candidate`，仅在
`mtp4-native-longinput` 上加入 `Q38_TP_BULK_COPY=1`。原有 profile 清理环境
后不会继承该开关；运行时和 profile 继续进入 session identity。未修改量化、
草稿宽度、前缀复用策略或正式默认配置。

### 局部双卡结果

真实 SM80 双卡、512×2560 元素、各三次，每次 30 轮，包含 producer、CPU
双 rank rendezvous、传输、本地归约和最终同步：

| 归约 | 原标量 P2P 均值 | 复制后归约均值 | 延迟下降 |
|---|---:|---:|---:|
| FP32，5,242,880 字节 | 4986.23 µs | 3798.04 µs | 23.83% |
| BF16，2,621,440 字节 | 2509.26 µs | 1926.21 µs | 23.24% |

逐位比较以旧标量 P2P 为 oracle，覆盖两个 stage group、同步和异步模式、
阈值两侧、非对齐指针、非 block 对齐尾部、守卫、延迟 producer、交替
nonblocking stream、排队中的 scratch 增长与小/大输入切换，并检查 CUDA
device restoration。`bulk-copy-test` 可重跑。

测试时另一个 27B 服务仍驻留 GPU 0，采样时没有计算活动。这些局部测量
不代表独占 GPU 的整模型 A/B，也不能推导已经获得相同比例的 TTFT 改善。

## 已撤回及未纳入的实验

GDN warp prefill 候选保持四单元 partial sum 与 32 项顺序归约，eager/Graph
的输出、状态和 prefix snapshot 逐位检查通过，但 512 行局部耗时由约
3485.85 µs 增至 4118.02 µs，已从生产源码撤回。GDN 源码和 fixture 的
SHA256 已核对恢复到原始版本。

既有 GDN 8 列分区也完成两个 rank、四种 projection layout、1/5/8/17/512
行、eager/Graph 的输出/状态/快照检查。单次局部比较为 8/16/32 列
3436.13/3486.05/3662.85 µs；8 列只有约 1.4% 收益，当前候选继续使用
已经合格的 16 列策略。

隔离的 GDN split-partial 实验保留原加法顺序，将 memory/core partial
放入不同 shared buffer，以减少每 token 的同步。逐位 fixture 通过，但
512 行局部耗时仅约 3423.85 µs，相对原 16 列约改善 1.8%，未纳入本次候选。

## 验证与运行状态

- Mac：C++ runtime/rank-worker、105 项 Python 检查通过，1 项 optional
  tokenizer fixture 缺依赖跳过。首次沙箱运行的 socket 权限错误已在授权的
  沙箱外重跑解决。
- Z4：候选完整 CUDA 运行时编译、105 项 Python 检查和独立 bulk GPU
  fixture 通过。原有异步 P2P fixture 在新开关启用时通过，覆盖 producer、
  inplace gather、BF16 RNE、多 group、排队调用、stream 切换和 reset。
- 整模型 oracle 门禁通过：五类输入含 8192 上下文、248320 维完整 BF16
  logits、十个强制拒绝前缀、九项完整质量回答、三类取消/deadline 恢复。
  Oracle 是此前合格的 Marlin scaled-BF16 后端，不是未量化官方权重。
- 新 profile 的三轮真实 HTTP 测量、短网页回归和 API 恢复检查均已完成。
- 新二进制安装到 Z4 `build/native-prefill/q38-cuda-runtime`，六个实现/测试
  文件同步至正式源码并核对 SHA256。旧 longinput 二进制和 profile 保留；
  原源码备份在服务器实验目录的 `install-backup/`。

服务器实验目录：`/home/icy/q38-native-prefill-20261011/`；
本地证据：`build/native-prefill-20261011/`。
二进制：正式源码目录 `build/native-prefill/q38-cuda-runtime`，隔离实验目录
中的原始构建副本也保留。SHA256
`ad7f2651dbc0594182c0ee8b08e34b871c63d2afbf540a78c64de66e0e5d0dd9`。

当前 `huihui-w4-vision-video` 容器在端口 30002 提供独立的 27B 文字/图片/
视频问答，GPU 0 约占 23 GiB。用户已授权暂停，整模型测试期间已停止该
容器，测试结束已恢复，端口 30002 `/health` 返回 200，GPU 0 显存
23339 MiB（包含其他原有 context）；局部测试与 prepare-only 校验没有
停止该服务。

354 个独立 segment、212,367,104,472 字节的完整 SHA256 校验已通过。
权重路径已迁移到 NAS，完整哈希校验和模型加载耗时单独记录，不计入
prefill、TTFT 或 decode。

正式模型准备命令使用完整 segment hash 校验、新 state 目录及该候选
profile。回退选择 `mtp4-native-longinput`，并用新身份/state；旧二进制完整
保留。新 profile 已通过上述门禁，但不由旧 profile 隐式启用。

```bash
cd /home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 .venv/bin/python tools/q38_tp_launch.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp-root /home/icy/q38-native-target160-20261004/gdn-qkvz-artifact \
  --runtime build/native-prefill/q38-cuda-runtime \
  --execution-profile mtp4-native-prefill-candidate \
  --ple-storage ram --ple-ram-budget-gib 64 --segment-hash-workers 2 \
  --state-dir /home/icy/q38-native-prefill-serving --port 30001
```

启动整模型仍需为双卡留足显存。回退改为
`--runtime build/native-longinput/q38-cuda-runtime --execution-profile mtp4-native-longinput`
并使用另一个新 state 目录。完整请求/SSE/WS、公式复核在本地证据目录的
`baseline/`、`candidate/`、`summary.json`；`full/` 保存 logits SHA256、质量、
拒绝和恢复记录；`installation.json` 保存复制后的二进制及源文件校验。
服务门禁结束前后 boot ID 一致，200W 保持；本次无新重启，历史整机重启
原因仍未查明。

后续 P2P 独立诊断与停 27B 后的复测见
[P2P 实测](p2p-diagnostic-20261011.md)。27B 此后再次停止并保持停止；以上
“已恢复”描述的是整模型测试结束时的状态。
