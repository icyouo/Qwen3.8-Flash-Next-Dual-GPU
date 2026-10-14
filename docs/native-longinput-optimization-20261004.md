# 长输入、SSE 与多轮前缀优化（2026-10-04）

同一条冻结的 4096-token 请求，用未修改的 `llm_speedtest` Python `/ws/test`
框架顺序对照：decode **73.73 → 83.14 tok/s（+12.76%）**，两次均输出
457 token，最终文本逐字一致。短提示词网页为 **178.61 → 178.88 tok/s**，
该请求性能保持。4K 首字等待仍约 **4.7 秒**，本次未显著改善 bulk prefill。

条件：HP Z4，双 64 GiB CMP170HX，SM80，驱动 610.43.02，CUDA 13.3，
每卡 200W；原有 W4/W8、RAM PLE（51,840,248,832 字节，预算 64 GiB）、
固定 MTP4、单并发、greedy、关闭 thinking、durability=off。未调整量化、
草稿宽度或模型上下文上限。

## 同请求结果及边界

| 请求/测量 | 旧版 | 候选版 |
|---|---:|---:|
| 4K 输入 / 457 输出 decode（tok/s） | 73.73 | 83.14 |
| 4K prefill（tok/s） | 874.83 | 868.44 |
| 4K TTFT（ms） | 4699.28 | 4728.39 |
| 网页 16 输入 / 1024 输出 decode（tok/s） | 178.6085 | 178.8752 |

4K 使用 upstream commit `eb19940368f0d811ef7b14f77f61d1c7e8676c7a`，
冻结 effective-request JSON，max_tokens=512，temperature=0，top_p=1。
token 数来自 API usage，无估算、无 reasoning；边界来自首个非空 content。
Mac 通过 SSH loopback 调用，适配层只设置 no-thinking，不改写 SSE。
正式测量请求完全相同，均先做 8 输出 token 的框架预热；预热文本和历史
kernel/cache 状态不能视为完全相同。只有一组顺序对照，不宣称稳定均值。

两次最终文本 SHA256：
`676c591a4cf676b89a230a53e41d9a7b09726348b0afcc67df602f52521b6e9c`。
两次最终 canonical=4553、target/stage0/stage1/draft=4552。
4K 指标窗口包含预热，不把接受率计数当作单独测量请求的完整草稿轨迹。
无 executor failure/rollback、CUDA Graph fallback、CUDA 分配失败增长或
进程 swap。两次最终文本相同不意味着每个预热 token 或 MTP 草稿完全一致。

短网页在 4K 对照后再预热 128 输出 token，各测一次，使用历史本机 SSE
首尾 content 时间方法。两次 1024 输出 token 均与既有冻结网页逐字一致。
该预热顺序不同于此前 172 tok/s 的三轮结果，不把 172→179 归因于本次改动。
与此前 vLLM 的 88.56 tok/s 也不是本轮重新配对的对照；其回答长度不同。

## 已实现的改动

1. **长位置 QSA score 的独立行分区**：仅对 1–7 个目标行、位置超过 2048
   的稀疏验证使用 8 个 CTA；短上下文保留原分区。各行 Q·K 的 FP32 累加
   次序保持，dense/sparse 全模型图分别捕获，位置仍按设备值动态 replay。
   GPU fixture 覆盖 M=1/4/5、两种本地头布局、0/2046/4096/8189/262137
   位置和 eager/graph；score 与完整 attention 输出逐位相同。
   单算子示例 M1 394.02→158.57µs、M4 302.47→138.11µs、M5
   303.95→145.75µs；这不是整个引擎的提速比例。
2. **bulk GDN 独立列分区**：prefill 使用 16 列，原配置为 32；小目标批次
   本来就是 16。512 行、双 rank、四种投影布局的 conv/recurrent 状态、
   prefix snapshot 和 BF16 输出 eager/graph 等价。局部 512 行示例
   3652.51→3475.56µs，但整段 4K TTFT 没有可确认的收益。
3. **增量解码**：ByteLevel tokenizer 使用其原生 `DecodeStream`，跨 token
   的 UTF-8 等待后续字节，最终只做一次官方累计 decode 核对及尾部 flush。
   其他 tokenizer 保留累计 fallback。Chinese/emoji、truncated bytes、
   字面 U+FFFD、停止 token 和 parser tags 已验证。454 个输出 token 的
   CPU fixture 从累计 15.58ms 降到 0.77ms；不能当作推理倍数收益。
4. **有界 SSE producer**：最多 2 个排队 committed event 加 1 个在途事件。
   producer 独占语义 writer lease，只发布已经提交的 token；消费者断开时
   只取消该请求的精确 RPC ID，并等待 producer drain/关闭。慢客户端不会
   创建无界输出队列或取消后来请求。midstream 错误用 SSE error + DONE，
   不再写第二套 HTTP headers。正常请求及 API cancel/disconnect 后恢复
   已通过真实 HTTP 验证。
5. **前缀诊断及受限操作**：增加 optional opcode 12，允许替换唯一尚未计算
   的末尾 stop token。必须 greedy、精确 canonical 长度/epoch/旧 token、
   各计算 frontier 同步、无在途事务；只改 canonical/sampler bookkeeping，
   不改已经计算的 GDN/QSA/PLE 状态。普通分歧继续完整重建；取消按原提交
   边界处理，RPC replay 不重复推进 epoch。**暂不默认启用**：数学与真实 4K
   对话下一轮的 seed/后续 token 相同，但全词表 BF16 logits 和 probe logits
   与完整重算不逐位相同，未通过本轮严格门禁。独立 sidecar 的
  `--reuse-terminal-prefix` 保留用于进一步诊断，正式 profile/launcher 不开。

前缀候选的 append logits：数学案例 235070/248320 行不同，relative L2
约 3.33%、最大绝对差 0.421875；4K 案例 240372 行不同，relative L2 约
5.25%、最大绝对差 0.734375。top1 相同，但不能把它概括成已验证的无损
复用。bulk 重算和增量历史的计算路径差异仍需要定位；本轮未定因。

新增配置为 `mtp4-native-longinput`。前四项的 GPU 及完整模型验收已经通过。
第一批候选二进制 SHA256：
`3b633d9e0e68f386c696c8f364e5bb5e3697cdd434e6b4bee0632d9ccbc873fa`，
与上述旧二进制 `7199f47e...` 的性能对照保存在独立原始目录。

## 当前瓶颈与可靠性

新诊断显示 4K 冷/热 prefill 仍约 4.79/4.77s，仅缓存展开的 BF16 权重不能
解释全部时间。旧版单 rank/stage 的 512 行区间，attention_finish 和
moe_routed_reduce 较长；这些粗粒度计时包含投影/状态更新/通信等待，不能
直接说都是 GDN 或归约计算。新增本地 GDN 投影、状态更新、norm/output、
collective 的独立标记，供后续分离真实关键路径。不能把双 rank 的并行时间
相加当作端到端比例。

长位置稀疏 target 的 QSA 明显比短位置慢，本轮已改善。MoE、GDN、输出头
及 MTP 接受率仍限制整体 decode；4K 哲学作文约三分之一草稿接受率，与
短网页的高接受率不同。固定 MTP4 保持不变，本轮没有扩大参数扫描。

Z4 安装 `/etc/systemd/journald.conf.d/90-q38-persistent-log.conf`，
Storage=persistent，SyncIntervalSec=5s，并 flush。Mac 对 kernel journal、
boot ID、GPU 功耗/温度、meminfo/vmstat/PSI 持续采集、每行 fsync，便于在
整机掉线后保留证据。本轮没有新重启；此前两次整机重启原因仍未查明，
未发现事件窗口内能定因的 OOM/Xid/panic/MCE。日志增强不能证明可靠性
问题已经修好，也不能排除采样间隔内的供电瞬态。

## 验证与证据

CPU runtime/RPC 包括前缀边界、epoch CAS、pre-cancel、sampler checksum、
cached seed 失效和 replay；104 项 Python 检查通过，Mac 跳过缺少 tokenizers
的 optional fixture，Z4 补跑该 fixture。完整模型包括五类输入及 8192
上下文、248320 维 BF16 logits、十个强制拒绝前缀、九项完整质量回答、三类
取消/deadline/reset 恢复。比较基准仍为此前 qualified Marlin scaled-BF16
后端，不宣称与未量化官方权重逐位相同。

Mac 原始目录：`/Users/icy/Homelab/.artifacts/q38-native-longinput-20261004/`；
Z4：`/home/icy/q38-native-longinput-20261004/`。

- `baseline/`、`candidate/`：请求、原始 SSE、WS 框架事件、metrics、profile
  与二进制 manifest，`candidate/api-gates.json` 包含真实取消/断线恢复。
- `profile/`：旧 qualified 二进制的诊断标记和冷/热 4K 请求；不作为速度成绩。
- `full/`：第一批全部 token/BF16/拒绝前缀/质量/取消门禁。
- `prefix-state/`、`prefix-full/`：受限前缀操作的新身份及验收结果。
- `incident/`：Mac 持续采集；日志配置与完整退出状态另存。

所有新完整启动身份均由官方 launcher 校验 354 独立 segment、
212,367,104,472 字节 SHA256 后生成。CUDA/CPU 编译与完整模型加载分开。
历史结果不覆盖；测试脚本错误与修正另保留日志，不把失败当作成功请求。

## 最终安装与复测

最终安装二进制：`build/native-longinput/q38-cuda-runtime`，SHA256
`3fde7f8d88db5063dba2998922d2a7485b3024416d0cf4dbb6e9a869e87f5f9d`。
该版本新增受限 RPC，但正式配置关闭前缀替换；此前 GPU/SSE 改动保持。
第二套官方身份重新完整校验 354 segment，随后在无 logits/event 诊断的
独立进程复测同一 4K 请求：**873.51 prefill / 83.59 decode tok/s**，
**4711.10ms TTFT**，4096 输入、457 输出。旧版、第一候选与最终安装版本
三次最终文本 SHA256 完全一致。短网页 16/1024 为 **176.8178 tok/s**。
这些是独立确认的一次测量；+12.76% 仍按最初配对结果计算。

最终版本的五类 token/BF16 输入、十个拒绝前缀、九项质量回答及三类取消
恢复均通过旧 oracle。真实 HTTP 再次检查 API cancel、取消恢复和消费者
断线后的下一请求恢复。CPU/rank-worker 与 Mac 104 项 Python 通过（1 项
optional 缺依赖跳过），Z4 同样 104 项全部通过。`source-verification.json`
确认 164 个生产输入源码文件的 Mac/Z4 SHA256 一致。最终 compiled GPU
fixture 的日志为 `qsa-fixture.log`、`gdn-fixture.log`。

正式启动使用新 state 目录：

```bash
cd /home/icy/Homelab/Qwen3.8-Flash-Next-Dual-GPU
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 .venv/bin/python tools/q38_tp_launch.py \
  --ready /home/icy/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --tp-root /home/icy/q38-native-target160-20261004/gdn-qkvz-artifact \
  --runtime build/native-longinput/q38-cuda-runtime \
  --execution-profile mtp4-native-longinput \
  --ple-storage ram --ple-ram-budget-gib 64 \
  --segment-hash-workers 2 \
  --state-dir /home/icy/q38-native-longinput-serving --port 30001
```

原 GPU 策略和二进制完整保留：改用
`--runtime build/native-mtp4-fast/q38-cuda-runtime --execution-profile mtp4-native-fast`
并用另一个新 state 目录即可回退 GPU 调度。旧二进制 SHA256 仍为
`7199f47ebf5d8dabf6b05611c14e55d9d66e7caf5e3dc8f4c95f2026f169613b`。
HTTP 支持保留独立回退开关：直接启动 sidecar 时使用
`--stream-decode cumulative --stream-buffer 0`。前缀实验开关不在正式启动
命令中；不能用它来宣称本轮 4K 性能结果。

测试进程退出，双卡显存释放、200W 限制保持，挖矿和其他 AI 服务继续停用。
本轮没有设置常驻 LLM 服务。`production/` 与 `FINAL.json` 保存最终确认，
`cleanup.json` 保存 boot ID/GPU/进程退出证据；历史 32K/4K 重启问题仍待定因。
