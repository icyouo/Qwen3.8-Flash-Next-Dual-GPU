# MoE prefill 候选测试与 Z4 重启：2026-10-11

> 2026-10-12 用户选择 16 位后，独立 BF16 通信候选完成有限质量/恢复及性能验证并安装，详见 [BF16 报告](bf16-transport-20261012.md)。该改动未合入本文的 bulk MoE 候选；原重启原因仍未解决。

本轮没有执行服务或主机重启命令。临时原生 CUDA 测试期间 Z4 整机重启，
boot 从 `befb0fc7-a99f-4c97-9b50-7c677a291e83` 变为
`c354ceb6-2af6-4725-845f-aa7b20afc984`。本轮候选未通过完整验收、未安装到正式源码或入口，
没有性能提升结论。之前已验证的 FP24 二进制和九个执行配置保持原样。

## 候选及已经完成的测试

实验在 `/home/icy/q38-moe-prefill-20261011/runtime-source` 隔离进行。
将已有 ≤7-token 的 Marlin BF16 直接 top-10 汇总扩展到 8–512-token prefill，
保留专家 BF16 舍入和 assignment 顺序 FP32 加权汇总，省去
5120×2560 FP32 中间数据的生成与再次读取（约 50 MiB）。该缓冲区仍已分配，
没有宣称减少峰值内存。显式候选开关为
`Q38_CUDA_MARLIN_PREFILL_DIRECT_REDUCE`，先在同一候选二进制上开关 A/B。

GPU fixture W4/W8、M=1/2/3/5/7/8/19/511/512、五次 owned-prefix 切换、
poisoned tails、反序 assignment 和非整系数权重、CUDA Graph 重放逐位一致。
512-token 四专家合成 fixture：W4 整段 0.315→0.280 ms，W8 0.370→0.336 ms。
这只是合成算子数据，不是整模型速度。运行时编译完成后才加载模型；
本轮没有 CPU/CUDA 编译与模型加载/推理重叠。

正式 HTTP 计划为固定 4096 输入/完整 457 输出、prefix=0、MTP4、chunk512、
RAM PLE64GiB、200W/卡，profiling 与 logit diagnostics 关闭。
旧 FP24 路径三次 TTFT 3385.43 / 3285.93 / 3265.26 ms，完整文本 SHA 与冻结答案一致。
新路径在测试中发生重启，`moe-direct/completed.json` 不存在，部分
`result.json` 和日志重启后含空洞/零字节。原始文件原样保留，不修复后计入成绩。
反转顺序复测、全模型 logits/质量/恢复门禁和新身份/分段验收均未执行。

候选 SHA256：`8fac6776896a481f9cf0e08857b2b8e3ef607197367fb3b447370d65fb9de0f2`。
保留正式 FP24 SHA256：`73269e71cd9edc3305f36adf9a1acc94860c0a54d76d27ea5055e7d12b0f5be9`。
五个涉及文件 Mac/Z4 正式源码均仍与候选开始前版本一致。

## 故障证据

Mac 每两秒采样并逐条 fsync，共 179 条，最后一条 UTC
`2026-10-11T15:06:37.279660+00:00`（北京时间 23:06:37）。该条 GPU55/53°C、
165.88/149.03W，200W 上限，显存58231/58591MiB；MemAvailable 约
187.56GiB。整个外采窗口 swap-in/out 与 OOM kill 增量为0。
外采和旧 boot 事件窗口没有记录 Xid、OOM、AER、panic、MCE 或正常关机。
旧 boot 的持久化日志止于北京时间23:06:33，新 boot 首条为23:07:41。
这些记录不能确定重启原因；同一 Z4 之前原生和 vLLM 均有整机重启历史。

`/sys/kernel/kexec_crash_loaded` 为0；此次未修改 kdump/pstore/驱动/功耗设置。
pstore 读取需要当前不可用的 sudo 认证，未读取；没有 crash dump 结论。

## 重启后的状态

- 无原生 CUDA 或本轮测试进程，未再次加载模型。
- `huihui-w4-vision-video` 仍停止，没有执行 27B 恢复。
- GPUs 各306MiB、200W；正式 FP24 二进制 SHA 未变。
- ComfyUI 两个容器自动运行；原先五个市场/Kafka 容器没有自动恢复，未对它们发启动或重启命令。
- `/mnt/nas` 和兼容路径均只读可用，仍指向同一 NAS source。
- Mac 外采已随 SSH 断开退出；断开的测试 SSH 客户端已结束。

证据：本地 `build/moe-prefill-20261011/incident/`、`incident-reboot/`、
`qualification/`；Z4 `/home/icy/q38-moe-prefill-20261011/`。
原有 FP24 已测性能详见[上一轮报告](prefill-exact-transport-20261011.md)，
本次故障不构成其长期稳定性合格证据。
