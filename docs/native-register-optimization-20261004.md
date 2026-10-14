# 固定 MTP4 寄存器 MMA 与批量输出头

2026-10-04。双 CMP170HX / 每卡 200W / P2P / 现有 W4/W8 / RAM PLE / MTP4。
**目标仍为超过 160 tok/s；这一版只达到约 81 tok/s。**

同一二进制、同一请求、每项预热后两次 HTTP/SSE decode 测量：

| 请求 | 路由版 tok/s | 寄存器版 tok/s |
|---|---:|---:|
| 网页，16 输入 / 1024 输出 | 73.97 | 81.15 |
| 代码，24 输入 / 512 输出 | 74.33 | 81.74 |
| 8296 输入 / 512 输出 | 47.33 | 51.34 |

保持原来的量化数值、FP32 G128 scale 累加和 BF16 激活边界。MoE 使用 SM80
m16n8k16 寄存器 MMA，减少 WMMA 共享内存往返。输出头的小批量 GEMV 复用
权重读取，跨卡完整词表汇总仍沿用原实现。两项独立开关为
`Q38_CUDA_MOE_REG_MMA=1`、`Q38_TP_COMPACT_HEAD_BATCH=1`。

108 项 MoE fixture 和 8 项批量 GEMV eager/graph fixture 逐位一致。
整模型五类输入的 token / 完整 BF16 logits、十个强制拒绝前缀、九个完整质量
回答、三类取消/超时/reset 恢复均通过。HTTP 内容和长度与对照一致。计时期间
VmSwap=0，无 major fault 增长，无 graph fallback。旧配置保留，测试未恢复挖矿。

可选生产二进制：`build/native-mtp4-register/q38-cuda-runtime`。
SHA256：`eff3a736477a3b9f414c17a59f381a23bfa86bfb3ebc825623ac5f146c0ed3ca`。
启动方式沿用 [路由版说明](native-route-optimization-20261004.md)，将 runtime 路径
替换为上述路径，execution profile 改为 `mtp4-register-candidate`，state-dir 使用
独立目录，重新生成身份。原始证据位于 Homelab 根目录
`.artifacts/q38-native-target160-20261004/qualified-reg-head/`；服务器镜像位于
`/home/icy/q38-native-target160-20261004/qualified-reg-head/`。

后续 Marlin 是独立的数值后端候选，尚未作为本版默认，也不能用局部算子加速
推算整模型吞吐。CMP 不支持 CUPTI tracing；细分耗时使用保留计算图的外部
CUDA event 节点，正式测速关闭事件诊断。
