# 参考引擎复用本项目 W4/W8 的真机结果

2026-09-20，在 hp-z4-server 双 CMP 170HX 上，参考引擎已直接读取现有
`Qwen3.8-Flash-Next-q38-cut25`，成功运行 TP2 + EP + MTP6 + CUDA Graph。
PLE 全量驻留主机内存。没有重新量化权重，没有依赖 NVFP4 下载完成，没有使用 GDS。

## 实测

两边均为同机、同一现有量化产物、P2P、RAM PLE、单请求、temperature=0、seed=0。
提示为 `写个网站网页`。非思考提示均为 16 tokens，思考提示均为 56 tokens。
后台 NVFP4 下载在两边测试期间均持续运行。表中速率为
`completion_tokens / (最后一个非空 SSE 到达时间 - 第一个非空 SSE 到达时间)`。
原始证据同时保留 N−1 口径、首 token 延迟和端到端速率。

| 工作负载 | 本项目原生 TP plain/eager | 参考引擎 + 本项目 W4/W8 | 比值 |
|---|---:|---:|---:|
| 非思考，1,024 token，第 1 轮 | 30.701 tok/s | 165.175 tok/s | |
| 非思考，1,024 token，第 2 轮 | 30.780 tok/s | 182.755 tok/s | |
| 非思考，1,024 token，第 3 轮 | 30.788 tok/s | 171.352 tok/s | |
| 非思考三轮均值 | **30.757 tok/s** | **173.094 tok/s** | **5.63×** |
| 思考，2,048 token，单轮 | **29.003 tok/s** | **76.247 tok/s** | **2.63×** |

参考侧非思考三轮首 token 延迟为 72.9 / 87.0 / 93.0 ms；端到端速率为
163.252 / 179.955 / 168.722 tok/s。思考测试输出预算用尽时仍在 reasoning 中，
不能把它描述为已完成网页答案。所有计时请求均以 `finish_reason=length` 结束。

这是成功的生成吞吐对照，不是逐 token 等价或全面质量评估：参考侧三轮长文本
内容不同，内容哈希也不同。检查过的输出是正常中文介绍和 HTML/CSS；短语义检查
两轮均得到 `391`、多轮 `400`，并正确解释蓝光散射。没有证明所有请求均确定性一致。

这些数字属于**参考引擎上的适配版本**；本项目独立 C++/CUDA TP 引擎仍是原来的
plain/eager 实现，没有因此自动获得 MTP、CUDA Graph 或 173 tok/s。
本次没有逐项消融实验，不能把总收益精确分摊给 MTP、图执行、Marlin 或 EP。

## 权重与内存适配

- 主模型 routed expert 第 2–45 层保留 W4 G128；第 0、1、46、47 层及 MTP
  保留 W8 G128。常驻线性层和 LM head 使用原有 W8。
- 将有符号 W4/W8 packed bytes 无损转换为 GPTQ/Marlin 所需的 biased packed
  排列，保留 BF16 scales。加载时流式重排，不导出第二份完整 checkpoint。
- 共享专家 down projection 的 K=640 在 TP2 下分成 320。为保持组边界对齐，
  将每个 G128 scale 复制两次，得到数值相同的 G64 表示；没有重新计算 scale。
- token embedding 从原有 W8 还原为 BF16，供参考模型的 embedding 接口使用。
  主模型和 MTP 的 LM head 显式接入 AutoGPTQ/Marlin，保持 W8。
- 参考库原来在 GDS runner 中安装的独立计算钩子改由 RAM adapter 安装；PLE
  使用原生异步 CPU offload。独立 CPU 进程显式加载 Q38 reader。
- PLE 为一份 owned CPU RAM：51,200,245,760 字节 FP8 数据与 640,003,072 字节
  每行 BF16 scales，共 48.279994 GiB。128 块完整校验后关闭源文件，查表时应用
  每行 scale，再向两个 GPU 发送 BF16 行。
- 容器禁用 swap，限制 96 GiB 主机内存；监督进程在主机可用内存低于 16 GiB
  时停止该测试容器。此限制不代表所有环境都适合直接复用相同参数。
- 未启用参考库的二次 INT8 draft 筛选/top-32 重排钩子，也未启用要求该接口的
  local argmax reduction。MTP 本身正常启用，目标模型负责验证。

## 验证证据

现有产物的 150 个唯一 segment 完整 SHA256 校验通过。四个真实专家矩阵的
整数值、scale bits、重排前后解量化值全部一致；两张 GPU 上 W4/W8、M=1/7/32
共 24 个 Marlin 测试通过。M=1 与 BF16 解量化参考一致，多行最大 relative L2
约 0.00272。共享专家两个 G64 分区相加的独立实矩阵测试最大 relative L2
约 0.00287；这是 BF16 运算误差检查，不是整模型等价证明。

整模型日志确认主模型和 MTP 加载完成，每卡模型占用 37.72 GiB；两 rank 均连接
同一 PLE worker。图记录确认目标验证图为 7 tokens/request，并有草稿图。
重复三条语义请求期间，PLE 进程 `read_bytes` 保持 24,721,670,144 不变，
新增物理磁盘读取为 0；`VmSwap=0`。包含 IPC 等一般读取的 `rchar` 计数仍增加
160 字节，不能把进程所有读取计数都称作零。

参考版本固定为
[`d73d8824a35a21dccd71e279f1f4442ee741eedf`](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/tree/d73d8824a35a21dccd71e279f1f4442ee741eedf)
（v0.1.7），镜像 `qwen-flash-sm80:0.1.7-own`。
适配源码只在独立镜像中；没有将外部推理实现复制进本项目的 C++/CUDA 引擎。

本地完整工作目录为 `../.artifacts/q38-reference-run-20260920`（相对本仓库根目录）：

- `own-adapter/`、`ram-adapter/`：适配源码和 Dockerfile。
- `prepare-own-view.py`：生成模型配置/tokenizer 的轻量视图。
- `launch-reference-ram.py`：隔离启动、内存检查及监督。
- `q38-own-before-reference-benchmark-20260920/`：原生引擎对照。
- `q38-reference-own-benchmark-20260920/`：参考侧原始 SSE、请求、输出和汇总。
- `q38-reference-own-smoke-20260920/`、`q38-reference-own-ram-io-smoke-20260920/`：语义检查。
- `own-evidence/`：镜像与容器信息、环境、日志、图记录和 PLE I/O 证据。

远端保留容器 `q38-reference-own-20260920`，监听配置为 `127.0.0.1:18420`。
其证据目录为 `/home/icy/q38-reference-own-evidence-20260920`；前三轮失败启动
分别以 `-attempt1`、`-attempt2`、`-attempt3` 保留，记录了修复过程。
再次测试须先释放两张 GPU，确保主机内存充足，并归档旧容器/证据后使用启动脚本。
测试结束后使用 `restore-own.sh` 恢复 30001 的原有 P2P + RAM 服务，并执行完整 hash 校验。

收尾已确认：参考容器正常退出（code 0）；原有服务完整校验、RAM 加载完成，
`/healthz` 返回 ok，算术答案为 391。原有非流式响应保留 `<|im_end|>` 结束标记，
原始响应记录在 `own-restoration-verified.json`；本次没有修改原生 API 序列化行为。

NVFP4 继续后台下载到单独的 noswap tmpfs，完成后脚本会进行完整文件校验。
收尾时下载曾因 Xet 网络请求错误退出，现已恢复并加入最多 24 次退避重试。
完整校验标记尚未生成，不能视为下载完成。本次成绩没有使用该 checkpoint，
也不是原版 NVFP4/GDS 部署的复现成绩。
