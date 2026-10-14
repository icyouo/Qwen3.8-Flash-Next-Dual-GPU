# 固定 MTP4 的集中改造与一次集成验收计划

状态：实施计划，尚未表示以下改造已完成。2026-09-21。

本计划按用户最新要求收敛原有优化计划：固定 4 个草稿 token，以完整候选版本
交付；取消宽度扫描、自适应策略测速和无穷尽的逐项性能消融。模块内检查随开发
完成，真实模型只做必要的统一正确性验收与整包性能对照。已经启动的 head-pack
固定 MTP4 任务自然收尾并恢复服务，不再追加同类小实验。

## 一、固定条件与现有证据

- 双 CMP170HX，P2P，现有 W4/W8 权重。主要执行模式为 RAM PLE；SSD PLE
  继续作为可用加载模式。MTP4 表示 4 次串行草稿、最多 5 行 target 验证，
  不意味着每轮一定发布 5 个 token。
- 固定参考提交 `d73d8824a35a21dccd71e279f1f4442ee741eedf`。本机适配参考库使用
  同一 W4/W8 和 RAM PLE：Graph+MTP4 非思考均值 162.8255、思考 93.6676 tok/s；
  仅 Graph 为 78.2997/78.0777，均关闭为 7.7116/7.4250。
  来源：Homelab 根目录 `.artifacts/q38-reference-ablation-v2-20260921/comparison.json`。
  这些请求都到达输出上限，不能当作完整回答质量验收。
- 原生最近固定 MTP4 的短提示约 57–69 tok/s，512 输入约 40、8192 输入约 56。
  这是诊断 RPC 的不同任务，不能直接与上述参考 HTTP 数值算公平倍数。
- 32.22 GiB 闲置 NVFP4 下载已校验迁出 tmpfs；head-sync/shared-MoE 两轮进程
  VmSwap 均为 0。极少数请求出现系统其他进程换入，仍按记录排除其性能归因。
- head-sync 未显示明显收益；shared-A MoE 首轮有效配对约慢 5%–6%，不纳入
  默认组合。此前融合和 KV-only 的时间因进程换页作废，正确性证据仍可复用。

参考数值是追赶目标，不承诺改完必达 163 tok/s。实际平均发布数为 p 时，追平
非思考参考需要整轮约 `1000*p/162.8255 ms`；必须计入 draft、PLE、verify、commit。

## 二、参考库机制与本次采用方式

| 参考机制 | 原生现状 | 集中改造 |
|---|---|---|
| FULL_AND_PIECEWISE 图执行、草稿图与 GDN 融合 | target 已有分段图；每次 collective 会结束 GPU 片段，回放调用 CPU wait_peer；当前 TP 测试草稿图关闭 | 建立两 rank 的静态执行计划，让 P2P 依赖成为图的一部分；补齐 TP 草稿及 5 行验证的覆盖 |
| TEP2 与 P2P | routed experts 已按专家分区，常驻 worker/异步 P2P 已有 | 复用现有分区与算术，移走热路径逐 collective 的 mutex/CV 配对 |
| PLE 多行规划、固定槽位和异步交接 | RAM/SSD 与批量接口已有 | 固定 5 行规划、复用 pinned staging、提前 H2D；每个槽位记录 generation 和消费事件 |
| HC GEMV、输入投影融合 | 原生已有 HC/PLE/投影候选，部分计时失效 | 复用已有实现；按 M=1/M=5 整理固定调度，避免重新造一组相同开关 |
| Marlin 等量化矩阵计算 | 原生 grouped MoE 已接通，量化格式和内核不同 | 针对现有 W4/W8/G128 做 M=5 的权重读取复用与融合，保持缩放和归并边界 |
| 全词表 INT8 草稿打分后 top-32 BF16 复核 | 原生输出头已为 W8，目标是保持现有输出语义 | 保持完整词表；优先合并 5 行规约与精确 argmax，不引入有限候选筛选 |

参考的 GDS 是存储路径选择；本次 RAM 推理继续从内存查表。参考配置中的 MTP6
改为固定 MTP4。参考历史优化百分比来自不同条件，不相加预测原生收益。

## 三、一次交付包的代码工作

### 1. 两 rank 联合图执行与通信计划：最高优先级

修改 `cuda_te_graph.cu`、`cuda_te.cu`、`cuda_tp_backend.cc` 及对应头文件。

- 初始化时将两 rank 的 collective 按 group、顺序、类型、元素数配对，冻结
  输入/输出/scratch 地址。图回放不再逐次经过 `wait_peer()`。
- 复用既有 peer 读取和 BF16/FP32 加法内核，将 producer-ready → peer-read →
  read-done → 写回建立为显式 GPU 依赖。优先用联合多设备图的依赖表达，避免
  将可复用旧 event 简单跨图连接而误读上一轮状态。独立传输 fixture 先验证可行性。
- 以状态 bank、执行形状和真实指针身份缓存计划。预分配 graph/scratch，图的
  生命周期覆盖全部资源；缓存销毁在 allocation 释放之前。
- 主机负责每个执行阶段的提交、异常和发布。保留取消检查边界；不将每层 CPU
  collective 包进 graph host callback 来假装去掉调度。
- 同时记录 GPU 图片段数、主机 collective 配对数及 graph fallback，验收必须
  能证明稳定回放经过新路径。单纯有 graph replay 计数不够。

这不是把现有 `cudaStreamBeginCapture` 范围放大。旧实现的 CPU 配对、动态分配
和输入生存期必须一起迁移。CUDA 支持图依赖和外部事件不等于本机此实现已通过。
若传输 fixture 发现图构建方案不可用，修复此实现再进入模型验收；不回到大量
微优化测速来替代主线。

### 2. 固定 MTP4 的执行与状态提交

修改 `cuda_backend.cu`、`cuda_tp_backend.cc`、`executor.cc` 和事务状态相关接口。

- 对草稿 M=1 和 target M=5 建立稳定工作区与图；草稿 4 步顺序固定，token、
  position、QSA 有效长度通过设备元数据更新，避免因输入值变化重建图。
- 4 步草稿均保持前一步 token/HC 依赖。读取草稿 IDs 后进行 CPU PLE 规划，
  再提交 target 验证；不声称有 CPU RAM 查表的整轮只需一次 GPU launch。
- 把已有前缀快照、最长接受前缀、QSA/GDN/PLE/HC 状态恢复组织成统一提交路径。
  只修复被接受状态确实需要的 QSA 缓存，整合已有 KV-only 候选，避免重算无用
  attention/MoE 输出。正确性不通过的分支留在原提交实现，不强行启用。
- 上下文尾部不足 5 行、EOS、普通采样沿用正确的边界路径；不再为其他 MTP
  宽度开发优化或做吞吐扫描。接受长度 0..4 的检查属于 MTP4 正确性。

### 3. 消除 TP 的 stage 中间主机往返

修改 `cuda_transport.cu`、`cuda_transport.h`、`cuda_tp_backend.cc` 及 boundary 接口。

- TP 两个 logical stage 在同一进程内为每个 rank 传递设备 lease，复用本 rank
  已有 HC，不再为了 PP 风格边界回读到 CPU、再上传 GPU。
- lease 包含 device、epoch、generation、范围及 producer/consumer 完成关系。
  两 rank 一致性使用设备端校验并保留主机可核验结果，不能删掉校验后直接传裸指针。
- 保留 PP 的传输实现；诊断导出按需回读，正常生成不回读完整中间张量。

### 4. PLE 的 RAM/SSD 共用流水线

修改 `ple.cc`、`ple_backend.cc`、`cuda_backend.cu` 及相关接口。

- 每轮将已知的 pending+4 草稿统一规划，复用行号/scale 缓冲及去重信息；使用
  正确的逐前缀历史，不对未知未来 token 预取，也不提前提交被拒绝的历史。
- RAM owner 保持一份完整只读表，填充固定 pinned staging；两 rank 的小批量
  H2D 使用专用 copy stream，消费图等待对应 generation 就绪。
- 固定缓冲槽循环使用，每槽在两个 rank 都完成读取后才可覆写；重置及取消
  先 drain，禁止把上一请求的 ready 标志误作本次就绪。
- SSD 保留现有读取后端，接入同一 staging/lease 合约。本轮不增加 GDS 依赖。

### 5. M=5 算子与输出头

修改 `q38_moe_prefill.cu`、`q38_kernels.cu`、`q38_hyper.cu`、`q38_ple.cu` 和后端调度。

- 针对 M=5 明确 W4/W8 dispatch、尾部掩码、专家空路由与 rank 内 FP32 partial
  布局；利用现有小矩阵路径复用量化权重读取，优先保持原缩放和归并顺序。
- HC、QKVZ/BA、PLE norm/projection/residual 使用已有融合候选；计算顺序变化
  独立记录，不能将“同一权重”误作逐位等价证明。
- 输出头合并 5 行 allreduce，保留全词表和原 argmax tie-break；复用正在运行
  的 head-pack 验证结果。该改动减少通信提交次数，不减少规约数据字节数。
- shared-A 候选保持关闭。改变 output 的分片轴会改变累加顺序和 artifact view，
  本交付包先不做这一额外重构；草稿 INT8 top-32 也不进入本包。

### 6. 单一配置、工具链和交付入口

修改 `q38_tp_launch.py`、runtime 参数/identity、构建配置和统一验收入口。

- 提供一个明确的固定 MTP4 组合配置，展开后的开关、权重、源码、编译器/CUDA、
  二进制身份全部落盘；消除互相冲突的隐式环境变量。旧组合保留为回退配置。
- 开发可以使用模块内部开关定位问题，用户只需选择 PLE RAM/SSD 和正常运行配置。
- 固定工具链；刚修复的 CUDA 13.3 `_NV_RSQRT_SPECIFIER` 兼容识别纳入构建。
- 使用已有有界并行完整 SHA256 校验缩短启动等待，仍逐文件验证，不跳过校验。
  decode 测速前不得编译、搬运大文件或加载第二份模型；记录进程驻留情况。

## 四、开发顺序与精简验收

开发依赖：冻结配置和工具链 → 联合图/P2P 合约 → MTP4/设备 boundary/PLE →
M=5 调度与输出头 → 统一配置和验收。可在代码层分提交，交付一个完整候选版本。
不因每个小函数变化就重启模型、跑一套全量对照。

1. 编译与模块检查：集中跑涉及到的 CUDA 算子/通信 fixtures、事务与配置测试。
   重点覆盖图回放 changing-input、旧 generation、scratch 复用和跨 rank 顺序。
2. 模型正确性一轮：复用现成的 MTP4 0..4 接受前缀与拒绝后续跑，检查完整 logits、
   QSA/GDN/PLE/HC 提交位置及双 rank 一致；包含 EOS、取消→重置→下一请求、
   上下文尾部、RAM/SSD 同输出和普通采样回退。复用已有短回答集合验最终答案。
   纯调度改动要求逐位一致；新算术路径与既有批量/逐行基线的误差分别报告，未
   解释的质量退化不能随整包上线。262144 上限契约不因缩减测速矩阵而被省略。
3. 性能仅原生基线和整包候选：相同 HTTP、模板、greedy、MTP4、输出预算，关闭
   profile。保留网页非思考 1024、网页思考 2048、固定代码及 8K 上下文四个案例，
   各预热一次、计时两次。只在失败、换页污染或差异不足以判断时针对该项重跑。
   参考网页两项复用已有固定 MTP4 结果；不再跑 Graph×MTP 或宽度矩阵。
4. 汇总 tok/s、TTFT、平均发布数、峰值内存及图回放/回退；必要时只做一次代表性
   profile 定位剩余差距。测试完成一次恢复/部署，保存旧二进制和完整回退命令。

一次交付表示集中实现、统一验收；编译或正确性失败仍需要修复与针对性复核。
不会把未经验证的多项改动直接替换服务，也不承诺第一版代码无需修改。

## 参考依据

- [参考启动配置（固定提交）](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/blob/d73d8824a35a21dccd71e279f1f4442ee741eedf/config/vllm-args.json)
- [参考优化与数据边界](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX#已集成优化)
- [参考 PLE 规划与异步生命周期](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/blob/d73d8824a35a21dccd71e279f1f4442ee741eedf/src/vllm_ple_gds_runner.py)
- [CUDA Graph API 与更新约束](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA Graph 外部事件约束](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html)
