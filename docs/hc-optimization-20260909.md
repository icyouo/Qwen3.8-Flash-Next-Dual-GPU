# HC / PLE 独立优化候选（2026-09-09）

本轮只实现能以原路径逐位一致性为门槛的 HC 优化。外部项目仅用于发现思路，不作为正确性依据，也未移植其实现。以本项目原 CUDA GEMV、HC elementwise kernel 的运算为参考。未运行真机；默认关闭，不宣称质量或性能已经验证。

## 实现

- `Q38_CUDA_HC_GEMV=1`：两行输出在同一 block 内复用输入值；每一行仍使用 256 个逻辑线程的列分配和 FP32 累加表达式，保留 warp 内及 warp 间两层归约树。
- `Q38_CUDA_HC_FUSED=1`：在上述 kernel 中融合 HC 的 SiLU/sigmoid。显式执行投影结果的 BF16 舍入，再按原 divisor/scale 计算激活，最终仍输出 BF16。开启此项已包含专用 GEMV，不必同时设置前项。
- 支持 batch=1、preserved BF16、输入宽度 320/10240；覆盖 base/MTP/final HC。FP32、量化矩阵、其他宽度及 tokens>1 的 prefill 批量投影保持原调用路径。
- 不增加权重副本，不修改 artifact、量化策略、采样、MTP 接受规则或事务状态。没有引入近似 exp、fast-math 或更低精度累加。

主要源码：`cuda/q38_kernels.cu`、`cuda/cuda_backend.cu`、`include/q38/cuda_kernels.h`。专用 API 显式返回是否支持当前布局，非支持布局不提交 kernel。

## 正确性门槛

`cuda/cuda_kernel_tests.cu::test_hc_gemv` 将新路径与本项目原 GPU kernel 对照，要求输出 BF16 words **逐位一致**，不使用宽松误差阈值。覆盖：

- 10240×320、320×10240、4×10240 等实际形状，以及单行和奇数行尾部。
- 原投影、SiLU/divisor=4、sigmoid/divisor=1/scale=1、sigmoid/divisor=4/scale=2。
- 随机值、混合指数和符号、正负零、BF16 极小值。
- eager 与 CUDA Graph 重放，在固定指针下更换输入；输出尾部 canary；不支持布局的拒绝。

这些测试是门禁，不是已经通过的 GPU 证明。即使算法保留求和顺序，也须验证实际编译代码；任何逐位不一致都需先修复，不通过扩大 tolerance 放行。

真机后续还必须完成 HC 中间输出及目标 raw logits 对比、固定 greedy/sampled 输出对照、两套 working bank、reset/cancel/rollback，以及端到端延迟/吞吐和显存检查。只有数值门禁通过且速度实测改善，才考虑默认开启。

## A/B

为当前源码生成匹配 READY，重新编译真机 runtime 和 CUDA fixtures 后执行：

```sh
make -j2 test cuda-check cuda-runtime cuda-test
compute-sanitizer --tool memcheck ./build/q38_cuda_kernel_tests
compute-sanitizer --tool racecheck ./build/q38_cuda_kernel_tests
python3 tools/q38_optimization_gate.py \
  --ready /path/to/current/READY.json \
  --tokens-json /path/to/tokens.json \
  --output /path/to/hc-evidence \
  --cases baseline hc hc-fused --run
```

不加 `--run` 只生成计划。HC 两个 case 独立隔离环境，保留固定输出预算。eager/graph 原始 logits 对比继续使用 README 中 `q38_logit_parity.py` 流程；不能只凭 token parity 放行。

## 本轮范围

本轮落实 HC GEMV 和保留中间舍入的激活融合。多 token target 验证、sampled MTP 涉及独立的事务/概率语义，尚未实现；低精度草稿筛选、近似候选裁剪已排除。既有 prefill、传输优化仍按 2026-09-06 清单待真机验证。本报告不代表所有可能的优化已经完成。

## 本地验证记录

- macOS C++ runtime regression 通过：`build/hc-cpu-tests.log`。
- Python 63/63 通过：`build/hc-python-tests.log`。初次沙箱内运行的三项 HTTP 测试因禁止绑定回环端口失败，放行本机端口操作后通过。
- A/B 默认只生成计划的路径检查通过：`build/hc-ab-plan.json`，没有启动 runtime。
- CUDA 12.8 / Linux ARM / SM80：完整 runtime、compile-check、新增 GPU fixtures 均已编译链接通过。日志为 `build/hc-cuda-compile.log`、`build/hc-cuda-fixtures.log`；首次 fixture 链接发现 Makefile 缺少原 `q38_hyper.cu`，补齐后通过。
- GPU fixtures 执行明确返回 `skipped (no GPU)`，不计为逐位一致性通过。上述 ARM 主机二进制不用于部署 x86_64 真机，真机仍需 CUDA 13.1 重新编译。
- 完整开发镜像解包曾因 Docker 虚拟盘空间不足失败，后改用官方 base 镜像加必要依赖完成编译；临时编译容器退出后自动删除。没有清理用户其他容器、镜像或卷。
- 源文件 hash 清单：`build/hc-manifest.json`。

## 第二批：HC / PLE 独立投影合并提交

新增两个默认关闭的开关：

- `Q38_CUDA_HC_GROUPED=1`：将共用归一化输入的 mix_down+SiLU 与 inject+sigmoid 合并成一次 kernel launch；仍使用上一批每行的原归约方式与显式 BF16 中间舍入。grid 按两个矩阵实际行数紧凑拼接，不把四行 inject 填充到 mix_down 的网格大小。mix_up 和后续 mix 保持依赖顺序；没有 inject 的 HC、tokens>1 或不支持的布局继续走原路径。与 `Q38_CUDA_HC_FUSED=1` 可组合，使 mix_up 也保留上一批融合。
- `Q38_CUDA_PLE_FUSED=1`：单 token 的 PLE key/value 共用原多投影 kernel 入口。继续使用原 W8/W4/BF16/FP32 数据和对应算术；tokens>1 的 prefill GEMM 不变，没有修改 PLE 行号、查表、I/O 或卷积状态。

实现前核对了 workspace：hyper_norm、low_rank、injection、mix_weights、mixed 使用独立分配，提前计算 inject 不会被 mix_up/mix 覆盖。合并 API 在提交前验证两个布局与输出，拒绝输入/输出或两个输出的范围重叠；不支持第二个矩阵时不会提前提交第一个。

同时修复多投影 API 的模式一致性：显式 `Q38_CUDA_DECODE_GEMV=scalar` 时回退到原 scalar GEMV，不再暗中调用 optimized quantized reduction。默认 optimized 模式保持原行为。

GPU 门禁增加了 HC 两矩阵原 kernel 对照（实际及奇数尾部、不同值域、eager/graph、失败无部分提交、alias 拒绝）；多投影测试覆盖 PLE 的 2560→10240 尺寸。必须在真机分别运行默认与 scalar 两个进程：

```sh
./build/q38_cuda_kernel_tests
Q38_CUDA_DECODE_GEMV=scalar ./build/q38_cuda_kernel_tests
```

A/B 新增 `hc-grouped`、`ple-fused`、`hc-ple`。`hc-grouped` case 同时开启 HC_FUSED，便于和 `hc-fused` 比较合并提交的增量；`hc-ple` 组合两项。示例：

```sh
python3 tools/q38_optimization_gate.py   --ready /path/to/current/READY.json --tokens-json /path/to/tokens.json   --output /path/to/grouped-evidence   --cases baseline hc-fused hc-grouped ple-fused hc-ple --run
```

这些新增项仍需 GPU 逐位、端到端 logits/质量以及性能验证，未改变默认启用策略。

第二批本地检查结果：C++ runtime 回归通过，Python 63/63 通过，CUDA 12.8/Linux ARM 的 SM80 runtime、compile-check、GPU fixtures 编译链接通过。默认和 scalar 两次 fixtures 执行均返回 `skipped (no GPU)`，不计为数值验证通过。日志：`build/grouped-cpu-tests.log`、`build/grouped-python-tests.log`、`build/grouped-cuda-compile.log`、`build/grouped-cuda-fixtures.log`；计划检查结果：`build/grouped-ab-plan.json`。

## 第三批：PLE 归一化与 residual 融合

这两项同时支持 decode 与 prefill（计算 tile 1..512 tokens），仍默认关闭：

| 开关 | 改动 | 数值约束 |
|---|---|---|
| `Q38_CUDA_PLE_NORM_PAIR=1` | key/query 两路 group RMSNorm 在一次 launch 中按 grid.z 分别执行 | 保留每一路原 256 线程列分配、平方和归约、epsilon、rsqrt 和 one-centered weight 运算 |
| `Q38_CUDA_PLE_RESIDUAL=1` | 卷积输出与随后 residual add 合并 | 保留卷积结果原 BF16 舍入后再做 residual 加法，状态更新顺序不变 |

PLE 两路输入、norm weights 和输出的实际 workspace 已核对独立。paired norm API 拒绝写入范围与任一路输入、weights 或另一输出重叠。卷积融合支持精确原地 residual 写回，拒绝部分重叠以及 state/weights/其他输入与输出重叠。没有新增权重副本、量化、近似激活或状态 bank。

原入口仍可用。归一化共用相同设备函数；卷积通过 compile-time `WithResidual=false/true` 分别实例化，以保留不启用融合时的原运算。详细 prefill profiling 开启融合后记录 `ple_conv_residual`，避免将合并时长错误归入旧独立阶段。

新增 GPU fixtures 覆盖 1、3、32、512 tokens：

- 两路 norm 与原两次调用逐位比较，覆盖多值域、正负零和极小值；固定指针、更换输入的 graph 重放。
- 卷积融合与原 conv+add 逐位比较输出，且在连续四次调用后逐次比较**完整卷积状态**。
- eager/graph、非零初始状态、residual 抵消、精确原地写回、非法重叠；tokens=1 同时对照原 decode 和原 prefill 入口。

真机 A/B 使用 `--cases baseline ple-norm ple-residual ple-ops`，其中 `ple-ops` 只组合这两项，以隔离上一批投影融合的影响。仍须完成 raw logits、完整生成质量、reset/cancel/rollback、内存 sanitizer 和实际性能门禁后才讨论启用。

第三批本地结果：C++ runtime 和 Python 63/63 通过；CUDA 12.8/Linux ARM/SM80 的完整 runtime、compile-check 与最终 GPU fixtures 编译链接通过。GPU fixtures 执行返回 `skipped (no GPU)`，逐位输出/状态及质量、性能均待真机验证。日志：`build/ple-ops-tests.log`、`build/ple-ops-cuda-compile.log`、`build/ple-ops-cuda-fixtures.log`；计划检查：`build/ple-ops-ab-plan.json`。源文件 hash：`build/ple-ops-manifest.json`。
