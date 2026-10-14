# 2026-09-06 prefill / decode 优化实现

本次完成代码实现和本地回归，尚未进行双 CMP 170HX 数值、显存或性能验证。所有新增 GPU 优化路径均通过显式开关启用；历史 README 性能数字不能作为本次改动的收益。CPU 侧 PLE 预取重构和 sampler tie-break 明确化属于默认代码变更。

## 源码基线

编辑目录原先是较旧的源码快照，没有 `.git`。本次先保存原文件，再从本地完整 Git 对象库恢复到 `ce412457027d5c4e8be6882253b86207c5c0656c`，最后叠加以下优化。

原始源码备份及逐文件导入记录位于：

- `/Users/icy/Homelab/.artifacts/q38-local-optimization-20260906/optimization-baseline/original-source.tar.gz`
- 同目录 `import-manifest.json`。

恢复的基线已经包含 grouped-MMQ Tensor Core MoE prefill、resident/partitioned GDN、tiled exact QSA 与 radix selection、优化 router / decode MoE、双 bank piecewise graph，以及 retained width-N MTP 的提前拒绝、在线前缀 checkpoint 和免 replay reconcile。这些不是本次重新实现的功能。

## 新增实现

| 路径 | 实现 | 启用方式 / 默认值 |
|---|---|---|
| Decode 多投影 | GDN qkv/z/b/a、QSA q/k/v/index、shared expert gate/up 合并 kernel launch；支持量化与保留精度矩阵，保留单投影回退 | `Q38_CUDA_FUSED_PROJECTIONS=1`，默认关闭 |
| Speculative target graph | 动态输入输出移出 graph，固定层体可复用于逐 token target 验证；按工作 bank 保存 graph | `--enable-piecewise-decode-graph`，默认关闭 |
| MTP draft/reconcile graph | 稳定 scratch 输入、输出；捕获前后静态片段，位置相关 QSA eager；捕获失败仅回退当前片段 | `Q38_CUDA_MTP_GRAPH=1`，需同时启用 MTP，默认关闭 |
| Adaptive MTP | 从已提交事务的接受深度、draft/target/wall 时间估计宽度，有限探索；reset 清空学习状态 | `--enable-mtp --enable-adaptive-mtp`，默认关闭；宽度不超过请求上限 |
| Compact sampling | GPU 精确排序取 top-k，只传 K 个 logits 和 token IDs；CPU 保留 top-p、RNG 与提交语义 | `--enable-compact-sampling`，默认关闭 |
| Prefill compute tile | 计算 tile 和 PLE I/O tile 解耦 | `Q38_CUDA_PREFILL_TILE`，默认 512，范围 1..512 |
| PLE I/O tile | 两个有界 ready tile；支持跨 tile 消费、取消后复用和错误传播 | `Q38_PLE_IO_TILE`，默认 512，范围 1..4096 tokens |
| Prefill sync window | 减少逐 tile stream synchronize；窗口末端、chunk 末端同步，保持取消检查 | `Q38_CUDA_PREFILL_SYNC_TILES`，默认 1，范围 1..8 |
| PLE 上传 | 独立 H2D stream/event，计算 stream 等待数据 ready | `Q38_CUDA_PLE_COPY_STREAM=1`，默认关闭 |
| Boundary D2H | 独立 copy stream；先 D2D 快照到租用 slot，避免源 buffer 重用污染；保留双端 checksum | `Q38_CUDA_ASYNC_BOUNDARY=1`，默认关闭 |
| Prefill cache | 启动时预热常用投影并封存 cache，避免运行中 miss 分配；保留容量与显存余量限制 | `Q38_CUDA_PREFILL_PREWARM=1`，默认关闭 |

Compact sampling 仅用于 `top-k-top-p`、top-k > 0、无 repetition/frequency/presence penalties 的配置。诊断 full logits、greedy 和其他不适用配置保持完整路径。相同 logit 的顺序明确为 token ID 升序；之前的 host `std::sort` 没有规定 tie 的次序，因此并不承诺与旧版本在 ties 时保持同一个随机采样输出。新 full/compact 路径使用相同顺序。

Adaptive MTP 只影响性能调度，不改变验证与提交规则；采样模式仍遵循现有 target-only fallback，A/B 脚本拒绝把非 greedy 配置当作 MTP 性能测试。

独立 boundary stream 增加有界 device slot 内存，并多一次 D2D；同步窗口改变取消响应上界；预热将部分初始化成本移到启动。这些都需实测收益，不能仅凭减少 launch/sync 数量就默认开启。

## 本地验证

- macOS：C++ runtime regression 通过，Python **63/63** 通过。
- Linux ARM：C++ runtime regression 通过，Python **63/63** 通过；容器允许 io_uring syscall。
- Linux ASan + UBSan（含 leak detection）C++ 回归通过，日志 `build/linux-asan-tests.log`。macOS ASan 在进入 main 前卡于 sanitizer 初始化，已终止；不记作通过。
- CUDA 12.8.1 Linux ARM：`sm_80` runtime、compile-check、GPU fixtures 完成编译链接。
- GPU fixtures 实际执行返回 `skipped (no GPU)`，不计为数值验证通过。
- 测试新增覆盖：adaptive 宽度/重置与 mock token parity；compact/full host RNG parity；PLE 非对齐切分、取消和错误；固定输出预算 A/B harness。GPU fixtures 新增 fused projection bitwise 比较、compact top-k 特殊值/边界和异步 boundary 源覆盖校验。
- 同时修复原 GPU fixture RMSNorm 输出 buffer 分配不足，以及 macOS snapshot sync 兼容性；Makefile 现在跟踪头文件依赖。

日志在 `build/local-tests.log`、`build/linux-tests.log`、`build/cuda-compile.log`。本地 Linux ARM 可执行文件不能部署到 x86_64 真机。CUDA 12.8 编译不替代 CUDA 13.1 验证。

本地复现 CUDA 编译可使用 `nvidia/cuda:12.8.1-devel-ubuntu24.04` 容器，安装 python3 后执行：

```sh
make -j2 cuda-runtime cuda-check build/cuda12/q38_cuda_kernel_tests \
  BUILD_DIR=build/cuda12 CUDA_HOME=/usr/local/cuda CUDA_COMPAT=0 \
  CUDA_HOST_CXX=g++ PYTHON=python3
```

CUDA 13.1 真机继续使用默认 compat 设置和独立 build 目录；不要混用不同 toolkit 生成的 compat headers。

## 下一轮真机验证入口

先按现有 artifact/identity 流程为当前源码生成匹配的 READY，保持量化 artifact、tokenizer、sampling config 一致。`validate_ready` 的 source/identity 检查必须通过，不绕过校验。

1. 真机重新编译：`make -j2 test cuda-check cuda-runtime cuda-test`。确认 GPU fixtures 没有 skip，并使用 `compute-sanitizer --tool memcheck ./build/q38_cuda_kernel_tests` 检查越界。
2. 用同一 token fixture 分别测试 baseline/fused/graph/tile128/transport/prefill，再测试 greedy 的 mtp-fixed/mtp-adaptive/mtp-graph/combined。
3. sampled compact 使用单独的匹配 READY，测试 `--cases baseline compact`。固定 RNG seed，覆盖 top-k/top-p/ties，并测试 penalties/diagnostics 的完整路径。
4. 按 README 的 `q38_logit_parity.py` 流程比较 eager/graph 原始 logits；不能只凭 token 一致认定数值一致。
5. 覆盖两个 state bank、reset、取消、回滚、接受/拒绝位置和 retained reconcile；检查 graph fallback、显存增长、PLE 错误、frontier 和 rollback counters。
6. 在 4K、32K、128K、262080 上重复运行；严格 262080+64 使用现有 strict gate。每组多次独立进程冷启动，并另做 warm decode；报告分布，不能把一次短样本当作最终收益。

新脚本默认只输出计划，不启动 GPU runtime：

```sh
python3 tools/q38_optimization_gate.py \
  --ready /path/to/current/READY.json \
  --tokens-json /path/to/tokens.json \
  --output /path/to/evidence \
  --cases baseline fused graph tile128 transport prefill mtp-fixed mtp-adaptive mtp-graph combined
```

实际执行时追加 `--run`。每个 case 使用独立进程、独立 socket；清除继承的优化环境变量。输出固定为 64 tokens（包含 seed，EOS 提前停止），decode 计时不含 seed，MTP 按剩余输出预算限制宽度。报告 token parity、prefill、TTFT、decode throughput、事务延迟分位数和每事务 token 数，以及 runtime/fixture hash 和完整 stats。token parity 不一致返回非零。

本次没有加入近似 QSA、改变历史检索语义或替换 GDN 数学算法。新路径是否默认启用，留到上述真机门禁通过后根据实测决定。

相对恢复基线的审阅补丁与逐文件 hash：`build/optimization-changes.patch`、`build/optimization-manifest.json`。
