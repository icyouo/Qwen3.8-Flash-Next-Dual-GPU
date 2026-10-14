Pinned vLLM v0.12.0 Marlin MoE (Apache-2.0), framework-free adaptation.
Upstream: https://github.com/vllm-project/vllm/tree/v0.12.0/csrc/moe/marlin_moe_wna16
ATen includes/checks are replaced by standard C++ in shared helpers. Full-K CTA grid bounds avoid inter-CTA reduction and preserve independent expert output.

Native dispatch change: force iters=k_tiles because the CUDA-graph grid is an upper bound with EP padding; this removes split-K/global-lock use even with many unowned/empty expert blocks.
