#ifndef Q38_CUDA_MOE_H
#define Q38_CUDA_MOE_H

#include "q38/cuda_weights.h"

#include <cstdint>

namespace q38 {

constexpr std::uint32_t kQ38MoeExperts = 512;
constexpr std::uint32_t kQ38MoeTopK = 10;
constexpr std::uint32_t kQ38MoeIntermediate = 640;

// gate_up_scratch: tokens*top_k*1280 BF16
// activated_scratch: tokens*top_k*640 BF16
// accumulation_scratch: tokens*2560 FP32
void cuda_moe_routed_bf16(const CudaTensorViewV1& gate_up_experts,
                          const CudaTensorViewV1& down_experts,
                          const std::uint16_t* hidden,
                          const std::int32_t* expert_ids,
                          const float* expert_weights,
                          std::uint32_t tokens,
                          std::uint32_t top_k,
                          std::uint16_t* gate_up_scratch,
                          std::uint16_t* activated_scratch,
                          float* accumulation_scratch,
                          std::uint16_t* routed_output,
                          void* stream,
                          int device);

void cuda_moe_combine_shared_bf16(const std::uint16_t* routed_output,
                                  const std::uint16_t* shared_output,
                                  const std::uint16_t* shared_gate,
                                  std::uint16_t* output,
                                  std::uint32_t tokens,
                                  void* stream,
                                  int device);

bool cuda_q38_moe_compiled();

}  // namespace q38

#endif
