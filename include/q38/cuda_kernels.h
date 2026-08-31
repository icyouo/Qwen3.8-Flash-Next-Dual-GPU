#ifndef Q38_CUDA_KERNELS_H
#define Q38_CUDA_KERNELS_H

#include "q38/cuda_weights.h"

#include <cstddef>
#include <cstdint>

namespace q38 {

// All activation pointers use raw BF16 words. Stream is a cudaStream_t kept
// opaque so CPU-only translation units never depend on CUDA headers.
void cuda_gemv_bf16(const CudaMatrixViewV1& matrix,
                    const std::uint16_t* input,
                    std::uint16_t* output,
                    std::uint32_t batch,
                    void* stream,
                    int device,
                    float alpha = 1.0f,
                    bool accumulate = false);

// Expand one compact matrix into row-major BF16 once for the tensor-core
// prefill cache.  Decode continues to consume the compact view directly.
void cuda_dequantize_matrix_bf16(const CudaMatrixViewV1& matrix,
                                 std::uint16_t* output,
                                 void* stream,
                                 int device);

void cuda_embedding_bf16(const CudaMatrixViewV1& embedding,
                         const std::int32_t* token_ids,
                         std::uint16_t* output,
                         std::uint32_t tokens,
                         void* stream,
                         int device);

void cuda_qwen38_rmsnorm_bf16(const std::uint16_t* input,
                              const void* weight,
                              bool weight_f32,
                              std::uint16_t* output,
                              std::uint32_t vectors,
                              std::uint32_t width,
                              float epsilon,
                              bool one_centered_weight,
                              void* stream,
                              int device);

void cuda_silu_gate_bf16(const std::uint16_t* gate_up,
                         std::uint16_t* output,
                         std::uint32_t vectors,
                         std::uint32_t width,
                         void* stream,
                         int device);

void cuda_silu_multiply_bf16(const std::uint16_t* gate,
                             const std::uint16_t* up,
                             std::uint16_t* output,
                             std::size_t elements,
                             void* stream,
                             int device);

void cuda_sigmoid_multiply_bf16(const std::uint16_t* values,
                                const std::uint16_t* gates,
                                std::uint16_t* output,
                                std::size_t elements,
                                void* stream,
                                int device);

void cuda_add_bf16(const std::uint16_t* left,
                   const std::uint16_t* right,
                   std::uint16_t* output,
                   std::size_t elements,
                   void* stream,
                   int device);

void cuda_argmax_bf16(const std::uint16_t* logits,
                      std::uint32_t count,
                      std::int32_t* output,
                      void* stream,
                      int device);

// Router probabilities are computed in FP32. Selected weights remain FP32 so
// expert accumulation does not lose precision before the final BF16 cast.
void cuda_topk_router_bf16(const std::uint16_t* logits,
                           std::int32_t* expert_ids,
                           float* expert_weights,
                           std::uint32_t tokens,
                           std::uint32_t experts,
                           std::uint32_t top_k,
                           bool normalize_top_k,
                           void* stream,
                           int device);

bool cuda_q38_kernels_compiled();

}  // namespace q38

#endif
