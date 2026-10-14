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

// Exact compact multirow GEMV; false leaves unsupported shapes untouched.
bool cuda_small_batch_gemv_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input, std::uint16_t* output, std::uint32_t batch,
    void* stream, int device);

// HC-only batch-1 candidate. Returns false without enqueueing for unsupported
// layouts. Keeps the reference 256-lane accumulation/reduction for each row;
// two rows reuse each input load. Optional activations retain the intermediate
// BF16 rounding of the separate GEMV + HC elementwise path.
enum class CudaHcActivation { kNone, kSilu, kSigmoid };
// Bounded BF16 tensor-core candidate for 2..7 HC rows; scratch capacity is
// 16 * 7 * 10240 FP32 elements. Returns false without enqueueing if unsupported.
bool cuda_try_hc_inject_batch_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input,std::uint16_t* output,unsigned tokens,void* stream,int device,
    CudaHcActivation activation,float divisor,float output_scale);
bool cuda_try_hc_mix_tensor_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* low,const std::uint16_t* normalized,std::uint16_t* mixed,
    unsigned tokens,void* stream,int device);
bool cuda_try_hc_tensor_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input, std::uint16_t* output, float* scratch,
    unsigned tokens, void* stream, int device, CudaHcActivation activation,
    float divisor, float output_scale);
bool cuda_try_hc_gemv_bf16(const CudaMatrixViewV1& matrix,
                           const std::uint16_t* input,
                           std::uint16_t* output,
                           void* stream, int device,
                           CudaHcActivation activation = CudaHcActivation::kNone,
                           float divisor = 1.0f, float output_scale = 1.0f);

// Independent HC projections of one normalized input. Both layouts must be
// supported before any work is enqueued; arithmetic matches the single lane.
struct CudaHcProjectionV1 {
    CudaMatrixViewV1 matrix;
    std::uint16_t* output = nullptr;
    CudaHcActivation activation = CudaHcActivation::kNone;
    float divisor = 1.0f;
    float output_scale = 1.0f;
};
// Batch independent HC rows with exactly the scalar pair-GEMV arithmetic.
bool cuda_try_hc_scalar_batch_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input,std::uint16_t* output,unsigned tokens,void* stream,int device,
    CudaHcActivation activation,float divisor,float output_scale);

bool cuda_try_hc_grouped_bf16(const CudaHcProjectionV1& first,
                               const CudaHcProjectionV1& second,
                               const std::uint16_t* input,
                               void* stream, int device);

// Expand one compact matrix into row-major BF16 once for the tensor-core
// prefill cache.  Decode continues to consume the compact view directly.
// Up to four matrices sharing one decode input, dispatched in one launch.
// Quantized matrices retain the optimized group-128 arithmetic; preserved
// matrices retain the reference block reduction. No artifact repack is needed.
struct CudaProjectionPackV1 {
    CudaMatrixViewV1 matrices[4]{};
    std::uint16_t* outputs[4]{};
    std::uint32_t count = 0;
};
void cuda_multi_projection_decode_bf16(
    const CudaProjectionPackV1& pack, const std::uint16_t* input,
    void* stream, int device);

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

// Stable lowest-ID tie handling; ignores NaNs like the single-row oracle.
// Each partial workspace needs rows * 128 elements; rows must be 1..7.
void cuda_argmax_bf16_batch(const std::uint16_t* logits,unsigned count,unsigned rows,
    float* partial_values,int* partial_ids,int* output,void* stream,int device);
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
