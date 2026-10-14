#ifndef Q38_CUDA_HYPER_H
#define Q38_CUDA_HYPER_H

#include <cstdint>

namespace q38 {

constexpr std::uint32_t kQ38HiddenWidth = 2560;
constexpr std::uint32_t kQ38HyperCount = 4;
constexpr std::uint32_t kQ38HyperWidth = 10240;
constexpr std::uint32_t kQ38HyperLowRank = 320;

void cuda_hyper_repeat_embedding_bf16(const std::uint16_t* embedding,
                                      std::uint16_t* hyper,
                                      std::uint32_t tokens,
                                      void* stream,
                                      int device);

void cuda_hyper_group_rmsnorm_bf16(const std::uint16_t* hyper,
                                   const std::uint16_t* norm_weight,
                                   std::uint16_t* normalized,
                                   std::uint32_t tokens,
                                   void* stream,
                                   int device);

// Two independent norms, preserving each original 256-thread reduction.
// Inputs may share storage; outputs must be disjoint from all inputs/weights.
struct CudaHyperNormV1 {
    const std::uint16_t* input = nullptr;
    const std::uint16_t* weight = nullptr;
    std::uint16_t* output = nullptr;
};
void cuda_hyper_pair_group_rmsnorm_bf16(
    const CudaHyperNormV1& first, const CudaHyperNormV1& second,
    std::uint32_t tokens, void* stream, int device);

void cuda_hyper_silu_scaled_bf16(const std::uint16_t* input,
                                 std::uint16_t* output,
                                 std::uint32_t tokens,
                                 std::uint32_t width,
                                 float divisor,
                                 void* stream,
                                 int device);

void cuda_hyper_sigmoid_bf16(const std::uint16_t* input,
                             std::uint16_t* output,
                             std::uint32_t tokens,
                             std::uint32_t width,
                             float input_divisor,
                             float output_scale,
                             void* stream,
                             int device);

void cuda_hyper_mix_bf16(const std::uint16_t* normalized_hyper,
                         const std::uint16_t* mix_weights,
                         std::uint16_t* mixed,
                         std::uint32_t tokens,
                         void* stream,
                         int device);

void cuda_hyper_inject_bf16(const std::uint16_t* original_hyper,
                            const std::uint16_t* block_output,
                            const std::uint16_t* injection_weights,
                            std::uint16_t* output_hyper,
                            std::uint32_t tokens,
                            void* stream,
                            int device);

// Accept raw BF16 HC injection projection. Keep its BF16 sigmoid*2 boundary
// inside the residual kernel, eliminating the separate elementwise launch.
void cuda_hyper_inject_from_projection_bf16(
    const std::uint16_t* original,const std::uint16_t* block,
    const std::uint16_t* projection,std::uint16_t* output,unsigned tokens,void* stream,int device);

bool cuda_q38_hyper_compiled();
// Fuse shared-expert combination and HC residual with explicit BF16 block
// and activated injection boundaries, matching the two existing kernels.
void cuda_hyper_moe_inject_bf16(const std::uint16_t* original,
    const std::uint16_t* routed, const std::uint16_t* shared,
    const std::uint16_t* shared_gate, const std::uint16_t* injection,
    std::uint16_t* output, unsigned tokens, void* stream, int device);
void cuda_hyper_add_embedding_bf16(const std::uint16_t* hyper,
    const std::uint16_t* embedding, std::uint16_t* output,
    unsigned tokens, void* stream, int device);

}  // namespace q38

#endif
