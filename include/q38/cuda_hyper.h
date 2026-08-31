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

bool cuda_q38_hyper_compiled();

}  // namespace q38

#endif
