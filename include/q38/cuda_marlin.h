#ifndef Q38_CUDA_MARLIN_H
#define Q38_CUDA_MARLIN_H
#include "q38/cuda_moe_prefill.h"
namespace q38 {
bool cuda_marlin_moe_enabled();
void cuda_marlin_repack_expert(void* data, void* scales, unsigned n, unsigned k,
    unsigned bits, void* data_scratch, void* scale_scratch, void* stream);
struct CudaMarlinWorkspace {
    std::int32_t *sorted=nullptr, *experts=nullptr, *padded=nullptr, *locks=nullptr;
    std::uint16_t *gate_up=nullptr, *down=nullptr;
    float* tmp=nullptr;
    // Sum of owned assignments; unowned rows form a contiguous packed tail.
    const std::uint32_t* owned_routes=nullptr;
    // Optional direct top10 reduction. Output then has routes/10 hidden rows;
    // preserves per-expert BF16 rounding and assignment-order FP32 addition.
    const std::uint32_t* reduce_assignment_to_packed=nullptr;
    const float* reduce_route_weights=nullptr;
};
void cuda_marlin_moe(const CudaTensorViewV1& gate, const CudaTensorViewV1& down,
    const std::uint16_t* packed_hidden, const Q38ExpertMmqTaskV1* tasks,
    unsigned max_tasks, unsigned routes, std::uint16_t* mid, float* output,
    CudaMarlinWorkspace workspace, void* stream);
}
#endif
