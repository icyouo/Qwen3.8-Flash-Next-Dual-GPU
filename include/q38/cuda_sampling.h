#ifndef Q38_CUDA_SAMPLING_H
#define Q38_CUDA_SAMPLING_H

#include <cstdint>
#include <memory>

namespace q38 {

// Preallocated exact top-k selection; RNG, top-p and publication remain in the
// transactional host sampler. No allocations occur during select().
class CudaSamplingWorkspace {
public:
    CudaSamplingWorkspace(int device, std::uint32_t vocabulary);
    ~CudaSamplingWorkspace();
    CudaSamplingWorkspace(const CudaSamplingWorkspace&) = delete;
    CudaSamplingWorkspace& operator=(const CudaSamplingWorkspace&) = delete;
    void select(const std::uint16_t* logits, std::uint32_t top_k,
                float temperature, void* stream);
    const std::uint16_t* logits() const;
    const std::int32_t* ids() const;
    std::uint64_t bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace q38
#endif
