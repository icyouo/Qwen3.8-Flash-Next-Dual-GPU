#ifndef Q38_CUDA_TRANSPORT_H
#define Q38_CUDA_TRANSPORT_H

#include "q38/backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace q38 {

struct CudaBoundaryRingOptions {
    int producer_device = 0;
    std::uint32_t slots = 3;
    std::uint32_t max_tokens = 4096;
    std::uint32_t hidden_width = 4 * 2560;
};

struct CudaBoundaryRingStats {
    std::uint64_t transfers = 0;
    std::uint64_t bytes = 0;
    std::uint64_t waits = 0;
};

// CUDA objects are hidden behind void stream/device pointers so the control
// plane and CPU tests never need CUDA headers or libcuda.
class CudaBoundaryRing {
public:
    explicit CudaBoundaryRing(CudaBoundaryRingOptions options);
    ~CudaBoundaryRing();

    CudaBoundaryRing(const CudaBoundaryRing&) = delete;
    CudaBoundaryRing& operator=(const CudaBoundaryRing&) = delete;
    CudaBoundaryRing(CudaBoundaryRing&&) noexcept;
    CudaBoundaryRing& operator=(CudaBoundaryRing&&) noexcept;

    std::shared_ptr<const BoundaryLease> copy_from_device(
        const std::uint16_t* device_source, std::size_t words,
        void* producer_stream);
    CudaBoundaryRingStats stats() const;
    std::uint64_t pinned_bytes() const;
    std::uint64_t device_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void cuda_copy_boundary_to_device(const BoundaryBuffer& boundary,
                                  std::uint16_t* device_destination,
                                  void* consumer_stream,
                                  int consumer_device);
bool cuda_transport_compiled();

}  // namespace q38

#endif
