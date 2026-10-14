#ifndef Q38_CUDA_PREFIX_SNAPSHOT_H
#define Q38_CUDA_PREFIX_SNAPSHOT_H

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace q38 {
// Row i stores the state after evaluated token i. The caller owns allocation
// and stream lifetime and provides storage disjoint from inputs/working state.
// Stride allows one layer to write into a full-model bank.
struct CudaPrefixSnapshotView {
    void* data = nullptr;
    std::uint64_t stride_bytes = 0;
    std::uint32_t count = 0;
};

inline void validate_prefix_snapshot(CudaPrefixSnapshotView view,
                                    std::uint32_t tokens,
                                    std::uint64_t state_bytes,
                                    std::uint64_t alignment) {
    if (!view.data && !view.stride_bytes && !view.count) return;
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    if (!alignment || !state_bytes || !view.data || !view.count || view.count > tokens || view.count > 64 ||
        view.stride_bytes < state_bytes || address % alignment ||
        view.stride_bytes % alignment ||
        address > std::numeric_limits<std::uintptr_t>::max() - state_bytes ||
        (view.count > 1 && view.stride_bytes >
            (std::numeric_limits<std::uintptr_t>::max() - address - state_bytes) /
                (view.count - 1)))
        throw std::invalid_argument("invalid CUDA prefix snapshot extent");
}
}  // namespace q38
#endif
