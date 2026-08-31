#ifndef Q38_CUDA_WEIGHTS_H
#define Q38_CUDA_WEIGHTS_H

#include "q38/device_artifact.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace q38 {

struct CudaDeviceWeightOptions {
    int device = 0;
    std::size_t staging_bytes = 64ull * 1024ull * 1024ull;
    std::uint32_t staging_buffers = 2;
    std::size_t arena_alignment = 4096;
    bool load_mtp = true;
};

struct CudaTensorViewV1 {
    const DeviceTensorV1* descriptor = nullptr;
    const void* data = nullptr;
    const void* scales = nullptr;

    bool empty() const { return descriptor == nullptr; }
};

struct CudaMatrixViewV1 {
    DeviceWeightFormatV1 format = DeviceWeightFormatV1::kInvalid;
    const void* data = nullptr;
    const void* scales = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t group_size = 0;
    bool preserved_f32 = false;
};

CudaMatrixViewV1 cuda_matrix_view(const CudaTensorViewV1& tensor,
                                  std::uint64_t outer_index = 0);

struct CudaDeviceWeightStats {
    std::uint64_t arena_bytes = 0;
    std::uint64_t uploaded_bytes = 0;
    std::uint64_t upload_chunks = 0;
    std::uint64_t staging_peak_pinned_bytes = 0;
    std::uint64_t w4_bytes = 0;
    std::uint64_t w8_bytes = 0;
    std::uint64_t preserved_bf16_bytes = 0;
    std::uint64_t preserved_f32_bytes = 0;
    std::uint64_t preserved_other_bytes = 0;
    std::uint32_t segment_count = 0;
    std::uint32_t tensor_count = 0;
    std::uint32_t host_only_tensor_count = 0;
    std::uint64_t host_only_bytes = 0;
    std::uint32_t excluded_tensor_count = 0;
    std::uint64_t excluded_bytes = 0;
};

// Uploads resident tensors from a mapped production stage artifact into one
// compact CUDA arena. FP8 PLE table tensors remain host/SSD-only. The source
// mapping is needed only for the duration of construction.
class CudaDeviceWeightStore {
public:
    CudaDeviceWeightStore(const DeviceWeightStore& source,
                          CudaDeviceWeightOptions options);
    ~CudaDeviceWeightStore();

    CudaDeviceWeightStore(const CudaDeviceWeightStore&) = delete;
    CudaDeviceWeightStore& operator=(const CudaDeviceWeightStore&) = delete;
    CudaDeviceWeightStore(CudaDeviceWeightStore&&) noexcept;
    CudaDeviceWeightStore& operator=(CudaDeviceWeightStore&&) noexcept;

    int device() const;
    std::uint32_t stage() const;
    CudaTensorViewV1 find(const std::string& name) const;
    CudaTensorViewV1 require(const std::string& name) const;
    CudaDeviceWeightStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool cuda_device_weights_compiled();

}  // namespace q38

#endif
