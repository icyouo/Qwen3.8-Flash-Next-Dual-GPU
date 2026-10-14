#include "q38/cuda_sampling.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/device/device_radix_sort.cuh>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38 {
namespace {
void check(cudaError_t status, const char* op) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
}
__global__ void sampling_keys(const std::uint16_t* logits,
                              std::uint64_t* keys, std::int32_t* ids,
                              std::uint32_t size, float temperature) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float value = __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(logits)[i]);
    if (!isfinite(value)) value = -CUDART_INF_F;
    value = __fdiv_rn(value, temperature);
    // Canonicalize signed zero: the host comparator treats +0 and -0 as ties.
    if (value == 0.0f) value = 0.0f;
    const auto bits = __float_as_uint(value);
    const auto ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    keys[i] = (static_cast<std::uint64_t>(ordered) << 32u) | (0xffffffffu - i);
    ids[i] = static_cast<std::int32_t>(i);
}
__global__ void gather_logits(const std::uint16_t* logits,
                              const std::int32_t* ids,
                              std::uint16_t* gathered, std::uint32_t count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gathered[i] = logits[ids[i]];
}
}  // namespace

struct CudaSamplingWorkspace::Impl {
    int device;
    std::uint32_t vocabulary;
    std::uint64_t allocated = 0;
    std::vector<void*> allocations;
    std::uint64_t *keys_in = nullptr, *keys_out = nullptr;
    std::int32_t *ids_in = nullptr, *ids_out = nullptr;
    std::uint16_t* gathered = nullptr;
    void* scratch = nullptr;
    std::size_t scratch_bytes = 0;
    cudaStream_t last_stream = nullptr;

    template <class T> T* allocate(std::size_t count) {
        T* p = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)),
              "cudaMalloc(sampling)");
        allocations.push_back(p);
        allocated += count * sizeof(T);
        return p;
    }
    Impl(int value_device, std::uint32_t size)
        : device(value_device), vocabulary(size) {
        if (device < 0 || size == 0 || size > 1000000)
            throw std::invalid_argument("invalid sampling workspace extent");
        check(cudaSetDevice(device), "cudaSetDevice(sampling)");
        try {
            keys_in = allocate<std::uint64_t>(size);
            keys_out = allocate<std::uint64_t>(size);
            ids_in = allocate<std::int32_t>(size);
            ids_out = allocate<std::int32_t>(size);
            gathered = allocate<std::uint16_t>(size);
            check(cub::DeviceRadixSort::SortPairsDescending(
                nullptr, scratch_bytes, keys_in, keys_out, ids_in, ids_out,
                size), "sampling sort workspace");
            scratch = allocate<std::uint8_t>(scratch_bytes);
        } catch (...) {
            for (auto* p : allocations) (void)cudaFree(p);
            throw;
        }
    }
    ~Impl() {
        (void)cudaSetDevice(device);
        if (last_stream) (void)cudaStreamSynchronize(last_stream);
        for (auto* p : allocations) (void)cudaFree(p);
    }
};

CudaSamplingWorkspace::CudaSamplingWorkspace(int device, std::uint32_t vocabulary)
    : impl_(std::make_unique<Impl>(device, vocabulary)) {}
CudaSamplingWorkspace::~CudaSamplingWorkspace() = default;

void CudaSamplingWorkspace::select(const std::uint16_t* logits,
                                  std::uint32_t top_k, float temperature,
                                  void* stream) {
    auto& s = *impl_;
    if (!logits || !stream || top_k == 0 || top_k > s.vocabulary ||
        !(temperature > 0.0f) || !std::isfinite(temperature))
        throw std::invalid_argument("invalid compact sampling request");
    check(cudaSetDevice(s.device), "cudaSetDevice(sampling select)");
    s.last_stream = reinterpret_cast<cudaStream_t>(stream);
    sampling_keys<<<(s.vocabulary + 255u) / 256u, 256, 0, s.last_stream>>>(
        logits, s.keys_in, s.ids_in, s.vocabulary, temperature);
    check(cudaPeekAtLastError(), "sampling_keys");
    check(cub::DeviceRadixSort::SortPairsDescending(
        s.scratch, s.scratch_bytes, s.keys_in, s.keys_out, s.ids_in, s.ids_out,
        s.vocabulary, 0, 64, s.last_stream), "sampling radix sort");
    gather_logits<<<(top_k + 255u) / 256u, 256, 0, s.last_stream>>>(
        logits, s.ids_out, s.gathered, top_k);
    check(cudaPeekAtLastError(), "gather_logits");
}
const std::uint16_t* CudaSamplingWorkspace::logits() const { return impl_->gathered; }
const std::int32_t* CudaSamplingWorkspace::ids() const { return impl_->ids_out; }
std::uint64_t CudaSamplingWorkspace::bytes() const { return impl_->allocated; }
}  // namespace q38
