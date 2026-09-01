#include "q38/cuda_ple.h"

#include "q38/cuda_hyper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace q38 {

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}
void select_device(int device) {
    if (device < 0) throw std::invalid_argument("invalid PLE device");
    check(cudaSetDevice(device), "cudaSetDevice(PLE)");
}
__device__ __forceinline__ float load(const std::uint16_t* source,
                                       std::uint64_t index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(source)[index]);
}
__device__ __forceinline__ void store(std::uint16_t* destination,
                                       std::uint64_t index, float value) {
    reinterpret_cast<__nv_bfloat16*>(destination)[index] =
        __float2bfloat16_rn(value);
}
__device__ float decode_e4m3fn(std::uint8_t encoded) {
    const float sign = (encoded & 0x80) ? -1.0f : 1.0f;
    const auto exponent = (encoded >> 3) & 0x0f;
    const auto mantissa = encoded & 0x07;
    if (exponent == 0)
        return sign * static_cast<float>(mantissa) * 0.001953125f;
    if (exponent == 15 && mantissa == 7) return 0.0f;
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                         static_cast<int>(exponent) - 7);
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}
__device__ float block_sum(float value) {
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? partial[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__global__ void fp8_rows_kernel(const std::uint8_t* rows,
                                 const std::uint16_t* row_scales,
                                 std::uint16_t* embedding,
                                 std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        const auto row = index / kQ38PleRowWidth;
        store(embedding, index,
              decode_e4m3fn(rows[index]) * load(row_scales, row));
    }
}

__global__ void ple_gate_kernel(const std::uint16_t* key,
                                const std::uint16_t* query,
                                const std::uint16_t* value,
                                std::uint16_t* gated) {
    const auto stream = blockIdx.x;
    const auto token = blockIdx.y;
    const auto hyper_base =
        static_cast<std::uint64_t>(token) * kQ38HyperWidth;
    const auto value_base =
        static_cast<std::uint64_t>(token) * kQ38HiddenWidth;
    float dot = 0.0f;
    for (std::uint32_t index = threadIdx.x; index < kQ38HiddenWidth;
         index += blockDim.x) {
        const auto offset =
            hyper_base + stream * kQ38HiddenWidth + index;
        dot += load(key, offset) * load(query, offset);
    }
    dot = block_sum(dot) * rsqrtf(static_cast<float>(kQ38HiddenWidth));
    __shared__ float gate;
    if (threadIdx.x == 0) {
        const float signed_root = copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)),
                                            dot);
        gate = 1.0f / (1.0f + expf(-signed_root));
    }
    __syncthreads();
    for (std::uint32_t index = threadIdx.x; index < kQ38HiddenWidth;
         index += blockDim.x)
        store(gated,
              hyper_base + stream * kQ38HiddenWidth + index,
              gate * load(value, value_base + index));
}

__global__ void ple_conv_decode_kernel(
    const std::uint16_t* gated, const std::uint16_t* normalized,
    const std::uint16_t* weight, std::uint16_t* state,
    std::uint16_t* output) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (channel >= kQ38HyperWidth) return;
    const auto state_base =
        static_cast<std::uint64_t>(channel) * kQ38PleConvState;
    const auto weight_base = static_cast<std::uint64_t>(channel) * 4;
    float convolved = load(state, state_base) * load(weight, weight_base) +
                      load(state, state_base + 3) *
                          load(weight, weight_base + 1) +
                      load(state, state_base + 6) *
                          load(weight, weight_base + 2) +
                      load(normalized, channel) *
                          load(weight, weight_base + 3);
    for (std::uint32_t index = 0; index + 1 < kQ38PleConvState; ++index)
        store(state, state_base + index, load(state, state_base + index + 1));
    store(state, state_base + kQ38PleConvState - 1,
          load(normalized, channel));
    const float silu = convolved / (1.0f + expf(-convolved));
    store(output, channel, load(gated, channel) + silu);
}

__global__ void ple_conv_prefill_kernel(
    const std::uint16_t* gated, const std::uint16_t* normalized,
    const std::uint16_t* weight, std::uint16_t* state,
    std::uint16_t* output, std::uint32_t tokens) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (channel >= kQ38HyperWidth) return;
    const auto state_base =
        static_cast<std::uint64_t>(channel) * kQ38PleConvState;
    const auto weight_base = static_cast<std::uint64_t>(channel) * 4;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto offset =
            static_cast<std::uint64_t>(token) * kQ38HyperWidth + channel;
        float convolved = load(state, state_base) * load(weight, weight_base) +
                          load(state, state_base + 3) *
                              load(weight, weight_base + 1) +
                          load(state, state_base + 6) *
                              load(weight, weight_base + 2) +
                          load(normalized, offset) *
                              load(weight, weight_base + 3);
        for (std::uint32_t index = 0; index + 1 < kQ38PleConvState; ++index)
            store(state, state_base + index,
                  load(state, state_base + index + 1));
        store(state, state_base + kQ38PleConvState - 1,
              load(normalized, offset));
        const float silu = convolved / (1.0f + expf(-convolved));
        store(output, offset, load(gated, offset) + silu);
    }
}

}  // namespace

struct CudaPleStateBank::Impl {
    int device;
    std::uint16_t* banks[2]{nullptr, nullptr};
    std::uint16_t* checkpoint = nullptr;
    std::uint32_t committed = 0;
    bool active = false;
    bool checkpoint_valid = false;
    std::uint64_t epoch = 0;
    static constexpr std::uint64_t words =
        static_cast<std::uint64_t>(kQ38HyperWidth) * kQ38PleConvState;

    Impl(int value_device, bool enable_checkpoint) : device(value_device) {
        select_device(device);
        try {
            check(cudaMalloc(reinterpret_cast<void**>(&banks[0]), words * 2),
                  "cudaMalloc(PLE bank 0)");
            check(cudaMalloc(reinterpret_cast<void**>(&banks[1]), words * 2),
                  "cudaMalloc(PLE bank 1)");
            if (enable_checkpoint)
                check(cudaMalloc(reinterpret_cast<void**>(&checkpoint),
                                 words * 2),
                      "cudaMalloc(PLE checkpoint)");
            check(cudaMemset(banks[0], 0, words * 2),
                  "cudaMemset(PLE bank 0)");
            check(cudaMemset(banks[1], 0, words * 2),
                  "cudaMemset(PLE bank 1)");
        } catch (...) {
            release();
            throw;
        }
    }
    ~Impl() { release(); }
    void release() noexcept {
        (void)cudaSetDevice(device);
        if (banks[0]) (void)cudaFree(banks[0]);
        if (banks[1]) (void)cudaFree(banks[1]);
        if (checkpoint) (void)cudaFree(checkpoint);
        banks[0] = banks[1] = nullptr;
        checkpoint = nullptr;
    }
};

CudaPleStateBank::CudaPleStateBank(int device, bool enable_checkpoint)
    : impl_(std::make_unique<Impl>(device, enable_checkpoint)) {}
CudaPleStateBank::~CudaPleStateBank() = default;
CudaPleStateBank::CudaPleStateBank(CudaPleStateBank&&) noexcept = default;
CudaPleStateBank& CudaPleStateBank::operator=(CudaPleStateBank&&) noexcept =
    default;

void CudaPleStateBank::begin(std::uint64_t epoch, void* stream) {
    if (!stream || epoch == 0 || impl_->active)
        throw std::logic_error("invalid PLE transaction begin");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->banks[1u - impl_->committed],
                          impl_->banks[impl_->committed], Impl::words * 2,
                          cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(PLE transaction clone)");
    impl_->active = true;
    impl_->epoch = epoch;
    impl_->checkpoint_valid = false;
}
void CudaPleStateBank::restore(void* stream) {
    if (!stream || !impl_->active)
        throw std::logic_error("invalid PLE transaction restore");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->banks[1u - impl_->committed],
                          impl_->banks[impl_->committed], Impl::words * 2,
                          cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(PLE transaction restore)");
}
void CudaPleStateBank::checkpoint(void* stream) {
    if (!stream || !impl_->active || !impl_->checkpoint)
        throw std::logic_error("invalid PLE speculative checkpoint");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->checkpoint,
                          impl_->banks[1u - impl_->committed], Impl::words * 2,
                          cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(PLE speculative checkpoint)");
    impl_->checkpoint_valid = true;
}
void CudaPleStateBank::restore_checkpoint(void* stream) {
    if (!stream || !impl_->active || !impl_->checkpoint_valid)
        throw std::logic_error("invalid PLE checkpoint restore");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->banks[1u - impl_->committed],
                          impl_->checkpoint, Impl::words * 2,
                          cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(PLE checkpoint restore)");
}
std::uint16_t* CudaPleStateBank::working() const {
    if (!impl_->active) throw std::logic_error("PLE state is not active");
    return impl_->banks[1u - impl_->committed];
}
void CudaPleStateBank::commit(std::uint64_t epoch) {
    if (!impl_->active || impl_->epoch != epoch)
        throw std::logic_error("PLE commit epoch mismatch");
    impl_->committed = 1u - impl_->committed;
    impl_->active = false;
    impl_->epoch = 0;
    impl_->checkpoint_valid = false;
}
void CudaPleStateBank::rollback(std::uint64_t epoch) {
    if (!impl_->active || impl_->epoch != epoch)
        throw std::logic_error("PLE rollback epoch mismatch");
    impl_->active = false;
    impl_->epoch = 0;
    impl_->checkpoint_valid = false;
}
void CudaPleStateBank::reset(void* stream) {
    if (!stream || impl_->active) throw std::logic_error("invalid PLE reset");
    select_device(impl_->device);
    check(cudaMemsetAsync(impl_->banks[0], 0, Impl::words * 2,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemsetAsync(PLE bank 0)");
    check(cudaMemsetAsync(impl_->banks[1], 0, Impl::words * 2,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemsetAsync(PLE bank 1)");
    if (impl_->checkpoint)
        check(cudaMemsetAsync(impl_->checkpoint, 0, Impl::words * 2,
                              reinterpret_cast<cudaStream_t>(stream)),
              "cudaMemsetAsync(PLE checkpoint)");
    impl_->committed = 0;
    impl_->checkpoint_valid = false;
}

std::uint64_t CudaPleStateBank::allocated_bytes() const {
    return (impl_->checkpoint ? 3ull : 2ull) * Impl::words *
           sizeof(std::uint16_t);
}

void cuda_ple_fp8_rows_to_bf16(const std::uint8_t* fp8_rows,
                               const std::uint16_t* row_scales,
                               std::uint16_t* embedding,
                               std::uint32_t tokens, void* stream,
                               int device) {
    if (!fp8_rows || !row_scales || !embedding || !stream || tokens == 0)
        throw std::invalid_argument("invalid PLE FP8 buffers");
    select_device(device);
    const auto elements = static_cast<std::uint64_t>(tokens) *
                          kQ38PleRowsPerToken * kQ38PleRowWidth;
    fp8_rows_kernel<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        fp8_rows, row_scales, embedding, elements);
    check(cudaPeekAtLastError(), "fp8_rows_kernel");
}

void cuda_ple_gate_bf16(const std::uint16_t* normalized_key,
                        const std::uint16_t* normalized_query,
                        const std::uint16_t* value,
                        std::uint16_t* gated_value, void* stream,
                        int device) {
    if (!normalized_key || !normalized_query || !value || !gated_value ||
        !stream)
        throw std::invalid_argument("invalid PLE gate buffers");
    select_device(device);
    ple_gate_kernel<<<dim3(kQ38HyperCount, 1), 256, 0,
                      reinterpret_cast<cudaStream_t>(stream)>>>(
        normalized_key, normalized_query, value, gated_value);
    check(cudaPeekAtLastError(), "ple_gate_kernel");
}

void cuda_ple_conv_decode_bf16(
    const std::uint16_t* gated_value,
    const std::uint16_t* normalized_gated_value,
    const std::uint16_t* conv_weight, std::uint16_t* conv_state,
    std::uint16_t* output, void* stream, int device) {
    if (!gated_value || !normalized_gated_value || !conv_weight ||
        !conv_state || !output || !stream)
        throw std::invalid_argument("invalid PLE convolution buffers");
    select_device(device);
    ple_conv_decode_kernel<<<(kQ38HyperWidth + 255) / 256, 256, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        gated_value, normalized_gated_value, conv_weight, conv_state, output);
    check(cudaPeekAtLastError(), "ple_conv_decode_kernel");
}

bool cuda_q38_ple_compiled() { return true; }

void cuda_ple_gate_prefill_bf16(
    const std::uint16_t* normalized_key,
    const std::uint16_t* normalized_query,
    const std::uint16_t* value,
    std::uint16_t* gated_value,
    std::uint32_t tokens,
    void* stream,
    int device) {
    if (!normalized_key || !normalized_query || !value || !gated_value ||
        !stream || tokens == 0)
        throw std::invalid_argument("invalid PLE prefill gate buffers");
    select_device(device);
    ple_gate_kernel<<<dim3(kQ38HyperCount, tokens), 256, 0,
                      reinterpret_cast<cudaStream_t>(stream)>>>(
        normalized_key, normalized_query, value, gated_value);
    check(cudaPeekAtLastError(), "PLE prefill gate");
}

void cuda_ple_conv_prefill_bf16(
    const std::uint16_t* gated_value,
    const std::uint16_t* normalized_gated_value,
    const std::uint16_t* conv_weight,
    std::uint16_t* conv_state,
    std::uint16_t* output,
    std::uint32_t tokens,
    void* stream,
    int device) {
    if (!gated_value || !normalized_gated_value || !conv_weight ||
        !conv_state || !output || !stream || tokens == 0)
        throw std::invalid_argument("invalid PLE prefill convolution buffers");
    select_device(device);
    ple_conv_prefill_kernel<<<(kQ38HyperWidth + 255) / 256, 256, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        gated_value, normalized_gated_value, conv_weight, conv_state, output,
        tokens);
    check(cudaPeekAtLastError(), "PLE prefill convolution");
}

}  // namespace q38
