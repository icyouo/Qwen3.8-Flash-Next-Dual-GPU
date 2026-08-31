#include "q38/cuda_hyper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace q38 {

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}
void select_device(int device) {
    if (device < 0) throw std::invalid_argument("invalid hyper device");
    check(cudaSetDevice(device), "cudaSetDevice(hyper)");
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

__global__ void repeat_embedding_kernel(const std::uint16_t* embedding,
                                         std::uint16_t* hyper) {
    const auto token = blockIdx.y;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= kQ38HyperWidth) return;
    store(hyper, static_cast<std::uint64_t>(token) * kQ38HyperWidth + index,
          load(embedding, static_cast<std::uint64_t>(token) * kQ38HiddenWidth +
                              index % kQ38HiddenWidth));
}

__global__ void group_rmsnorm_kernel(const std::uint16_t* hyper,
                                      const std::uint16_t* weight,
                                      std::uint16_t* output) {
    const auto stream = blockIdx.x;
    const auto token = blockIdx.y;
    const auto base = static_cast<std::uint64_t>(token) * kQ38HyperWidth +
                      stream * kQ38HiddenWidth;
    float squares = 0.0f;
    for (std::uint32_t index = threadIdx.x; index < kQ38HiddenWidth;
         index += blockDim.x) {
        const float value = load(hyper, base + index);
        squares += value * value;
    }
    const float inverse =
        rsqrtf(block_sum(squares) / kQ38HiddenWidth + 1.0e-6f);
    for (std::uint32_t index = threadIdx.x; index < kQ38HiddenWidth;
         index += blockDim.x)
        store(output, base + index,
              load(hyper, base + index) * inverse *
                  (1.0f + load(weight, stream * kQ38HiddenWidth + index)));
}

__global__ void elementwise_kernel(const std::uint16_t* input,
                                    std::uint16_t* output,
                                    std::uint64_t elements, float divisor,
                                    float output_scale, bool silu) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const float value = load(input, index) / divisor;
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    store(output, index, (silu ? value * sigmoid : sigmoid) * output_scale);
}

__global__ void mix_kernel(const std::uint16_t* normalized,
                           const std::uint16_t* weights,
                           std::uint16_t* mixed) {
    const auto token = blockIdx.y;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= kQ38HiddenWidth) return;
    const auto base = static_cast<std::uint64_t>(token) * kQ38HyperWidth;
    float value = 0.0f;
    for (std::uint32_t stream = 0; stream < kQ38HyperCount; ++stream) {
        const auto offset = base + stream * kQ38HiddenWidth + index;
        value += load(normalized, offset) * load(weights, offset);
    }
    store(mixed, static_cast<std::uint64_t>(token) * kQ38HiddenWidth + index,
          value / kQ38HyperCount);
}

__global__ void inject_kernel(const std::uint16_t* original,
                              const std::uint16_t* block_output,
                              const std::uint16_t* injection,
                              std::uint16_t* output) {
    const auto token = blockIdx.y;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= kQ38HyperWidth) return;
    const auto stream = index / kQ38HiddenWidth;
    const auto hidden = index % kQ38HiddenWidth;
    const auto hyper_offset =
        static_cast<std::uint64_t>(token) * kQ38HyperWidth + index;
    store(output, hyper_offset,
          load(original, hyper_offset) +
              load(block_output,
                   static_cast<std::uint64_t>(token) * kQ38HiddenWidth + hidden) *
                  load(injection,
                       static_cast<std::uint64_t>(token) * kQ38HyperCount +
                           stream));
}

void launch_elementwise(const std::uint16_t* input, std::uint16_t* output,
                        std::uint32_t tokens, std::uint32_t width,
                        float divisor, float scale, bool silu, void* stream,
                        int device) {
    if (!input || !output || !stream || tokens == 0 || width == 0 ||
        !(divisor > 0.0f))
        throw std::invalid_argument("invalid hyper elementwise buffers");
    select_device(device);
    const auto elements = static_cast<std::uint64_t>(tokens) * width;
    const auto blocks = (elements + 255) / 256;
    if (blocks > 0xffffffffu)
        throw std::overflow_error("hyper elementwise grid overflows");
    elementwise_kernel<<<static_cast<unsigned>(blocks), 256, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
        input, output, elements, divisor, scale, silu);
    check(cudaPeekAtLastError(), "hyper elementwise kernel");
}

}  // namespace

void cuda_hyper_repeat_embedding_bf16(const std::uint16_t* embedding,
                                      std::uint16_t* hyper,
                                      std::uint32_t tokens, void* stream,
                                      int device) {
    if (!embedding || !hyper || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper repeat buffers");
    select_device(device);
    repeat_embedding_kernel<<<dim3((kQ38HyperWidth + 255) / 256, tokens), 256,
                              0, reinterpret_cast<cudaStream_t>(stream)>>>(
        embedding, hyper);
    check(cudaPeekAtLastError(), "repeat_embedding_kernel");
}

void cuda_hyper_group_rmsnorm_bf16(
    const std::uint16_t* hyper, const std::uint16_t* norm_weight,
    std::uint16_t* normalized, std::uint32_t tokens, void* stream,
    int device) {
    if (!hyper || !norm_weight || !normalized || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper norm buffers");
    select_device(device);
    group_rmsnorm_kernel<<<dim3(kQ38HyperCount, tokens), 256, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
        hyper, norm_weight, normalized);
    check(cudaPeekAtLastError(), "group_rmsnorm_kernel");
}

void cuda_hyper_silu_scaled_bf16(
    const std::uint16_t* input, std::uint16_t* output, std::uint32_t tokens,
    std::uint32_t width, float divisor, void* stream, int device) {
    launch_elementwise(input, output, tokens, width, divisor, 1.0f, true,
                       stream, device);
}

void cuda_hyper_sigmoid_bf16(
    const std::uint16_t* input, std::uint16_t* output, std::uint32_t tokens,
    std::uint32_t width, float input_divisor, float output_scale, void* stream,
    int device) {
    launch_elementwise(input, output, tokens, width, input_divisor,
                       output_scale, false, stream, device);
}

void cuda_hyper_mix_bf16(const std::uint16_t* normalized_hyper,
                         const std::uint16_t* mix_weights,
                         std::uint16_t* mixed, std::uint32_t tokens,
                         void* stream, int device) {
    if (!normalized_hyper || !mix_weights || !mixed || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper mix buffers");
    select_device(device);
    mix_kernel<<<dim3((kQ38HiddenWidth + 255) / 256, tokens), 256, 0,
                     reinterpret_cast<cudaStream_t>(stream)>>>(
        normalized_hyper, mix_weights, mixed);
    check(cudaPeekAtLastError(), "hyper mix kernel");
}

void cuda_hyper_inject_bf16(
    const std::uint16_t* original_hyper,
    const std::uint16_t* block_output,
    const std::uint16_t* injection_weights,
    std::uint16_t* output_hyper, std::uint32_t tokens, void* stream,
    int device) {
    if (!original_hyper || !block_output || !injection_weights ||
        !output_hyper || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper injection buffers");
    select_device(device);
    inject_kernel<<<dim3((kQ38HyperWidth + 255) / 256, tokens), 256, 0,
                        reinterpret_cast<cudaStream_t>(stream)>>>(
        original_hyper, block_output, injection_weights, output_hyper);
    check(cudaPeekAtLastError(), "hyper injection kernel");
}

bool cuda_q38_hyper_compiled() { return true; }

}  // namespace q38
