#include "q38/cuda_moe.h"

#include "q38/cuda_hyper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace q38 {

namespace {

constexpr int kWarpSize = 32;
constexpr int kDecodeWarpsPerBlock = 8;
constexpr int kDecodeThreads = kWarpSize * kDecodeWarpsPerBlock;
constexpr std::uint32_t kQuantGroupSize = 128;

bool optimized_decode_moe_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("Q38_CUDA_DECODE_MOE");
        return !value || (std::strcmp(value, "scalar") != 0 &&
                          std::strcmp(value, "0") != 0 &&
                          std::strcmp(value, "false") != 0);
    }();
    return enabled;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}
void select_device(int device) {
    if (device < 0) throw std::invalid_argument("invalid MoE device");
    check(cudaSetDevice(device), "cudaSetDevice(MoE)");
}
__device__ __forceinline__ float load_bf16(const std::uint16_t* source,
                                            std::uint64_t index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(source)[index]);
}
__device__ __forceinline__ void store_bf16(std::uint16_t* destination,
                                            std::uint64_t index,
                                            float value) {
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
    return value;
}

__device__ float quant_weight(int format, const void* data,
                              const std::uint16_t* scales,
                              std::uint64_t matrix_elements,
                              std::uint64_t matrix_scale_elements,
                              std::uint32_t columns, std::int32_t expert,
                              std::uint32_t row, std::uint32_t column) {
    const auto local = static_cast<std::uint64_t>(row) * columns + column;
    const auto scale = load_bf16(
        scales, static_cast<std::uint64_t>(expert) * matrix_scale_elements +
                    static_cast<std::uint64_t>(row) * (columns / 128) +
                    column / 128);
    if (format == static_cast<int>(DeviceWeightFormatV1::kW8A16SymG128)) {
        const auto index = static_cast<std::uint64_t>(expert) * matrix_elements +
                           local;
        return static_cast<float>(
                   static_cast<const std::int8_t*>(data)[index]) *
               scale;
    }
    const auto nibble_index = static_cast<std::uint64_t>(expert) *
                                  (matrix_elements / 2) +
                              (local >> 1);
    const auto packed = static_cast<const std::uint8_t*>(data)[nibble_index];
    int quantized = (local & 1) ? packed >> 4 : packed & 0x0f;
    if (quantized >= 8) quantized -= 16;
    return static_cast<float>(quantized) * scale;
}

__device__ __forceinline__ int sign_extend_int4(unsigned value) {
    return static_cast<int>(value) - ((value & 0x8u) ? 16 : 0);
}

template <int WeightBits>
__device__ __forceinline__ float quantized_row_dot_warp(
    const void* data, const std::uint16_t* scales,
    std::uint64_t matrix_elements, std::uint64_t matrix_scale_elements,
    std::uint32_t columns, std::int32_t expert, std::uint32_t row,
    const std::uint16_t* input) {
    const auto lane = threadIdx.x & (kWarpSize - 1);
    const auto groups = columns / kQuantGroupSize;
    float sum = 0.0f;
    for (std::uint32_t group = 0; group < groups; ++group) {
        float scale = 0.0f;
        if (lane == 0)
            scale = load_bf16(
                scales, static_cast<std::uint64_t>(expert) *
                            matrix_scale_elements +
                            static_cast<std::uint64_t>(row) * groups + group);
        scale = __shfl_sync(0xffffffffu, scale, 0);

        const auto column = group * kQuantGroupSize + lane * 4;
        const float a0 = load_bf16(input, column);
        const float a1 = load_bf16(input, column + 1);
        const float a2 = load_bf16(input, column + 2);
        const float a3 = load_bf16(input, column + 3);
        const auto local = static_cast<std::uint64_t>(row) * columns + column;
        float dot = 0.0f;
        if constexpr (WeightBits == 8) {
            const auto* weights = static_cast<const std::int8_t*>(data) +
                                  static_cast<std::uint64_t>(expert) *
                                      matrix_elements +
                                  local;
            const auto packed = *reinterpret_cast<const char4*>(weights);
            dot = a0 * static_cast<float>(packed.x) +
                  a1 * static_cast<float>(packed.y) +
                  a2 * static_cast<float>(packed.z) +
                  a3 * static_cast<float>(packed.w);
        } else {
            const auto* weights = static_cast<const std::uint8_t*>(data) +
                                  static_cast<std::uint64_t>(expert) *
                                      (matrix_elements / 2) +
                                  (local >> 1);
            const auto packed =
                *reinterpret_cast<const std::uint16_t*>(weights);
            dot = a0 * static_cast<float>(sign_extend_int4(packed & 0x0fu)) +
                  a1 * static_cast<float>(
                           sign_extend_int4((packed >> 4) & 0x0fu)) +
                  a2 * static_cast<float>(
                           sign_extend_int4((packed >> 8) & 0x0fu)) +
                  a3 * static_cast<float>(
                           sign_extend_int4((packed >> 12) & 0x0fu));
        }
        sum = fmaf(dot, scale, sum);
    }
    return warp_sum(sum);
}

template <int WeightBits>
__global__ void expert_gate_up_decode_kernel(
    const void* data, const std::uint16_t* scales,
    const std::uint16_t* hidden, const std::int32_t* expert_ids,
    std::uint32_t top_k, std::uint16_t* output) {
    constexpr std::uint32_t rows = 2 * kQ38MoeIntermediate;
    constexpr std::uint32_t columns = kQ38HiddenWidth;
    extern __shared__ std::uint16_t cached_hidden[];
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        cached_hidden[column] = hidden[column];
    __syncthreads();

    const auto warp = threadIdx.x / kWarpSize;
    const auto lane = threadIdx.x & (kWarpSize - 1);
    const auto row = static_cast<std::uint32_t>(blockIdx.x) *
                         kDecodeWarpsPerBlock +
                     warp;
    const auto route = static_cast<std::uint32_t>(blockIdx.y);
    if (row >= rows || route >= top_k) return;
    const auto expert = expert_ids[route];
    if (expert < 0 || expert >= static_cast<std::int32_t>(kQ38MoeExperts)) {
        if (lane == 0) store_bf16(output, route * rows + row, 0.0f);
        return;
    }
    const auto sum = quantized_row_dot_warp<WeightBits>(
        data, scales, static_cast<std::uint64_t>(rows) * columns,
        static_cast<std::uint64_t>(rows) * (columns / kQuantGroupSize),
        columns, expert, row, cached_hidden);
    if (lane == 0) store_bf16(output, route * rows + row, sum);
}

template <int WeightBits>
__global__ void expert_down_decode_kernel(
    const void* data, const std::uint16_t* scales,
    const std::uint16_t* activated, const std::int32_t* expert_ids,
    const float* expert_weights, std::uint32_t top_k,
    std::uint16_t* routed_output) {
    constexpr std::uint32_t rows = kQ38HiddenWidth;
    constexpr std::uint32_t columns = kQ38MoeIntermediate;
    extern __shared__ std::uint16_t cached_activated[];
    const auto warp = threadIdx.x / kWarpSize;
    const auto lane = threadIdx.x & (kWarpSize - 1);
    const auto row = static_cast<std::uint32_t>(blockIdx.x) *
                         kDecodeWarpsPerBlock +
                     warp;
    float total = 0.0f;
    for (std::uint32_t route = 0; route < top_k; ++route) {
        const auto* route_input =
            activated + static_cast<std::uint64_t>(route) * columns;
        for (std::uint32_t column = threadIdx.x; column < columns;
             column += blockDim.x)
            cached_activated[column] = route_input[column];
        __syncthreads();
        const auto expert = expert_ids[route];
        if (row < rows && expert >= 0 &&
            expert < static_cast<std::int32_t>(kQ38MoeExperts)) {
            const auto sum = quantized_row_dot_warp<WeightBits>(
                data, scales, static_cast<std::uint64_t>(rows) * columns,
                static_cast<std::uint64_t>(rows) *
                    (columns / kQuantGroupSize),
                columns, expert, row, cached_activated);
            if (lane == 0) total += sum * expert_weights[route];
        }
        __syncthreads();
    }
    if (row < rows && lane == 0)
        store_bf16(routed_output, row, total);
}

__global__ void expert_gate_up_kernel(
    int format, const void* data, const std::uint16_t* scales,
    const std::uint16_t* hidden, const std::int32_t* expert_ids,
    std::uint32_t top_k, std::uint16_t* output) {
    constexpr std::uint32_t rows = 2 * kQ38MoeIntermediate;
    constexpr std::uint32_t columns = kQ38HiddenWidth;
    const auto row = blockIdx.x;
    const auto route = blockIdx.y;
    const auto token = route / top_k;
    const auto expert = expert_ids[route];
    if (expert < 0 || expert >= static_cast<std::int32_t>(kQ38MoeExperts)) {
        if (threadIdx.x == 0) store_bf16(output, route * rows + row, 0.0f);
        return;
    }
    float sum = 0.0f;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        sum += load_bf16(hidden,
                         static_cast<std::uint64_t>(token) * columns + column) *
               quant_weight(format, data, scales,
                            static_cast<std::uint64_t>(rows) * columns,
                            static_cast<std::uint64_t>(rows) * (columns / 128),
                            columns, expert, row, column);
    sum = block_sum(sum);
    if (threadIdx.x == 0) store_bf16(output, route * rows + row, sum);
}

__global__ void expert_silu_kernel(const std::uint16_t* gate_up,
                                   std::uint16_t* activated,
                                   std::uint64_t routes) {
    const auto route = blockIdx.x;
    if (route >= routes) return;
    for (std::uint32_t index = threadIdx.x; index < kQ38MoeIntermediate;
         index += blockDim.x) {
        const auto base = route * 2 * kQ38MoeIntermediate;
        const float gate = load_bf16(gate_up, base + index);
        const float up =
            load_bf16(gate_up, base + kQ38MoeIntermediate + index);
        store_bf16(activated, route * kQ38MoeIntermediate + index,
                   gate / (1.0f + expf(-gate)) * up);
    }
}

__global__ void expert_down_kernel(
    int format, const void* data, const std::uint16_t* scales,
    const std::uint16_t* activated, const std::int32_t* expert_ids,
    const float* expert_weights, std::uint32_t top_k, float* accumulation) {
    constexpr std::uint32_t rows = kQ38HiddenWidth;
    constexpr std::uint32_t columns = kQ38MoeIntermediate;
    const auto row = blockIdx.x;
    const auto route = blockIdx.y;
    const auto token = route / top_k;
    const auto expert = expert_ids[route];
    if (expert < 0 || expert >= static_cast<std::int32_t>(kQ38MoeExperts))
        return;
    float sum = 0.0f;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        sum += load_bf16(activated,
                         static_cast<std::uint64_t>(route) * columns + column) *
               quant_weight(format, data, scales,
                            static_cast<std::uint64_t>(rows) * columns,
                            static_cast<std::uint64_t>(rows) * (columns / 128),
                            columns, expert, row, column);
    sum = block_sum(sum);
    if (threadIdx.x == 0)
        atomicAdd(accumulation +
                      static_cast<std::uint64_t>(token) * rows + row,
                  sum * expert_weights[route]);
}

__global__ void float_to_bf16_kernel(const float* input,
                                      std::uint16_t* output,
                                      std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) store_bf16(output, index, input[index]);
}

__global__ void combine_shared_kernel(const std::uint16_t* routed,
                                      const std::uint16_t* shared,
                                      const std::uint16_t* gate,
                                      std::uint16_t* output,
                                      std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const auto token = index / kQ38HiddenWidth;
    const float gate_value = load_bf16(gate, token);
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    store_bf16(output, index,
               load_bf16(routed, index) + load_bf16(shared, index) * sigmoid);
}

void validate_expert_tensor(const CudaTensorViewV1& tensor,
                            std::uint32_t rows, std::uint32_t columns) {
    if (tensor.empty() || !tensor.data || !tensor.scales ||
        tensor.descriptor->shape.size() != 3 ||
        tensor.descriptor->shape[0] != kQ38MoeExperts ||
        tensor.descriptor->shape[1] != rows ||
        tensor.descriptor->shape[2] != columns ||
        tensor.descriptor->group_size != 128 ||
        (tensor.descriptor->format != DeviceWeightFormatV1::kW4A16SymG128 &&
         tensor.descriptor->format != DeviceWeightFormatV1::kW8A16SymG128))
        throw std::invalid_argument("invalid routed expert tensor");
}

}  // namespace

void cuda_moe_routed_bf16(
    const CudaTensorViewV1& gate_up_experts,
    const CudaTensorViewV1& down_experts, const std::uint16_t* hidden,
    const std::int32_t* expert_ids, const float* expert_weights,
    std::uint32_t tokens, std::uint32_t top_k,
    std::uint16_t* gate_up_scratch, std::uint16_t* activated_scratch,
    float* accumulation_scratch, std::uint16_t* routed_output, void* stream,
    int device) {
    validate_expert_tensor(gate_up_experts, 2 * kQ38MoeIntermediate,
                           kQ38HiddenWidth);
    validate_expert_tensor(down_experts, kQ38HiddenWidth,
                           kQ38MoeIntermediate);
    if (!hidden || !expert_ids || !expert_weights || tokens == 0 || top_k == 0 ||
        top_k > kQ38MoeTopK || !gate_up_scratch || !activated_scratch ||
        !accumulation_scratch || !routed_output || !stream)
        throw std::invalid_argument("invalid routed MoE buffers");
    if (static_cast<std::uint64_t>(tokens) * top_k > 65535)
        throw std::invalid_argument("routed MoE chunk exceeds CUDA grid");
    select_device(device);
    const auto routes = static_cast<std::uint32_t>(tokens * top_k);
    const auto stream_value = reinterpret_cast<cudaStream_t>(stream);
    if (tokens == 1 && optimized_decode_moe_enabled()) {
        const auto gate_blocks =
            (2 * kQ38MoeIntermediate + kDecodeWarpsPerBlock - 1) /
            kDecodeWarpsPerBlock;
        const dim3 gate_grid(gate_blocks, top_k);
        const auto hidden_shared_bytes =
            static_cast<std::size_t>(kQ38HiddenWidth) *
            sizeof(std::uint16_t);
        if (gate_up_experts.descriptor->format ==
            DeviceWeightFormatV1::kW8A16SymG128) {
            expert_gate_up_decode_kernel<8>
                <<<gate_grid, kDecodeThreads, hidden_shared_bytes,
                   stream_value>>>(
                    gate_up_experts.data,
                    static_cast<const std::uint16_t*>(gate_up_experts.scales),
                    hidden, expert_ids, top_k, gate_up_scratch);
        } else {
            expert_gate_up_decode_kernel<4>
                <<<gate_grid, kDecodeThreads, hidden_shared_bytes,
                   stream_value>>>(
                    gate_up_experts.data,
                    static_cast<const std::uint16_t*>(gate_up_experts.scales),
                    hidden, expert_ids, top_k, gate_up_scratch);
        }
        expert_silu_kernel<<<top_k, 256, 0, stream_value>>>(
            gate_up_scratch, activated_scratch, top_k);

        const auto down_blocks =
            (kQ38HiddenWidth + kDecodeWarpsPerBlock - 1) /
            kDecodeWarpsPerBlock;
        const auto down_shared_bytes =
            static_cast<std::size_t>(kQ38MoeIntermediate) *
            sizeof(std::uint16_t);
        if (down_experts.descriptor->format ==
            DeviceWeightFormatV1::kW8A16SymG128) {
            expert_down_decode_kernel<8>
                <<<down_blocks, kDecodeThreads, down_shared_bytes,
                   stream_value>>>(
                    down_experts.data,
                    static_cast<const std::uint16_t*>(down_experts.scales),
                    activated_scratch, expert_ids, expert_weights, top_k,
                    routed_output);
        } else {
            expert_down_decode_kernel<4>
                <<<down_blocks, kDecodeThreads, down_shared_bytes,
                   stream_value>>>(
                    down_experts.data,
                    static_cast<const std::uint16_t*>(down_experts.scales),
                    activated_scratch, expert_ids, expert_weights, top_k,
                    routed_output);
        }
        check(cudaPeekAtLastError(), "decode routed MoE kernels");
        return;
    }
    const auto accumulation_elements =
        static_cast<std::uint64_t>(tokens) * kQ38HiddenWidth;
    check(cudaMemsetAsync(accumulation_scratch, 0,
                          accumulation_elements * sizeof(float),
                          stream_value),
          "cudaMemsetAsync(MoE accumulation)");
    expert_gate_up_kernel<<<dim3(2 * kQ38MoeIntermediate, routes), 256, 0,
                              stream_value>>>(
        static_cast<int>(gate_up_experts.descriptor->format),
        gate_up_experts.data,
        static_cast<const std::uint16_t*>(gate_up_experts.scales), hidden,
        expert_ids, top_k, gate_up_scratch);
    expert_silu_kernel<<<routes, 256, 0, stream_value>>>(
        gate_up_scratch, activated_scratch, routes);
    expert_down_kernel<<<dim3(kQ38HiddenWidth, routes), 256, 0,
                           stream_value>>>(
        static_cast<int>(down_experts.descriptor->format), down_experts.data,
        static_cast<const std::uint16_t*>(down_experts.scales),
        activated_scratch, expert_ids, expert_weights, top_k,
        accumulation_scratch);
    const auto blocks = static_cast<unsigned>((accumulation_elements + 255) / 256);
    float_to_bf16_kernel<<<blocks, 256, 0, stream_value>>>(
        accumulation_scratch, routed_output, accumulation_elements);
    check(cudaPeekAtLastError(), "routed MoE kernels");
}

void cuda_moe_combine_shared_bf16(
    const std::uint16_t* routed_output,
    const std::uint16_t* shared_output,
    const std::uint16_t* shared_gate, std::uint16_t* output,
    std::uint32_t tokens, void* stream, int device) {
    if (!routed_output || !shared_output || !shared_gate || !output || !stream ||
        tokens == 0)
        throw std::invalid_argument("invalid shared MoE buffers");
    select_device(device);
    const auto elements = static_cast<std::uint64_t>(tokens) * kQ38HiddenWidth;
    combine_shared_kernel<<<static_cast<unsigned>((elements + 255) / 256), 256,
                            0, reinterpret_cast<cudaStream_t>(stream)>>>(
        routed_output, shared_output, shared_gate, output, elements);
    check(cudaPeekAtLastError(), "combine_shared_kernel");
}

bool cuda_q38_moe_compiled() { return true; }

}  // namespace q38
