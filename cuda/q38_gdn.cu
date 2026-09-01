#include "q38/cuda_gdn.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
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
    if (device < 0) throw std::invalid_argument("invalid CUDA GDN device");
    check(cudaSetDevice(device), "cudaSetDevice(GDN)");
}

__device__ __forceinline__ float bf16_load(const std::uint16_t* source,
                                            std::uint64_t index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(source)[index]);
}

__device__ __forceinline__ void bf16_store(std::uint16_t* destination,
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

__device__ float block_sum_128(float value) {
    __shared__ float partial[4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < 4 ? partial[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__global__ void gdn_conv_decode_kernel(
    const std::uint16_t* projected, const std::uint16_t* weight,
    std::uint16_t* state, std::uint16_t* activated) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (channel >= kQ38GdnQkvWidth) return;
    const auto base = static_cast<std::uint64_t>(channel) * kQ38GdnConvWidth;
    float convolved = 0.0f;
    for (std::uint32_t index = 0; index + 1 < kQ38GdnConvWidth; ++index) {
        const float shifted = bf16_load(state, base + index + 1);
        bf16_store(state, base + index, shifted);
        convolved += shifted * bf16_load(weight, base + index);
    }
    const float newest = bf16_load(projected, channel);
    bf16_store(state, base + kQ38GdnConvWidth - 1, newest);
    convolved += newest *
                 bf16_load(weight, base + kQ38GdnConvWidth - 1);
    const float silu = convolved / (1.0f + expf(-convolved));
    bf16_store(activated, channel, silu);
}

__global__ void gdn_conv_prefill_kernel(
    const std::uint16_t* projected, const std::uint16_t* weight,
    std::uint16_t* state, std::uint16_t* activated, std::uint32_t tokens) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (channel >= kQ38GdnQkvWidth) return;
    const auto state_base =
        static_cast<std::uint64_t>(channel) * kQ38GdnConvWidth;
    const auto weight_base = state_base;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        float convolved = 0.0f;
        for (std::uint32_t index = 0; index + 1 < kQ38GdnConvWidth; ++index) {
            const float shifted = bf16_load(state, state_base + index + 1);
            bf16_store(state, state_base + index, shifted);
            convolved += shifted * bf16_load(weight, weight_base + index);
        }
        const auto token_offset =
            static_cast<std::uint64_t>(token) * kQ38GdnQkvWidth + channel;
        const float newest = bf16_load(projected, token_offset);
        bf16_store(state, state_base + kQ38GdnConvWidth - 1, newest);
        convolved += newest *
                     bf16_load(weight, weight_base + kQ38GdnConvWidth - 1);
        bf16_store(activated, token_offset,
                   convolved / (1.0f + expf(-convolved)));
    }
}

__global__ void gdn_recurrent_decode_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* state, std::uint16_t* output) {
    const auto value_head = blockIdx.x;
    const auto element = threadIdx.x;
    const auto key_head = value_head / (kQ38GdnValueHeads / kQ38GdnKeyHeads);
    const auto query_base = key_head * kQ38GdnHeadWidth;
    const auto key_base = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_base;
    const auto value_base = 2 * kQ38GdnKeyHeads * kQ38GdnHeadWidth +
                            value_head * kQ38GdnHeadWidth;
    const float query_raw = bf16_load(qkv, query_base + element);
    const float key_raw = bf16_load(qkv, key_base + element);
    float query_norm = block_sum_128(query_raw * query_raw);
    float key_norm = block_sum_128(key_raw * key_raw);
    __shared__ float query_scale;
    __shared__ float key_scale;
    __shared__ float decay;
    __shared__ float beta;
    if (threadIdx.x == 0) {
        query_scale = rsqrtf(query_norm + 1.0e-6f) *
                      rsqrtf(static_cast<float>(kQ38GdnHeadWidth));
        key_scale = rsqrtf(key_norm + 1.0e-6f);
        const float a = bf16_load(projected_a, value_head);
        const float b = bf16_load(projected_b, value_head);
        const float shifted_a = a + bf16_load(dt_bias, value_head);
        const float softplus = shifted_a > 20.0f
                                   ? shifted_a
                                   : log1pf(expf(shifted_a));
        decay = expf(-expf(bf16_load(a_log, value_head)) * softplus);
        beta = 1.0f / (1.0f + expf(-b));
    }
    __syncthreads();
    float memory_value = 0.0f;
    const auto state_base = static_cast<std::uint64_t>(value_head) *
                            kQ38GdnHeadWidth * kQ38GdnHeadWidth;
    for (std::uint32_t key_element = 0; key_element < kQ38GdnHeadWidth;
         ++key_element) {
        const auto state_index = state_base +
                                 static_cast<std::uint64_t>(key_element) *
                                     kQ38GdnHeadWidth +
                                 element;
        const float decayed = state[state_index] * decay;
        state[state_index] = decayed;
        const float key = bf16_load(qkv, key_base + key_element) * key_scale;
        memory_value += decayed * key;
    }
    const float value = bf16_load(qkv, value_base + element);
    const float delta = (value - memory_value) * beta;
    float core = 0.0f;
    for (std::uint32_t key_element = 0; key_element < kQ38GdnHeadWidth;
         ++key_element) {
        const auto state_index = state_base +
                                 static_cast<std::uint64_t>(key_element) *
                                     kQ38GdnHeadWidth +
                                 element;
        const float key = bf16_load(qkv, key_base + key_element) * key_scale;
        const float updated = state[state_index] + key * delta;
        state[state_index] = updated;
        const float query_element =
            bf16_load(qkv, query_base + key_element) * query_scale;
        core += updated * query_element;
    }
    bf16_store(output, value_head * kQ38GdnHeadWidth + element, core);
}

__global__ void gdn_recurrent_prefill_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* state, std::uint16_t* output,
    std::uint32_t tokens) {
    const auto value_head = blockIdx.x;
    const auto element = threadIdx.x;
    const auto key_head = value_head / (kQ38GdnValueHeads / kQ38GdnKeyHeads);
    const auto query_local = key_head * kQ38GdnHeadWidth;
    const auto key_local = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_local;
    const auto value_local = 2 * kQ38GdnKeyHeads * kQ38GdnHeadWidth +
                             value_head * kQ38GdnHeadWidth;
    const auto state_base = static_cast<std::uint64_t>(value_head) *
                            kQ38GdnHeadWidth * kQ38GdnHeadWidth;
    __shared__ float query_scale;
    __shared__ float key_scale;
    __shared__ float decay;
    __shared__ float beta;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto token_base =
            static_cast<std::uint64_t>(token) * kQ38GdnQkvWidth;
        const float query_raw = bf16_load(qkv, token_base + query_local + element);
        const float key_raw = bf16_load(qkv, token_base + key_local + element);
        const float query_norm = block_sum_128(query_raw * query_raw);
        const float key_norm = block_sum_128(key_raw * key_raw);
        if (threadIdx.x == 0) {
            query_scale = rsqrtf(query_norm + 1.0e-6f) *
                          rsqrtf(static_cast<float>(kQ38GdnHeadWidth));
            key_scale = rsqrtf(key_norm + 1.0e-6f);
            const float a = bf16_load(
                projected_a,
                static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
                    value_head);
            const float b = bf16_load(
                projected_b,
                static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
                    value_head);
            const float shifted_a = a + bf16_load(dt_bias, value_head);
            const float softplus = shifted_a > 20.0f
                                       ? shifted_a
                                       : log1pf(expf(shifted_a));
            decay = expf(-expf(bf16_load(a_log, value_head)) * softplus);
            beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();
        float memory_value = 0.0f;
        for (std::uint32_t key_element = 0;
             key_element < kQ38GdnHeadWidth; ++key_element) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                element;
            const float decayed = state[state_index] * decay;
            state[state_index] = decayed;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            memory_value += decayed * key;
        }
        const float value = bf16_load(qkv, token_base + value_local + element);
        const float delta = (value - memory_value) * beta;
        float core = 0.0f;
        for (std::uint32_t key_element = 0;
             key_element < kQ38GdnHeadWidth; ++key_element) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                element;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            const float updated = state[state_index] + key * delta;
            state[state_index] = updated;
            const float query =
                bf16_load(qkv, token_base + query_local + key_element) *
                query_scale;
            core += updated * query;
        }
        bf16_store(output,
                   static_cast<std::uint64_t>(token) * kQ38GdnValueWidth +
                       value_head * kQ38GdnHeadWidth + element,
                   core);
        __syncthreads();
    }
}

constexpr std::uint32_t kGdnPrefillPartitions = 4;
constexpr std::uint32_t kGdnPartitionColumns =
    kQ38GdnHeadWidth / kGdnPrefillPartitions;

// The serial prefill kernel assigns one 128-thread CTA to a value head, which
// exposes only 24 CTAs on the whole device.  This variant tiles the recurrent
// state by output columns: four independent 1024-thread CTAs per head keep the
// state accesses coalesced while exposing 96 long-lived CTAs.  Each warp owns
// one key row across 32 adjacent output columns.
__global__ void gdn_recurrent_prefill_partitioned_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* state, std::uint16_t* output,
    std::uint32_t tokens) {
    const auto value_head = blockIdx.x;
    const auto partition = blockIdx.y;
    const auto lane = threadIdx.x & 31;
    const auto key_lane = threadIdx.x >> 5;
    const auto output_element = partition * kGdnPartitionColumns + lane;
    const auto key_head =
        value_head / (kQ38GdnValueHeads / kQ38GdnKeyHeads);
    const auto query_local = key_head * kQ38GdnHeadWidth;
    const auto key_local = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_local;
    const auto value_local = 2 * kQ38GdnKeyHeads * kQ38GdnHeadWidth +
                             value_head * kQ38GdnHeadWidth;
    const auto state_base = static_cast<std::uint64_t>(value_head) *
                            kQ38GdnHeadWidth * kQ38GdnHeadWidth;

    __shared__ float partial[kQ38GdnHeadWidth * kGdnPartitionColumns];
    __shared__ float query_norm_partials[4];
    __shared__ float key_norm_partials[4];
    __shared__ float deltas[kGdnPartitionColumns];
    __shared__ float query_scale;
    __shared__ float key_scale;
    __shared__ float decay;
    __shared__ float beta;

    for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto token_base =
            static_cast<std::uint64_t>(token) * kQ38GdnQkvWidth;
        float query_square = 0.0f;
        float key_square = 0.0f;
        if (threadIdx.x < kQ38GdnHeadWidth) {
            const float query_raw =
                bf16_load(qkv, token_base + query_local + threadIdx.x);
            const float key_raw =
                bf16_load(qkv, token_base + key_local + threadIdx.x);
            query_square = query_raw * query_raw;
            key_square = key_raw * key_raw;
        }
        query_square = warp_sum(query_square);
        key_square = warp_sum(key_square);
        if (threadIdx.x < kQ38GdnHeadWidth && lane == 0) {
            query_norm_partials[key_lane] = query_square;
            key_norm_partials[key_lane] = key_square;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float query_norm = 0.0f;
            float key_norm = 0.0f;
#pragma unroll
            for (std::uint32_t warp = 0; warp < 4; ++warp) {
                query_norm += query_norm_partials[warp];
                key_norm += key_norm_partials[warp];
            }
            query_scale = rsqrtf(query_norm + 1.0e-6f) *
                          rsqrtf(static_cast<float>(kQ38GdnHeadWidth));
            key_scale = rsqrtf(key_norm + 1.0e-6f);
            const float a = bf16_load(
                projected_a,
                static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
                    value_head);
            const float b = bf16_load(
                projected_b,
                static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
                    value_head);
            const float shifted_a = a + bf16_load(dt_bias, value_head);
            const float softplus = shifted_a > 20.0f
                                       ? shifted_a
                                       : log1pf(expf(shifted_a));
            decay =
                expf(-expf(bf16_load(a_log, value_head)) * softplus);
            beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();

        float memory_partial = 0.0f;
        for (std::uint32_t key_element = key_lane;
             key_element < kQ38GdnHeadWidth; key_element += 32) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                output_element;
            const float decayed = state[state_index] * decay;
            state[state_index] = decayed;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            memory_partial += decayed * key;
        }
        partial[key_lane * kGdnPartitionColumns + lane] = memory_partial;
        __syncthreads();
        if (key_lane == 0) {
            float memory = 0.0f;
#pragma unroll
            for (std::uint32_t row = 0; row < 32; ++row)
                memory += partial[row * kGdnPartitionColumns + lane];
            const float value =
                bf16_load(qkv, token_base + value_local + output_element);
            deltas[lane] = (value - memory) * beta;
        }
        __syncthreads();

        float core_partial = 0.0f;
        const float delta = deltas[lane];
        for (std::uint32_t key_element = key_lane;
             key_element < kQ38GdnHeadWidth; key_element += 32) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                output_element;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            const float updated = state[state_index] + key * delta;
            state[state_index] = updated;
            const float query =
                bf16_load(qkv, token_base + query_local + key_element) *
                query_scale;
            core_partial += updated * query;
        }
        partial[key_lane * kGdnPartitionColumns + lane] = core_partial;
        __syncthreads();
        if (key_lane == 0) {
            float core = 0.0f;
#pragma unroll
            for (std::uint32_t row = 0; row < 32; ++row)
                core += partial[row * kGdnPartitionColumns + lane];
            bf16_store(output,
                       static_cast<std::uint64_t>(token) *
                               kQ38GdnValueWidth +
                           value_head * kQ38GdnHeadWidth + output_element,
                       core);
        }
        __syncthreads();
    }
}

__global__ void gdn_prepare_prefill_parameters_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* query_scales, float* key_scales,
    float* decays, float* betas) {
    const auto key_head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto token_base =
        static_cast<std::uint64_t>(token) * kQ38GdnQkvWidth;
    const auto query_local = key_head * kQ38GdnHeadWidth;
    const auto key_local = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_local;
    const float query_raw =
        bf16_load(qkv, token_base + query_local + element);
    const float key_raw = bf16_load(qkv, token_base + key_local + element);
    const auto lane = threadIdx.x & 31;
    const auto warp = threadIdx.x >> 5;
    __shared__ float query_partials[4];
    __shared__ float key_partials[4];
    const float query_partial = warp_sum(query_raw * query_raw);
    const float key_partial = warp_sum(key_raw * key_raw);
    if (lane == 0) {
        query_partials[warp] = query_partial;
        key_partials[warp] = key_partial;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    float query_norm = 0.0f;
    float key_norm = 0.0f;
#pragma unroll
    for (std::uint32_t index = 0; index < 4; ++index) {
        query_norm += query_partials[index];
        key_norm += key_partials[index];
    }
    const auto key_scale_index =
        static_cast<std::uint64_t>(token) * kQ38GdnKeyHeads + key_head;
    query_scales[key_scale_index] =
        rsqrtf(query_norm + 1.0e-6f) *
        rsqrtf(static_cast<float>(kQ38GdnHeadWidth));
    key_scales[key_scale_index] = rsqrtf(key_norm + 1.0e-6f);

#pragma unroll
    for (std::uint32_t local = 0;
         local < kQ38GdnValueHeads / kQ38GdnKeyHeads; ++local) {
        const auto value_head =
            key_head * (kQ38GdnValueHeads / kQ38GdnKeyHeads) + local;
        const auto value_index =
            static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
            value_head;
        const float a = bf16_load(projected_a, value_index);
        const float b = bf16_load(projected_b, value_index);
        const float shifted_a = a + bf16_load(dt_bias, value_head);
        const float softplus = shifted_a > 20.0f
                                   ? shifted_a
                                   : log1pf(expf(shifted_a));
        decays[value_index] =
            expf(-expf(bf16_load(a_log, value_head)) * softplus);
        betas[value_index] = 1.0f / (1.0f + expf(-b));
    }
}

__global__ void gdn_recurrent_prefill_precomputed_kernel(
    const std::uint16_t* qkv, const float* query_scales,
    const float* key_scales, const float* decays, const float* betas,
    float* state, std::uint16_t* output, std::uint32_t tokens) {
    const auto value_head = blockIdx.x;
    const auto partition = blockIdx.y;
    const auto lane = threadIdx.x & 31;
    const auto key_lane = threadIdx.x >> 5;
    const auto output_element = partition * kGdnPartitionColumns + lane;
    const auto key_head =
        value_head / (kQ38GdnValueHeads / kQ38GdnKeyHeads);
    const auto query_local = key_head * kQ38GdnHeadWidth;
    const auto key_local = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_local;
    const auto value_local = 2 * kQ38GdnKeyHeads * kQ38GdnHeadWidth +
                             value_head * kQ38GdnHeadWidth;
    const auto state_base = static_cast<std::uint64_t>(value_head) *
                            kQ38GdnHeadWidth * kQ38GdnHeadWidth;
    __shared__ float partial[kQ38GdnHeadWidth * kGdnPartitionColumns];
    __shared__ float deltas[kGdnPartitionColumns];

    for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto token_base =
            static_cast<std::uint64_t>(token) * kQ38GdnQkvWidth;
        const auto key_scale_index =
            static_cast<std::uint64_t>(token) * kQ38GdnKeyHeads + key_head;
        const auto value_index =
            static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
            value_head;
        const float query_scale = query_scales[key_scale_index];
        const float key_scale = key_scales[key_scale_index];
        const float decay = decays[value_index];
        const float beta = betas[value_index];

        float memory_partial = 0.0f;
        for (std::uint32_t key_element = key_lane;
             key_element < kQ38GdnHeadWidth; key_element += 32) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                output_element;
            const float decayed = state[state_index] * decay;
            state[state_index] = decayed;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            memory_partial += decayed * key;
        }
        partial[key_lane * kGdnPartitionColumns + lane] = memory_partial;
        __syncthreads();
        if (key_lane == 0) {
            float memory = 0.0f;
#pragma unroll
            for (std::uint32_t row = 0; row < 32; ++row)
                memory += partial[row * kGdnPartitionColumns + lane];
            const float value =
                bf16_load(qkv, token_base + value_local + output_element);
            deltas[lane] = (value - memory) * beta;
        }
        __syncthreads();

        float core_partial = 0.0f;
        const float delta = deltas[lane];
        for (std::uint32_t key_element = key_lane;
             key_element < kQ38GdnHeadWidth; key_element += 32) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                output_element;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            const float updated = state[state_index] + key * delta;
            state[state_index] = updated;
            const float query =
                bf16_load(qkv, token_base + query_local + key_element) *
                query_scale;
            core_partial += updated * query;
        }
        partial[key_lane * kGdnPartitionColumns + lane] = core_partial;
        __syncthreads();
        if (key_lane == 0) {
            float core = 0.0f;
#pragma unroll
            for (std::uint32_t row = 0; row < 32; ++row)
                core += partial[row * kGdnPartitionColumns + lane];
            bf16_store(output,
                       static_cast<std::uint64_t>(token) *
                               kQ38GdnValueWidth +
                           value_head * kQ38GdnHeadWidth + output_element,
                       core);
        }
        __syncthreads();
    }
}

__global__ void gdn_output_norm_kernel(
    const std::uint16_t* core, const std::uint16_t* gate,
    const std::uint16_t* weight, std::uint16_t* output, float epsilon) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto index = static_cast<std::uint64_t>(token) * kQ38GdnValueWidth +
                       head * kQ38GdnHeadWidth + element;
    const float value = bf16_load(core, index);
    float squares = block_sum_128(value * value);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0)
        inverse_rms = rsqrtf(squares / kQ38GdnHeadWidth + epsilon);
    __syncthreads();
    const float gate_value = bf16_load(gate, index);
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    bf16_store(output, index,
               value * inverse_rms * bf16_load(weight, element) * sigmoid);
}

}  // namespace

struct CudaGdnStateBank::Impl {
    int device;
    std::uint32_t layers;
    std::byte* banks[2]{nullptr, nullptr};
    std::byte* checkpoint = nullptr;
    std::uint32_t committed = 0;
    bool active = false;
    bool checkpoint_valid = false;
    std::uint64_t active_epoch = 0;
    std::uint64_t conv_layer_bytes =
        static_cast<std::uint64_t>(kQ38GdnQkvWidth) * kQ38GdnConvWidth * 2;
    std::uint64_t recurrent_layer_bytes =
        static_cast<std::uint64_t>(kQ38GdnValueHeads) * kQ38GdnHeadWidth *
        kQ38GdnHeadWidth * sizeof(float);
    std::uint64_t bank_bytes = 0;

    Impl(int value_device, std::uint32_t value_layers,
         bool enable_checkpoint)
        : device(value_device), layers(value_layers) {
        if (layers == 0) throw std::invalid_argument("GDN state bank is empty");
        if (layers > std::numeric_limits<std::uint64_t>::max() /
                         (conv_layer_bytes + recurrent_layer_bytes))
            throw std::overflow_error("GDN state bank size overflows");
        bank_bytes = layers * (conv_layer_bytes + recurrent_layer_bytes);
        if (bank_bytes > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("GDN state bank exceeds address space");
        select_device(device);
        try {
            check(cudaMalloc(reinterpret_cast<void**>(&banks[0]), bank_bytes),
                  "cudaMalloc(GDN bank 0)");
            check(cudaMalloc(reinterpret_cast<void**>(&banks[1]), bank_bytes),
                  "cudaMalloc(GDN bank 1)");
            if (enable_checkpoint)
                check(cudaMalloc(reinterpret_cast<void**>(&checkpoint),
                                 bank_bytes),
                      "cudaMalloc(GDN checkpoint)");
            check(cudaMemset(banks[0], 0, bank_bytes), "cudaMemset(GDN bank 0)");
            check(cudaMemset(banks[1], 0, bank_bytes), "cudaMemset(GDN bank 1)");
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

    std::byte* layer_base(std::uint32_t bank, std::uint32_t layer) const {
        return banks[bank] +
               static_cast<std::uint64_t>(layer) *
                   (conv_layer_bytes + recurrent_layer_bytes);
    }
};

CudaGdnStateBank::CudaGdnStateBank(int device, std::uint32_t layers,
                                   bool enable_checkpoint)
    : impl_(std::make_unique<Impl>(device, layers, enable_checkpoint)) {}
CudaGdnStateBank::~CudaGdnStateBank() = default;
CudaGdnStateBank::CudaGdnStateBank(CudaGdnStateBank&&) noexcept = default;
CudaGdnStateBank& CudaGdnStateBank::operator=(CudaGdnStateBank&&) noexcept =
    default;

void CudaGdnStateBank::begin(std::uint64_t epoch, void* stream) {
    if (!stream || epoch == 0 || impl_->active)
        throw std::logic_error("invalid GDN transaction begin");
    select_device(impl_->device);
    const auto working = 1u - impl_->committed;
    check(cudaMemcpyAsync(impl_->banks[working], impl_->banks[impl_->committed],
                          impl_->bank_bytes, cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(GDN transaction clone)");
    impl_->active = true;
    impl_->active_epoch = epoch;
    impl_->checkpoint_valid = false;
}

void CudaGdnStateBank::restore(void* stream) {
    if (!stream || !impl_->active)
        throw std::logic_error("invalid GDN transaction restore");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->banks[1u - impl_->committed],
                          impl_->banks[impl_->committed], impl_->bank_bytes,
                          cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(GDN transaction restore)");
}

void CudaGdnStateBank::checkpoint(void* stream) {
    if (!stream || !impl_->active || !impl_->checkpoint)
        throw std::logic_error("invalid GDN speculative checkpoint");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->checkpoint,
                          impl_->banks[1u - impl_->committed],
                          impl_->bank_bytes, cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(GDN speculative checkpoint)");
    impl_->checkpoint_valid = true;
}

void CudaGdnStateBank::restore_checkpoint(void* stream) {
    if (!stream || !impl_->active || !impl_->checkpoint_valid)
        throw std::logic_error("invalid GDN checkpoint restore");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->banks[1u - impl_->committed],
                          impl_->checkpoint, impl_->bank_bytes,
                          cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(GDN checkpoint restore)");
}

CudaGdnLayerStateView CudaGdnStateBank::working(
    std::uint32_t local_layer) const {
    if (!impl_->active || local_layer >= impl_->layers)
        throw std::logic_error("invalid GDN working state request");
    auto* base = impl_->layer_base(1u - impl_->committed, local_layer);
    return CudaGdnLayerStateView{
        reinterpret_cast<std::uint16_t*>(base),
        reinterpret_cast<float*>(base + impl_->conv_layer_bytes)};
}

void CudaGdnStateBank::commit(std::uint64_t epoch) {
    if (!impl_->active || impl_->active_epoch != epoch)
        throw std::logic_error("GDN commit epoch mismatch");
    impl_->committed = 1u - impl_->committed;
    impl_->active = false;
    impl_->active_epoch = 0;
    impl_->checkpoint_valid = false;
}

void CudaGdnStateBank::rollback(std::uint64_t epoch) {
    if (!impl_->active || impl_->active_epoch != epoch)
        throw std::logic_error("GDN rollback epoch mismatch");
    impl_->active = false;
    impl_->active_epoch = 0;
    impl_->checkpoint_valid = false;
}

void CudaGdnStateBank::reset(void* stream) {
    if (!stream || impl_->active)
        throw std::logic_error("invalid GDN reset");
    select_device(impl_->device);
    check(cudaMemsetAsync(impl_->banks[0], 0, impl_->bank_bytes,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemsetAsync(GDN bank 0)");
    check(cudaMemsetAsync(impl_->banks[1], 0, impl_->bank_bytes,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemsetAsync(GDN bank 1)");
    if (impl_->checkpoint)
        check(cudaMemsetAsync(impl_->checkpoint, 0, impl_->bank_bytes,
                              reinterpret_cast<cudaStream_t>(stream)),
              "cudaMemsetAsync(GDN checkpoint)");
    impl_->committed = 0;
    impl_->checkpoint_valid = false;
}

std::uint32_t CudaGdnStateBank::layers() const { return impl_->layers; }
std::uint64_t CudaGdnStateBank::bytes_per_bank() const {
    return impl_->bank_bytes;
}
std::uint64_t CudaGdnStateBank::allocated_bytes() const {
    return impl_->bank_bytes * (impl_->checkpoint ? 3ull : 2ull);
}

void cuda_gdn_conv_decode_bf16(
    const std::uint16_t* projected_qkv, const std::uint16_t* conv_weight,
    std::uint16_t* conv_state, std::uint16_t* activated_qkv, void* stream,
    int device) {
    if (!projected_qkv || !conv_weight || !conv_state || !activated_qkv ||
        !stream)
        throw std::invalid_argument("invalid GDN convolution buffers");
    select_device(device);
    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks = (kQ38GdnQkvWidth + threads - 1) / threads;
    gdn_conv_decode_kernel<<<blocks, threads, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_qkv, conv_weight, conv_state, activated_qkv);
    check(cudaPeekAtLastError(), "gdn_conv_decode_kernel");
}

void cuda_gdn_recurrent_decode_bf16(
    const std::uint16_t* activated_qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* recurrent_state,
    std::uint16_t* core_output, void* stream, int device) {
    if (!activated_qkv || !projected_b || !projected_a || !a_log ||
        !dt_bias || !recurrent_state || !core_output || !stream)
        throw std::invalid_argument("invalid GDN recurrent buffers");
    select_device(device);
    gdn_recurrent_decode_kernel<<<kQ38GdnValueHeads, kQ38GdnHeadWidth, 0,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
        activated_qkv, projected_b, projected_a, a_log, dt_bias,
        recurrent_state, core_output);
    check(cudaPeekAtLastError(), "gdn_recurrent_decode_kernel");
}

void cuda_gdn_output_norm_bf16(
    const std::uint16_t* core_output, const std::uint16_t* gate_z,
    const std::uint16_t* norm_weight, std::uint16_t* output, float epsilon,
    void* stream, int device) {
    if (!core_output || !gate_z || !norm_weight || !output || !stream ||
        !(epsilon > 0.0f))
        throw std::invalid_argument("invalid GDN output norm buffers");
    select_device(device);
    gdn_output_norm_kernel<<<dim3(kQ38GdnValueHeads, 1), kQ38GdnHeadWidth, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        core_output, gate_z, norm_weight, output, epsilon);
    check(cudaPeekAtLastError(), "gdn_output_norm_kernel");
}

void cuda_gdn_conv_prefill_bf16(
    const std::uint16_t* projected_qkv, const std::uint16_t* conv_weight,
    std::uint16_t* conv_state, std::uint16_t* activated_qkv,
    std::uint32_t tokens, void* stream, int device) {
    if (!projected_qkv || !conv_weight || !conv_state || !activated_qkv ||
        !stream || tokens == 0)
        throw std::invalid_argument("invalid GDN prefill convolution buffers");
    select_device(device);
    gdn_conv_prefill_kernel<<<(kQ38GdnQkvWidth + 255) / 256, 256, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_qkv, conv_weight, conv_state, activated_qkv, tokens);
    check(cudaPeekAtLastError(), "GDN prefill convolution");
}

void cuda_gdn_recurrent_prefill_bf16(
    const std::uint16_t* activated_qkv,
    const std::uint16_t* projected_b,
    const std::uint16_t* projected_a,
    const std::uint16_t* a_log,
    const std::uint16_t* dt_bias,
    float* recurrent_state,
    std::uint16_t* core_output,
    std::uint32_t tokens,
    void* stream,
    int device) {
    if (!activated_qkv || !projected_b || !projected_a || !a_log ||
        !dt_bias || !recurrent_state || !core_output || !stream || tokens == 0)
        throw std::invalid_argument("invalid GDN prefill recurrent buffers");
    select_device(device);
    gdn_recurrent_prefill_kernel<<<kQ38GdnValueHeads, kQ38GdnHeadWidth, 0,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
        activated_qkv, projected_b, projected_a, a_log, dt_bias,
        recurrent_state, core_output, tokens);
    check(cudaPeekAtLastError(), "GDN prefill recurrent");
}

void cuda_gdn_recurrent_prefill_partitioned_bf16(
    const std::uint16_t* activated_qkv,
    const std::uint16_t* projected_b,
    const std::uint16_t* projected_a,
    const std::uint16_t* a_log,
    const std::uint16_t* dt_bias,
    float* recurrent_state,
    std::uint16_t* core_output,
    std::uint32_t tokens,
    void* stream,
    int device) {
    if (!activated_qkv || !projected_b || !projected_a || !a_log ||
        !dt_bias || !recurrent_state || !core_output || !stream || tokens == 0)
        throw std::invalid_argument(
            "invalid partitioned GDN prefill recurrent buffers");
    select_device(device);
    gdn_recurrent_prefill_partitioned_kernel<<<
        dim3(kQ38GdnValueHeads, kGdnPrefillPartitions), 1024, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        activated_qkv, projected_b, projected_a, a_log, dt_bias,
        recurrent_state, core_output, tokens);
    check(cudaPeekAtLastError(), "partitioned GDN prefill recurrent");
}

void cuda_gdn_recurrent_prefill_precomputed_bf16(
    const std::uint16_t* activated_qkv,
    const std::uint16_t* projected_b,
    const std::uint16_t* projected_a,
    const std::uint16_t* a_log,
    const std::uint16_t* dt_bias,
    float* recurrent_state,
    float* parameter_scratch,
    std::uint16_t* core_output,
    std::uint32_t tokens,
    void* stream,
    int device) {
    if (!activated_qkv || !projected_b || !projected_a || !a_log ||
        !dt_bias || !recurrent_state || !parameter_scratch || !core_output ||
        !stream || tokens == 0)
        throw std::invalid_argument(
            "invalid precomputed GDN prefill recurrent buffers");
    select_device(device);
    auto* query_scales = parameter_scratch;
    auto* key_scales = query_scales +
                       static_cast<std::size_t>(tokens) * kQ38GdnKeyHeads;
    auto* decays = key_scales +
                   static_cast<std::size_t>(tokens) * kQ38GdnKeyHeads;
    auto* betas = decays +
                  static_cast<std::size_t>(tokens) * kQ38GdnValueHeads;
    const auto stream_value = reinterpret_cast<cudaStream_t>(stream);
    gdn_prepare_prefill_parameters_kernel<<<
        dim3(kQ38GdnKeyHeads, tokens), kQ38GdnHeadWidth, 0, stream_value>>>(
        activated_qkv, projected_b, projected_a, a_log, dt_bias,
        query_scales, key_scales, decays, betas);
    gdn_recurrent_prefill_precomputed_kernel<<<
        dim3(kQ38GdnValueHeads, kGdnPrefillPartitions), 1024, 0,
        stream_value>>>(activated_qkv, query_scales, key_scales, decays,
                        betas, recurrent_state, core_output, tokens);
    check(cudaPeekAtLastError(), "precomputed GDN prefill recurrent");
}

void cuda_gdn_output_norm_prefill_bf16(
    const std::uint16_t* core_output,
    const std::uint16_t* gate_z,
    const std::uint16_t* norm_weight,
    std::uint16_t* output,
    std::uint32_t tokens,
    float epsilon,
    void* stream,
    int device) {
    if (!core_output || !gate_z || !norm_weight || !output || !stream ||
        tokens == 0 || !(epsilon > 0.0f))
        throw std::invalid_argument("invalid GDN prefill norm buffers");
    select_device(device);
    gdn_output_norm_kernel<<<dim3(kQ38GdnValueHeads, tokens),
                             kQ38GdnHeadWidth, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        core_output, gate_z, norm_weight, output, epsilon);
    check(cudaPeekAtLastError(), "GDN prefill output norm");
}

bool cuda_q38_gdn_compiled() { return true; }

}  // namespace q38
