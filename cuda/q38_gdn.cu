#include "q38/cuda_gdn.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <type_traits>
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

__device__ unsigned gdn_canonical_channel(unsigned local, unsigned rank) {
    if (local < 1024) return rank * 1024 + local;
    if (local < 2048) return 2048 + rank * 1024 + local - 1024;
    return 4096 + rank * 3072 + local - 2048;
}

template <bool Snapshot>
__global__ void gdn_conv_prefill_kernel(
    const std::uint16_t* projected, const std::uint16_t* weight,
    std::uint16_t* state, std::uint16_t* activated, std::uint32_t tokens, CudaPrefixSnapshotView snapshots = {}, int local_rank = -1, bool full_projection = false, unsigned projection_stride = 0) {
    const auto local_channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned width = local_rank < 0 ? kQ38GdnQkvWidth : kQ38GdnQkvWidth / 2;
    if (local_channel >= width) return;
    const auto channel = local_rank < 0 ? local_channel : gdn_canonical_channel(local_channel, local_rank);
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
        const float newest = bf16_load(projected, static_cast<std::uint64_t>(token) * (projection_stride?projection_stride:(full_projection?kQ38GdnQkvWidth:width)) + (full_projection?channel:local_channel));
        bf16_store(state, state_base + kQ38GdnConvWidth - 1, newest);
        if constexpr (Snapshot) {
            if (token < snapshots.count) {
                auto* target = reinterpret_cast<std::uint16_t*>(
                    static_cast<std::uint8_t*>(snapshots.data) + token * snapshots.stride_bytes);
                for (std::uint32_t i = 0; i < kQ38GdnConvWidth; ++i)
                    target[state_base + i] = state[state_base + i];
            }
        }
        convolved += newest *
                     bf16_load(weight, weight_base + kQ38GdnConvWidth - 1);
        bf16_store(activated, token_offset,
                   convolved / (1.0f + expf(-convolved)));
    }
}

__global__ void gdn_recurrent_decode_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* state, std::uint16_t* output, unsigned head_begin = 0, unsigned heads = kQ38GdnValueHeads, unsigned control_heads = 0) {
    const auto value_head = blockIdx.x + head_begin;
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
        const float a = bf16_load(projected_a, control_heads==kQ38GdnValueHeads ? value_head : blockIdx.x);
        const float b = bf16_load(projected_b, control_heads==kQ38GdnValueHeads ? value_head : blockIdx.x);
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
    bf16_store(output, blockIdx.x * kQ38GdnHeadWidth + element, core);
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

template <bool NormalizeQk = false>
__global__ void gdn_prepare_prefill_parameters_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* query_scales, float* key_scales,
    float* decays, float* betas, unsigned head_begin = 0, unsigned heads = kQ38GdnValueHeads, unsigned control_heads = 0, unsigned projection_stride = 0,
    float* prepared_queries = nullptr, float* prepared_keys = nullptr) {
    const unsigned stride=projection_stride?projection_stride:(control_heads?control_heads:heads);
    const auto key_head = blockIdx.x + head_begin / 3;
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
    __shared__ float normalized_scales[2];
    const float query_partial = warp_sum(query_raw * query_raw);
    const float key_partial = warp_sum(key_raw * key_raw);
    if (lane == 0) {
        query_partials[warp] = query_partial;
        key_partials[warp] = key_partial;
    }
    __syncthreads();
    if (threadIdx.x == 0) {

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
        const float a = bf16_load(projected_a, static_cast<std::uint64_t>(token) * stride + value_head - (control_heads==kQ38GdnValueHeads?0:head_begin));
        const float b = bf16_load(projected_b, static_cast<std::uint64_t>(token) * stride + value_head - (control_heads==kQ38GdnValueHeads?0:head_begin));
        const float shifted_a = a + bf16_load(dt_bias, value_head);
        const float softplus = shifted_a > 20.0f
                                   ? shifted_a
                                   : log1pf(expf(shifted_a));
        decays[value_index] =
            expf(-expf(bf16_load(a_log, value_head)) * softplus);
        betas[value_index] = 1.0f / (1.0f + expf(-b));
    }
    if constexpr (NormalizeQk) {
        normalized_scales[0] = query_scales[key_scale_index];
        normalized_scales[1] = key_scales[key_scale_index];
    }
    }
    if constexpr (NormalizeQk) {
        __syncthreads();
        const auto index = (static_cast<std::uint64_t>(token) *
                            kQ38GdnKeyHeads + key_head) * kQ38GdnHeadWidth + element;
        prepared_queries[index] = query_raw * normalized_scales[0];
        prepared_keys[index] = key_raw * normalized_scales[1];
    }
}


// Small target slabs: one CTA owns Q/K and their three value heads. Keep
// convolution BF16 boundaries and the original four-warp norm sum order.
template<bool Snapshot>
__global__ void gdn_conv_prepare_kernel(const std::uint16_t* projected,
    const std::uint16_t* weight, std::uint16_t* state, std::uint16_t* activated,
    const std::uint16_t* projected_b, const std::uint16_t* projected_a,
    const std::uint16_t* a_log, const std::uint16_t* dt_bias,
    float* query_scales, float* key_scales, float* decays, float* betas,
    unsigned tokens, unsigned rank, unsigned projection_stride,
    unsigned control_heads, unsigned control_stride, CudaPrefixSnapshotView snapshots) {
    const unsigned plane = threadIdx.x / 128, element = threadIdx.x % 128;
    const unsigned local_head = blockIdx.x;
    const unsigned key_head = rank * 8 + local_head;
    const unsigned local_channel = plane < 2 ? plane * 1024 + local_head * 128 + element
        : 2048 + (local_head * 3 + plane - 2) * 128 + element;
    const unsigned channel = gdn_canonical_channel(local_channel, rank);
    const unsigned state_base = channel * kQ38GdnConvWidth;
    __shared__ float qk[2][128], qp[4], kp[4];
    for (unsigned token = 0; token < tokens; ++token) {
        float convolved = 0.f;
        for (unsigned i = 0; i + 1 < kQ38GdnConvWidth; ++i) {
            const float shifted = bf16_load(state,state_base+i+1);
            bf16_store(state,state_base+i,shifted);
            convolved += shifted * bf16_load(weight,state_base+i);
        }
        const float newest = bf16_load(projected,token * projection_stride + local_channel);
        bf16_store(state,state_base+kQ38GdnConvWidth-1,newest);
        if constexpr(Snapshot) if(token<snapshots.count) {
            auto* target=reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(snapshots.data)+token*snapshots.stride_bytes);
            for(unsigned i=0;i<kQ38GdnConvWidth;++i)target[state_base+i]=state[state_base+i];
        }
        convolved += newest * bf16_load(weight,state_base+kQ38GdnConvWidth-1);
        const float value=__bfloat162float(__float2bfloat16_rn(convolved/(1.f+expf(-convolved))));
        bf16_store(activated,std::uint64_t(token)*kQ38GdnQkvWidth+channel,value);
        if(plane<2)qk[plane][element]=value;
        __syncthreads();
        if(plane==0){
            const float q=warp_sum(qk[0][element]*qk[0][element]);
            const float k=warp_sum(qk[1][element]*qk[1][element]);
            if((element&31)==0){qp[element/32]=q;kp[element/32]=k;}
        }
        __syncthreads();
        if(threadIdx.x==0){
            float qn=0.f,kn=0.f;
#pragma unroll
            for(unsigned i=0;i<4;++i){qn+=qp[i];kn+=kp[i];}
            query_scales[token*kQ38GdnKeyHeads+key_head]=rsqrtf(qn+1e-6f)*rsqrtf(float(kQ38GdnHeadWidth));
            key_scales[token*kQ38GdnKeyHeads+key_head]=rsqrtf(kn+1e-6f);
#pragma unroll
            for(unsigned local=0;local<3;++local){
                const unsigned h=key_head*3+local;
                const unsigned c=token*control_stride+h-(control_heads==48?0:rank*24);
                const float a=bf16_load(projected_a,c),b=bf16_load(projected_b,c);
                const float shifted_a=a+bf16_load(dt_bias,h);
                const float softplus=shifted_a>20.f?shifted_a:log1pf(expf(shifted_a));
                decays[token*kQ38GdnValueHeads+h]=expf(-expf(bf16_load(a_log,h))*softplus);
                betas[token*kQ38GdnValueHeads+h]=1.f/(1.f+expf(-b));
            }
        }
        __syncthreads();
    }
}

template <bool Snapshot, unsigned PartitionColumns = kGdnPartitionColumns,
          bool NormalizeQk = false>
__global__ void gdn_recurrent_prefill_precomputed_kernel(
    const std::uint16_t* qkv, const float* query_scales,
    const float* key_scales, const float* decays, const float* betas,
    float* state, std::uint16_t* output, std::uint32_t tokens, CudaPrefixSnapshotView snapshots = {}, unsigned head_begin = 0, unsigned heads = kQ38GdnValueHeads,
    const float* prepared_queries = nullptr, const float* prepared_keys = nullptr) {
    const auto value_head = blockIdx.x + head_begin;
    const auto partition = blockIdx.y;
    const auto lane = threadIdx.x % PartitionColumns;
    const auto key_lane = threadIdx.x / PartitionColumns;
    const auto output_element = partition * PartitionColumns + lane;
    const auto key_head =
        value_head / (kQ38GdnValueHeads / kQ38GdnKeyHeads);
    const auto query_local = key_head * kQ38GdnHeadWidth;
    const auto key_local = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_local;
    const auto value_local = 2 * kQ38GdnKeyHeads * kQ38GdnHeadWidth +
                             value_head * kQ38GdnHeadWidth;
    const auto state_base = static_cast<std::uint64_t>(value_head) *
                            kQ38GdnHeadWidth * kQ38GdnHeadWidth;
    __shared__ float partial[32 * PartitionColumns];
    __shared__ float deltas[PartitionColumns];

    // One 1024-thread CTA owns a complete 128x32 state-column tile.  Each
    // thread is therefore responsible for exactly four key rows.  Keep those
    // FP32 cells resident for the entire slab instead of round-tripping the
    // whole recurrent state through global memory twice per token.
    float resident_state[4];
#pragma unroll
    for (std::uint32_t item = 0; item < 4; ++item) {
        const auto key_element = key_lane + item * 32;
        const auto state_index =
            state_base + static_cast<std::uint64_t>(key_element) *
                             kQ38GdnHeadWidth +
            output_element;
        resident_state[item] = state[state_index];
    }

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
        float normalized_keys[4];
#pragma unroll
        for (std::uint32_t item = 0; item < 4; ++item) {
            const auto key_element = key_lane + item * 32;
            const float decayed = resident_state[item] * decay;
            resident_state[item] = decayed;
            if constexpr (NormalizeQk)
                normalized_keys[item] = prepared_keys[
                    key_scale_index * kQ38GdnHeadWidth + key_element];
            else
                normalized_keys[item] =
                    bf16_load(qkv, token_base + key_local + key_element) * key_scale;
            memory_partial += decayed * normalized_keys[item];
        }
        partial[key_lane * PartitionColumns + lane] = memory_partial;
        __syncthreads();
        if (key_lane == 0) {
            float memory = 0.0f;
#pragma unroll
            for (std::uint32_t row = 0; row < 32; ++row)
                memory += partial[row * PartitionColumns + lane];
            const float value =
                bf16_load(qkv, token_base + value_local + output_element);
            deltas[lane] = (value - memory) * beta;
        }
        __syncthreads();

        float core_partial = 0.0f;
        const float delta = deltas[lane];
#pragma unroll
        for (std::uint32_t item = 0; item < 4; ++item) {
            const auto key_element = key_lane + item * 32;
            const float updated =
                resident_state[item] + normalized_keys[item] * delta;
            resident_state[item] = updated;
            if constexpr (Snapshot) {
                if (token < snapshots.count) {
                    auto* target = reinterpret_cast<float*>(
                        static_cast<std::uint8_t*>(snapshots.data) + token * snapshots.stride_bytes);
                    target[state_base + static_cast<std::uint64_t>(key_element) *
                        kQ38GdnHeadWidth + output_element] = updated;
                }
            }
            const float query = NormalizeQk
                ? prepared_queries[key_scale_index * kQ38GdnHeadWidth + key_element]
                : bf16_load(qkv, token_base + query_local + key_element) * query_scale;
            core_partial += updated * query;
        }
        partial[key_lane * PartitionColumns + lane] = core_partial;
        __syncthreads();
        if (key_lane == 0) {
            float core = 0.0f;
#pragma unroll
            for (std::uint32_t row = 0; row < 32; ++row)
                core += partial[row * PartitionColumns + lane];
            bf16_store(output,
                       static_cast<std::uint64_t>(token) *
                               heads * kQ38GdnHeadWidth +
                           blockIdx.x * kQ38GdnHeadWidth + output_element,
                       core);
        }
        __syncthreads();
    }

#pragma unroll
    for (std::uint32_t item = 0; item < 4; ++item) {
        const auto key_element = key_lane + item * 32;
        const auto state_index =
            state_base + static_cast<std::uint64_t>(key_element) *
                             kQ38GdnHeadWidth +
            output_element;
        state[state_index] = resident_state[item];
    }
}

__global__ void gdn_output_norm_kernel(
    const std::uint16_t* core, const std::uint16_t* gate,
    const std::uint16_t* weight, std::uint16_t* output, float epsilon, unsigned heads = kQ38GdnValueHeads, unsigned gate_heads = 0, unsigned gate_begin = 0, unsigned projection_stride = 0) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto index = static_cast<std::uint64_t>(token) * heads * kQ38GdnHeadWidth +
                       head * kQ38GdnHeadWidth + element;
    const float value = bf16_load(core, index);
    float squares = block_sum_128(value * value);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0)
        inverse_rms = rsqrtf(squares / kQ38GdnHeadWidth + epsilon);
    __syncthreads();
    const float gate_value = bf16_load(gate, static_cast<std::uint64_t>(token)*(projection_stride?projection_stride:(gate_heads?gate_heads:heads)*kQ38GdnHeadWidth) + (head+gate_begin)*kQ38GdnHeadWidth+element);
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
    std::uint32_t tokens, void* stream, int device, CudaPrefixSnapshotView snapshots) {
    if (!projected_qkv || !conv_weight || !conv_state || !activated_qkv ||
        !stream || tokens == 0)
        throw std::invalid_argument("invalid GDN prefill convolution buffers");
    validate_prefix_snapshot(snapshots, tokens, std::uint64_t(kQ38GdnQkvWidth) * kQ38GdnConvWidth * 2, 2);
    select_device(device);
    if (snapshots.data) {
    gdn_conv_prefill_kernel<true><<<(kQ38GdnQkvWidth + 255) / 256, 256, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_qkv, conv_weight, conv_state, activated_qkv, tokens, snapshots);
    } else {
    gdn_conv_prefill_kernel<false><<<(kQ38GdnQkvWidth + 255) / 256, 256, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_qkv, conv_weight, conv_state, activated_qkv, tokens);
    }
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
    int device, CudaPrefixSnapshotView snapshots) {
    if (!activated_qkv || !projected_b || !projected_a || !a_log ||
        !dt_bias || !recurrent_state || !parameter_scratch || !core_output ||
        !stream || tokens == 0)
        throw std::invalid_argument(
            "invalid precomputed GDN prefill recurrent buffers");
    validate_prefix_snapshot(snapshots, tokens, std::uint64_t(kQ38GdnValueHeads) * kQ38GdnHeadWidth * kQ38GdnHeadWidth * 4, 4);
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
    if (snapshots.data) {
    gdn_recurrent_prefill_precomputed_kernel<true><<<
        dim3(kQ38GdnValueHeads, kGdnPrefillPartitions), 1024, 0,
        stream_value>>>(activated_qkv, query_scales, key_scales, decays,
                        betas, recurrent_state, core_output, tokens, snapshots);
    } else {
    gdn_recurrent_prefill_precomputed_kernel<false><<<
        dim3(kQ38GdnValueHeads, kGdnPrefillPartitions), 1024, 0,
        stream_value>>>(activated_qkv, query_scales, key_scales, decays,
                        betas, recurrent_state, core_output, tokens);
    }
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

void cuda_gdn_head_local_bf16(
    const std::uint16_t* projected_qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* gate_z,
    const std::uint16_t* conv_weight, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, const std::uint16_t* norm_weight,
    CudaGdnLayerStateView state, std::uint16_t* activated,
    std::uint16_t* core, std::uint16_t* output, float* parameter_scratch,
    unsigned tokens, unsigned rank, bool decode, void* stream, int device,
    CudaPrefixSnapshotView conv_snapshots, CudaPrefixSnapshotView recurrent_snapshots, CudaGdnHeadLocalProjectionLayout layout,
    float* normalized_qk) {
    if (rank > 1 || !tokens || (decode && tokens != 1) || !stream ||
        !projected_qkv || !projected_b || !projected_a || !gate_z || !conv_weight ||
        !a_log || !dt_bias || !norm_weight || !state.conv || !state.recurrent ||
        !activated || !core || !output || !parameter_scratch ||
        layout.qkv_stride < (layout.full_qkv ? 10240u : 5120u) ||
        (layout.control_heads != 24 && layout.control_heads != 48) ||
        layout.control_stride < layout.control_heads ||
        (layout.z_heads != 24 && layout.z_heads != 48) ||
        layout.z_head_begin != (layout.z_heads == 48 ? rank * 24 : 0) ||
        layout.z_stride < layout.z_heads * 128)
        throw std::invalid_argument("invalid head-local GDN inputs");
    validate_prefix_snapshot(conv_snapshots, tokens, std::uint64_t(kQ38GdnQkvWidth) * kQ38GdnConvWidth * 2, 2);
    validate_prefix_snapshot(recurrent_snapshots, tokens, std::uint64_t(kQ38GdnValueHeads) * kQ38GdnHeadWidth * kQ38GdnHeadWidth * 4, 4);
    select_device(device);
    auto st = reinterpret_cast<cudaStream_t>(stream);
    constexpr unsigned heads = 24;
    const unsigned begin = rank * heads;
    const auto* normalized_setting = std::getenv("Q38_CUDA_GDN_PREFILL_NORMALIZED_QK");
    const bool normalize_qk = !decode && tokens >= 8 && normalized_setting &&
        std::string(normalized_setting) == "1";
    if (normalize_qk && !normalized_qk)
        throw std::invalid_argument("normalized GDN prefill requires a separate FP32 Q/K buffer");
    auto* prepared_queries = normalized_qk;
    auto* prepared_keys = normalize_qk ? normalized_qk +
        static_cast<std::size_t>(tokens) * kQ38GdnKeyHeads * kQ38GdnHeadWidth : nullptr;
    auto* qs=parameter_scratch;
    auto* ks=qs+tokens*kQ38GdnKeyHeads;
    auto* decays=ks+tokens*kQ38GdnKeyHeads;
    auto* betas=decays+tokens*kQ38GdnValueHeads;
    const auto* setting=std::getenv("Q38_CUDA_GDN_CONV_PREPARE_FUSED");
    const bool fused=!decode && tokens<=7 && !layout.full_qkv && setting && std::string(setting)=="1";
    if(fused){
        if(conv_snapshots.data)
            gdn_conv_prepare_kernel<true><<<8,640,0,st>>>(projected_qkv,conv_weight,state.conv,activated,
                projected_b,projected_a,a_log,dt_bias,qs,ks,decays,betas,tokens,rank,layout.qkv_stride,
                layout.control_heads,layout.control_stride,conv_snapshots);
        else
            gdn_conv_prepare_kernel<false><<<8,640,0,st>>>(projected_qkv,conv_weight,state.conv,activated,
                projected_b,projected_a,a_log,dt_bias,qs,ks,decays,betas,tokens,rank,layout.qkv_stride,
                layout.control_heads,layout.control_stride,{});
    } else {
    if (conv_snapshots.data)
        gdn_conv_prefill_kernel<true><<<20,256,0,st>>>(projected_qkv,conv_weight,state.conv,activated,tokens,conv_snapshots,rank,layout.full_qkv,layout.qkv_stride);
    else
        gdn_conv_prefill_kernel<false><<<20,256,0,st>>>(projected_qkv,conv_weight,state.conv,activated,tokens,{},rank,layout.full_qkv,layout.qkv_stride);
    }
    if (decode) {
        gdn_recurrent_decode_kernel<<<heads,128,0,st>>>(activated,projected_b,projected_a,a_log,dt_bias,state.recurrent,core,begin,heads,layout.control_heads);
    } else {
        if (!fused) {
            if (normalize_qk)
                gdn_prepare_prefill_parameters_kernel<true><<<dim3(8,tokens),128,0,st>>>(activated,projected_b,projected_a,a_log,dt_bias,qs,ks,decays,betas,begin,heads,layout.control_heads,layout.control_stride,prepared_queries,prepared_keys);
            else
                gdn_prepare_prefill_parameters_kernel<false><<<dim3(8,tokens),128,0,st>>>(activated,projected_b,projected_a,a_log,dt_bias,qs,ks,decays,betas,begin,heads,layout.control_heads,layout.control_stride);
        }
        const auto* partition_setting=std::getenv("Q38_CUDA_GDN_PARTITION_COLUMNS");
        const auto* prefill_partition_setting=std::getenv("Q38_CUDA_GDN_PREFILL_PARTITION_COLUMNS");
        const unsigned columns=tokens>=8 && prefill_partition_setting
            ? unsigned(std::atoi(prefill_partition_setting))
            : tokens<=7 && partition_setting ? unsigned(std::atoi(partition_setting)) : 32u;
        if(columns!=8 && columns!=16 && columns!=32) throw std::invalid_argument("GDN partition columns must be 8/16/32");
        auto launch_variant=[&](auto tag, auto normalized_tag){
            constexpr unsigned Cols=decltype(tag)::value;
            constexpr bool Prepared=decltype(normalized_tag)::value;
            if(recurrent_snapshots.data)
                gdn_recurrent_prefill_precomputed_kernel<true,Cols,Prepared><<<dim3(heads,128/Cols),32*Cols,0,st>>>(activated,qs,ks,decays,betas,state.recurrent,core,tokens,recurrent_snapshots,begin,heads,prepared_queries,prepared_keys);
            else
                gdn_recurrent_prefill_precomputed_kernel<false,Cols,Prepared><<<dim3(heads,128/Cols),32*Cols,0,st>>>(activated,qs,ks,decays,betas,state.recurrent,core,tokens,{},begin,heads,prepared_queries,prepared_keys);
        };
        auto launch=[&](auto tag){
            if (normalize_qk) launch_variant(tag, std::true_type{});
            else launch_variant(tag, std::false_type{});
        };
        if(columns==8)launch(std::integral_constant<unsigned,8>{});
        else if(columns==16)launch(std::integral_constant<unsigned,16>{});
        else launch(std::integral_constant<unsigned,32>{});
    }
    gdn_output_norm_kernel<<<dim3(heads,tokens),128,0,st>>>(core,gate_z,norm_weight,output,1.0e-6f,heads,layout.z_heads,layout.z_head_begin,layout.z_stride);
    check(cudaPeekAtLastError(),"head-local GDN");
}

bool cuda_q38_gdn_compiled() { return true; }

}  // namespace q38
