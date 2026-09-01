#include "q38/cuda_qsa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef Q38_QSA_MAX_SCORE_CTAS
#define Q38_QSA_MAX_SCORE_CTAS 1024
#endif

namespace q38 {

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

void select_device(int device) {
    if (device < 0) throw std::invalid_argument("invalid CUDA QSA device");
    check(cudaSetDevice(device), "cudaSetDevice(QSA)");
}

void validate_cache(CudaQsaLayerStateView cache) {
    if (!cache.main_keys || !cache.main_values || !cache.raw_index_keys ||
        !cache.pooled_index_keys)
        throw std::invalid_argument("invalid CUDA QSA cache view");
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

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

template <int Threads>
__device__ float block_sum(float value) {
    __shared__ float partial[Threads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < Threads / 32 ? partial[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

template <int Threads>
__device__ float block_max(float value) {
    __shared__ float partial[Threads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < Threads / 32 ? partial[lane] : -3.402823466e+38F;
    if (warp == 0) value = warp_max(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__device__ float rope_value(const float* values, std::uint32_t element,
                            std::uint32_t rotary_width,
                            std::uint32_t position, float theta) {
    if (element >= rotary_width) return values[element];
    const auto half = rotary_width / 2;
    const auto frequency_index = element % half;
    const float exponent = static_cast<float>(2 * frequency_index) /
                           static_cast<float>(rotary_width);
    const float angle = static_cast<float>(position) / powf(theta, exponent);
    const float rotated = element < half ? -values[element + half]
                                         : values[element - half];
    return values[element] * cosf(angle) + rotated * sinf(angle);
}

__global__ void qsa_prepare_query_kernel(
    const std::uint16_t* projected, const std::uint16_t* norm_weight,
    std::uint16_t* query, std::uint16_t* gate,
    std::uint32_t first_position) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto projected_base =
        static_cast<std::uint64_t>(token) *
            kQ38QsaHeads * 2 * kQ38QsaHeadWidth +
        head * (2 * kQ38QsaHeadWidth);
    const float raw = bf16_load(projected, projected_base + element);
    const float norm = block_sum<256>(raw * raw);
    __shared__ float normalized[kQ38QsaHeadWidth];
    normalized[element] = raw * rsqrtf(norm / kQ38QsaHeadWidth + 1.0e-6f) *
                          (1.0f + bf16_load(norm_weight, element));
    __syncthreads();
    const auto output_base =
        static_cast<std::uint64_t>(token) * kQ38QsaHeads *
            kQ38QsaHeadWidth +
        head * kQ38QsaHeadWidth;
    bf16_store(query, output_base + element,
               rope_value(normalized, element, 64, first_position + token,
                          10000000.0f));
    bf16_store(gate, output_base + element,
               bf16_load(projected,
                         projected_base + kQ38QsaHeadWidth + element));
}

__global__ void qsa_prepare_kv_kernel(
    const std::uint16_t* projected_k, const std::uint16_t* projected_v,
    const std::uint16_t* norm_weight, std::uint16_t* cache_k,
    std::uint16_t* cache_v, std::uint32_t first_position) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto input =
        (static_cast<std::uint64_t>(token) * kQ38QsaKvHeads + head) *
            kQ38QsaHeadWidth +
        element;
    const float raw = bf16_load(projected_k, input);
    const float norm = block_sum<256>(raw * raw);
    __shared__ float normalized[kQ38QsaHeadWidth];
    normalized[element] = raw * rsqrtf(norm / kQ38QsaHeadWidth + 1.0e-6f) *
                          (1.0f + bf16_load(norm_weight, element));
    __syncthreads();
    const auto position = first_position + token;
    const auto output =
        (static_cast<std::uint64_t>(position) * kQ38QsaKvHeads + head) *
            kQ38QsaHeadWidth +
        element;
    bf16_store(cache_k, output,
               rope_value(normalized, element, 64, position, 10000000.0f));
    bf16_store(cache_v, output, bf16_load(projected_v, input));
}

__global__ void qsa_prepare_index_query_kernel(
    const std::uint16_t* projected, const std::uint16_t* norm_weight,
    std::uint16_t* query, std::uint32_t first_position) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto token_base = static_cast<std::uint64_t>(token) *
                            (kQ38QsaIndexerHeads + 1) *
                            kQ38QsaIndexerWidth;
    const auto input = token_base + head * kQ38QsaIndexerWidth + element;
    const float raw = bf16_load(projected, input);
    const float norm = block_sum<128>(raw * raw);
    __shared__ float normalized[kQ38QsaIndexerWidth];
    normalized[element] = raw * rsqrtf(norm / kQ38QsaIndexerWidth + 1.0e-6f) *
                          (1.0f + bf16_load(norm_weight, element));
    __syncthreads();
    bf16_store(query,
               static_cast<std::uint64_t>(token) *
                       kQ38QsaIndexerHeads * kQ38QsaIndexerWidth +
                   head * kQ38QsaIndexerWidth + element,
               rope_value(normalized, element, 64, first_position + token,
                          10000000.0f));
}

__global__ void qsa_store_raw_index_key_kernel(
    const std::uint16_t* projected, std::uint16_t* raw_cache,
    std::uint32_t first_position) {
    const auto token = blockIdx.x;
    const auto element = threadIdx.x;
    const auto position = first_position + token;
    const auto projected_base = static_cast<std::uint64_t>(token) *
                                (kQ38QsaIndexerHeads + 1) *
                                kQ38QsaIndexerWidth;
    bf16_store(raw_cache,
               static_cast<std::uint64_t>(position) * kQ38QsaIndexerWidth +
                   element,
               bf16_load(projected, projected_base +
                         kQ38QsaIndexerHeads * kQ38QsaIndexerWidth + element));
}

__global__ void qsa_pool_index_key_kernel(
    const std::uint16_t* raw_cache, const std::uint16_t* norm_weight,
    std::uint16_t* pooled_cache, std::uint32_t first_block) {
    const auto block = first_block + blockIdx.x;
    const auto element = threadIdx.x;
    const auto first_position = block * kQ38QsaBlockTokens;
    float average = 0.0f;
    for (std::uint32_t token = 0; token < kQ38QsaBlockTokens; ++token)
        average += bf16_load(
            raw_cache, (static_cast<std::uint64_t>(first_position + token) *
                            kQ38QsaIndexerWidth) +
                           element);
    average /= kQ38QsaBlockTokens;
    const float norm = block_sum<128>(average * average);
    __shared__ float normalized[kQ38QsaIndexerWidth];
    normalized[element] =
        average * rsqrtf(norm / kQ38QsaIndexerWidth + 1.0e-6f) *
        (1.0f + bf16_load(norm_weight, element));
    __syncthreads();
    bf16_store(pooled_cache,
               static_cast<std::uint64_t>(block) * kQ38QsaIndexerWidth +
                   element,
               rope_value(normalized, element, 64, first_position,
                          10000000.0f));
}

__global__ void qsa_attention_scores_kernel(
    const std::uint16_t* query, const std::uint16_t* keys,
    const std::int32_t* selected, std::uint32_t selected_count,
    float* scores) {
    const auto head = blockIdx.x;
    const auto tile = blockIdx.y;
    const auto warp = threadIdx.x >> 5;
    const auto lane = threadIdx.x & 31;
    constexpr std::uint32_t kWarpsPerBlock = 256 / 32;
    const auto kv_head = head / (kQ38QsaHeads / kQ38QsaKvHeads);
    const auto query_base = head * kQ38QsaHeadWidth;
    for (std::uint32_t selected_index =
             tile * kWarpsPerBlock + warp;
         selected_index < selected_count;
         selected_index += gridDim.y * kWarpsPerBlock) {
        const auto position = selected[selected_index];
        const auto key_base =
            (static_cast<std::uint64_t>(position) * kQ38QsaKvHeads +
             kv_head) *
            kQ38QsaHeadWidth;
        float dot = 0.0f;
        // A warp cooperatively reads one selected token.  The former layout
        // assigned one token to each lane, so lanes in a memory transaction
        // were separated by an entire KV-token stride.  Lane-striped head
        // dimensions turn both Q and K traffic into contiguous BF16 loads.
        for (std::uint32_t element = lane; element < kQ38QsaHeadWidth;
             element += 32)
            dot += bf16_load(query, query_base + element) *
                   bf16_load(keys, key_base + element);
        dot = warp_sum(dot);
        if (lane == 0)
            scores[static_cast<std::uint64_t>(head) * selected_count +
                   selected_index] =
                dot * rsqrtf(static_cast<float>(kQ38QsaHeadWidth));
    }
}

__global__ void qsa_attention_softmax_kernel(
    std::uint32_t selected_count, float* scores) {
    const auto head = blockIdx.x;
    const auto score_base = static_cast<std::uint64_t>(head) * selected_count;
    float local_max = -3.402823466e+38F;
    for (std::uint32_t index = threadIdx.x; index < selected_count;
         index += blockDim.x)
        local_max = fmaxf(local_max, scores[score_base + index]);
    const float maximum = block_max<256>(local_max);
    float local_sum = 0.0f;
    for (std::uint32_t index = threadIdx.x; index < selected_count;
         index += blockDim.x)
        local_sum += expf(scores[score_base + index] - maximum);
    const float denominator = block_sum<256>(local_sum);
    // Normalize each selected score once.  The prior implementation repeated
    // the same expf/divide for every one of the 256 value dimensions, or about
    // 12.6 million redundant exponentials per QSA layer at the 2051-token
    // budget.  Scores are scratch after softmax, so reusing them is safe.
    for (std::uint32_t index = threadIdx.x; index < selected_count;
         index += blockDim.x)
        scores[score_base + index] =
            expf(scores[score_base + index] - maximum) / denominator;
}

__global__ void qsa_attention_output_kernel(
    const std::uint16_t* values, const std::int32_t* selected,
    std::uint32_t selected_count, const float* scores,
    std::uint16_t* output) {
    const auto head = blockIdx.x;
    const auto element = blockIdx.y * blockDim.x + threadIdx.x;
    if (element >= kQ38QsaHeadWidth) return;
    const auto kv_head = head / (kQ38QsaHeads / kQ38QsaKvHeads);
    const auto score_base = static_cast<std::uint64_t>(head) * selected_count;
    float value = 0.0f;
    for (std::uint32_t index = 0; index < selected_count; ++index) {
        const auto position = selected[index];
        value += scores[score_base + index] *
                 bf16_load(values,
                           (static_cast<std::uint64_t>(position) *
                                kQ38QsaKvHeads +
                            kv_head) *
                                   kQ38QsaHeadWidth +
                               element);
    }
    bf16_store(output, head * kQ38QsaHeadWidth + element, value);
}

__device__ __forceinline__ int warp_sum_int(int value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ int block_sum_int(int value) {
    __shared__ int partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum_int(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? partial[lane] : 0;
    if (warp == 0) value = warp_sum_int(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float qsa_index_block_score(
    const std::uint16_t* query, std::uint64_t query_base,
    const std::uint16_t* pooled_keys, std::uint32_t block) {
    const auto lane = threadIdx.x & 31;
    const auto key_base =
        static_cast<std::uint64_t>(block) * kQ38QsaIndexerWidth;
    float dots[kQ38QsaIndexerHeads] = {};
    for (std::uint32_t element = lane; element < kQ38QsaIndexerWidth;
         element += 32) {
        const float key = bf16_load(pooled_keys, key_base + element);
#pragma unroll
        for (std::uint32_t head = 0; head < kQ38QsaIndexerHeads; ++head)
            dots[head] +=
                bf16_load(query,
                          query_base + head * kQ38QsaIndexerWidth + element) *
                key;
    }
    float score = 0.0f;
#pragma unroll
    for (std::uint32_t head = 0; head < kQ38QsaIndexerHeads; ++head)
        score += fmaxf(warp_sum(dots[head]), 0.0f);
    return score * rsqrtf(static_cast<float>(kQ38QsaIndexerWidth));
}

// Prefill assigns one cooperative block to each query token.  The token block
// walks its causal pooled history so hundreds of queries can run concurrently.
__global__ void qsa_index_scores_prefill_kernel(
    const std::uint16_t* query, const std::uint16_t* pooled_keys,
    std::uint32_t first_position, float* scores,
    std::uint32_t score_stride) {
    const auto token = blockIdx.x;
    const auto position = first_position + token;
    const auto complete_blocks = (position + 1) / kQ38QsaBlockTokens;
    const auto warp = threadIdx.x >> 5;
    const auto lane = threadIdx.x & 31;
    const auto query_base = static_cast<std::uint64_t>(token) *
                            kQ38QsaIndexerHeads * kQ38QsaIndexerWidth;
    for (std::uint32_t block = warp; block < complete_blocks; block += 8) {
        const float score =
            qsa_index_block_score(query, query_base, pooled_keys, block);
        if (lane == 0)
            scores[static_cast<std::uint64_t>(token) * score_stride + block] =
                score;
    }
}

// Decode has only one query, so assigning the whole scan to one CTA leaves the
// GPU almost idle and makes latency linear in context.  Spread one pooled block
// over each warp instead; the score remains bit-for-bit equivalent to the
// prefill path.
__global__ void qsa_index_scores_decode_kernel(
    const std::uint16_t* query, const std::uint16_t* pooled_keys,
    std::uint32_t complete_blocks, float* scores) {
    constexpr std::uint32_t kWarpsPerCta = 256 / 32;
    const auto warp = threadIdx.x >> 5;
    const auto lane = threadIdx.x & 31;
    float query_values[kQ38QsaIndexerHeads][kQ38QsaIndexerWidth / 32];
#pragma unroll
    for (std::uint32_t head = 0; head < kQ38QsaIndexerHeads; ++head) {
#pragma unroll
        for (std::uint32_t item = 0; item < kQ38QsaIndexerWidth / 32;
             ++item)
            query_values[head][item] = bf16_load(
                query, head * kQ38QsaIndexerWidth + item * 32 + lane);
    }
    const auto first_block = blockIdx.x * kWarpsPerCta + warp;
    const auto block_stride = gridDim.x * kWarpsPerCta;
    for (std::uint32_t block = first_block; block < complete_blocks;
         block += block_stride) {
        float dots[kQ38QsaIndexerHeads] = {};
#pragma unroll
        for (std::uint32_t item = 0; item < kQ38QsaIndexerWidth / 32;
             ++item) {
            const float key = bf16_load(
                pooled_keys,
                static_cast<std::uint64_t>(block) * kQ38QsaIndexerWidth +
                    item * 32 + lane);
#pragma unroll
            for (std::uint32_t head = 0; head < kQ38QsaIndexerHeads; ++head)
                dots[head] += query_values[head][item] * key;
        }
        float score = 0.0f;
#pragma unroll
        for (std::uint32_t head = 0; head < kQ38QsaIndexerHeads; ++head)
            score += fmaxf(warp_sum(dots[head]), 0.0f);
        if (lane == 0)
            scores[block] =
                score * rsqrtf(static_cast<float>(kQ38QsaIndexerWidth));
    }
}

__global__ void qsa_select_all_decode_kernel(
    std::uint32_t visible, std::int32_t* selected) {
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < visible; index += blockDim.x * gridDim.x)
        selected[index] = static_cast<std::int32_t>(index);
}

constexpr std::uint32_t kQsaRadixPrefixSlot = 256;
constexpr std::uint32_t kQsaRadixMaskSlot = 257;
constexpr std::uint32_t kQsaRadixRankSlot = 258;

__global__ void qsa_radix8_init_decode_kernel(std::uint32_t* scratch) {
    scratch[threadIdx.x] = 0;
    if (threadIdx.x == 0) {
        scratch[kQsaRadixPrefixSlot] = 0;
        scratch[kQsaRadixMaskSlot] = 0;
        scratch[kQsaRadixRankSlot] = kQ38QsaBlockBudget;
    }
}

__global__ void qsa_radix8_histogram_decode_kernel(
    const float* scores, std::uint32_t complete_blocks, int shift,
    std::uint32_t* scratch) {
    __shared__ std::uint32_t local[256];
    local[threadIdx.x] = 0;
    __syncthreads();
    const auto prefix = scratch[kQsaRadixPrefixSlot];
    const auto mask = scratch[kQsaRadixMaskSlot];
    for (std::uint32_t block = blockIdx.x * blockDim.x + threadIdx.x;
         block < complete_blocks; block += blockDim.x * gridDim.x) {
        const auto bits = __float_as_uint(scores[block]);
        if ((bits & mask) == prefix)
            atomicAdd(&local[(bits >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    if (local[threadIdx.x] != 0)
        atomicAdd(&scratch[threadIdx.x], local[threadIdx.x]);
}

__global__ void qsa_radix8_choose_decode_kernel(
    int shift, std::uint32_t* scratch) {
    __shared__ std::uint32_t histogram[256];
    histogram[threadIdx.x] = scratch[threadIdx.x];
    scratch[threadIdx.x] = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        auto prefix = scratch[kQsaRadixPrefixSlot];
        auto rank = scratch[kQsaRadixRankSlot];
        for (int bucket = 255; bucket >= 0; --bucket) {
            const auto count = histogram[bucket];
            if (rank <= count) {
                prefix |= static_cast<std::uint32_t>(bucket) << shift;
                break;
            }
            rank -= count;
        }
        scratch[kQsaRadixPrefixSlot] = prefix;
        scratch[kQsaRadixMaskSlot] |= 0xffu << shift;
        scratch[kQsaRadixRankSlot] = rank;
    }
}

// After the four parallel byte-wide histogram passes, gather in ascending block
// order.  Loading the threshold into shared memory first allows selected[] to
// serve as radix scratch without changing the public workspace contract.
__global__ void qsa_radix8_gather_decode_kernel(
    const float* scores, std::uint32_t complete_blocks,
    std::uint32_t position, std::int32_t* selected) {
    __shared__ std::uint32_t threshold;
    if (threadIdx.x == 0)
        threshold = reinterpret_cast<const std::uint32_t*>(selected)[
            kQsaRadixPrefixSlot];
    __syncthreads();
    __shared__ std::uint16_t offsets[256];
    __shared__ std::uint16_t chunk_count_shared;
    __shared__ std::uint32_t written;
    if (threadIdx.x == 0) written = 0;
    __syncthreads();
    for (int pass = 0; pass < 2; ++pass) {
        for (std::uint32_t base = 0; base < complete_blocks;
             base += blockDim.x) {
            const auto block = base + threadIdx.x;
            bool take = false;
            if (block < complete_blocks) {
                const auto bits = __float_as_uint(scores[block]);
                take = pass == 0 ? bits > threshold : bits == threshold;
            }
            offsets[threadIdx.x] = take ? 1 : 0;
            __syncthreads();
            if (threadIdx.x == 0) {
                std::uint16_t offset = 0;
                for (std::uint32_t lane = 0; lane < blockDim.x; ++lane) {
                    const auto flag = offsets[lane];
                    offsets[lane] = offset;
                    offset = static_cast<std::uint16_t>(offset + flag);
                }
                chunk_count_shared = offset;
            }
            __syncthreads();
            const auto chunk_count = chunk_count_shared;
            if (take) {
                const auto slot = written + offsets[threadIdx.x];
                if (slot < kQ38QsaBlockBudget) {
#pragma unroll
                    for (std::uint32_t row = 0; row < kQ38QsaBlockTokens;
                         ++row)
                        selected[slot * kQ38QsaBlockTokens + row] =
                            static_cast<std::int32_t>(
                                block * kQ38QsaBlockTokens + row);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
                written = min(static_cast<std::uint32_t>(kQ38QsaBlockBudget),
                              written + chunk_count);
            __syncthreads();
            if (written == kQ38QsaBlockBudget) break;
        }
        if (written == kQ38QsaBlockBudget) break;
    }
    const auto tail = (position + 1) % kQ38QsaBlockTokens;
    for (std::uint32_t index = threadIdx.x; index < tail;
         index += blockDim.x)
        selected[kQ38QsaBlockBudget * kQ38QsaBlockTokens + index] =
            static_cast<std::int32_t>(complete_blocks * kQ38QsaBlockTokens +
                                      index);
}

__global__ void qsa_select_prefill_kernel(
    const float* scores, std::uint32_t score_stride,
    std::uint32_t first_position, std::int32_t* selected) {
    const auto token = blockIdx.x;
    const auto position = first_position + token;
    const auto complete_blocks = (position + 1) / kQ38QsaBlockTokens;
    const auto selected_blocks =
        min(complete_blocks, static_cast<std::uint32_t>(kQ38QsaBlockBudget));
    auto* token_selected =
        selected + static_cast<std::uint64_t>(token) *
                       kQ38QsaMaximumSelected;
    const auto* token_scores =
        scores + static_cast<std::uint64_t>(token) * score_stride;

    std::uint32_t threshold_bits = 0;
    if (complete_blocks > selected_blocks) {
        std::uint32_t prefix = 0;
        std::uint32_t mask = 0;
        std::uint32_t rank = selected_blocks;
        for (int bit_index = 31; bit_index >= 0; --bit_index) {
            const auto bit = static_cast<std::uint32_t>(1u << bit_index);
            int local = 0;
            for (std::uint32_t block = threadIdx.x; block < complete_blocks;
                 block += blockDim.x) {
                const auto bits = __float_as_uint(token_scores[block]);
                if ((bits & mask) == prefix && (bits & bit) != 0) ++local;
            }
            const auto ones = static_cast<std::uint32_t>(block_sum_int(local));
            if (ones >= rank) {
                prefix |= bit;
            } else {
                rank -= ones;
            }
            mask |= bit;
        }
        threshold_bits = prefix;
    }

    __shared__ std::uint16_t offsets[256];
    __shared__ std::uint16_t chunk_count_shared;
    __shared__ std::uint32_t written;
    if (threadIdx.x == 0) written = 0;
    __syncthreads();
    for (int pass = 0; pass < 2; ++pass) {
        for (std::uint32_t base = 0; base < complete_blocks;
             base += blockDim.x) {
            const auto block = base + threadIdx.x;
            bool take = false;
            if (block < complete_blocks) {
                const auto bits = __float_as_uint(token_scores[block]);
                take = complete_blocks <= selected_blocks ||
                       (pass == 0 ? bits > threshold_bits
                                  : bits == threshold_bits);
            }
            offsets[threadIdx.x] = take ? 1 : 0;
            __syncthreads();
            if (threadIdx.x == 0) {
                std::uint16_t prefix = 0;
                for (std::uint32_t lane = 0; lane < blockDim.x; ++lane) {
                    const auto flag = offsets[lane];
                    offsets[lane] = prefix;
                    prefix = static_cast<std::uint16_t>(prefix + flag);
                }
                chunk_count_shared = prefix;
            }
            __syncthreads();
            const auto chunk_count = chunk_count_shared;
            if (take) {
                const auto slot = written + offsets[threadIdx.x];
                if (slot < selected_blocks) {
                    for (std::uint32_t row = 0; row < kQ38QsaBlockTokens;
                         ++row)
                        token_selected[slot * kQ38QsaBlockTokens + row] =
                            static_cast<std::int32_t>(
                                block * kQ38QsaBlockTokens + row);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
                written = min(selected_blocks, written + chunk_count);
            __syncthreads();
            if (written == selected_blocks) break;
        }
        if (written == selected_blocks) break;
    }
    const auto tail = (position + 1) % kQ38QsaBlockTokens;
    for (std::uint32_t index = threadIdx.x; index < tail;
         index += blockDim.x)
        token_selected[selected_blocks * kQ38QsaBlockTokens + index] =
            static_cast<std::int32_t>(complete_blocks * kQ38QsaBlockTokens +
                                      index);
}

__global__ void qsa_attention_scores_prefill_kernel(
    const std::uint16_t* query, const std::uint16_t* keys,
    const std::int32_t* selected, std::uint32_t first_position,
    float* scores) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto warp = threadIdx.x >> 5;
    const auto lane = threadIdx.x & 31;
    constexpr std::uint32_t kWarpsPerBlock = 256 / 32;
    const auto position = first_position + token;
    const auto complete_blocks = (position + 1) / kQ38QsaBlockTokens;
    const auto selected_count =
        min(complete_blocks, static_cast<std::uint32_t>(kQ38QsaBlockBudget)) *
            kQ38QsaBlockTokens +
        (position + 1) % kQ38QsaBlockTokens;
    const auto kv_head = head / (kQ38QsaHeads / kQ38QsaKvHeads);
    const auto* token_selected =
        selected + static_cast<std::uint64_t>(token) *
                       kQ38QsaMaximumSelected;
    const auto query_base =
        (static_cast<std::uint64_t>(token) * kQ38QsaHeads + head) *
        kQ38QsaHeadWidth;
    const auto score_base =
        (static_cast<std::uint64_t>(token) * kQ38QsaHeads + head) *
        kQ38QsaMaximumSelected;
    for (std::uint32_t selected_index = warp;
         selected_index < selected_count;
         selected_index += kWarpsPerBlock) {
        const auto source_position = token_selected[selected_index];
        const auto key_base =
            (static_cast<std::uint64_t>(source_position) * kQ38QsaKvHeads +
             kv_head) *
            kQ38QsaHeadWidth;
        float dot = 0.0f;
        for (std::uint32_t element = lane; element < kQ38QsaHeadWidth;
             element += 32)
            dot += bf16_load(query, query_base + element) *
                   bf16_load(keys, key_base + element);
        dot = warp_sum(dot);
        if (lane == 0)
            scores[score_base + selected_index] =
                dot * rsqrtf(static_cast<float>(kQ38QsaHeadWidth));
    }
}

__global__ void qsa_attention_output_prefill_kernel(
    const std::uint16_t* values, const std::int32_t* selected,
    std::uint32_t first_position, float* scores,
    std::uint16_t* output) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto position = first_position + token;
    const auto complete_blocks = (position + 1) / kQ38QsaBlockTokens;
    const auto selected_count =
        min(complete_blocks, static_cast<std::uint32_t>(kQ38QsaBlockBudget)) *
            kQ38QsaBlockTokens +
        (position + 1) % kQ38QsaBlockTokens;
    const auto kv_head = head / (kQ38QsaHeads / kQ38QsaKvHeads);
    const auto* token_selected =
        selected + static_cast<std::uint64_t>(token) *
                       kQ38QsaMaximumSelected;
    const auto score_base =
        (static_cast<std::uint64_t>(token) * kQ38QsaHeads + head) *
        kQ38QsaMaximumSelected;
    float local_max = -3.402823466e+38F;
    for (std::uint32_t index = threadIdx.x; index < selected_count;
         index += blockDim.x)
        local_max = fmaxf(local_max, scores[score_base + index]);
    const float maximum = block_max<256>(local_max);
    float local_sum = 0.0f;
    for (std::uint32_t index = threadIdx.x; index < selected_count;
         index += blockDim.x)
        local_sum += expf(scores[score_base + index] - maximum);
    const float denominator = block_sum<256>(local_sum);
    for (std::uint32_t index = threadIdx.x; index < selected_count;
         index += blockDim.x)
        scores[score_base + index] =
            expf(scores[score_base + index] - maximum) / denominator;
    __syncthreads();
    float value = 0.0f;
    for (std::uint32_t index = 0; index < selected_count; ++index) {
        const auto source_position = token_selected[index];
        value += scores[score_base + index] *
                 bf16_load(values,
                           (static_cast<std::uint64_t>(source_position) *
                                kQ38QsaKvHeads +
                            kv_head) *
                                   kQ38QsaHeadWidth +
                               element);
    }
    bf16_store(output,
               (static_cast<std::uint64_t>(token) * kQ38QsaHeads + head) *
                       kQ38QsaHeadWidth +
                   element,
               value);
}

}  // namespace

struct CudaQsaStateBank::Impl {
    int device;
    std::uint32_t layers;
    std::uint32_t capacity;
    std::byte* storage = nullptr;
    std::uint64_t layer_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t committed = 0;
    bool active = false;
    std::uint64_t epoch = 0;
    std::uint32_t evaluated = 0;

    Impl(int value_device, std::uint32_t value_layers,
         std::uint32_t value_capacity)
        : device(value_device), layers(value_layers), capacity(value_capacity) {
        if (layers == 0 || capacity == 0 || capacity > kQ38ContextLimit)
            throw std::invalid_argument("invalid QSA state dimensions");
        const std::uint64_t token_words =
            2ull * kQ38QsaKvHeads * kQ38QsaHeadWidth + kQ38QsaIndexerWidth;
        const std::uint64_t pooled_words =
            ((static_cast<std::uint64_t>(capacity) + kQ38QsaBlockTokens - 1) /
             kQ38QsaBlockTokens) *
            kQ38QsaIndexerWidth;
        layer_bytes = (static_cast<std::uint64_t>(capacity) * token_words +
                       pooled_words) *
                      2;
        if (layers > std::numeric_limits<std::uint64_t>::max() / layer_bytes)
            throw std::overflow_error("QSA state size overflows");
        total_bytes = layers * layer_bytes;
        if (total_bytes > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("QSA state exceeds address space");
        select_device(device);
        check(cudaMalloc(reinterpret_cast<void**>(&storage), total_bytes),
              "cudaMalloc(QSA state)");
    }

    ~Impl() {
        (void)cudaSetDevice(device);
        if (storage) (void)cudaFree(storage);
    }
};

CudaQsaStateBank::CudaQsaStateBank(int device, std::uint32_t layers,
                                   std::uint32_t capacity)
    : impl_(std::make_unique<Impl>(device, layers, capacity)) {}
CudaQsaStateBank::~CudaQsaStateBank() = default;
CudaQsaStateBank::CudaQsaStateBank(CudaQsaStateBank&&) noexcept = default;
CudaQsaStateBank& CudaQsaStateBank::operator=(CudaQsaStateBank&&) noexcept =
    default;

void CudaQsaStateBank::begin(std::uint64_t epoch) {
    if (epoch == 0 || impl_->active)
        throw std::logic_error("invalid QSA transaction begin");
    impl_->active = true;
    impl_->epoch = epoch;
    impl_->evaluated = 0;
}

CudaQsaLayerStateView CudaQsaStateBank::working(
    std::uint32_t local_layer) const {
    if (!impl_->active || local_layer >= impl_->layers)
        throw std::logic_error("invalid QSA working state request");
    auto* base = impl_->storage +
                 static_cast<std::uint64_t>(local_layer) * impl_->layer_bytes;
    const auto main_words = static_cast<std::uint64_t>(impl_->capacity) *
                            kQ38QsaKvHeads * kQ38QsaHeadWidth;
    const auto raw_words = static_cast<std::uint64_t>(impl_->capacity) *
                           kQ38QsaIndexerWidth;
    auto* words = reinterpret_cast<std::uint16_t*>(base);
    return CudaQsaLayerStateView{words, words + main_words,
                                 words + 2 * main_words,
                                 words + 2 * main_words + raw_words};
}

void CudaQsaStateBank::mark_evaluated(std::uint64_t epoch,
                                      std::uint32_t tokens) {
    if (!impl_->active || impl_->epoch != epoch || tokens == 0 ||
        impl_->committed + tokens > impl_->capacity)
        throw std::logic_error("invalid QSA evaluated extent");
    impl_->evaluated = tokens;
}

void CudaQsaStateBank::commit(std::uint64_t epoch,
                              std::uint32_t accepted_tokens) {
    if (!impl_->active || impl_->epoch != epoch || accepted_tokens == 0 ||
        accepted_tokens > impl_->evaluated)
        throw std::logic_error("invalid QSA commit extent");
    impl_->committed += accepted_tokens;
    impl_->active = false;
    impl_->epoch = 0;
    impl_->evaluated = 0;
}

void CudaQsaStateBank::rollback(std::uint64_t epoch) {
    if (!impl_->active || impl_->epoch != epoch)
        throw std::logic_error("QSA rollback epoch mismatch");
    impl_->active = false;
    impl_->epoch = 0;
    impl_->evaluated = 0;
}

void CudaQsaStateBank::reset() {
    if (impl_->active) throw std::logic_error("cannot reset active QSA state");
    impl_->committed = 0;
}

std::uint64_t CudaQsaStateBank::committed_tokens() const {
    return impl_->committed;
}
std::uint64_t CudaQsaStateBank::transaction_base() const {
    if (!impl_->active) throw std::logic_error("QSA transaction is not active");
    return impl_->committed;
}
std::uint32_t CudaQsaStateBank::capacity() const { return impl_->capacity; }
std::uint64_t CudaQsaStateBank::allocated_bytes() const {
    return impl_->total_bytes;
}

void cuda_qsa_prepare_main_decode_bf16(
    const std::uint16_t* projected_q_gate,
    const std::uint16_t* projected_k, const std::uint16_t* projected_v,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight, std::uint16_t* query,
    std::uint16_t* gate, CudaQsaLayerStateView cache,
    std::uint32_t position, void* stream, int device) {
    validate_cache(cache);
    if (!projected_q_gate || !projected_k || !projected_v || !q_norm_weight ||
        !k_norm_weight || !query || !gate || !stream ||
        position >= kQ38ContextLimit)
        throw std::invalid_argument("invalid QSA main preparation buffers");
    select_device(device);
    qsa_prepare_query_kernel<<<kQ38QsaHeads, kQ38QsaHeadWidth, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_q_gate, q_norm_weight, query, gate, position);
    qsa_prepare_kv_kernel<<<kQ38QsaKvHeads, kQ38QsaHeadWidth, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_k, projected_v, k_norm_weight, cache.main_keys,
        cache.main_values, position);
    check(cudaPeekAtLastError(), "QSA main preparation");
}

void cuda_qsa_prepare_index_decode_bf16(
    const std::uint16_t* projected_index_qk,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight, std::uint16_t* index_query,
    CudaQsaLayerStateView cache, std::uint32_t position, void* stream,
    int device) {
    validate_cache(cache);
    if (!projected_index_qk || !q_norm_weight || !k_norm_weight ||
        !index_query || !stream || position >= kQ38ContextLimit)
        throw std::invalid_argument("invalid QSA index preparation buffers");
    select_device(device);
    qsa_prepare_index_query_kernel<<<kQ38QsaIndexerHeads,
                                     kQ38QsaIndexerWidth, 0,
                                     reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_index_qk, q_norm_weight, index_query, position);
    qsa_store_raw_index_key_kernel<<<1, kQ38QsaIndexerWidth, 0,
                                     reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_index_qk, cache.raw_index_keys, position);
    if ((position + 1) % kQ38QsaBlockTokens == 0) {
        qsa_pool_index_key_kernel<<<1, kQ38QsaIndexerWidth, 0,
                                    reinterpret_cast<cudaStream_t>(stream)>>>(
            cache.raw_index_keys, k_norm_weight, cache.pooled_index_keys,
            position / kQ38QsaBlockTokens);
    }
    check(cudaPeekAtLastError(), "QSA index preparation");
}

std::uint32_t cuda_qsa_select_decode(
    const std::uint16_t* index_query, CudaQsaLayerStateView cache,
    std::uint32_t position, float* scratch_block_scores,
    std::int32_t* selected_indices, void* stream, int device) {
    validate_cache(cache);
    if (!index_query || !scratch_block_scores || !selected_indices || !stream ||
        position >= kQ38ContextLimit)
        throw std::invalid_argument("invalid QSA selection buffers");
    select_device(device);
    const auto visible = position + 1;
    const auto complete_blocks = visible / kQ38QsaBlockTokens;
    const auto selected_blocks =
        std::min<std::uint32_t>(complete_blocks, kQ38QsaBlockBudget);
    const auto tail = visible % kQ38QsaBlockTokens;
    const auto stream_value = reinterpret_cast<cudaStream_t>(stream);
    if (complete_blocks <= kQ38QsaBlockBudget) {
        const auto selected_count = selected_blocks * kQ38QsaBlockTokens + tail;
        qsa_select_all_decode_kernel<<<
            (selected_count + 255) / 256, 256, 0, stream_value>>>(
            selected_count, selected_indices);
    } else {
        constexpr std::uint32_t kWarpsPerCta = 256 / 32;
        // Keep enough independent warps to cover GA100 while still reusing the
        // fixed query across several history blocks at 128K and 256K.
        constexpr std::uint32_t kMaximumScoreCtas =
            Q38_QSA_MAX_SCORE_CTAS;
        const auto score_ctas = std::min<std::uint32_t>(
            (complete_blocks + kWarpsPerCta - 1) / kWarpsPerCta,
            kMaximumScoreCtas);
        qsa_index_scores_decode_kernel<<<
            score_ctas, 256, 0, stream_value>>>(
            index_query, cache.pooled_index_keys, complete_blocks,
            scratch_block_scores);
        auto* radix_scratch =
            reinterpret_cast<std::uint32_t*>(selected_indices);
        qsa_radix8_init_decode_kernel<<<1, 256, 0, stream_value>>>(
            radix_scratch);
        constexpr std::uint32_t kMaximumRadixCtas = 64;
        const auto radix_ctas = std::min<std::uint32_t>(
            (complete_blocks + 255) / 256, kMaximumRadixCtas);
        for (int shift = 24; shift >= 0; shift -= 8) {
            qsa_radix8_histogram_decode_kernel<<<
                radix_ctas, 256, 0, stream_value>>>(
                scratch_block_scores, complete_blocks, shift,
                radix_scratch);
            qsa_radix8_choose_decode_kernel<<<1, 256, 0, stream_value>>>(
                shift, radix_scratch);
        }
        qsa_radix8_gather_decode_kernel<<<1, 256, 0, stream_value>>>(
            scratch_block_scores, complete_blocks, position,
            selected_indices);
    }
    check(cudaPeekAtLastError(), "QSA token selection");
    return selected_blocks * kQ38QsaBlockTokens + tail;
}

void cuda_qsa_attention_decode_bf16(
    const std::uint16_t* query, CudaQsaLayerStateView cache,
    const std::int32_t* selected_indices, std::uint32_t selected_count,
    float* scratch_attention_scores, std::uint16_t* output, void* stream,
    int device) {
    validate_cache(cache);
    if (!query || !selected_indices || selected_count == 0 ||
        selected_count > kQ38QsaMaximumSelected ||
        !scratch_attention_scores || !output || !stream)
        throw std::invalid_argument("invalid QSA attention buffers");
    select_device(device);
    constexpr std::uint32_t kAttentionTiles = 4;
    const auto stream_value = reinterpret_cast<cudaStream_t>(stream);
    qsa_attention_scores_kernel<<<dim3(kQ38QsaHeads, kAttentionTiles), 256, 0,
                                  stream_value>>>(
        query, cache.main_keys, selected_indices, selected_count,
        scratch_attention_scores);
    qsa_attention_softmax_kernel<<<kQ38QsaHeads, 256, 0, stream_value>>>(
        selected_count, scratch_attention_scores);
    qsa_attention_output_kernel<<<dim3(kQ38QsaHeads, kAttentionTiles), 64, 0,
                                  stream_value>>>(
        cache.main_values, selected_indices, selected_count,
        scratch_attention_scores, output);
    check(cudaPeekAtLastError(), "QSA sparse attention");
}

void cuda_qsa_prepare_prefill_bf16(
    const std::uint16_t* projected_q_gate,
    const std::uint16_t* projected_k,
    const std::uint16_t* projected_v,
    const std::uint16_t* projected_index_qk,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight,
    const std::uint16_t* index_q_norm_weight,
    const std::uint16_t* index_k_norm_weight,
    std::uint16_t* query,
    std::uint16_t* gate,
    std::uint16_t* index_query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    void* stream,
    int device) {
    validate_cache(cache);
    if (!projected_q_gate || !projected_k || !projected_v ||
        !projected_index_qk || !q_norm_weight || !k_norm_weight ||
        !index_q_norm_weight || !index_k_norm_weight || !query || !gate ||
        !index_query || !stream || tokens == 0 ||
        first_position >= kQ38ContextLimit ||
        tokens > kQ38ContextLimit - first_position)
        throw std::invalid_argument("invalid QSA prefill preparation buffers");
    select_device(device);
    qsa_prepare_query_kernel<<<dim3(kQ38QsaHeads, tokens),
                               kQ38QsaHeadWidth, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_q_gate, q_norm_weight, query, gate, first_position);
    qsa_prepare_kv_kernel<<<dim3(kQ38QsaKvHeads, tokens),
                            kQ38QsaHeadWidth, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_k, projected_v, k_norm_weight, cache.main_keys,
        cache.main_values, first_position);
    qsa_prepare_index_query_kernel<<<
        dim3(kQ38QsaIndexerHeads, tokens), kQ38QsaIndexerWidth, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_index_qk, index_q_norm_weight, index_query,
        first_position);
    qsa_store_raw_index_key_kernel<<<tokens, kQ38QsaIndexerWidth, 0,
                                     reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_index_qk, cache.raw_index_keys, first_position);
    const auto completed_before = first_position / kQ38QsaBlockTokens;
    const auto completed_after =
        (first_position + tokens) / kQ38QsaBlockTokens;
    if (completed_after > completed_before) {
        qsa_pool_index_key_kernel<<<completed_after - completed_before,
                                    kQ38QsaIndexerWidth, 0,
                                    reinterpret_cast<cudaStream_t>(stream)>>>(
            cache.raw_index_keys, index_k_norm_weight,
            cache.pooled_index_keys, completed_before);
    }
    check(cudaPeekAtLastError(), "QSA prefill preparation");
}

void cuda_qsa_attention_prefill_bf16(
    const std::uint16_t* query,
    const std::uint16_t* index_query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    float* scratch_block_scores,
    std::uint32_t block_score_stride,
    std::int32_t* selected_indices,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device) {
    validate_cache(cache);
    const auto required_stride =
        (static_cast<std::uint64_t>(first_position) + tokens +
         kQ38QsaBlockTokens - 1) /
        kQ38QsaBlockTokens;
    if (!query || !index_query || !scratch_block_scores || !selected_indices ||
        !scratch_attention_scores || !output || !stream || tokens == 0 ||
        first_position >= kQ38ContextLimit ||
        tokens > kQ38ContextLimit - first_position ||
        block_score_stride < required_stride)
        throw std::invalid_argument("invalid QSA prefill attention buffers");
    select_device(device);
    qsa_index_scores_prefill_kernel<<<tokens, 256, 0,
                                      reinterpret_cast<cudaStream_t>(stream)>>>(
        index_query, cache.pooled_index_keys, first_position,
        scratch_block_scores, block_score_stride);
    qsa_select_prefill_kernel<<<tokens, 256, 0,
                                reinterpret_cast<cudaStream_t>(stream)>>>(
        scratch_block_scores, block_score_stride, first_position,
        selected_indices);
    qsa_attention_scores_prefill_kernel<<<dim3(kQ38QsaHeads, tokens), 256, 0,
                                          reinterpret_cast<cudaStream_t>(stream)>>>(
        query, cache.main_keys, selected_indices, first_position,
        scratch_attention_scores);
    qsa_attention_output_prefill_kernel<<<
        dim3(kQ38QsaHeads, tokens), kQ38QsaHeadWidth, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        cache.main_values, selected_indices, first_position,
        scratch_attention_scores, output);
    check(cudaPeekAtLastError(), "QSA prefill sparse attention");
}

bool cuda_q38_qsa_compiled() { return true; }

}  // namespace q38
