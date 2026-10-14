#include "q38/cuda_moe_prefill.h"

#include "q38/cuda_hyper.h"
#include "q38/cuda_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace q38 {

namespace {

constexpr std::uint32_t kRoutePlanThreads = kQ38RouteExperts;
constexpr std::uint32_t kMaxRoutes =
    kQ38PrefillSlabMaxTokens * kQ38RouteTopK;
constexpr std::uint16_t kInvalidExpert = 0xffffu;
constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kMmqThreads = 256;
constexpr std::uint32_t kMmqWarps = 8;
constexpr std::uint32_t kWmmaTile = 16;
constexpr std::uint32_t kGateUpOutputTiles = 10;
constexpr std::uint32_t kDownOutputTiles = 20;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

void select_device(int device) {
    if (device < 0) throw std::invalid_argument("invalid prefill MoE device");
    check(cudaSetDevice(device), "cudaSetDevice(prefill MoE)");
}

void validate_storage(const Q38RoutePlanStorageV1& plan) {
    if (!plan.header || !plan.expert_counts || !plan.expert_offsets ||
        !plan.expert_task_offsets || !plan.packed_assignment ||
        !plan.assignment_to_packed || !plan.tasks)
        throw std::invalid_argument("incomplete prefill MoE route-plan storage");
}

void validate_expert_tensor(const CudaTensorViewV1& tensor,
                            std::uint32_t rows, std::uint32_t columns) {
    if (tensor.empty() || !tensor.data || !tensor.scales ||
        tensor.descriptor->shape.size() != 3 ||
        (tensor.descriptor->shape[0] != kQ38RouteExperts &&
         tensor.descriptor->shape[0] * 2u != kQ38RouteExperts) ||
        tensor.descriptor->shape[1] != rows ||
        tensor.descriptor->shape[2] != columns ||
        tensor.descriptor->group_size != 128 ||
        (tensor.descriptor->format != DeviceWeightFormatV1::kW4A16SymG128 &&
         tensor.descriptor->format != DeviceWeightFormatV1::kW8A16SymG128))
        throw std::invalid_argument("invalid grouped prefill expert tensor");
}

void validate_grouped_mode(Q38PrefillMoeModeV1 mode) {
    if (mode != Q38PrefillMoeModeV1::kGroupedMmq &&
        mode != Q38PrefillMoeModeV1::kGroupedMmqSafe &&
        mode != Q38PrefillMoeModeV1::kGroupedFp32Scales)
        throw std::invalid_argument("invalid grouped prefill MoE mode");
}

bool shared_fp32_activation_tile_enabled() {
    const char* value = std::getenv("Q38_TP_MOE_FP32_SHARED_A");
    return value && std::strcmp(value, "1") == 0;
}

bool shared_moe_activation_tile_enabled() {
    const char* value = std::getenv("Q38_CUDA_PREFILL_MOE_A");
    return value && (std::strcmp(value, "shared") == 0 ||
                     std::strcmp(value, "1") == 0 ||
                     std::strcmp(value, "true") == 0);
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

template <int WeightBits, bool GroupScales = false>
__device__ __forceinline__ __nv_bfloat16 dequant_weight_bf16(
    const void* data, const std::uint16_t* scales,
    std::uint64_t matrix_elements, std::uint64_t matrix_scale_elements,
    std::uint32_t columns, std::uint32_t expert, std::uint32_t row,
    std::uint32_t column) {
    const std::uint64_t local =
        static_cast<std::uint64_t>(row) * columns + column;
    const float scale = load_bf16(
        scales, static_cast<std::uint64_t>(expert) * matrix_scale_elements +
                    static_cast<std::uint64_t>(row) * (columns / 128) +
                    column / 128);
    int quantized = 0;
    if constexpr (WeightBits == 8) {
        quantized = static_cast<const std::int8_t*>(data)[
            static_cast<std::uint64_t>(expert) * matrix_elements + local];
    } else {
        const std::uint8_t packed = static_cast<const std::uint8_t*>(data)[
            static_cast<std::uint64_t>(expert) * (matrix_elements / 2) +
            (local >> 1)];
        quantized = (local & 1) != 0 ? packed >> 4 : packed & 0x0f;
        if (quantized >= 8) quantized -= 16;
    }
    return __float2bfloat16_rn(static_cast<float>(quantized) * (GroupScales ? 1.0f : scale));
}

template <bool Tiled>
__device__ __forceinline__ __nv_bfloat162 dequant_weight_pair_w4_bf16(
    const void* data, std::uint64_t matrix_elements, std::uint32_t columns,
    std::uint32_t expert, std::uint32_t row, std::uint32_t first_column,
    float scale) {
    const std::uint64_t local =
        static_cast<std::uint64_t>(row) * columns + first_column;
    std::uint64_t packed_index = 0;
    if constexpr (Tiled) {
        constexpr std::uint32_t kTile = 16;
        constexpr std::uint32_t kTileBytes = kTile * kTile / 2;
        const auto tile =
            static_cast<std::uint64_t>(row / kTile) * (columns / kTile) +
            first_column / kTile;
        packed_index =
            static_cast<std::uint64_t>(expert) * (matrix_elements / 2) +
            tile * kTileBytes + (row % kTile) * (kTile / 2) +
            (first_column % kTile) / 2;
    } else {
        packed_index =
            static_cast<std::uint64_t>(expert) * (matrix_elements / 2) +
            (local >> 1);
    }
    const auto packed =
        static_cast<const std::uint8_t*>(data)[packed_index];
    int low = packed & 0x0f;
    int high = packed >> 4;
    if (low >= 8) low -= 16;
    if (high >= 8) high -= 16;
    return __floats2bfloat162_rn(static_cast<float>(low) * scale,
                                static_cast<float>(high) * scale);
}

__device__ __forceinline__ std::uint64_t hash_word(std::uint64_t hash,
                                                   std::uint64_t value) {
    hash ^= value;
    return hash * kFnvPrime;
}

__global__ void build_route_plan_kernel(
    const std::int32_t* expert_ids, std::uint32_t tokens,
    Q38RoutePlanStorageV1 plan, std::uint32_t maximum_tasks,
    std::uint32_t local_experts, bool allow_unowned,
    bool parallel_prefix, bool compute_hashes) {
    __shared__ std::uint16_t route_experts[kMaxRoutes];
    __shared__ std::uint32_t plan_status;
    __shared__ std::uint32_t warp_prefix[3][kQ38RouteExperts / 32];
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t routes = tokens * kQ38RouteTopK;

    if (lane == 0) {
        plan_status = static_cast<std::uint32_t>(Q38RoutePlanStatusV1::kOk);
        plan.header->magic = kQ38RoutePlanMagicV1;
        plan.header->version = kQ38RoutePlanVersionV1;
        plan.header->tokens = tokens;
        plan.header->routes = routes;
        plan.header->active_experts = 0;
        plan.header->task_count = 0;
        plan.header->status = plan_status;
        plan.header->flags = (allow_unowned ? kQ38RoutePlanAllowUnowned : 0) |
            (compute_hashes ? 0 : kQ38RoutePlanHashesOmitted);
        plan.header->route_hash = 0;
        plan.header->plan_hash = 0;
        plan.header->reserved0 = 0;
        plan.header->reserved1 = 0;
    }
    for (std::uint32_t assignment = lane; assignment < routes;
         assignment += blockDim.x) {
        const std::int32_t expert = expert_ids[assignment];
        const bool valid = expert >= 0 &&
                           expert < static_cast<std::int32_t>(local_experts);
        route_experts[assignment] =
            valid ? static_cast<std::uint16_t>(expert) : kInvalidExpert;
        plan.packed_assignment[assignment] = allow_unowned
            ? kQ38RouteUnownedAssignment : assignment;
        plan.assignment_to_packed[assignment] = valid || !allow_unowned
            ? assignment : kQ38RouteUnownedAssignment;
        if (!valid && !(allow_unowned && expert == -1))
            atomicCAS(&plan_status,
                      static_cast<std::uint32_t>(Q38RoutePlanStatusV1::kOk),
                      static_cast<std::uint32_t>(
                          Q38RoutePlanStatusV1::kInvalidExpert));
    }
    for (std::uint32_t task = lane; task < maximum_tasks;
         task += blockDim.x)
        plan.tasks[task] = Q38ExpertMmqTaskV1{0, 0, 0, 0};
    __syncthreads();

    std::uint32_t count = 0;
    for (std::uint32_t assignment = 0; assignment < routes; ++assignment) {
        if (route_experts[assignment] != lane) continue;
        // Preserve the assignment's stable rank within this expert while the
        // expert-owned lane is already doing the counting pass.  After the
        // global offsets exist, every thread can scatter assignments in O(N)
        // total work instead of making every expert scan all routes again.
        plan.assignment_to_packed[assignment] = count;
        ++count;
    }
    plan.expert_counts[lane] = static_cast<std::uint16_t>(count);
    __syncthreads();

    if (parallel_prefix) {
        const unsigned warp_lane = lane & 31u;
        const unsigned warp = lane / 32;
        const unsigned tasks = (count + kQ38MmqM - 1) / kQ38MmqM;
        const unsigned active = count != 0 ? 1u : 0u;
        unsigned count_scan = count, task_scan = tasks, active_scan = active;
#pragma unroll
        for (unsigned offset = 1; offset < 32; offset *= 2) {
            const unsigned c = __shfl_up_sync(0xffffffffu, count_scan, offset);
            const unsigned t = __shfl_up_sync(0xffffffffu, task_scan, offset);
            const unsigned a = __shfl_up_sync(0xffffffffu, active_scan, offset);
            if (warp_lane >= offset) {
                count_scan += c; task_scan += t; active_scan += a;
            }
        }
        if (warp_lane == 31) {
            warp_prefix[0][warp] = count_scan;
            warp_prefix[1][warp] = task_scan;
            warp_prefix[2][warp] = active_scan;
        }
        __syncthreads();
        if (warp == 0) {
            const unsigned c0 = warp_lane < kQ38RouteExperts / 32 ? warp_prefix[0][warp_lane] : 0;
            const unsigned t0 = warp_lane < kQ38RouteExperts / 32 ? warp_prefix[1][warp_lane] : 0;
            const unsigned a0 = warp_lane < kQ38RouteExperts / 32 ? warp_prefix[2][warp_lane] : 0;
            unsigned c = c0, t = t0, a = a0;
#pragma unroll
            for (unsigned offset = 1; offset < 32; offset *= 2) {
                const unsigned pc = __shfl_up_sync(0xffffffffu, c, offset);
                const unsigned pt = __shfl_up_sync(0xffffffffu, t, offset);
                const unsigned pa = __shfl_up_sync(0xffffffffu, a, offset);
                if (warp_lane >= offset) { c += pc; t += pt; a += pa; }
            }
            if (warp_lane < kQ38RouteExperts / 32) {
                warp_prefix[0][warp_lane] = c - c0;
                warp_prefix[1][warp_lane] = t - t0;
                warp_prefix[2][warp_lane] = a - a0;
            }
        }
        __syncthreads();
        count_scan += warp_prefix[0][warp];
        task_scan += warp_prefix[1][warp];
        active_scan += warp_prefix[2][warp];
        plan.expert_offsets[lane] = count_scan - count;
        plan.expert_task_offsets[lane] = task_scan - tasks;
        if (lane == kQ38RouteExperts - 1) {
            plan.expert_offsets[kQ38RouteExperts] = count_scan;
            plan.expert_task_offsets[kQ38RouteExperts] = task_scan;
            plan.header->active_experts = active_scan;
            plan.header->task_count = task_scan;
            plan.header->status = plan_status;
            if (((!allow_unowned && count_scan != routes) || count_scan > routes ||
                 task_scan > maximum_tasks) &&
                plan.header->status == static_cast<std::uint32_t>(Q38RoutePlanStatusV1::kOk))
                plan.header->status = static_cast<std::uint32_t>(Q38RoutePlanStatusV1::kInvalidShape);
        }
    } else if (lane == 0) {
        std::uint32_t assignment_offset = 0;
        std::uint32_t task_offset = 0;
        std::uint32_t active_experts = 0;
        for (std::uint32_t expert = 0; expert < kQ38RouteExperts; ++expert) {
            plan.expert_offsets[expert] = assignment_offset;
            plan.expert_task_offsets[expert] = task_offset;
            const std::uint32_t expert_count = plan.expert_counts[expert];
            assignment_offset += expert_count;
            task_offset += (expert_count + kQ38MmqM - 1) / kQ38MmqM;
            active_experts += expert_count != 0 ? 1u : 0u;
        }
        plan.expert_offsets[kQ38RouteExperts] = assignment_offset;
        plan.expert_task_offsets[kQ38RouteExperts] = task_offset;
        plan.header->active_experts = active_experts;
        plan.header->task_count = task_offset;
        plan.header->status = plan_status;
        if (((!allow_unowned && assignment_offset != routes) ||
             assignment_offset > routes || task_offset > maximum_tasks) &&
            plan.header->status == static_cast<std::uint32_t>(
                                       Q38RoutePlanStatusV1::kOk))
            plan.header->status = static_cast<std::uint32_t>(
                Q38RoutePlanStatusV1::kInvalidShape);
    }
    __syncthreads();

    for (std::uint32_t assignment = lane; assignment < routes;
         assignment += blockDim.x) {
        const auto expert = route_experts[assignment];
        if (expert == kInvalidExpert) continue;
        const std::uint32_t packed =
            plan.expert_offsets[expert] +
            plan.assignment_to_packed[assignment];
        plan.packed_assignment[packed] = assignment;
        plan.assignment_to_packed[assignment] = packed;
    }
    const std::uint32_t expert_count = plan.expert_counts[lane];
    const std::uint32_t expert_tasks =
        (expert_count + kQ38MmqM - 1) / kQ38MmqM;
    const std::uint32_t task_begin = plan.expert_task_offsets[lane];
    for (std::uint32_t tile = 0; tile < expert_tasks; ++tile) {
        const std::uint32_t consumed = tile * kQ38MmqM;
        const std::uint32_t remaining = expert_count - consumed;
        plan.tasks[task_begin + tile] = Q38ExpertMmqTaskV1{
            plan.expert_offsets[lane] + consumed,
            static_cast<std::uint16_t>(lane),
            static_cast<std::uint8_t>(remaining < kQ38MmqM ? remaining
                                                           : kQ38MmqM),
            0};
    }
    __syncthreads();

    if (lane == 0 && compute_hashes) {
        std::uint64_t route_hash = kFnvOffset;
        for (std::uint32_t assignment = 0; assignment < routes; ++assignment)
            route_hash = hash_word(route_hash, route_experts[assignment]);
        std::uint64_t plan_hash = kFnvOffset;
        for (std::uint32_t expert = 0; expert <= kQ38RouteExperts; ++expert) {
            plan_hash = hash_word(plan_hash, plan.expert_offsets[expert]);
            plan_hash =
                hash_word(plan_hash, plan.expert_task_offsets[expert]);
        }
        for (std::uint32_t packed_index = 0; packed_index < routes;
             ++packed_index)
            plan_hash =
                hash_word(plan_hash, plan.packed_assignment[packed_index]);
        for (std::uint32_t task = 0; task < plan.header->task_count; ++task) {
            const Q38ExpertMmqTaskV1 value = plan.tasks[task];
            const std::uint64_t low =
                static_cast<std::uint64_t>(value.packed_begin) |
                (static_cast<std::uint64_t>(value.expert) << 32);
            const std::uint64_t high =
                static_cast<std::uint64_t>(value.valid_rows) |
                (static_cast<std::uint64_t>(value.flags) << 8);
            plan_hash = hash_word(plan_hash, low);
            plan_hash = hash_word(plan_hash, high);
        }
        plan.header->route_hash = route_hash;
        plan.header->plan_hash = plan_hash;
    }
}

__global__ void pack_hidden_kernel(const std::uint16_t* hidden,
                                   const std::uint32_t* packed_assignment,
                                   std::uint32_t routes,
                                   std::uint16_t* packed_hidden) {
    const std::uint32_t packed = blockIdx.x;
    if (packed >= routes) return;
    const std::uint32_t assignment = packed_assignment[packed];
    if (assignment >= routes) return;
    const std::uint32_t token = assignment / kQ38RouteTopK;
    for (std::uint32_t column = threadIdx.x; column < kQ38HiddenWidth;
         column += blockDim.x)
        packed_hidden[static_cast<std::uint64_t>(packed) * kQ38HiddenWidth +
                      column] =
            hidden[static_cast<std::uint64_t>(token) * kQ38HiddenWidth +
                   column];
}

#include "q38_moe_reg_mma.cuh"

template <int WeightBits, bool Tiled = false, bool SharedA = true, bool GroupScales = false>
__global__ void grouped_gate_up_kernel(
    const void* data, const std::uint16_t* scales,
    const std::uint16_t* packed_hidden,
    const std::uint32_t* packed_assignment,
    const float* token_rank_weights, const Q38ExpertMmqTaskV1* tasks,
    std::uint16_t* weighted_mid) {
    using namespace nvcuda;
    const Q38ExpertMmqTaskV1 task = tasks[blockIdx.y];
    if (task.valid_rows == 0) return;
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t lane = threadIdx.x & 31;
    const std::uint32_t projection = warp / 4;
    const std::uint32_t output_slice = warp & 3;
    extern __shared__ unsigned char shared_raw[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    constexpr std::uint32_t kSharedATiles = SharedA ? 1 : kMmqWarps;
    auto* shared_b = shared_a + kSharedATiles * kWmmaTile * kWmmaTile;
    auto* shared_accum = reinterpret_cast<float*>(
        shared_b + kMmqWarps * kWmmaTile * kWmmaTile);
    auto* warp_a =
        shared_a + (SharedA ? 0 : warp * kWmmaTile * kWmmaTile);
    auto* warp_b = shared_b + warp * kWmmaTile * kWmmaTile;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major>
        a_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                   wmma::col_major>
        b_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    constexpr std::uint32_t rows = 2 * kQ38MoeIntermediate;
    constexpr std::uint32_t columns = kQ38HiddenWidth;
    constexpr std::uint64_t matrix_elements =
        static_cast<std::uint64_t>(rows) * columns;
    constexpr std::uint64_t matrix_scale_elements =
        static_cast<std::uint64_t>(rows) * (columns / 128);
    float w4_row_scales[4] = {};
    float group_totals[8] = {};
    for (std::uint32_t column_base = 0; column_base < columns;
         column_base += kWmmaTile) {
        constexpr std::uint32_t kPackedTileElements =
            kWmmaTile * kWmmaTile / 2;
        const std::uint32_t a_lane = SharedA ? threadIdx.x : lane;
        const std::uint32_t a_stride = SharedA ? blockDim.x : 32;
        for (std::uint32_t pair = a_lane; pair < kPackedTileElements;
             pair += a_stride) {
            const std::uint32_t local_row = pair / (kWmmaTile / 2);
            const std::uint32_t pair_column = pair % (kWmmaTile / 2);
            if (local_row < task.valid_rows) {
                const std::uint64_t packed_row =
                    task.packed_begin + local_row;
                reinterpret_cast<__nv_bfloat162*>(warp_a)[pair] =
                    reinterpret_cast<const __nv_bfloat162*>(packed_hidden)[
                        (packed_row * columns + column_base) / 2 +
                        pair_column];
            } else {
                reinterpret_cast<__nv_bfloat162*>(warp_a)[pair] =
                    __floats2bfloat162_rn(0.0f, 0.0f);
            }
        }
        if constexpr (WeightBits == 8) {
            for (std::uint32_t element = lane;
                 element < kWmmaTile * kWmmaTile; element += 32) {
                const std::uint32_t output_column = element / kWmmaTile;
                const std::uint32_t k_column = element % kWmmaTile;
                const std::uint32_t weight_row =
                    projection * kQ38MoeIntermediate + blockIdx.x * 64 +
                    output_slice * 16 + output_column;
                warp_b[element] = dequant_weight_bf16<8, GroupScales>(
                    data, scales, matrix_elements, matrix_scale_elements,
                    columns, task.expert, weight_row,
                    column_base + k_column);
            }
        }
        if constexpr (WeightBits == 4) {
#pragma unroll
            for (std::uint32_t iteration = 0; iteration < 4; ++iteration) {
                if ((column_base & 127u) != 0) continue;
                const std::uint32_t packed_element = lane + iteration * 32;
                const std::uint32_t output_column =
                    packed_element / (kWmmaTile / 2);
                const std::uint32_t weight_row =
                    projection * kQ38MoeIntermediate + blockIdx.x * 64 +
                    output_slice * 16 + output_column;
                float scale = 0.0f;
                if ((lane & 7u) == 0)
                    scale = load_bf16(
                        scales,
                        static_cast<std::uint64_t>(task.expert) *
                                matrix_scale_elements +
                            static_cast<std::uint64_t>(weight_row) *
                                (columns / 128) +
                            column_base / 128);
                w4_row_scales[iteration] = __shfl_sync(
                    0xffffffffu, scale, static_cast<int>(lane & ~7u));
            }
#pragma unroll
            for (std::uint32_t packed_element = lane;
                 packed_element < kPackedTileElements;
                 packed_element += 32) {
                const std::uint32_t output_column =
                    packed_element / (kWmmaTile / 2);
                const std::uint32_t pair_column =
                    packed_element % (kWmmaTile / 2);
                const std::uint32_t weight_row =
                    projection * kQ38MoeIntermediate + blockIdx.x * 64 +
                    output_slice * 16 + output_column;
                reinterpret_cast<__nv_bfloat162*>(warp_b)[packed_element] =
                    dequant_weight_pair_w4_bf16<Tiled>(
                        data, matrix_elements, columns, task.expert, weight_row,
                        column_base + pair_column * 2,
                        GroupScales ? 1.0f : w4_row_scales[packed_element / 32]);
            }
        }
        if constexpr (SharedA)
            __syncthreads();
        else
            __syncwarp();
        wmma::load_matrix_sync(a_fragment, warp_a, kWmmaTile);
        wmma::load_matrix_sync(b_fragment, warp_b, kWmmaTile);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        // Every warp consumes the same activation tile.  Do not let the first
        // warp reaching the next K tile overwrite it while a sibling warp is
        // still issuing its current WMMA load.
        if constexpr (SharedA) __syncthreads();
        if constexpr (GroupScales) {
            if ((column_base & 127u) == 112u) {
                // Integer W4/W8 quants are exact in BF16. Apply each G128
                // scale to its FP32 dot product, avoiding BF16(weight*scale).
                // Use documented row-major storage, never opaque fragment
                // register-to-coordinate mappings.
                wmma::store_matrix_sync(shared_accum + warp * 256,
                    accumulator, 16, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) {
                    const unsigned element = lane + i * 32;
                    const unsigned weight_row = projection * kQ38MoeIntermediate + blockIdx.x * 64 + output_slice * 16 + element % 16;
                    const float scale = load_bf16(scales,
                        static_cast<std::uint64_t>(task.expert) * matrix_scale_elements +
                        static_cast<std::uint64_t>(weight_row) * (columns / 128) + column_base / 128);
                    group_totals[i] += shared_accum[warp * 256 + element] * scale;
                }
                __syncwarp();
                wmma::fill_fragment(accumulator, 0.0f);
            }
        }
    }
    if constexpr (GroupScales) {
#pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            shared_accum[warp * 256 + lane + i * 32] = group_totals[i];
        __syncwarp();
    } else {
        wmma::store_matrix_sync(
            shared_accum + warp * kWmmaTile * kWmmaTile, accumulator, kWmmaTile,
            wmma::mem_row_major);
    }
    __syncthreads();

    if (warp < 4) {
        for (std::uint32_t element = lane;
             element < kWmmaTile * kWmmaTile; element += 32) {
            const std::uint32_t local_row = element / kWmmaTile;
            const std::uint32_t fragment_column = element % kWmmaTile;
            if (local_row >= task.valid_rows) continue;
            const std::uint32_t middle_column =
                warp * kWmmaTile + fragment_column;
            const float gate = __bfloat162float(__float2bfloat16_rn(
                shared_accum[warp * kWmmaTile * kWmmaTile + element]));
            const float up = __bfloat162float(__float2bfloat16_rn(
                shared_accum[(warp + 4) * kWmmaTile * kWmmaTile + element]));
            const float activated = __bfloat162float(__float2bfloat16_rn(
                gate / (1.0f + expf(-gate)) * up));
            const std::uint32_t packed_row = task.packed_begin + local_row;
            const std::uint32_t assignment = packed_assignment[packed_row];
            store_bf16(weighted_mid,
                       static_cast<std::uint64_t>(packed_row) *
                               kQ38MoeIntermediate +
                           blockIdx.x * 64 + middle_column,
                       token_rank_weights ? activated * token_rank_weights[assignment]
                                          : activated);
        }
    }
}

template <int WeightBits, bool Tiled = false, bool SharedA = true, bool GroupScales = false>
__global__ void grouped_down_kernel(
    const void* data, const std::uint16_t* scales,
    const std::uint16_t* weighted_mid, const Q38ExpertMmqTaskV1* tasks,
    float* route_output) {
    using namespace nvcuda;
    const Q38ExpertMmqTaskV1 task = tasks[blockIdx.y];
    if (task.valid_rows == 0) return;
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t lane = threadIdx.x & 31;
    const std::uint32_t output_base = blockIdx.x * 128 + warp * 16;

    extern __shared__ unsigned char shared_raw[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    constexpr std::uint32_t kSharedATiles = SharedA ? 1 : kMmqWarps;
    auto* shared_b = shared_a + kSharedATiles * kWmmaTile * kWmmaTile;
    auto* shared_accum = reinterpret_cast<float*>(
        shared_b + kMmqWarps * kWmmaTile * kWmmaTile);
    auto* warp_a =
        shared_a + (SharedA ? 0 : warp * kWmmaTile * kWmmaTile);
    auto* warp_b = shared_b + warp * kWmmaTile * kWmmaTile;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major>
        a_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                   wmma::col_major>
        b_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    constexpr std::uint32_t rows = kQ38HiddenWidth;
    constexpr std::uint32_t columns = kQ38MoeIntermediate;
    constexpr std::uint64_t matrix_elements =
        static_cast<std::uint64_t>(rows) * columns;
    constexpr std::uint64_t matrix_scale_elements =
        static_cast<std::uint64_t>(rows) * (columns / 128);
    float w4_row_scales[4] = {};
    float group_totals[8] = {};
    for (std::uint32_t column_base = 0; column_base < columns;
         column_base += kWmmaTile) {
        constexpr std::uint32_t kPackedTileElements =
            kWmmaTile * kWmmaTile / 2;
        const std::uint32_t a_lane = SharedA ? threadIdx.x : lane;
        const std::uint32_t a_stride = SharedA ? blockDim.x : 32;
        for (std::uint32_t pair = a_lane; pair < kPackedTileElements;
             pair += a_stride) {
            const std::uint32_t local_row = pair / (kWmmaTile / 2);
            const std::uint32_t pair_column = pair % (kWmmaTile / 2);
            if (local_row < task.valid_rows) {
                const std::uint64_t packed_row =
                    task.packed_begin + local_row;
                reinterpret_cast<__nv_bfloat162*>(warp_a)[pair] =
                    reinterpret_cast<const __nv_bfloat162*>(weighted_mid)[
                        (packed_row * columns + column_base) / 2 +
                        pair_column];
            } else {
                reinterpret_cast<__nv_bfloat162*>(warp_a)[pair] =
                    __floats2bfloat162_rn(0.0f, 0.0f);
            }
        }
        if constexpr (WeightBits == 8) {
            for (std::uint32_t element = lane;
                 element < kWmmaTile * kWmmaTile; element += 32) {
                const std::uint32_t output_column = element / kWmmaTile;
                const std::uint32_t k_column = element % kWmmaTile;
                const std::uint32_t weight_row =
                    blockIdx.x * 128 + warp * 16 + output_column;
                warp_b[element] = dequant_weight_bf16<8, GroupScales>(
                    data, scales, matrix_elements, matrix_scale_elements,
                    columns, task.expert, weight_row,
                    column_base + k_column);
            }
        }
        if constexpr (WeightBits == 4) {
#pragma unroll
            for (std::uint32_t iteration = 0; iteration < 4; ++iteration) {
                if ((column_base & 127u) != 0) continue;
                const std::uint32_t packed_element = lane + iteration * 32;
                const std::uint32_t output_column =
                    packed_element / (kWmmaTile / 2);
                const std::uint32_t weight_row =
                    blockIdx.x * 128 + warp * 16 + output_column;
                float scale = 0.0f;
                if ((lane & 7u) == 0)
                    scale = load_bf16(
                        scales,
                        static_cast<std::uint64_t>(task.expert) *
                                matrix_scale_elements +
                            static_cast<std::uint64_t>(weight_row) *
                                (columns / 128) +
                            column_base / 128);
                w4_row_scales[iteration] = __shfl_sync(
                    0xffffffffu, scale, static_cast<int>(lane & ~7u));
            }
#pragma unroll
            for (std::uint32_t packed_element = lane;
                 packed_element < kPackedTileElements;
                 packed_element += 32) {
                const std::uint32_t output_column =
                    packed_element / (kWmmaTile / 2);
                const std::uint32_t pair_column =
                    packed_element % (kWmmaTile / 2);
                const std::uint32_t weight_row =
                    blockIdx.x * 128 + warp * 16 + output_column;
                reinterpret_cast<__nv_bfloat162*>(warp_b)[packed_element] =
                    dequant_weight_pair_w4_bf16<Tiled>(
                        data, matrix_elements, columns, task.expert, weight_row,
                        column_base + pair_column * 2,
                        GroupScales ? 1.0f : w4_row_scales[packed_element / 32]);
            }
        }
        if constexpr (SharedA)
            __syncthreads();
        else
            __syncwarp();
        wmma::load_matrix_sync(a_fragment, warp_a, kWmmaTile);
        wmma::load_matrix_sync(b_fragment, warp_b, kWmmaTile);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        if constexpr (SharedA) __syncthreads();
        if constexpr (GroupScales) {
            if ((column_base & 127u) == 112u) {
                // Integer W4/W8 quants are exact in BF16. Apply each G128
                // scale to its FP32 dot product, avoiding BF16(weight*scale).
                // Use documented row-major storage, never opaque fragment
                // register-to-coordinate mappings.
                wmma::store_matrix_sync(shared_accum + warp * 256,
                    accumulator, 16, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) {
                    const unsigned element = lane + i * 32;
                    const unsigned weight_row = output_base + element % 16;
                    const float scale = load_bf16(scales,
                        static_cast<std::uint64_t>(task.expert) * matrix_scale_elements +
                        static_cast<std::uint64_t>(weight_row) * (columns / 128) + column_base / 128);
                    group_totals[i] += shared_accum[warp * 256 + element] * scale;
                }
                __syncwarp();
                wmma::fill_fragment(accumulator, 0.0f);
            }
        }
    }
    if constexpr (GroupScales) {
#pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            shared_accum[warp * 256 + lane + i * 32] = group_totals[i];
        __syncwarp();
    } else {
        wmma::store_matrix_sync(
            shared_accum + warp * kWmmaTile * kWmmaTile, accumulator, kWmmaTile,
            wmma::mem_row_major);
    }
    for (std::uint32_t element = lane;
         element < kWmmaTile * kWmmaTile; element += 32) {
        const std::uint32_t local_row = element / kWmmaTile;
        const std::uint32_t output_column = element % kWmmaTile;
        if (local_row >= task.valid_rows) continue;
        const std::uint32_t packed_row = task.packed_begin + local_row;
        route_output[static_cast<std::uint64_t>(packed_row) * rows +
                     output_base + output_column] =
            shared_accum[warp * kWmmaTile * kWmmaTile + element];
    }
}

__global__ void reduce_top10_f32_kernel(
    const float* route_output, const std::uint32_t* assignment_to_packed,
    float* output, const float* route_weights) {
    const std::uint32_t token = blockIdx.x;
    for (std::uint32_t column = threadIdx.x; column < kQ38HiddenWidth;
         column += blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (std::uint32_t slot = 0; slot < kQ38RouteTopK; ++slot) {
            const auto assignment = token * kQ38RouteTopK + slot;
            const auto packed = assignment_to_packed[assignment];
            if (packed != kQ38RouteUnownedAssignment)
                sum += route_output[static_cast<std::uint64_t>(packed) *
                                    kQ38HiddenWidth + column] *
                       (route_weights ? route_weights[assignment] : 1.0f);
        }
        output[static_cast<std::uint64_t>(token) * kQ38HiddenWidth + column] = sum;
    }
}

__global__ void reduce_top10_and_combine_shared_kernel(
    const float* route_output,
    const std::uint32_t* assignment_to_packed,
    const std::uint16_t* shared_output, const std::uint16_t* shared_gate,
    std::uint16_t* output) {
    const std::uint32_t token = blockIdx.x;
    const float gate = load_bf16(shared_gate, token);
    const float shared_multiplier = 1.0f / (1.0f + expf(-gate));
    for (std::uint32_t column = threadIdx.x; column < kQ38HiddenWidth;
         column += blockDim.x) {
        float routed = 0.0f;
#pragma unroll
        for (std::uint32_t rank = 0; rank < kQ38RouteTopK; ++rank) {
            const std::uint32_t assignment = token * kQ38RouteTopK + rank;
            const std::uint32_t packed =
                assignment_to_packed[assignment];
            routed += route_output[
                static_cast<std::uint64_t>(packed) * kQ38HiddenWidth + column];
        }
        store_bf16(output,
                   static_cast<std::uint64_t>(token) * kQ38HiddenWidth +
                       column,
                   routed +
                       load_bf16(
                           shared_output,
                           static_cast<std::uint64_t>(token) *
                                   kQ38HiddenWidth +
                               column) *
                           shared_multiplier);
    }
}

}  // namespace

Q38RoutePlanStorageV1 cuda_moe_route_plan_storage_v1(
    void* allocation, std::size_t allocation_bytes, std::uint32_t tokens) {
    if (!allocation || tokens == 0 || tokens > kQ38PrefillSlabMaxTokens)
        throw std::invalid_argument("invalid prefill MoE route-plan allocation");
    const std::size_t required = q38_moe_route_plan_bytes_v1(tokens);
    if (allocation_bytes < required)
        throw std::invalid_argument("prefill MoE route-plan allocation is too small");
    if (reinterpret_cast<std::uintptr_t>(allocation) %
            alignof(Q38RoutePlanHeaderV1) !=
        0)
        throw std::invalid_argument("prefill MoE route-plan allocation is unaligned");

    auto* cursor = static_cast<std::uint8_t*>(allocation);
    Q38RoutePlanStorageV1 result;
    result.header = reinterpret_cast<Q38RoutePlanHeaderV1*>(cursor);
    cursor += sizeof(Q38RoutePlanHeaderV1);
    result.expert_counts = reinterpret_cast<std::uint16_t*>(cursor);
    cursor += kQ38RouteExperts * sizeof(std::uint16_t);
    result.expert_offsets = reinterpret_cast<std::uint32_t*>(cursor);
    cursor += (kQ38RouteExperts + 1) * sizeof(std::uint32_t);
    result.expert_task_offsets = reinterpret_cast<std::uint32_t*>(cursor);
    cursor += (kQ38RouteExperts + 1) * sizeof(std::uint32_t);
    result.packed_assignment = reinterpret_cast<std::uint32_t*>(cursor);
    cursor += q38_moe_prefill_routes_v1(tokens) * sizeof(std::uint32_t);
    result.assignment_to_packed = reinterpret_cast<std::uint32_t*>(cursor);
    cursor += q38_moe_prefill_routes_v1(tokens) * sizeof(std::uint32_t);
    result.tasks = reinterpret_cast<Q38ExpertMmqTaskV1*>(cursor);
    return result;
}

void cuda_moe_build_route_plan_v1(const std::int32_t* expert_ids,
                                  std::uint32_t tokens,
                                  Q38RoutePlanStorageV1 plan, void* stream,
                                  int device, std::uint32_t local_experts,
                                  bool allow_unowned, bool compute_hashes) {
    if (!expert_ids || tokens == 0 || tokens > kQ38PrefillSlabMaxTokens ||
        !stream || local_experts == 0 || local_experts > kQ38RouteExperts)
        throw std::invalid_argument("invalid prefill MoE route-plan input");
    validate_storage(plan);
    select_device(device);
    const char* parallel = std::getenv("Q38_CUDA_MOE_ROUTE_PARALLEL");
    build_route_plan_kernel<<<1, kRoutePlanThreads, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        expert_ids, tokens, plan, q38_moe_prefill_max_tasks_v1(tokens),
        local_experts, allow_unowned,
        parallel && std::strcmp(parallel, "1") == 0, compute_hashes);
    check(cudaPeekAtLastError(), "build_route_plan_kernel");
}

void cuda_moe_pack_hidden_v1(const std::uint16_t* hidden,
                             const std::uint32_t* packed_assignment,
                             std::uint32_t routes,
                             std::uint16_t* packed_hidden, void* stream,
                             int device) {
    if (!hidden || !packed_assignment || routes == 0 ||
        routes > kMaxRoutes || routes % kQ38RouteTopK != 0 ||
        !packed_hidden || !stream)
        throw std::invalid_argument("invalid prefill MoE hidden-pack input");
    select_device(device);
    pack_hidden_kernel<<<routes, 256, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
        hidden, packed_assignment, routes, packed_hidden);
    check(cudaPeekAtLastError(), "pack_hidden_kernel");
}

void cuda_moe_grouped_gate_up_v1(
    const CudaTensorViewV1& gate_up_experts,
    const std::uint16_t* packed_hidden,
    const std::uint32_t* packed_assignment,
    const float* token_rank_weights, const Q38ExpertMmqTaskV1* tasks,
    std::uint32_t task_count, std::uint16_t* weighted_mid,
    Q38PrefillMoeModeV1 mode, void* stream, int device,
    bool apply_route_weights) {
    validate_expert_tensor(gate_up_experts, 2 * kQ38MoeIntermediate,
                           kQ38HiddenWidth);
    validate_grouped_mode(mode);
    if (!packed_hidden || !packed_assignment || !token_rank_weights || !tasks ||
        task_count == 0 ||
        task_count > q38_moe_prefill_max_tasks_v1(
                         kQ38PrefillSlabMaxTokens) ||
        !weighted_mid || !stream)
        throw std::invalid_argument("invalid grouped prefill gate/up input");
    // TP applies routing weights to FP32 down outputs, as its one-row lane
    // does. Avoid moving that multiplication across a BF16 rounding boundary.
    if (!apply_route_weights) token_rank_weights = nullptr;
    select_device(device);
    const bool shared_a = mode == Q38PrefillMoeModeV1::kGroupedFp32Scales
        ? shared_fp32_activation_tile_enabled() : shared_moe_activation_tile_enabled();
    const std::size_t a_tiles = shared_a ? 1 : kMmqWarps;
    const std::size_t shared_bytes =
        ((a_tiles + kMmqWarps) * kWmmaTile * kWmmaTile) *
            sizeof(__nv_bfloat16) +
        kMmqWarps * kWmmaTile * kWmmaTile * sizeof(float);
    const dim3 grid(kGateUpOutputTiles, task_count);
    const char* reg_mma = std::getenv("Q38_CUDA_MOE_REG_MMA");
    if (mode == Q38PrefillMoeModeV1::kGroupedFp32Scales && reg_mma && std::strcmp(reg_mma, "1") == 0) {
        if (gate_up_experts.descriptor->format == DeviceWeightFormatV1::kW8A16SymG128)
            reg_grouped_gate_up_kernel<8,false><<<grid,kMmqThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(gate_up_experts.data, static_cast<const std::uint16_t*>(gate_up_experts.scales), packed_hidden, packed_assignment, token_rank_weights, tasks, weighted_mid);
        else if (gate_up_experts.layout == CudaWeightLayoutV1::kMoeW4Tile16x16)
            reg_grouped_gate_up_kernel<4,true><<<grid,kMmqThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(gate_up_experts.data, static_cast<const std::uint16_t*>(gate_up_experts.scales), packed_hidden, packed_assignment, token_rank_weights, tasks, weighted_mid);
        else
            reg_grouped_gate_up_kernel<4,false><<<grid,kMmqThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(gate_up_experts.data, static_cast<const std::uint16_t*>(gate_up_experts.scales), packed_hidden, packed_assignment, token_rank_weights, tasks, weighted_mid);
        check(cudaPeekAtLastError(), "reg_grouped_gate_up_kernel");
        return;
    }
    if (mode == Q38PrefillMoeModeV1::kGroupedFp32Scales) {
        const auto launch = [&](auto share_tag) {
            constexpr bool SharedA = decltype(share_tag)::value;
            if (gate_up_experts.descriptor->format == DeviceWeightFormatV1::kW8A16SymG128)
                grouped_gate_up_kernel<8, false, SharedA, true><<<grid, kMmqThreads, shared_bytes,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
                    gate_up_experts.data, static_cast<const std::uint16_t*>(gate_up_experts.scales),
                    packed_hidden, packed_assignment, token_rank_weights, tasks, weighted_mid);
            else if (gate_up_experts.layout == CudaWeightLayoutV1::kMoeW4Tile16x16)
                grouped_gate_up_kernel<4, true, SharedA, true><<<grid, kMmqThreads, shared_bytes,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
                    gate_up_experts.data, static_cast<const std::uint16_t*>(gate_up_experts.scales),
                    packed_hidden, packed_assignment, token_rank_weights, tasks, weighted_mid);
            else
                grouped_gate_up_kernel<4, false, SharedA, true><<<grid, kMmqThreads, shared_bytes,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
                    gate_up_experts.data, static_cast<const std::uint16_t*>(gate_up_experts.scales),
                    packed_hidden, packed_assignment, token_rank_weights, tasks, weighted_mid);
        };
        if (shared_a) launch(std::true_type{});
        else launch(std::false_type{});
        check(cudaPeekAtLastError(), "grouped_gate_up_kernel(FP32 group scales)");
        return;
    }
    if (gate_up_experts.descriptor->format ==
        DeviceWeightFormatV1::kW8A16SymG128) {
        if (shared_a)
            grouped_gate_up_kernel<8, false, true><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                gate_up_experts.data,
                static_cast<const std::uint16_t*>(gate_up_experts.scales),
                packed_hidden, packed_assignment, token_rank_weights, tasks,
                weighted_mid);
        else
            grouped_gate_up_kernel<8, false, false><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                gate_up_experts.data,
                static_cast<const std::uint16_t*>(gate_up_experts.scales),
                packed_hidden, packed_assignment, token_rank_weights, tasks,
                weighted_mid);
    } else if (gate_up_experts.layout ==
               CudaWeightLayoutV1::kMoeW4Tile16x16) {
        if (shared_a)
            grouped_gate_up_kernel<4, true, true><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                gate_up_experts.data,
                static_cast<const std::uint16_t*>(gate_up_experts.scales),
                packed_hidden, packed_assignment, token_rank_weights, tasks,
                weighted_mid);
        else
            grouped_gate_up_kernel<4, true, false><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                gate_up_experts.data,
                static_cast<const std::uint16_t*>(gate_up_experts.scales),
                packed_hidden, packed_assignment, token_rank_weights, tasks,
                weighted_mid);
    } else {
        if (shared_a)
            grouped_gate_up_kernel<4, false, true><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                gate_up_experts.data,
                static_cast<const std::uint16_t*>(gate_up_experts.scales),
                packed_hidden, packed_assignment, token_rank_weights, tasks,
                weighted_mid);
        else
            grouped_gate_up_kernel<4, false, false><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                gate_up_experts.data,
                static_cast<const std::uint16_t*>(gate_up_experts.scales),
                packed_hidden, packed_assignment, token_rank_weights, tasks,
                weighted_mid);
    }
    check(cudaPeekAtLastError(), "grouped_gate_up_kernel");
}

void cuda_moe_grouped_down_v1(const CudaTensorViewV1& down_experts,
                              const std::uint16_t* weighted_mid,
                              const Q38ExpertMmqTaskV1* tasks,
                              std::uint32_t task_count, float* route_output,
                              Q38PrefillMoeModeV1 mode, void* stream,
                              int device) {
    validate_expert_tensor(down_experts, kQ38HiddenWidth,
                           kQ38MoeIntermediate);
    validate_grouped_mode(mode);
    if (!weighted_mid || !tasks || task_count == 0 ||
        task_count > q38_moe_prefill_max_tasks_v1(
                         kQ38PrefillSlabMaxTokens) ||
        !route_output || !stream)
        throw std::invalid_argument("invalid grouped prefill down input");
    select_device(device);
    const bool shared_a = mode == Q38PrefillMoeModeV1::kGroupedFp32Scales
        ? shared_fp32_activation_tile_enabled() : shared_moe_activation_tile_enabled();
    const std::size_t a_tiles = shared_a ? 1 : kMmqWarps;
    const std::size_t shared_bytes =
        ((a_tiles + kMmqWarps) * kWmmaTile * kWmmaTile) *
            sizeof(__nv_bfloat16) +
        kMmqWarps * kWmmaTile * kWmmaTile * sizeof(float);
    const dim3 grid(kDownOutputTiles, task_count);
    const char* reg_mma = std::getenv("Q38_CUDA_MOE_REG_MMA");
    if (mode == Q38PrefillMoeModeV1::kGroupedFp32Scales && reg_mma && std::strcmp(reg_mma, "1") == 0) {
        if (down_experts.descriptor->format == DeviceWeightFormatV1::kW8A16SymG128)
            reg_grouped_down_kernel<8,false><<<grid,kMmqThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(down_experts.data, static_cast<const std::uint16_t*>(down_experts.scales), weighted_mid, tasks, route_output);
        else if (down_experts.layout == CudaWeightLayoutV1::kMoeW4Tile16x16)
            reg_grouped_down_kernel<4,true><<<grid,kMmqThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(down_experts.data, static_cast<const std::uint16_t*>(down_experts.scales), weighted_mid, tasks, route_output);
        else
            reg_grouped_down_kernel<4,false><<<grid,kMmqThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(down_experts.data, static_cast<const std::uint16_t*>(down_experts.scales), weighted_mid, tasks, route_output);
        check(cudaPeekAtLastError(), "reg_grouped_down_kernel");
        return;
    }
    if (mode == Q38PrefillMoeModeV1::kGroupedFp32Scales) {
        const auto launch = [&](auto share_tag) {
            constexpr bool SharedA = decltype(share_tag)::value;
            if (down_experts.descriptor->format == DeviceWeightFormatV1::kW8A16SymG128)
                grouped_down_kernel<8, false, SharedA, true><<<grid, kMmqThreads, shared_bytes,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
                    down_experts.data, static_cast<const std::uint16_t*>(down_experts.scales),
                    weighted_mid, tasks, route_output);
            else if (down_experts.layout == CudaWeightLayoutV1::kMoeW4Tile16x16)
                grouped_down_kernel<4, true, SharedA, true><<<grid, kMmqThreads, shared_bytes,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
                    down_experts.data, static_cast<const std::uint16_t*>(down_experts.scales),
                    weighted_mid, tasks, route_output);
            else
                grouped_down_kernel<4, false, SharedA, true><<<grid, kMmqThreads, shared_bytes,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
                    down_experts.data, static_cast<const std::uint16_t*>(down_experts.scales),
                    weighted_mid, tasks, route_output);
        };
        if (shared_a) launch(std::true_type{});
        else launch(std::false_type{});
        check(cudaPeekAtLastError(), "grouped_down_kernel(FP32 group scales)");
        return;
    }
    if (down_experts.descriptor->format ==
        DeviceWeightFormatV1::kW8A16SymG128) {
        if (shared_a)
            grouped_down_kernel<8, false, true><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                down_experts.data,
                static_cast<const std::uint16_t*>(down_experts.scales),
                weighted_mid, tasks, route_output);
        else
            grouped_down_kernel<8, false, false><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                down_experts.data,
                static_cast<const std::uint16_t*>(down_experts.scales),
                weighted_mid, tasks, route_output);
    } else if (down_experts.layout ==
               CudaWeightLayoutV1::kMoeW4Tile16x16) {
        if (shared_a)
            grouped_down_kernel<4, true, true><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                down_experts.data,
                static_cast<const std::uint16_t*>(down_experts.scales),
                weighted_mid, tasks, route_output);
        else
            grouped_down_kernel<4, true, false><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                down_experts.data,
                static_cast<const std::uint16_t*>(down_experts.scales),
                weighted_mid, tasks, route_output);
    } else {
        if (shared_a)
            grouped_down_kernel<4, false, true><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                down_experts.data,
                static_cast<const std::uint16_t*>(down_experts.scales),
                weighted_mid, tasks, route_output);
        else
            grouped_down_kernel<4, false, false><<<
                grid, kMmqThreads, shared_bytes,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                down_experts.data,
                static_cast<const std::uint16_t*>(down_experts.scales),
                weighted_mid, tasks, route_output);
    }
    check(cudaPeekAtLastError(), "grouped_down_kernel");
}

void cuda_moe_reduce_top10_f32_v1(
    const float* route_output, const std::uint32_t* assignment_to_packed,
    std::uint32_t tokens, float* output, void* stream, int device,
    const float* route_weights) {
    if (!route_output || !assignment_to_packed || !output || !stream ||
        tokens == 0 || tokens > kQ38PrefillSlabMaxTokens)
        throw std::invalid_argument("invalid token-major MoE partial reduction");
    select_device(device);
    reduce_top10_f32_kernel<<<tokens, 256, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            route_output, assignment_to_packed, output, route_weights);
    check(cudaPeekAtLastError(), "reduce_top10_f32_kernel");
}

void cuda_moe_reduce_top10_and_combine_shared_v1(
    const float* route_output,
    const std::uint32_t* assignment_to_packed,
    const std::uint16_t* shared_output, const std::uint16_t* shared_gate,
    std::uint32_t tokens, std::uint16_t* output, void* stream, int device) {
    if (!route_output || !assignment_to_packed || !shared_output ||
        !shared_gate || tokens == 0 || tokens > kQ38PrefillSlabMaxTokens ||
        !output || !stream)
        throw std::invalid_argument("invalid grouped prefill reduction input");
    select_device(device);
    reduce_top10_and_combine_shared_kernel<<<
        tokens, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        route_output, assignment_to_packed, shared_output, shared_gate,
        output);
    check(cudaPeekAtLastError(),
          "reduce_top10_and_combine_shared_kernel");
}

bool cuda_q38_moe_prefill_compiled() { return true; }

}  // namespace q38
