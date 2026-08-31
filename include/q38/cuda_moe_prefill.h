#ifndef Q38_CUDA_MOE_PREFILL_H
#define Q38_CUDA_MOE_PREFILL_H

#include "q38/cuda_weights.h"

#include <cstddef>
#include <cstdint>

namespace q38 {

constexpr std::uint32_t kQ38RouteExperts = 512;
constexpr std::uint32_t kQ38RouteTopK = 10;
constexpr std::uint32_t kQ38MmqM = 16;
constexpr std::uint32_t kQ38PrefillSlabMaxTokens = 512;
constexpr std::uint32_t kQ38RoutePlanMagicV1 = 0x52383351u;  // "Q38R"
constexpr std::uint32_t kQ38RoutePlanVersionV1 = 1;

enum class Q38RoutePlanStatusV1 : std::uint32_t {
    kOk = 0,
    kInvalidExpert = 1,
    kInvalidShape = 2,
};

struct alignas(8) Q38ExpertMmqTaskV1 {
    std::uint32_t packed_begin;
    std::uint16_t expert;
    std::uint8_t valid_rows;
    std::uint8_t flags;
};
static_assert(sizeof(Q38ExpertMmqTaskV1) == 8,
              "Q38ExpertMmqTaskV1 ABI changed");

struct alignas(16) Q38RoutePlanHeaderV1 {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t tokens;
    std::uint32_t routes;
    std::uint32_t active_experts;
    std::uint32_t task_count;
    std::uint32_t status;
    std::uint32_t flags;
    std::uint64_t route_hash;
    std::uint64_t plan_hash;
    std::uint64_t reserved0;
    std::uint64_t reserved1;
};
static_assert(sizeof(Q38RoutePlanHeaderV1) == 64,
              "Q38RoutePlanHeaderV1 ABI changed");

struct Q38RoutePlanStorageV1 {
    Q38RoutePlanHeaderV1* header = nullptr;
    std::uint16_t* expert_counts = nullptr;
    std::uint32_t* expert_offsets = nullptr;
    std::uint32_t* expert_task_offsets = nullptr;
    std::uint32_t* packed_assignment = nullptr;
    std::uint32_t* assignment_to_packed = nullptr;
    Q38ExpertMmqTaskV1* tasks = nullptr;
};

struct Q38PrefillMoeWorkspaceV1 {
    Q38RoutePlanStorageV1 route;
    std::uint16_t* packed_hidden = nullptr;
    std::uint16_t* weighted_mid = nullptr;
    float* route_output = nullptr;
};

enum class Q38PrefillMoeModeV1 : std::uint32_t {
    kGroupedMmq = 1,
    kGroupedMmqSafe = 2,
    kLegacyAtomicDiagnostic = 3,
};

constexpr std::uint32_t q38_moe_prefill_routes_v1(std::uint32_t tokens) {
    return tokens * kQ38RouteTopK;
}

// Exact maximum sum_e ceil(count[e] / 16) for a fixed route count.
constexpr std::uint32_t q38_moe_prefill_max_tasks_v1(std::uint32_t tokens) {
    const std::uint32_t routes = q38_moe_prefill_routes_v1(tokens);
    const std::uint32_t initially_active =
        routes < kQ38RouteExperts ? routes : kQ38RouteExperts;
    return initially_active + (routes - initially_active) / kQ38MmqM;
}

constexpr std::size_t q38_moe_route_plan_bytes_v1(std::uint32_t tokens) {
    const std::size_t routes = q38_moe_prefill_routes_v1(tokens);
    return sizeof(Q38RoutePlanHeaderV1) +
           kQ38RouteExperts * sizeof(std::uint16_t) +
           2 * (kQ38RouteExperts + 1) * sizeof(std::uint32_t) +
           2 * routes * sizeof(std::uint32_t) +
           q38_moe_prefill_max_tasks_v1(tokens) *
               sizeof(Q38ExpertMmqTaskV1);
}

// Binds a single contiguous device allocation to the versioned plan views.
// The returned pointers are device addresses; this function does not access
// the allocation contents.
Q38RoutePlanStorageV1 cuda_moe_route_plan_storage_v1(
    void* allocation, std::size_t allocation_bytes, std::uint32_t tokens);

void cuda_moe_build_route_plan_v1(const std::int32_t* expert_ids,
                                  std::uint32_t tokens,
                                  Q38RoutePlanStorageV1 plan, void* stream,
                                  int device);

void cuda_moe_pack_hidden_v1(const std::uint16_t* hidden,
                             const std::uint32_t* packed_assignment,
                             std::uint32_t routes,
                             std::uint16_t* packed_hidden, void* stream,
                             int device);

void cuda_moe_grouped_gate_up_v1(
    const CudaTensorViewV1& gate_up_experts,
    const std::uint16_t* packed_hidden,
    const std::uint32_t* packed_assignment,
    const float* token_rank_weights, const Q38ExpertMmqTaskV1* tasks,
    std::uint32_t task_count, std::uint16_t* weighted_mid,
    Q38PrefillMoeModeV1 mode, void* stream, int device);

void cuda_moe_grouped_down_v1(const CudaTensorViewV1& down_experts,
                              const std::uint16_t* weighted_mid,
                              const Q38ExpertMmqTaskV1* tasks,
                              std::uint32_t task_count, float* route_output,
                              Q38PrefillMoeModeV1 mode, void* stream,
                              int device);

void cuda_moe_reduce_top10_and_combine_shared_v1(
    const float* route_output,
    const std::uint32_t* assignment_to_packed,
    const std::uint16_t* shared_output, const std::uint16_t* shared_gate,
    std::uint32_t tokens, std::uint16_t* output, void* stream, int device);

bool cuda_q38_moe_prefill_compiled();

}  // namespace q38

#endif
