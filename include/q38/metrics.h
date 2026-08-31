#ifndef Q38_METRICS_H
#define Q38_METRICS_H

#include "q38/contracts.h"
#include "q38/sampling.h"

#include <cstdint>
#include <string>

namespace q38 {

constexpr std::uint32_t kMetricsMagic = fourcc('Q', '3', '8', 'M');
constexpr std::uint64_t kMetricsSchemaHashV1 =
    UINT64_C(0x7133386d65740136);  // Final V1 field-order fingerprint.

struct ExecutorStatsV1 {
    std::uint64_t transactions = 0;
    std::uint64_t append_transactions = 0;
    std::uint64_t decode_transactions = 0;
    std::uint64_t speculative_transactions = 0;
    std::uint64_t evaluated_tokens = 0;
    std::uint64_t state_committed_tokens = 0;
    std::uint64_t published_tokens = 0;
    std::uint64_t drafted_tokens = 0;
    std::uint64_t rollbacks = 0;
    std::uint64_t failures = 0;
    std::uint64_t cancellations = 0;
    std::uint64_t deadline_exceeded = 0;
    std::uint64_t sampled_tokens = 0;
    std::uint64_t stage0_execute_ns = 0;
    std::uint64_t stage1_execute_ns = 0;
    std::uint64_t backend_commit_ns = 0;
    std::uint64_t draft_ns = 0;
    std::uint64_t sampling_ns = 0;
};

struct StageBackendMetricsV1 {
    std::uint64_t execute_calls = 0;
    std::uint64_t execute_tokens = 0;
    std::uint64_t commits = 0;
    std::uint64_t rollbacks = 0;
    std::uint64_t boundary_transfers = 0;
    std::uint64_t boundary_bytes = 0;
    std::uint64_t boundary_waits = 0;
    std::uint64_t ple_cache_hits = 0;
    std::uint64_t ple_cache_misses = 0;
    std::uint64_t ple_cache_evictions = 0;
    std::uint64_t ple_cache_resident_bytes = 0;
    std::uint64_t ple_cache_capacity_bytes = 0;
    std::uint64_t ple_requested_rows = 0;
    std::uint64_t ple_unique_page_requests = 0;
    std::uint64_t ple_useful_bytes = 0;
    std::uint64_t ple_scale_resident_bytes = 0;
    std::uint64_t ple_physical_read_bytes = 0;
    std::uint64_t ple_read_operations = 0;
    std::uint64_t ple_read_batches = 0;
    std::uint64_t ple_io_uring_submissions = 0;
    std::uint64_t ple_io_uring_completions = 0;
    std::uint64_t ple_direct_read_bytes = 0;
    std::uint64_t ple_read_errors = 0;
    std::uint64_t ple_maximum_queue_depth = 0;
    std::uint64_t ple_read_latency_p50_ns = 0;
    std::uint64_t ple_read_latency_p95_ns = 0;
    std::uint64_t ple_read_latency_p99_ns = 0;
    std::uint64_t ple_io_uring_enabled = 0;
    std::uint64_t ple_direct_io_enabled = 0;
    // Exact allocator ownership census.  Category fields are disjoint for
    // device allocations; host-only weights are deliberately excluded from
    // the CUDA total.  The two pinned fields account for all runtime-owned
    // cudaHostAlloc storage.
    std::uint64_t weight_arena_bytes = 0;
    std::uint64_t weight_uploaded_bytes = 0;
    std::uint64_t weight_host_only_bytes = 0;
    std::uint64_t weight_excluded_bytes = 0;
    std::uint64_t weight_staging_peak_pinned_bytes = 0;
    std::uint64_t weight_w4_bytes = 0;
    std::uint64_t weight_w8_bytes = 0;
    std::uint64_t weight_preserved_bf16_bytes = 0;
    std::uint64_t weight_preserved_f32_bytes = 0;
    std::uint64_t weight_preserved_other_bytes = 0;
    std::uint64_t qsa_state_bytes = 0;
    std::uint64_t gdn_state_bytes = 0;
    std::uint64_t ple_state_bytes = 0;
    std::uint64_t mtp_state_bytes = 0;
    std::uint64_t workspace_device_bytes = 0;
    std::uint64_t prefill_cache_device_bytes = 0;
    std::uint64_t boundary_pinned_bytes = 0;
    std::uint64_t workspace_pinned_bytes = 0;
    std::uint64_t cuda_tracked_allocated_bytes = 0;
    std::uint64_t cuda_tracked_peak_bytes = 0;
    std::uint64_t cuda_device_free_bytes = 0;
    std::uint64_t cuda_device_total_bytes = 0;
    std::uint64_t cuda_allocator_retries = 0;
    std::uint64_t cuda_allocation_failures = 0;
    std::uint64_t cuda_graph_held_bytes = 0;
};

struct HostMetricsV1 {
    std::uint64_t process_rss_bytes = 0;
    std::uint64_t process_rss_peak_bytes = 0;
    std::uint64_t process_anon_bytes = 0;
    std::uint64_t process_file_bytes = 0;
    std::uint64_t process_shared_bytes = 0;
    std::uint64_t process_swap_bytes = 0;
    std::uint64_t process_minor_faults = 0;
    std::uint64_t process_major_faults = 0;
    std::uint64_t voluntary_context_switches = 0;
    std::uint64_t involuntary_context_switches = 0;
    std::uint64_t system_mem_available_bytes = 0;
    std::uint64_t system_swap_free_bytes = 0;
    std::uint64_t system_swap_in_pages = 0;
    std::uint64_t system_swap_out_pages = 0;
    std::uint64_t cgroup_memory_current_bytes = 0;
    std::uint64_t cgroup_memory_peak_bytes = 0;
    std::uint64_t cgroup_memory_max_bytes = 0;
    std::uint64_t cgroup_oom_events = 0;
    std::uint64_t runtime_pinned_bytes = 0;
    std::uint64_t runtime_pinned_peak_bytes = 0;
};

struct LatencySummaryV1 {
    std::uint64_t count = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t minimum_ns = 0;
    std::uint64_t maximum_ns = 0;
    std::uint64_t p50_ns = 0;
    std::uint64_t p95_ns = 0;
    std::uint64_t p99_ns = 0;
};

struct alignas(8) MetricsSchemaV1 {
    std::uint32_t magic = kMetricsMagic;
    std::uint16_t version = kContractVersion;
    std::uint16_t header_bytes = sizeof(MetricsSchemaV1);
    std::uint64_t schema_hash = kMetricsSchemaHashV1;
    std::uint64_t timestamp_ns = 0;
    std::uint64_t identity_checksum = 0;
    SessionFrontiersV1 frontiers{};
    SamplerStateV1 sampler{};
    ExecutorStatsV1 executor{};
    StageBackendMetricsV1 stage0{};
    StageBackendMetricsV1 stage1{};
    HostMetricsV1 host{};
    LatencySummaryV1 transaction_latency{};
    LatencySummaryV1 stage0_latency{};
    LatencySummaryV1 stage1_latency{};
    LatencySummaryV1 commit_latency{};
    LatencySummaryV1 draft_latency{};
    LatencySummaryV1 sampling_latency{};
};

static_assert(sizeof(ExecutorStatsV1) == 18 * sizeof(std::uint64_t),
              "executor metrics ABI changed");
static_assert(sizeof(StageBackendMetricsV1) == 54 * sizeof(std::uint64_t),
              "stage metrics ABI changed");
static_assert(sizeof(HostMetricsV1) == 20 * sizeof(std::uint64_t),
              "host metrics ABI changed");

HostMetricsV1 collect_host_metrics();
std::string metrics_json(const MetricsSchemaV1& metrics);

}  // namespace q38

#endif
