#include "q38/metrics.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unordered_map>

namespace q38 {

namespace {

#ifdef __linux__
using CounterMap = std::unordered_map<std::string, std::uint64_t>;

CounterMap read_counters(const std::string& path, bool colon_delimited,
                         std::uint64_t multiplier = 1) {
    std::ifstream input(path);
    CounterMap result;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        std::string value;
        if (!(fields >> key >> value)) continue;
        if (colon_delimited && !key.empty() && key.back() == ':')
            key.pop_back();
        try {
            const auto parsed = std::stoull(value);
            if (parsed > std::numeric_limits<std::uint64_t>::max() /
                             multiplier)
                continue;
            result[key] = parsed * multiplier;
        } catch (...) {
        }
    }
    return result;
}

std::uint64_t counter(const CounterMap& values, const char* key) {
    const auto found = values.find(key);
    return found == values.end() ? 0 : found->second;
}

std::string cgroup_v2_path() {
    std::ifstream input("/proc/self/cgroup");
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("0::", 0) == 0)
            return std::string("/sys/fs/cgroup") + line.substr(3);
    }
    return {};
}

std::uint64_t read_scalar(const std::string& path) {
    std::ifstream input(path);
    std::string value;
    if (!(input >> value)) return 0;
    if (value == "max") return std::numeric_limits<std::uint64_t>::max();
    try {
        return std::stoull(value);
    } catch (...) {
        return 0;
    }
}
#endif

void append_latency(std::ostringstream& output, const char* name,
                    const LatencySummaryV1& value) {
    output << "\"" << name << "\":{";
    output << "\"count\":" << value.count
           << ",\"total_ns\":" << value.total_ns
           << ",\"minimum_ns\":" << value.minimum_ns
           << ",\"maximum_ns\":" << value.maximum_ns
           << ",\"p50_ns\":" << value.p50_ns
           << ",\"p95_ns\":" << value.p95_ns
           << ",\"p99_ns\":" << value.p99_ns << '}';
}

void append_stage(std::ostringstream& output, const char* name,
                  const StageBackendMetricsV1& value) {
    output << "\"" << name << "\":{";
    output << "\"execute_calls\":" << value.execute_calls
           << ",\"execute_tokens\":" << value.execute_tokens
           << ",\"commits\":" << value.commits
           << ",\"rollbacks\":" << value.rollbacks
           << ",\"boundary_transfers\":" << value.boundary_transfers
           << ",\"boundary_bytes\":" << value.boundary_bytes
           << ",\"boundary_waits\":" << value.boundary_waits
           << ",\"ple_cache_hits\":" << value.ple_cache_hits
           << ",\"ple_cache_misses\":" << value.ple_cache_misses
           << ",\"ple_cache_evictions\":" << value.ple_cache_evictions
           << ",\"ple_cache_resident_bytes\":"
           << value.ple_cache_resident_bytes
           << ",\"ple_cache_capacity_bytes\":"
           << value.ple_cache_capacity_bytes
           << ",\"ple_requested_rows\":" << value.ple_requested_rows
           << ",\"ple_unique_page_requests\":"
           << value.ple_unique_page_requests
           << ",\"ple_useful_bytes\":" << value.ple_useful_bytes
           << ",\"ple_scale_resident_bytes\":"
           << value.ple_scale_resident_bytes
           << ",\"ple_physical_read_bytes\":"
           << value.ple_physical_read_bytes
           << ",\"ple_read_operations\":" << value.ple_read_operations
           << ",\"ple_read_batches\":" << value.ple_read_batches
           << ",\"ple_io_uring_submissions\":"
           << value.ple_io_uring_submissions
           << ",\"ple_io_uring_completions\":"
           << value.ple_io_uring_completions
           << ",\"ple_direct_read_bytes\":"
           << value.ple_direct_read_bytes
           << ",\"ple_read_errors\":" << value.ple_read_errors
           << ",\"ple_maximum_queue_depth\":"
           << value.ple_maximum_queue_depth
           << ",\"ple_read_latency_p50_ns\":"
           << value.ple_read_latency_p50_ns
           << ",\"ple_read_latency_p95_ns\":"
           << value.ple_read_latency_p95_ns
           << ",\"ple_read_latency_p99_ns\":"
           << value.ple_read_latency_p99_ns
           << ",\"ple_io_uring_enabled\":"
           << value.ple_io_uring_enabled
           << ",\"ple_direct_io_enabled\":"
           << value.ple_direct_io_enabled
           << ",\"weight_arena_bytes\":" << value.weight_arena_bytes
           << ",\"weight_uploaded_bytes\":" << value.weight_uploaded_bytes
           << ",\"weight_host_only_bytes\":"
           << value.weight_host_only_bytes
           << ",\"weight_excluded_bytes\":"
           << value.weight_excluded_bytes
           << ",\"weight_staging_peak_pinned_bytes\":"
           << value.weight_staging_peak_pinned_bytes
           << ",\"weight_w4_bytes\":" << value.weight_w4_bytes
           << ",\"weight_w8_bytes\":" << value.weight_w8_bytes
           << ",\"weight_preserved_bf16_bytes\":"
           << value.weight_preserved_bf16_bytes
           << ",\"weight_preserved_f32_bytes\":"
           << value.weight_preserved_f32_bytes
           << ",\"weight_preserved_other_bytes\":"
           << value.weight_preserved_other_bytes
           << ",\"qsa_state_bytes\":" << value.qsa_state_bytes
           << ",\"gdn_state_bytes\":" << value.gdn_state_bytes
           << ",\"ple_state_bytes\":" << value.ple_state_bytes
           << ",\"mtp_state_bytes\":" << value.mtp_state_bytes
           << ",\"workspace_device_bytes\":"
           << value.workspace_device_bytes
           << ",\"prefill_cache_device_bytes\":"
           << value.prefill_cache_device_bytes
           << ",\"boundary_pinned_bytes\":"
           << value.boundary_pinned_bytes
           << ",\"workspace_pinned_bytes\":"
           << value.workspace_pinned_bytes
           << ",\"cuda_tracked_allocated_bytes\":"
           << value.cuda_tracked_allocated_bytes
           << ",\"cuda_tracked_peak_bytes\":"
           << value.cuda_tracked_peak_bytes
           << ",\"cuda_device_free_bytes\":"
           << value.cuda_device_free_bytes
           << ",\"cuda_device_total_bytes\":"
           << value.cuda_device_total_bytes
           << ",\"cuda_allocator_retries\":"
           << value.cuda_allocator_retries
           << ",\"cuda_allocation_failures\":"
           << value.cuda_allocation_failures
           << ",\"cuda_graph_held_bytes\":"
           << value.cuda_graph_held_bytes << '}';
}

void append_host(std::ostringstream& output, const HostMetricsV1& value) {
    output << "\"host\":{";
    output << "\"process_rss_bytes\":" << value.process_rss_bytes
           << ",\"process_rss_peak_bytes\":"
           << value.process_rss_peak_bytes
           << ",\"process_anon_bytes\":" << value.process_anon_bytes
           << ",\"process_file_bytes\":" << value.process_file_bytes
           << ",\"process_shared_bytes\":" << value.process_shared_bytes
           << ",\"process_swap_bytes\":" << value.process_swap_bytes
           << ",\"process_minor_faults\":" << value.process_minor_faults
           << ",\"process_major_faults\":" << value.process_major_faults
           << ",\"voluntary_context_switches\":"
           << value.voluntary_context_switches
           << ",\"involuntary_context_switches\":"
           << value.involuntary_context_switches
           << ",\"system_mem_available_bytes\":"
           << value.system_mem_available_bytes
           << ",\"system_swap_free_bytes\":"
           << value.system_swap_free_bytes
           << ",\"system_swap_in_pages\":"
           << value.system_swap_in_pages
           << ",\"system_swap_out_pages\":"
           << value.system_swap_out_pages
           << ",\"cgroup_memory_current_bytes\":"
           << value.cgroup_memory_current_bytes
           << ",\"cgroup_memory_peak_bytes\":"
           << value.cgroup_memory_peak_bytes
           << ",\"cgroup_memory_max_bytes\":"
           << value.cgroup_memory_max_bytes
           << ",\"cgroup_oom_events\":" << value.cgroup_oom_events
           << ",\"runtime_pinned_bytes\":"
           << value.runtime_pinned_bytes
           << ",\"runtime_pinned_peak_bytes\":"
           << value.runtime_pinned_peak_bytes << '}';
}

}  // namespace

HostMetricsV1 collect_host_metrics() {
    HostMetricsV1 result;
#ifdef __linux__
    constexpr std::uint64_t kKiB = 1024;
    const auto status = read_counters("/proc/self/status", true, kKiB);
    result.process_rss_bytes = counter(status, "VmRSS");
    result.process_rss_peak_bytes = counter(status, "VmHWM");
    result.process_anon_bytes = counter(status, "RssAnon");
    result.process_file_bytes = counter(status, "RssFile");
    result.process_shared_bytes = counter(status, "RssShmem");
    result.process_swap_bytes = counter(status, "VmSwap");
    const auto meminfo = read_counters("/proc/meminfo", true, kKiB);
    result.system_mem_available_bytes = counter(meminfo, "MemAvailable");
    result.system_swap_free_bytes = counter(meminfo, "SwapFree");
    const auto vmstat = read_counters("/proc/vmstat", false);
    result.system_swap_in_pages = counter(vmstat, "pswpin");
    result.system_swap_out_pages = counter(vmstat, "pswpout");
    const auto cgroup = cgroup_v2_path();
    if (!cgroup.empty()) {
        result.cgroup_memory_current_bytes =
            read_scalar(cgroup + "/memory.current");
        result.cgroup_memory_peak_bytes =
            read_scalar(cgroup + "/memory.peak");
        result.cgroup_memory_max_bytes = read_scalar(cgroup + "/memory.max");
        const auto events = read_counters(cgroup + "/memory.events", false);
        result.cgroup_oom_events = counter(events, "oom");
    }
#endif
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        result.process_minor_faults = usage.ru_minflt;
        result.process_major_faults = usage.ru_majflt;
        result.voluntary_context_switches = usage.ru_nvcsw;
        result.involuntary_context_switches = usage.ru_nivcsw;
    }
    return result;
}

std::string metrics_json(const MetricsSchemaV1& metrics) {
    std::ostringstream output;
    const auto& stats = metrics.executor;
    output << "{\"schema\":\"q38.metrics.v1\",\"schema_hash\":"
           << metrics.schema_hash << ",\"timestamp_ns\":"
           << metrics.timestamp_ns << ",\"identity_checksum\":"
           << metrics.identity_checksum << ",\"frontiers\":{"
           << "\"canonical\":" << metrics.frontiers.canonical
           << ",\"target\":" << metrics.frontiers.target
           << ",\"stage0\":" << metrics.frontiers.stage0
           << ",\"stage1\":" << metrics.frontiers.stage1
           << ",\"draft\":" << metrics.frontiers.draft
           << ",\"epoch\":" << metrics.frontiers.epoch << "},"
           << "\"executor\":{"
           << "\"transactions\":" << stats.transactions
           << ",\"append_transactions\":" << stats.append_transactions
           << ",\"decode_transactions\":" << stats.decode_transactions
           << ",\"speculative_transactions\":"
           << stats.speculative_transactions
           << ",\"evaluated_tokens\":" << stats.evaluated_tokens
           << ",\"state_committed_tokens\":"
           << stats.state_committed_tokens
           << ",\"published_tokens\":" << stats.published_tokens
           << ",\"drafted_tokens\":" << stats.drafted_tokens
           << ",\"rollbacks\":" << stats.rollbacks
           << ",\"failures\":" << stats.failures
           << ",\"cancellations\":" << stats.cancellations
           << ",\"deadline_exceeded\":" << stats.deadline_exceeded
           << ",\"sampled_tokens\":" << stats.sampled_tokens
           << ",\"stage0_execute_ns\":" << stats.stage0_execute_ns
           << ",\"stage1_execute_ns\":" << stats.stage1_execute_ns
           << ",\"backend_commit_ns\":" << stats.backend_commit_ns
           << ",\"draft_ns\":" << stats.draft_ns
           << ",\"sampling_ns\":" << stats.sampling_ns << "},"
           << "\"sampler\":{"
           << "\"rng_state\":" << metrics.sampler.rng_state
           << ",\"sampled_tokens\":" << metrics.sampler.sampled_tokens
           << ",\"canonical_tokens\":"
           << metrics.sampler.canonical_tokens
           << ",\"penalty_checksum\":"
           << metrics.sampler.penalty_checksum << "},";
    append_stage(output, "stage0", metrics.stage0);
    output << ',';
    append_stage(output, "stage1", metrics.stage1);
    output << ',';
    append_host(output, metrics.host);
    output << ",\"latency\":{";
    append_latency(output, "transaction", metrics.transaction_latency);
    output << ',';
    append_latency(output, "stage0", metrics.stage0_latency);
    output << ',';
    append_latency(output, "stage1", metrics.stage1_latency);
    output << ',';
    append_latency(output, "commit", metrics.commit_latency);
    output << ',';
    append_latency(output, "draft", metrics.draft_latency);
    output << ',';
    append_latency(output, "sampling", metrics.sampling_latency);
    output << "}}";
    return output.str();
}

}  // namespace q38
