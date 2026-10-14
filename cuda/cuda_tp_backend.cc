#include "q38/cuda_backend.h"
#include "q38/cuda_te.h"
#include "q38/paired_rank_worker.h"

#include <algorithm>
#include <array>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <thread>
#include <utility>

namespace q38 {
namespace {
// Keep each rank's snapshot alive until its own stage1 stream consumes it.
class PairedBoundaryLease final : public BoundaryLease {
public:
    PairedBoundaryLease(std::shared_ptr<const BoundaryLease> first,
                        std::shared_ptr<const BoundaryLease> second)
        : ranks_{std::move(first), std::move(second)} {
        if (!ranks_[0] || !ranks_[1] || ranks_[0]->size() != ranks_[1]->size() ||
            ranks_[0]->payload_checksum() != ranks_[1]->payload_checksum() ||
            !ranks_[0]->device_data(0) || !ranks_[1]->device_data(1))
            throw std::runtime_error("TP device boundary rank mismatch");
    }
    const std::uint16_t* data() const override { return ranks_[0]->data(); }
    std::size_t size() const override { return ranks_[0]->size(); }
    std::uint32_t slot() const override { return ranks_[0]->slot(); }
    std::uint64_t payload_checksum() const override { return ranks_[0]->payload_checksum(); }
    void wait_ready() const override {
        ranks_[0]->wait_ready(); ranks_[1]->wait_ready();
    }
    const std::uint16_t* device_data(int device) const override {
        return device >= 0 && device < 2 ? ranks_[device]->device_data(device) : nullptr;
    }
    void record_device_consumer(int device, void* stream) const override {
        if (device < 0 || device > 1) throw std::invalid_argument("invalid TP boundary device");
        ranks_[device]->record_device_consumer(device, stream);
    }
private:
    std::shared_ptr<const BoundaryLease> ranks_[2];
};
// One transactional executor owns both ranks. Each logical stage launches the
// matching rank backends and publishes only after they both finish and agree.
class TensorParallelStage final : public StageBackend {
public:
    TensorParallelStage(std::unique_ptr<StageBackend> first,
                        std::unique_ptr<StageBackend> second)
        : ranks_{std::move(first), std::move(second)} {
        const char* mode = std::getenv("Q38_TP_PERSISTENT_WORKER");
        if (mode && std::strcmp(mode, "0") != 0 && std::strcmp(mode, "false") != 0)
            worker_ = std::make_unique<PairedRankWorker>([] { te_bridge_abort(); });
    }
    Stage stage() const override { return ranks_[0]->stage(); }
    std::vector<std::int32_t> draft(
        std::int32_t pending, std::uint64_t position, std::uint32_t width,
        std::shared_ptr<CancellationToken> cancellation) override {
        std::vector<std::int32_t> tokens[2];
        both_cancellable([&](int r) { tokens[r] = ranks_[r]->draft(pending, position, width, cancellation); }, cancellation);
        if (tokens[0] != tokens[1]) {
            te_bridge_abort();
            throw std::runtime_error("TP2 draft token mismatch");
        }
        return std::move(tokens[0]);
    }
    void prefetch_transaction(const SessionTxnV1& txn,
                              const std::vector<std::int32_t>& ids,
                              std::shared_ptr<CancellationToken> cancel) override {
        both_cancellable([&](int r) { ranks_[r]->prefetch_transaction(txn, ids, cancel); }, cancel);
    }
    StageOutput execute(StageInput input) override {
        StageOutput out[2];
        both_cancellable([&](int r) { out[r] = ranks_[r]->execute(input); }, input.cancellation);
        if (out[0].state_commit_count != out[1].state_commit_count ||
            out[0].next_token != out[1].next_token ||
            out[0].logits_bf16 != out[1].logits_bf16 ||
            out[0].logit_token_ids != out[1].logit_token_ids ||
            out[0].target_predictions != out[1].target_predictions ||
            (stage() == Stage::kStage0 &&
             (out[0].boundary.size() != out[1].boundary.size() ||
              out[0].boundary.payload_checksum() != out[1].boundary.payload_checksum()))) {
            te_bridge_abort();
            throw std::runtime_error("TP2 rank output mismatch");
        }
        if (stage() == Stage::kStage0 && out[0].boundary.lease &&
            out[0].boundary.lease->device_data(0))
            out[0].boundary.lease = std::make_shared<PairedBoundaryLease>(
                out[0].boundary.lease, out[1].boundary.lease);
        return std::move(out[0]);
    }
    void commit(std::uint64_t epoch, std::uint32_t count) override {
        both([&](int r) { ranks_[r]->commit(epoch, count); });
    }
    void rollback(std::uint64_t epoch) override {
        both([&](int r) { ranks_[r]->rollback(epoch); });
    }
    void reset_session() override {
        both([&](int r) { ranks_[r]->reset_session(); });
        const auto* recovery = std::getenv("Q38_TP_RECOVER_CANCEL");
        if (recovery && std::strcmp(recovery, "1") == 0)
            te_bridge_reset_group(stage() == Stage::kStage0 ? 0 : 1);
    }
    StageBackendMetricsV1 metrics() const override {
        auto result = ranks_[0]->metrics();
        const auto peer = ranks_[1]->metrics();
        // Counts describe physical work on both ranks, memory is summed over
        // the two backends. Latency quantiles are conservatively the maximum.
        result.execute_calls += peer.execute_calls;
        result.execute_tokens += peer.execute_tokens;
        result.commits += peer.commits;
        result.rollbacks += peer.rollbacks;
        result.boundary_transfers += peer.boundary_transfers;
        result.boundary_bytes += peer.boundary_bytes;
        result.boundary_waits += peer.boundary_waits;
        // RAM mode shares one immutable store; each rank reports the same
        // physical ownership and request counters. Count it once per stage.
        if (!result.ple_ram_enabled) {
            result.ple_cache_hits += peer.ple_cache_hits;
            result.ple_cache_misses += peer.ple_cache_misses;
            result.ple_cache_evictions += peer.ple_cache_evictions;
            result.ple_cache_resident_bytes += peer.ple_cache_resident_bytes;
            result.ple_cache_capacity_bytes += peer.ple_cache_capacity_bytes;
            result.ple_requested_rows += peer.ple_requested_rows;
            result.ple_unique_page_requests += peer.ple_unique_page_requests;
            result.ple_useful_bytes += peer.ple_useful_bytes;
            result.ple_scale_resident_bytes += peer.ple_scale_resident_bytes;
            result.ple_physical_read_bytes += peer.ple_physical_read_bytes;
            result.ple_read_operations += peer.ple_read_operations;
            result.ple_read_batches += peer.ple_read_batches;
            result.ple_io_uring_submissions += peer.ple_io_uring_submissions;
            result.ple_io_uring_completions += peer.ple_io_uring_completions;
            result.ple_direct_read_bytes += peer.ple_direct_read_bytes;
            result.ple_read_errors += peer.ple_read_errors;
            result.ple_maximum_queue_depth = std::max(result.ple_maximum_queue_depth, peer.ple_maximum_queue_depth);
            result.ple_read_latency_p50_ns = std::max(result.ple_read_latency_p50_ns, peer.ple_read_latency_p50_ns);
            result.ple_read_latency_p95_ns = std::max(result.ple_read_latency_p95_ns, peer.ple_read_latency_p95_ns);
            result.ple_read_latency_p99_ns = std::max(result.ple_read_latency_p99_ns, peer.ple_read_latency_p99_ns);
            result.ple_io_uring_enabled = std::max(result.ple_io_uring_enabled, peer.ple_io_uring_enabled);
            result.ple_direct_io_enabled = std::max(result.ple_direct_io_enabled, peer.ple_direct_io_enabled);
        }
        result.weight_arena_bytes += peer.weight_arena_bytes;
        result.weight_uploaded_bytes += peer.weight_uploaded_bytes;
        result.weight_host_only_bytes += peer.weight_host_only_bytes;
        result.weight_excluded_bytes += peer.weight_excluded_bytes;
        result.weight_staging_peak_pinned_bytes += peer.weight_staging_peak_pinned_bytes;
        result.weight_w4_bytes += peer.weight_w4_bytes;
        result.weight_w8_bytes += peer.weight_w8_bytes;
        result.weight_preserved_bf16_bytes += peer.weight_preserved_bf16_bytes;
        result.weight_preserved_f32_bytes += peer.weight_preserved_f32_bytes;
        result.weight_preserved_other_bytes += peer.weight_preserved_other_bytes;
        result.qsa_state_bytes += peer.qsa_state_bytes;
        result.gdn_state_bytes += peer.gdn_state_bytes;
        result.ple_state_bytes += peer.ple_state_bytes;
        result.mtp_state_bytes += peer.mtp_state_bytes;
        result.workspace_device_bytes += peer.workspace_device_bytes;
        result.prefill_cache_device_bytes += peer.prefill_cache_device_bytes;
        result.boundary_pinned_bytes += peer.boundary_pinned_bytes;
        result.workspace_pinned_bytes += peer.workspace_pinned_bytes;
        result.cuda_tracked_allocated_bytes += peer.cuda_tracked_allocated_bytes;
        result.cuda_tracked_peak_bytes += peer.cuda_tracked_peak_bytes;
        result.cuda_device_free_bytes += peer.cuda_device_free_bytes;
        result.cuda_device_total_bytes += peer.cuda_device_total_bytes;
        result.cuda_allocator_retries += peer.cuda_allocator_retries;
        result.cuda_allocation_failures += peer.cuda_allocation_failures;
        result.cuda_graph_captures += peer.cuda_graph_captures;
        result.cuda_graph_replays += peer.cuda_graph_replays;
        result.cuda_graph_fallbacks += peer.cuda_graph_fallbacks;
        result.cuda_graph_nodes += peer.cuda_graph_nodes;
        result.cuda_graph_held_bytes += peer.cuda_graph_held_bytes;
        const auto bridge = te_bridge_stats(stage() == Stage::kStage0 ? 0 : 1);
        const auto* paired_graph = std::getenv("Q38_TP_PAIRED_GRAPH");
        if (paired_graph && std::strcmp(paired_graph, "1") == 0)
            std::fprintf(stderr, "{\"event\":\"tp_graph_census\",\"stage\":%u,\"paired_graph_calls\":%llu,\"host_collective_calls\":%llu}\n",
                unsigned(stage()), static_cast<unsigned long long>(bridge.paired_graph_calls),
                static_cast<unsigned long long>(bridge.allreduce_calls + bridge.allgather_calls));
        return result;
    }
private:
    template<class F> void both_cancellable(
        F action, const std::shared_ptr<CancellationToken>& cancellation) {
        try { both(std::move(action)); }
        catch (...) {
            // Both rank tasks have drained before both() rethrows. Preserve
            // the request's cancellation/deadline category even if the first
            // collected exception was the peer's aborted-collective error.
            if (cancellation) cancellation->throw_if_requested();
            throw;
        }
    }
    template<class F> void both(F action) {
        if (worker_) {
            try { worker_->run(std::move(action)); }
            catch (const std::exception& e) {
                throw std::runtime_error(std::string("TP2 rank failed: ") + e.what());
            }
            return;
        }
        std::exception_ptr errors[2];
        std::thread peer([&] {
            try { action(1); }
            catch (...) { errors[1] = std::current_exception(); te_bridge_abort(); }
        });
        try { action(0); }
        catch (...) { errors[0] = std::current_exception(); te_bridge_abort(); }
        peer.join();
        for (auto error : errors) if (error) {
            try { std::rethrow_exception(error); }
            catch (const std::exception& e) {
                // This prototype fails closed after cancellation/device errors;
                // another request cannot reuse a poisoned collective generation.
                throw std::runtime_error(std::string("TP2 rank failed: ") + e.what());
            }
        }
    }
    std::array<std::unique_ptr<StageBackend>, 2> ranks_;
    // Destroy worker before the rank backends it accesses.
    std::unique_ptr<PairedRankWorker> worker_;
};
} // namespace

std::unique_ptr<DualStageExecutor> make_cuda_tp_executor(
    const std::string& root, const std::string& ple_path,
    const ExecutorOptions& executor_options, std::uint64_t ple_cache_bytes,
    PleIoModeV1 ple_io_mode, std::uint32_t ple_queue_depth, bool enable_p2p,
    std::uint64_t ple_ram_budget_bytes, bool stream_ordered_p2p) {
    std::string identity_error;
    if (!validate_session_identity(executor_options.identity, &identity_error) ||
        (executor_options.identity.flags & kSessionIdentityDevelopment))
        throw std::invalid_argument("TP2 requires a production identity: " + identity_error);
    if ((executor_options.enable_adaptive_mtp && !executor_options.enable_batched_mtp) || executor_options.enable_compact_sampling)
        throw std::invalid_argument("TP2 adaptive MTP requires batch mode; compact sampling is unsupported");
    if (executor_options.enable_batched_mtp) {
        const char* batch = std::getenv("Q38_TP_BATCHED_TARGET");
        if (!batch || std::strcmp(batch, "1") != 0)
            throw std::invalid_argument("TP MTP requires batched target verification");
    }
    te_bridge_reset();
    te_bridge_use_p2p(enable_p2p, stream_ordered_p2p);
    PleStoreOptionsV1 ple_options;
    ple_options.cache_bytes = ple_cache_bytes;
    ple_options.io_mode = ple_io_mode;
    ple_options.queue_depth = ple_queue_depth;
    ple_options.ram_budget_bytes = ple_ram_budget_bytes;
    std::shared_ptr<PleStore> ram_store;
    if (ple_io_mode == PleIoModeV1::kRam)
        ram_store = std::make_shared<PleStore>(load_ple_layout(ple_path), ple_options);
    std::unique_ptr<StageBackend> ranks[2][2];
    for (int r = 0; r < 2; ++r) {
        const auto rank_root = root + "/rank" + std::to_string(r);
        auto first = load_device_stage_index(rank_root + "/stage0/index.q38d");
        auto second = load_device_stage_index(rank_root + "/stage1/index.q38d");
        validate_device_stage_pair(first, second);
        auto ple = ram_store ? ram_store : std::make_shared<PleStore>(load_ple_layout(ple_path), ple_options);
        CudaStageBackendOptions options;
        options.device = r;
        options.te_world_size = 2;
        options.te_rank = r;
        const char* graph = std::getenv("Q38_TP_PIECEWISE_GRAPH");
        options.enable_piecewise_decode_graph = graph && std::strcmp(graph, "1") == 0;
        options.max_transaction_tokens = executor_options.append_chunk_tokens;
        options.context_capacity = executor_options.context_limit;
        options.stage = Stage::kStage0;
        ranks[r][0] = std::make_unique<CudaStageBackend>(std::move(first), options, ple);
        options.stage = Stage::kStage1;
        options.enable_mtp = executor_options.enable_batched_mtp;
        ranks[r][1] = std::make_unique<CudaStageBackend>(std::move(second), options);
    }
    auto options = executor_options;
    options.backend_failure_is_fatal = true;
    options.backend_cancellation_is_fatal = true;
    const auto* recovery = std::getenv("Q38_TP_RECOVER_CANCEL");
    options.backend_cancellation_resettable = recovery && std::strcmp(recovery, "1") == 0;
    return std::make_unique<DualStageExecutor>(options,
        std::make_unique<TensorParallelStage>(std::move(ranks[0][0]), std::move(ranks[1][0])),
        std::make_unique<TensorParallelStage>(std::move(ranks[0][1]), std::move(ranks[1][1])));
}
} // namespace q38
