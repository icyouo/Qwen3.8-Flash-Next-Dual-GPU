#include "q38/ple_backend.h"

#include <stdexcept>
#include <utility>

namespace q38 {

namespace {

std::uint64_t hash_bytes(const std::vector<std::uint8_t>& bytes,
                         std::uint64_t seed) {
    auto digest = seed;
    for (const auto byte : bytes) {
        digest ^= byte;
        digest *= UINT64_C(1099511628211);
    }
    return digest;
}

}  // namespace

PleStage0Backend::PleStage0Backend(std::unique_ptr<StageBackend> inner,
                                   std::shared_ptr<PleStore> store,
                                   PleHashConfigV1 hash_config)
    : inner_(std::move(inner)), store_(std::move(store)),
      committed_hash_(hash_config) {
    if (!inner_ || inner_->stage() != Stage::kStage0 || !store_)
        throw std::invalid_argument("PLE backend requires stage0 and a store");
}

void PleStage0Backend::prefetch_transaction(
    const SessionTxnV1& txn,
    const std::vector<std::int32_t>& token_ids,
    std::shared_ptr<CancellationToken> cancellation) {
    inner_->prefetch_transaction(txn, token_ids, std::move(cancellation));
}

StageOutput PleStage0Backend::execute(StageInput input) {
    const auto epoch = input.txn.epoch;
    const auto expected = input.txn.evaluated_count;
    const auto token_ids = input.token_ids;
    const auto cancellation = input.cancellation;
    const auto offset = input.chunk_offset;
    const bool first = provisional_epoch_ == 0;
    if ((!first && (epoch != provisional_epoch_ ||
                    input.txn.evaluated_count != provisional_expected_ ||
                    offset != provisional_processed_)) ||
        (first && offset != 0))
        throw std::runtime_error("PLE backend received a noncontiguous chunk");
    auto output = inner_->execute(std::move(input));
    try {
        if (first) {
            candidate_hashes_.clear();
            candidate_hashes_.reserve(expected);
            provisional_epoch_ = epoch;
            provisional_expected_ = expected;
        }
        auto candidate = candidate_hashes_.empty()
                             ? committed_hash_
                             : candidate_hashes_.back();
        auto digest = output.state_digest;
        for (const auto token : token_ids) {
            if (cancellation) cancellation->throw_if_requested();
            const auto rows = candidate.rows({token});
            const auto bytes = store_->read_rows(rows);
            digest = hash_bytes(bytes, digest);
            candidate_hashes_.push_back(candidate);
        }
        if (cancellation) cancellation->throw_if_requested();
        output.state_digest = digest;
        provisional_processed_ += static_cast<std::uint32_t>(token_ids.size());
        return output;
    } catch (...) {
        candidate_hashes_.clear();
        throw;
    }
}

std::vector<std::int32_t> PleStage0Backend::draft(
    std::int32_t pending_token, std::uint64_t position,
    std::uint32_t max_draft,
    std::shared_ptr<CancellationToken> cancellation) {
    return inner_->draft(pending_token, position, max_draft,
                         std::move(cancellation));
}

void PleStage0Backend::commit(std::uint64_t epoch,
                              std::uint32_t state_commit_count) {
    if (epoch != provisional_epoch_ || state_commit_count == 0 ||
        provisional_processed_ != provisional_expected_ ||
        state_commit_count > candidate_hashes_.size())
        throw std::runtime_error("PLE commit does not match candidate state");
    inner_->commit(epoch, state_commit_count);
    committed_hash_ = candidate_hashes_[state_commit_count - 1u];
    candidate_hashes_.clear();
    provisional_epoch_ = 0;
    provisional_expected_ = 0;
    provisional_processed_ = 0;
}

void PleStage0Backend::rollback(std::uint64_t epoch) {
    inner_->rollback(epoch);
    if (provisional_epoch_ != 0 && epoch != provisional_epoch_)
        throw std::runtime_error("PLE rollback epoch mismatch");
    candidate_hashes_.clear();
    provisional_epoch_ = 0;
    provisional_expected_ = 0;
    provisional_processed_ = 0;
}

void PleStage0Backend::reset_session() {
    if (provisional_epoch_ != 0)
        throw std::logic_error("cannot reset an active PLE transaction");
    inner_->reset_session();
    committed_hash_.reset();
    candidate_hashes_.clear();
    provisional_expected_ = 0;
    provisional_processed_ = 0;
}

StageBackendMetricsV1 PleStage0Backend::metrics() const {
    auto result = inner_->metrics();
    const auto cache = store_->cache_stats();
    result.ple_cache_hits = cache.hits;
    result.ple_cache_misses = cache.misses;
    result.ple_cache_evictions = cache.evictions;
    result.ple_cache_resident_bytes = cache.resident_bytes;
    result.ple_cache_capacity_bytes = cache.capacity_bytes;
    result.ple_requested_rows = cache.requested_rows;
    result.ple_unique_page_requests = cache.unique_page_requests;
    result.ple_useful_bytes = cache.useful_bytes;
    result.ple_scale_resident_bytes = cache.scale_resident_bytes;
    result.ple_physical_read_bytes = cache.physical_read_bytes;
    result.ple_read_operations = cache.read_operations;
    result.ple_read_batches = cache.read_batches;
    result.ple_io_uring_submissions = cache.io_uring_submissions;
    result.ple_io_uring_completions = cache.io_uring_completions;
    result.ple_direct_read_bytes = cache.direct_read_bytes;
    result.ple_read_errors = cache.read_errors;
    result.ple_maximum_queue_depth = cache.maximum_queue_depth;
    result.ple_read_latency_p50_ns = cache.read_latency_p50_ns;
    result.ple_read_latency_p95_ns = cache.read_latency_p95_ns;
    result.ple_read_latency_p99_ns = cache.read_latency_p99_ns;
    result.ple_io_uring_enabled = cache.io_uring_enabled;
    result.ple_direct_io_enabled = cache.direct_io_enabled;
    return result;
}

}  // namespace q38
