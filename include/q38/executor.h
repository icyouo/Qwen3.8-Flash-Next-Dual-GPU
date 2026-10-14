#ifndef Q38_EXECUTOR_H
#define Q38_EXECUTOR_H

#include "q38/backend.h"
#include "q38/identity.h"
#include "q38/metrics.h"
#include "q38/sampling.h"
#include "q38/snapshot.h"
#include "q38/transaction.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace q38 {

struct ExecutorOptions {
    std::uint64_t session_hash = 1;
    std::uint32_t hidden_width = 4 * 2560;
    std::uint32_t vocab_size = 248320;
    // Matches the grouped-MMQ slab so stage 0 and stage 1 overlap across a
    // request through the existing three-slot boundary ring.
    std::uint32_t append_chunk_tokens = 512;
    std::uint32_t context_limit = 262144;
    // A production device backend cannot safely retry after CUDA, transport,
    // or device-state errors. CPU fixtures keep this false so rollback and
    // retry behavior can still be exercised independently.
    bool backend_failure_is_fatal = false;
    // TP aborts its collective generation on cooperative cancellation. Keep
    // the cancellation response typed, but forbid reuse of that executor.
    // PP backends with recoverable cancellation leave this disabled.
    bool backend_cancellation_is_fatal = false;
    // A cooperative cancellation still blocks generation until explicit reset.
    // Enabled only for backends that can drain and rebuild their bridge state.
    bool backend_cancellation_resettable = false;
    // Diagnostic-only: retain the raw BF16 logits from the most recently
    // committed non-speculative target transaction. Disabled by default so
    // ordinary greedy execution does not add a device-to-host logits copy.
    bool retain_last_logits_for_diagnostics = false;
    bool enable_adaptive_mtp = false;
    // Draft into an ephemeral backend branch, roll it back, then verify all
    // candidates in one target batch. Accepted MTP state is rebuilt at commit.
    bool enable_batched_mtp = false;
    bool enable_compact_sampling = false;
    SamplingConfigV1 sampling{};
    // Stop tokens are part of the parser/sampling identity.  Speculative
    // commits are truncated before the first accepted stop token so no token
    // after it can become canonical.
    std::vector<std::int32_t> stop_token_ids;
    SessionIdentityV1 identity{};
};

struct DecodeResult {
    CommitEventV1 commit{};
    std::vector<std::int32_t> published_tokens;
};

struct LogitSnapshotV1 {
    std::uint64_t transaction_epoch = 0;
    std::uint64_t target_frontier = 0;
    TxnKind transaction_kind = TxnKind::kInvalid;
    std::int32_t selected_token = -1;
    // Raw model-head output before repetition/frequency/presence penalties or
    // sampling transforms. Values preserve their exact BF16 bit patterns.
    std::vector<std::uint16_t> logits_bf16;
};

class DualStageExecutor {
public:
    DualStageExecutor(ExecutorOptions options,
                      std::unique_ptr<StageBackend> stage0,
                      std::unique_ptr<StageBackend> stage1);
    ~DualStageExecutor();

    DualStageExecutor(const DualStageExecutor&) = delete;
    DualStageExecutor& operator=(const DualStageExecutor&) = delete;

    void append(const std::vector<std::int32_t>& token_ids,
                std::shared_ptr<CancellationToken> cancellation = {});
    std::int32_t seed_decode(
        std::shared_ptr<CancellationToken> cancellation = {});
    DecodeResult decode_one(
        std::shared_ptr<CancellationToken> cancellation = {});
    DecodeResult speculative_step(
        std::uint32_t max_draft,
        std::shared_ptr<CancellationToken> cancellation = {});
    // Verify supplied candidates in one bounded target transaction. Used by
    // batched drafting and correctness fixtures independently of draft source.
    // Sampled sessions retain ordinary target sampling until rejection
    // sampling with draft probabilities is implemented.
    DecodeResult verify_candidates(
        const std::vector<std::int32_t>& candidates,
        std::shared_ptr<CancellationToken> cancellation = {});
    void reset_session();
    void replace_pending_terminal(std::uint32_t expected_canonical,
                                  std::uint32_t expected_epoch,
                                  std::int32_t expected_token,
                                  std::int32_t replacement,
                                  std::shared_ptr<CancellationToken> cancellation = {});

    const SessionFrontiersV1& frontiers() const;
    const std::vector<std::int32_t>& canonical_tokens() const;
    ExecutorStatsV1 stats() const;
    MetricsSchemaV1 metrics();
    SamplerStateV1 sampler_state() const;
    const SessionIdentityV1& identity() const;
    StateSnapshotV1 snapshot() const;
    std::optional<LogitSnapshotV1> diagnostic_logit_snapshot() const;
    void fail_closed();
    bool poisoned() const;
    bool cancellation_resettable() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<DualStageExecutor> make_mock_executor(
    const ExecutorOptions& options);

}  // namespace q38

#endif
