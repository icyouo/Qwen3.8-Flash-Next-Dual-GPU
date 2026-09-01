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
    // Diagnostic-only: retain the raw BF16 logits from the most recently
    // committed non-speculative target transaction. Disabled by default so
    // ordinary greedy execution does not add a device-to-host logits copy.
    bool retain_last_logits_for_diagnostics = false;
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
    void reset_session();

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<DualStageExecutor> make_mock_executor(
    const ExecutorOptions& options);

}  // namespace q38

#endif
