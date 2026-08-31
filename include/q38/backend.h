#ifndef Q38_BACKEND_H
#define Q38_BACKEND_H

#include "q38/contracts.h"
#include "q38/metrics.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace q38 {

class ExecutionCancelled : public std::runtime_error {
public:
    ExecutionCancelled() : std::runtime_error("execution cancelled") {}
};

class ExecutionDeadlineExceeded : public std::runtime_error {
public:
    ExecutionDeadlineExceeded()
        : std::runtime_error("execution deadline exceeded") {}
};

// Cooperative request cancellation with an atomic commit point.  Once seal()
// succeeds the coordinator is allowed to commit and later cancellation is too
// late; before that point every device tile may observe cancellation and roll
// the whole transaction back.
class CancellationToken {
public:
    explicit CancellationToken(std::uint32_t timeout_ms = 0)
        : has_deadline_(timeout_ms != 0),
          deadline_(timeout_ms == 0
                        ? std::chrono::steady_clock::time_point::max()
                        : std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms)) {}

    bool request_cancel() {
        auto expected = State::kActive;
        return state_.compare_exchange_strong(expected, State::kCancelled,
                                              std::memory_order_acq_rel);
    }

    bool sealed() const {
        return state_.load(std::memory_order_acquire) == State::kSealed;
    }

    void throw_if_requested() const {
        expire_deadline();
        const auto state = state_.load(std::memory_order_acquire);
        if (state == State::kDeadline)
            throw ExecutionDeadlineExceeded();
        if (state == State::kCancelled)
            throw ExecutionCancelled();
    }

    void seal() {
        throw_if_requested();
        auto expected = State::kActive;
        if (!state_.compare_exchange_strong(expected, State::kSealed,
                                            std::memory_order_acq_rel))
            throw_if_requested();
    }

private:
    enum class State : std::uint8_t {
        kActive,
        kCancelled,
        kDeadline,
        kSealed
    };

    void expire_deadline() const {
        if (!has_deadline_ ||
            std::chrono::steady_clock::now() < deadline_)
            return;
        auto expected = State::kActive;
        (void)state_.compare_exchange_strong(expected, State::kDeadline,
                                             std::memory_order_acq_rel);
    }

    bool has_deadline_ = false;
    std::chrono::steady_clock::time_point deadline_;
    mutable std::atomic<State> state_{State::kActive};
};

class BoundaryLease {
public:
    virtual ~BoundaryLease() = default;
    virtual const std::uint16_t* data() const = 0;
    virtual std::size_t size() const = 0;
    virtual void wait_ready() const = 0;
    virtual std::uint32_t slot() const { return 0; }
    virtual std::uint64_t payload_checksum() const { return 0; }
};

struct BoundaryBuffer {
    StageBoundaryFrameV1 frame{};
    // Owned storage is used by fixtures and CPU backends. Production CUDA
    // stages attach a pinned ring lease instead, avoiding an 80 MiB copy for a
    // 4096-token 4H prefill boundary.
    std::vector<std::uint16_t> bf16;
    std::shared_ptr<const BoundaryLease> lease;

    const std::uint16_t* data() const {
        return lease ? lease->data() : bf16.data();
    }
    std::size_t size() const { return lease ? lease->size() : bf16.size(); }
    void wait_ready() const {
        if (lease) lease->wait_ready();
    }
    std::uint64_t payload_checksum() const {
        return lease ? lease->payload_checksum()
                     : boundary_payload_checksum(bf16.data(), bf16.size());
    }
};

struct StageInput {
    SessionTxnV1 txn{};
    std::vector<std::int32_t> token_ids;
    BoundaryBuffer boundary{};
    // A transaction may be larger than a device workspace.  Chunks are
    // contiguous, ordered pieces of txn.evaluated_count and remain
    // provisional until the final chunk is coordinated and committed.
    std::uint32_t chunk_offset = 0;
    bool final_chunk = true;
    bool need_logits = false;
    std::shared_ptr<CancellationToken> cancellation;
};

struct StageOutput {
    BoundaryBuffer boundary{};
    std::uint32_t state_commit_count = 0;
    std::int32_t next_token = -1;
    std::uint64_t state_digest = 0;
    // Populated only for a final target step that requested transactional
    // host sampling. Greedy and intermediate prefill chunks keep this empty.
    std::vector<std::uint16_t> logits_bf16;
};

class StageBackend {
public:
    virtual ~StageBackend() = default;
    virtual Stage stage() const = 0;
    virtual StageOutput execute(StageInput input) = 0;
    virtual std::vector<std::int32_t> draft(std::int32_t pending_token,
                                            std::uint64_t position,
                                            std::uint32_t max_draft,
                                            std::shared_ptr<CancellationToken>
                                                cancellation = {});
    virtual void commit(std::uint64_t epoch,
                        std::uint32_t state_commit_count) = 0;
    virtual void rollback(std::uint64_t epoch) = 0;
    virtual StageBackendMetricsV1 metrics() const;
};

}  // namespace q38

#endif
