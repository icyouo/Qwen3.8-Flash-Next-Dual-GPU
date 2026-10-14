#include "q38/artifact.h"
#include "q38/adaptive_mtp.h"
#include "q38/contracts.h"
#include "q38/device_artifact.h"
#include "q38/executor.h"
#include "q38/identity.h"
#include "q38/mapped_weights.h"
#include "q38/mock_backend.h"
#include "q38/ple.h"
#include "q38/ple_prefetch.h"
#include "q38/rpc.h"
#include "q38/snapshot.h"
#include "q38/transaction.h"
#include "q38/tensor_index.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <future>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>
#include <string>

namespace {

int failures = 0;

class TestBoundaryLease final : public q38::BoundaryLease {
public:
    explicit TestBoundaryLease(std::size_t words) : words_(words, 7) {}
    const std::uint16_t* data() const override { return words_.data(); }
    std::size_t size() const override { return words_.size(); }
    void wait_ready() const override { waited_ = true; }
    bool waited() const { return waited_; }

private:
    std::vector<std::uint16_t> words_;
    mutable bool waited_ = false;
};

class FailOnceBackend final : public q38::StageBackend {
public:
    enum class Point { kExecute, kCommit };

    FailOnceBackend(std::unique_ptr<q38::StageBackend> inner, Point point,
                    std::uint32_t execute_call = 1)
        : inner_(std::move(inner)), point_(point),
          execute_call_(execute_call) {}

    q38::Stage stage() const override { return inner_->stage(); }

    q38::StageOutput execute(q38::StageInput input) override {
        ++execute_calls_;
        if (!failed_ && point_ == Point::kExecute &&
            execute_calls_ == execute_call_) {
            failed_ = true;
            throw std::runtime_error("injected execute failure");
        }
        return inner_->execute(std::move(input));
    }

    std::vector<std::int32_t> draft(std::int32_t pending_token,
                                    std::uint64_t position,
                                    std::uint32_t max_draft,
                                    std::shared_ptr<q38::CancellationToken>
                                        cancellation = {}) override {
        return inner_->draft(pending_token, position, max_draft,
                             std::move(cancellation));
    }

    void commit(std::uint64_t epoch, std::uint32_t count) override {
        if (!failed_ && point_ == Point::kCommit) {
            failed_ = true;
            throw std::runtime_error("injected commit failure");
        }
        inner_->commit(epoch, count);
    }

    void rollback(std::uint64_t epoch) override { inner_->rollback(epoch); }

private:
    std::unique_ptr<q38::StageBackend> inner_;
    Point point_;
    std::uint32_t execute_call_ = 1;
    std::uint32_t execute_calls_ = 0;
    bool failed_ = false;
};

class DraftFaultBackend final : public q38::StageBackend {
public:
    DraftFaultBackend(std::unique_ptr<q38::StageBackend> inner, int mode)
        : inner_(std::move(inner)), mode_(mode) {}
    q38::Stage stage() const override { return inner_->stage(); }
    q38::StageOutput execute(q38::StageInput input) override { return inner_->execute(std::move(input)); }
    std::vector<std::int32_t> draft(std::int32_t token, std::uint64_t position,
        std::uint32_t width, std::shared_ptr<q38::CancellationToken> cancel) override {
        auto result = inner_->draft(token, position, width, cancel);
        if (!faulted_) {
            faulted_ = true;
            if (mode_ == 0) cancel->request_cancel();
            if (mode_ == 1) throw std::runtime_error("injected draft failure");
            if (mode_ == 2) result.pop_back();
            if (mode_ == 3) result.front() = -1;
            if (mode_ == 4) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return result;
    }
    void commit(std::uint64_t epoch, std::uint32_t count) override { inner_->commit(epoch, count); }
    void rollback(std::uint64_t epoch) override { inner_->rollback(epoch); }
    void reset_session() override { inner_->reset_session(); }
private:
    std::unique_ptr<q38::StageBackend> inner_;
    int mode_;
    bool faulted_ = false;
};

class CancelAfterCandidateExecuteBackend final : public q38::StageBackend {
public:
    explicit CancelAfterCandidateExecuteBackend(std::unique_ptr<q38::StageBackend> inner)
        : inner_(std::move(inner)) {}
    q38::Stage stage() const override { return inner_->stage(); }
    q38::StageOutput execute(q38::StageInput input) override {
        const auto cancellation = input.cancellation;
        const bool candidate = input.txn.kind == q38::TxnKind::kSpeculative;
        auto output = inner_->execute(std::move(input));
        if (candidate && cancellation && !cancelled_) {
            cancelled_ = true;
            cancellation->request_cancel();
        }
        return output;
    }
    void commit(std::uint64_t epoch, std::uint32_t count) override {
        inner_->commit(epoch, count);
    }
    void rollback(std::uint64_t epoch) override { inner_->rollback(epoch); }
private:
    std::unique_ptr<q38::StageBackend> inner_;
    bool cancelled_ = false;
};

class WaitForCancellationBackend final : public q38::StageBackend {
public:
    WaitForCancellationBackend(std::unique_ptr<q38::StageBackend> inner,
                               std::shared_ptr<std::atomic<bool>> entered)
        : inner_(std::move(inner)), entered_(std::move(entered)) {}

    q38::Stage stage() const override { return inner_->stage(); }

    q38::StageOutput execute(q38::StageInput input) override {
        if (!input.cancellation)
            throw std::runtime_error("test backend requires cancellation");
        entered_->store(true, std::memory_order_release);
        for (;;) {
            input.cancellation->throw_if_requested();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::vector<std::int32_t> draft(
        std::int32_t pending_token, std::uint64_t position,
        std::uint32_t max_draft,
        std::shared_ptr<q38::CancellationToken> cancellation = {}) override {
        return inner_->draft(pending_token, position, max_draft,
                             std::move(cancellation));
    }

    void commit(std::uint64_t epoch, std::uint32_t count) override {
        inner_->commit(epoch, count);
    }
    void rollback(std::uint64_t epoch) override { inner_->rollback(epoch); }

private:
    std::unique_ptr<q38::StageBackend> inner_;
    std::shared_ptr<std::atomic<bool>> entered_;
};

class CancelAfterRetainedDraftBackend final : public q38::StageBackend {
public:
    CancelAfterRetainedDraftBackend(
        std::unique_ptr<q38::StageBackend> inner,
        std::shared_ptr<std::atomic<std::uint32_t>> rolled_back)
        : inner_(std::move(inner)), rolled_back_(std::move(rolled_back)) {}

    q38::Stage stage() const override { return inner_->stage(); }
    q38::StageOutput execute(q38::StageInput input) override {
        return inner_->execute(std::move(input));
    }
    std::vector<std::int32_t> draft(
        std::int32_t pending_token, std::uint64_t position,
        std::uint32_t max_draft,
        std::shared_ptr<q38::CancellationToken> cancellation = {}) override {
        return inner_->draft(pending_token, position, max_draft,
                             std::move(cancellation));
    }
    std::vector<std::int32_t> draft_retained(
        std::int32_t pending_token, std::uint64_t position,
        std::uint32_t max_draft, std::uint64_t transaction_epoch,
        std::shared_ptr<q38::CancellationToken> cancellation = {}) override {
        auto result = inner_->draft_retained(
            pending_token, position, max_draft, transaction_epoch,
            cancellation);
        if (!cancelled_once_ && cancellation) {
            cancelled_once_ = true;
            (void)cancellation->request_cancel();
        }
        return result;
    }
    void commit(std::uint64_t epoch, std::uint32_t count) override {
        inner_->commit(epoch, count);
    }
    void rollback(std::uint64_t epoch) override {
        inner_->rollback(epoch);
        rolled_back_->fetch_add(1, std::memory_order_relaxed);
    }
    q38::StageBackendMetricsV1 metrics() const override {
        return inner_->metrics();
    }

private:
    std::unique_ptr<q38::StageBackend> inner_;
    std::shared_ptr<std::atomic<std::uint32_t>> rolled_back_;
    bool cancelled_once_ = false;
};

struct DraftStage0OverlapState {
    std::atomic<bool> armed{false};
    std::atomic<bool> stage0_entered{false};
    std::atomic<bool> draft_entered{false};
};

void wait_for_overlap_peer(const std::atomic<bool>& peer) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (!peer.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error(
                "retained draft and stage0 did not overlap");
        std::this_thread::yield();
    }
}

class OverlapStage0Backend final : public q38::StageBackend {
public:
    OverlapStage0Backend(std::unique_ptr<q38::StageBackend> inner,
                         std::shared_ptr<DraftStage0OverlapState> state)
        : inner_(std::move(inner)), state_(std::move(state)) {}

    q38::Stage stage() const override { return inner_->stage(); }
    q38::StageOutput execute(q38::StageInput input) override {
        if (state_->armed.load(std::memory_order_acquire)) {
            state_->stage0_entered.store(true, std::memory_order_release);
            wait_for_overlap_peer(state_->draft_entered);
        }
        return inner_->execute(std::move(input));
    }
    void commit(std::uint64_t epoch, std::uint32_t count) override {
        inner_->commit(epoch, count);
    }
    void rollback(std::uint64_t epoch) override { inner_->rollback(epoch); }
    q38::StageBackendMetricsV1 metrics() const override {
        return inner_->metrics();
    }

private:
    std::unique_ptr<q38::StageBackend> inner_;
    std::shared_ptr<DraftStage0OverlapState> state_;
};

class OverlapDraftBackend final : public q38::StageBackend {
public:
    OverlapDraftBackend(std::unique_ptr<q38::StageBackend> inner,
                        std::shared_ptr<DraftStage0OverlapState> state)
        : inner_(std::move(inner)), state_(std::move(state)) {}

    q38::Stage stage() const override { return inner_->stage(); }
    q38::StageOutput execute(q38::StageInput input) override {
        return inner_->execute(std::move(input));
    }
    std::vector<std::int32_t> draft(
        std::int32_t pending_token, std::uint64_t position,
        std::uint32_t max_draft,
        std::shared_ptr<q38::CancellationToken> cancellation = {}) override {
        return inner_->draft(pending_token, position, max_draft,
                             std::move(cancellation));
    }
    std::vector<std::int32_t> draft_retained(
        std::int32_t pending_token, std::uint64_t position,
        std::uint32_t max_draft, std::uint64_t transaction_epoch,
        std::shared_ptr<q38::CancellationToken> cancellation = {}) override {
        if (state_->armed.load(std::memory_order_acquire)) {
            state_->draft_entered.store(true, std::memory_order_release);
            wait_for_overlap_peer(state_->stage0_entered);
        }
        return inner_->draft_retained(pending_token, position, max_draft,
                                      transaction_epoch,
                                      std::move(cancellation));
    }
    void commit(std::uint64_t epoch, std::uint32_t count) override {
        inner_->commit(epoch, count);
    }
    void rollback(std::uint64_t epoch) override { inner_->rollback(epoch); }
    q38::StageBackendMetricsV1 metrics() const override {
        return inner_->metrics();
    }

private:
    std::unique_ptr<q38::StageBackend> inner_;
    std::shared_ptr<DraftStage0OverlapState> state_;
};

#define CHECK(expr)                                                          \
    do {                                                                     \
        if (!(expr)) {                                                       \
            std::cerr << __FILE__ << ':' << __LINE__                         \
                      << ": CHECK failed: " #expr << '\n';                 \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

void ack_and_decide(q38::SessionTxnEngine* engine, std::uint32_t commit) {
    std::string error;
    const auto epoch = engine->active_transaction().epoch;
    const auto count = engine->active_transaction().evaluated_count;
    CHECK(engine->acknowledge(q38::Stage::kStage0, epoch, count, &error));
    CHECK(engine->acknowledge(q38::Stage::kStage1, epoch, count, &error));
    CHECK(engine->decide(commit, &error));
}

void test_boundary_contract() {
    q38::StageBoundaryFrameV1 frame;
    frame.session_hash = 7;
    frame.epoch = 9;
    frame.token_count = 1;
    frame.hidden_dtype = q38::DType::kBFloat16;
    frame.producer_status = q38::ProducerStatus::kReady;
    frame.hidden_width = 4 * 2560;
    frame.payload_bytes = 20480;
    std::string error;
    CHECK(q38::validate_boundary(frame, 8192, &error));
    frame.token_count = 4;
    frame.payload_bytes = 81920;
    CHECK(q38::validate_boundary(frame, 8192, &error));
    frame.payload_bytes = 40960;
    CHECK(!q38::validate_boundary(frame, 8192, &error));

    auto lease = std::make_shared<TestBoundaryLease>(10240);
    q38::BoundaryBuffer buffer;
    buffer.lease = lease;
    CHECK(buffer.size() == 10240);
    CHECK(buffer.data()[0] == 7);
    buffer.wait_ready();
    CHECK(lease->waited());

    std::array<std::uint16_t, 4> payload{1, 2, 3, 4};
    const auto checksum =
        q38::boundary_payload_checksum(payload.data(), payload.size());
    CHECK(checksum ==
          q38::boundary_payload_checksum(payload.data(), payload.size()));
    payload[2] ^= 1u;
    CHECK(checksum !=
          q38::boundary_payload_checksum(payload.data(), payload.size()));
}

void test_boundary_checksum_rejects_corruption() {
    constexpr std::uint32_t hidden = 16;
    q38::MockStageBackend stage0(q38::Stage::kStage0, hidden, 128);
    q38::MockStageBackend stage1(q38::Stage::kStage1, hidden, 128);
    q38::SessionTxnV1 txn;
    txn.session_hash = 0x38;
    txn.epoch = 1;
    txn.evaluated_count = 1;
    txn.kind = q38::TxnKind::kAppendKnown;
    txn.status = q38::TxnStatus::kPrepared;
    q38::StageInput input0;
    input0.txn = txn;
    input0.token_ids = {7};
    auto output0 = stage0.execute(std::move(input0));
    output0.boundary.bf16[0] ^= 1u;
    q38::StageInput input1;
    input1.txn = txn;
    input1.token_ids = {7};
    input1.boundary = std::move(output0.boundary);
    bool rejected = false;
    try {
        (void)stage1.execute(std::move(input1));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CHECK(rejected);
}

void test_transactions() {
    {
        q38::SessionTxnEngine atomic(0x170);
        std::string atomic_error;
        CHECK(atomic.prepare_append_known(8, &atomic_error));
        CHECK(atomic.frontiers().canonical == 8);
        CHECK(atomic.frontiers().target == 0);
        CHECK(atomic.active_transaction().base_canonical == 0);
        CHECK(atomic.rollback(&atomic_error));
        CHECK(atomic.frontiers().canonical == 0);
        CHECK(atomic.frontiers().target == 0);
        CHECK(atomic.frontiers().epoch == 1);
    }

    {
        q38::SessionTxnEngine continuation(0x171);
        std::string continuation_error;
        CHECK(continuation.prepare_append_known(4, &continuation_error));
        ack_and_decide(&continuation, 4);
        q38::CommitEventV1 continuation_event;
        CHECK(continuation.commit(&continuation_event, &continuation_error));
        CHECK(continuation.seed_decode_pending(&continuation_error));
        CHECK(continuation.frontiers().canonical == 5);
        CHECK(continuation.frontiers().target == 4);
        CHECK(continuation.prepare_append_known(3, 4, &continuation_error));
        CHECK(continuation.active_transaction().base_target == 4);
        CHECK(continuation.active_transaction().base_canonical == 5);
        CHECK(continuation.active_transaction().evaluated_count == 4);
        CHECK(continuation.frontiers().canonical == 8);
        CHECK(continuation.rollback(&continuation_error));
        CHECK(continuation.frontiers().canonical == 5);
        CHECK(continuation.frontiers().target == 4);
        CHECK(continuation.prepare_append_known(3, 4, &continuation_error));
        ack_and_decide(&continuation, 4);
        CHECK(continuation.commit(&continuation_event, &continuation_error));
        CHECK(continuation.frontiers().canonical == 8);
        CHECK(continuation.frontiers().target == 8);
    }

    q38::SessionTxnEngine engine(0x1234);
    std::string error;
    CHECK(engine.append_known(8, &error));
    CHECK(engine.prepare(q38::TxnKind::kAppendKnown, 3, &error));
    ack_and_decide(&engine, 3);
    q38::CommitEventV1 event;
    CHECK(engine.commit(&event, &error));
    CHECK(engine.frontiers().canonical == 8);
    CHECK(engine.frontiers().target == 3);
    CHECK(event.newly_published == 0);

    CHECK(engine.prepare(q38::TxnKind::kAppendKnown, 5, &error));
    ack_and_decide(&engine, 5);
    CHECK(engine.commit(&event, &error));
    CHECK(engine.frontiers().canonical == 8);
    CHECK(engine.frontiers().target == 8);

    CHECK(engine.seed_decode_pending(&error));
    CHECK(engine.prepare(q38::TxnKind::kDecode, 1, &error));
    ack_and_decide(&engine, 1);
    CHECK(engine.commit(&event, &error));
    CHECK(engine.frontiers().target == 9);
    CHECK(engine.frontiers().canonical == 10);
    CHECK(event.newly_published == 1);

    CHECK(engine.prepare(q38::TxnKind::kSpeculative, 4, &error));
    ack_and_decide(&engine, 3);
    CHECK(engine.commit(&event, &error));
    CHECK(engine.frontiers().target == 12);
    CHECK(engine.frontiers().canonical == 13);
    CHECK(event.newly_published == 3);

    const auto before = engine.frontiers();
    CHECK(engine.prepare(q38::TxnKind::kSpeculative, 6, &error));
    CHECK(engine.rollback(&error));
    CHECK(engine.frontiers().canonical == before.canonical);
    CHECK(engine.frontiers().target == before.target);
    CHECK(engine.frontiers().epoch == before.epoch + 1);
}

void test_rejects_bad_ack() {
    q38::SessionTxnEngine engine(99);
    std::string error;
    CHECK(engine.append_known(2, &error));
    CHECK(engine.prepare(q38::TxnKind::kAppendKnown, 2, &error));
    const auto epoch = engine.active_transaction().epoch;
    CHECK(!engine.acknowledge(q38::Stage::kStage0, epoch + 1, 2, &error));
    CHECK(engine.acknowledge(q38::Stage::kStage0, epoch, 2, &error));
    CHECK(!engine.acknowledge(q38::Stage::kStage0, epoch, 2, &error));
    CHECK(engine.acknowledge(q38::Stage::kStage1, epoch, 2, &error));
    CHECK(engine.decide(2, &error));
    q38::CommitEventV1 event;
    CHECK(engine.commit(&event, &error));
}

void test_shortens_only_unacknowledged_speculation() {
    q38::SessionTxnEngine engine(0x41);
    std::string error;
    CHECK(engine.append_known(1, &error));
    CHECK(engine.prepare(q38::TxnKind::kSpeculative, 5, &error));
    const auto epoch = engine.active_transaction().epoch;
    CHECK(!engine.shorten_speculative(0, &error));
    CHECK(!engine.shorten_speculative(5, &error));
    CHECK(engine.shorten_speculative(3, &error));
    CHECK(engine.active_transaction().evaluated_count == 3);
    CHECK(engine.acknowledge(q38::Stage::kStage0, epoch, 3, &error));
    CHECK(!engine.shorten_speculative(2, &error));
    CHECK(engine.acknowledge(q38::Stage::kStage1, epoch, 3, &error));
    CHECK(engine.decide(3, &error));
    q38::CommitEventV1 event;
    CHECK(engine.commit(&event, &error));
    CHECK(event.newly_published == 3);
    CHECK(engine.frontiers().target == 3);
    CHECK(engine.frontiers().canonical == 4);

    CHECK(engine.prepare(q38::TxnKind::kAppendKnown, 1, &error));
    CHECK(!engine.shorten_speculative(1, &error));
    CHECK(engine.rollback(&error));
}

void test_randomized_frontiers() {
    q38::SessionTxnEngine engine(0xabcdef);
    std::mt19937 rng(0x170);
    std::string error;

    for (int round = 0; round < 2000; ++round) {
        const std::uint32_t suffix = 1 + (rng() % 32);
        CHECK(engine.append_known(suffix, &error));
        while (engine.frontiers().target < engine.frontiers().canonical) {
            const auto pending = engine.frontiers().canonical -
                                 engine.frontiers().target;
            const auto chunk = static_cast<std::uint32_t>(
                pending < 7 ? pending : 1 + (rng() % 7));
            CHECK(engine.prepare(q38::TxnKind::kAppendKnown, chunk, &error));
            if ((rng() % 31) == 0) {
                CHECK(engine.rollback(&error));
                continue;
            }
            ack_and_decide(&engine, chunk);
            q38::CommitEventV1 event;
            CHECK(engine.commit(&event, &error));
            CHECK(event.newly_published == 0);
        }
        CHECK(engine.seed_decode_pending(&error));
        const std::uint32_t width = 1 + (rng() % 6);
        const auto kind = width == 1 ? q38::TxnKind::kDecode
                                     : q38::TxnKind::kSpeculative;
        CHECK(engine.prepare(kind, width, &error));
        const std::uint32_t accepted = kind == q38::TxnKind::kDecode
                                           ? 1
                                           : 1 + (rng() % width);
        ack_and_decide(&engine, accepted);
        q38::CommitEventV1 event;
        CHECK(engine.commit(&event, &error));
        CHECK(engine.frontiers().canonical - engine.frontiers().target == 1);
        CHECK(engine.frontiers().stage0 == engine.frontiers().target);
        CHECK(engine.frontiers().stage1 == engine.frontiers().target);
        CHECK(event.newly_published == accepted);
    }
}

void test_pending_terminal_replacement() {
    q38::ExecutorOptions options;
    options.vocab_size = 1024;
    options.context_limit = 128;
    auto probe = q38::make_mock_executor(options);
    probe->append({1,2,3});
    const auto first = probe->seed_decode();
    const auto terminal = probe->decode_one().published_tokens.back();
    options.stop_token_ids = {terminal};
    auto candidate = q38::make_mock_executor(options);
    candidate->append({1,2,3});
    CHECK(candidate->seed_decode() == first);
    CHECK(candidate->decode_one().published_tokens.back() == terminal);
    const auto before = candidate->frontiers();
    const auto original = candidate->canonical_tokens();
    auto rejected = [&](std::uint32_t count,std::uint32_t epoch,std::int32_t expected,std::int32_t replacement) {
        bool failed = false;
        try {candidate->replace_pending_terminal(count,epoch,expected,replacement);}
        catch(const std::invalid_argument&){failed=true;}
        CHECK(failed);
        CHECK(candidate->canonical_tokens()==original);
        CHECK(candidate->frontiers().epoch==before.epoch);
    };
    rejected(4,before.epoch,terminal,7);
    rejected(5,before.epoch+1,terminal,7);
    rejected(5,before.epoch,terminal+1,7);
    rejected(5,before.epoch,terminal,-1);
    auto cancel = std::make_shared<q38::CancellationToken>();
    cancel->request_cancel();
    bool cancelled=false;
    try {candidate->replace_pending_terminal(5,before.epoch,terminal,7,cancel);}
    catch(const q38::ExecutionCancelled&){cancelled=true;}
    CHECK(cancelled);
    CHECK(candidate->canonical_tokens()==original);
    candidate->replace_pending_terminal(5,before.epoch,terminal,7);
    CHECK(candidate->canonical_tokens()==std::vector<std::int32_t>({1,2,3,first,7}));
    CHECK(candidate->frontiers().target==before.target);
    CHECK(candidate->frontiers().draft==before.draft);
    CHECK(candidate->frontiers().epoch==before.epoch+1);
    bool stale_seed=false;
    try {(void)candidate->seed_decode();}catch(const std::runtime_error&){stale_seed=true;}
    CHECK(stale_seed);
    candidate->append({8,9});
    auto oracle=q38::make_mock_executor(options);
    oracle->append({1,2,3,first,7,8,9});
    CHECK(candidate->sampler_state().penalty_checksum==oracle->sampler_state().penalty_checksum);
    CHECK(candidate->seed_decode()==oracle->seed_decode());
    for(int i=0;i<4;i++)CHECK(candidate->speculative_step(4).published_tokens==oracle->speculative_step(4).published_tokens);
    CHECK(candidate->canonical_tokens()==oracle->canonical_tokens());
}

void test_mock_executor() {
    q38::ExecutorOptions options;
    options.append_chunk_tokens = 3;
    auto executor = q38::make_mock_executor(options);
    executor->append({1, 2, 3, 4, 5, 6, 7});
    CHECK(executor->frontiers().canonical == 7);
    CHECK(executor->frontiers().target == 7);
    CHECK(executor->canonical_tokens().size() == 7);
    const auto first = executor->seed_decode();
    CHECK(first >= 0);
    CHECK(executor->frontiers().canonical == 8);
    CHECK(executor->frontiers().target == 7);
    const auto plain = executor->decode_one();
    CHECK(plain.published_tokens.size() == 1);
    CHECK(executor->frontiers().canonical - executor->frontiers().target == 1);
    const auto before_suffix = executor->canonical_tokens();
    executor->append({90, 91, 92, 93});
    CHECK(executor->frontiers().canonical == before_suffix.size() + 4);
    CHECK(executor->frontiers().target == executor->frontiers().canonical);
    CHECK(executor->canonical_tokens().size() == before_suffix.size() + 4);
    CHECK(std::equal(before_suffix.begin(), before_suffix.end(),
                     executor->canonical_tokens().begin()));
    CHECK(executor->canonical_tokens()[before_suffix.size()] == 90);
    (void)executor->seed_decode();
    for (int i = 0; i < 50; ++i) {
        const auto speculative = executor->speculative_step(5);
        CHECK(!speculative.published_tokens.empty());
        CHECK(speculative.published_tokens.size() <= 6);
        CHECK(executor->frontiers().canonical - executor->frontiers().target == 1);
        CHECK(executor->frontiers().stage0 == executor->frontiers().target);
        CHECK(executor->frontiers().stage1 == executor->frontiers().target);
        CHECK(executor->canonical_tokens().size() ==
              executor->frontiers().canonical);
    }
}

void test_supplied_candidate_verification() {
    q38::ExecutorOptions options;
    options.hidden_width = 8;
    options.vocab_size = 1024;
    options.context_limit = 256;
    const std::vector<std::int32_t> prompt{11, 12, 13};
    auto oracle = q38::make_mock_executor(options);
    oracle->append(prompt); (void)oracle->seed_decode();
    std::vector<std::int32_t> continuation;
    for (unsigned i = 0; i < 65; ++i)
        continuation.push_back(oracle->decode_one().published_tokens.front());
    for (unsigned width : {1u, 2u, 4u, 6u, 16u, 64u}) {
        for (unsigned accepted = 0; accepted <= width; ++accepted) {
            auto executor = q38::make_mock_executor(options);
            auto reference = q38::make_mock_executor(options);
            executor->append(prompt); (void)executor->seed_decode();
            reference->append(prompt); (void)reference->seed_decode();
            std::vector<std::int32_t> draft(continuation.begin(), continuation.begin() + width);
            if (accepted < width) draft[accepted] = (draft[accepted] + 1) % options.vocab_size;
            const auto before = executor->metrics();
            const auto result = executor->verify_candidates(draft);
            CHECK(result.published_tokens == std::vector<std::int32_t>(
                continuation.begin(), continuation.begin() + accepted + 1));
            const auto after = executor->metrics();
            CHECK(after.stage0.execute_calls == before.stage0.execute_calls + 1);
            CHECK(after.stage1.execute_calls == before.stage1.execute_calls + 1);
            for (unsigned i = 0; i <= accepted; ++i) (void)reference->decode_one();
            CHECK(executor->canonical_tokens() == reference->canonical_tokens());
            CHECK(executor->frontiers().target == reference->frontiers().target);
            for (unsigned i = 0; i < 3; ++i)
                CHECK(executor->decode_one().published_tokens == reference->decode_one().published_tokens);
        }
    }
    options.stop_token_ids = {continuation[2]};
    auto stopped = q38::make_mock_executor(options);
    stopped->append(prompt); (void)stopped->seed_decode();
    auto result = stopped->verify_candidates(std::vector<std::int32_t>(continuation.begin(), continuation.begin() + 6));
    CHECK(result.published_tokens == std::vector<std::int32_t>(continuation.begin(), continuation.begin() + 3));
    options.stop_token_ids.clear();
    options.context_limit = 5;
    auto tail = q38::make_mock_executor(options);
    tail->append(prompt); (void)tail->seed_decode();
    CHECK(tail->verify_candidates({continuation[0], continuation[1]}).published_tokens.size() == 1);
    CHECK(tail->frontiers().canonical == options.context_limit);
    options.context_limit = 256;
    auto cancelled = q38::make_mock_executor(options);
    cancelled->append(prompt); (void)cancelled->seed_decode();
    const auto before = cancelled->canonical_tokens();
    auto cancellation = std::make_shared<q38::CancellationToken>();
    cancellation->request_cancel();
    bool rejected = false;
    try { (void)cancelled->verify_candidates({continuation[0]}, cancellation); }
    catch (const q38::ExecutionCancelled&) { rejected = true; }
    CHECK(rejected && cancelled->canonical_tokens() == before);
    for (const auto& invalid : {std::vector<std::int32_t>{-1}, std::vector<std::int32_t>(65, 1)}) {
        rejected = false;
        try { (void)cancelled->verify_candidates(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected && cancelled->canonical_tokens() == before);
    }
    CHECK(cancelled->verify_candidates({continuation[0]}).published_tokens.size() == 2);

    options.append_chunk_tokens = 2;
    auto bounded = q38::make_mock_executor(options);
    bounded->append(prompt); (void)bounded->seed_decode();
    const auto bounded_before = bounded->canonical_tokens();
    rejected = false;
    try { (void)bounded->verify_candidates({continuation[0], continuation[1]}); }
    catch (const std::length_error&) { rejected = true; }
    CHECK(rejected && bounded->canonical_tokens() == bounded_before);
    CHECK(bounded->verify_candidates({continuation[0]}).published_tokens.size() == 2);

    options.append_chunk_tokens = 512;
    for (const auto failing_stage : {q38::Stage::kStage0, q38::Stage::kStage1}) {
        auto backend = [&](q38::Stage stage) -> std::unique_ptr<q38::StageBackend> {
            auto inner = std::make_unique<q38::MockStageBackend>(
                stage, options.hidden_width, options.vocab_size);
            if (stage == failing_stage)
                return std::make_unique<FailOnceBackend>(std::move(inner),
                    FailOnceBackend::Point::kExecute, 2);
            return inner;
        };
        q38::DualStageExecutor failing(options, backend(q38::Stage::kStage0),
                                      backend(q38::Stage::kStage1));
        failing.append(prompt); (void)failing.seed_decode();
        const auto before_failure = failing.canonical_tokens();
        rejected = false;
        try { (void)failing.verify_candidates({continuation[0], continuation[1]}); }
        catch (const std::runtime_error&) { rejected = true; }
        CHECK(rejected && !failing.poisoned());
        CHECK(failing.canonical_tokens() == before_failure);
        CHECK(failing.frontiers().target == prompt.size());
        CHECK(failing.verify_candidates({continuation[0], continuation[1]}).published_tokens ==
              std::vector<std::int32_t>(continuation.begin(), continuation.begin() + 3));
    }
    for (const auto cancelling_stage : {q38::Stage::kStage0, q38::Stage::kStage1}) {
        auto backend = [&](q38::Stage stage) -> std::unique_ptr<q38::StageBackend> {
            auto inner = std::make_unique<q38::MockStageBackend>(
                stage, options.hidden_width, options.vocab_size);
            if (stage == cancelling_stage)
                return std::make_unique<CancelAfterCandidateExecuteBackend>(std::move(inner));
            return inner;
        };
        q38::DualStageExecutor interrupted(options, backend(q38::Stage::kStage0),
                                          backend(q38::Stage::kStage1));
        interrupted.append(prompt); (void)interrupted.seed_decode();
        const auto before_cancel = interrupted.canonical_tokens();
        rejected = false;
        try { (void)interrupted.verify_candidates({continuation[0], continuation[1]},
                  std::make_shared<q38::CancellationToken>()); }
        catch (const q38::ExecutionCancelled&) { rejected = true; }
        CHECK(rejected && !interrupted.poisoned());
        CHECK(interrupted.canonical_tokens() == before_cancel);
        CHECK(interrupted.frontiers().target == prompt.size());
        CHECK(interrupted.verify_candidates({continuation[0], continuation[1]}).published_tokens ==
              std::vector<std::int32_t>(continuation.begin(), continuation.begin() + 3));
    }
}

void test_batched_mtp_draft_faults() {
    for (int mode = 0; mode < 4; ++mode) {
        for (bool fatal : {false, true}) {
            q38::ExecutorOptions options;
            options.enable_batched_mtp = true;
            options.backend_failure_is_fatal = fatal;
            q38::DualStageExecutor executor(options,
                std::make_unique<q38::MockStageBackend>(q38::Stage::kStage0, options.hidden_width, options.vocab_size),
                std::make_unique<DraftFaultBackend>(std::make_unique<q38::MockStageBackend>(
                    q38::Stage::kStage1, options.hidden_width, options.vocab_size), mode));
            executor.append({11, 12, 13});
            (void)executor.seed_decode();
            const auto before = executor.metrics();
            bool caught = false;
            try { (void)executor.speculative_step(4, std::make_shared<q38::CancellationToken>()); }
            catch (const std::exception&) { caught = true; }
            CHECK(caught);
            CHECK(executor.frontiers().canonical == before.frontiers.canonical);
            CHECK(executor.frontiers().target == before.frontiers.target);
            CHECK(executor.metrics().stage0.execute_calls == before.stage0.execute_calls);
            CHECK(executor.poisoned() == (fatal && mode != 0));
            if (!executor.poisoned()) CHECK(!executor.speculative_step(4).published_tokens.empty());
        }
    }
}

void test_batched_mtp_boundaries() {
    q38::ExecutorOptions options;
    options.enable_batched_mtp = true;
    options.context_limit = 8;
    options.append_chunk_tokens = 8;
    auto oracle = q38::make_mock_executor(options);
    oracle->append({1, 2, 3});
    (void)oracle->seed_decode();
    const auto stop = oracle->decode_one().published_tokens.front();
    options.stop_token_ids = {stop};
    auto stopped = q38::make_mock_executor(options);
    stopped->append({1, 2, 3});
    (void)stopped->seed_decode();
    CHECK(stopped->speculative_step(6).published_tokens == std::vector<std::int32_t>{stop});
    options.stop_token_ids.clear();
    auto tail = q38::make_mock_executor(options);
    tail->append({1, 2, 3});
    (void)tail->seed_decode();
    while (tail->frontiers().canonical < options.context_limit)
        CHECK(!tail->speculative_step(6).published_tokens.empty());
    CHECK(tail->frontiers().canonical == options.context_limit);
    bool rejected = false;
    try { (void)tail->speculative_step(6); } catch (const std::length_error&) { rejected = true; }
    CHECK(rejected);
    options.context_limit = 32;
    options.append_chunk_tokens = 2;
    auto bounded = q38::make_mock_executor(options);
    bounded->append({1, 2, 3});
    (void)bounded->seed_decode();
    const auto before = bounded->frontiers();
    rejected = false;
    try { (void)bounded->speculative_step(2); } catch (const std::length_error&) { rejected = true; }
    CHECK(rejected);
    CHECK(bounded->frontiers().target == before.target);
    CHECK(bounded->stats().drafted_tokens == 0);
    auto cancel = std::make_shared<q38::CancellationToken>();
    cancel->request_cancel();
    rejected = false;
    try { (void)bounded->speculative_step(1, cancel); } catch (const q38::ExecutionCancelled&) { rejected = true; }
    CHECK(rejected);
    CHECK(!bounded->speculative_step(1).published_tokens.empty());
    options.sampling.mode = q38::SamplingModeV1::kTopKTopP;
    options.sampling.top_k = 3;
    options.vocab_size = 100;
    auto sampled = q38::make_mock_executor(options);
    sampled->append({1, 2, 3});
    (void)sampled->seed_decode();
    CHECK(sampled->speculative_step(6).published_tokens.size() == 1);
    CHECK(sampled->stats().drafted_tokens == 0);
    CHECK(sampled->stats().speculative_transactions == 0);
}

void test_batched_mtp_pipeline() {
    for (const auto width : {1u, 2u, 4u, 6u}) {
        q38::ExecutorOptions options;
        options.append_chunk_tokens = 8;
        options.enable_batched_mtp = true;
        auto candidate = q38::make_mock_executor(options);
        options.enable_batched_mtp = false;
        auto oracle = q38::make_mock_executor(options);
        for (int reset = 0; reset < 2; ++reset) {
            candidate->reset_session();
            oracle->reset_session();
            candidate->append({31, 32, 33, 34, 35, 36, 37});
            oracle->append({31, 32, 33, 34, 35, 36, 37});
            CHECK(candidate->seed_decode() == oracle->seed_decode());
            for (int step = 0; step < 128; ++step) {
                const auto before = candidate->metrics();
                const auto result = candidate->speculative_step(width);
                const auto after = candidate->metrics();
                CHECK(after.stage0.execute_calls == before.stage0.execute_calls + 1);
                CHECK(after.stage1.execute_calls == before.stage1.execute_calls + 1);
                CHECK(after.executor.drafted_tokens == before.executor.drafted_tokens + width);
                CHECK(after.executor.evaluated_tokens == before.executor.evaluated_tokens + width + 1);
                for (const auto token : result.published_tokens)
                    CHECK(oracle->decode_one().published_tokens == std::vector<std::int32_t>{token});
                CHECK(candidate->frontiers().target == oracle->frontiers().target);
                CHECK(candidate->frontiers().draft == candidate->frontiers().target);
            }
        }
    }
}

void test_width_one_speculative_pipeline() {
    q38::ExecutorOptions options;
    options.append_chunk_tokens = 4;
    auto executor = q38::make_mock_executor(options);
    executor->append({11, 12, 13, 14, 15, 16, 17});
    (void)executor->seed_decode();

    bool saw_accept = false;
    bool saw_reject = false;
    for (int step = 0; step < 128 && !(saw_accept && saw_reject); ++step) {
        const auto before = executor->metrics();
        const auto result = executor->speculative_step(1);
        const auto after = executor->metrics();
        const auto verified = result.published_tokens.size();
        CHECK(after.stage0.execute_calls == before.stage0.execute_calls + 2);
        CHECK(after.stage1.execute_calls ==
              before.stage1.execute_calls + verified);
        CHECK(after.executor.evaluated_tokens ==
              before.executor.evaluated_tokens + verified);
        CHECK(after.executor.state_committed_tokens ==
              before.executor.state_committed_tokens + verified);
        CHECK(result.published_tokens.size() == 1 ||
              result.published_tokens.size() == 2);
        saw_reject = saw_reject || result.published_tokens.size() == 1;
        saw_accept = saw_accept || result.published_tokens.size() == 2;
        CHECK(executor->frontiers().canonical -
                  executor->frontiers().target ==
              1);
        CHECK(executor->frontiers().stage0 == executor->frontiers().target);
        CHECK(executor->frontiers().stage1 == executor->frontiers().target);
        CHECK(executor->frontiers().draft == executor->frontiers().target);
    }
    CHECK(saw_accept);
    CHECK(saw_reject);
}

void test_width_n_speculative_pipeline() {
    q38::ExecutorOptions options;
    options.append_chunk_tokens = 4;
    auto executor = q38::make_mock_executor(options);
    executor->append({31, 32, 33, 34, 35, 36, 37});
    (void)executor->seed_decode();

    constexpr std::uint32_t width = 4;
    std::array<bool, width + 2> saw_published_extent{};
    const auto all_depths_seen = [&]() {
        return std::all_of(saw_published_extent.begin() + 1,
                           saw_published_extent.end(),
                           [](bool seen) { return seen; });
    };
    for (int step = 0; step < 4096 && !all_depths_seen(); ++step) {
        const auto before = executor->metrics();
        const auto result = executor->speculative_step(width);
        const auto after = executor->metrics();
        const auto verified = result.published_tokens.size();
        const auto stage0_rows =
            verified + (verified <= width ? 1u : 0u);
        CHECK(after.stage0.execute_calls ==
              before.stage0.execute_calls + stage0_rows);
        CHECK(after.stage1.execute_calls ==
              before.stage1.execute_calls + verified);
        CHECK(after.executor.evaluated_tokens ==
              before.executor.evaluated_tokens + verified);
        CHECK(after.executor.state_committed_tokens ==
              before.executor.state_committed_tokens + verified);
        CHECK(after.executor.drafted_tokens ==
              before.executor.drafted_tokens + width);
        CHECK((after.executor.accepted_draft_tokens -
                   before.executor.accepted_draft_tokens) +
                  (after.executor.rejected_draft_tokens -
                   before.executor.rejected_draft_tokens) ==
              width);
        CHECK(after.executor.maximum_draft_width == width);
        CHECK(result.published_tokens.size() >= 1);
        CHECK(result.published_tokens.size() <= width + 1);
        saw_published_extent[result.published_tokens.size()] = true;
        CHECK(executor->frontiers().canonical -
                  executor->frontiers().target ==
              1);
        CHECK(executor->frontiers().stage0 == executor->frontiers().target);
        CHECK(executor->frontiers().stage1 == executor->frontiers().target);
        CHECK(executor->frontiers().draft == executor->frontiers().target);
    }
    // Exercise rejection at every draft depth plus the all-accepted N+1 path.
    CHECK(all_depths_seen());
}

void test_retained_draft_cancellation_rolls_back_transaction() {
    q38::ExecutorOptions options;
    auto rolled_back = std::make_shared<std::atomic<std::uint32_t>>(0);
    auto stage0 = std::make_unique<q38::MockStageBackend>(
        q38::Stage::kStage0, options.hidden_width, options.vocab_size);
    auto stage1 = std::make_unique<CancelAfterRetainedDraftBackend>(
        std::make_unique<q38::MockStageBackend>(
            q38::Stage::kStage1, options.hidden_width, options.vocab_size),
        rolled_back);
    q38::DualStageExecutor executor(options, std::move(stage0),
                                    std::move(stage1));
    executor.append({1, 2, 3, 4});
    (void)executor.seed_decode();
    const auto before = executor.frontiers();
    auto cancellation = std::make_shared<q38::CancellationToken>();
    bool cancelled = false;
    try {
        (void)executor.speculative_step(4, cancellation);
    } catch (const q38::ExecutionCancelled&) {
        cancelled = true;
    }
    CHECK(cancelled);
    CHECK(rolled_back->load(std::memory_order_relaxed) == 1);
    CHECK(executor.frontiers().canonical == before.canonical);
    CHECK(executor.frontiers().target == before.target);
    const auto retry = executor.speculative_step(4);
    CHECK(!retry.published_tokens.empty());
}

void test_retained_draft_overlaps_first_stage0_row() {
    q38::ExecutorOptions options;
    auto state = std::make_shared<DraftStage0OverlapState>();
    auto stage0 = std::make_unique<OverlapStage0Backend>(
        std::make_unique<q38::MockStageBackend>(
            q38::Stage::kStage0, options.hidden_width, options.vocab_size),
        state);
    auto stage1 = std::make_unique<OverlapDraftBackend>(
        std::make_unique<q38::MockStageBackend>(
            q38::Stage::kStage1, options.hidden_width, options.vocab_size),
        state);
    q38::DualStageExecutor executor(options, std::move(stage0),
                                    std::move(stage1));
    executor.append({5, 6, 7, 8});
    (void)executor.seed_decode();
    state->armed.store(true, std::memory_order_release);
    const auto result = executor.speculative_step(1);
    CHECK(!result.published_tokens.empty());
    CHECK(state->stage0_entered.load(std::memory_order_acquire));
    CHECK(state->draft_entered.load(std::memory_order_acquire));
}

void test_speculative_stop_token_is_transactionally_truncated() {
    std::vector<std::int32_t> prompt;
    std::int32_t stop = -1;
    for (std::int32_t length = 1; length <= 16 && stop < 0; ++length) {
        prompt.assign(static_cast<std::size_t>(length), 17);
        auto baseline = q38::make_mock_executor({});
        baseline->append(prompt);
        (void)baseline->seed_decode();
        const auto baseline_step = baseline->speculative_step(5);
        if (baseline_step.published_tokens.size() > 1)
            stop = baseline_step.published_tokens.front();
    }
    CHECK(stop >= 0);

    q38::ExecutorOptions stop_options;
    stop_options.stop_token_ids = {stop};
    auto executor = q38::make_mock_executor(stop_options);
    executor->append(prompt);
    (void)executor->seed_decode();
    const auto before = executor->frontiers();
    const auto metrics_before = executor->metrics();
    const auto stopped = executor->speculative_step(5);
    const auto metrics_after = executor->metrics();
    CHECK(stopped.published_tokens.size() == 1);
    CHECK(stopped.published_tokens.front() == stop);
    CHECK(executor->frontiers().canonical == before.canonical + 1);
    CHECK(executor->frontiers().target == before.target + 1);
    CHECK(executor->frontiers().canonical - executor->frontiers().target == 1);
    CHECK(metrics_after.stage0.execute_calls ==
          metrics_before.stage0.execute_calls + 2);
    CHECK(metrics_after.stage1.execute_calls ==
          metrics_before.stage1.execute_calls + 1);
    CHECK(metrics_after.executor.evaluated_tokens ==
          metrics_before.executor.evaluated_tokens + 1);
}

void test_executor_context_limit() {
    q38::ExecutorOptions options;
    options.context_limit = 4;
    options.append_chunk_tokens = 2;

    {
        auto executor = q38::make_mock_executor(options);
        executor->append({1, 2, 3, 4});
        bool rejected = false;
        try {
            executor->append({5});
        } catch (const std::length_error&) {
            rejected = true;
        }
        CHECK(rejected);
        CHECK(executor->frontiers().canonical == 4);
        CHECK(executor->frontiers().target == 4);
        CHECK(executor->canonical_tokens().size() == 4);

        rejected = false;
        try {
            (void)executor->seed_decode();
        } catch (const std::length_error&) {
            rejected = true;
        }
        CHECK(rejected);
        CHECK(executor->frontiers().canonical == 4);
    }

    {
        auto executor = q38::make_mock_executor(options);
        executor->append({1, 2, 3});
        (void)executor->seed_decode();
        bool rejected = false;
        try {
            (void)executor->decode_one();
        } catch (const std::length_error&) {
            rejected = true;
        }
        CHECK(rejected);
        CHECK(executor->frontiers().canonical == 4);
        CHECK(executor->frontiers().target == 3);
    }

    {
        auto executor = q38::make_mock_executor(options);
        executor->append({1});
        (void)executor->seed_decode();
        (void)executor->speculative_step(64);
        CHECK(executor->frontiers().canonical <= options.context_limit);
        CHECK(executor->canonical_tokens().size() ==
              executor->frontiers().canonical);
    }

    {
        auto executor = q38::make_mock_executor(options);
        executor->append({1});
        (void)executor->seed_decode();
        bool rejected = false;
        try {
            (void)executor->speculative_step(65);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CHECK(rejected);
    }
}

void test_fixed_mtp4_full_context_contract() {
    q38::ExecutorOptions options;
    options.context_limit = 262144;
    options.append_chunk_tokens = 512;
    auto executor = q38::make_mock_executor(options);
    executor->append(std::vector<std::int32_t>(options.context_limit - 5, 1));
    (void)executor->seed_decode();
    while(executor->frontiers().canonical < options.context_limit) {
        const auto before = executor->frontiers().canonical;
        const auto step = executor->speculative_step(4);
        CHECK(!step.published_tokens.empty());
        CHECK(step.published_tokens.size() <= options.context_limit - before);
        CHECK(executor->frontiers().canonical <= options.context_limit);
    }
    const auto before = executor->canonical_tokens();
    bool rejected = false;
    try { executor->append({1}); } catch(const std::length_error&) { rejected = true; }
    CHECK(rejected);
    CHECK(executor->canonical_tokens() == before);
    rejected = false;
    try { (void)executor->speculative_step(4); } catch(const std::length_error&) { rejected = true; }
    CHECK(rejected);
    CHECK(executor->canonical_tokens() == before);
}

q38::ExecutorRpcRequestV1 rpc_request(
    q38::ExecutorRpcOpcodeV1 opcode, std::uint64_t request_id,
    std::uint64_t session_hash) {
    q38::ExecutorRpcRequestV1 request;
    request.header.request_id = request_id;
    request.header.session_hash = session_hash;
    request.header.opcode = opcode;
    return request;
}

void test_pending_terminal_replacement_rpc() {
    q38::ExecutorOptions options;
    options.vocab_size = 1024;
    auto probe = q38::make_mock_executor(options);
    probe->append({1,2,3});
    (void)probe->seed_decode();
    const auto terminal = probe->decode_one().published_tokens.back();
    options.stop_token_ids = {terminal};
    auto executor = q38::make_mock_executor(options);
    executor->append({1,2,3});
    (void)executor->seed_decode();
    (void)executor->decode_one();
    const auto before = executor->frontiers();
    q38::ExecutorRpcServiceV1 service(executor.get(), options.session_hash);
    auto request = rpc_request(q38::ExecutorRpcOpcodeV1::kReplacePendingTerminal, 1, options.session_hash);
    request.tokens = {terminal,7};
    request.header.token_count = 2;
    request.header.payload_bytes = 8;
    request.header.argument0 = static_cast<std::uint32_t>(before.canonical);
    request.header.argument1 = static_cast<std::uint32_t>(before.epoch);
    const auto first = service.handle(request);
    CHECK(first.header.status == q38::ExecutorRpcStatusV1::kOk);
    const auto epoch = executor->frontiers().epoch;
    CHECK(epoch == before.epoch+1);
    CHECK(service.handle(request).header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(executor->frontiers().epoch == epoch);
    request.tokens[1] = 8;
    CHECK(service.handle(request).header.status == q38::ExecutorRpcStatusV1::kBadRequest);
    request.header.request_id = 2;
    CHECK(service.handle(request).header.status == q38::ExecutorRpcStatusV1::kBadRequest);
    CHECK(executor->frontiers().epoch == epoch);
    CHECK(executor->canonical_tokens().back() == 7);
}

void test_candidate_verification_rpc() {
    q38::ExecutorOptions options;
    options.hidden_width = 8;
    auto executor = q38::make_mock_executor(options);
    executor->append({11, 12, 13}); (void)executor->seed_decode();
    q38::ExecutorRpcServiceV1 service(executor.get(), options.session_hash);
    auto request = rpc_request(q38::ExecutorRpcOpcodeV1::kVerifyCandidates, 1, options.session_hash);
    request.tokens = {17, 18, 19};
    request.header.token_count = 3;
    request.header.payload_bytes = 12;
    const auto first = service.handle(request);
    CHECK(first.header.status == q38::ExecutorRpcStatusV1::kOk);
    const auto frontier = executor->frontiers().canonical;
    CHECK(service.handle(request).tokens == first.tokens);
    CHECK(executor->frontiers().canonical == frontier);
    request.tokens[0] = 20;
    CHECK(service.handle(request).header.status == q38::ExecutorRpcStatusV1::kBadRequest);
    request.header.request_id = 2;
    request.header.argument0 = 1;
    CHECK(service.handle(request).header.status == q38::ExecutorRpcStatusV1::kBadRequest);
    CHECK(executor->frontiers().canonical == frontier);
}

void test_executor_rpc_service() {
    q38::ExecutorOptions options;
    options.session_hash = 0x170;
    options.context_limit = 8;
    auto executor = q38::make_mock_executor(options);
    q38::ExecutorRpcServiceV1 service(executor.get(), options.session_hash);

    auto append = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 1,
                              options.session_hash);
    append.tokens = {10, 11, 12};
    append.header.token_count = 3;
    append.header.payload_bytes = 3 * sizeof(std::int32_t);
    auto response = service.handle(append);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.header.request_id == 1);
    CHECK(response.header.frontiers.canonical == 3);
    CHECK(response.header.payload_bytes == 0);

    response = service.handle(append);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.header.frontiers.canonical == 3);
    CHECK(executor->canonical_tokens().size() == 3);

    auto conflicting_append = append;
    conflicting_append.tokens = {10, 11, 13};
    response = service.handle(conflicting_append);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kBadRequest);
    CHECK(response.message.find("already used") != std::string::npos);
    CHECK(response.header.frontiers.canonical == 3);

    auto seed = rpc_request(q38::ExecutorRpcOpcodeV1::kSeed, 2,
                            options.session_hash);
    response = service.handle(seed);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.tokens.size() == 1);
    CHECK(response.header.token_count == 1);
    CHECK(response.header.payload_bytes == sizeof(std::int32_t));

    auto decode = rpc_request(q38::ExecutorRpcOpcodeV1::kDecode, 3,
                              options.session_hash);
    decode.header.argument0 = 1;
    response = service.handle(decode);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.tokens.size() == 1);

    CHECK(response.header.frontiers.canonical == 5);
    const auto stats = executor->stats();
    CHECK(stats.transactions == 2);
    CHECK(stats.append_transactions == 1);
    CHECK(stats.decode_transactions == 1);
    CHECK(stats.evaluated_tokens == 4);
    CHECK(stats.state_committed_tokens == 4);
    CHECK(stats.published_tokens == 2);

    auto rpc_stats = rpc_request(q38::ExecutorRpcOpcodeV1::kStats, 31,
                                 options.session_hash);
    response = service.handle(rpc_stats);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.message.find("\"transactions\":2") !=
          std::string::npos);
    CHECK(response.header.message_bytes == response.message.size());

    auto wrong_session = rpc_request(q38::ExecutorRpcOpcodeV1::kState, 4,
                                     options.session_hash + 1);
    response = service.handle(wrong_session);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kBadRequest);
    CHECK(!response.message.empty());
    CHECK(response.header.message_bytes == response.message.size());
    CHECK(response.header.frontiers.canonical == 5);

    auto malformed = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 5,
                                 options.session_hash);
    malformed.tokens = {7};
    response = service.handle(malformed);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kBadRequest);
    CHECK(response.header.frontiers.canonical == 5);

    for (std::uint64_t request_id = 6; request_id <= 8; ++request_id) {
        decode.header.request_id = request_id;
        decode.header.argument0 = 1;
        response = service.handle(decode);
        CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
        CHECK(response.tokens.size() == 1);
    }
    decode.header.request_id = 9;
    response = service.handle(decode);
    CHECK(response.header.status ==
          q38::ExecutorRpcStatusV1::kCapacityExhausted);
    CHECK(response.header.frontiers.canonical == options.context_limit);
    CHECK(response.tokens.empty());

    auto reset = rpc_request(q38::ExecutorRpcOpcodeV1::kReset, 32,
                             options.session_hash);
    response = service.handle(reset);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.header.frontiers.canonical == 0);
    CHECK(response.header.frontiers.target == 0);
    CHECK(response.header.frontiers.stage0 == 0);
    CHECK(response.header.frontiers.stage1 == 0);
    CHECK(response.header.frontiers.draft == 0);
    CHECK(response.header.frontiers.epoch == 0);
    CHECK(executor->canonical_tokens().empty());
    CHECK(executor->sampler_state().canonical_tokens == 0);

    auto after_reset = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 33,
                                   options.session_hash);
    after_reset.tokens = {21, 22};
    after_reset.header.token_count = 2;
    after_reset.header.payload_bytes = 2 * sizeof(std::int32_t);
    response = service.handle(after_reset);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.header.frontiers.canonical == 2);
    CHECK(executor->canonical_tokens() == after_reset.tokens);

    auto stale = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 4,
                             options.session_hash);
    stale.tokens = {8};
    stale.header.token_count = 1;
    stale.header.payload_bytes = sizeof(std::int32_t);
    response = service.handle(stale);
    CHECK(response.header.status ==
          q38::ExecutorRpcStatusV1::kFailedPrecondition);
    CHECK(response.header.frontiers.canonical == 2);
}

void test_executor_rpc_logit_diagnostics() {
    q38::ExecutorOptions options;
    options.session_hash = 0x38;
    options.context_limit = 8;
    options.vocab_size = 16;
    options.retain_last_logits_for_diagnostics = true;
    auto executor = q38::make_mock_executor(options);
    q38::ExecutorRpcServiceV1 service(executor.get(), options.session_hash,
                                      {}, true);

    auto logits = rpc_request(q38::ExecutorRpcOpcodeV1::kLogits, 90,
                              options.session_hash);
    auto response = service.handle(logits);
    CHECK(response.header.status ==
          q38::ExecutorRpcStatusV1::kFailedPrecondition);
    CHECK(response.message.find("no committed") != std::string::npos);

    auto append = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 1,
                              options.session_hash);
    append.tokens = {2, 3};
    append.header.token_count = append.tokens.size();
    append.header.payload_bytes =
        append.tokens.size() * sizeof(std::int32_t);
    response = service.handle(append);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    const auto snapshot = executor->diagnostic_logit_snapshot();
    CHECK(snapshot.has_value());
    CHECK(snapshot->transaction_kind == q38::TxnKind::kAppendKnown);
    CHECK(snapshot->logits_bf16.size() == options.vocab_size);

    response = service.handle(logits);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(response.message.find("\"schema\":\"q38.logits.bf16.v1\"") !=
          std::string::npos);
    CHECK(response.message.find("\"element_count\":16") !=
          std::string::npos);
    CHECK(response.message.find("raw_bf16_le_hex") == std::string::npos);

    logits.header.argument0 = 1;
    response = service.handle(logits);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    const auto raw = response.message.find("\"raw_bf16_le_hex\":\"");
    CHECK(raw != std::string::npos);
    if (raw != std::string::npos) {
        const auto first = raw + std::string("\"raw_bf16_le_hex\":\"").size();
        const auto last = response.message.find('"', first);
        CHECK(last != std::string::npos);
        if (last != std::string::npos)
            CHECK(last - first == options.vocab_size * 4u);
    }

    auto invalid = logits;
    invalid.header.argument0 = 2;
    response = service.handle(invalid);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kBadRequest);

    auto seed = rpc_request(q38::ExecutorRpcOpcodeV1::kSeed, 2,
                            options.session_hash);
    response = service.handle(seed);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    auto decode = rpc_request(q38::ExecutorRpcOpcodeV1::kDecode, 3,
                              options.session_hash);
    decode.header.argument0 = 1;
    response = service.handle(decode);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(executor->diagnostic_logit_snapshot()->transaction_kind ==
          q38::TxnKind::kDecode);

    auto reset = rpc_request(q38::ExecutorRpcOpcodeV1::kReset, 4,
                             options.session_hash);
    response = service.handle(reset);
    CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    CHECK(!executor->diagnostic_logit_snapshot().has_value());

    q38::ExecutorRpcServiceV1 disabled(executor.get(), options.session_hash);
    logits.header.argument0 = 0;
    response = disabled.handle(logits);
    CHECK(response.header.status ==
          q38::ExecutorRpcStatusV1::kFailedPrecondition);
    CHECK(response.message.find("disabled") != std::string::npos);
}

void test_executor_rpc_cancel_and_deadline() {
    auto make_waiting_executor = [](q38::ExecutorOptions options,
                                    std::shared_ptr<std::atomic<bool>> entered) {
        auto stage0 = std::make_unique<q38::MockStageBackend>(
            q38::Stage::kStage0, options.hidden_width, options.vocab_size);
        auto stage1 = std::make_unique<WaitForCancellationBackend>(
            std::make_unique<q38::MockStageBackend>(
                q38::Stage::kStage1, options.hidden_width,
                options.vocab_size),
            std::move(entered));
        return std::make_unique<q38::DualStageExecutor>(
            options, std::move(stage0), std::move(stage1));
    };

    q38::ExecutorOptions options;
    options.session_hash = UINT64_C(0xca11ce1);
    options.append_chunk_tokens = 2;
    {
        auto entered = std::make_shared<std::atomic<bool>>(false);
        auto executor = make_waiting_executor(options, entered);
        q38::ExecutorRpcServiceV1 service(executor.get(), options.session_hash);
        auto append = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 100,
                                  options.session_hash);
        append.tokens = {1, 2, 3, 4, 5};
        append.header.token_count = 5;
        append.header.payload_bytes = 5 * sizeof(std::int32_t);
        auto writer = std::async(std::launch::async,
                                 [&service, append] {
                                     return service.handle(append);
                                 });
        const auto wait_limit = std::chrono::steady_clock::now() +
                                std::chrono::seconds(2);
        while (!entered->load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < wait_limit)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(entered->load(std::memory_order_acquire));

        const auto cancel = rpc_request(q38::ExecutorRpcOpcodeV1::kCancel,
                                        100, options.session_hash);
        const auto cancel_response = service.handle(cancel);
        CHECK(cancel_response.header.status ==
              q38::ExecutorRpcStatusV1::kOk);
        const auto writer_response = writer.get();
        CHECK(writer_response.header.status ==
              q38::ExecutorRpcStatusV1::kCancelled);
        CHECK(writer_response.header.frontiers.canonical == 0);
        CHECK(writer_response.header.frontiers.target == 0);
        CHECK(!executor->poisoned());
        CHECK(executor->canonical_tokens().empty());
        CHECK(executor->stats().cancellations == 1);
        CHECK(executor->stats().rollbacks == 1);
        CHECK(executor->sampler_state().sampled_tokens == 0);
        CHECK(executor->sampler_state().canonical_tokens == 0);

        const auto duplicate = service.handle(append);
        CHECK(duplicate.header.status ==
              q38::ExecutorRpcStatusV1::kCancelled);
        CHECK(executor->frontiers().canonical == 0);
    }

    {
        auto entered = std::make_shared<std::atomic<bool>>(false);
        auto executor = make_waiting_executor(options, entered);
        q38::ExecutorRpcServiceV1 service(executor.get(), options.session_hash);
        auto append = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 200,
                                  options.session_hash);
        append.tokens = {9, 8, 7};
        append.header.token_count = 3;
        append.header.payload_bytes = 3 * sizeof(std::int32_t);
        append.header.flags = q38::kExecutorRpcFlagDeadline;
        append.header.timeout_ms = 20;
        const auto response = service.handle(append);
        CHECK(entered->load(std::memory_order_acquire));
        CHECK(response.header.status ==
              q38::ExecutorRpcStatusV1::kDeadlineExceeded);
        CHECK(response.header.frontiers.canonical == 0);
        CHECK(response.header.frontiers.target == 0);
        CHECK(executor->stats().deadline_exceeded == 1);
        CHECK(executor->stats().rollbacks == 1);
    }
}

void test_fatal_cancellation_keeps_rpc_category() {
  for (bool recover : {false, true}) {
    for (bool deadline : {false, true}) {
        q38::ExecutorOptions options;
        options.enable_batched_mtp = true;
        options.backend_failure_is_fatal = true;
        options.backend_cancellation_is_fatal = true;
        options.backend_cancellation_resettable = recover;
        q38::DualStageExecutor executor(options,
            std::make_unique<q38::MockStageBackend>(q38::Stage::kStage0, options.hidden_width, options.vocab_size),
            std::make_unique<DraftFaultBackend>(std::make_unique<q38::MockStageBackend>(
                q38::Stage::kStage1, options.hidden_width, options.vocab_size), deadline ? 4 : 0));
        executor.append({1, 2, 3}); (void)executor.seed_decode();
        const auto before = executor.frontiers();
        q38::ExecutorRpcServiceV1 service(&executor, options.session_hash);
        auto request = rpc_request(q38::ExecutorRpcOpcodeV1::kSpeculative, 1, options.session_hash);
        request.header.argument0 = 1;
        request.header.argument1 = 4;
        if (deadline) {
            request.header.flags = q38::kExecutorRpcFlagDeadline;
            request.header.timeout_ms = 2;
        }
        auto response = service.handle(request);
        CHECK(response.header.status == (deadline ? q38::ExecutorRpcStatusV1::kDeadlineExceeded : q38::ExecutorRpcStatusV1::kCancelled));
        CHECK(executor.poisoned());
        CHECK(executor.frontiers().canonical == before.canonical);
        CHECK(executor.frontiers().target == before.target);
        response = service.handle(rpc_request(q38::ExecutorRpcOpcodeV1::kReset, 2, options.session_hash));
        CHECK(response.header.status == (recover ? q38::ExecutorRpcStatusV1::kOk : q38::ExecutorRpcStatusV1::kPoisoned));
        if (recover) {
            CHECK(!executor.poisoned());
            executor.append({1, 2, 3});
            (void)executor.seed_decode();
            executor.fail_closed();
            CHECK(!executor.cancellation_resettable());
        }
    }
  }
}

void test_batched_adaptive_executor() {
    q38::ExecutorOptions options;
    options.enable_batched_mtp = true;
    options.enable_adaptive_mtp = true;
    auto candidate = q38::make_mock_executor(options);
    options.enable_adaptive_mtp = false;
    auto oracle = q38::make_mock_executor(options);
    for (int reset = 0; reset < 2; ++reset) {
        candidate->reset_session(); oracle->reset_session();
        candidate->append({1, 2, 3}); oracle->append({1, 2, 3});
        CHECK(candidate->seed_decode() == oracle->seed_decode());
        for (int step = 0; step < 64; ++step) {
            const auto before = candidate->stats().drafted_tokens;
            const auto result = candidate->speculative_step(6);
            if (step == 0) CHECK(candidate->stats().drafted_tokens - before == 2);
            for (auto token : result.published_tokens)
                CHECK(oracle->decode_one().published_tokens == std::vector<std::int32_t>{token});
            CHECK(candidate->frontiers().target == oracle->frontiers().target);
        }
    }
}

void test_batched_mtp_width_policy() {
    q38::BatchedMtpWidthPolicy policy;
    CHECK(policy.choose(6) == 2);
    CHECK(policy.choose(1) <= 1);
    CHECK(policy.choose(0) == 0);
    const std::array<std::uint64_t, 7> cost{{50, 75, 90, 0, 150, 0, 180}};
    for (int workload = 0; workload < 3; ++workload) {
        int selected_best = 0;
        const auto best = workload == 0 ? 6u : workload == 1 ? 0u : 2u;
        for (int step = 0; step < 300; ++step) {
            const auto width = policy.choose(6);
            CHECK(width == 0 || width == 1 || width == 2 || width == 4 || width == 6);
            const auto published = workload == 0 ? width + 1 : workload == 1 ? 1u : std::min(width + 1, 3u);
            policy.observe(width, published, cost[width]);
            if (step >= 200 && width == best) ++selected_best;
        }
        CHECK(selected_best >= 80);
    }
    const auto observations = policy.observations();
    policy.observe(3, 1, 10);
    policy.observe(6, 8, 10);
    policy.observe(1, 0, 10);
    policy.observe(1, 1, 0);
    policy.observe(1, 1, 10, true);
    CHECK(policy.observations() == observations);
    for (std::uint32_t cap = 0; cap <= 64; ++cap) CHECK(policy.choose(cap) <= cap);
    policy.reset();
    CHECK(policy.observations() == 0);
    CHECK(policy.choose(6) == 2);
}

void test_batched_mtp_variable_acceptance_cost() {
    q38::BatchedMtpWidthPolicy policy;
    // Width 6 alternates one and seven publications at 70 time units per
    // transaction: its long-run cost is 17.5 per token, beating decode's 25.
    // Averaging per-transaction ratios instead gives (70 + 10) / 2 = 40,
    // incorrectly preferring ordinary decode for this bursty acceptance.
    for (int sample = 0; sample < 41; ++sample) {
        for (const auto width : {0u, 1u, 2u, 4u, 6u}) {
            const auto published = width == 6 ? (sample % 2 ? 7u : 1u) : 1u;
            const auto elapsed = width == 6 ? 70u : width == 0 ? 25u : 100u;
            policy.observe(width, published, elapsed);
        }
    }
    CHECK(policy.observations() == 205);
    CHECK(policy.choose(6) == 6);
}

void test_adaptive_mtp() {
    q38::AdaptiveMtpPolicy policy;
    CHECK(policy.choose(0) == 0);
    CHECK(policy.choose(1) == 1);
    CHECK(policy.choose(64) == 2);
    for (int i = 0; i < 65; ++i)
        policy.observe(4, 4, false, 10000000, 1000000, 4000000);
    CHECK(policy.choose(4) > 1);
    for (int i = 0; i < 129; ++i)
        policy.observe(4, 0, false, 10000000, 9000000, 1000000);
    CHECK(policy.choose(4) == 1);
    CHECK(policy.choose(1) == 1);
    const auto observations = policy.observations();
    policy.observe(65, 1, false, 1, 1, 1);
    CHECK(policy.observations() == observations);
    policy.reset();
    CHECK(policy.observations() == 0);

    q38::ExecutorOptions options;
    options.enable_adaptive_mtp = true;
    options.context_limit = 80;
    auto adaptive = q38::make_mock_executor(options);
    options.enable_adaptive_mtp = false;
    auto plain = q38::make_mock_executor(options);
    adaptive->append({1, 2, 3});
    plain->append({1, 2, 3});
    CHECK(adaptive->seed_decode() == plain->seed_decode());
    while (adaptive->frontiers().canonical < options.context_limit) {
        const auto result = adaptive->speculative_step(4);
        for (const auto token : result.published_tokens)
            CHECK(plain->decode_one().published_tokens.front() == token);
        CHECK(adaptive->canonical_tokens() == plain->canonical_tokens());
        CHECK(adaptive->frontiers().stage0 == adaptive->frontiers().target);
        CHECK(adaptive->frontiers().stage1 == adaptive->frontiers().target);
    }
    adaptive->reset_session();
    adaptive->append({4, 5});
    (void)adaptive->seed_decode();
    const auto before = adaptive->stats().drafted_tokens;
    (void)adaptive->speculative_step(4);
    CHECK(adaptive->stats().drafted_tokens - before == 2);
}

void test_compact_sampling() {
    auto bf16 = [](float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<std::uint16_t>(bits >> 16u);
    };
    for (const auto top_p : {0.4f, 0.95f, 1.0f}) {
        q38::SamplingConfigV1 config;
        config.mode = q38::SamplingModeV1::kTopKTopP;
        config.top_k = 3;
        config.top_p = top_p;
        config.temperature = 0.7f;
        q38::TransactionalSampler full(6, config), compact(6, config);
        const std::vector<std::uint16_t> logits{
            bf16(-1), bf16(2), bf16(2), bf16(0), bf16(3), bf16(-4)};
        const std::vector<std::uint16_t> selected{logits[4], logits[1], logits[2]};
        const std::vector<std::int32_t> ids{4, 1, 2};
        for (int step = 0; step < 128; ++step) {
            const auto a = full.prepare(logits, {1, 1, 4});
            const auto b = compact.prepare_compact(selected, ids);
            CHECK(a.token == b.token);
            CHECK(a.rng_before == b.rng_before && a.rng_after == b.rng_after);
            const auto retry = compact.prepare_compact(selected, ids);
            CHECK(retry.token == b.token && retry.rng_before == b.rng_before);
            full.commit(a);
            compact.commit(b);
            full.commit_token(a.token);
            compact.commit_token(b.token);
        }
        bool rejected = false;
        try { (void)compact.prepare_compact(selected, {4, 1, 1}); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
        config.presence_penalty = 0.1f;
        q38::TransactionalSampler penalized(6, config);
        CHECK(!penalized.can_sample_compact());
        rejected = false;
        try { (void)penalized.prepare_compact(selected, ids); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    }
}

void test_transactional_sampling() {
    const auto bf16 = [](float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<std::uint16_t>(
            (bits + UINT32_C(0x7fff) + ((bits >> 16u) & 1u)) >> 16u);
    };
    q38::SamplingConfigV1 config;
    config.mode = q38::SamplingModeV1::kTopKTopP;
    config.top_k = 3;
    config.top_p = 0.9f;
    config.temperature = 0.8f;
    config.repetition_penalty = 1.1f;
    config.frequency_penalty = 0.2f;
    config.presence_penalty = 0.1f;
    config.seed = 170;
    q38::TransactionalSampler sampler(4, config);
    const std::vector<std::uint16_t> logits{
        bf16(0.0f), bf16(1.0f), bf16(2.0f), bf16(3.0f)};
    const auto first = sampler.prepare(logits, {3, 3});
    const auto retry = sampler.prepare(logits, {3, 3});
    CHECK(first.token == retry.token);
    CHECK(first.rng_before == retry.rng_before);
    CHECK(first.rng_after == retry.rng_after);
    CHECK(sampler.state().sampled_tokens == 0);
    sampler.commit(first);
    CHECK(sampler.state().sampled_tokens == 1);
    sampler.commit_tokens({1, 2, 2});
    CHECK(sampler.state().canonical_tokens == 3);
    CHECK(sampler.state().penalty_checksum != 0);

    q38::ExecutorOptions options;
    options.vocab_size = 100;
    options.context_limit = 32;
    options.append_chunk_tokens = 2;
    options.sampling = config;
    auto executor = q38::make_mock_executor(options);
    executor->append({1, 2, 3, 4, 5});
    CHECK(executor->sampler_state().sampled_tokens == 1);
    CHECK(executor->sampler_state().canonical_tokens == 5);
    (void)executor->seed_decode();
    CHECK(executor->sampler_state().canonical_tokens == 6);
    (void)executor->decode_one();
    CHECK(executor->sampler_state().sampled_tokens == 2);
    CHECK(executor->sampler_state().canonical_tokens == 7);
    (void)executor->speculative_step(4);
    CHECK(executor->sampler_state().sampled_tokens == 3);
    CHECK(executor->stats().speculative_transactions == 0);
    CHECK(executor->stats().drafted_tokens == 0);
    const auto sampling_before = executor->sampler_state().sampled_tokens;
    const auto target_only = executor->verify_candidates({1, 2, 3, 4});
    CHECK(target_only.published_tokens.size() == 1);
    CHECK(executor->sampler_state().sampled_tokens == sampling_before + 1);
    CHECK(executor->stats().speculative_transactions == 0);
    CHECK(executor->stats().drafted_tokens == 0);
}

void test_session_identity_contract() {
    q38::SessionIdentityV1 identity;
    identity.session_hash = UINT64_C(0x170);
    identity.context_limit = 262144;
    const auto digest = q38::parse_digest256(std::string(64, 'a'));
    identity.model_checkpoint = digest;
    identity.tokenizer = digest;
    identity.chat_template = digest;
    identity.runtime = digest;
    identity.kernels = digest;
    identity.ple_layout = digest;
    identity.stage_plan = digest;
    identity.sampling_parser = digest;
    identity.identity_checksum = q38::session_identity_checksum(identity);
    std::string error;
    CHECK(q38::validate_session_identity(identity, &error));
    CHECK(q38::format_digest256(identity.model_checkpoint) ==
          std::string(64, 'a'));

    const auto path = std::string("/tmp/q38-identity-") +
                      std::to_string(getpid()) + ".q38i";
    {
        std::ofstream output(path);
        output << "Q38_SESSION_IDENTITY_V1\n"
               << "session_hash=" << identity.session_hash << '\n'
               << "model_checkpoint_sha256=" << std::string(64, 'a') << '\n'
               << "tokenizer_sha256=" << std::string(64, 'a') << '\n'
               << "chat_template_sha256=" << std::string(64, 'a') << '\n'
               << "runtime_sha256=" << std::string(64, 'a') << '\n'
               << "kernels_sha256=" << std::string(64, 'a') << '\n'
               << "ple_layout_sha256=" << std::string(64, 'a') << '\n'
               << "stage_plan_sha256=" << std::string(64, 'a') << '\n'
               << "sampling_parser_sha256=" << std::string(64, 'a') << '\n'
               << "context_limit=" << identity.context_limit << '\n'
               << "flags=0\nidentity_checksum="
               << identity.identity_checksum << '\n';
    }
    const auto loaded = q38::load_session_identity(path);
    CHECK(loaded.identity_checksum == identity.identity_checksum);
    CHECK(loaded.stage_plan == digest);
    std::remove(path.c_str());

    identity.context_limit = 1;
    CHECK(!q38::validate_session_identity(identity, &error));
}

void test_append_only_snapshot_journal() {
    q38::ExecutorOptions options;
    options.session_hash = UINT64_C(0x51a9);
    options.context_limit = 32;
    options.append_chunk_tokens = 2;
    auto executor = q38::make_mock_executor(options);
    const auto path = std::string("/tmp/q38-snapshot-") +
                      std::to_string(getpid()) + ".q38s";
    std::remove(path.c_str());
    {
        q38::SnapshotJournal journal(path, executor->identity());
        executor->append({1, 2, 3, 4, 5});
        journal.append(executor->snapshot());
        (void)executor->seed_decode();
        journal.append(executor->snapshot());
        (void)executor->decode_one();
        journal.append(executor->snapshot());
        const auto latest = journal.latest();
        CHECK(latest.header.sequence == 3);
        CHECK(latest.canonical_tokens == executor->canonical_tokens());
        CHECK(latest.header.frontiers.canonical == 7);
        std::string error;
        CHECK(q38::validate_state_snapshot(latest, executor->identity(),
                                           &error));
    }
    const auto compact_bytes =
        3 * sizeof(q38::StateSnapshotHeaderV1) + (5 + 1 + 1) * sizeof(std::int32_t);
    CHECK(std::filesystem::file_size(path) == compact_bytes);
    {
        std::ofstream trailing(path, std::ios::binary | std::ios::app);
        trailing.write("partial", 7);
    }
    {
        q38::SnapshotJournal recovered(path, executor->identity());
        const auto latest = recovered.latest();
        CHECK(latest.header.sequence == 3);
        CHECK(latest.canonical_tokens == executor->canonical_tokens());
    }
    std::remove(path.c_str());

    const auto service_path = path + ".service";
    std::remove(service_path.c_str());
    auto service_executor = q38::make_mock_executor(options);
    {
        q38::ExecutorRpcServiceV1 service(service_executor.get(),
                                          options.session_hash, service_path);
        auto append = rpc_request(q38::ExecutorRpcOpcodeV1::kAppend, 1,
                                  options.session_hash);
        append.tokens = {7, 8, 9};
        append.header.token_count = 3;
        append.header.payload_bytes = 3 * sizeof(std::int32_t);
        const auto response = service.handle(append);
        CHECK(response.header.status == q38::ExecutorRpcStatusV1::kOk);
    }
    {
        q38::SnapshotJournal recovered(service_path,
                                       service_executor->identity());
        CHECK(recovered.latest().canonical_tokens ==
              service_executor->canonical_tokens());
    }
    std::remove(service_path.c_str());
}

void test_metrics_schema() {
    q38::ExecutorOptions options;
    options.append_chunk_tokens = 2;
    auto executor = q38::make_mock_executor(options);
    executor->append({1, 2, 3, 4, 5});
    (void)executor->seed_decode();
    (void)executor->decode_one();
    const auto metrics = executor->metrics();
    CHECK(metrics.magic == q38::kMetricsMagic);
    CHECK(metrics.schema_hash == q38::kMetricsSchemaHashV1);
    CHECK(metrics.identity_checksum == executor->identity().identity_checksum);
    CHECK(metrics.executor.transactions == 2);
    CHECK(metrics.executor.published_tokens == 2);
    CHECK(metrics.stage0.execute_calls == 4);
    CHECK(metrics.stage1.execute_calls == 4);
    CHECK(metrics.stage0.execute_tokens == 6);
    CHECK(metrics.stage1.execute_tokens == 6);
    CHECK(metrics.transaction_latency.count == 2);
    CHECK(metrics.transaction_latency.p50_ns > 0);
    CHECK(metrics.stage0_latency.count == 4);
    CHECK(metrics.stage1_latency.count == 4);
    const auto json = q38::metrics_json(metrics);
    CHECK(json.find("\"schema\":\"q38.metrics.v1\"") !=
          std::string::npos);
    CHECK(json.find("\"p99_ns\":") != std::string::npos);
    CHECK(json.find("\"stage0\":{") != std::string::npos);
    CHECK(json.find("\"cuda_tracked_allocated_bytes\":") !=
          std::string::npos);
    CHECK(json.find("\"runtime_pinned_bytes\":") !=
          std::string::npos);
    CHECK(json.find("\"accepted_draft_tokens\":") !=
          std::string::npos);
    CHECK(json.find("\"maximum_draft_width\":") !=
          std::string::npos);
    CHECK(json.find("\"cuda_graph_captures\":") !=
          std::string::npos);
    CHECK(json.find("\"cuda_graph_replays\":") !=
          std::string::npos);
    CHECK(json.find("\"cuda_graph_fallbacks\":") !=
          std::string::npos);
    CHECK(json.find("\"cuda_graph_nodes\":") !=
          std::string::npos);
#ifdef __linux__
    CHECK(metrics.host.process_rss_bytes > 0);
    CHECK(metrics.host.system_mem_available_bytes > 0);
#endif
}

void test_stage_worker_startup_lifecycle() {
    q38::ExecutorOptions options;
    for (std::uint32_t attempt = 0; attempt < 128; ++attempt) {
        options.session_hash = attempt + 1;
        auto executor = q38::make_mock_executor(options);
        const auto metrics = executor->metrics();
        CHECK(metrics.magic == q38::kMetricsMagic);
    }
}

void test_artifact_manifest() {
    std::istringstream input(R"MANIFEST(
schema_version=1
model_id=Qwen3.8-Flash-Next
artifact_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
tokenizer_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
tensor_layout_version=q38-layout-v1
stage_plan_id=test-2-2
context_limit=262144
vocab_size=248320
model_hidden_width=2560
boundary_hidden_width=10240
layer_count=4
stage0_first=0
stage0_count=2
stage1_first=2
stage1_count=2
ple_layer=1
qsa_layers=1,3
qsa_main_dtype=bf16
ple_storage_dtype=fp8_e4m3
stage0_weights=s0
stage1_weights=s1
mtp_weights=mtp
ple_manifest=ple.json
)MANIFEST");
    const auto manifest = q38::parse_artifact_manifest(input);
    CHECK(manifest.stage0.count == 2);
    CHECK(manifest.stage1.first == 2);
    CHECK(manifest.qsa_layers.size() == 2);
    CHECK(manifest.boundary_hidden_width == 10240);
}

void test_tensor_index() {
    const auto base = std::string("/tmp/q38-index-") + std::to_string(getpid());
    const auto shard = base + ".bin";
    const auto index0_path = base + "-s0.q38i";
    const auto index1_path = base + "-s1.q38i";
    {
        std::ofstream output(shard, std::ios::binary);
        std::string bytes(128, '\0');
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<char>(i);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    const std::string hash(64, 'a');
    {
        std::ofstream output(index0_path);
        output << "Q38_TENSOR_INDEX_V1\nstage=0\ncut=1\nlayout_sha256="
               << hash << "\ntensor\ttoken_embd.weight\tBF16\t16\t8\t"
               << shard << "\t4,2\n";
    }
    {
        std::ofstream output(index1_path);
        output << "Q38_TENSOR_INDEX_V1\nstage=1\ncut=1\nlayout_sha256="
               << hash << "\ntensor\toutput.weight\tBF16\t16\t64\t"
               << shard << "\t4,2\n";
    }
    const auto stage0 = q38::load_tensor_index(index0_path);
    const auto stage1 = q38::load_tensor_index(index1_path);
    q38::validate_tensor_index_pair(stage0, stage1);
    CHECK(stage0.payload_bytes == 16);
    CHECK(stage1.tensors.at(0).absolute_offset == 64);
    {
        q38::MappedWeightStore weights(stage0);
        const auto embedding = weights.require("token_embd.weight");
        CHECK(embedding.size() == 16);
        CHECK(std::to_integer<unsigned>(embedding.data[0]) == 8);
        CHECK(std::to_integer<unsigned>(embedding.data[15]) == 23);
        CHECK(weights.find("missing").empty());
        CHECK(weights.shard_paths().size() == 1);
    }
    std::remove(index0_path.c_str());
    std::remove(index1_path.c_str());
    std::remove(shard.c_str());
}

void test_device_artifact_index_and_mapping() {
    const auto root = std::filesystem::path("/tmp") /
                      ("q38-device-" + std::to_string(getpid()));
    const auto stage0_root = root / "stage0";
    const auto stage1_root = root / "stage1";
    std::filesystem::create_directories(stage0_root / "segments");
    std::filesystem::create_directories(stage1_root / "segments");
    const auto segment0 = stage0_root / "segments" / "a.q38w";
    const auto segment1 = stage1_root / "segments" / "b.q38w";
    {
        std::ofstream output(segment0, std::ios::binary);
        for (int i = 0; i < 256; ++i) output.put(static_cast<char>(i));
    }
    {
        std::ofstream output(segment1, std::ios::binary);
        for (int i = 0; i < 256; ++i)
            output.put(static_cast<char>(255 - i));
    }
    const std::string commit(40, 'c');
    const std::string hash(64, 'a');
    const auto write_header = [&](std::ofstream* output, int stage) {
        *output << "Q38_DEVICE_INDEX_V1\n"
                << "stage=" << stage << "\ncut=25\n"
                << "source_repo=Qwen/Qwen3.8-Flash-Next\n"
                << "source_commit=" << commit << "\n"
                << "policy_sha256=" << hash << "\n"
                << "artifact_sha256=" << hash << "\n";
    };
    const auto index0_path = stage0_root / "index.q38d";
    const auto index1_path = stage1_root / "index.q38d";
    {
        std::ofstream output(index0_path);
        write_header(&output, 0);
        output << "segment\tsegments/a.q38w\t256\t" << hash << '\n'
               << "tensor\ttoken_embd.weight\tsource.embed\tBF16\tpreserve\t0\t0\t8\t16\t"
               << hash << "\t0\t0\t-\t4,2\n";
    }
    {
        std::ofstream output(index1_path);
        write_header(&output, 1);
        output << "segment\tsegments/b.q38w\t256\t" << hash << '\n'
               << "tensor\ttoken_embd.weight\tsource.embed\tBF16\tpreserve\t0\t0\t160\t16\t"
               << hash << "\t0\t0\t-\t4,2\n"
               << "tensor\toutput.weight\tsource.output\tBF16\tw4a16_sym_g128\t128\t0\t0\t128\t"
               << hash << "\t128\t4\t" << hash << "\t2,128\n";
    }
    const auto index0 = q38::load_device_stage_index(index0_path.string());
    const auto index1 = q38::load_device_stage_index(index1_path.string());
    q38::validate_device_stage_pair(index0, index1);
    {
        auto mismatched = index1;
        mismatched.tensors.at(0).data_sha256 = std::string(64, 'b');
        bool rejected = false;
        try {
            q38::validate_device_stage_pair(index0, mismatched);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        CHECK(rejected);
    }
    {
        auto unexpected = index0;
        unexpected.tensors.push_back(index1.tensors.at(1));
        bool rejected = false;
        try {
            q38::validate_device_stage_pair(unexpected, index1);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        CHECK(rejected);
    }
    CHECK(index0.tensors.size() == 1);
    CHECK(index1.tensors.at(1).scale_bytes == 4);
    {
        q38::DeviceWeightStore store(index0);
        const auto segment = store.segment(0);
        CHECK(!segment.empty());
        CHECK(segment.descriptor->bytes == 256);
        CHECK(std::to_integer<unsigned>(segment.data[8]) == 8);
        const auto view = store.require("token_embd.weight");
        CHECK(std::to_integer<unsigned>(view.data[0]) == 8);
        CHECK(std::to_integer<unsigned>(view.data[15]) == 23);
        CHECK(view.scales == nullptr);
    }
    {
        q38::DeviceWeightStore store(index1);
        const auto view = store.require("output.weight");
        CHECK(std::to_integer<unsigned>(view.data[0]) == 255);
        CHECK(std::to_integer<unsigned>(view.scales[0]) == 127);
    }
    std::filesystem::remove_all(root);
}

void test_execute_failure_rolls_back() {
    q38::ExecutorOptions options;
    options.append_chunk_tokens = 2;
    auto stage0 = std::make_unique<q38::MockStageBackend>(
        q38::Stage::kStage0, options.hidden_width, options.vocab_size);
    auto stage1 = std::make_unique<FailOnceBackend>(
        std::make_unique<q38::MockStageBackend>(
            q38::Stage::kStage1, options.hidden_width, options.vocab_size),
        FailOnceBackend::Point::kExecute, 2);
    q38::DualStageExecutor executor(options, std::move(stage0), std::move(stage1));
    bool failed = false;
    try {
        executor.append({1, 2, 3, 4, 5});
    } catch (const std::exception&) {
        failed = true;
    }
    CHECK(failed);
    CHECK(!executor.poisoned());
    CHECK(executor.frontiers().target == 0);
    CHECK(executor.frontiers().canonical == 0);
    CHECK(executor.canonical_tokens().empty());
    executor.append({1, 2, 3, 4, 5});
    CHECK(executor.frontiers().target == 5);
    CHECK(executor.frontiers().canonical == 5);
    CHECK(executor.canonical_tokens().size() == 5);
    CHECK(executor.stats().append_transactions == 2);
    CHECK(executor.stats().rollbacks == 1);

    q38::ExecutorOptions fatal_options;
    fatal_options.backend_failure_is_fatal = true;
    auto fatal_stage0 = std::make_unique<q38::MockStageBackend>(
        q38::Stage::kStage0, fatal_options.hidden_width,
        fatal_options.vocab_size);
    auto fatal_stage1 = std::make_unique<FailOnceBackend>(
        std::make_unique<q38::MockStageBackend>(
            q38::Stage::kStage1, fatal_options.hidden_width,
            fatal_options.vocab_size),
        FailOnceBackend::Point::kExecute);
    q38::DualStageExecutor fatal_executor(
        fatal_options, std::move(fatal_stage0), std::move(fatal_stage1));
    bool fatal_failed = false;
    try {
        fatal_executor.append({1});
    } catch (const std::exception&) {
        fatal_failed = true;
    }
    CHECK(fatal_failed);
    CHECK(fatal_executor.poisoned());
    CHECK(fatal_executor.frontiers().canonical == 0);
    CHECK(fatal_executor.frontiers().target == 0);
}

void test_commit_failure_poisoned() {
    q38::ExecutorOptions options;
    auto stage0 = std::make_unique<q38::MockStageBackend>(
        q38::Stage::kStage0, options.hidden_width, options.vocab_size);
    auto stage1 = std::make_unique<FailOnceBackend>(
        std::make_unique<q38::MockStageBackend>(
            q38::Stage::kStage1, options.hidden_width, options.vocab_size),
        FailOnceBackend::Point::kCommit);
    q38::DualStageExecutor executor(options, std::move(stage0), std::move(stage1));
    bool failed = false;
    try {
        executor.append({1});
    } catch (const std::exception&) {
        failed = true;
    }
    CHECK(failed);
    CHECK(executor.poisoned());
}

void test_ple_layout_hash_and_store() {
    const auto base = std::string("/tmp/q38-ple-") + std::to_string(getpid());
    const auto data_path = base + ".bin";
    const auto layout_path = base + ".q38p";
    {
        std::ofstream output(data_path, std::ios::binary);
        for (std::uint8_t row = 0; row < 16; ++row) {
            output.put(static_cast<char>(row));
            output.put(static_cast<char>(255u - row));
        }
    }
    {
        std::ofstream output(layout_path);
        output << "Q38_PLE_LAYOUT_V1\n"
               << "dtype=bf16\nalignment=2\nrow_stride=2\nrow_dimension=1\n"
               << "usable_rows=16\npadded_rows=16\nunigram_vocab=32\n"
               << "eos_token=31\nmultipliers=3,5,7\n"
               << "head_vocab_sizes=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
               << "head_offsets=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
               << "file\t0\t" << data_path << "\t32\t32\n"
               << "part\t0\t0\t0\t16\t0\t32\n";
    }
    const auto layout = q38::load_ple_layout(layout_path);
    q38::PleHashState hash(layout.hash);
    {
        auto config = layout.hash;
        for (std::uint32_t head = 0; head < q38::kPleHeads; ++head)
            config.head_vocab_sizes[head] = 97 + 2 * head;
        q38::PleHashState batched(config), appended(config);
        for (const auto& tokens : {std::vector<std::int32_t>{1, 2, 31, 4, 5},
                                   std::vector<std::int32_t>{31, 31, 7, 8, 9}}) {
            std::vector<std::uint64_t> planned;
            planned.reserve(tokens.size() * q38::kPleHeads);
            for (const auto token : tokens) appended.append_rows(token, planned);
            CHECK(planned == batched.rows(tokens));
        }
        std::vector<std::uint64_t> unchanged{17};
        bool rejected = false;
        try { appended.append_rows(-1, unchanged); }
        catch (const std::runtime_error&) { rejected = true; }
        CHECK(rejected && unchanged == std::vector<std::uint64_t>{17});
        CHECK(appended.rows({10}) == batched.rows({10}));
    }
    const auto rows = hash.rows({1, 2});
    CHECK(rows.size() == 32);
    for (std::uint32_t head = 0; head < q38::kPleHeads; ++head) {
        CHECK(rows[head] == head);
        CHECK(rows[q38::kPleHeads + head] == head);
    }
    q38::PleStore store(layout, 8, 8);
    const auto row = store.read_row(5);
    CHECK(row.size() == 2);
    CHECK(row[0] == 5);
    CHECK(row[1] == 250);
    const auto row_again = store.read_row(5);
    CHECK(row_again == row);
    const auto packed = store.read_rows({0, 15});
    CHECK(packed.size() == 4);
    CHECK(packed[0] == 0);
    CHECK(packed[2] == 15);
    std::array<std::uint8_t, 4> direct{};
    store.read_rows_into({7, 3}, direct.data(), direct.size());
    CHECK(direct[0] == 7);
    CHECK(direct[1] == 248);
    CHECK(direct[2] == 3);
    CHECK(direct[3] == 252);
    bool bad_extent = false;
    try {
        store.read_rows_into({1}, direct.data(), direct.size());
    } catch (const std::invalid_argument&) {
        bad_extent = true;
    }
    CHECK(bad_extent);
    const auto stats = store.cache_stats();
    CHECK(stats.hits >= 1);
    CHECK(stats.misses >= 2);
    CHECK(stats.evictions >= 1);
    CHECK(stats.capacity_bytes == 8);
    CHECK(stats.resident_bytes == 8);
    CHECK(stats.requested_rows == 4);
    CHECK(stats.unique_page_requests == 6);
    CHECK(stats.useful_bytes == 8);
    CHECK(stats.scale_resident_bytes == 0);
    {
        q38::PleStoreOptionsV1 options;
        options.io_mode = q38::PleIoModeV1::kRam;
        options.ram_budget_bytes = 31;
        bool rejected = false;
        try { q38::PleStore too_small(layout, options); }
        catch (const std::runtime_error&) { rejected = true; }
        CHECK(rejected);
        options.ram_budget_bytes = 32;
        options.cache_bytes = 1ull << 30u; // RAM must not also allocate an LRU.
        q38::PleStore memory(layout, options);
        const std::vector<std::uint64_t> rows{15, 0, 7, 7, 1};
        const auto expected = store.read_rows(rows);
        CHECK(memory.read_rows(rows) == expected);
        const auto before = memory.cache_stats();
        CHECK(before.ram_enabled == 1);
        CHECK(before.ram_resident_bytes == 32);
        CHECK(before.resident_bytes == 0 && before.capacity_bytes == 0);
        CHECK(before.io_uring_enabled == 0 && before.direct_io_enabled == 0);
        // A file-backed mapping or filesystem fallback would now fail/change.
        { std::ofstream truncate(data_path, std::ios::binary | std::ios::trunc); }
        CHECK(memory.read_rows(rows) == expected);
        CHECK(memory.read_row(5) == row);
        CHECK(memory.read_rows({}).empty());
        CHECK(memory.cache_stats().physical_read_bytes == before.physical_read_bytes);
        CHECK(memory.cache_stats().read_operations == before.read_operations);
        std::array<std::uint8_t,4> untouched{91,92,93,94};
        bool invalid = false;
        try { memory.read_rows_into({0,16},untouched.data(),untouched.size()); }
        catch (const std::runtime_error&) { invalid = true; }
        CHECK(invalid && untouched[0] == 91 && untouched[3] == 94);
    }
    std::remove(layout_path.c_str());
    std::remove(data_path.c_str());

    const auto scaled_data_path = base + "-scaled.bin";
    const auto scaled_layout_path = base + "-scaled.q38p";
    {
        std::ofstream output(scaled_data_path, std::ios::binary);
        for (std::uint8_t row = 0; row < 16; ++row)
            output.put(static_cast<char>(row));
        for (std::uint16_t row = 0; row < 16; ++row) {
            const std::uint16_t scale = static_cast<std::uint16_t>(0x3f80 + row);
            output.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
        }
    }
    {
        std::ofstream output(scaled_layout_path);
        output << "Q38_PLE_LAYOUT_V1\n"
               << "dtype=fp8_e4m3\nalignment=1\nrow_stride=1\n"
               << "row_dimension=1\nusable_rows=16\npadded_rows=16\n"
               << "unigram_vocab=32\neos_token=31\nmultipliers=3,5,7\n"
               << "head_vocab_sizes=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
               << "head_offsets=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
               << "file\t0\t" << scaled_data_path << "\t48\t48\n"
               << "part\t0\t0\t0\t16\t0\t16\t0\t16\t32\n";
    }
    const auto scaled_layout = q38::load_ple_layout(scaled_layout_path);
    q38::PleStore scaled_store(scaled_layout);
    std::array<std::uint16_t, 2> row_scales{};
    scaled_store.read_row_scales_into({0, 15}, row_scales.data(),
                                      row_scales.size());
    CHECK(row_scales[0] == 0x3f80);
    CHECK(row_scales[1] == 0x3f8f);
    const auto scaled_stats = scaled_store.cache_stats();
    CHECK(scaled_stats.scale_resident_bytes == 32);
    {
        auto shared_store = std::make_shared<q38::PleStore>(
            q38::load_ple_layout(scaled_layout_path), 16, 16);
        q38::PleAsyncReader reader(shared_store, 5);
        auto requested = std::make_shared<std::vector<std::uint64_t>>();
        for (std::uint64_t i = 0; i < 37; ++i) requested->push_back(i % 16);
        for (int repeat = 0; repeat < 3; ++repeat) {
            reader.submit(requested);
            bool out_of_order = false;
            std::uint8_t invalid_value = 0;
            std::uint16_t invalid_scale = 0;
            try { reader.take(1, 1, &invalid_value, 1, &invalid_scale, 1); }
            catch (const std::invalid_argument&) { out_of_order = true; }
            CHECK(out_of_order);
            std::size_t first = 0;
            for (const std::size_t count : {1u, 7u, 2u, 17u, 10u}) {
                std::vector<std::uint8_t> values(count);
                std::vector<std::uint16_t> scales(count);
                reader.take(first, count, values.data(), count, scales.data(), count);
                for (std::size_t i = 0; i < count; ++i) {
                    CHECK(values[i] == (first + i) % 16);
                    CHECK(scales[i] == 0x3f80 + (first + i) % 16);
                }
                first += count;
            }
            reader.submit(requested);
            reader.cancel(); // producer may be blocked on its two-tile bound
            auto invalid_rows = std::make_shared<std::vector<std::uint64_t>>(1, 9999);
            reader.submit(invalid_rows);
            bool read_failed = false;
            try { reader.take(0, 1, &invalid_value, 1, &invalid_scale, 1); }
            catch (const std::exception&) { read_failed = true; }
            CHECK(read_failed);
            reader.cancel();
        }
    }
    {
        q38::PleStoreOptionsV1 options;
        options.io_mode = q38::PleIoModeV1::kRam;
        options.ram_budget_bytes = 48;
        auto memory = std::make_shared<q38::PleStore>(scaled_layout, options);
        const auto expected = scaled_store.read_rows({0,15,7});
        { std::ofstream truncate(scaled_data_path, std::ios::binary | std::ios::trunc); }
        std::thread readers[2];
        std::exception_ptr errors[2];
        for(int r=0;r<2;++r) readers[r]=std::thread([&,r] {
            try {
                for(int i=0;i<20;++i) {
                    CHECK(memory->read_rows({0,15,7}) == expected);
                    std::uint16_t scales[3]{};
                    memory->read_row_scales_into({0,15,7},scales,3);
                    CHECK(scales[0]==0x3f80 && scales[1]==0x3f8f && scales[2]==0x3f87);
                }
            } catch (...) { errors[r]=std::current_exception(); }
        });
        for(auto& reader:readers) reader.join();
        for(auto error:errors) if(error) std::rethrow_exception(error);
        CHECK(memory->cache_stats().ram_resident_bytes == 16);
        CHECK(memory->cache_stats().scale_resident_bytes == 32);
        CHECK(memory->cache_stats().physical_read_bytes == 48);
        CHECK(memory->cache_stats().requested_rows == 120);
        q38::PleAsyncReader reader(memory, 2);
        auto rows = std::make_shared<std::vector<std::uint64_t>>(std::initializer_list<std::uint64_t>{0,15,7});
        reader.submit(rows);
        std::uint8_t values[3]{}; std::uint16_t scales[3]{};
        reader.take(0,3,values,3,scales,3);
        CHECK(values[0]==0 && values[1]==15 && values[2]==7);
        CHECK(scales[2]==0x3f87);
    }
    std::remove(scaled_layout_path.c_str());
    std::remove(scaled_data_path.c_str());

#ifdef __linux__
    const auto direct_data_path = base + "-direct.bin";
    const auto direct_layout_path = base + "-direct.q38p";
    {
        std::vector<std::uint8_t> data(16u * 4096u, 0);
        for (std::size_t row_index = 0; row_index < 16; ++row_index)
            data[row_index * 4096] = static_cast<std::uint8_t>(row_index);
        std::ofstream output(direct_data_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    {
        std::ofstream output(direct_layout_path);
        output << "Q38_PLE_LAYOUT_V1\n"
               << "dtype=bf16\nalignment=4096\nrow_stride=4096\n"
               << "row_dimension=2048\nusable_rows=16\npadded_rows=16\n"
               << "unigram_vocab=32\neos_token=31\nmultipliers=3,5,7\n"
               << "head_vocab_sizes=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
               << "head_offsets=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
               << "file\t0\t" << direct_data_path << "\t65536\t65536\n"
               << "part\t0\t0\t0\t16\t0\t65536\n";
    }
    auto direct_layout = q38::load_ple_layout(direct_layout_path);
    q38::PleStoreOptionsV1 direct_options;
    direct_options.cache_bytes = 4u * 4096u;
    direct_options.io_mode = q38::PleIoModeV1::kIoUringDirect;
    direct_options.queue_depth = 8;
    q38::PleStore direct_store(std::move(direct_layout), direct_options);
    const auto direct_rows = direct_store.read_rows(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    CHECK(direct_rows.size() == 16u * 4096u);
    CHECK(direct_rows[0] == 0);
    CHECK(direct_rows[3u * 4096u] == 3);
    CHECK(direct_rows[7u * 4096u] == 7);
    CHECK(direct_rows[15u * 4096u] == 15);
    const auto direct_stats = direct_store.cache_stats();
    CHECK(direct_stats.io_uring_enabled == 1);
    CHECK(direct_stats.direct_io_enabled == 1);
    CHECK(direct_stats.io_uring_submissions == 16);
    CHECK(direct_stats.io_uring_completions == 16);
    CHECK(direct_stats.maximum_queue_depth == 8);
    CHECK(direct_stats.direct_read_bytes == 16u * 4096u);
    CHECK(direct_stats.read_latency_p50_ns > 0);
    std::remove(direct_layout_path.c_str());
    std::remove(direct_data_path.c_str());
#endif
}

}  // namespace

int main() {
    test_boundary_contract();
    test_boundary_checksum_rejects_corruption();
    test_transactions();
    test_rejects_bad_ack();
    test_shortens_only_unacknowledged_speculation();
    test_randomized_frontiers();
    test_mock_executor();
    test_pending_terminal_replacement();
    test_pending_terminal_replacement_rpc();
    test_supplied_candidate_verification();
    test_candidate_verification_rpc();
    test_batched_mtp_boundaries();
    test_batched_mtp_draft_faults();
    test_batched_mtp_pipeline();
    test_width_one_speculative_pipeline();
    test_width_n_speculative_pipeline();
    test_retained_draft_cancellation_rolls_back_transaction();
    test_retained_draft_overlaps_first_stage0_row();
    test_speculative_stop_token_is_transactionally_truncated();
    test_executor_context_limit();
    test_fixed_mtp4_full_context_contract();
    test_executor_rpc_service();
    test_executor_rpc_logit_diagnostics();
    test_executor_rpc_cancel_and_deadline();
    test_transactional_sampling();
    test_fatal_cancellation_keeps_rpc_category();
    test_batched_adaptive_executor();
    test_batched_mtp_width_policy();
    test_batched_mtp_variable_acceptance_cost();
    test_adaptive_mtp();
    test_compact_sampling();
    test_session_identity_contract();
    test_append_only_snapshot_journal();
    test_metrics_schema();
    test_stage_worker_startup_lifecycle();
    test_artifact_manifest();
    test_tensor_index();
    test_device_artifact_index_and_mapping();
    test_execute_failure_rolls_back();
    test_commit_failure_poisoned();
    test_ple_layout_hash_and_store();
    if (failures) {
        std::cerr << "q38 runtime tests: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "q38 runtime tests: ok\n";
    return 0;
}
