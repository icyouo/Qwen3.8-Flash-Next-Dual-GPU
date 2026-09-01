#include "q38/executor.h"

#include "q38/mock_backend.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace q38 {

namespace {

class StageWorker {
public:
    explicit StageWorker(std::unique_ptr<StageBackend> backend)
        : backend_(std::move(backend)) {
        if (!backend_) throw std::invalid_argument("stage backend is null");
        // Start only after mutex_/ready_/tasks_/stopping_ have all completed
        // construction.  Starting from thread_'s member initializer races the
        // new worker against construction of the members it immediately uses.
        thread_ = std::thread([this] { run(); });
    }

    ~StageWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    Stage stage() const { return backend_->stage(); }

    template <typename Function>
    auto submit(Function function)
        -> std::future<typename std::invoke_result<Function, StageBackend&>::type> {
        using Result = typename std::invoke_result<Function, StageBackend&>::type;
        auto task = std::make_shared<std::packaged_task<Result(StageBackend&)>>(
            std::move(function));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("stage worker is stopping");
            tasks_.push_back([task](StageBackend& backend) { (*task)(backend); });
        }
        ready_.notify_one();
        return future;
    }

private:
    void run() {
        for (;;) {
            std::function<void(StageBackend&)> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task(*backend_);
        }
    }

    std::unique_ptr<StageBackend> backend_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void(StageBackend&)>> tasks_;
    bool stopping_ = false;
    std::thread thread_;
};

struct PairResult {
    CommitEventV1 commit{};
    StageOutput stage1{};
};

struct TimedStageOutput {
    StageOutput output{};
    std::uint64_t execute_ns = 0;
};

class LatencyTracker {
public:
    void record(std::uint64_t value) {
        total_ += value;
        minimum_ = values_.empty() ? value : std::min(minimum_, value);
        maximum_ = values_.empty() ? value : std::max(maximum_, value);
        values_.push_back(value);
        if (values_.size() > 4096) values_.pop_front();
        ++count_;
    }

    LatencySummaryV1 summary() const {
        LatencySummaryV1 result;
        result.count = count_;
        result.total_ns = total_;
        result.minimum_ns = minimum_;
        result.maximum_ns = maximum_;
        if (values_.empty()) return result;
        std::vector<std::uint64_t> sorted(values_.begin(), values_.end());
        std::sort(sorted.begin(), sorted.end());
        const auto percentile = [&](std::uint64_t numerator) {
            const auto rank =
                (sorted.size() * numerator + 99) / 100;
            const auto index = static_cast<std::size_t>(rank - 1);
            return sorted[index];
        };
        result.p50_ns = percentile(50);
        result.p95_ns = percentile(95);
        result.p99_ns = percentile(99);
        return result;
    }

private:
    std::deque<std::uint64_t> values_;
    std::uint64_t count_ = 0;
    std::uint64_t total_ = 0;
    std::uint64_t minimum_ = 0;
    std::uint64_t maximum_ = 0;
};

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

}  // namespace

struct DualStageExecutor::Impl {
    Impl(ExecutorOptions options_in, std::unique_ptr<StageBackend> stage0,
         std::unique_ptr<StageBackend> stage1)
        : options(options_in), engine(options.session_hash),
          sampler(options.vocab_size, options.sampling),
          worker0(std::move(stage0)), worker1(std::move(stage1)) {
        if (worker0.stage() != Stage::kStage0 ||
            worker1.stage() != Stage::kStage1)
            throw std::invalid_argument("executor stage ownership is reversed");
        if (options.append_chunk_tokens == 0)
            throw std::invalid_argument("append chunk size must be nonzero");
        if (options.context_limit == 0)
            throw std::invalid_argument("context limit must be nonzero");
        if (options.vocab_size == 0)
            throw std::invalid_argument("vocabulary size must be nonzero");
        for (const auto token : options.stop_token_ids) {
            if (token < 0 || static_cast<std::uint32_t>(token) >=
                                 options.vocab_size)
                throw std::invalid_argument("stop token is outside vocabulary");
        }
        std::sort(options.stop_token_ids.begin(), options.stop_token_ids.end());
        options.stop_token_ids.erase(
            std::unique(options.stop_token_ids.begin(),
                        options.stop_token_ids.end()),
            options.stop_token_ids.end());
        if (options.identity.identity_checksum == 0)
            options.identity = make_development_identity(
                options.session_hash, options.context_limit,
                options.sampling);
        std::string identity_error;
        if (!validate_session_identity(options.identity, &identity_error) ||
            options.identity.session_hash != options.session_hash ||
            options.identity.context_limit != options.context_limit)
            throw std::invalid_argument(
                "executor session identity mismatch: " + identity_error);
    }

    void ensure_healthy() const {
        if (poisoned) throw std::runtime_error("executor is fail-closed");
    }

    void rollback_prepared(std::uint64_t epoch) noexcept {
        ++stats.rollbacks;
        try {
            auto f0 = worker0.submit(
                [epoch](StageBackend& backend) { backend.rollback(epoch); });
            auto f1 = worker1.submit(
                [epoch](StageBackend& backend) { backend.rollback(epoch); });
            f0.get();
            f1.get();
            std::string error;
            if (engine.has_active_transaction()) engine.rollback(&error);
        } catch (...) {
            ++stats.failures;
            poisoned = true;
        }
    }

    PairResult run_transaction(TxnKind kind,
                               const std::vector<std::int32_t>& eval_tokens,
                               std::uint32_t chunk_tokens,
                               std::shared_ptr<CancellationToken> cancellation,
                               std::uint32_t appended_count = 0) {
        ensure_healthy();
        if (cancellation) cancellation->throw_if_requested();
        if (eval_tokens.empty())
            throw std::invalid_argument("cannot run an empty transaction");
        if (chunk_tokens == 0)
            throw std::invalid_argument("transaction chunk size is zero");
        const auto transaction_start = std::chrono::steady_clock::now();
        std::string error;
        const auto evaluated = static_cast<std::uint32_t>(eval_tokens.size());
        const bool prepared =
            kind == TxnKind::kAppendKnown
                ? engine.prepare_append_known(appended_count, evaluated, &error)
                : engine.prepare(kind, evaluated, &error);
        if (!prepared)
            throw std::runtime_error("prepare failed: " + error);
        const auto txn = engine.active_transaction();
        ++stats.transactions;
        stats.evaluated_tokens += txn.evaluated_count;
        switch (kind) {
        case TxnKind::kAppendKnown:
            ++stats.append_transactions;
            break;
        case TxnKind::kDecode:
            ++stats.decode_transactions;
            break;
        case TxnKind::kSpeculative:
            ++stats.speculative_transactions;
            break;
        case TxnKind::kInvalid:
            break;
        }

        try {
            std::optional<SamplerDecisionV1> sampler_decision;
            struct PendingStage0 {
                std::uint32_t offset = 0;
                std::uint32_t count = 0;
                bool final_chunk = false;
                std::vector<std::int32_t> tokens;
                std::future<TimedStageOutput> future;
            };
            auto submit_stage0 = [&](std::uint32_t offset) {
                auto pending = std::make_unique<PendingStage0>();
                pending->offset = offset;
                pending->count = std::min<std::uint32_t>(
                    chunk_tokens, txn.evaluated_count - offset);
                pending->final_chunk =
                    offset + pending->count == txn.evaluated_count;
                pending->tokens.assign(
                    eval_tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                    eval_tokens.begin() + static_cast<std::ptrdiff_t>(
                                              offset + pending->count));
                StageInput input;
                input.txn = txn;
                input.token_ids = pending->tokens;
                input.chunk_offset = pending->offset;
                input.final_chunk = pending->final_chunk;
                input.cancellation = cancellation;
                pending->future = worker0.submit(
                    [input = std::move(input)](
                        StageBackend& backend) mutable {
                        const auto start = std::chrono::steady_clock::now();
                        auto output = backend.execute(std::move(input));
                        return TimedStageOutput{std::move(output),
                                                elapsed_ns(start)};
                    });
                return pending;
            };

            StageOutput output1;
            auto pending0 = submit_stage0(0);
            while (pending0) {
                if (cancellation) cancellation->throw_if_requested();
                auto timed0 = pending0->future.get();
                stats.stage0_execute_ns += timed0.execute_ns;
                stage0_latency.record(timed0.execute_ns);
                auto output0 = std::move(timed0.output);
                if (cancellation) cancellation->throw_if_requested();

                if (!validate_boundary(output0.boundary.frame,
                                       pending0->count, &error))
                    throw std::runtime_error("stage0 boundary invalid: " + error);
                if (output0.boundary.frame.session_hash != txn.session_hash ||
                    output0.boundary.frame.epoch != txn.epoch ||
                    output0.boundary.frame.token_start !=
                        txn.base_target + pending0->offset ||
                    output0.boundary.frame.token_count != pending0->count ||
                    output0.boundary.frame.hidden_width != options.hidden_width)
                    throw std::runtime_error(
                        "stage0 boundary does not match the transaction chunk");

                StageInput input1;
                input1.txn = txn;
                input1.token_ids = std::move(pending0->tokens);
                input1.boundary = std::move(output0.boundary);
                input1.chunk_offset = pending0->offset;
                input1.final_chunk = pending0->final_chunk;
                input1.need_logits = sampler.sampled() &&
                                     pending0->final_chunk &&
                                     kind != TxnKind::kSpeculative;
                input1.cancellation = cancellation;
                const auto stage1_start = std::chrono::steady_clock::now();
                auto stage1 = worker1.submit(
                    [input = std::move(input1)](
                        StageBackend& backend) mutable {
                        return backend.execute(std::move(input));
                    });
                const auto next_offset = pending0->offset + pending0->count;
                auto next0 = next_offset < txn.evaluated_count
                                 ? submit_stage0(next_offset)
                                 : nullptr;
                output1 = stage1.get();
                const auto stage1_ns = elapsed_ns(stage1_start);
                stats.stage1_execute_ns += stage1_ns;
                stage1_latency.record(stage1_ns);
                if (cancellation) cancellation->throw_if_requested();
                pending0 = std::move(next0);
            }

            if (output1.next_token < 0 ||
                static_cast<std::uint32_t>(output1.next_token) >=
                    options.vocab_size)
                throw std::runtime_error(
                    "stage1 returned a token outside the vocabulary");

            if (sampler.sampled()) {
                if (kind == TxnKind::kSpeculative)
                    throw std::logic_error(
                        "sampled mode requires non-speculative target decode");
                const auto sampling_start = std::chrono::steady_clock::now();
                sampler_decision = sampler.prepare(
                    output1.logits_bf16,
                    kind == TxnKind::kAppendKnown
                        ? eval_tokens
                        : std::vector<std::int32_t>{});
                const auto sampling_ns = elapsed_ns(sampling_start);
                stats.sampling_ns += sampling_ns;
                sampling_latency.record(sampling_ns);
                output1.next_token = sampler_decision->token;
                if (output1.next_token < 0 ||
                    static_cast<std::uint32_t>(output1.next_token) >=
                        options.vocab_size)
                    throw std::runtime_error(
                        "sampler returned a token outside the vocabulary");
            }

            if (kind == TxnKind::kSpeculative &&
                !options.stop_token_ids.empty()) {
                // output1.state_commit_count includes the pending input row
                // and each target-accepted draft row. If eval[index] is a
                // stop, evaluate only eval[0..index-1] and publish that stop
                // as the new pending token. Both backends then commit the
                // shortened prefix; later drafts are replay-discarded.
                for (std::uint32_t index = 1;
                     index < output1.state_commit_count; ++index) {
                    if (std::binary_search(options.stop_token_ids.begin(),
                                           options.stop_token_ids.end(),
                                           eval_tokens[index])) {
                        output1.state_commit_count = index;
                        output1.next_token = eval_tokens[index];
                        break;
                    }
                }
            }

            if (cancellation) cancellation->seal();

            if (!engine.acknowledge(Stage::kStage0, txn.epoch,
                                    txn.evaluated_count, &error) ||
                !engine.acknowledge(Stage::kStage1, txn.epoch,
                                    txn.evaluated_count, &error))
                throw std::runtime_error("stage acknowledge failed: " + error);

            const std::uint32_t state_commit =
                kind == TxnKind::kSpeculative
                    ? output1.state_commit_count
                    : txn.evaluated_count;
            if (!engine.decide(state_commit, &error))
                throw std::runtime_error("transaction decision failed: " + error);

            CommitEventV1 event;
            if (!engine.commit(&event, &error))
                throw std::runtime_error("coordinator commit failed: " + error);

            try {
                const auto commit_start = std::chrono::steady_clock::now();
                auto c0 = worker0.submit([epoch = txn.epoch, state_commit](
                                             StageBackend& backend) {
                    backend.commit(epoch, state_commit);
                });
                auto c1 = worker1.submit([epoch = txn.epoch, state_commit](
                                             StageBackend& backend) {
                    backend.commit(epoch, state_commit);
                });
                c0.get();
                c1.get();
                if (sampler_decision) {
                    sampler.commit(*sampler_decision);
                    ++stats.sampled_tokens;
                }
                const auto commit_ns = elapsed_ns(commit_start);
                stats.backend_commit_ns += commit_ns;
                commit_latency.record(commit_ns);
            } catch (...) {
                poisoned = true;
                throw;
            }
            stats.state_committed_tokens += state_commit;
            stats.published_tokens += event.newly_published;
            transaction_latency.record(elapsed_ns(transaction_start));
            return PairResult{event, std::move(output1)};
        } catch (const ExecutionDeadlineExceeded&) {
            ++stats.failures;
            ++stats.deadline_exceeded;
            if (engine.has_active_transaction()) rollback_prepared(txn.epoch);
            transaction_latency.record(elapsed_ns(transaction_start));
            throw;
        } catch (const ExecutionCancelled&) {
            ++stats.failures;
            ++stats.cancellations;
            if (engine.has_active_transaction()) rollback_prepared(txn.epoch);
            transaction_latency.record(elapsed_ns(transaction_start));
            throw;
        } catch (...) {
            ++stats.failures;
            if (engine.has_active_transaction()) rollback_prepared(txn.epoch);
            if (options.backend_failure_is_fatal) poisoned = true;
            transaction_latency.record(elapsed_ns(transaction_start));
            throw;
        }
    }

    ExecutorOptions options;
    SessionTxnEngine engine;
    TransactionalSampler sampler;
    StageWorker worker0;
    StageWorker worker1;
    std::vector<std::int32_t> tokens;
    std::int32_t last_sample = -1;
    bool have_last_sample = false;
    bool poisoned = false;
    ExecutorStatsV1 stats{};
    LatencyTracker transaction_latency;
    LatencyTracker stage0_latency;
    LatencyTracker stage1_latency;
    LatencyTracker commit_latency;
    LatencyTracker draft_latency;
    LatencyTracker sampling_latency;
};

DualStageExecutor::DualStageExecutor(ExecutorOptions options,
                                     std::unique_ptr<StageBackend> stage0,
                                     std::unique_ptr<StageBackend> stage1)
    : impl_(std::make_unique<Impl>(options, std::move(stage0),
                                  std::move(stage1))) {}

DualStageExecutor::~DualStageExecutor() = default;

void DualStageExecutor::append(
    const std::vector<std::int32_t>& token_ids,
    std::shared_ptr<CancellationToken> cancellation) {
    impl_->ensure_healthy();
    if (cancellation) cancellation->throw_if_requested();
    if (token_ids.empty()) throw std::invalid_argument("cannot append zero tokens");
    for (const auto token : token_ids) {
        if (token < 0 || static_cast<std::uint32_t>(token) >=
                             impl_->options.vocab_size)
            throw std::invalid_argument("token is outside the configured vocabulary");
    }
    const auto canonical = impl_->engine.frontiers().canonical;
    if (token_ids.size() > impl_->options.context_limit -
                               std::min<std::uint64_t>(
                                   canonical, impl_->options.context_limit))
        throw std::length_error("append exceeds the configured context limit");
    const auto& frontiers = impl_->engine.frontiers();
    const auto pending = frontiers.canonical - frontiers.target;
    if (pending > 1)
        throw std::runtime_error("append has more than one pending token");
    std::vector<std::int32_t> evaluated;
    evaluated.reserve(token_ids.size() + static_cast<std::size_t>(pending));
    if (pending == 1)
        evaluated.push_back(impl_->tokens.at(frontiers.target));
    evaluated.insert(evaluated.end(), token_ids.begin(), token_ids.end());
    auto result = impl_->run_transaction(
        TxnKind::kAppendKnown, evaluated, impl_->options.append_chunk_tokens,
        std::move(cancellation), static_cast<std::uint32_t>(token_ids.size()));
    impl_->tokens.insert(impl_->tokens.end(), token_ids.begin(), token_ids.end());
    impl_->sampler.commit_tokens(token_ids);
    impl_->last_sample = result.stage1.next_token;
    impl_->have_last_sample = true;
}

std::int32_t DualStageExecutor::seed_decode(
    std::shared_ptr<CancellationToken> cancellation) {
    impl_->ensure_healthy();
    if (cancellation) {
        cancellation->throw_if_requested();
        cancellation->seal();
    }
    if (!impl_->have_last_sample)
        throw std::runtime_error("append must produce logits before decode seed");
    if (impl_->engine.frontiers().canonical >= impl_->options.context_limit)
        throw std::length_error("decode seed exceeds the context limit");
    std::string error;
    if (!impl_->engine.seed_decode_pending(&error))
        throw std::runtime_error("decode seed failed: " + error);
    impl_->tokens.push_back(impl_->last_sample);
    impl_->sampler.commit_token(impl_->last_sample);
    // The seed is a model-free publication of the logits decision already
    // produced by append.  It is not another evaluated transaction, but it is
    // a client-visible committed token and must be included in publication
    // accounting.
    ++impl_->stats.published_tokens;
    return impl_->last_sample;
}

DecodeResult DualStageExecutor::decode_one(
    std::shared_ptr<CancellationToken> cancellation) {
    impl_->ensure_healthy();
    const auto& f = impl_->engine.frontiers();
    if (f.canonical - f.target != 1)
        throw std::runtime_error("decode requires exactly one pending token");
    if (f.canonical >= impl_->options.context_limit)
        throw std::length_error("decode would exceed the context limit");
    std::vector<std::int32_t> eval{impl_->tokens.at(f.target)};
    auto pair = impl_->run_transaction(TxnKind::kDecode, eval, 1,
                                       std::move(cancellation));
    impl_->tokens.push_back(pair.stage1.next_token);
    impl_->sampler.commit_token(pair.stage1.next_token);
    impl_->last_sample = pair.stage1.next_token;
    return DecodeResult{pair.commit, {pair.stage1.next_token}};
}

DecodeResult DualStageExecutor::speculative_step(
    std::uint32_t max_draft,
    std::shared_ptr<CancellationToken> cancellation) {
    impl_->ensure_healthy();
    if (cancellation) cancellation->throw_if_requested();
    // Target-only fallback is mandatory until sampled rejection semantics can
    // consume both target and draft probabilities. Never substitute argmax.
    if (impl_->sampler.sampled())
        return decode_one(std::move(cancellation));
    if (max_draft == 0) return decode_one(std::move(cancellation));
    const auto& f = impl_->engine.frontiers();
    if (f.canonical - f.target != 1)
        throw std::runtime_error("speculative decode requires one pending token");
    if (f.canonical >= impl_->options.context_limit)
        throw std::length_error(
            "speculative decode would exceed the context limit");
    const auto remaining = impl_->options.context_limit - f.canonical;
    const auto capped_draft = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(max_draft, remaining - 1));
    if (capped_draft == 0) return decode_one(std::move(cancellation));
    const auto pending = impl_->tokens.at(f.target);
    const auto draft_start = std::chrono::steady_clock::now();
    const bool retained_draft = capped_draft == 1;
    if (retained_draft &&
        f.epoch == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("retained MTP draft epoch overflows");
    const auto retained_epoch = retained_draft ? f.epoch + 1 : 0;
    auto drafts = impl_->worker1.submit(
        [pending, position = f.target, capped_draft,
         retained_epoch, cancellation](StageBackend& backend) {
            return retained_epoch != 0
                       ? backend.draft_retained(pending, position,
                                                retained_epoch, cancellation)
                       : backend.draft(pending, position, capped_draft,
                                       cancellation);
        }).get();
    const auto draft_ns = elapsed_ns(draft_start);
    impl_->stats.draft_ns += draft_ns;
    impl_->draft_latency.record(draft_ns);
    std::vector<std::int32_t> eval;
    PairResult pair;
    try {
        if (cancellation) cancellation->throw_if_requested();
        impl_->stats.drafted_tokens += drafts.size();
        if ((retained_draft && drafts.size() != 1) ||
            drafts.size() > capped_draft)
            throw std::runtime_error(
                "draft backend exceeded the requested width");
        for (const auto token : drafts) {
            if (token < 0 || static_cast<std::uint32_t>(token) >=
                                 impl_->options.vocab_size)
                throw std::runtime_error(
                    "draft backend returned a token outside the vocabulary");
        }
        eval.reserve(drafts.size() + 1);
        eval.push_back(pending);
        eval.insert(eval.end(), drafts.begin(), drafts.end());
        pair = impl_->run_transaction(
            TxnKind::kSpeculative, eval,
            retained_draft ? 1u
                           : static_cast<std::uint32_t>(eval.size()),
            std::move(cancellation));
    } catch (...) {
        if (retained_draft) {
            try {
                impl_->worker1
                    .submit([retained_epoch](StageBackend& backend) {
                        backend.abandon_retained_draft(retained_epoch);
                    })
                    .get();
            } catch (...) {
                impl_->poisoned = true;
            }
        }
        throw;
    }

    DecodeResult result;
    result.commit = pair.commit;
    for (std::uint32_t i = 1; i < pair.commit.newly_published; ++i)
        result.published_tokens.push_back(eval.at(i));
    result.published_tokens.push_back(pair.stage1.next_token);
    impl_->tokens.insert(impl_->tokens.end(), result.published_tokens.begin(),
                         result.published_tokens.end());
    impl_->sampler.commit_tokens(result.published_tokens);
    impl_->last_sample = pair.stage1.next_token;
    return result;
}

const SessionFrontiersV1& DualStageExecutor::frontiers() const {
    return impl_->engine.frontiers();
}

const std::vector<std::int32_t>& DualStageExecutor::canonical_tokens() const {
    return impl_->tokens;
}

ExecutorStatsV1 DualStageExecutor::stats() const { return impl_->stats; }

MetricsSchemaV1 DualStageExecutor::metrics() {
    MetricsSchemaV1 result;
    result.timestamp_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    result.identity_checksum = impl_->options.identity.identity_checksum;
    result.frontiers = impl_->engine.frontiers();
    result.sampler = impl_->sampler.state();
    result.executor = impl_->stats;
    auto stage0 = impl_->worker0.submit(
        [](StageBackend& backend) { return backend.metrics(); });
    auto stage1 = impl_->worker1.submit(
        [](StageBackend& backend) { return backend.metrics(); });
    result.stage0 = stage0.get();
    result.stage1 = stage1.get();
    result.host = collect_host_metrics();
    result.host.runtime_pinned_bytes =
        result.stage0.boundary_pinned_bytes +
        result.stage0.workspace_pinned_bytes +
        result.stage1.boundary_pinned_bytes +
        result.stage1.workspace_pinned_bytes;
    result.host.runtime_pinned_peak_bytes = std::max(
        result.host.runtime_pinned_bytes,
        std::max(result.stage0.weight_staging_peak_pinned_bytes,
                 result.stage1.weight_staging_peak_pinned_bytes));
    result.transaction_latency = impl_->transaction_latency.summary();
    result.stage0_latency = impl_->stage0_latency.summary();
    result.stage1_latency = impl_->stage1_latency.summary();
    result.commit_latency = impl_->commit_latency.summary();
    result.draft_latency = impl_->draft_latency.summary();
    result.sampling_latency = impl_->sampling_latency.summary();
    return result;
}

SamplerStateV1 DualStageExecutor::sampler_state() const {
    return impl_->sampler.state();
}

const SessionIdentityV1& DualStageExecutor::identity() const {
    return impl_->options.identity;
}

StateSnapshotV1 DualStageExecutor::snapshot() const {
    if (impl_->engine.has_active_transaction())
        throw std::logic_error("cannot snapshot an active transaction");
    StateSnapshotV1 result;
    result.header.identity_checksum = impl_->options.identity.identity_checksum;
    result.header.frontiers = impl_->engine.frontiers();
    result.header.sampler = impl_->sampler.state();
    result.header.canonical_token_count = impl_->tokens.size();
    result.header.canonical_checksum =
        canonical_token_checksum(impl_->tokens);
    result.header.last_sample = impl_->last_sample;
    result.header.flags = kSnapshotColdRebuild |
                          (impl_->have_last_sample
                               ? kSnapshotHasLastSample
                               : 0u);
    result.canonical_tokens = impl_->tokens;
    return result;
}

void DualStageExecutor::fail_closed() { impl_->poisoned = true; }

bool DualStageExecutor::poisoned() const { return impl_->poisoned; }

std::unique_ptr<DualStageExecutor> make_mock_executor(
    const ExecutorOptions& options) {
    return std::make_unique<DualStageExecutor>(
        options,
        std::make_unique<MockStageBackend>(Stage::kStage0,
                                           options.hidden_width,
                                           options.vocab_size),
        std::make_unique<MockStageBackend>(Stage::kStage1,
                                           options.hidden_width,
                                           options.vocab_size));
}

}  // namespace q38
