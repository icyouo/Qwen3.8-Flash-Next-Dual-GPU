#include "q38/mock_backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace q38 {

namespace {

std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

std::uint64_t digest_tokens(const std::vector<std::int32_t>& tokens,
                            std::uint64_t seed) {
    std::uint64_t digest = seed;
    for (const auto token : tokens)
        digest = mix(digest ^ static_cast<std::uint32_t>(token));
    return digest;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto rounding = UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16u);
}

}  // namespace

std::vector<std::int32_t> StageBackend::draft(std::int32_t,
                                              std::uint64_t,
                                              std::uint32_t,
                                              std::shared_ptr<CancellationToken>) {
    throw std::runtime_error("draft is only valid on the final stage");
}

StageBackendMetricsV1 StageBackend::metrics() const { return {}; }

MockStageBackend::MockStageBackend(Stage stage, std::uint32_t hidden_width,
                                   std::uint32_t vocab_size)
    : stage_(stage), hidden_width_(hidden_width), vocab_size_(vocab_size) {
    if (hidden_width_ == 0 || vocab_size_ == 0)
        throw std::invalid_argument("mock backend dimensions must be nonzero");
}

std::int32_t MockStageBackend::predict(std::int32_t token,
                                       std::uint64_t position) const {
    const auto value = mix(static_cast<std::uint32_t>(token) ^
                           (position * UINT64_C(0x9e3779b97f4a7c15)));
    return static_cast<std::int32_t>(value % vocab_size_);
}

StageOutput MockStageBackend::execute(StageInput input) {
    if (input.cancellation) input.cancellation->throw_if_requested();
    if (input.txn.magic != kTxnMagic || input.txn.version != kContractVersion ||
        input.txn.status != TxnStatus::kPrepared)
        throw std::runtime_error("backend received an invalid transaction");
    const auto count = static_cast<std::uint32_t>(input.token_ids.size());
    if (count == 0 || input.chunk_offset > input.txn.evaluated_count ||
        count > input.txn.evaluated_count - input.chunk_offset ||
        input.final_chunk !=
            (input.chunk_offset + count == input.txn.evaluated_count))
        throw std::runtime_error("backend received an invalid chunk extent");
    ++metrics_.execute_calls;
    metrics_.execute_tokens += count;
    const bool first = provisional_epoch_ == 0;
    if (first) {
        if (input.txn.epoch <= committed_epoch_ || input.chunk_offset != 0 ||
            input.txn.base_target != committed_frontier_)
            throw std::runtime_error(
                "backend transaction epoch/frontier is not monotonic");
        provisional_epoch_ = input.txn.epoch;
        provisional_base_ = input.txn.base_target;
        provisional_expected_ = input.txn.evaluated_count;
        provisional_processed_ = 0;
        provisional_kind_ = input.txn.kind;
        provisional_digest_ =
            input.txn.epoch ^ static_cast<std::uint8_t>(stage_);
    } else if (input.txn.epoch != provisional_epoch_ ||
               input.txn.base_target != provisional_base_ ||
               input.txn.evaluated_count != provisional_expected_ ||
               input.txn.kind != provisional_kind_ ||
               input.chunk_offset != provisional_processed_) {
        throw std::runtime_error("backend chunk is not the next provisional range");
    }

    provisional_digest_ = digest_tokens(input.token_ids, provisional_digest_);
    provisional_processed_ += count;

    StageOutput output;
    output.state_digest = provisional_digest_;
    if (stage_ == Stage::kStage0) {
        auto& frame = output.boundary.frame;
        frame.session_hash = input.txn.session_hash;
        frame.epoch = input.txn.epoch;
        frame.token_start = input.txn.base_target + input.chunk_offset;
        frame.token_count = count;
        frame.hidden_dtype = DType::kBFloat16;
        frame.producer_status = ProducerStatus::kReady;
        frame.hidden_width = hidden_width_;
        frame.ring_slot = static_cast<std::uint32_t>(input.txn.epoch % 3u);
        frame.payload_bytes = static_cast<std::uint64_t>(hidden_width_) *
                              count * sizeof(std::uint16_t);
        output.boundary.bf16.resize(
            static_cast<std::size_t>(hidden_width_) * count);
        for (std::uint32_t row = 0; row < count; ++row) {
            const auto word = static_cast<std::uint16_t>(mix(
                provisional_digest_ ^ input.txn.base_target ^
                input.chunk_offset ^ row));
            std::fill_n(output.boundary.bf16.begin() +
                            static_cast<std::size_t>(row) * hidden_width_,
                        hidden_width_, word);
        }
        frame.payload_checksum = output.boundary.payload_checksum();
        output.state_commit_count = provisional_processed_;
        if (input.cancellation) input.cancellation->throw_if_requested();
        return output;
    }

    std::string error;
    if (!validate_boundary(input.boundary.frame, count, &error))
        throw std::runtime_error("stage1 rejected boundary: " + error);
    input.boundary.wait_ready();
    if (input.boundary.payload_checksum() !=
        input.boundary.frame.payload_checksum)
        throw std::runtime_error("stage1 boundary checksum mismatch");
    if (input.boundary.frame.epoch != input.txn.epoch ||
        input.boundary.frame.token_start !=
            input.txn.base_target + input.chunk_offset ||
        input.boundary.size() !=
            static_cast<std::size_t>(hidden_width_) * count)
        throw std::runtime_error("stage1 boundary identity mismatch");

    std::uint32_t committed = count;
    if (input.txn.kind == TxnKind::kSpeculative) {
        if (!first || input.chunk_offset != 0 || !input.final_chunk)
            throw std::runtime_error("speculative verify cannot be chunked");
        committed = 1;
        for (std::uint32_t i = 1; i < count; ++i) {
            const auto expected = predict(input.token_ids[i - 1],
                                          input.txn.base_target + i);
            if (input.token_ids[i] != expected) break;
            ++committed;
        }
    }
    output.state_commit_count = input.txn.kind == TxnKind::kSpeculative
                                    ? committed
                                    : provisional_processed_;
    output.next_token = predict(input.token_ids[committed - 1],
                                input.txn.base_target + input.chunk_offset +
                                    committed);
    if (input.need_logits) {
        if (!input.final_chunk || input.txn.kind == TxnKind::kSpeculative)
            throw std::runtime_error("mock logits requested for invalid lane");
        output.logits_bf16.assign(vocab_size_, float_to_bf16(-8.0f));
        output.logits_bf16[static_cast<std::uint32_t>(output.next_token)] =
            float_to_bf16(8.0f);
        output.logits_bf16[(static_cast<std::uint32_t>(output.next_token) + 1u) %
                           vocab_size_] = float_to_bf16(7.5f);
    }
    if (input.cancellation) input.cancellation->throw_if_requested();
    return output;
}

std::vector<std::int32_t> MockStageBackend::draft(
    std::int32_t pending_token, std::uint64_t position,
    std::uint32_t max_draft,
    std::shared_ptr<CancellationToken> cancellation) {
    if (cancellation) cancellation->throw_if_requested();
    if (stage_ != Stage::kStage1)
        throw std::runtime_error("stage0 cannot draft tokens");
    std::vector<std::int32_t> tokens;
    tokens.reserve(max_draft);
    auto previous = pending_token;
    const std::uint32_t mismatch = max_draft == 0
                                       ? 0
                                       : static_cast<std::uint32_t>(
                                             mix(position) % (max_draft + 1u));
    for (std::uint32_t i = 0; i < max_draft; ++i) {
        if (cancellation) cancellation->throw_if_requested();
        auto next = predict(previous, position + i + 1u);
        if (i == mismatch && mismatch < max_draft)
            next = static_cast<std::int32_t>((next + 1) % vocab_size_);
        tokens.push_back(next);
        previous = next;
    }
    return tokens;
}

void MockStageBackend::commit(std::uint64_t epoch,
                              std::uint32_t state_commit_count) {
    if (epoch != provisional_epoch_ || state_commit_count == 0 ||
        provisional_processed_ != provisional_expected_ ||
        state_commit_count > provisional_processed_)
        throw std::runtime_error("backend commit does not match provisional state");
    committed_frontier_ = provisional_base_ + state_commit_count;
    committed_epoch_ = epoch;
    provisional_epoch_ = 0;
    provisional_expected_ = 0;
    provisional_processed_ = 0;
    provisional_kind_ = TxnKind::kInvalid;
    provisional_digest_ = 0;
    ++metrics_.commits;
}

void MockStageBackend::rollback(std::uint64_t epoch) {
    if (provisional_epoch_ == 0) {
        if (epoch <= committed_epoch_)
            throw std::runtime_error("backend rollback epoch is stale");
        committed_epoch_ = epoch;
        ++metrics_.rollbacks;
        return;
    }
    if (epoch != provisional_epoch_)
        throw std::runtime_error("backend rollback epoch mismatch");
    committed_epoch_ = epoch;
    provisional_epoch_ = 0;
    provisional_expected_ = 0;
    provisional_processed_ = 0;
    provisional_kind_ = TxnKind::kInvalid;
    provisional_digest_ = 0;
    ++metrics_.rollbacks;
}

}  // namespace q38
