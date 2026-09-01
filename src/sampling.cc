#include "q38/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace q38 {

namespace {

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint64_t splitmix64(std::uint64_t state) {
    auto value = state;
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

std::uint64_t next_state(std::uint64_t state) {
    return state + UINT64_C(0x9e3779b97f4a7c15);
}

double uniform01(std::uint64_t state) {
    return static_cast<double>(splitmix64(state) >> 11u) *
           (1.0 / 9007199254740992.0);
}

}  // namespace

TransactionalSampler::TransactionalSampler(std::uint32_t vocabulary,
                                           SamplingConfigV1 config)
    : vocabulary_(vocabulary), config_(config), counts_(vocabulary, 0) {
    if (vocabulary_ == 0 || config_.version != 1)
        throw std::invalid_argument("invalid sampler dimensions/version");
    if (config_.mode != SamplingModeV1::kGreedy &&
        config_.mode != SamplingModeV1::kTopKTopP)
        throw std::invalid_argument("unsupported sampling mode");
    if (!(config_.temperature > 0.0f) ||
        !std::isfinite(config_.temperature) || !(config_.top_p > 0.0f) ||
        config_.top_p > 1.0f || !std::isfinite(config_.top_p) ||
        config_.top_k > vocabulary_ || config_.repetition_penalty < 1.0f ||
        !std::isfinite(config_.repetition_penalty) ||
        !std::isfinite(config_.frequency_penalty) ||
        !std::isfinite(config_.presence_penalty))
        throw std::invalid_argument("invalid sampling parameters");
    state_.rng_state = config_.seed;
}

SamplerDecisionV1 TransactionalSampler::prepare(
    const std::vector<std::uint16_t>& logits_bf16,
    const std::vector<std::int32_t>& additional_penalty_tokens) const {
    if (!sampled())
        throw std::logic_error("greedy sampler does not consume RNG");
    if (logits_bf16.size() != vocabulary_)
        throw std::invalid_argument("sampler logits size differs from vocabulary");

    std::unordered_map<std::int32_t, std::uint32_t> additional;
    for (const auto token : additional_penalty_tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= vocabulary_)
            throw std::invalid_argument("penalty token is outside vocabulary");
        ++additional[token];
    }

    std::vector<std::pair<float, std::int32_t>> candidates;
    candidates.reserve(vocabulary_);
    for (std::uint32_t token = 0; token < vocabulary_; ++token) {
        auto logit = bf16_to_float(logits_bf16[token]);
        if (!std::isfinite(logit))
            logit = -std::numeric_limits<float>::infinity();
        const auto extra = additional.find(static_cast<std::int32_t>(token));
        const std::uint64_t count =
            counts_[token] +
            (extra == additional.end() ? 0u : extra->second);
        if (count != 0) {
            if (config_.repetition_penalty != 1.0f)
                logit = logit > 0.0f
                            ? logit / config_.repetition_penalty
                            : logit * config_.repetition_penalty;
            logit -= config_.presence_penalty;
            logit -= config_.frequency_penalty * static_cast<float>(count);
        }
        candidates.emplace_back(logit / config_.temperature,
                                static_cast<std::int32_t>(token));
    }
    const auto descending = [](const auto& left, const auto& right) {
        return left.first > right.first;
    };
    if (config_.top_k != 0 && config_.top_k < candidates.size()) {
        std::nth_element(candidates.begin(),
                         candidates.begin() + config_.top_k,
                         candidates.end(), descending);
        candidates.resize(config_.top_k);
    }
    std::sort(candidates.begin(), candidates.end(), descending);
    if (candidates.empty() || !std::isfinite(candidates.front().first))
        throw std::runtime_error("sampler has no finite candidate logits");

    const auto maximum = candidates.front().first;
    std::vector<double> weights(candidates.size());
    double total = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        weights[index] = std::exp(
            static_cast<double>(candidates[index].first - maximum));
        total += weights[index];
    }
    if (!(total > 0.0) || !std::isfinite(total))
        throw std::runtime_error("sampler probability normalization failed");

    std::size_t retained = weights.size();
    if (config_.top_p < 1.0f) {
        double cumulative = 0.0;
        retained = 0;
        while (retained < weights.size()) {
            cumulative += weights[retained] / total;
            ++retained;
            if (cumulative >= config_.top_p) break;
        }
        total = 0.0;
        for (std::size_t index = 0; index < retained; ++index)
            total += weights[index];
    }

    const auto after = next_state(state_.rng_state);
    const auto draw = uniform01(after) * total;
    double cumulative = 0.0;
    std::size_t selected = retained - 1;
    for (std::size_t index = 0; index < retained; ++index) {
        cumulative += weights[index];
        if (draw < cumulative) {
            selected = index;
            break;
        }
    }
    return SamplerDecisionV1{candidates[selected].second, state_.rng_state,
                             after};
}

void TransactionalSampler::commit(const SamplerDecisionV1& decision) {
    if (!sampled() || decision.token < 0 ||
        static_cast<std::uint32_t>(decision.token) >= vocabulary_ ||
        decision.rng_before != state_.rng_state ||
        decision.rng_after != next_state(decision.rng_before))
        throw std::logic_error("sampler decision does not match committed state");
    state_.rng_state = decision.rng_after;
    ++state_.sampled_tokens;
}

void TransactionalSampler::commit_tokens(
    const std::vector<std::int32_t>& tokens) {
    for (const auto token : tokens) commit_token(token);
}

void TransactionalSampler::commit_token(std::int32_t token) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= vocabulary_)
        throw std::invalid_argument("committed penalty token is invalid");
    ++counts_[static_cast<std::uint32_t>(token)];
    ++state_.canonical_tokens;
    state_.penalty_checksum ^=
        splitmix64(static_cast<std::uint32_t>(token) ^
                   (state_.canonical_tokens *
                    UINT64_C(0x9e3779b97f4a7c15)));
}

void TransactionalSampler::reset() {
    state_ = SamplerStateV1{};
    state_.rng_state = config_.seed;
    std::fill(counts_.begin(), counts_.end(), 0);
}

}  // namespace q38
