#ifndef Q38_SAMPLING_H
#define Q38_SAMPLING_H

#include "q38/contracts.h"

#include <cstdint>
#include <vector>

namespace q38 {

constexpr std::uint32_t kSamplerStateMagic = fourcc('Q', '3', '8', 'S');

enum class SamplingModeV1 : std::uint16_t {
    kGreedy = 0,
    kTopKTopP = 1,
};

struct alignas(8) SamplingConfigV1 {
    SamplingModeV1 mode = SamplingModeV1::kGreedy;
    std::uint16_t version = 1;
    std::uint32_t top_k = 0;
    float top_p = 1.0f;
    float temperature = 1.0f;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::uint64_t seed = 1;
};

struct alignas(8) SamplerStateV1 {
    std::uint32_t magic = kSamplerStateMagic;
    std::uint16_t version = kContractVersion;
    std::uint16_t header_bytes = sizeof(SamplerStateV1);
    std::uint64_t rng_state = 1;
    std::uint64_t sampled_tokens = 0;
    std::uint64_t canonical_tokens = 0;
    std::uint64_t penalty_checksum = 0;
};

struct SamplerDecisionV1 {
    std::int32_t token = -1;
    std::uint64_t rng_before = 0;
    std::uint64_t rng_after = 0;
};

static_assert(sizeof(SamplingConfigV1) == 40,
              "sampling config ABI changed without a version bump");
static_assert(sizeof(SamplerStateV1) == 40,
              "sampler state ABI changed without a version bump");

class TransactionalSampler {
public:
    TransactionalSampler(std::uint32_t vocabulary,
                         SamplingConfigV1 config = {});

    bool sampled() const {
        return config_.mode != SamplingModeV1::kGreedy;
    }
    const SamplingConfigV1& config() const { return config_; }
    const SamplerStateV1& state() const { return state_; }

    SamplerDecisionV1 prepare(
        const std::vector<std::uint16_t>& logits_bf16,
        const std::vector<std::int32_t>& additional_penalty_tokens = {}) const;
    void commit(const SamplerDecisionV1& decision);
    void commit_tokens(const std::vector<std::int32_t>& tokens);
    void commit_token(std::int32_t token);
    void reset();

private:
    std::uint32_t vocabulary_ = 0;
    SamplingConfigV1 config_{};
    SamplerStateV1 state_{};
    std::vector<std::uint32_t> counts_;
};

}  // namespace q38

#endif
