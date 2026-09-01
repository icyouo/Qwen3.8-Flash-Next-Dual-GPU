#ifndef Q38_MOCK_BACKEND_H
#define Q38_MOCK_BACKEND_H

#include "q38/backend.h"

#include <cstdint>

namespace q38 {

class MockStageBackend final : public StageBackend {
public:
    MockStageBackend(Stage stage, std::uint32_t hidden_width,
                     std::uint32_t vocab_size);

    Stage stage() const override { return stage_; }
    StageOutput execute(StageInput input) override;
    std::vector<std::int32_t> draft(std::int32_t pending_token,
                                    std::uint64_t position,
                                    std::uint32_t max_draft,
                                    std::shared_ptr<CancellationToken>
                                        cancellation = {}) override;
    void checkpoint_speculative_prefix(
        std::uint64_t transaction_epoch,
        std::uint32_t prefix_tokens) override;
    void commit(std::uint64_t epoch,
                std::uint32_t state_commit_count) override;
    void rollback(std::uint64_t epoch) override;
    void reset_session() override;
    StageBackendMetricsV1 metrics() const override { return metrics_; }

private:
    std::int32_t predict(std::int32_t token, std::uint64_t position) const;

    Stage stage_;
    std::uint32_t hidden_width_;
    std::uint32_t vocab_size_;
    std::uint64_t committed_frontier_ = 0;
    std::uint64_t committed_epoch_ = 0;
    std::uint64_t provisional_epoch_ = 0;
    std::uint64_t provisional_base_ = 0;
    std::uint32_t provisional_expected_ = 0;
    std::uint32_t provisional_processed_ = 0;
    TxnKind provisional_kind_ = TxnKind::kInvalid;
    std::uint64_t provisional_digest_ = 0;
    std::vector<std::int32_t> provisional_tokens_;
    std::uint32_t speculative_checkpoint_ = 0;
    StageBackendMetricsV1 metrics_{};
};

}  // namespace q38

#endif
