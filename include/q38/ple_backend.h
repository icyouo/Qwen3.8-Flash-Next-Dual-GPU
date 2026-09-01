#ifndef Q38_PLE_BACKEND_H
#define Q38_PLE_BACKEND_H

#include "q38/backend.h"
#include "q38/ple.h"

#include <memory>
#include <vector>

namespace q38 {

// Decorates stage0 with the production PLE state rule: storage cache is
// non-semantic and survives rollback, while n-gram history is candidate state
// and commits only through the accepted target prefix.
class PleStage0Backend final : public StageBackend {
public:
    PleStage0Backend(std::unique_ptr<StageBackend> inner,
                     std::shared_ptr<PleStore> store,
                     PleHashConfigV1 hash_config);

    Stage stage() const override { return Stage::kStage0; }
    void prefetch_transaction(
        const SessionTxnV1& txn,
        const std::vector<std::int32_t>& token_ids,
        std::shared_ptr<CancellationToken> cancellation = {}) override;
    StageOutput execute(StageInput input) override;
    std::vector<std::int32_t> draft(std::int32_t pending_token,
                                    std::uint64_t position,
                                    std::uint32_t max_draft,
                                    std::shared_ptr<CancellationToken>
                                        cancellation = {}) override;
    void commit(std::uint64_t epoch,
                std::uint32_t state_commit_count) override;
    void rollback(std::uint64_t epoch) override;
    void reset_session() override;
    StageBackendMetricsV1 metrics() const override;

private:
    std::unique_ptr<StageBackend> inner_;
    std::shared_ptr<PleStore> store_;
    PleHashState committed_hash_;
    std::vector<PleHashState> candidate_hashes_;
    std::uint64_t provisional_epoch_ = 0;
    std::uint32_t provisional_expected_ = 0;
    std::uint32_t provisional_processed_ = 0;
};

}  // namespace q38

#endif
