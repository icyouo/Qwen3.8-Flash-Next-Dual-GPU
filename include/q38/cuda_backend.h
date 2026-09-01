#ifndef Q38_CUDA_BACKEND_H
#define Q38_CUDA_BACKEND_H

#include "q38/backend.h"
#include "q38/device_artifact.h"
#include "q38/executor.h"
#include "q38/ple.h"

#include <cstdint>
#include <memory>
#include <string>

namespace q38 {

struct CudaStageBackendOptions {
    Stage stage = Stage::kStage0;
    int device = 0;
    std::uint32_t max_transaction_tokens = 4096;
    std::uint32_t context_capacity = 262144;
    std::uint64_t prefill_matrix_cache_bytes = 8ull << 30u;
    bool enable_mtp = false;
};

class CudaStageBackend final : public StageBackend {
public:
    CudaStageBackend(DeviceStageIndexV1 index,
                     CudaStageBackendOptions options,
                     std::shared_ptr<PleStore> ple_store = nullptr);
    ~CudaStageBackend();

    CudaStageBackend(const CudaStageBackend&) = delete;
    CudaStageBackend& operator=(const CudaStageBackend&) = delete;

    Stage stage() const override;
    StageOutput execute(StageInput input) override;
    std::vector<std::int32_t> draft(std::int32_t pending_token,
                                    std::uint64_t position,
                                    std::uint32_t max_draft,
                                    std::shared_ptr<CancellationToken>
                                        cancellation = {}) override;
    std::vector<std::int32_t> draft_retained(
        std::int32_t pending_token, std::uint64_t position,
        std::uint64_t transaction_epoch,
        std::shared_ptr<CancellationToken> cancellation = {}) override;
    void abandon_retained_draft(
        std::uint64_t transaction_epoch) override;
    void commit(std::uint64_t epoch,
                std::uint32_t state_commit_count) override;
    void rollback(std::uint64_t epoch) override;
    void reset_session() override;
    StageBackendMetricsV1 metrics() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<DualStageExecutor> make_cuda_executor(
    const std::string& stage0_index,
    const std::string& stage1_index,
    const std::string& ple_layout,
    const ExecutorOptions& executor_options,
    int stage0_device = 0,
    int stage1_device = 1,
    std::uint64_t ple_cache_bytes = 8ull << 30u,
    PleIoModeV1 ple_io_mode = PleIoModeV1::kAuto,
    std::uint32_t ple_queue_depth = 64,
    bool enable_mtp = false);

bool cuda_q38_backend_compiled();

}  // namespace q38

#endif
