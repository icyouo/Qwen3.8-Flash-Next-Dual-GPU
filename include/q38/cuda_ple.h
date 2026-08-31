#ifndef Q38_CUDA_PLE_H
#define Q38_CUDA_PLE_H

#include <cstdint>
#include <memory>

namespace q38 {

constexpr std::uint32_t kQ38PleRowsPerToken = 16;
constexpr std::uint32_t kQ38PleRowWidth = 160;
constexpr std::uint32_t kQ38PleEmbeddingWidth = 2560;
constexpr std::uint32_t kQ38PleConvState = 9;

class CudaPleStateBank {
public:
    explicit CudaPleStateBank(int device);
    ~CudaPleStateBank();

    CudaPleStateBank(const CudaPleStateBank&) = delete;
    CudaPleStateBank& operator=(const CudaPleStateBank&) = delete;
    CudaPleStateBank(CudaPleStateBank&&) noexcept;
    CudaPleStateBank& operator=(CudaPleStateBank&&) noexcept;

    void begin(std::uint64_t epoch, void* stream);
    void restore(void* stream);
    std::uint16_t* working() const;
    void commit(std::uint64_t epoch);
    void rollback(std::uint64_t epoch);
    void reset(void* stream);
    std::uint64_t allocated_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void cuda_ple_fp8_rows_to_bf16(const std::uint8_t* fp8_rows,
                               const std::uint16_t* row_scales,
                               std::uint16_t* embedding,
                               std::uint32_t tokens,
                               void* stream,
                               int device);

void cuda_ple_gate_bf16(const std::uint16_t* normalized_key,
                        const std::uint16_t* normalized_query,
                        const std::uint16_t* value,
                        std::uint16_t* gated_value,
                        void* stream,
                        int device);

void cuda_ple_conv_decode_bf16(const std::uint16_t* gated_value,
                               const std::uint16_t* normalized_gated_value,
                               const std::uint16_t* conv_weight,
                               std::uint16_t* conv_state,
                               std::uint16_t* output,
                               void* stream,
                               int device);

void cuda_ple_gate_prefill_bf16(const std::uint16_t* normalized_key,
                                const std::uint16_t* normalized_query,
                                const std::uint16_t* value,
                                std::uint16_t* gated_value,
                                std::uint32_t tokens,
                                void* stream,
                                int device);

void cuda_ple_conv_prefill_bf16(
    const std::uint16_t* gated_value,
    const std::uint16_t* normalized_gated_value,
    const std::uint16_t* conv_weight,
    std::uint16_t* conv_state,
    std::uint16_t* output,
    std::uint32_t tokens,
    void* stream,
    int device);

bool cuda_q38_ple_compiled();

}  // namespace q38

#endif
