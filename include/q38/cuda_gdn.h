#ifndef Q38_CUDA_GDN_H
#define Q38_CUDA_GDN_H

#include <cstddef>
#include <cstdint>
#include <memory>

namespace q38 {

constexpr std::uint32_t kQ38GdnKeyHeads = 16;
constexpr std::uint32_t kQ38GdnValueHeads = 48;
constexpr std::uint32_t kQ38GdnHeadWidth = 128;
constexpr std::uint32_t kQ38GdnQkvWidth = 10240;
constexpr std::uint32_t kQ38GdnValueWidth = 6144;
constexpr std::uint32_t kQ38GdnConvWidth = 4;

struct CudaGdnLayerStateView {
    std::uint16_t* conv = nullptr;
    float* recurrent = nullptr;
};

// Transactional storage for all GDN layers owned by one pipeline stage. The
// working bank is cloned from committed state at begin; commit swaps banks and
// rollback simply discards working state.
class CudaGdnStateBank {
public:
    CudaGdnStateBank(int device, std::uint32_t layers);
    ~CudaGdnStateBank();

    CudaGdnStateBank(const CudaGdnStateBank&) = delete;
    CudaGdnStateBank& operator=(const CudaGdnStateBank&) = delete;
    CudaGdnStateBank(CudaGdnStateBank&&) noexcept;
    CudaGdnStateBank& operator=(CudaGdnStateBank&&) noexcept;

    void begin(std::uint64_t epoch, void* stream);
    void restore(void* stream);
    CudaGdnLayerStateView working(std::uint32_t local_layer) const;
    void commit(std::uint64_t epoch);
    void rollback(std::uint64_t epoch);
    void reset(void* stream);

    std::uint32_t layers() const;
    std::uint64_t bytes_per_bank() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void cuda_gdn_conv_decode_bf16(const std::uint16_t* projected_qkv,
                               const std::uint16_t* conv_weight,
                               std::uint16_t* conv_state,
                               std::uint16_t* activated_qkv,
                               void* stream,
                               int device);

void cuda_gdn_recurrent_decode_bf16(const std::uint16_t* activated_qkv,
                                    const std::uint16_t* projected_b,
                                    const std::uint16_t* projected_a,
                                    const std::uint16_t* a_log,
                                    const std::uint16_t* dt_bias,
                                    float* recurrent_state,
                                    std::uint16_t* core_output,
                                    void* stream,
                                    int device);

void cuda_gdn_output_norm_bf16(const std::uint16_t* core_output,
                               const std::uint16_t* gate_z,
                               const std::uint16_t* norm_weight,
                               std::uint16_t* output,
                               float epsilon,
                               void* stream,
                               int device);

// Layer-major prefill variants.  Projections remain batched GEMMs while these
// kernels advance the causal convolution/recurrent state in token order inside
// one launch per layer and tile.
void cuda_gdn_conv_prefill_bf16(const std::uint16_t* projected_qkv,
                                const std::uint16_t* conv_weight,
                                std::uint16_t* conv_state,
                                std::uint16_t* activated_qkv,
                                std::uint32_t tokens,
                                void* stream,
                                int device);

void cuda_gdn_recurrent_prefill_bf16(
    const std::uint16_t* activated_qkv,
    const std::uint16_t* projected_b,
    const std::uint16_t* projected_a,
    const std::uint16_t* a_log,
    const std::uint16_t* dt_bias,
    float* recurrent_state,
    std::uint16_t* core_output,
    std::uint32_t tokens,
    void* stream,
    int device);

void cuda_gdn_output_norm_prefill_bf16(
    const std::uint16_t* core_output,
    const std::uint16_t* gate_z,
    const std::uint16_t* norm_weight,
    std::uint16_t* output,
    std::uint32_t tokens,
    float epsilon,
    void* stream,
    int device);

bool cuda_q38_gdn_compiled();

}  // namespace q38

#endif
