#ifndef Q38_CUDA_QSA_H
#define Q38_CUDA_QSA_H

#include <cstddef>
#include <cstdint>
#include <memory>

namespace q38 {

constexpr std::uint32_t kQ38QsaHeads = 24;
constexpr std::uint32_t kQ38QsaKvHeads = 2;
constexpr std::uint32_t kQ38QsaHeadWidth = 256;
constexpr std::uint32_t kQ38QsaIndexerHeads = 4;
constexpr std::uint32_t kQ38QsaIndexerWidth = 128;
constexpr std::uint32_t kQ38QsaBlockTokens = 4;
constexpr std::uint32_t kQ38QsaTokenBudget = 2048;
constexpr std::uint32_t kQ38QsaBlockBudget = 512;
constexpr std::uint32_t kQ38ContextLimit = 262144;
constexpr std::uint32_t kQ38QsaMaximumSelected = 2051;

struct CudaQsaLayerStateView {
    std::uint16_t* main_keys = nullptr;
    std::uint16_t* main_values = nullptr;
    std::uint16_t* raw_index_keys = nullptr;
    std::uint16_t* pooled_index_keys = nullptr;
};

class CudaQsaStateBank {
public:
    CudaQsaStateBank(int device, std::uint32_t layers,
                     std::uint32_t capacity = kQ38ContextLimit);
    ~CudaQsaStateBank();

    CudaQsaStateBank(const CudaQsaStateBank&) = delete;
    CudaQsaStateBank& operator=(const CudaQsaStateBank&) = delete;
    CudaQsaStateBank(CudaQsaStateBank&&) noexcept;
    CudaQsaStateBank& operator=(CudaQsaStateBank&&) noexcept;

    void begin(std::uint64_t epoch);
    CudaQsaLayerStateView working(std::uint32_t local_layer) const;
    void mark_evaluated(std::uint64_t epoch, std::uint32_t tokens);
    void commit(std::uint64_t epoch, std::uint32_t accepted_tokens);
    void rollback(std::uint64_t epoch);
    void reset();

    std::uint64_t committed_tokens() const;
    std::uint64_t transaction_base() const;
    std::uint32_t capacity() const;
    std::uint64_t allocated_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// projected_q_gate is [24, 512], projected_k/v are [2, 256]. Output query
// and gate are each [24, 256]; normalized/rotated keys and values are appended
// to the layer cache at position.
void cuda_qsa_prepare_main_decode_bf16(
    const std::uint16_t* projected_q_gate,
    const std::uint16_t* projected_k,
    const std::uint16_t* projected_v,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight,
    std::uint16_t* query,
    std::uint16_t* gate,
    CudaQsaLayerStateView cache,
    std::uint32_t position,
    void* stream,
    int device);

// projected_index_qk is [4*128 + 128]. Raw K is cached before pooling, as in
// the official implementation. pooled_index_keys is updated every fourth token.
void cuda_qsa_prepare_index_decode_bf16(
    const std::uint16_t* projected_index_qk,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight,
    std::uint16_t* index_query,
    CudaQsaLayerStateView cache,
    std::uint32_t position,
    void* stream,
    int device);

// scratch_block_scores requires ceil((position+1)/4) floats. selected_indices
// requires kQ38QsaMaximumSelected int32 values. Returns selected count, which is
// known from position and does not synchronize the stream.
std::uint32_t cuda_qsa_select_decode(
    const std::uint16_t* index_query,
    CudaQsaLayerStateView cache,
    std::uint32_t position,
    float* scratch_block_scores,
    std::int32_t* selected_indices,
    void* stream,
    int device);

// scratch_attention_scores requires 24 * selected_count floats.
void cuda_qsa_attention_decode_bf16(
    const std::uint16_t* query,
    CudaQsaLayerStateView cache,
    const std::int32_t* selected_indices,
    std::uint32_t selected_count,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device);

// Tile-prefill path. Scratch score stride is ceil(context capacity / 4),
// selected stride is kQ38QsaMaximumSelected, and attention stride is
// kQ38QsaHeads*kQ38QsaMaximumSelected per token.
void cuda_qsa_prepare_prefill_bf16(
    const std::uint16_t* projected_q_gate,
    const std::uint16_t* projected_k,
    const std::uint16_t* projected_v,
    const std::uint16_t* projected_index_qk,
    const std::uint16_t* q_norm_weight,
    const std::uint16_t* k_norm_weight,
    const std::uint16_t* index_q_norm_weight,
    const std::uint16_t* index_k_norm_weight,
    std::uint16_t* query,
    std::uint16_t* gate,
    std::uint16_t* index_query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    void* stream,
    int device);

void cuda_qsa_attention_prefill_bf16(
    const std::uint16_t* query,
    const std::uint16_t* index_query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    float* scratch_block_scores,
    std::uint32_t block_score_stride,
    std::int32_t* selected_indices,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device);

// Fine-grained variants used by the opt-in prefill profiler.  Keeping the
// selector and sparse-attention launches separate also gives architecture
// experiments an explicit boundary without changing the default API.
void cuda_qsa_select_prefill(
    const std::uint16_t* index_query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    float* scratch_block_scores,
    std::uint32_t block_score_stride,
    std::int32_t* selected_indices,
    void* stream,
    int device);

void cuda_qsa_apply_prefill_bf16(
    const std::uint16_t* query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    const std::int32_t* selected_indices,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device);

void cuda_qsa_score_prefill_bf16(
    const std::uint16_t* query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    const std::int32_t* selected_indices,
    float* scratch_attention_scores,
    void* stream,
    int device);

void cuda_qsa_output_prefill_bf16(
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    const std::int32_t* selected_indices,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device);

// Grouped-query prefill path.  The twelve query heads attached to one KV head
// share explicit K/V tiles instead of relying on inter-CTA cache reuse.
void cuda_qsa_apply_grouped_prefill_bf16(
    const std::uint16_t* query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    const std::int32_t* selected_indices,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device);

// Experimental grouped fusion used only for A/B diagnostics.  The production
// default remains the faster split grouped path.
void cuda_qsa_apply_grouped_fused_prefill_bf16(
    const std::uint16_t* query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    const std::int32_t* selected_indices,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device);

void cuda_qsa_score_grouped_prefill_bf16(
    const std::uint16_t* query,
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    const std::int32_t* selected_indices,
    float* scratch_attention_scores,
    void* stream,
    int device);

void cuda_qsa_output_grouped_prefill_bf16(
    CudaQsaLayerStateView cache,
    std::uint32_t first_position,
    std::uint32_t tokens,
    const std::int32_t* selected_indices,
    float* scratch_attention_scores,
    std::uint16_t* output,
    void* stream,
    int device);

bool cuda_q38_qsa_compiled();

}  // namespace q38

#endif
