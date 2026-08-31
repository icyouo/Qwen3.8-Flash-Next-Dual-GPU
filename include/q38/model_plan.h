#ifndef Q38_MODEL_PLAN_H
#define Q38_MODEL_PLAN_H

#include "q38/tensor_index.h"

#include <cstdint>
#include <string>
#include <vector>

namespace q38 {

enum class Qwen38AttentionKindV1 : std::uint32_t {
    kGatedDeltaNet = 0,
    kQwenSparseAttention = 1,
};

struct Qwen38LayerPlanV1 {
    std::uint32_t layer = 0;
    Qwen38AttentionKindV1 attention =
        Qwen38AttentionKindV1::kGatedDeltaNet;
    std::vector<const IndexedTensorV1*> tensors;
};

struct Qwen38StagePlanV1 {
    std::uint32_t stage = 0;
    std::uint32_t cut = 0;
    std::uint32_t first_layer = 0;
    std::uint32_t layer_count = 0;
    std::vector<const IndexedTensorV1*> global_tensors;
    std::vector<Qwen38LayerPlanV1> layers;
    std::vector<const IndexedTensorV1*> auxiliary_tensors;
    std::uint64_t required_payload_bytes = 0;
    std::uint64_t auxiliary_payload_bytes = 0;
};

bool qwen38_layer_is_qsa(std::uint32_t layer);
Qwen38StagePlanV1 build_qwen38_stage_plan(const TensorIndexV1& index);

}  // namespace q38

#endif
