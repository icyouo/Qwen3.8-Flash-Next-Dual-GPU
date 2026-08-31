#include "q38/model_plan.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace q38 {

namespace {

constexpr std::uint32_t kLayers = 48;
constexpr std::uint64_t kHidden = 2560;
constexpr std::uint64_t kHcWidth = 10240;
constexpr std::uint64_t kHcLowRank = 320;
constexpr std::uint64_t kVocabulary = 248320;

enum class TypeClass {
    kF32,
    kBf16,
    kI64,
    kPlain,
    kMatrix,
};

struct Requirement {
    std::string suffix;
    TypeClass type;
    std::vector<std::uint64_t> dims;
};

bool type_matches(TypeClass expected, const std::string& actual) {
    switch (expected) {
        case TypeClass::kF32:
            return actual == "F32";
        case TypeClass::kBf16:
            return actual == "BF16";
        case TypeClass::kI64:
            return actual == "I64";
        case TypeClass::kPlain:
            return actual == "F32" || actual == "BF16";
        case TypeClass::kMatrix:
            return actual == "BF16" || actual == "Q8_0" ||
                   actual == "Q6_K" || actual == "Q5_K" ||
                   actual == "Q5_0" || actual == "Q4_K" ||
                   actual == "Q4_0" || actual == "Q3_K" ||
                   actual == "Q2_K" || actual == "IQ2_XXS";
    }
    return false;
}

const IndexedTensorV1* require_tensor(
    const std::unordered_map<std::string, const IndexedTensorV1*>& tensors,
    const std::string& name,
    TypeClass type,
    std::initializer_list<std::uint64_t> dims) {
    const auto found = tensors.find(name);
    if (found == tensors.end())
        throw std::runtime_error("Qwen3.8 stage plan is missing " + name);
    const auto* tensor = found->second;
    if (!type_matches(type, tensor->type_name))
        throw std::runtime_error("Qwen3.8 tensor " + name + " has type " +
                                 tensor->type_name);
    if (!std::equal(dims.begin(), dims.end(), tensor->dims.begin(),
                    tensor->dims.end()))
        throw std::runtime_error("Qwen3.8 tensor " + name +
                                 " has unexpected dimensions");
    return tensor;
}

void add_required(
    std::vector<const IndexedTensorV1*>* destination,
    std::set<std::string>* consumed,
    std::uint64_t* bytes,
    const std::unordered_map<std::string, const IndexedTensorV1*>& tensors,
    const std::string& name,
    TypeClass type,
    std::initializer_list<std::uint64_t> dims) {
    const auto* tensor = require_tensor(tensors, name, type, dims);
    destination->push_back(tensor);
    consumed->insert(name);
    *bytes += tensor->payload_bytes;
}

void add_hyper_connection(
    std::vector<const IndexedTensorV1*>* destination,
    std::set<std::string>* consumed,
    std::uint64_t* bytes,
    const std::unordered_map<std::string, const IndexedTensorV1*>& tensors,
    const std::string& prefix,
    bool inject) {
    add_required(destination, consumed, bytes, tensors, prefix + ".norm.weight",
                 TypeClass::kPlain, {kHcWidth});
    add_required(destination, consumed, bytes, tensors,
                 prefix + ".mix_down.weight", TypeClass::kBf16,
                 {kHcWidth, kHcLowRank});
    add_required(destination, consumed, bytes, tensors,
                 prefix + ".mix_up.weight", TypeClass::kBf16,
                 {kHcLowRank, kHcWidth});
    if (inject)
        add_required(destination, consumed, bytes, tensors,
                     prefix + ".inject.weight", TypeClass::kBf16,
                     {kHcWidth, 4});
}

void add_attention(
    Qwen38LayerPlanV1* layer,
    std::set<std::string>* consumed,
    std::uint64_t* bytes,
    const std::unordered_map<std::string, const IndexedTensorV1*>& tensors,
    const std::string& prefix) {
    auto* destination = &layer->tensors;
    if (layer->attention == Qwen38AttentionKindV1::kQwenSparseAttention) {
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_index_qk.weight", TypeClass::kMatrix,
                     {kHidden, 640});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_index_q_norm.weight", TypeClass::kPlain,
                     {128});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_index_k_norm.weight", TypeClass::kPlain,
                     {128});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_q.weight", TypeClass::kMatrix,
                     {kHidden, 12288});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_q_norm.weight", TypeClass::kPlain,
                     {256});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_k.weight", TypeClass::kMatrix,
                     {kHidden, 512});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_k_norm.weight", TypeClass::kPlain,
                     {256});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_v.weight", TypeClass::kMatrix,
                     {kHidden, 512});
        add_required(destination, consumed, bytes, tensors,
                     prefix + "attn_output.weight", TypeClass::kMatrix,
                     {6144, kHidden});
        return;
    }

    const auto linear = prefix + "linear_attn.";
    add_required(destination, consumed, bytes, tensors, linear + "a_log",
                 TypeClass::kPlain, {48});
    add_required(destination, consumed, bytes, tensors,
                 linear + "conv.weight", TypeClass::kBf16, {4, 1, 10240});
    add_required(destination, consumed, bytes, tensors, linear + "dt_bias",
                 TypeClass::kPlain, {48});
    add_required(destination, consumed, bytes, tensors,
                 linear + "in_a.weight", TypeClass::kMatrix, {kHidden, 48});
    add_required(destination, consumed, bytes, tensors,
                 linear + "in_b.weight", TypeClass::kMatrix, {kHidden, 48});
    add_required(destination, consumed, bytes, tensors,
                 linear + "qkv.weight", TypeClass::kMatrix,
                 {kHidden, 10240});
    add_required(destination, consumed, bytes, tensors,
                 linear + "z.weight", TypeClass::kMatrix, {kHidden, 6144});
    add_required(destination, consumed, bytes, tensors, linear + "norm.weight",
                 TypeClass::kPlain, {128});
    add_required(destination, consumed, bytes, tensors, linear + "out.weight",
                 TypeClass::kMatrix, {6144, kHidden});
}

void add_moe(
    Qwen38LayerPlanV1* layer,
    std::set<std::string>* consumed,
    std::uint64_t* bytes,
    const std::unordered_map<std::string, const IndexedTensorV1*>& tensors,
    const std::string& prefix) {
    auto* destination = &layer->tensors;
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_gate_inp.weight", TypeClass::kPlain,
                 {kHidden, 512});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_gate_exps.weight", TypeClass::kMatrix,
                 {kHidden, 640, 512});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_up_exps.weight", TypeClass::kMatrix,
                 {kHidden, 640, 512});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_down_exps.main.weight", TypeClass::kMatrix,
                 {512, kHidden, 512});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_down_exps.tail.weight", TypeClass::kMatrix,
                 {128, kHidden, 512});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_gate_shexp.weight", TypeClass::kMatrix,
                 {kHidden, 640});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_up_shexp.weight", TypeClass::kMatrix,
                 {kHidden, 640});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_down_shexp.weight", TypeClass::kMatrix,
                 {640, kHidden});
    add_required(destination, consumed, bytes, tensors,
                 prefix + "ffn_shexp_gate_inp.weight", TypeClass::kPlain,
                 {kHidden, 1});
}

void add_ple(
    Qwen38LayerPlanV1* layer,
    std::set<std::string>* consumed,
    std::uint64_t* bytes,
    const std::unordered_map<std::string, const IndexedTensorV1*>& tensors,
    const std::string& prefix) {
    auto* destination = &layer->tensors;
    const auto ple = prefix + "ple.";
    add_required(destination, consumed, bytes, tensors, ple + "conv.weight",
                 TypeClass::kBf16, {4, 1, kHcWidth});
    add_required(destination, consumed, bytes, tensors, ple + "key.weight",
                 TypeClass::kMatrix, {kHidden, kHcWidth});
    add_required(destination, consumed, bytes, tensors, ple + "value.weight",
                 TypeClass::kMatrix, {kHidden, kHidden});
    add_required(destination, consumed, bytes, tensors,
                 ple + "conv_norm.weight", TypeClass::kPlain, {kHcWidth});
    add_required(destination, consumed, bytes, tensors,
                 ple + "key_norm.weight", TypeClass::kPlain, {kHcWidth});
    add_required(destination, consumed, bytes, tensors,
                 ple + "query_norm.weight", TypeClass::kPlain, {kHcWidth});
    add_required(destination, consumed, bytes, tensors,
                 ple + "layer_multipliers", TypeClass::kI64, {3});
    add_required(destination, consumed, bytes, tensors,
                 ple + "head_offsets", TypeClass::kI64, {16});
    add_required(destination, consumed, bytes, tensors,
                 ple + "head_vocab_sizes", TypeClass::kI64, {16});
}

}  // namespace

bool qwen38_layer_is_qsa(std::uint32_t layer) {
    if (layer >= kLayers) throw std::out_of_range("Qwen3.8 layer is invalid");
    return layer % 4u == 3u;
}

Qwen38StagePlanV1 build_qwen38_stage_plan(const TensorIndexV1& index) {
    if (index.stage > 1 || index.cut == 0 || index.cut >= kLayers)
        throw std::runtime_error("invalid Qwen3.8 stage/cut");

    Qwen38StagePlanV1 plan;
    plan.stage = index.stage;
    plan.cut = index.cut;
    plan.first_layer = index.stage == 0 ? 0 : index.cut;
    const auto end_layer = index.stage == 0 ? index.cut : kLayers;
    plan.layer_count = end_layer - plan.first_layer;

    std::unordered_map<std::string, const IndexedTensorV1*> tensors;
    tensors.reserve(index.tensors.size());
    for (const auto& tensor : index.tensors) tensors.emplace(tensor.name, &tensor);
    std::set<std::string> consumed;

    if (index.stage == 0) {
        add_required(&plan.global_tensors, &consumed,
                     &plan.required_payload_bytes, tensors,
                     "token_embd.weight", TypeClass::kMatrix,
                     {kHidden, kVocabulary});
        add_hyper_connection(&plan.global_tensors, &consumed,
                             &plan.required_payload_bytes, tensors,
                             "hc_input", false);
    } else {
        add_required(&plan.global_tensors, &consumed,
                     &plan.required_payload_bytes, tensors,
                     "output.weight", TypeClass::kMatrix,
                     {kHidden, kVocabulary});
    }

    for (auto layer_index = plan.first_layer; layer_index < end_layer;
         ++layer_index) {
        Qwen38LayerPlanV1 layer;
        layer.layer = layer_index;
        layer.attention = qwen38_layer_is_qsa(layer_index)
                              ? Qwen38AttentionKindV1::kQwenSparseAttention
                              : Qwen38AttentionKindV1::kGatedDeltaNet;
        const auto prefix = "blk." + std::to_string(layer_index) + ".";
        add_hyper_connection(&layer.tensors, &consumed,
                             &plan.required_payload_bytes, tensors,
                             prefix + "hc_attn", true);
        add_attention(&layer, &consumed, &plan.required_payload_bytes, tensors,
                      prefix);
        if (layer_index == 1)
            add_ple(&layer, &consumed, &plan.required_payload_bytes, tensors,
                    prefix);
        add_moe(&layer, &consumed, &plan.required_payload_bytes, tensors,
                prefix);
        add_hyper_connection(&layer.tensors, &consumed,
                             &plan.required_payload_bytes, tensors,
                             prefix + "hc_ffn", true);
        plan.layers.push_back(std::move(layer));
    }

    for (const auto& tensor : index.tensors) {
        if (consumed.find(tensor.name) != consumed.end()) continue;
        plan.auxiliary_tensors.push_back(&tensor);
        plan.auxiliary_payload_bytes += tensor.payload_bytes;
    }
    if (plan.required_payload_bytes + plan.auxiliary_payload_bytes !=
        index.payload_bytes)
        throw std::runtime_error("Qwen3.8 stage byte census mismatch");
    return plan;
}

}  // namespace q38
