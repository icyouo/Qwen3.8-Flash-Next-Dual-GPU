#ifndef Q38_ARTIFACT_H
#define Q38_ARTIFACT_H

#include "q38/contracts.h"

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace q38 {

struct LayerRangeV1 {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
};

struct ModelArtifactManifestV1 {
    std::uint32_t schema_version = 0;
    std::string model_id;
    std::string artifact_sha256;
    std::string tokenizer_sha256;
    std::string tensor_layout_version;
    std::string stage_plan_id;
    std::uint32_t context_limit = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t model_hidden_width = 0;
    std::uint32_t boundary_hidden_width = 0;
    std::uint32_t layer_count = 0;
    LayerRangeV1 stage0{};
    LayerRangeV1 stage1{};
    std::uint32_t ple_layer = 0;
    std::vector<std::uint32_t> qsa_layers;
    DType qsa_main_dtype = DType::kInvalid;
    DType ple_storage_dtype = DType::kInvalid;
    std::string stage0_weights;
    std::string stage1_weights;
    std::string mtp_weights;
    std::string ple_manifest;
};

ModelArtifactManifestV1 parse_artifact_manifest(std::istream& input);
ModelArtifactManifestV1 load_artifact_manifest(const std::string& path);
void validate_artifact_manifest(const ModelArtifactManifestV1& manifest);

}  // namespace q38

#endif

