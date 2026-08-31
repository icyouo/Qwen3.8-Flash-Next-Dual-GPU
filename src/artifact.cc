#include "q38/artifact.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace q38 {

namespace {

std::string trim(std::string value) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
                value.end());
    return value;
}

std::uint32_t parse_u32(const std::string& key, const std::string& value) {
    std::size_t used = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &used, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("manifest key " + key + " is not an integer");
    }
    if (used != value.size() || parsed > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("manifest key " + key + " is outside uint32");
    return static_cast<std::uint32_t>(parsed);
}

DType parse_dtype(const std::string& key, const std::string& value) {
    static const std::unordered_map<std::string, DType> types{
        {"bf16", DType::kBFloat16}, {"f32", DType::kFloat32},
        {"f16", DType::kFloat16},   {"int8", DType::kInt8},
        {"fp8_e4m3", DType::kFp8E4M3},
    };
    const auto found = types.find(value);
    if (found == types.end())
        throw std::runtime_error("manifest key " + key + " has unknown dtype");
    return found->second;
}

std::vector<std::uint32_t> parse_u32_list(const std::string& key,
                                          const std::string& value) {
    std::vector<std::uint32_t> result;
    std::istringstream input(value);
    std::string item;
    while (std::getline(input, item, ',')) {
        item = trim(item);
        if (item.empty())
            throw std::runtime_error("manifest key " + key + " has an empty item");
        result.push_back(parse_u32(key, item));
    }
    if (result.empty())
        throw std::runtime_error("manifest key " + key + " is empty");
    return result;
}

bool is_hex_sha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

}  // namespace

ModelArtifactManifestV1 parse_artifact_manifest(std::istream& input) {
    std::unordered_map<std::string, std::string> values;
    std::string line;
    std::uint32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos)
            throw std::runtime_error("manifest line " +
                                     std::to_string(line_number) +
                                     " has no '='");
        auto key = trim(line.substr(0, equals));
        auto value = trim(line.substr(equals + 1));
        if (key.empty() || value.empty())
            throw std::runtime_error("manifest line " +
                                     std::to_string(line_number) +
                                     " has an empty key or value");
        if (!values.emplace(key, value).second)
            throw std::runtime_error("manifest has duplicate key " + key);
    }

    const std::set<std::string> required{
        "schema_version",       "model_id",          "artifact_sha256",
        "tokenizer_sha256",     "tensor_layout_version",
        "stage_plan_id",        "context_limit",     "vocab_size",
        "model_hidden_width",   "boundary_hidden_width",
        "layer_count",          "stage0_first",      "stage0_count",
        "stage1_first",         "stage1_count",      "ple_layer",
        "qsa_layers",           "qsa_main_dtype",    "ple_storage_dtype",
        "stage0_weights",       "stage1_weights",    "mtp_weights",
        "ple_manifest",
    };
    for (const auto& key : required) {
        if (values.find(key) == values.end())
            throw std::runtime_error("manifest is missing key " + key);
    }
    for (const auto& item : values) {
        if (required.find(item.first) == required.end())
            throw std::runtime_error("manifest has unknown key " + item.first);
    }

    ModelArtifactManifestV1 manifest;
    manifest.schema_version = parse_u32("schema_version", values.at("schema_version"));
    manifest.model_id = values.at("model_id");
    manifest.artifact_sha256 = values.at("artifact_sha256");
    manifest.tokenizer_sha256 = values.at("tokenizer_sha256");
    manifest.tensor_layout_version = values.at("tensor_layout_version");
    manifest.stage_plan_id = values.at("stage_plan_id");
    manifest.context_limit = parse_u32("context_limit", values.at("context_limit"));
    manifest.vocab_size = parse_u32("vocab_size", values.at("vocab_size"));
    manifest.model_hidden_width =
        parse_u32("model_hidden_width", values.at("model_hidden_width"));
    manifest.boundary_hidden_width =
        parse_u32("boundary_hidden_width", values.at("boundary_hidden_width"));
    manifest.layer_count = parse_u32("layer_count", values.at("layer_count"));
    manifest.stage0.first = parse_u32("stage0_first", values.at("stage0_first"));
    manifest.stage0.count = parse_u32("stage0_count", values.at("stage0_count"));
    manifest.stage1.first = parse_u32("stage1_first", values.at("stage1_first"));
    manifest.stage1.count = parse_u32("stage1_count", values.at("stage1_count"));
    manifest.ple_layer = parse_u32("ple_layer", values.at("ple_layer"));
    manifest.qsa_layers = parse_u32_list("qsa_layers", values.at("qsa_layers"));
    manifest.qsa_main_dtype =
        parse_dtype("qsa_main_dtype", values.at("qsa_main_dtype"));
    manifest.ple_storage_dtype =
        parse_dtype("ple_storage_dtype", values.at("ple_storage_dtype"));
    manifest.stage0_weights = values.at("stage0_weights");
    manifest.stage1_weights = values.at("stage1_weights");
    manifest.mtp_weights = values.at("mtp_weights");
    manifest.ple_manifest = values.at("ple_manifest");
    validate_artifact_manifest(manifest);
    return manifest;
}

ModelArtifactManifestV1 load_artifact_manifest(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open artifact manifest: " + path);
    return parse_artifact_manifest(input);
}

void validate_artifact_manifest(const ModelArtifactManifestV1& manifest) {
    if (manifest.schema_version != 1)
        throw std::runtime_error("unsupported artifact manifest version");
    if (manifest.model_id.empty() || manifest.tensor_layout_version.empty() ||
        manifest.stage_plan_id.empty())
        throw std::runtime_error("artifact identity fields must be nonempty");
    if (!is_hex_sha256(manifest.artifact_sha256) ||
        !is_hex_sha256(manifest.tokenizer_sha256))
        throw std::runtime_error("artifact hashes must be 64 hexadecimal characters");
    if (manifest.context_limit == 0 || manifest.vocab_size == 0 ||
        manifest.model_hidden_width == 0 || manifest.boundary_hidden_width == 0 ||
        manifest.layer_count == 0)
        throw std::runtime_error("artifact dimensions must be nonzero");
    if (manifest.boundary_hidden_width != 4u * manifest.model_hidden_width)
        throw std::runtime_error("boundary width must materialize four HC streams");
    if (manifest.stage0.first != 0 || manifest.stage0.count == 0 ||
        manifest.stage1.count == 0 ||
        manifest.stage1.first != manifest.stage0.count ||
        static_cast<std::uint64_t>(manifest.stage0.count) +
                manifest.stage1.count !=
            manifest.layer_count)
        throw std::runtime_error("stage ranges must be contiguous and exhaustive");
    if (manifest.ple_layer >= manifest.stage0.count)
        throw std::runtime_error("PLE consumer must be owned by stage0");
    std::set<std::uint32_t> qsa;
    for (const auto layer : manifest.qsa_layers) {
        if (layer >= manifest.layer_count || !qsa.insert(layer).second)
            throw std::runtime_error("QSA layer list is invalid or duplicated");
    }
    if (manifest.qsa_main_dtype != DType::kBFloat16)
        throw std::runtime_error("v1 requires BF16 main QSA K/V");
    if (manifest.ple_storage_dtype != DType::kBFloat16 &&
        manifest.ple_storage_dtype != DType::kFp8E4M3)
        throw std::runtime_error("v1 PLE storage must be BF16 or FP8 E4M3");
    if (manifest.stage0_weights.empty() || manifest.stage1_weights.empty() ||
        manifest.mtp_weights.empty() || manifest.ple_manifest.empty())
        throw std::runtime_error("artifact component paths must be nonempty");
}

}  // namespace q38

