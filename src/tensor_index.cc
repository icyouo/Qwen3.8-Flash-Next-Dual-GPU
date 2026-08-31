#include "q38/tensor_index.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace q38 {

namespace {

std::uint64_t parse_u64(const std::string& field, const std::string& value) {
    std::size_t used = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &used, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("tensor index " + field + " is not an integer");
    }
    if (used != value.size())
        throw std::runtime_error("tensor index " + field + " has trailing bytes");
    return parsed;
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> fields;
    std::istringstream input(value);
    std::string field;
    while (std::getline(input, field, separator)) fields.push_back(field);
    return fields;
}

bool is_sha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

}  // namespace

TensorIndexV1 load_tensor_index(const std::string& path,
                                bool verify_source_extents) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open tensor index: " + path);
    std::string line;
    if (!std::getline(input, line) || line != "Q38_TENSOR_INDEX_V1")
        throw std::runtime_error("tensor index has an invalid magic/version");

    TensorIndexV1 index;
    bool saw_stage = false;
    bool saw_cut = false;
    bool saw_layout = false;
    std::set<std::string> names;
    std::unordered_map<std::string, std::uint64_t> file_sizes;
    std::uint32_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') continue;
        if (line.rfind("stage=", 0) == 0) {
            if (saw_stage) throw std::runtime_error("duplicate tensor index stage");
            const auto value = parse_u64("stage", line.substr(6));
            if (value > 1) throw std::runtime_error("tensor index stage must be 0 or 1");
            index.stage = static_cast<std::uint32_t>(value);
            saw_stage = true;
            continue;
        }
        if (line.rfind("cut=", 0) == 0) {
            if (saw_cut) throw std::runtime_error("duplicate tensor index cut");
            const auto value = parse_u64("cut", line.substr(4));
            if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("tensor index cut is invalid");
            index.cut = static_cast<std::uint32_t>(value);
            saw_cut = true;
            continue;
        }
        if (line.rfind("layout_sha256=", 0) == 0) {
            if (saw_layout) throw std::runtime_error("duplicate tensor index layout hash");
            index.layout_sha256 = line.substr(14);
            if (!is_sha256(index.layout_sha256))
                throw std::runtime_error("tensor index layout hash is invalid");
            saw_layout = true;
            continue;
        }

        const auto fields = split(line, '\t');
        if (fields.size() != 7 || fields[0] != "tensor")
            throw std::runtime_error("tensor index line " +
                                     std::to_string(line_number) +
                                     " has an invalid record");
        IndexedTensorV1 tensor;
        tensor.name = fields[1];
        tensor.type_name = fields[2];
        tensor.payload_bytes = parse_u64("payload_bytes", fields[3]);
        tensor.absolute_offset = parse_u64("absolute_offset", fields[4]);
        tensor.shard_path = fields[5];
        for (const auto& dim : split(fields[6], ','))
            tensor.dims.push_back(parse_u64("dimension", dim));
        if (tensor.name.empty() || tensor.type_name.empty() ||
            tensor.payload_bytes == 0 || tensor.shard_path.empty() ||
            tensor.dims.empty())
            throw std::runtime_error("tensor index contains an empty tensor field");
        if (!names.insert(tensor.name).second)
            throw std::runtime_error("tensor index contains duplicate tensor " +
                                     tensor.name);
        if (index.payload_bytes > std::numeric_limits<std::uint64_t>::max() -
                                      tensor.payload_bytes)
            throw std::runtime_error("tensor index payload byte count overflows");
        index.payload_bytes += tensor.payload_bytes;

        if (verify_source_extents) {
            auto found = file_sizes.find(tensor.shard_path);
            if (found == file_sizes.end()) {
                std::error_code error;
                const auto size = std::filesystem::file_size(tensor.shard_path, error);
                if (error)
                    throw std::runtime_error("cannot stat tensor shard " +
                                             tensor.shard_path + ": " +
                                             error.message());
                found = file_sizes.emplace(tensor.shard_path, size).first;
            }
            if (tensor.absolute_offset > found->second ||
                tensor.payload_bytes > found->second - tensor.absolute_offset)
                throw std::runtime_error("tensor " + tensor.name +
                                         " exceeds its source shard");
        }
        index.tensors.push_back(std::move(tensor));
    }
    if (!saw_stage || !saw_cut || !saw_layout)
        throw std::runtime_error("tensor index is missing required headers");
    if (index.tensors.empty()) throw std::runtime_error("tensor index is empty");
    return index;
}

void validate_tensor_index_pair(const TensorIndexV1& stage0,
                                const TensorIndexV1& stage1) {
    if (stage0.stage != 0 || stage1.stage != 1)
        throw std::runtime_error("tensor index pair has reversed stage ownership");
    if (stage0.cut != stage1.cut)
        throw std::runtime_error("tensor index pair has different cuts");
    if (stage0.layout_sha256 != stage1.layout_sha256)
        throw std::runtime_error("tensor index pair has different layouts");
    std::set<std::string> names;
    for (const auto& tensor : stage0.tensors) names.insert(tensor.name);
    for (const auto& tensor : stage1.tensors) {
        if (!names.insert(tensor.name).second)
            throw std::runtime_error("tensor appears in both stages: " + tensor.name);
    }
}

}  // namespace q38

