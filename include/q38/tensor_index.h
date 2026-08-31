#ifndef Q38_TENSOR_INDEX_H
#define Q38_TENSOR_INDEX_H

#include <cstdint>
#include <string>
#include <vector>

namespace q38 {

struct IndexedTensorV1 {
    std::string name;
    std::string type_name;
    std::uint64_t payload_bytes = 0;
    std::uint64_t absolute_offset = 0;
    std::string shard_path;
    std::vector<std::uint64_t> dims;
};

struct TensorIndexV1 {
    std::uint32_t stage = 0;
    std::uint32_t cut = 0;
    std::string layout_sha256;
    std::vector<IndexedTensorV1> tensors;
    std::uint64_t payload_bytes = 0;
};

TensorIndexV1 load_tensor_index(const std::string& path,
                                bool verify_source_extents = true);
void validate_tensor_index_pair(const TensorIndexV1& stage0,
                                const TensorIndexV1& stage1);

}  // namespace q38

#endif

