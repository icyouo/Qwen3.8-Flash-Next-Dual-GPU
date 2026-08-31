#ifndef Q38_MAPPED_WEIGHTS_H
#define Q38_MAPPED_WEIGHTS_H

#include "q38/tensor_index.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace q38 {

struct TensorViewV1 {
    const IndexedTensorV1* descriptor = nullptr;
    const std::byte* data = nullptr;

    std::size_t size() const;
    bool empty() const;
};

// Maps each GGUF shard once and exposes stable, read-only tensor payload views.
// mmap is intentionally lazy: constructing a store does not read tensor pages
// into RAM.  The CUDA loader can therefore copy one tensor at a time through a
// bounded pinned staging buffer without ever materializing a second model copy.
class MappedWeightStore {
public:
    explicit MappedWeightStore(TensorIndexV1 index);
    ~MappedWeightStore();

    MappedWeightStore(const MappedWeightStore&) = delete;
    MappedWeightStore& operator=(const MappedWeightStore&) = delete;
    MappedWeightStore(MappedWeightStore&&) noexcept;
    MappedWeightStore& operator=(MappedWeightStore&&) noexcept;

    const TensorIndexV1& index() const;
    TensorViewV1 find(const std::string& name) const;
    TensorViewV1 require(const std::string& name) const;
    std::vector<std::string> shard_paths() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace q38

#endif
