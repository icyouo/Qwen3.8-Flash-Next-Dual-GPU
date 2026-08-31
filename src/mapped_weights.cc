#include "q38/mapped_weights.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace q38 {

namespace {

std::string system_error(const std::string& operation,
                         const std::string& path) {
    return operation + " " + path + ": " + std::strerror(errno);
}

struct Mapping {
    std::string path;
    int fd = -1;
    std::size_t size = 0;
    const std::byte* data = nullptr;

    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    Mapping(Mapping&& other) noexcept
        : path(std::move(other.path)),
          fd(std::exchange(other.fd, -1)),
          size(std::exchange(other.size, 0)),
          data(std::exchange(other.data, nullptr)) {}

    Mapping& operator=(Mapping&& other) noexcept {
        if (this == &other) return *this;
        reset();
        path = std::move(other.path);
        fd = std::exchange(other.fd, -1);
        size = std::exchange(other.size, 0);
        data = std::exchange(other.data, nullptr);
        return *this;
    }

    ~Mapping() { reset(); }

    void reset() noexcept {
        if (data != nullptr) ::munmap(const_cast<std::byte*>(data), size);
        if (fd >= 0) ::close(fd);
        data = nullptr;
        size = 0;
        fd = -1;
    }
};

Mapping map_read_only(const std::string& path) {
    Mapping result;
    result.path = path;
    result.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (result.fd < 0) throw std::runtime_error(system_error("open", path));

    struct stat info {};
    if (::fstat(result.fd, &info) != 0)
        throw std::runtime_error(system_error("fstat", path));
    if (info.st_size <= 0)
        throw std::runtime_error("weight shard is empty: " + path);
    result.size = static_cast<std::size_t>(info.st_size);

    void* mapped = ::mmap(nullptr, result.size, PROT_READ, MAP_PRIVATE,
                          result.fd, 0);
    if (mapped == MAP_FAILED)
        throw std::runtime_error(system_error("mmap", path));
    result.data = static_cast<const std::byte*>(mapped);
#ifdef MADV_DONTDUMP
    // Model weights are immutable artifacts and should not inflate a core dump.
    (void)::madvise(mapped, result.size, MADV_DONTDUMP);
#endif
    return result;
}

}  // namespace

struct MappedWeightStore::Impl {
    explicit Impl(TensorIndexV1 source_index)
        : index(std::move(source_index)) {}

    TensorIndexV1 index;
    std::vector<Mapping> mappings;
    std::unordered_map<std::string, std::size_t> mapping_by_path;
    std::unordered_map<std::string, std::size_t> tensor_by_name;
};

std::size_t TensorViewV1::size() const {
    return descriptor == nullptr
               ? 0
               : static_cast<std::size_t>(descriptor->payload_bytes);
}

bool TensorViewV1::empty() const { return descriptor == nullptr; }

MappedWeightStore::MappedWeightStore(TensorIndexV1 index)
    : impl_(std::make_unique<Impl>(std::move(index))) {
    for (std::size_t i = 0; i < impl_->index.tensors.size(); ++i) {
        const auto& tensor = impl_->index.tensors[i];
        if (!impl_->tensor_by_name.emplace(tensor.name, i).second)
            throw std::runtime_error("duplicate tensor in mapped store: " +
                                     tensor.name);
        if (impl_->mapping_by_path.find(tensor.shard_path) ==
            impl_->mapping_by_path.end()) {
            const auto mapping_index = impl_->mappings.size();
            impl_->mappings.push_back(map_read_only(tensor.shard_path));
            impl_->mapping_by_path.emplace(tensor.shard_path, mapping_index);
        }
        const auto& mapping = impl_->mappings.at(
            impl_->mapping_by_path.at(tensor.shard_path));
        if (tensor.absolute_offset > mapping.size ||
            tensor.payload_bytes > mapping.size - tensor.absolute_offset)
            throw std::runtime_error("tensor exceeds mapped shard: " +
                                     tensor.name);
    }
}

MappedWeightStore::~MappedWeightStore() = default;
MappedWeightStore::MappedWeightStore(MappedWeightStore&&) noexcept = default;
MappedWeightStore& MappedWeightStore::operator=(MappedWeightStore&&) noexcept =
    default;

const TensorIndexV1& MappedWeightStore::index() const { return impl_->index; }

TensorViewV1 MappedWeightStore::find(const std::string& name) const {
    const auto found = impl_->tensor_by_name.find(name);
    if (found == impl_->tensor_by_name.end()) return {};
    const auto& tensor = impl_->index.tensors.at(found->second);
    const auto& mapping = impl_->mappings.at(
        impl_->mapping_by_path.at(tensor.shard_path));
    return TensorViewV1{&tensor, mapping.data + tensor.absolute_offset};
}

TensorViewV1 MappedWeightStore::require(const std::string& name) const {
    auto result = find(name);
    if (result.empty()) throw std::runtime_error("missing tensor: " + name);
    return result;
}

std::vector<std::string> MappedWeightStore::shard_paths() const {
    std::vector<std::string> result;
    result.reserve(impl_->mappings.size());
    for (const auto& mapping : impl_->mappings) result.push_back(mapping.path);
    return result;
}

}  // namespace q38
