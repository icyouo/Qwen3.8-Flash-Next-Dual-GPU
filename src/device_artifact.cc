#include "q38/device_artifact.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace q38 {

namespace {

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> result;
    std::istringstream input(value);
    std::string item;
    while (std::getline(input, item, separator)) result.push_back(item);
    return result;
}

std::uint64_t parse_u64(const std::string& field, const std::string& value) {
    std::size_t used = 0;
    unsigned long long result = 0;
    try {
        result = std::stoull(value, &used, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("device index " + field + " is not an integer");
    }
    if (used != value.size())
        throw std::runtime_error("device index " + field + " has trailing bytes");
    return result;
}

bool is_hex(const std::string& value, std::size_t length) {
    return value.size() == length &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

DeviceWeightFormatV1 parse_format(const std::string& value) {
    if (value == "preserve") return DeviceWeightFormatV1::kPreserve;
    if (value == "w4a16_sym_g128")
        return DeviceWeightFormatV1::kW4A16SymG128;
    if (value == "w8a16_sym_g128")
        return DeviceWeightFormatV1::kW8A16SymG128;
    if (value == "fp8_e4m3fn") return DeviceWeightFormatV1::kFp8E4M3Fn;
    throw std::runtime_error("device index has unknown format " + value);
}

std::uint64_t checked_elements(const std::vector<std::uint64_t>& shape) {
    if (shape.empty()) throw std::runtime_error("device tensor shape is empty");
    std::uint64_t result = 1;
    for (const auto dimension : shape) {
        if (dimension == 0 || result > std::numeric_limits<std::uint64_t>::max() /
                                           dimension)
            throw std::runtime_error("device tensor shape overflows");
        result *= dimension;
    }
    return result;
}

std::uint64_t dtype_bytes(const std::string& dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "I64") return 8;
    if (dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3") return 1;
    throw std::runtime_error("device tensor has unknown source dtype " + dtype);
}

void validate_tensor_layout(const DeviceTensorV1& tensor) {
    const auto elements = checked_elements(tensor.shape);
    if (!is_hex(tensor.data_sha256, 64))
        throw std::runtime_error("device tensor data hash is invalid");
    if (tensor.data_bytes == 0)
        throw std::runtime_error("device tensor data extent is empty");
    switch (tensor.format) {
        case DeviceWeightFormatV1::kPreserve:
            if (tensor.group_size != 0 || tensor.scale_bytes != 0 ||
                tensor.data_bytes != elements * dtype_bytes(tensor.source_dtype))
                throw std::runtime_error("preserved device tensor layout is invalid");
            break;
        case DeviceWeightFormatV1::kW4A16SymG128:
        case DeviceWeightFormatV1::kW8A16SymG128: {
            if (tensor.group_size != 128 || tensor.shape.back() % 128 != 0 ||
                !is_hex(tensor.scale_sha256, 64))
                throw std::runtime_error("groupwise device tensor layout is invalid");
            const auto bits = tensor.format == DeviceWeightFormatV1::kW4A16SymG128
                                  ? 4u
                                  : 8u;
            if (elements > std::numeric_limits<std::uint64_t>::max() / bits ||
                tensor.data_bytes != elements * bits / 8u ||
                tensor.scale_bytes != elements / 128u * 2u)
                throw std::runtime_error("groupwise device tensor byte census is invalid");
            break;
        }
        case DeviceWeightFormatV1::kFp8E4M3Fn:
            if (tensor.group_size != 0 || tensor.shape.size() < 2 ||
                !is_hex(tensor.scale_sha256, 64) ||
                tensor.data_bytes != elements ||
                tensor.scale_bytes !=
                    (elements / tensor.shape.back()) * sizeof(std::uint16_t))
                throw std::runtime_error("FP8 device tensor layout is invalid");
            break;
        case DeviceWeightFormatV1::kInvalid:
            throw std::runtime_error("device tensor format is invalid");
    }
}

std::string system_error(const std::string& operation,
                         const std::string& path) {
    return operation + " " + path + ": " + std::strerror(errno);
}

struct Mapping {
    int fd = -1;
    std::size_t size = 0;
    const std::byte* data = nullptr;

    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& other) noexcept
        : fd(std::exchange(other.fd, -1)),
          size(std::exchange(other.size, 0)),
          data(std::exchange(other.data, nullptr)) {}
    Mapping& operator=(Mapping&& other) noexcept {
        if (this == &other) return *this;
        reset();
        fd = std::exchange(other.fd, -1);
        size = std::exchange(other.size, 0);
        data = std::exchange(other.data, nullptr);
        return *this;
    }
    ~Mapping() { reset(); }
    void reset() noexcept {
        if (data) ::munmap(const_cast<std::byte*>(data), size);
        if (fd >= 0) ::close(fd);
        fd = -1;
        size = 0;
        data = nullptr;
    }
};

Mapping map_segment(const std::string& path, std::uint64_t expected_bytes) {
    Mapping result;
    result.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (result.fd < 0) throw std::runtime_error(system_error("open", path));
    struct stat info {};
    if (::fstat(result.fd, &info) != 0)
        throw std::runtime_error(system_error("fstat", path));
    if (info.st_size < 0 || static_cast<std::uint64_t>(info.st_size) != expected_bytes)
        throw std::runtime_error("device segment size changed: " + path);
    result.size = static_cast<std::size_t>(info.st_size);
    void* address = ::mmap(nullptr, result.size, PROT_READ, MAP_PRIVATE,
                           result.fd, 0);
    if (address == MAP_FAILED)
        throw std::runtime_error(system_error("mmap", path));
    result.data = static_cast<const std::byte*>(address);
#ifdef MADV_DONTDUMP
    (void)::madvise(address, result.size, MADV_DONTDUMP);
#endif
    return result;
}

}  // namespace

DeviceStageIndexV1 load_device_stage_index(const std::string& path,
                                            bool verify_segment_extents) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open device stage index: " + path);
    std::string line;
    if (!std::getline(input, line) || line != "Q38_DEVICE_INDEX_V1")
        throw std::runtime_error("device stage index has invalid magic/version");

    DeviceStageIndexV1 index;
    index.index_directory =
        std::filesystem::absolute(std::filesystem::path(path)).parent_path().string();
    std::set<std::string> headers;
    std::set<std::string> names;
    std::uint32_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') continue;
        const auto equals = line.find('=');
        if (equals != std::string::npos && line.find('\t') == std::string::npos) {
            const auto key = line.substr(0, equals);
            const auto value = line.substr(equals + 1);
            if (!headers.insert(key).second)
                throw std::runtime_error("duplicate device index header " + key);
            if (key == "stage") {
                const auto parsed = parse_u64(key, value);
                if (parsed > 1) throw std::runtime_error("device stage must be 0 or 1");
                index.stage = static_cast<std::uint32_t>(parsed);
            } else if (key == "cut") {
                const auto parsed = parse_u64(key, value);
                if (parsed == 0 || parsed >= 48)
                    throw std::runtime_error("device cut must be in 1..47");
                index.cut = static_cast<std::uint32_t>(parsed);
            } else if (key == "source_repo") {
                index.source_repo = value;
            } else if (key == "source_commit") {
                index.source_commit = value;
            } else if (key == "policy_sha256") {
                index.policy_sha256 = value;
            } else if (key == "artifact_sha256") {
                index.artifact_sha256 = value;
            } else {
                throw std::runtime_error("unknown device index header " + key);
            }
            continue;
        }
        const auto fields = split(line, '\t');
        if (!fields.empty() && fields[0] == "segment") {
            if (fields.size() != 4)
                throw std::runtime_error("invalid device segment record at line " +
                                         std::to_string(line_number));
            DeviceSegmentV1 segment;
            segment.relative_path = fields[1];
            segment.bytes = parse_u64("segment bytes", fields[2]);
            segment.sha256 = fields[3];
            if (segment.relative_path.empty() || segment.bytes == 0 ||
                !is_hex(segment.sha256, 64))
                throw std::runtime_error("invalid device segment fields");
            if (verify_segment_extents) {
                const auto resolved = std::filesystem::path(index.index_directory) /
                                      segment.relative_path;
                std::error_code error;
                const auto bytes = std::filesystem::file_size(resolved, error);
                if (error || bytes != segment.bytes)
                    throw std::runtime_error("device segment extent differs: " +
                                             resolved.string());
            }
            if (index.mapped_bytes > std::numeric_limits<std::uint64_t>::max() -
                                         segment.bytes)
                throw std::runtime_error("device segment byte census overflows");
            index.mapped_bytes += segment.bytes;
            index.segments.push_back(std::move(segment));
            continue;
        }
        if (fields.size() != 14 || fields[0] != "tensor")
            throw std::runtime_error("invalid device tensor record at line " +
                                     std::to_string(line_number));
        DeviceTensorV1 tensor;
        tensor.name = fields[1];
        tensor.source_name = fields[2];
        tensor.source_dtype = fields[3];
        tensor.format = parse_format(fields[4]);
        tensor.group_size = static_cast<std::uint32_t>(
            parse_u64("group size", fields[5]));
        tensor.segment = static_cast<std::uint32_t>(
            parse_u64("segment index", fields[6]));
        tensor.data_offset = parse_u64("data offset", fields[7]);
        tensor.data_bytes = parse_u64("data bytes", fields[8]);
        tensor.data_sha256 = fields[9];
        tensor.scale_offset = parse_u64("scale offset", fields[10]);
        tensor.scale_bytes = parse_u64("scale bytes", fields[11]);
        tensor.scale_sha256 = fields[12] == "-" ? "" : fields[12];
        for (const auto& dimension : split(fields[13], ','))
            tensor.shape.push_back(parse_u64("shape", dimension));
        if (tensor.name.empty() || tensor.source_name.empty() ||
            !names.insert(tensor.name).second ||
            tensor.segment >= index.segments.size())
            throw std::runtime_error("invalid or duplicate device tensor identity");
        const auto segment_bytes = index.segments[tensor.segment].bytes;
        if (tensor.data_offset > segment_bytes ||
            tensor.data_bytes > segment_bytes - tensor.data_offset ||
            tensor.scale_offset > segment_bytes ||
            tensor.scale_bytes > segment_bytes - tensor.scale_offset)
            throw std::runtime_error("device tensor exceeds its segment");
        validate_tensor_layout(tensor);
        index.tensors.push_back(std::move(tensor));
    }

    const std::set<std::string> required_headers{
        "stage", "cut", "source_repo", "source_commit", "policy_sha256",
        "artifact_sha256"};
    if (headers != required_headers || index.source_repo.empty() ||
        !is_hex(index.source_commit, 40) || !is_hex(index.policy_sha256, 64) ||
        !is_hex(index.artifact_sha256, 64) || index.segments.empty() ||
        index.tensors.empty())
        throw std::runtime_error("device stage index is incomplete");
    return index;
}

void validate_device_stage_pair(const DeviceStageIndexV1& stage0,
                                const DeviceStageIndexV1& stage1) {
    if (stage0.stage != 0 || stage1.stage != 1 || stage0.cut != stage1.cut ||
        stage0.source_repo != stage1.source_repo ||
        stage0.source_commit != stage1.source_commit ||
        stage0.policy_sha256 != stage1.policy_sha256)
        throw std::runtime_error("device stage pair identity differs");
    std::unordered_map<std::string, const DeviceTensorV1*> stage0_tensors;
    for (const auto& tensor : stage0.tensors)
        stage0_tensors.emplace(tensor.name, &tensor);
    for (const auto& tensor : stage1.tensors) {
        const auto found = stage0_tensors.find(tensor.name);
        if (found == stage0_tensors.end()) continue;
        const auto& first = *found->second;
        const bool identical_replica =
            tensor.name == "token_embd.weight" &&
            tensor.source_name == first.source_name &&
            tensor.source_dtype == first.source_dtype &&
            tensor.format == first.format &&
            tensor.group_size == first.group_size &&
            tensor.data_bytes == first.data_bytes &&
            tensor.data_sha256 == first.data_sha256 &&
            tensor.scale_bytes == first.scale_bytes &&
            tensor.scale_sha256 == first.scale_sha256 &&
            tensor.shape == first.shape;
        if (!identical_replica)
            throw std::runtime_error("device tensor appears in both stages: " +
                                     tensor.name);
    }
}

struct DeviceWeightStore::Impl {
    explicit Impl(DeviceStageIndexV1 source) : index(std::move(source)) {}
    DeviceStageIndexV1 index;
    std::vector<Mapping> mappings;
    std::unordered_map<std::string, std::size_t> tensor_by_name;
};

bool DeviceTensorViewV1::empty() const { return descriptor == nullptr; }
bool DeviceSegmentViewV1::empty() const { return descriptor == nullptr; }

DeviceWeightStore::DeviceWeightStore(DeviceStageIndexV1 index)
    : impl_(std::make_unique<Impl>(std::move(index))) {
    impl_->mappings.reserve(impl_->index.segments.size());
    for (const auto& segment : impl_->index.segments) {
        const auto resolved = std::filesystem::path(impl_->index.index_directory) /
                              segment.relative_path;
        impl_->mappings.push_back(map_segment(resolved.string(), segment.bytes));
    }
    for (std::size_t i = 0; i < impl_->index.tensors.size(); ++i) {
        if (!impl_->tensor_by_name.emplace(impl_->index.tensors[i].name, i).second)
            throw std::runtime_error("duplicate device tensor in mapped store");
    }
}

DeviceWeightStore::~DeviceWeightStore() = default;
DeviceWeightStore::DeviceWeightStore(DeviceWeightStore&&) noexcept = default;
DeviceWeightStore& DeviceWeightStore::operator=(DeviceWeightStore&&) noexcept =
    default;

const DeviceStageIndexV1& DeviceWeightStore::index() const {
    return impl_->index;
}

DeviceSegmentViewV1 DeviceWeightStore::segment(std::size_t index) const {
    if (index >= impl_->mappings.size())
        throw std::out_of_range("device segment index is out of range");
    return DeviceSegmentViewV1{&impl_->index.segments[index],
                               impl_->mappings[index].data};
}

DeviceTensorViewV1 DeviceWeightStore::find(const std::string& name) const {
    const auto found = impl_->tensor_by_name.find(name);
    if (found == impl_->tensor_by_name.end()) return {};
    const auto& tensor = impl_->index.tensors[found->second];
    const auto& mapping = impl_->mappings[tensor.segment];
    return DeviceTensorViewV1{
        &tensor,
        mapping.data + tensor.data_offset,
        tensor.scale_bytes == 0 ? nullptr : mapping.data + tensor.scale_offset,
    };
}

DeviceTensorViewV1 DeviceWeightStore::require(const std::string& name) const {
    auto result = find(name);
    if (result.empty()) throw std::runtime_error("missing device tensor: " + name);
    return result;
}

}  // namespace q38
