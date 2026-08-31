#ifndef Q38_DEVICE_ARTIFACT_H
#define Q38_DEVICE_ARTIFACT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace q38 {

enum class DeviceWeightFormatV1 : std::uint32_t {
    kInvalid = 0,
    kPreserve = 1,
    kW4A16SymG128 = 2,
    kW8A16SymG128 = 3,
    kFp8E4M3Fn = 4,
};

struct DeviceSegmentV1 {
    std::string relative_path;
    std::uint64_t bytes = 0;
    std::string sha256;
};

struct DeviceTensorV1 {
    std::string name;
    std::string source_name;
    std::string source_dtype;
    DeviceWeightFormatV1 format = DeviceWeightFormatV1::kInvalid;
    std::uint32_t group_size = 0;
    std::uint32_t segment = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t data_bytes = 0;
    std::string data_sha256;
    std::uint64_t scale_offset = 0;
    std::uint64_t scale_bytes = 0;
    std::string scale_sha256;
    std::vector<std::uint64_t> shape;
};

struct DeviceStageIndexV1 {
    std::uint32_t stage = 0;
    std::uint32_t cut = 0;
    std::string source_repo;
    std::string source_commit;
    std::string policy_sha256;
    std::string artifact_sha256;
    std::string index_directory;
    std::vector<DeviceSegmentV1> segments;
    std::vector<DeviceTensorV1> tensors;
    std::uint64_t mapped_bytes = 0;
};

struct DeviceTensorViewV1 {
    const DeviceTensorV1* descriptor = nullptr;
    const std::byte* data = nullptr;
    const std::byte* scales = nullptr;

    bool empty() const;
};

struct DeviceSegmentViewV1 {
    const DeviceSegmentV1* descriptor = nullptr;
    const std::byte* data = nullptr;

    bool empty() const;
};

DeviceStageIndexV1 load_device_stage_index(
    const std::string& path, bool verify_segment_extents = true);
void validate_device_stage_pair(const DeviceStageIndexV1& stage0,
                                const DeviceStageIndexV1& stage1);

class DeviceWeightStore {
public:
    explicit DeviceWeightStore(DeviceStageIndexV1 index);
    ~DeviceWeightStore();

    DeviceWeightStore(const DeviceWeightStore&) = delete;
    DeviceWeightStore& operator=(const DeviceWeightStore&) = delete;
    DeviceWeightStore(DeviceWeightStore&&) noexcept;
    DeviceWeightStore& operator=(DeviceWeightStore&&) noexcept;

    const DeviceStageIndexV1& index() const;
    DeviceSegmentViewV1 segment(std::size_t index) const;
    DeviceTensorViewV1 find(const std::string& name) const;
    DeviceTensorViewV1 require(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace q38

#endif
