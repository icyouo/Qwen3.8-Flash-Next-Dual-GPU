#include "q38/cuda_weights.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace q38 {

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

std::uint64_t align_up(std::uint64_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("CUDA arena alignment must be a power of two");
    const auto mask = static_cast<std::uint64_t>(alignment - 1);
    if (value > std::numeric_limits<std::uint64_t>::max() - mask)
        throw std::overflow_error("CUDA weight arena size overflows");
    return (value + mask) & ~mask;
}

struct StagingBuffer {
    std::byte* host = nullptr;
    cudaEvent_t reusable = nullptr;
};

struct TensorPointers {
    const DeviceTensorV1* descriptor = nullptr;
    const void* data = nullptr;
    const void* scales = nullptr;
};

struct TensorPlacement {
    std::size_t tensor = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t scale_offset = 0;
};

}  // namespace

struct CudaDeviceWeightStore::Impl {
    CudaDeviceWeightOptions options;
    std::uint32_t stage = 0;
    std::byte* arena = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<StagingBuffer> staging;
    DeviceStageIndexV1 index;
    std::unordered_map<std::string, TensorPointers> tensors;
    CudaDeviceWeightStats statistics;

    explicit Impl(CudaDeviceWeightOptions value) : options(value) {}
    ~Impl() { release(); }

    void release() noexcept {
        (void)cudaSetDevice(options.device);
        if (stream) (void)cudaStreamSynchronize(stream);
        for (auto& buffer : staging) {
            if (buffer.reusable) (void)cudaEventDestroy(buffer.reusable);
            if (buffer.host) (void)cudaFreeHost(buffer.host);
            buffer.reusable = nullptr;
            buffer.host = nullptr;
        }
        if (stream) (void)cudaStreamDestroy(stream);
        if (arena) (void)cudaFree(arena);
        stream = nullptr;
        arena = nullptr;
    }

    void initialize(const DeviceWeightStore& source) {
        if (options.device < 0 || options.staging_bytes == 0 ||
            options.staging_buffers < 2 || options.staging_buffers > 8)
            throw std::invalid_argument("invalid CUDA device weight options");
        check(cudaSetDevice(options.device), "cudaSetDevice(weight upload)");
        index = source.index();
        stage = index.stage;
        statistics.segment_count =
            static_cast<std::uint32_t>(index.segments.size());

        std::uint64_t arena_bytes = 0;
        std::vector<TensorPlacement> placements;
        placements.reserve(index.tensors.size());
        for (std::size_t tensor_index = 0; tensor_index < index.tensors.size();
             ++tensor_index) {
            const auto& tensor = index.tensors[tensor_index];
            if (!options.load_mtp && tensor.name.rfind("mtp.", 0) == 0) {
                ++statistics.excluded_tensor_count;
                statistics.excluded_bytes +=
                    tensor.data_bytes + tensor.scale_bytes;
                continue;
            }
            if (tensor.format == DeviceWeightFormatV1::kFp8E4M3Fn) {
                ++statistics.host_only_tensor_count;
                statistics.host_only_bytes +=
                    tensor.data_bytes + tensor.scale_bytes;
                continue;
            }
            const auto payload_bytes = tensor.data_bytes + tensor.scale_bytes;
            switch (tensor.format) {
            case DeviceWeightFormatV1::kW4A16SymG128:
                statistics.w4_bytes += payload_bytes;
                break;
            case DeviceWeightFormatV1::kW8A16SymG128:
                statistics.w8_bytes += payload_bytes;
                break;
            case DeviceWeightFormatV1::kPreserve:
                if (tensor.source_dtype == "BF16")
                    statistics.preserved_bf16_bytes += payload_bytes;
                else if (tensor.source_dtype == "F32")
                    statistics.preserved_f32_bytes += payload_bytes;
                else
                    statistics.preserved_other_bytes += payload_bytes;
                break;
            case DeviceWeightFormatV1::kInvalid:
            case DeviceWeightFormatV1::kFp8E4M3Fn:
                throw std::runtime_error("invalid resident CUDA weight format");
            }
            arena_bytes = align_up(arena_bytes, options.arena_alignment);
            TensorPlacement placement;
            placement.tensor = tensor_index;
            placement.data_offset = arena_bytes;
            if (tensor.data_bytes > std::numeric_limits<std::uint64_t>::max() -
                                    arena_bytes)
                throw std::overflow_error("CUDA weight arena size overflows");
            arena_bytes += tensor.data_bytes;
            if (tensor.scale_bytes != 0) {
                arena_bytes = align_up(arena_bytes, 256);
                placement.scale_offset = arena_bytes;
                if (tensor.scale_bytes >
                    std::numeric_limits<std::uint64_t>::max() - arena_bytes)
                    throw std::overflow_error("CUDA weight arena size overflows");
                arena_bytes += tensor.scale_bytes;
            }
            placements.push_back(placement);
        }
        statistics.tensor_count =
            static_cast<std::uint32_t>(placements.size());
        arena_bytes = align_up(arena_bytes, options.arena_alignment);
        if (arena_bytes == 0 ||
            arena_bytes > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("CUDA weight arena cannot be allocated");
        statistics.arena_bytes = arena_bytes;

        check(cudaMalloc(reinterpret_cast<void**>(&arena),
                         static_cast<std::size_t>(arena_bytes)),
              "cudaMalloc(weight arena)");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate(weight upload)");
        staging.resize(options.staging_buffers);
        statistics.staging_peak_pinned_bytes =
            static_cast<std::uint64_t>(options.staging_buffers) *
            options.staging_bytes;
        for (auto& buffer : staging) {
            check(cudaHostAlloc(reinterpret_cast<void**>(&buffer.host),
                                options.staging_bytes, cudaHostAllocPortable),
                  "cudaHostAlloc(weight staging)");
            check(cudaEventCreateWithFlags(&buffer.reusable,
                                           cudaEventDisableTiming),
                  "cudaEventCreate(weight staging)");
            check(cudaEventRecord(buffer.reusable, stream),
                  "cudaEventRecord(weight staging initial)");
        }

        std::size_t cursor = 0;
        const auto upload_extent = [&](const std::byte* host,
                                       std::byte* device,
                                       std::uint64_t bytes) {
            std::uint64_t copied = 0;
            while (copied < bytes) {
                auto& buffer = staging[cursor++ % staging.size()];
                check(cudaEventSynchronize(buffer.reusable),
                      "cudaEventSynchronize(weight staging)");
                const auto remaining = bytes - copied;
                const auto chunk = static_cast<std::size_t>(
                    std::min<std::uint64_t>(remaining, options.staging_bytes));
                std::memcpy(buffer.host, host + copied, chunk);
                check(cudaMemcpyAsync(device + copied, buffer.host, chunk,
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(weight upload)");
                check(cudaEventRecord(buffer.reusable, stream),
                      "cudaEventRecord(weight staging)");
                copied += chunk;
                statistics.uploaded_bytes += chunk;
                ++statistics.upload_chunks;
            }
        };
        tensors.reserve(placements.size());
        for (const auto& placement : placements) {
            const auto& tensor = index.tensors[placement.tensor];
            const auto source_view = source.require(tensor.name);
            upload_extent(source_view.data, arena + placement.data_offset,
                          tensor.data_bytes);
            if (tensor.scale_bytes != 0)
                upload_extent(source_view.scales,
                              arena + placement.scale_offset,
                              tensor.scale_bytes);
            TensorPointers pointers{
                &tensor,
                arena + placement.data_offset,
                tensor.scale_bytes == 0
                    ? nullptr
                    : static_cast<const void*>(arena + placement.scale_offset),
            };
            if (!tensors.emplace(tensor.name, pointers).second)
                throw std::runtime_error("duplicate CUDA tensor name");
        }
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize(weight upload)");

        for (auto& buffer : staging) {
            check(cudaEventDestroy(buffer.reusable),
                  "cudaEventDestroy(weight staging)");
            buffer.reusable = nullptr;
            check(cudaFreeHost(buffer.host), "cudaFreeHost(weight staging)");
            buffer.host = nullptr;
        }
        staging.clear();
        check(cudaStreamDestroy(stream), "cudaStreamDestroy(weight upload)");
        stream = nullptr;
    }
};

CudaDeviceWeightStore::CudaDeviceWeightStore(const DeviceWeightStore& source,
                                               CudaDeviceWeightOptions options)
    : impl_(std::make_unique<Impl>(options)) {
    try {
        impl_->initialize(source);
    } catch (...) {
        impl_->release();
        throw;
    }
}

CudaDeviceWeightStore::~CudaDeviceWeightStore() = default;
CudaDeviceWeightStore::CudaDeviceWeightStore(CudaDeviceWeightStore&&) noexcept =
    default;
CudaDeviceWeightStore& CudaDeviceWeightStore::operator=(
    CudaDeviceWeightStore&&) noexcept = default;

int CudaDeviceWeightStore::device() const { return impl_->options.device; }
std::uint32_t CudaDeviceWeightStore::stage() const { return impl_->stage; }

CudaTensorViewV1 CudaDeviceWeightStore::find(const std::string& name) const {
    const auto found = impl_->tensors.find(name);
    if (found == impl_->tensors.end()) return {};
    return CudaTensorViewV1{found->second.descriptor, found->second.data,
                            found->second.scales};
}

CudaTensorViewV1 CudaDeviceWeightStore::require(const std::string& name) const {
    auto result = find(name);
    if (result.empty()) throw std::runtime_error("missing CUDA tensor: " + name);
    return result;
}

CudaDeviceWeightStats CudaDeviceWeightStore::stats() const {
    return impl_->statistics;
}

CudaMatrixViewV1 cuda_matrix_view(const CudaTensorViewV1& tensor,
                                  std::uint64_t outer_index) {
    if (tensor.empty() || !tensor.data || tensor.descriptor->shape.size() < 2)
        throw std::invalid_argument("CUDA tensor is not a matrix");
    const auto& descriptor = *tensor.descriptor;
    const auto rows64 = descriptor.shape[descriptor.shape.size() - 2];
    const auto columns64 = descriptor.shape.back();
    if (rows64 > std::numeric_limits<std::uint32_t>::max() ||
        columns64 > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("CUDA matrix dimensions exceed kernel ABI");
    std::uint64_t matrices = 1;
    for (std::size_t i = 0; i + 2 < descriptor.shape.size(); ++i) {
        if (matrices > std::numeric_limits<std::uint64_t>::max() /
                           descriptor.shape[i])
            throw std::overflow_error("CUDA matrix count overflows");
        matrices *= descriptor.shape[i];
    }
    if (outer_index >= matrices)
        throw std::out_of_range("CUDA matrix outer index is out of range");
    const auto elements = rows64 * columns64;
    std::uint64_t data_stride = 0;
    std::uint64_t scale_stride = 0;
    bool preserved_f32 = false;
    switch (descriptor.format) {
        case DeviceWeightFormatV1::kPreserve:
            if (descriptor.source_dtype == "BF16") {
                data_stride = elements * 2;
            } else if (descriptor.source_dtype == "F32") {
                data_stride = elements * 4;
                preserved_f32 = true;
            } else {
                throw std::invalid_argument(
                    "preserved CUDA matrix must be BF16 or F32");
            }
            break;
        case DeviceWeightFormatV1::kW4A16SymG128:
            data_stride = elements / 2;
            scale_stride = elements / descriptor.group_size * 2;
            break;
        case DeviceWeightFormatV1::kW8A16SymG128:
            data_stride = elements;
            scale_stride = elements / descriptor.group_size * 2;
            break;
        default:
            throw std::invalid_argument("CUDA tensor format is not a matrix format");
    }
    const auto* data = static_cast<const std::byte*>(tensor.data) +
                       outer_index * data_stride;
    const void* scales = nullptr;
    if (scale_stride != 0)
        scales = static_cast<const std::byte*>(tensor.scales) +
                 outer_index * scale_stride;
    return CudaMatrixViewV1{descriptor.format,
                            data,
                            scales,
                            static_cast<std::uint32_t>(rows64),
                            static_cast<std::uint32_t>(columns64),
                            descriptor.group_size,
                            preserved_f32};
}

bool cuda_device_weights_compiled() { return true; }

}  // namespace q38
