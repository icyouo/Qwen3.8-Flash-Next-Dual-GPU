#include "q38/cuda_transport.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q38 {

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

__device__ __forceinline__ std::uint64_t boundary_mix64(
    std::uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31u;
    return value;
}

__global__ void boundary_checksum_kernel(
    const std::uint16_t* words, std::size_t word_count,
    unsigned long long* checksum) {
    __shared__ unsigned long long partial[256];
    const auto thread = static_cast<std::size_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const auto stride = static_cast<std::size_t>(gridDim.x * blockDim.x);
    std::uint64_t local = 0;
    if (thread == 0) {
        local = boundary_mix64(
            static_cast<std::uint64_t>(word_count) ^
            UINT64_C(0x713338424f554e44));
    }
    for (std::size_t index = thread; index < word_count; index += stride) {
        const auto tagged =
            (static_cast<std::uint64_t>(words[index]) << 48u) ^
            static_cast<std::uint64_t>(index) ^
            UINT64_C(0x9e3779b97f4a7c15);
        local ^= boundary_mix64(tagged);
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2u; offset != 0; offset /= 2u) {
        if (threadIdx.x < offset)
            partial[threadIdx.x] ^= partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicXor(checksum, partial[0]);
}

void launch_boundary_checksum(const std::uint16_t* words,
                              std::size_t word_count,
                              unsigned long long* checksum,
                              cudaStream_t stream) {
    constexpr unsigned kThreads = 256;
    const auto needed = (word_count + kThreads - 1u) / kThreads;
    const auto blocks = static_cast<unsigned>(
        std::max<std::size_t>(1, std::min<std::size_t>(256, needed)));
    check(cudaMemsetAsync(checksum, 0, sizeof(*checksum), stream),
          "cudaMemsetAsync(boundary checksum)");
    boundary_checksum_kernel<<<blocks, kThreads, 0, stream>>>(
        words, word_count, checksum);
    check(cudaGetLastError(), "boundary_checksum_kernel");
}

struct RingState {
    struct Slot {
        std::uint16_t* host = nullptr;
        unsigned long long* checksum_host = nullptr;
        unsigned long long* checksum_device = nullptr;
        cudaEvent_t checksum_ready = nullptr;
        cudaEvent_t ready = nullptr;
        bool leased = false;
    };

    explicit RingState(CudaBoundaryRingOptions value) : options(value) {
        if (options.slots < 2 || options.slots > 16 ||
            options.max_tokens == 0 || options.hidden_width == 0)
            throw std::invalid_argument("invalid CUDA boundary ring options");
        if (static_cast<std::uint64_t>(options.max_tokens) *
                options.hidden_width > SIZE_MAX / sizeof(std::uint16_t))
            throw std::overflow_error("CUDA boundary ring capacity overflows");
        capacity_words = static_cast<std::size_t>(options.max_tokens) *
                         options.hidden_width;
        check(cudaSetDevice(options.producer_device), "cudaSetDevice(producer)");
        try {
            slots.resize(options.slots);
            for (auto& slot : slots) {
                check(cudaHostAlloc(reinterpret_cast<void**>(&slot.host),
                                    capacity_words * sizeof(std::uint16_t),
                                    cudaHostAllocPortable),
                      "cudaHostAlloc(boundary)");
                check(cudaHostAlloc(
                          reinterpret_cast<void**>(&slot.checksum_host),
                          sizeof(*slot.checksum_host), cudaHostAllocPortable),
                      "cudaHostAlloc(boundary checksum)");
                check(cudaMalloc(
                          reinterpret_cast<void**>(&slot.checksum_device),
                          sizeof(*slot.checksum_device)),
                      "cudaMalloc(boundary checksum)");
                check(cudaEventCreateWithFlags(&slot.checksum_ready,
                                               cudaEventDisableTiming),
                      "cudaEventCreate(boundary checksum)");
                check(cudaEventCreateWithFlags(&slot.ready,
                                               cudaEventDisableTiming),
                      "cudaEventCreate(boundary)");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~RingState() { release(); }

    void release() noexcept {
        (void)cudaSetDevice(options.producer_device);
        for (auto& slot : slots) {
            if (slot.checksum_ready)
                (void)cudaEventDestroy(slot.checksum_ready);
            if (slot.ready) (void)cudaEventDestroy(slot.ready);
            if (slot.checksum_device)
                (void)cudaFree(slot.checksum_device);
            if (slot.checksum_host)
                (void)cudaFreeHost(slot.checksum_host);
            if (slot.host) (void)cudaFreeHost(slot.host);
            slot.checksum_ready = nullptr;
            slot.ready = nullptr;
            slot.checksum_device = nullptr;
            slot.checksum_host = nullptr;
            slot.host = nullptr;
        }
    }

    CudaBoundaryRingOptions options;
    std::size_t capacity_words = 0;
    std::vector<Slot> slots;
    mutable std::mutex mutex;
    std::condition_variable available;
    CudaBoundaryRingStats statistics;
    std::size_t cursor = 0;
};

class RingLease final : public BoundaryLease {
public:
    RingLease(std::shared_ptr<RingState> state, std::size_t slot,
              std::size_t words)
        : state_(std::move(state)), slot_(slot), words_(words) {}

    ~RingLease() override {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->slots[slot_].leased = false;
        }
        state_->available.notify_one();
    }

    const std::uint16_t* data() const override {
        return state_->slots[slot_].host;
    }
    std::size_t size() const override { return words_; }
    std::uint32_t slot() const override {
        return static_cast<std::uint32_t>(slot_);
    }
    void wait_ready() const override {
        check(cudaSetDevice(state_->options.producer_device),
              "cudaSetDevice(boundary wait)");
        check(cudaEventSynchronize(state_->slots[slot_].ready),
              "cudaEventSynchronize(boundary)");
        const auto actual = boundary_payload_checksum(
            state_->slots[slot_].host, words_);
        if (actual != *state_->slots[slot_].checksum_host)
            throw std::runtime_error("CUDA boundary payload checksum mismatch");
    }
    std::uint64_t payload_checksum() const override {
        return *state_->slots[slot_].checksum_host;
    }

private:
    std::shared_ptr<RingState> state_;
    std::size_t slot_;
    std::size_t words_;
};

}  // namespace

struct CudaBoundaryRing::Impl {
    explicit Impl(CudaBoundaryRingOptions options)
        : state(std::make_shared<RingState>(options)) {}
    std::shared_ptr<RingState> state;
};

CudaBoundaryRing::CudaBoundaryRing(CudaBoundaryRingOptions options)
    : impl_(std::make_unique<Impl>(options)) {}
CudaBoundaryRing::~CudaBoundaryRing() = default;
CudaBoundaryRing::CudaBoundaryRing(CudaBoundaryRing&&) noexcept = default;
CudaBoundaryRing& CudaBoundaryRing::operator=(CudaBoundaryRing&&) noexcept =
    default;

std::shared_ptr<const BoundaryLease> CudaBoundaryRing::copy_from_device(
    const std::uint16_t* device_source, std::size_t words,
    void* producer_stream) {
    if (!device_source || !producer_stream || words == 0 ||
        words > impl_->state->capacity_words)
        throw std::invalid_argument("invalid CUDA boundary transfer");
    auto& state = *impl_->state;
    std::unique_lock<std::mutex> lock(state.mutex);
    std::size_t selected = state.slots.size();
    for (;;) {
        for (std::size_t attempt = 0; attempt < state.slots.size(); ++attempt) {
            const auto candidate = (state.cursor + attempt) % state.slots.size();
            if (!state.slots[candidate].leased) {
                selected = candidate;
                state.cursor = (candidate + 1) % state.slots.size();
                break;
            }
        }
        if (selected != state.slots.size()) break;
        ++state.statistics.waits;
        state.available.wait(lock);
    }
    state.slots[selected].leased = true;
    lock.unlock();

    try {
        check(cudaSetDevice(state.options.producer_device),
              "cudaSetDevice(boundary copy)");
        auto stream = reinterpret_cast<cudaStream_t>(producer_stream);
        launch_boundary_checksum(device_source, words,
                                 state.slots[selected].checksum_device,
                                 stream);
        check(cudaMemcpyAsync(state.slots[selected].checksum_host,
                              state.slots[selected].checksum_device,
                              sizeof(*state.slots[selected].checksum_host),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(boundary checksum)");
        check(cudaEventRecord(state.slots[selected].checksum_ready, stream),
              "cudaEventRecord(boundary checksum)");
        check(cudaMemcpyAsync(state.slots[selected].host, device_source,
                              words * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(boundary D2H)");
        check(cudaEventRecord(state.slots[selected].ready, stream),
              "cudaEventRecord(boundary)");
        check(cudaEventSynchronize(state.slots[selected].checksum_ready),
              "cudaEventSynchronize(boundary checksum)");
    } catch (...) {
        {
            std::lock_guard<std::mutex> restore(state.mutex);
            state.slots[selected].leased = false;
        }
        state.available.notify_one();
        throw;
    }
    {
        std::lock_guard<std::mutex> update(state.mutex);
        ++state.statistics.transfers;
        state.statistics.bytes += words * sizeof(std::uint16_t);
    }
    return std::make_shared<RingLease>(impl_->state, selected, words);
}

CudaBoundaryRingStats CudaBoundaryRing::stats() const {
    std::lock_guard<std::mutex> lock(impl_->state->mutex);
    return impl_->state->statistics;
}

std::uint64_t CudaBoundaryRing::pinned_bytes() const {
    const auto& state = *impl_->state;
    return static_cast<std::uint64_t>(state.slots.size()) *
           (state.capacity_words * sizeof(std::uint16_t) +
            sizeof(unsigned long long));
}

std::uint64_t CudaBoundaryRing::device_bytes() const {
    return static_cast<std::uint64_t>(impl_->state->slots.size()) *
           sizeof(unsigned long long);
}

void cuda_copy_boundary_to_device(const BoundaryBuffer& boundary,
                                  std::uint16_t* device_destination,
                                  void* consumer_stream,
                                  int consumer_device) {
    if (!device_destination || !consumer_stream || boundary.size() == 0)
        throw std::invalid_argument("invalid CUDA boundary consumer transfer");
    boundary.wait_ready();
    if (boundary.frame.payload_checksum != boundary.payload_checksum())
        throw std::runtime_error("CUDA boundary frame checksum mismatch");
    check(cudaSetDevice(consumer_device), "cudaSetDevice(boundary consumer)");
    check(cudaMemcpyAsync(device_destination, boundary.data(),
                          boundary.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice,
                          reinterpret_cast<cudaStream_t>(consumer_stream)),
          "cudaMemcpyAsync(boundary H2D)");
}

bool cuda_transport_compiled() { return true; }

}  // namespace q38
