// Verified direct peer reads/writes and memcpy, without a model or NCCL.
// Sources are immutable and distinct from every output, including duplex runs.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__host__ __device__ std::uint32_t pattern(std::size_t i, unsigned seed) {
    std::uint32_t x = std::uint32_t(i) ^ ((seed + 1u) * 0x9e3779b9u);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
__global__ void fill(std::uint32_t* data, std::size_t n, unsigned seed) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += std::size_t(blockDim.x) * gridDim.x) data[i] = pattern(i, seed);
}
template<bool Vector, bool RemoteWrite>
__global__ void copy(const std::uint32_t* src, std::uint32_t* dst, std::size_t n) {
    constexpr unsigned width = Vector ? 4 : 1;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / width;
         i += std::size_t(blockDim.x) * gridDim.x) {
        if constexpr (Vector) reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
        else dst[i] = src[i];
    }
    if constexpr (RemoteWrite) __threadfence_system();
}
__global__ void verify(const std::uint32_t* data, std::size_t n,
                       std::size_t guard, unsigned owner, int source,
                       unsigned* errors) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n + 2 * guard;
         i += std::size_t(blockDim.x) * gridDim.x) {
        const bool payload = source >= 0 && i >= guard && i < guard + n;
        const auto expected = pattern(i, payload ? unsigned(source) : owner + 8);
        if (data[i] != expected) atomicAdd(errors, 1u);
    }
}
__global__ void verify_source(const std::uint32_t* data, std::size_t n,
                              unsigned owner, unsigned* errors) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += std::size_t(blockDim.x) * gridDim.x)
        if (data[i] != pattern(i, owner)) atomicAdd(errors, 1u);
}
enum class Mode { Memcpy, ReadScalar, ReadVector, WriteScalar, WriteVector, Local };
const char* name(Mode mode) {
    switch (mode) {
    case Mode::Memcpy: return "peer_memcpy";
    case Mode::ReadScalar: return "peer_read_scalar";
    case Mode::ReadVector: return "peer_read_vector";
    case Mode::WriteScalar: return "peer_write_scalar";
    case Mode::WriteVector: return "peer_write_vector";
    case Mode::Local: return "local_memcpy";
    }
    throw std::logic_error("mode");
}
struct Buffers {
    static constexpr std::size_t guard = 32; // 128-byte alignment and full guards.
    std::size_t n;
    std::uint32_t* src[2]{};
    std::uint32_t* dst[2]{};
    unsigned* errors[2]{};
    cudaStream_t streams[2]{};
    cudaEvent_t begin[2]{}, end[2]{};
    explicit Buffers(std::size_t bytes) : n(bytes / 4) {
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r));
            check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
            check(cudaEventCreate(&begin[r])); check(cudaEventCreate(&end[r]));
            check(cudaMalloc(&src[r], (n + 2 * guard) * 4));
            check(cudaMalloc(&dst[r], (n + 2 * guard) * 4));
            check(cudaMalloc(&errors[r], sizeof(unsigned)));
            fill<<<256, 256, 0, streams[r]>>>(src[r], n + 2 * guard, r);
            check(cudaPeekAtLastError()); check(cudaStreamSynchronize(streams[r]));
        }
    }
    ~Buffers() {
        for (int r = 0; r < 2; ++r) {
            cudaSetDevice(r); cudaStreamSynchronize(streams[r]);
            cudaFree(src[r]); cudaFree(dst[r]); cudaFree(errors[r]);
            cudaEventDestroy(begin[r]); cudaEventDestroy(end[r]);
            cudaStreamDestroy(streams[r]);
        }
    }
    int context(Mode mode, int source) const {
        return mode == Mode::WriteScalar || mode == Mode::WriteVector || mode == Mode::Local ? source : 1 - source;
    }
    void enqueue(Mode mode, int source) {
        const int dest = mode == Mode::Local ? source : 1 - source;
        const int ctx = context(mode, source);
        check(cudaSetDevice(ctx));
        const auto* input = src[source] + guard;
        auto* output = dst[dest] + guard;
        switch (mode) {
        case Mode::Memcpy:
            check(cudaMemcpyPeerAsync(output, dest, input, source, n * 4, streams[ctx])); break;
        case Mode::Local:
            check(cudaMemcpyAsync(output, input, n * 4, cudaMemcpyDeviceToDevice, streams[ctx])); break;
        case Mode::ReadScalar: copy<false, false><<<256, 256, 0, streams[ctx]>>>(input, output, n); break;
        case Mode::ReadVector: copy<true, false><<<256, 256, 0, streams[ctx]>>>(input, output, n); break;
        case Mode::WriteScalar: copy<false, true><<<256, 256, 0, streams[ctx]>>>(input, output, n); break;
        case Mode::WriteVector: copy<true, true><<<256, 256, 0, streams[ctx]>>>(input, output, n); break;
        }
        check(cudaPeekAtLastError());
    }
    void run(Mode mode, int direction, unsigned trial) {
        const bool duplex = direction == 2;
        const unsigned rounds = n * 4 <= 262144 ? 300 : n * 4 <= 5242880 ? 40 : 12;
        std::vector<int> sources = duplex ? std::vector<int>{0, 1} : std::vector<int>{direction};
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r));
            fill<<<256, 256, 0, streams[r]>>>(dst[r], n + 2 * guard, r + 8);
            check(cudaPeekAtLastError()); check(cudaStreamSynchronize(streams[r]));
        }
        for (unsigned k = 0; k < 3; ++k) for (int s : sources) enqueue(mode, s);
        for (int r = 0; r < 2; ++r) { check(cudaSetDevice(r)); check(cudaStreamSynchronize(streams[r])); }
        const auto start = std::chrono::steady_clock::now();
        for (int s : sources) { const int ctx = context(mode, s); check(cudaSetDevice(ctx)); check(cudaEventRecord(begin[ctx], streams[ctx])); }
        for (unsigned k = 0; k < rounds; ++k) for (int s : sources) enqueue(mode, s);
        for (int s : sources) { const int ctx = context(mode, s); check(cudaSetDevice(ctx)); check(cudaEventRecord(end[ctx], streams[ctx])); }
        double device_us[2]{};
        for (int s : sources) {
            const int ctx = context(mode, s); check(cudaSetDevice(ctx)); check(cudaEventSynchronize(end[ctx]));
            float ms = 0; check(cudaEventElapsedTime(&ms, begin[ctx], end[ctx])); device_us[s] = double(ms) * 1000 / rounds;
        }
        const double wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / rounds;
        for (int r = 0; r < 2; ++r) {
            int expected = -1;
            for (int s : sources) if (r == (mode == Mode::Local ? s : 1 - s)) expected = s;
            check(cudaSetDevice(r)); check(cudaMemsetAsync(errors[r], 0, sizeof(unsigned), streams[r]));
            verify<<<256, 256, 0, streams[r]>>>(dst[r], n, guard, r, expected, errors[r]);
            verify_source<<<256, 256, 0, streams[r]>>>(src[r], n + 2 * guard, r, errors[r]);
            check(cudaPeekAtLastError());
            unsigned mismatches = 0;
            check(cudaMemcpyAsync(&mismatches, errors[r], sizeof(unsigned), cudaMemcpyDeviceToHost, streams[r]));
            check(cudaStreamSynchronize(streams[r]));
            if (mismatches) throw std::runtime_error("payload/source/guard/other-GPU sentinel mismatch");
        }
        const double us = std::max(device_us[0], device_us[1]);
        std::printf("{\"type\":\"raw\",\"mode\":\"%s\",\"direction\":%d,\"trial\":%u,\"bytes_per_direction\":%zu,\"rounds\":%u,\"device_us\":%.6f,\"wall_us\":%.6f,\"per_direction_GBps\":%.6f,\"aggregate_wall_GBps\":%.6f,\"verified\":true}\n",
            name(mode), direction, trial, n * 4, rounds, us, wall_us, double(n * 4) / us / 1000, double(n * 4 * sources.size()) / wall_us / 1000);
    }
};
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const bool prefill_only = argc == 2 && std::string(argv[1]) == "--prefill-only";
        if (argc > 1 && !prefill_only) throw std::runtime_error("usage: q38_p2p_bench [--prefill-only]");
        int count = 0; check(cudaGetDeviceCount(&count));
        if (count != 2) throw std::runtime_error("requires exactly two visible GPUs");
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r)); int capable = 0;
            check(cudaDeviceCanAccessPeer(&capable, r, 1 - r));
            if (!capable) throw std::runtime_error("bidirectional peer access unavailable");
            check(cudaDeviceEnablePeerAccess(1 - r, 0));
        }
        const std::vector<Mode> modes{Mode::Memcpy, Mode::ReadScalar, Mode::ReadVector, Mode::WriteScalar, Mode::WriteVector, Mode::Local};
        for (std::size_t bytes : {4096ul, 65536ul, 262144ul, 2621440ul, 5242880ul, 16777216ul, 67108864ul}) {
            if (prefill_only && bytes != 2621440 && bytes != 5242880) continue;
            Buffers buffers(bytes);
            for (unsigned trial = 0; trial < 5; ++trial)
                for (unsigned j = 0; j < modes.size(); ++j) {
                    const auto mode = modes[(j + trial) % modes.size()];
                    for (int direction : {0, 1, 2}) buffers.run(mode, direction, trial);
                }
        }
        std::puts("P2P_RAW_PASS");
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
