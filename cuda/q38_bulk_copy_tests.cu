#include "q38/cuda_te.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class F> void ranks(F fn) {
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int r = 0; r < 2; ++r) workers[r] = std::thread([&, r] {
        try { check(cudaSetDevice(r)); fn(r); }
        catch (...) { q38::te_bridge_abort(); errors[r] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
}
template <typename T> __device__ T encode(float value) { return value; }
template <> __device__ std::uint16_t encode(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
template <typename T> __global__ void produce(T* output, unsigned count,
                                             int rank, unsigned generation,
                                             bool delay) {
    if (delay) {
        const auto start = clock64();
        while (clock64() - start < 1000000) {}
    }
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        float value = float(int((i * 17u + rank * 43u + generation * 11u) % 211u) - 105) / 32.f;
        if (i % 257 == 0) value = rank ? -0.f : 0.f;
        if (i % 257 == 1) value = rank ? 3.f / 256.f : 1.f;
        output[i] = encode<T>(value);
    }
}
template <typename T> void reduce(int rank, T* data, unsigned count,
                                int group, cudaStream_t stream);
template <> void reduce(int rank, float* data, unsigned count,
                        int group, cudaStream_t stream) {
    q38::te_bridge_allreduce_sum_f32(rank, data, count, group, stream);
}
template <> void reduce(int rank, std::uint16_t* data, unsigned count,
                        int group, cudaStream_t stream) {
    q38::te_bridge_allreduce_sum_bf16(rank, data, count, group, stream);
}
template <typename T> std::vector<T> queued(bool bulk, int group) {
    setenv("Q38_TP_BULK_COPY", bulk ? "1" : "0", 1);
    // Cross threshold, grow/reuse scratch with earlier generations in flight,
    // use unaligned pointers and non-block-aligned tails, then return to small.
    const unsigned sizes[] = {65535, 65536, 65537, 131072, 131073,
                              1310721, 32771, 1310721, 65536};
    std::vector<T> payloads[2];
    ranks([&](int rank) {
        cudaStream_t streams[2];
        for (auto& stream : streams) check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::vector<T*> buffers;
        for (unsigned generation = 0; generation < sizeof(sizes) / sizeof(sizes[0]); ++generation) {
            const auto count = sizes[generation];
            auto stream = streams[generation % 2];
            T* allocation;
            check(cudaMalloc(&allocation, (count + 2) * sizeof(T)));
            buffers.push_back(allocation);
            check(cudaMemsetAsync(allocation, 0x55, (count + 2) * sizeof(T), stream));
            auto* data = allocation + 1;
            produce<<<32, 256, 0, stream>>>(data, count, rank, generation,
                                           generation % 3 == unsigned(rank));
            check(cudaPeekAtLastError());
            reduce(rank, data, count, group, stream);
            int restored = -1;
            check(cudaGetDevice(&restored));
            if (restored != rank) throw std::runtime_error("device restoration");
        }
        for (auto stream : streams) check(cudaStreamSynchronize(stream));
        for (unsigned generation = 0; generation < buffers.size(); ++generation) {
            std::vector<T> values(sizes[generation] + 2);
            check(cudaMemcpy(values.data(), buffers[generation], values.size() * sizeof(T), cudaMemcpyDeviceToHost));
            T sentinel;
            std::memset(&sentinel, 0x55, sizeof(T));
            if (std::memcmp(&values.front(), &sentinel, sizeof(T)) ||
                std::memcmp(&values.back(), &sentinel, sizeof(T)))
                throw std::runtime_error("bulk reduction guard overwritten");
            payloads[rank].insert(payloads[rank].end(), values.begin() + 1, values.end() - 1);
            check(cudaFree(buffers[generation]));
        }
        for (auto stream : streams) check(cudaStreamDestroy(stream));
    });
    if (payloads[0].size() != payloads[1].size() ||
        std::memcmp(payloads[0].data(), payloads[1].data(), payloads[0].size() * sizeof(T)))
        throw std::runtime_error("rank outputs differ");
    q38::te_bridge_reset_group(group);
    return payloads[0];
}
template <typename T> void parity(int group) {
    const auto reference = queued<T>(false, group);
    const auto candidate = queued<T>(true, group);
    if (reference.size() != candidate.size() ||
        std::memcmp(reference.data(), candidate.data(), reference.size() * sizeof(T)))
        throw std::runtime_error("bulk copy differs from scalar P2P oracle");
}
template <typename T> double benchmark(bool bulk) {
    setenv("Q38_TP_BULK_COPY", bulk ? "1" : "0", 1);
    constexpr unsigned count = 512 * 2560, rounds = 30;
    double elapsed[2] = {};
    ranks([&](int rank) {
        T* data;
        check(cudaMalloc(&data, count * sizeof(T)));
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (unsigned i = 0; i < 3; ++i) {
            produce<<<32, 256, 0, stream>>>(data, count, rank, i, false);
            reduce(rank, data, count, 0, stream);
        }
        check(cudaStreamSynchronize(stream));
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < rounds; ++i) {
            produce<<<32, 256, 0, stream>>>(data, count, rank, i, false);
            reduce(rank, data, count, 0, stream);
        }
        check(cudaStreamSynchronize(stream));
        elapsed[rank] = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / rounds;
        check(cudaFree(data));
        check(cudaStreamDestroy(stream));
    });
    q38::te_bridge_reset();
    return std::max(elapsed[0], elapsed[1]);
}
}
int main() {
    try {
        q38::te_bridge_use_p2p(true, true);
        for (int group = 0; group < 2; ++group) {
            parity<float>(group);
            parity<std::uint16_t>(group);
        }
        for (bool ordered : {false, true}) {
            q38::te_bridge_use_p2p(true, ordered);
            parity<float>(0);
            parity<std::uint16_t>(1);
        }
        q38::te_bridge_use_p2p(true, true);
        for (unsigned trial = 0; trial < 3; ++trial) {
            const auto f0 = benchmark<float>(false), f1 = benchmark<float>(true);
            const auto b0 = benchmark<std::uint16_t>(false), b1 = benchmark<std::uint16_t>(true);
            std::cout << "{\"trial\":" << trial << ",\"tokens\":512,\"f32_scalar_us\":" << f0
                      << ",\"f32_copy_us\":" << f1 << ",\"bf16_scalar_us\":" << b0
                      << ",\"bf16_copy_us\":" << b1 << "}\n";
        }
        std::cout << "BULK_COPY_PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
