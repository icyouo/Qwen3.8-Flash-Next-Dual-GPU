// Real native bridge timing, including rank rendezvous and producer ordering.
// Optional NCCL reference uses the same producers, shapes and rank threads.
#include "q38/cuda_te.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#ifdef Q38_P2P_BENCH_NCCL
#include <nccl.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
#ifdef Q38_P2P_BENCH_NCCL
void check_nccl(ncclResult_t result) {
    if (result != ncclSuccess) throw std::runtime_error(ncclGetErrorString(result));
}
#endif
__host__ __device__ float value(std::size_t i, int rank) {
    return float(int((i * 17u + rank * 43u) % 409u) - 204) / 64.f + (rank ? 1.f / 512.f : 0.f);
}
std::uint16_t round_bf16(float input) {
    std::uint32_t bits; std::memcpy(&bits, &input, sizeof(bits));
    return std::uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float decode_bf16(std::uint16_t input) {
    const std::uint32_t bits = std::uint32_t(input) << 16;
    float output; std::memcpy(&output, &bits, sizeof(output)); return output;
}
template<typename T> __device__ T encode(float input) { return input; }
template<> __device__ std::uint16_t encode(float input) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(input));
}
template<typename T> __global__ void produce(T* data, std::size_t n, int rank) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += std::size_t(blockDim.x) * gridDim.x) data[i] = encode<T>(value(i, rank));
}
template<typename T> T expected(std::size_t i) { return value(i, 0) + value(i, 1); }
template<> std::uint16_t expected(std::size_t i) {
    return round_bf16(decode_bf16(round_bf16(value(i, 0))) + decode_bf16(round_bf16(value(i, 1))));
}
template<typename T> void reduce(int rank, T* data, std::size_t n, int group, cudaStream_t stream);
template<> void reduce(int rank, float* data, std::size_t n, int group, cudaStream_t stream) {
    q38::te_bridge_allreduce_sum_f32(rank, data, n, group, stream);
}
template<> void reduce(int rank, std::uint16_t* data, std::size_t n, int group, cudaStream_t stream) {
    q38::te_bridge_allreduce_sum_bf16(rank, data, n, group, stream);
}
template<typename T> struct Worker {
    static constexpr std::size_t guard = 32;
    int rank, group;
    std::size_t n;
    T* allocation{};
    cudaStream_t stream{};
    cudaEvent_t begin{}, end{};
    double wall_us{}, device_us{};
    Worker(int r, int g, std::size_t count) : rank(r), group(g), n(count) {
        check(cudaSetDevice(rank));
        check(cudaMalloc(&allocation, (n + 2 * guard) * sizeof(T)));
        check(cudaMemset(allocation, 0x55, (n + 2 * guard) * sizeof(T)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    }
    ~Worker() {
        cudaSetDevice(rank); cudaStreamSynchronize(stream);
        cudaEventDestroy(begin); cudaEventDestroy(end);
        cudaStreamDestroy(stream); cudaFree(allocation);
    }
    T* data() { return allocation + guard; }
    void verify() {
        check(cudaSetDevice(rank));
        std::vector<T> actual(n + 2 * guard);
        check(cudaMemcpy(actual.data(), allocation, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        T sentinel; std::memset(&sentinel, 0x55, sizeof(T));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const T want = i < guard || i >= n + guard ? sentinel : expected<T>(i - guard);
            if (std::memcmp(&want, &actual[i], sizeof(T)))
                throw std::runtime_error("collective full-payload/guard mismatch");
        }
    }
};
template<typename T> void bench(const std::string& mode, unsigned tokens,
                                unsigned groups, unsigned trial) {
    const std::size_t n = std::size_t(tokens) * 2560;
    const unsigned rounds = tokens <= 5 ? 200 : tokens < 512 ? 30 : 15;
    const bool nccl_mode = mode == "nccl";
    if (!nccl_mode) {
        q38::te_bridge_use_p2p(mode != "host", mode != "host");
        setenv("Q38_TP_BULK_COPY", mode == "bulk" ? "1" : "0", 1);
    }
#ifdef Q38_P2P_BENCH_NCCL
    ncclComm_t comms[2][2]{};
    if (nccl_mode) {
        const int devices[2]{0, 1};
        for (unsigned g = 0; g < groups; ++g) check_nccl(ncclCommInitAll(comms[g], 2, devices));
    }
#else
    if (nccl_mode) throw std::runtime_error("NCCL reference not compiled");
#endif
    std::vector<Worker<T>*> workers;
    for (unsigned g = 0; g < groups; ++g)
        for (int r = 0; r < 2; ++r) workers.push_back(new Worker<T>(r, g, n));
    std::atomic<unsigned> ready{0};
    std::atomic<bool> failed{false};
    std::vector<std::exception_ptr> errors(workers.size());
    std::vector<std::thread> threads;
    for (unsigned index = 0; index < workers.size(); ++index) threads.emplace_back([&, index] {
        auto& w = *workers[index];
        try {
            check(cudaSetDevice(w.rank));
            const auto step = [&] {
                produce<<<32, 256, 0, w.stream>>>(w.data(), n, w.rank);
                check(cudaPeekAtLastError());
#ifdef Q38_P2P_BENCH_NCCL
                if (nccl_mode) check_nccl(ncclAllReduce(w.data(), w.data(), n,
                    sizeof(T) == 4 ? ncclFloat32 : ncclBfloat16, ncclSum, comms[w.group][w.rank], w.stream));
                else
#endif
                    reduce(w.rank, w.data(), n, w.group, w.stream);
            };
            for (unsigned j = 0; j < 3; ++j) step();
            check(cudaStreamSynchronize(w.stream));
            ++ready;
            while (ready < workers.size()) {
                if (failed) throw std::runtime_error("peer worker failed");
                std::this_thread::yield();
            }
            const auto start = std::chrono::steady_clock::now();
            check(cudaEventRecord(w.begin, w.stream));
            for (unsigned j = 0; j < rounds; ++j) step();
            check(cudaEventRecord(w.end, w.stream)); check(cudaEventSynchronize(w.end));
            w.wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / rounds;
            float ms = 0; check(cudaEventElapsedTime(&ms, w.begin, w.end)); w.device_us = double(ms) * 1000 / rounds;
            w.verify();
        } catch (...) { failed = true; q38::te_bridge_abort(); errors[index] = std::current_exception(); }
    });
    for (auto& t : threads) t.join();
    for (const auto& e : errors) if (e) std::rethrow_exception(e);
    double wall_us = 0, device_us = 0;
    for (auto* w : workers) { wall_us = std::max(wall_us, w->wall_us); device_us = std::max(device_us, w->device_us); }
    std::printf("{\"type\":\"collective\",\"mode\":\"%s\",\"dtype\":\"%s\",\"tokens\":%u,\"groups\":%u,\"trial\":%u,\"rounds\":%u,\"bytes_per_rank\":%zu,\"wall_us\":%.6f,\"device_interval_us\":%.6f,\"effective_per_group_GBps\":%.6f,\"verified\":true}\n",
        mode.c_str(), sizeof(T) == 4 ? "f32" : "bf16", tokens, groups, trial, rounds,
        n * sizeof(T), wall_us, device_us, double(n * sizeof(T)) / wall_us / 1000);
    if (!nccl_mode) q38::te_bridge_reset();
#ifdef Q38_P2P_BENCH_NCCL
    if (nccl_mode) for (unsigned g = 0; g < groups; ++g)
        for (int r = 0; r < 2; ++r) { check(cudaSetDevice(r)); check_nccl(ncclCommDestroy(comms[g][r])); }
#endif
    for (auto* w : workers) delete w;
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        int count = 0; check(cudaGetDeviceCount(&count));
        if (count != 2) throw std::runtime_error("requires exactly two visible GPUs");
        bool nccl_only = false, prefill_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--nccl-only") nccl_only = true;
            else if (arg == "--prefill-only") prefill_only = true;
            else throw std::runtime_error("usage: q38_collective_bench [--nccl-only] [--prefill-only]");
        }
        std::vector<std::string> modes = nccl_only ? std::vector<std::string>{} : std::vector<std::string>{"host", "scalar", "bulk"};
#ifdef Q38_P2P_BENCH_NCCL
        int version = 0; check_nccl(ncclGetVersion(&version));
        std::printf("{\"type\":\"nccl_version\",\"version\":%d}\n", version);
        modes.push_back("nccl");
#endif
        if (modes.empty()) throw std::runtime_error("NCCL reference not compiled");
        for (unsigned tokens : {1u, 5u, 128u, 512u}) {
            if (prefill_only && tokens != 512) continue;
            for (unsigned groups : {1u, 2u})
                for (unsigned trial = 0; trial < 5; ++trial)
                    for (unsigned j = 0; j < modes.size(); ++j) {
                        const auto& mode = modes[(j + trial) % modes.size()];
                        bench<float>(mode, tokens, groups, trial);
                        bench<std::uint16_t>(mode, tokens, groups, trial);
                    }
        }
        std::puts("P2P_COLLECTIVE_PASS");
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
