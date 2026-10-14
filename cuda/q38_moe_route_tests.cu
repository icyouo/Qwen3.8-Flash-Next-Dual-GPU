#include "q38/cuda_moe_prefill.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void select(bool parallel) {
    if (setenv("Q38_CUDA_MOE_ROUTE_PARALLEL", parallel ? "1" : "0", 1))
        throw std::runtime_error("setenv failed");
}
std::vector<std::uint8_t> snapshot(void* p, std::size_t bytes, cudaStream_t stream) {
    std::vector<std::uint8_t> result(bytes);
    check(cudaMemcpyAsync(result.data(), p, bytes, cudaMemcpyDeviceToHost, stream), "snapshot");
    check(cudaStreamSynchronize(stream), "snapshot ready");
    return result;
}
void equal(const std::vector<std::uint8_t>& expected,
           std::vector<std::uint8_t> actual, bool omitted) {
    if (omitted) {
        q38::Q38RoutePlanHeaderV1 original{}, header{};
        std::memcpy(&original, expected.data(), sizeof(original));
        std::memcpy(&header, actual.data(), sizeof(header));
        if (header.route_hash || header.plan_hash ||
            header.flags != (original.flags | q38::kQ38RoutePlanHashesOmitted))
            throw std::runtime_error("invalid omitted-hash header");
        header.flags = original.flags;
        header.route_hash = original.route_hash;
        header.plan_hash = original.plan_hash;
        std::memcpy(actual.data(), &header, sizeof(header));
    }
    if (actual != expected) throw std::runtime_error("route-plan byte parity failed");
}
void fixture(unsigned tokens, cudaStream_t stream) {
    const unsigned routes = q38::q38_moe_prefill_routes_v1(tokens);
    const auto bytes = q38::q38_moe_route_plan_bytes_v1(tokens);
    void* allocation{};
    std::int32_t* device_ids{};
    check(cudaMalloc(&allocation, bytes), "plan allocation");
    check(cudaMemset(allocation, 0, bytes), "initialize plan");
    check(cudaMalloc(reinterpret_cast<void**>(&device_ids), routes * sizeof(*device_ids)), "ids allocation");
    const auto plan = q38::cuda_moe_route_plan_storage_v1(allocation, bytes, tokens);
    std::vector<std::int32_t> ids(routes);
    cudaGraph_t graphs[3]{};
    cudaGraphExec_t runs[3]{};
    for (unsigned v = 0; v < 3; ++v) {
        select(v != 0);
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
        q38::cuda_moe_build_route_plan_v1(device_ids, tokens, plan, stream, 0, 256, true, v != 2);
        check(cudaStreamEndCapture(stream, &graphs[v]), "capture end");
        check(cudaGraphInstantiate(&runs[v], graphs[v], 0), "instantiate");
    }
    unsigned cases = 0;
    for (unsigned local : {256u, 512u}) {
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
            for (unsigned i = 0; i < routes; ++i) {
                ids[i] = pattern == 1 ? 3 : pattern == 2 ? int((i / 17) % 7) : int((i * 37 + tokens) % local);
                if (pattern == 3 && i % 3 == 0) ids[i] = -1;
                if (pattern == 4) ids[i] = -1;
            }
            if (pattern == 5) ids[0] = -2;
            check(cudaMemcpyAsync(device_ids, ids.data(), routes * sizeof(ids[0]), cudaMemcpyHostToDevice, stream), "ids upload");
            select(false);
            q38::cuda_moe_build_route_plan_v1(device_ids, tokens, plan, stream, 0, local, true);
            const auto expected = snapshot(allocation, bytes, stream);
            for (unsigned v = 1; v < 3; ++v) {
                select(true);
                q38::cuda_moe_build_route_plan_v1(device_ids, tokens, plan, stream, 0, local, true, v != 2);
                equal(expected, snapshot(allocation, bytes, stream), v == 2);
                ++cases;
            }
            if (local == 256) {
                for (unsigned v = 0; v < 3; ++v) {
                    // Two queued generations reuse the same plan. Every graph
                    // reads the newly uploaded IDs, rather than capture-time data.
                    check(cudaGraphLaunch(runs[v], stream), "graph replay 1");
                    check(cudaGraphLaunch(runs[v], stream), "graph replay 2");
                    equal(expected, snapshot(allocation, bytes, stream), v == 2);
                    ++cases;
                }
            }
        }
    }
    for (unsigned i = 0; i < routes; ++i) ids[i] = int((i * 37 + tokens) % 256);
    check(cudaMemcpyAsync(device_ids, ids.data(), routes * sizeof(ids[0]), cudaMemcpyHostToDevice, stream), "timing input");
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "start event"); check(cudaEventCreate(&stop), "stop event");
    std::vector<float> times[3];
    for (unsigned repeat = 0; repeat < 5; ++repeat)
        for (unsigned order = 0; order < 3; ++order) {
            const unsigned v = (repeat + order) % 3;
            for (unsigned i = 0; i < 10; ++i) check(cudaGraphLaunch(runs[v], stream), "warmup");
            check(cudaEventRecord(start, stream), "timing start");
            for (unsigned i = 0; i < 200; ++i) check(cudaGraphLaunch(runs[v], stream), "timed replay");
            check(cudaEventRecord(stop, stream), "timing stop");
            check(cudaEventSynchronize(stop), "timing ready");
            float ms; check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
            times[v].push_back(ms / 200.f);
        }
    float medians[3]{};
    for (unsigned v = 0; v < 3; ++v) { std::sort(times[v].begin(), times[v].end()); medians[v] = times[v][2]; }
    std::cout << std::setprecision(8) << "{\"tokens\":" << tokens << ",\"parity_cases\":" << cases
        << ",\"baseline_ms\":" << medians[0] << ",\"parallel_ms\":" << medians[1]
        << ",\"runtime_ms\":" << medians[2] << ",\"speedup\":" << medians[0] / medians[2] << "}\n";
    (void)cudaEventDestroy(start); (void)cudaEventDestroy(stop);
    for (unsigned v = 0; v < 3; ++v) { (void)cudaGraphExecDestroy(runs[v]); (void)cudaGraphDestroy(graphs[v]); }
    (void)cudaFree(device_ids); (void)cudaFree(allocation);
}
}

int main() {
    try {
        cudaStream_t stream{};
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        for (unsigned tokens : {1u, 2u, 5u, 16u, 128u, 512u}) fixture(tokens, stream);
        check(cudaStreamDestroy(stream), "destroy stream");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
