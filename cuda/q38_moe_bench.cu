#include "q38/cuda_moe.h"
#include "q38/cuda_hyper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

std::uint16_t bf16(float value) {
    const auto converted = __float2bfloat16_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&converted);
}

float fp32(std::uint16_t value) {
    return __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(&value));
}

template <typename T>
T* upload(const std::vector<T>& host) {
    T* result = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&result),
                     host.size() * sizeof(T)),
          "cudaMalloc(MoE benchmark upload)");
    check(cudaMemcpy(result, host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy(MoE benchmark upload)");
    return result;
}

template <typename T>
T* allocate(std::size_t count) {
    T* result = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)),
          "cudaMalloc(MoE benchmark buffer)");
    return result;
}

std::vector<std::uint8_t> make_w4(std::size_t elements,
                                  std::uint32_t salt) {
    std::vector<std::uint8_t> result(elements / 2);
    for (std::size_t index = 0; index < elements; index += 2) {
        const int low_value =
            -7 + static_cast<int>((index * 29 + salt) % 15);
        const int high_value =
            -7 + static_cast<int>(((index + 1) * 29 + salt) % 15);
        result[index / 2] = static_cast<std::uint8_t>(
            (static_cast<unsigned>(low_value) & 0xfu) |
            ((static_cast<unsigned>(high_value) & 0xfu) << 4));
    }
    return result;
}

int parse_iterations(int argc, char** argv) {
    if (argc <= 1) return 200;
    const auto value = std::strtol(argv[1], nullptr, 10);
    if (value <= 0 || value > 100000)
        throw std::invalid_argument("invalid iteration count");
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto iterations = parse_iterations(argc, argv);
        const char* policy = std::getenv("Q38_CUDA_DECODE_MOE");
        constexpr std::uint32_t top_k = q38::kQ38MoeTopK;
        constexpr std::uint32_t gate_rows =
            2 * q38::kQ38MoeIntermediate;
        constexpr std::uint32_t gate_columns = q38::kQ38HiddenWidth;
        constexpr std::uint32_t down_rows = q38::kQ38HiddenWidth;
        constexpr std::uint32_t down_columns = q38::kQ38MoeIntermediate;
        const auto gate_elements =
            static_cast<std::size_t>(gate_rows) * gate_columns;
        const auto down_elements =
            static_cast<std::size_t>(down_rows) * down_columns;

        auto gate_data = make_w4(gate_elements, 5);
        auto down_data = make_w4(down_elements, 11);
        std::vector<std::uint16_t> gate_scales(
            gate_rows * (gate_columns / 128), bf16(0.0125f));
        std::vector<std::uint16_t> down_scales(
            down_rows * (down_columns / 128), bf16(0.0175f));
        std::vector<std::uint16_t> hidden(gate_columns);
        for (std::size_t index = 0; index < hidden.size(); ++index)
            hidden[index] = bf16(
                std::sin(static_cast<float>(index) * 0.013f) * 0.2f);
        std::vector<std::int32_t> experts(top_k, 0);
        std::vector<float> route_weights(top_k, 1.0f / top_k);

        auto* d_gate_data = upload(gate_data);
        auto* d_down_data = upload(down_data);
        auto* d_gate_scales = upload(gate_scales);
        auto* d_down_scales = upload(down_scales);
        auto* d_hidden = upload(hidden);
        auto* d_experts = upload(experts);
        auto* d_route_weights = upload(route_weights);

        q38::DeviceTensorV1 gate_descriptor;
        gate_descriptor.format = q38::DeviceWeightFormatV1::kW4A16SymG128;
        gate_descriptor.group_size = 128;
        gate_descriptor.shape = {q38::kQ38MoeExperts, gate_rows,
                                 gate_columns};
        q38::DeviceTensorV1 down_descriptor;
        down_descriptor.format = q38::DeviceWeightFormatV1::kW4A16SymG128;
        down_descriptor.group_size = 128;
        down_descriptor.shape = {q38::kQ38MoeExperts, down_rows,
                                 down_columns};
        const q38::CudaTensorViewV1 gate_view{&gate_descriptor, d_gate_data,
                                              d_gate_scales};
        const q38::CudaTensorViewV1 down_view{&down_descriptor, d_down_data,
                                              d_down_scales};

        auto* gate_scratch = allocate<std::uint16_t>(
            static_cast<std::size_t>(top_k) * gate_rows);
        auto* activated = allocate<std::uint16_t>(
            static_cast<std::size_t>(top_k) * down_columns);
        auto* accumulation = allocate<float>(down_rows);
        auto* output = allocate<std::uint16_t>(down_rows);
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate(MoE benchmark)");

        const auto launch = [&] {
            q38::cuda_moe_routed_bf16(
                gate_view, down_view, d_hidden, d_experts, d_route_weights, 1,
                top_k, gate_scratch, activated, accumulation, output,
                reinterpret_cast<void*>(stream), 0);
        };
        for (int index = 0; index < 20; ++index) launch();
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize(MoE warmup)");
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check(cudaEventCreate(&start), "cudaEventCreate(MoE start)");
        check(cudaEventCreate(&stop), "cudaEventCreate(MoE stop)");
        check(cudaEventRecord(start, stream), "cudaEventRecord(MoE start)");
        for (int index = 0; index < iterations; ++index) launch();
        check(cudaEventRecord(stop, stream), "cudaEventRecord(MoE stop)");
        check(cudaEventSynchronize(stop), "cudaEventSynchronize(MoE stop)");
        float elapsed_ms = 0.0f;
        check(cudaEventElapsedTime(&elapsed_ms, start, stop),
              "cudaEventElapsedTime(MoE)");
        elapsed_ms /= static_cast<float>(iterations);

        std::vector<std::uint16_t> host_output(down_rows);
        check(cudaMemcpy(host_output.data(), output,
                         host_output.size() * sizeof(std::uint16_t),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(MoE output)");
        double checksum = 0.0;
        for (const auto value : host_output) checksum += fp32(value);
        const double active_weight_bytes =
            static_cast<double>(top_k) *
            (static_cast<double>(gate_elements + down_elements) / 2.0);
        std::cout << "format=W4A16_G128 policy="
                  << (policy ? policy : "warp") << " tokens=1 top_k=" << top_k
                  << " ms=" << std::fixed << std::setprecision(4)
                  << elapsed_ms << " active_weight_GB/s="
                  << std::setprecision(3)
                  << active_weight_bytes / (elapsed_ms * 1.0e6)
                  << " checksum=" << checksum << '\n';

        (void)cudaEventDestroy(stop);
        (void)cudaEventDestroy(start);
        (void)cudaStreamDestroy(stream);
        (void)cudaFree(output);
        (void)cudaFree(accumulation);
        (void)cudaFree(activated);
        (void)cudaFree(gate_scratch);
        (void)cudaFree(d_route_weights);
        (void)cudaFree(d_experts);
        (void)cudaFree(d_hidden);
        (void)cudaFree(d_down_scales);
        (void)cudaFree(d_gate_scales);
        (void)cudaFree(d_down_data);
        (void)cudaFree(d_gate_data);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "q38 MoE benchmark: " << error.what() << '\n';
        return 1;
    }
}
