#include "q38/cuda_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
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
          "cudaMalloc(benchmark upload)");
    check(cudaMemcpy(result, host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy(benchmark upload)");
    return result;
}

__device__ __forceinline__ float load_bf16(const std::uint16_t* source,
                                            std::uint64_t index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(source)[index]);
}

__device__ __forceinline__ void store_bf16(std::uint16_t* destination,
                                            std::uint64_t index,
                                            float value) {
    reinterpret_cast<__nv_bfloat16*>(destination)[index] =
        __float2bfloat16_rn(value);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ float block_sum(float value) {
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? partial[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    return value;
}

__device__ __forceinline__ float reference_weight(
    int bits, const void* data, const std::uint16_t* scales,
    std::uint32_t columns, std::uint32_t row, std::uint32_t column) {
    const auto element = static_cast<std::uint64_t>(row) * columns + column;
    const auto groups = columns / 128;
    const auto scale = load_bf16(
        scales, static_cast<std::uint64_t>(row) * groups + column / 128);
    if (bits == 8)
        return static_cast<float>(
                   static_cast<const std::int8_t*>(data)[element]) *
               scale;
    const auto packed = static_cast<const std::uint8_t*>(data)[element >> 1];
    int value = (element & 1) ? packed >> 4 : packed & 0x0f;
    if (value >= 8) value -= 16;
    return static_cast<float>(value) * scale;
}

__global__ void scalar_reference_kernel(
    int bits, const void* data, const std::uint16_t* scales,
    std::uint32_t rows, std::uint32_t columns, const std::uint16_t* input,
    std::uint16_t* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= rows) return;
    float sum = 0.0f;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        sum += load_bf16(input, column) *
               reference_weight(bits, data, scales, columns, row, column);
    sum = block_sum(sum);
    if (threadIdx.x == 0) store_bf16(output, row, sum);
}

template <typename Launch>
float measure(cudaStream_t stream, int warmup, int iterations, Launch launch) {
    for (int index = 0; index < warmup; ++index) launch();
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int index = 0; index < iterations; ++index) launch();
    check(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop),
          "cudaEventElapsedTime");
    (void)cudaEventDestroy(stop);
    (void)cudaEventDestroy(start);
    return elapsed_ms / static_cast<float>(iterations);
}

int parse_positive(const char* text, const char* label) {
    const auto value = std::strtol(text, nullptr, 10);
    if (value <= 0 || value > 1000000)
        throw std::invalid_argument(std::string("invalid ") + label);
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int bits = argc > 1 ? parse_positive(argv[1], "bits") : 8;
        const auto rows = static_cast<std::uint32_t>(
            argc > 2 ? parse_positive(argv[2], "rows") : 2560);
        const auto columns = static_cast<std::uint32_t>(
            argc > 3 ? parse_positive(argv[3], "columns") : 2560);
        const int iterations =
            argc > 4 ? parse_positive(argv[4], "iterations") : 500;
        if ((bits != 4 && bits != 8) || columns % 128 != 0)
            throw std::invalid_argument(
                "bits must be 4 or 8 and columns must be divisible by 128");

        std::vector<std::uint16_t> input(columns);
        for (std::uint32_t column = 0; column < columns; ++column)
            input[column] = bf16(
                std::sin(static_cast<float>(column) * 0.017f) * 0.5f);
        const auto elements = static_cast<std::size_t>(rows) * columns;
        std::vector<std::int8_t> logical(elements);
        const int minimum = bits == 8 ? -127 : -7;
        const int span = bits == 8 ? 255 : 15;
        for (std::size_t index = 0; index < elements; ++index)
            logical[index] = static_cast<std::int8_t>(
                minimum + static_cast<int>((index * 17 + index / columns) %
                                           span));
        std::vector<std::uint8_t> packed;
        if (bits == 4) {
            packed.resize(elements / 2);
            for (std::size_t index = 0; index < elements; index += 2) {
                const auto low = static_cast<unsigned>(logical[index]) & 0xfu;
                const auto high =
                    static_cast<unsigned>(logical[index + 1]) & 0xfu;
                packed[index / 2] =
                    static_cast<std::uint8_t>(low | (high << 4));
            }
        }
        const auto groups = columns / 128;
        std::vector<std::uint16_t> scales(
            static_cast<std::size_t>(rows) * groups);
        for (std::size_t index = 0; index < scales.size(); ++index)
            scales[index] = bf16((bits == 8 ? 0.003f : 0.04f) *
                                 (1.0f + static_cast<float>(index % 7) *
                                             0.0625f));

        void* device_data = bits == 8
                                ? static_cast<void*>(upload(logical))
                                : static_cast<void*>(upload(packed));
        auto* device_scales = upload(scales);
        auto* device_input = upload(input);
        std::uint16_t* optimized_output = nullptr;
        std::uint16_t* reference_output = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&optimized_output),
                         rows * sizeof(std::uint16_t)),
              "cudaMalloc(optimized output)");
        check(cudaMalloc(reinterpret_cast<void**>(&reference_output),
                         rows * sizeof(std::uint16_t)),
              "cudaMalloc(reference output)");
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate");

        const auto format =
            bits == 8 ? q38::DeviceWeightFormatV1::kW8A16SymG128
                      : q38::DeviceWeightFormatV1::kW4A16SymG128;
        const q38::CudaMatrixViewV1 matrix{format, device_data, device_scales,
                                           rows, columns, 128, false};
        const auto optimized_ms = measure(stream, 20, iterations, [&] {
            q38::cuda_gemv_bf16(matrix, device_input, optimized_output, 1,
                                reinterpret_cast<void*>(stream), 0);
        });
        const auto scalar_ms = measure(stream, 20, iterations, [&] {
            scalar_reference_kernel<<<rows, 256, 0, stream>>>(
                bits, device_data, device_scales, rows, columns, device_input,
                reference_output);
            check(cudaPeekAtLastError(), "scalar_reference_kernel");
        });

        std::vector<std::uint16_t> optimized(rows);
        std::vector<std::uint16_t> reference(rows);
        check(cudaMemcpy(optimized.data(), optimized_output,
                         rows * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(optimized result)");
        check(cudaMemcpy(reference.data(), reference_output,
                         rows * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(reference result)");
        float maximum_absolute = 0.0f;
        for (std::size_t index = 0; index < optimized.size(); ++index)
            maximum_absolute = std::max(
                maximum_absolute,
                std::fabs(fp32(optimized[index]) - fp32(reference[index])));

        const double bytes =
            (bits == 8 ? static_cast<double>(elements)
                       : static_cast<double>(elements) / 2.0) +
            static_cast<double>(scales.size() * sizeof(std::uint16_t)) +
            static_cast<double>(columns * sizeof(std::uint16_t));
        const double optimized_gbs = bytes / (optimized_ms * 1.0e6);
        const double scalar_gbs = bytes / (scalar_ms * 1.0e6);
        std::cout << "format=W" << bits << "A16_G128"
                  << " rows=" << rows << " columns=" << columns
                  << " optimized_ms=" << std::fixed << std::setprecision(4)
                  << optimized_ms << " scalar_ms=" << scalar_ms
                  << " speedup=" << std::setprecision(3)
                  << scalar_ms / optimized_ms
                  << " optimized_GB/s=" << optimized_gbs
                  << " scalar_GB/s=" << scalar_gbs
                  << " max_abs=" << maximum_absolute << '\n';

        (void)cudaStreamDestroy(stream);
        (void)cudaFree(reference_output);
        (void)cudaFree(optimized_output);
        (void)cudaFree(device_input);
        (void)cudaFree(device_scales);
        (void)cudaFree(device_data);
        return maximum_absolute <= 0.125f ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "q38 decode benchmark: " << error.what() << '\n';
        return 1;
    }
}
