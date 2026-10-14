#include "q38/cuda_kernels.h"
#include "q38/cuda_padded_gemm.h"
#include "q38/cuda_sampling.h"
#include "q38/cuda_moe.h"
#include "q38/cuda_moe_prefill.h"
#include "q38/cuda_transport.h"
#include "q38/cuda_gdn.h"
#include "q38/cuda_hyper.h"
#include "q38/cuda_ple.h"
#include "q38/cuda_qsa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

std::uint16_t bf16(float value) {
    __nv_bfloat16 converted = __float2bfloat16_rn(value);
    return *reinterpret_cast<std::uint16_t*>(&converted);
}

float fp32(std::uint16_t value) {
    return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&value));
}

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)),
          "cudaMalloc(test)");
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy(test upload)");
    return device;
}

template <typename T>
T* allocate(std::size_t count, bool zero = false) {
    T* result = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)),
          "cudaMalloc(test buffer)");
    if (zero)
        check(cudaMemset(result, 0, count * sizeof(T)),
              "cudaMemset(test buffer)");
    return result;
}

void require_device_bits(const std::uint16_t* actual,
                         const std::uint16_t* expected, std::size_t words,
                         cudaStream_t stream, const char* label,
                         std::size_t guard_words = 0) {
    std::vector<std::uint16_t> a(words + guard_words), b(words);
    check(cudaMemcpyAsync(a.data(), actual, a.size() * 2,
                          cudaMemcpyDeviceToHost, stream), label);
    check(cudaMemcpyAsync(b.data(), expected, b.size() * 2,
                          cudaMemcpyDeviceToHost, stream), label);
    check(cudaStreamSynchronize(stream), label);
    if (!std::equal(b.begin(), b.end(), a.begin()) ||
        !std::all_of(a.begin() + words, a.end(), [](auto v) { return v == 0x1234; }))
        throw std::runtime_error(std::string(label) + " bitwise/guard mismatch");
}

void test_padded_small_gemm(cudaStream_t stream) {
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("create small GEMM test handle");
    for (const unsigned columns : {128u, 320u, 640u, 2560u}) {
        const unsigned rows = columns == 320 ? 17 : 64;
        std::vector<std::uint16_t> weights(rows * columns);
        for (unsigned i = 0; i < weights.size(); ++i)
            weights[i] = bf16(float(int(i % 11) - 5) * .125f);
        auto* dw = upload(weights);
        auto* scratch = allocate<std::uint16_t>(8 * (columns + rows));
        for (const unsigned batch : {2u, 3u, 5u, 7u}) {
            std::vector<std::uint16_t> input(batch * columns);
            for (unsigned i = 0; i < input.size(); ++i)
                input[i] = bf16(float(int(i % 7) - 3) * .125f);
            auto* dx = upload(input);
            auto* dy = upload(std::vector<std::uint16_t>(batch * rows + 16, 0x1234));
            check(cudaStreamSynchronize(nullptr), "small GEMM fixture upload ready");
            check(cudaMemsetAsync(scratch, 0xff, 8 * (columns + rows) * 2, stream), "poison GEMM scratch");
            q38::cuda_padded_gemm_bf16(handle, dw, dx, dy + 8, scratch,
                scratch + 8 * columns, rows, columns, batch, stream);
            std::vector<std::uint16_t> actual(batch * rows + 16);
            check(cudaMemcpyAsync(actual.data(), dy, actual.size() * 2,
                cudaMemcpyDeviceToHost, stream), "copy small GEMM fixture");
            check(cudaStreamSynchronize(stream), "small GEMM fixture ready");
            for (unsigned t = 0; t < batch; ++t)
                for (unsigned row = 0; row < rows; ++row) {
                    float expected = 0;
                    for (unsigned k = 0; k < columns; ++k)
                        expected += fp32(input[t * columns + k]) * fp32(weights[row * columns + k]);
                    if (actual[8 + t * rows + row] != bf16(expected))
                        throw std::runtime_error("padded small GEMM differs from CPU matrix product");
                }
            for (unsigned guard = 0; guard < 8; ++guard)
                if (actual[guard] != 0x1234 || actual[8 + batch * rows + guard] != 0x1234)
                    throw std::runtime_error("padded small GEMM overwrote output guard");
            (void)cudaFree(dx); (void)cudaFree(dy);
        }
        (void)cudaFree(dw); (void)cudaFree(scratch);
    }
    (void)cublasDestroy(handle);
}

void test_paired_norm(cudaStream_t stream) {
    for (const auto tokens : {1u, 3u, 32u, 512u}) {
        const auto words = static_cast<std::size_t>(tokens) * q38::kQ38HyperWidth;
        std::vector<std::uint16_t> weights0(q38::kQ38HyperWidth), weights1(weights0.size());
        for (std::size_t i = 0; i < weights0.size(); ++i) {
            weights0[i] = bf16((static_cast<int>(i % 13) - 6) * 0.125f);
            weights1[i] = bf16((static_cast<int>(i % 17) - 8) * 0.125f);
        }
        auto* w0 = upload(weights0);
        auto* w1 = upload(weights1);
        auto* x0 = allocate<std::uint16_t>(words);
        auto* x1 = allocate<std::uint16_t>(words);
        auto* y0 = upload(std::vector<std::uint16_t>(words + 8, 0x1234));
        auto* y1 = upload(std::vector<std::uint16_t>(words + 8, 0x1234));
        auto* r0 = allocate<std::uint16_t>(words);
        auto* r1 = allocate<std::uint16_t>(words);
        const q38::CudaHyperNormV1 first{x0, w0, y0}, second{x1, w1, y1};
        bool rejected = false;
        try { q38::cuda_hyper_pair_group_rmsnorm_bf16(
            first, {x1, w1, y0 + 1}, tokens, stream, 0); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("paired norm accepted alias");
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "norm capture begin");
        q38::cuda_hyper_pair_group_rmsnorm_bf16(first, second, tokens, stream, 0);
        check(cudaStreamEndCapture(stream, &graph), "norm capture end");
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "norm instantiate");
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<std::uint16_t> a(words), b(words);
            for (std::size_t i = 0; i < words; ++i) {
                a[i] = pattern == 0 ? bf16((static_cast<int>(i % 193) - 96) / 97.0f)
                    : pattern == 1 ? bf16(std::ldexp((i & 1) ? -1.0f : 1.0f,
                                                    static_cast<int>(i % 41) - 20))
                    : static_cast<std::uint16_t>((i & 1 ? 0x8000 : 0) | (pattern == 3 ? 1 : 0));
            }
            std::reverse_copy(a.begin(), a.end(), b.begin());
            check(cudaMemcpyAsync(x0, a.data(), words * 2, cudaMemcpyHostToDevice, stream), "norm input0");
            check(cudaMemcpyAsync(x1, b.data(), words * 2, cudaMemcpyHostToDevice, stream), "norm input1");
            q38::cuda_hyper_group_rmsnorm_bf16(x0, w0, r0, tokens, stream, 0);
            q38::cuda_hyper_group_rmsnorm_bf16(x1, w1, r1, tokens, stream, 0);
            if (pattern & 1) check(cudaGraphLaunch(executable, stream), "norm replay");
            else q38::cuda_hyper_pair_group_rmsnorm_bf16(first, second, tokens, stream, 0);
            require_device_bits(y0, r0, words, stream, "paired key norm", 8);
            require_device_bits(y1, r1, words, stream, "paired query norm", 8);
        }
        check(cudaGraphExecDestroy(executable), "norm exec destroy");
        check(cudaGraphDestroy(graph), "norm graph destroy");
        for (auto* ptr : {w0, w1, x0, x1, y0, y1, r0, r1}) check(cudaFree(ptr), "norm free");
    }
}

void test_ple_residual(cudaStream_t stream) {
    constexpr std::size_t state_words = q38::kQ38HyperWidth * q38::kQ38PleConvState;
    for (const auto tokens : {1u, 3u, 32u, 512u}) {
        const auto words = static_cast<std::size_t>(tokens) * q38::kQ38HyperWidth;
        std::vector<std::uint16_t> weights(q38::kQ38HyperWidth * 4), state(state_words);
        for (std::size_t i = 0; i < weights.size(); ++i)
            weights[i] = bf16((static_cast<int>(i % 19) - 9) * 0.015625f);
        for (std::size_t i = 0; i < state.size(); ++i)
            state[i] = bf16((static_cast<int>(i % 23) - 11) * 0.03125f);
        auto* w = upload(weights);
        auto* ref_state = upload(state);
        auto* fused_state = upload(state);
        auto* gated = allocate<std::uint16_t>(words);
        auto* normalized = allocate<std::uint16_t>(words);
        auto* residual = upload(std::vector<std::uint16_t>(words + 8, 0x1234));
        auto* output = upload(std::vector<std::uint16_t>(words + 8, 0x1234));
        auto* temporary = allocate<std::uint16_t>(words);
        auto* expected = allocate<std::uint16_t>(words);
        bool rejected = false;
        try { q38::cuda_ple_conv_residual_bf16(gated, normalized, w, fused_state,
                residual, residual + 1, tokens, stream, 0); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("PLE residual accepted partial alias");
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "residual capture begin");
        q38::cuda_ple_conv_residual_bf16(gated, normalized, w, fused_state,
                                       residual, output, tokens, stream, 0);
        check(cudaStreamEndCapture(stream, &graph), "residual capture end");
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "residual instantiate");
        for (int step = 0; step < 4; ++step) {
            std::vector<std::uint16_t> g(words), n(words), r(words);
            for (std::size_t i = 0; i < words; ++i) {
                g[i] = bf16((static_cast<int>((i + step * 7) % 137) - 68) * 0.015625f);
                n[i] = bf16((static_cast<int>((i + step * 11) % 97) - 48) * 0.03125f);
                r[i] = bf16(-fp32(g[i]) + (static_cast<int>(i % 5) - 2) * 0.00390625f);
            }
            check(cudaMemcpyAsync(gated, g.data(), words * 2, cudaMemcpyHostToDevice, stream), "residual gated");
            check(cudaMemcpyAsync(normalized, n.data(), words * 2, cudaMemcpyHostToDevice, stream), "residual normalized");
            check(cudaMemcpyAsync(residual, r.data(), words * 2, cudaMemcpyHostToDevice, stream), "residual input");
            // Exercise both original entry points for a one-token prefill
            // tail, since the fused dispatcher uses the decode kernel there.
            if (tokens == 1 && (step & 1) == 0)
                q38::cuda_ple_conv_decode_bf16(gated, normalized, w, ref_state, temporary, stream, 0);
            else
                q38::cuda_ple_conv_prefill_bf16(gated, normalized, w, ref_state, temporary, tokens, stream, 0);
            q38::cuda_add_bf16(residual, temporary, expected, words, stream, 0);
            if (step == 1) check(cudaGraphLaunch(executable, stream), "residual replay");
            else q38::cuda_ple_conv_residual_bf16(gated, normalized, w, fused_state,
                      residual, step == 3 ? residual : output, tokens, stream, 0);
            require_device_bits(step == 3 ? residual : output, expected, words,
                                stream, "PLE residual output", 8);
            require_device_bits(fused_state, ref_state, state_words, stream, "PLE residual state");
        }
        check(cudaGraphExecDestroy(executable), "residual exec destroy");
        check(cudaGraphDestroy(graph), "residual graph destroy");
        for (auto* ptr : {w, ref_state, fused_state, gated, normalized, residual, output, temporary, expected})
            check(cudaFree(ptr), "residual free");
    }
}

void test_hc_gemv(cudaStream_t stream) {
    // Production HC shapes, inject rows, and odd-row tails. Each fixture is
    // compared with the existing GPU kernel, not a reimplemented CPU oracle.
    const std::pair<std::uint32_t, std::uint32_t> shapes[] = {
        {1, 320}, {17, 320}, {10240, 320},
        {4, 10240}, {17, 10240}, {320, 10240}};
    std::uint32_t random = 0x89abcdefu;
    auto next = [&] {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };
    for (const auto& shape : shapes) {
        const auto rows = shape.first, columns = shape.second;
        std::vector<std::uint16_t> weights(
            static_cast<std::size_t>(rows) * columns);
        for (auto& w : weights) {
            const auto bits = next();
            w = bf16(std::ldexp((static_cast<int>(bits % 511) - 255) / 256.0f,
                                static_cast<int>((bits >> 16) % 31) - 15));
        }
        auto* w = upload(weights);
        auto* x = allocate<std::uint16_t>(columns);
        auto* y = upload(std::vector<std::uint16_t>(rows + 8, 0x1234));
        auto* reference = allocate<std::uint16_t>(rows);
        q38::CudaMatrixViewV1 matrix{
            q38::DeviceWeightFormatV1::kPreserve, w, nullptr,
            rows, columns, 0, false};
        // Unsupported layouts must not enter the specialized path.
        auto unsupported = matrix;
        unsupported.preserved_f32 = true;
        if (q38::cuda_try_hc_gemv_bf16(unsupported, x, y, stream, 0))
            throw std::runtime_error("HC kernel accepted FP32 weights");
        unsupported = matrix;
        unsupported.columns = 319;
        if (q38::cuda_try_hc_gemv_bf16(unsupported, x, y, stream, 0))
            throw std::runtime_error("HC kernel accepted unsupported columns");
        for (int mode = 0; mode < 4; ++mode) {
            const auto activation = mode == 0 ? q38::CudaHcActivation::kNone
                : mode == 1 ? q38::CudaHcActivation::kSilu
                            : q38::CudaHcActivation::kSigmoid;
            const float divisor = (mode == 1 || mode == 3) ? 4.0f : 1.0f;
            const float scale = mode == 3 ? 2.0f : 1.0f;
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                  "HC capture begin");
            if (!q38::cuda_try_hc_gemv_bf16(matrix, x, y, stream, 0, activation, divisor, scale))
                throw std::runtime_error("HC kernel rejected supported shape");
            check(cudaStreamEndCapture(stream, &graph), "HC capture end");
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                  "HC graph instantiate");
            for (int pattern = 0; pattern < 4; ++pattern) {
                std::vector<std::uint16_t> input(columns);
                for (std::uint32_t col = 0; col < columns; ++col) {
                    const auto bits = next();
                    input[col] = pattern == 0 ? bf16((static_cast<int>(bits % 997) - 498) / 499.0f)
                        : pattern == 1 ? bf16(std::ldexp((col & 1) ? -1.0f : 1.0f,
                                               static_cast<int>(col % 41) - 20))
                        : pattern == 2 ? static_cast<std::uint16_t>((col & 1) ? 0x8000 : 0)
                        : static_cast<std::uint16_t>((col & 1) ? 0x8001 : 1);
                }
                check(cudaMemcpyAsync(x, input.data(), columns * sizeof(*x),
                    cudaMemcpyHostToDevice, stream), "HC input");
                q38::cuda_gemv_bf16(matrix, x, reference, 1, stream, 0);
                if (mode == 1)
                    q38::cuda_hyper_silu_scaled_bf16(reference, reference, 1,
                        rows, divisor, stream, 0);
                else if (mode != 0)
                    q38::cuda_hyper_sigmoid_bf16(reference, reference, 1,
                        rows, divisor, scale, stream, 0);
                for (int replay = 0; replay < 2; ++replay) {
                    if (replay) check(cudaGraphLaunch(executable, stream), "HC replay");
                    else (void)q38::cuda_try_hc_gemv_bf16(matrix, x, y, stream, 0, activation, divisor, scale);
                    std::vector<std::uint16_t> actual(rows + 8), expected(rows);
                    check(cudaMemcpyAsync(actual.data(), y, actual.size() * sizeof(*y),
                        cudaMemcpyDeviceToHost, stream), "HC result");
                    check(cudaMemcpyAsync(expected.data(), reference,
                        expected.size() * sizeof(*reference), cudaMemcpyDeviceToHost,
                        stream), "HC reference");
                    check(cudaStreamSynchronize(stream), "HC sync");
                    if (!std::equal(expected.begin(), expected.end(), actual.begin()))
                        throw std::runtime_error("HC GEMV is not bitwise equal to reference");
                    if (!std::all_of(actual.begin() + rows, actual.end(),
                                     [](auto value) { return value == 0x1234; }))
                        throw std::runtime_error("HC GEMV overwrote tail guard");
                }
            }
            check(cudaGraphExecDestroy(executable), "HC graph exec destroy");
            check(cudaGraphDestroy(graph), "HC graph destroy");
        }
        for (void* ptr : {static_cast<void*>(w), static_cast<void*>(x),
                         static_cast<void*>(y), static_cast<void*>(reference)})
            check(cudaFree(ptr), "HC test free");
    }
}

void test_hc_grouped(cudaStream_t stream) {
    for (const auto columns : {320u, 10240u}) {
        for (const auto rows : {1u, 319u, 320u}) {
            const auto inject_rows = rows == 320 ? 4u : 3u;
            std::vector<std::uint16_t> weights(
                static_cast<std::size_t>(rows + inject_rows) * columns);
            for (std::size_t j = 0; j < weights.size(); ++j)
                weights[j] = bf16(std::ldexp(
                    (static_cast<int>((j * 73u + 29u) % 509u) - 254) / 255.0f,
                    static_cast<int>(j % 17) - 8));
            auto* w = upload(weights);
            auto* x = allocate<std::uint16_t>(columns);
            auto* y = upload(std::vector<std::uint16_t>(rows + 8, 0x1234));
            auto* z = upload(std::vector<std::uint16_t>(inject_rows + 8, 0x1234));
            auto* ref_y = allocate<std::uint16_t>(rows);
            auto* ref_z = allocate<std::uint16_t>(inject_rows);
            q38::CudaHcProjectionV1 down{
                {q38::DeviceWeightFormatV1::kPreserve, w, nullptr,
                 rows, columns, 0, false}, y, q38::CudaHcActivation::kSilu, 4, 1};
            q38::CudaHcProjectionV1 inject{
                {q38::DeviceWeightFormatV1::kPreserve,
                 w + static_cast<std::size_t>(rows) * columns, nullptr,
                 inject_rows, columns, 0, false}, z,
                q38::CudaHcActivation::kSigmoid, 4, 2};
            auto invalid = inject;
            invalid.matrix.preserved_f32 = true;
            if (q38::cuda_try_hc_grouped_bf16(down, invalid, x, stream, 0))
                throw std::runtime_error("grouped HC accepted unsupported second matrix");
            invalid = inject;
            invalid.output = y + rows - 1;
            bool rejected = false;
            try { (void)q38::cuda_try_hc_grouped_bf16(down, invalid, x, stream, 0); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("grouped HC accepted overlapping outputs");
            std::vector<std::uint16_t> untouched(rows + 8);
            check(cudaMemcpy(untouched.data(), y, untouched.size() * 2,
                             cudaMemcpyDeviceToHost), "HC unsupported output");
            if (!std::all_of(untouched.begin(), untouched.end(),
                             [](auto v) { return v == 0x1234; }))
                throw std::runtime_error("grouped HC enqueued partial work on rejection");
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                  "grouped HC capture begin");
            if (!q38::cuda_try_hc_grouped_bf16(down, inject, x, stream, 0))
                throw std::runtime_error("grouped HC rejected supported matrices");
            check(cudaStreamEndCapture(stream, &graph), "grouped HC capture end");
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                  "grouped HC graph instantiate");
            for (int pattern = 0; pattern < 4; ++pattern) {
                std::vector<std::uint16_t> input(columns);
                for (std::uint32_t j = 0; j < columns; ++j)
                    input[j] = pattern == 0 ? bf16((static_cast<int>(j % 97) - 48) / 49.0f)
                        : pattern == 1 ? bf16(std::ldexp((j & 1) ? -1.0f : 1.0f,
                                                        static_cast<int>(j % 41) - 20))
                        : pattern == 2 ? static_cast<std::uint16_t>((j & 1) ? 0x8000 : 0)
                        : static_cast<std::uint16_t>((j & 1) ? 0x8001 : 1);
                check(cudaMemcpyAsync(x, input.data(), columns * 2,
                    cudaMemcpyHostToDevice, stream), "grouped HC input");
                q38::cuda_gemv_bf16(down.matrix, x, ref_y, 1, stream, 0);
                q38::cuda_hyper_silu_scaled_bf16(ref_y, ref_y, 1, rows, 4, stream, 0);
                q38::cuda_gemv_bf16(inject.matrix, x, ref_z, 1, stream, 0);
                q38::cuda_hyper_sigmoid_bf16(ref_z, ref_z, 1, inject_rows, 4, 2, stream, 0);
                for (int replay = 0; replay < 2; ++replay) {
                    if (replay) check(cudaGraphLaunch(executable, stream), "grouped HC replay");
                    else (void)q38::cuda_try_hc_grouped_bf16(down, inject, x, stream, 0);
                    for (int which = 0; which < 2; ++which) {
                        const auto count = which ? inject_rows : rows;
                        std::vector<std::uint16_t> actual(count + 8), expected(count);
                        check(cudaMemcpyAsync(actual.data(), which ? z : y,
                            actual.size() * 2, cudaMemcpyDeviceToHost, stream), "grouped HC result");
                        check(cudaMemcpyAsync(expected.data(), which ? ref_z : ref_y,
                            expected.size() * 2, cudaMemcpyDeviceToHost, stream), "grouped HC reference");
                        check(cudaStreamSynchronize(stream), "grouped HC sync");
                        if (!std::equal(expected.begin(), expected.end(), actual.begin()) ||
                            !std::all_of(actual.begin() + count, actual.end(),
                                         [](auto v) { return v == 0x1234; }))
                            throw std::runtime_error("grouped HC parity/tail failure");
                    }
                }
            }
            check(cudaGraphExecDestroy(executable), "grouped HC exec destroy");
            check(cudaGraphDestroy(graph), "grouped HC graph destroy");
            for (auto* ptr : {w, x, y, z, ref_y, ref_z})
                check(cudaFree(ptr), "grouped HC free");
        }
    }
}

void test_multi_projection(cudaStream_t stream,
                           std::uint32_t columns = 256,
                           std::uint32_t rows = 13) {
    std::vector<std::uint16_t> input(columns), scales(rows * (columns / 128));
    std::vector<std::int8_t> w8(rows * columns);
    std::vector<std::uint8_t> w4(rows * columns / 2);
    std::vector<std::uint16_t> preserved(rows * columns);
    std::vector<float> f32(rows * columns);
    for (std::uint32_t i = 0; i < columns; ++i)
        input[i] = bf16((static_cast<int>(i % 17) - 8) * 0.125f);
    for (std::size_t i = 0; i < scales.size(); ++i)
        scales[i] = bf16(0.01f * (i % 7 + 1));
    for (std::size_t i = 0; i < w8.size(); ++i) {
        w8[i] = static_cast<std::int8_t>(static_cast<int>(i % 31) - 15);
        f32[i] = (static_cast<int>(i % 23) - 11) * 0.03125f;
        preserved[i] = bf16(f32[i]);
    }
    for (std::size_t i = 0; i < w4.size(); ++i)
        w4[i] = static_cast<std::uint8_t>((i % 16) | ((15 - i % 16) << 4));
    auto* x = upload(input);
    auto* scales_device = upload(scales);
    auto* w8_device = upload(w8);
    auto* w4_device = upload(w4);
    auto* bf16_device = upload(preserved);
    auto* f32_device = upload(f32);
    q38::CudaProjectionPackV1 pack;
    pack.count = 4;
    pack.matrices[0] = {q38::DeviceWeightFormatV1::kW8A16SymG128,
                        w8_device, scales_device, rows, columns, 128, false};
    pack.matrices[1] = {q38::DeviceWeightFormatV1::kW4A16SymG128,
                        w4_device, scales_device, rows, columns, 128, false};
    pack.matrices[2] = {q38::DeviceWeightFormatV1::kPreserve,
                        bf16_device, nullptr, rows, columns, 0, false};
    pack.matrices[3] = {q38::DeviceWeightFormatV1::kPreserve,
                        f32_device, nullptr, rows, columns, 0, true};
    auto* reference = allocate<std::uint16_t>(rows);
    for (auto& out : pack.outputs)
        out = upload(std::vector<std::uint16_t>(rows + 4, 0x1234));
    q38::cuda_multi_projection_decode_bf16(pack, x, stream, 0);
    for (std::uint32_t i = 0; i < pack.count; ++i) {
        q38::cuda_gemv_bf16(pack.matrices[i], x, reference, 1, stream, 0);
        std::vector<std::uint16_t> a(rows + 4), b(rows);
        check(cudaMemcpyAsync(a.data(), pack.outputs[i], a.size() * 2,
            cudaMemcpyDeviceToHost, stream), "projection test output");
        check(cudaMemcpyAsync(b.data(), reference, b.size() * 2,
            cudaMemcpyDeviceToHost, stream), "projection test reference");
        check(cudaStreamSynchronize(stream), "projection test synchronize");
        if (!std::equal(b.begin(), b.end(), a.begin()))
            throw std::runtime_error("fused projection differs from separate GEMV");
        if (!std::all_of(a.begin() + rows, a.end(),
                         [](auto value) { return value == 0x1234; }))
            throw std::runtime_error("fused projection overwrote tail canary");
    }
    for (auto* out : pack.outputs) (void)cudaFree(out);
    for (void* p : {static_cast<void*>(x), static_cast<void*>(scales_device),
                   static_cast<void*>(w8_device), static_cast<void*>(w4_device),
                   static_cast<void*>(bf16_device), static_cast<void*>(f32_device),
                   static_cast<void*>(reference)}) (void)cudaFree(p);
}

void test_compact_sampling(cudaStream_t stream) {
    constexpr std::uint32_t size = 4099;
    q38::CudaSamplingWorkspace scratch(0, size);
    const auto allocated = scratch.bytes();
    std::vector<std::uint16_t> logits(size);
    for (std::uint32_t i = 0; i < size; ++i)
        logits[i] = bf16((static_cast<int>((i * 37) % 131) - 65) * 0.0625f);
    logits[0] = 0x7fc1;  // NaN is excluded like the host sampler.
    logits[1] = 0x7f80;
    logits[2] = 0xff80;
    logits[3] = 0x8000;
    logits[4] = 0;
    auto* device_logits = upload(logits);
    for (float temperature : {0.7f, 1.0f, 1000.0f}) {
        std::vector<std::int32_t> expected(size);
        std::iota(expected.begin(), expected.end(), 0);
        auto value = [&](std::int32_t id) {
            const float raw = fp32(logits[id]);
            return (std::isfinite(raw) ? raw : -std::numeric_limits<float>::infinity()) /
                   temperature;
        };
        std::sort(expected.begin(), expected.end(), [&](auto a, auto b) {
            return value(a) > value(b) || (value(a) == value(b) && a < b);
        });
        for (std::uint32_t k : {1u, 10u, 64u, size}) {
            scratch.select(device_logits, k, temperature, stream);
            std::vector<std::int32_t> ids(k);
            std::vector<std::uint16_t> selected(k);
            check(cudaMemcpyAsync(ids.data(), scratch.ids(), k * sizeof(std::int32_t),
                cudaMemcpyDeviceToHost, stream), "compact sampling IDs");
            check(cudaMemcpyAsync(selected.data(), scratch.logits(), k * 2,
                cudaMemcpyDeviceToHost, stream), "compact sampling logits");
            check(cudaStreamSynchronize(stream), "compact sampling synchronize");
            if (!std::equal(ids.begin(), ids.end(), expected.begin()))
                throw std::runtime_error("compact sampling top-k/tie order differs");
            for (std::uint32_t i = 0; i < k; ++i)
                if (selected[i] != logits[ids[i]])
                    throw std::runtime_error("compact sampling altered raw logits");
            if (scratch.bytes() != allocated)
                throw std::runtime_error("compact sampling grew its workspace");
        }
    }
    (void)cudaFree(device_logits);
}

void require_close(const std::vector<std::uint16_t>& left,
                   const std::vector<std::uint16_t>& right,
                   float tolerance, const char* label) {
    if (left.size() != right.size())
        throw std::runtime_error(std::string(label) + " size differs");
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (std::fabs(fp32(left[index]) - fp32(right[index])) > tolerance)
            throw std::runtime_error(std::string(label) + " differs at " +
                                     std::to_string(index));
    }
}

std::vector<std::uint8_t> tile_w4_16x16(
    const std::vector<std::uint8_t>& row_major, std::uint32_t rows,
    std::uint32_t columns) {
    if (rows % 16 != 0 || columns % 16 != 0 ||
        row_major.size() != static_cast<std::size_t>(rows) * columns / 2)
        throw std::invalid_argument("invalid W4 tile test fixture");
    std::vector<std::uint8_t> tiled(row_major.size());
    std::size_t output = 0;
    for (std::uint32_t row_base = 0; row_base < rows; row_base += 16) {
        for (std::uint32_t column_base = 0; column_base < columns;
             column_base += 16) {
            for (std::uint32_t row = 0; row < 16; ++row) {
                const auto source =
                    static_cast<std::size_t>(row_base + row) *
                        (columns / 2) +
                    column_base / 2;
                std::copy_n(row_major.begin() + source, 8,
                            tiled.begin() + output);
                output += 8;
            }
        }
    }
    return tiled;
}

void test_quantized_decode_gemv(q38::DeviceWeightFormatV1 format,
                                cudaStream_t stream) {
    constexpr std::uint32_t rows = 13;
    constexpr std::uint32_t columns = 384;
    constexpr std::uint32_t group_size = 128;
    constexpr std::uint32_t groups = columns / group_size;
    const bool w8 = format == q38::DeviceWeightFormatV1::kW8A16SymG128;
    const float alpha = w8 ? 1.0f : 0.625f;
    const bool accumulate = !w8;

    std::vector<std::uint16_t> input(columns);
    for (std::uint32_t column = 0; column < columns; ++column)
        input[column] = bf16(
            std::sin(static_cast<float>(column) * 0.071f) * 0.75f);
    std::vector<std::uint16_t> scales(rows * groups);
    for (std::uint32_t row = 0; row < rows; ++row)
        for (std::uint32_t group = 0; group < groups; ++group)
            scales[static_cast<std::size_t>(row) * groups + group] = bf16(
                (w8 ? 0.003f : 0.0375f) *
                (1.0f + static_cast<float>((row + group) % 5) * 0.125f));

    std::vector<std::int8_t> logical_weights(
        static_cast<std::size_t>(rows) * columns);
    for (std::size_t index = 0; index < logical_weights.size(); ++index) {
        const int span = w8 ? 255 : 15;
        const int minimum = w8 ? -127 : -7;
        logical_weights[index] = static_cast<std::int8_t>(
            minimum + static_cast<int>((index * 37 + index / columns * 11) %
                                       span));
    }
    std::vector<std::uint8_t> packed;
    const void* device_weight_data = nullptr;
    std::int8_t* device_w8 = nullptr;
    std::uint8_t* device_w4 = nullptr;
    if (w8) {
        device_w8 = upload(logical_weights);
        device_weight_data = device_w8;
    } else {
        packed.resize(logical_weights.size() / 2);
        for (std::size_t index = 0; index < logical_weights.size(); index += 2) {
            const auto low = static_cast<unsigned>(logical_weights[index]) & 0xfu;
            const auto high =
                static_cast<unsigned>(logical_weights[index + 1]) & 0xfu;
            packed[index / 2] = static_cast<std::uint8_t>(low | (high << 4));
        }
        device_w4 = upload(packed);
        device_weight_data = device_w4;
    }
    auto* device_scales = upload(scales);
    auto* device_input = upload(input);
    std::vector<std::uint16_t> initial(rows, bf16(0.25f));
    auto* device_output = upload(initial);
    q38::CudaMatrixViewV1 matrix{format, device_weight_data, device_scales,
                                 rows, columns, group_size, false};
    q38::cuda_gemv_bf16(matrix, device_input, device_output, 1,
                        reinterpret_cast<void*>(stream), 0, alpha,
                        accumulate);
    std::vector<std::uint16_t> actual(rows);
    check(cudaMemcpyAsync(actual.data(), device_output,
                          actual.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(quantized decode GEMV)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(quantized decode GEMV)");

    for (std::uint32_t row = 0; row < rows; ++row) {
        float expected = accumulate ? fp32(initial[row]) : 0.0f;
        float dot = 0.0f;
        for (std::uint32_t column = 0; column < columns; ++column) {
            const auto scale = fp32(
                scales[static_cast<std::size_t>(row) * groups +
                       column / group_size]);
            dot += fp32(input[column]) *
                   static_cast<float>(logical_weights[
                       static_cast<std::size_t>(row) * columns + column]) *
                   scale;
        }
        expected += alpha * dot;
        if (std::fabs(fp32(actual[row]) - expected) > 0.075f)
            throw std::runtime_error(
                std::string(w8 ? "W8" : "W4") +
                " optimized decode GEMV differs at row " +
                std::to_string(row));
    }

    (void)cudaFree(device_output);
    (void)cudaFree(device_input);
    (void)cudaFree(device_scales);
    if (device_w4) (void)cudaFree(device_w4);
    if (device_w8) (void)cudaFree(device_w8);
}

void test_decode_router_topk(cudaStream_t stream) {
    constexpr std::uint32_t tokens = 2;
    constexpr std::uint32_t experts = 512;
    constexpr std::uint32_t top_k = 10;
    std::vector<std::uint16_t> logits(tokens * experts);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            float value = std::sin(static_cast<float>(expert) * 0.037f +
                                   static_cast<float>(token) * 0.31f);
            if (expert == 3 || expert == 7) value = 2.0f;
            logits[static_cast<std::size_t>(token) * experts + expert] =
                bf16(value);
        }
    }
    auto* device_logits = upload(logits);
    auto* device_ids = allocate<std::int32_t>(tokens * top_k);
    auto* device_weights = allocate<float>(tokens * top_k);
    q38::cuda_topk_router_bf16(
        device_logits, device_ids, device_weights, tokens, experts, top_k,
        true, reinterpret_cast<void*>(stream), 0);
    std::vector<std::int32_t> ids(tokens * top_k);
    std::vector<float> weights(tokens * top_k);
    check(cudaMemcpyAsync(ids.data(), device_ids,
                          ids.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(router ids)");
    check(cudaMemcpyAsync(weights.data(), device_weights,
                          weights.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(router weights)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(router top-k)");

    for (std::uint32_t token = 0; token < tokens; ++token) {
        std::vector<std::pair<float, std::int32_t>> ordered;
        ordered.reserve(experts);
        for (std::uint32_t expert = 0; expert < experts; ++expert)
            ordered.emplace_back(
                fp32(logits[static_cast<std::size_t>(token) * experts +
                            expert]),
                static_cast<std::int32_t>(expert));
        std::sort(ordered.begin(), ordered.end(), [](const auto& left,
                                                     const auto& right) {
            if (left.first != right.first) return left.first > right.first;
            return left.second < right.second;
        });
        float denominator = 0.0f;
        for (std::uint32_t rank = 0; rank < top_k; ++rank)
            denominator += std::exp(ordered[rank].first - ordered[0].first);
        for (std::uint32_t rank = 0; rank < top_k; ++rank) {
            const auto index = static_cast<std::size_t>(token) * top_k + rank;
            if (ids[index] != ordered[rank].second)
                throw std::runtime_error("decode router top-k ids differ");
            const float expected =
                std::exp(ordered[rank].first - ordered[0].first) / denominator;
            if (std::fabs(weights[index] - expected) > 1.0e-5f)
                throw std::runtime_error("decode router top-k weights differ");
        }
    }

    (void)cudaFree(device_weights);
    (void)cudaFree(device_ids);
    (void)cudaFree(device_logits);
}

std::vector<std::uint8_t> make_quantized_matrix(
    std::size_t elements, std::uint32_t salt,
    q38::DeviceWeightFormatV1 format) {
    const bool w8 = format == q38::DeviceWeightFormatV1::kW8A16SymG128;
    std::vector<std::uint8_t> result(w8 ? elements : elements / 2);
    for (std::size_t index = 0; index < elements; index += w8 ? 1 : 2) {
        const int low_value =
            (w8 ? -127 : -7) +
            static_cast<int>((index * 13 + salt) % (w8 ? 255 : 15));
        if (w8) {
            result[index] = static_cast<std::uint8_t>(
                static_cast<std::int8_t>(low_value));
            continue;
        }
        const int high_value =
            -7 + static_cast<int>(((index + 1) * 13 + salt) % 15);
        const auto low = static_cast<unsigned>(low_value) & 0xfu;
        const auto high = static_cast<unsigned>(high_value) & 0xfu;
        result[index / 2] =
            static_cast<std::uint8_t>(low | (high << 4));
    }
    return result;
}

void test_moe_decode_parity(q38::DeviceWeightFormatV1 format,
                            cudaStream_t stream) {
    constexpr std::uint32_t top_k = q38::kQ38MoeTopK;
    constexpr std::uint32_t gate_rows = 2 * q38::kQ38MoeIntermediate;
    constexpr std::uint32_t gate_columns = q38::kQ38HiddenWidth;
    constexpr std::uint32_t down_rows = q38::kQ38HiddenWidth;
    constexpr std::uint32_t down_columns = q38::kQ38MoeIntermediate;
    const auto gate_elements =
        static_cast<std::size_t>(gate_rows) * gate_columns;
    const auto down_elements =
        static_cast<std::size_t>(down_rows) * down_columns;
    const bool w8 = format == q38::DeviceWeightFormatV1::kW8A16SymG128;
    auto gate_data = make_quantized_matrix(gate_elements, 3, format);
    auto down_data = make_quantized_matrix(down_elements, 7, format);
    std::vector<std::uint16_t> gate_scales(
        gate_rows * (gate_columns / 128), bf16(w8 ? 0.00075f : 0.0125f));
    std::vector<std::uint16_t> down_scales(
        down_rows * (down_columns / 128), bf16(w8 ? 0.001f : 0.0175f));
    std::vector<std::uint16_t> hidden(gate_columns);
    for (std::size_t index = 0; index < hidden.size(); ++index)
        hidden[index] = bf16(
            std::sin(static_cast<float>(index) * 0.013f) * 0.2f);
    std::vector<std::uint16_t> hidden_pair = hidden;
    hidden_pair.insert(hidden_pair.end(), hidden.begin(), hidden.end());
    std::vector<std::int32_t> experts(top_k, 0);
    std::vector<std::int32_t> expert_pair = experts;
    expert_pair.insert(expert_pair.end(), experts.begin(), experts.end());
    std::vector<float> route_weights(top_k, 1.0f / top_k);
    std::vector<float> route_weight_pair = route_weights;
    route_weight_pair.insert(route_weight_pair.end(), route_weights.begin(),
                             route_weights.end());

    auto* d_gate_data = upload(gate_data);
    auto* d_down_data = upload(down_data);
    auto* d_gate_scales = upload(gate_scales);
    auto* d_down_scales = upload(down_scales);
    auto* d_hidden = upload(hidden);
    auto* d_hidden_pair = upload(hidden_pair);
    auto* d_experts = upload(experts);
    auto* d_expert_pair = upload(expert_pair);
    auto* d_route_weights = upload(route_weights);
    auto* d_route_weight_pair = upload(route_weight_pair);

    q38::DeviceTensorV1 gate_descriptor;
    gate_descriptor.format = format;
    gate_descriptor.group_size = 128;
    gate_descriptor.shape = {q38::kQ38MoeExperts, gate_rows, gate_columns};
    q38::DeviceTensorV1 down_descriptor;
    down_descriptor.format = format;
    down_descriptor.group_size = 128;
    down_descriptor.shape = {q38::kQ38MoeExperts, down_rows, down_columns};
    q38::CudaTensorViewV1 gate_view{&gate_descriptor, d_gate_data,
                                    d_gate_scales};
    q38::CudaTensorViewV1 down_view{&down_descriptor, d_down_data,
                                    d_down_scales};

    const auto gate_route_words =
        static_cast<std::size_t>(top_k) * gate_rows;
    const auto activated_route_words =
        static_cast<std::size_t>(top_k) * down_columns;
    auto* gate_scratch = allocate<std::uint16_t>(gate_route_words);
    auto* gate_scratch_pair = allocate<std::uint16_t>(2 * gate_route_words);
    auto* activated = allocate<std::uint16_t>(activated_route_words);
    auto* activated_pair =
        allocate<std::uint16_t>(2 * activated_route_words);
    auto* accumulation = allocate<float>(down_rows);
    auto* accumulation_pair = allocate<float>(2 * down_rows);
    auto* output = allocate<std::uint16_t>(down_rows);
    auto* output_pair = allocate<std::uint16_t>(2 * down_rows);

    q38::cuda_moe_routed_bf16(
        gate_view, down_view, d_hidden, d_experts, d_route_weights, 1, top_k,
        gate_scratch, activated, accumulation, output,
        reinterpret_cast<void*>(stream), 0);
    q38::cuda_moe_routed_bf16(
        gate_view, down_view, d_hidden_pair, d_expert_pair, d_route_weight_pair,
        2, top_k, gate_scratch_pair, activated_pair, accumulation_pair,
        output_pair, reinterpret_cast<void*>(stream), 0);
    std::vector<std::uint16_t> decode(down_rows);
    std::vector<std::uint16_t> reference(down_rows);
    check(cudaMemcpyAsync(decode.data(), output,
                          decode.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(MoE decode)");
    check(cudaMemcpyAsync(reference.data(), output_pair,
                          reference.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(MoE reference)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(MoE parity)");
    const std::string parity_label =
        std::string(w8 ? "W8" : "W4") + " MoE decode parity";
    require_close(decode, reference, 0.075f, parity_label.c_str());

    // Expert parallel reduction must sum FP32 partials before its sole BF16
    // rounding boundary. Mask complementary routes on the same fixture and
    // verify reconstruction of the complete routed result.
    auto* partial = allocate<float>(down_rows);
    std::vector<float> full_f32(down_rows), partial0(down_rows), partial1(down_rows);
    const auto run_partial = [&](const std::vector<std::int32_t>& ids,
                                  std::vector<float>& values) {
        check(cudaMemcpyAsync(d_experts, ids.data(), top_k * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice, stream), "upload EP mask");
        q38::cuda_moe_routed_bf16(
            gate_view, down_view, d_hidden, d_experts, d_route_weights, 1, top_k,
            gate_scratch, activated, accumulation, output,
            reinterpret_cast<void*>(stream), 0, partial);
        check(cudaMemcpyAsync(values.data(), partial, down_rows * sizeof(float),
                              cudaMemcpyDeviceToHost, stream), "read EP partial");
        check(cudaStreamSynchronize(stream), "wait EP partial");
    };
    run_partial(experts, full_f32);
    auto mask0 = experts, mask1 = experts;
    for (std::uint32_t i = 0; i < top_k; ++i) {
        if (i % 2) mask0[i] = -1;
        else mask1[i] = -1;
    }
    run_partial(mask0, partial0);
    run_partial(mask1, partial1);
    for (std::uint32_t i = 0; i < down_rows; ++i) {
        if (bf16(full_f32[i]) != decode[i])
            throw std::runtime_error("EP FP32 changed unsharded rounding");
        if (std::fabs(partial0[i] + partial1[i] - full_f32[i]) >
            1e-5f * std::max(1.0f, std::fabs(full_f32[i])))
            throw std::runtime_error("EP masked FP32 reconstruction differs");
    }
    (void)cudaFree(partial);

    (void)cudaFree(output_pair);
    (void)cudaFree(output);
    (void)cudaFree(accumulation_pair);
    (void)cudaFree(accumulation);
    (void)cudaFree(activated_pair);
    (void)cudaFree(activated);
    (void)cudaFree(gate_scratch_pair);
    (void)cudaFree(gate_scratch);
    (void)cudaFree(d_route_weight_pair);
    (void)cudaFree(d_route_weights);
    (void)cudaFree(d_expert_pair);
    (void)cudaFree(d_experts);
    (void)cudaFree(d_hidden_pair);
    (void)cudaFree(d_hidden);
    (void)cudaFree(d_down_scales);
    (void)cudaFree(d_gate_scales);
    (void)cudaFree(d_down_data);
    (void)cudaFree(d_gate_data);
}

void verify_moe_route_plan(std::uint32_t tokens, bool repeated_expert,
                           cudaStream_t stream) {
    const std::uint32_t routes = q38::q38_moe_prefill_routes_v1(tokens);
    std::vector<std::int32_t> experts(routes);
    for (std::uint32_t assignment = 0; assignment < routes; ++assignment) {
        experts[assignment] = repeated_expert
                                  ? 7
                                  : static_cast<std::int32_t>(
                                        (assignment * 73u +
                                         (assignment / q38::kQ38RouteTopK) *
                                             17u) %
                                        q38::kQ38RouteExperts);
    }
    auto* device_experts = upload(experts);
    const std::size_t plan_bytes =
        q38::q38_moe_route_plan_bytes_v1(tokens);
    auto* device_allocation = allocate<std::uint8_t>(plan_bytes);
    const auto plan = q38::cuda_moe_route_plan_storage_v1(
        device_allocation, plan_bytes, tokens);

    q38::cuda_moe_build_route_plan_v1(
        device_experts, tokens, plan, reinterpret_cast<void*>(stream), 0);
    std::vector<std::uint8_t> first(plan_bytes);
    check(cudaMemcpyAsync(first.data(), device_allocation, plan_bytes,
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan first run)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(route plan first run)");
    check(cudaMemsetAsync(device_allocation, 0xa5, plan_bytes, stream),
          "cudaMemset(route plan determinism)");
    q38::cuda_moe_build_route_plan_v1(
        device_experts, tokens, plan, reinterpret_cast<void*>(stream), 0);
    std::vector<std::uint8_t> second(plan_bytes);
    check(cudaMemcpyAsync(second.data(), device_allocation, plan_bytes,
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan second run)");

    q38::Q38RoutePlanHeaderV1 header{};
    std::vector<std::uint16_t> counts(q38::kQ38RouteExperts);
    std::vector<std::uint32_t> offsets(q38::kQ38RouteExperts + 1);
    std::vector<std::uint32_t> task_offsets(q38::kQ38RouteExperts + 1);
    std::vector<std::uint32_t> packed(routes);
    std::vector<std::uint32_t> inverse(routes);
    std::vector<q38::Q38ExpertMmqTaskV1> tasks(
        q38::q38_moe_prefill_max_tasks_v1(tokens));
    check(cudaMemcpyAsync(&header, plan.header, sizeof(header),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan header)");
    check(cudaMemcpyAsync(counts.data(), plan.expert_counts,
                          counts.size() * sizeof(counts.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan counts)");
    check(cudaMemcpyAsync(offsets.data(), plan.expert_offsets,
                          offsets.size() * sizeof(offsets.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan offsets)");
    check(cudaMemcpyAsync(task_offsets.data(), plan.expert_task_offsets,
                          task_offsets.size() * sizeof(task_offsets.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan task offsets)");
    check(cudaMemcpyAsync(packed.data(), plan.packed_assignment,
                          packed.size() * sizeof(packed.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan packed assignments)");
    check(cudaMemcpyAsync(inverse.data(), plan.assignment_to_packed,
                          inverse.size() * sizeof(inverse.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan inverse assignments)");
    check(cudaMemcpyAsync(tasks.data(), plan.tasks,
                          tasks.size() * sizeof(tasks.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(route plan tasks)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(route plan validation)");

    if (first != second)
        throw std::runtime_error("route plan is not bitwise deterministic");
    if (header.magic != q38::kQ38RoutePlanMagicV1 ||
        header.version != q38::kQ38RoutePlanVersionV1 ||
        header.tokens != tokens || header.routes != routes ||
        header.status !=
            static_cast<std::uint32_t>(q38::Q38RoutePlanStatusV1::kOk))
        throw std::runtime_error("route plan header differs");

    std::vector<std::uint16_t> expected_counts(q38::kQ38RouteExperts, 0);
    for (const std::int32_t expert : experts)
        ++expected_counts[static_cast<std::size_t>(expert)];
    std::vector<std::uint32_t> expected_offsets(q38::kQ38RouteExperts + 1, 0);
    std::vector<std::uint32_t> expected_task_offsets(
        q38::kQ38RouteExperts + 1, 0);
    for (std::uint32_t expert = 0; expert < q38::kQ38RouteExperts; ++expert) {
        expected_offsets[expert + 1] =
            expected_offsets[expert] + expected_counts[expert];
        expected_task_offsets[expert + 1] =
            expected_task_offsets[expert] +
            (expected_counts[expert] + q38::kQ38MmqM - 1) / q38::kQ38MmqM;
    }
    if (counts != expected_counts || offsets != expected_offsets ||
        task_offsets != expected_task_offsets ||
        header.task_count != expected_task_offsets.back() ||
        header.task_count > q38::q38_moe_prefill_max_tasks_v1(tokens))
        throw std::runtime_error("route plan counts or task bounds differ");

    std::vector<std::uint32_t> expected_packed;
    expected_packed.reserve(routes);
    for (std::uint32_t expert = 0; expert < q38::kQ38RouteExperts; ++expert)
        for (std::uint32_t assignment = 0; assignment < routes; ++assignment)
            if (experts[assignment] == static_cast<std::int32_t>(expert))
                expected_packed.push_back(assignment);
    if (packed != expected_packed)
        throw std::runtime_error("route plan stable expert packing differs");
    for (std::uint32_t position = 0; position < routes; ++position)
        if (inverse[packed[position]] != position)
            throw std::runtime_error("route plan inverse mapping differs");

    for (std::uint32_t expert = 0; expert < q38::kQ38RouteExperts; ++expert) {
        const std::uint32_t count = expected_counts[expert];
        const std::uint32_t expert_tasks =
            (count + q38::kQ38MmqM - 1) / q38::kQ38MmqM;
        for (std::uint32_t tile = 0; tile < expert_tasks; ++tile) {
            const auto& task = tasks[expected_task_offsets[expert] + tile];
            const std::uint32_t remaining = count - tile * q38::kQ38MmqM;
            const std::uint32_t expected_rows =
                std::min(remaining, q38::kQ38MmqM);
            if (task.packed_begin !=
                    expected_offsets[expert] + tile * q38::kQ38MmqM ||
                task.expert != expert || task.valid_rows != expected_rows ||
                task.flags != 0)
                throw std::runtime_error("route plan MMQ task differs");
        }
    }

    (void)cudaFree(device_allocation);
    (void)cudaFree(device_experts);
}

void test_tp_moe_route_plan() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "count TP route fixture devices");
    for (const unsigned tokens : {1u, 2u, 7u, 17u, 512u}) {
        const unsigned routes = tokens * q38::kQ38RouteTopK;
        for (int pattern = 0; pattern < 3; ++pattern) {
            std::vector<int> global(routes);
            std::vector<float> weights(routes);
            for (unsigned i = 0; i < routes; ++i) weights[i] = float(i % 4 + 1) / 8.0f;
            for (unsigned i = 0; i < routes; ++i)
                global[i] = pattern == 0 ? int((i * 173 + 255) % 512)
                    : pattern == 1 ? 255 : (i / 10 % 2 ? 256 : 3);
            std::vector<float> total(tokens * q38::kQ38HiddenWidth, 0.0f);
            for (int rank = 0; rank < 2; ++rank) {
                const auto device = rank % device_count;
                check(cudaSetDevice(device), "select TP route fixture rank");
                cudaStream_t stream;
                check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                      "create TP route fixture stream");
                std::vector<std::int32_t> local(routes, -1);
                std::vector<std::uint32_t> expected_packed;
                for (unsigned i = 0; i < routes; ++i)
                    if (global[i] / 256 == rank) local[i] = global[i] % 256;
                for (int expert = 0; expert < 256; ++expert)
                    for (unsigned i = 0; i < routes; ++i)
                        if (local[i] == expert) expected_packed.push_back(i);
                const auto owned = expected_packed.size();
                expected_packed.resize(routes, q38::kQ38RouteUnownedAssignment);
                auto* ids = upload(local);
                auto* device_weights = upload(weights);
                const auto bytes = q38::q38_moe_route_plan_bytes_v1(tokens);
                auto* allocation = allocate<std::uint8_t>(bytes);
                const auto plan = q38::cuda_moe_route_plan_storage_v1(allocation, bytes, tokens);
                // upload() uses the default stream; the fixture deliberately
                // consumes on a nonblocking stream, so establish readiness.
                check(cudaStreamSynchronize(nullptr), "TP fixture input upload ready");
                q38::cuda_moe_build_route_plan_v1(ids, tokens, plan, stream, device, 256, true);
                std::vector<std::uint32_t> packed(routes), inverse(routes);
                q38::Q38RoutePlanHeaderV1 header;
                check(cudaMemcpyAsync(&header, plan.header, sizeof(header), cudaMemcpyDeviceToHost, stream), "TP route header");
                check(cudaMemcpyAsync(packed.data(), plan.packed_assignment, routes * 4, cudaMemcpyDeviceToHost, stream), "TP packed assignments");
                check(cudaMemcpyAsync(inverse.data(), plan.assignment_to_packed, routes * 4, cudaMemcpyDeviceToHost, stream), "TP inverse assignments");
                check(cudaStreamSynchronize(stream), "TP plan ready");
                if (header.status || header.flags != q38::kQ38RoutePlanAllowUnowned || packed != expected_packed)
                    throw std::runtime_error("TP sparse expert route plan differs");
                for (unsigned i = 0; i < routes; ++i) {
                    if (local[i] == -1) {
                        if (inverse[i] != q38::kQ38RouteUnownedAssignment)
                            throw std::runtime_error("unowned route retained stale inverse");
                    } else if (inverse[i] >= owned || packed[inverse[i]] != i)
                        throw std::runtime_error("TP owned route inverse differs");
                }
                // Poison unused rows so accidentally reading an unowned route
                // cannot pass as a conveniently zero-initialized contribution.
                std::vector<float> route_output(routes * q38::kQ38HiddenWidth, NAN);
                for (unsigned p = 0; p < owned; ++p)
                    for (unsigned column = 0; column < q38::kQ38HiddenWidth; ++column)
                        route_output[p * q38::kQ38HiddenWidth + column] =
                            float(int(packed[p] + 1) * (int(column % 7) - 3));
                auto* routed = upload(route_output);
                auto* partial = allocate<float>(total.size());
                check(cudaStreamSynchronize(nullptr), "TP fixture routed upload ready");
                q38::cuda_moe_reduce_top10_f32_v1(routed, plan.assignment_to_packed,
                    tokens, partial, stream, device, device_weights);
                std::vector<float> actual(total.size());
                check(cudaMemcpyAsync(actual.data(), partial, actual.size() * 4,
                    cudaMemcpyDeviceToHost, stream), "TP token-major partial");
                check(cudaStreamSynchronize(stream), "TP partial ready");
                for (unsigned token = 0; token < tokens; ++token)
                    for (unsigned column = 0; column < q38::kQ38HiddenWidth; ++column) {
                        float expected = 0;
                        for (unsigned slot = 0; slot < 10; ++slot) {
                            const auto assignment = token * 10 + slot;
                            if (local[assignment] >= 0)
                                expected += float(int(assignment + 1) * (int(column % 7) - 3)) * weights[assignment];
                        }
                        const auto index = token * q38::kQ38HiddenWidth + column;
                        if (actual[index] != expected)
                            throw std::runtime_error("TP token-major FP32 partial differs: tokens=" +
                                std::to_string(tokens) + " pattern=" + std::to_string(pattern) +
                                " rank=" + std::to_string(rank) + " token=" + std::to_string(token) +
                                " column=" + std::to_string(column) + " actual=" +
                                std::to_string(actual[index]) + " expected=" + std::to_string(expected));
                        total[index] += actual[index];
                    }
                // -1 is the only permitted missing-route sentinel; an
                // out-of-range expert must still produce an invalid plan.
                local[0] = 256;
                check(cudaMemcpyAsync(ids, local.data(), routes * 4, cudaMemcpyHostToDevice, stream), "TP invalid route input");
                q38::cuda_moe_build_route_plan_v1(ids, tokens, plan, stream, device, 256, true);
                check(cudaMemcpyAsync(&header, plan.header, sizeof(header), cudaMemcpyDeviceToHost, stream), "TP invalid header");
                check(cudaStreamSynchronize(stream), "TP invalid plan ready");
                if (header.status != static_cast<unsigned>(q38::Q38RoutePlanStatusV1::kInvalidExpert))
                    throw std::runtime_error("TP accepted an out-of-range local expert");
                (void)cudaFree(ids); (void)cudaFree(allocation);
                (void)cudaFree(device_weights);
                (void)cudaFree(routed); (void)cudaFree(partial);
                (void)cudaStreamDestroy(stream);
            }
            for (unsigned token = 0; token < tokens; ++token)
                for (unsigned column = 0; column < q38::kQ38HiddenWidth; ++column) {
                    float expected = 0;
                    for (unsigned slot = 0; slot < 10; ++slot)
                        expected += float(int(token * 10 + slot + 1) * (int(column % 7) - 3)) * weights[token * 10 + slot];
                    if (total[token * q38::kQ38HiddenWidth + column] != expected)
                        throw std::runtime_error("TP rank partial sum differs from unpartitioned routes");
                }
        }
    }
    check(cudaSetDevice(0), "restore route fixture device");
}

void test_moe_route_plan(cudaStream_t stream) {
    if (q38::q38_moe_prefill_max_tasks_v1(128) != 560 ||
        q38::q38_moe_prefill_max_tasks_v1(256) != 640 ||
        q38::q38_moe_prefill_max_tasks_v1(512) != 800 ||
        q38::q38_moe_route_plan_bytes_v1(512) != 52552)
        throw std::runtime_error("route plan compile-time bounds differ");
    verify_moe_route_plan(1, false, stream);
    verify_moe_route_plan(128, false, stream);
    verify_moe_route_plan(512, false, stream);
    verify_moe_route_plan(512, true, stream);

    constexpr std::uint32_t tokens = 3;
    const std::uint32_t routes = q38::q38_moe_prefill_routes_v1(tokens);
    std::vector<std::int32_t> experts(routes);
    for (std::uint32_t assignment = 0; assignment < routes; ++assignment)
        experts[assignment] = static_cast<std::int32_t>(
            (assignment * 19u + 5u) % q38::kQ38RouteExperts);
    std::vector<std::uint16_t> hidden(
        static_cast<std::size_t>(tokens) * q38::kQ38HiddenWidth);
    for (std::uint32_t token = 0; token < tokens; ++token)
        for (std::uint32_t column = 0; column < q38::kQ38HiddenWidth; ++column)
            hidden[static_cast<std::size_t>(token) * q38::kQ38HiddenWidth +
                   column] = bf16(static_cast<float>(token * 4 + column % 4));
    auto* device_experts = upload(experts);
    auto* device_hidden = upload(hidden);
    const std::size_t plan_bytes =
        q38::q38_moe_route_plan_bytes_v1(tokens);
    auto* device_allocation = allocate<std::uint8_t>(plan_bytes);
    const auto plan = q38::cuda_moe_route_plan_storage_v1(
        device_allocation, plan_bytes, tokens);
    auto* device_packed_hidden = allocate<std::uint16_t>(
        static_cast<std::size_t>(routes) * q38::kQ38HiddenWidth);
    q38::cuda_moe_build_route_plan_v1(
        device_experts, tokens, plan, reinterpret_cast<void*>(stream), 0);
    q38::cuda_moe_pack_hidden_v1(
        device_hidden, plan.packed_assignment, routes, device_packed_hidden,
        reinterpret_cast<void*>(stream), 0);
    std::vector<std::uint32_t> packed(routes);
    std::vector<std::uint16_t> packed_hidden(
        static_cast<std::size_t>(routes) * q38::kQ38HiddenWidth);
    check(cudaMemcpyAsync(packed.data(), plan.packed_assignment,
                          packed.size() * sizeof(packed.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(hidden-pack assignments)");
    check(cudaMemcpyAsync(packed_hidden.data(), device_packed_hidden,
                          packed_hidden.size() * sizeof(packed_hidden.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(hidden-pack output)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(hidden-pack test)");
    for (std::uint32_t position = 0; position < routes; ++position) {
        const std::uint32_t token = packed[position] / q38::kQ38RouteTopK;
        for (std::uint32_t column = 0; column < q38::kQ38HiddenWidth; ++column)
            if (packed_hidden[static_cast<std::size_t>(position) *
                                  q38::kQ38HiddenWidth +
                              column] !=
                hidden[static_cast<std::size_t>(token) *
                           q38::kQ38HiddenWidth +
                       column])
                throw std::runtime_error("route-plan hidden packing differs");
    }

    experts[4] = -1;
    check(cudaMemcpyAsync(device_experts, experts.data(),
                          experts.size() * sizeof(experts.front()),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpy(invalid route expert)");
    q38::cuda_moe_build_route_plan_v1(
        device_experts, tokens, plan, reinterpret_cast<void*>(stream), 0);
    q38::Q38RoutePlanHeaderV1 header{};
    check(cudaMemcpyAsync(&header, plan.header, sizeof(header),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(invalid route header)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(invalid route test)");
    if (header.status != static_cast<std::uint32_t>(
                             q38::Q38RoutePlanStatusV1::kInvalidExpert))
        throw std::runtime_error("invalid route expert did not fail closed");

    (void)cudaFree(device_packed_hidden);
    (void)cudaFree(device_allocation);
    (void)cudaFree(device_hidden);
    (void)cudaFree(device_experts);
}

void test_grouped_moe_prefill(q38::DeviceWeightFormatV1 format,
                              cudaStream_t stream) {
    constexpr std::uint32_t tokens = 2;
    constexpr std::uint32_t routes = tokens * q38::kQ38RouteTopK;
    constexpr std::uint32_t gate_rows = 2 * q38::kQ38MoeIntermediate;
    constexpr std::uint32_t gate_columns = q38::kQ38HiddenWidth;
    constexpr std::uint32_t down_rows = q38::kQ38HiddenWidth;
    constexpr std::uint32_t down_columns = q38::kQ38MoeIntermediate;
    const bool w8 = format == q38::DeviceWeightFormatV1::kW8A16SymG128;
    const std::size_t gate_elements =
        static_cast<std::size_t>(gate_rows) * gate_columns;
    const std::size_t down_elements =
        static_cast<std::size_t>(down_rows) * down_columns;
    std::vector<std::uint8_t> gate_data(w8 ? gate_elements
                                           : gate_elements / 2,
                                        0);
    std::vector<std::uint8_t> down_data(w8 ? down_elements
                                           : down_elements / 2,
                                        0);
    const auto set_quantized = [w8](std::vector<std::uint8_t>* target,
                                    std::size_t logical_index,
                                    std::int8_t value) {
        if (w8) {
            (*target)[logical_index] = static_cast<std::uint8_t>(value);
            return;
        }
        const std::size_t packed_index = logical_index / 2;
        const std::uint8_t nibble =
            static_cast<std::uint8_t>(value) & 0x0fu;
        if ((logical_index & 1u) == 0)
            (*target)[packed_index] = static_cast<std::uint8_t>(
                ((*target)[packed_index] & 0xf0u) | nibble);
        else
            (*target)[packed_index] = static_cast<std::uint8_t>(
                ((*target)[packed_index] & 0x0fu) | (nibble << 4));
    };
    set_quantized(&gate_data, 0, 1);
    set_quantized(&gate_data,
                  static_cast<std::size_t>(q38::kQ38MoeIntermediate) *
                      gate_columns,
                  1);
    set_quantized(&down_data, 0, 1);
    std::vector<std::uint16_t> gate_scales(
        gate_rows * (gate_columns / 128), bf16(1.0f));
    std::vector<std::uint16_t> down_scales(
        down_rows * (down_columns / 128), bf16(1.0f));
    std::vector<std::uint16_t> hidden(
        static_cast<std::size_t>(tokens) * gate_columns, bf16(0.0f));
    hidden[0] = bf16(1.0f);
    hidden[gate_columns] = bf16(2.0f);
    std::vector<std::int32_t> experts(routes, 0);
    std::vector<float> route_weights(routes, 1.0f / q38::kQ38RouteTopK);
    std::vector<std::uint16_t> shared_output(
        static_cast<std::size_t>(tokens) * down_rows, bf16(0.0f));
    std::vector<std::uint16_t> shared_gate(tokens, bf16(0.0f));

    auto* device_gate_data = upload(gate_data);
    auto* device_down_data = upload(down_data);
    auto* device_gate_scales = upload(gate_scales);
    auto* device_down_scales = upload(down_scales);
    auto* device_hidden = upload(hidden);
    auto* device_experts = upload(experts);
    auto* device_route_weights = upload(route_weights);
    auto* device_shared_output = upload(shared_output);
    auto* device_shared_gate = upload(shared_gate);

    q38::DeviceTensorV1 gate_descriptor;
    gate_descriptor.format = format;
    gate_descriptor.group_size = 128;
    gate_descriptor.shape = {q38::kQ38RouteExperts, gate_rows, gate_columns};
    q38::DeviceTensorV1 down_descriptor;
    down_descriptor.format = format;
    down_descriptor.group_size = 128;
    down_descriptor.shape = {q38::kQ38RouteExperts, down_rows, down_columns};
    q38::CudaTensorViewV1 gate_view{&gate_descriptor, device_gate_data,
                                    device_gate_scales};
    q38::CudaTensorViewV1 down_view{&down_descriptor, device_down_data,
                                    device_down_scales};

    const std::size_t plan_bytes =
        q38::q38_moe_route_plan_bytes_v1(tokens);
    auto* device_plan_allocation = allocate<std::uint8_t>(plan_bytes);
    const auto plan = q38::cuda_moe_route_plan_storage_v1(
        device_plan_allocation, plan_bytes, tokens);
    auto* device_packed_hidden = allocate<std::uint16_t>(
        static_cast<std::size_t>(routes) * gate_columns);
    auto* device_weighted_mid = allocate<std::uint16_t>(
        static_cast<std::size_t>(routes) * down_columns);
    auto* device_weighted_mid_safe = allocate<std::uint16_t>(
        static_cast<std::size_t>(routes) * down_columns);
    auto* device_route_output =
        allocate<float>(static_cast<std::size_t>(routes) * down_rows);
    auto* device_route_output_safe =
        allocate<float>(static_cast<std::size_t>(routes) * down_rows);
    auto* device_output =
        allocate<std::uint16_t>(static_cast<std::size_t>(tokens) * down_rows);
    auto* device_output_safe =
        allocate<std::uint16_t>(static_cast<std::size_t>(tokens) * down_rows);

    q38::cuda_moe_build_route_plan_v1(
        device_experts, tokens, plan, reinterpret_cast<void*>(stream), 0);
    q38::Q38RoutePlanHeaderV1 header{};
    check(cudaMemcpyAsync(&header, plan.header, sizeof(header),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(grouped MoE route header)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(grouped MoE route header)");
    if (header.status != 0 || header.task_count != 2)
        throw std::runtime_error("grouped MoE route-plan fixture differs");
    q38::cuda_moe_pack_hidden_v1(
        device_hidden, plan.packed_assignment, routes, device_packed_hidden,
        reinterpret_cast<void*>(stream), 0);

    const auto run_mode = [&](const q38::CudaTensorViewV1& selected_gate,
                              const q38::CudaTensorViewV1& selected_down,
                              q38::Q38PrefillMoeModeV1 mode,
                              std::uint16_t* weighted_mid,
                              float* route_output,
                              std::uint16_t* output) {
        q38::cuda_moe_grouped_gate_up_v1(
            selected_gate, device_packed_hidden, plan.packed_assignment,
            device_route_weights, plan.tasks, header.task_count, weighted_mid,
            mode, reinterpret_cast<void*>(stream), 0);
        q38::cuda_moe_grouped_down_v1(
            selected_down, weighted_mid, plan.tasks, header.task_count,
            route_output, mode, reinterpret_cast<void*>(stream), 0);
        q38::cuda_moe_reduce_top10_and_combine_shared_v1(
            route_output, plan.assignment_to_packed, device_shared_output,
            device_shared_gate, tokens, output,
            reinterpret_cast<void*>(stream), 0);
    };
    run_mode(gate_view, down_view, q38::Q38PrefillMoeModeV1::kGroupedMmq,
             device_weighted_mid, device_route_output, device_output);
    run_mode(gate_view, down_view,
             q38::Q38PrefillMoeModeV1::kGroupedMmqSafe,
             device_weighted_mid_safe, device_route_output_safe,
             device_output_safe);

    std::vector<std::uint16_t> weighted_mid(
        static_cast<std::size_t>(routes) * down_columns);
    std::vector<std::uint16_t> weighted_mid_safe(weighted_mid.size());
    std::vector<float> route_output(
        static_cast<std::size_t>(routes) * down_rows);
    std::vector<float> route_output_safe(route_output.size());
    std::vector<std::uint16_t> output(
        static_cast<std::size_t>(tokens) * down_rows);
    std::vector<std::uint16_t> output_safe(output.size());
    check(cudaMemcpyAsync(weighted_mid.data(), device_weighted_mid,
                          weighted_mid.size() * sizeof(weighted_mid.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(grouped MoE middle)");
    check(cudaMemcpyAsync(
              weighted_mid_safe.data(), device_weighted_mid_safe,
              weighted_mid_safe.size() * sizeof(weighted_mid_safe.front()),
              cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(grouped MoE safe middle)");
    check(cudaMemcpyAsync(route_output.data(), device_route_output,
                          route_output.size() * sizeof(route_output.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(grouped MoE route output)");
    check(cudaMemcpyAsync(
              route_output_safe.data(), device_route_output_safe,
              route_output_safe.size() * sizeof(route_output_safe.front()),
              cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(grouped MoE safe route output)");
    check(cudaMemcpyAsync(output.data(), device_output,
                          output.size() * sizeof(output.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(grouped MoE output)");
    check(cudaMemcpyAsync(output_safe.data(), device_output_safe,
                          output_safe.size() * sizeof(output_safe.front()),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(grouped MoE safe output)");

    std::uint8_t* device_gate_tiled = nullptr;
    std::uint8_t* device_down_tiled = nullptr;
    std::vector<std::uint16_t> output_tiled;
    if (!w8) {
        const auto gate_tiled =
            tile_w4_16x16(gate_data, gate_rows, gate_columns);
        const auto down_tiled =
            tile_w4_16x16(down_data, down_rows, down_columns);
        device_gate_tiled = upload(gate_tiled);
        device_down_tiled = upload(down_tiled);
        const q38::CudaTensorViewV1 tiled_gate_view{
            &gate_descriptor, device_gate_tiled, device_gate_scales,
            q38::CudaWeightLayoutV1::kMoeW4Tile16x16};
        const q38::CudaTensorViewV1 tiled_down_view{
            &down_descriptor, device_down_tiled, device_down_scales,
            q38::CudaWeightLayoutV1::kMoeW4Tile16x16};
        run_mode(tiled_gate_view, tiled_down_view,
                 q38::Q38PrefillMoeModeV1::kGroupedMmq,
                 device_weighted_mid_safe, device_route_output_safe,
                 device_output_safe);
        output_tiled.resize(output.size());
        check(cudaMemcpyAsync(output_tiled.data(), device_output_safe,
                              output_tiled.size() *
                                  sizeof(output_tiled.front()),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpy(tiled grouped MoE output)");
    }
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(grouped MoE fixture)");
    if (weighted_mid != weighted_mid_safe ||
        route_output != route_output_safe || output != output_safe)
        throw std::runtime_error(
            "grouped and safe MoE modes are not bitwise equal");
    if (!w8 && output != output_tiled)
        throw std::runtime_error(
            "row-major and tiled W4 grouped MoE are not bitwise equal");
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const float gate = fp32(hidden[static_cast<std::size_t>(token) *
                                         gate_columns]);
        const float activated = fp32(
            bf16(gate / (1.0f + std::exp(-gate)) * gate));
        const float weighted = fp32(bf16(
            activated * (1.0f / static_cast<float>(q38::kQ38RouteTopK))));
        float routed = 0.0f;
        for (std::uint32_t rank = 0; rank < q38::kQ38RouteTopK; ++rank)
            routed += weighted;
        for (std::uint32_t column = 0; column < down_rows; ++column) {
            const std::uint16_t expected =
                column == 0 ? bf16(routed) : bf16(0.0f);
            const std::uint16_t actual =
                output[static_cast<std::size_t>(token) * down_rows + column];
            if (actual != expected)
                throw std::runtime_error(
                    std::string(w8 ? "W8" : "W4") +
                    " grouped MoE arithmetic differs at token " +
                    std::to_string(token) + " column " +
                    std::to_string(column) + ": expected " +
                    std::to_string(fp32(expected)) + " got " +
                    std::to_string(fp32(actual)));
        }
    }

    // TP must apply routing weights after down, without rounding weighted
    // activations to BF16. Exercise the real W4/W8 kernels with this policy.
    auto* device_partial = allocate<float>(static_cast<std::size_t>(tokens) * down_rows);
    for (const auto mode : {q38::Q38PrefillMoeModeV1::kGroupedMmq,
                            q38::Q38PrefillMoeModeV1::kGroupedMmqSafe,
                            q38::Q38PrefillMoeModeV1::kGroupedFp32Scales}) {
        q38::cuda_moe_grouped_gate_up_v1(
            gate_view, device_packed_hidden, plan.packed_assignment,
            device_route_weights, plan.tasks, header.task_count,
            device_weighted_mid, mode, stream, 0, false);
        q38::cuda_moe_grouped_down_v1(
            down_view, device_weighted_mid, plan.tasks, header.task_count,
            device_route_output, mode, stream, 0);
        q38::cuda_moe_reduce_top10_f32_v1(
            device_route_output, plan.assignment_to_packed, tokens,
            device_partial, stream, 0, device_route_weights);
        std::vector<float> partial(static_cast<std::size_t>(tokens) * down_rows);
        check(cudaMemcpyAsync(partial.data(), device_partial, partial.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream), "TP grouped partial");
        check(cudaStreamSynchronize(stream), "TP grouped partial ready");
        for (unsigned token = 0; token < tokens; ++token) {
            const float gate = fp32(hidden[static_cast<std::size_t>(token) * gate_columns]);
            const float activated = fp32(bf16(gate / (1.0f + std::exp(-gate)) * gate));
            float expected = 0.0f;
            for (unsigned slot = 0; slot < q38::kQ38RouteTopK; ++slot)
                expected += activated * (1.0f / static_cast<float>(q38::kQ38RouteTopK));
            for (unsigned column = 0; column < down_rows; ++column)
                if (std::abs(partial[token * down_rows + column] - (column == 0 ? expected : 0.0f)) > 1e-6f)
                    throw std::runtime_error("TP grouped post-down route weighting differs");
        }
    }
    // Non-power-of-two scales on every output column and every G128
    // group expose dequant-to-BF16 rounding and incorrect scale coordinates.
    std::fill(gate_data.begin(), gate_data.end(), 0);
    std::fill(down_data.begin(), down_data.end(), 0);
    std::fill(hidden.begin(), hidden.end(), bf16(0));
    const auto quant = [w8](unsigned row, unsigned group) {
        const int magnitude = w8 ? 57 + int(row % 71) : 1 + int(row % 7);
        return (row + group) % 3 == 0 ? -magnitude : magnitude;
    };
    for (unsigned row = 0; row < gate_rows; ++row)
        for (unsigned group = 0; group < gate_columns / 128; ++group) {
            for (const unsigned offset : {0u, 63u, 127u})
                set_quantized(&gate_data, static_cast<std::size_t>(row) * gate_columns + group * 128 + offset,
                              static_cast<std::int8_t>(quant(row + offset, group)));
            gate_scales[row * (gate_columns / 128) + group] =
                bf16((w8 ? .001f : .01f) * (1.0f + float((row + group) % 11) * .13f));
        }
    for (unsigned row = 0; row < down_rows; ++row)
        for (unsigned group = 0; group < down_columns / 128; ++group) {
            for (const unsigned offset : {0u, 63u, 127u})
                set_quantized(&down_data, static_cast<std::size_t>(row) * down_columns + group * 128 + offset,
                              static_cast<std::int8_t>(quant(row + offset, group)));
            down_scales[row * (down_columns / 128) + group] =
                bf16((w8 ? .0005f : .005f) * (1.0f + float((row + group) % 9) * .17f));
        }
    for (unsigned token = 0; token < tokens; ++token)
        for (unsigned group = 0; group < gate_columns / 128; ++group)
            for (const unsigned offset : {0u, 63u, 127u})
                hidden[token * gate_columns + group * 128 + offset] = bf16(.25f + float((token + group + offset) % 4) * .125f);
    const auto upload_into = [&](auto* destination, const auto& source) {
        check(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(source[0]),
                              cudaMemcpyHostToDevice, stream), "upload FP32 group scale fixture");
    };
    upload_into(device_gate_data, gate_data); upload_into(device_down_data, down_data);
    upload_into(device_gate_scales, gate_scales); upload_into(device_down_scales, down_scales);
    upload_into(device_hidden, hidden);
    q38::cuda_moe_pack_hidden_v1(device_hidden, plan.packed_assignment, routes,
                                device_packed_hidden, stream, 0);
    for (int tiled = 0; tiled < (w8 ? 1 : 2); ++tiled) {
        q38::CudaTensorViewV1 test_gate = gate_view, test_down = down_view;
        std::vector<std::uint8_t> gate_tiled, down_tiled;
        if (tiled) {
            gate_tiled = tile_w4_16x16(gate_data, gate_rows, gate_columns);
            down_tiled = tile_w4_16x16(down_data, down_rows, down_columns);
            upload_into(device_gate_tiled, gate_tiled); upload_into(device_down_tiled, down_tiled);
            test_gate.data = device_gate_tiled; test_down.data = device_down_tiled;
            test_gate.layout = test_down.layout = q38::CudaWeightLayoutV1::kMoeW4Tile16x16;
        }
        const auto mode = q38::Q38PrefillMoeModeV1::kGroupedFp32Scales;
        const char* original_flag = std::getenv("Q38_TP_MOE_FP32_SHARED_A");
        const bool had_flag = original_flag != nullptr;
        const std::string saved_flag = original_flag ? original_flag : "";
        const auto run_shared_variant = [&](bool shared) {
            if (setenv("Q38_TP_MOE_FP32_SHARED_A", shared ? "1" : "0", 1) != 0)
                throw std::runtime_error("set shared FP32 activation flag failed");
            q38::cuda_moe_grouped_gate_up_v1(test_gate, device_packed_hidden,
                plan.packed_assignment, device_route_weights, plan.tasks, header.task_count,
                device_weighted_mid, mode, stream, 0, false);
            q38::cuda_moe_grouped_down_v1(test_down, device_weighted_mid, plan.tasks,
                header.task_count, device_route_output, mode, stream, 0);
            q38::cuda_moe_reduce_top10_f32_v1(device_route_output, plan.assignment_to_packed,
                tokens, device_partial, stream, 0, device_route_weights);
            std::vector<float> partial(tokens * down_rows);
            check(cudaMemcpyAsync(partial.data(), device_partial, partial.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream), "FP32 group scale result");
            check(cudaStreamSynchronize(stream), "FP32 group scale fixture ready");
            return partial;
        };
        const auto partial = run_shared_variant(false);
        const auto shared_partial = run_shared_variant(true);
        if (had_flag) setenv("Q38_TP_MOE_FP32_SHARED_A", saved_flag.c_str(), 1);
        else unsetenv("Q38_TP_MOE_FP32_SHARED_A");
        if (std::memcmp(partial.data(), shared_partial.data(), partial.size() * sizeof(float)) != 0)
            throw std::runtime_error("shared FP32 activation changed MoE output bits");
        for (unsigned token = 0; token < tokens; ++token) {
            std::vector<float> projected(gate_rows), activation(down_columns);
            for (unsigned row = 0; row < gate_rows; ++row) {
                float sum = 0;
                for (unsigned group = 0; group < gate_columns / 128; ++group) {
                    float dot = 0;
                    for (const unsigned offset : {0u, 63u, 127u})
                        dot += float(quant(row + offset, group)) * fp32(hidden[token * gate_columns + group * 128 + offset]);
                    sum = std::fma(dot, fp32(gate_scales[row * (gate_columns / 128) + group]), sum);
                }
                projected[row] = fp32(bf16(sum));
            }
            for (unsigned row = 0; row < down_columns; ++row)
                activation[row] = fp32(bf16(projected[row] / (1.0f + std::exp(-projected[row])) * projected[row + down_columns]));
            for (unsigned row = 0; row < down_rows; ++row) {
                float sum = 0;
                for (unsigned group = 0; group < down_columns / 128; ++group) {
                    float dot = 0;
                    for (const unsigned offset : {0u, 63u, 127u})
                        dot += float(quant(row + offset, group)) * activation[group * 128 + offset];
                    sum = std::fma(dot, fp32(down_scales[row * (down_columns / 128) + group]), sum);
                }
                float expected = 0;
                for (unsigned slot = 0; slot < q38::kQ38RouteTopK; ++slot)
                    expected = std::fma(sum, route_weights[token * q38::kQ38RouteTopK + slot], expected);
                if (std::abs(partial[token * down_rows + row] - expected) > 1e-5f * (1.0f + std::abs(expected)))
                    throw std::runtime_error("FP32 grouped scale arithmetic differs: row=" +
                        std::to_string(row) + " expected=" + std::to_string(expected) +
                        " actual=" + std::to_string(partial[token * down_rows + row]));
            }
        }
    }
    (void)cudaFree(device_partial);
    (void)cudaFree(device_output_safe);
    (void)cudaFree(device_output);
    (void)cudaFree(device_route_output_safe);
    (void)cudaFree(device_route_output);
    (void)cudaFree(device_weighted_mid_safe);
    (void)cudaFree(device_weighted_mid);
    (void)cudaFree(device_packed_hidden);
    (void)cudaFree(device_plan_allocation);
    (void)cudaFree(device_shared_gate);
    (void)cudaFree(device_shared_output);
    (void)cudaFree(device_route_weights);
    (void)cudaFree(device_experts);
    (void)cudaFree(device_hidden);
    (void)cudaFree(device_down_scales);
    (void)cudaFree(device_gate_scales);
    (void)cudaFree(device_down_data);
    (void)cudaFree(device_gate_data);
    if (device_down_tiled) (void)cudaFree(device_down_tiled);
    if (device_gate_tiled) (void)cudaFree(device_gate_tiled);
}

void test_boundary_transport_checksum(cudaStream_t stream, bool asynchronous = false) {
    constexpr std::size_t words = 4096;
    std::vector<std::uint16_t> source(words);
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<std::uint16_t>(index * 37u + 11u);
    auto* device_source = upload(source);
    auto* device_destination = allocate<std::uint16_t>(words);
    q38::CudaBoundaryRingOptions options;
    options.producer_device = 0;
    options.slots = 2;
    options.max_tokens = 1;
    options.hidden_width = words;
    options.asynchronous_copy = asynchronous;
    q38::CudaBoundaryRing ring(options);

    q38::BoundaryBuffer boundary;
    boundary.frame.session_hash = 0x38;
    boundary.frame.epoch = 1;
    boundary.frame.token_count = 1;
    boundary.frame.hidden_dtype = q38::DType::kBFloat16;
    boundary.frame.producer_status = q38::ProducerStatus::kReady;
    boundary.frame.hidden_width = words;
    boundary.frame.payload_bytes = words * sizeof(std::uint16_t);
    boundary.lease = ring.copy_from_device(device_source, words, stream);
    if (asynchronous)
        check(cudaMemsetAsync(device_source, 0, words * sizeof(std::uint16_t), stream),
              "overwrite boundary producer buffer after handoff");
    boundary.frame.ring_slot = boundary.lease->slot();
    boundary.frame.payload_checksum = boundary.payload_checksum();
    if (boundary.frame.payload_checksum !=
        q38::boundary_payload_checksum(source.data(), source.size()))
        throw std::runtime_error("GPU/CPU boundary checksum differs");
    q38::cuda_copy_boundary_to_device(boundary, device_destination, stream, 0);
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(boundary round trip)");
    std::vector<std::uint16_t> round_trip(words);
    check(cudaMemcpy(round_trip.data(), device_destination,
                     words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
          "cudaMemcpy(boundary round trip)");
    if (round_trip != source)
        throw std::runtime_error("boundary round trip differs");

    check(cudaMemcpyAsync(device_source, source.data(), words * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice, stream), "restore boundary test source");
    auto corrupt = ring.copy_from_device(device_source, words, stream);
    corrupt->wait_ready();
    const_cast<std::uint16_t*>(corrupt->data())[17] ^= 1u;
    bool rejected = false;
    try {
        corrupt->wait_ready();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("corrupt boundary checksum was accepted");
    check(cudaFree(device_destination), "cudaFree(boundary destination)");
    check(cudaFree(device_source), "cudaFree(boundary source)");
}

void test_device_boundary_reuse(cudaStream_t stream) {
    constexpr std::size_t words = 64, rounds = 32;
    q38::CudaBoundaryRingOptions options;
    options.producer_device = 0; options.slots = 2;
    options.max_tokens = 1; options.hidden_width = words; options.device_only = true;
    q38::CudaBoundaryRing ring(options);
    if (ring.pinned_bytes() != 2 * sizeof(unsigned long long))
        throw std::runtime_error("device boundary allocated a host payload");
    std::vector<std::uint16_t> expected(words * rounds);
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = (i * 13 + 5) % 65536;
    auto* source = allocate<std::uint16_t>(words);
    auto* destination = allocate<std::uint16_t>(words * rounds);
    cudaStream_t consumer;
    check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "create boundary test consumer");
    for (std::size_t i = 0; i < rounds; ++i) {
        check(cudaMemcpyAsync(source, expected.data() + i * words, words * 2,
                              cudaMemcpyHostToDevice, stream), "set device boundary input");
        q38::BoundaryBuffer boundary;
        boundary.lease = ring.copy_from_device(source, words, stream);
        boundary.frame.payload_checksum = boundary.payload_checksum();
        if (boundary.payload_checksum() != q38::boundary_payload_checksum(expected.data() + i * words, words))
            throw std::runtime_error("device boundary checksum differs");
        check(cudaMemsetAsync(source, 0, words * 2, stream), "overwrite leased producer input");
        q38::cuda_copy_boundary_to_device(boundary, destination + i * words, consumer, 0);
        // Lease released while the other stream can still be consuming its slot.
    }
    check(cudaStreamSynchronize(consumer), "finish device boundary consumers");
    std::vector<std::uint16_t> actual(expected.size());
    check(cudaMemcpy(actual.data(), destination, actual.size() * 2, cudaMemcpyDeviceToHost), "read boundary generations");
    if (actual != expected) throw std::runtime_error("device boundary slot reused before consumer finished");
    check(cudaStreamDestroy(consumer), "destroy boundary test consumer");
    check(cudaFree(source), "free boundary test input");
    check(cudaFree(destination), "free boundary test output");
}

void test_gdn_prefill(cudaStream_t stream) {
    constexpr std::uint32_t tokens = 3;
    std::vector<std::uint16_t> projected(
        static_cast<std::size_t>(tokens) * q38::kQ38GdnQkvWidth);
    std::vector<std::uint16_t> gate(
        static_cast<std::size_t>(tokens) * q38::kQ38GdnValueWidth);
    std::vector<std::uint16_t> a(
        static_cast<std::size_t>(tokens) * q38::kQ38GdnValueHeads);
    std::vector<std::uint16_t> b(a.size());
    for (std::size_t i = 0; i < projected.size(); ++i)
        projected[i] = bf16(std::sin(static_cast<float>(i % 251) * 0.017f) *
                            0.08f);
    for (std::size_t i = 0; i < gate.size(); ++i)
        gate[i] = bf16(std::cos(static_cast<float>(i % 127) * 0.023f) * 0.1f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = bf16(-0.2f + static_cast<float>(i % 11) * 0.01f);
        b[i] = bf16(0.1f - static_cast<float>(i % 7) * 0.01f);
    }
    std::vector<std::uint16_t> conv(
        static_cast<std::size_t>(q38::kQ38GdnQkvWidth) *
        q38::kQ38GdnConvWidth,
        bf16(0.05f));
    std::vector<std::uint16_t> a_log(q38::kQ38GdnValueHeads, bf16(-1.0f));
    std::vector<std::uint16_t> dt(q38::kQ38GdnValueHeads, bf16(-0.5f));
    std::vector<std::uint16_t> norm(q38::kQ38GdnHeadWidth, bf16(1.0f));

    auto* d_projected = upload(projected);
    auto* d_gate = upload(gate);
    auto* d_a = upload(a);
    auto* d_b = upload(b);
    auto* d_conv = upload(conv);
    auto* d_a_log = upload(a_log);
    auto* d_dt = upload(dt);
    auto* d_norm = upload(norm);
    const auto conv_words = static_cast<std::size_t>(q38::kQ38GdnQkvWidth) *
                            q38::kQ38GdnConvWidth;
    const auto recurrent_values =
        static_cast<std::size_t>(q38::kQ38GdnValueHeads) *
        q38::kQ38GdnHeadWidth * q38::kQ38GdnHeadWidth;
    auto* state_conv_batch = allocate<std::uint16_t>(conv_words, true);
    auto* state_conv_decode = allocate<std::uint16_t>(conv_words, true);
    auto* state_rec_batch = allocate<float>(recurrent_values, true);
    auto* state_rec_partitioned = allocate<float>(recurrent_values, true);
    auto* state_rec_precomputed = allocate<float>(recurrent_values, true);
    auto* state_rec_decode = allocate<float>(recurrent_values, true);
    const auto qkv_values =
        static_cast<std::size_t>(tokens) * q38::kQ38GdnQkvWidth;
    const auto core_values =
        static_cast<std::size_t>(tokens) * q38::kQ38GdnValueWidth;
    auto* activated_batch = allocate<std::uint16_t>(qkv_values);
    auto* activated_decode = allocate<std::uint16_t>(qkv_values);
    auto* core_batch = allocate<std::uint16_t>(core_values);
    auto* core_partitioned = allocate<std::uint16_t>(core_values);
    auto* core_precomputed = allocate<std::uint16_t>(core_values);
    auto* core_decode = allocate<std::uint16_t>(core_values);
    auto* output_batch = allocate<std::uint16_t>(core_values);
    auto* output_partitioned = allocate<std::uint16_t>(core_values);
    auto* output_precomputed = allocate<std::uint16_t>(core_values);
    auto* output_decode = allocate<std::uint16_t>(core_values);
    auto* gdn_parameters = allocate<float>(
        q38::q38_gdn_prefill_parameter_floats(tokens));

    q38::cuda_gdn_conv_prefill_bf16(
        d_projected, d_conv, state_conv_batch, activated_batch, tokens,
        reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_recurrent_prefill_bf16(
        activated_batch, d_b, d_a, d_a_log, d_dt, state_rec_batch,
        core_batch, tokens, reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_recurrent_prefill_partitioned_bf16(
        activated_batch, d_b, d_a, d_a_log, d_dt,
        state_rec_partitioned, core_partitioned, tokens,
        reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_recurrent_prefill_precomputed_bf16(
        activated_batch, d_b, d_a, d_a_log, d_dt,
        state_rec_precomputed, gdn_parameters, core_precomputed, tokens,
        reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_output_norm_prefill_bf16(
        core_batch, d_gate, d_norm, output_batch, tokens, 1.0e-6f,
        reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_output_norm_prefill_bf16(
        core_partitioned, d_gate, d_norm, output_partitioned, tokens,
        1.0e-6f, reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_output_norm_prefill_bf16(
        core_precomputed, d_gate, d_norm, output_precomputed, tokens,
        1.0e-6f, reinterpret_cast<void*>(stream), 0);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        q38::cuda_gdn_conv_decode_bf16(
            d_projected + static_cast<std::size_t>(token) *
                              q38::kQ38GdnQkvWidth,
            d_conv, state_conv_decode,
            activated_decode + static_cast<std::size_t>(token) *
                                   q38::kQ38GdnQkvWidth,
            reinterpret_cast<void*>(stream), 0);
        q38::cuda_gdn_recurrent_decode_bf16(
            activated_decode + static_cast<std::size_t>(token) *
                                   q38::kQ38GdnQkvWidth,
            d_b + static_cast<std::size_t>(token) * q38::kQ38GdnValueHeads,
            d_a + static_cast<std::size_t>(token) * q38::kQ38GdnValueHeads,
            d_a_log, d_dt, state_rec_decode,
            core_decode + static_cast<std::size_t>(token) *
                              q38::kQ38GdnValueWidth,
            reinterpret_cast<void*>(stream), 0);
        q38::cuda_gdn_output_norm_bf16(
            core_decode + static_cast<std::size_t>(token) *
                              q38::kQ38GdnValueWidth,
            d_gate + static_cast<std::size_t>(token) *
                         q38::kQ38GdnValueWidth,
            d_norm,
            output_decode + static_cast<std::size_t>(token) *
                                q38::kQ38GdnValueWidth,
            1.0e-6f, reinterpret_cast<void*>(stream), 0);
    }
    std::vector<std::uint16_t> host_batch(core_values);
    std::vector<std::uint16_t> host_partitioned(core_values);
    std::vector<std::uint16_t> host_precomputed(core_values);
    std::vector<std::uint16_t> host_decode(core_values);
    check(cudaMemcpyAsync(host_batch.data(), output_batch,
                          core_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(GDN batch)");
    check(cudaMemcpyAsync(host_decode.data(), output_decode,
                          core_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(GDN decode)");
    check(cudaMemcpyAsync(host_partitioned.data(), output_partitioned,
                          core_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(GDN partitioned)");
    check(cudaMemcpyAsync(host_precomputed.data(), output_precomputed,
                          core_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(GDN precomputed)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(GDN test)");
    require_close(host_batch, host_decode, 0.01f, "GDN prefill parity");
    require_close(host_partitioned, host_batch, 0.01f,
                  "GDN partitioned prefill parity");
    if (host_precomputed != host_partitioned)
        throw std::runtime_error(
            "GDN precomputed and partitioned prefill are not bitwise equal");

    (void)cudaFree(gdn_parameters);
    (void)cudaFree(output_decode);
    (void)cudaFree(output_precomputed);
    (void)cudaFree(output_partitioned);
    (void)cudaFree(output_batch);
    (void)cudaFree(core_decode);
    (void)cudaFree(core_precomputed);
    (void)cudaFree(core_partitioned);
    (void)cudaFree(core_batch);
    (void)cudaFree(activated_decode);
    (void)cudaFree(activated_batch);
    (void)cudaFree(state_rec_decode);
    (void)cudaFree(state_rec_precomputed);
    (void)cudaFree(state_rec_partitioned);
    (void)cudaFree(state_rec_batch);
    (void)cudaFree(state_conv_decode);
    (void)cudaFree(state_conv_batch);
    (void)cudaFree(d_norm);
    (void)cudaFree(d_dt);
    (void)cudaFree(d_a_log);
    (void)cudaFree(d_conv);
    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
    (void)cudaFree(d_gate);
    (void)cudaFree(d_projected);
}

void test_ple_prefill(cudaStream_t stream) {
    constexpr std::uint32_t tokens = 3;
    const auto values = static_cast<std::size_t>(tokens) * q38::kQ38HyperWidth;
    std::vector<std::uint16_t> gated(values);
    std::vector<std::uint16_t> normalized(values);
    for (std::size_t i = 0; i < values; ++i) {
        gated[i] = bf16(std::sin(static_cast<float>(i % 191) * 0.02f) * 0.1f);
        normalized[i] =
            bf16(std::cos(static_cast<float>(i % 173) * 0.015f) * 0.08f);
    }
    std::vector<std::uint16_t> weight(
        static_cast<std::size_t>(q38::kQ38HyperWidth) * 4, bf16(0.125f));
    auto* d_gated = upload(gated);
    auto* d_normalized = upload(normalized);
    auto* d_weight = upload(weight);
    const auto state_values =
        static_cast<std::size_t>(q38::kQ38HyperWidth) * q38::kQ38PleConvState;
    auto* state_batch = allocate<std::uint16_t>(state_values, true);
    auto* state_decode = allocate<std::uint16_t>(state_values, true);
    auto* output_batch = allocate<std::uint16_t>(values);
    auto* output_decode = allocate<std::uint16_t>(values);
    q38::cuda_ple_conv_prefill_bf16(
        d_gated, d_normalized, d_weight, state_batch, output_batch, tokens,
        reinterpret_cast<void*>(stream), 0);
    for (std::uint32_t token = 0; token < tokens; ++token)
        q38::cuda_ple_conv_decode_bf16(
            d_gated + static_cast<std::size_t>(token) * q38::kQ38HyperWidth,
            d_normalized +
                static_cast<std::size_t>(token) * q38::kQ38HyperWidth,
            d_weight, state_decode,
            output_decode +
                static_cast<std::size_t>(token) * q38::kQ38HyperWidth,
            reinterpret_cast<void*>(stream), 0);
    std::vector<std::uint16_t> host_batch(values);
    std::vector<std::uint16_t> host_decode(values);
    check(cudaMemcpyAsync(host_batch.data(), output_batch,
                          values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(PLE batch)");
    check(cudaMemcpyAsync(host_decode.data(), output_decode,
                          values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(PLE decode)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(PLE test)");
    require_close(host_batch, host_decode, 0.001f, "PLE prefill parity");
    (void)cudaFree(output_decode);
    (void)cudaFree(output_batch);
    (void)cudaFree(state_decode);
    (void)cudaFree(state_batch);
    (void)cudaFree(d_weight);
    (void)cudaFree(d_normalized);
    (void)cudaFree(d_gated);
}

void test_qsa_prefill(cudaStream_t stream) {
    constexpr std::uint32_t tokens = 8;
    constexpr std::uint32_t capacity = 16;
    const auto query_values = static_cast<std::size_t>(tokens) *
                              q38::kQ38QsaHeads *
                              q38::kQ38QsaHeadWidth;
    const auto kv_values = static_cast<std::size_t>(tokens) *
                           q38::kQ38QsaKvHeads *
                           q38::kQ38QsaHeadWidth;
    const auto index_values = static_cast<std::size_t>(tokens) *
                              (q38::kQ38QsaIndexerHeads + 1) *
                              q38::kQ38QsaIndexerWidth;
    const auto index_query_values = static_cast<std::size_t>(tokens) *
                                    q38::kQ38QsaIndexerHeads *
                                    q38::kQ38QsaIndexerWidth;
    std::vector<std::uint16_t> projected_q_gate(query_values * 2);
    std::vector<std::uint16_t> projected_k(kv_values);
    std::vector<std::uint16_t> projected_v(kv_values);
    std::vector<std::uint16_t> projected_index(index_values);
    for (std::size_t index = 0; index < projected_q_gate.size(); ++index)
        projected_q_gate[index] = bf16(
            std::sin(static_cast<float>(index % 509) * 0.011f) * 0.08f);
    for (std::size_t index = 0; index < projected_k.size(); ++index) {
        projected_k[index] = bf16(
            std::cos(static_cast<float>(index % 251) * 0.017f) * 0.07f);
        projected_v[index] = bf16(
            std::sin(static_cast<float>(index % 193) * 0.019f) * 0.09f);
    }
    for (std::size_t index = 0; index < projected_index.size(); ++index)
        projected_index[index] = bf16(
            std::cos(static_cast<float>(index % 127) * 0.023f) * 0.06f);
    std::vector<std::uint16_t> main_norm(q38::kQ38QsaHeadWidth,
                                         bf16(0.0f));
    std::vector<std::uint16_t> index_norm(q38::kQ38QsaIndexerWidth,
                                          bf16(0.0f));

    auto* d_q_gate = upload(projected_q_gate);
    auto* d_k = upload(projected_k);
    auto* d_v = upload(projected_v);
    auto* d_index = upload(projected_index);
    auto* d_main_norm = upload(main_norm);
    auto* d_index_norm = upload(index_norm);
    auto* query_batch = allocate<std::uint16_t>(query_values);
    auto* gate_batch = allocate<std::uint16_t>(query_values);
    auto* index_query_batch = allocate<std::uint16_t>(index_query_values);
    auto* output_batch = allocate<std::uint16_t>(query_values);
    auto* output_grouped = allocate<std::uint16_t>(query_values);
    auto* output_grouped_fused = allocate<std::uint16_t>(query_values);
    auto* query_decode = allocate<std::uint16_t>(query_values);
    auto* gate_decode = allocate<std::uint16_t>(query_values);
    auto* index_query_decode = allocate<std::uint16_t>(index_query_values);
    auto* output_decode = allocate<std::uint16_t>(query_values);
    constexpr std::uint32_t batch_score_stride =
        (tokens + q38::kQ38QsaBlockTokens - 1) /
        q38::kQ38QsaBlockTokens;
    auto* batch_block_scores =
        allocate<float>(tokens * batch_score_stride);
    auto* batch_selected = allocate<std::int32_t>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaMaximumSelected);
    auto* batch_attention = allocate<float>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaHeads *
        q38::kQ38QsaMaximumSelected);
    auto* grouped_attention = allocate<float>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaHeads *
        q38::kQ38QsaMaximumSelected);
    auto* grouped_fused_attention = allocate<float>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaHeads *
        q38::kQ38QsaMaximumSelected);
    auto* decode_block_scores = allocate<float>(
        q38::kQ38ContextLimit / q38::kQ38QsaBlockTokens);
    auto* decode_selected =
        allocate<std::int32_t>(q38::kQ38QsaMaximumSelected);
    auto* decode_attention = allocate<float>(
        q38::kQ38QsaHeads * q38::kQ38QsaMaximumSelected);

    q38::CudaQsaStateBank batch_state(0, 1, capacity);
    q38::CudaQsaStateBank decode_state(0, 1, capacity);
    batch_state.begin(1);
    decode_state.begin(1);
    const auto batch_cache = batch_state.working(0);
    const auto decode_cache = decode_state.working(0);
    q38::cuda_qsa_prepare_prefill_bf16(
        d_q_gate, d_k, d_v, d_index, d_main_norm, d_main_norm, d_index_norm,
        d_index_norm, query_batch, gate_batch, index_query_batch, batch_cache,
        0, tokens, reinterpret_cast<void*>(stream), 0);
    q38::cuda_qsa_attention_prefill_bf16(
        query_batch, index_query_batch, batch_cache, 0, tokens,
        batch_block_scores, batch_score_stride, batch_selected,
        batch_attention, output_batch, reinterpret_cast<void*>(stream), 0);
    q38::cuda_qsa_apply_grouped_prefill_bf16(
        query_batch, batch_cache, 0, tokens, batch_selected,
        grouped_attention, output_grouped, reinterpret_cast<void*>(stream),
        0);
    q38::cuda_qsa_apply_grouped_fused_prefill_bf16(
        query_batch, batch_cache, 0, tokens, batch_selected,
        grouped_fused_attention, output_grouped_fused,
        reinterpret_cast<void*>(stream), 0);

    for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto query_offset = static_cast<std::size_t>(token) *
                                  q38::kQ38QsaHeads *
                                  q38::kQ38QsaHeadWidth;
        const auto kv_offset = static_cast<std::size_t>(token) *
                               q38::kQ38QsaKvHeads *
                               q38::kQ38QsaHeadWidth;
        const auto index_offset = static_cast<std::size_t>(token) *
                                  (q38::kQ38QsaIndexerHeads + 1) *
                                  q38::kQ38QsaIndexerWidth;
        const auto index_query_offset = static_cast<std::size_t>(token) *
                                        q38::kQ38QsaIndexerHeads *
                                        q38::kQ38QsaIndexerWidth;
        q38::cuda_qsa_prepare_main_decode_bf16(
            d_q_gate + query_offset * 2, d_k + kv_offset, d_v + kv_offset,
            d_main_norm, d_main_norm, query_decode + query_offset,
            gate_decode + query_offset, decode_cache, token,
            reinterpret_cast<void*>(stream), 0);
        q38::cuda_qsa_prepare_index_decode_bf16(
            d_index + index_offset, d_index_norm, d_index_norm,
            index_query_decode + index_query_offset, decode_cache, token,
            reinterpret_cast<void*>(stream), 0);
        const auto selected = q38::cuda_qsa_select_decode(
            index_query_decode + index_query_offset, decode_cache, token,
            decode_block_scores, decode_selected,
            reinterpret_cast<void*>(stream), 0);
        q38::cuda_qsa_attention_decode_bf16(
            query_decode + query_offset, decode_cache, decode_selected,
            selected, decode_attention, output_decode + query_offset,
            reinterpret_cast<void*>(stream), 0);
    }
    std::vector<std::uint16_t> host_batch(query_values);
    std::vector<std::uint16_t> host_grouped(query_values);
    std::vector<std::uint16_t> host_grouped_fused(query_values);
    std::vector<std::uint16_t> host_decode(query_values);
    check(cudaMemcpyAsync(host_batch.data(), output_batch,
                          query_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA batch)");
    check(cudaMemcpyAsync(host_decode.data(), output_decode,
                          query_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA decode)");
    check(cudaMemcpyAsync(host_grouped.data(), output_grouped,
                          query_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA grouped prefill)");
    check(cudaMemcpyAsync(host_grouped_fused.data(), output_grouped_fused,
                          query_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA fused grouped prefill)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(QSA test)");
    require_close(host_batch, host_decode, 0.01f, "QSA prefill parity");
    require_close(host_grouped, host_batch, 0.01f,
                  "QSA grouped prefill parity");
    if (host_grouped_fused != host_grouped)
        throw std::runtime_error(
            "QSA split and fused grouped prefill are not bitwise equal");

    (void)cudaFree(decode_attention);
    (void)cudaFree(decode_selected);
    (void)cudaFree(decode_block_scores);
    (void)cudaFree(batch_attention);
    (void)cudaFree(grouped_attention);
    (void)cudaFree(grouped_fused_attention);
    (void)cudaFree(batch_selected);
    (void)cudaFree(batch_block_scores);
    (void)cudaFree(output_decode);
    (void)cudaFree(index_query_decode);
    (void)cudaFree(gate_decode);
    (void)cudaFree(query_decode);
    (void)cudaFree(output_batch);
    (void)cudaFree(output_grouped);
    (void)cudaFree(output_grouped_fused);
    (void)cudaFree(index_query_batch);
    (void)cudaFree(gate_batch);
    (void)cudaFree(query_batch);
    (void)cudaFree(d_index_norm);
    (void)cudaFree(d_main_norm);
    (void)cudaFree(d_index);
    (void)cudaFree(d_v);
    (void)cudaFree(d_k);
    (void)cudaFree(d_q_gate);
}

void test_qsa_radix_selection(cudaStream_t stream) {
    constexpr std::uint32_t position =
        (q38::kQ38QsaBlockBudget + 1024) * q38::kQ38QsaBlockTokens - 1;
    constexpr std::uint32_t capacity = position + 1;
    constexpr std::uint32_t score_stride =
        (capacity + q38::kQ38QsaBlockTokens - 1) /
        q38::kQ38QsaBlockTokens;
    q38::CudaQsaStateBank state(0, 1, capacity);
    state.begin(1);
    const auto cache = state.working(0);
    const auto main_values = static_cast<std::size_t>(capacity) *
                             q38::kQ38QsaKvHeads *
                             q38::kQ38QsaHeadWidth;
    const auto pooled_values = static_cast<std::size_t>(score_stride) *
                               q38::kQ38QsaIndexerWidth;
    std::vector<std::uint16_t> keys(main_values);
    std::vector<std::uint16_t> values(main_values);
    std::vector<std::uint16_t> pooled(pooled_values);
    for (std::size_t index = 0; index < main_values; ++index) {
        keys[index] = bf16(
            std::sin(static_cast<float>(index % 337) * 0.013f) * 0.05f);
        values[index] = bf16(
            std::cos(static_cast<float>(index % 281) * 0.017f) * 0.07f);
    }
    for (std::size_t index = 0; index < pooled_values; ++index)
        pooled[index] = bf16(
            std::sin(static_cast<float>((index * 17) % 401) * 0.019f) *
            0.08f);
    check(cudaMemcpyAsync(cache.main_keys, keys.data(),
                          keys.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpy(QSA radix keys)");
    check(cudaMemcpyAsync(cache.main_values, values.data(),
                          values.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpy(QSA radix values)");
    check(cudaMemcpyAsync(cache.pooled_index_keys, pooled.data(),
                          pooled.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpy(QSA radix pooled keys)");

    std::vector<std::uint16_t> query_values(
        q38::kQ38QsaHeads * q38::kQ38QsaHeadWidth);
    std::vector<std::uint16_t> index_values(
        q38::kQ38QsaIndexerHeads * q38::kQ38QsaIndexerWidth);
    for (std::size_t index = 0; index < query_values.size(); ++index)
        query_values[index] = bf16(
            std::cos(static_cast<float>(index % 229) * 0.021f) * 0.09f);
    for (std::size_t index = 0; index < index_values.size(); ++index)
        index_values[index] = bf16(
            std::sin(static_cast<float>(index % 149) * 0.027f) * 0.06f);
    auto* query = upload(query_values);
    auto* index_query = upload(index_values);
    auto* batch_scores = allocate<float>(score_stride);
    auto* decode_scores = allocate<float>(
        q38::kQ38ContextLimit / q38::kQ38QsaBlockTokens);
    auto* batch_selected =
        allocate<std::int32_t>(q38::kQ38QsaMaximumSelected);
    auto* decode_selected =
        allocate<std::int32_t>(q38::kQ38QsaMaximumSelected);
    auto* batch_attention = allocate<float>(
        q38::kQ38QsaHeads * q38::kQ38QsaMaximumSelected);
    auto* grouped_attention = allocate<float>(
        q38::kQ38QsaHeads * q38::kQ38QsaMaximumSelected);
    auto* decode_attention = allocate<float>(
        q38::kQ38QsaHeads * q38::kQ38QsaMaximumSelected);
    auto* batch_output = allocate<std::uint16_t>(query_values.size());
    auto* grouped_output = allocate<std::uint16_t>(query_values.size());
    auto* decode_output = allocate<std::uint16_t>(query_values.size());

    q38::cuda_qsa_attention_prefill_bf16(
        query, index_query, cache, position, 1, batch_scores, score_stride,
        batch_selected, batch_attention, batch_output,
        reinterpret_cast<void*>(stream), 0);
    q38::cuda_qsa_apply_grouped_prefill_bf16(
        query, cache, position, 1, batch_selected, grouped_attention,
        grouped_output, reinterpret_cast<void*>(stream), 0);
    const auto selected_count = q38::cuda_qsa_select_decode(
        index_query, cache, position, decode_scores, decode_selected,
        reinterpret_cast<void*>(stream), 0);
    if (selected_count !=
        q38::kQ38QsaBlockBudget * q38::kQ38QsaBlockTokens)
        throw std::runtime_error("QSA radix selection count differs");
    q38::cuda_qsa_attention_decode_bf16(
        query, cache, decode_selected, selected_count, decode_attention,
        decode_output, reinterpret_cast<void*>(stream), 0);

    std::vector<std::int32_t> host_batch_selected(selected_count);
    std::vector<std::int32_t> host_decode_selected(selected_count);
    std::vector<std::uint16_t> host_batch_output(query_values.size());
    std::vector<std::uint16_t> host_grouped_output(query_values.size());
    std::vector<std::uint16_t> host_decode_output(query_values.size());
    check(cudaMemcpyAsync(host_batch_selected.data(), batch_selected,
                          selected_count * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA radix batch selection)");
    check(cudaMemcpyAsync(host_decode_selected.data(), decode_selected,
                          selected_count * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA radix decode selection)");
    check(cudaMemcpyAsync(host_batch_output.data(), batch_output,
                          host_batch_output.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA radix batch output)");
    check(cudaMemcpyAsync(host_decode_output.data(), decode_output,
                          host_decode_output.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA radix decode output)");
    check(cudaMemcpyAsync(host_grouped_output.data(), grouped_output,
                          host_grouped_output.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA radix grouped output)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(QSA radix test)");
    if (host_batch_selected != host_decode_selected)
        throw std::runtime_error("QSA radix selected blocks differ");
    require_close(host_batch_output, host_decode_output, 0.01f,
                  "QSA radix attention parity");
    require_close(host_grouped_output, host_batch_output, 0.01f,
                  "QSA radix grouped attention parity");

    (void)cudaFree(decode_output);
    (void)cudaFree(grouped_output);
    (void)cudaFree(batch_output);
    (void)cudaFree(decode_attention);
    (void)cudaFree(batch_attention);
    (void)cudaFree(grouped_attention);
    (void)cudaFree(decode_selected);
    (void)cudaFree(batch_selected);
    (void)cudaFree(decode_scores);
    (void)cudaFree(batch_scores);
    (void)cudaFree(index_query);
    (void)cudaFree(query);
}

void test_qsa_radix_tie_selection(cudaStream_t stream) {
    constexpr std::uint32_t complete_blocks = q38::kQ38QsaBlockBudget + 32;
    constexpr std::uint32_t capacity =
        complete_blocks * q38::kQ38QsaBlockTokens;
    constexpr std::uint32_t position = capacity - 1;
    q38::CudaQsaStateBank state(0, 1, capacity);
    state.begin(1);
    const auto cache = state.working(0);
    check(cudaMemsetAsync(cache.pooled_index_keys, 0,
                          static_cast<std::size_t>(complete_blocks) *
                              q38::kQ38QsaIndexerWidth *
                              sizeof(std::uint16_t),
                          stream),
          "cudaMemset(QSA radix tie keys)");
    auto* query = allocate<std::uint16_t>(
        q38::kQ38QsaIndexerHeads * q38::kQ38QsaIndexerWidth);
    check(cudaMemsetAsync(query, 0,
                          q38::kQ38QsaIndexerHeads *
                              q38::kQ38QsaIndexerWidth *
                              sizeof(std::uint16_t),
                          stream),
          "cudaMemset(QSA radix tie query)");
    auto* scores = allocate<float>(complete_blocks);
    auto* selected =
        allocate<std::int32_t>(q38::kQ38QsaMaximumSelected);
    const auto selected_count = q38::cuda_qsa_select_decode(
        query, cache, position, scores, selected,
        reinterpret_cast<void*>(stream), 0);
    if (selected_count !=
        q38::kQ38QsaBlockBudget * q38::kQ38QsaBlockTokens)
        throw std::runtime_error("QSA radix tie count differs");
    std::vector<std::int32_t> host_selected(selected_count);
    check(cudaMemcpyAsync(host_selected.data(), selected,
                          selected_count * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA radix tie selection)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(QSA radix tie test)");
    for (std::uint32_t index = 0; index < selected_count; ++index) {
        if (host_selected[index] != static_cast<std::int32_t>(index))
            throw std::runtime_error("QSA radix tie stability differs");
    }
    (void)cudaFree(selected);
    (void)cudaFree(scores);
    (void)cudaFree(query);
}

void test_qsa_prefill_radix_batch_ties(cudaStream_t stream) {
    constexpr std::uint32_t tokens = 4;
    constexpr std::uint32_t complete_blocks =
        q38::kQ38QsaBlockBudget + 32;
    constexpr std::uint32_t capacity =
        complete_blocks * q38::kQ38QsaBlockTokens;
    constexpr std::uint32_t first_position = capacity - tokens;
    constexpr std::uint32_t score_stride = complete_blocks;
    q38::CudaQsaStateBank state(0, 1, capacity);
    state.begin(1);
    const auto cache = state.working(0);
    const auto main_values = static_cast<std::size_t>(capacity) *
                             q38::kQ38QsaKvHeads *
                             q38::kQ38QsaHeadWidth;
    check(cudaMemsetAsync(cache.main_keys, 0,
                          main_values * sizeof(std::uint16_t), stream),
          "cudaMemset(QSA prefill radix tie keys)");
    check(cudaMemsetAsync(cache.main_values, 0,
                          main_values * sizeof(std::uint16_t), stream),
          "cudaMemset(QSA prefill radix tie values)");
    check(cudaMemsetAsync(cache.pooled_index_keys, 0,
                          static_cast<std::size_t>(complete_blocks) *
                              q38::kQ38QsaIndexerWidth *
                              sizeof(std::uint16_t),
                          stream),
          "cudaMemset(QSA prefill radix tie pooled keys)");

    auto* query = allocate<std::uint16_t>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaHeads *
        q38::kQ38QsaHeadWidth);
    auto* index_query = allocate<std::uint16_t>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaIndexerHeads *
        q38::kQ38QsaIndexerWidth);
    check(cudaMemsetAsync(
              query, 0,
              static_cast<std::size_t>(tokens) * q38::kQ38QsaHeads *
                  q38::kQ38QsaHeadWidth * sizeof(std::uint16_t),
              stream),
          "cudaMemset(QSA prefill radix tie query)");
    check(cudaMemsetAsync(
              index_query, 0,
              static_cast<std::size_t>(tokens) * q38::kQ38QsaIndexerHeads *
                  q38::kQ38QsaIndexerWidth * sizeof(std::uint16_t),
              stream),
          "cudaMemset(QSA prefill radix tie index query)");
    auto* scores = allocate<float>(
        static_cast<std::size_t>(tokens) * score_stride);
    auto* selected = allocate<std::int32_t>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaMaximumSelected);
    auto* attention = allocate<float>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaHeads *
        q38::kQ38QsaMaximumSelected);
    auto* output = allocate<std::uint16_t>(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaHeads *
        q38::kQ38QsaHeadWidth);
    q38::cuda_qsa_attention_prefill_bf16(
        query, index_query, cache, first_position, tokens, scores,
        score_stride, selected, attention, output,
        reinterpret_cast<void*>(stream), 0);

    std::vector<std::int32_t> host_selected(
        static_cast<std::size_t>(tokens) * q38::kQ38QsaMaximumSelected);
    check(cudaMemcpyAsync(host_selected.data(), selected,
                          host_selected.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA prefill radix tie selection)");
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(QSA prefill radix tie test)");
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto position = first_position + token;
        const auto token_complete_blocks =
            (position + 1) / q38::kQ38QsaBlockTokens;
        const auto tail = (position + 1) % q38::kQ38QsaBlockTokens;
        const auto base = static_cast<std::size_t>(token) *
                          q38::kQ38QsaMaximumSelected;
        for (std::uint32_t index = 0;
             index < q38::kQ38QsaTokenBudget; ++index) {
            if (host_selected[base + index] !=
                static_cast<std::int32_t>(index))
                throw std::runtime_error(
                    "QSA prefill radix tie block order differs");
        }
        for (std::uint32_t index = 0; index < tail; ++index) {
            const auto expected =
                token_complete_blocks * q38::kQ38QsaBlockTokens + index;
            if (host_selected[base + q38::kQ38QsaTokenBudget + index] !=
                static_cast<std::int32_t>(expected))
                throw std::runtime_error(
                    "QSA prefill radix tie tail differs");
        }
    }

    (void)cudaFree(output);
    (void)cudaFree(attention);
    (void)cudaFree(selected);
    (void)cudaFree(scores);
    (void)cudaFree(index_query);
    (void)cudaFree(query);
}

}  // namespace

int main() {
    int devices = 0;
    const auto probe = cudaGetDeviceCount(&devices);
    if (probe != cudaSuccess || devices == 0) {
        std::cout << "q38 CUDA kernel tests: skipped (no GPU)\n";
        return 0;
    }
    try {
        check(cudaSetDevice(0), "cudaSetDevice(test)");
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "cudaStreamCreate(test)");
        test_quantized_decode_gemv(
            q38::DeviceWeightFormatV1::kW8A16SymG128, stream);
        test_quantized_decode_gemv(
            q38::DeviceWeightFormatV1::kW4A16SymG128, stream);
        test_decode_router_topk(stream);
        test_moe_decode_parity(
            q38::DeviceWeightFormatV1::kW8A16SymG128, stream);
        test_moe_decode_parity(
            q38::DeviceWeightFormatV1::kW4A16SymG128, stream);
        test_moe_route_plan(stream);
        test_tp_moe_route_plan();
        test_grouped_moe_prefill(
            q38::DeviceWeightFormatV1::kW8A16SymG128, stream);
        test_grouped_moe_prefill(
            q38::DeviceWeightFormatV1::kW4A16SymG128, stream);
        std::vector<std::int8_t> weights(2 * 128, 1);
        for (std::size_t index = 128; index < weights.size(); ++index)
            weights[index] = index & 1 ? -1 : 1;
        std::vector<std::uint16_t> scales{bf16(1.0f), bf16(1.0f)};
        std::vector<std::uint16_t> input(128, bf16(1.0f));
        auto* device_weights = upload(weights);
        auto* device_scales = upload(scales);
        auto* device_input = upload(input);
        std::uint16_t* device_output = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&device_output), 128 * sizeof(std::uint16_t)),
              "cudaMalloc(test output)");
        q38::CudaMatrixViewV1 matrix{
            q38::DeviceWeightFormatV1::kW8A16SymG128,
            device_weights, device_scales, 2, 128, 128, false};
        q38::cuda_gemv_bf16(matrix, device_input, device_output, 1,
                            reinterpret_cast<void*>(stream), 0);
        std::vector<std::uint16_t> output(2);
        check(cudaMemcpyAsync(output.data(), device_output, 4,
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpy(test result)");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(test)");
        if (std::fabs(fp32(output[0]) - 128.0f) > 1.0f ||
            std::fabs(fp32(output[1])) > 0.1f)
            throw std::runtime_error("W8 GEMV result differs");

        std::vector<std::uint16_t> norm_weight(128, bf16(0.0f));
        auto* device_norm_weight = upload(norm_weight);
        q38::cuda_qwen38_rmsnorm_bf16(
            device_input, device_norm_weight, false, device_output, 1, 128,
            1.0e-6f, true, reinterpret_cast<void*>(stream), 0);
        check(cudaMemcpyAsync(output.data(), device_output, 4,
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpy(norm result)");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(norm)");
        if (std::fabs(fp32(output[0]) - 1.0f) > 0.02f)
            throw std::runtime_error("RMSNorm result differs");

        test_padded_small_gemm(stream);
        test_paired_norm(stream);
        test_ple_residual(stream);
        test_hc_gemv(stream);
        test_hc_grouped(stream);
        test_multi_projection(stream);
        test_multi_projection(stream, 2560, 10240);
        test_compact_sampling(stream);
        test_boundary_transport_checksum(stream);
        test_boundary_transport_checksum(stream, true);
        test_device_boundary_reuse(stream);
        test_gdn_prefill(stream);
        test_ple_prefill(stream);
        test_qsa_prefill(stream);
        test_qsa_radix_selection(stream);
        test_qsa_radix_tie_selection(stream);
        test_qsa_prefill_radix_batch_ties(stream);

        (void)cudaFree(device_norm_weight);
        (void)cudaFree(device_output);
        (void)cudaFree(device_input);
        (void)cudaFree(device_scales);
        (void)cudaFree(device_weights);
        (void)cudaStreamDestroy(stream);
        std::cout << "q38 CUDA kernel tests: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "q38 CUDA kernel tests: " << error.what() << '\n';
        return 1;
    }
}
