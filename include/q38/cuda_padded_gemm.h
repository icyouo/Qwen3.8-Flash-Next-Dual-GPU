#ifndef Q38_CUDA_PADDED_GEMM_H
#define Q38_CUDA_PADDED_GEMM_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace q38 {

// Caller owns non-overlapping scratch for exactly 8 input/output rows and
// retains it until stream completion. Only real rows are read and published.
inline void cuda_padded_gemm_bf16(
    cublasHandle_t handle, const std::uint16_t* weights,
    const std::uint16_t* input, std::uint16_t* output,
    std::uint16_t* padded_input, std::uint16_t* padded_output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch,
    cudaStream_t stream, void* workspace = nullptr, std::size_t workspace_bytes = 0) {
    if (!handle || !weights || !input || !output || !padded_input || !padded_output ||
        !stream || batch < 2 || batch >= 8 || !rows || !columns ||
        rows > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
        columns > static_cast<unsigned>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid padded small GEMM");
    const auto checked = [](cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("padded GEMM copy: ") + cudaGetErrorString(status));
    };
    checked(cudaMemcpyAsync(padded_input, input,
        static_cast<std::size_t>(batch) * columns * 2, cudaMemcpyDeviceToDevice, stream));
    checked(cudaMemsetAsync(padded_input + static_cast<std::size_t>(batch) * columns,
        0, static_cast<std::size_t>(8 - batch) * columns * 2, stream));
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("padded GEMM stream selection failed");
    if (workspace && cublasSetWorkspace(handle, workspace, workspace_bytes) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("padded GEMM workspace selection failed");
    const float alpha = 1.0f, beta = 0.0f;
    if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 8, columns,
        &alpha, weights, CUDA_R_16BF, columns, padded_input, CUDA_R_16BF, columns,
        &beta, padded_output, CUDA_R_16BF, rows, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("padded small GEMM failed");
    checked(cudaMemcpyAsync(output, padded_output,
        static_cast<std::size_t>(batch) * rows * 2, cudaMemcpyDeviceToDevice, stream));
}

}  // namespace q38
#endif
