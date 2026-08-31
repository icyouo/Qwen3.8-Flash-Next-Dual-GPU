#include "q38/cuda_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace q38 {

namespace {

constexpr int kThreads = 256;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

void validate_stream(void* stream, int device) {
    if (!stream || device < 0)
        throw std::invalid_argument("invalid CUDA kernel launch context");
    check(cudaSetDevice(device), "cudaSetDevice(kernel)");
}

__device__ __forceinline__ float bf16_load(const std::uint16_t* source,
                                            std::uint64_t index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(source)[index]);
}

__device__ __forceinline__ void bf16_store(std::uint16_t* destination,
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
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? partial[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    return value;
}

__device__ __forceinline__ float matrix_weight(
    int format, const void* data, const void* scales, bool preserved_f32,
    std::uint32_t columns, std::uint32_t group_size, std::uint32_t row,
    std::uint32_t column) {
    const auto element = static_cast<std::uint64_t>(row) * columns + column;
    if (format == static_cast<int>(DeviceWeightFormatV1::kPreserve)) {
        if (preserved_f32) return static_cast<const float*>(data)[element];
        return bf16_load(static_cast<const std::uint16_t*>(data), element);
    }
    const auto scale_index = static_cast<std::uint64_t>(row) *
                                 (columns / group_size) +
                             column / group_size;
    const float scale =
        bf16_load(static_cast<const std::uint16_t*>(scales), scale_index);
    if (format == static_cast<int>(DeviceWeightFormatV1::kW8A16SymG128))
        return static_cast<float>(
                   static_cast<const std::int8_t*>(data)[element]) *
               scale;
    const auto packed = static_cast<const std::uint8_t*>(data)[element >> 1];
    int quantized = (element & 1) ? (packed >> 4) : (packed & 0x0f);
    if (quantized >= 8) quantized -= 16;
    return static_cast<float>(quantized) * scale;
}

__global__ void gemv_bf16_kernel(
    int format, const void* data, const void* scales, bool preserved_f32,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t group_size,
    const std::uint16_t* input, std::uint16_t* output, float alpha,
    bool accumulate) {
    const auto row = blockIdx.x;
    const auto batch = blockIdx.y;
    if (row >= rows) return;
    float sum = 0.0f;
    const auto input_base = static_cast<std::uint64_t>(batch) * columns;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        sum += bf16_load(input, input_base + column) *
               matrix_weight(format, data, scales, preserved_f32, columns,
                             group_size, row, column);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        const auto output_index = static_cast<std::uint64_t>(batch) * rows + row;
        const float previous = accumulate ? bf16_load(output, output_index) : 0.0f;
        bf16_store(output, output_index, previous + alpha * sum);
    }
}

__global__ void dequantize_matrix_bf16_kernel(
    int format, const void* data, const void* scales, bool preserved_f32,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t group_size,
    std::uint16_t* output) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(rows) * columns;
    if (index >= elements) return;
    const auto row = static_cast<std::uint32_t>(index / columns);
    const auto column = static_cast<std::uint32_t>(index % columns);
    bf16_store(output, index,
               matrix_weight(format, data, scales, preserved_f32, columns,
                             group_size, row, column));
}

__global__ void embedding_bf16_kernel(
    int format, const void* data, const void* scales, bool preserved_f32,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t group_size,
    const std::int32_t* token_ids, std::uint16_t* output) {
    const auto token = blockIdx.x;
    const auto row_signed = token_ids[token];
    if (row_signed < 0 || static_cast<std::uint32_t>(row_signed) >= rows) return;
    const auto row = static_cast<std::uint32_t>(row_signed);
    const auto output_base = static_cast<std::uint64_t>(token) * columns;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        bf16_store(output, output_base + column,
                   matrix_weight(format, data, scales, preserved_f32, columns,
                                 group_size, row, column));
    }
}

__global__ void rmsnorm_bf16_kernel(
    const std::uint16_t* input, const void* weight, bool weight_f32,
    std::uint16_t* output, std::uint32_t width, float epsilon,
    bool one_centered_weight) {
    const auto vector = blockIdx.x;
    const auto base = static_cast<std::uint64_t>(vector) * width;
    float squares = 0.0f;
    for (std::uint32_t index = threadIdx.x; index < width;
         index += blockDim.x) {
        const float value = bf16_load(input, base + index);
        squares += value * value;
    }
    squares = block_sum(squares);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0)
        inverse_rms = rsqrtf(squares / static_cast<float>(width) + epsilon);
    __syncthreads();
    for (std::uint32_t index = threadIdx.x; index < width;
         index += blockDim.x) {
        float scale = weight_f32
                          ? static_cast<const float*>(weight)[index]
                          : bf16_load(static_cast<const std::uint16_t*>(weight),
                                      index);
        if (one_centered_weight) scale += 1.0f;
        bf16_store(output, base + index,
                   bf16_load(input, base + index) * inverse_rms * scale);
    }
}

__global__ void silu_gate_bf16_kernel(const std::uint16_t* gate_up,
                                       std::uint16_t* output,
                                       std::uint32_t width) {
    const auto vector = blockIdx.x;
    const auto input_base = static_cast<std::uint64_t>(vector) * width * 2;
    const auto output_base = static_cast<std::uint64_t>(vector) * width;
    for (std::uint32_t index = threadIdx.x; index < width;
         index += blockDim.x) {
        const float gate = bf16_load(gate_up, input_base + index);
        const float up = bf16_load(gate_up, input_base + width + index);
        const float silu = gate / (1.0f + expf(-gate));
        bf16_store(output, output_base + index, silu * up);
    }
}

__global__ void silu_multiply_bf16_kernel(const std::uint16_t* gate,
                                           const std::uint16_t* up,
                                           std::uint16_t* output,
                                           std::size_t elements) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const float value = bf16_load(gate, index);
    bf16_store(output, index,
               value / (1.0f + expf(-value)) * bf16_load(up, index));
}

__global__ void sigmoid_multiply_bf16_kernel(
    const std::uint16_t* values, const std::uint16_t* gates,
    std::uint16_t* output, std::size_t elements) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const float value = bf16_load(values, index);
    const float gate = bf16_load(gates, index);
    bf16_store(output, index, value / (1.0f + expf(-gate)));
}

__global__ void add_bf16_kernel(const std::uint16_t* left,
                                const std::uint16_t* right,
                                std::uint16_t* output,
                                std::size_t elements) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements)
        bf16_store(output, index,
                   bf16_load(left, index) + bf16_load(right, index));
}

__global__ void argmax_bf16_kernel(const std::uint16_t* logits,
                                   std::uint32_t count,
                                   std::int32_t* output) {
    __shared__ float values[256];
    __shared__ std::int32_t indices[256];
    float best = -3.402823466e+38F;
    std::int32_t best_index = -1;
    for (std::uint32_t index = threadIdx.x; index < count;
         index += blockDim.x) {
        const float value = bf16_load(logits, index);
        if (value > best ||
            (value == best &&
             (best_index < 0 ||
              index < static_cast<std::uint32_t>(best_index)))) {
            best = value;
            best_index = static_cast<std::int32_t>(index);
        }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = values[threadIdx.x + stride];
            const auto other_index = indices[threadIdx.x + stride];
            if (other > values[threadIdx.x] ||
                (other == values[threadIdx.x] && other_index >= 0 &&
                 (indices[threadIdx.x] < 0 ||
                  other_index < indices[threadIdx.x]))) {
                values[threadIdx.x] = other;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[0] = indices[0];
}

__global__ void topk_router_bf16_kernel(
    const std::uint16_t* logits, std::int32_t* expert_ids,
    float* expert_weights, std::uint32_t experts, std::uint32_t top_k,
    bool normalize_top_k) {
    if (threadIdx.x != 0) return;
    constexpr std::uint32_t kMaximumTopK = 32;
    const auto token = blockIdx.x;
    const auto base = static_cast<std::uint64_t>(token) * experts;
    float selected_values[kMaximumTopK];
    std::int32_t selected_ids[kMaximumTopK];
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
        selected_values[rank] = -3.402823466e+38F;
        selected_ids[rank] = -1;
    }
    float maximum = -3.402823466e+38F;
    for (std::uint32_t expert = 0; expert < experts; ++expert)
        maximum = fmaxf(maximum, bf16_load(logits, base + expert));
    float all_sum = 0.0f;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        const float probability_numerator =
            expf(bf16_load(logits, base + expert) - maximum);
        all_sum += probability_numerator;
        std::uint32_t position = top_k;
        for (std::uint32_t rank = 0; rank < top_k; ++rank) {
            if (probability_numerator > selected_values[rank]) {
                position = rank;
                break;
            }
        }
        if (position == top_k) continue;
        for (std::uint32_t rank = top_k - 1; rank > position; --rank) {
            selected_values[rank] = selected_values[rank - 1];
            selected_ids[rank] = selected_ids[rank - 1];
        }
        selected_values[position] = probability_numerator;
        selected_ids[position] = static_cast<std::int32_t>(expert);
    }
    float selected_sum = 0.0f;
    for (std::uint32_t rank = 0; rank < top_k; ++rank)
        selected_sum += selected_values[rank];
    const float denominator = normalize_top_k ? selected_sum : all_sum;
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
        const auto output = static_cast<std::uint64_t>(token) * top_k + rank;
        expert_ids[output] = selected_ids[rank];
        expert_weights[output] = selected_values[rank] / denominator;
    }
}

void validate_matrix(const CudaMatrixViewV1& matrix) {
    if (!matrix.data || matrix.rows == 0 || matrix.columns == 0)
        throw std::invalid_argument("invalid CUDA matrix view");
    if (matrix.format == DeviceWeightFormatV1::kW4A16SymG128 ||
        matrix.format == DeviceWeightFormatV1::kW8A16SymG128) {
        if (!matrix.scales || matrix.group_size != 128 ||
            matrix.columns % matrix.group_size != 0)
            throw std::invalid_argument("invalid groupwise CUDA matrix view");
    } else if (matrix.format != DeviceWeightFormatV1::kPreserve) {
        throw std::invalid_argument("unsupported CUDA matrix format");
    }
}

}  // namespace

void cuda_gemv_bf16(const CudaMatrixViewV1& matrix,
                    const std::uint16_t* input, std::uint16_t* output,
                    std::uint32_t batch, void* stream, int device, float alpha,
                    bool accumulate) {
    validate_matrix(matrix);
    validate_stream(stream, device);
    if (!input || !output || batch == 0)
        throw std::invalid_argument("invalid BF16 GEMV buffers");
    const dim3 grid(matrix.rows, batch);
    gemv_bf16_kernel<<<grid, kThreads, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<int>(matrix.format), matrix.data, matrix.scales,
        matrix.preserved_f32, matrix.rows, matrix.columns, matrix.group_size,
        input, output, alpha, accumulate);
    check(cudaPeekAtLastError(), "gemv_bf16_kernel");
}

void cuda_dequantize_matrix_bf16(const CudaMatrixViewV1& matrix,
                                 std::uint16_t* output,
                                 void* stream,
                                 int device) {
    validate_matrix(matrix);
    validate_stream(stream, device);
    if (!output)
        throw std::invalid_argument("invalid matrix dequantization output");
    const auto elements =
        static_cast<std::uint64_t>(matrix.rows) * matrix.columns;
    const auto blocks = (elements + kThreads - 1) / kThreads;
    if (blocks > std::numeric_limits<unsigned>::max())
        throw std::overflow_error("matrix dequantization grid overflows");
    dequantize_matrix_bf16_kernel<<<
        static_cast<unsigned>(blocks), kThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<int>(matrix.format), matrix.data, matrix.scales,
        matrix.preserved_f32, matrix.rows, matrix.columns, matrix.group_size,
        output);
    check(cudaPeekAtLastError(), "dequantize_matrix_bf16_kernel");
}

void cuda_embedding_bf16(const CudaMatrixViewV1& embedding,
                         const std::int32_t* token_ids, std::uint16_t* output,
                         std::uint32_t tokens, void* stream, int device) {
    validate_matrix(embedding);
    validate_stream(stream, device);
    if (!token_ids || !output || tokens == 0)
        throw std::invalid_argument("invalid BF16 embedding buffers");
    embedding_bf16_kernel<<<tokens, kThreads, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<int>(embedding.format), embedding.data, embedding.scales,
        embedding.preserved_f32, embedding.rows, embedding.columns,
        embedding.group_size, token_ids, output);
    check(cudaPeekAtLastError(), "embedding_bf16_kernel");
}

void cuda_qwen38_rmsnorm_bf16(
    const std::uint16_t* input, const void* weight, bool weight_f32,
    std::uint16_t* output, std::uint32_t vectors, std::uint32_t width,
    float epsilon, bool one_centered_weight, void* stream, int device) {
    validate_stream(stream, device);
    if (!input || !weight || !output || vectors == 0 || width == 0 ||
        !(epsilon > 0.0f))
        throw std::invalid_argument("invalid Qwen3.8 RMSNorm buffers");
    rmsnorm_bf16_kernel<<<vectors, kThreads, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
        input, weight, weight_f32, output, width, epsilon,
        one_centered_weight);
    check(cudaPeekAtLastError(), "rmsnorm_bf16_kernel");
}

void cuda_silu_gate_bf16(const std::uint16_t* gate_up,
                         std::uint16_t* output, std::uint32_t vectors,
                         std::uint32_t width, void* stream, int device) {
    validate_stream(stream, device);
    if (!gate_up || !output || vectors == 0 || width == 0)
        throw std::invalid_argument("invalid SiLU gate buffers");
    silu_gate_bf16_kernel<<<vectors, kThreads, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
        gate_up, output, width);
    check(cudaPeekAtLastError(), "silu_gate_bf16_kernel");
}

void cuda_silu_multiply_bf16(const std::uint16_t* gate,
                             const std::uint16_t* up,
                             std::uint16_t* output,
                             std::size_t elements,
                             void* stream,
                             int device) {
    validate_stream(stream, device);
    if (!gate || !up || !output || elements == 0)
        throw std::invalid_argument("invalid separated SiLU buffers");
    const auto blocks = (elements + kThreads - 1) / kThreads;
    if (blocks > std::numeric_limits<unsigned>::max())
        throw std::overflow_error("separated SiLU grid overflows");
    silu_multiply_bf16_kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                                reinterpret_cast<cudaStream_t>(stream)>>>(
        gate, up, output, elements);
    check(cudaPeekAtLastError(), "silu_multiply_bf16_kernel");
}

void cuda_sigmoid_multiply_bf16(const std::uint16_t* values,
                                const std::uint16_t* gates,
                                std::uint16_t* output, std::size_t elements,
                                void* stream, int device) {
    validate_stream(stream, device);
    if (!values || !gates || !output || elements == 0)
        throw std::invalid_argument("invalid sigmoid multiply buffers");
    const auto blocks64 = (elements + kThreads - 1) / kThreads;
    if (blocks64 > std::numeric_limits<unsigned>::max())
        throw std::overflow_error("sigmoid multiply grid overflows");
    sigmoid_multiply_bf16_kernel<<<static_cast<unsigned>(blocks64), kThreads, 0,
                                   reinterpret_cast<cudaStream_t>(stream)>>>(
        values, gates, output, elements);
    check(cudaPeekAtLastError(), "sigmoid_multiply_bf16_kernel");
}

void cuda_add_bf16(const std::uint16_t* left, const std::uint16_t* right,
                   std::uint16_t* output, std::size_t elements, void* stream,
                   int device) {
    validate_stream(stream, device);
    if (!left || !right || !output || elements == 0)
        throw std::invalid_argument("invalid BF16 add buffers");
    const auto blocks64 = (elements + kThreads - 1) / kThreads;
    if (blocks64 > std::numeric_limits<unsigned>::max())
        throw std::overflow_error("BF16 add grid overflows");
    add_bf16_kernel<<<static_cast<unsigned>(blocks64), kThreads, 0,
                      reinterpret_cast<cudaStream_t>(stream)>>>(
        left, right, output, elements);
    check(cudaPeekAtLastError(), "add_bf16_kernel");
}

void cuda_argmax_bf16(const std::uint16_t* logits, std::uint32_t count,
                      std::int32_t* output, void* stream, int device) {
    validate_stream(stream, device);
    if (!logits || !output || count == 0)
        throw std::invalid_argument("invalid BF16 argmax buffers");
    argmax_bf16_kernel<<<1, 256, 0,
                        reinterpret_cast<cudaStream_t>(stream)>>>(logits, count,
                                                                  output);
    check(cudaPeekAtLastError(), "argmax_bf16_kernel");
}

void cuda_topk_router_bf16(
    const std::uint16_t* logits, std::int32_t* expert_ids,
    float* expert_weights, std::uint32_t tokens, std::uint32_t experts,
    std::uint32_t top_k, bool normalize_top_k, void* stream, int device) {
    validate_stream(stream, device);
    if (!logits || !expert_ids || !expert_weights || tokens == 0 ||
        experts == 0 || top_k == 0 || top_k > experts || top_k > 32)
        throw std::invalid_argument("invalid top-k router buffers");
    topk_router_bf16_kernel<<<tokens, 32, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        logits, expert_ids, expert_weights, experts, top_k, normalize_top_k);
    check(cudaPeekAtLastError(), "topk_router_bf16_kernel");
}

bool cuda_q38_kernels_compiled() { return true; }

}  // namespace q38
