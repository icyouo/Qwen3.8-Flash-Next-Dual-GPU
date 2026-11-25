#include "q38/cuda_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace q38 {

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kDecodeWarpsPerBlock = kThreads / kWarpSize;
constexpr std::uint32_t kQuantGroupSize = 128;
constexpr std::uint32_t kMaximumDecodeSharedColumns =
    (48u * 1024u) / sizeof(std::uint16_t);

bool optimized_decode_gemv_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("Q38_CUDA_DECODE_GEMV");
        return !value || (std::strcmp(value, "scalar") != 0 &&
                          std::strcmp(value, "0") != 0 &&
                          std::strcmp(value, "false") != 0);
    }();
    return enabled;
}

bool optimized_decode_topk_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("Q38_CUDA_DECODE_TOPK");
        return !value || (std::strcmp(value, "scalar") != 0 &&
                          std::strcmp(value, "0") != 0 &&
                          std::strcmp(value, "false") != 0);
    }();
    return enabled;
}

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

__device__ float hc_epilogue(float sum, int activation,
                              float divisor, float output_scale) {
    const float projected = 0.0f + 1.0f * sum;
    if (activation == static_cast<int>(CudaHcActivation::kNone))
        return projected;
    // The unfused path writes a BF16 tensor before elementwise_kernel loads
    // it. Preserve that rounding boundary explicitly inside the fused kernel.
    const float value = __bfloat162float(__float2bfloat16_rn(projected)) / divisor;
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    return (activation == static_cast<int>(CudaHcActivation::kSilu)
                ? value * sigmoid : sigmoid) * output_scale;
}

#include "q38_hc_mma.cuh"

__global__ void hc_inject_batch_kernel(const std::uint16_t* weights,const std::uint16_t* input,
    std::uint16_t* output,unsigned rows,unsigned columns,int activation,float divisor,float scale){
    const unsigned row=blockIdx.x,token=blockIdx.y;
    float sum=0.f;
    for(unsigned col=threadIdx.x;col<columns;col+=blockDim.x)
        sum+=bf16_load(input,static_cast<std::uint64_t>(token)*columns+col)*bf16_load(weights,static_cast<std::uint64_t>(row)*columns+col);
    sum=block_sum(sum);
    if(!threadIdx.x)bf16_store(output,token*rows+row,hc_epilogue(sum,activation,divisor,scale));
}

// Independently derived from gemv_bf16_kernel above. Each row retains exactly
// the same column ownership, accumulation expression and two-level reduction.
// Pairing rows reuses the activation load without a new weight representation.
__device__ void hc_pair_gemv_bf16_impl(
    const std::uint16_t* weights, std::uint32_t rows, std::uint32_t columns,
    const std::uint16_t* input, std::uint16_t* output,
    int activation, float divisor, float output_scale, std::uint32_t first) {
    const bool second_valid = first + 1u < rows;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kThreads) {
        const float x = bf16_load(input, column);
        sum0 += x * bf16_load(weights,
            static_cast<std::uint64_t>(first) * columns + column);
        if (second_valid)
            sum1 += x * bf16_load(weights,
                static_cast<std::uint64_t>(first + 1u) * columns + column);
    }
    __shared__ float partial[2][kThreads / kWarpSize];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    if (lane == 0) {
        partial[0][warp] = sum0;
        partial[1][warp] = sum1;
    }
    __syncthreads();
    if (warp == 0) {
        sum0 = lane < kThreads / kWarpSize ? partial[0][lane] : 0.0f;
        sum1 = lane < kThreads / kWarpSize ? partial[1][lane] : 0.0f;
        sum0 = warp_sum(sum0);
        sum1 = warp_sum(sum1);
        if (lane == 0) {
            // Keep the reference's previous + alpha * sum, including signed
            // zero behavior (this API has alpha=1 and accumulate=false).
            bf16_store(output, first, hc_epilogue(sum0, activation, divisor, output_scale));
            if (second_valid)
                bf16_store(output, first + 1u, hc_epilogue(sum1, activation, divisor, output_scale));
        }
    }
}

template<unsigned Batch>
__global__ void hc_scalar_batch_kernel(const std::uint16_t* weights,unsigned rows,unsigned columns,
    const std::uint16_t* input,std::uint16_t* output,int activation,float divisor,float scale){
    const unsigned first=blockIdx.x*2,lane=threadIdx.x&31,warp=threadIdx.x/32;
    const bool second=first+1<rows;
    float s0[Batch]={},s1[Batch]={};
    for(unsigned column=threadIdx.x;column<columns;column+=256){
        const float w0=bf16_load(weights,static_cast<std::uint64_t>(first)*columns+column);
        const float w1=second?bf16_load(weights,static_cast<std::uint64_t>(first+1)*columns+column):0.f;
        #pragma unroll
        for(unsigned t=0;t<Batch;++t){const float x=bf16_load(input,static_cast<std::uint64_t>(t)*columns+column);s0[t]+=x*w0;if(second)s1[t]+=x*w1;}
    }
    __shared__ float partial[Batch][2][8];
    #pragma unroll
    for(unsigned t=0;t<Batch;++t){s0[t]=warp_sum(s0[t]);s1[t]=warp_sum(s1[t]);if(lane==0){partial[t][0][warp]=s0[t];partial[t][1][warp]=s1[t];}}
    __syncthreads();
    if(warp==0){
        #pragma unroll
        for(unsigned t=0;t<Batch;++t){
            const float a=warp_sum(lane<8?partial[t][0][lane]:0.f);
            const float b=warp_sum(lane<8?partial[t][1][lane]:0.f);
            if(lane==0){bf16_store(output,static_cast<std::uint64_t>(t)*rows+first,hc_epilogue(a,activation,divisor,scale));if(second)bf16_store(output,static_cast<std::uint64_t>(t)*rows+first+1,hc_epilogue(b,activation,divisor,scale));}
        }
    }
}

__global__ void hc_pair_gemv_bf16_kernel(
    const std::uint16_t* weights, std::uint32_t rows, std::uint32_t columns,
    const std::uint16_t* input, std::uint16_t* output,
    int activation, float divisor, float output_scale) {
    hc_pair_gemv_bf16_impl(weights, rows, columns, input, output,
        activation, divisor, output_scale, blockIdx.x * 2u);
}

__global__ void hc_grouped_bf16_kernel(
    CudaHcProjectionV1 first, CudaHcProjectionV1 second,
    const std::uint16_t* input, std::uint32_t first_blocks) {
    // A compact concatenated grid avoids padding the four-row inject matrix
    // to the much larger mix_down grid. Selection is uniform within a block.
    const bool use_first = blockIdx.x < first_blocks;
    const auto projection = use_first ? first : second;
    const auto block = use_first ? blockIdx.x : blockIdx.x - first_blocks;
    hc_pair_gemv_bf16_impl(
        static_cast<const std::uint16_t*>(projection.matrix.data),
        projection.matrix.rows, projection.matrix.columns, input,
        projection.output, static_cast<int>(projection.activation),
        projection.divisor, projection.output_scale, block * 2u);
}

__device__ __forceinline__ int sign_extend_int4(unsigned value) {
    return static_cast<int>(value) - ((value & 0x8u) ? 16 : 0);
}

// Batch-1 decode is a GEMV, not a small GEMM.  Giving a 16-row WMMA tile one
// real row wastes 15/16 of the tensor-core work.  This kernel instead assigns
// one output row to each warp, shares the activation across eight rows, loads
// one BF16 scale per 128-value group, and consumes four adjacent weights per
// lane.  It uses the production artifact layout directly; no repack or second
// weight ABI is required.
template <int WeightBits>
__device__ void decode_gemv_group128_impl(
    const void* data, const std::uint16_t* scales, std::uint32_t rows,
    std::uint32_t columns, const std::uint16_t* input,
    std::uint16_t* output, float alpha, bool accumulate, std::uint16_t* cached_input) {
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        cached_input[column] = input[column];
    __syncthreads();

    const auto lane = threadIdx.x & (kWarpSize - 1);
    const auto warp = threadIdx.x / kWarpSize;
    const auto row = static_cast<std::uint32_t>(blockIdx.x) *
                         kDecodeWarpsPerBlock +
                     warp;
    if (row >= rows) return;

    const auto groups = columns / kQuantGroupSize;
    float sum = 0.0f;
    for (std::uint32_t group = 0; group < groups; ++group) {
        float scale = 0.0f;
        if (lane == 0)
            scale = bf16_load(scales,
                              static_cast<std::uint64_t>(row) * groups + group);
        scale = __shfl_sync(0xffffffffu, scale, 0);

        const auto column = group * kQuantGroupSize + lane * 4;
        const float a0 = bf16_load(cached_input, column);
        const float a1 = bf16_load(cached_input, column + 1);
        const float a2 = bf16_load(cached_input, column + 2);
        const float a3 = bf16_load(cached_input, column + 3);
        float dot = 0.0f;
        if constexpr (WeightBits == 8) {
            const auto* weights = static_cast<const std::int8_t*>(data) +
                                  static_cast<std::uint64_t>(row) * columns +
                                  column;
            const auto packed =
                *reinterpret_cast<const char4*>(weights);
            dot = a0 * static_cast<float>(packed.x) +
                  a1 * static_cast<float>(packed.y) +
                  a2 * static_cast<float>(packed.z) +
                  a3 * static_cast<float>(packed.w);
        } else {
            const auto element = static_cast<std::uint64_t>(row) * columns +
                                 column;
            const auto packed = *reinterpret_cast<const std::uint16_t*>(
                static_cast<const std::uint8_t*>(data) + (element >> 1));
            dot = a0 * static_cast<float>(sign_extend_int4(packed & 0x0fu)) +
                  a1 * static_cast<float>(
                           sign_extend_int4((packed >> 4) & 0x0fu)) +
                  a2 * static_cast<float>(
                           sign_extend_int4((packed >> 8) & 0x0fu)) +
                  a3 * static_cast<float>(
                           sign_extend_int4((packed >> 12) & 0x0fu));
        }
        sum = fmaf(dot, scale, sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        const float previous = accumulate ? bf16_load(output, row) : 0.0f;
        bf16_store(output, row, previous + alpha * sum);
    }
}

template <int WeightBits>
__global__ void decode_gemv_group128_kernel(
    const void* data, const std::uint16_t* scales, std::uint32_t rows,
    std::uint32_t columns, const std::uint16_t* input,
    std::uint16_t* output, float alpha, bool accumulate) {
    extern __shared__ std::uint16_t cached_input[];
    decode_gemv_group128_impl<WeightBits>(data, scales, rows, columns,
        input, output, alpha, accumulate, cached_input);
}

#include "q38_gemv_batch.cuh"

__global__ void multi_projection_decode_kernel(
    CudaProjectionPackV1 pack, const std::uint16_t* input) {
    const auto matrix = pack.matrices[blockIdx.y];
    auto* output = pack.outputs[blockIdx.y];
    extern __shared__ std::uint16_t cached_input[];
    if (matrix.format == DeviceWeightFormatV1::kW4A16SymG128) {
        decode_gemv_group128_impl<4>(matrix.data,
            static_cast<const std::uint16_t*>(matrix.scales), matrix.rows,
            matrix.columns, input, output, 1.0f, false, cached_input);
    } else if (matrix.format == DeviceWeightFormatV1::kW8A16SymG128) {
        decode_gemv_group128_impl<8>(matrix.data,
            static_cast<const std::uint16_t*>(matrix.scales), matrix.rows,
            matrix.columns, input, output, 1.0f, false, cached_input);
    } else {
        // Preserve the scalar reference reduction for BF16/FP32 projections.
        const auto row = blockIdx.x;
        if (row >= matrix.rows) return;
        float sum = 0.0f;
        for (std::uint32_t column = threadIdx.x; column < matrix.columns;
             column += blockDim.x)
            sum += bf16_load(input, column) * matrix_weight(
                static_cast<int>(matrix.format), matrix.data, matrix.scales,
                matrix.preserved_f32, matrix.columns, matrix.group_size,
                row, column);
        sum = block_sum(sum);
        if (threadIdx.x == 0) bf16_store(output, row, sum);
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

__device__ __forceinline__ void sigmoid_multiply(
    const std::uint16_t* values, const std::uint16_t* gates,
    std::uint16_t* output, std::size_t source, std::size_t target) {
    const float value = bf16_load(values, source);
    const float gate = bf16_load(gates, source);
    bf16_store(output, target, value / (1.0f + expf(-gate)));
}

__global__ void sigmoid_multiply_bf16_kernel(
    const std::uint16_t* values, const std::uint16_t* gates,
    std::uint16_t* output, std::size_t elements) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    sigmoid_multiply(values, gates, output, index, index);
}

__global__ void sigmoid_multiply_slice_bf16_kernel(
    const std::uint16_t* values, const std::uint16_t* gates,
    std::uint16_t* output, std::uint32_t input_width,
    std::uint32_t first_column, std::uint32_t output_width) {
    const auto column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width) return;
    const auto token = static_cast<std::size_t>(blockIdx.y);
    sigmoid_multiply(values, gates, output,
                     token * input_width + first_column + column,
                     token * output_width + column);
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

__device__ void stable_max(float& value,int& id,float other,int other_id){
    if(other>value||(other==value&&other_id>=0&&(id<0||other_id<id))){value=other;id=other_id;}
}
__global__ void argmax_partials_kernel(const std::uint16_t* logits,unsigned count,unsigned rows,
    float* partial_values,int* partial_ids){
    __shared__ float values[256];__shared__ int ids[256];
    float best=-3.402823466e+38F;int id=-1;
    const unsigned row=blockIdx.y;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        stable_max(best,id,bf16_load(logits,static_cast<std::uint64_t>(row)*count+i),int(i));
    values[threadIdx.x]=best;ids[threadIdx.x]=id;__syncthreads();
    for(unsigned step=128;step;step/=2){if(threadIdx.x<step)
        stable_max(values[threadIdx.x],ids[threadIdx.x],values[threadIdx.x+step],ids[threadIdx.x+step]);__syncthreads();}
    if(!threadIdx.x){partial_values[row*128+blockIdx.x]=values[0];partial_ids[row*128+blockIdx.x]=ids[0];}
}
__global__ void argmax_finalize_kernel(const float* input,const int* indices,int* output){
    __shared__ float values[256];__shared__ int ids[256];const unsigned row=blockIdx.x,t=threadIdx.x;
    values[t]=t<128?input[row*128+t]:-3.402823466e+38F;ids[t]=t<128?indices[row*128+t]:-1;__syncthreads();
    for(unsigned step=128;step;step/=2){if(t<step)stable_max(values[t],ids[t],values[t+step],ids[t+step]);__syncthreads();}
    if(!t)output[row]=ids[0];
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

#include "q38_topk_warp.cuh"

// Qwen3.8 decode routes one token over exactly 512 experts.  The reference
// kernel above intentionally handles arbitrary shapes, but only thread 0 does
// its 512 exponentials and insertion sort.  This specialization loads two
// logits per thread, sorts them in shared memory with stable expert-id tie
// breaking, and evaluates expf only for the selected top-k entries.
__global__ void topk_router_512_decode_kernel(
    const std::uint16_t* logits, std::int32_t* expert_ids,
    float* expert_weights, std::uint32_t top_k) {
    constexpr std::uint32_t kExperts = 512;
    __shared__ float values[kExperts];
    __shared__ std::int32_t ids[kExperts];
    __shared__ float numerators[32];
    __shared__ float denominator;
    const auto token = blockIdx.x;
    const auto base = static_cast<std::uint64_t>(token) * kExperts;
    for (std::uint32_t expert = threadIdx.x; expert < kExperts;
         expert += blockDim.x) {
        values[expert] = bf16_load(logits, base + expert);
        ids[expert] = static_cast<std::int32_t>(expert);
    }
    __syncthreads();

    // Ascending bitonic order by (logit, -expert_id).  Reading from the end
    // therefore produces descending logits and the lowest id first on ties,
    // matching the reference insertion order.
    for (std::uint32_t width = 2; width <= kExperts; width <<= 1) {
        for (std::uint32_t stride = width >> 1; stride != 0; stride >>= 1) {
            for (std::uint32_t left = threadIdx.x; left < kExperts;
                 left += blockDim.x) {
                const auto right = left ^ stride;
                if (right <= left) continue;
                const bool ascending = (left & width) == 0;
                const bool left_better =
                    values[left] > values[right] ||
                    (values[left] == values[right] && ids[left] < ids[right]);
                const bool right_better =
                    values[right] > values[left] ||
                    (values[right] == values[left] && ids[right] < ids[left]);
                if ((ascending && left_better) ||
                    (!ascending && right_better)) {
                    const auto value = values[left];
                    values[left] = values[right];
                    values[right] = value;
                    const auto id = ids[left];
                    ids[left] = ids[right];
                    ids[right] = id;
                }
            }
            __syncthreads();
        }
    }

    float numerator = 0.0f;
    if (threadIdx.x < top_k) {
        const auto sorted = kExperts - 1 - threadIdx.x;
        numerator = expf(values[sorted] - values[kExperts - 1]);
        numerators[threadIdx.x] = numerator;
        expert_ids[static_cast<std::uint64_t>(token) * top_k + threadIdx.x] =
            ids[sorted];
    }
    __syncthreads();
    // Preserve the reference kernel's rank-order FP32 accumulation.  Only ten
    // additions are serialized in production, while identical rounding keeps
    // greedy trajectories from drifting solely because routing weights used a
    // different reduction tree.
    if (threadIdx.x == 0) {
        denominator = 0.0f;
        for (std::uint32_t rank = 0; rank < top_k; ++rank)
            denominator += numerators[rank];
    }
    __syncthreads();
    if (threadIdx.x < top_k)
        expert_weights[static_cast<std::uint64_t>(token) * top_k +
                       threadIdx.x] = numerators[threadIdx.x] / denominator;
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

bool cuda_small_batch_gemv_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input, std::uint16_t* output, std::uint32_t batch,
    void* stream, int device) {
    if (matrix.group_size != 128 || (batch != 2 && batch != 3 && batch != 5 && batch != 7) ||
        static_cast<std::size_t>(matrix.columns)*batch*2 > 48*1024 ||
        (matrix.format != DeviceWeightFormatV1::kW4A16SymG128 && matrix.format != DeviceWeightFormatV1::kW8A16SymG128)) return false;
    validate_matrix(matrix);validate_stream(stream,device);
    if (!input || !output) throw std::invalid_argument("invalid small batch buffers");
    if(try_head_pipelined_gemv(matrix,input,output,batch,stream))return true;
    auto launch=[&](auto bits_tag,auto batch_tag){
        constexpr int Bits=decltype(bits_tag)::value;constexpr unsigned Batch=decltype(batch_tag)::value;
        small_batch_gemv_group128_kernel<Bits,Batch><<<(matrix.rows+7)/8,256,
            static_cast<std::size_t>(matrix.columns)*Batch*2,reinterpret_cast<cudaStream_t>(stream)>>>(
            matrix.data,static_cast<const std::uint16_t*>(matrix.scales),matrix.rows,matrix.columns,input,output);
    };
    auto shape=[&](auto bits){
        if(batch==2)launch(bits,std::integral_constant<unsigned,2>{});
        else if(batch==3)launch(bits,std::integral_constant<unsigned,3>{});
        else if(batch==5)launch(bits,std::integral_constant<unsigned,5>{});
        else launch(bits,std::integral_constant<unsigned,7>{});
    };
    if(matrix.format==DeviceWeightFormatV1::kW8A16SymG128)shape(std::integral_constant<int,8>{});
    else shape(std::integral_constant<int,4>{});
    check(cudaPeekAtLastError(),"small batch compact GEMV");return true;
}

void cuda_gemv_bf16(const CudaMatrixViewV1& matrix,
                    const std::uint16_t* input, std::uint16_t* output,
                    std::uint32_t batch, void* stream, int device, float alpha,
                    bool accumulate) {
    validate_matrix(matrix);
    validate_stream(stream, device);
    if (!input || !output || batch == 0)
        throw std::invalid_argument("invalid BF16 GEMV buffers");
    if (batch==1 && alpha==1.f && !accumulate &&
        try_head_pipelined_gemv(matrix,input,output,batch,stream))return;
    if (batch == 1 && optimized_decode_gemv_enabled() &&
        matrix.group_size == kQuantGroupSize &&
        matrix.columns <= kMaximumDecodeSharedColumns &&
        (matrix.format == DeviceWeightFormatV1::kW4A16SymG128 ||
         matrix.format == DeviceWeightFormatV1::kW8A16SymG128)) {
        const auto blocks =
            (matrix.rows + kDecodeWarpsPerBlock - 1) / kDecodeWarpsPerBlock;
        const auto shared_bytes =
            static_cast<std::size_t>(matrix.columns) * sizeof(std::uint16_t);
        if (matrix.format == DeviceWeightFormatV1::kW8A16SymG128) {
            decode_gemv_group128_kernel<8>
                <<<blocks, kThreads, shared_bytes,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
                    matrix.data,
                    static_cast<const std::uint16_t*>(matrix.scales),
                    matrix.rows, matrix.columns, input, output, alpha,
                    accumulate);
        } else {
            decode_gemv_group128_kernel<4>
                <<<blocks, kThreads, shared_bytes,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
                    matrix.data,
                    static_cast<const std::uint16_t*>(matrix.scales),
                    matrix.rows, matrix.columns, input, output, alpha,
                    accumulate);
        }
        check(cudaPeekAtLastError(), "decode_gemv_group128_kernel");
        return;
    }
    const dim3 grid(matrix.rows, batch);
    gemv_bf16_kernel<<<grid, kThreads, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<int>(matrix.format), matrix.data, matrix.scales,
        matrix.preserved_f32, matrix.rows, matrix.columns, matrix.group_size,
        input, output, alpha, accumulate);
    check(cudaPeekAtLastError(), "gemv_bf16_kernel");
}

bool cuda_try_hc_inject_batch_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input,std::uint16_t* output,unsigned tokens,void* stream,int device,
    CudaHcActivation activation,float divisor,float scale){
    if(matrix.format!=DeviceWeightFormatV1::kPreserve || matrix.preserved_f32 ||
       matrix.rows!=4 || matrix.columns!=10240 || tokens<2 || tokens>7 || !matrix.data || !input || !output || !stream)return false;
    check(cudaSetDevice(device),"select HC inject device");
    hc_inject_batch_kernel<<<dim3(matrix.rows,tokens),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const std::uint16_t*>(matrix.data),input,output,matrix.rows,matrix.columns,int(activation),divisor,scale);
    check(cudaPeekAtLastError(),"HC inject batch");return true;
}

bool cuda_try_hc_mix_tensor_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* low,const std::uint16_t* normalized,std::uint16_t* mixed,
    unsigned tokens,void* stream,int device){
    if(matrix.format!=DeviceWeightFormatV1::kPreserve || matrix.preserved_f32 || matrix.rows!=10240 || matrix.columns!=320 || tokens<2 || tokens>7 || !stream || !matrix.data || !low || !normalized || !mixed)return false;
    check(cudaSetDevice(device),"select HC mixed tensor device");
    hc_mma_mix_kernel<<<320,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const std::uint16_t*>(matrix.data),low,normalized,mixed,tokens);
    check(cudaPeekAtLastError(),"HC fused tensor mixture");return true;
}
bool cuda_try_hc_tensor_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input, std::uint16_t* output, float* scratch,
    unsigned tokens, void* stream, int device, CudaHcActivation activation,
    float divisor, float scale) {
    if(matrix.format!=DeviceWeightFormatV1::kPreserve || matrix.preserved_f32 ||
        !matrix.data || matrix.rows%32 || matrix.columns%16 || tokens<2 || tokens>7 ||
        !input || !output || !scratch || !stream) return false;
    check(cudaSetDevice(device),"set HC tensor device");
    const unsigned splits=matrix.columns>=10240?16:1;
    auto st=reinterpret_cast<cudaStream_t>(stream);
    if(splits==1)
        hc_mma_projection_kernel<false><<<dim3(matrix.rows/32,1),128,0,st>>>(
            static_cast<const std::uint16_t*>(matrix.data),input,output,scratch,matrix.rows,matrix.columns,tokens,splits,int(activation),divisor,scale);
    else {
        hc_mma_projection_kernel<true><<<dim3(matrix.rows/32,splits),128,0,st>>>(
            static_cast<const std::uint16_t*>(matrix.data),input,output,scratch,matrix.rows,matrix.columns,tokens,splits,int(activation),divisor,scale);
        hc_mma_reduce_kernel<<<(matrix.rows*tokens+255)/256,256,0,st>>>(scratch,output,matrix.rows,tokens,splits,int(activation),divisor,scale);
    }
    check(cudaPeekAtLastError(),"HC tensor core projection");
    return true;
}

bool cuda_try_hc_gemv_bf16(const CudaMatrixViewV1& matrix,
                           const std::uint16_t* input,
                           std::uint16_t* output,
                           void* stream, int device,
                           CudaHcActivation activation, float divisor,
                           float output_scale) {
    if (matrix.format != DeviceWeightFormatV1::kPreserve ||
        matrix.preserved_f32 ||
        (matrix.columns != 320 && matrix.columns != 10240))
        return false;
    validate_matrix(matrix);
    validate_stream(stream, device);
    if (!input || !output || !(divisor > 0.0f) || !std::isfinite(divisor) ||
        !std::isfinite(output_scale) ||
        (activation != CudaHcActivation::kNone &&
         activation != CudaHcActivation::kSilu &&
         activation != CudaHcActivation::kSigmoid))
        throw std::invalid_argument("invalid HC GEMV buffers/epilogue");
    hc_pair_gemv_bf16_kernel<<<(matrix.rows + 1u) / 2u, kThreads, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const std::uint16_t*>(matrix.data), matrix.rows,
        matrix.columns, input, output, static_cast<int>(activation),
        divisor, output_scale);
    check(cudaPeekAtLastError(), "hc_pair_gemv_bf16_kernel");
    return true;
}

bool cuda_try_hc_scalar_batch_bf16(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input,std::uint16_t* output,unsigned tokens,void* stream,int device,
    CudaHcActivation activation,float divisor,float scale) {
    if(matrix.format!=DeviceWeightFormatV1::kPreserve || matrix.preserved_f32 ||
       (matrix.columns!=320 && matrix.columns!=10240) ||
       (tokens!=2 && tokens!=3 && tokens!=5 && tokens!=7))return false;
    validate_matrix(matrix);validate_stream(stream,device);
    if(!input || !output || !(divisor>0.f) || !std::isfinite(divisor) || !std::isfinite(scale))throw std::invalid_argument("invalid scalar HC batch");
    auto launch=[&](auto tag){constexpr unsigned Batch=decltype(tag)::value;
        hc_scalar_batch_kernel<Batch><<<(matrix.rows+1)/2,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const std::uint16_t*>(matrix.data),matrix.rows,matrix.columns,input,output,int(activation),divisor,scale);};
    if(tokens==2)launch(std::integral_constant<unsigned,2>{});
    else if(tokens==3)launch(std::integral_constant<unsigned,3>{});
    else if(tokens==5)launch(std::integral_constant<unsigned,5>{});
    else launch(std::integral_constant<unsigned,7>{});
    check(cudaPeekAtLastError(),"scalar HC batch");return true;
}

bool cuda_try_hc_grouped_bf16(const CudaHcProjectionV1& first,
                               const CudaHcProjectionV1& second,
                               const std::uint16_t* input,
                               void* stream, int device) {
    if (first.matrix.columns != second.matrix.columns) return false;
    for (const auto* projection : {&first, &second}) {
        const auto& matrix = projection->matrix;
        if (matrix.format != DeviceWeightFormatV1::kPreserve ||
            matrix.preserved_f32 ||
            (matrix.columns != 320 && matrix.columns != 10240))
            return false;
    }
    for (const auto* projection : {&first, &second}) {
        validate_matrix(projection->matrix);
        const auto activation = projection->activation;
        if (!projection->output || !(projection->divisor > 0.0f) ||
            !std::isfinite(projection->divisor) ||
            !std::isfinite(projection->output_scale) ||
            (activation != CudaHcActivation::kNone &&
             activation != CudaHcActivation::kSilu &&
             activation != CudaHcActivation::kSigmoid))
            throw std::invalid_argument("invalid grouped HC output/epilogue");
    }
    const auto overlaps = [](const std::uint16_t* a, std::uint32_t count_a,
                             const std::uint16_t* b, std::uint32_t count_b) {
        const auto begin_a = reinterpret_cast<std::uintptr_t>(a);
        const auto begin_b = reinterpret_cast<std::uintptr_t>(b);
        return begin_a <= begin_b
            ? begin_b - begin_a < static_cast<std::uint64_t>(count_a) * 2
            : begin_a - begin_b < static_cast<std::uint64_t>(count_b) * 2;
    };
    if (!input || overlaps(first.output, first.matrix.rows,
                            second.output, second.matrix.rows) ||
        overlaps(input, first.matrix.columns, first.output, first.matrix.rows) ||
        overlaps(input, first.matrix.columns, second.output, second.matrix.rows))
        throw std::invalid_argument("invalid grouped HC input/alias");
    validate_stream(stream, device);
    const auto first_blocks = (static_cast<std::uint64_t>(first.matrix.rows) + 1) / 2;
    const auto total_blocks = first_blocks +
        (static_cast<std::uint64_t>(second.matrix.rows) + 1) / 2;
    if (total_blocks > 2147483647u)
        throw std::overflow_error("grouped HC grid exceeds CUDA limit");
    hc_grouped_bf16_kernel<<<static_cast<unsigned>(total_blocks), kThreads, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
        first, second, input, static_cast<std::uint32_t>(first_blocks));
    check(cudaPeekAtLastError(), "hc_grouped_bf16_kernel");
    return true;
}

void cuda_multi_projection_decode_bf16(
    const CudaProjectionPackV1& pack, const std::uint16_t* input,
    void* stream, int device) {
    if (pack.count == 0 || pack.count > 4 || !input)
        throw std::invalid_argument("invalid projection pack");
    validate_stream(stream, device);
    std::uint32_t blocks = 0;
    const auto columns = pack.matrices[0].columns;
    for (std::uint32_t i = 0; i < pack.count; ++i) {
        const auto& m = pack.matrices[i];
        validate_matrix(m);
        if (m.columns != columns || !pack.outputs[i] || columns > 16384 ||
            (m.format != DeviceWeightFormatV1::kPreserve && m.group_size != 128))
            throw std::invalid_argument("incompatible projection pack");
        const auto rows = m.format == DeviceWeightFormatV1::kPreserve
                              ? m.rows : (m.rows + 7u) / 8u;
        blocks = std::max(blocks, rows);
    }
    if (!optimized_decode_gemv_enabled()) {
        for (std::uint32_t i = 0; i < pack.count; ++i)
            cuda_gemv_bf16(pack.matrices[i], input, pack.outputs[i], 1,
                           stream, device);
        return;
    }
    multi_projection_decode_kernel<<<dim3(blocks, pack.count), 256,
        columns * sizeof(std::uint16_t),
        reinterpret_cast<cudaStream_t>(stream)>>>(pack, input);
    check(cudaPeekAtLastError(), "multi_projection_decode_kernel");
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

void cuda_sigmoid_multiply_slice_bf16(
    const std::uint16_t* values, const std::uint16_t* gates,
    std::uint16_t* output, std::uint32_t tokens, std::uint32_t input_width,
    std::uint32_t first_column, std::uint32_t output_width, void* stream,
    int device) {
    validate_stream(stream, device);
    if (!values || !gates || !output || tokens == 0 || tokens > 65535 ||
        output_width == 0 || first_column >= input_width ||
        output_width > input_width - first_column)
        throw std::invalid_argument("invalid sigmoid multiply slice");
    const dim3 grid((output_width - 1) / kThreads + 1, tokens);
    sigmoid_multiply_slice_bf16_kernel<<<grid, kThreads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            values, gates, output, input_width, first_column, output_width);
    check(cudaPeekAtLastError(), "sigmoid_multiply_slice_bf16_kernel");
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

void cuda_argmax_bf16_batch(const std::uint16_t* logits,unsigned count,unsigned rows,
    float* partial_values,int* partial_ids,int* output,void* stream,int device){
    validate_stream(stream,device);
    if(!logits||!partial_values||!partial_ids||!output||!count||!rows||rows>7)
        throw std::invalid_argument("invalid batched argmax buffers");
    auto st=reinterpret_cast<cudaStream_t>(stream);
    argmax_partials_kernel<<<dim3(128,rows),256,0,st>>>(logits,count,rows,partial_values,partial_ids);
    check(cudaPeekAtLastError(),"argmax partials");
    argmax_finalize_kernel<<<rows,256,0,st>>>(partial_values,partial_ids,output);
    check(cudaPeekAtLastError(),"argmax finalization");
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
    const auto* warp = std::getenv("Q38_CUDA_TOPK_WARP");
    if (experts == 512 && normalize_top_k && warp && std::strcmp(warp,"1")==0) {
        topk_router_512_warp_kernel<<<tokens,32,0,reinterpret_cast<cudaStream_t>(stream)>>>(logits,expert_ids,expert_weights,top_k);
        check(cudaPeekAtLastError(),"topk_router_512_warp_kernel");
    } else if (experts == 512 && normalize_top_k && optimized_decode_topk_enabled()) {
        topk_router_512_decode_kernel<<<
            tokens, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            logits, expert_ids, expert_weights, top_k);
        check(cudaPeekAtLastError(), "topk_router_512_decode_kernel");
    } else {
        topk_router_bf16_kernel<<<tokens, 32, 0,
                                 reinterpret_cast<cudaStream_t>(stream)>>>(
            logits, expert_ids, expert_weights, experts, top_k,
            normalize_top_k);
        check(cudaPeekAtLastError(), "topk_router_bf16_kernel");
    }
}

bool cuda_q38_kernels_compiled() { return true; }

}  // namespace q38
