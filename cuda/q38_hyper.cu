#include "q38/cuda_hyper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace q38 {

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}
void select_device(int device) {
    if (device < 0) throw std::invalid_argument("invalid hyper device");
    check(cudaSetDevice(device), "cudaSetDevice(hyper)");
}
__device__ __forceinline__ float load(const std::uint16_t* source,
                                       std::uint64_t index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(source)[index]);
}
__device__ __forceinline__ void store(std::uint16_t* destination,
                                       std::uint64_t index, float value) {
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
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__global__ void repeat_embedding_kernel(const std::uint16_t* embedding,
                                         std::uint16_t* hyper) {
    const auto token = blockIdx.y;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= kQ38HyperWidth) return;
    store(hyper, static_cast<std::uint64_t>(token) * kQ38HyperWidth + index,
          load(embedding, static_cast<std::uint64_t>(token) * kQ38HiddenWidth +
                              index % kQ38HiddenWidth));
}

__device__ void group_rmsnorm_impl(const std::uint16_t* hyper,
                                      const std::uint16_t* weight,
                                      std::uint16_t* output,
                                      std::uint32_t stream,
                                      std::uint32_t token) {
    const auto base = static_cast<std::uint64_t>(token) * kQ38HyperWidth +
                      stream * kQ38HiddenWidth;
    float squares = 0.0f;
    for (std::uint32_t index = threadIdx.x; index < kQ38HiddenWidth;
         index += blockDim.x) {
        const float value = load(hyper, base + index);
        squares += value * value;
    }
    const float inverse =
        rsqrtf(block_sum(squares) / kQ38HiddenWidth + 1.0e-6f);
    for (std::uint32_t index = threadIdx.x; index < kQ38HiddenWidth;
         index += blockDim.x)
        store(output, base + index,
              load(hyper, base + index) * inverse *
                  (1.0f + load(weight, stream * kQ38HiddenWidth + index)));
}

__global__ void group_rmsnorm_kernel(const std::uint16_t* hyper,
                                      const std::uint16_t* weight,
                                      std::uint16_t* output) {
    group_rmsnorm_impl(hyper, weight, output, blockIdx.x, blockIdx.y);
}

__global__ void pair_group_rmsnorm_kernel(CudaHyperNormV1 first,
                                           CudaHyperNormV1 second) {
    const auto norm = blockIdx.z == 0 ? first : second;
    group_rmsnorm_impl(norm.input, norm.weight, norm.output,
                       blockIdx.x, blockIdx.y);
}

__global__ void elementwise_kernel(const std::uint16_t* input,
                                    std::uint16_t* output,
                                    std::uint64_t elements, float divisor,
                                    float output_scale, bool silu) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const float value = load(input, index) / divisor;
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    store(output, index, (silu ? value * sigmoid : sigmoid) * output_scale);
}

__global__ void mix_kernel(const std::uint16_t* normalized,
                           const std::uint16_t* weights,
                           std::uint16_t* mixed) {
    const auto token = blockIdx.y;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= kQ38HiddenWidth) return;
    const auto base = static_cast<std::uint64_t>(token) * kQ38HyperWidth;
    float value = 0.0f;
    for (std::uint32_t stream = 0; stream < kQ38HyperCount; ++stream) {
        const auto offset = base + stream * kQ38HiddenWidth + index;
        value += load(normalized, offset) * load(weights, offset);
    }
    store(mixed, static_cast<std::uint64_t>(token) * kQ38HiddenWidth + index,
          value / kQ38HyperCount);
}

template<bool FromProjection = false>
__global__ void inject_kernel(const std::uint16_t* original,
                              const std::uint16_t* block_output,
                              const std::uint16_t* injection,
                              std::uint16_t* output) {
    const auto token = blockIdx.y;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    __shared__ float activated_gate;
    if constexpr(FromProjection) {
        if(threadIdx.x==0){
            const auto gate_index=static_cast<std::uint64_t>(token)*kQ38HyperCount+index/kQ38HiddenWidth;
            const float x=load(injection,gate_index)/float(kQ38HyperCount);
            const float sigmoid=1.f/(1.f+expf(-x));
            activated_gate=__bfloat162float(__float2bfloat16_rn(sigmoid*2.f));
        }
        __syncthreads();
    }
    if (index >= kQ38HyperWidth) return;
    const auto stream = index / kQ38HiddenWidth;
    const auto hidden = index % kQ38HiddenWidth;
    const auto hyper_offset =
        static_cast<std::uint64_t>(token) * kQ38HyperWidth + index;
    const float gate=FromProjection?activated_gate:load(injection,
                       static_cast<std::uint64_t>(token) * kQ38HyperCount+stream);
    store(output, hyper_offset,
          load(original, hyper_offset) +
              load(block_output,
                   static_cast<std::uint64_t>(token) * kQ38HiddenWidth + hidden) *
                  gate);
}

__global__ void add_embedding_kernel(const std::uint16_t* hyper,
    const std::uint16_t* embedding, std::uint16_t* output) {
    const unsigned token=blockIdx.y,index=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned h=token*kQ38HyperWidth+index,e=token*kQ38HiddenWidth+index%kQ38HiddenWidth;
    store(output,h,load(hyper,h)+load(embedding,e));
}
__global__ void moe_inject_kernel(const std::uint16_t* original,
    const std::uint16_t* routed, const std::uint16_t* shared,
    const std::uint16_t* shared_gate, const std::uint16_t* injection,
    std::uint16_t* output) {
    const unsigned token = blockIdx.y;
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float expert_gate, hc_gate;
    if (threadIdx.x == 0) {
        expert_gate = 1.f / (1.f + expf(-load(shared_gate, token)));
        const float x = load(injection, token * kQ38HyperCount + index / kQ38HiddenWidth) / float(kQ38HyperCount);
        const float sigmoid = 1.f / (1.f + expf(-x));
        hc_gate = __bfloat162float(__float2bfloat16_rn(sigmoid * 2.f));
    }
    __syncthreads();
    const unsigned hidden_index = token * kQ38HiddenWidth + index % kQ38HiddenWidth;
    const float combined = __bfloat162float(__float2bfloat16_rn(
        load(routed, hidden_index) + load(shared, hidden_index) * expert_gate));
    const unsigned hyper_index = token * kQ38HyperWidth + index;
    store(output, hyper_index, load(original, hyper_index) + combined * hc_gate);
}

void launch_elementwise(const std::uint16_t* input, std::uint16_t* output,
                        std::uint32_t tokens, std::uint32_t width,
                        float divisor, float scale, bool silu, void* stream,
                        int device) {
    if (!input || !output || !stream || tokens == 0 || width == 0 ||
        !(divisor > 0.0f))
        throw std::invalid_argument("invalid hyper elementwise buffers");
    select_device(device);
    const auto elements = static_cast<std::uint64_t>(tokens) * width;
    const auto blocks = (elements + 255) / 256;
    if (blocks > 0xffffffffu)
        throw std::overflow_error("hyper elementwise grid overflows");
    elementwise_kernel<<<static_cast<unsigned>(blocks), 256, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
        input, output, elements, divisor, scale, silu);
    check(cudaPeekAtLastError(), "hyper elementwise kernel");
}

}  // namespace

void cuda_hyper_repeat_embedding_bf16(const std::uint16_t* embedding,
                                      std::uint16_t* hyper,
                                      std::uint32_t tokens, void* stream,
                                      int device) {
    if (!embedding || !hyper || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper repeat buffers");
    select_device(device);
    repeat_embedding_kernel<<<dim3((kQ38HyperWidth + 255) / 256, tokens), 256,
                              0, reinterpret_cast<cudaStream_t>(stream)>>>(
        embedding, hyper);
    check(cudaPeekAtLastError(), "repeat_embedding_kernel");
}

void cuda_hyper_group_rmsnorm_bf16(
    const std::uint16_t* hyper, const std::uint16_t* norm_weight,
    std::uint16_t* normalized, std::uint32_t tokens, void* stream,
    int device) {
    if (!hyper || !norm_weight || !normalized || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper norm buffers");
    select_device(device);
    group_rmsnorm_kernel<<<dim3(kQ38HyperCount, tokens), 256, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
        hyper, norm_weight, normalized);
    check(cudaPeekAtLastError(), "group_rmsnorm_kernel");
}

void cuda_hyper_pair_group_rmsnorm_bf16(
    const CudaHyperNormV1& first, const CudaHyperNormV1& second,
    std::uint32_t tokens, void* stream, int device) {
    if (!first.input || !first.weight || !first.output || !second.input ||
        !second.weight || !second.output || !stream || tokens == 0 || tokens > 65535)
        throw std::invalid_argument("invalid paired hyper norm buffers/extent");
    const auto words = static_cast<std::uint64_t>(tokens) * kQ38HyperWidth;
    const auto overlap = [](const std::uint16_t* a, std::uint64_t a_words,
                            const std::uint16_t* b, std::uint64_t b_words) {
        const auto x = reinterpret_cast<std::uintptr_t>(a);
        const auto y = reinterpret_cast<std::uintptr_t>(b);
        return x <= y ? y - x < a_words * 2 : x - y < b_words * 2;
    };
    for (const auto* norm : {&first, &second}) {
        if (overlap(norm->output, words, first.input, words) ||
            overlap(norm->output, words, second.input, words) ||
            overlap(norm->output, words, first.weight, kQ38HyperWidth) ||
            overlap(norm->output, words, second.weight, kQ38HyperWidth))
            throw std::invalid_argument("paired hyper norm output overlaps read data");
    }
    if (overlap(first.output, words, second.output, words))
        throw std::invalid_argument("paired hyper norm outputs overlap");
    select_device(device);
    pair_group_rmsnorm_kernel<<<dim3(kQ38HyperCount, tokens, 2), 256, 0,
                                reinterpret_cast<cudaStream_t>(stream)>>>(first, second);
    check(cudaPeekAtLastError(), "pair_group_rmsnorm_kernel");
}

void cuda_hyper_silu_scaled_bf16(
    const std::uint16_t* input, std::uint16_t* output, std::uint32_t tokens,
    std::uint32_t width, float divisor, void* stream, int device) {
    launch_elementwise(input, output, tokens, width, divisor, 1.0f, true,
                       stream, device);
}

void cuda_hyper_sigmoid_bf16(
    const std::uint16_t* input, std::uint16_t* output, std::uint32_t tokens,
    std::uint32_t width, float input_divisor, float output_scale, void* stream,
    int device) {
    launch_elementwise(input, output, tokens, width, input_divisor,
                       output_scale, false, stream, device);
}

void cuda_hyper_mix_bf16(const std::uint16_t* normalized_hyper,
                         const std::uint16_t* mix_weights,
                         std::uint16_t* mixed, std::uint32_t tokens,
                         void* stream, int device) {
    if (!normalized_hyper || !mix_weights || !mixed || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper mix buffers");
    select_device(device);
    mix_kernel<<<dim3((kQ38HiddenWidth + 255) / 256, tokens), 256, 0,
                     reinterpret_cast<cudaStream_t>(stream)>>>(
        normalized_hyper, mix_weights, mixed);
    check(cudaPeekAtLastError(), "hyper mix kernel");
}

void cuda_hyper_inject_bf16(
    const std::uint16_t* original_hyper,
    const std::uint16_t* block_output,
    const std::uint16_t* injection_weights,
    std::uint16_t* output_hyper, std::uint32_t tokens, void* stream,
    int device) {
    if (!original_hyper || !block_output || !injection_weights ||
        !output_hyper || !stream || tokens == 0)
        throw std::invalid_argument("invalid hyper injection buffers");
    select_device(device);
    inject_kernel<false><<<dim3((kQ38HyperWidth + 255) / 256, tokens), 256, 0,
                        reinterpret_cast<cudaStream_t>(stream)>>>(
        original_hyper, block_output, injection_weights, output_hyper);
    check(cudaPeekAtLastError(), "hyper injection kernel");
}

void cuda_hyper_inject_from_projection_bf16(
    const std::uint16_t* original,const std::uint16_t* block,
    const std::uint16_t* projection,std::uint16_t* output,unsigned tokens,void* stream,int device) {
    if(!original||!block||!projection||!output||!tokens||!stream)
        throw std::invalid_argument("invalid fused hyper injection buffers");
    select_device(device);
    // Every 256-element CTA lies wholly within one 2560-element HC stream.
    inject_kernel<true><<<dim3(kQ38HyperWidth/256,tokens),256,0,
        reinterpret_cast<cudaStream_t>(stream)>>>(original,block,projection,output);
    check(cudaPeekAtLastError(),"fused hyper injection kernel");
}

bool cuda_q38_hyper_compiled() { return true; }
void cuda_hyper_add_embedding_bf16(const std::uint16_t* hyper,
    const std::uint16_t* embedding, std::uint16_t* output,
    unsigned tokens, void* stream, int device) {
    if(!hyper||!embedding||!output||!tokens||!stream)
        throw std::invalid_argument("invalid HC broadcast addition");
    select_device(device);
    add_embedding_kernel<<<dim3(kQ38HyperWidth/256,tokens),256,0,
        reinterpret_cast<cudaStream_t>(stream)>>>(hyper,embedding,output);
    check(cudaPeekAtLastError(),"HC broadcast addition");
}
void cuda_hyper_moe_inject_bf16(const std::uint16_t* original,
    const std::uint16_t* routed, const std::uint16_t* shared,
    const std::uint16_t* shared_gate, const std::uint16_t* injection,
    std::uint16_t* output, unsigned tokens, void* stream, int device) {
    if (!original || !routed || !shared || !shared_gate || !injection || !output || !tokens || !stream)
        throw std::invalid_argument("invalid fused MoE HC residual buffers");
    select_device(device);
    moe_inject_kernel<<<dim3(kQ38HyperWidth/256,tokens),256,0,
        reinterpret_cast<cudaStream_t>(stream)>>>(original,routed,shared,shared_gate,injection,output);
    check(cudaPeekAtLastError(),"fused MoE HC residual");
}

}  // namespace q38
