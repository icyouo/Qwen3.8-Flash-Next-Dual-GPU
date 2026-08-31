#include "q38/cuda_gdn.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace q38 {

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

void select_device(int device) {
    if (device < 0) throw std::invalid_argument("invalid CUDA GDN device");
    check(cudaSetDevice(device), "cudaSetDevice(GDN)");
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

__device__ float block_sum_128(float value) {
    __shared__ float partial[4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < 4 ? partial[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__global__ void gdn_conv_decode_kernel(
    const std::uint16_t* projected, const std::uint16_t* weight,
    std::uint16_t* state, std::uint16_t* activated) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (channel >= kQ38GdnQkvWidth) return;
    const auto base = static_cast<std::uint64_t>(channel) * kQ38GdnConvWidth;
    float convolved = 0.0f;
    for (std::uint32_t index = 0; index + 1 < kQ38GdnConvWidth; ++index) {
        const float shifted = bf16_load(state, base + index + 1);
        bf16_store(state, base + index, shifted);
        convolved += shifted * bf16_load(weight, base + index);
    }
    const float newest = bf16_load(projected, channel);
    bf16_store(state, base + kQ38GdnConvWidth - 1, newest);
    convolved += newest *
                 bf16_load(weight, base + kQ38GdnConvWidth - 1);
    const float silu = convolved / (1.0f + expf(-convolved));
    bf16_store(activated, channel, silu);
}

__global__ void gdn_conv_prefill_kernel(
    const std::uint16_t* projected, const std::uint16_t* weight,
    std::uint16_t* state, std::uint16_t* activated, std::uint32_t tokens) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (channel >= kQ38GdnQkvWidth) return;
    const auto state_base =
        static_cast<std::uint64_t>(channel) * kQ38GdnConvWidth;
    const auto weight_base = state_base;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        float convolved = 0.0f;
        for (std::uint32_t index = 0; index + 1 < kQ38GdnConvWidth; ++index) {
            const float shifted = bf16_load(state, state_base + index + 1);
            bf16_store(state, state_base + index, shifted);
            convolved += shifted * bf16_load(weight, weight_base + index);
        }
        const auto token_offset =
            static_cast<std::uint64_t>(token) * kQ38GdnQkvWidth + channel;
        const float newest = bf16_load(projected, token_offset);
        bf16_store(state, state_base + kQ38GdnConvWidth - 1, newest);
        convolved += newest *
                     bf16_load(weight, weight_base + kQ38GdnConvWidth - 1);
        bf16_store(activated, token_offset,
                   convolved / (1.0f + expf(-convolved)));
    }
}

__global__ void gdn_recurrent_decode_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* state, std::uint16_t* output) {
    const auto value_head = blockIdx.x;
    const auto element = threadIdx.x;
    const auto key_head = value_head / (kQ38GdnValueHeads / kQ38GdnKeyHeads);
    const auto query_base = key_head * kQ38GdnHeadWidth;
    const auto key_base = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_base;
    const auto value_base = 2 * kQ38GdnKeyHeads * kQ38GdnHeadWidth +
                            value_head * kQ38GdnHeadWidth;
    const float query_raw = bf16_load(qkv, query_base + element);
    const float key_raw = bf16_load(qkv, key_base + element);
    float query_norm = block_sum_128(query_raw * query_raw);
    float key_norm = block_sum_128(key_raw * key_raw);
    __shared__ float query_scale;
    __shared__ float key_scale;
    __shared__ float decay;
    __shared__ float beta;
    if (threadIdx.x == 0) {
        query_scale = rsqrtf(query_norm + 1.0e-6f) *
                      rsqrtf(static_cast<float>(kQ38GdnHeadWidth));
        key_scale = rsqrtf(key_norm + 1.0e-6f);
        const float a = bf16_load(projected_a, value_head);
        const float b = bf16_load(projected_b, value_head);
        const float shifted_a = a + bf16_load(dt_bias, value_head);
        const float softplus = shifted_a > 20.0f
                                   ? shifted_a
                                   : log1pf(expf(shifted_a));
        decay = expf(-expf(bf16_load(a_log, value_head)) * softplus);
        beta = 1.0f / (1.0f + expf(-b));
    }
    __syncthreads();
    float memory_value = 0.0f;
    const auto state_base = static_cast<std::uint64_t>(value_head) *
                            kQ38GdnHeadWidth * kQ38GdnHeadWidth;
    for (std::uint32_t key_element = 0; key_element < kQ38GdnHeadWidth;
         ++key_element) {
        const auto state_index = state_base +
                                 static_cast<std::uint64_t>(key_element) *
                                     kQ38GdnHeadWidth +
                                 element;
        const float decayed = state[state_index] * decay;
        state[state_index] = decayed;
        const float key = bf16_load(qkv, key_base + key_element) * key_scale;
        memory_value += decayed * key;
    }
    const float value = bf16_load(qkv, value_base + element);
    const float delta = (value - memory_value) * beta;
    float core = 0.0f;
    for (std::uint32_t key_element = 0; key_element < kQ38GdnHeadWidth;
         ++key_element) {
        const auto state_index = state_base +
                                 static_cast<std::uint64_t>(key_element) *
                                     kQ38GdnHeadWidth +
                                 element;
        const float key = bf16_load(qkv, key_base + key_element) * key_scale;
        const float updated = state[state_index] + key * delta;
        state[state_index] = updated;
        const float query_element =
            bf16_load(qkv, query_base + key_element) * query_scale;
        core += updated * query_element;
    }
    bf16_store(output, value_head * kQ38GdnHeadWidth + element, core);
}

__global__ void gdn_recurrent_prefill_kernel(
    const std::uint16_t* qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* state, std::uint16_t* output,
    std::uint32_t tokens) {
    const auto value_head = blockIdx.x;
    const auto element = threadIdx.x;
    const auto key_head = value_head / (kQ38GdnValueHeads / kQ38GdnKeyHeads);
    const auto query_local = key_head * kQ38GdnHeadWidth;
    const auto key_local = kQ38GdnKeyHeads * kQ38GdnHeadWidth + query_local;
    const auto value_local = 2 * kQ38GdnKeyHeads * kQ38GdnHeadWidth +
                             value_head * kQ38GdnHeadWidth;
    const auto state_base = static_cast<std::uint64_t>(value_head) *
                            kQ38GdnHeadWidth * kQ38GdnHeadWidth;
    __shared__ float query_scale;
    __shared__ float key_scale;
    __shared__ float decay;
    __shared__ float beta;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        const auto token_base =
            static_cast<std::uint64_t>(token) * kQ38GdnQkvWidth;
        const float query_raw = bf16_load(qkv, token_base + query_local + element);
        const float key_raw = bf16_load(qkv, token_base + key_local + element);
        const float query_norm = block_sum_128(query_raw * query_raw);
        const float key_norm = block_sum_128(key_raw * key_raw);
        if (threadIdx.x == 0) {
            query_scale = rsqrtf(query_norm + 1.0e-6f) *
                          rsqrtf(static_cast<float>(kQ38GdnHeadWidth));
            key_scale = rsqrtf(key_norm + 1.0e-6f);
            const float a = bf16_load(
                projected_a,
                static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
                    value_head);
            const float b = bf16_load(
                projected_b,
                static_cast<std::uint64_t>(token) * kQ38GdnValueHeads +
                    value_head);
            const float shifted_a = a + bf16_load(dt_bias, value_head);
            const float softplus = shifted_a > 20.0f
                                       ? shifted_a
                                       : log1pf(expf(shifted_a));
            decay = expf(-expf(bf16_load(a_log, value_head)) * softplus);
            beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();
        float memory_value = 0.0f;
        for (std::uint32_t key_element = 0;
             key_element < kQ38GdnHeadWidth; ++key_element) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                element;
            const float decayed = state[state_index] * decay;
            state[state_index] = decayed;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            memory_value += decayed * key;
        }
        const float value = bf16_load(qkv, token_base + value_local + element);
        const float delta = (value - memory_value) * beta;
        float core = 0.0f;
        for (std::uint32_t key_element = 0;
             key_element < kQ38GdnHeadWidth; ++key_element) {
            const auto state_index =
                state_base + static_cast<std::uint64_t>(key_element) *
                                 kQ38GdnHeadWidth +
                element;
            const float key =
                bf16_load(qkv, token_base + key_local + key_element) *
                key_scale;
            const float updated = state[state_index] + key * delta;
            state[state_index] = updated;
            const float query =
                bf16_load(qkv, token_base + query_local + key_element) *
                query_scale;
            core += updated * query;
        }
        bf16_store(output,
                   static_cast<std::uint64_t>(token) * kQ38GdnValueWidth +
                       value_head * kQ38GdnHeadWidth + element,
                   core);
        __syncthreads();
    }
}

__global__ void gdn_output_norm_kernel(
    const std::uint16_t* core, const std::uint16_t* gate,
    const std::uint16_t* weight, std::uint16_t* output, float epsilon) {
    const auto head = blockIdx.x;
    const auto token = blockIdx.y;
    const auto element = threadIdx.x;
    const auto index = static_cast<std::uint64_t>(token) * kQ38GdnValueWidth +
                       head * kQ38GdnHeadWidth + element;
    const float value = bf16_load(core, index);
    float squares = block_sum_128(value * value);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0)
        inverse_rms = rsqrtf(squares / kQ38GdnHeadWidth + epsilon);
    __syncthreads();
    const float gate_value = bf16_load(gate, index);
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    bf16_store(output, index,
               value * inverse_rms * bf16_load(weight, element) * sigmoid);
}

}  // namespace

struct CudaGdnStateBank::Impl {
    int device;
    std::uint32_t layers;
    std::byte* banks[2]{nullptr, nullptr};
    std::uint32_t committed = 0;
    bool active = false;
    std::uint64_t active_epoch = 0;
    std::uint64_t conv_layer_bytes =
        static_cast<std::uint64_t>(kQ38GdnQkvWidth) * kQ38GdnConvWidth * 2;
    std::uint64_t recurrent_layer_bytes =
        static_cast<std::uint64_t>(kQ38GdnValueHeads) * kQ38GdnHeadWidth *
        kQ38GdnHeadWidth * sizeof(float);
    std::uint64_t bank_bytes = 0;

    Impl(int value_device, std::uint32_t value_layers)
        : device(value_device), layers(value_layers) {
        if (layers == 0) throw std::invalid_argument("GDN state bank is empty");
        if (layers > std::numeric_limits<std::uint64_t>::max() /
                         (conv_layer_bytes + recurrent_layer_bytes))
            throw std::overflow_error("GDN state bank size overflows");
        bank_bytes = layers * (conv_layer_bytes + recurrent_layer_bytes);
        if (bank_bytes > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("GDN state bank exceeds address space");
        select_device(device);
        try {
            check(cudaMalloc(reinterpret_cast<void**>(&banks[0]), bank_bytes),
                  "cudaMalloc(GDN bank 0)");
            check(cudaMalloc(reinterpret_cast<void**>(&banks[1]), bank_bytes),
                  "cudaMalloc(GDN bank 1)");
            check(cudaMemset(banks[0], 0, bank_bytes), "cudaMemset(GDN bank 0)");
            check(cudaMemset(banks[1], 0, bank_bytes), "cudaMemset(GDN bank 1)");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }
    void release() noexcept {
        (void)cudaSetDevice(device);
        if (banks[0]) (void)cudaFree(banks[0]);
        if (banks[1]) (void)cudaFree(banks[1]);
        banks[0] = banks[1] = nullptr;
    }

    std::byte* layer_base(std::uint32_t bank, std::uint32_t layer) const {
        return banks[bank] +
               static_cast<std::uint64_t>(layer) *
                   (conv_layer_bytes + recurrent_layer_bytes);
    }
};

CudaGdnStateBank::CudaGdnStateBank(int device, std::uint32_t layers)
    : impl_(std::make_unique<Impl>(device, layers)) {}
CudaGdnStateBank::~CudaGdnStateBank() = default;
CudaGdnStateBank::CudaGdnStateBank(CudaGdnStateBank&&) noexcept = default;
CudaGdnStateBank& CudaGdnStateBank::operator=(CudaGdnStateBank&&) noexcept =
    default;

void CudaGdnStateBank::begin(std::uint64_t epoch, void* stream) {
    if (!stream || epoch == 0 || impl_->active)
        throw std::logic_error("invalid GDN transaction begin");
    select_device(impl_->device);
    const auto working = 1u - impl_->committed;
    check(cudaMemcpyAsync(impl_->banks[working], impl_->banks[impl_->committed],
                          impl_->bank_bytes, cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(GDN transaction clone)");
    impl_->active = true;
    impl_->active_epoch = epoch;
}

void CudaGdnStateBank::restore(void* stream) {
    if (!stream || !impl_->active)
        throw std::logic_error("invalid GDN transaction restore");
    select_device(impl_->device);
    check(cudaMemcpyAsync(impl_->banks[1u - impl_->committed],
                          impl_->banks[impl_->committed], impl_->bank_bytes,
                          cudaMemcpyDeviceToDevice,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemcpyAsync(GDN transaction restore)");
}

CudaGdnLayerStateView CudaGdnStateBank::working(
    std::uint32_t local_layer) const {
    if (!impl_->active || local_layer >= impl_->layers)
        throw std::logic_error("invalid GDN working state request");
    auto* base = impl_->layer_base(1u - impl_->committed, local_layer);
    return CudaGdnLayerStateView{
        reinterpret_cast<std::uint16_t*>(base),
        reinterpret_cast<float*>(base + impl_->conv_layer_bytes)};
}

void CudaGdnStateBank::commit(std::uint64_t epoch) {
    if (!impl_->active || impl_->active_epoch != epoch)
        throw std::logic_error("GDN commit epoch mismatch");
    impl_->committed = 1u - impl_->committed;
    impl_->active = false;
    impl_->active_epoch = 0;
}

void CudaGdnStateBank::rollback(std::uint64_t epoch) {
    if (!impl_->active || impl_->active_epoch != epoch)
        throw std::logic_error("GDN rollback epoch mismatch");
    impl_->active = false;
    impl_->active_epoch = 0;
}

void CudaGdnStateBank::reset(void* stream) {
    if (!stream || impl_->active)
        throw std::logic_error("invalid GDN reset");
    select_device(impl_->device);
    check(cudaMemsetAsync(impl_->banks[0], 0, impl_->bank_bytes,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemsetAsync(GDN bank 0)");
    check(cudaMemsetAsync(impl_->banks[1], 0, impl_->bank_bytes,
                          reinterpret_cast<cudaStream_t>(stream)),
          "cudaMemsetAsync(GDN bank 1)");
    impl_->committed = 0;
}

std::uint32_t CudaGdnStateBank::layers() const { return impl_->layers; }
std::uint64_t CudaGdnStateBank::bytes_per_bank() const {
    return impl_->bank_bytes;
}

void cuda_gdn_conv_decode_bf16(
    const std::uint16_t* projected_qkv, const std::uint16_t* conv_weight,
    std::uint16_t* conv_state, std::uint16_t* activated_qkv, void* stream,
    int device) {
    if (!projected_qkv || !conv_weight || !conv_state || !activated_qkv ||
        !stream)
        throw std::invalid_argument("invalid GDN convolution buffers");
    select_device(device);
    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks = (kQ38GdnQkvWidth + threads - 1) / threads;
    gdn_conv_decode_kernel<<<blocks, threads, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_qkv, conv_weight, conv_state, activated_qkv);
    check(cudaPeekAtLastError(), "gdn_conv_decode_kernel");
}

void cuda_gdn_recurrent_decode_bf16(
    const std::uint16_t* activated_qkv, const std::uint16_t* projected_b,
    const std::uint16_t* projected_a, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias, float* recurrent_state,
    std::uint16_t* core_output, void* stream, int device) {
    if (!activated_qkv || !projected_b || !projected_a || !a_log ||
        !dt_bias || !recurrent_state || !core_output || !stream)
        throw std::invalid_argument("invalid GDN recurrent buffers");
    select_device(device);
    gdn_recurrent_decode_kernel<<<kQ38GdnValueHeads, kQ38GdnHeadWidth, 0,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
        activated_qkv, projected_b, projected_a, a_log, dt_bias,
        recurrent_state, core_output);
    check(cudaPeekAtLastError(), "gdn_recurrent_decode_kernel");
}

void cuda_gdn_output_norm_bf16(
    const std::uint16_t* core_output, const std::uint16_t* gate_z,
    const std::uint16_t* norm_weight, std::uint16_t* output, float epsilon,
    void* stream, int device) {
    if (!core_output || !gate_z || !norm_weight || !output || !stream ||
        !(epsilon > 0.0f))
        throw std::invalid_argument("invalid GDN output norm buffers");
    select_device(device);
    gdn_output_norm_kernel<<<dim3(kQ38GdnValueHeads, 1), kQ38GdnHeadWidth, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        core_output, gate_z, norm_weight, output, epsilon);
    check(cudaPeekAtLastError(), "gdn_output_norm_kernel");
}

void cuda_gdn_conv_prefill_bf16(
    const std::uint16_t* projected_qkv, const std::uint16_t* conv_weight,
    std::uint16_t* conv_state, std::uint16_t* activated_qkv,
    std::uint32_t tokens, void* stream, int device) {
    if (!projected_qkv || !conv_weight || !conv_state || !activated_qkv ||
        !stream || tokens == 0)
        throw std::invalid_argument("invalid GDN prefill convolution buffers");
    select_device(device);
    gdn_conv_prefill_kernel<<<(kQ38GdnQkvWidth + 255) / 256, 256, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
        projected_qkv, conv_weight, conv_state, activated_qkv, tokens);
    check(cudaPeekAtLastError(), "GDN prefill convolution");
}

void cuda_gdn_recurrent_prefill_bf16(
    const std::uint16_t* activated_qkv,
    const std::uint16_t* projected_b,
    const std::uint16_t* projected_a,
    const std::uint16_t* a_log,
    const std::uint16_t* dt_bias,
    float* recurrent_state,
    std::uint16_t* core_output,
    std::uint32_t tokens,
    void* stream,
    int device) {
    if (!activated_qkv || !projected_b || !projected_a || !a_log ||
        !dt_bias || !recurrent_state || !core_output || !stream || tokens == 0)
        throw std::invalid_argument("invalid GDN prefill recurrent buffers");
    select_device(device);
    gdn_recurrent_prefill_kernel<<<kQ38GdnValueHeads, kQ38GdnHeadWidth, 0,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
        activated_qkv, projected_b, projected_a, a_log, dt_bias,
        recurrent_state, core_output, tokens);
    check(cudaPeekAtLastError(), "GDN prefill recurrent");
}

void cuda_gdn_output_norm_prefill_bf16(
    const std::uint16_t* core_output,
    const std::uint16_t* gate_z,
    const std::uint16_t* norm_weight,
    std::uint16_t* output,
    std::uint32_t tokens,
    float epsilon,
    void* stream,
    int device) {
    if (!core_output || !gate_z || !norm_weight || !output || !stream ||
        tokens == 0 || !(epsilon > 0.0f))
        throw std::invalid_argument("invalid GDN prefill norm buffers");
    select_device(device);
    gdn_output_norm_kernel<<<dim3(kQ38GdnValueHeads, tokens),
                             kQ38GdnHeadWidth, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
        core_output, gate_z, norm_weight, output, epsilon);
    check(cudaPeekAtLastError(), "GDN prefill output norm");
}

bool cuda_q38_gdn_compiled() { return true; }

}  // namespace q38
