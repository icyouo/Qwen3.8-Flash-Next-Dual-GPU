#include "q38/cuda_kernels.h"
#include "q38/cuda_transport.h"
#include "q38/cuda_gdn.h"
#include "q38/cuda_hyper.h"
#include "q38/cuda_ple.h"
#include "q38/cuda_qsa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
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

void test_boundary_transport_checksum(cudaStream_t stream) {
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
    auto* state_rec_decode = allocate<float>(recurrent_values, true);
    const auto qkv_values =
        static_cast<std::size_t>(tokens) * q38::kQ38GdnQkvWidth;
    const auto core_values =
        static_cast<std::size_t>(tokens) * q38::kQ38GdnValueWidth;
    auto* activated_batch = allocate<std::uint16_t>(qkv_values);
    auto* activated_decode = allocate<std::uint16_t>(qkv_values);
    auto* core_batch = allocate<std::uint16_t>(core_values);
    auto* core_decode = allocate<std::uint16_t>(core_values);
    auto* output_batch = allocate<std::uint16_t>(core_values);
    auto* output_decode = allocate<std::uint16_t>(core_values);

    q38::cuda_gdn_conv_prefill_bf16(
        d_projected, d_conv, state_conv_batch, activated_batch, tokens,
        reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_recurrent_prefill_bf16(
        activated_batch, d_b, d_a, d_a_log, d_dt, state_rec_batch,
        core_batch, tokens, reinterpret_cast<void*>(stream), 0);
    q38::cuda_gdn_output_norm_prefill_bf16(
        core_batch, d_gate, d_norm, output_batch, tokens, 1.0e-6f,
        reinterpret_cast<void*>(stream), 0);
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
    std::vector<std::uint16_t> host_decode(core_values);
    check(cudaMemcpyAsync(host_batch.data(), output_batch,
                          core_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(GDN batch)");
    check(cudaMemcpyAsync(host_decode.data(), output_decode,
                          core_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(GDN decode)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(GDN test)");
    require_close(host_batch, host_decode, 0.01f, "GDN prefill parity");

    (void)cudaFree(output_decode);
    (void)cudaFree(output_batch);
    (void)cudaFree(core_decode);
    (void)cudaFree(core_batch);
    (void)cudaFree(activated_decode);
    (void)cudaFree(activated_batch);
    (void)cudaFree(state_rec_decode);
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
    std::vector<std::uint16_t> host_decode(query_values);
    check(cudaMemcpyAsync(host_batch.data(), output_batch,
                          query_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA batch)");
    check(cudaMemcpyAsync(host_decode.data(), output_decode,
                          query_values * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpy(QSA decode)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(QSA test)");
    require_close(host_batch, host_decode, 0.01f, "QSA prefill parity");

    (void)cudaFree(decode_attention);
    (void)cudaFree(decode_selected);
    (void)cudaFree(decode_block_scores);
    (void)cudaFree(batch_attention);
    (void)cudaFree(batch_selected);
    (void)cudaFree(batch_block_scores);
    (void)cudaFree(output_decode);
    (void)cudaFree(index_query_decode);
    (void)cudaFree(gate_decode);
    (void)cudaFree(query_decode);
    (void)cudaFree(output_batch);
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
        (q38::kQ38QsaBlockBudget + 2) * q38::kQ38QsaBlockTokens - 1;
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
    auto* decode_attention = allocate<float>(
        q38::kQ38QsaHeads * q38::kQ38QsaMaximumSelected);
    auto* batch_output = allocate<std::uint16_t>(query_values.size());
    auto* decode_output = allocate<std::uint16_t>(query_values.size());

    q38::cuda_qsa_attention_prefill_bf16(
        query, index_query, cache, position, 1, batch_scores, score_stride,
        batch_selected, batch_attention, batch_output,
        reinterpret_cast<void*>(stream), 0);
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
    check(cudaStreamSynchronize(stream),
          "cudaStreamSynchronize(QSA radix test)");
    if (host_batch_selected != host_decode_selected)
        throw std::runtime_error("QSA radix selected blocks differ");
    require_close(host_batch_output, host_decode_output, 0.01f,
                  "QSA radix attention parity");

    (void)cudaFree(decode_output);
    (void)cudaFree(batch_output);
    (void)cudaFree(decode_attention);
    (void)cudaFree(batch_attention);
    (void)cudaFree(decode_selected);
    (void)cudaFree(batch_selected);
    (void)cudaFree(decode_scores);
    (void)cudaFree(batch_scores);
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
        std::vector<std::int8_t> weights(2 * 128, 1);
        for (std::size_t index = 128; index < weights.size(); ++index)
            weights[index] = index & 1 ? -1 : 1;
        std::vector<std::uint16_t> scales{bf16(1.0f), bf16(1.0f)};
        std::vector<std::uint16_t> input(128, bf16(1.0f));
        auto* device_weights = upload(weights);
        auto* device_scales = upload(scales);
        auto* device_input = upload(input);
        std::uint16_t* device_output = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&device_output), 4),
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

        test_boundary_transport_checksum(stream);
        test_gdn_prefill(stream);
        test_ple_prefill(stream);
        test_qsa_prefill(stream);
        test_qsa_radix_selection(stream);

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
