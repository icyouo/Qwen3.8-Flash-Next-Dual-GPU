#include "q38/cuda_gdn.h"
#include "q38/cuda_ple.h"
#include "q38/cuda_hyper.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() { return static_cast<T*>(data); }
};
std::uint16_t bf16(float f) {
    const auto v = __float2bfloat16_rn(f);
    std::uint16_t bits; std::memcpy(&bits, &v, 2); return bits;
}
void fill_bf16(Buffer& buffer, std::size_t words) {
    std::vector<std::uint16_t> values(words);
    for (std::size_t i = 0; i < words; ++i)
        values[i] = bf16(float(int(i % 29) - 14) / 128.f);
    check(cudaMemcpy(buffer.data, values.data(), words*2, cudaMemcpyHostToDevice));
}
std::vector<std::uint8_t> read(void* p, std::size_t bytes, cudaStream_t stream) {
    std::vector<std::uint8_t> result(bytes);
    check(cudaMemcpyAsync(result.data(), p, bytes, cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream)); return result;
}
template<class Run>
void verify(const char* name, std::size_t state_bytes, std::size_t output_words,
            bool fp32_state, cudaStream_t stream, Run run) {
    Buffer initial(state_bytes), state(state_bytes), reference(state_bytes);
    if (fp32_state) {
        std::vector<float> values(state_bytes / 4);
        for (std::size_t i=0; i<values.size(); ++i) values[i]=float(int(i%17)-8)/1024.f;
        check(cudaMemcpy(initial.data, values.data(), state_bytes, cudaMemcpyHostToDevice));
    } else fill_bf16(initial, state_bytes/2);
    constexpr unsigned maximum=7;
    const auto stride=state_bytes+64;
    Buffer snapshots(maximum*stride+64), output(maximum*output_words*2), oracle(maximum*output_words*2);
    check(cudaDeviceSynchronize());
    for (unsigned tokens : {1u,2u,3u,5u,7u}) {
        for (unsigned saved : {1u,tokens}) {
            check(cudaMemsetAsync(snapshots.data, 0xa5, maximum*stride+64, stream));
            check(cudaMemcpyAsync(state.data,initial.data,state_bytes,cudaMemcpyDeviceToDevice,stream));
            run(state.data,output.as<std::uint16_t>(),tokens,{snapshots.data,stride,saved});
            const auto captured=read(snapshots.data,maximum*stride+64,stream);
            const auto with_capture=read(output.data,tokens*output_words*2,stream);
            const auto final_state=read(state.data,state_bytes,stream);
            check(cudaMemcpyAsync(reference.data,initial.data,state_bytes,cudaMemcpyDeviceToDevice,stream));
            run(reference.data,oracle.as<std::uint16_t>(),tokens,{});
            if (with_capture!=read(oracle.data,tokens*output_words*2,stream) ||
                final_state!=read(reference.data,state_bytes,stream))
                throw std::runtime_error(std::string(name)+" capture changed arithmetic");
            for (unsigned prefix=1; prefix<=saved; ++prefix) {
                check(cudaMemcpyAsync(reference.data,initial.data,state_bytes,cudaMemcpyDeviceToDevice,stream));
                run(reference.data,oracle.as<std::uint16_t>(),prefix,{});
                const auto expected=read(reference.data,state_bytes,stream);
                if (!std::equal(expected.begin(),expected.end(),captured.begin()+(prefix-1)*stride))
                    throw std::runtime_error(std::string(name)+" prefix state differs");
                // Restoring the prefix must be a byte-exact state operation.
                check(cudaMemcpyAsync(state.data,snapshots.as<std::uint8_t>()+(prefix-1)*stride,
                                      state_bytes,cudaMemcpyDeviceToDevice,stream));
                if (read(state.data,state_bytes,stream)!=expected)
                    throw std::runtime_error("prefix restore differs");
            }
            for (std::size_t i=0; i<captured.size(); ++i)
                if ((i/stride>=saved || i%stride>=state_bytes) && captured[i]!=0xa5)
                    throw std::runtime_error("snapshot wrote outside requested state rows");
        }
    }
    for (auto bad : {q38::CudaPrefixSnapshotView{snapshots.data,state_bytes-1,1},
                     q38::CudaPrefixSnapshotView{snapshots.data,stride,8},
                     q38::CudaPrefixSnapshotView{nullptr,stride,1},
                     q38::CudaPrefixSnapshotView{snapshots.data,UINT64_MAX,2}}) {
        bool rejected=false;
        try { run(state.data,output.as<std::uint16_t>(),7,bad); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) throw std::runtime_error("invalid snapshot view accepted");
    }
    std::cout << name << ": exact prefix/output/state and guards passed\n";
}
}
int main() {
 try {
    check(cudaSetDevice(0)); cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    constexpr unsigned m=7;
    Buffer qkv(m*q38::kQ38GdnQkvWidth*2), conv(q38::kQ38GdnQkvWidth*q38::kQ38GdnConvWidth*2);
    Buffer a(m*q38::kQ38GdnValueHeads*2), b(m*q38::kQ38GdnValueHeads*2);
    Buffer alog(q38::kQ38GdnValueHeads*2), dt(q38::kQ38GdnValueHeads*2);
    Buffer parameters(q38::q38_gdn_prefill_parameter_floats(m)*4);
    fill_bf16(qkv,m*q38::kQ38GdnQkvWidth); fill_bf16(conv,q38::kQ38GdnQkvWidth*q38::kQ38GdnConvWidth);
    fill_bf16(a,m*q38::kQ38GdnValueHeads); fill_bf16(b,m*q38::kQ38GdnValueHeads);
    fill_bf16(alog,q38::kQ38GdnValueHeads); fill_bf16(dt,q38::kQ38GdnValueHeads);
    verify("GDN conv",q38::kQ38GdnQkvWidth*q38::kQ38GdnConvWidth*2,
        q38::kQ38GdnQkvWidth,false,stream,[&](void* state,auto* out,unsigned tokens,q38::CudaPrefixSnapshotView snapshots) {
        q38::cuda_gdn_conv_prefill_bf16(qkv.as<std::uint16_t>(),conv.as<std::uint16_t>(),
            static_cast<std::uint16_t*>(state),out,tokens,stream,0,snapshots);
    });
    verify("GDN recurrent",q38::kQ38GdnValueHeads*q38::kQ38GdnHeadWidth*q38::kQ38GdnHeadWidth*4,
        q38::kQ38GdnValueWidth,true,stream,[&](void* state,auto* out,unsigned tokens,q38::CudaPrefixSnapshotView snapshots) {
        q38::cuda_gdn_recurrent_prefill_precomputed_bf16(qkv.as<std::uint16_t>(),b.as<std::uint16_t>(),
            a.as<std::uint16_t>(),alog.as<std::uint16_t>(),dt.as<std::uint16_t>(),static_cast<float*>(state),
            parameters.as<float>(),out,tokens,stream,0,snapshots);
    });
    Buffer gated(m*q38::kQ38HyperWidth*2), normalized(m*q38::kQ38HyperWidth*2);
    Buffer weight(q38::kQ38HyperWidth*4*2), residual(m*q38::kQ38HyperWidth*2);
    fill_bf16(gated,m*q38::kQ38HyperWidth); fill_bf16(normalized,m*q38::kQ38HyperWidth);
    fill_bf16(weight,q38::kQ38HyperWidth*4); fill_bf16(residual,m*q38::kQ38HyperWidth);
    for (bool fused : {false,true}) {
        verify(fused?"PLE fused":"PLE conv",q38::kQ38HyperWidth*q38::kQ38PleConvState*2,
            q38::kQ38HyperWidth,false,stream,[&](void* state,auto* out,unsigned tokens,q38::CudaPrefixSnapshotView snapshots) {
            if (fused) q38::cuda_ple_conv_residual_bf16(gated.as<std::uint16_t>(),normalized.as<std::uint16_t>(),
                weight.as<std::uint16_t>(),static_cast<std::uint16_t*>(state),residual.as<std::uint16_t>(),
                out,tokens,stream,0,snapshots);
            else q38::cuda_ple_conv_prefill_bf16(gated.as<std::uint16_t>(),normalized.as<std::uint16_t>(),
                weight.as<std::uint16_t>(),static_cast<std::uint16_t*>(state),out,tokens,stream,0,snapshots);
        });
    }
    check(cudaStreamDestroy(stream));
    return 0;
 } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
