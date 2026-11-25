#include "q38/cuda_te.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace q38 {
namespace {
thread_local TeCollectiveRecorder* collective_recorder = nullptr;
void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
struct DeviceGuard {
    int device = 0;
    DeviceGuard() { check(cudaGetDevice(&device), "get TE device"); }
    ~DeviceGuard() { cudaSetDevice(device); }
};

__global__ void add_f32_kernel(float* local, const float* peer, std::size_t n) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) local[i] += peer[i];
}
__global__ void add_bf16_kernel(std::uint16_t* local, const std::uint16_t* peer,
                               std::size_t n) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        auto* out = reinterpret_cast<__nv_bfloat16*>(local);
        const auto* in = reinterpret_cast<const __nv_bfloat16*>(peer);
        out[i] = __float2bfloat16_rn(__bfloat162float(out[i]) + __bfloat162float(in[i]));
    }
}
__global__ void round_partial_bf16_kernel(const float* input,
    std::uint16_t* output, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(input[i]));
}

// Per-group high-water buffers. No cudaMalloc/cudaHostAlloc in steady decode.
struct Buffers {
    void* host[2] = {};
    void* output = nullptr;
    void* scratch[2] = {};
    std::size_t host_capacity = 0;
    std::size_t scratch_capacity = 0;
    cudaEvent_t ready[2] = {}, read_done[2] = {};
    cudaEvent_t output_done[2] = {};
    bool peer_pending = false;
    void drain_peer() {
        if (!peer_pending) return;
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r), "select P2P drain device");
            check(cudaEventSynchronize(output_done[r]), "drain P2P output");
        }
        peer_pending = false;
    }
    ~Buffers() {
        int previous = 0;
        cudaGetDevice(&previous);
        for (int r = 0; r < 2; ++r) {
            if (host[r]) cudaFreeHost(host[r]);
            cudaSetDevice(r);
            if (scratch[r]) cudaFree(scratch[r]);
            if (ready[r]) cudaEventDestroy(ready[r]);
            if (read_done[r]) cudaEventDestroy(read_done[r]);
            if (output_done[r]) cudaEventDestroy(output_done[r]);
        }
        if (output) cudaFreeHost(output);
        cudaSetDevice(previous);
    }
    void ensure_peer(std::size_t bytes) {
        if (bytes > scratch_capacity) drain_peer();
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r), "select P2P scratch device");
            if (!ready[r]) check(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming), "create P2P ready event");
            if (!read_done[r]) check(cudaEventCreateWithFlags(&read_done[r], cudaEventDisableTiming), "create P2P read event");
            if (!output_done[r]) check(cudaEventCreateWithFlags(&output_done[r], cudaEventDisableTiming), "create P2P output event");
            if (bytes > scratch_capacity) {
                if (scratch[r]) check(cudaFree(scratch[r]), "free P2P scratch");
                scratch[r] = nullptr;
                check(cudaMalloc(&scratch[r], bytes), "allocate P2P scratch");
            }
        }
        if (bytes > scratch_capacity) scratch_capacity = bytes;
    }
    void ensure(std::size_t bytes, bool reduce) {
        if (bytes > host_capacity) {
            for (auto& p : host) {
                if (p) check(cudaFreeHost(p), "free TE pinned input");
                p = nullptr;
                check(cudaHostAlloc(&p, bytes, cudaHostAllocPortable), "allocate TE pinned input");
            }
            if (output) check(cudaFreeHost(output), "free TE pinned output");
            output = nullptr;
            check(cudaHostAlloc(&output, bytes * 2, cudaHostAllocPortable), "allocate TE pinned output");
            host_capacity = bytes;
        }
        if (reduce && bytes > scratch_capacity) {
            for (int r = 0; r < 2; ++r) {
                check(cudaSetDevice(r), "select TE scratch device");
                if (scratch[r]) check(cudaFree(scratch[r]), "free TE scratch");
                scratch[r] = nullptr;
                check(cudaMalloc(&scratch[r], bytes), "allocate TE scratch");
            }
            scratch_capacity = bytes;
        }
    }
};
enum class Op { kGatherBf16, kSumBf16, kSumF32, kGatherF32, kPairedGraph, kSumF32ToBf16 };
std::atomic<bool> bridge_p2p{false};
std::atomic<bool> bridge_stream_ordered{false};

template <typename T, bool Gather>
__global__ void peer_collect_kernel(const T* rank0, const T* rank1, T* output,
                                    std::size_t elems, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        if constexpr (Gather) {
            const auto row = i / (2 * elems);
            const auto column = i % (2 * elems);
            output[i] = column < elems ? rank0[row * elems + column]
                                     : rank1[row * elems + column - elems];
        } else {
            output[i] = rank0[i] + rank1[i];
        }
    }
}
template <>
__global__ void peer_collect_kernel<std::uint16_t, false>(
    const std::uint16_t* rank0, const std::uint16_t* rank1,
    std::uint16_t* output, std::size_t, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const float a = __bfloat162float(__ushort_as_bfloat16(rank0[i]));
        const float b = __bfloat162float(__ushort_as_bfloat16(rank1[i]));
        output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(a + b));
    }
}

constexpr unsigned kBf16ExponentRawPacket = 1u << 8;
constexpr unsigned kBf16ExponentZero = 15u;

__device__ unsigned bf16_exponent_code(unsigned word, unsigned base) {
    return (word & 0x7fffu) == 0 ? kBf16ExponentZero : ((word >> 7u) & 0xffu) - base;
}

__global__ void pack_bulk_bf16(const std::uint16_t* input, std::uint8_t* packets,
                              std::uint16_t* raw_tail, std::size_t blocks) {
    constexpr unsigned warps_per_cta = 8;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5u;
    auto* headers = reinterpret_cast<std::uint16_t*>(
        packets + blocks * TeBf16BulkPackStorage::kPayloadBytes);
    for (auto block = static_cast<std::size_t>(blockIdx.x) * warps_per_cta + warp;
         block < blocks; block += static_cast<std::size_t>(gridDim.x) * warps_per_cta) {
        const auto* words = input + block * TeBf16BulkPackStorage::kBlockWords;
        const unsigned first = words[lane];
        const unsigned second = words[lane + 32u];
        unsigned minimum = 256, maximum = 0;
        if (first & 0x7fffu) {
            minimum = (first >> 7u) & 0xffu;
            maximum = minimum;
        }
        if (second & 0x7fffu) {
            const unsigned exponent = (second >> 7u) & 0xffu;
            minimum = min(minimum, exponent);
            maximum = max(maximum, exponent);
        }
        for (unsigned mask = 16; mask != 0; mask >>= 1) {
            minimum = min(minimum, __shfl_xor_sync(0xffffffffu, minimum, mask));
            maximum = max(maximum, __shfl_xor_sync(0xffffffffu, maximum, mask));
        }
        if (minimum == 256) minimum = 0;
        auto* packet = packets + block * TeBf16BulkPackStorage::kPayloadBytes;
        if (maximum - minimum < kBf16ExponentZero) {
            if (lane == 0) headers[block] = static_cast<std::uint16_t>(minimum);
            packet[lane] = (first & 0x7fu) | ((first >> 8u) & 0x80u);
            packet[lane + 32u] = (second & 0x7fu) | ((second >> 8u) & 0x80u);
            packet[64u + lane] = bf16_exponent_code(first, minimum) |
                (bf16_exponent_code(second, minimum) << 4u);
        } else {
            if (lane == 0) headers[block] = kBf16ExponentRawPacket;
            auto* raw_prefix = reinterpret_cast<std::uint16_t*>(packet);
            raw_prefix[lane] = static_cast<std::uint16_t>(first);
            if (lane < 16u) raw_prefix[lane + 32u] = static_cast<std::uint16_t>(second);
            else raw_tail[block * 16u + lane - 16u] = static_cast<std::uint16_t>(second);
        }
    }
}

__global__ void sum_packed_bulk_bf16(std::uint16_t* local,
    const std::uint8_t* peer_packets, const std::uint16_t* peer_raw_tail,
    std::size_t blocks, int rank) {
    using Pack = TeBf16BulkPackStorage;
    const auto* headers = reinterpret_cast<const std::uint16_t*>(
        peer_packets + blocks * Pack::kPayloadBytes);
    const auto count = blocks * Pack::kBlockWords;
    for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto block = i / Pack::kBlockWords;
        const auto word = i % Pack::kBlockWords;
        const auto header = headers[block];
        const auto* packet = peer_packets + block * Pack::kPayloadBytes;
        std::uint16_t peer;
        if (header == kBf16ExponentRawPacket) {
            peer = word < 48 ? reinterpret_cast<const std::uint16_t*>(packet)[word]
                : peer_raw_tail[block * 16u + word - 48u];
        } else {
            auto exponent_code = packet[64u + (word & 31u)];
            exponent_code = (exponent_code >> (word >= 32u ? 4u : 0u)) & 15u;
            const unsigned exponent = exponent_code == kBf16ExponentZero ? 0 : header + exponent_code;
            const unsigned sign_and_mantissa = packet[word];
            peer = ((sign_and_mantissa & 0x80u) << 8u) | (exponent << 7u) |
                (sign_and_mantissa & 0x7fu);
        }
        const auto first = rank == 0 ? local[i] : peer;
        const auto second = rank == 0 ? peer : local[i];
        const float a = __bfloat162float(__ushort_as_bfloat16(first));
        const float b = __bfloat162float(__ushort_as_bfloat16(second));
        local[i] = __bfloat16_as_ushort(__float2bfloat16_rn(a + b));
    }
}
// Preserve the scalar two-rank arithmetic, issuing aligned peer reads in
// full 16-byte lanes. Unaligned/tiny row layouts retain the scalar path.
template<bool F32, bool Gather>
__global__ void peer_collect_vector_kernel(const void* rank0,const void* rank1,
    void* output,std::size_t elems,std::size_t count) {
    constexpr unsigned width=F32?4:8;
    for(std::size_t v=blockIdx.x*blockDim.x+threadIdx.x;v<count/width;
        v+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if constexpr(Gather) {
            const auto per=elems/width,row=v/(2*per),col=v%(2*per);
            reinterpret_cast<uint4*>(output)[v]=col<per?
                reinterpret_cast<const uint4*>(rank0)[row*per+col]:
                reinterpret_cast<const uint4*>(rank1)[row*per+col-per];
        }else if constexpr(F32) {
            const float4 a=reinterpret_cast<const float4*>(rank0)[v];
            const float4 b=reinterpret_cast<const float4*>(rank1)[v];
            reinterpret_cast<float4*>(output)[v]=make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
        }else {
            const uint4 a=reinterpret_cast<const uint4*>(rank0)[v];
            const uint4 b=reinterpret_cast<const uint4*>(rank1)[v];uint4 result;
            const auto* ap=reinterpret_cast<const __nv_bfloat162*>(&a);
            const auto* bp=reinterpret_cast<const __nv_bfloat162*>(&b);
            auto* rp=reinterpret_cast<__nv_bfloat162*>(&result);
            #pragma unroll
            for(unsigned j=0;j<4;++j){const float2 x=__bfloat1622float2(ap[j]),y=__bfloat1622float2(bp[j]);
                rp[j]=__floats2bfloat162_rn(x.x+y.x,x.y+y.y);}
            reinterpret_cast<uint4*>(output)[v]=result;
        }
    }
}
bool peer_vector_enabled(const TeCollectiveCall& a,const TeCollectiveCall& b,
    void* scratch[2],std::size_t count,bool fp32,bool gather) {
    const auto* p=std::getenv("Q38_TP_PEER_VECTOR");
    const unsigned width=fp32?4:8;
    return p&&std::strcmp(p,"1")==0&&count%width==0&&(!gather||a.elements%width==0)&&
        !(reinterpret_cast<std::uintptr_t>(a.input)%16)&&!(reinterpret_cast<std::uintptr_t>(b.input)%16)&&
        !(reinterpret_cast<std::uintptr_t>(scratch[0])%16)&&!(reinterpret_cast<std::uintptr_t>(scratch[1])%16);
}
__device__ unsigned flag_acquire(const unsigned* flag) {
    unsigned value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(flag) : "memory");
    return value;
}
__device__ void flag_release(unsigned* flag, unsigned value) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(flag), "r"(value) : "memory");
}
__device__ void flag_wait(const unsigned* flag, unsigned generation) {
    const auto start = clock64();
    while (flag_acquire(flag) != generation) {
        // A failed peer launch is a device failure, never a valid partial
        // collective. Bound the wait and poison the CUDA context on failure.
        if (clock64() - start > 5000000000ull) asm volatile("trap;");
        __nanosleep(64);
    }
}
__global__ void graph_signal_kernel(unsigned* flags, const unsigned* peer, bool ready) {
    const auto prior = flag_acquire(flags);
    const auto generation = ready ? prior + 1 : prior;
    const auto index = ready ? 0 : 1;
    flag_release(flags + index, generation);
    // One signaling CTA avoids hundreds of simultaneous PCIe flag polls.
    // The following graph node begins only after this acquire completes.
    flag_wait(peer + index, generation);
}

// A single CTA can retain both generation barriers and scratch commit for
// bounded graph payloads, avoiding three graph nodes without a grid barrier.
// The same rank-ordered arithmetic and BF16 RNE boundary apply as above.
template<bool F32, bool Gather, bool Vector>
__global__ void graph_collect_fused_kernel(
    const void* rank0, const void* rank1, void* scratch, void* output,
    std::size_t elems, std::size_t count, unsigned* flags, const unsigned* peer) {
    __shared__ unsigned generation;
    if (threadIdx.x == 0) {
        generation = flag_acquire(flags) + 1u;
        flag_release(flags, generation);
        flag_wait(peer, generation);
    }
    __syncthreads();
    constexpr unsigned width = Vector ? (F32 ? 4u : 8u) : 1u;
    for (std::size_t i=threadIdx.x; i<count/width; i+=blockDim.x) {
        if constexpr (Vector && Gather) {
            const auto per=elems/width, row=i/(2*per), col=i%(2*per);
            reinterpret_cast<uint4*>(scratch)[i] = col<per ?
                reinterpret_cast<const uint4*>(rank0)[row*per+col] :
                reinterpret_cast<const uint4*>(rank1)[row*per+col-per];
        } else if constexpr (Vector && F32) {
            const auto a=reinterpret_cast<const float4*>(rank0)[i];
            const auto b=reinterpret_cast<const float4*>(rank1)[i];
            reinterpret_cast<float4*>(scratch)[i]=make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
        } else if constexpr (Vector) {
            const uint4 a=reinterpret_cast<const uint4*>(rank0)[i];
            const uint4 b=reinterpret_cast<const uint4*>(rank1)[i];
            uint4 result;
            const auto* ap=reinterpret_cast<const __nv_bfloat162*>(&a);
            const auto* bp=reinterpret_cast<const __nv_bfloat162*>(&b);
            auto* rp=reinterpret_cast<__nv_bfloat162*>(&result);
            #pragma unroll
            for(unsigned j=0;j<4;++j) {
                const float2 x=__bfloat1622float2(ap[j]), y=__bfloat1622float2(bp[j]);
                rp[j]=__floats2bfloat162_rn(x.x+y.x,x.y+y.y);
            }
            reinterpret_cast<uint4*>(scratch)[i]=result;
        } else if constexpr (Gather) {
            const auto row=i/(2*elems), col=i%(2*elems);
            static_cast<std::uint16_t*>(scratch)[i] = col<elems ?
                static_cast<const std::uint16_t*>(rank0)[row*elems+col] :
                static_cast<const std::uint16_t*>(rank1)[row*elems+col-elems];
        } else if constexpr (F32) {
            static_cast<float*>(scratch)[i]=static_cast<const float*>(rank0)[i]+static_cast<const float*>(rank1)[i];
        } else {
            const float a=__bfloat162float(__ushort_as_bfloat16(static_cast<const std::uint16_t*>(rank0)[i]));
            const float b=__bfloat162float(__ushort_as_bfloat16(static_cast<const std::uint16_t*>(rank1)[i]));
            static_cast<std::uint16_t*>(scratch)[i]=__bfloat16_as_ushort(__float2bfloat16_rn(a+b));
        }
    }
    // Every original input has been read before either rank may overwrite it.
    __syncthreads();
    if (threadIdx.x == 0) {
        flag_release(flags+1, generation);
        flag_wait(peer+1, generation);
    }
    __syncthreads();
    for (std::size_t i=threadIdx.x; i<count/width; i+=blockDim.x) {
        if constexpr (Vector)
            reinterpret_cast<uint4*>(output)[i]=reinterpret_cast<const uint4*>(scratch)[i];
        else if constexpr (F32)
            static_cast<float*>(output)[i]=static_cast<const float*>(scratch)[i];
        else
            static_cast<std::uint16_t*>(output)[i]=static_cast<const std::uint16_t*>(scratch)[i];
    }
}
void peer_collective(Buffers& buffers, Op op, void* ptr[2], void* full[2],
                     cudaStream_t streams[2], std::size_t elems,
                     std::size_t tokens, std::size_t bytes) {
    const bool gather = op == Op::kGatherBf16 || op == Op::kGatherF32;
    const bool fp32 = op == Op::kSumF32 || op == Op::kGatherF32;
    const auto count = elems * tokens * (gather ? 2 : 1);
    const auto output_bytes = bytes * (gather ? 2 : 1);
    buffers.ensure_peer(output_bytes);
    // A previous generation can still be using scratch. The streams can
    // change between calls, so stream-local order alone is insufficient.
    // Establish both waits before recording either new output_done event.
    if (buffers.peer_pending) {
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r), "select P2P scratch reuse device");
            check(cudaStreamWaitEvent(streams[r], buffers.output_done[r], 0),
                  "wait P2P previous scratch consumer");
        }
    }
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select P2P producer device");
        check(cudaEventRecord(buffers.ready[r], streams[r]), "record P2P producer");
    }
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>((count + 255) / 256, 256));
    // Large eager prefill reductions benefit from the copy engine rather
    // than scalar SM loads over PCIe. Keep short/graph collectives unchanged.
    const auto* bulk_setting = std::getenv("Q38_TP_BULK_COPY");
    const bool bulk_copy = !gather && bytes >= (256u << 10u) &&
        bulk_setting && std::strcmp(bulk_setting, "1") == 0;
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select P2P reader device");
        check(cudaStreamWaitEvent(streams[r], buffers.ready[1-r], 0), "wait P2P peer producer");
        if (bulk_copy)
            check(cudaMemcpyPeerAsync(buffers.scratch[r], r, ptr[1-r], 1-r,
                                      bytes, streams[r]), "copy P2P reduction input");
        else if (fp32 && gather)
            peer_collect_kernel<float, true><<<blocks, 256, 0, streams[r]>>>(
                static_cast<const float*>(ptr[0]), static_cast<const float*>(ptr[1]),
                static_cast<float*>(buffers.scratch[r]), elems, count);
        else if (fp32)
            peer_collect_kernel<float, false><<<blocks, 256, 0, streams[r]>>>(
                static_cast<const float*>(ptr[0]), static_cast<const float*>(ptr[1]),
                static_cast<float*>(buffers.scratch[r]), elems, count);
        else if (gather)
            peer_collect_kernel<std::uint16_t, true><<<blocks, 256, 0, streams[r]>>>(
                static_cast<const std::uint16_t*>(ptr[0]), static_cast<const std::uint16_t*>(ptr[1]),
                static_cast<std::uint16_t*>(buffers.scratch[r]), elems, count);
        else
            peer_collect_kernel<std::uint16_t, false><<<blocks, 256, 0, streams[r]>>>(
                static_cast<const std::uint16_t*>(ptr[0]), static_cast<const std::uint16_t*>(ptr[1]),
                static_cast<std::uint16_t*>(buffers.scratch[r]), elems, count);
        check(cudaPeekAtLastError(), "P2P collective kernel");
        check(cudaEventRecord(buffers.read_done[r], streams[r]), "record P2P input read completion");
    }
    // Neither input may be overwritten while the other GPU still reads it.
    // Scratch also permits full==primary for token-major AllGather.
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select P2P output device");
        check(cudaStreamWaitEvent(streams[r], buffers.read_done[1-r], 0), "wait P2P peer input lifetime");
        if (bulk_copy) {
            // Both copies precede either overwrite. Use rank0 + rank1 on
            // both devices, retaining the original FP32/BF16 arithmetic.
            const auto* rank0 = r == 0 ? ptr[r] : buffers.scratch[r];
            const auto* rank1 = r == 0 ? buffers.scratch[r] : ptr[r];
            if (fp32)
                peer_collect_kernel<float, false><<<blocks, 256, 0, streams[r]>>>(
                    static_cast<const float*>(rank0), static_cast<const float*>(rank1),
                    static_cast<float*>(ptr[r]), elems, count);
            else
                peer_collect_kernel<std::uint16_t, false><<<blocks, 256, 0, streams[r]>>>(
                    static_cast<const std::uint16_t*>(rank0), static_cast<const std::uint16_t*>(rank1),
                    static_cast<std::uint16_t*>(ptr[r]), elems, count);
            check(cudaPeekAtLastError(), "local P2P bulk reduction");
        } else
            check(cudaMemcpyAsync(gather ? full[r] : ptr[r], buffers.scratch[r],
                                  output_bytes, cudaMemcpyDeviceToDevice, streams[r]), "commit P2P collective output");
    }
    if (bridge_stream_ordered.load()) {
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r), "select P2P output event device");
            check(cudaEventRecord(buffers.output_done[r], streams[r]),
                  "record P2P output completion");
        }
        buffers.peer_pending = true;
    } else {
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r), "select P2P completion device");
            check(cudaStreamSynchronize(streams[r]), "wait P2P collective output");
        }
        buffers.peer_pending = false;
    }
}
__global__ void local_f32_sum_to_bf16(const float* rank0, const float* rank1,
    std::uint16_t* output, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(rank0[i] + rank1[i]));
}

// A transport encoding, not a new arithmetic format. Each 256-value block
// shares its maximum finite exponent. Values use sign + five exponent-delta
// bits + 18 mantissa bits (three bytes), omitting only five FP32 mantissa bits.
// Delta 31 encodes an exact signed zero or a full-FP32 peer-read escape.
// For ordinary finite values both endpoints sharing the encoded bits bound
// the exact FP32. Equal BF16 endpoint sums prove exact rounding by monotonic
// RNE; ambiguous intervals read the original FP32. Specials always escape.
__global__ void pack_f32_fp24(const float* input, unsigned char* output,
                            std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x)*256 + threadIdx.x;
    const unsigned bits = i < count ? __float_as_uint(input[i]) : 0u;
    const unsigned exponent = (bits >> 23) & 255u;
    unsigned maximum = exponent == 255u ? 0u : exponent;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (unsigned step=16;step;step>>=1)
        maximum = max(maximum,__shfl_down_sync(0xffffffffu,maximum,step));
    __shared__ unsigned maxima[8];
    if (!lane) maxima[warp] = maximum;
    __syncthreads();
    maximum = threadIdx.x < 8 ? maxima[lane] : 0u;
    if (!warp) {
        for (unsigned step=16;step;step>>=1)
            maximum = max(maximum,__shfl_down_sync(0xffffffffu,maximum,step));
        if (!lane) maxima[0] = maximum;
    }
    __syncthreads();
    maximum = maxima[0];
    if (!threadIdx.x) output[3*count+blockIdx.x] = maximum;
    if (i >= count) return;
    const unsigned sign = (bits >> 8) & 0x800000u;
    unsigned word;
    if (!(bits & 0x7fffffffu)) word = sign | (31u << 18);
    else if (exponent == 255u || maximum-exponent >= 31u)
        word = sign | (31u << 18) | 1u;
    else word = sign | ((maximum-exponent) << 18) |
        ((bits & 0x7fffffu) >> 5);
    output[3*i] = word;
    output[3*i+1] = word >> 8;
    output[3*i+2] = word >> 16;
}
__global__ void local_fp24_sum_to_bf16(const float* local,
    const unsigned char* packed_peer, const volatile float* exact_peer,
    std::uint16_t* output, std::size_t count, bool local_rank0) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const unsigned word = unsigned(packed_peer[3*i]) |
            (unsigned(packed_peer[3*i+1]) << 8) |
            (unsigned(packed_peer[3*i+2]) << 16);
        const unsigned delta = (word >> 18) & 31u;
        const unsigned sign = (word & 0x800000u) << 8;
        const unsigned mantissa = (word & 0x3ffffu) << 5;
        const float value = local[i];
        const unsigned local_bits = __float_as_uint(value);
        if (delta == 31u && !mantissa) {
            const float peer = __uint_as_float(sign);
            output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(
                local_rank0 ? __fadd_rn(value,peer) : __fadd_rn(peer,value)));
            continue;
        }
        if (delta < 31u && (local_bits & 0x7f800000u) != 0x7f800000u) {
            const unsigned maximum = packed_peer[3*count+i/256];
            const unsigned bits = sign | ((maximum-delta) << 23) | mantissa;
            const float a = __uint_as_float(bits);
            const float b = __uint_as_float(bits | 31u);
            const auto low = __bfloat16_as_ushort(__float2bfloat16_rn(
                local_rank0 ? __fadd_rn(value,a) : __fadd_rn(a,value)));
            const auto high = __bfloat16_as_ushort(__float2bfloat16_rn(
                local_rank0 ? __fadd_rn(value,b) : __fadd_rn(b,value)));
            if (low == high) { output[i] = low; continue; }
        }
        const float peer = exact_peer[i];
        output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(
            local_rank0 ? __fadd_rn(value,peer) : __fadd_rn(peer,value)));
    }
}

void peer_f32_sum_to_bf16(Buffers& buffers, void* ptr[2], void* output[2],
    cudaStream_t streams[2], std::size_t count) {
    const std::size_t split = (count + 1) / 2;
    const std::size_t offsets[2] = {0, split};
    const std::size_t sizes[2] = {split, count - split};
    const auto* fp24_setting = std::getenv("Q38_TP_RSAG_FP24");
    const bool fp24 = fp24_setting && std::strcmp(fp24_setting,"1") == 0;
    const auto packed_capacity = split*3 + (split+255)/256;
    buffers.ensure_peer(fp24 ? packed_capacity*2 : split*sizeof(float));
    if (buffers.peer_pending) {
        for (int rank = 0; rank < 2; ++rank) {
            check(cudaSetDevice(rank), "select RSAG scratch reuse device");
            check(cudaStreamWaitEvent(streams[rank], buffers.output_done[rank], 0),
                  "wait RSAG previous scratch consumer");
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        check(cudaSetDevice(rank), "select RSAG producer device");
        if (fp24 && sizes[1-rank]) {
            const auto blocks = static_cast<unsigned>((sizes[1-rank]+255)/256);
            pack_f32_fp24<<<blocks,256,0,streams[rank]>>>(
                static_cast<const float*>(ptr[rank]) + offsets[1-rank],
                static_cast<unsigned char*>(buffers.scratch[rank]) + packed_capacity,
                sizes[1-rank]);
            check(cudaPeekAtLastError(),"pack RSAG FP32 block-exponent24");
        }
        check(cudaEventRecord(buffers.ready[rank], streams[rank]), "record RSAG producer");
    }
    for (int rank = 0; rank < 2; ++rank) {
        check(cudaSetDevice(rank), "select RSAG reduction device");
        check(cudaStreamWaitEvent(streams[rank], buffers.ready[1-rank], 0),
              "wait RSAG peer producer");
        if (sizes[rank]) {
            const auto* source = fp24 ?
                static_cast<const void*>(static_cast<const unsigned char*>(buffers.scratch[1-rank]) + packed_capacity) :
                static_cast<const void*>(static_cast<const float*>(ptr[1-rank]) + offsets[rank]);
            check(cudaMemcpyPeerAsync(buffers.scratch[rank], rank,source,1-rank,
                (fp24 ? sizes[rank]*3 + (sizes[rank]+255)/256 : sizes[rank]*sizeof(float)), streams[rank]), "copy RSAG FP32 half");
            const auto* local = static_cast<const float*>(ptr[rank]) + offsets[rank];
            const auto* remote = static_cast<const float*>(buffers.scratch[rank]);
            const auto blocks = static_cast<unsigned>(std::min<std::size_t>((sizes[rank]+255)/256, 256));
            if (fp24)
                local_fp24_sum_to_bf16<<<blocks,256,0,streams[rank]>>>(local,
                    static_cast<const unsigned char*>(buffers.scratch[rank]),
                    static_cast<const float*>(ptr[1-rank]) + offsets[rank],
                    static_cast<std::uint16_t*>(output[rank]) + offsets[rank],sizes[rank],rank==0);
            else
                local_f32_sum_to_bf16<<<blocks, 256, 0, streams[rank]>>>(
                    rank == 0 ? local : remote, rank == 0 ? remote : local,
                    static_cast<std::uint16_t*>(output[rank]) + offsets[rank], sizes[rank]);
            check(cudaPeekAtLastError(), "reduce RSAG FP32 half and round BF16");
        }
        check(cudaEventRecord(buffers.read_done[rank], streams[rank]), "record RSAG reduced half");
    }
    for (int rank = 0; rank < 2; ++rank) {
        check(cudaSetDevice(rank), "select RSAG gather device");
        check(cudaStreamWaitEvent(streams[rank], buffers.read_done[1-rank], 0),
              "wait RSAG peer reduced half");
        if (sizes[1-rank])
            check(cudaMemcpyPeerAsync(static_cast<std::uint16_t*>(output[rank]) + offsets[1-rank], rank,
                static_cast<const std::uint16_t*>(output[1-rank]) + offsets[1-rank], 1-rank,
                sizes[1-rank] * sizeof(std::uint16_t), streams[rank]), "gather RSAG BF16 half");
        check(cudaEventRecord(buffers.output_done[rank], streams[rank]), "record RSAG gathered output");
    }
    // The other rank may still be reading our reduced output half. Neither
    // caller can overwrite that output or a source until both gathers finish.
    for (int rank = 0; rank < 2; ++rank) {
        check(cudaSetDevice(rank), "select RSAG output lifetime device");
        check(cudaStreamWaitEvent(streams[rank], buffers.output_done[1-rank], 0),
              "wait RSAG peer output lifetime");
    }
    if (bridge_stream_ordered.load()) {
        buffers.peer_pending = true;
    } else {
        for (int rank = 0; rank < 2; ++rank) {
            check(cudaSetDevice(rank), "select RSAG synchronous output device");
            check(cudaStreamSynchronize(streams[rank]), "finish RSAG synchronous output");
        }
        buffers.peer_pending = false;
    }
}

void collective(Buffers& buffers, Op op, void* ptr[2], void* full[2],
                cudaStream_t streams[2], std::size_t elems, std::size_t tokens,
                bool p2p = false) {
    DeviceGuard guard;
    if (op == Op::kSumF32ToBf16) {
        if (!p2p || tokens != 1 || elems > std::numeric_limits<std::size_t>::max()/8)
            throw std::invalid_argument("RSAG requires eager P2P and a bounded extent");
        for (int rank = 0; rank < 2; ++rank) {
            const auto source = reinterpret_cast<std::uintptr_t>(ptr[rank]);
            const auto output = reinterpret_cast<std::uintptr_t>(full[rank]);
            if ((source <= output && output-source < elems*sizeof(float)) ||
                (output < source && source-output < elems*sizeof(std::uint16_t)))
                throw std::invalid_argument("RSAG input and output must not overlap");
        }
        peer_f32_sum_to_bf16(buffers, ptr, full, streams, elems);
        return;
    }
    const bool gather = op == Op::kGatherBf16 || op == Op::kGatherF32;
    const std::size_t unit = op == Op::kSumF32 || op == Op::kGatherF32 ? 4 : 2;
    if (tokens == 0 || elems > std::numeric_limits<std::size_t>::max() / tokens / unit / 2)
        throw std::invalid_argument("TE collective extent overflow");
    const auto bytes = elems * tokens * unit;
    if (p2p) {
        peer_collective(buffers, op, ptr, full, streams, elems, tokens, bytes);
        return;
    }
    const bool host_sum = !gather && elems * tokens <= 16384;
    buffers.ensure(bytes, !gather && !host_sum);
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select TE input device");
        check(cudaMemcpyAsync(buffers.host[r], ptr[r], bytes, cudaMemcpyDeviceToHost,
                              streams[r]), "TE producer stream D2H");
    }
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select TE input wait device");
        check(cudaStreamSynchronize(streams[r]), "wait TE producer stream D2H");
    }
    if (gather) {
        const auto row_bytes = elems * unit;
        auto* output = static_cast<char*>(buffers.output);
        for (std::size_t t = 0; t < tokens; ++t)
            for (int r = 0; r < 2; ++r)
                std::memcpy(output + (t * 2 + r) * row_bytes,
                            static_cast<char*>(buffers.host[r]) + t * row_bytes, row_bytes);
    }
    if (host_sum) {
        if (op == Op::kSumBf16) {
            const auto* a = static_cast<const std::uint16_t*>(buffers.host[0]);
            const auto* b = static_cast<const std::uint16_t*>(buffers.host[1]);
            auto* out = static_cast<std::uint16_t*>(buffers.output);
            for (std::size_t i = 0; i < elems * tokens; ++i) {
                std::uint32_t ai = static_cast<std::uint32_t>(a[i]) << 16;
                std::uint32_t bi = static_cast<std::uint32_t>(b[i]) << 16;
                float av, bv;
                std::memcpy(&av, &ai, sizeof(av)); std::memcpy(&bv, &bi, sizeof(bv));
                const float sum = av + bv;
                std::uint32_t bits; std::memcpy(&bits, &sum, sizeof(bits));
                // IEEE round-to-nearest-even, with canonical quiet NaN.
                out[i] = (bits & 0x7fffffffU) > 0x7f800000U ? 0x7fffU :
                    static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
            }
        } else {
            auto* out = static_cast<float*>(buffers.output);
            const auto* a = static_cast<const float*>(buffers.host[0]);
            const auto* b = static_cast<const float*>(buffers.host[1]);
            for (std::size_t i = 0; i < elems * tokens; ++i) out[i] = a[i] + b[i];
        }
    }
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select TE output device");
        if (gather || host_sum) {
            check(cudaMemcpyAsync(gather ? full[r] : ptr[r], buffers.output, gather ? bytes * 2 : bytes,
                                  cudaMemcpyHostToDevice, streams[r]), "TE gathered H2D");
        } else {
            check(cudaMemcpyAsync(buffers.scratch[r], buffers.host[1-r], bytes,
                                  cudaMemcpyHostToDevice, streams[r]), "TE peer H2D");
            const auto count = elems * tokens;
            const auto grid = static_cast<unsigned>((count + 255) / 256);
            if (op == Op::kSumBf16)
                add_bf16_kernel<<<grid, 256, 0, streams[r]>>>(
                    static_cast<std::uint16_t*>(ptr[r]),
                    static_cast<const std::uint16_t*>(buffers.scratch[r]), count);
            else
                add_f32_kernel<<<grid, 256, 0, streams[r]>>>(
                    static_cast<float*>(ptr[r]), static_cast<const float*>(buffers.scratch[r]), count);
            check(cudaPeekAtLastError(), "TE sum kernel");
        }
    }
    // The peer may immediately reuse its input or the pinned buffers; finish
    // both outputs before releasing either rank from this generation.
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select TE output wait device");
        check(cudaStreamSynchronize(streams[r]), "wait TE output");
    }
}
struct BridgeState {
    std::mutex mu;
    std::condition_variable cv;
    unsigned arrived = 0;
    std::uint64_t generation = 0;
    Op op = Op::kGatherBf16;
    std::size_t elems = 0, tokens = 0;
    void* ptr[2] = {};
    void* full[2] = {};
    cudaStream_t streams[2] = {};
    std::exception_ptr failure;
    Buffers buffers;
    TeHostBounceStats stats;
    TePairedAction paired_action = nullptr;
};
BridgeState& bridge_for(int group) {
    static BridgeState states[2];
    if (group < 0 || group > 1) throw std::invalid_argument("invalid TE group");
    return states[group];
}
void wait_peer(int group, int rank, Op op, std::size_t elems, void* primary,
               void* secondary, std::size_t tokens, void* stream) {
    if (rank < 0 || rank > 1) throw std::invalid_argument("invalid TE rank");
    auto& st = bridge_for(group);
    std::unique_lock<std::mutex> lock(st.mu);
    if (st.failure) std::rethrow_exception(st.failure);
    const auto generation = st.generation;
    const bool gather = op == Op::kGatherBf16;
    const bool output_required = gather || op == Op::kSumF32ToBf16;
    if (!primary || (output_required && !secondary) || elems == 0 || tokens == 0 ||
        (st.arrived & (1u << rank)) ||
        (st.arrived && (st.op != op || st.elems != elems || st.tokens != tokens))) {
        st.failure = std::make_exception_ptr(std::runtime_error("TE collective rank/op/extent mismatch"));
        st.cv.notify_all();
        std::rethrow_exception(st.failure);
    }
    st.op = op; st.elems = elems; st.tokens = tokens;
    st.ptr[rank] = primary; st.full[rank] = secondary;
    st.streams[rank] = reinterpret_cast<cudaStream_t>(stream);
    st.arrived |= 1u << rank;
    if (st.arrived != 3) {
        if (!st.cv.wait_for(lock, std::chrono::seconds(30), [&] {
                return st.generation != generation || st.failure;
            })) {
            st.failure = std::make_exception_ptr(std::runtime_error("TE collective peer timeout"));
            st.cv.notify_all();
        }
        if (st.failure) std::rethrow_exception(st.failure);
        return;
    }
    try {
        const auto start = std::chrono::steady_clock::now();
        collective(st.buffers, op, st.ptr, st.full, st.streams, elems, tokens,
                   bridge_p2p.load());
        st.stats.collective_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (op == Op::kGatherBf16) ++st.stats.allgather_calls;
        else ++st.stats.allreduce_calls;
        if (op == Op::kSumF32ToBf16) st.stats.bytes_moved += elems * 6;
        else st.stats.bytes_moved += elems * tokens * (op == Op::kSumF32 ? 4 : 2) *
                               (bridge_p2p.load() ? 2 : (op == Op::kGatherBf16 ? 6 : 4));
        st.arrived = 0;
        ++st.generation;
    } catch (...) {
        st.failure = std::current_exception();
    }
    st.cv.notify_all();
    if (st.failure) std::rethrow_exception(st.failure);
}
} // namespace

void te_host_bounce_allreduce_sum_f32(float* device0, float* device1,
                                      std::size_t elements, TeHostBounceStats* stats) {
    if (!elements) return;
    static Buffers buffers;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    void* ptr[2] = {device0, device1}; void* full[2] = {};
    cudaStream_t streams[2] = {};
    collective(buffers, Op::kSumF32, ptr, full, streams, elements, 1);
    if (stats) { ++stats->allreduce_calls; stats->bytes_moved += elements * sizeof(float) * 4; }
}
void te_host_bounce_allgather_f32(float* device0, float* device1,
                                  float* full0, float* full1,
                                  std::size_t elements, TeHostBounceStats* stats) {
    if (!elements) return;
    static Buffers buffers;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    void* ptr[2] = {device0, device1}; void* full[2] = {full0, full1};
    cudaStream_t streams[2] = {};
    collective(buffers, Op::kGatherF32, ptr, full, streams, elements, 1);
    if (stats) { ++stats->allgather_calls; stats->bytes_moved += elements * sizeof(float) * 6; }
}
bool cuda_te_compiled() { return true; }

static void add_packed_bf16_reduce(cudaGraph_t graph, const TeCollectiveCall calls[2],
    void* scratch[2], const cudaGraphNode_t* inputs, std::size_t input_count,
    void* completed[2], const TeBf16BulkPackStorage& packed, std::size_t count) {
    using Pack = TeBf16BulkPackStorage;
    auto blocks = count / Pack::kBlockWords;
    const auto packet_bytes = blocks * Pack::kStoredBytes;
    cudaGraphNode_t encoded[2], received[2];
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select bulk BF16 pack device");
        auto* input = calls[r].input;
        auto* packets = packed.source[r];
        auto* fallback_tail = packed.fallback_tail[r];
        void* arguments[] = {&input, &packets, &fallback_tail, &blocks};
        cudaKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(pack_bulk_bf16);
        params.gridDim = dim3(std::min<std::size_t>((blocks + 7) / 8, 256));
        params.blockDim = dim3(256);
        params.kernelParams = arguments;
        check(cudaGraphAddKernelNode(&encoded[r], graph, inputs, input_count, &params),
              "add paired bulk BF16 packing");
    }
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select bulk BF16 receive device");
        check(cudaGraphAddMemcpyNode1D(&received[r], graph, &encoded[1-r], 1,
            scratch[r], packed.source[1-r], packet_bytes, cudaMemcpyDeviceToDevice),
            "add paired bulk BF16 packet copy");
    }
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select bulk BF16 reduce device");
        auto* input = calls[r].input;
        auto* peer_packets = scratch[r];
        auto* peer_fallback_tail = packed.fallback_tail[1-r];
        void* arguments[] = {&input, &peer_packets, &peer_fallback_tail, &blocks, &r};
        cudaKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(sum_packed_bulk_bf16);
        params.gridDim = dim3(std::min<std::size_t>((count + 255) / 256, 256));
        params.blockDim = dim3(256);
        params.kernelParams = arguments;
        cudaGraphNode_t reduced;
        // Fallback reads are from a separate peer tail, never from either
        // original input. The next collective waits on both reduce tails
        // before reusing the packet and fallback allocations.
        check(cudaGraphAddKernelNode(&reduced, graph, received, 2, &params),
              "add paired bulk BF16 decode and sum");
        completed[r] = reduced;
    }
}

void te_graph_add_collective(void* opaque_graph, const TeCollectiveCall calls[2],
                             void* scratch[2], void* dependencies[2], void* completed[2],
                             bool bulk_prefill, const TeBf16BulkPackStorage* packed) {
    DeviceGuard guard;
    const auto& a = calls[0];
    const auto& b = calls[1];
    const bool gather = a.kind == TeCollectiveCall::Kind::kGatherBf16;
    const bool fp32 = a.kind == TeCollectiveCall::Kind::kSumF32;
    const std::size_t unit = fp32 ? 4 : 2;
    if (!opaque_graph || !scratch[0] || !scratch[1] || a.rank != 0 || b.rank != 1 ||
        (!gather && !fp32 && a.kind != TeCollectiveCall::Kind::kSumBf16) ||
        a.kind != b.kind || a.group != b.group || a.elements != b.elements ||
        a.tokens != b.tokens || !a.tokens || !a.elements || !a.input || !b.input ||
        (gather && (!a.output || !b.output)) ||
        a.elements > std::numeric_limits<std::size_t>::max() / a.tokens / unit / 2)
        throw std::invalid_argument("TE graph collective descriptors differ");
    const auto count = a.elements * a.tokens * (gather ? 2 : 1);
    const auto* bulk_copy = std::getenv("Q38_TP_BULK_COPY");
    const bool staged_sum = bulk_prefill && !gather && count * unit >= (256u << 10u) &&
        bulk_copy && std::strcmp(bulk_copy, "1") == 0;
    const bool vector=peer_vector_enabled(a,b,scratch,count,fp32,gather);
    const unsigned width=vector?(fp32?4:8):1;
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>((count/width + 255) / 256, 256));
    auto graph = static_cast<cudaGraph_t>(opaque_graph);
    cudaGraphNode_t inputs[2], reads[2];
    std::size_t input_count = 0;
    for (int r = 0; r < 2; ++r)
        if (dependencies[r]) inputs[input_count++] = static_cast<cudaGraphNode_t>(dependencies[r]);
    if (staged_sum && !fp32 && packed && count % TeBf16BulkPackStorage::kBlockWords == 0) {
        add_packed_bf16_reduce(graph, calls, scratch, inputs, input_count,
                               completed, *packed, count);
        return;
    }
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select TE graph reader device");
        if (staged_sum) {
            check(cudaGraphAddMemcpyNode1D(&reads[r], graph, inputs, input_count,
                scratch[r], calls[1-r].input, count * unit, cudaMemcpyDeviceToDevice),
                "add paired TE bulk peer copy");
            continue;
        }
        void* first = a.input;
        void* second = b.input;
        void* output = scratch[r];
        auto elements = a.elements;
        auto total = count;
        void* arguments[] = {&first, &second, &output, &elements, &total};
        cudaKernelNodeParams params{};
        params.func = vector?(fp32?reinterpret_cast<void*>(peer_collect_vector_kernel<true,false>):gather?reinterpret_cast<void*>(peer_collect_vector_kernel<false,true>):reinterpret_cast<void*>(peer_collect_vector_kernel<false,false>)):fp32 ? reinterpret_cast<void*>(peer_collect_kernel<float, false>) :
            gather ? reinterpret_cast<void*>(peer_collect_kernel<std::uint16_t, true>) :
                     reinterpret_cast<void*>(peer_collect_kernel<std::uint16_t, false>);
        params.gridDim = dim3(blocks); params.blockDim = dim3(256);
        params.kernelParams = arguments;
        check(cudaGraphAddKernelNode(&reads[r], graph, inputs, input_count, &params),
              "add TE graph peer read");
    }
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select TE graph commit device");
        void* input = scratch[r];
        void* output = gather ? calls[r].output : calls[r].input;
        cudaGraphNode_t node;
        if (staged_sum) {
            // Each copy must complete before either rank can overwrite a
            // producer. Sum local and staged input in the eager rank order.
            void* first = r == 0 ? output : input;
            void* second = r == 0 ? input : output;
            auto elements = a.elements;
            auto total = count;
            void* arguments[] = {&first, &second, &output, &elements, &total};
            cudaKernelNodeParams params{};
            params.func = fp32 ? reinterpret_cast<void*>(peer_collect_kernel<float, false>) :
                reinterpret_cast<void*>(peer_collect_kernel<std::uint16_t, false>);
            params.gridDim = dim3(std::min<std::size_t>((count + 255) / 256, 256));
            params.blockDim = dim3(256);
            params.kernelParams = arguments;
            check(cudaGraphAddKernelNode(&node, graph, reads, 2, &params),
                  "add paired TE bulk local sum");
            completed[r] = node;
            continue;
        }
        // Match eager P2P's device-copy commit, retaining both reader fences.
        check(cudaGraphAddMemcpyNode1D(&node, graph, reads, 2, output, input,
            count * unit, cudaMemcpyDeviceToDevice), "add TE graph commit");
        completed[r] = node;
    }
}
void te_graph_add_device_collective(void* graphs[2], const TeCollectiveCall calls[2],
                                    void* scratch[2], void* flags[2],
                                    void* dependencies[2], void* completed[2]) {
    DeviceGuard guard;
    const auto& a = calls[0]; const auto& b = calls[1];
    const bool gather = a.kind == TeCollectiveCall::Kind::kGatherBf16;
    const bool fp32 = a.kind == TeCollectiveCall::Kind::kSumF32;
    const std::size_t unit = fp32 ? 4 : 2;
    if (!graphs[0] || !graphs[1] || !flags[0] || !flags[1] || !scratch[0] || !scratch[1] ||
        a.rank != 0 || b.rank != 1 || a.kind != b.kind || a.group != b.group ||
        (!gather && !fp32 && a.kind != TeCollectiveCall::Kind::kSumBf16) ||
        a.elements != b.elements || a.tokens != b.tokens || !a.tokens || !a.elements ||
        !a.input || !b.input || (gather && (!a.output || !b.output)) ||
        a.elements > std::numeric_limits<std::size_t>::max() / a.tokens / unit / 2)
        throw std::invalid_argument("TE device graph collective descriptors differ");
    const auto count = a.elements * a.tokens * (gather ? 2 : 1);
    const bool vector=peer_vector_enabled(a,b,scratch,count,fp32,gather);
    const unsigned width=vector?(fp32?4:8):1;
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>((count/width + 255) / 256, 256));
    for (int r = 0; r < 2; ++r) {
        check(cudaSetDevice(r), "select TE device graph rank");
        auto graph = static_cast<cudaGraph_t>(graphs[r]);
        auto tail = static_cast<cudaGraphNode_t>(dependencies[r]);
        auto add = [&](void* function, void** arguments, unsigned grid, unsigned block) {
            cudaKernelNodeParams params{};
            params.func = function; params.kernelParams = arguments;
            params.gridDim = dim3(grid); params.blockDim = dim3(block);
            cudaGraphNode_t next;
            check(cudaGraphAddKernelNode(&next, graph, tail ? &tail : nullptr,
                tail ? 1 : 0, &params), "add TE device graph fence/kernel");
            tail = next;
        };
        auto* own = static_cast<unsigned*>(flags[r]);
        auto* peer = static_cast<unsigned*>(flags[1-r]);
        const auto* fused_flag=std::getenv("Q38_TP_PEER_FUSED");
        if (fused_flag && std::strcmp(fused_flag,"1")==0 && count*unit <= (256u<<10)) {
            const unsigned lane=fp32?4:8;
            // Vectorization is safe only for aligned whole rows. Odd/tiny
            // payloads preserve the scalar loads and leave guard tails intact.
            const bool aligned=count%lane==0 && (!gather || a.elements%lane==0) &&
                !(reinterpret_cast<std::uintptr_t>(a.input)%16) &&
                !(reinterpret_cast<std::uintptr_t>(b.input)%16) &&
                !(reinterpret_cast<std::uintptr_t>(scratch[r])%16) &&
                !(reinterpret_cast<std::uintptr_t>(gather?calls[r].output:calls[r].input)%16);
            void* first=a.input; void* second=b.input; void* temporary=scratch[r];
            void* destination=gather?calls[r].output:calls[r].input;
            auto elements=a.elements; auto total=count;
            void* args[]={&first,&second,&temporary,&destination,&elements,&total,&own,&peer};
            void* function=aligned ?
                (fp32?reinterpret_cast<void*>(graph_collect_fused_kernel<true,false,true>):
                 gather?reinterpret_cast<void*>(graph_collect_fused_kernel<false,true,true>):
                        reinterpret_cast<void*>(graph_collect_fused_kernel<false,false,true>)) :
                (fp32?reinterpret_cast<void*>(graph_collect_fused_kernel<true,false,false>):
                 gather?reinterpret_cast<void*>(graph_collect_fused_kernel<false,true,false>):
                        reinterpret_cast<void*>(graph_collect_fused_kernel<false,false,false>));
            add(function,args,1,1024);
            completed[r]=tail;
            continue;
        }
        bool ready = true;
        void* signal_arguments[] = {&own, &peer, &ready};
        add(reinterpret_cast<void*>(graph_signal_kernel), signal_arguments, 1, 1);
        void* first = a.input; void* second = b.input; void* output = scratch[r];
        auto elements = a.elements; auto total = count;
        void* read_arguments[] = {&first, &second, &output, &elements, &total};
        add(vector?(fp32?reinterpret_cast<void*>(peer_collect_vector_kernel<true,false>):gather?reinterpret_cast<void*>(peer_collect_vector_kernel<false,true>):reinterpret_cast<void*>(peer_collect_vector_kernel<false,false>)):fp32 ? reinterpret_cast<void*>(peer_collect_kernel<float, false>) :
            gather ? reinterpret_cast<void*>(peer_collect_kernel<std::uint16_t, true>) :
                     reinterpret_cast<void*>(peer_collect_kernel<std::uint16_t, false>),
            read_arguments, blocks, 256);
        ready = false;
        add(reinterpret_cast<void*>(graph_signal_kernel), signal_arguments, 1, 1);
        void* input = scratch[r]; output = gather ? calls[r].output : calls[r].input;
        cudaGraphNode_t committed;
        check(cudaGraphAddMemcpyNode1D(&committed, graph, &tail, 1, output, input,
            count * unit, cudaMemcpyDeviceToDevice), "add rank-local TE commit");
        tail = committed;
        completed[r] = tail;
    }
}
bool te_bridge_p2p_enabled() { return bridge_p2p.load(); }
static void pair_host_action(int group, int rank, void* context,
                             TePairedAction action, bool graph_replay) {
    if (rank < 0 || rank > 1) throw std::invalid_argument("invalid paired graph rank");
    auto& st = bridge_for(group);
    std::unique_lock<std::mutex> lock(st.mu);
    if (st.failure) std::rethrow_exception(st.failure);
    const auto generation = st.generation;
    if (!context || !action || !bridge_p2p.load() || (st.arrived & (1u << rank)) ||
        (st.arrived && (st.op != Op::kPairedGraph || st.paired_action != action))) {
        st.failure = std::make_exception_ptr(std::runtime_error("TE paired graph rendezvous mismatch"));
        st.cv.notify_all();
        std::rethrow_exception(st.failure);
    }
    st.op = Op::kPairedGraph;
    st.paired_action = action;
    st.ptr[rank] = context;
    st.arrived |= 1u << rank;
    if (st.arrived != 3) {
        if (!st.cv.wait_for(lock, std::chrono::seconds(30), [&] {
                return st.generation != generation || st.failure;
            })) {
            st.failure = std::make_exception_ptr(std::runtime_error("TE paired graph peer timeout"));
            st.cv.notify_all();
        }
    } else {
        try {
            action(st.ptr[0], st.ptr[1]);
            if (graph_replay) ++st.stats.paired_graph_calls;
            st.arrived = 0;
            ++st.generation;
        } catch (...) { st.failure = std::current_exception(); }
        st.cv.notify_all();
    }
    if (st.failure) std::rethrow_exception(st.failure);
}
void te_bridge_pair_graph(int group, int rank, void* context, TePairedAction action) {
    pair_host_action(group, rank, context, action, true);
}
bool te_bridge_pair_prefill_ready(int group, int rank, bool captured) {
    pair_host_action(group, rank, &captured, [](void* first, void* second) {
        auto& rank0 = *static_cast<bool*>(first);
        auto& rank1 = *static_cast<bool*>(second);
        const bool ready = rank0 && rank1;
        rank0 = ready;
        rank1 = ready;
    }, false);
    return captured;
}
TeBridge* te_bridge_global() { static TeBridge bridge; return &bridge; }
void te_bridge_use_p2p(bool enabled, bool stream_ordered) {
    if (stream_ordered && !enabled)
        throw std::invalid_argument("stream-ordered collectives require P2P");
    auto& first = bridge_for(0);
    auto& second = bridge_for(1);
    std::scoped_lock lock(first.mu, second.mu);
    if (first.arrived || second.arrived)
        throw std::logic_error("cannot change active TE transport");
    DeviceGuard transport_guard;
    first.buffers.drain_peer();
    second.buffers.drain_peer();
    if (enabled) {
        DeviceGuard guard;
        for (int r = 0; r < 2; ++r) {
            check(cudaSetDevice(r), "select TE P2P device");
            int capable = 0;
            check(cudaDeviceCanAccessPeer(&capable, r, 1-r), "query TE P2P access");
            if (!capable) throw std::runtime_error("TE P2P transport requires bidirectional peer access");
            const auto status = cudaDeviceEnablePeerAccess(1-r, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
            else check(status, "enable TE P2P access");
        }
    }
    bridge_p2p.store(enabled);
    bridge_stream_ordered.store(stream_ordered);
}
void te_bridge_reset() {
    te_bridge_reset_group(0);
    te_bridge_reset_group(1);
}
void te_bridge_reset_group(int group) {
    DeviceGuard guard;
    {
        auto& st = bridge_for(group);
        std::lock_guard<std::mutex> lock(st.mu);
        if (st.arrived && !st.failure) throw std::logic_error("cannot reset active TE bridge");
        st.buffers.drain_peer();
        st.arrived = 0; st.failure = nullptr; ++st.generation;
    }
}
TeHostBounceStats te_bridge_stats(int group) {
    if (group < -1 || group > 1) throw std::invalid_argument("invalid TE stats group");
    TeHostBounceStats total;
    for (int g = group < 0 ? 0 : group; g < (group < 0 ? 2 : group + 1); ++g) {
        auto& st = bridge_for(g);
        std::lock_guard<std::mutex> lock(st.mu);
        total.allgather_calls += st.stats.allgather_calls;
        total.allreduce_calls += st.stats.allreduce_calls;
        total.bytes_moved += st.stats.bytes_moved;
        total.collective_ns += st.stats.collective_ns;
        total.paired_graph_calls += st.stats.paired_graph_calls;
    }
    return total;
}
void te_bridge_abort() {
    for (int g = 0; g < 2; ++g) {
        auto& st = bridge_for(g);
        std::lock_guard<std::mutex> lock(st.mu);
        if (!st.failure) st.failure = std::make_exception_ptr(std::runtime_error("TE peer aborted"));
        st.cv.notify_all();
    }
}
void te_bridge_allgather_bf16(int rank, std::uint16_t* half, std::uint16_t* full,
                              std::size_t elems, std::size_t tokens, int group, void* stream) {
    if (collective_recorder) {
        collective_recorder->record({TeCollectiveCall::Kind::kGatherBf16,
            rank, half, full, elems, tokens, group, stream});
        return;
    }
    wait_peer(group, rank, Op::kGatherBf16, elems, half, full, tokens, stream);
}
void te_bridge_allreduce_sum_bf16(int rank, std::uint16_t* buf,
                                  std::size_t elems, int group, void* stream) {
    if (collective_recorder) {
        collective_recorder->record({TeCollectiveCall::Kind::kSumBf16,
            rank, buf, nullptr, elems, 1, group, stream});
        return;
    }
    wait_peer(group, rank, Op::kSumBf16, elems, buf, nullptr, 1, stream);
}
void te_bridge_allreduce_sum_f32(int rank, float* buf, std::size_t elems,
                                 int group, void* stream) {
    if (collective_recorder) {
        collective_recorder->record({TeCollectiveCall::Kind::kSumF32,
            rank, buf, nullptr, elems, 1, group, stream});
        return;
    }
    wait_peer(group, rank, Op::kSumF32, elems, buf, nullptr, 1, stream);
}
void te_bridge_set_recorder(TeCollectiveRecorder* recorder) {
    if (recorder && collective_recorder)
        throw std::logic_error("nested TE collective recording is invalid");
    collective_recorder = recorder;
}
void te_bridge_allreduce_f32_to_bf16(int rank, const float* input,
    std::uint16_t* output, std::size_t elements, int group, void* stream) {
    if (collective_recorder || !bridge_p2p.load())
        throw std::logic_error("RSAG is an eager P2P collective only");
    wait_peer(group, rank, Op::kSumF32ToBf16, elements,
              const_cast<float*>(input), output, 1, stream);
}
bool te_bridge_f32_to_bf16_available() {
    return collective_recorder == nullptr && bridge_p2p.load();
}
void te_bridge_allreduce_f32_as_bf16(int rank, const float* input,
    std::uint16_t* output, std::size_t elements, int group, void* stream,
    const std::function<void()>& enqueue) {
    if (rank < 0 || rank > 1 || group < 0 || group > 1 || !input || !output ||
        !elements || elements > std::numeric_limits<std::size_t>::max()/8)
        throw std::invalid_argument("invalid BF16 communication extent");
    const auto source = reinterpret_cast<std::uintptr_t>(input);
    const auto target = reinterpret_cast<std::uintptr_t>(output);
    if ((source <= target && target-source < elements*sizeof(float)) ||
        (target < source && source-target < elements*sizeof(std::uint16_t)))
        throw std::invalid_argument("BF16 communication input/output overlap");
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(
        (elements+255)/256, 256));
    round_partial_bf16_kernel<<<blocks,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        input, output, elements);
    check(cudaPeekAtLastError(), "round rank partial before BF16 communication");
    if (enqueue && collective_recorder) {
        collective_recorder->record_parallel({TeCollectiveCall::Kind::kSumBf16,
            rank, output, nullptr, elements, 1, group, stream}, enqueue);
        return;
    }
    if (enqueue) enqueue();
    te_bridge_allreduce_sum_bf16(rank, output, elements, group, stream);
}
void te_bridge_allreduce_sum_f32_parallel(int rank, float* buf,
    std::size_t elements, int group, void* stream,
    const std::function<void()>& enqueue) {
    if (!enqueue) throw std::invalid_argument("empty parallel TE branch");
    if (collective_recorder) {
        collective_recorder->record_parallel({TeCollectiveCall::Kind::kSumF32,
            rank, buf, nullptr, elements, 1, group, stream}, enqueue);
        return;
    }
    enqueue();
    te_bridge_allreduce_sum_f32(rank, buf, elements, group, stream);
}
void te_bridge_replay(const TeCollectiveCall& call) {
    if (collective_recorder)
        throw std::logic_error("cannot replay TE collective while recording");
    switch (call.kind) {
    case TeCollectiveCall::Kind::kGatherBf16:
        te_bridge_allgather_bf16(call.rank, static_cast<std::uint16_t*>(call.input),
            static_cast<std::uint16_t*>(call.output), call.elements, call.tokens,
            call.group, call.stream);
        return;
    case TeCollectiveCall::Kind::kSumBf16:
        te_bridge_allreduce_sum_bf16(call.rank, static_cast<std::uint16_t*>(call.input),
            call.elements, call.group, call.stream);
        return;
    case TeCollectiveCall::Kind::kSumF32:
        te_bridge_allreduce_sum_f32(call.rank, static_cast<float*>(call.input),
            call.elements, call.group, call.stream);
        return;
    }
    throw std::logic_error("invalid recorded TE collective");
}
} // namespace q38
