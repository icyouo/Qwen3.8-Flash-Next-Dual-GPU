#include "q38/cuda_te.h"
#include "q38/cuda_kernels.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstring>

void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool v, const char* why) { if(!v) throw std::runtime_error(why); }
__global__ void delayed_fill(std::uint16_t* p, int n, int rank, unsigned long long delay) {
    const auto start = clock64();
    while (clock64() - start < delay) {}
    for(int i=threadIdx.x; i<n; i+=blockDim.x)
        p[i] = __bfloat16_as_ushort(__float2bfloat16_rn(float(rank*16+i+1)));
}
template<class F> void ranks(F fn) {
    std::exception_ptr errors[2];
    std::thread a([&]{try{fn(0);}catch(...){q38::te_bridge_abort(); errors[0]=std::current_exception();}});
    std::thread b([&]{try{fn(1);}catch(...){q38::te_bridge_abort(); errors[1]=std::current_exception();}});
    a.join(); b.join();
    for(auto e:errors) if(e) std::rethrow_exception(e);
}
__global__ void fill_f32(float* value, float input, unsigned long long delay) {
    const auto start = clock64();
    while (clock64() - start < delay) {}
    *value = input;
}
int main(int argc, char** argv) {
    try {
        const bool ordered = argc == 2 && std::string(argv[1]) == "p2p-async";
        const bool peer = ordered || (argc == 2 && std::string(argv[1]) == "p2p");
        if (argc > 2 || (argc == 2 && !peer)) throw std::runtime_error("usage: q38-te-tests [p2p|p2p-async]");
        q38::te_bridge_reset();
        q38::te_bridge_use_p2p(peer, ordered);
        std::uint16_t* data[2]; cudaStream_t streams[2];
        for(int r=0;r<2;++r) {
            check(cudaSetDevice(r)); check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
            check(cudaMalloc(&data[r], 1024*2)); check(cudaMemset(data[r],0,1024*2));
        }
        ranks([&](int r) {
            check(cudaSetDevice(r));
            // Three token rows and in-place gather: default-stream copies would
            // read old zeros while this delayed nonblocking producer is running.
            delayed_fill<<<1,32,0,streams[r]>>>(data[r],12,r,20000000);
            q38::te_bridge_allgather_bf16(r,data[r],data[r],4,3,0,streams[r]);
            int current=-1; check(cudaGetDevice(&current)); require(current==r,"device not restored");
            check(cudaStreamSynchronize(streams[r]));
            std::uint16_t out[24]; check(cudaMemcpy(out,data[r],sizeof(out),cudaMemcpyDeviceToHost));
            for(int t=0;t<3;++t) for(int rank=0;rank<2;++rank) for(int i=0;i<4;++i)
                require(out[t*8+rank*4+i] == __bfloat16_as_ushort(__float2bfloat16_rn(float(rank*16+t*4+i+1))),"producer stream / token gather mismatch");
        });
        ranks([&](int r) {
            check(cudaSetDevice(r));
            // 1 + 3/256 rounds to 1.015625 (RNE); truncation returns 1.0078125.
            const std::uint16_t value=__bfloat16_as_ushort(__float2bfloat16_rn(r ? 3.0f/256 : 1.0f));
            check(cudaMemcpyAsync(data[r],&value,2,cudaMemcpyHostToDevice,streams[r]));
            q38::te_bridge_allreduce_sum_bf16(r,data[r],1,0,streams[r]);
            check(cudaStreamSynchronize(streams[r]));
            std::uint16_t result; check(cudaMemcpy(&result,data[r],2,cudaMemcpyDeviceToHost));
            require(result == __bfloat16_as_ushort(__float2bfloat16_rn(1.0f+3.0f/256)),"BF16 RNE mismatch");
        });
        // Exercise both stage groups concurrently and repeated generations.
        std::thread groups[2]; std::exception_ptr group_error[2];
        for(int g=0;g<2;++g) groups[g]=std::thread([&,g]{try{
            ranks([&](int r){check(cudaSetDevice(r));
                cudaStream_t s; check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
                float* p; check(cudaMalloc(&p,32*sizeof(float)));
                for(int it=0;it<100;++it) {
                    const float value=float(r+it); check(cudaMemcpyAsync(p,&value,4,cudaMemcpyHostToDevice,s));
                    q38::te_bridge_allreduce_sum_f32(r,p,1,g,s);
                    check(cudaStreamSynchronize(s));
                    float out; check(cudaMemcpy(&out,p,4,cudaMemcpyDeviceToHost)); require(out==1+2*it,"generation/group mismatch");
                }
                check(cudaFree(p)); check(cudaStreamDestroy(s));
            });
        }catch(...){group_error[g]=std::current_exception();}});
        for(auto& g:groups) g.join(); for(auto e:group_error) if(e) std::rethrow_exception(e);
        // Grow scratch beyond the small-reduction path and check every element,
        // including the non-block-aligned tail, with different rank payloads.
        ranks([&](int r) {
            check(cudaSetDevice(r));
            constexpr int n=32771;
            float* fp; std::uint16_t* bp;
            check(cudaMalloc(&fp,(n+1)*sizeof(float)));
            check(cudaMalloc(&bp,(n+1)*sizeof(std::uint16_t)));
            std::vector<float> f(n+1), fo(n+1);
            std::vector<std::uint16_t> b(n+1), bo(n+1);
            auto value=[](int i,int rank) { return float((i*17+rank*43)%211-105)/32.0f; };
            for(int i=0;i<n;++i) {
                f[i]=value(i,r); b[i]=__bfloat16_as_ushort(__float2bfloat16_rn(f[i]));
            }
            f[n]=12345.0f; b[n]=0x55aa;
            check(cudaMemcpyAsync(fp,f.data(),f.size()*sizeof(float),cudaMemcpyHostToDevice,streams[r]));
            check(cudaMemcpyAsync(bp,b.data(),b.size()*2,cudaMemcpyHostToDevice,streams[r]));
            q38::te_bridge_allreduce_sum_f32(r,fp,n,0,streams[r]);
            q38::te_bridge_allreduce_sum_bf16(r,bp,n,0,streams[r]);
            check(cudaStreamSynchronize(streams[r]));
            check(cudaMemcpy(fo.data(),fp,fo.size()*sizeof(float),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(bo.data(),bp,bo.size()*2,cudaMemcpyDeviceToHost));
            for(int i=0;i<n;++i) {
                require(fo[i]==value(i,0)+value(i,1),"large F32 sum mismatch");
                const auto a=__float2bfloat16_rn(value(i,0)), c=__float2bfloat16_rn(value(i,1));
                require(bo[i]==__bfloat16_as_ushort(__float2bfloat16_rn(__bfloat162float(a)+__bfloat162float(c))),"large BF16 sum mismatch");
            }
            require(fo[n]==12345.0f && bo[n]==0x55aa,"collective tail overwritten");
            check(cudaFree(fp)); check(cudaFree(bp));
        });
        // MTP4 head payload: coalescing five vocabulary reductions must retain
        // exact BF16 results, including scratch growth with work in flight.
        ranks([&](int r) {
            check(cudaSetDevice(r));
            constexpr std::size_t vocab = 248320, rows = 5, n = vocab * rows;
            std::vector<std::uint16_t> input(n + 1), sequential(n + 1), packed(n + 1);
            for (std::size_t i = 0; i < n; ++i)
                input[i] = __bfloat16_as_ushort(__float2bfloat16_rn(
                    float(int((i * 17 + r * 43) % 211) - 105) / 32.0f));
            input[n] = 0x55aa;
            std::uint16_t *a, *b;
            check(cudaMalloc(&a, input.size() * 2));
            check(cudaMalloc(&b, input.size() * 2));
            check(cudaMemcpyAsync(a, input.data(), input.size() * 2,
                                  cudaMemcpyHostToDevice, streams[r]));
            check(cudaMemcpyAsync(b, input.data(), input.size() * 2,
                                  cudaMemcpyHostToDevice, streams[r]));
            for (std::size_t row = 0; row < rows; ++row)
                q38::te_bridge_allreduce_sum_bf16(r, a + row * vocab, vocab, 0, streams[r]);
            q38::te_bridge_allreduce_sum_bf16(r, b, n, 0, streams[r]);
            check(cudaStreamSynchronize(streams[r]));
            check(cudaMemcpy(sequential.data(), a, input.size() * 2, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(packed.data(), b, input.size() * 2, cudaMemcpyDeviceToHost));
            require(sequential == packed, "packed MTP4 vocabulary reductions differ");
            require(packed[n] == 0x55aa, "packed MTP4 vocabulary tail overwritten");
            check(cudaFree(a)); check(cudaFree(b));
        });
        // Keep many generations in flight, alternate streams and disjoint
        // input/output buffers, delay producers asymmetrically. A scratch
        // reuse race can pass individually synchronized tests above.
        ranks([&](int r) {
            check(cudaSetDevice(r));
            constexpr int rounds = 1024;
            cudaStream_t lanes[2]; float* values; float* results;
            for (auto& lane : lanes)
                check(cudaStreamCreateWithFlags(&lane, cudaStreamNonBlocking));
            check(cudaMalloc(&values, rounds * sizeof(float)));
            check(cudaMalloc(&results, rounds * sizeof(float)));
            for (int generation = 0; generation < rounds; ++generation) {
                auto lane = lanes[generation % 2];
                fill_f32<<<1,1,0,lane>>>(values + generation,
                    float(generation * 4 + r),
                    generation % 17 == r ? 1000000 : 0);
                q38::te_bridge_allreduce_sum_f32(r, values + generation,
                    1, 0, lane);
                check(cudaMemcpyAsync(results + generation, values + generation,
                    sizeof(float), cudaMemcpyDeviceToDevice, lane));
            }
            for (auto lane : lanes) check(cudaStreamSynchronize(lane));
            std::vector<float> actual(rounds);
            check(cudaMemcpy(actual.data(), results, rounds * sizeof(float), cudaMemcpyDeviceToHost));
            for (int generation = 0; generation < rounds; ++generation)
                require(actual[generation] == float(generation * 8 + 1),
                        "queued cross-stream generation mismatch");
            check(cudaFree(values)); check(cudaFree(results));
            for (auto lane : lanes) check(cudaStreamDestroy(lane));
        });
        // A mismatched collective must wake both ranks, not strand a waiter.
        bool failed[2]={false,false}; std::thread mismatch[2];
        for(int r=0;r<2;++r) mismatch[r]=std::thread([&,r]{try {
            check(cudaSetDevice(r));
            q38::te_bridge_allreduce_sum_bf16(r,data[r],r+1,0,streams[r]);
        }catch(...){failed[r]=true;}});
        for(auto& t:mismatch)t.join(); require(failed[0]&&failed[1],"mismatch did not fail both ranks");
        q38::te_bridge_reset();
        // Optimized router on both GPUs, nonblocking streams, exact stable ties.
        ranks([&](int r){check(cudaSetDevice(r));
            std::int32_t* ids; float* weights;
            check(cudaMalloc(&ids,10*4));check(cudaMalloc(&weights,10*4));
            std::vector<std::uint16_t> logits(512);
            for(int i=0;i<512;++i)logits[i]=__bfloat16_as_ushort(__float2bfloat16_rn(float(i%16)));
            check(cudaMemcpyAsync(data[r],logits.data(),1024,cudaMemcpyHostToDevice,streams[r]));
            q38::cuda_topk_router_bf16(data[r],ids,weights,1,512,10,true,streams[r],r);
            check(cudaStreamSynchronize(streams[r]));
            std::int32_t out[10];check(cudaMemcpy(out,ids,40,cudaMemcpyDeviceToHost));
            for(int i=0;i<10;++i)require(out[i]==15+16*i,"optimized router mismatch");
            check(cudaFree(ids));check(cudaFree(weights));
        });
        for(int r=0;r<2;++r){check(cudaSetDevice(r));check(cudaFree(data[r]));check(cudaStreamDestroy(streams[r]));}
        std::cout << "PASS transport=" << (ordered ? "p2p-async" : peer ? "p2p" : "host")
                  << ": producer streams, in-place token gather, BF16 RNE, device restoration, 200 group reductions, 1024 queued cross-stream generations, large F32/BF16 payloads and tails, peer failure, dual-GPU optimized router\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
