#include "q38/cuda_te_graph.h"
#include "q38/cuda_te.h"
#include "q38/cuda_padded_gemm.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <array>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdlib>
#include <chrono>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void fill(float* values, std::uint16_t* half, const int* seed, int rank) {
    const auto i=threadIdx.x;
    if (i<17) values[i]=float(*seed+rank*2+i);
    if (i<8) reinterpret_cast<__nv_bfloat16*>(half)[i]=__float2bfloat16_rn(float(*seed+rank*8+i%4));
}
__global__ void independent(float* output, const int* seed, int rank) {
    const auto start = clock64();
    while (clock64() - start < (rank ? 200000ull : 10000ull)) {}
    if (threadIdx.x < 17) output[threadIdx.x] = float((*seed + rank*3 + threadIdx.x)*7);
}
__global__ void finish(float* values, const float* independent_output) {
    if (threadIdx.x<17) values[threadIdx.x]=values[threadIdx.x]*.5f+1.f+independent_output[threadIdx.x];
}
__global__ void widen(const std::uint16_t* partial, float* values) {
    if (threadIdx.x < 17)
        values[threadIdx.x] = __bfloat162float(__ushort_as_bfloat16(partial[threadIdx.x]));
}
struct Rank {
    int device;
    bool bf16_partial;
    cudaStream_t stream = nullptr;
    float* values=nullptr;
    float* branch=nullptr;
    std::uint16_t *half=nullptr,*full=nullptr;
    std::uint16_t* partial=nullptr;
    int* seed=nullptr;
    std::unique_ptr<q38::CudaTeGraphTape> tape;
    explicit Rank(int d,bool bf16=false):device(d),bf16_partial(bf16) {
        check(cudaSetDevice(d)); check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaMalloc(&values,17*4));check(cudaMalloc(&half,8*2));check(cudaMalloc(&full,16*2));check(cudaMalloc(&seed,4));
        check(cudaMalloc(&branch,17*4));
        check(cudaMalloc(&partial,17*2));
        check(cudaMemsetAsync(values,0,17*4,stream));check(cudaMemsetAsync(half,0,8*2,stream));
        check(cudaMemsetAsync(full,0,16*2,stream));check(cudaMemsetAsync(seed,0,4,stream));
        check(cudaStreamSynchronize(stream));
    }
    ~Rank() {
        cudaSetDevice(device); cudaStreamSynchronize(stream); tape.reset();
        cudaFree(partial);cudaFree(branch);cudaFree(values);cudaFree(half);cudaFree(full);cudaFree(seed);cudaStreamDestroy(stream);
    }
    void enqueue(int group) {
        fill<<<1,32,0,stream>>>(values,half,seed,device);
        auto parallel = [&] {
            independent<<<1,32,0,stream>>>(branch,seed,device);
            check(cudaPeekAtLastError());
        };
        if (bf16_partial) {
            q38::te_bridge_allreduce_f32_as_bf16(device,values,partial,17,group,stream,parallel);
            widen<<<1,32,0,stream>>>(partial,values);
        } else q38::te_bridge_allreduce_sum_f32_parallel(device,values,17,group,stream,parallel);
        q38::te_bridge_allgather_bf16(device,half,full,4,2,group,stream);
        q38::te_bridge_allreduce_sum_bf16(device,full,16,group,stream);
        finish<<<1,32,0,stream>>>(values,branch);
        check(cudaPeekAtLastError());
    }
    void verify(int input) {
        std::array<float,17> result;
        std::array<__nv_bfloat16,16> gathered;
        check(cudaMemcpyAsync(result.data(),values,sizeof(result),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(gathered.data(),full,sizeof(gathered),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for (unsigned i=0;i<result.size();++i)
            if (result[i]!=float(input+2+i+7*(input+device*3+i))) throw std::runtime_error("TE tape changing input/parallel join arithmetic differs");
        for (unsigned i=0;i<gathered.size();++i) {
            const auto rank=(i%8)/4;
            if (__bfloat162float(gathered[i])!=float(2*(input+rank*8+i%4)))
                throw std::runtime_error("TE tape gather layout/reduction differs");
        }
    }
};
template<class F> void pair(F fn) {
    std::exception_ptr errors[2];
    std::thread peer([&]{try{fn(1);}catch(...){errors[1]=std::current_exception();q38::te_bridge_abort();}});
    try{fn(0);}catch(...){errors[0]=std::current_exception();q38::te_bridge_abort();}
    peer.join();for(auto error:errors)if(error)std::rethrow_exception(error);
}

struct GemmRank {
    int device;
    int rows, columns;
    bool fixed_workspace;
    cudaStream_t stream{};
    cublasHandle_t handle{};
    void* workspace{};
    std::uint16_t *weights{}, *input{}, *output{}, *scratch{};
    std::unique_ptr<q38::CudaTeGraphTape> tape;
    static constexpr std::size_t workspace_bytes=32u<<20u;
    explicit GemmRank(int d,int r=32,int c=16,bool fixed=true):device(d),rows(r),columns(c),fixed_workspace(fixed) {
        check(cudaSetDevice(d));check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("test cuBLAS create");
        check(cudaMalloc(&workspace,workspace_bytes));
        check(cudaMalloc(&weights,rows*columns*2));check(cudaMalloc(&input,5*columns*2));
        check(cudaMalloc(&output,5*rows*2));check(cudaMalloc(&scratch,8*(columns+rows)*2));
        std::vector<__nv_bfloat16> w(rows*columns);
        for(unsigned i=0;i<w.size();++i)w[i]=__float2bfloat16_rn(float(int(i%13)-6)/16.f);
        check(cudaMemcpy(weights,w.data(),w.size()*2,cudaMemcpyHostToDevice));
    }
    ~GemmRank() {
        cudaSetDevice(device);cudaStreamSynchronize(stream);tape.reset();
        cublasDestroy(handle);cudaFree(workspace);cudaFree(weights);cudaFree(input);
        cudaFree(output);cudaFree(scratch);cudaStreamDestroy(stream);
    }
    void seed(int generation) {
        check(cudaSetDevice(device));
        std::vector<__nv_bfloat16> x(5*columns);
        for(unsigned i=0;i<x.size();++i)x[i]=__float2bfloat16_rn(float(int(i%7)+generation+device)/8.f);
        check(cudaMemcpy(input,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    }
    void enqueue() {
        q38::cuda_padded_gemm_bf16(handle,weights,input,output,scratch,scratch+8*columns,
                                 rows,columns,5,stream,fixed_workspace?workspace:nullptr,fixed_workspace?workspace_bytes:0);
        q38::te_bridge_allreduce_sum_bf16(device,output,5*rows,0,stream);
    }
    std::vector<std::uint16_t> result() {
        std::vector<std::uint16_t> values(5*rows);
        check(cudaMemcpyAsync(values.data(),output,values.size()*2,cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));return values;
    }
};
void test_gemm_collective_graph() {
    setenv("Q38_TP_PAIRED_GRAPH","1",1);
    q38::te_bridge_reset();q38::te_bridge_use_p2p(true,true);
    GemmRank first(0),second(1);GemmRank* ranks[]={&first,&second};
    // Warm library internals before capture, just as model graph preparation does.
    pair([&](int r){ranks[r]->seed(0);ranks[r]->enqueue();ranks[r]->result();});
    for(auto* rank:ranks) {
        check(cudaSetDevice(rank->device));
        rank->tape=std::make_unique<q38::CudaTeGraphTape>(rank->device,rank->stream,[&]{rank->enqueue();});
    }
    for(int generation=1;generation<=8;++generation) {
        std::array<std::vector<std::uint16_t>,2> expected;
        pair([&](int r){ranks[r]->seed(generation);ranks[r]->enqueue();expected[r]=ranks[r]->result();});
        pair([&](int r){ranks[r]->seed(generation);ranks[r]->tape->launch(ranks[r]->stream);
            if(ranks[r]->result()!=expected[r])throw std::runtime_error("cuBLAS paired graph differs from eager");});
    }
    std::cout<<"cuBLAS M=5 plus paired collective graph passed\n";
}
void profile_collective_graph() {
    for(int mode:{4}) {
        setenv("Q38_TP_PAIRED_GRAPH",mode>=3?"1":"0",1);
        setenv("Q38_TP_GPU_HANDSHAKE",mode==4?"1":"0",1);
        q38::te_bridge_reset();q38::te_bridge_use_p2p(true,true);
        GemmRank first(0,10240,2560,mode>=3),second(1,10240,2560,mode>=3);
        GemmRank* ranks[]={&first,&second};
        pair([&](int r){ranks[r]->seed(1);ranks[r]->enqueue();ranks[r]->result();});
        for(auto* rank:ranks) {
            check(cudaSetDevice(rank->device));
            rank->tape=std::make_unique<q38::CudaTeGraphTape>(rank->device,rank->stream,[&]{
                for(int i=0;i<16;++i)rank->enqueue();
            });
        }
        pair([&](int r){ranks[r]->tape->launch(ranks[r]->stream);ranks[r]->result();});
        const auto start=std::chrono::steady_clock::now();
        pair([&](int r){for(int i=0;i<128;++i)ranks[r]->tape->launch(ranks[r]->stream);ranks[r]->result();});
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"M=5 cuBLAS+collective profile mode="<<mode<<" graphs=128 GEMMs_per_graph=16 milliseconds="<<elapsed<<'\n';
    }
    for(int mode:{4}) {
        setenv("Q38_TP_PAIRED_GRAPH",mode>=3?"1":"0",1);
        setenv("Q38_TP_GPU_HANDSHAKE",mode==4?"1":"0",1);
        q38::te_bridge_reset();q38::te_bridge_use_p2p(true,true);
        Rank first(0),second(1);Rank* ranks[]={&first,&second};
        for(auto* rank:ranks) {
            check(cudaSetDevice(rank->device));
            rank->tape=std::make_unique<q38::CudaTeGraphTape>(rank->device,rank->stream,[&]{
                for(int i=0;i<16;++i)rank->enqueue(0);
            });
        }
        pair([&](int r){ranks[r]->tape->launch(ranks[r]->stream);ranks[r]->verify(0);});
        const auto start=std::chrono::steady_clock::now();
        pair([&](int r){
            for(int i=0;i<128;++i)ranks[r]->tape->launch(ranks[r]->stream);
            ranks[r]->verify(0);
        });
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"collective profile mode="<<mode<<" graphs=128 collectives_per_graph=48 milliseconds="<<elapsed<<'\n';
    }
    for(int mode:{4}) {
        setenv("Q38_TP_PAIRED_GRAPH",mode>=3?"1":"0",1);
        setenv("Q38_TP_GPU_HANDSHAKE",mode==4?"1":"0",1);
        q38::te_bridge_reset();q38::te_bridge_use_p2p(true,true);
        Rank first(0),second(1);Rank* ranks[]={&first,&second};
        for(auto* rank:ranks) {
            check(cudaSetDevice(rank->device));
            check(cudaFree(rank->values));check(cudaFree(rank->half));check(cudaFree(rank->full));
            check(cudaMalloc(&rank->values,102400*4));check(cudaMalloc(&rank->half,12800*2));check(cudaMalloc(&rank->full,25600*2));
            check(cudaMemset(rank->values,0,102400*4));check(cudaMemset(rank->half,0,12800*2));check(cudaMemset(rank->full,0,25600*2));
            rank->tape=std::make_unique<q38::CudaTeGraphTape>(rank->device,rank->stream,[&]{
                for(int i=0;i<16;++i) {
                    q38::te_bridge_allreduce_sum_f32(rank->device,rank->values,12800,0,rank->stream);
                    q38::te_bridge_allgather_bf16(rank->device,rank->half,rank->full,2560,5,0,rank->stream);
                    q38::te_bridge_allreduce_sum_bf16(rank->device,rank->full,12800,0,rank->stream);
                }
            });
        }
        pair([&](int r){ranks[r]->tape->launch(ranks[r]->stream);check(cudaStreamSynchronize(ranks[r]->stream));});
        const auto start=std::chrono::steady_clock::now();
        pair([&](int r){for(int i=0;i<128;++i)ranks[r]->tape->launch(ranks[r]->stream);check(cudaStreamSynchronize(ranks[r]->stream));});
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"M=5 collective profile mode="<<mode<<" graphs=128 collectives_per_graph=48 milliseconds="<<elapsed<<'\n';
    }
}
}
int main() {
 try {
    int devices=0;check(cudaGetDeviceCount(&devices));if(devices<2)throw std::runtime_error("two CUDA devices required");
    for (bool bf16_partial : {false,true}) for (const auto mode : {0,1,2,3,4}) {
        setenv("Q38_TP_PAIRED_GRAPH", mode >= 3 ? "1" : "0", 1);
        setenv("Q38_TP_GPU_HANDSHAKE", mode == 4 ? "1" : "0", 1);
        q38::te_bridge_reset();q38::te_bridge_use_p2p(mode!=0,mode>=2);
        for(int group:{0,1}) {
            Rank first(0,bf16_partial),second(1,bf16_partial);Rank* ranks[]={&first,&second};
            const auto before=q38::te_bridge_stats();
            // Sequential construction would deadlock if it rendezvoused with a peer.
            for(auto* rank:ranks) {
                check(cudaSetDevice(rank->device));
                rank->tape=std::make_unique<q38::CudaTeGraphTape>(rank->device,rank->stream,[&]{rank->enqueue(group);});
                if(rank->tape->graphs()!=(mode>=3?(bf16_partial?4u:3u):(bf16_partial?3u:2u)) ||
                   rank->tape->collectives()!=3 || rank->tape->nodes()!=(bf16_partial?5u:3u))
                    throw std::runtime_error("TE tape segment census differs");
                std::array<float,17> untouched;
                check(cudaMemcpy(untouched.data(),rank->values,sizeof(untouched),cudaMemcpyDeviceToHost));
                for(auto value:untouched)if(value!=0.f)throw std::runtime_error("capture advanced GPU state");
            }
            const auto after=q38::te_bridge_stats();
            if(before.allreduce_calls!=after.allreduce_calls || before.allgather_calls!=after.allgather_calls)
                throw std::runtime_error("capture executed a collective");
            for(int iteration=0;iteration<128;++iteration) {
                const int seed=iteration%23;
                pair([&](int r){auto& rank=*ranks[r];check(cudaSetDevice(r));
                    check(cudaMemcpyAsync(rank.seed,&seed,4,cudaMemcpyHostToDevice,rank.stream));
                    rank.tape->launch(rank.stream);rank.verify(seed);});
            }
            const auto replayed=q38::te_bridge_stats();
            if (mode>=3 && (replayed.paired_graph_calls-after.paired_graph_calls!=128 ||
                replayed.allreduce_calls!=after.allreduce_calls ||
                replayed.allgather_calls!=after.allgather_calls))
                throw std::runtime_error("paired tape replay retained per-collective host calls");
            // No host output wait between generations: the next replay must
            // consume its new inputs and wait for both prior readers/consumers.
            std::array<int,32> queued;
            for(int i=0;i<32;++i) queued[i]=i%19;
            for(int i=0;i<32;++i) pair([&](int r){auto& rank=*ranks[r];check(cudaSetDevice(r));
                check(cudaMemcpyAsync(rank.seed,&queued[i],4,cudaMemcpyHostToDevice,rank.stream));
                rank.tape->launch(rank.stream);});
            pair([&](int r){check(cudaSetDevice(r));ranks[r]->verify(queued.back());});
            // A failed capture must leave no active recorder or partial GPU work.
            for(auto* rank:ranks) {
                bool failed=false;
                try { q38::CudaTeGraphTape bad(rank->device,rank->stream,[&]{rank->enqueue(group);throw std::runtime_error("injected capture failure");}); }
                catch(const std::runtime_error&){failed=true;}
                if(!failed)throw std::runtime_error("capture failure was swallowed");
            }
            pair([&](int r){check(cudaSetDevice(r));ranks[r]->enqueue(group);ranks[r]->verify(queued.back());});
            if (mode >= 3) {
                // A mismatched peer must wake both ranks, not launch a partial
                // graph. After explicit drain/reset a replacement tape works.
                check(cudaSetDevice(1));
                second.tape = std::make_unique<q38::CudaTeGraphTape>(1, second.stream, [&] {
                    q38::te_bridge_allreduce_sum_f32(1, second.values, 16, group, second.stream);
                });
                bool mismatch = false;
                try { pair([&](int r){ranks[r]->tape->launch(ranks[r]->stream);}); }
                catch (const std::exception&) { mismatch = true; }
                if (!mismatch) throw std::runtime_error("paired graph mismatch was accepted");
                for (int r=0;r<2;++r) {check(cudaSetDevice(r));check(cudaStreamSynchronize(ranks[r]->stream));}
                q38::te_bridge_reset();
                second.tape = std::make_unique<q38::CudaTeGraphTape>(1, second.stream, [&]{second.enqueue(group);});
                pair([&](int r){ranks[r]->tape->launch(ranks[r]->stream);check(cudaSetDevice(r));ranks[r]->verify(queued.back());});
            }
            std::cout<<"TE graph tape mode="<<mode<<" group="<<group<<" bf16_partial="<<bf16_partial<<" passed\n";
        }
    }
    for(const char* mode:{"0","1"}) {
        setenv("Q38_TP_GPU_HANDSHAKE",mode,1);
        test_gemm_collective_graph();
    }
    if(std::getenv("Q38_TE_GRAPH_BENCH"))profile_collective_graph();
    q38::te_bridge_reset();
    unsetenv("Q38_TP_PAIRED_GRAPH");
    return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
