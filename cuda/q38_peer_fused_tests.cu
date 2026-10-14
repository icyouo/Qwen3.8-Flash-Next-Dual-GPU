#include "q38/cuda_te.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <stdexcept>
#include <exception>

void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__host__ __device__ float value(unsigned i,int rank,int generation) {
    return float(int((i*17u+unsigned(rank)*43u+unsigned(generation)*3u)%211u)-105)/32.f;
}
__global__ void producer(void* input,unsigned count,int rank,bool fp32,int* counter,bool delay) {
    __shared__ int generation;
    if(!threadIdx.x) {generation=++*counter; if(delay && generation%13==0) {
        const auto start=clock64(); while(clock64()-start<200000ull) {} }}
    __syncthreads();
    for(unsigned i=threadIdx.x;i<count;i+=blockDim.x) {
        if(fp32) static_cast<float*>(input)[i]=value(i,rank,generation);
        else static_cast<std::uint16_t*>(input)[i]=__bfloat16_as_ushort(__float2bfloat16_rn(value(i,rank,generation)));
    }
}
__global__ void validate(const void* output,unsigned count,unsigned elements,bool fp32,bool gather,int* counter,int* failures) {
    const int generation=*counter;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        bool mismatch;
        if(fp32) mismatch=static_cast<const float*>(output)[i]!=value(i,0,generation)+value(i,1,generation);
        else {
            const auto row=i/(2*elements),col=i%(2*elements);
            const float expected=gather?value(row*elements+col%elements,col/elements,generation):
                __bfloat162float(__float2bfloat16_rn(value(i,0,generation)))+
                __bfloat162float(__float2bfloat16_rn(value(i,1,generation)));
            mismatch=static_cast<const std::uint16_t*>(output)[i]!=__bfloat16_as_ushort(__float2bfloat16_rn(expected));
        }
        if(mismatch)atomicAdd(failures,1);
    }
}
struct Rank {
    void *base{},*result{},*scratch{},*flags{};
    int *counter{},*failures{}; cudaStream_t stream{}; cudaGraph_t graph{}; cudaGraphExec_t exec{};
};
struct Result {std::vector<unsigned char> payload[2]; double milliseconds{};};
Result run(bool fused,q38::TeCollectiveCall::Kind kind,unsigned elements,unsigned tokens,
           bool inplace,bool unaligned,bool delay,int repetitions) {
    setenv("Q38_TP_PEER_FUSED",fused?"1":"0",1);
    bool fp32=kind==q38::TeCollectiveCall::Kind::kSumF32;
    bool gather=kind==q38::TeCollectiveCall::Kind::kGatherBf16;
    const unsigned unit=fp32?4:2, n=elements*tokens, total=n*(gather?2:1), offset=unaligned?unit:0;
    Rank ranks[2]; void* graphs[2],*scratch[2];
    q38::TeCollectiveCall calls[2];
    constexpr int steps=16;
    for(int r=0;r<2;++r) {
        auto& a=ranks[r];check(cudaSetDevice(r));check(cudaStreamCreateWithFlags(&a.stream,cudaStreamNonBlocking));
        check(cudaMalloc(&a.base,(total+2)*unit));check(cudaMalloc(&a.result,(total+2)*unit));
        check(cudaMalloc(&a.scratch,(total+2)*unit));check(cudaMalloc(&a.flags,steps*2*sizeof(unsigned)));
        check(cudaMalloc(&a.counter,sizeof(int)));check(cudaMalloc(&a.failures,sizeof(int)));check(cudaMemset(a.failures,0,sizeof(int)));check(cudaMemset(a.counter,0,sizeof(int)));
        check(cudaMemset(a.base,0x55,(total+2)*unit));check(cudaMemset(a.result,0x55,(total+2)*unit));
        check(cudaMemset(a.scratch,0x55,(total+2)*unit));check(cudaMemset(a.flags,0,steps*2*sizeof(unsigned)));
        check(cudaGraphCreate(&a.graph,0));graphs[r]=a.graph;scratch[r]=static_cast<unsigned char*>(a.scratch)+offset;
        calls[r]={kind,r,static_cast<unsigned char*>(a.base)+offset,
            inplace?static_cast<unsigned char*>(a.base)+offset:static_cast<unsigned char*>(a.result)+offset,
            elements,tokens,0,a.stream};
        int can=0;check(cudaDeviceCanAccessPeer(&can,r,1-r));if(!can)throw std::runtime_error("no CUDA P2P");
        auto e=cudaDeviceEnablePeerAccess(1-r,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(e);
    }
    void* tails[2]={};
    for(int step=0;step<steps;++step) {
        void* flags[2];
        for(int r=0;r<2;++r) {
            check(cudaSetDevice(r));auto& a=ranks[r];cudaKernelNodeParams p{};
            auto input=calls[r].input;auto count=n;auto rank=r;auto counter=a.counter;auto waits=delay && r==1;
            void* args[]={&input,&count,&rank,&fp32,&counter,&waits};
            p.func=reinterpret_cast<void*>(producer);p.kernelParams=args;p.gridDim=dim3(1);p.blockDim=dim3(256);
            cudaGraphNode_t node;auto tail=static_cast<cudaGraphNode_t>(tails[r]);
            check(cudaGraphAddKernelNode(&node,a.graph,tail?&tail:nullptr,tail?1:0,&p));tails[r]=node;
            flags[r]=static_cast<unsigned*>(a.flags)+step*2;
        }
        void* done[2];q38::te_graph_add_device_collective(graphs,calls,scratch,flags,tails,done);
        for(int r=0;r<2;++r) {
            check(cudaSetDevice(r));auto& a=ranks[r];cudaKernelNodeParams p{};
            auto output=gather?calls[r].output:calls[r].input;auto count=total;auto width=elements;
            auto counter=a.counter;auto failures=a.failures;
            void* args[]={&output,&count,&width,&fp32,&gather,&counter,&failures};
            p.func=reinterpret_cast<void*>(validate);p.kernelParams=args;p.gridDim=dim3(32);p.blockDim=dim3(256);
            cudaGraphNode_t node;auto tail=static_cast<cudaGraphNode_t>(done[r]);
            check(cudaGraphAddKernelNode(&node,a.graph,&tail,1,&p));tails[r]=node;
        }
    }
    for(int r=0;r<2;++r) {check(cudaSetDevice(r));check(cudaGraphInstantiate(&ranks[r].exec,ranks[r].graph,0));}
    std::exception_ptr errors[2];
    const auto start=std::chrono::steady_clock::now();
    std::thread workers[2];
    for(int r=0;r<2;++r)workers[r]=std::thread([&,r]{try {
        check(cudaSetDevice(r));for(int i=0;i<repetitions;++i)check(cudaGraphLaunch(ranks[r].exec,ranks[r].stream));
        check(cudaStreamSynchronize(ranks[r].stream));
    }catch(...){errors[r]=std::current_exception();}});
    for(auto& t:workers)t.join();for(auto e:errors)if(e)std::rethrow_exception(e);
    Result result;result.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    for(int r=0;r<2;++r) {
        check(cudaSetDevice(r));auto& a=ranks[r];auto out=gather&&!inplace?a.result:a.base;
        result.payload[r].resize((total+2)*unit);check(cudaMemcpy(result.payload[r].data(),out,result.payload[r].size(),cudaMemcpyDeviceToHost));
        const auto* bytes=result.payload[r].data();
        for(unsigned i=0;i<offset;++i)if(bytes[i]!=0x55)throw std::runtime_error("prefix guard overwritten");
        for(unsigned i=offset+total*unit;i<(total+2)*unit;++i)if(bytes[i]!=0x55)throw std::runtime_error("output guard overwritten");
        int errors=0;check(cudaMemcpy(&errors,a.failures,sizeof(int),cudaMemcpyDeviceToHost));if(errors)throw std::runtime_error("queued intermediate collective mismatch");
        int generation=0;check(cudaMemcpy(&generation,a.counter,sizeof(int),cudaMemcpyDeviceToHost));
        if(generation!=steps*repetitions)throw std::runtime_error("producer generation mismatch");
        for(unsigned i=0;i<total;++i) {
            if(fp32) {float got;std::memcpy(&got,bytes+offset+i*unit,unit);
                const float expected=value(i,0,generation)+value(i,1,generation);
                if(got!=expected)throw std::runtime_error("F32 collective differs");
            }else {
                std::uint16_t got;std::memcpy(&got,bytes+offset+i*unit,unit);
                const auto row=i/(2*elements),col=i%(2*elements);
                const float expected=gather?value(row*elements+col%elements,col/elements,generation):
                    __bfloat162float(__float2bfloat16_rn(value(i,0,generation)))+
                    __bfloat162float(__float2bfloat16_rn(value(i,1,generation)));
                if(got!=__bfloat16_as_ushort(__float2bfloat16_rn(expected)))throw std::runtime_error("BF16 sum/gather differs");
            }
        }
        std::vector<unsigned char> temporary((total+2)*unit);check(cudaMemcpy(temporary.data(),a.scratch,temporary.size(),cudaMemcpyDeviceToHost));
        for(unsigned i=offset+total*unit;i<(total+2)*unit;++i)if(temporary[i]!=0x55)throw std::runtime_error("scratch guard overwritten");
        check(cudaGraphExecDestroy(a.exec));check(cudaGraphDestroy(a.graph));check(cudaFree(a.base));check(cudaFree(a.result));
        check(cudaFree(a.scratch));check(cudaFree(a.flags));check(cudaFree(a.counter));check(cudaFree(a.failures));check(cudaStreamDestroy(a.stream));
    }
    return result;
}
int main() {try {
    using Kind=q38::TeCollectiveCall::Kind;
    for(auto kind:{Kind::kSumF32,Kind::kSumBf16,Kind::kGatherBf16})
    for(auto elements:{1u,17u,160u,2560u,10240u,32771u})
    for(auto tokens:{1u,5u,7u}) {
        // Include odd and misaligned tails, in-place gathers and unequal
        // producers; queue generations without a host synchronization.
        for(bool inplace:{false,true}) {
            if(kind!=Kind::kGatherBf16 && !inplace)continue;
            const bool unaligned=elements==17,delay=elements<=160;
            const auto old=run(false,kind,elements,tokens,inplace,unaligned,delay,8);
            const auto next=run(true,kind,elements,tokens,inplace,unaligned,delay,8);
            for(int r=0;r<2;++r)if(old.payload[r]!=next.payload[r])throw std::runtime_error("fused collective bit parity");
        }
    }
    for(auto kind:{Kind::kSumF32,Kind::kSumBf16,Kind::kGatherBf16})
    for(auto elements:{160u,2560u,10240u}) {
        const auto old=run(false,kind,elements,5,true,false,false,128);
        const auto next=run(true,kind,elements,5,true,false,false,128);
        std::cout<<"{\"kind\":"<<int(kind)<<",\"elements\":"<<elements<<",\"tokens\":5,\"old_ms\":"<<old.milliseconds
            <<",\"fused_ms\":"<<next.milliseconds<<"}\n";
    }
    std::cout<<"PEER_FUSED_PASS: rank-ordered sums, token-major gathers, RNE, in-place buffers, unaligned tails, delayed producers, queued generations, bounded fallback\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
