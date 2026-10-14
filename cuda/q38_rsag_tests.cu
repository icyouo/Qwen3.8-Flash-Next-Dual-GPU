#include "q38/cuda_te.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class F> void ranks(F fn) {
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int rank=0; rank<2; ++rank) workers[rank]=std::thread([&,rank] {
        try { check(cudaSetDevice(rank)); fn(rank); }
        catch (...) { q38::te_bridge_abort(); errors[rank]=std::current_exception(); }
    });
    for (auto& worker:workers) worker.join();
    for (auto error:errors) if(error) std::rethrow_exception(error);
}
__global__ void produce(float* output, unsigned count, int rank, unsigned generation, bool special, bool delay) {
    if (delay) {
        const auto start=clock64();
        while(clock64()-start<1000000) {}
    }
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        unsigned random = i*747796405u + rank*2891336453u + generation*277803737u;
        random = ((random >> ((random >> 28u)+4u)) ^ random)*277803737u;
        random = (random >> 22u) ^ random;
        float value=float(int(random%65537u)-32768)*0.00009371321f;
        if (special && generation%3==0 && i%5==0) value=__uint_as_float(random);
        if (special && i%5==1) {
            // Enumerate all omitted-byte values around cancellation and BF16
            // half-way boundaries, alternating endpoint parity and signs.
            const unsigned word=0x3f800000u+((i/256u)%512u)*256u+(i%256u);
            value=rank ? __uint_as_float(word) :
                (i%2 ? -__uint_as_float(word & ~255u) : 0.00390625f);
        }
        if (special && i%257<16 && (generation%3==0 || i<16)) {
            const unsigned a[16]={0x00000000,0x80000000,0x3f800000,0x3f810000,
                0x3f808000,0x3f818000,0x7f800000,0xff800000,0x7fc12345,
                0x7fa12345,0x00000001,0x80000001,0x007fffff,0x00800000,
                0x7f7fffff,0xff7fffff};
            const unsigned b[16]={0x80000000,0x80000000,0x3b800000,0x3b800000,
                0x00000000,0x00000000,0xff800000,0x7f800000,0x3f800000,
                0x7fc22222,0x00000001,0x00000001,0x00000001,0x807fffff,
                0x7f7fffff,0x7f7fffff};
            value=__uint_as_float(rank?b[i%257]:a[i%257]);
        }
        output[i]=value;
    }
}
__global__ void round_bf16(const float* input,std::uint16_t* output,unsigned count) {
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        output[i]=__bfloat16_as_ushort(__float2bfloat16_rn(input[i]));
}
struct Allocation { float* input; float* source; float* control; std::uint16_t* output; std::uint16_t* snapshot; unsigned count; };

std::vector<std::uint16_t> queued(bool candidate,int group) {
    const unsigned sizes[]={1,2,3,127,256,65535,65536,65537,131073,1310721,32771,1310721,65536};
    std::vector<std::uint16_t> payloads[2];
    ranks([&](int rank) {
        cudaStream_t streams[2];
        for(auto& stream:streams) check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        std::vector<Allocation> allocations;
        for(unsigned generation=0;generation<sizeof(sizes)/sizeof(sizes[0]);++generation) {
            Allocation a{};a.count=sizes[generation];
            auto stream=streams[generation%2];
            const auto fbytes=(a.count+2)*sizeof(float),bbytes=(a.count+2)*sizeof(std::uint16_t);
            check(cudaMalloc(&a.input,fbytes));check(cudaMalloc(&a.source,fbytes));
            check(cudaMalloc(&a.control,fbytes));check(cudaMalloc(&a.output,bbytes));check(cudaMalloc(&a.snapshot,bbytes));
            check(cudaMemsetAsync(a.input,0x55,fbytes,stream));
            check(cudaMemsetAsync(a.output,0x55,bbytes,stream));
            produce<<<32,256,0,stream>>>(a.input+1,a.count,rank,generation,true,generation%3==unsigned(rank));
            check(cudaPeekAtLastError());
            check(cudaMemcpyAsync(a.source,a.input,fbytes,cudaMemcpyDeviceToDevice,stream));
            if(candidate) {
                if(generation%3==0) {
                    check(cudaMemcpyAsync(a.control,a.input,fbytes,cudaMemcpyDeviceToDevice,stream));
                    q38::te_bridge_allreduce_sum_f32(rank,a.control+1,a.count,group,stream);
                }
                q38::te_bridge_allreduce_f32_to_bf16(rank,a.input+1,a.output+1,a.count,group,stream);
            } else {
                q38::te_bridge_allreduce_sum_f32(rank,a.input+1,a.count,group,stream);
                round_bf16<<<32,256,0,stream>>>(a.input+1,a.output+1,a.count);
                check(cudaPeekAtLastError());
            }
            int restored=-1;check(cudaGetDevice(&restored));
            if(restored!=rank)throw std::runtime_error("RSAG changed caller device");
            // Save before intentionally overwriting the output on its producer
            // stream: the peer's BF16 gather must already have finished.
            check(cudaMemcpyAsync(a.snapshot,a.output,bbytes,cudaMemcpyDeviceToDevice,stream));
            check(cudaMemsetAsync(a.output,0xaa,bbytes,stream));
            allocations.push_back(a);
        }
        for(auto stream:streams)check(cudaStreamSynchronize(stream));
        for(auto a:allocations) {
            std::vector<std::uint16_t> output(a.count+2);
            check(cudaMemcpy(output.data(),a.snapshot,output.size()*2,cudaMemcpyDeviceToHost));
            if(output.front()!=0x5555||output.back()!=0x5555)throw std::runtime_error("RSAG output guard overwritten");
            payloads[rank].insert(payloads[rank].end(),output.begin()+1,output.end()-1);
            std::vector<unsigned> input(a.count+2),source(a.count+2);
            check(cudaMemcpy(input.data(),a.input,input.size()*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(source.data(),a.source,source.size()*4,cudaMemcpyDeviceToHost));
            if(input.front()!=0x55555555||input.back()!=0x55555555)throw std::runtime_error("RSAG input guard overwritten");
            if(candidate&&std::memcmp(input.data(),source.data(),input.size()*4))throw std::runtime_error("RSAG mutated FP32 input");
            check(cudaFree(a.input));check(cudaFree(a.source));check(cudaFree(a.control));
            check(cudaFree(a.output));check(cudaFree(a.snapshot));
        }
        for(auto stream:streams)check(cudaStreamDestroy(stream));
    });
    if(payloads[0]!=payloads[1])throw std::runtime_error("RSAG rank results differ");
    q38::te_bridge_reset_group(group);
    return payloads[0];
}
double benchmark(bool candidate,int group) {
    constexpr unsigned count=512*2560,rounds=30;
    double elapsed[2]={};
    ranks([&](int rank) {
        float* input;std::uint16_t* output;
        check(cudaMalloc(&input,count*4));check(cudaMalloc(&output,count*2));
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto run=[&](unsigned generation) {
            produce<<<32,256,0,stream>>>(input,count,rank,generation,false,false);
            if(candidate)q38::te_bridge_allreduce_f32_to_bf16(rank,input,output,count,group,stream);
            else {
                q38::te_bridge_allreduce_sum_f32(rank,input,count,group,stream);
                round_bf16<<<32,256,0,stream>>>(input,output,count);
            }
            check(cudaPeekAtLastError());
        };
        for(unsigned i=0;i<3;++i)run(i);
        check(cudaStreamSynchronize(stream));
        auto start=std::chrono::steady_clock::now();
        for(unsigned i=0;i<rounds;++i)run(i);
        check(cudaStreamSynchronize(stream));
        elapsed[rank]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/rounds;
        check(cudaFree(input));check(cudaFree(output));check(cudaStreamDestroy(stream));
    });
    return std::max(elapsed[0],elapsed[1]);
}
}
int main() {
    try {
        setenv("Q38_TP_BULK_COPY","1",1);
        setenv("Q38_TP_RSAG_FP24","1",1);
        for(bool ordered:{false,true}) {
            q38::te_bridge_use_p2p(true,ordered);
            for(int group=0;group<2;++group) {
                const auto oracle=queued(false,group);
                setenv("Q38_TP_RSAG_FP24","0",1);
                const auto plain=queued(true,group);
                setenv("Q38_TP_RSAG_FP24","1",1);
                const auto candidate=queued(true,group);
                if(oracle!=plain)throw std::runtime_error("plain RSAG oracle mismatch");
                if(oracle!=candidate)throw std::runtime_error("RSAG differs from rank-ordered FP32 sum + BF16 RNE oracle");
            }
        }
        q38::te_bridge_use_p2p(true,true);
        for(unsigned trial=0;trial<5;++trial) {
            auto baseline=benchmark(false,0);
            setenv("Q38_TP_RSAG_FP24","0",1);
            auto plain=benchmark(true,0);
            setenv("Q38_TP_RSAG_FP24","1",1);
            auto candidate=benchmark(true,0);
            q38::te_bridge_reset();
            double baseline_groups[2]={},plain_groups[2]={},candidate_groups[2]={};std::exception_ptr error[2];
            auto concurrent=[&](bool mode,double (&times)[2]) {
                std::thread workers[2];
                for(int g=0;g<2;++g)workers[g]=std::thread([&,g] {
                    try{times[g]=benchmark(mode,g);}catch(...){error[g]=std::current_exception();q38::te_bridge_abort();}
                });
                for(auto& w:workers)w.join();
                for(auto e:error)if(e)std::rethrow_exception(e);
                q38::te_bridge_reset();
            };
            concurrent(false,baseline_groups);
            setenv("Q38_TP_RSAG_FP24","0",1);concurrent(true,plain_groups);
            setenv("Q38_TP_RSAG_FP24","1",1);concurrent(true,candidate_groups);
            std::cout<<"{\"trial\":"<<trial<<",\"tokens\":512,\"f32_allreduce_round_us\":"<<baseline
                     <<",\"f32_rsag_bf16_us\":"<<candidate<<",\"two_groups_baseline_us\":"<<std::max(baseline_groups[0],baseline_groups[1])
                     <<",\"plain_rsag_us\":"<<plain<<",\"two_groups_plain_rsag_us\":"<<std::max(plain_groups[0],plain_groups[1])
                     <<",\"two_groups_rsag_us\":"<<std::max(candidate_groups[0],candidate_groups[1])<<"}\n";
        }
        std::cout<<"RSAG_PARITY_PASS\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
