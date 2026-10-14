#include "q38/cuda_moe.h"
#include "q38/cuda_moe_prefill.h"
#include "q38/cuda_hyper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
std::uint16_t bf16(float value) {
    const auto b = __float2bfloat16_rn(value);
    std::uint16_t result;
    std::memcpy(&result, &b, sizeof(result));
    return result;
}
struct Buffers {
    std::vector<void*> pointers;
    template <typename T> T* allocate(std::size_t count) {
        T* p = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "allocate");
        pointers.push_back(p);
        return p;
    }
    template <typename T> T* upload(const std::vector<T>& values) {
        auto* p = allocate<T>(values.size());
        check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
        return p;
    }
    ~Buffers() { for (void* p : pointers) (void)cudaFree(p); }
};
template <typename T> std::vector<T> read(const T* p, std::size_t n, cudaStream_t stream) {
    std::vector<T> values(n);
    check(cudaMemcpyAsync(values.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, stream), "read");
    check(cudaStreamSynchronize(stream), "read ready");
    return values;
}
template <typename T> void equal(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)))
        throw std::runtime_error(std::string("bit parity failed: ") + label);
}

void fixture(int bits, bool tiled, cudaStream_t stream) {
    Buffers memory;
    constexpr unsigned experts = q38::kQ38MoeTopK;
    constexpr unsigned rows = 2 * q38::kQ38MoeIntermediate;
    constexpr unsigned columns = q38::kQ38HiddenWidth;
    constexpr unsigned middle = q38::kQ38MoeIntermediate;
    const auto weights = [&](unsigned n, unsigned k, unsigned salt) {
        const std::size_t elements = std::size_t(experts) * n * k;
        std::vector<std::uint8_t> data(elements * bits / 8);
        for (std::size_t i = 0; i < elements; ++i) {
            const int q = int((i * 17 + i / k * 3 + salt) % (bits == 8 ? 255 : 16)) - (bits == 8 ? 127 : 8);
            if (bits == 8) data[i] = static_cast<std::uint8_t>(q);
            else data[i / 2] |= (unsigned(q) & 15u) << ((i & 1u) * 4);
        }
        if (tiled) {
            std::vector<std::uint8_t> repacked(data.size());
            for (unsigned e = 0; e < experts; ++e)
                for (unsigned r = 0; r < n; ++r)
                    for (unsigned c = 0; c < k; c += 2) {
                        const auto target = std::size_t(e) * n * k / 2 +
                            (std::size_t(r / 16) * (k / 16) + c / 16) * 128 +
                            (r % 16) * 8 + (c % 16) / 2;
                        repacked[target] = data[(std::size_t(e) * n * k + std::size_t(r) * k + c) / 2];
                    }
            data.swap(repacked);
        }
        return data;
    };
    const auto scales = [&](unsigned n, unsigned k, float base) {
        std::vector<std::uint16_t> data(std::size_t(experts) * n * (k / 128));
        for (std::size_t i = 0; i < data.size(); ++i)
            data[i] = bf16(base * (1.0f + float(i % 11) * .125f));
        return data;
    };
    q38::DeviceTensorV1 gate_desc, down_desc;
    gate_desc.format = down_desc.format = bits == 8 ? q38::DeviceWeightFormatV1::kW8A16SymG128 : q38::DeviceWeightFormatV1::kW4A16SymG128;
    gate_desc.group_size = down_desc.group_size = 128;
    gate_desc.shape = {q38::kQ38MoeExperts / 2, rows, columns};
    down_desc.shape = {q38::kQ38MoeExperts / 2, columns, middle};
    q38::CudaTensorViewV1 gate{&gate_desc, memory.upload(weights(rows, columns, 3)), memory.upload(scales(rows, columns, bits == 8 ? .0004f : .006f))};
    q38::CudaTensorViewV1 down{&down_desc, memory.upload(weights(columns, middle, 11)), memory.upload(scales(columns, middle, bits == 8 ? .0006f : .008f))};
    if (tiled) gate.layout = down.layout = q38::CudaWeightLayoutV1::kMoeW4Tile16x16;
    constexpr unsigned routes=50;
    auto* hidden=memory.allocate<std::uint16_t>(routes*columns);
    auto* assignment=memory.allocate<std::uint32_t>(routes);
    auto* route_weights=memory.allocate<float>(routes);
    auto* tasks=memory.allocate<q38::Q38ExpertMmqTaskV1>(routes);
    auto* mid=memory.allocate<std::uint16_t>(routes*middle);
    auto* route=memory.allocate<float>(routes*columns);
    std::vector<std::uint16_t> input(routes*columns);
    std::vector<std::uint32_t> assignments(routes);
    std::vector<float> routing(routes);
    std::vector<q38::Q38ExpertMmqTaskV1> plan(routes);
    auto update=[&](unsigned count,unsigned pattern){
        for(unsigned i=0;i<input.size();++i)input[i]=bf16(std::sin(float(i*13+pattern*31)*.017f)*.2f);
        for(unsigned i=0;i<routes;++i){assignments[i]=i;routing[i]=.01f+float((i*3+pattern)%7)*.025f;}
        unsigned cursor=0;
        for(unsigned i=0;i<routes;++i){
            unsigned rows=count==1?1:count==5?((i%3)+1):16;
            rows=std::min(rows,routes-cursor);
            plan[i]={cursor,std::uint16_t(i%experts),std::uint8_t(rows),0};cursor+=rows;
        }
        if(pattern==2)for(unsigned i=0;i<routes;i+=3)plan[i].valid_rows=0;
        if(pattern==3)for(auto& t:plan)t.valid_rows=0;
        check(cudaMemcpyAsync(hidden,input.data(),input.size()*2,cudaMemcpyHostToDevice,stream),"input");
        check(cudaMemcpyAsync(assignment,assignments.data(),routes*4,cudaMemcpyHostToDevice,stream),"assignment");
        check(cudaMemcpyAsync(route_weights,routing.data(),routes*4,cudaMemcpyHostToDevice,stream),"routing");
        check(cudaMemcpyAsync(tasks,plan.data(),plan.size()*sizeof(plan[0]),cudaMemcpyHostToDevice,stream),"tasks");
        check(cudaMemsetAsync(mid,0,routes*middle*2,stream),"init mid");
        check(cudaMemsetAsync(route,0,routes*columns*4,stream),"init routes");
        check(cudaStreamSynchronize(stream),"inputs ready");
    };
    auto select=[](bool value){setenv("Q38_CUDA_MOE_REG_MMA",value?"1":"0",1);};
    auto launch=[&](bool weights){
        q38::cuda_moe_grouped_gate_up_v1(gate,hidden,assignment,route_weights,tasks,routes,mid,
            q38::Q38PrefillMoeModeV1::kGroupedFp32Scales,stream,0,weights);
        q38::cuda_moe_grouped_down_v1(down,mid,tasks,routes,route,
            q38::Q38PrefillMoeModeV1::kGroupedFp32Scales,stream,0);
    };
    unsigned cases=0;
    for(unsigned count:{1u,5u,16u}){
        for(unsigned pattern=0;pattern<4;++pattern)for(bool weighted:{false,true}){
            update(count,pattern);select(false);launch(weighted);
            const auto m=read(mid,routes*middle,stream);const auto r=read(route,routes*columns,stream);
            select(true);launch(weighted);
            equal(m,read(mid,routes*middle,stream),"activated");
            equal(r,read(route,routes*columns,stream),"FP32 down");++cases;
        }
        update(count,0);
        cudaGraph_t graphs[2]{};cudaGraphExec_t runs[2]{};
        for(unsigned v=0;v<2;++v){select(v);check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture");
            for(unsigned i=0;i<20;++i)launch(false);
            check(cudaStreamEndCapture(stream,&graphs[v]),"end capture");check(cudaGraphInstantiate(&runs[v],graphs[v],0),"instantiate");}
        for(unsigned pattern=0;pattern<4;++pattern){update(count,pattern);
            check(cudaGraphLaunch(runs[0],stream),"base replay");const auto m=read(mid,routes*middle,stream);const auto r=read(route,routes*columns,stream);
            check(cudaGraphLaunch(runs[1],stream),"reg replay");equal(m,read(mid,routes*middle,stream),"graph activated");equal(r,read(route,routes*columns,stream),"graph down");++cases;}
        update(count,0);float times[2];cudaEvent_t a,b;check(cudaEventCreate(&a),"event");check(cudaEventCreate(&b),"event");
        for(unsigned v=0;v<2;++v){for(unsigned i=0;i<5;++i)check(cudaGraphLaunch(runs[v],stream),"warmup");
            check(cudaEventRecord(a,stream),"start");for(unsigned i=0;i<20;++i)check(cudaGraphLaunch(runs[v],stream),"timed");check(cudaEventRecord(b,stream),"stop");check(cudaEventSynchronize(b),"ready");check(cudaEventElapsedTime(&times[v],a,b),"elapsed");times[v]/=400.f;}
        std::cout<<"{\"bits\":"<<bits<<",\"tiled\":"<<tiled<<",\"rows_per_task\":"<<count<<",\"baseline_ms\":"<<times[0]<<",\"candidate_ms\":"<<times[1]<<",\"speedup\":"<<times[0]/times[1]<<"}\n";
        cudaEventDestroy(a);cudaEventDestroy(b);for(unsigned v=0;v<2;++v){cudaGraphExecDestroy(runs[v]);cudaGraphDestroy(graphs[v]);}
    }
    std::cout<<"PARITY_CASES "<<bits<<" "<<tiled<<" "<<cases<<"\n";
}
}
int main(){try{cudaStream_t s;check(cudaSetDevice(0),"device");check(cudaStreamCreate(&s),"stream");fixture(4,false,s);fixture(4,true,s);fixture(8,false,s);check(cudaStreamDestroy(s),"destroy");std::cout<<"REG_MMA_PASS\n";}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
