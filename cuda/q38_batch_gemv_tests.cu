#include "q38/cuda_kernels.h"
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

void fixture(int bits,unsigned n,unsigned k,cudaStream_t stream){
 Buffers mem;std::vector<std::uint8_t> bytes(std::size_t(n)*k*bits/8);
 for(std::size_t i=0;i<bytes.size();++i)bytes[i]=std::uint8_t(i*97+i/k*13);
 std::vector<std::uint16_t> scales(std::size_t(n)*k/128),input(7*k);
 for(unsigned i=0;i<scales.size();++i)scales[i]=bf16(.002f+float(i%17)*.000125f);
 for(unsigned i=0;i<input.size();++i)input[i]=bf16(std::sin(float(i*13)*.013f));
 q38::CudaMatrixViewV1 w;w.format=bits==4?q38::DeviceWeightFormatV1::kW4A16SymG128:q38::DeviceWeightFormatV1::kW8A16SymG128;
 w.rows=n;w.columns=k;w.group_size=128;w.data=mem.upload(bytes);w.scales=mem.upload(scales);
 auto* x=mem.upload(input);auto* y=mem.allocate<std::uint16_t>(7*n);
 for(unsigned m:{2u,3u,5u,7u}){
  auto launch=[&](bool batch){if(batch){if(!q38::cuda_small_batch_gemv_bf16(w,x,y,m,stream,0))throw std::runtime_error("dispatch");}
    else for(unsigned t=0;t<m;++t)q38::cuda_gemv_bf16(w,x+t*k,y+t*n,1,stream,0);};
  launch(false);const auto expected=read(y,m*n,stream);launch(true);equal(expected,read(y,m*n,stream),"eager batch logits");
  cudaGraph_t g[2]{};cudaGraphExec_t e[2]{};
  for(unsigned v=0;v<2;++v){check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture");for(unsigned i=0;i<10;++i)launch(v);check(cudaStreamEndCapture(stream,&g[v]),"end");check(cudaGraphInstantiate(&e[v],g[v],0),"instantiate");}
  for(unsigned v=0;v<2;++v){check(cudaGraphLaunch(e[v],stream),"replay");equal(expected,read(y,m*n,stream),"graph batch logits");}
  cudaEvent_t a,b;check(cudaEventCreate(&a),"event");check(cudaEventCreate(&b),"event");float timing[2];
  for(unsigned v=0;v<2;++v){for(unsigned i=0;i<5;++i)check(cudaGraphLaunch(e[v],stream),"warm");check(cudaEventRecord(a,stream),"start");for(unsigned i=0;i<10;++i)check(cudaGraphLaunch(e[v],stream),"timed");check(cudaEventRecord(b,stream),"stop");check(cudaEventSynchronize(b),"ready");check(cudaEventElapsedTime(&timing[v],a,b),"elapsed");timing[v]/=100.f;}
  std::cout<<"{\"bits\":"<<bits<<",\"N\":"<<n<<",\"K\":"<<k<<",\"M\":"<<m<<",\"baseline_ms\":"<<timing[0]<<",\"batch_ms\":"<<timing[1]<<",\"speedup\":"<<timing[0]/timing[1]<<"}\n";
  cudaEventDestroy(a);cudaEventDestroy(b);for(unsigned v=0;v<2;++v){cudaGraphExecDestroy(e[v]);cudaGraphDestroy(g[v]);}
 }
}
}
int main(){try{cudaStream_t s;check(cudaSetDevice(0),"device");check(cudaStreamCreate(&s),"stream");fixture(8,248320,1280,s);fixture(4,1024,2560,s);check(cudaStreamDestroy(s),"destroy");std::cout<<"BATCH_GEMV_PASS\n";}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
