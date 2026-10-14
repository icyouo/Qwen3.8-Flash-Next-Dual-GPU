#include "q38/cuda_kernels.h"
#include "q38/cuda_hyper.h"
#include "q38/cuda_padded_gemm.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iostream>
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


void fixture(unsigned n,unsigned k,unsigned m,q38::CudaHcActivation activation,cudaStream_t s){
 Buffers mem;std::vector<std::uint16_t>w(n*k),x(m*k),normx(m*10240);for(unsigned i=0;i<normx.size();++i)normx[i]=bf16(std::sin(float(i%10789)*.19f)*.73f);for(unsigned i=0;i<w.size();++i)w[i]=bf16(std::sin(float(i%10007)*.137f)*.014f);for(unsigned i=0;i<x.size();++i)x[i]=bf16(std::cos(float(i)*.173f)*.31f);
 auto* dn=mem.upload(normx);auto* mix=mem.allocate<std::uint16_t>(m*2560);auto*dw=mem.upload(w),*dx=mem.upload(x),*dy=mem.allocate<std::uint16_t>(m*n),*pad=mem.allocate<std::uint16_t>(8*(k+n));auto*partial=mem.allocate<float>(16*7*10240);auto*workspace=mem.allocate<char>(32<<20);
 q38::CudaMatrixViewV1 mat;mat.format=q38::DeviceWeightFormatV1::kPreserve;mat.data=dw;mat.rows=n;mat.columns=k;
 cublasHandle_t h;cublasCreate(&h);cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH);
 const float divisor=(activation==q38::CudaHcActivation::kSilu||n==4)?4.f:1.f;
 auto launch=[&](bool candidate){if(candidate){if(!q38::cuda_try_hc_scalar_batch_bf16(mat,dx,dy,m,s,0,activation,divisor,n==4?2.f:1.f))throw std::runtime_error("HC dispatch");}
 else for(unsigned row=0;row<m;++row)if(!q38::cuda_try_hc_gemv_bf16(mat,dx+row*k,dy+row*n,s,0,activation,divisor,n==4?2.f:1.f))throw std::runtime_error("HC scalar dispatch");};
 launch(false);auto ref=read(dy,m*n,s);launch(true);auto got=read(dy,m*n,s);double err=0,power=0,maxerr=0;unsigned bits=0;for(unsigned i=0;i<got.size();++i){__nv_bfloat16 a,b;std::memcpy(&a,&got[i],2);std::memcpy(&b,&ref[i],2);double d=__bfloat162float(a)-__bfloat162float(b);err+=d*d;power+=std::pow(__bfloat162float(b),2);maxerr=std::max(maxerr,std::abs(d));bits+=got[i]!=ref[i];}double l2=std::sqrt(err/std::max(power,1e-20));if(bits!=0)throw std::runtime_error("HC tensor numerical mismatch");
 cudaGraph_t graphs[2];cudaGraphExec_t execs[2];for(unsigned v=0;v<2;++v){check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");for(unsigned i=0;i<30;++i)launch(v);check(cudaStreamEndCapture(s,&graphs[v]),"end");check(cudaGraphInstantiate(&execs[v],graphs[v],0),"instantiate");check(cudaGraphLaunch(execs[v],s),"parity replay");equal(v?got:ref,read(dy,m*n,s),"graph projection");}
 cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);float timing[2];for(unsigned v=0;v<2;++v){for(unsigned i=0;i<4;++i)cudaGraphLaunch(execs[v],s);cudaEventRecord(a,s);for(unsigned i=0;i<8;++i)cudaGraphLaunch(execs[v],s);cudaEventRecord(b,s);cudaEventSynchronize(b);cudaEventElapsedTime(&timing[v],a,b);timing[v]/=240.f;}
 std::cout<<"{\"N\":"<<n<<",\"K\":"<<k<<",\"M\":"<<m<<",\"rel_l2\":"<<l2<<",\"max_error\":"<<maxerr<<",\"changed_bf16_words\":"<<bits<<",\"old_ms\":"<<timing[0]<<",\"tensor_ms\":"<<timing[1]<<"}\n";
 for(unsigned v=0;v<2;++v){cudaGraphExecDestroy(execs[v]);cudaGraphDestroy(graphs[v]);}cudaEventDestroy(a);cudaEventDestroy(b);cublasDestroy(h);
}
}
int main(){try{cudaStream_t s;cudaSetDevice(0);cudaStreamCreate(&s);for(unsigned m:{2u,3u,5u,7u}){fixture(320,10240,m,q38::CudaHcActivation::kSilu,s);fixture(10240,320,m,q38::CudaHcActivation::kSigmoid,s);}cudaStreamDestroy(s);std::cout<<"HC_SCALAR_BATCH_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
