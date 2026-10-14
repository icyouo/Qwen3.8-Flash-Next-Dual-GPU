#include "q38/cuda_kernels.h"
#include "q38/cuda_padded_gemm.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
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


template<typename T>std::vector<T>file(const char* name){std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(name);auto size=f.tellg();std::vector<T> out(std::size_t(size)/sizeof(T));f.seekg(0);f.read((char*)out.data(),size);return out;}
void fixture(cudaStream_t s){
 Buffers mem; std::vector<q38::CudaMatrixViewV1> mats(5);std::vector<std::uint16_t*> weights,outputs;
 const char* names[]={"qkv","z","in_b","in_a","packed"}; unsigned sizes[]={10240,6144,48,48,8240};
 for(unsigned i=0;i<5;++i){auto&v=mats[i];v.format=q38::DeviceWeightFormatV1::kW8A16SymG128;v.group_size=128;v.rows=sizes[i];v.columns=2560;v.data=mem.upload(file<char>((std::string(names[i])+".q").c_str()));v.scales=mem.upload(file<std::uint16_t>((std::string(names[i])+".scale").c_str()));auto*w=mem.allocate<std::uint16_t>(sizes[i]*2560);weights.push_back(w);outputs.push_back(mem.allocate<std::uint16_t>(7*sizes[i]));q38::cuda_dequantize_matrix_bf16(v,w,s,0);}
 auto*pad=mem.allocate<std::uint16_t>(8*(10240+2560));auto*work=mem.allocate<char>(32<<20);cublasHandle_t h;cublasCreate(&h);cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH);
 std::vector<std::uint16_t> hostx(7*2560);for(unsigned i=0;i<hostx.size();++i)hostx[i]=bf16(std::sin(float(i%13337)*.117f)*.4f);auto*x=mem.upload(hostx);
 for(unsigned m:{1u,2u,3u,5u,7u}) {
 auto one=[&](unsigned i){if(m==1)q38::cuda_gemv_bf16(mats[i],x,outputs[i],1,s,0);else q38::cuda_padded_gemm_bf16(h,weights[i],x,outputs[i],pad,pad+8*2560,sizes[i],2560,m,s,work,32<<20);};
 for(unsigned i=0;i<5;++i)one(i);
 std::vector<std::vector<std::uint16_t>> ys;for(unsigned i=0;i<5;++i)ys.push_back(read(outputs[i],m*sizes[i],s));unsigned changed=0;float maxerr=0;double err=0,power=0;
 for(unsigned t=0;t<m;++t)for(unsigned i=0;i<8240;++i){unsigned part=i<5120?0:i<8192?1:i<8216?2:3;unsigned row=part==0?(i<1024?i:i<2048?2048+i-1024:4096+i-2048):i-(part==1?5120:part==2?8192:8216);auto a=ys[part][t*sizes[part]+row],b=ys[4][t*8240+i];changed+=a!=b;__nv_bfloat16 av,bv;std::memcpy(&av,&a,2);std::memcpy(&bv,&b,2);float x=__bfloat162float(av),y=__bfloat162float(bv);maxerr=std::max(maxerr,std::abs(x-y));err+=double(x-y)*(x-y);power+=double(x)*x;}
 float us[2];for(unsigned mode=0;mode<2;++mode){cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);for(unsigned j=0;j<48;++j){if(mode)one(4);else for(unsigned i=0;i<4;++i)one(i);}cudaGraph_t g;cudaStreamEndCapture(s,&g);cudaGraphExec_t e;check(cudaGraphInstantiate(&e,g,nullptr,nullptr,0),"instantiate");cudaGraphLaunch(e,s);cudaStreamSynchronize(s);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a,s);for(unsigned j=0;j<100;++j)cudaGraphLaunch(e,s);cudaEventRecord(b,s);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);us[mode]=ms*1000/100/48;cudaGraphExecDestroy(e);cudaGraphDestroy(g);cudaEventDestroy(a);cudaEventDestroy(b);}
 std::cout<<"{\"M\":"<<m<<",\"changed\":"<<changed<<",\"max_error\":"<<maxerr<<",\"relative_l2\":"<<std::sqrt(err/power)<<",\"original_us\":"<<us[0]<<",\"packed_us\":"<<us[1]<<"}\n";
 if(changed)throw std::runtime_error("packed projection parity failed");
 }
 cublasDestroy(h);
}
}
int main(){try{cudaStream_t s;cudaSetDevice(0);cudaStreamCreate(&s);fixture(s);cudaStreamDestroy(s);}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
