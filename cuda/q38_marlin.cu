// Framework-free integration of pinned vLLM v0.12.0 Marlin (Apache-2.0).
#include "q38/cuda_marlin.h"
#include "vendor/marlin/kernel.h"
#include "vendor/marlin/marlin_template.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
namespace q38 {
namespace {
void check(cudaError_t s,const char* op){if(s!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(s));}
template<unsigned Bits>
__global__ void repack(const unsigned char* src,unsigned* dst,unsigned n,unsigned k){
    const unsigned tile=blockIdx.x, lane=threadIdx.x%32, warp=threadIdx.x/32;
    const unsigned nt=tile%(n/64), kt=tile/(n/64);
    const unsigned c=nt*64+warp*16+lane/4, r=kt*16+(lane%4)*2;
    const unsigned offsets[4]={0,1,8,9}; unsigned vals[8];
    #pragma unroll
    for(unsigned i=0;i<8;++i){
        const auto index=static_cast<unsigned long long>(c+(i/4)*8)*k+r+offsets[i%4];
        unsigned q;
        if constexpr(Bits==4)q=(src[index/2]>>((index%2)*4))&15;
        else q=src[index];
        vals[i]=q^(1u<<(Bits-1));
    }
    if constexpr(Bits==4){
        const unsigned perm[8]={0,2,4,6,1,3,5,7};unsigned word=0;
        #pragma unroll
        for(unsigned i=0;i<8;++i)word|=vals[perm[i]]<<(4*i);
        dst[tile*128+lane*4+warp]=word;
    }else{
        const unsigned perm[4]={0,2,1,3};unsigned a=0,b=0;
        #pragma unroll
        for(unsigned i=0;i<4;++i){a|=vals[perm[i]]<<(8*i);b|=vals[4+perm[i]]<<(8*i);}
        dst[tile*256+lane*8+warp*2]=a;dst[tile*256+lane*8+warp*2+1]=b;
    }
}
__global__ void repack_scales(const unsigned short* src,unsigned short* dst,unsigned n,unsigned groups){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*groups)return;
    unsigned col=i%n,g=i/n,p=col%64;unsigned original=col-p+p/8+(p%8)*8;
    dst[i]=src[original*groups+g];
}
__global__ void marlin_plan(const Q38ExpertMmqTaskV1* tasks,unsigned max_tasks,unsigned routes,
    int* sorted,int* experts,int* padded){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0)*padded=max_tasks*16;
    if(i>=max_tasks*16)return;
    auto t=tasks[i/16];unsigned row=i%16;
    sorted[i]=row<t.valid_rows?int(t.packed_begin+row):int(routes);
    if(row%8==0)experts[i/8]=row<t.valid_rows?int(t.expert):-1;
}
// Compact the padded eight-row Marlin blocks without a CPU count readback.
// sum(ceil(valid_rows/8)) <= routes for the bounded 1..7-token branch.
__global__ void marlin_plan_small(const Q38ExpertMmqTaskV1* tasks,unsigned max_tasks,unsigned routes,
    int* sorted,int* experts,int* padded){
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x/32,i=threadIdx.x;
    const auto t=i<max_tasks?tasks[i]:Q38ExpertMmqTaskV1{};
    const unsigned blocks=(t.valid_rows+7)/8;
    unsigned prefix=blocks;
    for(unsigned step=1;step<32;step*=2){unsigned other=__shfl_up_sync(0xffffffff,prefix,step);if(lane>=step)prefix+=other;}
    __shared__ unsigned totals[4];if(lane==31)totals[warp]=prefix;__syncthreads();
    unsigned prior=0;for(unsigned w=0;w<warp;++w)prior+=totals[w];
    const unsigned begin=prior+prefix-blocks;
    if(i==127){unsigned total=prior+prefix;*padded=total*8;}
    for(unsigned b=0;b<blocks;++b){experts[begin+b]=t.expert;for(unsigned row=0;row<8;++row){unsigned local=b*8+row;sorted[(begin+b)*8+row]=local<t.valid_rows?int(t.packed_begin+local):int(routes);}}
}
__global__ void marlin_activation(unsigned short* gate,unsigned short* mid,unsigned routes,const unsigned* owned=nullptr){
    const auto i=static_cast<unsigned long long>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<unsigned long long>(routes)*640)return;
    const auto base=(i/640)*1280+i%640;
    if(owned && i/640>=*owned){gate[base]=0;gate[base+640]=0;mid[i]=0;return;}
    float g=__bfloat162float(__ushort_as_bfloat16(gate[base]));
    float u=__bfloat162float(__ushort_as_bfloat16(gate[base+640]));
    mid[i]=__bfloat16_as_ushort(__float2bfloat16_rn(g/(1.f+expf(-g))*u));
}
__global__ void to_float(unsigned short* input,float* output,unsigned count,const unsigned* owned=nullptr){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    if(owned && i/2560>=*owned){input[i]=0;output[i]=0.f;return;}
    output[i]=__bfloat162float(__ushort_as_bfloat16(input[i]));
}
__global__ void reduce_bf16_top10(unsigned short* input, float* output,
    const unsigned* mapping, const float* weights, unsigned routes,
    const unsigned* owned) {
    const unsigned token = blockIdx.y;
    // Preserve initialized diagnostic BF16 tails without materializing a
    // routes x hidden FP32 buffer. Unowned assignments are never consumed.
    for (unsigned i = (token * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; i < routes * 2560;
         i += gridDim.x * gridDim.y * blockDim.x)
        if (i / 2560 >= *owned) input[i] = 0;
    const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < 2560) {
        float sum = 0.f;
#pragma unroll
        for (unsigned slot = 0; slot < 10; ++slot) {
            const unsigned assignment = token * 10 + slot;
            const unsigned packed = mapping[assignment];
            if (packed != kQ38RouteUnownedAssignment)
                sum += __bfloat162float(__ushort_as_bfloat16(input[packed * 2560 + column])) * weights[assignment];
        }
        output[token * 2560 + column] = sum;
    }
}
using Kernel=void(*)(MARLIN_KERNEL_PARAMS);
template<vllm::ScalarTypeId W,bool Persistent=false,bool Wide=false> Kernel select(){constexpr auto F=vllm::kBFloat16.id();return MARLIN_NAMESPACE_NAME::Marlin<F,W,F,F,Wide?256:128,1,Wide?8:4,8,true,4,8,false,Persistent>;}
void launch(const CudaTensorViewV1& w,const unsigned short* a,unsigned short* c,
    unsigned routes,unsigned tasks,CudaMarlinWorkspace ws,cudaStream_t stream){
    if(w.layout!=CudaWeightLayoutV1::kMarlinG128)throw std::runtime_error("Marlin expert layout mismatch");
    const auto n=unsigned(w.descriptor->shape[1]),k=unsigned(w.descriptor->shape[2]);
    const auto bits=w.descriptor->format==DeviceWeightFormatV1::kW4A16SymG128?4:8;
    const auto* persistent_setting=std::getenv("Q38_CUDA_MARLIN_PERSISTENT");
    const bool persistent=routes<=70 && persistent_setting && std::strcmp(persistent_setting,"1")==0;
    const auto* wide_setting=std::getenv("Q38_CUDA_MARLIN_WIDE_N");
    const bool wide=routes<=70 && wide_setting && std::strcmp(wide_setting,"1")==0;
    if(wide && persistent)throw std::invalid_argument("independent Marlin experiments cannot be combined");
    auto kernel=wide ? (bits==4?select<vllm::kU4B8.id(),false,true>():select<vllm::kU8B128.id(),false,true>())
                       : persistent ? (bits==4?select<vllm::kU4B8.id(),true>():select<vllm::kU8B128.id(),true>())
                           : (bits==4?select<vllm::kU4B8.id()>():select<vllm::kU8B128.id()>());
    const unsigned tile_n=wide?128:64;
    const int shared=256+4*16*128*2+4*128*tile_n*bits/8+4*tile_n*2;
    // Attribute setup happens during eager warmup too; capture permits this
    // host-only function. No allocation, locks or inter-CTA reductions.
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared),"Marlin shared-memory attribute");
    const unsigned tiles=tasks*(n/tile_n);
    kernel<<<persistent ? (tiles<128?tiles:128) : tiles,wide?256:128,shared,stream>>>((const int4*)a,(const int4*)w.data,(int4*)c,(int4*)ws.tmp,
        nullptr,nullptr,(const int4*)w.scales,nullptr,nullptr,nullptr,
        ws.sorted,ws.experts,ws.padded,nullptr,1,false,true,k/128,routes,n,k,ws.locks,false,false,true);
    check(cudaGetLastError(),"Marlin grouped expert launch");
}
}
bool cuda_marlin_moe_enabled(){const char* p=std::getenv("Q38_CUDA_MARLIN_MOE");return p&&std::strcmp(p,"1")==0;}
void cuda_marlin_repack_expert(void* data,void* scales,unsigned n,unsigned k,unsigned bits,
    void* data_scratch,void* scale_scratch,void* opaque){
    if((bits!=4&&bits!=8)||n%64||k%128||!data||!scales||!opaque)throw std::invalid_argument("unsupported native Marlin expert");
    auto stream=reinterpret_cast<cudaStream_t>(opaque);unsigned tiles=(n/64)*(k/16);
    if(bits==4)repack<4><<<tiles,128,0,stream>>>((const unsigned char*)data,(unsigned*)data_scratch,n,k);
    else repack<8><<<tiles,128,0,stream>>>((const unsigned char*)data,(unsigned*)data_scratch,n,k);
    check(cudaGetLastError(),"lossless Marlin repack");
    repack_scales<<<(n*(k/128)+255)/256,256,0,stream>>>((const unsigned short*)scales,(unsigned short*)scale_scratch,n,k/128);
    check(cudaGetLastError(),"lossless Marlin scale permutation");
    check(cudaMemcpyAsync(data,data_scratch,static_cast<size_t>(n)*k*bits/8,cudaMemcpyDeviceToDevice,stream),"install Marlin expert layout");
    check(cudaMemcpyAsync(scales,scale_scratch,static_cast<size_t>(n)*(k/128)*2,cudaMemcpyDeviceToDevice,stream),"install Marlin scale layout");
}
void cuda_marlin_moe(const CudaTensorViewV1& gate,const CudaTensorViewV1& down,
    const unsigned short* input,const Q38ExpertMmqTaskV1* tasks,unsigned max_tasks,unsigned routes,
    unsigned short* mid,float* output,CudaMarlinWorkspace ws,void* opaque){
    auto stream=reinterpret_cast<cudaStream_t>(opaque);
    const char* compact_flag=std::getenv("Q38_CUDA_MARLIN_COMPACT_PLAN");
    const bool compact=routes<=70 && max_tasks<=70 && compact_flag && std::strcmp(compact_flag,"1")==0;
    if(compact)marlin_plan_small<<<1,128,0,stream>>>(tasks,max_tasks,routes,ws.sorted,ws.experts,ws.padded);
    else marlin_plan<<<(max_tasks*16+255)/256,256,0,stream>>>(tasks,max_tasks,routes,ws.sorted,ws.experts,ws.padded);
    check(cudaGetLastError(),"build graph-local Marlin plan");
    const auto* setting=std::getenv("Q38_CUDA_MARLIN_FUSED_TAIL");
    const bool fused_tail=ws.owned_routes && setting && std::strcmp(setting,"1")==0;
    // Clear unowned packed tail: consumers skip it, but all device reads and
    // diagnostic outputs remain initialized across reject/reset generations.
    if(!fused_tail)check(cudaMemsetAsync(ws.gate_up,0,static_cast<size_t>(routes)*1280*2,stream),"clear Marlin gate output");
    if(!fused_tail)check(cudaMemsetAsync(ws.down,0,static_cast<size_t>(routes)*2560*2,stream),"clear Marlin down output");
    launch(gate,input,ws.gate_up,routes,max_tasks*(compact?1:2),ws,stream);
    marlin_activation<<<(routes*640+255)/256,256,0,stream>>>(ws.gate_up,mid,routes,fused_tail?ws.owned_routes:nullptr);
    check(cudaGetLastError(),"Marlin SiLU boundary");
    launch(down,mid,ws.down,routes,max_tasks*(compact?1:2),ws,stream);
    if (ws.reduce_assignment_to_packed) {
        if (routes > 70 || routes % 10 || !ws.owned_routes || !ws.reduce_route_weights)
            throw std::invalid_argument("unsupported Marlin direct reduction");
        reduce_bf16_top10<<<dim3(10,routes/10),256,0,stream>>>(ws.down,output,
            ws.reduce_assignment_to_packed,ws.reduce_route_weights,routes,ws.owned_routes);
    } else to_float<<<(routes*2560+255)/256,256,0,stream>>>(ws.down,output,routes*2560,fused_tail?ws.owned_routes:nullptr);
    check(cudaGetLastError(),"Marlin FP32 reduction input");
}
}
