#include "q38/cuda_qsa.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
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


void fixture(unsigned m,cudaStream_t s){
 Buffers mem;auto fill=[](std::size_t n,float scale){std::vector<std::uint16_t>x(n);for(std::size_t i=0;i<n;++i)x[i]=bf16(std::sin(float(i%10007)*.113f)*scale);return x;};
 const auto host_qg=fill(m*12288,.3f);auto*qg=mem.upload(host_qg);auto*k=mem.upload(fill(m*512,.2f));auto*v=mem.upload(fill(m*512,.4f));auto*iqk=mem.upload(fill(m*640,.3f));auto*qn=mem.upload(fill(256,.1f));auto*kn=mem.upload(fill(256,.1f));auto*in=mem.upload(fill(128,.1f));
 q38::CudaQsaLayerStateView cache{mem.upload(fill(262144ull*512,.1f)),mem.upload(fill(262144ull*512,.2f)),mem.upload(fill(262144ull*128,.1f)),mem.upload(fill(65536ull*128,.1f))};
 auto*q=mem.allocate<std::uint16_t>(m*6144);auto*g=mem.allocate<std::uint16_t>(m*6144);auto*iq=mem.allocate<std::uint16_t>(m*512);auto*o=mem.allocate<std::uint16_t>(m*6144);auto*blocks=mem.allocate<float>(m*65536);auto*selected=mem.allocate<std::int32_t>(m*2051);auto*scores=mem.allocate<float>(m*24*2051);auto*position=mem.allocate<unsigned>(1);
 auto launch=[&](unsigned pos,const std::uint16_t* projected,bool ungrouped=false){
  if(m==1){q38::cuda_qsa_prepare_main_decode_bf16(projected,k,v,qn,kn,q,g,cache,pos,s,0);q38::cuda_qsa_prepare_index_decode_bf16(iqk,in,in,iq,cache,pos,s,0);auto count=q38::cuda_qsa_select_decode(iq,cache,pos,blocks,selected,s,0);q38::cuda_qsa_attention_decode_bf16(q,cache,selected,count,scores,o,s,0);}
  else{q38::cuda_qsa_prepare_prefill_bf16(projected,k,v,iqk,qn,kn,in,in,q,g,iq,cache,pos,m,s,0);q38::cuda_qsa_select_prefill(iq,cache,pos,m,blocks,65536,selected,s,0);if(ungrouped)q38::cuda_qsa_apply_prefill_bf16(q,cache,pos,m,selected,scores,o,s,0);else q38::cuda_qsa_apply_grouped_prefill_bf16(q,cache,pos,m,selected,scores,o,s,0);}
 };
 auto owned_equal=[&](const std::vector<std::uint16_t>&a,const std::vector<std::uint16_t>&b,unsigned rank,const char*label){for(unsigned t=0;t<m;++t){auto first=t*6144+rank*3072;for(unsigned i=first;i<first+3072;++i)if(a[i]!=b[i])throw std::runtime_error(std::string("owned parity: ")+label);for(unsigned i=t*6144+(1-rank)*3072;i<t*6144+(1-rank)*3072+3072;++i)if(b[i]!=65535)throw std::runtime_error("unowned scratch was written");}};
 for(unsigned rank=0;rank<2;++rank)for(unsigned packed=0;packed<2;++packed){
  std::vector<std::uint16_t> host_local(m*6144);for(unsigned t=0;t<m;++t)std::copy_n(host_qg.data()+t*12288+rank*6144,6144,host_local.data()+t*6144);auto*local=mem.upload(host_local);
  const auto layout=q38::CudaQsaHeadLayout{rank*12u,12u,packed?rank*12u:0u,packed?12u:24u};const auto*projected=packed?local:qg;
  cudaGraph_t graphs[2]{};cudaGraphExec_t execs[2]{};
  if(m>1){for(unsigned sparse=0;sparse<2;++sparse){check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");q38::cuda_qsa_set_graph_position(position,m);q38::cuda_qsa_set_head_layout(layout);launch(sparse?4096:0,projected,true);q38::cuda_qsa_set_head_layout({});q38::cuda_qsa_set_graph_position(nullptr,0);check(cudaStreamEndCapture(s,&graphs[sparse]),"end");check(cudaGraphInstantiate(&execs[sparse],graphs[sparse],0),"instantiate");}}
  const std::vector<unsigned> positions=m==5?std::vector<unsigned>{0,3,2043,2044,2047,2048,4093,8189,262137}:std::vector<unsigned>{3,2048-m,2049-m,4093};
  for(unsigned pos:positions){
   q38::cuda_qsa_set_head_layout({});launch(pos,qg);auto eq=read(q,m*6144,s),eg=read(g,m*6144,s),eiq=read(iq,m*512,s),eo=read(o,m*6144,s);auto esi=read(selected,m*2051,s);
   auto compare=[&]{owned_equal(eq,read(q,m*6144,s),rank,"query");owned_equal(eg,read(g,m*6144,s),rank,"gate");equal(eiq,read(iq,m*512,s),"index query");owned_equal(eo,read(o,m*6144,s),rank,"attention output");auto got=read(selected,m*2051,s);for(unsigned t=0;t<m;++t){unsigned visible=pos+t+1,count=std::min(visible/4,512u)*4+visible%4;for(unsigned i=0;i<count;++i)if(got[t*2051+i]!=esi[t*2051+i])throw std::runtime_error("selected indices");}};
   auto poison=[&]{check(cudaMemsetAsync(q,255,m*6144*2,s),"poison q");check(cudaMemsetAsync(g,255,m*6144*2,s),"poison g");check(cudaMemsetAsync(o,255,m*6144*2,s),"poison o");check(cudaMemsetAsync(scores,255,m*24*2051*4,s),"poison scores");};
   poison();q38::cuda_qsa_set_head_layout(layout);launch(pos,projected,true);q38::cuda_qsa_set_head_layout({});compare();
   if(m>1){poison();check(cudaMemcpyAsync(position,&pos,4,cudaMemcpyHostToDevice,s),"position");check(cudaGraphLaunch(execs[pos+m>2048],s),"replay");compare();}
   std::cout<<"{\"M\":"<<m<<",\"rank\":"<<rank<<",\"packed\":"<<packed<<",\"position\":"<<pos<<",\"bit_parity\":true}\n";
  }
  if(m>1)for(unsigned i=0;i<2;++i){cudaGraphExecDestroy(execs[i]);cudaGraphDestroy(graphs[i]);}
 }
}

}
int main(){try{cudaStream_t s;cudaSetDevice(0);cudaStreamCreate(&s);for(unsigned m:{1u,2u,3u,5u,7u})fixture(m,s);cudaStreamDestroy(s);std::cout<<"QSA_LOCAL_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
