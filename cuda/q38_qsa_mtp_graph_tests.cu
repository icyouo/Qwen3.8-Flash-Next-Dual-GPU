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
 for(unsigned sparse=0;sparse<2;++sparse) {
  auto candidate=[&](unsigned pos){q38::cuda_qsa_prepare_prefill_bf16(qg,k,v,iqk,qn,kn,in,in,q,g,iq,cache,pos,1,s,0);
    q38::cuda_qsa_select_prefill(iq,cache,pos,1,blocks,65536,selected,s,0);
    q38::cuda_qsa_apply_prefill_bf16(q,cache,pos,1,selected,scores,o,s,0);};
  cudaGraph_t graph;cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");
  q38::cuda_qsa_set_graph_position(position,1);candidate(sparse?4096:0);q38::cuda_qsa_set_graph_position(nullptr,0);
  check(cudaStreamEndCapture(s,&graph),"end");check(cudaGraphInstantiate(&exec,graph,0),"instantiate");
  for(unsigned pos:{0u,1u,2u,3u,511u,1023u,2047u,2048u,2051u,8189u,262143u}) {
   if((pos+1>2048)!=bool(sparse))continue;
   launch(pos,qg);const auto eq=read(q,6144,s),eg=read(g,6144,s),ei=read(iq,512,s),eo=read(o,6144,s);
   const unsigned count=std::min((pos+1)/4,512u)*4+(pos+1)%4;
   const auto es=read(selected,count,s);
   auto compare=[&]{equal(eq,read(q,6144,s),"query");equal(eg,read(g,6144,s),"gate");
    equal(ei,read(iq,512,s),"index query");equal(eo,read(o,6144,s),"MTP attention output");
    equal(es,read(selected,count,s),"selected indices");};
   candidate(pos);compare();
   check(cudaMemsetAsync(q,255,6144*2,s),"poison query");check(cudaMemsetAsync(o,255,6144*2,s),"poison output");
   check(cudaMemcpyAsync(position,&pos,4,cudaMemcpyHostToDevice,s),"position");check(cudaGraphLaunch(exec,s),"replay");compare();
   std::cout<<"{\"position\":"<<pos<<",\"bit_parity\":true}\n";
  }
  cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
 }

}

}
int main(){try{cudaStream_t s;cudaSetDevice(0);cudaStreamCreate(&s);fixture(1,s);cudaStreamDestroy(s);std::cout<<"QSA_MTP_GRAPH_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
