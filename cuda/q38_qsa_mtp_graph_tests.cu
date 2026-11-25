#include "q38/cuda_qsa.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
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
    if (n == 0) return values;
    check(cudaMemcpyAsync(values.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, stream), "read");
    check(cudaStreamSynchronize(stream), "read ready");
    return values;
}
template <typename T> void equal(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    if (a.size() != b.size() || (!a.empty() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T))))
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
  auto candidate=[&](unsigned pos){
    q38::cuda_qsa_prepare_prefill_bf16(qg,k,v,iqk,qn,kn,in,in,q,g,iq,cache,pos,m,s,0);
    q38::cuda_qsa_select_prefill(iq,cache,pos,m,blocks,65536,selected,s,0);
    if(m==1) q38::cuda_qsa_apply_prefill_bf16(q,cache,pos,m,selected,scores,o,s,0);
    else q38::cuda_qsa_apply_grouped_prefill_bf16(q,cache,pos,m,selected,scores,o,s,0);};
  cudaGraph_t graph;cudaGraphExec_t exec;
  if (m == 1) setenv("Q38_CUDA_QSA_TILE_MTP_INDEX", "1", 1);
  check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");
  q38::cuda_qsa_set_graph_position(position,m);candidate(sparse?4096:0);q38::cuda_qsa_set_graph_position(nullptr,0);
  check(cudaStreamEndCapture(s,&graph),"end");check(cudaGraphInstantiate(&exec,graph,0),"instantiate");
  unsetenv("Q38_CUDA_QSA_TILE_MTP_INDEX");
  const auto positions = m==1 ? std::vector<unsigned>{0,1,2,3,511,1023,2047,2048,2051,8189,262143}
                              : std::vector<unsigned>{0,512,1536,1537,1540,4096};
  for(unsigned pos:positions) {
   if((pos+m>2048)!=bool(sparse))continue;
   check(cudaMemsetAsync(selected,255,m*2051*sizeof(std::int32_t),s),"initialize selector");
   launch(pos,qg);const auto eq=read(q,m*6144,s),eg=read(g,m*6144,s),ei=read(iq,m*512,s),eo=read(o,m*6144,s);
   auto read_selected = [&] {
      const auto rows = read(selected, m * 2051, s);
      std::vector<std::int32_t> active;
      for (unsigned row = 0; row < m; ++row) {
          const auto visible = pos + row + 1;
          const auto count = std::min(visible / 4, 512u) * 4 + visible % 4;
          const auto start = rows.begin() + row * 2051;
          active.insert(active.end(), start, start + count);
      }
      return active;
   };
   const auto es=read_selected();
   auto compare=[&]{equal(eq,read(q,m*6144,s),"query");equal(eg,read(g,m*6144,s),"gate");
    equal(ei,read(iq,m*512,s),"index query");equal(eo,read(o,m*6144,s),"attention output");
    equal(es,read_selected(),"selected indices");};
   candidate(pos);compare();
   check(cudaMemsetAsync(q,255,m*6144*2,s),"poison query");check(cudaMemsetAsync(o,255,m*6144*2,s),"poison output");
   check(cudaMemcpyAsync(position,&pos,4,cudaMemcpyHostToDevice,s),"position");check(cudaGraphLaunch(exec,s),"replay");compare();
   std::cout<<"{\"position\":"<<pos<<",\"rows\":"<<m<<",\"bit_parity\":true}\n";
  }
  cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
 }

 for (unsigned pos : {0u, 3u, 2051u}) {
    q38::cuda_qsa_prepare_prefill_bf16(qg,k,v,iqk,qn,kn,in,in,q,g,iq,cache,pos,m,s,0);
    const auto full_keys = read(cache.main_keys + pos * 512, m * 512, s);
    const auto full_values = read(cache.main_values + pos * 512, m * 512, s);
    const auto full_raw = read(cache.raw_index_keys + pos * 128, m * 128, s);
    const auto pooled = (pos + m) / 4 - pos / 4;
    const auto full_pooled = read(cache.pooled_index_keys + (pos / 4) * 128, pooled * 128, s);
    check(cudaMemsetAsync(cache.main_keys + pos * 512,255,m*512*2,s),"poison keys");
    check(cudaMemsetAsync(cache.main_values + pos * 512,255,m*512*2,s),"poison values");
    check(cudaMemsetAsync(cache.raw_index_keys + pos * 128,255,m*128*2,s),"poison raw index keys");
    if (pooled) check(cudaMemsetAsync(cache.pooled_index_keys + (pos / 4) * 128,255,pooled*128*2,s),"poison pooled index keys");
    q38::cuda_qsa_prepare_prefill_bf16(nullptr,k,v,iqk,nullptr,kn,nullptr,in,
        nullptr,nullptr,nullptr,cache,pos,m,s,0,true);
    equal(full_keys,read(cache.main_keys + pos * 512,m * 512,s),"cache-only main keys");
    equal(full_values,read(cache.main_values + pos * 512,m * 512,s),"cache-only main values");
    equal(full_raw,read(cache.raw_index_keys + pos * 128,m * 128,s),"cache-only raw index keys");
    equal(full_pooled,read(cache.pooled_index_keys + (pos / 4) * 128,pooled * 128,s),"cache-only pooled index keys");
 }

}

}
int main(){try{cudaStream_t s;cudaSetDevice(0);cudaStreamCreate(&s);fixture(1,s);fixture(512,s);cudaStreamDestroy(s);std::cout<<"QSA_MTP_GRAPH_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
