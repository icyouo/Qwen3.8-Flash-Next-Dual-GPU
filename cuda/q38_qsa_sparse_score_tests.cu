#include "q38/cuda_qsa.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t s,const char* label){if(s!=cudaSuccess)throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(s));}
struct Memory{
 std::vector<void*> p;
 template<class T>T* alloc(std::size_t n){T*v=nullptr;check(cudaMalloc((void**)&v,n*sizeof(T)),"alloc");p.push_back(v);return v;}
 template<class T>T* upload(const std::vector<T>&v){auto*d=alloc<T>(v.size());check(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"upload");return d;}
 ~Memory(){for(auto*v:p)cudaFree(v);}
};
std::vector<std::uint16_t> fill(std::size_t n){std::vector<std::uint16_t>v(n);for(std::size_t i=0;i<n;i++){auto b=__float2bfloat16_rn(std::sin(float(i%10007)*.113f)*.2f);std::memcpy(&v[i],&b,2);}return v;}
template<class T>std::vector<T> read(const T*p,std::size_t n,cudaStream_t s){std::vector<T>v(n);check(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s),"read");check(cudaStreamSynchronize(s),"sync");return v;}
template<class T>void exact(const std::vector<T>&a,const std::vector<T>&b){if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))throw std::runtime_error("score/output bit parity failed");}
void fixture(unsigned m,cudaStream_t s){
 Memory mem;auto*q=mem.upload(fill(m*6144));auto*k=mem.upload(fill(4096*512));auto*v=mem.upload(fill(4096*512));auto*d=mem.alloc<std::uint16_t>(1);
 q38::CudaQsaLayerStateView cache{k,v,d,d};
 std::vector<std::int32_t>indices(m*2051);for(unsigned t=0;t<m;t++)for(unsigned i=0;i<2051;i++)indices[t*2051+i]=(i*13+t*17)%4096;
 auto*selected=mem.upload(indices);auto*scores=mem.alloc<float>(m*24*2051);auto*out=mem.alloc<std::uint16_t>(m*6144);auto*position=mem.alloc<unsigned>(1);
 setenv("Q38_CUDA_QSA_SCORE_TILES","1",1);setenv("Q38_CUDA_QSA_VALUE_STAGING","8",1);
 auto launch=[&](unsigned pos,unsigned tiles){auto setting=std::to_string(tiles);setenv("Q38_CUDA_QSA_SPARSE_SCORE_TILES",setting.c_str(),1);q38::cuda_qsa_apply_prefill_bf16(q,cache,pos,m,selected,scores,out,s,0);};
 for(auto layout:{q38::CudaQsaHeadLayout{},q38::CudaQsaHeadLayout{12,12,12,12}}){
  q38::cuda_qsa_set_head_layout(layout);
  for(unsigned pos:{0u,2046u,4096u,8189u,262137u}){
   check(cudaMemsetAsync(scores,0,m*24*2051*4,s),"clear scores");check(cudaMemsetAsync(out,0,m*6144*2,s),"clear output");launch(pos,1);
   auto es=read(scores,m*24*2051,s);auto eo=read(out,m*6144,s);
   for(unsigned tiles:{4u,8u}){
    check(cudaMemsetAsync(scores,0,m*24*2051*4,s),"clear scores");check(cudaMemsetAsync(out,0,m*6144*2,s),"clear output");launch(pos,tiles);exact(es,read(scores,m*24*2051,s));exact(eo,read(out,m*6144,s));
    cudaGraph_t graph;cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");q38::cuda_qsa_set_graph_position(position,m);launch(4096,tiles);q38::cuda_qsa_set_graph_position(nullptr,0);check(cudaStreamEndCapture(s,&graph),"end capture");check(cudaGraphInstantiate(&exec,graph,0),"instantiate");
    check(cudaMemcpyAsync(position,&pos,4,cudaMemcpyHostToDevice,s),"position");check(cudaMemsetAsync(scores,0,m*24*2051*4,s),"clear scores");check(cudaMemsetAsync(out,0,m*6144*2,s),"clear output");check(cudaGraphLaunch(exec,s),"replay");exact(es,read(scores,m*24*2051,s));exact(eo,read(out,m*6144,s));cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
   }
  }
 }
 q38::cuda_qsa_set_head_layout({0,12,0,12});
 for(unsigned tiles:{1u,4u,8u}){
  for(int i=0;i<10;i++)launch(4096,tiles);check(cudaStreamSynchronize(s),"warmup");cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a,s);for(int i=0;i<100;i++)launch(4096,tiles);cudaEventRecord(b,s);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);
  std::cout<<"{\"tokens\":"<<m<<",\"sparse_score_tiles\":"<<tiles<<",\"attention_us\":"<<ms*10<<",\"bit_parity\":true}\n";
 }
 q38::cuda_qsa_set_head_layout({});
}
}
int main(){try{cudaSetDevice(0);cudaStream_t s;check(cudaStreamCreate(&s),"stream");for(unsigned m:{1u,4u,5u})fixture(m,s);cudaStreamDestroy(s);std::cout<<"QSA_SPARSE_SCORE_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
