#include "q38/cuda_marlin.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Buffers{std::vector<void*> all;template<class T>T* alloc(std::size_t n){T*p;check(cudaMalloc(&p,n*sizeof(T)));all.push_back(p);return p;}~Buffers(){for(auto p:all)cudaFree(p);}};
unsigned short bf16(float x){return __bfloat16_as_ushort(__float2bfloat16_rn(x));}
template<class T>std::vector<T> read(T*p,std::size_t n,cudaStream_t s){std::vector<T>x(n);check(cudaMemcpyAsync(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));return x;}
template<class T>void parity(const std::vector<T>&a,const std::vector<T>&b,const char*label){if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))throw std::runtime_error(label);}
void fixture(unsigned bits,cudaStream_t stream){
 const char* wide_setting=std::getenv("Q38_CUDA_MARLIN_WIDE_N");const bool wide=wide_setting&&std::strcmp(wide_setting,"1")==0;
 const char* ps=std::getenv("Q38_CUDA_MARLIN_PERSISTENT");const bool persistent=ps&&std::strcmp(ps,"1")==0;
 Buffers mem;auto*scratch=mem.alloc<unsigned char>(1280*2560);auto*ss=mem.alloc<unsigned short>(1280*20);
 q38::DeviceTensorV1 gd,dd;gd.shape={4,1280,2560};dd.shape={4,2560,640};gd.format=dd.format=bits==4?q38::DeviceWeightFormatV1::kW4A16SymG128:q38::DeviceWeightFormatV1::kW8A16SymG128;
 auto weight=[&](q38::DeviceTensorV1&desc){unsigned n=desc.shape[1],k=desc.shape[2];std::vector<unsigned char>q(4*n*k*bits/8);for(unsigned i=0;i<q.size();++i)q[i]=i*97+i/2560*13;
  std::vector<unsigned short>s(4*n*k/128);for(unsigned i=0;i<s.size();++i)s[i]=bf16(.002f+(i%17)*.000125f);
  auto*d=mem.alloc<unsigned char>(q.size());auto*ds=mem.alloc<unsigned short>(s.size());check(cudaMemcpyAsync(d,q.data(),q.size(),cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(ds,s.data(),s.size()*2,cudaMemcpyHostToDevice,stream));
  for(unsigned e=0;e<4;++e)q38::cuda_marlin_repack_expert(d+e*n*k*bits/8,ds+e*n*k/128,n,k,bits,scratch,ss,stream);
  check(cudaStreamSynchronize(stream));return q38::CudaTensorViewV1{&desc,d,ds,q38::CudaWeightLayoutV1::kMarlinG128};};
 auto gate=weight(gd),down=weight(dd);
 for(unsigned m:{1u,2u,3u,5u,7u,512u}){
  unsigned routes=m*10,nt=q38::q38_moe_prefill_max_tasks_v1(m);auto*tasks=mem.alloc<q38::Q38ExpertMmqTaskV1>(nt);auto*owned=mem.alloc<unsigned>(1);
  auto*x=mem.alloc<unsigned short>(routes*2560);auto*mid=mem.alloc<unsigned short>(routes*640);auto*out=mem.alloc<float>(routes*2560);
  auto*mapping=mem.alloc<unsigned>(routes);auto*weights=mem.alloc<float>(routes);
  auto*reference=mem.alloc<float>(m*2560);auto*reduced=mem.alloc<float>(m*2560);
  q38::CudaMarlinWorkspace ws;ws.sorted=mem.alloc<int>(nt*16);ws.experts=mem.alloc<int>(nt*2);ws.padded=mem.alloc<int>(1);ws.locks=mem.alloc<int>(4096);ws.gate_up=mem.alloc<unsigned short>(routes*1280);ws.down=mem.alloc<unsigned short>(routes*2560);ws.tmp=mem.alloc<float>(nt*16*2560);ws.owned_routes=owned;
  auto launch=[&]{q38::cuda_marlin_moe(gate,down,x,tasks,nt,routes,mid,out,ws,stream);};
  auto launch_reduced=[&]{auto rws=ws;rws.reduce_assignment_to_packed=mapping;rws.reduce_route_weights=weights;
   q38::cuda_marlin_moe(gate,down,x,tasks,nt,routes,mid,reduced,rws,stream);};
  setenv("Q38_CUDA_MARLIN_WIDE_N",wide?"1":"0",1);
  setenv("Q38_CUDA_MARLIN_PERSISTENT",persistent?"1":"0",1);
  setenv("Q38_CUDA_MARLIN_COMPACT_PLAN","1",1);setenv("Q38_CUDA_MARLIN_FUSED_TAIL","1",1);
  cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,0));
  cudaGraph_t rg=nullptr;cudaGraphExec_t re=nullptr;
  if(m<=7){check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch_reduced();check(cudaStreamEndCapture(stream,&rg));check(cudaGraphInstantiate(&re,rg,0));}
  for(unsigned n:{routes,0u,routes*3/5,routes/5,routes}){
   std::vector<q38::Q38ExpertMmqTaskV1>plan(nt);unsigned p=0,i=0;while(p<n){unsigned count=std::min(16u,n-p);plan[i++]={p,static_cast<unsigned short>((i-1)%4),static_cast<unsigned char>(count),0};p+=count;}
   std::vector<unsigned short>input(routes*2560);for(unsigned j=0;j<input.size();++j)input[j]=bf16(std::sin((j+n)*.013f)*.2f);
   check(cudaMemcpyAsync(tasks,plan.data(),nt*8,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(owned,&n,4,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(x,input.data(),input.size()*2,cudaMemcpyHostToDevice,stream));
   setenv("Q38_CUDA_MARLIN_WIDE_N","0",1);setenv("Q38_CUDA_MARLIN_PERSISTENT","0",1);setenv("Q38_CUDA_MARLIN_FUSED_TAIL","0",1);launch();auto a=read(out,routes*2560,stream);auto am=read(mid,routes*640,stream);auto ag=read(ws.gate_up,routes*1280,stream);auto ad=read(ws.down,routes*2560,stream);
   check(cudaMemsetAsync(ws.gate_up,0xff,routes*1280*2,stream));check(cudaMemsetAsync(ws.down,0xff,routes*2560*2,stream));check(cudaMemsetAsync(mid,0xff,routes*640*2,stream));check(cudaGraphLaunch(exec,stream));
   parity(a,read(out,routes*2560,stream),"route output");parity(am,read(mid,routes*640,stream),"activation");parity(ag,read(ws.gate_up,routes*1280,stream),"gate/up initialized");parity(ad,read(ws.down,routes*2560,stream),"down initialized");
   if(m<=7){std::vector<unsigned>map(routes);std::vector<float>w(routes);
    // Reverse packed order and leave arbitrary holes: preserve assignment
    // order even when route packing changes between graph generations.
    for(unsigned j=0;j<routes;++j){map[j]=routes-1-j<n?routes-1-j:q38::kQ38RouteUnownedAssignment;w[j]=float((j+n)%13)*.03125f;}
    check(cudaMemcpyAsync(mapping,map.data(),routes*4,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(weights,w.data(),routes*4,cudaMemcpyHostToDevice,stream));
    q38::cuda_moe_reduce_top10_f32_v1(out,mapping,m,reference,stream,0,weights);
    auto expected=read(reference,m*2560,stream);
    check(cudaMemsetAsync(ws.down,0xff,routes*2560*2,stream));check(cudaGraphLaunch(re,stream));
    parity(expected,read(reduced,m*2560,stream),"direct BF16 top10 reduction");parity(ad,read(ws.down,routes*2560,stream),"direct initialized down tail");
   }
  }
  setenv("Q38_CUDA_MARLIN_WIDE_N",wide?"1":"0",1);
  setenv("Q38_CUDA_MARLIN_PERSISTENT",persistent?"1":"0",1);
  float timing[2]={};if(m==5){for(int v=0;v<2;++v){setenv("Q38_CUDA_MARLIN_FUSED_TAIL",v?"1":"0",1);cudaGraph_t g;cudaGraphExec_t e;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(unsigned r=0;r<20;++r)launch();check(cudaStreamEndCapture(stream,&g));check(cudaGraphInstantiate(&e,g,0));for(unsigned r=0;r<3;++r)check(cudaGraphLaunch(e,stream));cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,stream));for(unsigned r=0;r<20;++r)check(cudaGraphLaunch(e,stream));check(cudaEventRecord(b,stream));check(cudaEventSynchronize(b));check(cudaEventElapsedTime(&timing[v],a,b));timing[v]/=400.f;cudaEventDestroy(a);cudaEventDestroy(b);cudaGraphExecDestroy(e);cudaGraphDestroy(g);}}
  if(re)cudaGraphExecDestroy(re);if(rg)cudaGraphDestroy(rg);
  cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);std::cout<<"{\"bits\":"<<bits<<",\"M\":"<<m<<",\"changing_owned_prefix_poisoned_tail_graph_bit_parity\":true,\"direct_reduction_bit_parity\":"<<(m<=7?"true":"null")<<",\"baseline_ms\":"<<timing[0]<<",\"fused_ms\":"<<timing[1]<<"}\n";
 }
}
}
int main(){try{cudaStream_t s;check(cudaStreamCreate(&s));fixture(4,s);fixture(8,s);cudaStreamDestroy(s);std::cout<<"MARLIN_TAIL_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
