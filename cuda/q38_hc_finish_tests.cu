#include "q38/cuda_hyper.h"
#include "q38/cuda_moe.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace{
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
struct Buffers{std::vector<void*>p;template<class T>T*alloc(std::size_t n){T*x;check(cudaMalloc(&x,n*sizeof(T)));p.push_back(x);return x;}~Buffers(){for(auto x:p)cudaFree(x);}};
std::vector<unsigned short>values(std::size_t n,float scale){std::vector<unsigned short>x(n);for(unsigned i=0;i<n;++i)x[i]=__bfloat16_as_ushort(__float2bfloat16_rn(std::sin(i*.137f)*scale));return x;}
std::vector<unsigned short>read(unsigned short*p,unsigned n,cudaStream_t s){std::vector<unsigned short>x(n);check(cudaMemcpyAsync(x.data(),p,n*2,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));return x;}
void fixture(unsigned m,cudaStream_t s){Buffers b;auto*orig=b.alloc<unsigned short>(m*10240),*block=b.alloc<unsigned short>(m*2560),*raw=b.alloc<unsigned short>(m*4),*gate=b.alloc<unsigned short>(m*4),*out=b.alloc<unsigned short>(m*10240);
 auto x=values(m*10240,2.f),y=values(m*2560,.3f),z=values(m*4,50.f);check(cudaMemcpyAsync(orig,x.data(),x.size()*2,cudaMemcpyHostToDevice,s));check(cudaMemcpyAsync(block,y.data(),y.size()*2,cudaMemcpyHostToDevice,s));check(cudaMemcpyAsync(raw,z.data(),z.size()*2,cudaMemcpyHostToDevice,s));
 auto launch=[&](bool fused){if(fused)q38::cuda_hyper_inject_from_projection_bf16(orig,block,raw,out,m,s,0);else{q38::cuda_hyper_sigmoid_bf16(raw,gate,m,4,4.f,2.f,s,0);q38::cuda_hyper_inject_bf16(orig,block,gate,out,m,s,0);}};
 launch(false);auto expected=read(out,m*10240,s);for(bool graph:{false,true}){if(!graph)launch(true);else{cudaGraph_t g;cudaGraphExec_t e;check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));launch(true);check(cudaStreamEndCapture(s,&g));check(cudaGraphInstantiate(&e,g,0));check(cudaMemsetAsync(out,0xff,m*10240*2,s));check(cudaGraphLaunch(e,s));check(cudaStreamSynchronize(s));cudaGraphExecDestroy(e);cudaGraphDestroy(g);}if(expected!=read(out,m*10240,s))throw std::runtime_error("HC finish BF16 parity");}
 float timing[2]{};if(m==5){for(unsigned f=0;f<2;++f){cudaGraph_t g;cudaGraphExec_t e;check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));for(unsigned i=0;i<100;++i)launch(f);check(cudaStreamEndCapture(s,&g));check(cudaGraphInstantiate(&e,g,0));for(unsigned i=0;i<3;++i)check(cudaGraphLaunch(e,s));cudaEvent_t a,c;check(cudaEventCreate(&a));check(cudaEventCreate(&c));check(cudaEventRecord(a,s));for(unsigned i=0;i<20;++i)check(cudaGraphLaunch(e,s));check(cudaEventRecord(c,s));check(cudaEventSynchronize(c));check(cudaEventElapsedTime(&timing[f],a,c));timing[f]/=2000.f;cudaEventDestroy(a);cudaEventDestroy(c);cudaGraphExecDestroy(e);cudaGraphDestroy(g);}}
 auto*shared=b.alloc<unsigned short>(m*2560),*shared_gate=b.alloc<unsigned short>(m),*combined=b.alloc<unsigned short>(m*2560);
 auto sy=values(m*2560,.7f),sg=values(m,16.f);check(cudaMemcpyAsync(shared,sy.data(),sy.size()*2,cudaMemcpyHostToDevice,s));check(cudaMemcpyAsync(shared_gate,sg.data(),sg.size()*2,cudaMemcpyHostToDevice,s));
 q38::cuda_moe_combine_shared_bf16(block,shared,shared_gate,combined,m,s,0);
 q38::cuda_hyper_inject_from_projection_bf16(orig,combined,raw,out,m,s,0);
 auto combined_expected=read(out,m*10240,s);
 for(bool graph:{false,true}){auto fused=[&]{q38::cuda_hyper_moe_inject_bf16(orig,block,shared,shared_gate,raw,out,m,s,0);};
  if(!graph)fused();else{cudaGraph_t g;cudaGraphExec_t e;check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));fused();check(cudaStreamEndCapture(s,&g));check(cudaGraphInstantiate(&e,g,0));check(cudaMemsetAsync(out,0xff,m*10240*2,s));check(cudaGraphLaunch(e,s));check(cudaStreamSynchronize(s));cudaGraphExecDestroy(e);cudaGraphDestroy(g);}
  if(combined_expected!=read(out,m*10240,s))throw std::runtime_error("MoE HC finish BF16 parity");
 }
 std::cout<<"{\"M\":"<<m<<",\"bitwise_eager_graph\":true,\"moe_combine_finish_bit_parity\":true,\"baseline_ms\":"<<timing[0]<<",\"fused_ms\":"<<timing[1]<<"}\n";
}
}
int main(){try{cudaStream_t s;check(cudaStreamCreate(&s));for(unsigned m:{1u,2u,3u,5u,7u,512u})fixture(m,s);cudaStreamDestroy(s);std::cout<<"HC_FINISH_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
