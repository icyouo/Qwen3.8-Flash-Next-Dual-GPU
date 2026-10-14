#include "q38/cuda_kernels.h"
#include "q38/cuda_hyper.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t s) { if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s)); }
struct Buffers {
 std::vector<void*> all;
 template<class T>T* alloc(std::size_t n){T*p;check(cudaMalloc(&p,n*sizeof(T)));all.push_back(p);return p;}
 template<class T>T* upload(const std::vector<T>& x){auto*p=alloc<T>(x.size());check(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
 ~Buffers(){for(auto p:all)cudaFree(p);}
};
unsigned short bf16(float x){return __bfloat16_as_ushort(__float2bfloat16_rn(x));}
std::vector<unsigned short> read(unsigned short*p,unsigned n,cudaStream_t s){std::vector<unsigned short>x(n);check(cudaMemcpyAsync(x.data(),p,n*2,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));return x;}
void fixture(unsigned bits,cudaStream_t s){
 Buffers b;constexpr unsigned rows=17,width=2560;
 q38::CudaMatrixViewV1 matrix;matrix.rows=rows;matrix.columns=width;
 if(bits){std::vector<unsigned char>q(rows*width*bits/8);for(unsigned i=0;i<q.size();++i)q[i]=i*73+i/width*17;
  std::vector<unsigned short>sc(rows*20);for(unsigned i=0;i<sc.size();++i)sc[i]=bf16(.007f+(i%19)*.003f);
  matrix.data=b.upload(q);matrix.scales=b.upload(sc);matrix.group_size=128;
  matrix.format=bits==4?q38::DeviceWeightFormatV1::kW4A16SymG128:q38::DeviceWeightFormatV1::kW8A16SymG128;
 }else{std::vector<unsigned short>x(rows*width);for(unsigned i=0;i<x.size();++i)x[i]=bf16(std::sin(i*.017f)*2.f);matrix.data=b.upload(x);matrix.format=q38::DeviceWeightFormatV1::kPreserve;}
 std::vector<int>ids(rows);for(unsigned i=0;i<rows;++i)ids[i]=(i*7)%rows;
 std::vector<unsigned short>w(width);for(unsigned i=0;i<width;++i)w[i]=bf16(std::cos(i*.023f)*.1f);
 auto*token=b.upload(ids);auto*norm=b.upload(w);auto*cache=b.alloc<unsigned short>(rows*width);
 auto*raw=b.alloc<unsigned short>(rows*width);auto*out=b.alloc<unsigned short>(rows*width);
 q38::cuda_embedding_bf16(matrix,token,raw,rows,s,0);
 q38::cuda_qwen38_rmsnorm_bf16(raw,norm,false,out,rows,width,1e-6f,true,s,0);
 auto expected=read(out,rows*width,s);
 q38::cuda_dequantize_matrix_bf16(matrix,cache,s,0);
 q38::cuda_qwen38_rmsnorm_bf16(cache,norm,false,cache,rows,width,1e-6f,true,s,0);
 auto normalized=matrix;normalized.data=cache;normalized.scales=nullptr;normalized.format=q38::DeviceWeightFormatV1::kPreserve;normalized.preserved_f32=false;
 cudaGraph_t graph;cudaGraphExec_t exec;
 check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));q38::cuda_embedding_bf16(normalized,token,out,rows,s,0);check(cudaStreamEndCapture(s,&graph));check(cudaGraphInstantiate(&exec,graph,0));check(cudaGraphLaunch(exec,s));
 if(expected!=read(out,rows*width,s))throw std::runtime_error("MTP normalized embedding cache BF16 parity");
 cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
 auto*hyper=b.alloc<unsigned short>(rows*10240);auto*repeat=b.alloc<unsigned short>(rows*10240);auto*result=b.alloc<unsigned short>(rows*10240);
 std::vector<unsigned short>h(rows*10240);for(unsigned i=0;i<h.size();++i)h[i]=bf16(std::sin(i*.019f)*3.f);check(cudaMemcpy(hyper,h.data(),h.size()*2,cudaMemcpyHostToDevice));
 q38::cuda_hyper_repeat_embedding_bf16(out,repeat,rows,s,0);q38::cuda_add_bf16(hyper,repeat,result,h.size(),s,0);auto sum=read(result,h.size(),s);
 q38::cuda_hyper_add_embedding_bf16(hyper,out,hyper,rows,s,0);
 if(sum!=read(hyper,h.size(),s))throw std::runtime_error("MTP in-place broadcast BF16 parity");
 std::cout<<"MTP input bits="<<bits<<" cache and broadcast bit parity passed\n";
}
}
int main(){try{cudaStream_t s;check(cudaStreamCreate(&s));for(unsigned bits:{0u,4u,8u})fixture(bits,s);cudaStreamDestroy(s);std::cout<<"MTP_INPUT_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
