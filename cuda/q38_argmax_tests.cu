#include "q38/cuda_kernels.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
std::uint16_t bf16(float x){return __bfloat16_as_ushort(__float2bfloat16_rn(x));}
}
int main(){try{
 cudaStream_t stream;check(cudaStreamCreate(&stream));
 constexpr unsigned rows=7,max_count=124160;
 std::uint16_t* data;float* partial;int *ids,*old,*now;
 check(cudaMalloc(&data,rows*max_count*2));check(cudaMalloc(&partial,rows*128*4));
 check(cudaMalloc(&ids,rows*128*4));check(cudaMalloc(&old,rows*4));check(cudaMalloc(&now,rows*4));
 for(unsigned count:{1u,31u,257u,32769u,max_count}){
  std::vector<std::uint16_t> host(rows*count);
  for(unsigned row=0;row<rows;++row){auto first=host.begin()+row*count;
   std::fill_n(first,count,bf16(-std::numeric_limits<float>::infinity()));
   if(row==0)std::fill_n(first,count,bf16(std::numeric_limits<float>::quiet_NaN()));
   if(row==1)std::fill_n(first,count,bf16(-2.f));
   if(row==2){first[count-1]=bf16(7.f);first[count/2]=bf16(7.f);first[0]=bf16(std::numeric_limits<float>::quiet_NaN());}
   if(row==3){first[count-1]=bf16(std::numeric_limits<float>::infinity());first[0]=first[count-1];}
   if(row==4)for(unsigned i=0;i<count;++i)first[i]=bf16(i%2?-0.f:0.f);
   if(row==5)first[count-1]=bf16(-1e30f);
  }
  check(cudaMemcpyAsync(data,host.data(),host.size()*2,cudaMemcpyHostToDevice,stream));
  auto launch=[&]{for(unsigned row=0;row<rows;++row)q38::cuda_argmax_bf16(data+row*count,count,old+row,stream,0);
   q38::cuda_argmax_bf16_batch(data,count,rows,partial,ids,now,stream,0);};
  for(bool graph:{false,true}){
   if(!graph)launch();else{cudaGraph_t g;cudaGraphExec_t e;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();check(cudaStreamEndCapture(stream,&g));check(cudaGraphInstantiate(&e,g,0));
    check(cudaMemsetAsync(now,0x7f,rows*4,stream));check(cudaMemsetAsync(ids,0x7f,rows*128*4,stream));check(cudaGraphLaunch(e,stream));check(cudaStreamSynchronize(stream));cudaGraphExecDestroy(e);cudaGraphDestroy(g);}
   std::vector<int>a(rows),b(rows);check(cudaMemcpyAsync(a.data(),old,rows*4,cudaMemcpyDeviceToHost,stream));check(cudaMemcpyAsync(b.data(),now,rows*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
   if(a!=b)throw std::runtime_error("stable argmax parity failed");
   if(b[0]!=-1||b[1]!=0||b[3]!=0||b[4]!=0||b[5]!=int(count-1)||b[6]!=-1)throw std::runtime_error("argmax edge-case contract failed");
  }
  std::cout<<"{\"count\":"<<count<<",\"ties_nan_infinity_signed_zero_graph_parity\":true}\n";
 }
 cudaFree(data);cudaFree(partial);cudaFree(ids);cudaFree(old);cudaFree(now);cudaStreamDestroy(stream);
 std::cout<<"ARGMAX_PASS\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
