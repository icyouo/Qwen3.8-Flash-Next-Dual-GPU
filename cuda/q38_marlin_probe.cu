#include "q38/cuda_marlin.h"
extern "C" void q38_test_repack(void* data,void* scales,int n,int k,int bits,void* scratch,void* scale_scratch,void* stream){q38::cuda_marlin_repack_expert(data,scales,n,k,bits,scratch,scale_scratch,stream);}
extern "C" void q38_test_moe(const void* gate,const void* gs,const void* down,const void* ds,int bits,
 const unsigned short* input,const void* tasks,int max_tasks,int routes,unsigned short* mid,float* output,
 int* sorted,int* experts,int* padded,int* locks,unsigned short* gout,unsigned short* dout,float* tmp,void* stream){
 q38::DeviceTensorV1 gd,dd;gd.shape={32,1280,2560};dd.shape={32,2560,640};
 gd.format=dd.format=bits==4?q38::DeviceWeightFormatV1::kW4A16SymG128:q38::DeviceWeightFormatV1::kW8A16SymG128;
 q38::CudaTensorViewV1 g{&gd,gate,gs,q38::CudaWeightLayoutV1::kMarlinG128},d{&dd,down,ds,q38::CudaWeightLayoutV1::kMarlinG128};
 q38::cuda_marlin_moe(g,d,input,(const q38::Q38ExpertMmqTaskV1*)tasks,max_tasks,routes,mid,output,
 {sorted,experts,padded,locks,gout,dout,tmp},stream);
}
#include "q38/cuda_kernels.h"
extern "C" void q38_test_topk(const unsigned short* logits,int* ids,float* weights,int m,void* stream){q38::cuda_topk_router_bf16(logits,ids,weights,m,512,10,true,stream,0);}
