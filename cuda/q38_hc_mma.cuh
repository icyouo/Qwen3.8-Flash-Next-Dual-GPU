// Direct BF16 m16n8k16 tensor-core HC projection. No weight conversion.
// Small batches retain the GEMM -> BF16 -> activation rounding boundary.
__device__ __forceinline__ float hc_mma_activation(float value,int activation,float divisor,float scale){
 if(activation==int(CudaHcActivation::kNone))return value;
 const float x=__bfloat162float(__float2bfloat16_rn(value))/divisor;
 const float sig=1.f/(1.f+expf(-x));
 return (activation==int(CudaHcActivation::kSilu)?x*sig:sig)*scale;
}
template<bool Partial>
__global__ void hc_mma_projection_kernel(const std::uint16_t* weight,const std::uint16_t* input,
 std::uint16_t* output,float* partial,unsigned rows,unsigned columns,unsigned tokens,unsigned splits,
 int activation,float divisor,float scale){
 const unsigned lane=threadIdx.x&31,warp=threadIdx.x/32,group=lane/4,quad=lane%4;
 const unsigned base=blockIdx.x*32+warp*8;
 float d[4]={};
 const unsigned steps=columns/16/splits,first=blockIdx.y*steps;
 for(unsigned step=first;step<first+steps;++step){
  const unsigned col=step*16+quad*2;
  unsigned a[4]={};
  if(group<tokens){const auto* p=reinterpret_cast<const unsigned*>(input+group*columns+col);a[0]=p[0];a[2]=p[4];}
  // tokens <=7, so the second 8-row part of A remains exactly zero.
  const auto* w=reinterpret_cast<const unsigned*>(weight+(base+group)*columns+col);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(w[0]),"r"(w[4]));
 }
 if(group<tokens)for(unsigned i=0;i<2;++i){
  const unsigned n=base+quad*2+i;
  if(n>=rows)continue;
  if constexpr(Partial)partial[(blockIdx.y*tokens+group)*rows+n]=d[i];
  else reinterpret_cast<__nv_bfloat16*>(output)[group*rows+n]=__float2bfloat16_rn(hc_mma_activation(d[i],activation,divisor,scale));
 }
}
__global__ void hc_mma_reduce_kernel(const float* partial,std::uint16_t* output,unsigned rows,unsigned tokens,unsigned splits,int activation,float divisor,float scale){
 const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=rows*tokens)return;
 float sum=0.f;for(unsigned s=0;s<splits;++s)sum+=partial[s*rows*tokens+i];
 reinterpret_cast<__nv_bfloat16*>(output)[i]=__float2bfloat16_rn(hc_mma_activation(sum,activation,divisor,scale));
}

// Compute four stream gates and their original rank-order FP32 mixture in
// one CTA. The projection and sigmoid each retain their BF16 boundaries.
__global__ void hc_mma_mix_kernel(const std::uint16_t* weight,const std::uint16_t* low,
 const std::uint16_t* normalized,std::uint16_t* mixed,unsigned tokens){
 const unsigned lane=threadIdx.x&31,warp=threadIdx.x/32,group=lane/4,quad=lane%4;
 const unsigned hidden=blockIdx.x*8,base=warp*2560+hidden;
 float d[4]={};
 for(unsigned step=0;step<20;++step){const unsigned col=step*16+quad*2;unsigned a[4]={};
  if(group<tokens){const auto*p=reinterpret_cast<const unsigned*>(low+group*320+col);a[0]=p[0];a[2]=p[4];}
  const auto*w=reinterpret_cast<const unsigned*>(weight+(base+group)*320+col);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
   : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(w[0]),"r"(w[4]));
 }
 __shared__ float partial[7][4][8];
 if(group<tokens)for(unsigned j=0;j<2;++j){const unsigned n=quad*2+j;
  float gate=__bfloat162float(__float2bfloat16_rn(hc_mma_activation(d[j],int(CudaHcActivation::kSigmoid),1.f,1.f)));
  partial[group][warp][n]=gate*bf16_load(normalized,group*10240+warp*2560+hidden+n);
 }
 __syncthreads();
 if(threadIdx.x<tokens*8){const unsigned t=threadIdx.x/8,n=threadIdx.x%8;float sum=0.f;
  for(unsigned j=0;j<4;++j)sum+=partial[t][j][n];bf16_store(mixed,t*2560+hidden+n,sum/4.f);
 }
}
