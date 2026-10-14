// Reuse each compact weight load across independently reduced token rows.
// Each row follows decode_gemv_group128_impl's accumulation order exactly.
template<int Bits, unsigned Batch>
__global__ void small_batch_gemv_group128_kernel(const void* data,
    const std::uint16_t* scales,unsigned rows,unsigned columns,
    const std::uint16_t* input,std::uint16_t* output) {
    extern __shared__ std::uint16_t cached[];
    for(unsigned i=threadIdx.x;i<Batch*columns;i+=blockDim.x)cached[i]=input[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x/32;
    const unsigned row=blockIdx.x*8+warp;if(row>=rows)return;
    const unsigned groups=columns/128;
    float sums[Batch]={};
    for(unsigned group=0;group<groups;++group){
        float scale=lane==0?bf16_load(scales,static_cast<std::uint64_t>(row)*groups+group):0.f;
        scale=__shfl_sync(0xffffffffu,scale,0);
        const unsigned col=group*128+lane*4;
        float q0,q1,q2,q3;
        if constexpr(Bits==8){
            const char4 q=*reinterpret_cast<const char4*>(static_cast<const std::int8_t*>(data)+static_cast<std::uint64_t>(row)*columns+col);
            q0=float(q.x);q1=float(q.y);q2=float(q.z);q3=float(q.w);
        }else{
            const auto packed=*reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(data)+(static_cast<std::uint64_t>(row)*columns+col)/2);
            q0=float(sign_extend_int4(packed&15));q1=float(sign_extend_int4((packed>>4)&15));q2=float(sign_extend_int4((packed>>8)&15));q3=float(sign_extend_int4((packed>>12)&15));
        }
        #pragma unroll
        for(unsigned t=0;t<Batch;++t){
            const auto base=t*columns+col;
            const float dot=bf16_load(cached,base)*q0+bf16_load(cached,base+1)*q1+bf16_load(cached,base+2)*q2+bf16_load(cached,base+3)*q3;
            sums[t]=fmaf(dot,scale,sums[t]);
        }
    }
    #pragma unroll
    for(unsigned t=0;t<Batch;++t){const float sum=warp_sum(sums[t]);if(lane==0)bf16_store(output,static_cast<std::uint64_t>(t)*rows+row,sum);}
}

// Fixed head dimensions expose independent weight loads to the compiler.
// Keep each token's group/FMA order identical to the qualified compact GEMV.
template<unsigned Batch, unsigned Unroll, unsigned Warps = 8>
__global__ void head_pipelined_gemv_kernel(const std::int8_t* __restrict__ data,
    const std::uint16_t* __restrict__ scales,
    const std::uint16_t* __restrict__ input,
    std::uint16_t* __restrict__ output) {
    constexpr unsigned columns=2560,rows=124160,groups=20;
    extern __shared__ std::uint16_t cached[];
    for(unsigned i=threadIdx.x;i<Batch*columns;i+=blockDim.x)cached[i]=input[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x/32;
    const unsigned row=blockIdx.x*Warps+warp;
    if(row>=rows)return;
    float sums[Batch]={};
    #pragma unroll Unroll
    for(unsigned group=0;group<groups;++group){
        float scale=lane==0?bf16_load(scales,static_cast<std::uint64_t>(row)*groups+group):0.f;
        scale=__shfl_sync(0xffffffffu,scale,0);
        const unsigned col=group*128+lane*4;
        const char4 q=*reinterpret_cast<const char4*>(data+static_cast<std::uint64_t>(row)*columns+col);
        const float q0=float(q.x),q1=float(q.y),q2=float(q.z),q3=float(q.w);
        #pragma unroll
        for(unsigned t=0;t<Batch;++t){
            const auto base=t*columns+col;
            const float dot=bf16_load(cached,base)*q0+bf16_load(cached,base+1)*q1+bf16_load(cached,base+2)*q2+bf16_load(cached,base+3)*q3;
            sums[t]=fmaf(dot,scale,sums[t]);
        }
    }
    #pragma unroll
    for(unsigned t=0;t<Batch;++t){const float sum=warp_sum(sums[t]);if(lane==0)bf16_store(output,static_cast<std::uint64_t>(t)*rows+row,Batch==1?0.f+1.f*sum:sum);}
}

bool try_head_pipelined_gemv(const CudaMatrixViewV1& matrix,
    const std::uint16_t* input,std::uint16_t* output,unsigned batch,void* stream){
    const auto* value=std::getenv("Q38_CUDA_HEAD_PIPELINED");
    const int unroll=value?std::atoi(value):0;
    if((unroll!=2 && unroll!=4 && unroll!=10 && unroll!=20) ||
       matrix.rows!=124160 || matrix.columns!=2560 || matrix.group_size!=128 ||
       matrix.format!=DeviceWeightFormatV1::kW8A16SymG128 ||
       (batch!=1 && batch!=2 && batch!=3 && batch!=5 && batch!=7))return false;
    auto launch=[&](auto batch_tag,auto unroll_tag){
        constexpr unsigned Batch=decltype(batch_tag)::value,Unroll=decltype(unroll_tag)::value;
        const auto* setting=std::getenv("Q38_CUDA_HEAD_WARPS");
        const int warps=setting && std::strcmp(setting,"auto")==0 ? (Batch==1?4:16) : (setting?std::atoi(setting):8);
        if(warps!=4 && warps!=8 && warps!=16)throw std::invalid_argument("head warps must be 4/8/16");
        auto schedule=[&](auto w){constexpr unsigned Warps=decltype(w)::value;
            head_pipelined_gemv_kernel<Batch,Unroll,Warps><<<(124160+Warps-1)/Warps,32*Warps,Batch*2560*2,reinterpret_cast<cudaStream_t>(stream)>>>(
                static_cast<const std::int8_t*>(matrix.data),static_cast<const std::uint16_t*>(matrix.scales),input,output);
        };
        if(warps==4)schedule(std::integral_constant<unsigned,4>{});
        else if(warps==16)schedule(std::integral_constant<unsigned,16>{});
        else schedule(std::integral_constant<unsigned,8>{});
    };
    auto config=[&](auto b){
        if(unroll==2)launch(b,std::integral_constant<unsigned,2>{});
        else if(unroll==4)launch(b,std::integral_constant<unsigned,4>{});
        else if(unroll==10)launch(b,std::integral_constant<unsigned,10>{});
        else launch(b,std::integral_constant<unsigned,20>{});
    };
    if(batch==1)config(std::integral_constant<unsigned,1>{});
    else if(batch==2)config(std::integral_constant<unsigned,2>{});
    else if(batch==3)config(std::integral_constant<unsigned,3>{});
    else if(batch==5)config(std::integral_constant<unsigned,5>{});
    else config(std::integral_constant<unsigned,7>{});
    check(cudaPeekAtLastError(),"pipelined head GEMV");return true;
}
