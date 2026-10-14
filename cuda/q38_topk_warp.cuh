// Stable top-k for the 512-expert router. One warp owns one token;
// register-held logits avoid the 45 CTA barriers of full bitonic sorting.
__global__ void topk_router_512_warp_kernel(const std::uint16_t* logits,
    std::int32_t* ids,float* weights,std::uint32_t top_k){
    const unsigned lane=threadIdx.x;const unsigned token=blockIdx.x;
    float values[16],numerators[32];
    #pragma unroll
    for(unsigned i=0;i<16;++i)values[i]=bf16_load(logits,token*512ull+i*32+lane);
    float maximum=0.f,denominator=0.f;
    for(unsigned rank=0;rank<top_k;++rank){
        float best=-3.402823466e+38F;int id=2147483647;
        #pragma unroll
        for(unsigned i=0;i<16;++i){const int candidate=i*32+lane;
            if(values[i]>best||(values[i]==best&&candidate<id)){best=values[i];id=candidate;}}
        #pragma unroll
        for(unsigned mask=16;mask;mask>>=1){
            const float other=__shfl_xor_sync(0xffffffff,best,mask);
            const int other_id=__shfl_xor_sync(0xffffffff,id,mask);
            if(other>best||(other==best&&other_id<id)){best=other;id=other_id;}
        }
        if(rank==0)maximum=best;
        if(lane==0){ids[token*top_k+rank]=id;numerators[rank]=expf(best-maximum);denominator+=numerators[rank];}
        if((unsigned(id)&31u)==lane)values[unsigned(id)/32]=-INFINITY;
    }
    if(lane==0)for(unsigned rank=0;rank<top_k;++rank)weights[token*top_k+rank]=numerators[rank]/denominator;
}
