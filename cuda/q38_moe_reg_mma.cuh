// SM80 m16n8k16 register layout follows NVIDIA PTX ISA section 9.7.16.5.8.
// This path keeps integer quants exact and applies each G128 scale in FP32.
// It does not materialize or round the scaled weights to BF16.
__device__ __forceinline__ unsigned reg_bf16_pair(__nv_bfloat162 value) {
    unsigned bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}
__device__ __forceinline__ void reg_mma_16x8(float* d, const unsigned* a,
                                           unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template<int Bits, bool Tiled>
__device__ __forceinline__ unsigned reg_quant_pair(const void* data,
    unsigned rows, unsigned columns, unsigned expert, unsigned row, unsigned col) {
    if constexpr (Bits == 4)
        return reg_bf16_pair(dequant_weight_pair_w4_bf16<Tiled>(data,
            static_cast<std::uint64_t>(rows) * columns, columns, expert, row, col, 1.0f));
    else {
        const auto offset = (static_cast<std::uint64_t>(expert) * rows + row) * columns + col;
        const auto* q = static_cast<const std::int8_t*>(data);
        return reg_bf16_pair(__floats2bfloat162_rn(float(q[offset]), float(q[offset + 1])));
    }
}
template<int Bits, bool Tiled, unsigned Rows, unsigned Columns>
__device__ __forceinline__ void reg_group_dot(const void* data,
    const std::uint16_t* scales, const std::uint16_t* input,
    Q38ExpertMmqTaskV1 task, unsigned output_base, float* totals) {
    const unsigned lane = threadIdx.x & 31, group = lane >> 2, quad = lane & 3;
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) totals[i] = 0.0f;
    for (unsigned g = 0; g < Columns / 128; ++g) {
        float dots[8] = {};
        #pragma unroll
        for (unsigned step = 0; step < 8; ++step) {
            const unsigned col = g * 128 + step * 16 + quad * 2;
            unsigned a[4] = {};
            if (group < task.valid_rows) {
                const auto* row = reinterpret_cast<const unsigned*>(input +
                    (static_cast<std::uint64_t>(task.packed_begin) + group) * Columns + col);
                a[0] = row[0]; a[2] = row[4];
            }
            if (group + 8 < task.valid_rows) {
                const auto* row = reinterpret_cast<const unsigned*>(input +
                    (static_cast<std::uint64_t>(task.packed_begin) + group + 8) * Columns + col);
                a[1] = row[0]; a[3] = row[4];
            }
            const unsigned r = output_base + group;
            reg_mma_16x8(dots, a,
                reg_quant_pair<Bits,Tiled>(data,Rows,Columns,task.expert,r,col),
                reg_quant_pair<Bits,Tiled>(data,Rows,Columns,task.expert,r,col+8));
            reg_mma_16x8(dots + 4, a,
                reg_quant_pair<Bits,Tiled>(data,Rows,Columns,task.expert,r+8,col),
                reg_quant_pair<Bits,Tiled>(data,Rows,Columns,task.expert,r+8,col+8));
        }
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const unsigned r = output_base + (i / 4) * 8 + quad * 2 + (i & 1);
            const float scale = load_bf16(scales,
                (static_cast<std::uint64_t>(task.expert) * Rows + r) * (Columns / 128) + g);
            totals[i] += dots[i] * scale;
        }
    }
}
template<int Bits, bool Tiled>
__global__ void reg_grouped_gate_up_kernel(const void* data,
    const std::uint16_t* scales, const std::uint16_t* packed_hidden,
    const std::uint32_t* packed_assignment, const float* weights,
    const Q38ExpertMmqTaskV1* tasks, std::uint16_t* mid) {
    const auto task = tasks[blockIdx.y];
    if (!task.valid_rows) return;
    const unsigned warp=threadIdx.x/32,lane=threadIdx.x&31,quad=lane&3,group=lane>>2;
    const unsigned out=(warp/4)*kQ38MoeIntermediate+blockIdx.x*64+(warp&3)*16;
    float totals[8];
    reg_group_dot<Bits,Tiled,2*kQ38MoeIntermediate,kQ38HiddenWidth>(
        data,scales,packed_hidden,task,out,totals);
    __shared__ float result[8*256];
    #pragma unroll
    for (unsigned i=0;i<8;++i) {
        const unsigned row=group+((i%4)/2)*8,col=(i/4)*8+quad*2+(i&1);
        result[warp*256+row*16+col]=totals[i];
    }
    __syncthreads();
    if (warp<4) for (unsigned e=lane;e<256;e+=32) {
        const unsigned row=e/16;
        if (row>=task.valid_rows) continue;
        const float gate=__bfloat162float(__float2bfloat16_rn(result[warp*256+e]));
        const float up=__bfloat162float(__float2bfloat16_rn(result[(warp+4)*256+e]));
        const float active=__bfloat162float(__float2bfloat16_rn(gate/(1.f+expf(-gate))*up));
        const unsigned packed=task.packed_begin+row;
        store_bf16(mid,static_cast<std::uint64_t>(packed)*kQ38MoeIntermediate+
            blockIdx.x*64+warp*16+e%16,weights?active*weights[packed_assignment[packed]]:active);
    }
}
template<int Bits, bool Tiled>
__global__ void reg_grouped_down_kernel(const void* data,
    const std::uint16_t* scales, const std::uint16_t* mid,
    const Q38ExpertMmqTaskV1* tasks, float* output) {
    const auto task=tasks[blockIdx.y];if(!task.valid_rows)return;
    const unsigned warp=threadIdx.x/32,lane=threadIdx.x&31,quad=lane&3,group=lane>>2;
    const unsigned out=blockIdx.x*128+warp*16;
    float totals[8];
    reg_group_dot<Bits,Tiled,kQ38HiddenWidth,kQ38MoeIntermediate>(data,scales,mid,task,out,totals);
    #pragma unroll
    for(unsigned i=0;i<8;++i){
        const unsigned row=group+((i%4)/2)*8,col=(i/4)*8+quad*2+(i&1);
        if(row<task.valid_rows)
            output[(static_cast<std::uint64_t>(task.packed_begin)+row)*kQ38HiddenWidth+out+col]=totals[i];
    }
}
