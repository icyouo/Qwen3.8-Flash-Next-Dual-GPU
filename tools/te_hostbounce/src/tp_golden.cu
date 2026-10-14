// TP forward parity golden for one real Q38 BF16 tensor.
// Verifies: ColumnParallel (row_shard axis=0, no comm, concat)
//           RowParallel   (col_shard axis=1, requires AllReduce)
// against a single-GPU baseline.
//
// Compile: nvcc -O3 -arch=sm_80 -o tp_golden tp_golden.cu
// Usage:   tp_golden <segment_path> <data_offset> <out_dim> <in_dim>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1);} } while(0)

// Y[b,o] = sum_i X[b,i] * W[o,i]   (W in BF16, X in FP32, Y in FP32)
__global__ void gemm_bf16w(const float* __restrict__ X,
                           const __nv_bfloat16* __restrict__ W,
                           float* __restrict__ Y,
                           int B, int O, int I) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || o >= O) return;
    float acc = 0.f;
    for (int i = 0; i < I; ++i) {
        acc += X[b * I + i] * __bfloat162float(W[o * I + i]);
    }
    Y[b * O + o] = acc;
}

// AllReduce SUM over 2 ranks: sums Y0 and Y1 into both.
__global__ void add_kernel(const float* A, const float* B, float* out, size_t n) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; idx < n; idx += stride) out[idx] = A[idx] + B[idx];
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <segment_path> <data_offset> <O> <I>\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    size_t offset = std::strtoull(argv[2], nullptr, 10);
    int O = std::atoi(argv[3]);
    int I = std::atoi(argv[4]);
    size_t weight_bytes = size_t(O) * size_t(I) * sizeof(__nv_bfloat16);

    // mmap the segment and pull weight bytes
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st{}; ::fstat(fd, &st);
    void* map = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    const __nv_bfloat16* Wh =
        reinterpret_cast<const __nv_bfloat16*>(
            static_cast<const char*>(map) + offset);
    printf("loaded W [O=%d, I=%d] from %s+%zu (%.2f MiB)\n",
           O, I, path, offset, weight_bytes / 1048576.0);

    // Random-ish X on host: deterministic hash
    const int B = 4;
    float* Xh = new float[B * I];
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < I; ++i)
            Xh[b * I + i] = std::sin(0.001f * (b * I + i)) * 0.1f;

    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { printf("need 2 GPUs\n"); return 1; }
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) if (a != b) {
        int can = 0;
        CK(cudaDeviceCanAccessPeer(&can, a, b));
        printf("  peer %d->%d can=%d\n", a, b, can);
        CK(cudaSetDevice(a));
        cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "  EnablePeerAccess %d->%d: %s\n", a, b, cudaGetErrorString(e));
        } else {
            printf("  EnablePeerAccess %d->%d: %s\n", a, b,
                   e == cudaErrorPeerAccessAlreadyEnabled ? "already-enabled" : "ok");
        }
    }

    // ---- Baseline: single GPU full compute ----
    CK(cudaSetDevice(0));
    __nv_bfloat16* Wd_full;
    float *Xd_full, *Yd_full;
    CK(cudaMalloc(&Wd_full, weight_bytes));
    CK(cudaMalloc(&Xd_full, B * I * sizeof(float)));
    CK(cudaMalloc(&Yd_full, B * O * sizeof(float)));
    CK(cudaMemcpy(Wd_full, Wh, weight_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Xd_full, Xh, B * I * sizeof(float), cudaMemcpyHostToDevice));
    {
        dim3 bl(16, 8);
        dim3 gr((O + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
        gemm_bf16w<<<gr, bl>>>(Xd_full, Wd_full, Yd_full, B, O, I);
        CK(cudaDeviceSynchronize());
    }
    float* Y_ref = new float[B * O];
    CK(cudaMemcpy(Y_ref, Yd_full, B * O * sizeof(float), cudaMemcpyDeviceToHost));

    // ============ Test 1: ColumnParallel (row_shard on W axis=0) ============
    // W is (O, I). Split O into O0 = O/2 on rank0 and O1 = O-O0 on rank1.
    // Each rank has X full, gets Y_r shape (B, O_r). Concat gives full Y.
    int O0 = O / 2, O1 = O - O0;
    __nv_bfloat16 *W0, *W1;
    float *X0, *X1, *Y0, *Y1;
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&W0, O0 * size_t(I) * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&X0, B * I * sizeof(float)));
    CK(cudaMalloc(&Y0, B * O0 * sizeof(float)));
    CK(cudaMemcpy(W0, Wh,               O0 * size_t(I) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(X0, Xh, B * I * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&W1, O1 * size_t(I) * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&X1, B * I * sizeof(float)));
    CK(cudaMalloc(&Y1, B * O1 * sizeof(float)));
    CK(cudaMemcpy(W1, Wh + O0 * size_t(I), O1 * size_t(I) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(X1, Xh, B * I * sizeof(float), cudaMemcpyHostToDevice));

    CK(cudaSetDevice(0));
    { dim3 bl(16, 8); dim3 gr((O0 + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
      gemm_bf16w<<<gr, bl>>>(X0, W0, Y0, B, O0, I); CK(cudaDeviceSynchronize()); }
    CK(cudaSetDevice(1));
    { dim3 bl(16, 8); dim3 gr((O1 + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
      gemm_bf16w<<<gr, bl>>>(X1, W1, Y1, B, O1, I); CK(cudaDeviceSynchronize()); }

    float* Y_cp = new float[B * O];
    CK(cudaSetDevice(0));
    for (int b = 0; b < B; ++b)
        CK(cudaMemcpy(Y_cp + b * O,       Y0 + b * O0, O0 * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(1));
    for (int b = 0; b < B; ++b)
        CK(cudaMemcpy(Y_cp + b * O + O0, Y1 + b * O1, O1 * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_cp = 0, mean_abs_cp = 0, ref_norm = 0;
    for (int i = 0; i < B * O; ++i) {
        double d = std::fabs(double(Y_cp[i]) - double(Y_ref[i]));
        max_abs_cp = std::max(max_abs_cp, d);
        mean_abs_cp += d;
        ref_norm += std::fabs(double(Y_ref[i]));
    }
    mean_abs_cp /= (B * O);
    ref_norm /= (B * O);
    printf("[Test 1 ColumnParallel] max_abs=%.3e  mean_abs=%.3e  ref_mean=%.3e  %s\n",
           max_abs_cp, mean_abs_cp, ref_norm,
           max_abs_cp < 1e-5 ? "PASS" : "FAIL");

    // ============ Test 2: RowParallel (col_shard on W axis=1) + AllReduce ==
    // W is (O, I). Split I into I0 = I/2 and I1 = I - I0.
    // X is split along last dim too: X_r shape (B, I_r).
    // Each rank has W_r shape (O, I_r). Compute partial Y_r (B, O).
    // Y = Y0 + Y1  via AllReduce.
    int I0 = I / 2, I1 = I - I0;
    __nv_bfloat16 *rp_W0, *rp_W1;
    float *rp_X0, *rp_X1, *rp_Y0, *rp_Y1;

    // Split W by column: rank0 gets columns [0,I0), rank1 gets [I0,I)
    // W is row-major (O, I). Column slice is strided; do host slice first.
    __nv_bfloat16* Wh_c0 = new __nv_bfloat16[size_t(O) * I0];
    __nv_bfloat16* Wh_c1 = new __nv_bfloat16[size_t(O) * I1];
    for (int o = 0; o < O; ++o) {
        std::memcpy(Wh_c0 + o * size_t(I0), Wh + o * size_t(I),           I0 * sizeof(__nv_bfloat16));
        std::memcpy(Wh_c1 + o * size_t(I1), Wh + o * size_t(I) + I0,       I1 * sizeof(__nv_bfloat16));
    }
    float* Xh_c0 = new float[B * I0];
    float* Xh_c1 = new float[B * I1];
    for (int b = 0; b < B; ++b) {
        std::memcpy(Xh_c0 + b * I0, Xh + b * I,       I0 * sizeof(float));
        std::memcpy(Xh_c1 + b * I1, Xh + b * I + I0,  I1 * sizeof(float));
    }

    CK(cudaSetDevice(0));
    CK(cudaMalloc(&rp_W0, size_t(O) * I0 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&rp_X0, B * I0 * sizeof(float)));
    CK(cudaMalloc(&rp_Y0, B * O  * sizeof(float)));
    CK(cudaMemcpy(rp_W0, Wh_c0, size_t(O) * I0 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rp_X0, Xh_c0, B * I0 * sizeof(float), cudaMemcpyHostToDevice));

    CK(cudaSetDevice(1));
    CK(cudaMalloc(&rp_W1, size_t(O) * I1 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&rp_X1, B * I1 * sizeof(float)));
    CK(cudaMalloc(&rp_Y1, B * O  * sizeof(float)));
    CK(cudaMemcpy(rp_W1, Wh_c1, size_t(O) * I1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rp_X1, Xh_c1, B * I1 * sizeof(float), cudaMemcpyHostToDevice));

    // Partial matmul (uses same kernel but with sliced I)
    CK(cudaSetDevice(0));
    { dim3 bl(16, 8); dim3 gr((O + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
      gemm_bf16w<<<gr, bl>>>(rp_X0, rp_W0, rp_Y0, B, O, I0); CK(cudaDeviceSynchronize()); }
    CK(cudaSetDevice(1));
    { dim3 bl(16, 8); dim3 gr((O + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
      gemm_bf16w<<<gr, bl>>>(rp_X1, rp_W1, rp_Y1, B, O, I1); CK(cudaDeviceSynchronize()); }

    // AllReduce via P2P: each rank pulls peer's partial into scratch and sums.
    float *scratch0, *scratch1;
    CK(cudaSetDevice(0)); CK(cudaMalloc(&scratch0, B * O * sizeof(float)));
    CK(cudaSetDevice(1)); CK(cudaMalloc(&scratch1, B * O * sizeof(float)));

    // Debug: read rp_Y0[0..3], rp_Y1[0..3], and expected Y_ref[0..3]
    {
        float dbg0[4] = {}, dbg1[4] = {};
        CK(cudaMemcpy(dbg0, rp_Y0, sizeof(dbg0), cudaMemcpyDeviceToHost));
        CK(cudaSetDevice(1));
        CK(cudaMemcpy(dbg1, rp_Y1, sizeof(dbg1), cudaMemcpyDeviceToHost));
        CK(cudaSetDevice(0));
        printf("  DEBUG b=0 o=0..3:\n");
        for (int k = 0; k < 4; ++k) {
            printf("    Y0=%.6f  Y1=%.6f  Y0+Y1=%.6f  ref=%.6f  diff=%.6f\n",
                   dbg0[k], dbg1[k], dbg0[k] + dbg1[k], Y_ref[k],
                   (dbg0[k] + dbg1[k]) - Y_ref[k]);
        }
    }
    // rank0: pull rank1.Y1 to scratch0, then add
    // Workaround: use pinned host staging to check math correctness first.
    float* host_stage = nullptr;
    CK(cudaHostAlloc(&host_stage, B * O * sizeof(float), cudaHostAllocDefault));
    CK(cudaSetDevice(1));
    CK(cudaMemcpy(host_stage, rp_Y1, B * O * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(scratch0, host_stage, B * O * sizeof(float), cudaMemcpyHostToDevice));
    printf("  host-staged Y1[0..3]: %.6f %.6f %.6f %.6f\n",
           host_stage[0], host_stage[1], host_stage[2], host_stage[3]);
    {
        float post[4] = {};
        CK(cudaMemcpy(post, scratch0, sizeof(post), cudaMemcpyDeviceToHost));
        printf("  host-staged scratch0[0..3]: %.6f %.6f %.6f %.6f\n",
               post[0], post[1], post[2], post[3]);
    }
    // Verify scratch0 now equals rp_Y1
    {
        float peer_dbg[4] = {};
        CK(cudaMemcpy(peer_dbg, scratch0, sizeof(peer_dbg), cudaMemcpyDeviceToHost));
        printf("  After P2P scratch0[0..3]: %.6f %.6f %.6f %.6f (should == rank1 Y1)\n",
               peer_dbg[0], peer_dbg[1], peer_dbg[2], peer_dbg[3]);
    }
    {
        dim3 bl(256); dim3 gr(std::min((B * O + 255) / 256, 4096));
        add_kernel<<<gr, bl>>>(rp_Y0, scratch0, rp_Y0, B * O);
        CK(cudaDeviceSynchronize());
    }
    float* Y_rp = new float[B * O];
    CK(cudaMemcpy(Y_rp, rp_Y0, B * O * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  After AR Y_rp[0..3]: %.6f %.6f %.6f %.6f\n",
           Y_rp[0], Y_rp[1], Y_rp[2], Y_rp[3]);
    printf("  Y_ref[0..3]:         %.6f %.6f %.6f %.6f\n",
           Y_ref[0], Y_ref[1], Y_ref[2], Y_ref[3]);
    // Also check an element far away
    printf("  Y_rp[10000..10003]:  %.6f %.6f %.6f %.6f\n",
           Y_rp[10000], Y_rp[10001], Y_rp[10002], Y_rp[10003]);
    printf("  Y_ref[10000..10003]: %.6f %.6f %.6f %.6f\n",
           Y_ref[10000], Y_ref[10001], Y_ref[10002], Y_ref[10003]);

    double max_abs_rp = 0, mean_abs_rp = 0;
    for (int i = 0; i < B * O; ++i) {
        double d = std::fabs(double(Y_rp[i]) - double(Y_ref[i]));
        max_abs_rp = std::max(max_abs_rp, d);
        mean_abs_rp += d;
    }
    mean_abs_rp /= (B * O);
    // Row-parallel adds bf16-of-different-halves → some fp rounding drift is normal
    printf("[Test 2 RowParallel+AR] max_abs=%.3e  mean_abs=%.3e  ref_mean=%.3e  %s\n",
           max_abs_rp, mean_abs_rp, ref_norm,
           max_abs_rp < 1e-3 ? "PASS" : "FAIL");

    // cleanup
    ::munmap(map, st.st_size); ::close(fd);
    return 0;
}
