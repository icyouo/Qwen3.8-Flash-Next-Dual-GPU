// W8A16-sym-g128 TP golden with host-bounce AllReduce (no P2P required).
// Usage: tp_golden_w8 <seg> <data_off> <scale_off> <O> <I> <group>
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
#include <algorithm>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

__device__ __forceinline__ float bf16_to_f(const __nv_bfloat16* p, size_t i){
  return __bfloat162float(p[i]);
}

// Y[b,o] = sum_i X[b,i] * (int8(W[o,i]) * scale[o, i/G])
__global__ void gemm_w8(const float* __restrict__ X,
                        const int8_t* __restrict__ W,
                        const __nv_bfloat16* __restrict__ S,
                        float* __restrict__ Y,
                        int B, int O, int I, int G) {
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || o >= O) return;
  float acc = 0.f;
  const int groups = I / G;
  for (int i = 0; i < I; ++i) {
    float scale = bf16_to_f(S, size_t(o) * groups + (i / G));
    acc += X[b * I + i] * (float(W[size_t(o) * I + i]) * scale);
  }
  Y[b * O + o] = acc;
}

__global__ void add_kernel(const float* A, const float* B, float* out, size_t n) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) out[idx] = A[idx] + B[idx];
}

static void host_bounce_sum(float* y0_dev, float* y1_dev, int nfloats) {
  float* stage = nullptr;
  CK(cudaHostAlloc(&stage, size_t(nfloats) * sizeof(float), cudaHostAllocDefault));
  float* scratch0 = nullptr;
  CK(cudaSetDevice(0));
  CK(cudaMalloc(&scratch0, size_t(nfloats) * sizeof(float)));
  CK(cudaSetDevice(1));
  CK(cudaMemcpy(stage, y1_dev, size_t(nfloats) * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(scratch0, stage, size_t(nfloats) * sizeof(float), cudaMemcpyHostToDevice));
  dim3 bl(256);
  dim3 gr(std::min((nfloats + 255) / 256, 4096));
  add_kernel<<<gr, bl>>>(y0_dev, scratch0, y0_dev, nfloats);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(scratch0));
  CK(cudaFreeHost(stage));
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s <seg> <data_off> <scale_off> <O> <I> <group>\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  size_t data_off = std::strtoull(argv[2], nullptr, 10);
  size_t scale_off = std::strtoull(argv[3], nullptr, 10);
  int O = std::atoi(argv[4]);
  int I = std::atoi(argv[5]);
  int G = std::atoi(argv[6]);
  if (I % G) { fprintf(stderr, "I must be multiple of group\n"); return 1; }
  size_t w_bytes = size_t(O) * size_t(I); // int8
  size_t s_bytes = size_t(O) * size_t(I / G) * sizeof(__nv_bfloat16);

  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  struct stat st{}; ::fstat(fd, &st);
  void* map = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap"); return 1; }
  const int8_t* Wh = reinterpret_cast<const int8_t*>(static_cast<char*>(map) + data_off);
  const __nv_bfloat16* Sh = reinterpret_cast<const __nv_bfloat16*>(static_cast<char*>(map) + scale_off);
  printf("W8 [O=%d I=%d G=%d] data@%zu scales@%zu (%.2f MiB weights)\n",
         O, I, G, data_off, scale_off, w_bytes / 1048576.0);

  const int B = 4;
  float* Xh = new float[B * I];
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < I; ++i)
      Xh[b * I + i] = std::sin(0.001f * (b * I + i)) * 0.1f;

  int ndev = 0; CK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) { printf("need 2 GPUs\n"); return 1; }

  // ---- Baseline on GPU0 ----
  CK(cudaSetDevice(0));
  int8_t* Wd; __nv_bfloat16* Sd; float *Xd, *Yd;
  CK(cudaMalloc(&Wd, w_bytes));
  CK(cudaMalloc(&Sd, s_bytes));
  CK(cudaMalloc(&Xd, B * I * sizeof(float)));
  CK(cudaMalloc(&Yd, B * O * sizeof(float)));
  CK(cudaMemcpy(Wd, Wh, w_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(Sd, Sh, s_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(Xd, Xh, B * I * sizeof(float), cudaMemcpyHostToDevice));
  { dim3 bl(16,8); dim3 gr((O+bl.x-1)/bl.x, (B+bl.y-1)/bl.y);
    gemm_w8<<<gr,bl>>>(Xd,Wd,Sd,Yd,B,O,I,G); CK(cudaDeviceSynchronize()); }
  float* Y_ref = new float[B * O];
  CK(cudaMemcpy(Y_ref, Yd, B * O * sizeof(float), cudaMemcpyDeviceToHost));

  // ---- Test 1 ColumnParallel: split rows ----
  int O0 = O / 2, O1 = O - O0;
  int groups = I / G;
  CK(cudaSetDevice(0));
  int8_t *W0; __nv_bfloat16 *S0; float *X0, *Y0;
  CK(cudaMalloc(&W0, size_t(O0)*I));
  CK(cudaMalloc(&S0, size_t(O0)*groups*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&X0, B*I*sizeof(float)));
  CK(cudaMalloc(&Y0, B*O0*sizeof(float)));
  CK(cudaMemcpy(W0, Wh, size_t(O0)*I, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(S0, Sh, size_t(O0)*groups*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(X0, Xh, B*I*sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaSetDevice(1));
  int8_t *W1; __nv_bfloat16 *S1; float *X1, *Y1;
  CK(cudaMalloc(&W1, size_t(O1)*I));
  CK(cudaMalloc(&S1, size_t(O1)*groups*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&X1, B*I*sizeof(float)));
  CK(cudaMalloc(&Y1, B*O1*sizeof(float)));
  CK(cudaMemcpy(W1, Wh + size_t(O0)*I, size_t(O1)*I, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(S1, Sh + size_t(O0)*groups, size_t(O1)*groups*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(X1, Xh, B*I*sizeof(float), cudaMemcpyHostToDevice));

  CK(cudaSetDevice(0));
  { dim3 bl(16,8); dim3 gr((O0+bl.x-1)/bl.x,(B+bl.y-1)/bl.y);
    gemm_w8<<<gr,bl>>>(X0,W0,S0,Y0,B,O0,I,G); CK(cudaDeviceSynchronize()); }
  CK(cudaSetDevice(1));
  { dim3 bl(16,8); dim3 gr((O1+bl.x-1)/bl.x,(B+bl.y-1)/bl.y);
    gemm_w8<<<gr,bl>>>(X1,W1,S1,Y1,B,O1,I,G); CK(cudaDeviceSynchronize()); }

  float* Y_cp = new float[B*O];
  CK(cudaSetDevice(0));
  for (int b=0;b<B;++b) CK(cudaMemcpy(Y_cp+b*O, Y0+b*O0, O0*sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(1));
  for (int b=0;b<B;++b) CK(cudaMemcpy(Y_cp+b*O+O0, Y1+b*O1, O1*sizeof(float), cudaMemcpyDeviceToHost));
  double max_cp=0, mean_cp=0, ref_mean=0;
  for (int i=0;i<B*O;++i){
    double d=std::fabs(double(Y_cp[i])-double(Y_ref[i]));
    max_cp=std::max(max_cp,d); mean_cp+=d; ref_mean+=std::fabs(double(Y_ref[i]));
  }
  mean_cp/=(B*O); ref_mean/=(B*O);
  printf("[W8 ColumnParallel] max_abs=%.3e mean_abs=%.3e ref_mean=%.3e %s\n",
         max_cp, mean_cp, ref_mean, max_cp<1e-5?"PASS":"FAIL");

  // ---- Test 2 RowParallel: split cols (group-aligned) + host-bounce AR ----
  int I0 = I / 2, I1 = I - I0;
  if (I0 % G || I1 % G) { fprintf(stderr, "col split not group-aligned\n"); return 1; }
  int g0 = I0 / G, g1 = I1 / G;
  int8_t* Wh0 = new int8_t[size_t(O)*I0];
  int8_t* Wh1 = new int8_t[size_t(O)*I1];
  __nv_bfloat16* Sh0 = new __nv_bfloat16[size_t(O)*g0];
  __nv_bfloat16* Sh1 = new __nv_bfloat16[size_t(O)*g1];
  float* Xh0 = new float[B*I0];
  float* Xh1 = new float[B*I1];
  for (int o=0;o<O;++o){
    std::memcpy(Wh0+size_t(o)*I0, Wh+size_t(o)*I, size_t(I0));
    std::memcpy(Wh1+size_t(o)*I1, Wh+size_t(o)*I+I0, size_t(I1));
    std::memcpy(Sh0+size_t(o)*g0, Sh+size_t(o)*groups, size_t(g0)*sizeof(__nv_bfloat16));
    std::memcpy(Sh1+size_t(o)*g1, Sh+size_t(o)*groups+g0, size_t(g1)*sizeof(__nv_bfloat16));
  }
  for (int b=0;b<B;++b){
    std::memcpy(Xh0+b*I0, Xh+b*I, I0*sizeof(float));
    std::memcpy(Xh1+b*I1, Xh+b*I+I0, I1*sizeof(float));
  }

  CK(cudaSetDevice(0));
  int8_t *rpW0; __nv_bfloat16 *rpS0; float *rpX0, *rpY0;
  CK(cudaMalloc(&rpW0, size_t(O)*I0));
  CK(cudaMalloc(&rpS0, size_t(O)*g0*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&rpX0, B*I0*sizeof(float)));
  CK(cudaMalloc(&rpY0, B*O*sizeof(float)));
  CK(cudaMemcpy(rpW0, Wh0, size_t(O)*I0, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rpS0, Sh0, size_t(O)*g0*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rpX0, Xh0, B*I0*sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaSetDevice(1));
  int8_t *rpW1; __nv_bfloat16 *rpS1; float *rpX1, *rpY1;
  CK(cudaMalloc(&rpW1, size_t(O)*I1));
  CK(cudaMalloc(&rpS1, size_t(O)*g1*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&rpX1, B*I1*sizeof(float)));
  CK(cudaMalloc(&rpY1, B*O*sizeof(float)));
  CK(cudaMemcpy(rpW1, Wh1, size_t(O)*I1, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rpS1, Sh1, size_t(O)*g1*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rpX1, Xh1, B*I1*sizeof(float), cudaMemcpyHostToDevice));

  CK(cudaSetDevice(0));
  { dim3 bl(16,8); dim3 gr((O+bl.x-1)/bl.x,(B+bl.y-1)/bl.y);
    gemm_w8<<<gr,bl>>>(rpX0,rpW0,rpS0,rpY0,B,O,I0,G); CK(cudaDeviceSynchronize()); }
  CK(cudaSetDevice(1));
  { dim3 bl(16,8); dim3 gr((O+bl.x-1)/bl.x,(B+bl.y-1)/bl.y);
    gemm_w8<<<gr,bl>>>(rpX1,rpW1,rpS1,rpY1,B,O,I1,G); CK(cudaDeviceSynchronize()); }

  host_bounce_sum(rpY0, rpY1, B*O);
  float* Y_rp = new float[B*O];
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(Y_rp, rpY0, B*O*sizeof(float), cudaMemcpyDeviceToHost));
  double max_rp=0, mean_rp=0;
  for (int i=0;i<B*O;++i){
    double d=std::fabs(double(Y_rp[i])-double(Y_ref[i]));
    max_rp=std::max(max_rp,d); mean_rp+=d;
  }
  mean_rp/=(B*O);
  printf("[W8 RowParallel+hostAR] max_abs=%.3e mean_abs=%.3e ref_mean=%.3e %s\n",
         max_rp, mean_rp, ref_mean, max_rp<1e-3?"PASS":"FAIL");

  ::munmap(map, st.st_size); ::close(fd);
  return 0;
}
