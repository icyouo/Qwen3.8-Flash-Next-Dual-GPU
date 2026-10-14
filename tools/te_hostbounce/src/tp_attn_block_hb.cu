// S3.4: one linear_attn projection block with TP + host-bounce AllReduce.
// Layout (blk.0): QKV [10240,2560]=Q(2048)|K(2048)|V(6144); Z[6144,2560]; Out[2560,6144]
// Middle stand-in (elementwise, head-local): core = V * sigmoid(Z)
// Real GDN conv/recurrent is deferred; this proves TP cut points + host AR.
//
// Usage: tp_attn_block_hb <seg.q38w>
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
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

__device__ __forceinline__ float bf16_to_f(const __nv_bfloat16* p, size_t i){
  return __bfloat162float(p[i]);
}

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

// Y[b,o] = sum_i X[b,i] * W[o,i]  (W already dequantized to float on host for Out? use W8)
// RowParallel out: same gemm_w8 with I_local

__global__ void gate_mul(const float* V, const float* Z, float* core, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float z = Z[i];
    float sig = 1.f / (1.f + expf(-z));
    core[i] = V[i] * sig;
  }
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

struct W8View {
  const int8_t* w = nullptr;
  const __nv_bfloat16* s = nullptr;
  int O = 0, I = 0, G = 128;
};

static void run_gemm(int dev, const float* Xh, int B, int I,
                     const W8View& m, float* Yh) {
  CK(cudaSetDevice(dev));
  int8_t* Wd; __nv_bfloat16* Sd; float *Xd, *Yd;
  size_t w_bytes = size_t(m.O) * m.I;
  size_t s_bytes = size_t(m.O) * (m.I / m.G) * sizeof(__nv_bfloat16);
  CK(cudaMalloc(&Wd, w_bytes));
  CK(cudaMalloc(&Sd, s_bytes));
  CK(cudaMalloc(&Xd, size_t(B) * I * sizeof(float)));
  CK(cudaMalloc(&Yd, size_t(B) * m.O * sizeof(float)));
  CK(cudaMemcpy(Wd, m.w, w_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(Sd, m.s, s_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(Xd, Xh, size_t(B) * I * sizeof(float), cudaMemcpyHostToDevice));
  dim3 bl(16, 8);
  dim3 gr((m.O + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
  gemm_w8<<<gr, bl>>>(Xd, Wd, Sd, Yd, B, m.O, m.I, m.G);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(Yh, Yd, size_t(B) * m.O * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(Xd)); CK(cudaFree(Yd));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <seg.q38w>\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  // Offsets from model-00001-of-00131.q38.json (blk.0)
  const size_t qkv_d = 13553664, qkv_s = 39768064; // 10240 x 2560
  const size_t z_d = 40177664, z_s = 55906304;     // 6144 x 2560
  const size_t out_d = 56156160, out_s = 71884800;  // 2560 x 6144
  const int H = 2560, QKV = 10240, V = 6144, G = 128;
  const int QK = 4096; // Q+K prefix before V
  const int B = 4;

  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  struct stat st{}; ::fstat(fd, &st);
  void* map = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap"); return 1; }
  auto base = static_cast<char*>(map);

  W8View qkv{reinterpret_cast<int8_t*>(base + qkv_d),
             reinterpret_cast<__nv_bfloat16*>(base + qkv_s), QKV, H, G};
  W8View z{reinterpret_cast<int8_t*>(base + z_d),
           reinterpret_cast<__nv_bfloat16*>(base + z_s), V, H, G};
  W8View out{reinterpret_cast<int8_t*>(base + out_d),
             reinterpret_cast<__nv_bfloat16*>(base + out_s), H, V, G};

  float* Xh = new float[B * H];
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < H; ++i)
      Xh[b * H + i] = std::sin(0.001f * (b * H + i)) * 0.1f;

  int ndev = 0; CK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) { printf("need 2 GPUs\n"); return 1; }

  // ---- Baseline on GPU0 ----
  float* qkv_y = new float[B * QKV];
  float* z_y = new float[B * V];
  float* Y_ref = new float[B * H];
  run_gemm(0, Xh, B, H, qkv, qkv_y);
  run_gemm(0, Xh, B, H, z, z_y);
  float* core = new float[B * V];
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < V; ++i) {
      float vv = qkv_y[b * QKV + QK + i];
      float zz = z_y[b * V + i];
      core[b * V + i] = vv * (1.f / (1.f + std::exp(-zz)));
    }
  // Out gemm: treat core as X with I=V, W out OxI = HxV
  {
    CK(cudaSetDevice(0));
    int8_t* Wd; __nv_bfloat16* Sd; float *Xd, *Yd;
    CK(cudaMalloc(&Wd, size_t(H) * V));
    CK(cudaMalloc(&Sd, size_t(H) * (V / G) * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&Xd, size_t(B) * V * sizeof(float)));
    CK(cudaMalloc(&Yd, size_t(B) * H * sizeof(float)));
    CK(cudaMemcpy(Wd, out.w, size_t(H) * V, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd, out.s, size_t(H) * (V / G) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Xd, core, size_t(B) * V * sizeof(float), cudaMemcpyHostToDevice));
    dim3 bl(16, 8);
    dim3 gr((H + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
    gemm_w8<<<gr, bl>>>(Xd, Wd, Sd, Yd, B, H, V, G);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(Y_ref, Yd, size_t(B) * H * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(Xd)); CK(cudaFree(Yd));
  }
  double ref_mean = 0;
  for (int i = 0; i < B * H; ++i) ref_mean += std::fabs(double(Y_ref[i]));
  ref_mean /= (B * H);
  printf("S3.4 baseline ref_mean=%.3e (middle=V*sigmoid(Z) stand-in)\n", ref_mean);

  // ---- TP2: ColumnParallel V/Z heads + RowParallel Out + host AR ----
  const int V0 = V / 2, V1 = V - V0; // 3072 / 3072, group-aligned
  // qkv V rows start at QK
  W8View qkv0{qkv.w + size_t(QK) * H,
              qkv.s + size_t(QK) * (H / G), V0, H, G};
  W8View qkv1{qkv.w + size_t(QK + V0) * H,
              qkv.s + size_t(QK + V0) * (H / G), V1, H, G};
  W8View z0{z.w, z.s, V0, H, G};
  W8View z1{z.w + size_t(V0) * H, z.s + size_t(V0) * (H / G), V1, H, G};

  float* v0 = new float[B * V0];
  float* v1 = new float[B * V1];
  float* zz0 = new float[B * V0];
  float* zz1 = new float[B * V1];
  run_gemm(0, Xh, B, H, qkv0, v0);
  run_gemm(1, Xh, B, H, qkv1, v1);
  run_gemm(0, Xh, B, H, z0, zz0);
  run_gemm(1, Xh, B, H, z1, zz1);

  float* core0 = new float[B * V0];
  float* core1 = new float[B * V1];
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < V0; ++i) {
      float sig = 1.f / (1.f + std::exp(-zz0[b * V0 + i]));
      core0[b * V0 + i] = v0[b * V0 + i] * sig;
    }
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < V1; ++i) {
      float sig = 1.f / (1.f + std::exp(-zz1[b * V1 + i]));
      core1[b * V1 + i] = v1[b * V1 + i] * sig;
    }

  // Out RowParallel: split columns — pack Wh0/Wh1 and scales
  int g0 = V0 / G, g1 = V1 / G, groups = V / G;
  int8_t* Wh0 = new int8_t[size_t(H) * V0];
  int8_t* Wh1 = new int8_t[size_t(H) * V1];
  auto* Sh0 = new __nv_bfloat16[size_t(H) * g0];
  auto* Sh1 = new __nv_bfloat16[size_t(H) * g1];
  for (int o = 0; o < H; ++o) {
    std::memcpy(Wh0 + size_t(o) * V0, out.w + size_t(o) * V, size_t(V0));
    std::memcpy(Wh1 + size_t(o) * V1, out.w + size_t(o) * V + V0, size_t(V1));
    std::memcpy(Sh0 + size_t(o) * g0, out.s + size_t(o) * groups, size_t(g0) * sizeof(__nv_bfloat16));
    std::memcpy(Sh1 + size_t(o) * g1, out.s + size_t(o) * groups + g0, size_t(g1) * sizeof(__nv_bfloat16));
  }

  float *Y0d = nullptr, *Y1d = nullptr;
  CK(cudaSetDevice(0));
  {
    int8_t* Wd; __nv_bfloat16* Sd; float *Xd;
    CK(cudaMalloc(&Wd, size_t(H) * V0));
    CK(cudaMalloc(&Sd, size_t(H) * g0 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&Xd, size_t(B) * V0 * sizeof(float)));
    CK(cudaMalloc(&Y0d, size_t(B) * H * sizeof(float)));
    CK(cudaMemcpy(Wd, Wh0, size_t(H) * V0, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd, Sh0, size_t(H) * g0 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Xd, core0, size_t(B) * V0 * sizeof(float), cudaMemcpyHostToDevice));
    dim3 bl(16, 8);
    dim3 gr((H + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
    gemm_w8<<<gr, bl>>>(Xd, Wd, Sd, Y0d, B, H, V0, G);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(Xd));
  }
  CK(cudaSetDevice(1));
  {
    int8_t* Wd; __nv_bfloat16* Sd; float *Xd;
    CK(cudaMalloc(&Wd, size_t(H) * V1));
    CK(cudaMalloc(&Sd, size_t(H) * g1 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&Xd, size_t(B) * V1 * sizeof(float)));
    CK(cudaMalloc(&Y1d, size_t(B) * H * sizeof(float)));
    CK(cudaMemcpy(Wd, Wh1, size_t(H) * V1, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd, Sh1, size_t(H) * g1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Xd, core1, size_t(B) * V1 * sizeof(float), cudaMemcpyHostToDevice));
    dim3 bl(16, 8);
    dim3 gr((H + bl.x - 1) / bl.x, (B + bl.y - 1) / bl.y);
    gemm_w8<<<gr, bl>>>(Xd, Wd, Sd, Y1d, B, H, V1, G);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(Xd));
  }

  host_bounce_sum(Y0d, Y1d, B * H);
  float* Y_tp = new float[B * H];
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(Y_tp, Y0d, size_t(B) * H * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs = 0, mean_abs = 0;
  for (int i = 0; i < B * H; ++i) {
    double d = std::fabs(double(Y_tp[i]) - double(Y_ref[i]));
    max_abs = std::max(max_abs, d);
    mean_abs += d;
  }
  mean_abs /= (B * H);
  printf("[S3.4 AttnBlock TP+hostAR] max_abs=%.3e mean_abs=%.3e ref_mean=%.3e %s\n",
         max_abs, mean_abs, ref_mean, max_abs < 1e-3 ? "PASS" : "FAIL");
  printf("note: middle is V*sigmoid(Z) stand-in; swap for cuda_gdn_* next\n");

  CK(cudaSetDevice(0)); CK(cudaFree(Y0d));
  CK(cudaSetDevice(1)); CK(cudaFree(Y1d));
  ::munmap(map, st.st_size); ::close(fd);
  return max_abs < 1e-3 ? 0 : 1;
}
