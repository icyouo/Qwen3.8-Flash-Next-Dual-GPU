// S3.4b: real cuda_gdn_* decode + TP projections + host-bounce AR on out.
// Flow: ColumnParallel W8 gemv (qkv/z/a/b) -> host AllGather -> GDN on GPU0
//       -> RowParallel out + host AR. Compares to unreplicated full path.
#include "q38/cuda_gdn.h"
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

using q38::kQ38GdnQkvWidth;
using q38::kQ38GdnValueWidth;
using q38::kQ38GdnValueHeads;
using q38::kQ38GdnHeadWidth;
using q38::kQ38GdnConvWidth;
using q38::kQ38GdnKeyHeads;

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

__device__ __forceinline__ float bf16_to_f(const __nv_bfloat16* p, size_t i){
  return __bfloat162float(p[i]);
}

__global__ void gemv_w8_bf16(const __nv_bfloat16* __restrict__ X,
                             const int8_t* __restrict__ W,
                             const __nv_bfloat16* __restrict__ S,
                             __nv_bfloat16* __restrict__ Y,
                             int O, int I, int G) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float acc = 0.f;
  const int groups = I / G;
  for (int i = 0; i < I; ++i) {
    float scale = bf16_to_f(S, size_t(o) * groups + (i / G));
    acc += __bfloat162float(X[i]) * (float(W[size_t(o) * I + i]) * scale);
  }
  Y[o] = __float2bfloat16_rn(acc);
}

__global__ void gemv_w8_f32_from_bf16(const __nv_bfloat16* __restrict__ X,
                                      const int8_t* __restrict__ W,
                                      const __nv_bfloat16* __restrict__ S,
                                      float* __restrict__ Y,
                                      int O, int I, int G) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float acc = 0.f;
  const int groups = I / G;
  for (int i = 0; i < I; ++i) {
    float scale = bf16_to_f(S, size_t(o) * groups + (i / G));
    acc += __bfloat162float(X[i]) * (float(W[size_t(o) * I + i]) * scale);
  }
  Y[o] = acc;
}

__global__ void add_f32(const float* A, const float* B, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = A[i] + B[i];
}

static void host_bounce_sum_f32(float* y0, float* y1, int n) {
  float* stage = nullptr;
  CK(cudaHostAlloc(&stage, size_t(n) * sizeof(float), cudaHostAllocDefault));
  float* scratch = nullptr;
  CK(cudaSetDevice(0));
  CK(cudaMalloc(&scratch, size_t(n) * sizeof(float)));
  CK(cudaSetDevice(1));
  CK(cudaMemcpy(stage, y1, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(scratch, stage, size_t(n) * sizeof(float), cudaMemcpyHostToDevice));
  add_f32<<<(n + 255) / 256, 256>>>(y0, scratch, y0, n);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(scratch));
  CK(cudaFreeHost(stage));
}

struct W8 { const int8_t* w; const __nv_bfloat16* s; int O, I, G; };

static void upload_w8(int dev, const W8& m, int8_t** Wd, __nv_bfloat16** Sd) {
  CK(cudaSetDevice(dev));
  CK(cudaMalloc(Wd, size_t(m.O) * m.I));
  CK(cudaMalloc(Sd, size_t(m.O) * (m.I / m.G) * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(*Wd, m.w, size_t(m.O) * m.I, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(*Sd, m.s, size_t(m.O) * (m.I / m.G) * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <seg.q38w>\n", argv[0]); return 2; }
  const char* path = argv[1];
  // blk.0 offsets from segment json
  const size_t a_log_off = 13209600;      // bf16[48]
  const size_t conv_off = 13213696;       // bf16[10240,1,4]
  const size_t dt_off = 13295616;         // bf16[48]
  const size_t ina_d = 13299712, ina_s = 13422592; // 48x2560 w8
  const size_t inb_d = 13426688, inb_s = 13549568;
  const size_t qkv_d = 13553664, qkv_s = 39768064; // 10240x2560
  const size_t z_d = 40177664, z_s = 55906304;     // 6144x2560
  const size_t norm_off = 56152064;                // bf16[128]
  const size_t out_d = 56156160, out_s = 71884800; // 2560x6144
  const int H = 2560, G = 128;

  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  struct stat st{}; ::fstat(fd, &st);
  void* map = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap"); return 1; }
  auto base = static_cast<char*>(map);

  W8 qkv{reinterpret_cast<int8_t*>(base + qkv_d),
         reinterpret_cast<__nv_bfloat16*>(base + qkv_s), (int)kQ38GdnQkvWidth, H, G};
  W8 z{reinterpret_cast<int8_t*>(base + z_d),
       reinterpret_cast<__nv_bfloat16*>(base + z_s), (int)kQ38GdnValueWidth, H, G};
  W8 ina{reinterpret_cast<int8_t*>(base + ina_d),
         reinterpret_cast<__nv_bfloat16*>(base + ina_s), (int)kQ38GdnValueHeads, H, G};
  W8 inb{reinterpret_cast<int8_t*>(base + inb_d),
         reinterpret_cast<__nv_bfloat16*>(base + inb_s), (int)kQ38GdnValueHeads, H, G};
  W8 out{reinterpret_cast<int8_t*>(base + out_d),
         reinterpret_cast<__nv_bfloat16*>(base + out_s), H, (int)kQ38GdnValueWidth, G};
  const auto* a_log = reinterpret_cast<__nv_bfloat16*>(base + a_log_off);
  const auto* dt_bias = reinterpret_cast<__nv_bfloat16*>(base + dt_off);
  const auto* conv_w = reinterpret_cast<__nv_bfloat16*>(base + conv_off);
  const auto* norm_w = reinterpret_cast<__nv_bfloat16*>(base + norm_off);

  // Random-ish input in bf16
  std::vector<__nv_bfloat16> Xh(H);
  for (int i = 0; i < H; ++i)
    Xh[i] = __float2bfloat16_rn(std::sin(0.001f * i) * 0.1f);

  int ndev = 0; CK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) { printf("need 2 GPUs\n"); return 1; }

  cudaStream_t stream0;
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&stream0));

  // ---- Upload full weights to GPU0 for baseline ----
  int8_t *qkvW=nullptr,*zW=nullptr,*aW=nullptr,*bW=nullptr,*outW=nullptr;
  __nv_bfloat16 *qkvS=nullptr,*zS=nullptr,*aS=nullptr,*bS=nullptr,*outS=nullptr;
  upload_w8(0, qkv, &qkvW, &qkvS);
  upload_w8(0, z, &zW, &zS);
  upload_w8(0, ina, &aW, &aS);
  upload_w8(0, inb, &bW, &bS);
  upload_w8(0, out, &outW, &outS);

  __nv_bfloat16 *Xd=nullptr, *qkv_y=nullptr, *z_y=nullptr, *a_y=nullptr, *b_y=nullptr;
  __nv_bfloat16 *conv_wd=nullptr, *a_log_d=nullptr, *dt_d=nullptr, *norm_d=nullptr;
  __nv_bfloat16 *conv_state=nullptr, *act=nullptr, *core=nullptr, *gated=nullptr;
  float *rec_state=nullptr;
  CK(cudaMalloc(&Xd, H * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(Xd, Xh.data(), H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&qkv_y, kQ38GdnQkvWidth * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&z_y, kQ38GdnValueWidth * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&a_y, kQ38GdnValueHeads * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&b_y, kQ38GdnValueHeads * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&conv_wd, kQ38GdnQkvWidth * kQ38GdnConvWidth * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(conv_wd, conv_w, kQ38GdnQkvWidth * kQ38GdnConvWidth * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
  CK(cudaMalloc(&a_log_d, kQ38GdnValueHeads * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&dt_d, kQ38GdnValueHeads * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&norm_d, kQ38GdnHeadWidth * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(a_log_d, a_log, kQ38GdnValueHeads * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dt_d, dt_bias, kQ38GdnValueHeads * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(norm_d, norm_w, kQ38GdnHeadWidth * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&conv_state, kQ38GdnQkvWidth * kQ38GdnConvWidth * sizeof(__nv_bfloat16)));
  CK(cudaMemset(conv_state, 0, kQ38GdnQkvWidth * kQ38GdnConvWidth * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&act, kQ38GdnQkvWidth * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&core, kQ38GdnValueWidth * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&gated, kQ38GdnValueWidth * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&rec_state, size_t(kQ38GdnValueHeads) * kQ38GdnHeadWidth * kQ38GdnHeadWidth * sizeof(float)));
  CK(cudaMemset(rec_state, 0, size_t(kQ38GdnValueHeads) * kQ38GdnHeadWidth * kQ38GdnHeadWidth * sizeof(float)));

  auto launch_gemv_bf16 = [&](const __nv_bfloat16* x, int8_t* W, __nv_bfloat16* S,
                              __nv_bfloat16* y, int O, int I) {
    gemv_w8_bf16<<<(O + 255) / 256, 256, 0, stream0>>>(x, W, S, y, O, I, G);
  };
  launch_gemv_bf16(Xd, qkvW, qkvS, qkv_y, kQ38GdnQkvWidth, H);
  launch_gemv_bf16(Xd, zW, zS, z_y, kQ38GdnValueWidth, H);
  launch_gemv_bf16(Xd, aW, aS, a_y, kQ38GdnValueHeads, H);
  launch_gemv_bf16(Xd, bW, bS, b_y, kQ38GdnValueHeads, H);
  CK(cudaStreamSynchronize(stream0));

  q38::cuda_gdn_conv_decode_bf16(
      reinterpret_cast<const std::uint16_t*>(qkv_y),
      reinterpret_cast<const std::uint16_t*>(conv_wd),
      reinterpret_cast<std::uint16_t*>(conv_state),
      reinterpret_cast<std::uint16_t*>(act), stream0, 0);
  q38::cuda_gdn_recurrent_decode_bf16(
      reinterpret_cast<const std::uint16_t*>(act),
      reinterpret_cast<const std::uint16_t*>(b_y),
      reinterpret_cast<const std::uint16_t*>(a_y),
      reinterpret_cast<const std::uint16_t*>(a_log_d),
      reinterpret_cast<const std::uint16_t*>(dt_d),
      rec_state,
      reinterpret_cast<std::uint16_t*>(core), stream0, 0);
  q38::cuda_gdn_output_norm_bf16(
      reinterpret_cast<const std::uint16_t*>(core),
      reinterpret_cast<const std::uint16_t*>(z_y),
      reinterpret_cast<const std::uint16_t*>(norm_d),
      reinterpret_cast<std::uint16_t*>(gated), 1e-6f, stream0, 0);
  CK(cudaStreamSynchronize(stream0));

  float* Y_ref = nullptr;
  CK(cudaMalloc(&Y_ref, H * sizeof(float)));
  gemv_w8_f32_from_bf16<<<(H + 255) / 256, 256, 0, stream0>>>(
      gated, outW, outS, Y_ref, H, kQ38GdnValueWidth, G);
  CK(cudaStreamSynchronize(stream0));
  std::vector<float> Y_ref_h(H);
  CK(cudaMemcpy(Y_ref_h.data(), Y_ref, H * sizeof(float), cudaMemcpyDeviceToHost));
  double ref_mean = 0;
  for (float v : Y_ref_h) ref_mean += std::fabs(v);
  ref_mean /= H;
  printf("S3.4b baseline (real GDN) ref_mean=%.3e\n", ref_mean);

  // ---- TP path: ColumnParallel halves of qkv/z/a/b, host gather, reuse GDN on GPU0 ----
  // Reset GDN state for fair compare
  CK(cudaMemset(conv_state, 0, kQ38GdnQkvWidth * kQ38GdnConvWidth * sizeof(__nv_bfloat16)));
  CK(cudaMemset(rec_state, 0, size_t(kQ38GdnValueHeads) * kQ38GdnHeadWidth * kQ38GdnHeadWidth * sizeof(float)));

  const int qkv0 = kQ38GdnQkvWidth / 2, qkv1 = kQ38GdnQkvWidth - qkv0; // 5120/5120
  const int v0 = kQ38GdnValueWidth / 2, v1 = kQ38GdnValueWidth - v0;
  const int h0 = kQ38GdnValueHeads / 2, h1 = kQ38GdnValueHeads - h0;

  // Host buffers for gathered activations
  std::vector<__nv_bfloat16> qkv_h(kQ38GdnQkvWidth), z_h(kQ38GdnValueWidth);
  std::vector<__nv_bfloat16> a_h(kQ38GdnValueHeads), b_h(kQ38GdnValueHeads);

  auto run_partial = [&](int dev, const W8& full, int row0, int rows,
                         std::vector<__nv_bfloat16>& host_out, int host_off) {
    CK(cudaSetDevice(dev));
    W8 part{full.w + size_t(row0) * full.I,
            full.s + size_t(row0) * (full.I / full.G), rows, full.I, full.G};
    int8_t* Wd; __nv_bfloat16 *Sd, *xd, *yd;
    CK(cudaMalloc(&Wd, size_t(rows) * full.I));
    CK(cudaMalloc(&Sd, size_t(rows) * (full.I / full.G) * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&xd, H * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&yd, rows * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(Wd, part.w, size_t(rows) * full.I, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd, part.s, size_t(rows) * (full.I / full.G) * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xd, Xh.data(), H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gemv_w8_bf16<<<(rows + 255) / 256, 256>>>(xd, Wd, Sd, yd, rows, full.I, G);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host_out.data() + host_off, yd, rows * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));
    CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(xd)); CK(cudaFree(yd));
  };

  run_partial(0, qkv, 0, qkv0, qkv_h, 0);
  run_partial(1, qkv, qkv0, qkv1, qkv_h, qkv0);
  run_partial(0, z, 0, v0, z_h, 0);
  run_partial(1, z, v0, v1, z_h, v0);
  run_partial(0, ina, 0, h0, a_h, 0);
  run_partial(1, ina, h0, h1, a_h, h0);
  run_partial(0, inb, 0, h0, b_h, 0);
  run_partial(1, inb, h0, h1, b_h, h0);

  CK(cudaSetDevice(0));
  CK(cudaMemcpy(qkv_y, qkv_h.data(), kQ38GdnQkvWidth * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(z_y, z_h.data(), kQ38GdnValueWidth * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(a_y, a_h.data(), kQ38GdnValueHeads * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b_y, b_h.data(), kQ38GdnValueHeads * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  q38::cuda_gdn_conv_decode_bf16(
      reinterpret_cast<const std::uint16_t*>(qkv_y),
      reinterpret_cast<const std::uint16_t*>(conv_wd),
      reinterpret_cast<std::uint16_t*>(conv_state),
      reinterpret_cast<std::uint16_t*>(act), stream0, 0);
  q38::cuda_gdn_recurrent_decode_bf16(
      reinterpret_cast<const std::uint16_t*>(act),
      reinterpret_cast<const std::uint16_t*>(b_y),
      reinterpret_cast<const std::uint16_t*>(a_y),
      reinterpret_cast<const std::uint16_t*>(a_log_d),
      reinterpret_cast<const std::uint16_t*>(dt_d),
      rec_state,
      reinterpret_cast<std::uint16_t*>(core), stream0, 0);
  q38::cuda_gdn_output_norm_bf16(
      reinterpret_cast<const std::uint16_t*>(core),
      reinterpret_cast<const std::uint16_t*>(z_y),
      reinterpret_cast<const std::uint16_t*>(norm_d),
      reinterpret_cast<std::uint16_t*>(gated), 1e-6f, stream0, 0);
  CK(cudaStreamSynchronize(stream0));

  // RowParallel out: split gated + out cols across GPUs, host AR
  std::vector<__nv_bfloat16> gated_h(kQ38GdnValueWidth);
  CK(cudaMemcpy(gated_h.data(), gated, kQ38GdnValueWidth * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));

  int g0 = v0 / G, g1 = v1 / G, groups = kQ38GdnValueWidth / G;
  int8_t* Wh0 = new int8_t[size_t(H) * v0];
  int8_t* Wh1 = new int8_t[size_t(H) * v1];
  auto* Sh0 = new __nv_bfloat16[size_t(H) * g0];
  auto* Sh1 = new __nv_bfloat16[size_t(H) * g1];
  for (int o = 0; o < H; ++o) {
    std::memcpy(Wh0 + size_t(o) * v0, out.w + size_t(o) * kQ38GdnValueWidth, size_t(v0));
    std::memcpy(Wh1 + size_t(o) * v1, out.w + size_t(o) * kQ38GdnValueWidth + v0, size_t(v1));
    std::memcpy(Sh0 + size_t(o) * g0, out.s + size_t(o) * groups, size_t(g0) * sizeof(__nv_bfloat16));
    std::memcpy(Sh1 + size_t(o) * g1, out.s + size_t(o) * groups + g0, size_t(g1) * sizeof(__nv_bfloat16));
  }

  float *Y0d = nullptr, *Y1d = nullptr;
  CK(cudaSetDevice(0));
  {
    int8_t* Wd; __nv_bfloat16 *Sd, *xd;
    CK(cudaMalloc(&Wd, size_t(H) * v0));
    CK(cudaMalloc(&Sd, size_t(H) * g0 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&xd, v0 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&Y0d, H * sizeof(float)));
    CK(cudaMemcpy(Wd, Wh0, size_t(H) * v0, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd, Sh0, size_t(H) * g0 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xd, gated_h.data(), v0 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gemv_w8_f32_from_bf16<<<(H + 255) / 256, 256>>>(xd, Wd, Sd, Y0d, H, v0, G);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(xd));
  }
  CK(cudaSetDevice(1));
  {
    int8_t* Wd; __nv_bfloat16 *Sd, *xd;
    CK(cudaMalloc(&Wd, size_t(H) * v1));
    CK(cudaMalloc(&Sd, size_t(H) * g1 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&xd, v1 * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&Y1d, H * sizeof(float)));
    CK(cudaMemcpy(Wd, Wh1, size_t(H) * v1, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd, Sh1, size_t(H) * g1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xd, gated_h.data() + v0, v1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gemv_w8_f32_from_bf16<<<(H + 255) / 256, 256>>>(xd, Wd, Sd, Y1d, H, v1, G);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(xd));
  }
  host_bounce_sum_f32(Y0d, Y1d, H);
  std::vector<float> Y_tp(H);
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(Y_tp.data(), Y0d, H * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs = 0, mean_abs = 0;
  for (int i = 0; i < H; ++i) {
    double d = std::fabs(double(Y_tp[i]) - double(Y_ref_h[i]));
    max_abs = std::max(max_abs, d);
    mean_abs += d;
  }
  mean_abs /= H;
  printf("[S3.4b realGDN+TP+hostAR] max_abs=%.3e mean_abs=%.3e ref_mean=%.3e %s\n",
         max_abs, mean_abs, ref_mean, max_abs < 1e-2 ? "PASS" : "FAIL");

  ::munmap(map, st.st_size); ::close(fd);
  return max_abs < 1e-2 ? 0 : 1;
}
