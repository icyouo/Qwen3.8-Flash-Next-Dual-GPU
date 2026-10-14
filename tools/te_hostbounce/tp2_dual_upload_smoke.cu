// Load TP2-sharded blk.0.linear_attn.out.weight to 2 GPUs, host-bounce AR vs full.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

struct TensorMeta {
  std::string name;
  int segment = -1;
  uint64_t data_off = 0, data_bytes = 0;
  uint64_t scale_off = 0, scale_bytes = 0;
  std::vector<int> shape;
};

struct StageIdx {
  std::vector<std::string> seg_paths;
  std::vector<uint64_t> seg_bytes;
  std::vector<TensorMeta> tensors;
};

static std::vector<std::string> split_tab(const std::string& ln) {
  std::vector<std::string> out; std::string cur;
  for (char c : ln) {
    if (c == '\t') { out.push_back(cur); cur.clear(); }
    else cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

static StageIdx load_index(const std::string& path) {
  StageIdx st;
  std::ifstream in(path);
  std::string ln;
  std::getline(in, ln);
  if (ln != "Q38_DEVICE_INDEX_V1") { fprintf(stderr, "bad magic %s\n", path.c_str()); std::exit(1); }
  while (std::getline(in, ln)) {
    if (ln.empty() || ln.find('=') != std::string::npos && ln.find('\t') == std::string::npos) continue;
    auto p = split_tab(ln);
    if (p[0] == "segment") {
      st.seg_paths.push_back(p[1]);
      st.seg_bytes.push_back(std::stoull(p[2]));
    } else if (p[0] == "tensor") {
      TensorMeta t;
      t.name = p[1];
      t.segment = std::stoi(p[6]);
      t.data_off = std::stoull(p[7]);
      t.data_bytes = std::stoull(p[8]);
      t.scale_off = std::stoull(p[10]);
      t.scale_bytes = std::stoull(p[11]);
      std::stringstream ss(p[13]);
      std::string d;
      while (std::getline(ss, d, ',')) t.shape.push_back(std::stoi(d));
      st.tensors.push_back(std::move(t));
    }
  }
  return st;
}

static const TensorMeta* find_t(const StageIdx& st, const char* name) {
  for (auto& t : st.tensors) if (t.name == name) return &t;
  return nullptr;
}

static std::vector<uint8_t> read_extent(const std::string& root, const StageIdx& st, const TensorMeta& t, bool scale) {
  auto path = root + "/" + st.seg_paths[t.segment];
  std::ifstream in(path, std::ios::binary);
  auto off = scale ? t.scale_off : t.data_off;
  auto n = scale ? t.scale_bytes : t.data_bytes;
  in.seekg(static_cast<std::streamoff>(off));
  std::vector<uint8_t> buf(n);
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  if (!in) { fprintf(stderr, "read fail %s\n", path.c_str()); std::exit(1); }
  return buf;
}

__device__ __forceinline__ float bf16_load(const __nv_bfloat16* p, size_t i) {
  return __bfloat162float(p[i]);
}

__global__ void gemv_w8(const float* x, const int8_t* W, const __nv_bfloat16* S,
                        float* y, int O, int I, int G) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float acc = 0.f;
  int groups = I / G;
  for (int i = 0; i < I; ++i) {
    float sc = bf16_load(S, size_t(o) * groups + (i / G));
    acc += x[i] * (float(W[size_t(o) * I + i]) * sc);
  }
  y[o] = acc;
}

__global__ void add_f(const float* a, const float* b, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] + b[i];
}

static void host_bounce_sum(float* y0, float* y1, int n) {
  float* stage = nullptr;
  CK(cudaHostAlloc(&stage, size_t(n) * sizeof(float), 0));
  float* scratch = nullptr;
  CK(cudaSetDevice(0));
  CK(cudaMalloc(&scratch, size_t(n) * sizeof(float)));
  CK(cudaSetDevice(1));
  CK(cudaMemcpy(stage, y1, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(scratch, stage, size_t(n) * sizeof(float), cudaMemcpyHostToDevice));
  add_f<<<(n + 255) / 256, 256>>>(y0, scratch, y0, n);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(scratch));
  CK(cudaFreeHost(stage));
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <source_stage0> <tp2_rank0_stage0> <tp2_rank1_stage0>\n", argv[0]);
    return 2;
  }
  const char* name = "blk.0.linear_attn.out.weight";
  std::string src_root = argv[1];
  std::string r0_root = argv[2];
  std::string r1_root = argv[3];
  auto src = load_index(src_root + "/index.q38d");
  auto r0 = load_index(r0_root + "/index.q38d");
  auto r1 = load_index(r1_root + "/index.q38d");
  auto* ts = find_t(src, name);
  auto* t0 = find_t(r0, name);
  auto* t1 = find_t(r1, name);
  if (!ts || !t0 || !t1) { fprintf(stderr, "tensor missing\n"); return 1; }
  int O = ts->shape[0], I = ts->shape[1], G = 128;
  int I0 = t0->shape[1], I1 = t1->shape[1];
  if (I0 + I1 != I) { fprintf(stderr, "col split mismatch %d+%d != %d\n", I0, I1, I); return 1; }

  auto W = read_extent(src_root, src, *ts, false);
  auto S = read_extent(src_root, src, *ts, true);
  auto W0 = read_extent(r0_root, r0, *t0, false);
  auto S0 = read_extent(r0_root, r0, *t0, true);
  auto W1 = read_extent(r1_root, r1, *t1, false);
  auto S1 = read_extent(r1_root, r1, *t1, true);

  std::vector<float> x(I);
  for (int i = 0; i < I; ++i) x[i] = std::sin(0.001f * i) * 0.1f;
  std::vector<float> x0(x.begin(), x.begin() + I0);
  std::vector<float> x1(x.begin() + I0, x.end());

  // baseline GPU0 full
  CK(cudaSetDevice(0));
  int8_t *Wd; __nv_bfloat16 *Sd; float *xd, *yd;
  CK(cudaMalloc(&Wd, W.size())); CK(cudaMalloc(&Sd, S.size()));
  CK(cudaMalloc(&xd, I * sizeof(float))); CK(cudaMalloc(&yd, O * sizeof(float)));
  CK(cudaMemcpy(Wd, W.data(), W.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(Sd, S.data(), S.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(xd, x.data(), I * sizeof(float), cudaMemcpyHostToDevice));
  gemv_w8<<<(O + 255) / 256, 256>>>(xd, Wd, Sd, yd, O, I, G);
  CK(cudaDeviceSynchronize());
  std::vector<float> yref(O);
  CK(cudaMemcpy(yref.data(), yd, O * sizeof(float), cudaMemcpyDeviceToHost));

  // TP: row-parallel out on 2 GPUs
  float *y0d = nullptr, *y1d = nullptr;
  CK(cudaSetDevice(0));
  {
    int8_t *W; __nv_bfloat16 *Sc; float *xx;
    CK(cudaMalloc(&W, W0.size())); CK(cudaMalloc(&Sc, S0.size()));
    CK(cudaMalloc(&xx, I0 * sizeof(float))); CK(cudaMalloc(&y0d, O * sizeof(float)));
    CK(cudaMemcpy(W, W0.data(), W0.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sc, S0.data(), S0.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xx, x0.data(), I0 * sizeof(float), cudaMemcpyHostToDevice));
    gemv_w8<<<(O + 255) / 256, 256>>>(xx, W, Sc, y0d, O, I0, G);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(W)); CK(cudaFree(Sc)); CK(cudaFree(xx));
  }
  CK(cudaSetDevice(1));
  {
    int8_t *W; __nv_bfloat16 *Sc; float *xx;
    CK(cudaMalloc(&W, W1.size())); CK(cudaMalloc(&Sc, S1.size()));
    CK(cudaMalloc(&xx, I1 * sizeof(float))); CK(cudaMalloc(&y1d, O * sizeof(float)));
    CK(cudaMemcpy(W, W1.data(), W1.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sc, S1.data(), S1.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xx, x1.data(), I1 * sizeof(float), cudaMemcpyHostToDevice));
    gemv_w8<<<(O + 255) / 256, 256>>>(xx, W, Sc, y1d, O, I1, G);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(W)); CK(cudaFree(Sc)); CK(cudaFree(xx));
  }
  host_bounce_sum(y0d, y1d, O);
  std::vector<float> ytp(O);
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(ytp.data(), y0d, O * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs = 0, mean = 0, refm = 0;
  for (int i = 0; i < O; ++i) {
    double d = std::fabs(double(ytp[i]) - double(yref[i]));
    max_abs = std::max(max_abs, d);
    mean += d;
    refm += std::fabs(double(yref[i]));
  }
  mean /= O; refm /= O;
  printf("[TP2 dual-upload hostAR] %s O=%d I=%d+%d max_abs=%.3e mean_abs=%.3e ref_mean=%.3e %s\n",
         name, O, I0, I1, max_abs, mean, refm, max_abs < 1e-3 ? "PASS" : "FAIL");
  return max_abs < 1e-3 ? 0 : 1;
}
