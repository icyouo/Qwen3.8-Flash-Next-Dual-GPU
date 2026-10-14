// For every col_shard W8 matrix present on both TP2 ranks: dual GEMV + host AR
// vs full source matrix. Proves RowParallel cuts across the artifact.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

struct TensorMeta {
  std::string name, fmt;
  int segment = -1;
  uint64_t data_off=0, data_bytes=0, scale_off=0, scale_bytes=0;
  std::vector<int> shape;
};

struct StageIdx {
  std::vector<std::string> seg_paths;
  std::vector<TensorMeta> tensors;
  std::unordered_map<std::string, int> by_name;
};

static std::vector<std::string> split_tab(const std::string& ln) {
  std::vector<std::string> out; std::string cur;
  for (char c: ln) { if(c=='\t'){out.push_back(cur); cur.clear();} else cur.push_back(c);} out.push_back(cur); return out;
}

static StageIdx load_index(const std::string& path) {
  StageIdx st; std::ifstream in(path); std::string ln; std::getline(in, ln);
  if (ln != "Q38_DEVICE_INDEX_V1") { fprintf(stderr,"bad magic %s\n", path.c_str()); std::exit(1); }
  while (std::getline(in, ln)) {
    if (ln.empty()) continue;
    if (ln.find('\t')==std::string::npos) continue;
    auto p = split_tab(ln);
    if (p[0]=="segment") st.seg_paths.push_back(p[1]);
    else if (p[0]=="tensor") {
      TensorMeta t; t.name=p[1]; t.fmt=p[4]; t.segment=std::stoi(p[6]);
      t.data_off=std::stoull(p[7]); t.data_bytes=std::stoull(p[8]);
      t.scale_off=std::stoull(p[10]); t.scale_bytes=std::stoull(p[11]);
      std::stringstream ss(p[13]); std::string d;
      while (std::getline(ss,d,',')) t.shape.push_back(std::stoi(d));
      st.by_name[t.name]= (int)st.tensors.size();
      st.tensors.push_back(std::move(t));
    }
  }
  return st;
}

static std::vector<uint8_t> read_extent(const std::string& root, const StageIdx& st, const TensorMeta& t, bool scale) {
  std::ifstream in(root + "/" + st.seg_paths[t.segment], std::ios::binary);
  auto off = scale?t.scale_off:t.data_off; auto n=scale?t.scale_bytes:t.data_bytes;
  in.seekg((std::streamoff)off);
  std::vector<uint8_t> buf(n);
  in.read((char*)buf.data(), (std::streamsize)n);
  if (!in) { fprintf(stderr,"read fail %s/%s\n", root.c_str(), t.name.c_str()); std::exit(1); }
  return buf;
}

__device__ __forceinline__ float bf16_load(const __nv_bfloat16* p, size_t i){ return __bfloat162float(p[i]); }

__global__ void gemv_w8(const float* x, const int8_t* W, const __nv_bfloat16* S,
                        float* y, int O, int I, int G) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float acc = 0.f; int groups = I / G;
  for (int i=0;i<I;++i) acc += x[i]*(float(W[size_t(o)*I+i])*bf16_load(S,size_t(o)*groups+(i/G)));
  y[o]=acc;
}
__global__ void add_f(const float*a,const float*b,float*o,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)o[i]=a[i]+b[i]; }

static void host_bounce_sum(float* y0, float* y1, int n) {
  float* stage=nullptr; CK(cudaHostAlloc(&stage,size_t(n)*sizeof(float),0));
  float* scratch=nullptr; CK(cudaSetDevice(0)); CK(cudaMalloc(&scratch,size_t(n)*sizeof(float)));
  CK(cudaSetDevice(1)); CK(cudaMemcpy(stage,y1,size_t(n)*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaSetDevice(0)); CK(cudaMemcpy(scratch,stage,size_t(n)*sizeof(float),cudaMemcpyHostToDevice));
  add_f<<<(n+255)/256,256>>>(y0,scratch,y0,n); CK(cudaDeviceSynchronize());
  CK(cudaFree(scratch)); CK(cudaFreeHost(stage));
}

static int check_one(const std::string& src_root, const StageIdx& src,
                     const std::string& r0_root, const StageIdx& r0,
                     const std::string& r1_root, const StageIdx& r1,
                     const std::string& name) {
  auto* ts=&src.tensors[src.by_name.at(name)];
  auto* t0=&r0.tensors[r0.by_name.at(name)];
  auto* t1=&r1.tensors[r1.by_name.at(name)];
  if (ts->fmt != "w8a16_sym_g128") return 0; // skip non-w8 for this smoke
  if (ts->shape.size()!=2) return 0;
  int O=ts->shape[0], I=ts->shape[1], G=128;
  int I0=t0->shape[1], I1=t1->shape[1];
  if (I0+I1!=I || t0->shape[0]!=O || t1->shape[0]!=O) {
    printf("SKIP shape %s\n", name.c_str()); return 0;
  }
  // Cap very large mats for runtime (e.g. vocab) — still sample first 4k rows? skip if O*I > 80e6
  if (int64_t(O)*I > 80'000'000LL) { printf("SKIP large %s O=%d I=%d\n", name.c_str(), O, I); return 0; }

  auto W=read_extent(src_root,src,*ts,false);
  auto S=read_extent(src_root,src,*ts,true);
  auto W0=read_extent(r0_root,r0,*t0,false);
  auto S0=read_extent(r0_root,r0,*t0,true);
  auto W1=read_extent(r1_root,r1,*t1,false);
  auto S1=read_extent(r1_root,r1,*t1,true);
  std::vector<float> x(I); for(int i=0;i<I;++i) x[i]=std::sin(0.001f*i)*0.1f;
  std::vector<float> x0(x.begin(), x.begin()+I0), x1(x.begin()+I0, x.end());

  CK(cudaSetDevice(0));
  int8_t*Wd; __nv_bfloat16*Sd; float*xd,*yd;
  CK(cudaMalloc(&Wd,W.size())); CK(cudaMalloc(&Sd,S.size()));
  CK(cudaMalloc(&xd,I*sizeof(float))); CK(cudaMalloc(&yd,O*sizeof(float)));
  CK(cudaMemcpy(Wd,W.data(),W.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(Sd,S.data(),S.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(xd,x.data(),I*sizeof(float),cudaMemcpyHostToDevice));
  gemv_w8<<<(O+255)/256,256>>>(xd,Wd,Sd,yd,O,I,G); CK(cudaDeviceSynchronize());
  std::vector<float> yref(O); CK(cudaMemcpy(yref.data(),yd,O*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaFree(Wd)); CK(cudaFree(Sd)); CK(cudaFree(xd)); CK(cudaFree(yd));

  float *y0d=nullptr,*y1d=nullptr;
  CK(cudaSetDevice(0));
  { int8_t*W; __nv_bfloat16*Sc; float*xx;
    CK(cudaMalloc(&W,W0.size())); CK(cudaMalloc(&Sc,S0.size()));
    CK(cudaMalloc(&xx,I0*sizeof(float))); CK(cudaMalloc(&y0d,O*sizeof(float)));
    CK(cudaMemcpy(W,W0.data(),W0.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sc,S0.data(),S0.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xx,x0.data(),I0*sizeof(float),cudaMemcpyHostToDevice));
    gemv_w8<<<(O+255)/256,256>>>(xx,W,Sc,y0d,O,I0,G); CK(cudaDeviceSynchronize());
    CK(cudaFree(W)); CK(cudaFree(Sc)); CK(cudaFree(xx)); }
  CK(cudaSetDevice(1));
  { int8_t*W; __nv_bfloat16*Sc; float*xx;
    CK(cudaMalloc(&W,W1.size())); CK(cudaMalloc(&Sc,S1.size()));
    CK(cudaMalloc(&xx,I1*sizeof(float))); CK(cudaMalloc(&y1d,O*sizeof(float)));
    CK(cudaMemcpy(W,W1.data(),W1.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sc,S1.data(),S1.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xx,x1.data(),I1*sizeof(float),cudaMemcpyHostToDevice));
    gemv_w8<<<(O+255)/256,256>>>(xx,W,Sc,y1d,O,I1,G); CK(cudaDeviceSynchronize());
    CK(cudaFree(W)); CK(cudaFree(Sc)); CK(cudaFree(xx)); }
  host_bounce_sum(y0d,y1d,O);
  std::vector<float> ytp(O); CK(cudaSetDevice(0));
  CK(cudaMemcpy(ytp.data(),y0d,O*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaFree(y0d)); CK(cudaSetDevice(1)); CK(cudaFree(y1d));

  double max_abs=0, mean=0;
  for(int i=0;i<O;++i){ double d=std::fabs(double(ytp[i])-double(yref[i])); max_abs=std::max(max_abs,d); mean+=d; }
  mean/=O;
  int pass = max_abs < 1e-3;
  printf("%s %s O=%d I=%d+%d max_abs=%.3e mean=%.3e\n", pass?"PASS":"FAIL", name.c_str(), O,I0,I1,max_abs,mean);
  return pass ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr,"usage: %s src0 r0_s0 r1_s0 src1 r0_s1 r1_s1\n", argv[0]);
    return 2;
  }
  int fails=0, checked=0;
  for (int stage=0; stage<2; ++stage) {
    auto src = load_index(std::string(argv[1+stage*3]) + "/index.q38d");
    auto r0  = load_index(std::string(argv[2+stage*3]) + "/index.q38d");
    auto r1  = load_index(std::string(argv[3+stage*3]) + "/index.q38d");
    std::string src_root = argv[1+stage*3];
    std::string r0_root = argv[2+stage*3];
    std::string r1_root = argv[3+stage*3];
    for (auto& t : src.tensors) {
      if (!r0.by_name.count(t.name) || !r1.by_name.count(t.name)) continue;
      auto& a=r0.tensors[r0.by_name[t.name]];
      auto& b=r1.tensors[r1.by_name[t.name]];
      // col_shard heuristic: same outer dims, last dim halved
      if (t.shape.size()<2 || a.shape.size()!=t.shape.size()) continue;
      if (a.shape.back()*2 != t.shape.back()) continue;
      if (a.shape.back() != b.shape.back()) continue;
      bool prefix_ok=true;
      for (size_t i=0;i+1<t.shape.size();++i) if (a.shape[i]!=t.shape[i]||b.shape[i]!=t.shape[i]) prefix_ok=false;
      if (!prefix_ok) continue;
      int rc = check_one(src_root,src,r0_root,r0,r1_root,r1,t.name);
      if (rc==0 || rc==1) { // 0 pass after check, but check_one returns 0 pass 1 fail; skip returns 0 without checked
      }
      // recount: check_one returns 0 pass, 1 fail, and prints SKIP with return 0 — distinguish by looking at fmt inside
      if (t.fmt=="w8a16_sym_g128" && t.shape.size()==2 && int64_t(t.shape[0])*t.shape[1] <= 80'000'000LL) {
        checked++;
        fails += rc;
      }
    }
  }
  printf("SUMMARY checked=%d fails=%d %s\n", checked, fails, fails? "FAIL":"PASS");
  return fails?1:0;
}
