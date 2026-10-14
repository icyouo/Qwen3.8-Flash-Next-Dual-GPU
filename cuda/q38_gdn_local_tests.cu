#include "q38/cuda_gdn.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iostream>
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
std::uint16_t bf16(float value) {
    const auto b = __float2bfloat16_rn(value);
    std::uint16_t result;
    std::memcpy(&result, &b, sizeof(result));
    return result;
}
struct Buffers {
    std::vector<void*> pointers;
    template <typename T> T* allocate(std::size_t count) {
        T* p = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "allocate");
        pointers.push_back(p);
        return p;
    }
    template <typename T> T* upload(const std::vector<T>& values) {
        auto* p = allocate<T>(values.size());
        check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
        return p;
    }
    ~Buffers() { for (void* p : pointers) (void)cudaFree(p); }
};
template <typename T> std::vector<T> read(const T* p, std::size_t n, cudaStream_t stream) {
    std::vector<T> values(n);
    check(cudaMemcpyAsync(values.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, stream), "read");
    check(cudaStreamSynchronize(stream), "read ready");
    return values;
}
template <typename T> void equal(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)))
        throw std::runtime_error(std::string("bit parity failed: ") + label);
}


unsigned canonical(unsigned local,unsigned r){if(local<1024)return r*1024+local;if(local<2048)return 2048+r*1024+local-1024;return 4096+r*3072+local-2048;}
void fixture(unsigned m,unsigned rank,bool decode,cudaStream_t s,unsigned layout_kind){
 const bool full_controls=layout_kind!=0;const bool packed=layout_kind==2;const bool full_qkv=layout_kind==3;
 Buffers mem;auto fill=[](std::size_t n,float scale){std::vector<std::uint16_t> x(n);for(std::size_t i=0;i<n;++i)x[i]=bf16(std::sin(float(i%10007)*.173f)*scale);return x;};
 auto q=fill(m*10240,.4f),b=fill(m*48,.2f),a=fill(m*48,.15f),z=fill(m*6144,.3f),w=fill(10240*4,.2f),al=fill(48,.3f),dt=fill(48,.1f),norm=fill(128,.4f),initc=fill(10240*4,.3f);
 std::vector<float> initr(48*128*128);for(unsigned i=0;i<initr.size();++i)initr[i]=std::sin(float(i)*.007f)*.01f;
 std::vector<std::uint16_t> lq(m*5120),lb(m*24),la(m*24),lz(m*3072);
 for(unsigned t=0;t<m;++t){for(unsigned i=0;i<5120;++i)lq[t*5120+i]=q[t*10240+canonical(i,rank)];for(unsigned i=0;i<24;++i){lb[t*24+i]=b[t*48+rank*24+i];la[t*24+i]=a[t*48+rank*24+i];}for(unsigned i=0;i<3072;++i)lz[t*3072+i]=z[t*6144+rank*3072+i];}
 std::vector<std::uint16_t> pq(m*8192);
 for(unsigned t=0;t<m;++t){std::copy_n(lq.begin()+t*5120,5120,pq.begin()+t*8192);std::copy_n(lz.begin()+t*3072,3072,pq.begin()+t*8192+5120);}
 auto*dpq=mem.upload(pq);
 q38::CudaGdnHeadLocalProjectionLayout layout;
 layout.full_qkv=full_qkv;layout.qkv_stride=packed?8192:(full_qkv?10240:5120);
 layout.control_heads=full_controls?48:24;layout.control_stride=layout.control_heads;
 layout.z_heads=full_controls&&!packed?48:24;layout.z_head_begin=layout.z_heads==48?rank*24:0;
 layout.z_stride=packed?8192:layout.z_heads*128;
 auto*dq=mem.upload(q),*db=mem.upload(b),*da=mem.upload(a),*dz=mem.upload(z),*dw=mem.upload(w),*dal=mem.upload(al),*ddt=mem.upload(dt),*dnorm=mem.upload(norm),*dlq=mem.upload(lq),*dlb=mem.upload(lb),*dla=mem.upload(la),*dlz=mem.upload(lz);
 q38::CudaGdnLayerStateView full{mem.upload(initc),mem.upload(initr)},local{mem.upload(initc),mem.upload(initr)};
 auto*af=mem.allocate<std::uint16_t>(m*10240);auto*ac=mem.allocate<std::uint16_t>(m*10240);auto*cf=mem.allocate<std::uint16_t>(m*6144);auto*cc=mem.allocate<std::uint16_t>(m*3072);auto*of=mem.allocate<std::uint16_t>(m*6144);auto*oc=mem.allocate<std::uint16_t>(m*3072);auto*params=mem.allocate<float>(q38::q38_gdn_prefill_parameter_floats(m));const auto norm_count=2ull*m*16*128;auto*norm_storage=mem.upload(std::vector<float>(norm_count+128,.375f));auto*normalized=norm_storage+64;
 constexpr unsigned nc=10240*4,nr=48*128*128;const unsigned count=decode?0:std::min(m,4u);
 auto*scf=mem.allocate<std::uint16_t>(std::max(count,1u)*nc);auto*scc=mem.allocate<std::uint16_t>(std::max(count,1u)*nc);auto*srf=mem.allocate<float>(std::max(count,1u)*nr);auto*src=mem.allocate<float>(std::max(count,1u)*nr);
 q38::CudaPrefixSnapshotView vc=count?q38::CudaPrefixSnapshotView{scf,nc*2,count}:q38::CudaPrefixSnapshotView{},vr=count?q38::CudaPrefixSnapshotView{srf,nr*4,count}:q38::CudaPrefixSnapshotView{},lc=count?q38::CudaPrefixSnapshotView{scc,nc*2,count}:q38::CudaPrefixSnapshotView{},lr=count?q38::CudaPrefixSnapshotView{src,nr*4,count}:q38::CudaPrefixSnapshotView{};
 auto launch=[&](bool own){if(own)q38::cuda_gdn_head_local_bf16(packed?dpq:(full_qkv?dq:dlq),full_controls?db:dlb,full_controls?da:dla,packed?dpq+5120:(full_controls?dz:dlz),dw,dal,ddt,dnorm,local,ac,cc,oc,params,m,rank,decode,s,0,lc,lr,layout,normalized);
 else if(decode){q38::cuda_gdn_conv_decode_bf16(dq,dw,full.conv,af,s,0);q38::cuda_gdn_recurrent_decode_bf16(af,db,da,dal,ddt,full.recurrent,cf,s,0);q38::cuda_gdn_output_norm_bf16(cf,dz,dnorm,of,1e-6f,s,0);}
 else{q38::cuda_gdn_conv_prefill_bf16(dq,dw,full.conv,af,m,s,0,vc);q38::cuda_gdn_recurrent_prefill_precomputed_bf16(af,db,da,dal,ddt,full.recurrent,params,cf,m,s,0,vr);q38::cuda_gdn_output_norm_prefill_bf16(cf,dz,dnorm,of,m,1e-6f,s,0);}};
 if (!decode && m>=8) {
  bool rejected=false;
  try { q38::cuda_gdn_head_local_bf16(packed?dpq:(full_qkv?dq:dlq),full_controls?db:dlb,full_controls?da:dla,packed?dpq+5120:(full_controls?dz:dlz),dw,dal,ddt,dnorm,local,ac,cc,oc,params,m,rank,decode,s,0,lc,lr,layout,nullptr); }
  catch (const std::invalid_argument&) { rejected=true; }
  if (!rejected) throw std::runtime_error("missing normalized Q/K cache accepted");
 }
 for(unsigned graph=0;graph<2;++graph){
  for(auto state:{full,local}){check(cudaMemcpyAsync(state.conv,initc.data(),nc*2,cudaMemcpyHostToDevice,s),"reset conv");check(cudaMemcpyAsync(state.recurrent,initr.data(),nr*4,cudaMemcpyHostToDevice,s),"reset recur");}check(cudaStreamSynchronize(s),"reset");
  for(unsigned own=0;own<2;++own){if(!graph)launch(own);else{cudaGraph_t g;cudaGraphExec_t e;check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");launch(own);check(cudaStreamEndCapture(s,&g),"end");check(cudaGraphInstantiate(&e,g,0),"instantiate");check(cudaGraphLaunch(e,s),"replay");check(cudaStreamSynchronize(s),"replay ready");cudaGraphExecDestroy(e);cudaGraphDestroy(g);}}
  const auto normalized_cells=read(norm_storage,norm_count+128,s);
  for(unsigned i=0;i<64;++i) if(normalized_cells[i]!=.375f||normalized_cells[64+norm_count+i]!=.375f)throw std::runtime_error("normalized Q/K cache guard changed");
  for(unsigned plane=0;plane<2;++plane)for(unsigned t=0;t<m;++t)for(unsigned h=(1-rank)*8;h<(2-rank)*8;++h)for(unsigned e=0;e<128;++e)
   if(normalized_cells[64+(std::size_t(plane)*m*16+t*16+h)*128+e]!=.375f)throw std::runtime_error("normalized Q/K unowned head changed");
  const auto yf=read(of,m*6144,s),yc=read(oc,m*3072,s);std::vector<std::uint16_t> expected;
  for(unsigned t=0;t<m;++t)expected.insert(expected.end(),yf.begin()+t*6144+rank*3072,yf.begin()+t*6144+(rank+1)*3072);equal(expected,yc,"output");
  const auto fc=read(full.conv,nc,s),lcells=read(local.conv,nc,s);for(unsigned i=0;i<5120;++i)for(unsigned c=0;c<4;++c)if(fc[canonical(i,rank)*4+c]!=lcells[canonical(i,rank)*4+c])throw std::runtime_error("conv state");
  const auto fr=read(full.recurrent,nr,s),lrcells=read(local.recurrent,nr,s);std::vector<float> ex(fr.begin()+rank*nr/2,fr.begin()+(rank+1)*nr/2),got(lrcells.begin()+rank*nr/2,lrcells.begin()+(rank+1)*nr/2);equal(ex,got,"recurrent state");
  if(count){const auto fcs=read(scf,count*nc,s),lcs=read(scc,count*nc,s);for(unsigned t=0;t<count;++t)for(unsigned i=0;i<5120;++i)for(unsigned c=0;c<4;++c)if(fcs[t*nc+canonical(i,rank)*4+c]!=lcs[t*nc+canonical(i,rank)*4+c])throw std::runtime_error("conv prefix");const auto frs=read(srf,count*nr,s),lrs=read(src,count*nr,s);for(unsigned t=0;t<count;++t){std::vector<float> x(frs.begin()+t*nr+rank*nr/2,frs.begin()+t*nr+(rank+1)*nr/2),y(lrs.begin()+t*nr+rank*nr/2,lrs.begin()+t*nr+(rank+1)*nr/2);equal(x,y,"recurrent prefix");}}
 }
 if(m==512 && layout_kind==3){
  const auto initial_conv=read(local.conv,nc,s);const auto initial_recurrent=read(local.recurrent,nr,s);
  auto* reset_conv=mem.upload(initial_conv);auto*reset_recurrent=mem.upload(initial_recurrent);
  auto reset=[&]{check(cudaMemcpyAsync(local.conv,reset_conv,nc*2,cudaMemcpyDeviceToDevice,s),"bench reset conv");check(cudaMemcpyAsync(local.recurrent,reset_recurrent,nr*4,cudaMemcpyDeviceToDevice,s),"bench reset recur");};
  for(unsigned enabled:{0u,1u}) for(unsigned columns:{32u,16u,8u}){
   const auto setting=std::to_string(columns);setenv("Q38_CUDA_GDN_PREFILL_PARTITION_COLUMNS",setting.c_str(),1);setenv("Q38_CUDA_GDN_PREFILL_NORMALIZED_QK",enabled?"1":"0",1);
   for(int i=0;i<3;i++){reset();launch(true);}check(cudaStreamSynchronize(s),"bench warm");
   cudaEvent_t a,b;check(cudaEventCreate(&a),"bench event");check(cudaEventCreate(&b),"bench event");std::vector<float> samples;
   for(int i=0;i<20;i++){reset();check(cudaEventRecord(a,s),"bench start");launch(true);check(cudaEventRecord(b,s),"bench end");check(cudaEventSynchronize(b),"bench ready");float ms;check(cudaEventElapsedTime(&ms,a,b),"bench elapsed");samples.push_back(ms*1000);}
   cudaEventDestroy(a);cudaEventDestroy(b);std::sort(samples.begin(),samples.end());
   std::cout<<"{\"bulk_tokens\":"<<m<<",\"rank\":"<<rank<<",\"normalized_qk\":"<<enabled<<",\"partition_columns\":"<<columns<<",\"head_local_no_reset_median_us\":"<<(samples[9]+samples[10])/2<<"}\n";
  }
  setenv("Q38_CUDA_GDN_PREFILL_PARTITION_COLUMNS","16",1);setenv("Q38_CUDA_GDN_PREFILL_NORMALIZED_QK","1",1);
 }
 std::cout<<"{\"M\":"<<m<<",\"rank\":"<<rank<<",\"decode\":"<<decode<<",\"eager_graph_owned_bit_parity\":true}\n";
}
}
int main(){try{setenv("Q38_CUDA_GDN_PREFILL_NORMALIZED_QK","1",1);setenv("Q38_CUDA_GDN_PREFILL_PARTITION_COLUMNS","16",1);cudaStream_t s;check(cudaSetDevice(0),"device");check(cudaStreamCreate(&s),"stream");for(unsigned r:{0u,1u}){for(unsigned full:{0u,1u,2u,3u}){fixture(1,r,true,s,full);for(unsigned m:{1u,5u,8u,15u,127u,512u,513u})fixture(m,r,false,s,full);}}cudaStreamDestroy(s);std::cout<<"GDN_LOCAL_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
