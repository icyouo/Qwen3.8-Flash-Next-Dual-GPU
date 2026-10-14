// P2P-based AllReduce microbench on 2 GPUs, no NCCL.
// Compile: nvcc -O3 -arch=sm_80 -o p2p_ar p2p_ar.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1);} } while(0)

// BF16-style: use __nv_bfloat16 for realism; here we use float32 for simplicity
// (kernel size dominates; format only affects β at very high bandwidth)
__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out, size_t n) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; idx < n; idx += stride) out[idx] = a[idx] + b[idx];
}

struct DevBuf {
    int dev;
    float* ptr = nullptr;
    size_t n = 0;
};

static void allocate(DevBuf& b, int dev, size_t elems) {
    CK(cudaSetDevice(dev));
    b.dev = dev; b.n = elems;
    CK(cudaMalloc(&b.ptr, elems * sizeof(float)));
    CK(cudaMemset(b.ptr, 0, elems * sizeof(float)));
}
static void freebuf(DevBuf& b) {
    CK(cudaSetDevice(b.dev));
    CK(cudaFree(b.ptr));
    b.ptr = nullptr;
}

// Two-rank ring AllReduce:
//   1) each rank sends its buf to peer's scratch
//   2) each rank kernel-adds scratch into local
//   3) each rank has the reduced result
static void allreduce_pair(DevBuf& a0, DevBuf& a1,
                           DevBuf& scratch0, DevBuf& scratch1,
                           cudaStream_t s0, cudaStream_t s1) {
    size_t bytes = a0.n * sizeof(float);
    // Cross-copy: peer's data → local scratch
    CK(cudaSetDevice(a0.dev));
    CK(cudaMemcpyPeerAsync(scratch0.ptr, a0.dev, a1.ptr, a1.dev, bytes, s0));
    CK(cudaSetDevice(a1.dev));
    CK(cudaMemcpyPeerAsync(scratch1.ptr, a1.dev, a0.ptr, a0.dev, bytes, s1));
    // Local reductions
    dim3 block(256);
    dim3 grid(std::min<size_t>((a0.n + 255) / 256, 4096));
    CK(cudaSetDevice(a0.dev));
    add_kernel<<<grid, block, 0, s0>>>(a0.ptr, scratch0.ptr, a0.ptr, a0.n);
    CK(cudaSetDevice(a1.dev));
    add_kernel<<<grid, block, 0, s1>>>(a1.ptr, scratch1.ptr, a1.ptr, a1.n);
}

static double bench(size_t elems, int iters, bool warmup_only = false) {
    DevBuf a0{}, a1{}, s0b{}, s1b{};
    allocate(a0, 0, elems);
    allocate(a1, 1, elems);
    allocate(s0b, 0, elems);
    allocate(s1b, 1, elems);
    cudaStream_t st0, st1;
    CK(cudaSetDevice(0)); CK(cudaStreamCreate(&st0));
    CK(cudaSetDevice(1)); CK(cudaStreamCreate(&st1));
    // warmup
    for (int i = 0; i < 5; ++i) allreduce_pair(a0, a1, s0b, s1b, st0, st1);
    CK(cudaStreamSynchronize(st0));
    CK(cudaStreamSynchronize(st1));
    if (warmup_only) {
        freebuf(a0); freebuf(a1); freebuf(s0b); freebuf(s1b);
        cudaStreamDestroy(st0); cudaStreamDestroy(st1);
        return 0.0;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) allreduce_pair(a0, a1, s0b, s1b, st0, st1);
    CK(cudaStreamSynchronize(st0));
    CK(cudaStreamSynchronize(st1));
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    freebuf(a0); freebuf(a1); freebuf(s0b); freebuf(s1b);
    cudaStreamDestroy(st0); cudaStreamDestroy(st1);
    return sec / iters;  // seconds per iter
}

// Simulate 48 back-to-back layer AllReduces to see whether launches serialize
static double bench_48layers(size_t elems, int iters) {
    // Allocate 48 independent buffer pairs to prevent WAW dependency
    const int L = 48;
    std::vector<DevBuf> a0(L), a1(L), s0(L), s1(L);
    for (int i = 0; i < L; ++i) {
        allocate(a0[i], 0, elems);
        allocate(a1[i], 1, elems);
        allocate(s0[i], 0, elems);
        allocate(s1[i], 1, elems);
    }
    cudaStream_t st0, st1;
    CK(cudaSetDevice(0)); CK(cudaStreamCreate(&st0));
    CK(cudaSetDevice(1)); CK(cudaStreamCreate(&st1));
    // warmup
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < L; ++i)
            allreduce_pair(a0[i], a1[i], s0[i], s1[i], st0, st1);
    CK(cudaStreamSynchronize(st0));
    CK(cudaStreamSynchronize(st1));
    auto t0 = std::chrono::steady_clock::now();
    for (int j = 0; j < iters; ++j) {
        for (int i = 0; i < L; ++i)
            allreduce_pair(a0[i], a1[i], s0[i], s1[i], st0, st1);
    }
    CK(cudaStreamSynchronize(st0));
    CK(cudaStreamSynchronize(st1));
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    for (int i = 0; i < L; ++i) {
        freebuf(a0[i]); freebuf(a1[i]); freebuf(s0[i]); freebuf(s1[i]);
    }
    cudaStreamDestroy(st0); cudaStreamDestroy(st1);
    return sec / iters;  // seconds per (48-layer forward) iter
}

int main() {
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    printf("Devices: %d\n", ndev);
    if (ndev < 2) { printf("Need 2 GPUs\n"); return 1; }
    int can01 = 0, can10 = 0;
    CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    printf("CanAccessPeer 0->1=%d  1->0=%d\n", can01, can10);
    if (can01) { CK(cudaSetDevice(0)); cudaError_t e = cudaDeviceEnablePeerAccess(1, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
            fprintf(stderr, "enable 0->1: %s\n", cudaGetErrorString(e)); }
    if (can10) { CK(cudaSetDevice(1)); cudaError_t e = cudaDeviceEnablePeerAccess(0, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
            fprintf(stderr, "enable 1->0: %s\n", cudaGetErrorString(e)); }

    // Single AllReduce sweep
    struct Row { const char* label; size_t elems; int iters; };
    Row rows[] = {
        {"1 KiB",     256,           5000},
        {"4 KiB",     1024,          5000},
        {"10 KiB",    2560,          5000},   // ~ per-layer AR at d_model=2560, BF16 (halved for FP32 sim)
        {"30 KiB",    7680,          3000},   // 3-layer fused
        {"100 KiB",   25600,         2000},
        {"1 MiB",     262144,        500},
        {"10 MiB",    2621440,       100},
        {"100 MiB",   26214400,      20},
    };
    printf("\n=== single AllReduce ===\n");
    printf("%-10s  %12s  %10s  %10s\n", "size", "per_call_us", "GB/s_bidir", "GB/s_uni");
    for (auto& r : rows) {
        double sec = bench(r.elems, r.iters);
        double bytes = double(r.elems) * sizeof(float);
        // AllReduce moves 2× data on the wire (bidirectional exchange)
        double gbs_bidir = 2.0 * bytes / sec / 1e9;
        double gbs_uni = bytes / sec / 1e9;
        printf("%-10s  %12.2f  %10.2f  %10.2f\n",
               r.label, sec * 1e6, gbs_bidir, gbs_uni);
    }

    // 48-layer chained AllReduce (simulates one token forward's comm cost)
    printf("\n=== 48-layer chained (per-token comm budget) ===\n");
    printf("%-10s  %12s  %12s\n", "per_layer_size", "48x_wall_ms", "per_layer_us");
    struct L48 { const char* label; size_t elems; };
    L48 l48[] = {
        {"10 KiB", 2560},
        {"30 KiB (fused 3)", 7680},
        {"100 KiB (fused 10)", 25600},
    };
    for (auto& r : l48) {
        double sec = bench_48layers(r.elems, 200);
        printf("%-20s  %12.3f  %12.2f\n",
               r.label, sec * 1e3, sec / 48.0 * 1e6);
    }

    return 0;
}
