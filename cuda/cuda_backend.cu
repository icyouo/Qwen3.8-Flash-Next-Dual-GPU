#include "q38/cuda_te_graph.h"
#include "q38/cuda_backend.h"
#include "q38/cuda_te.h"
#include "q38/cuda_sampling.h"
#include "q38/cuda_padded_gemm.h"
#include "q38/ple_prefetch.h"

#include "q38/cuda_gdn.h"
#include "q38/cuda_hyper.h"
#include "q38/cuda_kernels.h"
#include "q38/cuda_moe.h"
#include "q38/cuda_moe_prefill.h"
#include "q38/cuda_marlin.h"
#include "q38/cuda_ple.h"
#include "q38/cuda_qsa.h"
#include "q38/cuda_transport.h"
#include "q38/cuda_weights.h"
#include "q38/production_contract.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <map>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace q38 {

namespace {

constexpr std::uint32_t kVocabulary = 248320;
constexpr std::uint32_t kLayers = 48;
constexpr std::uint32_t kCut = kQ38ProductionCut;
// The grouped-MMQ lane needs enough assignments per expert to amortize weight
// dequantization and Tensor-Core tiles.  This is also the cancellation and
// stage-boundary slab size; decode remains a separate batch-1 lane.
constexpr std::uint32_t kPrefillTile = kQ38PrefillSlabMaxTokens;
constexpr std::uint32_t kSmallBoundaryTokens = 128;

__global__ void te_remap_experts_kernel(std::int32_t* ids, float* weights,
                                        std::size_t count, int base, int local) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int id = ids[i] - base;
    ids[i] = id >= 0 && id < local ? id : -1;
    if (ids[i] < 0) weights[i] = 0.0f;
}

__global__ void te_round_f32_bf16_kernel(const float* input, std::uint16_t* output,
                                        std::size_t count) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) reinterpret_cast<__nv_bfloat16*>(output)[i] = __float2bfloat16_rn(input[i]);
}

__global__ void head_candidate_kernel(const std::uint16_t* logits,const std::int32_t* local_ids,
    std::uint16_t* candidates,unsigned rows,unsigned tokens,unsigned rank){
    const unsigned row=threadIdx.x;if(row>=tokens)return;const unsigned local=local_ids[row],id=rank*rows+local;
    candidates[row*4]=logits[static_cast<std::uint64_t>(row)*rows+local];candidates[row*4+1]=id&65535u;candidates[row*4+2]=id>>16;candidates[row*4+3]=0;
}
__global__ void head_choose_kernel(const std::uint16_t* candidates,std::int32_t* output,unsigned tokens){
    const unsigned row=threadIdx.x;if(row>=tokens)return;const auto* c=candidates+row*8;
    const unsigned a=c[1]|(unsigned(c[2])<<16),b=c[5]|(unsigned(c[6])<<16);
    const float av=__bfloat162float(__ushort_as_bfloat16(c[0])),bv=__bfloat162float(__ushort_as_bfloat16(c[4]));output[row]=bv>av||(bv==av&&b<a)?b:a;
}

inline int te_bridge_group_for(const CudaStageBackendOptions& opt) {
    return opt.stage == Stage::kStage1 ? 1 : 0;
}

inline float te_bf16_to_float(std::uint16_t bits) {
    const std::uint32_t bits32 = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}
inline void te_debug_probe_bf16(const char* tag, int rank, const std::uint16_t* device_ptr,
                                cudaStream_t stream) {
    static const bool enabled = [] {
        const char* v = std::getenv("Q38_TE_DEBUG");
        return v && v[0] && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0;
    }();
    if (!enabled) return;
    if (!device_ptr) return;
    std::uint16_t bits = 0;
    if (cudaMemcpyAsync(&bits, device_ptr, sizeof(bits), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess)
        return;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    std::fprintf(stderr, "te-probe rank=%d %s bits=0x%04x value=%g\n",
                 rank, tag, bits, te_bf16_to_float(bits));
    std::fflush(stderr);
}

bool profile_requested(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0;
}

std::uint32_t bounded_setting(const char* name, std::uint32_t fallback,
                              std::uint32_t maximum) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoul(value, &end, 10);
    if (value == end || *end != '\0' || parsed == 0 || parsed > maximum)
        throw std::invalid_argument(std::string("invalid ") + name);
    return static_cast<std::uint32_t>(parsed);
}
std::uint32_t prefill_tile_tokens() {
    static const auto value = bounded_setting("Q38_CUDA_PREFILL_TILE", 512, 512);
    return value;
}
std::uint32_t ple_io_tile_tokens() {
    static const auto value = bounded_setting("Q38_PLE_IO_TILE", 512, 4096);
    return value;
}
std::uint32_t prefill_sync_tiles() {
    static const auto value = bounded_setting("Q38_CUDA_PREFILL_SYNC_TILES", 1, 8);
    return value;
}

bool grouped_qsa_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("Q38_CUDA_PREFILL_QSA");
        if (!value) return true;
        return std::strcmp(value, "legacy") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

bool fused_grouped_qsa_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("Q38_CUDA_PREFILL_QSA_FUSED");
        return value && std::strcmp(value, "split") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

bool partitioned_gdn_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("Q38_CUDA_PREFILL_GDN");
        if (!value) return true;
        return std::strcmp(value, "serial") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

bool precomputed_gdn_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("Q38_CUDA_PREFILL_GDN");
        if (!value) return true;
        return std::strcmp(value, "partitioned") != 0 &&
               std::strcmp(value, "serial") != 0 &&
               std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

Q38PrefillMoeModeV1 prefill_moe_mode() {
    static const Q38PrefillMoeModeV1 mode = [] {
        const char* value = std::getenv("Q38_CUDA_PREFILL_MOE");
        if (!value || std::strcmp(value, "grouped") == 0 ||
            std::strcmp(value, "1") == 0)
            return Q38PrefillMoeModeV1::kGroupedMmq;
        if (std::strcmp(value, "legacy") == 0 ||
            std::strcmp(value, "0") == 0)
            return Q38PrefillMoeModeV1::kLegacyAtomicDiagnostic;
        if (std::strcmp(value, "safe") == 0 ||
            std::strcmp(value, "grouped_safe") == 0)
            return Q38PrefillMoeModeV1::kGroupedMmqSafe;
        throw std::invalid_argument("invalid Q38_CUDA_PREFILL_MOE mode");
    }();
    return mode;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}

__global__ void graph_clock_marker(std::uint64_t* timestamp) {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    *timestamp = value;
}

// Opt-in event profiling for real-model batch-1 decode.  It has no CUDA
// events, records, synchronization, or logging cost unless explicitly enabled.
// Repeated labels are aggregated across all layers owned by the stage.
class CudaEventProfile {
public:
    CudaEventProfile(int value_device, std::string value_type, bool external = false)
        : device_(value_device), type_(std::move(value_type)), external_(external) {
        if (external_) {
            check(cudaSetDevice(device_), "select graph clock device");
            check(cudaMalloc(reinterpret_cast<void**>(&device_clocks_), clock_capacity_ * sizeof(std::uint64_t)), "allocate graph clock markers");
        }
    }
    ~CudaEventProfile() {
        (void)cudaSetDevice(device_);
        for (auto event : events_) (void)cudaEventDestroy(event);
        if (device_clocks_) (void)cudaFree(device_clocks_);
    }

    CudaEventProfile(const CudaEventProfile&) = delete;
    CudaEventProfile& operator=(const CudaEventProfile&) = delete;

    void begin(cudaStream_t stream) {
        labels_.clear();
        event_count_ = 0;
        active_ = true;
        record(stream);
    }

    void mark(const char* label, cudaStream_t stream) {
        if (!active_) return;
        labels_.emplace_back(label);
        record(stream);
    }

    void finish(Stage stage, std::uint64_t position, cudaStream_t stream,
                std::uint32_t tokens = 1, bool reusable = false) {
        if (!active_) return;
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize(decode profile)");
        std::vector<std::uint64_t> clocks;
        if (external_) {
            clocks.resize(event_count_);
            check(cudaMemcpy(clocks.data(), device_clocks_, event_count_*sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "read graph clock markers");
        }
        std::vector<std::pair<std::string, double>> totals;
        double total_ms = 0.0;
        for (std::size_t index = 0; index < labels_.size(); ++index) {
            float elapsed_ms = 0.0f;
            if (external_) elapsed_ms = static_cast<float>(clocks[index+1]-clocks[index]) * 1.0e-6f;
            else check(cudaEventElapsedTime(&elapsed_ms, events_[index],
                                       events_[index + 1]),
                  "cudaEventElapsedTime(decode profile)");
            total_ms += elapsed_ms;
            const auto found = std::find_if(
                totals.begin(), totals.end(), [&](const auto& entry) {
                    return entry.first == labels_[index];
                });
            if (found == totals.end())
                totals.emplace_back(labels_[index], elapsed_ms);
            else
                found->second += elapsed_ms;
        }
        std::ostringstream record;
        record << "{\"type\":\"q38_cuda_" << type_
                  << "_profile\",\"stage\":"
                  << static_cast<unsigned>(stage) << ",\"device\":" << device_
                  << ",\"tokens\":" << tokens << ",\"position\":"
                  << position << ",\"total_ms\":" << total_ms
                  << ",\"categories_ms\":{";
        for (std::size_t index = 0; index < totals.size(); ++index) {
            if (index) record << ',';
            record << '\"' << totals[index].first << "\":"
                      << totals[index].second;
        }
        record << "}";
        if (external_ && type_ == "prefill" &&
            profile_requested("Q38_CUDA_PROFILE_PREFILL_TIMELINE")) {
            record << ",\"intervals_ns\":[";
            for (std::size_t index = 0; index < labels_.size(); ++index) {
                if (index) record << ',';
                record << "{\"label\":\"" << labels_[index]
                       << "\",\"start_ns\":" << clocks[index]
                       << ",\"end_ns\":" << clocks[index + 1] << '}';
            }
            record << ']';
        }
        record << "}\n";
        // Rank threads finish together. Publish each JSON record atomically
        // through stdio rather than interleaving individual stream inserts.
        std::fputs(record.str().c_str(), stderr);
        std::fflush(stderr);
        active_ = reusable;
    }

    bool active() const { return active_; }

private:
    void record(cudaStream_t stream) {
        if (external_) {
            if (event_count_ >= clock_capacity_) throw std::runtime_error("graph clock marker capacity exceeded");
            graph_clock_marker<<<1,1,0,stream>>>(device_clocks_ + event_count_++);
            check(cudaPeekAtLastError(), "graph clock marker");
            return;
        }
        if (event_count_ == events_.size()) {
            cudaEvent_t event = nullptr;
            check(cudaEventCreate(&event),
                  "cudaEventCreate(decode profile)");
            events_.push_back(event);
        }
        check(cudaEventRecordWithFlags(events_[event_count_++], stream,
                                        external_ ? cudaEventRecordExternal : cudaEventRecordDefault),
              "cudaEventRecord(decode profile)");
    }

    static constexpr std::size_t clock_capacity_ = 4096;
    std::uint64_t* device_clocks_ = nullptr;
    int device_ = 0;
    std::string type_;
    bool external_ = false;
    bool active_ = false;
    std::size_t event_count_ = 0;
    std::vector<cudaEvent_t> events_;
    std::vector<std::string> labels_;
};

// One captured GPU-only fragment.  It deliberately owns no tensor storage:
// every node points at immutable weights or the backend's lifetime-stable
// workspace/state allocations.  The owner must therefore destroy graph execs
// before releasing those allocations.
class CapturedCudaGraph {
public:
    CapturedCudaGraph(int value_device, cudaStream_t stream,
                      const std::function<void()>& enqueue, bool tp = false,
                      std::unique_ptr<CudaEventProfile> profile = {},
                      bool bulk_prefill = false)
        : device_(value_device), profile_(std::move(profile)) {
        if (tp) {
            tape_ = std::make_unique<CudaTeGraphTape>(value_device, stream, enqueue, bulk_prefill);
            nodes_ = tape_->nodes();
            return;
        }
        check(cudaSetDevice(device_), "cudaSetDevice(graph capture)");
        // Capture is lazy and happens after transaction inputs/state clones
        // have been enqueued.  Drain that setup once so it stays outside the
        // graph and capture begins from an unambiguous stream boundary.
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize(graph capture boundary)");
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
              "cudaStreamBeginCapture(decode fragment)");
        bool capturing = true;
        try {
            enqueue();
            const auto end_status = cudaStreamEndCapture(stream, &graph_);
            // EndCapture terminates capture even when it reports an
            // invalidated graph. Do not call it a second time from cleanup.
            capturing = false;
            check(end_status, "cudaStreamEndCapture(decode fragment)");
            if (!graph_)
                throw std::runtime_error(
                    "CUDA decode fragment capture returned a null graph");
            check(cudaGraphGetNodes(graph_, nullptr, &nodes_),
                  "cudaGraphGetNodes(decode fragment)");
            check(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0),
                  "cudaGraphInstantiate(decode fragment)");
            // The executable owns the instantiated node state. The source
            // graph is no longer needed after node census and instantiation.
            check(cudaGraphDestroy(graph_),
                  "cudaGraphDestroy(instantiated decode fragment)");
            graph_ = nullptr;
        } catch (...) {
            if (capturing) {
                cudaGraph_t discarded = nullptr;
                (void)cudaStreamEndCapture(stream, &discarded);
                if (discarded) (void)cudaGraphDestroy(discarded);
                (void)cudaGetLastError();
            }
            release();
            throw;
        }
    }

    ~CapturedCudaGraph() { release(); }

    CapturedCudaGraph(const CapturedCudaGraph&) = delete;
    CapturedCudaGraph& operator=(const CapturedCudaGraph&) = delete;

    void launch(cudaStream_t stream, const std::function<void()>& cancel = {}) const {
        if (tape_) { tape_->launch(stream, cancel); return; }
        if (cancel) cancel();
        if (!exec_) throw std::logic_error("CUDA decode graph is unavailable");
        check(cudaGraphLaunch(exec_, stream),
              "cudaGraphLaunch(decode fragment)");
    }

    void profile_finish(Stage stage, std::uint64_t position,
                        cudaStream_t stream, std::uint32_t tokens) {
        if (profile_) profile_->finish(stage, position, stream, tokens, true);
    }

    std::size_t nodes() const { return nodes_; }
    std::size_t graphs() const { return tape_ ? tape_->graphs() : 1; }
    std::size_t held_bytes() const { return tape_ ? tape_->held_bytes() : 0; }

private:
    void release() noexcept {
        (void)cudaSetDevice(device_);
        tape_.reset();
        if (exec_) (void)cudaGraphExecDestroy(exec_);
        if (graph_) (void)cudaGraphDestroy(graph_);
        exec_ = nullptr;
        graph_ = nullptr;
        nodes_ = 0;
    }

    std::unique_ptr<CudaEventProfile> profile_;
    std::unique_ptr<CudaTeGraphTape> tape_;
    int device_ = 0;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    std::size_t nodes_ = 0;
};

std::uint64_t elements(const DeviceTensorV1& tensor) {
    std::uint64_t result = 1;
    for (const auto dimension : tensor.shape) {
        if (dimension == 0 ||
            result > std::numeric_limits<std::uint64_t>::max() / dimension)
            throw std::runtime_error("CUDA tensor shape overflows");
        result *= dimension;
    }
    return result;
}

const std::uint16_t* require_bf16(const CudaDeviceWeightStore& weights,
                                  const std::string& name,
                                  std::uint64_t expected_elements) {
    const auto tensor = weights.require(name);
    if (tensor.descriptor->format != DeviceWeightFormatV1::kPreserve ||
        tensor.descriptor->source_dtype != "BF16" ||
        elements(*tensor.descriptor) != expected_elements)
        throw std::runtime_error("invalid BF16 control tensor " + name);
    return static_cast<const std::uint16_t*>(tensor.data);
}

thread_local int g_te_bind_world_size = 1;

// When te_world_size==2, TP2 shards may halve rows (row_shard) or columns
// (col_shard). Accept exact full shape or exact half along one axis.
CudaMatrixViewV1 require_matrix(const CudaDeviceWeightStore& weights,
                                const std::string& name,
                                std::uint32_t rows,
                                std::uint32_t columns) {
    auto result = cuda_matrix_view(weights.require(name));
    const int te_world_size = g_te_bind_world_size;
    const bool exact = result.rows == rows && result.columns == columns;
    const bool half_rows =
        te_world_size == 2 && result.rows * 2u == rows &&
        result.columns == columns;
    const bool half_cols =
        te_world_size == 2 && result.rows == rows &&
        result.columns * 2u == columns;
    if (!exact && !half_rows && !half_cols)
        throw std::runtime_error(
            "invalid CUDA matrix dimensions " + name + " got " +
            std::to_string(result.rows) + "x" + std::to_string(result.columns) +
            " want " + std::to_string(rows) + "x" + std::to_string(columns) +
            (te_world_size == 2 ? " (te half ok)" : ""));
    return result;
}

struct HyperWeights {
    const std::uint16_t* norm = nullptr;
    CudaMatrixViewV1 mix_down;
    CudaMatrixViewV1 mix_up;
    CudaMatrixViewV1 inject;
    bool has_inject = false;
};

HyperWeights bind_hyper(const CudaDeviceWeightStore& weights,
                        const std::string& prefix, bool inject) {
    HyperWeights result;
    result.norm = require_bf16(weights, prefix + ".norm.weight", kQ38HyperWidth);
    result.mix_down = require_matrix(weights, prefix + ".mix_down.weight",
                                     kQ38HyperLowRank, kQ38HyperWidth);
    result.mix_up = require_matrix(weights, prefix + ".mix_up.weight",
                                   kQ38HyperWidth, kQ38HyperLowRank);
    result.has_inject = inject;
    if (inject)
        result.inject = require_matrix(weights, prefix + ".inject.weight",
                                       kQ38HyperCount, kQ38HyperWidth);
    return result;
}

struct GdnWeights {
    CudaMatrixViewV1 qkv;
    CudaMatrixViewV1 qkv_prefill_full;
    CudaMatrixViewV1 packed;
    CudaMatrixViewV1 z;
    CudaMatrixViewV1 b;
    CudaMatrixViewV1 a;
    CudaMatrixViewV1 output;
    const std::uint16_t* conv = nullptr;
    const std::uint16_t* a_log = nullptr;
    const std::uint16_t* dt_bias = nullptr;
    const std::uint16_t* norm = nullptr;
};

class ScopedQsaHeadLayout {
public:
    explicit ScopedQsaHeadLayout(CudaQsaHeadLayout layout)
        : previous_(cuda_qsa_set_head_layout(layout)) {}
    ~ScopedQsaHeadLayout() { (void)cuda_qsa_set_head_layout(previous_); }
    ScopedQsaHeadLayout(const ScopedQsaHeadLayout&) = delete;
    ScopedQsaHeadLayout& operator=(const ScopedQsaHeadLayout&) = delete;
private:
    CudaQsaHeadLayout previous_;
};

struct QsaWeights {
    CudaMatrixViewV1 index_qk;
    CudaMatrixViewV1 q_gate;
    CudaMatrixViewV1 q_gate_local;
    CudaMatrixViewV1 k;
    CudaMatrixViewV1 v;
    CudaMatrixViewV1 output;
    const std::uint16_t* index_q_norm = nullptr;
    const std::uint16_t* index_k_norm = nullptr;
    const std::uint16_t* q_norm = nullptr;
    const std::uint16_t* k_norm = nullptr;
};

struct MoeWeights {
    CudaMatrixViewV1 router;
    CudaTensorViewV1 gate_up_experts;
    CudaTensorViewV1 down_experts;
    CudaMatrixViewV1 shared_gate;
    CudaMatrixViewV1 shared_up;
    CudaMatrixViewV1 shared_down;
    CudaMatrixViewV1 shared_output_gate;
};

struct PleWeights {
    CudaMatrixViewV1 key;
    CudaMatrixViewV1 value;
    const std::uint16_t* key_norm = nullptr;
    const std::uint16_t* query_norm = nullptr;
    const std::uint16_t* conv_norm = nullptr;
    const std::uint16_t* conv = nullptr;
};

struct LayerWeights {
    std::uint32_t global = 0;
    std::uint32_t state_local = 0;
    bool qsa = false;
    bool has_ple = false;
    HyperWeights attention_hyper;
    HyperWeights moe_hyper;
    GdnWeights gdn;
    QsaWeights sparse;
    MoeWeights moe;
    PleWeights ple;
};

MoeWeights bind_moe(const CudaDeviceWeightStore& weights,
                    const std::string& prefix) {
    MoeWeights result;
    result.router = require_matrix(weights, prefix + "ffn_gate_inp.weight",
                                   kQ38MoeExperts, kQ38HiddenWidth);
    result.gate_up_experts = weights.require(prefix + "ffn_gate_up_exps.weight");
    result.down_experts = weights.require(prefix + "ffn_down_exps.weight");
    result.shared_gate = require_matrix(
        weights, prefix + "ffn_gate_shexp.weight", kQ38MoeIntermediate,
        kQ38HiddenWidth);
    result.shared_up = require_matrix(
        weights, prefix + "ffn_up_shexp.weight", kQ38MoeIntermediate,
        kQ38HiddenWidth);
    result.shared_down = require_matrix(
        weights, prefix + "ffn_down_shexp.weight", kQ38HiddenWidth,
        kQ38MoeIntermediate);
    result.shared_output_gate = require_matrix(
        weights, prefix + "ffn_shexp_gate_inp.weight", 1,
        kQ38HiddenWidth);
    return result;
}

LayerWeights bind_layer(const CudaDeviceWeightStore& weights,
                        std::uint32_t layer, std::uint32_t state_local) {
    LayerWeights result;
    result.global = layer;
    result.state_local = state_local;
    result.qsa = layer % 4 == 3;
    result.has_ple = layer == 1;
    const auto prefix = "blk." + std::to_string(layer) + ".";
    result.attention_hyper = bind_hyper(weights, prefix + "hc_attn", true);
    result.moe_hyper = bind_hyper(weights, prefix + "hc_ffn", true);
    if (result.qsa) {
        result.sparse.index_qk = require_matrix(
            weights, prefix + "attn_index_qk.weight", 640, kQ38HiddenWidth);
        result.sparse.index_q_norm = require_bf16(
            weights, prefix + "attn_index_q_norm.weight", kQ38QsaIndexerWidth);
        result.sparse.index_k_norm = require_bf16(
            weights, prefix + "attn_index_k_norm.weight", kQ38QsaIndexerWidth);
        result.sparse.q_gate = require_matrix(
            weights, prefix + "attn_q.weight", 12288, kQ38HiddenWidth);
        result.sparse.q_norm = require_bf16(
            weights, prefix + "attn_q_norm.weight", kQ38QsaHeadWidth);
        result.sparse.k = require_matrix(weights, prefix + "attn_k.weight", 512,
                                         kQ38HiddenWidth);
        result.sparse.k_norm = require_bf16(
            weights, prefix + "attn_k_norm.weight", kQ38QsaHeadWidth);
        result.sparse.v = require_matrix(weights, prefix + "attn_v.weight", 512,
                                         kQ38HiddenWidth);
        result.sparse.output = require_matrix(
            weights, prefix + "attn_output.weight", kQ38HiddenWidth, 6144);
    } else {
        result.gdn.a_log = require_bf16(weights, prefix + "linear_attn.a_log", 48);
        result.gdn.dt_bias =
            require_bf16(weights, prefix + "linear_attn.dt_bias", 48);
        result.gdn.conv = require_bf16(
            weights, prefix + "linear_attn.conv.weight",
            static_cast<std::uint64_t>(kQ38GdnQkvWidth) * kQ38GdnConvWidth);
        result.gdn.qkv = require_matrix(
            weights, prefix + "linear_attn.qkv.weight", kQ38GdnQkvWidth,
            kQ38HiddenWidth);
        if(!weights.find(prefix+"linear_attn.qkv_prefill_full.weight").empty())
            result.gdn.qkv_prefill_full=require_matrix(weights,prefix+"linear_attn.qkv_prefill_full.weight",10240,kQ38HiddenWidth);
        if(!weights.find(prefix+"linear_attn.qkvzba_packed.weight").empty())
            result.gdn.packed=require_matrix(weights,prefix+"linear_attn.qkvzba_packed.weight",8192,kQ38HiddenWidth);
        result.gdn.z = require_matrix(weights, prefix + "linear_attn.z.weight",
                                      kQ38GdnValueWidth, kQ38HiddenWidth);
        result.gdn.b = require_matrix(weights, prefix + "linear_attn.in_b.weight",
                                      kQ38GdnValueHeads, kQ38HiddenWidth);
        result.gdn.a = require_matrix(weights, prefix + "linear_attn.in_a.weight",
                                      kQ38GdnValueHeads, kQ38HiddenWidth);
        result.gdn.norm = require_bf16(
            weights, prefix + "linear_attn.norm.weight", kQ38GdnHeadWidth);
        result.gdn.output = require_matrix(
            weights, prefix + "linear_attn.out.weight", kQ38HiddenWidth,
            kQ38GdnValueWidth);
    }
    if (result.has_ple) {
        result.ple.key = require_matrix(weights, prefix + "ple.key.weight",
                                        kQ38HyperWidth, kQ38PleEmbeddingWidth);
        result.ple.value = require_matrix(weights, prefix + "ple.value.weight",
                                          kQ38HiddenWidth,
                                          kQ38PleEmbeddingWidth);
        result.ple.key_norm = require_bf16(
            weights, prefix + "ple.key_norm.weight", kQ38HyperWidth);
        result.ple.query_norm = require_bf16(
            weights, prefix + "ple.query_norm.weight", kQ38HyperWidth);
        result.ple.conv_norm = require_bf16(
            weights, prefix + "ple.conv_norm.weight", kQ38HyperWidth);
        result.ple.conv = require_bf16(
            weights, prefix + "ple.conv.weight",
            static_cast<std::uint64_t>(kQ38HyperWidth) * 4);
    }
    result.moe = bind_moe(weights, prefix);
    return result;
}

LayerWeights bind_mtp_layer(const CudaDeviceWeightStore& weights) {
    LayerWeights result;
    result.global = kLayers;
    result.state_local = 0;
    result.qsa = true;
    const std::string prefix = "mtp.blk.0.";
    result.attention_hyper = bind_hyper(weights, prefix + "hc_attn", true);
    result.moe_hyper = bind_hyper(weights, prefix + "hc_ffn", true);
    result.sparse.index_qk = require_matrix(
        weights, prefix + "attn_index_qk.weight", 640, kQ38HiddenWidth);
    result.sparse.index_q_norm = require_bf16(
        weights, prefix + "attn_index_q_norm.weight", kQ38QsaIndexerWidth);
    result.sparse.index_k_norm = require_bf16(
        weights, prefix + "attn_index_k_norm.weight", kQ38QsaIndexerWidth);
    result.sparse.q_gate = require_matrix(
        weights, prefix + "attn_q.weight", 12288, kQ38HiddenWidth);
    result.sparse.q_norm = require_bf16(
        weights, prefix + "attn_q_norm.weight", kQ38QsaHeadWidth);
    result.sparse.k = require_matrix(weights, prefix + "attn_k.weight", 512,
                                     kQ38HiddenWidth);
    result.sparse.k_norm = require_bf16(
        weights, prefix + "attn_k_norm.weight", kQ38QsaHeadWidth);
    result.sparse.v = require_matrix(weights, prefix + "attn_v.weight", 512,
                                     kQ38HiddenWidth);
    result.sparse.output = require_matrix(
        weights, prefix + "attn_output.weight", kQ38HiddenWidth, 6144);
    result.moe = bind_moe(weights, prefix);
    return result;
}

struct MtpWeights {
    CudaMatrixViewV1 embedding;
    CudaMatrixViewV1 fc_embedding;
    CudaMatrixViewV1 fc_hidden;
    const std::uint16_t* embedding_norm = nullptr;
    const std::uint16_t* hidden_norm = nullptr;
    HyperWeights output_hyper;
    LayerWeights layer;
};

MtpWeights bind_mtp(const CudaDeviceWeightStore& weights) {
    MtpWeights result;
    result.embedding = require_matrix(weights, "token_embd.weight", kVocabulary,
                                      kQ38HiddenWidth);
    result.fc_embedding = require_matrix(
        weights, "mtp.fc_embedding.weight", kQ38HiddenWidth,
        kQ38HiddenWidth);
    result.fc_hidden = require_matrix(weights, "mtp.fc_hidden.weight",
                                      kQ38HiddenWidth, kQ38HiddenWidth);
    result.embedding_norm = require_bf16(
        weights, "mtp.fc_embedding_norm.weight", kQ38HiddenWidth);
    result.hidden_norm = require_bf16(weights, "mtp.fc_hidden_norm.weight",
                                      kQ38HyperWidth);
    result.output_hyper = bind_hyper(weights, "mtp.hc_input", false);
    result.layer = bind_mtp_layer(weights);
    return result;
}

// One persistent producer per stage0 backend.  It warms the non-semantic PLE
// page cache for a complete known-token transaction, allowing I/O for later
// device-workspace chunks to overlap the current GPU slab.

class PrefillMatrixCache {
public:
    PrefillMatrixCache(int value_device, std::uint64_t value_max_bytes)
        : device_(value_device), max_bytes_(value_max_bytes) {
        check(cudaSetDevice(device_), "cudaSetDevice(prefill cache)");
        check_cublas(cublasCreate(&handle_), "cublasCreate(prefill)");
        try {
            check_cublas(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH),
                         "cublasSetMathMode(prefill)");
            if (profile_requested("Q38_TP_PAIRED_GRAPH")) {
                blas_workspace_bytes_ = 32ull << 20u;
                check(cudaMalloc(&blas_workspace_, blas_workspace_bytes_),
                      "allocate persistent cuBLAS workspace");
                resident_bytes_ += blas_workspace_bytes_;
            }
        } catch (...) {
            if (blas_workspace_) (void)cudaFree(blas_workspace_);
            (void)cublasDestroy(handle_);
            throw;
        }
    }

    ~PrefillMatrixCache() {
        (void)cudaSetDevice(device_);
        // Workspace is declared after this cache and therefore drains its
        // stream first.  The synchronize is a defensive guard for constructor
        // failure and future alternate call sites.
        (void)cudaDeviceSynchronize();
        for (auto& item : matrices_) (void)cudaFree(item.second);
        if (small_scratch_) (void)cudaFree(small_scratch_);
        for (auto* scratch : retired_small_scratch_) (void)cudaFree(scratch);
        if (handle_) (void)cublasDestroy(handle_);
        if (blas_workspace_) (void)cudaFree(blas_workspace_);
    }

    PrefillMatrixCache(const PrefillMatrixCache&) = delete;
    PrefillMatrixCache& operator=(const PrefillMatrixCache&) = delete;

    std::uint64_t resident_bytes() const { return resident_bytes_; }
    std::uint64_t allocation_failures() const {
        return allocation_failures_;
    }

    void seal() { sealed_ = true; }

    bool sealed() const { return sealed_; }

    void warm(const CudaMatrixViewV1& matrix, cudaStream_t stream) {
        (void)require(matrix, stream);
    }

    void warm_small(const CudaMatrixViewV1& matrix, cudaStream_t stream) {
        if (require(matrix, stream)) (void)reserve_small_scratch(matrix, stream);
    }

    bool gemm(const CudaMatrixViewV1& matrix, const std::uint16_t* input,
              std::uint16_t* output, std::uint32_t batch,
              cudaStream_t stream, bool borrowed_small = false) {
        if (!input || !output || batch < 2 || !stream)
            throw std::invalid_argument("invalid prefill GEMM buffers");
        const auto* expanded = require(matrix, stream);
        if (!expanded) return false;
        if (batch < 8 && !borrowed_small) {
            if (!reserve_small_scratch(matrix, stream)) return false;
            cuda_padded_gemm_bf16(handle_, expanded, input, output, small_scratch_,
                small_scratch_ + static_cast<std::size_t>(8) * matrix.columns,
                matrix.rows, matrix.columns, batch, stream,
                blas_workspace_, blas_workspace_bytes_);
            return true;
        }
        check_cublas(cublasSetStream(handle_, stream),
                     "cublasSetStream(prefill)");
        // SetStream resets the handle workspace. Bind it again before every
        // GEMM so capture contains no cuBLAS temporary allocation nodes.
        if (blas_workspace_)
            check_cublas(cublasSetWorkspace(handle_, blas_workspace_, blas_workspace_bytes_),
                         "cublasSetWorkspace(prefill)");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Row-major output [batch, rows] is column-major [rows, batch].
        // Row-major W [rows, columns] is column-major [columns, rows], so
        // transpose that view and multiply by input's [columns, batch] view.
        const cublasStatus_t st = cublasGemmEx(
                handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(matrix.rows), static_cast<int>(borrowed_small ? 8 : batch),
                static_cast<int>(matrix.columns), &alpha, expanded,
                CUDA_R_16BF, static_cast<int>(matrix.columns), input,
                CUDA_R_16BF, static_cast<int>(matrix.columns), &beta, output,
                CUDA_R_16BF, static_cast<int>(matrix.rows),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr,
                "prefill GemmEx fail status=%d rows=%u cols=%u batch=%u "
                "expanded=%p input=%p output=%p\n",
                static_cast<int>(st), matrix.rows, matrix.columns, batch,
                (void*)expanded, (void*)input, (void*)output);
            std::fflush(stderr);
        }
        check_cublas(st, "cublasGemmEx(prefill)");
        return true;
    }

private:
    bool reserve_small_scratch(const CudaMatrixViewV1& matrix, cudaStream_t stream) {
        const auto bytes = (static_cast<std::uint64_t>(matrix.columns) + matrix.rows) * 8 * 2;
        if (bytes <= small_scratch_bytes_) return true;
        if (sealed_ || bytes > max_bytes_ - std::min(max_bytes_, resident_bytes_))
            return false;
        std::uint16_t* replacement = nullptr;
        const auto allocated = cudaMalloc(reinterpret_cast<void**>(&replacement), bytes);
        if (allocated == cudaErrorMemoryAllocation) {
            ++allocation_failures_;
            (void)cudaGetLastError();
            return false;
        }
        check(allocated, "allocate small GEMM scratch");
        const auto completed = cudaStreamSynchronize(stream);
        if (completed != cudaSuccess) {
            (void)cudaFree(replacement);
            check(completed, "drain small GEMM scratch before growth");
        }
        // A lazy MTP graph can be captured before target warmup grows this
        // scratch. Keep prior allocations alive for those captured pointers.
        if (small_scratch_) {
            try { retired_small_scratch_.push_back(small_scratch_); }
            catch (...) { (void)cudaFree(replacement); throw; }
        }
        resident_bytes_ += bytes;
        small_scratch_ = replacement;
        small_scratch_bytes_ = bytes;
        return true;
    }
    const std::uint16_t* require(const CudaMatrixViewV1& matrix,
                                 cudaStream_t stream) {
        if (matrix.format == DeviceWeightFormatV1::kPreserve &&
            !matrix.preserved_f32)
            return static_cast<const std::uint16_t*>(matrix.data);
        const auto found = matrices_.find(matrix.data);
        if (found != matrices_.end()) return found->second;
        if (sealed_ || disabled_.count(matrix.data) != 0) return nullptr;
        const auto elements =
            static_cast<std::uint64_t>(matrix.rows) * matrix.columns;
        if (elements > std::numeric_limits<std::size_t>::max() /
                           sizeof(std::uint16_t))
            throw std::overflow_error("prefill matrix cache size overflows");
        const auto bytes = elements * sizeof(std::uint16_t);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes),
              "cudaMemGetInfo(prefill matrix)");
        constexpr std::uint64_t kRequiredHeadroom = 4ull << 30u;
        if (bytes > max_bytes_ - std::min(max_bytes_, resident_bytes_) ||
            free_bytes <= kRequiredHeadroom ||
            bytes > free_bytes - kRequiredHeadroom) {
            disabled_.insert(matrix.data);
            return nullptr;
        }
        std::uint16_t* expanded = nullptr;
        const auto allocation =
            cudaMalloc(reinterpret_cast<void**>(&expanded), bytes);
        if (allocation == cudaErrorMemoryAllocation) {
            // The free-memory snapshot above is advisory: another CUDA
            // allocation may win the race, and the driver can reject a large
            // contiguous allocation even when aggregate free memory is high
            // enough.  Tensor-Core prefill is an optimization, so pin this
            // matrix to the compact GEMV path instead of failing the request.
            ++allocation_failures_;
            (void)cudaGetLastError();
            disabled_.insert(matrix.data);
            return nullptr;
        }
        if (allocation != cudaSuccess) ++allocation_failures_;
        check(allocation, "cudaMalloc(prefill matrix)");
        try {
            cuda_dequantize_matrix_bf16(
                matrix, expanded, reinterpret_cast<void*>(stream), device_);
            matrices_.emplace(matrix.data, expanded);
            resident_bytes_ += bytes;
            return expanded;
        } catch (...) {
            (void)cudaFree(expanded);
            throw;
        }
    }

    int device_;
    cublasHandle_t handle_ = nullptr;
    void* blas_workspace_ = nullptr;
    std::size_t blas_workspace_bytes_ = 0;
    bool sealed_ = false;
    std::unordered_map<const void*, std::uint16_t*> matrices_;
    std::unordered_set<const void*> disabled_;
    std::uint64_t resident_bytes_ = 0;
    std::uint64_t max_bytes_ = 0;
    std::uint64_t allocation_failures_ = 0;
    std::uint16_t* small_scratch_ = nullptr;
    std::vector<std::uint16_t*> retired_small_scratch_;
    std::uint64_t small_scratch_bytes_ = 0;
};

class Workspace {
public:
    Workspace(int value_device, Stage stage, std::uint32_t max_tokens,
              bool enable_mtp, bool enable_tp)
        : device(value_device), maximum_tokens(max_tokens) {
        check(cudaSetDevice(device), "cudaSetDevice(workspace)");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate(stage)");
        try {
            if (stage == Stage::kStage0 && profile_requested("Q38_CUDA_PLE_COPY_STREAM")) {
                check(cudaStreamCreateWithFlags(&ple_copy_stream, cudaStreamNonBlocking),
                      "cudaStreamCreate(PLE copy)");
                check(cudaEventCreateWithFlags(&ple_copy_ready, cudaEventDisableTiming),
                      "cudaEventCreate(PLE copy)");
                check(cudaEventCreateWithFlags(&ple_consumer_done, cudaEventDisableTiming),
                      "cudaEventCreate(PLE consumer)");
            }
            qsa_graph_position = allocate<std::uint32_t>(1);
            tokens = allocate<std::int32_t>(max_tokens);
            predictions = allocate<std::int32_t>(max_tokens);
            if (enable_mtp && profile_requested("Q38_CUDA_MTP_GRAPH")) {
                mtp_graph_hc = allocate<std::uint16_t>(kQ38HyperWidth);
                mtp_graph_token = allocate<std::int32_t>(1);
                mtp_graph_prediction = allocate<std::int32_t>(1);
            }
            boundary = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(max_tokens) * kQ38HyperWidth);
            hyper_a = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperWidth);
            hyper_b = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperWidth);
            hyper_norm = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperWidth);
            mix_weights = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperWidth);
            mixed = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HiddenWidth);
            low_rank = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperLowRank);
            injection = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperCount);
            block_output = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HiddenWidth);
            proj0 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * 12288);
            proj1 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperWidth);
            proj2 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HyperWidth);
            proj3 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38GdnValueWidth);
            proj4 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38GdnValueWidth);
            proj5 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HiddenWidth);
            proj6 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HiddenWidth);
            small0 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * 1280);
            small1 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * 1280);
            small2 = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * 1280);
            expert_ids = allocate<std::int32_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38MoeTopK);
            expert_weights = allocate<float>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38MoeTopK);
            const std::uint32_t legacy_moe_tokens =
                prefill_moe_mode() ==
                        Q38PrefillMoeModeV1::kLegacyAtomicDiagnostic
                    ? kPrefillTile
                    : 1;
            moe_gate_up = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(legacy_moe_tokens) *
                kQ38MoeTopK * 2 *
                kQ38MoeIntermediate);
            moe_activated = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(legacy_moe_tokens) *
                kQ38MoeTopK *
                kQ38MoeIntermediate);
            if (enable_tp) te_routed = allocate<float>(static_cast<std::uint64_t>(kPrefillTile) * kQ38HiddenWidth);
            moe_accumulation = allocate<float>(
                static_cast<std::uint64_t>(legacy_moe_tokens) *
                kQ38HiddenWidth);
            if (prefill_moe_mode() !=
                Q38PrefillMoeModeV1::kLegacyAtomicDiagnostic) {
                const auto route_plan_bytes =
                    q38_moe_route_plan_bytes_v1(kPrefillTile);
                moe_route_plan_allocation =
                    allocate<std::uint8_t>(route_plan_bytes);
                moe_route_plan = cuda_moe_route_plan_storage_v1(
                    moe_route_plan_allocation, route_plan_bytes,
                    kPrefillTile);
                moe_packed_hidden = allocate<std::uint16_t>(
                    static_cast<std::uint64_t>(kPrefillTile) *
                    kQ38MoeTopK * kQ38HiddenWidth);
                moe_weighted_mid = allocate<std::uint16_t>(
                    static_cast<std::uint64_t>(kPrefillTile) *
                    kQ38MoeTopK * kQ38MoeIntermediate);
                moe_route_output = allocate<float>(
                    static_cast<std::uint64_t>(kPrefillTile) *
                    kQ38MoeTopK * kQ38HiddenWidth);
                if (cuda_marlin_moe_enabled()) {
                    if (!enable_tp) throw std::runtime_error("Marlin candidate requires TP2");
                    const auto tasks=q38_moe_prefill_max_tasks_v1(kPrefillTile);
                    marlin.sorted=allocate<std::int32_t>(tasks*16);
                    marlin.experts=allocate<std::int32_t>(tasks*2);
                    marlin.padded=allocate<std::int32_t>(1);
                    marlin.locks=allocate<std::int32_t>(4096);
                    marlin.gate_up=allocate<std::uint16_t>(static_cast<std::uint64_t>(kPrefillTile)*kQ38MoeTopK*1280);
                    marlin.down=allocate<std::uint16_t>(static_cast<std::uint64_t>(kPrefillTile)*kQ38MoeTopK*2560);
                    marlin.tmp=allocate<float>(tasks*16*2560ull);
                    check(cudaMemsetAsync(marlin.locks,0,4096*sizeof(std::int32_t),stream),"initialize Marlin locks");
                }
            }
            selected_indices = allocate<std::int32_t>(
                static_cast<std::uint64_t>(kPrefillTile) *
                kQ38QsaMaximumSelected);
            block_scores = allocate<float>(
                static_cast<std::uint64_t>(kPrefillTile) *
                (kQ38ContextLimit / kQ38QsaBlockTokens));
            if (enable_tp && profile_requested("Q38_CUDA_GDN_PREFILL_NORMALIZED_QK"))
                gdn_normalized_qk = allocate<float>(2ull * kPrefillTile *
                    kQ38GdnKeyHeads * kQ38GdnHeadWidth);
            attention_scores = allocate<float>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38QsaHeads *
                kQ38QsaMaximumSelected);
            if (stage == Stage::kStage1)
                logits = allocate<std::uint16_t>(kVocabulary);
            if (stage == Stage::kStage1 && enable_tp &&
                (profile_requested("Q38_TP_HEAD_BATCH_REDUCE") || profile_requested("Q38_TP_VOCAB_ROWS"))) {
                head_batch_logits = allocate<std::uint16_t>(7ull * kVocabulary);
                head_local_ids=allocate<std::int32_t>(7);
                head_partial_values=allocate<float>(7*128);
                head_partial_ids=allocate<std::int32_t>(7*128);
                head_candidates=allocate<std::uint16_t>(7*4);
                head_candidates_full=allocate<std::uint16_t>(7*8);
            }
            if (stage == Stage::kStage1 && enable_mtp) {
                final_hc = allocate<std::uint16_t>(
                    static_cast<std::uint64_t>(max_tokens) * kQ38HyperWidth);
                mtp_pending_hc = allocate<std::uint16_t>(kQ38HyperWidth);
                mtp_candidate_hc = allocate<std::uint16_t>(kQ38HyperWidth);
            }
            if (stage == Stage::kStage0)
                ple_rows = allocate<std::uint8_t>(
                    static_cast<std::uint64_t>(max_tokens) *
                    kQ38PleRowsPerToken * kQ38PleRowWidth);
            if (stage == Stage::kStage0)
                ple_scales = allocate<std::uint16_t>(
                    static_cast<std::uint64_t>(max_tokens) *
                    kQ38PleRowsPerToken);
            check(cudaHostAlloc(reinterpret_cast<void**>(&host_predictions),
                                max_tokens * sizeof(std::int32_t),
                                cudaHostAllocPortable),
                  "cudaHostAlloc(predictions)");
            pinned_bytes +=
                static_cast<std::uint64_t>(max_tokens) * sizeof(std::int32_t);
            if (stage == Stage::kStage0)
                check(cudaHostAlloc(reinterpret_cast<void**>(&host_ple_rows),
                                    static_cast<std::uint64_t>(max_tokens) *
                                        kQ38PleRowsPerToken * kQ38PleRowWidth,
                                    cudaHostAllocPortable),
                      "cudaHostAlloc(PLE rows)");
            if (stage == Stage::kStage0)
                pinned_bytes += static_cast<std::uint64_t>(max_tokens) *
                                kQ38PleRowsPerToken * kQ38PleRowWidth;
            if (stage == Stage::kStage0)
                check(cudaHostAlloc(
                          reinterpret_cast<void**>(&host_ple_scales),
                          static_cast<std::uint64_t>(max_tokens) *
                              kQ38PleRowsPerToken * sizeof(std::uint16_t),
                          cudaHostAllocPortable),
                      "cudaHostAlloc(PLE scales)");
            if (stage == Stage::kStage0)
                pinned_bytes += static_cast<std::uint64_t>(max_tokens) *
                                kQ38PleRowsPerToken * sizeof(std::uint16_t);
            if (stage == Stage::kStage1 && enable_mtp)
                mtp_state_bytes = 2ull * kQ38HyperWidth *
                                  sizeof(std::uint16_t);
        } catch (...) {
            release();
            throw;
        }
    }

    ~Workspace() { release(); }

    template <typename T>
    T* allocate(std::uint64_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("CUDA workspace allocation overflows");
        T* result = nullptr;
        const auto bytes = count * sizeof(T);
        check(cudaMalloc(reinterpret_cast<void**>(&result), bytes),
              "cudaMalloc(workspace)");
        try {
            check(cudaMemsetAsync(result,0,bytes,stream),"initialize workspace allocation");
            allocations.push_back({result,bytes});
        } catch (...) { (void)cudaFree(result); throw; }
        device_bytes += bytes;
        return result;
    }

    bool owns_range(const void* pointer,std::uint64_t bytes) const {
        const auto p=reinterpret_cast<std::uintptr_t>(pointer);
        for(const auto& allocation:allocations){const auto base=reinterpret_cast<std::uintptr_t>(allocation.first);
            if(p>=base && p-base<=allocation.second && bytes<=allocation.second-(p-base))return true;
        }
        return false;
    }
    bool borrowed_gemm_ranges(const void* input,const void* output,unsigned columns,unsigned rows) const {
        const auto in_bytes=16ull*columns,out_bytes=16ull*rows;
        if(!owns_range(input,in_bytes)||!owns_range(output,out_bytes))return false;
        const auto a=reinterpret_cast<std::uintptr_t>(input),b=reinterpret_cast<std::uintptr_t>(output);
        return a<=b ? in_bytes<=b-a : out_bytes<=a-b;
    }
    void release() noexcept {
        (void)cudaSetDevice(device);
        if (stream) (void)cudaStreamSynchronize(stream);
        if (ple_copy_stream) (void)cudaStreamSynchronize(ple_copy_stream);
        if (ple_consumer_done) (void)cudaEventDestroy(ple_consumer_done);
        if (ple_copy_ready) (void)cudaEventDestroy(ple_copy_ready);
        if (ple_copy_stream) (void)cudaStreamDestroy(ple_copy_stream);
        ple_copy_ready = nullptr;
        ple_copy_stream = nullptr;
        if (host_predictions) (void)cudaFreeHost(host_predictions);
        if (host_ple_rows) (void)cudaFreeHost(host_ple_rows);
        if (host_ple_scales) (void)cudaFreeHost(host_ple_scales);
        for (const auto& allocation : allocations) (void)cudaFree(allocation.first);
        allocations.clear();
        if (stream) (void)cudaStreamDestroy(stream);
        stream = nullptr;
        host_predictions = nullptr;
        host_ple_rows = nullptr;
        host_ple_scales = nullptr;
        device_bytes = 0;
        pinned_bytes = 0;
        mtp_state_bytes = 0;
    }

    int device;
    std::uint32_t maximum_tokens;
    std::uint64_t device_bytes = 0;
    std::uint64_t pinned_bytes = 0;
    std::uint64_t mtp_state_bytes = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t ple_copy_stream = nullptr;
    cudaEvent_t ple_consumer_done = nullptr;
    bool ple_copy_pending = false;
    cudaEvent_t ple_copy_ready = nullptr;
    std::vector<std::pair<void*,std::uint64_t>> allocations;
    std::int32_t* tokens = nullptr;
    std::int32_t* predictions = nullptr;
    std::int32_t* host_predictions = nullptr;
    std::uint16_t* mtp_graph_hc = nullptr;
    std::uint16_t* mtp_normalized_embedding = nullptr;
    std::int32_t* mtp_graph_token = nullptr;
    std::int32_t* mtp_graph_prediction = nullptr;
    std::uint8_t* ple_rows = nullptr;
    std::uint8_t* host_ple_rows = nullptr;
    std::uint16_t* ple_scales = nullptr;
    std::uint16_t* host_ple_scales = nullptr;
    std::uint16_t* boundary = nullptr;
    std::uint32_t* qsa_graph_position = nullptr;
    std::uint16_t* hyper_a = nullptr;
    std::uint16_t* hyper_b = nullptr;
    std::uint16_t* hyper_norm = nullptr;
    std::uint16_t* mix_weights = nullptr;
    std::uint16_t* mixed = nullptr;
    std::uint16_t* low_rank = nullptr;
    std::uint16_t* injection = nullptr;
    std::uint16_t* block_output = nullptr;
    std::uint16_t* proj0 = nullptr;
    std::uint16_t* proj1 = nullptr;
    std::uint16_t* proj2 = nullptr;
    std::uint16_t* proj3 = nullptr;
    std::uint16_t* proj4 = nullptr;
    std::uint16_t* proj5 = nullptr;
    std::uint16_t* proj6 = nullptr;
    std::uint16_t* small0 = nullptr;
    std::uint16_t* small1 = nullptr;
    std::uint16_t* small2 = nullptr;
    std::int32_t* expert_ids = nullptr;
    float* expert_weights = nullptr;
    CudaMarlinWorkspace marlin;
    std::uint16_t* moe_gate_up = nullptr;
    std::uint16_t* moe_activated = nullptr;
    float* moe_accumulation = nullptr;
    float* te_routed = nullptr;
    std::uint8_t* moe_route_plan_allocation = nullptr;
    Q38RoutePlanStorageV1 moe_route_plan;
    std::uint16_t* moe_packed_hidden = nullptr;
    std::uint16_t* moe_weighted_mid = nullptr;
    float* moe_route_output = nullptr;
    std::int32_t* selected_indices = nullptr;
    float* block_scores = nullptr;
    float* gdn_normalized_qk = nullptr;
    float* attention_scores = nullptr;
    std::uint16_t* logits = nullptr;
    std::uint16_t* head_batch_logits = nullptr;
    std::int32_t* head_local_ids=nullptr;
    float* head_partial_values=nullptr;
    std::int32_t* head_partial_ids=nullptr;
    std::uint16_t *head_candidates=nullptr,*head_candidates_full=nullptr;
    std::uint16_t* final_hc = nullptr;
    std::uint16_t* mtp_pending_hc = nullptr;
    std::uint16_t* mtp_candidate_hc = nullptr;
};

}  // namespace

// Fixed six rejected-suffix prefixes cover the qualified MTP widths 1/2/4/6.
// Allocation persists across transactions; each kernel writes its layer slice.
struct TargetPrefixStorage {
    int device;
    std::uint8_t* data = nullptr;
    std::uint64_t gdn_bytes, ple_bytes, stride;
    static constexpr std::uint32_t capacity = 6;
    TargetPrefixStorage(int d, std::uint64_t gdn, std::uint64_t ple)
        : device(d), gdn_bytes(gdn), ple_bytes(ple), stride(gdn + ple) {
        check(cudaSetDevice(device), "set prefix storage device");
        check(cudaMalloc(reinterpret_cast<void**>(&data), stride * capacity),
              "allocate target prefix storage");
        check(cudaMemset(data,0,stride*capacity),"initialize prefix storage");
    }
    ~TargetPrefixStorage() {
        (void)cudaSetDevice(device);
        if (data) (void)cudaFree(data);
    }
    CudaPrefixSnapshotView view(std::uint64_t offset, std::uint32_t count) {
        if (count == 0 || count > capacity || offset >= stride)
            throw std::logic_error("invalid target prefix storage view");
        return {data + offset, stride, count};
    }
};

struct CudaStageBackend::Impl {
    enum class DecodeStaticActionKind : std::uint8_t {
        kInput,
        kPle,
        kPrepareAttention,
        kGdn,
        kFinish,
        kPrepareMoe,
        kMoe,
        kOutput,
        kQsa,
        kMoeFinish,
    };

    struct DecodeStaticAction {
        DecodeStaticActionKind kind = DecodeStaticActionKind::kInput;
        std::uint32_t layer = 0;
        std::uint16_t* first = nullptr;
        std::uint16_t* second = nullptr;
    };

    struct DecodeGraphVariant {
        std::uint16_t* output_hc = nullptr;
        std::uintptr_t gdn_working = 0;
        std::uintptr_t ple_working = 0;
        std::uint32_t tokens = 1;
        bool sparse_qsa = false;
        std::uintptr_t prefix_storage = 0;
        std::vector<std::unique_ptr<CapturedCudaGraph>> fragments;
        std::vector<std::uint32_t> qsa_layers;
    };

    Impl(DeviceStageIndexV1 source_index, CudaStageBackendOptions value_options,
         std::shared_ptr<PleStore> value_ple)
        : options(value_options),
          weights(make_weights_for_device(std::move(source_index),
                                          value_options.device,
                                          value_options.enable_mtp)),
          prefill_cache(value_options.device,
                        value_options.prefill_matrix_cache_bytes),
          workspace(options.device, options.stage,
                    options.max_transaction_tokens, options.enable_mtp, options.te_world_size == 2),
          ple_store(std::move(value_ple)) {
        if (options.device < 0 || options.max_transaction_tokens == 0 ||
            options.context_capacity == 0 ||
            options.context_capacity > kQ38ContextLimit ||
            weights.stage() != static_cast<std::uint32_t>(options.stage))
            throw std::invalid_argument("invalid CUDA stage backend options");
        const bool marked_gdn = !weights.find("q38.tp.gdn_heads.stage" + std::to_string(unsigned(options.stage)) + ".rank0").empty() ||
                                !weights.find("q38.tp.gdn_heads.stage" + std::to_string(unsigned(options.stage)) + ".rank1").empty();
        const bool local_gdn = profile_requested("Q38_TP_GDN_LOCAL");
        if(marked_gdn != local_gdn || (local_gdn &&
            (options.te_world_size!=2 || weights.find("q38.tp.gdn_heads.stage" + std::to_string(unsigned(options.stage)) + ".rank" +
                 std::to_string(options.te_rank)).empty())))
            throw std::invalid_argument("GDN head ownership does not match execution policy/rank");
        const std::string packed_marker="q38.tp.gdn_packed.stage"+std::to_string(unsigned(options.stage))+".rank";
        const bool marked_packed=!weights.find(packed_marker+"0").empty() || !weights.find(packed_marker+"1").empty();
        if(marked_packed!=profile_requested("Q38_TP_GDN_PACKED") || (marked_packed && (!local_gdn || weights.find(packed_marker+std::to_string(options.te_rank)).empty())))
            throw std::invalid_argument("packed GDN artifact does not match execution policy/rank");
        const auto first = options.stage == Stage::kStage0 ? 0u : kCut;
        const auto end = options.stage == Stage::kStage0 ? kCut : kLayers;
        std::uint32_t gdn_local = 0;
        std::uint32_t qsa_local = 0;
        g_te_bind_world_size = options.te_world_size > 0 ? options.te_world_size : 1;
        for (auto layer = first; layer < end; ++layer) {
            const bool qsa = layer % 4 == 3;
            layers.push_back(bind_layer(weights, layer,
                                        qsa ? qsa_local++ : gdn_local++));
        }
        if (profile_requested("Q38_TP_QSA_LOCAL")) {
            if (options.te_world_size != 2 || options.te_rank < 0 || options.te_rank > 1)
                throw std::invalid_argument("QSA local heads require TP2");
            for (auto& layer : layers) {
                if (!layer.qsa) continue;
                auto view = layer.sparse.q_gate;
                if (view.rows != 12288 || view.columns != 2560 || layer.sparse.output.columns != 3072)
                    throw std::invalid_argument("QSA local projection layout mismatch");
                const unsigned bytes_per_weight = view.format == DeviceWeightFormatV1::kPreserve
                    ? (view.preserved_f32 ? 4u : 2u)
                    : view.format == DeviceWeightFormatV1::kW8A16SymG128 ? 1u : 0u;
                if (!bytes_per_weight) throw std::invalid_argument("unsupported QSA projection row view");
                const std::size_t first_row = static_cast<std::size_t>(options.te_rank) * 6144;
                view.data = static_cast<const unsigned char*>(view.data) + first_row * view.columns * bytes_per_weight;
                if (view.scales) view.scales = static_cast<const std::uint16_t*>(view.scales) + first_row * (view.columns / view.group_size);
                view.rows = 6144;
                layer.sparse.q_gate_local = view;
            }
        }
        // te bind stays active through embedding/output/mtp binds

        const bool retained_prefix_checkpoint =
            options.enable_speculative_checkpoint &&
            options.stage == Stage::kStage0;
        gdn_state = std::make_unique<CudaGdnStateBank>(
            options.device, gdn_local, retained_prefix_checkpoint);
        qsa_state = std::make_unique<CudaQsaStateBank>(
            options.device, qsa_local, options.context_capacity);
        if (options.stage == Stage::kStage0) {
            if (!ple_store || ple_store->layout().storage_dtype != DType::kFp8E4M3 ||
                ple_store->layout().row_dimension != kQ38PleRowWidth)
                throw std::invalid_argument("stage0 requires FP8 PLE layout");
            ple_state = std::make_unique<CudaPleStateBank>(
                options.device, retained_prefix_checkpoint);
            committed_ple_hash =
                std::make_unique<PleHashState>(ple_store->layout().hash);
            ple_reader = std::make_unique<PleAsyncReader>(
                ple_store, static_cast<std::size_t>(ple_io_tile_tokens()) * kQ38PleRowsPerToken);
            embedding = require_matrix(weights, "token_embd.weight", kVocabulary,
                                       kQ38HiddenWidth);
            small_boundary_ring = std::make_unique<CudaBoundaryRing>(
                CudaBoundaryRingOptions{options.device, 3,
                                        kSmallBoundaryTokens,
                                        kQ38HyperWidth,
                                        profile_requested("Q38_CUDA_ASYNC_BOUNDARY"),
                                        options.te_world_size == 2 && profile_requested("Q38_TP_DEVICE_BOUNDARY")});
            prefill_boundary_ring = std::make_unique<CudaBoundaryRing>(
                CudaBoundaryRingOptions{options.device, 3,
                                        options.max_transaction_tokens,
                                        kQ38HyperWidth,
                                        profile_requested("Q38_CUDA_ASYNC_BOUNDARY"),
                                        options.te_world_size == 2 && profile_requested("Q38_TP_DEVICE_BOUNDARY")});
        } else {
            if (options.enable_compact_sampling)
                compact_sampling = std::make_unique<CudaSamplingWorkspace>(
                    options.device, kVocabulary);
            output = require_matrix(weights, "output.weight", kVocabulary,
                                    kQ38HiddenWidth);
            if ((output.rows*2u==kVocabulary)!=profile_requested("Q38_TP_VOCAB_ROWS"))throw std::runtime_error("LM head layout policy mismatch");
            final_hyper = bind_hyper(weights, "hc_input", false);
            if (options.enable_mtp) {
                mtp = std::make_unique<MtpWeights>(bind_mtp(weights));
                mtp_qsa_state = std::make_unique<CudaQsaStateBank>(
                    options.device, 1, options.context_capacity);
                if (profile_requested("Q38_CUDA_MTP_INPUT_CACHE")) {
                    const auto count=std::uint64_t(mtp->embedding.rows)*mtp->embedding.columns;
                    workspace.mtp_normalized_embedding=workspace.allocate<std::uint16_t>(count);
                    cuda_dequantize_matrix_bf16(mtp->embedding,workspace.mtp_normalized_embedding,
                        workspace.stream,options.device);
                    cuda_qwen38_rmsnorm_bf16(workspace.mtp_normalized_embedding,mtp->embedding_norm,false,
                        workspace.mtp_normalized_embedding,mtp->embedding.rows,mtp->embedding.columns,
                        1e-6f,true,workspace.stream,options.device);
                    check(cudaStreamSynchronize(workspace.stream),"MTP normalized embedding cache ready");
                }
            }
        }
        // Validate scheduling knobs before the first request or reader thread
        // uses them. Cache prewarming is optional and bounded by the same cap.
        (void)prefill_tile_tokens();
        (void)ple_io_tile_tokens();
        (void)prefill_sync_tiles();
        if (profile_requested("Q38_CUDA_PREFILL_PREWARM")) {
            auto warm = [&](const CudaMatrixViewV1& matrix) {
                if (matrix.data) prefill_cache.warm(matrix, workspace.stream);
            };
            auto hyper = [&](const HyperWeights& weights) {
                warm(weights.mix_down); warm(weights.mix_up); warm(weights.inject);
            };
            auto layer = [&](const LayerWeights& weights) {
                hyper(weights.attention_hyper); hyper(weights.moe_hyper);
                warm(weights.gdn.packed); warm(weights.gdn.qkv);
                    if(weights.gdn.qkv_prefill_full.data)warm(weights.gdn.qkv_prefill_full); warm(weights.gdn.z);
                warm(weights.gdn.a); warm(weights.gdn.b); warm(weights.gdn.output);
                warm(weights.sparse.q_gate);
                if(profile_requested("Q38_TP_QSA_LOCAL_PROJECTION"))warm(weights.sparse.q_gate_local); warm(weights.sparse.k);
                warm(weights.sparse.v); warm(weights.sparse.index_qk);
                warm(weights.sparse.output);
                warm(weights.moe.router); warm(weights.moe.shared_gate);
                warm(weights.moe.shared_up); warm(weights.moe.shared_down);
                warm(weights.moe.shared_output_gate);
                warm(weights.ple.key); warm(weights.ple.value);
            };
            for (const auto& weights : layers) layer(weights);
            if (mtp) {
                layer(mtp->layer); hyper(mtp->output_hyper);
                warm(mtp->fc_hidden); warm(mtp->fc_embedding);
            }
            check(cudaStreamSynchronize(workspace.stream),
                  "cudaStreamSynchronize(prefill prewarm)");
            prefill_cache.seal();
        }
        g_te_bind_world_size = 1;
        if (profile_requested("Q38_CUDA_PROFILE_DECODE"))
            decode_profile = std::make_unique<CudaEventProfile>(
                options.device, "decode");
        if (profile_requested("Q38_CUDA_PROFILE_PREFILL") ||
            profile_requested("Q38_CUDA_PROFILE_GRAPHS"))
            prefill_profile = std::make_unique<CudaEventProfile>(
                options.device, "prefill", profile_requested("Q38_CUDA_PROFILE_PREFILL_TIMELINE"));
    }

    void profile_mark(const char* label) {
        if (capturing_profile_events) return;
        if (decode_profile && decode_profile->active())
            decode_profile->mark(label, workspace.stream);
        if (prefill_profile && prefill_profile->active())
            prefill_profile->mark(label, workspace.stream);
    }

    bool detailed_prefill_profile() const {
        return prefill_profile && prefill_profile->active() &&
               profile_requested("Q38_CUDA_PROFILE_PREFILL_DETAIL");
    }

    void cuda_gemv_bf16(const CudaMatrixViewV1& matrix,
                         const std::uint16_t* input,
                         std::uint16_t* output,
                         std::uint32_t batch,
                         void* stream,
                         int device,
                         float alpha = 1.0f,
                         bool accumulate = false) {
        // BF16 tensor-op GemmEx wants N multiple of 8; tiny prefill batches
        // (e.g. TE smoke with 2 tokens) must use the regular GEMV path.
        const bool small_gemm = options.te_world_size == 2 && batch >= 2 &&
            profile_requested("Q38_TP_SMALL_GEMM");
        if ((batch >= 8 || small_gemm) && alpha == 1.0f && !accumulate &&
            device == options.device &&
            stream == reinterpret_cast<void*>(workspace.stream)) {
            const bool borrowed=batch<8 && profile_requested("Q38_CUDA_SMALL_GEMM_BORROWED") &&
                workspace.borrowed_gemm_ranges(input,output,matrix.columns,matrix.rows);
            if (prefill_cache.gemm(matrix, input, output, batch,
                                   workspace.stream,borrowed))
                return;
        }
        q38::cuda_gemv_bf16(matrix, input, output, batch, stream, device,
                            alpha, accumulate);
    }

    void project(const std::uint16_t* input, std::uint32_t tokens,
                 std::initializer_list<std::pair<CudaMatrixViewV1,
                                                std::uint16_t*>> projections) {
        if (tokens == 1 && profile_requested("Q38_CUDA_FUSED_PROJECTIONS")) {
            CudaProjectionPackV1 pack;
            for (const auto& item : projections) {
                if (pack.count == 4)
                    throw std::logic_error("projection pack exceeds four");
                pack.matrices[pack.count] = item.first;
                pack.outputs[pack.count++] = item.second;
            }
            cuda_multi_projection_decode_bf16(pack, input, workspace.stream,
                                               options.device);
            return;
        }
        for (const auto& item : projections)
            cuda_gemv_bf16(item.first, input, item.second, tokens,
                          workspace.stream, options.device);
    }

    void project_hc(const CudaMatrixViewV1& matrix,
                    const std::uint16_t* input, std::uint16_t* output,
                    std::uint32_t tokens, CudaHcActivation activation,
                    float divisor, float output_scale) {
        if(profile_requested("Q38_CUDA_HC_INJECT_BATCH") &&
            cuda_try_hc_inject_batch_bf16(matrix,input,output,tokens,workspace.stream,
                options.device,activation,divisor,output_scale))return;
        static const bool enabled = profile_requested("Q38_CUDA_HC_GEMV");
        static const bool fused = profile_requested("Q38_CUDA_HC_FUSED");
        const bool specialized = (enabled || fused) && tokens == 1 &&
            cuda_try_hc_gemv_bf16(matrix, input, output,
                workspace.stream, options.device,
                fused ? activation : CudaHcActivation::kNone,
                divisor, output_scale);
        if (specialized && fused) return;
        if (!specialized)
            cuda_gemv_bf16(matrix, input, output, tokens,
                           workspace.stream, options.device);
        if (activation == CudaHcActivation::kSilu)
            cuda_hyper_silu_scaled_bf16(output, output, tokens, matrix.rows,
                divisor, workspace.stream, options.device);
        else if (activation == CudaHcActivation::kSigmoid)
            cuda_hyper_sigmoid_bf16(output, output, tokens, matrix.rows,
                divisor, output_scale, workspace.stream, options.device);
    }

    bool fuse_hyper_finish() const {
        return options.te_world_size==2 && profile_requested("Q38_CUDA_HC_FINISH_FUSED");
    }

    void prepare_hyper(const HyperWeights& hyper, const std::uint16_t* input,
                       std::uint32_t tokens = 1) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        cuda_hyper_group_rmsnorm_bf16(input, hyper.norm, workspace.hyper_norm,
                                      tokens, stream, options.device);
        static const bool grouped_enabled = profile_requested("Q38_CUDA_HC_GROUPED");
        // Grouped HC assumes full mix_down/inject rows; skip under TE.
        const bool grouped = grouped_enabled && tokens == 1 && hyper.has_inject &&
            options.te_world_size != 2 &&
            cuda_try_hc_grouped_bf16(
                {hyper.mix_down, workspace.low_rank, CudaHcActivation::kSilu,
                 static_cast<float>(kQ38HyperCount), 1.0f},
                {hyper.inject, workspace.injection, CudaHcActivation::kSigmoid,
                 static_cast<float>(kQ38HyperCount), 2.0f},
                workspace.hyper_norm, stream, options.device);
        if (!grouped)
            project_hc(hyper.mix_down, workspace.hyper_norm, workspace.low_rank,
                       tokens, CudaHcActivation::kSilu, kQ38HyperCount, 1.0f);
        if (options.te_world_size == 2 &&
            hyper.mix_down.rows * 2u == kQ38HyperLowRank)
            te_bridge_allgather_bf16(options.te_rank, workspace.low_rank,
                                     workspace.low_rank, hyper.mix_down.rows,
                                     tokens, te_bridge_group_for(options), workspace.stream);
        const bool fused_mix=profile_requested("Q38_CUDA_HC_MIX_TENSOR") &&
            cuda_try_hc_mix_tensor_bf16(hyper.mix_up,workspace.low_rank,workspace.hyper_norm,
                workspace.mixed,tokens,workspace.stream,options.device);
        if(!fused_mix){
            project_hc(hyper.mix_up, workspace.low_rank, workspace.mix_weights,
                       tokens, CudaHcActivation::kSigmoid, 1.0f, 1.0f);
            if (options.te_world_size == 2 && hyper.mix_up.rows * 2u == kQ38HyperWidth)
                te_bridge_allgather_bf16(options.te_rank,workspace.mix_weights,workspace.mix_weights,hyper.mix_up.rows,tokens,te_bridge_group_for(options),workspace.stream);
            cuda_hyper_mix_bf16(workspace.hyper_norm, workspace.mix_weights,
                                workspace.mixed,tokens,stream,options.device);
        }
        if (hyper.has_inject && !grouped) {
            project_hc(hyper.inject, workspace.hyper_norm, workspace.injection,
                       tokens, fuse_hyper_finish()?CudaHcActivation::kNone:CudaHcActivation::kSigmoid, kQ38HyperCount, 2.0f);
            if (options.te_world_size == 2 &&
                hyper.inject.rows * 2u == kQ38HyperCount)
                te_bridge_allgather_bf16(options.te_rank, workspace.injection,
                                         workspace.injection, hyper.inject.rows,
                                         tokens, te_bridge_group_for(options), workspace.stream);
        }
    }

    std::uint16_t* finish_hyper(const std::uint16_t* original,
                                std::uint16_t* destination,
                                std::uint32_t tokens = 1) {
        if(fuse_hyper_finish())
            cuda_hyper_inject_from_projection_bf16(original,workspace.block_output,
                workspace.injection,destination,tokens,workspace.stream,options.device);
        else cuda_hyper_inject_bf16(
            original, workspace.block_output, workspace.injection, destination,
            tokens, reinterpret_cast<void*>(workspace.stream), options.device);
        return destination;
    }

    bool fuse_moe_finish(unsigned tokens) const {
        return tokens > 1 && tokens <= 7 && fuse_hyper_finish() &&
            profile_requested("Q38_CUDA_MOE_HC_FINISH_FUSED");
    }
    void finish_moe_hyper(const std::uint16_t* original,
                          std::uint16_t* destination, unsigned tokens = 1) {
        if (!fuse_moe_finish(tokens)) { finish_hyper(original,destination,tokens); return; }
        cuda_hyper_moe_inject_bf16(original,workspace.proj6,workspace.proj5,
            workspace.small1,workspace.injection,destination,tokens,workspace.stream,options.device);
    }

    void project_ple(const LayerWeights& layer, const std::uint16_t* input,
                     std::uint16_t* key, std::uint16_t* value,
                     std::uint32_t tokens) {
        static const bool enabled = profile_requested("Q38_CUDA_PLE_FUSED");
        if (enabled && tokens == 1) {
            CudaProjectionPackV1 pack;
            pack.count = 2;
            pack.matrices[0] = layer.ple.key;
            pack.matrices[1] = layer.ple.value;
            pack.outputs[0] = key;
            pack.outputs[1] = value;
            cuda_multi_projection_decode_bf16(pack, input,
                                               workspace.stream, options.device);
            return;
        }
        cuda_gemv_bf16(layer.ple.key, input, key, tokens,
                       workspace.stream, options.device);
        cuda_gemv_bf16(layer.ple.value, input, value, tokens,
                       workspace.stream, options.device);
    }

    void normalize_ple(const LayerWeights& layer, const std::uint16_t* key,
                       const std::uint16_t* query, std::uint16_t* normalized_key,
                       std::uint16_t* normalized_query, std::uint32_t tokens) {
        static const bool paired = profile_requested("Q38_CUDA_PLE_NORM_PAIR");
        if (paired) {
            cuda_hyper_pair_group_rmsnorm_bf16(
                {key, layer.ple.key_norm, normalized_key},
                {query, layer.ple.query_norm, normalized_query},
                tokens, workspace.stream, options.device);
        } else {
            cuda_hyper_group_rmsnorm_bf16(key, layer.ple.key_norm,
                normalized_key, tokens, workspace.stream, options.device);
            cuda_hyper_group_rmsnorm_bf16(query, layer.ple.query_norm,
                normalized_query, tokens, workspace.stream, options.device);
        }
    }

    bool ple_residual_fused() const {
        static const bool enabled = profile_requested("Q38_CUDA_PLE_RESIDUAL");
        return enabled;
    }

    void run_ple(const LayerWeights& layer, std::uint16_t* hyper,
                 std::uint16_t* destination, std::uint32_t token_index) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const auto* rows = workspace.ple_rows +
                           static_cast<std::uint64_t>(token_index) *
                               kQ38PleRowsPerToken * kQ38PleRowWidth;
        const auto* scales = workspace.ple_scales +
                             static_cast<std::uint64_t>(token_index) *
                                 kQ38PleRowsPerToken;
        cuda_ple_fp8_rows_to_bf16(rows, scales, workspace.proj5, 1, stream,
                                  options.device);
        project_ple(layer, workspace.proj5, workspace.proj1, workspace.proj6, 1);
        normalize_ple(layer, workspace.proj1, hyper, workspace.hyper_norm,
                      workspace.proj2, 1);
        cuda_ple_gate_bf16(workspace.hyper_norm, workspace.proj2,
                           workspace.proj6, workspace.mix_weights, stream,
                           options.device);
        cuda_hyper_group_rmsnorm_bf16(
            workspace.mix_weights, layer.ple.conv_norm, workspace.proj1, 1,
            stream, options.device);
        if (ple_residual_fused()) {
            cuda_ple_conv_residual_bf16(workspace.mix_weights, workspace.proj1,
                layer.ple.conv, ple_state->working(), hyper, destination,
                1, stream, options.device);
        } else {
            cuda_ple_conv_decode_bf16(
                workspace.mix_weights, workspace.proj1, layer.ple.conv,
                ple_state->working(), workspace.proj2, stream, options.device);
            cuda_add_bf16(hyper, workspace.proj2, destination, kQ38HyperWidth,
                          stream, options.device);
        }
    }

    CudaPrefixSnapshotView gdn_prefix_view(const LayerWeights& layer, bool recurrent) {
        if (!capture_target_prefixes) return {};
        const auto layer_bytes = gdn_state->bytes_per_bank() / gdn_state->layers();
        const auto conv_bytes = std::uint64_t(kQ38GdnQkvWidth) * kQ38GdnConvWidth * 2;
        return target_prefixes->view(layer.state_local * layer_bytes +
            (recurrent ? conv_bytes : 0), target_prefix_count);
    }
    CudaPrefixSnapshotView ple_prefix_view() {
        return capture_target_prefixes
            ? target_prefixes->view(target_prefixes->gdn_bytes, target_prefix_count)
            : CudaPrefixSnapshotView{};
    }

    void run_ple_prefill(const LayerWeights& layer,
                         const std::uint16_t* hyper,
                         std::uint16_t* destination,
                         std::uint32_t token_offset,
                         std::uint32_t tokens) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const auto* rows = workspace.ple_rows +
                           static_cast<std::uint64_t>(token_offset) *
                               kQ38PleRowsPerToken * kQ38PleRowWidth;
        const auto* scales = workspace.ple_scales +
                             static_cast<std::uint64_t>(token_offset) *
                                 kQ38PleRowsPerToken;
        cuda_ple_fp8_rows_to_bf16(rows, scales, workspace.proj5, tokens,
                                  stream, options.device);
        if (detailed_prefill_profile()) profile_mark("ple_unpack");
        project_ple(layer, workspace.proj5, workspace.hyper_norm,
                    workspace.proj6, tokens);
        if (detailed_prefill_profile()) profile_mark("ple_key_value");
        normalize_ple(layer, workspace.hyper_norm, hyper, workspace.proj1,
                      workspace.proj2, tokens);
        if (detailed_prefill_profile()) profile_mark("ple_norms");
        cuda_ple_gate_prefill_bf16(
            workspace.proj1, workspace.proj2, workspace.proj6,
            workspace.mix_weights, tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("ple_gate");
        cuda_hyper_group_rmsnorm_bf16(
            workspace.mix_weights, layer.ple.conv_norm, workspace.hyper_norm,
            tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("ple_conv_norm");
        if (ple_residual_fused()) {
            cuda_ple_conv_residual_bf16(workspace.mix_weights, workspace.hyper_norm,
                layer.ple.conv, ple_state->working(), hyper, destination,
                tokens, stream, options.device, ple_prefix_view());
            if (detailed_prefill_profile()) profile_mark("ple_conv_residual");
        } else {
            cuda_ple_conv_prefill_bf16(
                workspace.mix_weights, workspace.hyper_norm, layer.ple.conv,
                ple_state->working(), workspace.proj2, tokens, stream,
                options.device, ple_prefix_view());
            if (detailed_prefill_profile()) profile_mark("ple_conv");
            cuda_add_bf16(hyper, workspace.proj2, destination,
                          static_cast<std::size_t>(tokens) * kQ38HyperWidth,
                          stream, options.device);
            if (detailed_prefill_profile()) profile_mark("ple_residual");
        }
    }

    bool run_gdn_local(const LayerWeights& layer, unsigned tokens, bool decode) {
        if(!profile_requested("Q38_TP_GDN_LOCAL")) return false;
        const bool packed=profile_requested("Q38_TP_GDN_PACKED");
        if(options.te_world_size!=2 || layer.gdn.qkv.rows!=(packed?10240:5120) ||
           layer.gdn.z.rows!=6144 || layer.gdn.b.rows!=48 || layer.gdn.a.rows!=48 ||
           layer.gdn.output.columns!=3072 || (!decode && !precomputed_gdn_prefill_enabled()) ||
           (packed && layer.gdn.packed.rows!=8192))
            throw std::logic_error("head-local GDN profile/artifact mismatch");
        auto stream=reinterpret_cast<void*>(workspace.stream);
        const bool bulk=tokens>=8;
        // Changing the GEMM shape can change BF16 rounding on a short final
        // tile. Qualify the 512-token shape separately and keep the tail path.
        const bool packed_bulk = tokens == 512 &&
            profile_requested("Q38_CUDA_GDN_PREFILL_PACKED_QKVZ");
        const bool fused = packed && (!bulk || packed_bulk);
        const auto qkv=packed?layer.gdn.qkv:(bulk?layer.gdn.qkv_prefill_full:layer.gdn.qkv);
        if (bulk && !fused && !qkv.data)
            throw std::logic_error("missing exact bulk QKV projection");
        if(fused){
            cuda_gemv_bf16(layer.gdn.packed,workspace.mixed,workspace.proj1,tokens,stream,options.device);
            project(workspace.mixed,tokens,{{layer.gdn.b,workspace.small0},{layer.gdn.a,workspace.small1}});
        } else project(workspace.mixed,tokens,{{qkv,workspace.proj1},{layer.gdn.z,workspace.proj3},{layer.gdn.b,workspace.small0},{layer.gdn.a,workspace.small1}});
        if (detailed_prefill_profile()) profile_mark("gdn_local_projection");
        const auto state=gdn_state->working(layer.state_local);
        CudaGdnHeadLocalProjectionLayout projection_layout;
        projection_layout.full_qkv = bulk && !fused;
        projection_layout.qkv_stride = fused ? 8192 : qkv.rows;
        projection_layout.control_heads = 48;
        projection_layout.control_stride = 48;
        projection_layout.z_heads = fused ? 24 : 48;
        projection_layout.z_head_begin = fused ? 0 : options.te_rank * 24;
        projection_layout.z_stride = fused ? 8192 : 6144;
        cuda_gdn_head_local_bf16(workspace.proj1,workspace.small0,
            workspace.small1,fused?workspace.proj1+5120:workspace.proj3,
            layer.gdn.conv,layer.gdn.a_log,layer.gdn.dt_bias,layer.gdn.norm,
            state,workspace.proj2,workspace.proj4,workspace.proj0,workspace.block_scores,
            tokens,options.te_rank,decode,stream,options.device,
            decode?CudaPrefixSnapshotView{}:gdn_prefix_view(layer,false),
            decode?CudaPrefixSnapshotView{}:gdn_prefix_view(layer,true),projection_layout,
            workspace.gdn_normalized_qk);
        if (detailed_prefill_profile()) profile_mark("gdn_local_recurrent_norm");
        cuda_gemv_bf16(layer.gdn.output,workspace.proj0,workspace.block_output,tokens,stream,options.device);
        if (detailed_prefill_profile()) profile_mark("gdn_local_output");
        te_bridge_allreduce_sum_bf16(options.te_rank,workspace.block_output,
            static_cast<std::size_t>(tokens)*kQ38HiddenWidth,te_bridge_group_for(options),workspace.stream);
        if (detailed_prefill_profile()) profile_mark("gdn_local_collective");
        return true;
    }

    void run_gdn(const LayerWeights& layer) {
        if(run_gdn_local(layer,1,true)) return;
        auto stream = reinterpret_cast<void*>(workspace.stream);
        project(workspace.mixed, 1, {
            {layer.gdn.qkv, workspace.proj1},
            {layer.gdn.z, workspace.proj3},
            {layer.gdn.b, workspace.small0},
            {layer.gdn.a, workspace.small1}});
        if (options.te_world_size == 2) {
            // S3.4b: AllGather row-parallel projections before full GDN.
            if (layer.gdn.qkv.rows * 2u == kQ38GdnQkvWidth)
                te_bridge_allgather_bf16(options.te_rank, workspace.proj1,
                                         workspace.proj1, layer.gdn.qkv.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
            if (layer.gdn.z.rows * 2u == kQ38GdnValueWidth)
                te_bridge_allgather_bf16(options.te_rank, workspace.proj3,
                                         workspace.proj3, layer.gdn.z.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
            if (layer.gdn.b.rows * 2u == kQ38GdnValueHeads)
                te_bridge_allgather_bf16(options.te_rank, workspace.small0,
                                         workspace.small0, layer.gdn.b.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
            if (layer.gdn.a.rows * 2u == kQ38GdnValueHeads)
                te_bridge_allgather_bf16(options.te_rank, workspace.small1,
                                         workspace.small1, layer.gdn.a.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
        }
        const auto state = gdn_state->working(layer.state_local);
        cuda_gdn_conv_decode_bf16(
            workspace.proj1, layer.gdn.conv, state.conv, workspace.proj2,
            stream, options.device);
        cuda_gdn_recurrent_decode_bf16(
            workspace.proj2, workspace.small0, workspace.small1,
            layer.gdn.a_log, layer.gdn.dt_bias, state.recurrent,
            workspace.proj4, stream, options.device);
        cuda_gdn_output_norm_bf16(
            workspace.proj4, workspace.proj3, layer.gdn.norm, workspace.proj0,
            1.0e-6f, stream, options.device);
        const std::uint16_t* out_in = workspace.proj0;
        if (options.te_world_size == 2 &&
            layer.gdn.output.columns * 2u == kQ38GdnValueWidth)
            out_in = workspace.proj0 +
                     static_cast<std::size_t>(options.te_rank) *
                         layer.gdn.output.columns;
        cuda_gemv_bf16(layer.gdn.output, out_in, workspace.block_output, 1,
                       stream, options.device);
        if (options.te_world_size == 2)
            te_bridge_allreduce_sum_bf16(options.te_rank, workspace.block_output,
                                         kQ38HiddenWidth, te_bridge_group_for(options), workspace.stream);
    }

    void run_gdn_prefill(const LayerWeights& layer, std::uint32_t tokens) {
        if(run_gdn_local(layer,tokens,false)) return;
        auto stream = reinterpret_cast<void*>(workspace.stream);
        project(workspace.mixed, tokens, {
            {layer.gdn.qkv, workspace.proj1},
            {layer.gdn.z, workspace.proj3},
            {layer.gdn.b, workspace.small0},
            {layer.gdn.a, workspace.small1}});
        if (options.te_world_size == 2) {
            // Prefill TE: token-major AllGather (per-token half -> full).
            if (layer.gdn.qkv.rows * 2u == kQ38GdnQkvWidth)
                te_bridge_allgather_bf16(options.te_rank, workspace.proj1,
                                         workspace.proj1, layer.gdn.qkv.rows,
                                         tokens, te_bridge_group_for(options), workspace.stream);
            if (layer.gdn.z.rows * 2u == kQ38GdnValueWidth)
                te_bridge_allgather_bf16(options.te_rank, workspace.proj3,
                                         workspace.proj3, layer.gdn.z.rows,
                                         tokens, te_bridge_group_for(options), workspace.stream);
            if (layer.gdn.b.rows * 2u == kQ38GdnValueHeads)
                te_bridge_allgather_bf16(options.te_rank, workspace.small0,
                                         workspace.small0, layer.gdn.b.rows,
                                         tokens, te_bridge_group_for(options), workspace.stream);
            if (layer.gdn.a.rows * 2u == kQ38GdnValueHeads)
                te_bridge_allgather_bf16(options.te_rank, workspace.small1,
                                         workspace.small1, layer.gdn.a.rows,
                                         tokens, te_bridge_group_for(options), workspace.stream);
        }
        if (detailed_prefill_profile()) profile_mark("gdn_project");
        const auto state = gdn_state->working(layer.state_local);
        cuda_gdn_conv_prefill_bf16(
            workspace.proj1, layer.gdn.conv, state.conv, workspace.proj2,
            tokens, stream, options.device, gdn_prefix_view(layer, false));
        if (detailed_prefill_profile()) profile_mark("gdn_conv");
        if (precomputed_gdn_prefill_enabled())
            cuda_gdn_recurrent_prefill_precomputed_bf16(
                workspace.proj2, workspace.small0, workspace.small1,
                layer.gdn.a_log, layer.gdn.dt_bias, state.recurrent,
                workspace.block_scores, workspace.proj4, tokens, stream,
                options.device, gdn_prefix_view(layer, true));
        else if (partitioned_gdn_prefill_enabled())
            cuda_gdn_recurrent_prefill_partitioned_bf16(
                workspace.proj2, workspace.small0, workspace.small1,
                layer.gdn.a_log, layer.gdn.dt_bias, state.recurrent,
                workspace.proj4, tokens, stream, options.device);
        else
            cuda_gdn_recurrent_prefill_bf16(
                workspace.proj2, workspace.small0, workspace.small1,
                layer.gdn.a_log, layer.gdn.dt_bias, state.recurrent,
                workspace.proj4, tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("gdn_recurrent");
        cuda_gdn_output_norm_prefill_bf16(
            workspace.proj4, workspace.proj3, layer.gdn.norm, workspace.proj0,
            tokens, 1.0e-6f, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("gdn_output_norm");
        {
            const std::uint16_t* out_in = workspace.proj0;
            if (options.te_world_size == 2 &&
                layer.gdn.output.columns * 2u == kQ38GdnValueWidth)
            {
                const auto width = layer.gdn.output.columns;
                check(cudaMemcpy2DAsync(
                    workspace.proj2, width * sizeof(std::uint16_t),
                    workspace.proj0 + options.te_rank * width,
                    width * 2 * sizeof(std::uint16_t),
                    width * sizeof(std::uint16_t), tokens,
                    cudaMemcpyDeviceToDevice, workspace.stream),
                    "pack token-major TE column input");
                out_in = workspace.proj2;
            }
            cuda_gemv_bf16(layer.gdn.output, out_in, workspace.block_output,
                           tokens, stream, options.device);
        }
        if (options.te_world_size == 2)
            te_bridge_allreduce_sum_bf16(
                options.te_rank, workspace.block_output,
                static_cast<std::size_t>(kQ38HiddenWidth) * tokens, te_bridge_group_for(options), workspace.stream);
        if (detailed_prefill_profile()) profile_mark("gdn_output");
    }

    void run_qsa(const LayerWeights& layer, std::uint32_t position,
                 CudaQsaStateBank& state_bank) {

        const bool local = profile_requested("Q38_TP_QSA_LOCAL") &&
            options.te_world_size == 2 && layer.sparse.output.columns == 3072;
        const bool local_projection = local && 1 < 8 && profile_requested("Q38_TP_QSA_LOCAL_PROJECTION");
        ScopedQsaHeadLayout head_scope(local
            ? CudaQsaHeadLayout{unsigned(options.te_rank) * 12u,12u,
                local_projection ? unsigned(options.te_rank) * 12u : 0u,
                local_projection ? 12u : 24u}
            : CudaQsaHeadLayout{});
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const auto state = state_bank.working(layer.state_local);
        project(workspace.mixed, 1, {
            {local_projection ? layer.sparse.q_gate_local : layer.sparse.q_gate, workspace.proj0},
            {layer.sparse.k, workspace.small0},
            {layer.sparse.v, workspace.small1},
            {layer.sparse.index_qk, workspace.small2}});
        if (options.te_world_size == 2) {
            if (layer.sparse.q_gate.rows * 2u == 12288u)
                te_bridge_allgather_bf16(options.te_rank, workspace.proj0,
                                         workspace.proj0, layer.sparse.q_gate.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
            // k/v/index often replicated or smaller; AG when halved
            if (layer.sparse.k.rows * 2u == 512u)
                te_bridge_allgather_bf16(options.te_rank, workspace.small0,
                                         workspace.small0, layer.sparse.k.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
            if (layer.sparse.v.rows * 2u == 512u)
                te_bridge_allgather_bf16(options.te_rank, workspace.small1,
                                         workspace.small1, layer.sparse.v.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
            if (layer.sparse.index_qk.rows * 2u == 640u)
                te_bridge_allgather_bf16(options.te_rank, workspace.small2,
                                         workspace.small2,
                                         layer.sparse.index_qk.rows, /*tokens=*/1, te_bridge_group_for(options), workspace.stream);
        }
        profile_mark("qsa_project");
        cuda_qsa_prepare_main_decode_bf16(
            workspace.proj0, workspace.small0, workspace.small1,
            layer.sparse.q_norm, layer.sparse.k_norm, workspace.proj3,
            workspace.proj4, state, position, stream, options.device);
        cuda_qsa_prepare_index_decode_bf16(
            workspace.small2, layer.sparse.index_q_norm,
            layer.sparse.index_k_norm, workspace.small0, state, position,
            stream, options.device);
        profile_mark("qsa_prepare");
        const auto selected = cuda_qsa_select_decode(
            workspace.small0, state, position, workspace.block_scores,
            workspace.selected_indices, stream, options.device);
        profile_mark("qsa_select");
        cuda_qsa_attention_decode_bf16(
            workspace.proj3, state, workspace.selected_indices, selected,
            workspace.attention_scores, workspace.proj0, stream,
            options.device);
        profile_mark("qsa_attention");
        cuda_sigmoid_multiply_bf16(
            workspace.proj0, workspace.proj4, workspace.proj3,
            kQ38GdnValueWidth, stream, options.device);
        profile_mark("qsa_gate");
        {
            const std::uint16_t* out_in = workspace.proj3;
            if (options.te_world_size == 2 &&
                layer.sparse.output.columns * 2u == 6144u)
                out_in = workspace.proj3 +
                         static_cast<std::size_t>(options.te_rank) *
                             layer.sparse.output.columns;
            cuda_gemv_bf16(layer.sparse.output, out_in, workspace.block_output,
                           1, stream, options.device);
        }
        // Replicated MTP QSA already produced the full output on each rank.
        if (options.te_world_size == 2 &&
            layer.sparse.output.columns * 2u == 6144u)
            te_bridge_allreduce_sum_bf16(options.te_rank, workspace.block_output,
                                         kQ38HiddenWidth, te_bridge_group_for(options), workspace.stream);
        profile_mark("qsa_output");
    }

    void run_qsa_prefill(const LayerWeights& layer,
                         std::uint32_t first_position,
                         std::uint32_t tokens,
                         CudaQsaStateBank& state_bank,
                         bool cache_only = false) {

        const bool local = profile_requested("Q38_TP_QSA_LOCAL") &&
            options.te_world_size == 2 && layer.sparse.output.columns == 3072;
        const bool local_projection = local && tokens < 8 && profile_requested("Q38_TP_QSA_LOCAL_PROJECTION");
        ScopedQsaHeadLayout head_scope(local
            ? CudaQsaHeadLayout{unsigned(options.te_rank) * 12u,12u,
                local_projection ? unsigned(options.te_rank) * 12u : 0u,
                local_projection ? 12u : 24u}
            : CudaQsaHeadLayout{});
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const auto state = state_bank.working(layer.state_local);
        const bool omit_query = cache_only && profile_requested("Q38_CUDA_MTP_KV_ONLY");
        if (omit_query) {
            project(workspace.mixed, tokens, {
                {layer.sparse.k, workspace.small0},
                {layer.sparse.v, workspace.small1},
                {layer.sparse.index_qk, workspace.small2}});
        } else {
            project(workspace.mixed, tokens, {
                {local_projection ? layer.sparse.q_gate_local : layer.sparse.q_gate, workspace.proj0},
                {layer.sparse.k, workspace.small0},
                {layer.sparse.v, workspace.small1},
                {layer.sparse.index_qk, workspace.small2}});
        }
        if (options.te_world_size == 2) {
            if (!omit_query && layer.sparse.q_gate.rows * 2u == 12288u)
                te_bridge_allgather_bf16(options.te_rank, workspace.proj0,
                                         workspace.proj0,
                                         layer.sparse.q_gate.rows, tokens, te_bridge_group_for(options), workspace.stream);
            if (layer.sparse.k.rows * 2u == 512u)
                te_bridge_allgather_bf16(options.te_rank, workspace.small0,
                                         workspace.small0, layer.sparse.k.rows,
                                         tokens, te_bridge_group_for(options), workspace.stream);
            if (layer.sparse.v.rows * 2u == 512u)
                te_bridge_allgather_bf16(options.te_rank, workspace.small1,
                                         workspace.small1, layer.sparse.v.rows,
                                         tokens, te_bridge_group_for(options), workspace.stream);
            if (layer.sparse.index_qk.rows * 2u == 640u)
                te_bridge_allgather_bf16(options.te_rank, workspace.small2,
                                         workspace.small2,
                                         layer.sparse.index_qk.rows, tokens, te_bridge_group_for(options), workspace.stream);
        }
        if (detailed_prefill_profile()) profile_mark("qsa_project");
        cuda_qsa_prepare_prefill_bf16(
            workspace.proj0, workspace.small0, workspace.small1,
            workspace.small2, layer.sparse.q_norm, layer.sparse.k_norm,
            layer.sparse.index_q_norm, layer.sparse.index_k_norm,
            workspace.proj3, workspace.proj4, workspace.proj1, state,
            first_position, tokens, stream, options.device, omit_query);
        if (detailed_prefill_profile()) profile_mark("qsa_prepare");
        // Preparation writes all persistent QSA state (K/V, raw index keys,
        // pooled index keys). MTP commit repair only needs those writes; the
        // attention result and following feed-forward output are discarded.
        if (cache_only) return;
        cuda_qsa_select_prefill(
            workspace.proj1, state, first_position, tokens,
            workspace.block_scores,
            kQ38ContextLimit / kQ38QsaBlockTokens,
            workspace.selected_indices, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("qsa_select");
        const bool grouped_qsa = grouped_qsa_prefill_enabled() &&
            !(tokens<=7 && profile_requested("Q38_CUDA_QSA_SMALL_UNGROUPED"));
        const bool fused_grouped_qsa =
            grouped_qsa && fused_grouped_qsa_prefill_enabled();
        if (fused_grouped_qsa) {
            cuda_qsa_apply_grouped_fused_prefill_bf16(
                workspace.proj3, state, first_position, tokens,
                workspace.selected_indices, workspace.attention_scores,
                workspace.proj0, stream, options.device);
            if (detailed_prefill_profile())
                profile_mark("qsa_attention_fused");
        } else if (grouped_qsa) {
            cuda_qsa_score_grouped_prefill_bf16(
                workspace.proj3, state, first_position, tokens,
                workspace.selected_indices, workspace.attention_scores,
                stream, options.device);
        } else {
            cuda_qsa_score_prefill_bf16(
                workspace.proj3, state, first_position, tokens,
                workspace.selected_indices, workspace.attention_scores,
                stream, options.device);
        }
        if (!fused_grouped_qsa) {
            if (detailed_prefill_profile()) profile_mark("qsa_scores");
            if (grouped_qsa)
                cuda_qsa_output_grouped_prefill_bf16(
                    state, first_position, tokens,
                    workspace.selected_indices, workspace.attention_scores,
                    workspace.proj0, stream, options.device);
            else
                cuda_qsa_output_prefill_bf16(
                    state, first_position, tokens,
                    workspace.selected_indices, workspace.attention_scores,
                    workspace.proj0, stream, options.device);
            if (detailed_prefill_profile()) profile_mark("qsa_value");
        }
        constexpr auto qsa_width = kQ38QsaHeads * kQ38QsaHeadWidth;
        const auto width = layer.sparse.output.columns;
        const bool sharded_output = options.te_world_size == 2 &&
            width * 2u == qsa_width;
        const bool compact_gate = sharded_output && tokens >= 8 &&
            profile_requested("Q38_CUDA_QSA_PREFILL_COMPACT_GATE");
        const std::uint16_t* out_in = workspace.proj3;
        if (compact_gate) {
            cuda_sigmoid_multiply_slice_bf16(
                workspace.proj0, workspace.proj4, workspace.proj2,
                tokens, qsa_width, options.te_rank * width, width, stream,
                options.device);
            if (detailed_prefill_profile()) profile_mark("qsa_gate");
            out_in = workspace.proj2;
        } else {
            cuda_sigmoid_multiply_bf16(
                workspace.proj0, workspace.proj4, workspace.proj3,
                static_cast<std::size_t>(tokens) * qsa_width, stream,
                options.device);
            if (detailed_prefill_profile()) profile_mark("qsa_gate");
            if (sharded_output) {
                check(cudaMemcpy2DAsync(
                    workspace.proj2, width * sizeof(std::uint16_t),
                    workspace.proj3 + options.te_rank * width,
                    qsa_width * sizeof(std::uint16_t),
                    width * sizeof(std::uint16_t), tokens,
                    cudaMemcpyDeviceToDevice, workspace.stream),
                    "pack token-major TE column input");
                out_in = workspace.proj2;
            }
        }
        cuda_gemv_bf16(layer.sparse.output, out_in, workspace.block_output,
                       tokens, stream, options.device);
        // Replicated MTP QSA already produced the full output on each rank.
        if (sharded_output)
            te_bridge_allreduce_sum_bf16(
                options.te_rank, workspace.block_output,
                static_cast<std::size_t>(kQ38HiddenWidth) * tokens, te_bridge_group_for(options), workspace.stream);
        if (detailed_prefill_profile()) profile_mark("qsa_output");
    }

    void run_moe_shared(const LayerWeights& layer, std::uint32_t tokens) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        project(workspace.mixed, tokens, {
            {layer.moe.shared_gate, workspace.small2},
            {layer.moe.shared_up, workspace.small1}});
        if (detailed_prefill_profile())
            profile_mark("moe_shared_projection");
        if (options.te_world_size == 2 &&
            layer.moe.shared_gate.rows * 2u == kQ38MoeIntermediate) {
            // Shared expert is row-sharded on the intermediate dim; gather
            // before SiLU so the replicated down proj sees full width 640.
            te_bridge_allgather_bf16(options.te_rank, workspace.small2,
                                     workspace.small2,
                                     layer.moe.shared_gate.rows, tokens,
                                     te_bridge_group_for(options), workspace.stream);
            te_bridge_allgather_bf16(options.te_rank, workspace.small1,
                                     workspace.small1,
                                     layer.moe.shared_up.rows, tokens,
                                     te_bridge_group_for(options), workspace.stream);
            if (tokens == 1)
                te_debug_probe_bf16("moe_shared_after_ag", options.te_rank,
                                    workspace.small2, workspace.stream);
        }
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark(detailed_prefill_profile() ? "moe_shared_gather"
                                                 : "moe_shared_gate_up");
        cuda_silu_multiply_bf16(
            workspace.small2, workspace.small1, workspace.small0,
            static_cast<std::size_t>(tokens) * kQ38MoeIntermediate, stream,
            options.device);
        if (options.te_world_size == 2 && tokens == 1)
            te_debug_probe_bf16("moe_shared_after_silu", options.te_rank,
                                workspace.small0, workspace.stream);
        cuda_gemv_bf16(layer.moe.shared_down, workspace.small0,
                       workspace.proj5, tokens, stream, options.device);
        if (options.te_world_size == 2 && tokens == 1)
            te_debug_probe_bf16("moe_shared_after_down", options.te_rank,
                                workspace.proj5, workspace.stream);
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_shared_down");
        cuda_gemv_bf16(layer.moe.shared_output_gate, workspace.mixed,
                       workspace.small1, tokens, stream, options.device);
    }

    void run_moe(const LayerWeights& layer, std::uint32_t tokens = 1) {
        bool shared_complete = false;
        const bool direct_reduce = options.te_world_size == 2 && tokens <= 7 &&
            cuda_marlin_moe_enabled() && profile_requested("Q38_CUDA_MARLIN_DIRECT_REDUCE");
        auto stream = reinterpret_cast<void*>(workspace.stream);
        cuda_gemv_bf16(layer.moe.router, workspace.mixed, workspace.small0,
                       tokens, stream, options.device);
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_router");
        cuda_topk_router_bf16(workspace.small0, workspace.expert_ids,
                              workspace.expert_weights, tokens, kQ38MoeExperts,
                              kQ38MoeTopK, true, stream, options.device);
        if (options.te_world_size == 2) {
            const auto local_experts = layer.moe.gate_up_experts.descriptor->shape[0];
            te_remap_experts_kernel<<<(tokens * kQ38MoeTopK + 255) / 256, 256,
                                      0, workspace.stream>>>(
                workspace.expert_ids, workspace.expert_weights,
                tokens * kQ38MoeTopK, options.te_rank * local_experts,
                local_experts);
            check(cudaPeekAtLastError(), "te_remap_experts_kernel");
        }
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_topk");
        const bool grouped_prefill =
            (tokens > 1 || cuda_marlin_moe_enabled()) && (options.te_world_size != 2 ||
                          profile_requested("Q38_TP_GROUPED_MOE")) &&
            workspace.moe_route_plan.header != nullptr;
        const bool detailed_prefill_profile =
            grouped_prefill && prefill_profile && prefill_profile->active() &&
            (profile_requested("Q38_CUDA_PROFILE_PREFILL_DETAIL") ||
             profile_requested("Q38_CUDA_PROFILE_PREFILL_MOE_DETAIL"));
        const auto grouped_mode = options.te_world_size == 2 &&
            profile_requested("Q38_TP_MOE_FP32_SCALES")
                ? Q38PrefillMoeModeV1::kGroupedFp32Scales : prefill_moe_mode();
        if (grouped_prefill) {
            cuda_moe_build_route_plan_v1(
                workspace.expert_ids, tokens, workspace.moe_route_plan,
                stream, options.device,
                static_cast<std::uint32_t>(layer.moe.gate_up_experts.descriptor->shape[0]),
                options.te_world_size == 2,
                !profile_requested("Q38_CUDA_MOE_ROUTE_HASH_SKIP"));
            if (detailed_prefill_profile) profile_mark("moe_route_plan");
            cuda_moe_pack_hidden_v1(
                workspace.mixed, workspace.moe_route_plan.packed_assignment,
                tokens * kQ38MoeTopK, workspace.moe_packed_hidden, stream,
                options.device);
            if (detailed_prefill_profile) profile_mark("moe_pack_hidden");
            if (cuda_marlin_moe_enabled()) {
                cuda_marlin_moe(layer.moe.gate_up_experts,layer.moe.down_experts,
                    workspace.moe_packed_hidden,workspace.moe_route_plan.tasks,
                    q38_moe_prefill_max_tasks_v1(tokens),tokens*kQ38MoeTopK,
                    workspace.moe_weighted_mid,direct_reduce ? workspace.te_routed : workspace.moe_route_output,
                    [&]{auto ws=workspace.marlin;
                        ws.owned_routes=workspace.moe_route_plan.expert_offsets+kQ38RouteExperts;
                        if (direct_reduce) {
                            ws.reduce_assignment_to_packed = workspace.moe_route_plan.assignment_to_packed;
                            ws.reduce_route_weights = workspace.expert_weights;
                        }
                        return ws;}(),stream);
                if (detailed_prefill_profile) profile_mark("moe_marlin");
            } else {
            cuda_moe_grouped_gate_up_v1(
                layer.moe.gate_up_experts, workspace.moe_packed_hidden,
                workspace.moe_route_plan.packed_assignment,
                workspace.expert_weights, workspace.moe_route_plan.tasks,
                q38_moe_prefill_max_tasks_v1(tokens),
                workspace.moe_weighted_mid, grouped_mode, stream,
                options.device, options.te_world_size != 2);
            if (detailed_prefill_profile) profile_mark("moe_gate_up");
            cuda_moe_grouped_down_v1(
                layer.moe.down_experts, workspace.moe_weighted_mid,
                workspace.moe_route_plan.tasks,
                q38_moe_prefill_max_tasks_v1(tokens),
                workspace.moe_route_output, grouped_mode, stream,
                options.device);
            if (detailed_prefill_profile) profile_mark("moe_down");
            }
            if (options.te_world_size == 2 && !direct_reduce)
                cuda_moe_reduce_top10_f32_v1(
                    workspace.moe_route_output,
                    workspace.moe_route_plan.assignment_to_packed,
                    tokens, workspace.te_routed, stream, options.device,
                    workspace.expert_weights);
        } else {
            // The decode workspace has one route row. Iterate rows for TP
            // prefill instead of passing a batch that overruns its scratch.
            const auto batch = options.te_world_size == 2 ? 1u : tokens;
            for (std::uint32_t first = 0; first < tokens; first += batch)
                cuda_moe_routed_bf16(
                    layer.moe.gate_up_experts, layer.moe.down_experts,
                    workspace.mixed + first * kQ38HiddenWidth,
                    workspace.expert_ids + first * kQ38MoeTopK,
                    workspace.expert_weights + first * kQ38MoeTopK,
                    batch, kQ38MoeTopK, workspace.moe_gate_up,
                    workspace.moe_activated, workspace.moe_accumulation,
                    workspace.proj6 + first * kQ38HiddenWidth,
                    stream, options.device,
                    options.te_world_size == 2 ? workspace.te_routed + first * kQ38HiddenWidth : nullptr);
        }
        if (options.te_world_size == 2) {
            const auto count = static_cast<std::size_t>(kQ38HiddenWidth) * tokens;
            if (grouped_prefill && profile_requested("Q38_TP_MOE_VALIDATE")) {
                // Compare the two operators on the SAME layer input. Model
                // logits alone conflate local arithmetic error with later
                // router changes and recurrent-state propagation. Diagnostic
                // only: preserve the grouped output used by this invocation.
                std::vector<float> grouped(count), row(count);
                check(cudaMemcpyAsync(grouped.data(), workspace.te_routed,
                    count * sizeof(float), cudaMemcpyDeviceToHost, workspace.stream),
                    "copy grouped MoE validation output");
                check(cudaStreamSynchronize(workspace.stream),
                    "grouped MoE validation ready");
                for (std::uint32_t first = 0; first < tokens; ++first)
                    cuda_moe_routed_bf16(
                        layer.moe.gate_up_experts, layer.moe.down_experts,
                        workspace.mixed + first * kQ38HiddenWidth,
                        workspace.expert_ids + first * kQ38MoeTopK,
                        workspace.expert_weights + first * kQ38MoeTopK,
                        1, kQ38MoeTopK, workspace.moe_gate_up,
                        workspace.moe_activated, workspace.moe_accumulation,
                        workspace.proj6 + first * kQ38HiddenWidth,
                        stream, options.device,
                        workspace.te_routed + first * kQ38HiddenWidth);
                check(cudaMemcpyAsync(row.data(), workspace.te_routed,
                    count * sizeof(float), cudaMemcpyDeviceToHost, workspace.stream),
                    "copy row MoE validation output");
                check(cudaStreamSynchronize(workspace.stream),
                    "row MoE validation ready");
                double squared_error = 0, squared_reference = 0, maximum = 0;
                std::size_t unequal = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    if (!std::isfinite(grouped[i]) || !std::isfinite(row[i]))
                        throw std::runtime_error("nonfinite TP MoE validation output");
                    const double delta = static_cast<double>(grouped[i]) - row[i];
                    squared_error += delta * delta;
                    squared_reference += static_cast<double>(row[i]) * row[i];
                    maximum = std::max(maximum, std::abs(delta));
                    unequal += grouped[i] != row[i];
                }
                std::fprintf(stderr,
                    "tp-moe-validation rank=%d tensor=%s tokens=%u max_abs=%.9g "
                    "rmse=%.9g relative_l2=%.9g unequal=%zu count=%zu\n",
                    options.te_rank, layer.moe.gate_up_experts.descriptor->name.c_str(),
                    tokens, maximum, std::sqrt(squared_error / count),
                    std::sqrt(squared_error / std::max(squared_reference, 1e-300)),
                    unequal, count);
                check(cudaMemcpyAsync(workspace.te_routed, grouped.data(),
                    count * sizeof(float), cudaMemcpyHostToDevice, workspace.stream),
                    "restore grouped MoE validation output");
                // grouped owns pageable host storage and must outlive upload.
                check(cudaStreamSynchronize(workspace.stream),
                    "grouped MoE validation restore ready");
            }
            if (profile_requested("Q38_TP_MOE_BF16_REDUCE")) {
                if (tokens > 1 && tokens <= 7 &&
                    layer.moe.shared_gate.rows == kQ38MoeIntermediate &&
                    profile_requested("Q38_TP_MOE_SHARED_OVERLAP")) {
                    te_bridge_allreduce_f32_as_bf16(options.te_rank, workspace.te_routed,
                        workspace.proj6, count, te_bridge_group_for(options), workspace.stream,
                        [&] { run_moe_shared(layer, tokens); });
                    shared_complete = true;
                } else te_bridge_allreduce_f32_as_bf16(options.te_rank, workspace.te_routed,
                    workspace.proj6, count, te_bridge_group_for(options), workspace.stream);
            } else if (tokens >= 8 && profile_requested("Q38_TP_F32_BF16_RSAG") &&
                te_bridge_f32_to_bf16_available()) {
                te_bridge_allreduce_f32_to_bf16(options.te_rank, workspace.te_routed,
                    workspace.proj6, count, te_bridge_group_for(options), workspace.stream);
            } else {
                if (tokens > 1 && tokens <= 7 &&
                    layer.moe.shared_gate.rows == kQ38MoeIntermediate &&
                    profile_requested("Q38_TP_MOE_SHARED_OVERLAP")) {
                    te_bridge_allreduce_sum_f32_parallel(options.te_rank, workspace.te_routed,
                        count, te_bridge_group_for(options), workspace.stream,
                        [&] { run_moe_shared(layer, tokens); });
                    shared_complete = true;
                } else te_bridge_allreduce_sum_f32(options.te_rank, workspace.te_routed,
                    count, te_bridge_group_for(options), workspace.stream);
                te_round_f32_bf16_kernel<<<(count + 255) / 256, 256, 0, workspace.stream>>>(
                    workspace.te_routed, workspace.proj6, count);
                check(cudaPeekAtLastError(), "round TE routed FP32 sum");
            }
            if (detailed_prefill_profile)
                profile_mark("moe_routed_reduce");
        }
        if (!detailed_prefill_profile &&
            (tokens == 1 ||
             (prefill_profile && prefill_profile->active())))
            profile_mark("moe_routed");
        if (!shared_complete) run_moe_shared(layer, tokens);
        if (fuse_moe_finish(tokens)) return;
        if (grouped_prefill && options.te_world_size != 2) {
            cuda_moe_reduce_top10_and_combine_shared_v1(
                workspace.moe_route_output,
                workspace.moe_route_plan.assignment_to_packed,
                workspace.proj5, workspace.small1, tokens,
                workspace.block_output, stream, options.device);
        } else {
            cuda_moe_combine_shared_bf16(
                workspace.proj6, workspace.proj5, workspace.small1,
                workspace.block_output, tokens, stream, options.device);
        }
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_shared_combine");
    }

    void enqueue_decode_static_action(const DecodeStaticAction& action,
                                      std::uint32_t data_index = 0,
                                      std::uint32_t result_index = 0,
                                      std::uint32_t tokens = 1) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const auto require_layer = [&]() -> const LayerWeights& {
            if (action.layer >= layers.size())
                throw std::logic_error(
                    "piecewise decode action has an invalid layer");
            return layers[action.layer];
        };
        switch (action.kind) {
        case DecodeStaticActionKind::kInput:
            if (options.stage == Stage::kStage0) {
                cuda_embedding_bf16(embedding, workspace.tokens + data_index,
                                    workspace.proj5, 1, stream,
                                    options.device);
                cuda_hyper_repeat_embedding_bf16(
                    workspace.proj5, workspace.hyper_a, 1, stream,
                    options.device);
            } else {
                check(cudaMemcpyAsync(
                          workspace.hyper_a, workspace.boundary +
                              static_cast<std::uint64_t>(data_index) * kQ38HyperWidth,
                          kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(graph stage boundary token)");
            }
            return;
        case DecodeStaticActionKind::kPle:
            if (tokens == 1) run_ple(require_layer(), action.first, action.second, 0);
            else run_ple_prefill(require_layer(), action.first, action.second, 0, tokens);
            return;
        case DecodeStaticActionKind::kPrepareAttention:
            prepare_hyper(require_layer().attention_hyper, action.first, tokens);
            return;
        case DecodeStaticActionKind::kQsa:
            throw std::logic_error("QSA graph action requires a scoped device position");
        case DecodeStaticActionKind::kGdn:
            if (tokens == 1) run_gdn(require_layer());
            else run_gdn_prefill(require_layer(), tokens);
            return;
        case DecodeStaticActionKind::kFinish:
            finish_hyper(action.first, action.second, tokens);
            return;
        case DecodeStaticActionKind::kPrepareMoe:
            prepare_hyper(require_layer().moe_hyper, action.first, tokens);
            return;
        case DecodeStaticActionKind::kMoeFinish:
            finish_moe_hyper(action.first, action.second, tokens);
            return;
        case DecodeStaticActionKind::kMoe:
            run_moe(require_layer(), tokens);
            return;
        case DecodeStaticActionKind::kOutput:
            if (options.stage == Stage::kStage0) {
                check(cudaMemcpyAsync(
                          workspace.boundary, action.first,
                          kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(graph stage0 boundary output)");
            } else {
                if (mtp)
                    check(cudaMemcpyAsync(
                              workspace.final_hc +
                                  static_cast<std::uint64_t>(result_index) * kQ38HyperWidth,
                              action.first,
                              kQ38HyperWidth * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice, workspace.stream),
                          "cudaMemcpyAsync(graph final target HC)");
                prepare_hyper(final_hyper, action.first);
                {
                    const std::uint16_t* lin = workspace.mixed;
                    if (options.te_world_size == 2 &&
                        output.columns * 2u == kQ38HiddenWidth)
                        lin = workspace.mixed +
                              static_cast<std::size_t>(options.te_rank) *
                                  output.columns;
                    cuda_gemv_bf16(output, lin, workspace.logits, 1, stream,
                                   options.device);
                }
                select_head(workspace.logits,workspace.predictions + result_index);
            }
            return;
        }
        throw std::logic_error("unknown piecewise decode action");
    }

    std::pair<std::uintptr_t, std::uintptr_t> decode_graph_state_key() const {
        const auto gdn = gdn_state->working(0);
        const auto gdn_key = reinterpret_cast<std::uintptr_t>(gdn.conv);
        const auto ple_key = ple_state
                                  ? reinterpret_cast<std::uintptr_t>(
                                        ple_state->working())
                                  : std::uintptr_t{0};
        return {gdn_key, ple_key};
    }

    DecodeGraphVariant* find_decode_graph_variant(
        std::uintptr_t gdn_working, std::uintptr_t ple_working,
        std::uint32_t tokens = 1, bool sparse_qsa = false) const {
        const auto prefix = tokens > 1 && capture_target_prefixes
            ? reinterpret_cast<std::uintptr_t>(target_prefixes->data) : 0;
        for (const auto& variant : decode_graphs) {
            if (variant->gdn_working == gdn_working &&
                variant->ple_working == ple_working && variant->tokens == tokens &&
                variant->prefix_storage == prefix && variant->sparse_qsa == sparse_qsa)
                return variant.get();
        }
        return nullptr;
    }

    DecodeGraphVariant& build_decode_graph_variant(
        std::uintptr_t gdn_working, std::uintptr_t ple_working,
        std::uint32_t tokens = 1, bool sparse_qsa = false,
        bool bulk_prefill = false) {
        // Each transaction alternates between two GDN/PLE working banks.  A
        // graph records raw kernel pointers, so keep one variant per pointer
        // pair rather than replaying a graph against the wrong bank.
        const auto variant_limit = options.te_world_size == 1 ? 2u :
            profile_requested("Q38_TP_PREFILL_GRAPH") ? 24u : 20u;
        if (decode_graphs.size() >= variant_limit)
            throw std::runtime_error(
                "piecewise graph exceeded bounded shape/bank variants");

        auto variant = std::make_unique<DecodeGraphVariant>();
        variant->gdn_working = gdn_working;
        variant->ple_working = ple_working;
        variant->tokens = tokens;
        variant->sparse_qsa = sparse_qsa;
        const bool whole = bulk_prefill ||
            (tokens > 1 && profile_requested("Q38_TP_WHOLE_TARGET_GRAPH"));
        variant->prefix_storage = tokens > 1 && capture_target_prefixes
            ? reinterpret_cast<std::uintptr_t>(target_prefixes->data) : 0;
        std::vector<std::vector<DecodeStaticAction>> pieces(1);
        auto* current = workspace.hyper_a;
        auto* alternate = workspace.hyper_b;

        for (std::uint32_t layer_index = 0;
             layer_index < layers.size(); ++layer_index) {
            const auto& layer = layers[layer_index];
            if (layer.has_ple) {
                pieces.back().push_back({DecodeStaticActionKind::kPle,
                                         layer_index, current, alternate});
                std::swap(current, alternate);
            }
            pieces.back().push_back(
                {DecodeStaticActionKind::kPrepareAttention, layer_index,
                 current, nullptr});
            if (layer.qsa) {
                if(whole)pieces.back().push_back({DecodeStaticActionKind::kQsa,layer_index,nullptr,nullptr});
                else {variant->qsa_layers.push_back(layer_index);pieces.emplace_back();}
            } else {
                pieces.back().push_back(
                    {DecodeStaticActionKind::kGdn, layer_index, nullptr,
                     nullptr});
            }
            pieces.back().push_back({DecodeStaticActionKind::kFinish,
                                     layer_index, current, alternate});
            std::swap(current, alternate);
            pieces.back().push_back(
                {DecodeStaticActionKind::kPrepareMoe, layer_index, current,
                 nullptr});
            pieces.back().push_back(
                {DecodeStaticActionKind::kMoe, layer_index, nullptr,
                 nullptr});
            pieces.back().push_back({DecodeStaticActionKind::kMoeFinish,
                                     layer_index, current, alternate});
            std::swap(current, alternate);
        }
        variant->output_hc = current;
        if (pieces.size() != variant->qsa_layers.size() + 1)
            throw std::logic_error(
                "piecewise decode fragment plan is inconsistent");

        std::uint64_t nodes = 0;
        for (const auto& piece : pieces) {
            if (piece.empty())
                throw std::logic_error(
                    "piecewise decode produced an empty graph fragment");
            auto profile = profile_requested("Q38_CUDA_PROFILE_GRAPHS") &&
                profile_requested("Q38_CUDA_PROFILE_GRAPH_INTERNAL")
                ? std::make_unique<CudaEventProfile>(options.device, "graph", true) : nullptr;
            auto* event_profile = profile.get();
            auto graph = std::make_unique<CapturedCudaGraph>(
                options.device, workspace.stream, [&]() {
                    capturing_profile_events = true;
                    if(whole)cuda_qsa_set_graph_position(workspace.qsa_graph_position,tokens);
                    try {
                        if (event_profile) event_profile->begin(workspace.stream);
                        for (const auto& action : piece) {
                            if(action.kind==DecodeStaticActionKind::kQsa)
                                run_qsa_prefill(layers[action.layer],sparse_qsa?4096u:0u,tokens,*qsa_state);
                            else enqueue_decode_static_action(action, 0, 0, tokens);
                            if (event_profile) {
                                static constexpr const char* labels[] = {
                                    "input", "ple", "attention_hyper", "gdn", "finish",
                                    "moe_hyper", "moe", "output", "qsa", "finish"};
                                event_profile->mark(labels[static_cast<unsigned>(action.kind)],
                                                    workspace.stream);
                            }
                        }
                    } catch (...) {cuda_qsa_set_graph_position(nullptr,0);capturing_profile_events = false; throw; }
                    cuda_qsa_set_graph_position(nullptr,0);
                    capturing_profile_events = false;
                }, options.te_world_size == 2, std::move(profile), bulk_prefill);
            nodes += graph->nodes();
            variant->fragments.push_back(std::move(graph));
        }
        for (const auto& fragment : variant->fragments)
            metrics.cuda_graph_captures += fragment->graphs();
        metrics.cuda_graph_nodes += nodes;
        auto* result = variant.get();
        decode_graphs.push_back(std::move(variant));
        return *result;
    }

    bool run_one_piecewise(std::uint32_t data_index, std::uint32_t ple_index,
                           std::uint32_t result_index, std::uint32_t position) {
        if (decode_graph_failed) {
            ++metrics.cuda_graph_fallbacks;
            return false;
        }
        check_cancelled();
        // PLE I/O and both H2D copies stay outside capture.  During capture
        // run_ple only records the GPU conversion/projection/convolution.
        if (options.stage == Stage::kStage0)
            ensure_ple_ready(result_index, ple_index, 1);

        const auto [gdn_key, ple_key] = decode_graph_state_key();
        auto* variant = find_decode_graph_variant(gdn_key, ple_key);
        if (!variant) {
            // Surface setup/copy failures before classifying a later error as
            // a graph-capture fallback.
            check(cudaStreamSynchronize(workspace.stream),
                  "cudaStreamSynchronize(piecewise graph setup)");
            try {
                variant = &build_decode_graph_variant(gdn_key, ple_key);
            } catch (const std::exception& error) {
                decode_graph_failed = true;
                ++metrics.cuda_graph_fallbacks;
                std::cerr
                    << "{\"type\":\"q38_cuda_graph_fallback\",\"stage\":"
                    << static_cast<unsigned>(options.stage)
                    << ",\"error\":\"" << error.what() << "\"}\n";
                return false;
            }
        }
        if (!variant ||
            variant->fragments.size() != variant->qsa_layers.size() + 1)
            throw std::logic_error(
                "piecewise decode graph variant is incomplete");

        // Input/output offsets vary for retained target rows. Keep only these
        // endpoints eager; every captured layer body retains stable pointers.
        enqueue_decode_static_action(
            {DecodeStaticActionKind::kInput, 0, nullptr, nullptr}, data_index);
        for (std::size_t index = 0; index < variant->qsa_layers.size();
             ++index) {
            check_cancelled();
            variant->fragments[index]->launch(workspace.stream, [this] { check_cancelled(); });
            metrics.cuda_graph_replays += variant->fragments[index]->graphs();
            run_qsa(layers[variant->qsa_layers[index]], position,
                    *qsa_state);
        }
        check_cancelled();
        variant->fragments.back()->launch(workspace.stream, [this] { check_cancelled(); });
        metrics.cuda_graph_replays += variant->fragments.back()->graphs();
        enqueue_decode_static_action(
            {DecodeStaticActionKind::kOutput, 0, variant->output_hc, nullptr},
            data_index, result_index);
        return true;
    }

    bool run_batch_piecewise(std::uint32_t tokens, std::uint32_t position,
                             std::uint16_t*& output_hc,
                             bool bulk_prefill = false) {
        if (!(bulk_prefill ? prefill_cache.sealed() : batch_graph_warmed))
            return false;
        auto& failed = bulk_prefill ? prefill_graph_failed : batch_graph_failed;
        if (failed) { ++metrics.cuda_graph_fallbacks; return false; }
        const bool whole = bulk_prefill || profile_requested("Q38_TP_WHOLE_TARGET_GRAPH");
        const bool sparse = whole && position + tokens > kQ38QsaTokenBudget;
        if(whole)check(cudaMemcpyAsync(workspace.qsa_graph_position,&position,sizeof(position),
            cudaMemcpyHostToDevice,workspace.stream),"upload target graph position");
        const auto [gdn, ple] = decode_graph_state_key();
        auto* variant = find_decode_graph_variant(gdn, ple, tokens, sparse);
        if (!variant) {
            try { variant = &build_decode_graph_variant(gdn, ple, tokens, sparse, bulk_prefill); }
            catch (const std::exception& error) {
                std::cerr << "{\"type\":\""
                    << (bulk_prefill ? "q38_tp_prefill_graph_fallback" : "q38_tp_batch_graph_fallback")
                    << "\",\"stage\":"
                    << unsigned(options.stage) << ",\"tokens\":" << tokens
                    << ",\"error\":\"" << error.what() << "\"}\n";
            }
        }
        bool captured = variant != nullptr;
        if (bulk_prefill) {
            // Capture records no GPU work. Agree before either rank can
            // enter a paired replay or queue its eager collectives.
            captured = te_bridge_pair_prefill_ready(te_bridge_group_for(options),
                                                    options.te_rank, captured);
        }
        if (!captured) {
            failed = true;
            ++metrics.cuda_graph_fallbacks;
            return false;
        }
        for (std::size_t index = 0; index < variant->qsa_layers.size(); ++index) {
            variant->fragments[index]->launch(workspace.stream, [this] { check_cancelled(); });
            metrics.cuda_graph_replays += variant->fragments[index]->graphs();
            if (profile_requested("Q38_CUDA_PROFILE_GRAPHS")) profile_mark("graph_fragment");
            run_qsa_prefill(layers[variant->qsa_layers[index]], position, tokens, *qsa_state);
            if (profile_requested("Q38_CUDA_PROFILE_GRAPHS")) profile_mark("qsa");
        }
        variant->fragments.back()->launch(workspace.stream, [this] { check_cancelled(); });
        metrics.cuda_graph_replays += variant->fragments.back()->graphs();
        output_hc = variant->output_hc;
        if (profile_requested("Q38_CUDA_PROFILE_GRAPHS")) {
            profile_mark("graph_fragment");
            profiled_batch_variant = variant;
        }
        if (profile_requested("Q38_TP_BATCH_GRAPH_DIAGNOSTICS"))
            std::fprintf(stderr, "tp-batch-graph replay rank=%d stage=%u tokens=%u\n",
                options.te_rank, unsigned(options.stage), tokens);
        return true;
    }

    void select_head(const std::uint16_t* logits,std::int32_t* predictions,unsigned tokens=1) {
        auto stream=reinterpret_cast<void*>(workspace.stream);
        if(output.rows==kVocabulary){
            if(options.te_world_size==2)te_bridge_allreduce_sum_bf16(options.te_rank,const_cast<std::uint16_t*>(logits),static_cast<std::size_t>(tokens)*kVocabulary,te_bridge_group_for(options),workspace.stream);
            for(unsigned row=0;row<tokens;++row)cuda_argmax_bf16(logits+row*static_cast<std::uint64_t>(kVocabulary),kVocabulary,predictions+row,stream,options.device);
            return;
        }
        if(tokens>7||output.rows*2u!=kVocabulary||output.columns!=kQ38HiddenWidth||options.te_world_size!=2)throw std::logic_error("invalid vocabulary-row head shape");
        if(profile_requested("Q38_TP_HEAD_PARALLEL_ARGMAX"))
            cuda_argmax_bf16_batch(logits,output.rows,tokens,workspace.head_partial_values,workspace.head_partial_ids,workspace.head_local_ids,stream,options.device);
        else for(unsigned row=0;row<tokens;++row)cuda_argmax_bf16(logits+row*static_cast<std::uint64_t>(output.rows),output.rows,workspace.head_local_ids+row,stream,options.device);
        head_candidate_kernel<<<1,32,0,workspace.stream>>>(logits,workspace.head_local_ids,workspace.head_candidates,output.rows,tokens,options.te_rank);
        check(cudaPeekAtLastError(),"local complete-vocabulary maximum");
        te_bridge_allgather_bf16(options.te_rank,workspace.head_candidates,workspace.head_candidates_full,4,tokens,te_bridge_group_for(options),workspace.stream);
        head_choose_kernel<<<1,32,0,workspace.stream>>>(workspace.head_candidates_full,predictions,tokens);
        check(cudaPeekAtLastError(),"global stable vocabulary maximum");
    }
    bool prepare_head_hyper_batch(const std::uint16_t* current,unsigned rows) {
        if(!profile_requested("Q38_TP_HEAD_HYPER_BATCH") ||
           (rows!=2 && rows!=3 && rows!=5 && rows!=7) ||
           final_hyper.has_inject || final_hyper.mix_down.rows!=320 ||
           final_hyper.mix_up.rows!=10240 || final_hyper.mix_down.columns!=10240 ||
           final_hyper.mix_up.columns!=320)return false;
        auto stream=reinterpret_cast<void*>(workspace.stream);
        cuda_hyper_group_rmsnorm_bf16(current,final_hyper.norm,workspace.hyper_norm,rows,stream,options.device);
        if(!cuda_try_hc_scalar_batch_bf16(final_hyper.mix_down,workspace.hyper_norm,workspace.low_rank,rows,stream,options.device,CudaHcActivation::kSilu,4.f,1.f))throw std::logic_error("head down batch layout");
        if(!cuda_try_hc_scalar_batch_bf16(final_hyper.mix_up,workspace.low_rank,workspace.mix_weights,rows,stream,options.device,CudaHcActivation::kSigmoid,1.f,1.f))throw std::logic_error("head up batch layout");
        cuda_hyper_mix_bf16(workspace.hyper_norm,workspace.mix_weights,workspace.mixed,rows,stream,options.device);
        return true;
    }

    bool run_row_head_batch(const std::uint16_t* current,unsigned tokens,unsigned first,bool all_predictions){
        if(output.rows==kVocabulary)return false;
        auto stream=reinterpret_cast<void*>(workspace.stream);const unsigned rows=all_predictions?tokens:1;
        if(rows>7)throw std::logic_error("row head publication exceeds verification width");
        if(!capturing_head_graph && !head_graph_failed && all_predictions && first==0 &&
           (rows==2 || rows==3 || rows==5 || rows==7) && profile_requested("Q38_TP_HEAD_GRAPH") &&
           options.enable_piecewise_decode_graph && !std::getenv("Q38_TE_DEBUG")) {
            const auto key=std::make_pair(reinterpret_cast<std::uintptr_t>(current),rows);
            auto found=head_graphs.find(key);
            if(found==head_graphs.end()) {
                if(!head_graph_warmed){capturing_head_graph=true;try{run_row_head_batch(current,tokens,first,all_predictions);}catch(...){capturing_head_graph=false;throw;}capturing_head_graph=false;head_graph_warmed=true;return true;}
                try {
                    if(head_graphs.size()>=8)throw std::runtime_error("head graph cache bound exceeded");
                    auto graph=std::make_unique<CapturedCudaGraph>(options.device,workspace.stream,[&]{
                        capturing_head_graph=true;
                        try{run_row_head_batch(current,tokens,first,all_predictions);}catch(...){capturing_head_graph=false;throw;}
                        capturing_head_graph=false;
                    },options.te_world_size==2);
                    metrics.cuda_graph_captures+=graph->graphs();metrics.cuda_graph_nodes+=graph->nodes();
                    found=head_graphs.emplace(key,std::move(graph)).first;
                }catch(const std::exception& error){head_graph_failed=true;std::cerr<<"head graph fallback: "<<error.what()<<"\n";}
            }
            if(found!=head_graphs.end()){found->second->launch(workspace.stream,[this]{check_cancelled();});metrics.cuda_graph_replays+=found->second->graphs();return true;}
        }
        const bool batch_hyper=all_predictions && prepare_head_hyper_batch(current,rows);
        if(batch_hyper)check(cudaMemcpyAsync(workspace.proj0,workspace.mixed,static_cast<std::uint64_t>(rows)*output.columns*2,cudaMemcpyDeviceToDevice,workspace.stream),"pack batched row head input");
        else for(unsigned row=0;row<rows;++row){
            const unsigned source_row=all_predictions?row:tokens-1;
            prepare_hyper(final_hyper,current+static_cast<std::uint64_t>(source_row)*kQ38HyperWidth);
            if(rows==1)cuda_gemv_bf16(output,workspace.mixed,workspace.head_batch_logits,1,stream,options.device);
            else check(cudaMemcpyAsync(workspace.proj0+static_cast<std::uint64_t>(row)*output.columns,workspace.mixed,output.columns*2,cudaMemcpyDeviceToDevice,workspace.stream),"pack row head input");
        }
        if(rows>1&&!cuda_small_batch_gemv_bf16(output,workspace.proj0,workspace.head_batch_logits,rows,stream,options.device))
            for(unsigned row=0;row<rows;++row)cuda_gemv_bf16(output,workspace.proj0+static_cast<std::uint64_t>(row)*output.columns,
                workspace.head_batch_logits+static_cast<std::uint64_t>(row)*output.rows,1,stream,options.device);
        select_head(workspace.head_batch_logits,workspace.predictions+first+(all_predictions?0:tokens-1),rows);
        check(cudaMemcpyAsync(workspace.logits,workspace.head_batch_logits+(rows-1)*static_cast<std::uint64_t>(output.rows),output.rows*2,cudaMemcpyDeviceToDevice,workspace.stream),"retain local logits");
        return true;
    }

    void check_cancelled() const {
        if (provisional_cancellation)
            provisional_cancellation->throw_if_requested();
    }

    void run_one(std::uint32_t data_index, std::uint32_t ple_index,
                 std::uint32_t result_index, std::uint32_t position,
                 bool write_result) {
        check_cancelled();
        auto stream = reinterpret_cast<void*>(workspace.stream);
        if (options.stage == Stage::kStage0) {
            cuda_embedding_bf16(embedding, workspace.tokens + data_index,
                                workspace.proj5, 1, stream, options.device);
            cuda_hyper_repeat_embedding_bf16(
                workspace.proj5, workspace.hyper_a, 1, stream, options.device);
        } else {
            check(cudaMemcpyAsync(
                      workspace.hyper_a,
                      workspace.boundary +
                          static_cast<std::uint64_t>(data_index) *
                              kQ38HyperWidth,
                      kQ38HyperWidth * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToDevice, workspace.stream),
                  "cudaMemcpyAsync(stage boundary token)");
        }
        profile_mark("input");
        std::uint16_t* current = workspace.hyper_a;
        std::uint16_t* alternate = workspace.hyper_b;
        for (const auto& layer : layers) {
            check_cancelled();
            if (layer.has_ple) {
                ensure_ple_ready(result_index, ple_index, 1);
                run_ple(layer, current, alternate, ple_index);
                std::swap(current, alternate);
                profile_mark("ple");
            }
            prepare_hyper(layer.attention_hyper, current);
            profile_mark("attention_hyper");
            if (layer.qsa)
                run_qsa(layer, position, *qsa_state);
            else
                run_gdn(layer);
            profile_mark(layer.qsa ? "qsa" : "gdn");
            finish_hyper(current, alternate);
            std::swap(current, alternate);
            profile_mark("attention_finish");
            if (options.te_world_size == 2 && &layer == &layers.front())
                te_debug_probe_bf16("after_attn0", options.te_rank, current,
                                    workspace.stream);

            prepare_hyper(layer.moe_hyper, current);
            profile_mark("moe_hyper");
            run_moe(layer);
            finish_moe_hyper(current, alternate);
            std::swap(current, alternate);
            profile_mark("moe_finish");
            if (options.te_world_size == 2 && &layer == &layers.front())
                te_debug_probe_bf16("after_moe0", options.te_rank, current,
                                    workspace.stream);
        }
        if (options.stage == Stage::kStage0) {
            if (write_result)
                check(cudaMemcpyAsync(
                          workspace.boundary +
                              static_cast<std::uint64_t>(ple_index) *
                                  kQ38HyperWidth,
                          current, kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(stage0 boundary output)");
        } else {
            if (mtp)
                check(cudaMemcpyAsync(
                          workspace.final_hc +
                              static_cast<std::uint64_t>(result_index) *
                                  kQ38HyperWidth,
                          current, kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(final target HC)");
            if (write_result) {
                if (options.te_world_size == 2)
                    te_debug_probe_bf16("before_final_hyper", options.te_rank,
                                        current, workspace.stream);
                prepare_hyper(final_hyper, current);
                if (options.te_world_size == 2)
                    te_debug_probe_bf16("after_final_hyper_mixed", options.te_rank,
                                        workspace.mixed, workspace.stream);
                {
                    const std::uint16_t* lin = workspace.mixed;
                    if (options.te_world_size == 2 &&
                        output.columns * 2u == kQ38HiddenWidth)
                        lin = workspace.mixed +
                              static_cast<std::size_t>(options.te_rank) *
                                  output.columns;
                    cuda_gemv_bf16(output, lin, workspace.logits, 1, stream,
                                   options.device);
                }
                select_head(workspace.logits,workspace.predictions + result_index);
                if (options.te_world_size == 2 &&
                    (!profile_requested("Q38_TP_DEFER_HEAD_SYNC") ||
                     std::getenv("Q38_TE_DEBUG"))) {
                    check(cudaStreamSynchronize(workspace.stream),
                          "sync te-run-one-head");
                    std::int32_t pred = -7;
                    check(cudaMemcpy(&pred, workspace.predictions + result_index,
                                     sizeof(pred), cudaMemcpyDeviceToHost),
                          "D2H te-run-one-head");
                    if (std::getenv("Q38_TE_DEBUG")) std::fprintf(stderr,
                        "te-run-one-head done rank=%d pred=%d\n",
                        options.te_rank, pred);
                    std::fflush(stderr);
                }
            }
        }
        profile_mark(options.stage == Stage::kStage0 ? "stage_output"
                                                      : "final_head");
    }

    void run_prefill(std::uint32_t count, std::uint32_t chunk_offset,
                     bool write_result) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const bool all_predictions = provisional_kind == TxnKind::kSpeculative;
        capture_target_prefixes = all_predictions && chunk_offset == 0 &&
            count == provisional_expected && count > 1 && count <= 7 &&
            options.te_world_size == 2 && precomputed_gdn_prefill_enabled() &&
            profile_requested("Q38_TP_PREFIX_SNAPSHOTS");
        if (capture_target_prefixes) {
            if (!target_prefixes)
                target_prefixes = std::make_unique<TargetPrefixStorage>(
                    options.device, gdn_state->bytes_per_bank(), ple_state ?
                    std::uint64_t(kQ38HyperWidth) * kQ38PleConvState * 2 : 0);
            target_prefix_count = count - 1;
        }
        for (std::uint32_t first = 0; first < count;
             first += prefill_tile_tokens()) {
            check_cancelled();
            const auto tokens =
                std::min<std::uint32_t>(prefill_tile_tokens(), count - first);
            // Profile two representative full tiles per chunk: one after the
            // lazy matrix-cache warmup and one at the largest context in the
            // chunk. Also profile speculative verification and short prompts;
            // their small row counts are separate shapes in the JSON records.
            const bool profile_prefill =
                prefill_profile && (all_predictions || count < prefill_tile_tokens() ||
                    (tokens == prefill_tile_tokens() &&
                     (first == prefill_tile_tokens() || first + tokens == count)));
            if (profile_prefill)
                prefill_profile->begin(workspace.stream);
            if (options.stage == Stage::kStage0) {
                cuda_embedding_bf16(embedding, workspace.tokens + first,
                                    workspace.proj5, tokens, stream,
                                    options.device);
                cuda_hyper_repeat_embedding_bf16(
                    workspace.proj5, workspace.hyper_a, tokens, stream,
                    options.device);
            } else {
                check(cudaMemcpyAsync(
                          workspace.hyper_a,
                          workspace.boundary +
                              static_cast<std::uint64_t>(first) *
                                  kQ38HyperWidth,
                          static_cast<std::size_t>(tokens) * kQ38HyperWidth *
                              sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(stage boundary tile)");
            }
            if (profile_prefill) profile_mark("input");
            std::uint16_t* current = workspace.hyper_a;
            std::uint16_t* alternate = workspace.hyper_b;
            const bool graph_eligible = options.te_world_size == 2 && all_predictions &&
                write_result && chunk_offset == 0 && first == 0 && count == provisional_expected &&
                (tokens == 2 || tokens == 3 || tokens == 5 || tokens == 7) &&
                options.enable_piecewise_decode_graph && profile_requested("Q38_TP_BATCHED_GRAPH") &&
                !decode_profile && (!prefill_profile || profile_requested("Q38_CUDA_PROFILE_GRAPHS"));
            const bool bulk_graph_eligible = provisional_kind == TxnKind::kAppendKnown &&
                tokens == kQ38QsaPrefillGraphTokens && options.te_world_size == 2 && te_bridge_p2p_enabled() &&
                prefill_cache.sealed() && !prefill_graph_failed &&
                options.enable_piecewise_decode_graph && !decode_profile && !prefill_profile &&
                profile_requested("Q38_TP_PAIRED_GRAPH") &&
                profile_requested("Q38_TP_PREFILL_GRAPH");
            if ((graph_eligible || bulk_graph_eligible) && options.stage == Stage::kStage0)
                ensure_ple_ready(chunk_offset + first, first, tokens);
            const bool graphed = (graph_eligible || bulk_graph_eligible) &&
                run_batch_piecewise(tokens,
                    static_cast<std::uint32_t>(provisional_base + chunk_offset + first),
                    current, bulk_graph_eligible);
            if (!graphed) for (const auto& layer : layers) {
                check_cancelled();
                if (layer.has_ple) {
                    ensure_ple_ready(chunk_offset + first, first, tokens);
                    run_ple_prefill(layer, current, alternate, first, tokens);
                    std::swap(current, alternate);
                    if (profile_prefill && !detailed_prefill_profile())
                        profile_mark("ple");
                }
                prepare_hyper(layer.attention_hyper, current, tokens);
                if (profile_prefill) profile_mark("attention_hyper");
                if (layer.qsa)
                    run_qsa_prefill(
                        layer,
                        static_cast<std::uint32_t>(provisional_base +
                                                   chunk_offset + first),
                        tokens, *qsa_state);
                else
                    run_gdn_prefill(layer, tokens);
                if (profile_prefill && !detailed_prefill_profile())
                    profile_mark(layer.qsa ? "qsa" : "gdn");
                finish_hyper(current, alternate, tokens);
                std::swap(current, alternate);
                if (profile_prefill) profile_mark("attention_finish");

                prepare_hyper(layer.moe_hyper, current, tokens);
                if (profile_prefill) profile_mark("moe_hyper");
                run_moe(layer, tokens);
                finish_moe_hyper(current, alternate, tokens);
                std::swap(current, alternate);
                if (profile_prefill) profile_mark("moe_finish");
            }
            // First eligible eager batch warms every matrix/handle and the
            // largest padded GEMM scratch. Freeze addresses before recording.
            if (graph_eligible && !batch_graph_warmed) {
                prefill_cache.seal();
                batch_graph_warmed = true;
            }
            if (options.stage == Stage::kStage0) {
                if (write_result)
                    check(cudaMemcpyAsync(
                              workspace.boundary +
                                  static_cast<std::uint64_t>(first) *
                                      kQ38HyperWidth,
                              current,
                              static_cast<std::size_t>(tokens) *
                                  kQ38HyperWidth * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice, workspace.stream),
                          "cudaMemcpyAsync(stage0 boundary tile)");
            } else {
                if (mtp)
                    check(cudaMemcpyAsync(
                              workspace.final_hc +
                                  static_cast<std::uint64_t>(first) *
                                      kQ38HyperWidth,
                              current,
                              static_cast<std::size_t>(tokens) *
                                  kQ38HyperWidth * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice, workspace.stream),
                          "cudaMemcpyAsync(final target HC tile)");
                if (write_result && (all_predictions || first + tokens == count) &&
                    !run_row_head_batch(current,tokens,first,all_predictions)) {
                  // Keep the original per-row projection and BF16 rounding.
                  // Only coalesce independent vocabulary reductions; debug
                  // readbacks retain the original row-by-row execution path.
                  const bool packed_head = all_predictions && tokens <= 7 &&
                      options.te_world_size == 2 && workspace.head_batch_logits &&
                      !std::getenv("Q38_TE_DEBUG");
                  const bool batched_projection = packed_head && tokens > 1 &&
                      profile_requested("Q38_TP_COMPACT_HEAD_BATCH") &&
                      (tokens == 2 || tokens == 3 || tokens == 5 || tokens == 7) &&
                      output.group_size == 128 && output.columns * tokens * 2 <= 48 * 1024;
                  for (std::uint32_t head_row = all_predictions ? 0u : tokens - 1;
                       head_row < tokens; ++head_row) {
                    auto* head_logits = packed_head
                        ? workspace.head_batch_logits +
                              static_cast<std::uint64_t>(head_row) * kVocabulary
                        : workspace.logits;
                    const auto* last =
                        current + static_cast<std::uint64_t>(head_row) *
                                      kQ38HyperWidth;
                    prepare_hyper(final_hyper, last);
                    {
                        const std::uint16_t* lin = workspace.mixed;
                        if (options.te_world_size == 2 &&
                            output.columns * 2u == kQ38HiddenWidth)
                            lin = workspace.mixed +
                                  static_cast<std::size_t>(options.te_rank) *
                                      output.columns;
                        if (batched_projection)
                            check(cudaMemcpyAsync(workspace.proj0 + static_cast<std::size_t>(head_row)*output.columns,
                                lin,output.columns*sizeof(std::uint16_t),cudaMemcpyDeviceToDevice,workspace.stream),
                                "pack compact head inputs");
                        else cuda_gemv_bf16(output, lin, head_logits, 1, stream,
                                       options.device);
                    }
                    if (packed_head) continue;
                    if (options.te_world_size == 2)
                        te_bridge_allreduce_sum_bf16(
                            options.te_rank, workspace.logits, kVocabulary,
                            te_bridge_group_for(options), workspace.stream);
                    cuda_argmax_bf16(
                        workspace.logits, kVocabulary,
                        workspace.predictions + first + head_row, stream,
                        options.device);
                    if (options.te_world_size == 2 &&
                        (!profile_requested("Q38_TP_DEFER_HEAD_SYNC") ||
                         std::getenv("Q38_TE_DEBUG"))) {
                        check(cudaStreamSynchronize(workspace.stream),
                              "sync te pred");
                        std::int32_t pred = -999;
                        check(cudaMemcpy(&pred,
                                         workspace.predictions + first + head_row,
                                         sizeof(pred),
                                         cudaMemcpyDeviceToHost),
                              "D2H te pred");
                        // Peek a few logits for finiteness
                        std::uint16_t logit0 = 0;
                        check(cudaMemcpy(&logit0, workspace.logits, 2,
                                         cudaMemcpyDeviceToHost),
                              "D2H logit0");
                        if (std::getenv("Q38_TE_DEBUG")) std::fprintf(stderr,
                            "te-prefill-head rank=%d group=%d pred=%d "
                            "logit0_bf16=0x%04x out_cols=%u\n",
                            options.te_rank, te_bridge_group_for(options),
                            pred, logit0, output.columns);
                        std::fflush(stderr);
                    }
                  }
                  if (batched_projection && !cuda_small_batch_gemv_bf16(output, workspace.proj0,
                        workspace.head_batch_logits, tokens, stream, options.device))
                      throw std::logic_error("compact head batch dispatch rejected validated shape");
                  if (packed_head) {
                    te_bridge_allreduce_sum_bf16(
                        options.te_rank, workspace.head_batch_logits,
                        static_cast<std::size_t>(tokens) * kVocabulary,
                        te_bridge_group_for(options), workspace.stream);
                    for (std::uint32_t row = 0; row < tokens; ++row)
                        cuda_argmax_bf16(
                            workspace.head_batch_logits +
                                static_cast<std::uint64_t>(row) * kVocabulary,
                            kVocabulary, workspace.predictions + first + row,
                            stream, options.device);
                    // Preserve the existing last-row workspace contract.
                    check(cudaMemcpyAsync(workspace.logits,
                              workspace.head_batch_logits +
                                  static_cast<std::uint64_t>(tokens - 1) * kVocabulary,
                              kVocabulary * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice, workspace.stream),
                          "cudaMemcpyAsync(packed head last logits)");
                  }
                }
            }
            if (profile_prefill) {
                profile_mark(options.stage == Stage::kStage0
                                 ? "stage_output"
                                 : "final_head");
                prefill_profile->finish(
                    options.stage, provisional_base + chunk_offset + first,
                    workspace.stream, tokens);
                if (graphed && profiled_batch_variant)
                    for (const auto& fragment : profiled_batch_variant->fragments)
                        fragment->profile_finish(options.stage, provisional_base + chunk_offset + first,
                                                 workspace.stream, tokens);
            }
            if (first + tokens == count ||
                ((first / prefill_tile_tokens() + 1) % prefill_sync_tiles()) == 0)
                check(cudaStreamSynchronize(workspace.stream),
                      "cudaStreamSynchronize(prefill cancellation window)");
            check_cancelled();
        }
        qsa_state->mark_evaluated(provisional_epoch, chunk_offset + count);
        target_prefixes_valid = capture_target_prefixes;
        if (target_prefixes_valid && profile_requested("Q38_TP_PREFIX_DIAGNOSTICS"))
            std::fprintf(stderr, "tp-prefix capture rank=%d stage=%u epoch=%llu count=%u bytes=%llu\n",
                options.te_rank, unsigned(options.stage), (unsigned long long)provisional_epoch,
                target_prefix_count, (unsigned long long)(target_prefixes->stride * TargetPrefixStorage::capacity));
        capture_target_prefixes = false;
    }

    void run_tokens(std::uint32_t count, std::uint32_t chunk_offset,
                    bool write_result) {
        const bool batched_target = options.te_world_size == 2 &&
            provisional_kind == TxnKind::kSpeculative && chunk_offset == 0 &&
            count <= prefill_tile_tokens() && profile_requested("Q38_TP_BATCHED_TARGET");
        if ((provisional_kind == TxnKind::kAppendKnown || batched_target) && count > 1) {
            run_prefill(count, chunk_offset, write_result);
            return;
        }
        const bool graph_row = count == 1 && write_result &&
            (provisional_kind == TxnKind::kDecode ||
             provisional_kind == TxnKind::kSpeculative);
        const auto data_index = provisional_kind == TxnKind::kSpeculative
                                    ? chunk_offset : 0u;
        if (options.enable_piecewise_decode_graph && !decode_profile && graph_row &&
            run_one_piecewise(data_index, 0, chunk_offset,
                static_cast<std::uint32_t>(provisional_base + chunk_offset))) {
            qsa_state->mark_evaluated(provisional_epoch, chunk_offset + 1);
            return;
        }
        for (std::uint32_t token = 0; token < count; ++token) {
            const auto data_index =
                provisional_kind == TxnKind::kSpeculative
                    ? chunk_offset + token
                    : token;
            run_one(data_index, token, chunk_offset + token,
                    static_cast<std::uint32_t>(provisional_base +
                                               chunk_offset + token),
                    write_result);
        }
        qsa_state->mark_evaluated(provisional_epoch, chunk_offset + count);
    }

    template <class Enqueue>
    void run_mtp_fragment(std::size_t index, bool enabled, Enqueue enqueue) {
        if (!enabled || mtp_graph_failed) {
            enqueue();
            return;
        }
        if (!mtp_graphs[index]) {
            try {
                mtp_graphs[index] = std::make_unique<CapturedCudaGraph>(
                    options.device, workspace.stream, enqueue, options.te_world_size == 2);
                ++metrics.cuda_graph_captures;
                metrics.cuda_graph_nodes += mtp_graphs[index]->nodes();
            } catch (const std::exception& error) {
                mtp_graph_failed = true;
                ++metrics.cuda_graph_fallbacks;
                std::fprintf(stderr, "MTP graph fallback rank=%d fragment=%zu: %s\n",
                             options.te_rank, index, error.what());
                enqueue();
                return;
            }
        }
        mtp_graphs[index]->launch(workspace.stream);
        ++metrics.cuda_graph_replays;
    }

    void run_mtp_step(const std::uint16_t* target_hc,
                      const std::int32_t* token_device,
                      std::uint32_t position, bool produce_prediction,
                      std::int32_t* prediction_output = nullptr,
                      std::uint16_t* output_hc = nullptr) {
        if (!mtp || !mtp_qsa_state || !target_hc || !token_device)
            throw std::logic_error("MTP step is unavailable");
        if (produce_prediction && !prediction_output)
            throw std::logic_error("MTP prediction output is unavailable");
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const bool graph = workspace.mtp_graph_hc && !decode_profile;
        auto* requested_prediction = prediction_output;
        if (graph) {
            check(cudaMemcpyAsync(workspace.mtp_graph_hc, target_hc,
                kQ38HyperWidth * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                workspace.stream), "cudaMemcpyAsync(MTP graph HC)");
            check(cudaMemcpyAsync(workspace.mtp_graph_token, token_device,
                sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
                workspace.stream), "cudaMemcpyAsync(MTP graph token)");
            target_hc = workspace.mtp_graph_hc;
            token_device = workspace.mtp_graph_token;
            prediction_output = workspace.mtp_graph_prediction;
            if (!mtp_graphs[0] && !mtp_graph_failed)
                prefill_cache.warm_small(mtp->fc_hidden, workspace.stream);
        }
        auto input_fragment = [&] {
            // Official Qwen MTP fusion: project the normalized current-token
            // embedding once, project each of the four normalized target HC
            // streams with the shared hidden projection, broadcast the embedding
            // projection, then add the two 4H tensors.
            if(workspace.mtp_normalized_embedding) {
                auto normalized=mtp->embedding;
                normalized.data=workspace.mtp_normalized_embedding;
                normalized.format=DeviceWeightFormatV1::kPreserve;
                normalized.preserved_f32=false; normalized.scales=nullptr;
                cuda_embedding_bf16(normalized,token_device,workspace.proj6,1,stream,options.device);
            } else {
            cuda_embedding_bf16(mtp->embedding, token_device, workspace.proj5, 1,
                                stream, options.device);
            cuda_qwen38_rmsnorm_bf16(
                workspace.proj5, mtp->embedding_norm, false, workspace.proj6, 1,
                kQ38HiddenWidth, 1.0e-6f, true, stream, options.device);
            }
            cuda_gemv_bf16(mtp->fc_embedding, workspace.proj6, workspace.proj5, 1,
                           stream, options.device);
            cuda_qwen38_rmsnorm_bf16(
                target_hc, mtp->hidden_norm, false, workspace.hyper_norm, 1,
                kQ38HyperWidth, 1.0e-6f, true, stream, options.device);
            cuda_gemv_bf16(mtp->fc_hidden, workspace.hyper_norm,
                           workspace.hyper_a, kQ38HyperCount, stream,
                           options.device);
            if(profile_requested("Q38_CUDA_MTP_INPUT_CACHE"))
                cuda_hyper_add_embedding_bf16(workspace.hyper_a,workspace.proj5,workspace.hyper_a,1,stream,options.device);
            else {
            cuda_hyper_repeat_embedding_bf16(
                workspace.proj5, workspace.hyper_b, 1, stream, options.device);
            cuda_add_bf16(workspace.hyper_a, workspace.hyper_b,
                          workspace.hyper_a, kQ38HyperWidth, stream,
                          options.device);
            }

            prepare_hyper(mtp->layer.attention_hyper, workspace.hyper_a);
        };
        std::uint16_t* current = workspace.hyper_a;
        std::uint16_t* alternate = workspace.hyper_b;
        auto output_fragment = [&] {
            finish_hyper(current, alternate);
            std::swap(current, alternate);
            prepare_hyper(mtp->layer.moe_hyper, current);
            run_moe(mtp->layer);
            finish_moe_hyper(current, alternate);
            std::swap(current, alternate);

            if (produce_prediction) {
                prepare_hyper(mtp->output_hyper, current);
                {
                    const std::uint16_t* lin = workspace.mixed;
                    if (options.te_world_size == 2 &&
                        output.columns * 2u == kQ38HiddenWidth)
                        lin = workspace.mixed +
                              static_cast<std::size_t>(options.te_rank) *
                                  output.columns;
                    cuda_gemv_bf16(output, lin, workspace.logits, 1, stream,
                                   options.device);
                }
                select_head(workspace.logits,prediction_output);
            }
        };
        const bool whole=graph && profile_requested("Q38_TP_WHOLE_MTP_GRAPH") && !mtp_whole_graph_failed;
        if(whole) {
            check(cudaMemcpyAsync(workspace.qsa_graph_position,&position,sizeof(position),cudaMemcpyHostToDevice,
                workspace.stream),"upload MTP graph position");
            const bool sparse=position+1>kQ38QsaTokenBudget;
            // Reconciliation only consumes this row's QSA cache writes.
            const bool repair_cache_only = !produce_prediction && !output_hc &&
                profile_requested("Q38_CUDA_MTP_KV_ONLY");
            auto& captured = mtp_whole_graphs[(sparse ? 3 : 0) +
                (repair_cache_only ? 2 : static_cast<unsigned>(produce_prediction))];
            if(!captured) {
                try {
                    captured=std::make_unique<CapturedCudaGraph>(options.device,workspace.stream,[&]{
                        capturing_profile_events=true;
                        try {
                            input_fragment();
                            cuda_qsa_set_graph_position(workspace.qsa_graph_position,1);
                            run_qsa_prefill(mtp->layer, sparse ? 4096u : 0u, 1,
                                            *mtp_qsa_state, repair_cache_only);
                            cuda_qsa_set_graph_position(nullptr,0);
                            if (!repair_cache_only) output_fragment();
                        }catch(...){cuda_qsa_set_graph_position(nullptr,0);capturing_profile_events=false;throw;}
                        capturing_profile_events=false;
                    },options.te_world_size==2);
                    ++metrics.cuda_graph_captures;metrics.cuda_graph_nodes+=captured->nodes();
                }catch(const std::exception& error){mtp_whole_graph_failed=true;++metrics.cuda_graph_fallbacks;
                    std::fprintf(stderr,"MTP whole graph fallback rank=%d: %s\n",options.te_rank,error.what());}
            }
            if(captured){captured->launch(workspace.stream);++metrics.cuda_graph_replays;}
            else{run_mtp_fragment(0,graph,input_fragment);run_qsa(mtp->layer,position,*mtp_qsa_state);
                run_mtp_fragment(produce_prediction?1:2,graph,output_fragment);}
        }else {
            run_mtp_fragment(0,graph,input_fragment);run_qsa(mtp->layer,position,*mtp_qsa_state);
            run_mtp_fragment(produce_prediction?1:2,graph,output_fragment);
        }
        // There are two HC swaps in the output fragment, so current returns
        // to hyper_a on eager execution and on every replay.
        if (graph && produce_prediction)
            check(cudaMemcpyAsync(requested_prediction, workspace.mtp_graph_prediction,
                sizeof(std::int32_t), cudaMemcpyDeviceToDevice, workspace.stream),
                "cudaMemcpyAsync(MTP graph prediction)");
        if (output_hc)
            check(cudaMemcpyAsync(output_hc, current,
                                  kQ38HyperWidth * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToDevice, workspace.stream),
                  "cudaMemcpyAsync(MTP recurrent HC)");
    }

    std::vector<std::int32_t> run_mtp_draft(
        std::int32_t pending_token, std::uint64_t position,
        std::uint32_t max_draft, std::uint64_t retained_epoch,
        const std::shared_ptr<CancellationToken>& cancellation) {
        if (cancellation) cancellation->throw_if_requested();
        if (options.stage != Stage::kStage1)
            throw std::runtime_error("stage0 cannot draft tokens");
        if (!mtp)
            throw std::logic_error("MTP is disabled for this executor");
        if (max_draft == 0 || max_draft > kMaximumMtpDraftWidth ||
            max_draft > workspace.maximum_tokens ||
            pending_token < 0 ||
            static_cast<std::uint32_t>(pending_token) >= kVocabulary ||
            position != committed_frontier || provisional_epoch != 0 ||
            !mtp_pending_valid || !mtp_qsa_state ||
            mtp_qsa_state->committed_tokens() + 1 != position ||
            max_draft > mtp_qsa_state->capacity() -
                            mtp_qsa_state->committed_tokens() ||
            mtp_transaction_active)
            throw std::runtime_error("MTP draft frontier is not ready");

        check(cudaSetDevice(options.device), "cudaSetDevice(MTP draft)");
        constexpr std::uint64_t kEphemeralDraftEpoch =
            std::numeric_limits<std::uint64_t>::max();
        const auto epoch = retained_epoch == 0 ? kEphemeralDraftEpoch
                                                : retained_epoch;
        bool active = false;
        try {
            check(cudaMemcpyAsync(workspace.tokens, &pending_token,
                                  sizeof(pending_token), cudaMemcpyHostToDevice,
                                  workspace.stream),
                  "cudaMemcpyAsync(MTP token)");
            mtp_qsa_state->begin(epoch);
            active = true;
            const auto first_position = mtp_qsa_state->committed_tokens();
            for (std::uint32_t index = 0; index < max_draft; ++index) {
                if (cancellation) cancellation->throw_if_requested();
                const auto* target_hc = index == 0
                                            ? workspace.mtp_pending_hc
                                            : workspace.mtp_candidate_hc;
                const auto* token_device = index == 0
                                               ? workspace.tokens
                                               : workspace.predictions +
                                                     index - 1;
                run_mtp_step(
                    target_hc, token_device,
                    static_cast<std::uint32_t>(first_position + index), true,
                    workspace.predictions + index,
                    workspace.mtp_candidate_hc);
                mtp_qsa_state->mark_evaluated(epoch, index + 1);
            }
            check(cudaMemcpyAsync(workspace.host_predictions,
                                  workspace.predictions,
                                  static_cast<std::uint64_t>(max_draft) *
                                      sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  workspace.stream),
                  "cudaMemcpyAsync(MTP predictions)");
            check(cudaStreamSynchronize(workspace.stream),
                  "cudaStreamSynchronize(MTP draft)");
            if (cancellation) cancellation->throw_if_requested();
            if (retained_epoch != 0) {
                mtp_transaction_active = true;
                mtp_transaction_epoch = retained_epoch;
                mtp_transaction_rows = max_draft;
                mtp_reconciled_rows = 1;
                active = false;
            } else {
                mtp_qsa_state->rollback(epoch);
                active = false;
            }
            if (options.te_rank == 0 && profile_requested("Q38_MTP_DRAFT_DIAGNOSTICS")) {
                std::string record = "{\"event\":\"mtp_draft\",\"position\":" +
                    std::to_string(position) + ",\"tokens\":[";
                for (std::uint32_t i = 0; i < max_draft; ++i) {
                    if (i) record += ',';
                    record += std::to_string(workspace.host_predictions[i]);
                }
                record += "]}\n";
                std::fputs(record.c_str(), stderr);
                std::fflush(stderr);
            }
            return std::vector<std::int32_t>(workspace.host_predictions,
                                             workspace.host_predictions +
                                                 max_draft);
        } catch (...) {
            if (active) mtp_qsa_state->rollback(epoch);
            throw;
        }
    }

    void run_mtp_prefill(std::uint64_t first_position,
                         std::uint32_t evaluated,
                         bool had_pending) {
        if (!mtp || !mtp_qsa_state || evaluated == 0)
            throw std::logic_error("invalid MTP prefill extent");
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const bool cache_only = options.te_world_size == 2 &&
            profile_requested("Q38_TP_MTP_KV_ONLY_COMMIT");
        for (std::uint32_t first = 0; first < evaluated;
             first += kPrefillTile) {
            check_cancelled();
            const auto tokens =
                std::min<std::uint32_t>(kPrefillTile, evaluated - first);
            const auto token_index = first + (had_pending ? 0u : 1u);
            if (had_pending && first == 0) {
                check(cudaMemcpyAsync(
                          workspace.hyper_a, workspace.mtp_candidate_hc,
                          kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(MTP prefix pending HC)");
                if (tokens > 1)
                    check(cudaMemcpyAsync(
                              workspace.hyper_a + kQ38HyperWidth,
                              workspace.final_hc,
                              static_cast<std::size_t>(tokens - 1) *
                                  kQ38HyperWidth * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice, workspace.stream),
                          "cudaMemcpyAsync(MTP prefix target HC)");
            } else {
                const auto target_index =
                    had_pending ? first - 1u : first;
                check(cudaMemcpyAsync(
                          workspace.hyper_a,
                          workspace.final_hc +
                              static_cast<std::uint64_t>(target_index) *
                                  kQ38HyperWidth,
                          static_cast<std::size_t>(tokens) * kQ38HyperWidth *
                              sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(MTP target HC tile)");
            }

            cuda_embedding_bf16(mtp->embedding,
                                workspace.tokens + token_index,
                                workspace.proj5, tokens, stream,
                                options.device);
            cuda_qwen38_rmsnorm_bf16(
                workspace.proj5, mtp->embedding_norm, false, workspace.proj6,
                tokens, kQ38HiddenWidth, 1.0e-6f, true, stream,
                options.device);
            cuda_gemv_bf16(mtp->fc_embedding, workspace.proj6,
                           workspace.proj5, tokens, stream, options.device);
            cuda_qwen38_rmsnorm_bf16(
                workspace.hyper_a, mtp->hidden_norm, false,
                workspace.hyper_norm, tokens, kQ38HyperWidth, 1.0e-6f, true,
                stream, options.device);
            cuda_gemv_bf16(mtp->fc_hidden, workspace.hyper_norm,
                           workspace.hyper_a, tokens * kQ38HyperCount, stream,
                           options.device);
            cuda_hyper_repeat_embedding_bf16(
                workspace.proj5, workspace.hyper_b, tokens, stream,
                options.device);
            cuda_add_bf16(
                workspace.hyper_a, workspace.hyper_b, workspace.hyper_a,
                static_cast<std::size_t>(tokens) * kQ38HyperWidth, stream,
                options.device);

            std::uint16_t* current = workspace.hyper_a;
            std::uint16_t* alternate = workspace.hyper_b;
            prepare_hyper(mtp->layer.attention_hyper, current, tokens);
            run_qsa_prefill(
                mtp->layer,
                static_cast<std::uint32_t>(first_position + first), tokens,
                *mtp_qsa_state, cache_only);
            if (!cache_only) {
                finish_hyper(current, alternate, tokens);
                std::swap(current, alternate);
                prepare_hyper(mtp->layer.moe_hyper, current, tokens);
                run_moe(mtp->layer, tokens);
                finish_moe_hyper(current, alternate, tokens);
            }
            check(cudaStreamSynchronize(workspace.stream),
                  "cudaStreamSynchronize(MTP cancellation tile)");
            check_cancelled();
        }
    }

    // Advance MTP only over target-accepted rows.  A prefix of N target tokens
    // owns N-1 committed MTP QSA rows plus the final target HC kept pending.
    // This makes a rejected speculative suffix invisible to future drafts.
    void prepare_mtp_commit(std::uint64_t epoch, std::uint32_t accepted) {
        if (!mtp || !mtp_qsa_state || accepted == 0)
            throw std::logic_error("invalid MTP commit preparation");
        const auto prior_pairs = mtp_qsa_state->committed_tokens();
        if (prior_pairs + (mtp_pending_valid ? 1u : 0u) != provisional_base)
            throw std::runtime_error("MTP target frontier differs");

        mtp_candidate_pending_valid = mtp_pending_valid;
        if (mtp_pending_valid)
            check(cudaMemcpyAsync(workspace.mtp_candidate_hc,
                                  workspace.mtp_pending_hc,
                                  kQ38HyperWidth * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToDevice,
                                  workspace.stream),
                  "cudaMemcpyAsync(MTP committed to candidate HC)");

        const std::uint32_t evaluated =
            accepted - (mtp_pending_valid ? 0u : 1u);
        const bool retained_draft = mtp_transaction_active;
        if (retained_draft) {
            if (mtp_transaction_epoch != epoch ||
                mtp_transaction_rows == 0 || !mtp_pending_valid ||
                provisional_kind != TxnKind::kSpeculative ||
                provisional_expected != mtp_transaction_rows + 1 ||
                evaluated == 0 || evaluated > mtp_reconciled_rows)
                throw std::runtime_error(
                    "retained MTP draft does not match target commit");
        } else {
            mtp_transaction_rows = 0;
        }
        if (evaluated != 0 && !retained_draft) {
            mtp_qsa_state->begin(epoch);
            mtp_transaction_active = true;
            mtp_transaction_epoch = epoch;
            try {
                run_mtp_prefill(prior_pairs, evaluated, mtp_pending_valid);
                mtp_qsa_state->mark_evaluated(epoch, evaluated);
                mtp_transaction_rows = evaluated;
            } catch (...) {
                mtp_qsa_state->rollback(epoch);
                mtp_transaction_active = false;
                mtp_transaction_epoch = 0;
                throw;
            }
        } else if (retained_draft) {
            // Row zero was retained from canonical target HC. Each deeper
            // accepted row was reconciled online immediately after the
            // preceding target row verified its draft, overlapping GPU0's
            // one-row lookahead. Commit only the reconciled prefix; there is
            // no serial MTP repair loop on the commit path.
            mtp_transaction_rows = evaluated;
        }
        check(cudaMemcpyAsync(
                  workspace.mtp_candidate_hc,
                  workspace.final_hc +
                      static_cast<std::uint64_t>(accepted - 1) *
                          kQ38HyperWidth,
                  kQ38HyperWidth * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToDevice, workspace.stream),
              "cudaMemcpyAsync(MTP pending HC)");
        mtp_candidate_pending_valid = true;
        try {
            check(cudaStreamSynchronize(workspace.stream),
                  "cudaStreamSynchronize(MTP commit preparation)");
        } catch (...) {
            if (mtp_transaction_active) {
                mtp_qsa_state->rollback(mtp_transaction_epoch);
                mtp_transaction_active = false;
                mtp_transaction_epoch = 0;
                mtp_transaction_rows = 0;
            }
            throw;
        }
    }

    // Append transactions can span many device-workspace chunks.  Advance the
    // draft model after each target chunk while keeping its QSA rows and final
    // HC provisional under the same request epoch.
    void advance_mtp_append(std::uint64_t epoch, std::uint32_t chunk_offset,
                            std::uint32_t count) {
        if (!mtp || !mtp_qsa_state || count == 0)
            throw std::logic_error("invalid chunked MTP append");
        const auto prior_pairs = mtp_qsa_state->committed_tokens();
        if (prior_pairs + (mtp_pending_valid ? 1u : 0u) != provisional_base ||
            prior_pairs + mtp_transaction_rows +
                    (mtp_candidate_pending_valid ? 1u : 0u) !=
                provisional_base + chunk_offset)
            throw std::runtime_error("chunked MTP frontier differs");

        const auto evaluated =
            count - (mtp_candidate_pending_valid ? 0u : 1u);
        if (evaluated != 0) {
            if (!mtp_transaction_active) {
                mtp_qsa_state->begin(epoch);
                mtp_transaction_active = true;
                mtp_transaction_epoch = epoch;
            }
            run_mtp_prefill(prior_pairs + mtp_transaction_rows, evaluated,
                            mtp_candidate_pending_valid);
            mtp_transaction_rows += evaluated;
            mtp_qsa_state->mark_evaluated(epoch, mtp_transaction_rows);
        }
        check(cudaMemcpyAsync(
                  workspace.mtp_candidate_hc,
                  workspace.final_hc +
                      static_cast<std::uint64_t>(count - 1) * kQ38HyperWidth,
                  kQ38HyperWidth * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToDevice, workspace.stream),
              "cudaMemcpyAsync(chunked MTP candidate HC)");
        mtp_candidate_pending_valid = true;
    }

    void start_ple_prefetch(
        const SessionTxnV1& txn,
        const std::vector<std::int32_t>& token_ids,
        const std::shared_ptr<CancellationToken>& cancellation) {
        if (options.stage != Stage::kStage0 || !ple_reader ||
            !committed_ple_hash || txn.kind != TxnKind::kAppendKnown ||
            txn.status != TxnStatus::kPrepared || txn.epoch == 0 ||
            txn.epoch <= committed_epoch ||
            txn.base_target != committed_frontier ||
            token_ids.size() != txn.evaluated_count || token_ids.empty() ||
            token_ids.size() <= options.max_transaction_tokens ||
            provisional_epoch != 0 || states_active || ple_read_active)
            throw std::logic_error("invalid PLE transaction read-ahead");
        if (cancellation) cancellation->throw_if_requested();
        auto candidate = *committed_ple_hash;
        auto rows = candidate.rows(token_ids);
        if (cancellation) cancellation->throw_if_requested();
        ple_read_rows =
            std::make_shared<const std::vector<std::uint64_t>>(
                std::move(rows));
        ple_read_token_base = 0;
        ple_read_epoch = txn.epoch;
        ple_reader->submit(ple_read_rows);
        ple_read_active = true;
    }

    void start_ple_read(const std::vector<std::int32_t>& token_ids,
                        std::uint32_t token_offset) {
        if (!ple_reader)
            throw std::logic_error("PLE reader is unavailable");
        auto candidate = candidate_ple_hashes.empty()
                             ? *committed_ple_hash
                             : candidate_ple_hashes.back();
        std::vector<std::uint64_t> chunk_rows;
        chunk_rows.reserve(token_ids.size() * kQ38PleRowsPerToken);
        candidate_ple_hashes.reserve(candidate_ple_hashes.size() + token_ids.size());
        for (const auto token : token_ids) {
            candidate.append_rows(token, chunk_rows);
            candidate_ple_hashes.push_back(candidate);
        }
        if (ple_read_active) {
            if (ple_read_epoch != provisional_epoch || !ple_read_rows ||
                token_offset < ple_read_token_base)
                throw std::logic_error("PLE read-ahead transaction differs");
            const auto first = static_cast<std::size_t>(
                                   token_offset - ple_read_token_base) *
                               kQ38PleRowsPerToken;
            if (first > ple_read_rows->size() ||
                chunk_rows.size() > ple_read_rows->size() - first ||
                !std::equal(chunk_rows.begin(), chunk_rows.end(),
                            ple_read_rows->begin() +
                                static_cast<std::ptrdiff_t>(first)))
                throw std::logic_error("PLE read-ahead rows differ");
            return;
        }
        ple_read_rows = std::make_shared<const std::vector<std::uint64_t>>(
            std::move(chunk_rows));
        ple_read_token_base = token_offset;
        ple_read_epoch = provisional_epoch;
        ple_reader->submit(ple_read_rows);
        ple_read_active = true;
    }

    void ensure_ple_ready(std::uint32_t transaction_token_offset,
                          std::uint32_t workspace_token_offset,
                          std::uint32_t tokens) {
        if (!ple_read_active) return;
        if (!ple_read_rows ||
            transaction_token_offset < ple_read_token_base)
            throw std::logic_error("PLE read-ahead range is unavailable");
        const auto local_token_offset =
            transaction_token_offset - ple_read_token_base;
        const auto first_row =
            static_cast<std::size_t>(local_token_offset) *
            kQ38PleRowsPerToken;
        const auto row_count =
            static_cast<std::size_t>(tokens) * kQ38PleRowsPerToken;
        const auto required_rows = first_row + row_count;
        if (first_row > ple_read_rows->size() ||
            row_count > ple_read_rows->size() - first_row)
            throw std::logic_error("PLE read-ahead extent is invalid");
        const auto workspace_first_row =
            static_cast<std::size_t>(workspace_token_offset) *
            kQ38PleRowsPerToken;
        const auto byte_offset = workspace_first_row * kQ38PleRowWidth;
        const auto byte_count = row_count * kQ38PleRowWidth;
        if (workspace.ple_copy_stream && workspace.ple_copy_pending) {
            // The reader may overwrite pinned staging only after its prior
            // H2D completes. Device staging also waits for prior GPU consumers.
            check(cudaEventSynchronize(workspace.ple_copy_ready), "wait PLE host staging reuse");
            check(cudaEventRecord(workspace.ple_consumer_done, workspace.stream),
                  "record PLE device staging consumer");
            check(cudaStreamWaitEvent(workspace.ple_copy_stream, workspace.ple_consumer_done, 0),
                  "wait PLE device staging reuse");
        }
        try {
            ple_reader->take(
                first_row, row_count,
                workspace.host_ple_rows + byte_offset, byte_count,
                workspace.host_ple_scales + workspace_first_row, row_count);
        } catch (...) {
            ple_read_active = false;
            ple_read_rows.reset();
            throw;
        }
        const auto copy_stream = workspace.ple_copy_stream
                                     ? workspace.ple_copy_stream : workspace.stream;
        check(cudaMemcpyAsync(workspace.ple_rows + byte_offset,
                              workspace.host_ple_rows + byte_offset,
                              byte_count, cudaMemcpyHostToDevice,
                              copy_stream),
              "cudaMemcpyAsync(PLE row tile)");
        check(cudaMemcpyAsync(
                  workspace.ple_scales + workspace_first_row,
                  workspace.host_ple_scales + workspace_first_row,
                  row_count * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice, copy_stream),
              "cudaMemcpyAsync(PLE scale tile)");
        if (workspace.ple_copy_stream) {
            check(cudaEventRecord(workspace.ple_copy_ready, copy_stream),
                  "cudaEventRecord(PLE upload)");
            check(cudaStreamWaitEvent(workspace.stream, workspace.ple_copy_ready, 0),
                  "cudaStreamWaitEvent(PLE upload)");
            workspace.ple_copy_pending = true;
        }
        if (required_rows == ple_read_rows->size()) {
            ple_read_active = false;
            ple_read_rows.reset();
            ple_read_epoch = 0;
            ple_read_token_base = 0;
        }
    }

    void drain_ple_read() noexcept {
        if (!ple_read_active) return;
        ple_reader->cancel();
        ple_read_active = false;
        ple_read_rows.reset();
        ple_read_epoch = 0;
        ple_read_token_base = 0;
    }

    static CudaDeviceWeightStore make_weights_for_device(
        DeviceStageIndexV1 index, int device, bool enable_mtp) {
        if (index.cut != kCut || index.source_repo != kQ38OfficialSourceRepo ||
            index.source_commit != kQ38OfficialSourceCommit ||
            index.policy_sha256 != kQ38AmperePolicySha256)
            throw std::invalid_argument(
                "CUDA artifact does not match the pinned production contract");
        DeviceWeightStore host(std::move(index));
        CudaDeviceWeightOptions weight_options;
        weight_options.device = device;
        weight_options.load_mtp = enable_mtp;
        return CudaDeviceWeightStore(host, weight_options);
    }

    CudaStageBackendOptions options;
    CudaDeviceWeightStore weights;
    PrefillMatrixCache prefill_cache;
    Workspace workspace;
    std::vector<LayerWeights> layers;
    std::unique_ptr<CudaGdnStateBank> gdn_state;
    std::unique_ptr<TargetPrefixStorage> target_prefixes;
    bool capture_target_prefixes = false;
    bool target_prefixes_valid = false;
    std::uint32_t target_prefix_count = 0;
    std::unique_ptr<CudaQsaStateBank> qsa_state;
    std::unique_ptr<CudaPleStateBank> ple_state;
    std::unique_ptr<PleAsyncReader> ple_reader;
    std::unique_ptr<CudaQsaStateBank> mtp_qsa_state;
    std::shared_ptr<PleStore> ple_store;
    std::unique_ptr<PleHashState> committed_ple_hash;
    std::vector<PleHashState> candidate_ple_hashes;
    std::unique_ptr<CudaBoundaryRing> small_boundary_ring;
    std::unique_ptr<CudaBoundaryRing> prefill_boundary_ring;
    CudaMatrixViewV1 embedding;
    CudaMatrixViewV1 output;
    HyperWeights final_hyper;
    std::unique_ptr<MtpWeights> mtp;
    std::unique_ptr<CudaSamplingWorkspace> compact_sampling;
    std::vector<std::int32_t> provisional_tokens;
    std::shared_ptr<CancellationToken> provisional_cancellation;
    std::uint64_t committed_frontier = 0;
    std::uint64_t committed_epoch = 0;
    std::uint64_t provisional_epoch = 0;
    std::uint64_t provisional_base = 0;
    std::uint32_t provisional_expected = 0;
    std::uint32_t provisional_processed = 0;
    std::uint32_t speculative_checkpoint_count = 0;
    TxnKind provisional_kind = TxnKind::kInvalid;
    bool states_active = false;
    bool mtp_pending_valid = false;
    bool mtp_candidate_pending_valid = false;
    bool mtp_transaction_active = false;
    std::uint64_t mtp_transaction_epoch = 0;
    std::uint32_t mtp_transaction_rows = 0;
    std::uint32_t mtp_reconciled_rows = 0;
    bool ple_read_active = false;
    std::shared_ptr<const std::vector<std::uint64_t>> ple_read_rows;
    std::uint32_t ple_read_token_base = 0;
    std::uint64_t ple_read_epoch = 0;
    std::map<std::pair<std::uintptr_t,unsigned>,std::unique_ptr<CapturedCudaGraph>> head_graphs;
    bool capturing_head_graph=false,head_graph_warmed=false,head_graph_failed=false;
    bool capturing_profile_events = false;
    DecodeGraphVariant* profiled_batch_variant = nullptr;
    std::unique_ptr<CudaEventProfile> decode_profile;
    std::unique_ptr<CudaEventProfile> prefill_profile;
    mutable std::uint64_t tracked_peak_bytes = 0;
    StageBackendMetricsV1 metrics{};
    bool decode_graph_failed = false;
    bool batch_graph_failed = false;
    bool batch_graph_warmed = false;
    bool prefill_graph_failed = false;
    bool mtp_graph_failed = false;
    std::unique_ptr<CapturedCudaGraph> mtp_graphs[3];
    std::unique_ptr<CapturedCudaGraph> mtp_whole_graphs[6];
    bool mtp_whole_graph_failed=false;
    std::vector<std::unique_ptr<DecodeGraphVariant>> decode_graphs;
};

CudaStageBackend::CudaStageBackend(DeviceStageIndexV1 index,
                                   CudaStageBackendOptions options,
                                   std::shared_ptr<PleStore> ple_store)
    : impl_(std::make_unique<Impl>(std::move(index), options,
                                   std::move(ple_store))) {}
CudaStageBackend::~CudaStageBackend() = default;

Stage CudaStageBackend::stage() const { return impl_->options.stage; }

void CudaStageBackend::prefetch_transaction(
    const SessionTxnV1& txn,
    const std::vector<std::int32_t>& token_ids,
    std::shared_ptr<CancellationToken> cancellation) {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device),
          "cudaSetDevice(PLE transaction read-ahead)");
    state.start_ple_prefetch(txn, token_ids, cancellation);
}

StageOutput CudaStageBackend::execute(StageInput input) {
    auto& state = *impl_;
    if (input.cancellation) input.cancellation->throw_if_requested();
    check(cudaSetDevice(state.options.device), "cudaSetDevice(stage execute)");
    const auto count = static_cast<std::uint32_t>(input.token_ids.size());
    if (input.txn.magic != kTxnMagic || input.txn.version != kContractVersion ||
        input.txn.status != TxnStatus::kPrepared ||
        input.token_ids.empty() ||
        input.token_ids.size() > state.options.max_transaction_tokens)
        throw std::runtime_error("CUDA backend received invalid transaction");
    if (input.chunk_offset > input.txn.evaluated_count ||
        count > input.txn.evaluated_count - input.chunk_offset ||
        input.final_chunk !=
            (input.chunk_offset + count == input.txn.evaluated_count) ||
        input.txn.base_target + input.txn.evaluated_count >
            state.options.context_capacity)
        throw std::runtime_error("CUDA backend received invalid chunk extent");
    ++state.metrics.execute_calls;
    state.metrics.execute_tokens += count;
    const bool profile_decode =
        state.decode_profile && count == 1 &&
        input.txn.kind == TxnKind::kDecode;
    if (profile_decode)
        state.decode_profile->begin(state.workspace.stream);

    const bool first = state.provisional_epoch == 0;
    if (first) {
        if (input.txn.epoch <= state.committed_epoch ||
            input.chunk_offset != 0 ||
            input.txn.base_target != state.committed_frontier ||
            state.qsa_state->committed_tokens() != input.txn.base_target)
            throw std::runtime_error("CUDA backend frontier/epoch mismatch");
        state.provisional_epoch = input.txn.epoch;
        state.provisional_base = input.txn.base_target;
        state.provisional_expected = input.txn.evaluated_count;
        state.provisional_processed = 0;
        state.speculative_checkpoint_count = 0;
        state.target_prefixes_valid = false;
        state.capture_target_prefixes = false;
        state.target_prefix_count = 0;
        state.provisional_kind = input.txn.kind;
        state.provisional_tokens.clear();
        state.provisional_tokens.reserve(input.txn.evaluated_count);
        state.provisional_cancellation = input.cancellation;
        state.candidate_ple_hashes.clear();
        if (state.committed_ple_hash)
            state.candidate_ple_hashes.reserve(input.txn.evaluated_count);
        const bool retained_mtp_draft =
            state.options.stage == Stage::kStage1 && state.mtp &&
            input.txn.kind == TxnKind::kSpeculative &&
            state.mtp_transaction_active &&
            state.mtp_transaction_epoch == input.txn.epoch &&
            state.mtp_transaction_rows != 0 &&
            state.mtp_transaction_rows + 1 == input.txn.evaluated_count;
        if (state.mtp_transaction_active && !retained_mtp_draft)
            throw std::runtime_error(
                "retained MTP draft does not match target transaction");
        if (!retained_mtp_draft) {
            state.mtp_transaction_active = false;
            state.mtp_transaction_epoch = 0;
            state.mtp_transaction_rows = 0;
            state.mtp_reconciled_rows = 0;
        }
        state.mtp_candidate_pending_valid = false;
        state.gdn_state->begin(input.txn.epoch, state.workspace.stream);
        state.qsa_state->begin(input.txn.epoch);
        if (state.ple_state)
            state.ple_state->begin(input.txn.epoch, state.workspace.stream);
        state.states_active = true;
        if (state.options.stage == Stage::kStage1 && state.mtp &&
            input.txn.kind == TxnKind::kAppendKnown) {
            state.mtp_candidate_pending_valid = state.mtp_pending_valid;
            if (state.mtp_pending_valid)
                check(cudaMemcpyAsync(
                          state.workspace.mtp_candidate_hc,
                          state.workspace.mtp_pending_hc,
                          kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, state.workspace.stream),
                      "cudaMemcpyAsync(MTP append candidate seed)");
        }
    } else if (input.txn.epoch != state.provisional_epoch ||
               input.txn.base_target != state.provisional_base ||
               input.txn.evaluated_count != state.provisional_expected ||
               input.txn.kind != state.provisional_kind ||
               input.chunk_offset != state.provisional_processed ||
               input.cancellation != state.provisional_cancellation ||
               !state.states_active) {
        throw std::runtime_error(
            "CUDA backend chunk is not the next provisional range");
    }
    if (profile_decode) state.profile_mark("state_begin");

    if (state.options.stage == Stage::kStage1) {
        std::string error;
        if (!validate_boundary(input.boundary.frame, count, &error) ||
            input.boundary.frame.epoch != input.txn.epoch ||
            input.boundary.frame.token_start !=
                input.txn.base_target + input.chunk_offset ||
            input.boundary.frame.token_count != count ||
            input.boundary.frame.hidden_width != kQ38HyperWidth)
            throw std::runtime_error("CUDA stage1 boundary invalid: " + error);
        auto* boundary_destination = state.workspace.boundary;
        if (input.txn.kind == TxnKind::kSpeculative)
            boundary_destination +=
                static_cast<std::uint64_t>(input.chunk_offset) *
                kQ38HyperWidth;
        cuda_copy_boundary_to_device(input.boundary, boundary_destination,
                                     reinterpret_cast<void*>(
                                         state.workspace.stream),
                                     state.options.device);
    } else {
        state.start_ple_read(input.token_ids, input.chunk_offset);
    }

    state.provisional_tokens.insert(state.provisional_tokens.end(),
                                    input.token_ids.begin(),
                                    input.token_ids.end());
    try {
        auto* token_destination = state.workspace.tokens;
        if (input.txn.kind == TxnKind::kSpeculative)
            token_destination += input.chunk_offset;
        check(cudaMemcpyAsync(token_destination, input.token_ids.data(),
                              input.token_ids.size() * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice, state.workspace.stream),
              "cudaMemcpyAsync(tokens)");
        state.run_tokens(count, input.chunk_offset, true);
        if (state.options.stage == Stage::kStage1 && state.mtp &&
            input.txn.kind == TxnKind::kAppendKnown)
            state.advance_mtp_append(input.txn.epoch, input.chunk_offset,
                                     count);
        state.provisional_processed += count;

        StageOutput result;
        result.state_digest = input.txn.epoch ^ input.txn.base_target ^
                              input.chunk_offset ^ count;
        if (state.options.stage == Stage::kStage0) {
            auto& frame = result.boundary.frame;
            frame.session_hash = input.txn.session_hash;
            frame.epoch = input.txn.epoch;
            frame.token_start = input.txn.base_target + input.chunk_offset;
            frame.token_count = count;
            frame.hidden_dtype = DType::kBFloat16;
            frame.producer_status = ProducerStatus::kReady;
            frame.hidden_width = kQ38HyperWidth;
            frame.payload_bytes = static_cast<std::uint64_t>(
                                      count) *
                                  kQ38HyperWidth * sizeof(std::uint16_t);
            auto* ring = count <= kSmallBoundaryTokens
                             ? state.small_boundary_ring.get()
                             : state.prefill_boundary_ring.get();
            result.boundary.lease = ring->copy_from_device(
                state.workspace.boundary,
                static_cast<std::size_t>(count) * kQ38HyperWidth,
                state.workspace.stream);
            if (profile_decode) {
                state.profile_mark("boundary");
                state.decode_profile->finish(
                    state.options.stage,
                    input.txn.base_target + input.chunk_offset,
                    state.workspace.stream);
            }
            frame.ring_slot = result.boundary.lease->slot();
            frame.payload_checksum = result.boundary.payload_checksum();
            result.state_commit_count = state.provisional_processed;
            return result;
        }
        const auto prediction_offset =
            input.txn.kind == TxnKind::kSpeculative
                ? input.chunk_offset
                : 0u;
        check(cudaMemcpyAsync(
                  state.workspace.host_predictions + prediction_offset,
                  state.workspace.predictions + prediction_offset,
                  count * sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost, state.workspace.stream),
              "cudaMemcpyAsync(predictions)");
        if (input.need_logits) {
            if(state.output.rows*2u==kVocabulary)te_bridge_allgather_bf16(state.options.te_rank,state.workspace.logits,state.workspace.logits,state.output.rows,1,te_bridge_group_for(state.options),state.workspace.stream);
            if (!input.final_chunk ||
                input.txn.kind == TxnKind::kSpeculative)
                throw std::runtime_error(
                    "CUDA logits requested for an invalid execution lane");
            if (input.compact_top_k != 0 && state.compact_sampling) {
                state.compact_sampling->select(state.workspace.logits,
                    input.compact_top_k, input.sampling_temperature,
                    state.workspace.stream);
                result.logits_bf16.resize(input.compact_top_k);
                result.logit_token_ids.resize(input.compact_top_k);
                check(cudaMemcpyAsync(result.logits_bf16.data(),
                    state.compact_sampling->logits(),
                    input.compact_top_k * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost, state.workspace.stream),
                    "cudaMemcpyAsync(compact logits)");
                check(cudaMemcpyAsync(result.logit_token_ids.data(),
                    state.compact_sampling->ids(),
                    input.compact_top_k * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost, state.workspace.stream),
                    "cudaMemcpyAsync(compact logit IDs)");
            } else {
                result.logits_bf16.resize(kVocabulary);
                check(cudaMemcpyAsync(result.logits_bf16.data(),
                    state.workspace.logits, kVocabulary * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost, state.workspace.stream),
                    "cudaMemcpyAsync(final logits)");
            }
        }
        if (profile_decode) state.profile_mark("result_copy");
        check(cudaStreamSynchronize(state.workspace.stream),
              "cudaStreamSynchronize(stage1 execute)");
        if (profile_decode)
            state.decode_profile->finish(
                state.options.stage,
                input.txn.base_target + input.chunk_offset,
                state.workspace.stream);
        std::uint32_t accepted = count;
        if (input.txn.kind == TxnKind::kSpeculative) {
            result.target_predictions.assign(
                state.workspace.host_predictions + prediction_offset,
                state.workspace.host_predictions + prediction_offset + count);
            if (!input.final_chunk) {
                result.state_commit_count = state.provisional_processed;
                result.next_token = state.workspace.host_predictions[
                    state.provisional_processed - 1];
                return result;
            }
            accepted = 1;
            while (accepted < state.provisional_processed &&
                   state.provisional_tokens[accepted] ==
                       state.workspace.host_predictions[accepted - 1])
                ++accepted;
        }
        result.state_commit_count =
            input.txn.kind == TxnKind::kSpeculative
                ? accepted
                : state.provisional_processed;
        result.next_token = state.workspace.host_predictions[
            input.txn.kind == TxnKind::kSpeculative ? accepted - 1
                                                     : count - 1];
        if (state.options.te_world_size == 2 &&
            state.options.stage == Stage::kStage1) {
            if (std::getenv("Q38_TE_DEBUG")) std::fprintf(stderr,
                "te-debug rank=%d next_token=%d count=%d pred0=%d vocab=%u "
                "output_cols=%u mixed_te=%d\n",
                state.options.te_rank, result.next_token, count,
                state.workspace.host_predictions[0], kVocabulary,
                state.output.columns,
                state.output.columns * 2u == kQ38HiddenWidth ? 1 : 0);
            std::fflush(stderr);
        }
        return result;
    } catch (...) {
        if (state.options.te_world_size == 2) te_bridge_abort();
        state.drain_ple_read();
        throw;
    }
}

std::vector<std::int32_t> CudaStageBackend::draft(
    std::int32_t pending_token, std::uint64_t position,
    std::uint32_t max_draft,
    std::shared_ptr<CancellationToken> cancellation) {
    auto& state = *impl_;
    if (max_draft == 0) return {};
    return state.run_mtp_draft(pending_token, position, max_draft, 0,
                               cancellation);
}

std::vector<std::int32_t> CudaStageBackend::draft_retained(
    std::int32_t pending_token, std::uint64_t position,
    std::uint32_t max_draft, std::uint64_t transaction_epoch,
    std::shared_ptr<CancellationToken> cancellation) {
    if (max_draft == 0 || transaction_epoch == 0)
        throw std::invalid_argument("retained MTP draft extent is invalid");
    return impl_->run_mtp_draft(pending_token, position, max_draft,
                                transaction_epoch, cancellation);
}

void CudaStageBackend::checkpoint_speculative_prefix(
    std::uint64_t transaction_epoch, std::uint32_t prefix_tokens) {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device),
          "cudaSetDevice(speculative prefix checkpoint)");
    if (state.options.stage != Stage::kStage0 ||
        state.provisional_epoch != transaction_epoch ||
        state.provisional_kind != TxnKind::kSpeculative ||
        !state.states_active || prefix_tokens == 0 ||
        prefix_tokens != state.provisional_processed ||
        prefix_tokens >= state.provisional_expected)
        throw std::runtime_error("speculative prefix checkpoint is invalid");
    state.gdn_state->checkpoint(state.workspace.stream);
    if (state.ple_state)
        state.ple_state->checkpoint(state.workspace.stream);
    state.speculative_checkpoint_count = prefix_tokens;
}

void CudaStageBackend::reconcile_retained_draft(
    std::uint64_t transaction_epoch, std::uint32_t target_row) {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device),
          "cudaSetDevice(reconcile retained MTP draft)");
    if (state.options.stage != Stage::kStage1 || !state.mtp ||
        !state.mtp_qsa_state || !state.mtp_transaction_active ||
        state.mtp_transaction_epoch != transaction_epoch ||
        state.provisional_epoch != transaction_epoch ||
        state.provisional_kind != TxnKind::kSpeculative ||
        target_row >= state.mtp_transaction_rows ||
        state.provisional_processed != target_row + 1 ||
        state.mtp_reconciled_rows != target_row + 1)
        throw std::runtime_error("retained MTP reconciliation is invalid");
    state.check_cancelled();
    const auto prior_pairs = state.mtp_qsa_state->committed_tokens();
    state.run_mtp_step(
        state.workspace.final_hc +
            static_cast<std::uint64_t>(target_row) * kQ38HyperWidth,
        state.workspace.predictions + target_row,
        static_cast<std::uint32_t>(prior_pairs + target_row + 1), false);
    state.mtp_qsa_state->mark_evaluated(transaction_epoch, target_row + 2);
    state.mtp_reconciled_rows = target_row + 2;
}

void CudaStageBackend::abandon_retained_draft(
    std::uint64_t transaction_epoch) {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device),
          "cudaSetDevice(abandon MTP draft)");
    if (!state.mtp_transaction_active) return;
    if (state.mtp_transaction_epoch != transaction_epoch ||
        state.mtp_transaction_rows == 0)
        throw std::runtime_error("retained MTP draft epoch mismatch");
    state.mtp_qsa_state->rollback(state.mtp_transaction_epoch);
    state.mtp_transaction_active = false;
    state.mtp_transaction_epoch = 0;
    state.mtp_transaction_rows = 0;
    state.mtp_reconciled_rows = 0;
}

void CudaStageBackend::commit(std::uint64_t epoch,
                              std::uint32_t state_commit_count) {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device), "cudaSetDevice(stage commit)");
    const bool complete =
        state.provisional_processed == state.provisional_expected;
    const bool online_exact =
        state.provisional_kind == TxnKind::kSpeculative &&
        state.provisional_processed == state_commit_count &&
        state.provisional_processed < state.provisional_expected;
    const bool online_lookahead =
        state.options.stage == Stage::kStage0 &&
        state.provisional_kind == TxnKind::kSpeculative &&
        state.provisional_processed == state_commit_count + 1 &&
        state.speculative_checkpoint_count == state_commit_count;
    if (epoch != state.provisional_epoch || !state.states_active ||
        state_commit_count == 0 ||
        state_commit_count > state.provisional_processed ||
        (!complete && !online_exact && !online_lookahead))
        throw std::runtime_error("CUDA backend commit does not match transaction");
    if (state_commit_count < state.provisional_processed) {
        if (state.target_prefixes_valid && state_commit_count <= state.target_prefix_count) {
            const auto* prefix = state.target_prefixes->data +
                (state_commit_count - 1) * state.target_prefixes->stride;
            const auto layer_bytes = state.gdn_state->bytes_per_bank() / state.gdn_state->layers();
            for (std::uint32_t layer = 0; layer < state.gdn_state->layers(); ++layer)
                check(cudaMemcpyAsync(state.gdn_state->working(layer).conv,
                    prefix + layer * layer_bytes, layer_bytes,
                    cudaMemcpyDeviceToDevice, state.workspace.stream),
                    "restore evaluated GDN prefix");
            if (state.ple_state)
                check(cudaMemcpyAsync(state.ple_state->working(),
                    prefix + state.target_prefixes->gdn_bytes, state.target_prefixes->ple_bytes,
                    cudaMemcpyDeviceToDevice, state.workspace.stream),
                    "restore evaluated PLE prefix");
            check(cudaStreamSynchronize(state.workspace.stream), "target prefix restore ready");
            if (profile_requested("Q38_TP_PREFIX_DIAGNOSTICS"))
                std::fprintf(stderr, "tp-prefix restore rank=%d stage=%u epoch=%llu count=%u\n",
                    state.options.te_rank, unsigned(state.options.stage),
                    (unsigned long long)epoch, state_commit_count);
        } else if (online_lookahead) {
            // Stage0 may be exactly one row ahead when stage1 observes the
            // first mismatch. Restore the prefix captured before that row;
            // QSA is append-only and commits the shorter extent directly.
            state.gdn_state->restore_checkpoint(state.workspace.stream);
            if (state.ple_state)
                state.ple_state->restore_checkpoint(state.workspace.stream);
            check(cudaStreamSynchronize(state.workspace.stream),
                  "cudaStreamSynchronize(speculative checkpoint restore)");
        } else {
            // Non-retained speculative callers still use the conservative
            // replay path until they adopt online prefix checkpoints.
            state.gdn_state->restore(state.workspace.stream);
            if (state.ple_state)
                state.ple_state->restore(state.workspace.stream);
            if (state.options.stage == Stage::kStage0 && state.ple_reader) {
                state.drain_ple_read();
                state.candidate_ple_hashes.clear();
                state.start_ple_read(std::vector<std::int32_t>(
                    state.provisional_tokens.begin(),
                    state.provisional_tokens.begin() + state_commit_count),
                    0);
            }
            state.run_tokens(state_commit_count, 0, false);
            check(cudaStreamSynchronize(state.workspace.stream),
                  "cudaStreamSynchronize(partial replay)");
        }
    }
    if (state.options.stage == Stage::kStage1 && state.mtp) {
        if (state.provisional_kind == TxnKind::kAppendKnown) {
            if (state_commit_count != state.provisional_expected ||
                !state.mtp_candidate_pending_valid)
                throw std::runtime_error(
                    "chunked append MTP state is incomplete");
        } else {
            state.prepare_mtp_commit(epoch, state_commit_count);
        }
    }
    state.gdn_state->commit(epoch);
    state.qsa_state->commit(epoch, state_commit_count);
    if (state.ple_state) {
        state.ple_state->commit(epoch);
        *state.committed_ple_hash =
            state.candidate_ple_hashes[state_commit_count - 1];
    }
    if (state.mtp_transaction_active) {
        state.mtp_qsa_state->commit(state.mtp_transaction_epoch,
                                    state.mtp_transaction_rows);
        state.mtp_transaction_active = false;
        state.mtp_transaction_epoch = 0;
        state.mtp_transaction_rows = 0;
        state.mtp_reconciled_rows = 0;
    }
    if (state.options.stage == Stage::kStage1 && state.mtp) {
        check(cudaMemcpyAsync(state.workspace.mtp_pending_hc,
                              state.workspace.mtp_candidate_hc,
                              kQ38HyperWidth * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice,
                              state.workspace.stream),
              "cudaMemcpyAsync(commit MTP pending HC)");
        check(cudaStreamSynchronize(state.workspace.stream),
              "cudaStreamSynchronize(commit MTP pending HC)");
        state.mtp_pending_valid = state.mtp_candidate_pending_valid;
    }
    state.committed_frontier = state.provisional_base + state_commit_count;
    state.committed_epoch = epoch;
    state.provisional_epoch = 0;
    state.provisional_expected = 0;
    state.provisional_processed = 0;
    state.speculative_checkpoint_count = 0;
    state.provisional_kind = TxnKind::kInvalid;
    state.provisional_tokens.clear();
    state.provisional_cancellation.reset();
    state.candidate_ple_hashes.clear();
    state.states_active = false;
    state.mtp_candidate_pending_valid = false;
    ++state.metrics.commits;
}

void CudaStageBackend::rollback(std::uint64_t epoch) {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device), "cudaSetDevice(stage rollback)");
    if (state.provisional_epoch == 0) {
        if (state.mtp_transaction_active) {
            if (state.mtp_transaction_epoch != epoch)
                throw std::runtime_error(
                    "retained MTP rollback epoch mismatch");
            state.mtp_qsa_state->rollback(state.mtp_transaction_epoch);
            state.mtp_transaction_active = false;
            state.mtp_transaction_epoch = 0;
            state.mtp_transaction_rows = 0;
        }
        if (epoch <= state.committed_epoch)
            throw std::runtime_error("CUDA backend rollback epoch is stale");
        state.committed_epoch = epoch;
        ++state.metrics.rollbacks;
        return;
    }
    if (epoch != state.provisional_epoch)
        throw std::runtime_error("CUDA backend rollback epoch mismatch");
    if (state.states_active) {
        state.gdn_state->rollback(epoch);
        state.qsa_state->rollback(epoch);
        if (state.ple_state) state.ple_state->rollback(epoch);
    }
    if (state.mtp_transaction_active) {
        state.mtp_qsa_state->rollback(state.mtp_transaction_epoch);
        state.mtp_transaction_active = false;
        state.mtp_transaction_epoch = 0;
        state.mtp_transaction_rows = 0;
    }
    state.committed_epoch = epoch;
    state.provisional_epoch = 0;
    state.provisional_expected = 0;
    state.provisional_processed = 0;
    state.speculative_checkpoint_count = 0;
    state.provisional_kind = TxnKind::kInvalid;
    state.provisional_tokens.clear();
    state.provisional_cancellation.reset();
    state.candidate_ple_hashes.clear();
    state.states_active = false;
    state.mtp_candidate_pending_valid = false;
    ++state.metrics.rollbacks;
}

void CudaStageBackend::reset_session() {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device), "cudaSetDevice(stage reset)");
    if (state.provisional_epoch != 0 || state.states_active ||
        state.mtp_transaction_active)
        throw std::logic_error("cannot reset an active CUDA transaction");

    state.drain_ple_read();
    auto* stream = reinterpret_cast<void*>(state.workspace.stream);
    state.gdn_state->reset(stream);
    state.qsa_state->reset();
    if (state.ple_state) state.ple_state->reset(stream);
    if (state.committed_ple_hash) state.committed_ple_hash->reset();
    if (state.mtp_qsa_state) state.mtp_qsa_state->reset();
    check(cudaStreamSynchronize(state.workspace.stream),
          "cudaStreamSynchronize(stage reset)");

    state.committed_frontier = 0;
    state.committed_epoch = 0;
    state.provisional_base = 0;
    state.provisional_expected = 0;
    state.provisional_processed = 0;
    state.provisional_kind = TxnKind::kInvalid;
    state.provisional_tokens.clear();
    state.provisional_cancellation.reset();
    state.candidate_ple_hashes.clear();
    state.mtp_pending_valid = false;
    state.mtp_candidate_pending_valid = false;
    state.mtp_transaction_epoch = 0;
    state.mtp_transaction_rows = 0;
    state.mtp_reconciled_rows = 0;
}

StageBackendMetricsV1 CudaStageBackend::metrics() const {
    const auto& state = *impl_;
    auto result = state.metrics;
    const auto weight_stats = state.weights.stats();
    result.weight_arena_bytes = weight_stats.arena_bytes;
    result.weight_uploaded_bytes = weight_stats.uploaded_bytes;
    result.weight_host_only_bytes = weight_stats.host_only_bytes;
    result.weight_excluded_bytes = weight_stats.excluded_bytes;
    result.weight_staging_peak_pinned_bytes =
        weight_stats.staging_peak_pinned_bytes;
    result.weight_w4_bytes = weight_stats.w4_bytes;
    result.weight_w8_bytes = weight_stats.w8_bytes;
    result.weight_preserved_bf16_bytes =
        weight_stats.preserved_bf16_bytes;
    result.weight_preserved_f32_bytes = weight_stats.preserved_f32_bytes;
    result.weight_preserved_other_bytes =
        weight_stats.preserved_other_bytes;
    result.qsa_state_bytes = state.qsa_state->allocated_bytes();
    result.gdn_state_bytes = state.gdn_state->allocated_bytes() +
        (state.target_prefixes ? state.target_prefixes->gdn_bytes * TargetPrefixStorage::capacity : 0);
    result.ple_state_bytes =
        (state.ple_state ? state.ple_state->allocated_bytes() : 0) +
        (state.target_prefixes ? state.target_prefixes->ple_bytes * TargetPrefixStorage::capacity : 0);
    result.mtp_state_bytes = state.workspace.mtp_state_bytes +
        (state.mtp_qsa_state ? state.mtp_qsa_state->allocated_bytes() : 0);
    result.workspace_device_bytes =
        state.workspace.device_bytes - state.workspace.mtp_state_bytes +
        (state.compact_sampling ? state.compact_sampling->bytes() : 0);
    result.prefill_cache_device_bytes = state.prefill_cache.resident_bytes();
    result.cuda_graph_held_bytes = 0;
    for (const auto& variant : state.decode_graphs)
        for (const auto& fragment : variant->fragments)
            result.cuda_graph_held_bytes += fragment->held_bytes();
    for (const auto& graph : state.mtp_graphs)
        if (graph) result.cuda_graph_held_bytes += graph->held_bytes();
    for (const auto& graph : state.mtp_whole_graphs)
        if (graph) result.cuda_graph_held_bytes += graph->held_bytes();
    result.workspace_pinned_bytes = state.workspace.pinned_bytes;
    result.cuda_allocation_failures =
        state.prefill_cache.allocation_failures();
    if (state.options.stage == Stage::kStage0) {
        const auto small = state.small_boundary_ring->stats();
        const auto prefill = state.prefill_boundary_ring->stats();
        result.boundary_transfers = small.transfers + prefill.transfers;
        result.boundary_bytes = small.bytes + prefill.bytes;
        result.boundary_waits = small.waits + prefill.waits;
        result.boundary_pinned_bytes =
            state.small_boundary_ring->pinned_bytes() +
            state.prefill_boundary_ring->pinned_bytes();
        result.workspace_device_bytes +=
            state.small_boundary_ring->device_bytes() +
            state.prefill_boundary_ring->device_bytes();
        if (state.ple_store) {
            const auto cache = state.ple_store->cache_stats();
            result.ple_cache_hits = cache.hits;
            result.ple_cache_misses = cache.misses;
            result.ple_cache_evictions = cache.evictions;
            result.ple_cache_resident_bytes = cache.resident_bytes;
            result.ple_cache_capacity_bytes = cache.capacity_bytes;
            result.ple_requested_rows = cache.requested_rows;
            result.ple_unique_page_requests = cache.unique_page_requests;
            result.ple_useful_bytes = cache.useful_bytes;
            result.ple_scale_resident_bytes = cache.scale_resident_bytes;
            result.ple_physical_read_bytes = cache.physical_read_bytes;
            result.ple_read_operations = cache.read_operations;
            result.ple_read_batches = cache.read_batches;
            result.ple_io_uring_submissions = cache.io_uring_submissions;
            result.ple_io_uring_completions = cache.io_uring_completions;
            result.ple_direct_read_bytes = cache.direct_read_bytes;
            result.ple_read_errors = cache.read_errors;
            result.ple_maximum_queue_depth = cache.maximum_queue_depth;
            result.ple_read_latency_p50_ns = cache.read_latency_p50_ns;
            result.ple_read_latency_p95_ns = cache.read_latency_p95_ns;
            result.ple_read_latency_p99_ns = cache.read_latency_p99_ns;
            result.ple_io_uring_enabled = cache.io_uring_enabled;
            result.ple_direct_io_enabled = cache.direct_io_enabled;
            result.ple_ram_enabled = cache.ram_enabled;
            result.ple_ram_resident_bytes = cache.ram_resident_bytes;
        }
    }
    result.cuda_tracked_allocated_bytes =
        result.weight_arena_bytes + result.qsa_state_bytes +
        result.gdn_state_bytes + result.ple_state_bytes +
        result.mtp_state_bytes + result.workspace_device_bytes +
        result.prefill_cache_device_bytes + result.cuda_graph_held_bytes;
    state.tracked_peak_bytes =
        std::max(state.tracked_peak_bytes,
                 result.cuda_tracked_allocated_bytes);
    result.cuda_tracked_peak_bytes = state.tracked_peak_bytes;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaSetDevice(state.options.device) == cudaSuccess &&
        cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        result.cuda_device_free_bytes = free_bytes;
        result.cuda_device_total_bytes = total_bytes;
    } else {
        (void)cudaGetLastError();
    }
    return result;
}

std::unique_ptr<DualStageExecutor> make_cuda_executor(
    const std::string& stage0_path, const std::string& stage1_path,
    const std::string& ple_path, const ExecutorOptions& executor_options,
    int stage0_device, int stage1_device, std::uint64_t ple_cache_bytes,
    PleIoModeV1 ple_io_mode, std::uint32_t ple_queue_depth,
    bool enable_mtp, bool enable_piecewise_decode_graph,
    int te_world_size, int te_rank, std::uint64_t ple_ram_budget_bytes) {
    std::string identity_error;
    if (!validate_session_identity(executor_options.identity,
                                   &identity_error) ||
        (executor_options.identity.flags &
         kSessionIdentityDevelopment) != 0)
        throw std::invalid_argument(
            "CUDA runtime requires a production session identity: " +
            identity_error);
    auto stage0_index = load_device_stage_index(stage0_path);
    auto stage1_index = load_device_stage_index(stage1_path);
    validate_device_stage_pair(stage0_index, stage1_index);
    PleStoreOptionsV1 ple_options;
    ple_options.cache_bytes = ple_cache_bytes;
    ple_options.io_mode = ple_io_mode;
    ple_options.queue_depth = ple_queue_depth;
    ple_options.ram_budget_bytes = ple_ram_budget_bytes;
    auto ple = std::make_shared<PleStore>(load_ple_layout(ple_path),
                                          ple_options);
    CudaStageBackendOptions options0;
    options0.stage = Stage::kStage0;
    options0.device = stage0_device;
    options0.max_transaction_tokens = executor_options.append_chunk_tokens;
    options0.context_capacity = executor_options.context_limit;
    options0.enable_speculative_checkpoint = enable_mtp;
    options0.enable_piecewise_decode_graph = enable_piecewise_decode_graph;
    options0.enable_compact_sampling = executor_options.enable_compact_sampling;
    options0.te_world_size = te_world_size;
    options0.te_rank = te_rank;
    CudaStageBackendOptions options1 = options0;
    options1.stage = Stage::kStage1;
    options1.device = stage1_device;
    options1.enable_mtp = enable_mtp;
    options1.enable_speculative_checkpoint = false;
    options1.te_world_size = te_world_size;
    options1.te_rank = te_rank;
    auto production_executor_options = executor_options;
    production_executor_options.backend_failure_is_fatal = true;
    return std::make_unique<DualStageExecutor>(
        std::move(production_executor_options),
        std::make_unique<CudaStageBackend>(std::move(stage0_index), options0,
                                           ple),
        std::make_unique<CudaStageBackend>(std::move(stage1_index), options1));
}

bool cuda_q38_backend_compiled() { return true; }

}  // namespace q38
