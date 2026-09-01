#include "q38/cuda_backend.h"

#include "q38/cuda_gdn.h"
#include "q38/cuda_hyper.h"
#include "q38/cuda_kernels.h"
#include "q38/cuda_moe.h"
#include "q38/cuda_moe_prefill.h"
#include "q38/cuda_ple.h"
#include "q38/cuda_qsa.h"
#include "q38/cuda_transport.h"
#include "q38/cuda_weights.h"
#include "q38/production_contract.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
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

bool profile_requested(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0;
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

// Opt-in event profiling for real-model batch-1 decode.  It has no CUDA
// events, records, synchronization, or logging cost unless explicitly enabled.
// Repeated labels are aggregated across all layers owned by the stage.
class CudaEventProfile {
public:
    CudaEventProfile(int value_device, std::string value_type)
        : device_(value_device), type_(std::move(value_type)) {}
    ~CudaEventProfile() {
        (void)cudaSetDevice(device_);
        for (auto event : events_) (void)cudaEventDestroy(event);
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

    void finish(Stage stage, std::uint64_t position, cudaStream_t stream) {
        if (!active_) return;
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize(decode profile)");
        std::vector<std::pair<std::string, double>> totals;
        double total_ms = 0.0;
        for (std::size_t index = 0; index < labels_.size(); ++index) {
            float elapsed_ms = 0.0f;
            check(cudaEventElapsedTime(&elapsed_ms, events_[index],
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
        std::cerr << "{\"type\":\"q38_cuda_" << type_
                  << "_profile\",\"stage\":"
                  << static_cast<unsigned>(stage) << ",\"position\":"
                  << position << ",\"total_ms\":" << total_ms
                  << ",\"categories_ms\":{";
        for (std::size_t index = 0; index < totals.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << '\"' << totals[index].first << "\":"
                      << totals[index].second;
        }
        std::cerr << "}}\n";
        active_ = false;
    }

    bool active() const { return active_; }

private:
    void record(cudaStream_t stream) {
        if (event_count_ == events_.size()) {
            cudaEvent_t event = nullptr;
            check(cudaEventCreate(&event),
                  "cudaEventCreate(decode profile)");
            events_.push_back(event);
        }
        check(cudaEventRecord(events_[event_count_++], stream),
              "cudaEventRecord(decode profile)");
    }

    int device_ = 0;
    std::string type_;
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
                      const std::function<void()>& enqueue)
        : device_(value_device) {
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

    void launch(cudaStream_t stream) const {
        if (!exec_) throw std::logic_error("CUDA decode graph is unavailable");
        check(cudaGraphLaunch(exec_, stream),
              "cudaGraphLaunch(decode fragment)");
    }

    std::size_t nodes() const { return nodes_; }

private:
    void release() noexcept {
        (void)cudaSetDevice(device_);
        if (exec_) (void)cudaGraphExecDestroy(exec_);
        if (graph_) (void)cudaGraphDestroy(graph_);
        exec_ = nullptr;
        graph_ = nullptr;
        nodes_ = 0;
    }

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

CudaMatrixViewV1 require_matrix(const CudaDeviceWeightStore& weights,
                                const std::string& name,
                                std::uint32_t rows,
                                std::uint32_t columns) {
    auto result = cuda_matrix_view(weights.require(name));
    if (result.rows != rows || result.columns != columns)
        throw std::runtime_error("invalid CUDA matrix dimensions " + name);
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
    CudaMatrixViewV1 z;
    CudaMatrixViewV1 b;
    CudaMatrixViewV1 a;
    CudaMatrixViewV1 output;
    const std::uint16_t* conv = nullptr;
    const std::uint16_t* a_log = nullptr;
    const std::uint16_t* dt_bias = nullptr;
    const std::uint16_t* norm = nullptr;
};

struct QsaWeights {
    CudaMatrixViewV1 index_qk;
    CudaMatrixViewV1 q_gate;
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
class PleAsyncReader {
public:
    explicit PleAsyncReader(std::shared_ptr<PleStore> value_store)
        : store_(std::move(value_store)) {
        if (!store_) throw std::invalid_argument("PLE async store is null");
        worker_ = std::thread([this] { run(); });
    }

    ~PleAsyncReader() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            cancel_requested_ = true;
        }
        ready_.notify_all();
        finished_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    PleAsyncReader(const PleAsyncReader&) = delete;
    PleAsyncReader& operator=(const PleAsyncReader&) = delete;

    void submit(std::shared_ptr<const std::vector<std::uint64_t>> rows) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!rows || rows->empty())
            throw std::invalid_argument("PLE async row set is empty");
        if (request_ready_ || working_ || result_ready_ || stopping_)
            throw std::logic_error("PLE async reader already has work");
        rows_ = std::move(rows);
        error_ = nullptr;
        total_rows_ = rows_->size();
        cancel_requested_ = false;
        request_ready_ = true;
        ready_.notify_one();
    }

    void take(std::size_t first_row, std::size_t row_count,
              std::uint8_t* output, std::size_t output_bytes,
              std::uint16_t* scale_output,
              std::size_t output_scales) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto row_bytes = store_->layout().row_stride_bytes;
        if (row_count == 0 || first_row > total_rows_ ||
            row_count > total_rows_ - first_row || !output ||
            output_bytes != row_count * row_bytes || !scale_output ||
            output_scales != row_count)
            throw std::invalid_argument("PLE async take extent is invalid");
        finished_.wait(lock, [this, first_row] {
            return (!ready_tiles_.empty() &&
                    ready_tiles_.front().first_row == first_row) ||
                   result_ready_ || stopping_;
        });
        if (ready_tiles_.empty() ||
            ready_tiles_.front().first_row != first_row) {
            auto error = error_;
            if (result_ready_) {
                result_ready_ = false;
                error_ = nullptr;
            }
            lock.unlock();
            if (error) std::rethrow_exception(error);
            throw std::runtime_error("PLE async reader stopped early");
        }
        auto tile = std::move(ready_tiles_.front());
        ready_tiles_.pop_front();
        ready_.notify_one();
        if (tile.rows != row_count || tile.bytes.size() != output_bytes ||
            tile.scales.size() != output_scales)
            throw std::logic_error("PLE async tile extent differs");
        const bool final = first_row + row_count == total_rows_;
        if (final) {
            finished_.wait(lock, [this] { return result_ready_ || stopping_; });
            const auto error = error_;
            result_ready_ = false;
            error_ = nullptr;
            if (error) std::rethrow_exception(error);
        }
        lock.unlock();
        std::copy(tile.bytes.begin(), tile.bytes.end(), output);
        std::copy(tile.scales.begin(), tile.scales.end(), scale_output);
    }

    void cancel() noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!request_ready_ && !working_ && !result_ready_) return;
        cancel_requested_ = true;
        ready_.notify_all();
        finished_.notify_all();
        finished_.wait(lock, [this] {
            return result_ready_ ||
                   (!request_ready_ && !working_) || stopping_;
        });
        result_ready_ = false;
        error_ = nullptr;
        rows_.reset();
        ready_tiles_.clear();
        total_rows_ = 0;
        cancel_requested_ = false;
    }

private:
    struct PleTileResult {
        std::size_t first_row = 0;
        std::size_t rows = 0;
        std::vector<std::uint8_t> bytes;
        std::vector<std::uint16_t> scales;
    };

    void run() noexcept {
        for (;;) {
            std::shared_ptr<const std::vector<std::uint64_t>> rows;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock,
                            [this] { return stopping_ || request_ready_; });
                if (stopping_ && !request_ready_) return;
                rows = std::move(rows_);
                request_ready_ = false;
                working_ = true;
            }
            std::exception_ptr error;
            try {
                if (!rows || rows->empty())
                    throw std::invalid_argument("PLE async row set vanished");
                constexpr std::size_t kRowsPerTile =
                    static_cast<std::size_t>(kPrefillTile) *
                    kQ38PleRowsPerToken;
                for (std::size_t first = 0; first < rows->size();
                     first += kRowsPerTile) {
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        ready_.wait(lock, [this] {
                            return ready_tiles_.size() < 2 ||
                                   cancel_requested_ || stopping_;
                        });
                        if (cancel_requested_ || stopping_) break;
                    }
                    const auto count =
                        std::min<std::size_t>(kRowsPerTile,
                                              rows->size() - first);
                    std::vector<std::uint64_t> tile(
                        rows->begin() + static_cast<std::ptrdiff_t>(first),
                        rows->begin() +
                            static_cast<std::ptrdiff_t>(first + count));
                    PleTileResult result;
                    result.first_row = first;
                    result.rows = count;
                    result.bytes = store_->read_rows(tile);
                    result.scales.resize(count);
                    store_->read_row_scales_into(
                        tile, result.scales.data(), result.scales.size());
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        ready_tiles_.push_back(std::move(result));
                        finished_.notify_all();
                    }
                }
            } catch (...) {
                error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                working_ = false;
                result_ready_ = true;
                error_ = error;
            }
            finished_.notify_one();
        }
    }

    std::shared_ptr<PleStore> store_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::shared_ptr<const std::vector<std::uint64_t>> rows_;
    std::deque<PleTileResult> ready_tiles_;
    std::exception_ptr error_;
    std::size_t total_rows_ = 0;
    bool request_ready_ = false;
    bool working_ = false;
    bool result_ready_ = false;
    bool cancel_requested_ = false;
    bool stopping_ = false;
};

class PrefillMatrixCache {
public:
    PrefillMatrixCache(int value_device, std::uint64_t value_max_bytes)
        : device_(value_device), max_bytes_(value_max_bytes) {
        check(cudaSetDevice(device_), "cudaSetDevice(prefill cache)");
        check_cublas(cublasCreate(&handle_), "cublasCreate(prefill)");
        check_cublas(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH),
                     "cublasSetMathMode(prefill)");
    }

    ~PrefillMatrixCache() {
        (void)cudaSetDevice(device_);
        // Workspace is declared after this cache and therefore drains its
        // stream first.  The synchronize is a defensive guard for constructor
        // failure and future alternate call sites.
        (void)cudaDeviceSynchronize();
        for (auto& item : matrices_) (void)cudaFree(item.second);
        if (handle_) (void)cublasDestroy(handle_);
    }

    PrefillMatrixCache(const PrefillMatrixCache&) = delete;
    PrefillMatrixCache& operator=(const PrefillMatrixCache&) = delete;

    std::uint64_t resident_bytes() const { return resident_bytes_; }
    std::uint64_t allocation_failures() const {
        return allocation_failures_;
    }

    bool gemm(const CudaMatrixViewV1& matrix, const std::uint16_t* input,
              std::uint16_t* output, std::uint32_t batch,
              cudaStream_t stream) {
        if (!input || !output || batch < 2 || !stream)
            throw std::invalid_argument("invalid prefill GEMM buffers");
        const auto* expanded = require(matrix, stream);
        if (!expanded) return false;
        check_cublas(cublasSetStream(handle_, stream),
                     "cublasSetStream(prefill)");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Row-major output [batch, rows] is column-major [rows, batch].
        // Row-major W [rows, columns] is column-major [columns, rows], so
        // transpose that view and multiply by input's [columns, batch] view.
        check_cublas(
            cublasGemmEx(
                handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(matrix.rows), static_cast<int>(batch),
                static_cast<int>(matrix.columns), &alpha, expanded,
                CUDA_R_16BF, static_cast<int>(matrix.columns), input,
                CUDA_R_16BF, static_cast<int>(matrix.columns), &beta, output,
                CUDA_R_16BF, static_cast<int>(matrix.rows),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx(prefill)");
        return true;
    }

private:
    const std::uint16_t* require(const CudaMatrixViewV1& matrix,
                                 cudaStream_t stream) {
        if (matrix.format == DeviceWeightFormatV1::kPreserve &&
            !matrix.preserved_f32)
            return static_cast<const std::uint16_t*>(matrix.data);
        const auto found = matrices_.find(matrix.data);
        if (found != matrices_.end()) return found->second;
        if (disabled_.count(matrix.data) != 0) return nullptr;
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
    std::unordered_map<const void*, std::uint16_t*> matrices_;
    std::unordered_set<const void*> disabled_;
    std::uint64_t resident_bytes_ = 0;
    std::uint64_t max_bytes_ = 0;
    std::uint64_t allocation_failures_ = 0;
};

class Workspace {
public:
    Workspace(int value_device, Stage stage, std::uint32_t max_tokens,
              bool enable_mtp)
        : device(value_device), maximum_tokens(max_tokens) {
        check(cudaSetDevice(device), "cudaSetDevice(workspace)");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreate(stage)");
        try {
            tokens = allocate<std::int32_t>(max_tokens);
            predictions = allocate<std::int32_t>(max_tokens);
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
            }
            selected_indices = allocate<std::int32_t>(
                static_cast<std::uint64_t>(kPrefillTile) *
                kQ38QsaMaximumSelected);
            block_scores = allocate<float>(
                static_cast<std::uint64_t>(kPrefillTile) *
                (kQ38ContextLimit / kQ38QsaBlockTokens));
            attention_scores = allocate<float>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38QsaHeads *
                kQ38QsaMaximumSelected);
            if (stage == Stage::kStage1)
                logits = allocate<std::uint16_t>(kVocabulary);
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
        allocations.push_back(result);
        device_bytes += bytes;
        return result;
    }

    void release() noexcept {
        (void)cudaSetDevice(device);
        if (stream) (void)cudaStreamSynchronize(stream);
        if (host_predictions) (void)cudaFreeHost(host_predictions);
        if (host_ple_rows) (void)cudaFreeHost(host_ple_rows);
        if (host_ple_scales) (void)cudaFreeHost(host_ple_scales);
        for (auto* allocation : allocations) (void)cudaFree(allocation);
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
    std::vector<void*> allocations;
    std::int32_t* tokens = nullptr;
    std::int32_t* predictions = nullptr;
    std::int32_t* host_predictions = nullptr;
    std::uint8_t* ple_rows = nullptr;
    std::uint8_t* host_ple_rows = nullptr;
    std::uint16_t* ple_scales = nullptr;
    std::uint16_t* host_ple_scales = nullptr;
    std::uint16_t* boundary = nullptr;
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
    std::uint16_t* moe_gate_up = nullptr;
    std::uint16_t* moe_activated = nullptr;
    float* moe_accumulation = nullptr;
    std::uint8_t* moe_route_plan_allocation = nullptr;
    Q38RoutePlanStorageV1 moe_route_plan;
    std::uint16_t* moe_packed_hidden = nullptr;
    std::uint16_t* moe_weighted_mid = nullptr;
    float* moe_route_output = nullptr;
    std::int32_t* selected_indices = nullptr;
    float* block_scores = nullptr;
    float* attention_scores = nullptr;
    std::uint16_t* logits = nullptr;
    std::uint16_t* final_hc = nullptr;
    std::uint16_t* mtp_pending_hc = nullptr;
    std::uint16_t* mtp_candidate_hc = nullptr;
};

}  // namespace

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
    };

    struct DecodeStaticAction {
        DecodeStaticActionKind kind = DecodeStaticActionKind::kInput;
        std::uint32_t layer = 0;
        std::uint16_t* first = nullptr;
        std::uint16_t* second = nullptr;
    };

    struct DecodeGraphVariant {
        std::uintptr_t gdn_working = 0;
        std::uintptr_t ple_working = 0;
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
                    options.max_transaction_tokens, options.enable_mtp),
          ple_store(std::move(value_ple)) {
        if (options.device < 0 || options.max_transaction_tokens == 0 ||
            options.context_capacity == 0 ||
            options.context_capacity > kQ38ContextLimit ||
            weights.stage() != static_cast<std::uint32_t>(options.stage))
            throw std::invalid_argument("invalid CUDA stage backend options");
        const auto first = options.stage == Stage::kStage0 ? 0u : kCut;
        const auto end = options.stage == Stage::kStage0 ? kCut : kLayers;
        std::uint32_t gdn_local = 0;
        std::uint32_t qsa_local = 0;
        for (auto layer = first; layer < end; ++layer) {
            const bool qsa = layer % 4 == 3;
            layers.push_back(bind_layer(weights, layer,
                                        qsa ? qsa_local++ : gdn_local++));
        }
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
            ple_reader = std::make_unique<PleAsyncReader>(ple_store);
            embedding = require_matrix(weights, "token_embd.weight", kVocabulary,
                                       kQ38HiddenWidth);
            small_boundary_ring = std::make_unique<CudaBoundaryRing>(
                CudaBoundaryRingOptions{options.device, 3,
                                        kSmallBoundaryTokens,
                                        kQ38HyperWidth});
            prefill_boundary_ring = std::make_unique<CudaBoundaryRing>(
                CudaBoundaryRingOptions{options.device, 3,
                                        options.max_transaction_tokens,
                                        kQ38HyperWidth});
        } else {
            output = require_matrix(weights, "output.weight", kVocabulary,
                                    kQ38HiddenWidth);
            final_hyper = bind_hyper(weights, "hc_input", false);
            if (options.enable_mtp) {
                mtp = std::make_unique<MtpWeights>(bind_mtp(weights));
                mtp_qsa_state = std::make_unique<CudaQsaStateBank>(
                    options.device, 1, options.context_capacity);
            }
        }
        if (profile_requested("Q38_CUDA_PROFILE_DECODE"))
            decode_profile = std::make_unique<CudaEventProfile>(
                options.device, "decode");
        if (profile_requested("Q38_CUDA_PROFILE_PREFILL"))
            prefill_profile = std::make_unique<CudaEventProfile>(
                options.device, "prefill");
    }

    void profile_mark(const char* label) {
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
        if (batch > 1 && alpha == 1.0f && !accumulate &&
            device == options.device &&
            stream == reinterpret_cast<void*>(workspace.stream)) {
            if (prefill_cache.gemm(matrix, input, output, batch,
                                   workspace.stream))
                return;
        }
        q38::cuda_gemv_bf16(matrix, input, output, batch, stream, device,
                            alpha, accumulate);
    }

    void prepare_hyper(const HyperWeights& hyper, const std::uint16_t* input,
                       std::uint32_t tokens = 1) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        cuda_hyper_group_rmsnorm_bf16(input, hyper.norm, workspace.hyper_norm,
                                      tokens, stream, options.device);
        cuda_gemv_bf16(hyper.mix_down, workspace.hyper_norm,
                       workspace.low_rank, tokens, stream, options.device);
        cuda_hyper_silu_scaled_bf16(
            workspace.low_rank, workspace.low_rank, tokens,
            kQ38HyperLowRank, kQ38HyperCount, stream, options.device);
        cuda_gemv_bf16(hyper.mix_up, workspace.low_rank, workspace.mix_weights,
                       tokens, stream, options.device);
        cuda_hyper_sigmoid_bf16(workspace.mix_weights, workspace.mix_weights,
                                tokens, kQ38HyperWidth, 1.0f, 1.0f, stream,
                                options.device);
        cuda_hyper_mix_bf16(workspace.hyper_norm, workspace.mix_weights,
                            workspace.mixed, tokens, stream, options.device);
        if (hyper.has_inject) {
            cuda_gemv_bf16(hyper.inject, workspace.hyper_norm,
                           workspace.injection, tokens, stream, options.device);
            cuda_hyper_sigmoid_bf16(
                workspace.injection, workspace.injection, tokens,
                kQ38HyperCount, kQ38HyperCount, 2.0f, stream, options.device);
        }
    }

    std::uint16_t* finish_hyper(const std::uint16_t* original,
                                std::uint16_t* destination,
                                std::uint32_t tokens = 1) {
        cuda_hyper_inject_bf16(
            original, workspace.block_output, workspace.injection, destination,
            tokens, reinterpret_cast<void*>(workspace.stream), options.device);
        return destination;
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
        cuda_gemv_bf16(layer.ple.key, workspace.proj5, workspace.proj1, 1,
                       stream, options.device);
        cuda_gemv_bf16(layer.ple.value, workspace.proj5, workspace.proj6, 1,
                       stream, options.device);
        cuda_hyper_group_rmsnorm_bf16(
            workspace.proj1, layer.ple.key_norm, workspace.hyper_norm, 1,
            stream, options.device);
        cuda_hyper_group_rmsnorm_bf16(
            hyper, layer.ple.query_norm, workspace.proj2, 1, stream,
            options.device);
        cuda_ple_gate_bf16(workspace.hyper_norm, workspace.proj2,
                           workspace.proj6, workspace.mix_weights, stream,
                           options.device);
        cuda_hyper_group_rmsnorm_bf16(
            workspace.mix_weights, layer.ple.conv_norm, workspace.proj1, 1,
            stream, options.device);
        cuda_ple_conv_decode_bf16(
            workspace.mix_weights, workspace.proj1, layer.ple.conv,
            ple_state->working(), workspace.proj2, stream, options.device);
        cuda_add_bf16(hyper, workspace.proj2, destination, kQ38HyperWidth,
                      stream, options.device);
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
        cuda_gemv_bf16(layer.ple.key, workspace.proj5, workspace.hyper_norm,
                       tokens, stream, options.device);
        cuda_gemv_bf16(layer.ple.value, workspace.proj5, workspace.proj6,
                       tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("ple_key_value");
        cuda_hyper_group_rmsnorm_bf16(
            workspace.hyper_norm, layer.ple.key_norm, workspace.proj1, tokens,
            stream, options.device);
        cuda_hyper_group_rmsnorm_bf16(
            hyper, layer.ple.query_norm, workspace.proj2, tokens, stream,
            options.device);
        if (detailed_prefill_profile()) profile_mark("ple_norms");
        cuda_ple_gate_prefill_bf16(
            workspace.proj1, workspace.proj2, workspace.proj6,
            workspace.mix_weights, tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("ple_gate");
        cuda_hyper_group_rmsnorm_bf16(
            workspace.mix_weights, layer.ple.conv_norm, workspace.hyper_norm,
            tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("ple_conv_norm");
        cuda_ple_conv_prefill_bf16(
            workspace.mix_weights, workspace.hyper_norm, layer.ple.conv,
            ple_state->working(), workspace.proj2, tokens, stream,
            options.device);
        if (detailed_prefill_profile()) profile_mark("ple_conv");
        cuda_add_bf16(hyper, workspace.proj2, destination,
                      static_cast<std::size_t>(tokens) * kQ38HyperWidth,
                      stream, options.device);
        if (detailed_prefill_profile()) profile_mark("ple_residual");
    }

    void run_gdn(const LayerWeights& layer) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        cuda_gemv_bf16(layer.gdn.qkv, workspace.mixed, workspace.proj1, 1,
                       stream, options.device);
        cuda_gemv_bf16(layer.gdn.z, workspace.mixed, workspace.proj3, 1,
                       stream, options.device);
        cuda_gemv_bf16(layer.gdn.b, workspace.mixed, workspace.small0, 1,
                       stream, options.device);
        cuda_gemv_bf16(layer.gdn.a, workspace.mixed, workspace.small1, 1,
                       stream, options.device);
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
        cuda_gemv_bf16(layer.gdn.output, workspace.proj0,
                       workspace.block_output, 1, stream, options.device);
    }

    void run_gdn_prefill(const LayerWeights& layer, std::uint32_t tokens) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        cuda_gemv_bf16(layer.gdn.qkv, workspace.mixed, workspace.proj1,
                       tokens, stream, options.device);
        cuda_gemv_bf16(layer.gdn.z, workspace.mixed, workspace.proj3, tokens,
                       stream, options.device);
        cuda_gemv_bf16(layer.gdn.b, workspace.mixed, workspace.small0, tokens,
                       stream, options.device);
        cuda_gemv_bf16(layer.gdn.a, workspace.mixed, workspace.small1, tokens,
                       stream, options.device);
        if (detailed_prefill_profile()) profile_mark("gdn_project");
        const auto state = gdn_state->working(layer.state_local);
        cuda_gdn_conv_prefill_bf16(
            workspace.proj1, layer.gdn.conv, state.conv, workspace.proj2,
            tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("gdn_conv");
        if (precomputed_gdn_prefill_enabled())
            cuda_gdn_recurrent_prefill_precomputed_bf16(
                workspace.proj2, workspace.small0, workspace.small1,
                layer.gdn.a_log, layer.gdn.dt_bias, state.recurrent,
                workspace.block_scores, workspace.proj4, tokens, stream,
                options.device);
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
        cuda_gemv_bf16(layer.gdn.output, workspace.proj0,
                       workspace.block_output, tokens, stream,
                       options.device);
        if (detailed_prefill_profile()) profile_mark("gdn_output");
    }

    void run_qsa(const LayerWeights& layer, std::uint32_t position,
                 CudaQsaStateBank& state_bank) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const auto state = state_bank.working(layer.state_local);
        cuda_gemv_bf16(layer.sparse.q_gate, workspace.mixed, workspace.proj0, 1,
                       stream, options.device);
        cuda_gemv_bf16(layer.sparse.k, workspace.mixed, workspace.small0, 1,
                       stream, options.device);
        cuda_gemv_bf16(layer.sparse.v, workspace.mixed, workspace.small1, 1,
                       stream, options.device);
        cuda_gemv_bf16(layer.sparse.index_qk, workspace.mixed,
                       workspace.small2, 1, stream, options.device);
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
        cuda_gemv_bf16(layer.sparse.output, workspace.proj3,
                       workspace.block_output, 1, stream, options.device);
        profile_mark("qsa_output");
    }

    void run_qsa_prefill(const LayerWeights& layer,
                         std::uint32_t first_position,
                         std::uint32_t tokens,
                         CudaQsaStateBank& state_bank) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        const auto state = state_bank.working(layer.state_local);
        cuda_gemv_bf16(layer.sparse.q_gate, workspace.mixed, workspace.proj0,
                       tokens, stream, options.device);
        cuda_gemv_bf16(layer.sparse.k, workspace.mixed, workspace.small0,
                       tokens, stream, options.device);
        cuda_gemv_bf16(layer.sparse.v, workspace.mixed, workspace.small1,
                       tokens, stream, options.device);
        cuda_gemv_bf16(layer.sparse.index_qk, workspace.mixed,
                       workspace.small2, tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("qsa_project");
        cuda_qsa_prepare_prefill_bf16(
            workspace.proj0, workspace.small0, workspace.small1,
            workspace.small2, layer.sparse.q_norm, layer.sparse.k_norm,
            layer.sparse.index_q_norm, layer.sparse.index_k_norm,
            workspace.proj3, workspace.proj4, workspace.proj1, state,
            first_position, tokens, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("qsa_prepare");
        cuda_qsa_select_prefill(
            workspace.proj1, state, first_position, tokens,
            workspace.block_scores,
            kQ38ContextLimit / kQ38QsaBlockTokens,
            workspace.selected_indices, stream, options.device);
        if (detailed_prefill_profile()) profile_mark("qsa_select");
        const bool grouped_qsa = grouped_qsa_prefill_enabled();
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
        cuda_sigmoid_multiply_bf16(
            workspace.proj0, workspace.proj4, workspace.proj3,
            static_cast<std::size_t>(tokens) * kQ38GdnValueWidth, stream,
            options.device);
        if (detailed_prefill_profile()) profile_mark("qsa_gate");
        cuda_gemv_bf16(layer.sparse.output, workspace.proj3,
                       workspace.block_output, tokens, stream,
                       options.device);
        if (detailed_prefill_profile()) profile_mark("qsa_output");
    }

    void run_moe(const LayerWeights& layer, std::uint32_t tokens = 1) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        cuda_gemv_bf16(layer.moe.router, workspace.mixed, workspace.small0,
                       tokens, stream, options.device);
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_router");
        cuda_topk_router_bf16(workspace.small0, workspace.expert_ids,
                              workspace.expert_weights, tokens, kQ38MoeExperts,
                              kQ38MoeTopK, true, stream, options.device);
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_topk");
        const bool grouped_prefill =
            tokens > 1 && workspace.moe_route_plan.header != nullptr;
        const bool detailed_prefill_profile =
            grouped_prefill && prefill_profile && prefill_profile->active() &&
            (profile_requested("Q38_CUDA_PROFILE_PREFILL_DETAIL") ||
             profile_requested("Q38_CUDA_PROFILE_PREFILL_MOE_DETAIL"));
        if (grouped_prefill) {
            cuda_moe_build_route_plan_v1(
                workspace.expert_ids, tokens, workspace.moe_route_plan,
                stream, options.device);
            if (detailed_prefill_profile) profile_mark("moe_route_plan");
            cuda_moe_pack_hidden_v1(
                workspace.mixed, workspace.moe_route_plan.packed_assignment,
                tokens * kQ38MoeTopK, workspace.moe_packed_hidden, stream,
                options.device);
            if (detailed_prefill_profile) profile_mark("moe_pack_hidden");
            cuda_moe_grouped_gate_up_v1(
                layer.moe.gate_up_experts, workspace.moe_packed_hidden,
                workspace.moe_route_plan.packed_assignment,
                workspace.expert_weights, workspace.moe_route_plan.tasks,
                q38_moe_prefill_max_tasks_v1(tokens),
                workspace.moe_weighted_mid, prefill_moe_mode(), stream,
                options.device);
            if (detailed_prefill_profile) profile_mark("moe_gate_up");
            cuda_moe_grouped_down_v1(
                layer.moe.down_experts, workspace.moe_weighted_mid,
                workspace.moe_route_plan.tasks,
                q38_moe_prefill_max_tasks_v1(tokens),
                workspace.moe_route_output, prefill_moe_mode(), stream,
                options.device);
            if (detailed_prefill_profile) profile_mark("moe_down");
        } else {
            cuda_moe_routed_bf16(
                layer.moe.gate_up_experts, layer.moe.down_experts,
                workspace.mixed, workspace.expert_ids,
                workspace.expert_weights, tokens, kQ38MoeTopK,
                workspace.moe_gate_up, workspace.moe_activated,
                workspace.moe_accumulation, workspace.proj6, stream,
                options.device);
        }
        if (!detailed_prefill_profile &&
            (tokens == 1 ||
             (prefill_profile && prefill_profile->active())))
            profile_mark("moe_routed");
        cuda_gemv_bf16(layer.moe.shared_gate, workspace.mixed,
                       workspace.small2, tokens, stream, options.device);
        cuda_gemv_bf16(layer.moe.shared_up, workspace.mixed,
                       workspace.small1, tokens, stream, options.device);
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_shared_gate_up");
        cuda_silu_multiply_bf16(
            workspace.small2, workspace.small1, workspace.small0,
            static_cast<std::size_t>(tokens) * kQ38MoeIntermediate, stream,
            options.device);
        cuda_gemv_bf16(layer.moe.shared_down, workspace.small0,
                       workspace.proj5, tokens, stream, options.device);
        if (tokens == 1 ||
            (prefill_profile && prefill_profile->active()))
            profile_mark("moe_shared_down");
        cuda_gemv_bf16(layer.moe.shared_output_gate, workspace.mixed,
                       workspace.small1, tokens, stream, options.device);
        if (grouped_prefill) {
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

    void enqueue_decode_static_action(const DecodeStaticAction& action) {
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
                cuda_embedding_bf16(embedding, workspace.tokens,
                                    workspace.proj5, 1, stream,
                                    options.device);
                cuda_hyper_repeat_embedding_bf16(
                    workspace.proj5, workspace.hyper_a, 1, stream,
                    options.device);
            } else {
                check(cudaMemcpyAsync(
                          workspace.hyper_a, workspace.boundary,
                          kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(graph stage boundary token)");
            }
            return;
        case DecodeStaticActionKind::kPle:
            run_ple(require_layer(), action.first, action.second, 0);
            return;
        case DecodeStaticActionKind::kPrepareAttention:
            prepare_hyper(require_layer().attention_hyper, action.first);
            return;
        case DecodeStaticActionKind::kGdn:
            run_gdn(require_layer());
            return;
        case DecodeStaticActionKind::kFinish:
            finish_hyper(action.first, action.second);
            return;
        case DecodeStaticActionKind::kPrepareMoe:
            prepare_hyper(require_layer().moe_hyper, action.first);
            return;
        case DecodeStaticActionKind::kMoe:
            run_moe(require_layer());
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
                              workspace.final_hc, action.first,
                              kQ38HyperWidth * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToDevice, workspace.stream),
                          "cudaMemcpyAsync(graph final target HC)");
                prepare_hyper(final_hyper, action.first);
                cuda_gemv_bf16(output, workspace.mixed, workspace.logits, 1,
                               stream, options.device);
                cuda_argmax_bf16(workspace.logits, kVocabulary,
                                 workspace.predictions, stream,
                                 options.device);
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
        std::uintptr_t gdn_working, std::uintptr_t ple_working) const {
        for (const auto& variant : decode_graphs) {
            if (variant->gdn_working == gdn_working &&
                variant->ple_working == ple_working)
                return variant.get();
        }
        return nullptr;
    }

    DecodeGraphVariant& build_decode_graph_variant(
        std::uintptr_t gdn_working, std::uintptr_t ple_working) {
        // Each transaction alternates between two GDN/PLE working banks.  A
        // graph records raw kernel pointers, so keep one variant per pointer
        // pair rather than replaying a graph against the wrong bank.
        if (decode_graphs.size() >= 2)
            throw std::runtime_error(
                "piecewise decode observed more than two state-bank variants");

        auto variant = std::make_unique<DecodeGraphVariant>();
        variant->gdn_working = gdn_working;
        variant->ple_working = ple_working;
        std::vector<std::vector<DecodeStaticAction>> pieces(1);
        auto* current = workspace.hyper_a;
        auto* alternate = workspace.hyper_b;
        pieces.back().push_back(
            {DecodeStaticActionKind::kInput, 0, nullptr, nullptr});

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
                variant->qsa_layers.push_back(layer_index);
                pieces.emplace_back();
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
            pieces.back().push_back({DecodeStaticActionKind::kFinish,
                                     layer_index, current, alternate});
            std::swap(current, alternate);
        }
        pieces.back().push_back(
            {DecodeStaticActionKind::kOutput, 0, current, nullptr});
        if (pieces.size() != variant->qsa_layers.size() + 1)
            throw std::logic_error(
                "piecewise decode fragment plan is inconsistent");

        std::uint64_t nodes = 0;
        for (const auto& piece : pieces) {
            if (piece.empty())
                throw std::logic_error(
                    "piecewise decode produced an empty graph fragment");
            auto graph = std::make_unique<CapturedCudaGraph>(
                options.device, workspace.stream, [&]() {
                    for (const auto& action : piece)
                        enqueue_decode_static_action(action);
                });
            nodes += graph->nodes();
            variant->fragments.push_back(std::move(graph));
        }
        metrics.cuda_graph_captures += variant->fragments.size();
        metrics.cuda_graph_nodes += nodes;
        auto* result = variant.get();
        decode_graphs.push_back(std::move(variant));
        return *result;
    }

    bool run_one_piecewise(std::uint32_t position) {
        if (decode_graph_failed) {
            ++metrics.cuda_graph_fallbacks;
            return false;
        }
        check_cancelled();
        // PLE I/O and both H2D copies stay outside capture.  During capture
        // run_ple only records the GPU conversion/projection/convolution.
        if (options.stage == Stage::kStage0) ensure_ple_ready(0, 0, 1);

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

        for (std::size_t index = 0; index < variant->qsa_layers.size();
             ++index) {
            check_cancelled();
            variant->fragments[index]->launch(workspace.stream);
            ++metrics.cuda_graph_replays;
            run_qsa(layers[variant->qsa_layers[index]], position,
                    *qsa_state);
        }
        check_cancelled();
        variant->fragments.back()->launch(workspace.stream);
        ++metrics.cuda_graph_replays;
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

            prepare_hyper(layer.moe_hyper, current);
            profile_mark("moe_hyper");
            run_moe(layer);
            finish_hyper(current, alternate);
            std::swap(current, alternate);
            profile_mark("moe_finish");
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
                prepare_hyper(final_hyper, current);
                cuda_gemv_bf16(output, workspace.mixed, workspace.logits, 1,
                               stream, options.device);
                cuda_argmax_bf16(workspace.logits, kVocabulary,
                                 workspace.predictions + result_index, stream,
                                 options.device);
            }
        }
        profile_mark(options.stage == Stage::kStage0 ? "stage_output"
                                                      : "final_head");
    }

    void run_prefill(std::uint32_t count, std::uint32_t chunk_offset,
                     bool write_result) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        for (std::uint32_t first = 0; first < count;
             first += kPrefillTile) {
            check_cancelled();
            const auto tokens =
                std::min<std::uint32_t>(kPrefillTile, count - first);
            // Profile two representative full tiles per chunk: one after the
            // lazy matrix-cache warmup and one at the largest context in the
            // chunk.  Tail tiles have different occupancy and are excluded.
            const bool profile_prefill =
                prefill_profile && tokens == kPrefillTile &&
                (first == kPrefillTile || first + tokens == count);
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
            for (const auto& layer : layers) {
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
                finish_hyper(current, alternate, tokens);
                std::swap(current, alternate);
                if (profile_prefill) profile_mark("moe_finish");
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
                if (write_result && first + tokens == count) {
                    const auto* last =
                        current + static_cast<std::uint64_t>(tokens - 1) *
                                      kQ38HyperWidth;
                    prepare_hyper(final_hyper, last);
                    cuda_gemv_bf16(output, workspace.mixed, workspace.logits,
                                   1, stream, options.device);
                    cuda_argmax_bf16(
                        workspace.logits, kVocabulary,
                        workspace.predictions + count - 1, stream,
                        options.device);
                }
            }
            if (profile_prefill) {
                profile_mark(options.stage == Stage::kStage0
                                 ? "stage_output"
                                 : "final_head");
                prefill_profile->finish(
                    options.stage, provisional_base + chunk_offset + first,
                    workspace.stream);
            }
            check(cudaStreamSynchronize(workspace.stream),
                  "cudaStreamSynchronize(prefill cancellation tile)");
            check_cancelled();
        }
        qsa_state->mark_evaluated(provisional_epoch, chunk_offset + count);
    }

    void run_tokens(std::uint32_t count, std::uint32_t chunk_offset,
                    bool write_result) {
        if (provisional_kind == TxnKind::kAppendKnown && count > 1) {
            run_prefill(count, chunk_offset, write_result);
            return;
        }
        if (options.enable_piecewise_decode_graph && !decode_profile &&
            provisional_kind == TxnKind::kDecode && count == 1 &&
            chunk_offset == 0 && write_result &&
            run_one_piecewise(static_cast<std::uint32_t>(provisional_base))) {
            qsa_state->mark_evaluated(provisional_epoch, 1);
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

        // Official Qwen MTP fusion: project the normalized current-token
        // embedding once, project each of the four normalized target HC
        // streams with the shared hidden projection, broadcast the embedding
        // projection, then add the two 4H tensors.
        cuda_embedding_bf16(mtp->embedding, token_device, workspace.proj5, 1,
                            stream, options.device);
        cuda_qwen38_rmsnorm_bf16(
            workspace.proj5, mtp->embedding_norm, false, workspace.proj6, 1,
            kQ38HiddenWidth, 1.0e-6f, true, stream, options.device);
        cuda_gemv_bf16(mtp->fc_embedding, workspace.proj6, workspace.proj5, 1,
                       stream, options.device);
        cuda_qwen38_rmsnorm_bf16(
            target_hc, mtp->hidden_norm, false, workspace.hyper_norm, 1,
            kQ38HyperWidth, 1.0e-6f, true, stream, options.device);
        cuda_gemv_bf16(mtp->fc_hidden, workspace.hyper_norm,
                       workspace.hyper_a, kQ38HyperCount, stream,
                       options.device);
        cuda_hyper_repeat_embedding_bf16(
            workspace.proj5, workspace.hyper_b, 1, stream, options.device);
        cuda_add_bf16(workspace.hyper_a, workspace.hyper_b,
                      workspace.hyper_a, kQ38HyperWidth, stream,
                      options.device);

        std::uint16_t* current = workspace.hyper_a;
        std::uint16_t* alternate = workspace.hyper_b;
        prepare_hyper(mtp->layer.attention_hyper, current);
        run_qsa(mtp->layer, position, *mtp_qsa_state);
        finish_hyper(current, alternate);
        std::swap(current, alternate);
        prepare_hyper(mtp->layer.moe_hyper, current);
        run_moe(mtp->layer);
        finish_hyper(current, alternate);
        std::swap(current, alternate);

        if (produce_prediction) {
            prepare_hyper(mtp->output_hyper, current);
            cuda_gemv_bf16(output, workspace.mixed, workspace.logits, 1,
                           stream, options.device);
            cuda_argmax_bf16(workspace.logits, kVocabulary,
                             prediction_output, stream, options.device);
        }
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
                *mtp_qsa_state);
            finish_hyper(current, alternate, tokens);
            std::swap(current, alternate);
            prepare_hyper(mtp->layer.moe_hyper, current, tokens);
            run_moe(mtp->layer, tokens);
            finish_hyper(current, alternate, tokens);
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
        for (const auto token : token_ids) {
            const auto rows = candidate.rows({token});
            chunk_rows.insert(chunk_rows.end(), rows.begin(), rows.end());
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
        check(cudaMemcpyAsync(workspace.ple_rows + byte_offset,
                              workspace.host_ple_rows + byte_offset,
                              byte_count, cudaMemcpyHostToDevice,
                              workspace.stream),
              "cudaMemcpyAsync(PLE row tile)");
        check(cudaMemcpyAsync(
                  workspace.ple_scales + workspace_first_row,
                  workspace.host_ple_scales + workspace_first_row,
                  row_count * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice, workspace.stream),
              "cudaMemcpyAsync(PLE scale tile)");
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
    std::unique_ptr<CudaEventProfile> decode_profile;
    std::unique_ptr<CudaEventProfile> prefill_profile;
    mutable std::uint64_t tracked_peak_bytes = 0;
    StageBackendMetricsV1 metrics{};
    bool decode_graph_failed = false;
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
            if (!input.final_chunk ||
                input.txn.kind == TxnKind::kSpeculative)
                throw std::runtime_error(
                    "CUDA logits requested for an invalid execution lane");
            result.logits_bf16.resize(kVocabulary);
            check(cudaMemcpyAsync(result.logits_bf16.data(),
                                  state.workspace.logits,
                                  kVocabulary * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost,
                                  state.workspace.stream),
                  "cudaMemcpyAsync(final logits)");
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
        return result;
    } catch (...) {
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
        if (online_lookahead) {
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
    result.gdn_state_bytes = state.gdn_state->allocated_bytes();
    result.ple_state_bytes =
        state.ple_state ? state.ple_state->allocated_bytes() : 0;
    result.mtp_state_bytes = state.workspace.mtp_state_bytes +
        (state.mtp_qsa_state ? state.mtp_qsa_state->allocated_bytes() : 0);
    result.workspace_device_bytes =
        state.workspace.device_bytes - state.workspace.mtp_state_bytes;
    result.prefill_cache_device_bytes = state.prefill_cache.resident_bytes();
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
    bool enable_mtp, bool enable_piecewise_decode_graph) {
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
    auto ple = std::make_shared<PleStore>(load_ple_layout(ple_path),
                                          ple_options);
    CudaStageBackendOptions options0;
    options0.stage = Stage::kStage0;
    options0.device = stage0_device;
    options0.max_transaction_tokens = executor_options.append_chunk_tokens;
    options0.context_capacity = executor_options.context_limit;
    options0.enable_speculative_checkpoint = enable_mtp;
    options0.enable_piecewise_decode_graph = enable_piecewise_decode_graph;
    CudaStageBackendOptions options1 = options0;
    options1.stage = Stage::kStage1;
    options1.device = stage1_device;
    options1.enable_mtp = enable_mtp;
    options1.enable_speculative_checkpoint = false;
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
