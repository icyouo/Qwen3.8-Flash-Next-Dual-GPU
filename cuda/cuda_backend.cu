#include "q38/cuda_backend.h"

#include "q38/cuda_gdn.h"
#include "q38/cuda_hyper.h"
#include "q38/cuda_kernels.h"
#include "q38/cuda_moe.h"
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
constexpr std::uint32_t kPrefillTile = 32;
constexpr std::uint32_t kSmallBoundaryTokens = 128;

bool decode_profile_requested() {
    const char* value = std::getenv("Q38_CUDA_PROFILE_DECODE");
    return value && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0;
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
class DecodeEventProfile {
public:
    explicit DecodeEventProfile(int value_device) : device_(value_device) {}
    ~DecodeEventProfile() {
        (void)cudaSetDevice(device_);
        for (auto event : events_) (void)cudaEventDestroy(event);
    }

    DecodeEventProfile(const DecodeEventProfile&) = delete;
    DecodeEventProfile& operator=(const DecodeEventProfile&) = delete;

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
        std::cerr << "{\"type\":\"q38_cuda_decode_profile\",\"stage\":"
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
    bool active_ = false;
    std::size_t event_count_ = 0;
    std::vector<cudaEvent_t> events_;
    std::vector<std::string> labels_;
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

// One persistent producer per stage0 backend.  It fills CUDA-pinned host
// memory directly, so token-0 layer-0 kernels can run while the 16 PLE rows
// for the transaction are fetched from SSD/cache.  No temporary row vectors
// or per-transaction thread creation are involved.
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
        }
        ready_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    PleAsyncReader(const PleAsyncReader&) = delete;
    PleAsyncReader& operator=(const PleAsyncReader&) = delete;

    void submit(std::vector<std::uint64_t> rows, std::uint8_t* output,
                std::size_t output_bytes, std::uint16_t* scale_output,
                std::size_t output_scales) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (request_ready_ || working_ || result_ready_ || stopping_)
            throw std::logic_error("PLE async reader already has work");
        rows_ = std::move(rows);
        output_ = output;
        output_bytes_ = output_bytes;
        scale_output_ = scale_output;
        output_scales_ = output_scales;
        error_ = nullptr;
        completed_rows_ = 0;
        total_rows_ = rows_.size();
        request_ready_ = true;
        ready_.notify_one();
    }

    void wait_until(std::size_t required_rows) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (required_rows == 0 || required_rows > total_rows_)
            throw std::invalid_argument("PLE async wait extent is invalid");
        finished_.wait(lock, [this, required_rows] {
            return completed_rows_ >= required_rows || result_ready_ ||
                   stopping_;
        });
        if (completed_rows_ < required_rows) {
            auto error = error_;
            if (result_ready_) {
                result_ready_ = false;
                error_ = nullptr;
            }
            lock.unlock();
            if (error) std::rethrow_exception(error);
            throw std::runtime_error("PLE async reader stopped early");
        }
        if (required_rows == total_rows_) {
            finished_.wait(lock, [this] { return result_ready_ || stopping_; });
            auto error = error_;
            result_ready_ = false;
            error_ = nullptr;
            lock.unlock();
            if (error) std::rethrow_exception(error);
        }
    }

    void wait() { wait_until(total_rows_); }

private:
    void run() noexcept {
        for (;;) {
            std::vector<std::uint64_t> rows;
            std::uint8_t* output = nullptr;
            std::size_t output_bytes = 0;
            std::uint16_t* scale_output = nullptr;
            std::size_t output_scales = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock,
                            [this] { return stopping_ || request_ready_; });
                if (stopping_ && !request_ready_) return;
                rows = std::move(rows_);
                output = output_;
                output_bytes = output_bytes_;
                scale_output = scale_output_;
                output_scales = output_scales_;
                request_ready_ = false;
                working_ = true;
            }
            std::exception_ptr error;
            try {
                const auto row_bytes = store_->layout().row_stride_bytes;
                if (!output || !scale_output ||
                    output_bytes != rows.size() * row_bytes ||
                    output_scales != rows.size())
                    throw std::invalid_argument(
                        "PLE async output extent differs");
                constexpr std::size_t kRowsPerTile =
                    static_cast<std::size_t>(kPrefillTile) *
                    kQ38PleRowsPerToken;
                for (std::size_t first = 0; first < rows.size();
                     first += kRowsPerTile) {
                    const auto count =
                        std::min<std::size_t>(kRowsPerTile, rows.size() - first);
                    std::vector<std::uint64_t> tile(
                        rows.begin() + static_cast<std::ptrdiff_t>(first),
                        rows.begin() +
                            static_cast<std::ptrdiff_t>(first + count));
                    store_->read_rows_into(
                        tile, output + first * row_bytes, count * row_bytes);
                    store_->read_row_scales_into(
                        tile, scale_output + first, count);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        completed_rows_ = first + count;
                    }
                    finished_.notify_all();
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
    std::vector<std::uint64_t> rows_;
    std::uint8_t* output_ = nullptr;
    std::size_t output_bytes_ = 0;
    std::uint16_t* scale_output_ = nullptr;
    std::size_t output_scales_ = 0;
    std::exception_ptr error_;
    std::size_t completed_rows_ = 0;
    std::size_t total_rows_ = 0;
    bool request_ready_ = false;
    bool working_ = false;
    bool result_ready_ = false;
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
            moe_gate_up = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38MoeTopK * 2 *
                kQ38MoeIntermediate);
            moe_activated = allocate<std::uint16_t>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38MoeTopK *
                kQ38MoeIntermediate);
            moe_accumulation = allocate<float>(
                static_cast<std::uint64_t>(kPrefillTile) * kQ38HiddenWidth);
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
        gdn_state = std::make_unique<CudaGdnStateBank>(options.device, gdn_local);
        qsa_state = std::make_unique<CudaQsaStateBank>(
            options.device, qsa_local, options.context_capacity);
        if (options.stage == Stage::kStage0) {
            if (!ple_store || ple_store->layout().storage_dtype != DType::kFp8E4M3 ||
                ple_store->layout().row_dimension != kQ38PleRowWidth)
                throw std::invalid_argument("stage0 requires FP8 PLE layout");
            ple_state = std::make_unique<CudaPleStateBank>(options.device);
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
        if (decode_profile_requested())
            decode_profile =
                std::make_unique<DecodeEventProfile>(options.device);
    }

    void profile_mark(const char* label) {
        if (decode_profile && decode_profile->active())
            decode_profile->mark(label, workspace.stream);
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
        cuda_gemv_bf16(layer.ple.key, workspace.proj5, workspace.hyper_norm,
                       tokens, stream, options.device);
        cuda_gemv_bf16(layer.ple.value, workspace.proj5, workspace.proj6,
                       tokens, stream, options.device);
        cuda_hyper_group_rmsnorm_bf16(
            workspace.hyper_norm, layer.ple.key_norm, workspace.proj1, tokens,
            stream, options.device);
        cuda_hyper_group_rmsnorm_bf16(
            hyper, layer.ple.query_norm, workspace.proj2, tokens, stream,
            options.device);
        cuda_ple_gate_prefill_bf16(
            workspace.proj1, workspace.proj2, workspace.proj6,
            workspace.mix_weights, tokens, stream, options.device);
        cuda_hyper_group_rmsnorm_bf16(
            workspace.mix_weights, layer.ple.conv_norm, workspace.hyper_norm,
            tokens, stream, options.device);
        cuda_ple_conv_prefill_bf16(
            workspace.mix_weights, workspace.hyper_norm, layer.ple.conv,
            ple_state->working(), workspace.proj2, tokens, stream,
            options.device);
        cuda_add_bf16(hyper, workspace.proj2, destination,
                      static_cast<std::size_t>(tokens) * kQ38HyperWidth,
                      stream, options.device);
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
        const auto state = gdn_state->working(layer.state_local);
        cuda_gdn_conv_prefill_bf16(
            workspace.proj1, layer.gdn.conv, state.conv, workspace.proj2,
            tokens, stream, options.device);
        cuda_gdn_recurrent_prefill_bf16(
            workspace.proj2, workspace.small0, workspace.small1,
            layer.gdn.a_log, layer.gdn.dt_bias, state.recurrent,
            workspace.proj4, tokens, stream, options.device);
        cuda_gdn_output_norm_prefill_bf16(
            workspace.proj4, workspace.proj3, layer.gdn.norm, workspace.proj0,
            tokens, 1.0e-6f, stream, options.device);
        cuda_gemv_bf16(layer.gdn.output, workspace.proj0,
                       workspace.block_output, tokens, stream,
                       options.device);
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
        cuda_qsa_prepare_main_decode_bf16(
            workspace.proj0, workspace.small0, workspace.small1,
            layer.sparse.q_norm, layer.sparse.k_norm, workspace.proj3,
            workspace.proj4, state, position, stream, options.device);
        cuda_qsa_prepare_index_decode_bf16(
            workspace.small2, layer.sparse.index_q_norm,
            layer.sparse.index_k_norm, workspace.small0, state, position,
            stream, options.device);
        const auto selected = cuda_qsa_select_decode(
            workspace.small0, state, position, workspace.block_scores,
            workspace.selected_indices, stream, options.device);
        cuda_qsa_attention_decode_bf16(
            workspace.proj3, state, workspace.selected_indices, selected,
            workspace.attention_scores, workspace.proj0, stream,
            options.device);
        cuda_sigmoid_multiply_bf16(
            workspace.proj0, workspace.proj4, workspace.proj3,
            kQ38GdnValueWidth, stream, options.device);
        cuda_gemv_bf16(layer.sparse.output, workspace.proj3,
                       workspace.block_output, 1, stream, options.device);
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
        cuda_qsa_prepare_prefill_bf16(
            workspace.proj0, workspace.small0, workspace.small1,
            workspace.small2, layer.sparse.q_norm, layer.sparse.k_norm,
            layer.sparse.index_q_norm, layer.sparse.index_k_norm,
            workspace.proj3, workspace.proj4, workspace.proj1, state,
            first_position, tokens, stream, options.device);
        cuda_qsa_attention_prefill_bf16(
            workspace.proj3, workspace.proj1, state, first_position, tokens,
            workspace.block_scores,
            kQ38ContextLimit / kQ38QsaBlockTokens,
            workspace.selected_indices, workspace.attention_scores,
            workspace.proj0, stream, options.device);
        cuda_sigmoid_multiply_bf16(
            workspace.proj0, workspace.proj4, workspace.proj3,
            static_cast<std::size_t>(tokens) * kQ38GdnValueWidth, stream,
            options.device);
        cuda_gemv_bf16(layer.sparse.output, workspace.proj3,
                       workspace.block_output, tokens, stream,
                       options.device);
    }

    void run_moe(const LayerWeights& layer, std::uint32_t tokens = 1) {
        auto stream = reinterpret_cast<void*>(workspace.stream);
        cuda_gemv_bf16(layer.moe.router, workspace.mixed, workspace.small0,
                       tokens, stream, options.device);
        if (tokens == 1) profile_mark("moe_router");
        cuda_topk_router_bf16(workspace.small0, workspace.expert_ids,
                              workspace.expert_weights, tokens, kQ38MoeExperts,
                              kQ38MoeTopK, true, stream, options.device);
        if (tokens == 1) profile_mark("moe_topk");
        cuda_moe_routed_bf16(
            layer.moe.gate_up_experts, layer.moe.down_experts, workspace.mixed,
            workspace.expert_ids, workspace.expert_weights, tokens,
            kQ38MoeTopK, workspace.moe_gate_up, workspace.moe_activated,
            workspace.moe_accumulation, workspace.proj6, stream,
            options.device);
        if (tokens == 1) profile_mark("moe_routed");
        cuda_gemv_bf16(layer.moe.shared_gate, workspace.mixed,
                       workspace.small2, tokens, stream, options.device);
        cuda_gemv_bf16(layer.moe.shared_up, workspace.mixed,
                       workspace.small1, tokens, stream, options.device);
        if (tokens == 1) profile_mark("moe_shared_gate_up");
        cuda_silu_multiply_bf16(
            workspace.small2, workspace.small1, workspace.small0,
            static_cast<std::size_t>(tokens) * kQ38MoeIntermediate, stream,
            options.device);
        cuda_gemv_bf16(layer.moe.shared_down, workspace.small0,
                       workspace.proj5, tokens, stream, options.device);
        if (tokens == 1) profile_mark("moe_shared_down");
        cuda_gemv_bf16(layer.moe.shared_output_gate, workspace.mixed,
                       workspace.small1, tokens, stream, options.device);
        cuda_moe_combine_shared_bf16(
            workspace.proj6, workspace.proj5, workspace.small1,
            workspace.block_output, tokens, stream, options.device);
        if (tokens == 1) profile_mark("moe_shared_combine");
    }

    void check_cancelled() const {
        if (provisional_cancellation)
            provisional_cancellation->throw_if_requested();
    }

    void run_one(std::uint32_t token_index, std::uint32_t position,
                 bool write_result) {
        check_cancelled();
        auto stream = reinterpret_cast<void*>(workspace.stream);
        if (options.stage == Stage::kStage0) {
            cuda_embedding_bf16(embedding, workspace.tokens + token_index,
                                workspace.proj5, 1, stream, options.device);
            cuda_hyper_repeat_embedding_bf16(
                workspace.proj5, workspace.hyper_a, 1, stream, options.device);
        } else {
            check(cudaMemcpyAsync(
                      workspace.hyper_a,
                      workspace.boundary +
                          static_cast<std::uint64_t>(token_index) *
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
                ensure_ple_ready(token_index, 1);
                run_ple(layer, current, alternate, token_index);
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
                              static_cast<std::uint64_t>(token_index) *
                                  kQ38HyperWidth,
                          current, kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(stage0 boundary output)");
        } else {
            if (mtp)
                check(cudaMemcpyAsync(
                          workspace.final_hc +
                              static_cast<std::uint64_t>(token_index) *
                                  kQ38HyperWidth,
                          current, kQ38HyperWidth * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, workspace.stream),
                      "cudaMemcpyAsync(final target HC)");
            if (write_result) {
                prepare_hyper(final_hyper, current);
                cuda_gemv_bf16(output, workspace.mixed, workspace.logits, 1,
                               stream, options.device);
                cuda_argmax_bf16(workspace.logits, kVocabulary,
                                 workspace.predictions + token_index, stream,
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
            std::uint16_t* current = workspace.hyper_a;
            std::uint16_t* alternate = workspace.hyper_b;
            for (const auto& layer : layers) {
                check_cancelled();
                if (layer.has_ple) {
                    ensure_ple_ready(first, tokens);
                    run_ple_prefill(layer, current, alternate, first, tokens);
                    std::swap(current, alternate);
                }
                prepare_hyper(layer.attention_hyper, current, tokens);
                if (layer.qsa)
                    run_qsa_prefill(
                        layer,
                        static_cast<std::uint32_t>(provisional_base +
                                                   chunk_offset + first),
                        tokens, *qsa_state);
                else
                    run_gdn_prefill(layer, tokens);
                finish_hyper(current, alternate, tokens);
                std::swap(current, alternate);

                prepare_hyper(layer.moe_hyper, current, tokens);
                run_moe(layer, tokens);
                finish_hyper(current, alternate, tokens);
                std::swap(current, alternate);
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
        for (std::uint32_t token = 0; token < count; ++token)
            run_one(token,
                    static_cast<std::uint32_t>(provisional_base +
                                               chunk_offset + token),
                    write_result);
        qsa_state->mark_evaluated(provisional_epoch, chunk_offset + count);
    }

    void run_mtp_step(const std::uint16_t* target_hc,
                      const std::int32_t* token_device,
                      std::uint32_t position, bool produce_prediction) {
        if (!mtp || !mtp_qsa_state || !target_hc || !token_device)
            throw std::logic_error("MTP step is unavailable");
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
                             workspace.predictions, stream, options.device);
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
        mtp_transaction_rows = 0;
        if (evaluated != 0) {
            mtp_qsa_state->begin(epoch);
            mtp_transaction_active = true;
            try {
                run_mtp_prefill(prior_pairs, evaluated, mtp_pending_valid);
                mtp_qsa_state->mark_evaluated(epoch, evaluated);
                mtp_transaction_rows = evaluated;
            } catch (...) {
                mtp_qsa_state->rollback(epoch);
                mtp_transaction_active = false;
                throw;
            }
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
                mtp_qsa_state->rollback(epoch);
                mtp_transaction_active = false;
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

    void start_ple_read(const std::vector<std::int32_t>& token_ids) {
        if (!ple_reader || ple_read_active)
            throw std::logic_error("PLE reader is not ready");
        auto candidate = candidate_ple_hashes.empty()
                             ? *committed_ple_hash
                             : candidate_ple_hashes.back();
        std::vector<std::uint64_t> all_rows;
        all_rows.reserve(token_ids.size() * kQ38PleRowsPerToken);
        for (const auto token : token_ids) {
            const auto rows = candidate.rows({token});
            all_rows.insert(all_rows.end(), rows.begin(), rows.end());
            candidate_ple_hashes.push_back(candidate);
        }
        ple_read_bytes = all_rows.size() * kQ38PleRowWidth;
        ple_read_scales = all_rows.size();
        ple_reader->submit(std::move(all_rows), workspace.host_ple_rows,
                           ple_read_bytes, workspace.host_ple_scales,
                           ple_read_scales);
        ple_read_active = true;
    }

    void ensure_ple_ready(std::uint32_t token_offset,
                          std::uint32_t tokens) {
        if (!ple_read_active) return;
        const auto first_row =
            static_cast<std::size_t>(token_offset) * kQ38PleRowsPerToken;
        const auto row_count =
            static_cast<std::size_t>(tokens) * kQ38PleRowsPerToken;
        const auto required_rows = first_row + row_count;
        try {
            ple_reader->wait_until(required_rows);
            if (required_rows == ple_read_scales) ple_read_active = false;
        } catch (...) {
            ple_read_active = false;
            throw;
        }
        const auto byte_offset = first_row * kQ38PleRowWidth;
        const auto byte_count = row_count * kQ38PleRowWidth;
        check(cudaMemcpyAsync(workspace.ple_rows + byte_offset,
                              workspace.host_ple_rows + byte_offset,
                              byte_count, cudaMemcpyHostToDevice,
                              workspace.stream),
              "cudaMemcpyAsync(PLE row tile)");
        check(cudaMemcpyAsync(
                  workspace.ple_scales + first_row,
                  workspace.host_ple_scales + first_row,
                  row_count * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice, workspace.stream),
              "cudaMemcpyAsync(PLE scale tile)");
    }

    void drain_ple_read() noexcept {
        if (!ple_read_active) return;
        try {
            ple_reader->wait();
        } catch (...) {
        }
        ple_read_active = false;
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
    TxnKind provisional_kind = TxnKind::kInvalid;
    bool states_active = false;
    bool mtp_pending_valid = false;
    bool mtp_candidate_pending_valid = false;
    bool mtp_transaction_active = false;
    std::uint32_t mtp_transaction_rows = 0;
    bool ple_read_active = false;
    std::size_t ple_read_bytes = 0;
    std::size_t ple_read_scales = 0;
    std::unique_ptr<DecodeEventProfile> decode_profile;
    mutable std::uint64_t tracked_peak_bytes = 0;
    StageBackendMetricsV1 metrics{};
};

CudaStageBackend::CudaStageBackend(DeviceStageIndexV1 index,
                                   CudaStageBackendOptions options,
                                   std::shared_ptr<PleStore> ple_store)
    : impl_(std::make_unique<Impl>(std::move(index), options,
                                   std::move(ple_store))) {}
CudaStageBackend::~CudaStageBackend() = default;

Stage CudaStageBackend::stage() const { return impl_->options.stage; }

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
        state.provisional_kind = input.txn.kind;
        state.provisional_tokens.clear();
        state.provisional_tokens.reserve(input.txn.evaluated_count);
        state.provisional_cancellation = input.cancellation;
        state.candidate_ple_hashes.clear();
        if (state.committed_ple_hash)
            state.candidate_ple_hashes.reserve(input.txn.evaluated_count);
        state.mtp_transaction_active = false;
        state.mtp_transaction_rows = 0;
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
        cuda_copy_boundary_to_device(input.boundary, state.workspace.boundary,
                                     reinterpret_cast<void*>(
                                         state.workspace.stream),
                                     state.options.device);
    } else {
        state.start_ple_read(input.token_ids);
    }

    state.provisional_tokens.insert(state.provisional_tokens.end(),
                                    input.token_ids.begin(),
                                    input.token_ids.end());
    try {
        check(cudaMemcpyAsync(state.workspace.tokens, input.token_ids.data(),
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
        check(cudaMemcpyAsync(
                  state.workspace.host_predictions, state.workspace.predictions,
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
            if (!first || input.chunk_offset != 0 || !input.final_chunk)
                throw std::runtime_error(
                    "CUDA speculative verification cannot be chunked");
            accepted = 1;
            while (accepted < count &&
                   input.token_ids[accepted] ==
                       state.workspace.host_predictions[accepted - 1])
                ++accepted;
        }
        result.state_commit_count =
            input.txn.kind == TxnKind::kSpeculative
                ? accepted
                : state.provisional_processed;
        result.next_token = state.workspace.host_predictions[accepted - 1];
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
    if (cancellation) cancellation->throw_if_requested();
    if (state.options.stage != Stage::kStage1)
        throw std::runtime_error("stage0 cannot draft tokens");
    if (!state.mtp)
        throw std::logic_error("MTP is disabled for this executor");
    if (max_draft == 0) return {};
    check(cudaSetDevice(state.options.device), "cudaSetDevice(MTP draft)");
    if (pending_token < 0 ||
        static_cast<std::uint32_t>(pending_token) >= kVocabulary ||
        position != state.committed_frontier || state.provisional_epoch != 0 ||
        !state.mtp_pending_valid || !state.mtp_qsa_state ||
        state.mtp_qsa_state->committed_tokens() + 1 != position)
        throw std::runtime_error("MTP draft frontier is not ready");

    // Draft writes the append-only cache row at the next logical position,
    // reads the proposal, then rolls the logical length back.  Target commit
    // later recomputes this row using only the accepted prefix, so a rejected
    // proposal can never leak into semantic state.
    constexpr std::uint64_t kDraftEpoch =
        std::numeric_limits<std::uint64_t>::max();
    bool active = false;
    try {
        check(cudaMemcpyAsync(state.workspace.tokens, &pending_token,
                              sizeof(pending_token), cudaMemcpyHostToDevice,
                              state.workspace.stream),
              "cudaMemcpyAsync(MTP token)");
        state.mtp_qsa_state->begin(kDraftEpoch);
        active = true;
        state.run_mtp_step(
            state.workspace.mtp_pending_hc, state.workspace.tokens,
            static_cast<std::uint32_t>(
                state.mtp_qsa_state->committed_tokens()),
            true);
        state.mtp_qsa_state->mark_evaluated(kDraftEpoch, 1);
        check(cudaMemcpyAsync(state.workspace.host_predictions,
                              state.workspace.predictions,
                              sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                              state.workspace.stream),
              "cudaMemcpyAsync(MTP prediction)");
        check(cudaStreamSynchronize(state.workspace.stream),
              "cudaStreamSynchronize(MTP draft)");
        if (cancellation) cancellation->throw_if_requested();
        state.mtp_qsa_state->rollback(kDraftEpoch);
        active = false;
        return {state.workspace.host_predictions[0]};
    } catch (...) {
        if (active) state.mtp_qsa_state->rollback(kDraftEpoch);
        throw;
    }
}

void CudaStageBackend::commit(std::uint64_t epoch,
                              std::uint32_t state_commit_count) {
    auto& state = *impl_;
    check(cudaSetDevice(state.options.device), "cudaSetDevice(stage commit)");
    if (epoch != state.provisional_epoch || !state.states_active ||
        state.provisional_processed != state.provisional_expected ||
        state_commit_count == 0 ||
        state_commit_count > state.provisional_processed)
        throw std::runtime_error("CUDA backend commit does not match transaction");
    if (state_commit_count < state.provisional_processed) {
        state.gdn_state->restore(state.workspace.stream);
        if (state.ple_state) state.ple_state->restore(state.workspace.stream);
        state.run_tokens(state_commit_count, 0, false);
        check(cudaStreamSynchronize(state.workspace.stream),
              "cudaStreamSynchronize(partial replay)");
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
        state.mtp_qsa_state->commit(epoch, state.mtp_transaction_rows);
        state.mtp_transaction_active = false;
        state.mtp_transaction_rows = 0;
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
        state.mtp_qsa_state->rollback(epoch);
        state.mtp_transaction_active = false;
        state.mtp_transaction_rows = 0;
    }
    state.committed_epoch = epoch;
    state.provisional_epoch = 0;
    state.provisional_expected = 0;
    state.provisional_processed = 0;
    state.provisional_kind = TxnKind::kInvalid;
    state.provisional_tokens.clear();
    state.provisional_cancellation.reset();
    state.candidate_ple_hashes.clear();
    state.states_active = false;
    state.mtp_candidate_pending_valid = false;
    ++state.metrics.rollbacks;
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
    result.gdn_state_bytes = 2ull * state.gdn_state->bytes_per_bank();
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
    bool enable_mtp) {
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
    CudaStageBackendOptions options1 = options0;
    options1.stage = Stage::kStage1;
    options1.device = stage1_device;
    options1.enable_mtp = enable_mtp;
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
