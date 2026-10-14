#ifndef Q38_CUDA_TE_H
#define Q38_CUDA_TE_H

#include <cstddef>
#include <cstdint>
#include <functional>

namespace q38 {

struct TeHostBounceStats {
    std::uint64_t allreduce_calls = 0;
    std::uint64_t allgather_calls = 0;
    std::uint64_t bytes_moved = 0;
    std::uint64_t collective_ns = 0;
    std::uint64_t paired_graph_calls = 0;
};

void te_host_bounce_allreduce_sum_f32(float* device0, float* device1,
                                      std::size_t elements,
                                      TeHostBounceStats* stats = nullptr);

void te_host_bounce_allgather_f32(float* device0_shard, float* device1_shard,
                                  float* device0_full, float* device1_full,
                                  std::size_t shard_elements,
                                  TeHostBounceStats* stats = nullptr);

bool cuda_te_compiled();

// Thread-local recording only: no rendezvous or GPU execution/state advancement.
// A recorder splits a GPU capture at these CPU collective boundaries.
struct TeCollectiveCall {
    enum class Kind { kGatherBf16, kSumBf16, kSumF32 };
    Kind kind;
    int rank;
    void* input;
    void* output;
    std::size_t elements;
    std::size_t tokens;
    int group;
    void* stream;
};
class TeCollectiveRecorder {
public:
    virtual ~TeCollectiveRecorder() = default;
    virtual void record(const TeCollectiveCall& call) = 0;
    virtual void record_parallel(const TeCollectiveCall& call,
                                 const std::function<void()>& enqueue) {
        enqueue();
        record(call);
    }
};
void te_bridge_set_recorder(TeCollectiveRecorder* recorder);
void te_bridge_replay(const TeCollectiveCall& call);

// One rendezvous for a complete paired graph fragment. The action receives
// rank-ordered contexts and must enqueue dependencies on both producer streams.
// Uses the same poison/reset contract as ordinary collectives.
using TePairedAction = void (*)(void*, void*);
void te_bridge_pair_graph(int group, int rank, void* context, TePairedAction action);
bool te_bridge_p2p_enabled();

// CUDA handles are opaque here to keep the public runtime header CUDA-free.
// Add exact peer-read and commit nodes; both reads precede either overwrite.
void te_graph_add_collective(void* graph, const TeCollectiveCall calls[2],
                             void* scratch[2], void* dependencies[2],
                             void* completed[2]);
// Rank-local graphs with GPU-side generation fences. Each flags allocation
// contains two uint32 words, initially zero, and is unique to this collective.
void te_graph_add_device_collective(void* graphs[2], const TeCollectiveCall calls[2],
                                    void* scratch[2], void* flags[2],
                                    void* dependencies[2], void* completed[2]);

// Dual-rank lockstep bridge for host-bounce collectives during decode.
// Both ranks must call the matching te_bridge_* entry in the same order.
struct TeBridge {
    int world_size = 1;
};

// Process-wide TE bridge (one DualRank TE runtime per process).
TeBridge* te_bridge_global();
void te_bridge_reset();
// Caller has joined both rank tasks and drained their streams for this group.
void te_bridge_reset_group(int group);
// Configure only while both groups are idle. P2P has no host-staging fallback.
void te_bridge_use_p2p(bool enabled, bool stream_ordered = false);
// Poison both groups and wake peers after a rank failure. Reset only when idle.
void te_bridge_abort();
TeHostBounceStats te_bridge_stats(int group = -1);

// Collectives copy on each rank's producer stream and return after both
// outputs are ready in the default synchronous mode. Opt-in stream_ordered
// P2P returns after enqueue: subsequent consumers must use that rank's producer
// stream or explicitly wait for it. Host publication still needs synchronization.
// The caller's current CUDA device is preserved. Reset/transport change drains
// queued peer work; do not free input/output allocations before their stream ends.
// Rank registers its device pointer, waits for peer, runs collective, returns.
// half/full are device pointers on the calling rank's device.
// shard_elements = per-token half width; tokens>1 does token-major interleave
// (rank0_t || rank1_t) per token, not flat [all_r0 | all_r1].
// group: 0=stage0, 1=stage1 — independent 2-rank barriers so PP workers
// on the same TE rank do not steal each other's arrivals.
void te_bridge_allgather_bf16(int rank, std::uint16_t* half, std::uint16_t* full,
                              std::size_t shard_elements,
                              std::size_t tokens = 1, int group = 0,
                              void* stream = nullptr);
void te_bridge_allreduce_sum_bf16(int rank, std::uint16_t* buf,
                                  std::size_t elements, int group = 0,
                              void* stream = nullptr);
void te_bridge_allreduce_sum_f32(int rank, float* buf,
                                 std::size_t elements, int group = 0,
                              void* stream = nullptr);
// Eager P2P: reduce-scatter in FP32, round each sum once, then gather BF16.
// Inputs remain unchanged. Output is a separate rank-local elements array.
// Both inputs and outputs must remain alive through the producer stream.
// Recording/Graph and host-bounce transports are intentionally unsupported.
void te_bridge_allreduce_f32_to_bf16(int rank, const float* input,
    std::uint16_t* output, std::size_t elements, int group = 0,
    void* stream = nullptr);
bool te_bridge_f32_to_bf16_available();
// BF16 communication policy: round each rank's FP32 partial before reduction.
// This intentionally differs from summing the original FP32 partials first.
// Inputs are immutable and outputs must not overlap them. Supports eager,
// host-bounce and Graph recording, including an independent parallel branch.
void te_bridge_allreduce_f32_as_bf16(int rank, const float* input,
    std::uint16_t* output, std::size_t elements, int group = 0,
    void* stream = nullptr, const std::function<void()>& enqueue = {});
// Enqueue an independent GPU branch synchronously. Paired graph recording
// forks it alongside the collective and joins both before subsequent work.
// The branch must not access collective buffers or record other collectives.
// Eager execution is ordered.
void te_bridge_allreduce_sum_f32_parallel(int rank, float* buf,
    std::size_t elements, int group, void* stream,
    const std::function<void()>& enqueue);

}  // namespace q38

#endif
