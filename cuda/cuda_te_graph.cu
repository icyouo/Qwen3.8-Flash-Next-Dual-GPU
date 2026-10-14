#include "q38/cuda_te_graph.h"
#include "q38/cuda_te.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace q38 {
namespace {
void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
}
struct Piece {
    cudaGraphExec_t graph = nullptr;
    cudaGraph_t source = nullptr;
    TeCollectiveCall call{};
    std::size_t nodes = 0;
    std::unique_ptr<Piece> parallel;
    ~Piece() {
        if (graph) (void)cudaGraphExecDestroy(graph);
        if (source) (void)cudaGraphDestroy(source);
    }
};
}
struct CudaTeGraphTape::Impl final : TeCollectiveRecorder {
    int device;
    cudaStream_t stream;
    bool capturing = false;
    bool branch_recording = false;
    std::vector<std::unique_ptr<Piece>> pieces;
    std::size_t graph_count = 0, node_count = 0, collective_count = 0;
    bool paired = false;
    int group = -1;
    struct Joint {
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t graph = nullptr;
        bool device_fences = false;
        cudaGraph_t rank_graph[2] = {};
        cudaGraphExec_t rank_exec[2] = {};
        cudaEvent_t rank_done[2] = {};
        unsigned* flags[2] = {};
        bool rank_pending[2] = {}, rank_recorded[2] = {};
        std::size_t flag_bytes = 0;
        void* scratch[2] = {};
        std::size_t scratch_bytes = 0;
        cudaStream_t streams[2] = {};
        cudaEvent_t incoming = nullptr, done = nullptr;
        bool pending = false;
        bool completion_recorded = false;
        void drain() {
            if (device_fences) {
                std::exception_ptr error;
                for (int r = 0; r < 2; ++r) if (rank_pending[r]) {
                    try {
                        check(cudaSetDevice(r), "select rank-local TE drain device");
                        if (rank_recorded[r]) check(cudaEventSynchronize(rank_done[r]), "drain rank-local TE graph");
                        else check(cudaStreamSynchronize(streams[r]), "drain interrupted rank-local graph");
                    } catch (...) { if (!error) error = std::current_exception(); }
                    rank_pending[r] = false;
                }
                if (error) std::rethrow_exception(error);
                return;
            }
            if (!pending) return;
            check(cudaSetDevice(0), "select paired TE drain device");
            if (completion_recorded) check(cudaEventSynchronize(done), "drain paired TE completion");
            else check(cudaStreamSynchronize(streams[0]), "drain interrupted paired TE launch");
            pending = false;
        }
        ~Joint() {
            int previous = 0;
            (void)cudaGetDevice(&previous);
            (void)cudaSetDevice(0);
            try { drain(); } catch (...) {}
            if (exec) (void)cudaGraphExecDestroy(exec);
            if (graph) (void)cudaGraphDestroy(graph);
            if (done) (void)cudaEventDestroy(done);
            for (int r = 0; r < 2; ++r) {
                (void)cudaSetDevice(r);
                if (rank_exec[r]) (void)cudaGraphExecDestroy(rank_exec[r]);
                if (rank_graph[r]) (void)cudaGraphDestroy(rank_graph[r]);
                if (rank_done[r]) (void)cudaEventDestroy(rank_done[r]);
                if (flags[r]) (void)cudaFree(flags[r]);
                if (scratch[r]) (void)cudaFree(scratch[r]);
            }
            if (incoming) (void)cudaEventDestroy(incoming);
            (void)cudaSetDevice(previous);
        }
        void build(Impl& a, Impl& b) {
            if (a.device != 0 || b.device != 1 || a.group != b.group ||
                a.collective_count != b.collective_count || !a.collective_count)
                throw std::logic_error("paired TE tape identity differs");
            Impl* ranks[] = {&a, &b};
            const auto* fence_mode = std::getenv("Q38_TP_GPU_HANDSHAKE");
            device_fences = fence_mode && std::strcmp(fence_mode, "1") == 0;
            std::size_t bytes = 0;
            for (const auto& piece : a.pieces) if (!piece->source) {
                const auto& call = piece->call;
                const std::size_t unit = call.kind == TeCollectiveCall::Kind::kSumF32 ? 4 : 2;
                if (call.elements > std::numeric_limits<std::size_t>::max() / call.tokens / unit / 2)
                    throw std::overflow_error("paired TE graph scratch overflow");
                bytes = std::max(bytes, call.elements * call.tokens * unit *
                    (call.kind == TeCollectiveCall::Kind::kGatherBf16 ? 2 : 1));
            }
            for (int r = 0; r < 2; ++r) {
                check(cudaSetDevice(r), "select paired TE allocation device");
                streams[r] = ranks[r]->stream;
                check(cudaMalloc(&scratch[r], bytes), "allocate paired TE graph scratch");
                if (device_fences) {
                    flag_bytes = a.collective_count * 2 * sizeof(unsigned);
                    check(cudaMalloc(&flags[r], flag_bytes), "allocate TE graph generations");
                    check(cudaMemset(flags[r], 0, flag_bytes), "initialize TE graph generations");
                    check(cudaGraphCreate(&rank_graph[r], 0), "create rank-local TE graph");
                    check(cudaEventCreateWithFlags(&rank_done[r], cudaEventDisableTiming), "create rank-local TE completion");
                }
            }
            scratch_bytes = bytes;
            if (!device_fences) {
                check(cudaEventCreateWithFlags(&incoming, cudaEventDisableTiming), "create paired TE input event");
                check(cudaSetDevice(0), "select paired TE graph device");
                check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "create paired TE completion event");
                check(cudaGraphCreate(&graph, 0), "create paired TE graph");
            }
            std::size_t cursor[2] = {};
            void* tails[2] = {};
            for (std::size_t index = 0; index <= a.collective_count; ++index) {
                TeCollectiveCall calls[2]{};
                Piece* collective_pieces[2]{};
                for (int r = 0; r < 2; ++r) {
                    check(cudaSetDevice(r), "select paired TE child device");
                    auto& pieces = ranks[r]->pieces;
                    while (cursor[r] < pieces.size() && pieces[cursor[r]]->source) {
                        auto dependency = static_cast<cudaGraphNode_t>(tails[r]);
                        cudaGraphNode_t child;
                        check(cudaGraphAddChildGraphNode(&child, device_fences ? rank_graph[r] : graph, dependency ? &dependency : nullptr,
                            dependency ? 1 : 0, pieces[cursor[r]++]->source), "add paired TE GPU fragment");
                        tails[r] = child;
                    }
                    if (index < a.collective_count) {
                        if (cursor[r] == pieces.size()) throw std::logic_error("paired TE collective missing");
                        collective_pieces[r] = pieces[cursor[r]++].get();
                        calls[r] = collective_pieces[r]->call;
                    } else if (cursor[r] != pieces.size()) {
                        throw std::logic_error("paired TE collective count differs");
                    }
                }
                if (index < a.collective_count) {
                    void* branches[2]{};
                    if (bool(collective_pieces[0]->parallel) != bool(collective_pieces[1]->parallel))
                        throw std::logic_error("paired TE branch descriptors differ");
                    for (int r = 0; r < 2; ++r) if (collective_pieces[r]->parallel) {
                        check(cudaSetDevice(r), "select parallel TE child device");
                        const auto dependency = static_cast<cudaGraphNode_t>(tails[r]);
                        cudaGraphNode_t child;
                        check(cudaGraphAddChildGraphNode(&child, device_fences ? rank_graph[r] : graph,
                            dependency ? &dependency : nullptr, dependency ? 1 : 0,
                            collective_pieces[r]->parallel->source), "add independent TE child");
                        branches[r] = child;
                    }
                    if (device_fences) {
                        void* graphs[] = {rank_graph[0], rank_graph[1]};
                        void* signals[] = {flags[0] + index * 2, flags[1] + index * 2};
                        te_graph_add_device_collective(graphs, calls, scratch, signals, tails, tails);
                    } else te_graph_add_collective(graph, calls, scratch, tails, tails);
                    for (int r = 0; r < 2; ++r) if (branches[r]) {
                        check(cudaSetDevice(r), "select parallel TE join device");
                        cudaGraphNode_t dependencies[] = {static_cast<cudaGraphNode_t>(tails[r]),
                                                          static_cast<cudaGraphNode_t>(branches[r])};
                        cudaGraphNode_t joined;
                        check(cudaGraphAddEmptyNode(&joined, device_fences ? rank_graph[r] : graph,
                            dependencies, 2), "join parallel TE work and collective");
                        tails[r] = joined;
                    }
                }
            }
            if (device_fences) {
                for (int r = 0; r < 2; ++r) {
                    check(cudaSetDevice(r), "instantiate rank-local TE device");
                    check(cudaGraphInstantiate(&rank_exec[r], rank_graph[r], nullptr, nullptr, 0), "instantiate rank-local TE graph");
                    check(cudaGraphDestroy(rank_graph[r]), "release rank-local TE source");
                    rank_graph[r] = nullptr;
                }
                return;
            }
            check(cudaSetDevice(0), "instantiate paired TE graph device");
            check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate paired TE graph");
            check(cudaGraphDestroy(graph), "release paired TE graph source");
            graph = nullptr;
        }
        void launch() {
            if (device_fences) {
                // Each stream orders its own inputs and consumers. Matching
                // generation/read-done kernels establish cross-device order.
                for (int r = 0; r < 2; ++r) {
                    check(cudaSetDevice(r), "select rank-local TE launch device");
                    check(cudaGraphLaunch(rank_exec[r], streams[r]), "launch rank-local TE graph");
                    rank_pending[r] = true; rank_recorded[r] = false;
                    check(cudaEventRecord(rank_done[r], streams[r]), "record rank-local TE completion");
                    rank_recorded[r] = true;
                }
                return;
            }
            // Stream1 includes its prior wait on done. Record incoming BEFORE
            // overwriting done, so repeated queued generations cannot cycle.
            check(cudaSetDevice(1), "select paired TE input device");
            check(cudaEventRecord(incoming, streams[1]), "record paired TE input readiness");
            check(cudaSetDevice(0), "select paired TE launch device");
            check(cudaStreamWaitEvent(streams[0], incoming, 0), "wait paired TE peer inputs");
            check(cudaGraphLaunch(exec, streams[0]), "launch paired TE graph");
            // Mark pending before completion-event creation can fail. On error
            // the bridge is poisoned and destruction drains the launch stream.
            pending = true;
            completion_recorded = false;
            check(cudaEventRecord(done, streams[0]), "record paired TE completion");
            completion_recorded = true;
            check(cudaSetDevice(1), "select paired TE consumer device");
            check(cudaStreamWaitEvent(streams[1], done, 0), "wait paired TE output readiness");
        }
    };
    std::shared_ptr<Joint> joint;
    static void replay_pair(void* first, void* second) {
        auto& a = *static_cast<Impl*>(first);
        auto& b = *static_cast<Impl*>(second);
        int previous = 0;
        check(cudaGetDevice(&previous), "get paired TE current device");
        try {
            if (!a.joint || a.joint != b.joint) {
                auto next = std::make_shared<Joint>();
                next->build(a, b);
                a.joint = next; b.joint = next;
            }
            a.joint->launch();
        } catch (...) { (void)cudaSetDevice(previous); throw; }
        check(cudaSetDevice(previous), "restore paired TE current device");
    }
    Impl(int d, void* s, const std::function<void()>& enqueue)
        : device(d), stream(reinterpret_cast<cudaStream_t>(s)) {
        if (device < 0 || !stream || !enqueue)
            throw std::invalid_argument("invalid TE graph tape construction");
        const auto* mode = std::getenv("Q38_TP_PAIRED_GRAPH");
        paired = mode && std::strcmp(mode, "1") == 0;
        if (paired && (!te_bridge_p2p_enabled() || device > 1))
            throw std::invalid_argument("paired TE graphs require TP2 P2P");
        check(cudaSetDevice(device), "select TE graph device");
        check(cudaStreamSynchronize(stream), "TE graph setup boundary");
        te_bridge_set_recorder(this);
        try {
            begin();
            enqueue();
            end();
            te_bridge_set_recorder(nullptr);
        } catch (...) {
            te_bridge_set_recorder(nullptr);
            if (capturing) {
                cudaGraph_t discarded = nullptr;
                (void)cudaStreamEndCapture(stream, &discarded);
                if (discarded) (void)cudaGraphDestroy(discarded);
                capturing = false;
                (void)cudaGetLastError();
            }
            throw;
        }
    }
    ~Impl() {
        (void)cudaSetDevice(device);
        if (joint) { try { joint->drain(); } catch (...) {} }
        joint.reset();
        (void)cudaSetDevice(device);
        pieces.clear();
    }
    void begin() {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
              "begin TE GPU segment");
        capturing = true;
    }
    void end() {
        cudaGraph_t source = nullptr;
        const auto status = cudaStreamEndCapture(stream, &source);
        capturing = false;
        if (status != cudaSuccess) {
            if (source) (void)cudaGraphDestroy(source);
            check(status, "end TE GPU segment");
        }
        if (!source) throw std::runtime_error("TE capture returned no source graph");
        try {
            auto piece = std::make_unique<Piece>();
            check(cudaGraphGetNodes(source, nullptr, &piece->nodes), "count TE graph nodes");
            // Adjacent collectives produce empty GPU segments. Do not launch
            // an empty graph or lose the intervening collective order.
            if (piece->nodes) {
                check(cudaGraphInstantiate(&piece->graph, source, nullptr, nullptr, 0),
                      "instantiate TE GPU segment");
                // Transfer ownership instead of cloning: ordinary cuBLAS
                // captures may contain allocation nodes, which cannot clone.
                piece->source = source;
                source = nullptr;
                const auto nodes = piece->nodes;
                pieces.push_back(std::move(piece));
                ++graph_count; node_count += nodes;
            }
        } catch (...) {
            if (source) (void)cudaGraphDestroy(source);
            throw;
        }
        if (source) check(cudaGraphDestroy(source), "release TE source graph");
    }
    void validate(const TeCollectiveCall& call) const {
        if (branch_recording || !capturing || call.stream != stream || call.rank != device || call.rank > 1 ||
            call.group < 0 || call.group > 1 || !call.input || !call.elements || !call.tokens ||
            (group >= 0 && call.group != group) ||
            (call.kind == TeCollectiveCall::Kind::kGatherBf16 && !call.output))
            throw std::logic_error("invalid collective in TE graph tape");
    }
    void record(const TeCollectiveCall& call) override {
        validate(call);
        group = call.group;
        end();
        auto piece = std::make_unique<Piece>();
        piece->call = call;
        pieces.push_back(std::move(piece));
        ++collective_count;
        begin();
    }
    void record_parallel(const TeCollectiveCall& call,
                         const std::function<void()>& enqueue) override {
        if (!paired) { enqueue(); record(call); return; }
        validate(call);
        group = call.group;
        end();
        const auto previous = pieces.size();
        begin();
        branch_recording = true;
        try { enqueue(); } catch (...) { branch_recording = false; throw; }
        branch_recording = false;
        end();
        if (pieces.size() != previous + 1)
            throw std::logic_error("empty independent TE branch");
        auto piece = std::make_unique<Piece>();
        piece->parallel = std::move(pieces.back());
        pieces.pop_back();
        piece->call = call;
        pieces.push_back(std::move(piece));
        ++collective_count;
        begin();
    }
};
CudaTeGraphTape::CudaTeGraphTape(int device, void* stream,
                               const std::function<void()>& enqueue)
    : impl_(std::make_unique<Impl>(device, stream, enqueue)) {}
CudaTeGraphTape::~CudaTeGraphTape() = default;
void CudaTeGraphTape::launch(void* stream, const std::function<void()>& cancel) const {
    if (stream != impl_->stream) throw std::logic_error("TE graph replay stream differs");
    check(cudaSetDevice(impl_->device), "select TE graph replay device");
    if (impl_->paired && impl_->collective_count) {
        if (cancel) cancel();
        te_bridge_pair_graph(impl_->group, impl_->device, impl_.get(), Impl::replay_pair);
        return;
    }
    for (const auto& piece : impl_->pieces) {
        if (cancel) cancel();
        if (piece->graph)
            check(cudaGraphLaunch(piece->graph, impl_->stream), "launch TE GPU segment");
        else te_bridge_replay(piece->call);
    }
}
std::size_t CudaTeGraphTape::nodes() const { return impl_->node_count; }
std::size_t CudaTeGraphTape::graphs() const { return impl_->graph_count; }
std::size_t CudaTeGraphTape::collectives() const { return impl_->collective_count; }
std::size_t CudaTeGraphTape::held_bytes() const {
    // Each rank reports its own device's share of the shared paired plan.
    return impl_->joint ? impl_->joint->scratch_bytes + impl_->joint->flag_bytes : 0;
}
}
